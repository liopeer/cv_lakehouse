#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""An embeddings run embeds only what no earlier run embedded with the same model."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray

from lakehouse_core.lake_files import remove_tree
from lakehouse_cv.contract.box_identity import derive_box_id
from lakehouse_cv.contract.silver_tables import (
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_SCHEMA,
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
)
from lakehouse_cv.defs import corrections as corrections_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs import silver_embeddings as embeddings_defs
from lakehouse_cv.defs.corrections import build_corrections_asset
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.transforms.embeddings import EMBEDDING_DIMENSION, Crop, Embedder
from tests.export_fakes import EXPORT_URL, FakeExportServer
from tests.lake_runs import (
    find_embedding_files,
    find_silver_files,
    materialize_assets,
    materialize_bronze_links,
)

DATASET = "wider_face"
PARADE = "0--Parade/a.jpg"
HANDSHAKING = "1--Handshaking/b.jpg"
FACE = derive_box_id(
    dataset=DATASET, split="train", file_name=PARADE, source_box_index=0
)
DRAWN = "aaaaaaaa-0000-0000-0000-000000000001"


class ContentEmbedder(Embedder):
    """Give each input a vector of its own, and remember every request."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.crops: list[Crop] = []

    def embed_images(self, paths: Sequence[str]) -> NDArray[np.float32]:
        self.paths.extend(paths)
        return _vectors_of([hash(path) for path in paths])

    def embed_crops(self, crops: Sequence[Crop]) -> NDArray[np.float32]:
        self.crops.extend(crops)
        return _vectors_of([hash(crop) for crop in crops])

    def forget_requests(self) -> None:
        self.paths.clear()
        self.crops.clear()


def _vectors_of(seeds: list[int]) -> NDArray[np.float32]:
    vectors = np.zeros(shape=(len(seeds), EMBEDDING_DIMENSION), dtype=np.float32)
    for row, seed in enumerate(seeds):
        vectors[row, seed % EMBEDDING_DIMENSION] = 1.0
    return vectors


class _Lake:
    """A lake with linked bronze, a fake export, and one embedder for every run."""

    def __init__(self, resource: CvLakeResource, server: FakeExportServer) -> None:
        self.resource = resource
        self.server = server
        self.embedder = ContentEmbedder()

    def build_silver(self, rows: list[dict] | None = None) -> None:
        """Publish the rows as an event file, if any, and rebuild silver."""
        if rows is not None:
            self.server.publish(dataset=DATASET, rows=rows)
        self.embedder.forget_requests()
        materialize_assets(
            lake=self.resource,
            assets=[
                build_corrections_asset(DATASET),
                silver_defs.build_silver_asset(DATASET),
                embeddings_defs.build_embeddings_asset(DATASET),
            ],
        )

    def read_crop_vectors(self) -> dict[str, list[float]]:
        table = pq.read_table(
            crop_embeddings_file(
                build_dir=find_embedding_files(lake=self.resource, name=DATASET),
                split="train",
            )
        )
        assert table.schema == CROP_EMBEDDING_SCHEMA
        return dict(
            zip(
                table.column("box_id").to_pylist(),
                table.column("embedding").to_pylist(),
                strict=True,
            )
        )

    def read_image_vectors(self) -> pa.Table:
        table = pq.read_table(
            embeddings_file(
                build_dir=find_embedding_files(lake=self.resource, name=DATASET),
                split="train",
            )
        )
        assert table.schema == EMBEDDING_SCHEMA
        return table


@pytest.fixture
def reuse_lake(
    lake: CvLakeResource,
    bronze_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> _Lake:
    server = FakeExportServer()
    resource = CvLakeResource(
        root=lake.root,
        download_workers=lake.download_workers,
        request_timeout_seconds=lake.request_timeout_seconds,
        triton_url="fake:0",
        studio_export_url=EXPORT_URL,
    )
    reuse_lake = _Lake(resource=resource, server=server)
    monkeypatch.setattr(
        target=corrections_defs,
        name="open_export_client",
        value=lambda timeout: server.client(),
    )
    monkeypatch.setattr(
        target=embeddings_defs,
        name="TritonEmbedder",
        value=lambda url: reuse_lake.embedder,
    )
    materialize_bronze_links(lake=resource, sources=bronze_sources)
    reuse_lake.build_silver()
    return reuse_lake


def _event(
    box_id: str,
    *,
    file_name: str = PARADE,
    label_name: str = "face",
    geometry: tuple[float, float, float, float] = (10.0, 20.0, 30.0, 40.0),
    is_deleted: bool = False,
) -> dict:
    x, y, w, h = (None, None, None, None) if is_deleted else geometry
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": file_name,
        "box_id": box_id,
        "is_deleted": is_deleted,
        "label_name": None if is_deleted else label_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    }


def test_the_first_run_embeds_everything(reuse_lake: _Lake) -> None:
    # Three train images and one val image. Two train boxes and one val box.
    assert len(reuse_lake.embedder.paths) == 4
    assert len(reuse_lake.embedder.crops) == 3


def test_a_rebuild_with_no_change_embeds_nothing(reuse_lake: _Lake) -> None:
    images = reuse_lake.read_image_vectors()
    crops = reuse_lake.read_crop_vectors()

    reuse_lake.build_silver()

    assert (reuse_lake.embedder.paths, reuse_lake.embedder.crops) == ([], [])
    assert reuse_lake.read_image_vectors() == images
    assert reuse_lake.read_crop_vectors() == crops


def test_a_relabelled_box_keeps_its_vector(reuse_lake: _Lake) -> None:
    crops = reuse_lake.read_crop_vectors()

    reuse_lake.build_silver([_event(FACE, label_name="head")])

    assert reuse_lake.embedder.crops == []
    assert reuse_lake.read_crop_vectors() == crops


def test_a_moved_box_is_the_only_one_embedded_again(reuse_lake: _Lake) -> None:
    crops = reuse_lake.read_crop_vectors()

    reuse_lake.build_silver([_event(FACE, geometry=(12.0, 22.0, 33.0, 44.0))])

    (crop,) = reuse_lake.embedder.crops
    assert (crop.x, crop.y, crop.width, crop.height) == (12, 22, 33, 44)
    assert reuse_lake.embedder.paths == []
    after = reuse_lake.read_crop_vectors()
    assert after[FACE] == _vectors_of([hash(crop)])[0].tolist()
    assert {box: v for box, v in after.items() if box != FACE} == {
        box: v for box, v in crops.items() if box != FACE
    }


def test_a_drawn_box_is_embedded_and_a_deleted_box_loses_its_row(
    reuse_lake: _Lake,
) -> None:
    reuse_lake.build_silver(
        [
            _event(FACE, is_deleted=True),
            _event(DRAWN, file_name=HANDSHAKING, geometry=(1.0, 2.0, 3.0, 4.0)),
        ]
    )

    (crop,) = reuse_lake.embedder.crops
    assert crop.path.endswith(HANDSHAKING)
    vectors = reuse_lake.read_crop_vectors()
    assert DRAWN in vectors and FACE not in vectors
    # One row per box, in the order of the id.
    boxes = pq.read_table(
        boxes_file(
            build_dir=find_silver_files(lake=reuse_lake.resource, name=DATASET),
            split="train",
        )
    )
    assert list(vectors) == sorted(boxes.column("box_id").to_pylist())


def test_a_rebuild_on_new_code_reuses_every_vector(
    reuse_lake: _Lake, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vector depends on the pixels, the crop and the model, and on no other code."""
    images = reuse_lake.read_image_vectors()
    crops = reuse_lake.read_crop_vectors()
    monkeypatch.setattr(target=silver_defs, name="SILVER_LOGIC_VERSION", value="next")

    reuse_lake.build_silver()

    assert (reuse_lake.embedder.paths, reuse_lake.embedder.crops) == ([], [])
    assert reuse_lake.read_image_vectors() == images
    assert reuse_lake.read_crop_vectors() == crops


def test_a_rebuild_leaves_no_working_file_behind(reuse_lake: _Lake) -> None:
    reuse_lake.build_silver()
    embeddings_dir = reuse_lake.resource.paths.embeddings_dir(DATASET)
    names = {
        path.relative_to(embeddings_dir).parts[0]
        for path in embeddings_dir.rglob("*")
        if path.is_file()
    }
    assert names == {"_embeddings.json", "builds"}
    current = find_embedding_files(lake=reuse_lake.resource, name=DATASET)
    assert {path.name for path in current.rglob("*") if path.is_file()} == {
        "train.parquet",
        "val.parquet",
    }


def test_the_vectors_that_silver_kept_before_this_asset_are_reused(
    reuse_lake: _Lake,
) -> None:
    """A lake of the layout before keeps its vectors beside silver."""
    crops = reuse_lake.read_crop_vectors()
    current = find_embedding_files(lake=reuse_lake.resource, name=DATASET)
    legacy_dir = reuse_lake.resource.paths.silver_dir(DATASET)
    for table in ("embeddings", "crop_embeddings"):
        (current / table).rename(legacy_dir / table)
    remove_tree(reuse_lake.resource.paths.embeddings_dir(DATASET))

    reuse_lake.build_silver()

    assert (reuse_lake.embedder.paths, reuse_lake.embedder.crops) == ([], [])
    assert reuse_lake.read_crop_vectors() == crops
