#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The silver embedding files: sorted, reused, and resumed after a stop."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from numpy.typing import NDArray
from upath import UPath

from lakehouse_core.lake_store import LakeStore
from lakehouse_cv.contract.box_identity import derive_box_id, derive_image_id
from lakehouse_cv.contract.silver_tables import (
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_SCHEMA,
    EMBEDDING_VERSION_METADATA_KEY,
    SilverBox,
    SilverImage,
)
from lakehouse_cv.transforms import embedding_files
from lakehouse_cv.transforms.embedding_files import (
    CROP_VECTORS,
    IMAGE_VECTORS,
    SplitVectorFiles,
    VectorCount,
    VectorTable,
    split_key_ranges,
    write_vectors,
)
from lakehouse_cv.transforms.embeddings import Crop
from lakehouse_cv.transforms.progress_log import ProgressLog
from lakehouse_cv.transforms.silver_writer import write_split
from tests.fakes import FakeEmbedder

NUM_IMAGES = 20


class _StoppingEmbedder(FakeEmbedder):
    """Fail on the request after the first `requests_before_stop` ones."""

    def __init__(self, requests_before_stop: int) -> None:
        super().__init__()
        self._requests_left = requests_before_stop

    def embed_images(self, paths: Sequence[str]) -> NDArray[np.float32]:
        if self._requests_left == 0:
            raise ConnectionError("The server went away.")
        self._requests_left -= 1
        return super().embed_images(paths)


def _write_silver(silver_dir: Path, boxes: Sequence[tuple[float, ...]] = ()) -> None:
    """Write images whose file order is not the order of their ids."""
    images = [
        SilverImage(
            file_name=f"{index:02d}.jpg",
            width=100,
            height=100,
            boxes=tuple(
                SilverBox(
                    box_id=derive_box_id(
                        dataset="d",
                        split="train",
                        file_name=f"{index:02d}.jpg",
                        source_box_index=box_index,
                    ),
                    box_index=box_index,
                    class_id=0,
                    class_name="face",
                    source_class="face",
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                )
                for box_index, (x, y, w, h) in enumerate(boxes)
            ),
        )
        for index in range(NUM_IMAGES)
    ]
    write_split(build_dir=UPath(silver_dir), dataset="d", split="train", images=images)


def _files(tmp_path: Path, table: VectorTable = IMAGE_VECTORS) -> SplitVectorFiles:
    """Rebuild one target in place, as a run that reuses its own last file."""
    target = table.target_file(build_dir=UPath(tmp_path), split="train")
    return SplitVectorFiles(
        rows=table.rows_file(build_dir=UPath(tmp_path), split="train"),
        target=target,
        reusable=(target,),
        parts_dir=UPath(tmp_path) / "parts" / target.parent.name,
    )


def _write(
    tmp_path: Path, embedder: FakeEmbedder, table: VectorTable = IMAGE_VECTORS
) -> VectorCount:
    return write_vectors(
        store=LakeStore(root=str(tmp_path), storage_options={}),
        embedder=embedder,
        table=table,
        files=_files(tmp_path=tmp_path, table=table),
        locate_image=lambda name: f"/lake/{name}",
        log=lambda message: None,
    )


def _read(tmp_path: Path, table: VectorTable = IMAGE_VECTORS):
    return pq.read_table(_files(tmp_path=tmp_path, table=table).target)


def test_the_vectors_are_sorted_by_the_id_and_name_the_model(tmp_path: Path) -> None:
    _write_silver(tmp_path, boxes=[(1.0, 2.0, 3.0, 4.0)])
    embedder = FakeEmbedder()

    images = _write(tmp_path, embedder)
    crops = _write(tmp_path, embedder, table=CROP_VECTORS)

    assert images == VectorCount(written=NUM_IMAGES, embedded=NUM_IMAGES)
    assert crops == VectorCount(written=NUM_IMAGES, embedded=NUM_IMAGES)
    image_table = _read(tmp_path)
    crop_table = _read(tmp_path, table=CROP_VECTORS)
    assert image_table.schema == EMBEDDING_SCHEMA
    assert crop_table.schema == CROP_EMBEDDING_SCHEMA
    expected_ids = sorted(
        derive_image_id(dataset="d", split="train", file_name=f"{index:02d}.jpg")
        for index in range(NUM_IMAGES)
    )
    assert image_table.column("image_id").to_pylist() == expected_ids
    box_ids = crop_table.column("box_id").to_pylist()
    assert box_ids == sorted(box_ids)
    assert image_table.schema.metadata == {
        EMBEDDING_MODEL_METADATA_KEY: b"mobileclip_s0",
        EMBEDDING_VERSION_METADATA_KEY: b"1",
    }
    assert not _files(tmp_path).parts_dir.exists()


def test_a_crop_rounds_half_to_even_and_names_at_least_one_pixel(
    tmp_path: Path,
) -> None:
    _write_silver(tmp_path, boxes=[(10.4, 10.6, 5.5, 0.2), (2.5, 3.5, 0.6, 1.5)])
    embedder = FakeEmbedder()

    _write(tmp_path, embedder, table=CROP_VECTORS)

    rectangles = {(crop.x, crop.y, crop.width, crop.height) for crop in embedder.crops}
    assert rectangles == {(10, 11, 6, 1), (2, 4, 1, 2)}
    assert Crop(path="/lake/00.jpg", x=10, y=11, width=6, height=1) in embedder.crops


def test_a_second_run_embeds_nothing(tmp_path: Path) -> None:
    _write_silver(tmp_path)
    _write(tmp_path, FakeEmbedder())
    before = _read(tmp_path)
    embedder = FakeEmbedder()

    count = _write(tmp_path, embedder)

    assert count == VectorCount(written=NUM_IMAGES, embedded=0)
    assert embedder.paths == []
    assert _read(tmp_path) == before


def test_a_stopped_run_keeps_every_closed_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target=embedding_files, name="ITEMS_PER_REQUEST", value=4)
    monkeypatch.setattr(target=embedding_files, name="ROWS_PER_PART", value=4)
    _write_silver(tmp_path)

    with pytest.raises(expected_exception=ConnectionError):
        _write(tmp_path, _StoppingEmbedder(requests_before_stop=3))

    files = _files(tmp_path)
    assert not files.target.exists()
    assert len(list(files.parts_dir.iterdir())) == 3
    embedder = FakeEmbedder()
    count = _write(tmp_path, embedder)
    assert count == VectorCount(written=NUM_IMAGES, embedded=NUM_IMAGES - 12)
    assert len(set(embedder.paths)) == NUM_IMAGES - 12
    assert not files.parts_dir.exists()


def test_a_part_that_a_stopped_run_left_open_is_removed(tmp_path: Path) -> None:
    _write_silver(tmp_path)
    parts_dir = _files(tmp_path).parts_dir
    parts_dir.mkdir(parents=True)
    (parts_dir / "part-0.writing").write_bytes(b"half a file")
    (parts_dir / "part-1.parquet").write_bytes(b"not parquet")

    count = _write(tmp_path, FakeEmbedder())

    assert count == VectorCount(written=NUM_IMAGES, embedded=NUM_IMAGES)


@pytest.mark.parametrize(
    argnames=("name", "key"),
    argvalues=[
        ("EMBEDDING_MODEL", EMBEDDING_MODEL_METADATA_KEY),
        ("EMBEDDING_VERSION", EMBEDDING_VERSION_METADATA_KEY),
    ],
)
def test_a_vector_of_another_model_or_version_is_embedded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, key: bytes
) -> None:
    _write_silver(tmp_path)
    _write(tmp_path, FakeEmbedder())
    monkeypatch.setattr(target=embedding_files, name=name, value="next")

    count = _write(tmp_path, FakeEmbedder())

    assert count == VectorCount(written=NUM_IMAGES, embedded=NUM_IMAGES)
    metadata = _read(tmp_path).schema.metadata
    assert metadata is not None
    assert metadata[key] == b"next"


def test_the_target_is_written_in_ranges_of_the_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(target=embedding_files, name="ROWS_PER_RANGE", value=1)
    _write_silver(tmp_path)

    count = _write(tmp_path, FakeEmbedder())

    assert count.written == NUM_IMAGES
    ids = _read(tmp_path).column("image_id").to_pylist()
    assert ids == sorted(set(ids))


def test_the_key_ranges_cover_every_string() -> None:
    ranges = split_key_ranges(0)

    assert len(ranges) == 16
    assert ranges[0] == (None, "1")
    assert ranges[-1] == ("f", None)
    assert all(
        upper == lower
        for (_, upper), (lower, _) in zip(ranges, ranges[1:], strict=False)
    )
    assert len(split_key_ranges(16 * embedding_files.ROWS_PER_RANGE + 1)) == 256


def test_the_progress_log_reports_the_rate_and_the_time_left() -> None:
    messages: list[str] = []
    now = [0.0]
    progress = ProgressLog(
        log=messages.append,
        label="train crop embeddings",
        total=1000,
        interval_seconds=60.0,
        clock=lambda: now[0],
    )

    now[0] = 30.0
    progress.advance(100)
    now[0] = 60.0
    progress.advance(100)

    now[0] = 75.0
    progress.finish()

    assert messages == [
        "train crop embeddings: 200 of 1,000 (20.0%), 3/s, 0:04:00 left",
        "train crop embeddings: 200 done in 0:01:15",
    ]
