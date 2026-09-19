#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Run bronze and silver end to end on the fixtures. Nothing downloads."""

import shutil
from collections import Counter
from pathlib import Path

import dagster as dg
import pyarrow.parquet as pq
import pytest

from cv_lakehouse.defs import silver as silver_defs
from cv_lakehouse.defs.bronze import build_bronze_asset, detect_materialized_splits
from cv_lakehouse.defs.resources import LakeResource
from cv_lakehouse.manifests import (
    BRONZE_MANIFEST,
    SILVER_MANIFEST,
    BronzeManifest,
    SilverManifest,
    read_manifest,
)
from cv_lakehouse.silver_schema import (
    BOX_SCHEMA,
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_SCHEMA,
    IMAGE_SCHEMA,
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
    images_file,
)
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME
from tests.fakes import FakeEmbedder
from tests.fixtures import make_all, make_wider_face

DATASETS = ("open_images", "wider_face", "pp4av")


@pytest.fixture
def lake(tmp_path: Path) -> LakeResource:
    return LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def embedding_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LakeResource:
    """A lake with a Triton server that is a fake, so no test needs a GPU."""
    monkeypatch.setattr(
        target=silver_defs, name="TritonEmbedder", value=lambda url: FakeEmbedder()
    )
    return LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
        triton_url="fake:0",
    )


@pytest.fixture
def bronze_sources(tmp_path: Path) -> dict[str, Path]:
    return make_all(tmp_path / "external")


def _materialize_bronze(lake: LakeResource, sources: dict[str, Path]) -> None:
    for name in DATASETS:
        asset = build_bronze_asset(name)
        result = dg.materialize(
            assets=[asset],
            resources={"lake": lake},
            run_config=dg.RunConfig(
                ops={asset.op.name: {"config": {"source_dir": str(sources[name])}}}
            ),
        )
        assert result.success


def _materialize(lake: LakeResource, assets: list) -> dg.ExecuteInProcessResult:
    result = dg.materialize(assets=assets, resources={"lake": lake})
    assert result.success
    return result


def test_bronze_links_instead_of_downloading(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    _materialize_bronze(lake=lake, sources=bronze_sources)
    for name in DATASETS:
        assert lake.paths.bronze_dir(name).is_symlink()
        assert lake.paths.bronze_dir(name).resolve() == bronze_sources[name].resolve()


def test_bronze_reports_only_the_splits_on_disk(tmp_path: Path) -> None:
    bronze = make_wider_face(tmp_path)
    shutil.rmtree(bronze / "WIDER_val")
    splits = detect_materialized_splits(
        source=SOURCE_BY_NAME["wider_face"], bronze_dir=bronze
    )
    assert splits == ("train",)


def test_bronze_rejects_a_link_to_an_incomplete_copy(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    """WIDER_test holds no label, and bronze needs it all the same."""
    shutil.rmtree(bronze_sources["wider_face"] / "WIDER_test")
    asset = build_bronze_asset("wider_face")
    result = dg.materialize(
        assets=[asset],
        resources={"lake": lake},
        run_config=dg.RunConfig(
            ops={
                asset.op.name: {
                    "config": {"source_dir": str(bronze_sources["wider_face"])}
                }
            }
        ),
        raise_on_error=False,
    )
    assert not result.success
    assert not (lake.paths.bronze_dir("wider_face") / "_bronze.json").exists()


def test_the_bronze_manifest_lists_every_published_file(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    _materialize_bronze(lake=lake, sources=bronze_sources)
    manifest = read_manifest(
        path=lake.paths.bronze_dir("pp4av") / BRONZE_MANIFEST, model=BronzeManifest
    )
    assert manifest.published_files == list(SOURCE_BY_NAME["pp4av"].published_files)


def test_silver_normalizes_every_dataset(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    _materialize_bronze(lake=lake, sources=bronze_sources)
    _materialize(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )

    pp4av = lake.paths.silver_dir("pp4av")
    manifest = read_manifest(path=pp4av / SILVER_MANIFEST, model=SilverManifest)
    assert set(manifest.splits) == {"test", "fisheye"}
    assert classes_of(silver_dir=pp4av, split="test") == {"face": 1, "license_plate": 1}

    silver_dir = lake.paths.silver_dir("wider_face")
    wider = read_manifest(path=silver_dir / SILVER_MANIFEST, model=SilverManifest)
    assert num_rows(images_file(silver_dir=silver_dir, split="train")) == 3
    # Both boxes of a.jpg, one of them the region the annotators rejected. Only the all
    # zero line of c.jpg drops, which `test_sources_wider_face` asserts on the
    # normalizer, the only place that can still see a dropped row.
    assert num_rows(boxes_file(silver_dir=silver_dir, split="train")) == 2
    assert wider.commercial_use is False
    assert "attr_invalid" in attrs_of(silver_dir=silver_dir, split="train")
    assert "attr_is_group_of" not in attrs_of(silver_dir=silver_dir, split="train")

    # Bronze holds a cat. Silver keeps its box under other, for a consumer to judge.
    open_images = lake.paths.silver_dir("open_images")
    # Three faces: one real, plus the crowd box and the depiction, both flagged.
    assert classes_of(silver_dir=open_images, split="validation") == {
        "face": 3,
        "license_plate": 1,
        "other": 1,
    }
    assert attrs_of(silver_dir=open_images, split="validation") == {
        "attr_is_group_of",
        "attr_is_depiction",
    }


def test_silver_writes_no_embedding_without_a_server(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    """A laptop with no Triton server still builds the layer."""
    _materialize_bronze(lake=lake, sources=bronze_sources)
    _materialize(lake=lake, assets=[silver_defs.build_silver_asset("wider_face")])

    silver_dir = lake.paths.silver_dir("wider_face")
    manifest = read_manifest(path=silver_dir / SILVER_MANIFEST, model=SilverManifest)
    assert manifest.embedding_model is None
    # Every split, not only the first: the layer is built and no vector file exists.
    assert manifest.splits
    for split in manifest.splits:
        assert images_file(silver_dir=silver_dir, split=split).exists()
        assert not embeddings_file(silver_dir=silver_dir, split=split).exists()
        assert not crop_embeddings_file(silver_dir=silver_dir, split=split).exists()


def test_silver_embeds_every_image_and_every_box(
    embedding_lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    _materialize_bronze(lake=embedding_lake, sources=bronze_sources)
    _materialize(
        lake=embedding_lake, assets=[silver_defs.build_silver_asset("wider_face")]
    )

    silver_dir = embedding_lake.paths.silver_dir("wider_face")
    manifest = read_manifest(path=silver_dir / SILVER_MANIFEST, model=SilverManifest)
    assert manifest.embedding_model == "mobileclip_s0"

    images = pq.read_table(embeddings_file(silver_dir=silver_dir, split="train"))
    crops = pq.read_table(crop_embeddings_file(silver_dir=silver_dir, split="train"))
    assert images.schema == EMBEDDING_SCHEMA
    assert crops.schema == CROP_EMBEDDING_SCHEMA
    # One vector per image row and one per box row, which is the join these files are
    # for. Nothing else records that the embedding pass covered the whole split.
    assert images.num_rows == num_rows(
        images_file(silver_dir=silver_dir, split="train")
    )
    assert crops.num_rows == num_rows(boxes_file(silver_dir=silver_dir, split="train"))
    # The key of the boxes file, so a join needs nothing else.
    assert crops.column("file_name").to_pylist() == boxes_of(silver_dir)


def boxes_of(silver_dir: Path) -> list[str]:
    return (
        pq.read_table(boxes_file(silver_dir=silver_dir, split="train"))
        .column("file_name")
        .to_pylist()
    )


def classes_of(silver_dir: Path, split: str) -> dict[str, int]:
    """Count one split's boxes per class. The manifest no longer carries the tally."""
    names = (
        pq.read_table(boxes_file(silver_dir=silver_dir, split=split))
        .column("class_name")
        .to_pylist()
    )
    return dict(Counter(names))


def num_rows(path: Path) -> int:
    return pq.read_metadata(path).num_rows


def attrs_of(silver_dir: Path, split: str) -> set[str]:
    """The `attr_*` columns this split fills. The manifest no longer names them."""
    table = pq.read_table(boxes_file(silver_dir=silver_dir, split=split))
    return {
        name
        for name in table.column_names
        if name.startswith("attr_") and table.column(name).null_count < table.num_rows
    }


def test_rematerializing_without_a_server_removes_the_embeddings(
    embedding_lake: LakeResource, lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    """A vector left behind would describe a row that no longer exists.

    Both fixtures resolve to the same root, so the second run rebuilds the first.
    """
    _materialize_bronze(lake=embedding_lake, sources=bronze_sources)
    _materialize(
        lake=embedding_lake, assets=[silver_defs.build_silver_asset("wider_face")]
    )
    silver_dir = embedding_lake.paths.silver_dir("wider_face")
    assert embeddings_file(silver_dir=silver_dir, split="train").exists()

    _materialize(lake=lake, assets=[silver_defs.build_silver_asset("wider_face")])

    assert not embeddings_file(silver_dir=silver_dir, split="train").exists()
    assert not crop_embeddings_file(silver_dir=silver_dir, split="train").exists()
    manifest = read_manifest(path=silver_dir / SILVER_MANIFEST, model=SilverManifest)
    assert manifest.embedding_model is None


def test_silver_checks_pass(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    _materialize_bronze(lake=lake, sources=bronze_sources)
    for name in DATASETS:
        result = dg.materialize(
            assets=[
                silver_defs.build_silver_asset(name),
                *silver_defs.build_silver_checks(name),
            ],
            resources={"lake": lake},
        )
        assert result.success
        evaluations = result.get_asset_check_evaluations()
        assert len(evaluations) == 2
        assert all(evaluation.passed for evaluation in evaluations)


def test_silver_writes_parquet_a_plain_reader_can_open(
    lake: LakeResource, bronze_sources: dict[str, Path]
) -> None:
    """The layer has to be usable without our code, which is the point of Parquet."""
    _materialize_bronze(lake=lake, sources=bronze_sources)
    _materialize(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )

    boxes = pq.read_table(
        boxes_file(silver_dir=lake.paths.silver_dir("wider_face"), split="train")
    )
    assert boxes.schema == BOX_SCHEMA
    assert boxes.column("attr_invalid").to_pylist() == [False, True]
    # Exact floats, not the integers LightlyStudio stores.
    assert boxes.column("w").to_pylist() == [30.0, 5.0]

    images = pq.read_table(
        images_file(silver_dir=lake.paths.silver_dir("wider_face"), split="train")
    )
    assert images.schema == IMAGE_SCHEMA
    # b.jpg carries no box and still has a row, so a negative is not lost.
    assert images.column("file_name").to_pylist() == [
        "0--Parade/a.jpg",
        "1--Handshaking/b.jpg",
        "2--Demonstration/c.jpg",
    ]
