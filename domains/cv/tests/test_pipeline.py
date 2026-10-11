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
from upath import UPath

from lakehouse_core.bronze_asset import build_bronze_asset
from lakehouse_core.bronze_manifest import BRONZE_MANIFEST
from lakehouse_core.manifest_files import read_manifest
from lakehouse_cv.contract.manifests import (
    SILVER_MANIFEST,
    CvBronzeManifest,
    SilverManifest,
)
from lakehouse_cv.contract.silver_tables import (
    BOX_SCHEMA,
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_SCHEMA,
    IMAGE_SCHEMA,
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
    images_file,
)
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs import silver_embeddings as embeddings_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.base import detect_materialized_splits
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from tests.fixtures import make_wider_face
from tests.lake_runs import (
    DATASETS,
    find_embedding_files,
    find_silver_files,
    materialize_assets,
    materialize_bronze_links,
    run_assets,
)


def test_bronze_links_instead_of_downloading(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    for name in DATASETS:
        bronze_dir = lake.paths.bronze_dir(name)
        manifest = read_manifest(
            path=bronze_dir / BRONZE_MANIFEST, model=CvBronzeManifest
        )
        assert manifest.path == str(bronze_sources[name])
        assert [path.name for path in bronze_dir.iterdir()] == [BRONZE_MANIFEST]


def test_bronze_reports_only_the_splits_on_disk(tmp_path: Path) -> None:
    bronze = make_wider_face(tmp_path)
    shutil.rmtree(str(bronze / "WIDER_val"))
    splits = detect_materialized_splits(
        source=SOURCE_BY_NAME["wider_face"], bronze_dir=bronze
    )
    assert splits == ("train",)


def test_bronze_rejects_a_link_to_an_incomplete_copy(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    """WIDER_test holds no label, and bronze needs it all the same."""
    shutil.rmtree(bronze_sources["wider_face"] / "WIDER_test")
    asset = build_bronze_asset(SOURCE_BY_NAME["wider_face"])
    result = run_assets(
        assets=[asset],
        lake=lake,
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
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    manifest = read_manifest(
        path=lake.paths.bronze_dir("pp4av") / BRONZE_MANIFEST, model=CvBronzeManifest
    )
    assert manifest.published_files == list(SOURCE_BY_NAME["pp4av"].published_files)


def test_silver_normalizes_every_dataset(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )

    manifest = read_manifest(
        path=lake.paths.silver_dir("pp4av") / SILVER_MANIFEST, model=SilverManifest
    )
    pp4av = find_silver_files(lake=lake, name="pp4av")
    assert set(manifest.splits) == {"test", "fisheye"}
    assert classes_of(build_dir=pp4av, split="test") == {"face": 1, "license_plate": 1}

    wider = find_silver_files(lake=lake, name="wider_face")
    assert num_rows(images_file(build_dir=wider, split="train")) == 3
    # Both boxes of a.jpg, one of them the region the annotators rejected. Only the all
    # zero line of c.jpg drops, which `test_sources_wider_face` asserts on the
    # normalizer, the only place that can still see a dropped row.
    assert num_rows(boxes_file(build_dir=wider, split="train")) == 2
    assert not read_manifest(
        path=lake.paths.silver_dir("wider_face") / SILVER_MANIFEST, model=SilverManifest
    ).commercial_use
    assert "attr_invalid" in attrs_of(build_dir=wider, split="train")
    assert "attr_is_group_of" not in attrs_of(build_dir=wider, split="train")

    # Bronze holds a cat. Silver keeps its box under other, for a consumer to judge.
    open_images = find_silver_files(lake=lake, name="open_images")
    # Three faces: one real, plus the crowd box and the depiction, both flagged.
    assert classes_of(build_dir=open_images, split="validation") == {
        "face": 3,
        "license_plate": 1,
        "other": 1,
    }
    assert attrs_of(build_dir=open_images, split="validation") == {
        "attr_is_group_of",
        "attr_is_depiction",
    }


def test_silver_writes_no_embedding(
    embedding_lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    """Silver means the same, whatever the environment."""
    materialize_bronze_links(lake=embedding_lake, sources=bronze_sources)
    materialize_assets(
        lake=embedding_lake, assets=[silver_defs.build_silver_asset("wider_face")]
    )

    names = {
        path.name
        for path in find_silver_files(lake=embedding_lake, name="wider_face").iterdir()
    }
    assert names == {"_build.json", "images", "boxes"}


def test_the_embeddings_fail_without_a_server(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("wider_face")])

    result = run_assets(
        assets=[embeddings_defs.build_embeddings_asset("wider_face")],
        lake=lake,
        raise_on_error=False,
    )

    assert not result.success
    assert not lake.paths.embeddings_dir("wider_face").exists()


def test_the_embeddings_cover_every_image_and_every_box(
    embedding_lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=embedding_lake, sources=bronze_sources)
    materialize_assets(
        lake=embedding_lake,
        assets=[
            silver_defs.build_silver_asset("wider_face"),
            embeddings_defs.build_embeddings_asset("wider_face"),
        ],
    )

    silver = find_silver_files(lake=embedding_lake, name="wider_face")
    vectors = find_embedding_files(lake=embedding_lake, name="wider_face")
    images = pq.read_table(embeddings_file(build_dir=vectors, split="train"))
    crops = pq.read_table(crop_embeddings_file(build_dir=vectors, split="train"))
    assert images.schema == EMBEDDING_SCHEMA
    assert crops.schema == CROP_EMBEDDING_SCHEMA
    # One vector per image row and one per box row, which is the join these files are
    # for. Nothing else records that the embedding pass covered the whole split.
    assert images.num_rows == num_rows(images_file(build_dir=silver, split="train"))
    assert crops.num_rows == num_rows(boxes_file(build_dir=silver, split="train"))
    # The keys of the boxes file, so a join needs nothing else, in the order of the id.
    boxes = pq.read_table(boxes_file(build_dir=silver, split="train"))
    pairs = zip(
        boxes.column("box_id").to_pylist(),
        boxes.column("file_name").to_pylist(),
        strict=True,
    )
    assert crops.select(["box_id", "file_name"]).to_pylist() == [
        {"box_id": box_id, "file_name": file_name}
        for box_id, file_name in sorted(pairs)
    ]


def classes_of(build_dir: UPath, split: str) -> dict[str, int]:
    """Count one split's boxes per class. The manifest no longer carries the tally."""
    names = (
        pq.read_table(boxes_file(build_dir=build_dir, split=split))
        .column("class_name")
        .to_pylist()
    )
    return dict(Counter(names))


def num_rows(path: UPath) -> int:
    return pq.read_metadata(path).num_rows


def attrs_of(build_dir: UPath, split: str) -> set[str]:
    """The `attr_*` columns this split fills. The manifest no longer names them."""
    table = pq.read_table(boxes_file(build_dir=build_dir, split=split))
    return {
        name
        for name in table.column_names
        if name.startswith("attr_") and table.column(name).null_count < table.num_rows
    }


def test_a_silver_run_with_an_invalid_box_keeps_the_last_build(
    lake: CvLakeResource,
    bronze_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("wider_face")])
    manifest_path = lake.paths.silver_dir("wider_face") / SILVER_MANIFEST
    before = manifest_path.read_text()
    # New rules, so a new build, and rules that reject every box.
    monkeypatch.setattr(target=silver_defs, name="SILVER_LOGIC_VERSION", value="next")
    monkeypatch.setattr(
        target=silver_defs, name="EDGE_TOLERANCE_PIXELS", value=-1_000.0
    )

    result = run_assets(
        assets=[silver_defs.build_silver_asset("wider_face")],
        lake=lake,
        raise_on_error=False,
    )

    assert not result.success
    assert manifest_path.read_text() == before
    builds = lake.paths.silver_dir("wider_face") / "builds"
    assert len(list(builds.iterdir())) == 1


def test_silver_keeps_the_current_build_and_the_one_before(
    lake: CvLakeResource,
    bronze_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    for version in ("a", "b", "c"):
        monkeypatch.setattr(
            target=silver_defs, name="SILVER_LOGIC_VERSION", value=version
        )
        materialize_assets(
            lake=lake, assets=[silver_defs.build_silver_asset("wider_face")]
        )

    builds = lake.paths.silver_dir("wider_face") / "builds"
    assert len(list(builds.iterdir())) == 2
    assert find_silver_files(lake=lake, name="wider_face").exists()


def test_a_silver_run_on_the_same_inputs_writes_nothing(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    first = materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset("wider_face")]
    )
    boxes = boxes_file(
        build_dir=find_silver_files(lake=lake, name="wider_face"), split="train"
    )
    written = boxes.stat().st_mtime_ns

    second = materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset("wider_face")]
    )

    assert boxes.stat().st_mtime_ns == written
    versions = [
        result.get_asset_materialization_events()[0].materialization.tags
        for result in (first, second)
    ]
    assert versions[0] == versions[1]
    (materialization,) = second.get_asset_materialization_events()
    assert materialization.materialization.metadata["is_reused"].value is True


def test_silver_checks_pass(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    for name in DATASETS:
        result = run_assets(
            assets=[
                silver_defs.build_silver_asset(name),
                *silver_defs.build_silver_checks(name),
            ],
            lake=lake,
        )
        assert result.success
        evaluations = result.get_asset_check_evaluations()
        assert len(evaluations) == 3
        assert all(evaluation.passed for evaluation in evaluations)


def test_silver_writes_parquet_a_plain_reader_can_open(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    """The layer has to be usable without our code, which is the point of Parquet."""
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )

    wider = find_silver_files(lake=lake, name="wider_face")
    boxes = pq.read_table(boxes_file(build_dir=wider, split="train"))
    assert boxes.schema == BOX_SCHEMA
    assert boxes.column("attr_invalid").to_pylist() == [False, True]
    # Exact floats, not the integers LightlyStudio stores.
    assert boxes.column("w").to_pylist() == [30.0, 5.0]

    images = pq.read_table(images_file(build_dir=wider, split="train"))
    assert images.schema == IMAGE_SCHEMA
    # b.jpg carries no box and still has a row, so a negative is not lost.
    assert images.column("file_name").to_pylist() == [
        "0--Parade/a.jpg",
        "1--Handshaking/b.jpg",
        "2--Demonstration/c.jpg",
    ]
