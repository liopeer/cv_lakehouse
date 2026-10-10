#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Build gold on the fixture lake."""

from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import duckdb
import pyarrow.parquet as pq
import pytest

from lakehouse_cv.contract.gold_tables import (
    GOLD_BOX_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    gold_boxes_file,
    gold_images_file,
)
from lakehouse_cv.defs import gold as gold_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.gold_build import build_gold, read_gold_manifest
from lakehouse_cv.transforms.layer_builds import read_silver_manifest
from tests.lake_runs import (
    DATASETS,
    materialize_assets,
    materialize_bronze_links,
    read_silver_build_ids,
    run_assets,
)

SPECS = [source.spec for source in SOURCE_BY_NAME.values()]
FIRST_BUILD = datetime(2026, 1, 1, tzinfo=UTC)
SECOND_BUILD = datetime(2026, 1, 2, tzinfo=UTC)


@pytest.fixture
def silver_lake(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> CvLakeResource:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )
    return lake


def _build(lake: CvLakeResource, built_at: datetime, code_version: str = "test") -> str:
    build = build_gold(
        store=lake.store,
        paths=lake.paths,
        specs=SPECS,
        silver_build_ids=read_silver_build_ids(
            lake=lake, names=[spec.name for spec in SPECS]
        ),
        code_version=code_version,
        built_at=built_at,
    )
    return build.manifest.build_id


def _query(lake: CvLakeResource, table: str, select: str) -> list[tuple]:
    manifest = read_gold_manifest(lake.paths)
    assert manifest is not None
    files = lake.paths.gold_build_dir(manifest.build_id) / table / "*" / "*.parquet"
    with duckdb.connect() as connection:
        return connection.execute(
            f"select {select} from read_parquet('{files}') group by all order by all"
        ).fetchall()


def test_gold_unions_every_silver_dataset(silver_lake: CvLakeResource) -> None:
    result = materialize_assets(
        lake=silver_lake,
        assets=[gold_defs.build_gold_asset(), *gold_defs.build_gold_checks()],
    )
    assert all(check.passed for check in result.get_asset_check_evaluations())

    assert _query(lake=silver_lake, table="images", select="dataset, count(*)") == [
        ("open_images", 3),
        ("pp4av", 2),
        ("wider_face", 4),
    ]


def test_gold_writes_its_schema_to_every_file(silver_lake: CvLakeResource) -> None:
    gold_dir = silver_lake.paths.gold_build_dir(
        _build(lake=silver_lake, built_at=FIRST_BUILD)
    )
    manifest = read_gold_manifest(silver_lake.paths)
    assert manifest is not None
    for dataset in manifest.datasets:
        for split in dataset.splits:
            names = {"build_dir": gold_dir, "dataset": dataset.dataset}
            images = pq.read_schema(gold_images_file(**names, split=split.split))
            boxes = pq.read_schema(gold_boxes_file(**names, split=split.split))
            assert images == GOLD_IMAGE_SCHEMA
            assert boxes == GOLD_BOX_SCHEMA


def test_gold_gives_every_row_the_role_of_its_split(
    silver_lake: CvLakeResource,
) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    assert _query(lake=silver_lake, table="images", select="dataset, split, role") == [
        ("open_images", "test", "test"),
        ("open_images", "train", "train"),
        ("open_images", "validation", "val"),
        ("pp4av", "fisheye", "test"),
        ("pp4av", "test", "test"),
        ("wider_face", "train", "train"),
        ("wider_face", "val", "val"),
    ]


def test_gold_drops_every_flagged_box(silver_lake: CvLakeResource) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    counts = _query(lake=silver_lake, table="boxes", select="dataset, split, count(*)")
    assert counts == [
        # Silver holds five boxes here. The crowd box and the depiction drop out.
        ("open_images", "test", 1),
        ("open_images", "train", 1),
        ("open_images", "validation", 3),
        ("pp4av", "fisheye", 1),
        ("pp4av", "test", 2),
        # Silver holds two. The region the annotators rejected drops out.
        ("wider_face", "train", 1),
        ("wider_face", "val", 1),
    ]


def test_gold_names_every_pixel_relative_to_the_lake_root(
    silver_lake: CvLakeResource,
) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    paths = _query(lake=silver_lake, table="images", select="image_path")
    assert len(paths) == 9
    assert all((silver_lake.paths.root / path).is_file() for (path,) in paths)


def test_a_rebuild_from_the_same_silver_gives_the_same_rows(
    silver_lake: CvLakeResource,
) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    before = {
        table: _query(lake=silver_lake, table=table, select="*")
        for table in ("images", "boxes")
    }

    _build(lake=silver_lake, built_at=SECOND_BUILD)

    for table, rows in before.items():
        assert _query(lake=silver_lake, table=table, select="*") == rows


def test_gold_keeps_the_current_build_and_the_one_before(
    silver_lake: CvLakeResource,
) -> None:
    first = _build(lake=silver_lake, built_at=FIRST_BUILD, code_version="a")
    second = _build(lake=silver_lake, built_at=SECOND_BUILD, code_version="b")
    third = _build(lake=silver_lake, built_at=SECOND_BUILD, code_version="c")

    builds = silver_lake.paths.gold_dir() / "builds"
    assert {path.name for path in builds.iterdir()} == {second, third}
    assert first not in {second, third}


def test_a_build_of_the_same_silver_and_code_is_the_same_build(
    silver_lake: CvLakeResource,
) -> None:
    first = _build(lake=silver_lake, built_at=FIRST_BUILD)
    files = silver_lake.paths.gold_build_dir(first) / "boxes" / "pp4av" / "test.parquet"
    written = files.stat().st_mtime_ns

    assert _build(lake=silver_lake, built_at=SECOND_BUILD) == first
    assert files.stat().st_mtime_ns == written
    manifest = read_gold_manifest(silver_lake.paths)
    assert manifest is not None
    assert manifest.built_at == FIRST_BUILD


def test_gold_reads_the_silver_build_that_dagster_recorded(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    """A silver manifest that moved after the run is no input of gold."""
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("pp4av")])
    recorded = read_silver_manifest(paths=lake.paths, name="pp4av").build_id
    manifest_path = lake.paths.silver_dir("pp4av") / "_silver.json"
    manifest_path.write_text(manifest_path.read_text().replace(recorded, "elsewhere"))
    asset = gold_defs.build_gold_asset()

    result = run_assets(
        lake=lake,
        assets=[asset],
        run_config=dg.RunConfig(
            ops={asset.op.name: gold_defs.GoldConfig(datasets=["pp4av"])}
        ),
    )

    assert result.success
    gold = read_gold_manifest(lake.paths)
    assert gold is not None
    assert [dataset.silver_build_id for dataset in gold.datasets] == [recorded]
    (materialization,) = result.get_asset_materialization_events()
    tags = materialization.materialization.tags or {}
    assert tags["dagster/data_version"] == gold.build_id


def test_gold_records_the_silver_build_of_each_dataset(
    silver_lake: CvLakeResource,
) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    manifest = read_gold_manifest(silver_lake.paths)
    assert manifest is not None
    for dataset in manifest.datasets:
        silver = read_silver_manifest(paths=silver_lake.paths, name=dataset.dataset)
        assert dataset.silver_build_id == silver.build_id


def test_gold_fails_on_a_dataset_with_no_silver(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("pp4av")])

    result = run_assets(
        assets=[gold_defs.build_gold_asset()],
        lake=lake,
        raise_on_error=False,
    )

    assert not result.success
    assert read_gold_manifest(lake.paths) is None


def test_gold_joins_the_datasets_of_its_config(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("pp4av")])
    asset = gold_defs.build_gold_asset()

    result = run_assets(
        assets=[asset],
        lake=lake,
        run_config=dg.RunConfig(
            ops={asset.op.name: gold_defs.GoldConfig(datasets=["pp4av"])}
        ),
    )

    assert result.success
    manifest = read_gold_manifest(lake.paths)
    assert manifest is not None
    assert [dataset.dataset for dataset in manifest.datasets] == ["pp4av"]


def test_gold_rejects_a_dataset_that_is_not_registered(lake: CvLakeResource) -> None:
    asset = gold_defs.build_gold_asset()
    result = run_assets(
        assets=[asset],
        lake=lake,
        run_config=dg.RunConfig(
            ops={asset.op.name: gold_defs.GoldConfig(datasets=["no_such_dataset"])}
        ),
        raise_on_error=False,
    )
    assert not result.success
