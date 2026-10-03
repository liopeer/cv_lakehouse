#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Build gold on the fixture lake."""

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lakehouse_cv.contract.gold_tables import (
    GOLD_BOX_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    gold_boxes_file,
    gold_images_file,
)
from lakehouse_cv.contract.silver_tables import boxes_file
from lakehouse_cv.defs import gold as gold_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.gold_build import build_gold_version, read_gold_manifest
from tests.lake_runs import DATASETS, materialize_assets, materialize_bronze_links

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


def _build(lake: CvLakeResource, built_at: datetime) -> int:
    build = build_gold_version(
        paths=lake.paths, specs=SPECS, code_version="test", built_at=built_at
    )
    return build.manifest.version


def _query(lake: CvLakeResource, table: str, select: str) -> list[tuple]:
    manifest = read_gold_manifest(lake.paths)
    assert manifest is not None
    files = lake.paths.gold_version_dir(manifest.version) / table / "*" / "*.parquet"
    with duckdb.connect() as connection:
        connection.execute("set TimeZone = 'UTC'")
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
    version_dir = silver_lake.paths.gold_version_dir(
        _build(lake=silver_lake, built_at=FIRST_BUILD)
    )
    manifest = read_gold_manifest(silver_lake.paths)
    assert manifest is not None
    for dataset in manifest.datasets:
        for split in dataset.splits:
            names = {"version_dir": version_dir, "dataset": dataset.dataset}
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


def test_a_rebuild_with_no_change_keeps_changed_at(silver_lake: CvLakeResource) -> None:
    assert _build(lake=silver_lake, built_at=FIRST_BUILD) == 1
    assert _build(lake=silver_lake, built_at=SECOND_BUILD) == 2

    for table in ("images", "boxes"):
        assert _query(lake=silver_lake, table=table, select="changed_at") == [
            (FIRST_BUILD,)
        ]


def test_a_changed_box_gets_the_time_of_its_build(silver_lake: CvLakeResource) -> None:
    _build(lake=silver_lake, built_at=FIRST_BUILD)
    path = boxes_file(
        silver_dir=silver_lake.paths.silver_dir("wider_face"), split="val"
    )
    table = pq.read_table(path)
    moved = table.set_column(
        table.schema.get_field_index("x"),
        table.schema.field("x"),
        pa.array([99.0], type=pa.float64()),
    )
    pq.write_table(table=moved, where=path)

    _build(lake=silver_lake, built_at=SECOND_BUILD)

    assert _query(
        lake=silver_lake, table="boxes", select="dataset, split, changed_at, count(*)"
    ) == [
        ("open_images", "test", FIRST_BUILD, 1),
        ("open_images", "train", FIRST_BUILD, 1),
        ("open_images", "validation", FIRST_BUILD, 3),
        ("pp4av", "fisheye", FIRST_BUILD, 1),
        ("pp4av", "test", FIRST_BUILD, 2),
        ("wider_face", "train", FIRST_BUILD, 1),
        ("wider_face", "val", SECOND_BUILD, 1),
    ]


def test_gold_keeps_the_current_version_and_the_one_before(
    silver_lake: CvLakeResource,
) -> None:
    for built_at in (FIRST_BUILD, SECOND_BUILD, SECOND_BUILD):
        _build(lake=silver_lake, built_at=built_at)
    versions = silver_lake.paths.gold_dir() / "versions"
    assert sorted(path.name for path in versions.iterdir()) == ["2", "3"]


def test_gold_leaves_out_a_dataset_with_no_silver(
    lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(lake=lake, assets=[silver_defs.build_silver_asset("pp4av")])

    build = build_gold_version(
        paths=lake.paths, specs=SPECS, code_version="test", built_at=FIRST_BUILD
    )

    assert [dataset.dataset for dataset in build.manifest.datasets] == ["pp4av"]
    assert set(build.skipped_datasets) == {"open_images", "wider_face"}
    assert (build.num_images, build.num_boxes) == (2, 3)


def test_gold_fails_with_no_silver_at_all(lake: CvLakeResource) -> None:
    with pytest.raises(expected_exception=RuntimeError, match="No silver"):
        build_gold_version(
            paths=lake.paths, specs=SPECS, code_version="test", built_at=FIRST_BUILD
        )
