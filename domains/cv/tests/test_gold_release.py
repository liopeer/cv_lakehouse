#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Freeze the val and test rows of gold, and read them back through the API."""

from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from lakehouse_cv.contract.manifests import RELEASE_MANIFEST
from lakehouse_cv.contract.silver_tables import boxes_file
from lakehouse_cv.defs import gold as gold_defs
from lakehouse_cv.defs import gold_releases as release_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.gold_api.app import create_app
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.gold_build import build_gold
from lakehouse_cv.transforms.gold_release import (
    IdenticalReleaseError,
    find_changed_release_files,
    list_release_numbers,
    read_release_manifest,
    write_eval_release,
)
from tests.lake_runs import (
    DATASETS,
    find_silver_files,
    materialize_assets,
    materialize_bronze_links,
)

SPECS = [source.spec for source in SOURCE_BY_NAME.values()]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def gold_lake(lake: CvLakeResource, bronze_sources: dict[str, Path]) -> CvLakeResource:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    materialize_assets(
        lake=lake, assets=[silver_defs.build_silver_asset(name) for name in DATASETS]
    )
    _build_gold(lake)
    return lake


def _build_gold(lake: CvLakeResource) -> None:
    build_gold(
        store=lake.store,
        paths=lake.paths,
        specs=SPECS,
        code_version="test",
        built_at=NOW,
    )


def _move_the_wider_face_val_box(lake: CvLakeResource) -> None:
    """Change one eval box in silver, as a correction does, and rebuild gold."""
    path = boxes_file(
        build_dir=find_silver_files(lake=lake, name="wider_face"), split="val"
    )
    table = pq.read_table(path)
    moved = table.set_column(
        table.schema.get_field_index("x"),
        table.schema.field("x"),
        pa.array([99.0], type=pa.float64()),
    )
    pq.write_table(table=moved, where=path)
    _build_gold(lake)


def _read_val_x(client: TestClient, release: str) -> float:
    response = client.get(
        url="/v1/boxes",
        params={"dataset": "wider_face", "role": "val", "release": release},
    )
    assert response.status_code == 200, response.text
    (row,) = response.json()["rows"]
    return row["x"]


def test_a_release_holds_the_val_and_test_files_and_no_train_file(
    gold_lake: CvLakeResource,
) -> None:
    manifest = write_eval_release(paths=gold_lake.paths, created_at=NOW)

    assert manifest.release == 1
    assert sorted(pinned.path for pinned in manifest.files) == [
        "boxes/open_images/test.parquet",
        "boxes/open_images/validation.parquet",
        "boxes/pp4av/fisheye.parquet",
        "boxes/pp4av/test.parquet",
        "boxes/wider_face/val.parquet",
        "images/open_images/test.parquet",
        "images/open_images/validation.parquet",
        "images/pp4av/fisheye.parquet",
        "images/pp4av/test.parquet",
        "images/wider_face/val.parquet",
    ]
    release_dir = gold_lake.paths.gold_release_dir(1)
    assert all((release_dir / pinned.path).is_file() for pinned in manifest.files)
    assert read_release_manifest(paths=gold_lake.paths, release=1) == manifest


def test_a_gold_rebuild_leaves_a_release_as_it_was(gold_lake: CvLakeResource) -> None:
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    release_file = gold_lake.paths.gold_release_dir(1) / "boxes/wider_face/val.parquet"
    before = release_file.read_bytes()

    _move_the_wider_face_val_box(gold_lake)
    # Gold keeps two builds, so a third build removes the one the release came from.
    _build_gold(gold_lake)

    assert release_file.read_bytes() == before
    assert find_changed_release_files(gold_lake.paths) == []


def test_a_release_with_the_rows_of_the_last_one_is_refused(
    gold_lake: CvLakeResource,
) -> None:
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    _build_gold(gold_lake)

    with pytest.raises(expected_exception=IdenticalReleaseError, match="release 1"):
        write_eval_release(paths=gold_lake.paths, created_at=NOW)
    assert list_release_numbers(gold_lake.paths) == [1]


def test_a_change_to_an_eval_row_gives_the_next_release(
    gold_lake: CvLakeResource,
) -> None:
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    _move_the_wider_face_val_box(gold_lake)

    manifest = write_eval_release(paths=gold_lake.paths, created_at=NOW)

    assert manifest.release == 2
    assert list_release_numbers(gold_lake.paths) == [1, 2]


def test_a_staging_directory_with_a_manifest_is_no_release(
    gold_lake: CvLakeResource,
) -> None:
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    release_dir = gold_lake.paths.gold_release_dir(1)
    staging_dir = release_dir.with_name(f".{release_dir.name}.staging")
    staging_dir.mkdir()
    (staging_dir / RELEASE_MANIFEST).write_text("{}")

    assert list_release_numbers(gold_lake.paths) == [1]


def test_a_release_file_that_changed_is_reported(gold_lake: CvLakeResource) -> None:
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    path = gold_lake.paths.gold_release_dir(1) / "boxes/pp4av/test.parquet"
    path.chmod(0o644)
    path.write_bytes(b"not the pinned bytes")

    assert find_changed_release_files(gold_lake.paths) == [
        "0001/boxes/pp4av/test.parquet"
    ]


def test_the_asset_publishes_a_release_and_refuses_an_equal_one(
    gold_lake: CvLakeResource,
) -> None:
    assets = [
        gold_defs.build_gold_asset(),
        release_defs.build_eval_release_asset(),
        *release_defs.build_eval_release_checks(),
    ]
    result = materialize_assets(lake=gold_lake, assets=assets)

    (check,) = result.get_asset_check_evaluations()
    assert check.passed
    assert list_release_numbers(gold_lake.paths) == [1]
    again = dg.materialize(
        assets=assets, resources={"lake": gold_lake}, raise_on_error=False
    )
    assert not again.success
    assert list_release_numbers(gold_lake.paths) == [1]


def test_the_api_serves_the_frozen_rows_of_a_release(gold_lake: CvLakeResource) -> None:
    client = TestClient(create_app(gold_lake.store))
    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    _move_the_wider_face_val_box(gold_lake)

    assert _read_val_x(client=client, release="1") == 5.0
    assert _read_val_x(client=client, release="draft") == 99.0

    write_eval_release(paths=gold_lake.paths, created_at=NOW)
    assert _read_val_x(client=client, release="1") == 5.0
    assert _read_val_x(client=client, release="2") == 99.0
    releases = client.get("/v1/releases").json()
    assert [release["release"] for release in releases] == [1, 2]
    assert client.get("/v1/releases/2").json()["gold_code_version"] == "test"
    assert client.get("/v1/releases/7").status_code == 404


def test_a_request_for_eval_rows_must_name_a_release(gold_lake: CvLakeResource) -> None:
    client = TestClient(create_app(gold_lake.store))

    assert client.get("/v1/boxes").status_code == 400
    assert client.get(url="/v1/boxes", params={"role": "val"}).status_code == 400
    assert client.get(url="/v1/boxes", params={"release": "7"}).status_code == 404
    assert client.get(url="/v1/boxes", params={"release": "newest"}).status_code == 422
    # Train rows are those of the current gold, so they need no release.
    train = client.get(url="/v1/boxes", params={"role": "train"})
    assert train.status_code == 200
    assert len(train.json()["rows"]) == 2


def test_a_release_joins_the_train_rows_of_the_current_gold(
    gold_lake: CvLakeResource,
) -> None:
    client = TestClient(create_app(gold_lake.store))
    write_eval_release(paths=gold_lake.paths, created_at=NOW)

    rows = client.get(url="/v1/boxes", params={"release": "1"}).json()["rows"]

    assert len(rows) == 10
    assert {row["role"] for row in rows} == {"train", "val", "test"}


def test_a_release_holds_no_vector(gold_lake: CvLakeResource) -> None:
    client = TestClient(create_app(gold_lake.store))
    write_eval_release(paths=gold_lake.paths, created_at=NOW)

    assert client.get(url="/v1/embeddings", params={"release": "1"}).status_code == 400
