#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fetch event files into bronze from a fake export."""

import dagster as dg
import pyarrow.parquet as pq
import pytest

from lakehouse_cv.contract.correction_actions import EVENT_SCHEMA
from lakehouse_cv.defs import corrections as corrections_defs
from lakehouse_cv.defs.corrections import (
    build_corrections_asset,
    read_corrections_manifest,
)
from lakehouse_cv.defs.resources import CvLakeResource
from tests.export_fakes import EXPORT_URL, FakeExportServer
from tests.lake_runs import run_assets

DATASET = "wider_face"
MOVED = {
    "dataset": DATASET,
    "split": "train",
    "file_name": "0--Parade/a.jpg",
    "box_id": "00000000-0000-0000-0000-000000000001",
    "is_deleted": False,
    "label_name": "face",
    "x": 1.0,
    "y": 2.0,
    "w": 3.0,
    "h": 4.0,
}
DELETED = {
    "dataset": DATASET,
    "split": "train",
    "file_name": "0--Parade/a.jpg",
    "box_id": "00000000-0000-0000-0000-000000000002",
    "is_deleted": True,
}


@pytest.fixture
def export_server(monkeypatch: pytest.MonkeyPatch) -> FakeExportServer:
    server = FakeExportServer()
    monkeypatch.setattr(
        target=corrections_defs,
        name="open_export_client",
        value=lambda timeout: server.client(),
    )
    return server


@pytest.fixture
def export_lake(lake: CvLakeResource) -> CvLakeResource:
    return CvLakeResource(
        root=lake.root,
        download_workers=lake.download_workers,
        request_timeout_seconds=lake.request_timeout_seconds,
        studio_export_url=EXPORT_URL,
    )


def _materialize(
    lake: CvLakeResource, raise_on_error: bool = True
) -> dg.ExecuteInProcessResult:
    return run_assets(
        assets=[build_corrections_asset(DATASET)],
        lake=lake,
        raise_on_error=raise_on_error,
    )


def _read_data_version(result: dg.ExecuteInProcessResult) -> str | None:
    (materialization,) = result.get_asset_materialization_events()
    tags = materialization.materialization.tags or {}
    return tags.get("dagster/data_version")


def test_a_lake_with_no_export_url_fetches_nothing(lake: CvLakeResource) -> None:
    result = _materialize(lake)

    assert result.success
    assert read_corrections_manifest(paths=lake.paths, name=DATASET).event_files == []
    assert _read_data_version(result) == "none"


def _find_event_file(
    lake: CvLakeResource, server: FakeExportServer, event_file_id: str
):
    return (
        lake.paths.corrections_dir(DATASET)
        / "events"
        / server.chain_id
        / event_file_id
        / "events.parquet"
    )


def test_an_event_file_lands_in_bronze_as_published(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    event_file_id = export_server.publish(dataset=DATASET, rows=[MOVED, DELETED])

    result = _materialize(export_lake)

    (event_file,) = read_corrections_manifest(
        paths=export_lake.paths, name=DATASET
    ).event_files
    assert event_file.event_file_id == event_file_id
    assert (event_file.sequence, event_file.parent_event_file_id) == (1, None)
    assert (event_file.after_log_sequence, event_file.last_log_sequence) == (0, 2)
    assert event_file.chain_id == export_server.chain_id
    path = export_lake.paths.corrections_dir(DATASET) / event_file.file.path
    assert path.read_bytes() == next(iter(export_server.files.values()))
    assert pq.read_table(path).schema == EVENT_SCHEMA
    assert _read_data_version(result) == event_file_id


def test_a_later_run_fetches_only_the_new_event_file(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    first = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    first_file = _find_event_file(
        lake=export_lake, server=export_server, event_file_id=first
    )
    first_modified = first_file.stat().st_mtime_ns
    second = export_server.publish(dataset=DATASET, rows=[DELETED])

    result = _materialize(export_lake)

    manifest = read_corrections_manifest(paths=export_lake.paths, name=DATASET)
    assert [event_file.event_file_id for event_file in manifest.event_files] == [
        first,
        second,
    ]
    assert manifest.event_files[1].parent_event_file_id == first
    assert first_file.stat().st_mtime_ns == first_modified
    assert _read_data_version(result) == second


def test_a_run_with_no_new_event_file_keeps_the_data_version(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    event_file_id = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)

    assert _read_data_version(_materialize(export_lake)) == event_file_id


def test_a_stored_event_file_that_the_export_changed_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.listings.clear()
    export_server.publish(dataset=DATASET, rows=[DELETED])

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_a_stored_event_file_that_changed_on_disk_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    event_file_id = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    path = _find_event_file(
        lake=export_lake, server=export_server, event_file_id=event_file_id
    )
    path.write_bytes(b"not the pinned bytes")

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_an_export_that_lost_its_log_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    """The same database lists fewer files, and silver must not lose an edit."""
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.listings.clear()

    assert not _materialize(lake=export_lake, raise_on_error=False).success
    assert (
        len(
            read_corrections_manifest(paths=export_lake.paths, name=DATASET).event_files
        )
        == 1
    )


def test_an_event_file_with_the_wrong_parent_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.publish(dataset=DATASET, rows=[DELETED])
    export_server.listings[DATASET][1]["parent_event_file_id"] = "000001-other"

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_an_event_file_with_a_gap_in_the_log_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.publish(dataset=DATASET, rows=[DELETED])
    export_server.listings[DATASET][1]["after_log_sequence"] = 5

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_a_new_database_starts_a_chain_and_bronze_keeps_the_old_one(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    old = export_server.publish(dataset=DATASET, rows=[MOVED])
    old_chain = export_server.chain_id
    _materialize(export_lake)
    export_server.start_new_chain()

    assert _materialize(export_lake).success
    new = export_server.publish(dataset=DATASET, rows=[DELETED])
    _materialize(export_lake)

    manifest = read_corrections_manifest(paths=export_lake.paths, name=DATASET)
    assert [
        (event_file.chain_id, event_file.event_file_id)
        for event_file in manifest.event_files
    ] == [(old_chain, old), (export_server.chain_id, new)]


def test_a_download_with_the_wrong_bytes_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    (path,) = export_server.files
    export_server.files[path] = b"x" * len(export_server.files[path])

    assert not _materialize(lake=export_lake, raise_on_error=False).success
    assert (
        read_corrections_manifest(paths=export_lake.paths, name=DATASET).event_files
        == []
    )
