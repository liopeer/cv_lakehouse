#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fetch correction snapshots into bronze from a fake export."""

import dagster as dg
import pyarrow.parquet as pq
import pytest

from lakehouse_cv.contract.correction_actions import CORRECTION_SCHEMA, CorrectionAction
from lakehouse_cv.defs import corrections as corrections_defs
from lakehouse_cv.defs.corrections import (
    build_corrections_asset,
    read_corrections_manifest,
)
from lakehouse_cv.defs.resources import CvLakeResource
from tests.export_fakes import EXPORT_URL, FakeExportServer

DATASET = "wider_face"
MOVED = {
    "dataset": DATASET,
    "split": "train",
    "file_name": "0--Parade/a.jpg",
    "box_id": "00000000-0000-0000-0000-000000000001",
    "action": CorrectionAction.UPDATE,
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
    "action": CorrectionAction.DELETE,
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
    return dg.materialize(
        assets=[build_corrections_asset(DATASET)],
        resources={"lake": lake},
        raise_on_error=raise_on_error,
    )


def _read_data_version(result: dg.ExecuteInProcessResult) -> str | None:
    (materialization,) = result.get_asset_materialization_events()
    tags = materialization.materialization.tags or {}
    return tags.get("dagster/data_version")


def test_a_lake_with_no_export_url_fetches_nothing(lake: CvLakeResource) -> None:
    result = _materialize(lake)

    assert result.success
    assert read_corrections_manifest(paths=lake.paths, name=DATASET).snapshots == []
    assert _read_data_version(result) == "none"


def test_a_snapshot_lands_in_bronze_as_published(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    snapshot_id = export_server.publish(dataset=DATASET, rows=[MOVED, DELETED])

    result = _materialize(export_lake)

    (snapshot,) = read_corrections_manifest(
        paths=export_lake.paths, name=DATASET
    ).snapshots
    assert snapshot.snapshot_id == snapshot_id
    assert (snapshot.sequence, snapshot.parent_snapshot_id) == (1, None)
    path = export_lake.paths.corrections_dir(DATASET) / snapshot.file.path
    assert path.read_bytes() == next(iter(export_server.files.values()))
    assert pq.read_table(path).schema == CORRECTION_SCHEMA
    assert _read_data_version(result) == snapshot_id


def test_a_later_run_fetches_only_the_new_snapshot(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    first = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    first_file = (
        export_lake.paths.corrections_dir(DATASET)
        / "snapshots"
        / first
        / "corrections.parquet"
    )
    first_modified = first_file.stat().st_mtime_ns
    second = export_server.publish(dataset=DATASET, rows=[MOVED, DELETED])

    result = _materialize(export_lake)

    manifest = read_corrections_manifest(paths=export_lake.paths, name=DATASET)
    assert [snapshot.snapshot_id for snapshot in manifest.snapshots] == [first, second]
    assert manifest.snapshots[1].parent_snapshot_id == first
    assert first_file.stat().st_mtime_ns == first_modified
    assert _read_data_version(result) == second


def test_a_run_with_no_new_snapshot_keeps_the_data_version(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    snapshot_id = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)

    assert _read_data_version(_materialize(export_lake)) == snapshot_id


def test_a_stored_snapshot_that_the_export_changed_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.listings.clear()
    export_server.publish(dataset=DATASET, rows=[DELETED])

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_a_stored_snapshot_that_changed_on_disk_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    snapshot_id = export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    path = (
        export_lake.paths.corrections_dir(DATASET)
        / "snapshots"
        / snapshot_id
        / "corrections.parquet"
    )
    path.write_bytes(b"not the pinned bytes")

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_an_export_that_lost_its_history_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    """A new LightlyStudio database lists nothing, and silver must not lose a fix."""
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.listings.clear()

    assert not _materialize(lake=export_lake, raise_on_error=False).success
    assert (
        len(read_corrections_manifest(paths=export_lake.paths, name=DATASET).snapshots)
        == 1
    )


def test_a_snapshot_with_the_wrong_parent_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    _materialize(export_lake)
    export_server.publish(dataset=DATASET, rows=[MOVED, DELETED])
    export_server.listings[DATASET][1]["parent_snapshot_id"] = "000001-other"

    assert not _materialize(lake=export_lake, raise_on_error=False).success


def test_a_download_with_the_wrong_bytes_fails_the_run(
    export_lake: CvLakeResource, export_server: FakeExportServer
) -> None:
    export_server.publish(dataset=DATASET, rows=[MOVED])
    (path,) = export_server.files
    export_server.files[path] = b"x" * len(export_server.files[path])

    assert not _materialize(lake=export_lake, raise_on_error=False).success
    assert (
        read_corrections_manifest(paths=export_lake.paths, name=DATASET).snapshots == []
    )
