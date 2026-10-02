#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Find the corrections of a curator, publish them, and take them back from gold."""

import hashlib
import io
import threading

import psycopg
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from cv_lakehouse_studio.corrections import (
    CORRECTION_SCHEMA,
    detect_corrections,
    list_snapshots,
    publish_snapshot_if_changed,
)
from cv_lakehouse_studio.export_api import create_export_app
from cv_lakehouse_studio.sync_loop import sync_every_dataset
from tests import curator
from tests.fakes import BOX_1, BOX_2, DATASET, IMAGE_B, FakeGoldClient, make_box


@pytest.fixture
def synced(database_url: str) -> FakeGoldClient:
    client = FakeGoldClient()
    sync_every_dataset(client=client, image_base="/lake")
    return client


def _detect(database_url: str) -> list[dict]:
    with psycopg.connect(database_url) as connection:
        table = detect_corrections(connection=connection, dataset=DATASET)
    assert table.schema == CORRECTION_SCHEMA
    return table.to_pylist()


def _publish(database_url: str) -> str | None:
    with psycopg.connect(database_url) as connection:
        snapshot = publish_snapshot_if_changed(connection=connection, dataset=DATASET)
    return None if snapshot is None else snapshot.snapshot_id


def _correction(box_id: str, action: str, **fields) -> dict:
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": "a.jpg",
        "box_id": box_id,
        "action": action,
        "label_name": None,
        "x": None,
        "y": None,
        "w": None,
        "h": None,
        **fields,
    }


@pytest.mark.usefixtures("synced")
def test_an_untouched_dataset_has_no_correction(database_url: str) -> None:
    assert _detect(database_url) == []
    assert _publish(database_url) is None


@pytest.mark.usefixtures("synced")
def test_a_relabelled_box_reports_the_label_and_no_geometry(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    assert _detect(database_url) == [
        _correction(box_id=BOX_1, action="update", label_name="other")
    ]


@pytest.mark.usefixtures("synced")
def test_a_moved_box_reports_the_geometry_and_no_label(database_url: str) -> None:
    curator.move_box(box_id=BOX_1, x=11, y=21, width=30, height=40)
    assert _detect(database_url) == [
        _correction(box_id=BOX_1, action="update", x=11.0, y=21.0, w=30.0, h=40.0)
    ]


@pytest.mark.usefixtures("synced")
def test_a_deleted_box_reports_a_delete(database_url: str) -> None:
    curator.delete_box(BOX_2)
    assert _detect(database_url) == [_correction(box_id=BOX_2, action="delete")]


@pytest.mark.usefixtures("synced")
def test_a_drawn_box_reports_an_add_under_its_own_id(database_url: str) -> None:
    drawn = curator.draw_box(
        image_id=IMAGE_B, label_name="face", x=1, y=2, width=3, height=4
    )
    assert _detect(database_url) == [
        _correction(
            box_id=drawn,
            action="add",
            split="validation",
            file_name="b.jpg",
            label_name="face",
            x=1.0,
            y=2.0,
            w=3.0,
            h=4.0,
        )
    ]


@pytest.mark.usefixtures("synced")
def test_a_box_that_a_curator_put_back_reports_nothing(database_url: str) -> None:
    curator.move_box(box_id=BOX_1, x=99, y=21, width=30, height=40)
    curator.move_box(box_id=BOX_1, x=10, y=21, width=30, height=40)
    assert _detect(database_url) == []


@pytest.mark.usefixtures("synced")
def test_a_snapshot_is_published_once_per_change(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    first = _publish(database_url)
    assert first is not None and first.startswith("000001-")
    assert _publish(database_url) is None

    curator.delete_box(BOX_2)
    second = _publish(database_url)

    assert second is not None and second.startswith("000002-")
    with psycopg.connect(database_url) as connection:
        snapshots = list_snapshots(connection=connection, dataset=DATASET)
    assert [snapshot.parent_snapshot_id for snapshot in snapshots] == [None, first]
    # A snapshot holds every live correction, so the second one holds both.
    assert [snapshot.row_count for snapshot in snapshots] == [1, 2]


@pytest.mark.usefixtures("synced")
def test_corrections_that_a_curator_took_back_give_an_empty_snapshot(
    database_url: str,
) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    _publish(database_url)
    curator.relabel_box(box_id=BOX_1, label_name="face")

    assert _publish(database_url) is not None
    with psycopg.connect(database_url) as connection:
        snapshots = list_snapshots(connection=connection, dataset=DATASET)
    assert [snapshot.row_count for snapshot in snapshots] == [1, 0]


@pytest.mark.usefixtures("synced")
def test_the_export_lists_and_serves_what_it_pinned(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    client = TestClient(
        create_export_app(database_url=database_url, sync_lock=threading.Lock())
    )

    # The listing itself looks for new corrections.
    (listed,) = client.get(f"/v1/datasets/{DATASET}/snapshots").json()
    content = client.get(f"/{listed['path']}").content

    assert (listed["sequence"], listed["parent_snapshot_id"]) == (1, None)
    assert (len(content), hashlib.sha256(content).hexdigest()) == (
        listed["size"],
        listed["sha256"],
    )
    table = pq.read_table(io.BytesIO(content))
    assert table.schema == CORRECTION_SCHEMA
    assert table.column("label_name").to_pylist() == ["other"]
    # A second listing finds no change, so it publishes nothing.
    assert len(client.get(f"/v1/datasets/{DATASET}/snapshots").json()) == 1
    assert (
        client.get(
            f"/v1/datasets/{DATASET}/snapshots/nope/corrections.parquet"
        ).status_code
        == 404
    )


def test_the_export_answers_503_before_the_first_sync(database_url: str) -> None:
    client = TestClient(
        create_export_app(database_url=database_url, sync_lock=threading.Lock())
    )
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get(f"/v1/datasets/{DATASET}/snapshots").status_code == 503


def test_the_corrections_stay_live_after_they_come_back_in_gold(
    database_url: str, synced: FakeGoldClient
) -> None:
    """The lake applies a snapshot to the untouched source on every silver build.

    So a correction must stay in every later snapshot, also once gold holds it.
    """
    curator.move_box(box_id=BOX_1, x=11, y=21, width=30, height=40)
    curator.delete_box(BOX_2)
    drawn = curator.draw_box(
        image_id=IMAGE_B, label_name="face", x=1, y=2, width=3, height=4
    )
    before = _detect(database_url)
    assert _publish(database_url) is not None

    # Silver applied the snapshot, and gold was rebuilt from it.
    synced.boxes = [
        make_box(box_id=BOX_1, class_name="face", x=11.0, y=21.0, w=30.0, h=40.0),
        {
            **make_box(
                box_id=drawn,
                class_name="face",
                x=1.0,
                y=2.0,
                w=3.0,
                h=4.0,
                origin="studio",
            ),
            "image_id": IMAGE_B,
        },
    ]
    synced.publish_new_version()
    (report,) = sync_every_dataset(client=synced, image_base="/lake")

    assert (report.num_new_boxes, report.num_updated_boxes) == (0, 0)
    assert report.num_removed_boxes == 0
    assert _detect(database_url) == before
    assert _publish(database_url) is None

    # The curator deletes the drawn box. The sync must not bring it back from gold.
    curator.delete_box(drawn)
    synced.publish_new_version()
    sync_every_dataset(client=synced, image_base="/lake")
    assert {row["box_id"] for row in _detect(database_url)} == {BOX_1, BOX_2}
