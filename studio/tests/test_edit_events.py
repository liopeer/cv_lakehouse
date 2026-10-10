#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Log the edits of a curator, and publish them as event files."""

import hashlib
import io
from datetime import UTC, datetime

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from cv_lakehouse_studio.edit_events import (
    EVENT_SCHEMA,
    list_event_files,
    publish_event_file_if_new,
)
from cv_lakehouse_studio.export_api import create_export_app
from cv_lakehouse_studio.sync_loop import sync_every_dataset
from tests import curator
from tests.fakes import BOX_1, BOX_2, DATASET, IMAGE_B, FakeGoldClient


@pytest.fixture
def synced(database_url: str) -> FakeGoldClient:
    client = FakeGoldClient()
    sync_every_dataset(client=client, image_base="/lake")
    return client


def _publish(database_url: str) -> list[dict] | None:
    """Publish an event file, and return its rows without the log entries."""
    with psycopg.connect(database_url) as connection:
        event_file = publish_event_file_if_new(connection=connection, dataset=DATASET)
        if event_file is None:
            return None
        row = connection.execute(
            "select content from lakehouse_sync.event_file where event_file_id = %s",
            (event_file.event_file_id,),
        ).fetchone()
    assert row is not None
    table = pq.read_table(io.BytesIO(bytes(row[0])))
    assert table.schema == EVENT_SCHEMA
    return [
        {key: value for key, value in event.items() if key != "log_sequence"}
        for event in table.to_pylist()
    ]


def _event(box_id: str, **fields) -> dict:
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": "a.jpg",
        "box_id": box_id,
        "is_deleted": False,
        "label_name": "face",
        "x": 10.0,
        "y": 21.0,
        "w": 30.0,
        "h": 40.0,
        **fields,
    }


def _tombstone(box_id: str) -> dict:
    return _event(
        box_id, is_deleted=True, label_name=None, x=None, y=None, w=None, h=None
    )


@pytest.mark.usefixtures("synced")
def test_an_untouched_dataset_has_no_event_file(database_url: str) -> None:
    assert _publish(database_url) is None


@pytest.mark.usefixtures("synced")
def test_a_relabelled_box_is_an_event_with_its_whole_box(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    assert _publish(database_url) == [_event(BOX_1, label_name="other")]


@pytest.mark.usefixtures("synced")
def test_a_moved_box_is_an_event_with_its_whole_box(database_url: str) -> None:
    curator.move_box(box_id=BOX_1, x=11, y=21, width=30, height=40)
    assert _publish(database_url) == [_event(BOX_1, x=11.0)]


@pytest.mark.usefixtures("synced")
def test_a_deleted_box_is_a_tombstone(database_url: str) -> None:
    curator.delete_box(BOX_2)
    assert _publish(database_url) == [_tombstone(BOX_2)]


@pytest.mark.usefixtures("synced")
def test_a_drawn_box_is_an_event_under_its_own_id(database_url: str) -> None:
    drawn = curator.draw_box(
        image_id=IMAGE_B, label_name="face", x=1, y=2, width=3, height=4
    )
    assert _publish(database_url) == [
        _event(
            drawn,
            split="validation",
            file_name="b.jpg",
            x=1.0,
            y=2.0,
            w=3.0,
            h=4.0,
        )
    ]


@pytest.mark.usefixtures("synced")
def test_a_box_edited_twice_is_one_event_of_the_box_as_it_is(database_url: str) -> None:
    curator.move_box(box_id=BOX_1, x=99, y=21, width=30, height=40)
    curator.move_box(box_id=BOX_1, x=10, y=21, width=30, height=40)
    assert _publish(database_url) == [_event(BOX_1)]


@pytest.mark.usefixtures("synced")
def test_an_event_file_is_published_once_per_edit(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    assert _publish(database_url) is not None
    assert _publish(database_url) is None

    curator.delete_box(BOX_2)
    assert _publish(database_url) == [_tombstone(BOX_2)]

    with psycopg.connect(database_url) as connection:
        first, second = list_event_files(connection=connection, dataset=DATASET)
    assert second.parent_event_file_id == first.event_file_id
    assert second.after_log_sequence == first.last_log_sequence
    assert second.last_log_sequence > first.last_log_sequence


def test_the_sync_publishes_no_event(database_url: str) -> None:
    client = FakeGoldClient()
    sync_every_dataset(client=client, image_base="/lake")
    del client.boxes[1]
    sync_every_dataset(client=client, image_base="/lake")

    assert _publish(database_url) is None


def test_the_first_event_file_takes_over_the_latest_snapshot(
    database_url: str, synced: FakeGoldClient
) -> None:
    """A correction made before the log is in the first event file."""
    curator.delete_box(BOX_2)
    with psycopg.connect(database_url) as connection:
        # The sync was the one to log nothing so far. Drop the delete from the log, as
        # a database before the triggers holds no entry.
        connection.execute("delete from lakehouse_sync.edit_log")
        connection.execute(
            """
            insert into lakehouse_sync.snapshot
                (dataset, sequence, snapshot_id, parent_snapshot_id, created_at,
                 row_count, size, sha256, content)
            values (%s, 1, '000001-old', null, %s, 1, 0, '', %s)
            """,
            (DATASET, datetime(2026, 1, 1, tzinfo=UTC), _serialize_snapshot([BOX_2])),
        )
    curator.relabel_box(box_id=BOX_1, label_name="other")

    assert _publish(database_url) == [
        _tombstone(BOX_2),
        _event(BOX_1, label_name="other"),
    ]


def _serialize_snapshot(box_ids: list[str]) -> bytes:
    table = pa.table(
        {
            "dataset": [DATASET] * len(box_ids),
            "split": ["train"] * len(box_ids),
            "file_name": ["a.jpg"] * len(box_ids),
            "box_id": box_ids,
            "action": ["delete"] * len(box_ids),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table=table, where=sink)
    return sink.getvalue().to_pybytes()


@pytest.mark.usefixtures("synced")
def test_the_export_lists_and_serves_what_it_pinned(database_url: str) -> None:
    curator.relabel_box(box_id=BOX_1, label_name="other")
    client = TestClient(create_export_app(database_url=database_url))

    # The listing itself publishes the new edits.
    listing = client.get(f"/v1/datasets/{DATASET}/events").json()
    (listed,) = listing["event_files"]
    content = client.get(f"/{listed['path']}").content

    assert listing["chain_id"]
    assert (listed["sequence"], listed["parent_event_file_id"]) == (1, None)
    assert listed["after_log_sequence"] == 0
    assert (len(content), hashlib.sha256(content).hexdigest()) == (
        listed["size"],
        listed["sha256"],
    )
    table = pq.read_table(io.BytesIO(content))
    assert table.column("label_name").to_pylist() == ["other"]
    # A second listing finds no new edit, so it publishes nothing.
    assert len(client.get(f"/v1/datasets/{DATASET}/events").json()["event_files"]) == 1
    assert (
        client.get(f"/v1/datasets/{DATASET}/events/nope/events.parquet").status_code
        == 404
    )


def test_the_export_answers_503_before_the_first_sync(database_url: str) -> None:
    client = TestClient(create_export_app(database_url=database_url))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get(f"/v1/datasets/{DATASET}/events").status_code == 503


def test_the_edits_of_a_split_dataset_come_under_the_gold_dataset(
    database_url: str,
) -> None:
    sync_every_dataset(client=FakeGoldClient(), image_base="/lake", max_images=1)
    curator.relabel_box(box_id=BOX_1, label_name="other")
    assert _publish(database_url) == [_event(BOX_1, label_name="other")]
