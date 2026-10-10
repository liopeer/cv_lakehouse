#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Publish the logged edits of the curators as immutable event files.

The triggers of `sync_state` log every edit of a curator. An event file holds the
boxes of the log entries after the last file of the dataset: each box as LightlyStudio
holds it now, or a tombstone when it is gone. A file names the range of the log that it
covers, and the file before it. A stored file never changes.

The first file of a dataset also holds the boxes of its latest snapshot of ADR 0009, so
no correction made before the log is lost.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from cv_lakehouse_studio.sync_state import SYNC_SCHEMA

EVENTS_FILE = "events.parquet"

# The contract with `lakehouse_cv.contract.correction_actions`. A tombstone has no label
# and no box.
EVENT_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="log_sequence", type=pa.int64(), nullable=False),
        pa.field(name="is_deleted", type=pa.bool_(), nullable=False),
        pa.field(name="label_name", type=pa.string()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
    ]
)


@dataclass(frozen=True)
class EventFile:
    event_file_id: str
    sequence: int
    parent_event_file_id: str | None
    after_log_sequence: int
    last_log_sequence: int
    created_at: datetime
    row_count: int
    size: int
    sha256: str


def read_chain_id(connection: psycopg.Connection) -> str:
    row = connection.execute(
        f"select chain_id from {SYNC_SCHEMA}.event_chain"
    ).fetchone()
    if row is None:
        raise RuntimeError("The sync has not created the event chain yet.")
    return row[0]


def publish_event_file_if_new(
    connection: psycopg.Connection, dataset: str
) -> EventFile | None:
    """Store a new event file when the log holds an edit after the last file."""
    with connection.transaction():
        # The lock waits for every transaction that writes the log, and holds the next
        # ones until this one ends. Every entry up to the last one has then committed,
        # so no later file skips an entry that commits late.
        connection.execute(f"lock table {SYNC_SCHEMA}.edit_log in exclusive mode")
        files = list_event_files(connection=connection, dataset=dataset)
        latest = files[-1] if files else None
        after = 0 if latest is None else latest.last_log_sequence
        last = _read_last_log_sequence(connection)
        events = read_events(
            connection=connection,
            dataset=dataset,
            after_log_sequence=after,
            last_log_sequence=last,
            snapshot_boxes=(
                _read_snapshot_boxes(connection=connection, dataset=dataset)
                if latest is None
                else _EMPTY_SNAPSHOT_BOXES
            ),
        )
        if events.num_rows == 0:
            return None
        content = _serialize_events(events)
        sha256 = hashlib.sha256(content).hexdigest()
        sequence = 1 if latest is None else latest.sequence + 1
        event_file = EventFile(
            event_file_id=f"{sequence:06d}-{sha256[:12]}",
            sequence=sequence,
            parent_event_file_id=None if latest is None else latest.event_file_id,
            after_log_sequence=after,
            last_log_sequence=last,
            created_at=datetime.now(tz=UTC),
            row_count=events.num_rows,
            size=len(content),
            sha256=sha256,
        )
        connection.execute(
            f"""
            insert into {SYNC_SCHEMA}.event_file
                (dataset, sequence, event_file_id, parent_event_file_id,
                 after_log_sequence, last_log_sequence, created_at, row_count, size,
                 sha256, content)
            values (%(dataset)s, %(sequence)s, %(event_file_id)s,
                    %(parent_event_file_id)s, %(after_log_sequence)s,
                    %(last_log_sequence)s, %(created_at)s, %(row_count)s, %(size)s,
                    %(sha256)s, %(content)s)
            """,
            {"dataset": dataset, "content": content, **event_file.__dict__},
        )
    return event_file


def read_events(
    *,
    connection: psycopg.Connection,
    dataset: str,
    after_log_sequence: int,
    last_log_sequence: int,
    snapshot_boxes: pa.Table,
) -> pa.Table:
    """Return the edited boxes of a part of the log, in the order of the log."""
    rows = connection.execute(
        _EVENT_QUERY,
        {
            "dataset": dataset,
            "after": after_log_sequence,
            "last": last_log_sequence,
            "box_ids": snapshot_boxes.column("box_id").to_pylist(),
            "splits": snapshot_boxes.column("split").to_pylist(),
            "file_names": snapshot_boxes.column("file_name").to_pylist(),
        },
    ).fetchall()
    columns = list(zip(*rows, strict=True)) or [[] for _ in EVENT_SCHEMA]
    return pa.table(
        [
            pa.array(list(column), type=field.type)
            for column, field in zip(columns, EVENT_SCHEMA, strict=True)
        ],
        schema=EVENT_SCHEMA,
    )


def list_event_files(connection: psycopg.Connection, dataset: str) -> list[EventFile]:
    rows = connection.execute(
        f"""
        select event_file_id, sequence, parent_event_file_id, after_log_sequence,
               last_log_sequence, created_at, row_count, size, sha256
        from {SYNC_SCHEMA}.event_file
        where dataset = %(dataset)s
        order by sequence
        """,
        {"dataset": dataset},
    ).fetchall()
    return [
        EventFile(
            event_file_id=row[0],
            sequence=row[1],
            parent_event_file_id=row[2],
            after_log_sequence=row[3],
            last_log_sequence=row[4],
            created_at=row[5],
            row_count=row[6],
            size=row[7],
            sha256=row[8],
        )
        for row in rows
    ]


def read_event_file_content(
    *, connection: psycopg.Connection, dataset: str, event_file_id: str
) -> bytes | None:
    row = connection.execute(
        f"""
        select content from {SYNC_SCHEMA}.event_file
        where dataset = %(dataset)s and event_file_id = %(event_file_id)s
        """,
        {"dataset": dataset, "event_file_id": event_file_id},
    ).fetchone()
    return None if row is None else bytes(row[0])


def _read_last_log_sequence(connection: psycopg.Connection) -> int:
    row = connection.execute(
        f"select coalesce(max(sequence), 0) from {SYNC_SCHEMA}.edit_log"
    ).fetchone()
    return 0 if row is None else int(row[0])


_EMPTY_SNAPSHOT_BOXES = pa.table(
    {
        "split": pa.array([], type=pa.string()),
        "file_name": pa.array([], type=pa.string()),
        "box_id": pa.array([], type=pa.string()),
    }
)


def _read_snapshot_boxes(connection: psycopg.Connection, dataset: str) -> pa.Table:
    """Return the boxes of the latest snapshot of ADR 0009, or none."""
    row = connection.execute(
        f"""
        select content from {SYNC_SCHEMA}.snapshot
        where dataset = %(dataset)s
        order by sequence desc
        limit 1
        """,
        {"dataset": dataset},
    ).fetchone()
    if row is None:
        return _EMPTY_SNAPSHOT_BOXES
    return pq.read_table(io.BytesIO(bytes(row[0]))).select(
        ["split", "file_name", "box_id"]
    )


def _serialize_events(events: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table=events, where=sink)
    return sink.getvalue().to_pybytes()


# Every logged box, with the image of the box as the log or LightlyStudio names it. A
# box of a snapshot has its image in the snapshot, and comes first, as the log entry 0.
_EVENT_QUERY = f"""
with logged as (
    select sample_id, max(sequence) as log_sequence,
           (array_agg(image_id order by sequence desc)
               filter (where image_id is not null))[1] as image_id
    from {SYNC_SCHEMA}.edit_log
    where sequence > %(after)s and sequence <= %(last)s
    group by sample_id
),
box as (
    select base.sample_id, base.parent_sample_id as image_id,
           label.annotation_label_name as label,
           detection.x, detection.y, detection.width, detection.height
    from annotation_base base
    join object_detection_annotation detection on detection.sample_id = base.sample_id
    join annotation_label label
        on label.annotation_label_id = base.annotation_label_id
),
edited as (
    select logged.sample_id, logged.log_sequence, image.dataset, image.split,
           image.file_name
    from logged
    left join box on box.sample_id = logged.sample_id
    join {SYNC_SCHEMA}.loaded_image image
        on image.image_id = coalesce(box.image_id, logged.image_id)
    -- An annotation that is no box, such as a classification, is no event.
    where box.sample_id is not null
       or not exists (
           select from annotation_base other where other.sample_id = logged.sample_id
       )
    union all
    select snapshot_box.box_id::uuid, 0, %(dataset)s, snapshot_box.split,
           snapshot_box.file_name
    from unnest(%(box_ids)s::text[], %(splits)s::text[], %(file_names)s::text[])
        as snapshot_box(box_id, split, file_name)
    where snapshot_box.box_id::uuid not in (select sample_id from logged)
)
select edited.dataset, edited.split, edited.file_name, edited.sample_id::text,
       edited.log_sequence, box.sample_id is null, box.label, box.x::float8,
       box.y::float8, box.width::float8, box.height::float8
from edited
left join box on box.sample_id = edited.sample_id
where edited.dataset = %(dataset)s
order by edited.log_sequence, edited.sample_id
"""
