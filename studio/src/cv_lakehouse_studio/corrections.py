#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Find what the curators changed, and publish it as immutable snapshots.

LightlyStudio records no change. The sync records every box as it last wrote it, so a
correction is the difference between LightlyStudio and that record:

- A box that differs from its record was relabelled or moved.
- A recorded box that is gone was deleted.
- A box with no record was drawn by a curator.

The difference holds every live correction, not only the new ones. So a snapshot is
complete on its own, and the lake applies the latest one to the untouched source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import psycopg
import pyarrow as pa
import pyarrow.parquet as pq

from cv_lakehouse_studio.sync_state import SYNC_SCHEMA

CORRECTIONS_FILE = "corrections.parquet"

# The contract with `cv_lakehouse.sources.correction_snapshots`. A null `label_name`
# is an unchanged class, and a null box is an unmoved box.
CORRECTION_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="action", type=pa.string(), nullable=False),
        pa.field(name="label_name", type=pa.string()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
    ]
)


@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    sequence: int
    parent_snapshot_id: str | None
    created_at: datetime
    row_count: int
    size: int
    sha256: str


def detect_corrections(connection: psycopg.Connection, dataset: str) -> pa.Table:
    """Return every live correction of one dataset, in the order of the box id."""
    rows = connection.execute(_CORRECTION_QUERY, {"dataset": dataset}).fetchall()
    columns = list(zip(*rows, strict=True)) or [[] for _ in CORRECTION_SCHEMA]
    return pa.table(
        [
            pa.array(list(column), type=field.type)
            for column, field in zip(columns, CORRECTION_SCHEMA, strict=True)
        ],
        schema=CORRECTION_SCHEMA,
    )


def publish_snapshot_if_changed(
    connection: psycopg.Connection, dataset: str
) -> Snapshot | None:
    """Store a new snapshot when the corrections differ from the latest one.

    A dataset with no correction and no snapshot gets none. Corrections that a curator
    took back give a snapshot with fewer rows, down to none.
    """
    corrections = detect_corrections(connection=connection, dataset=dataset)
    content = _serialize_corrections(corrections)
    sha256 = hashlib.sha256(content).hexdigest()
    snapshots = list_snapshots(connection=connection, dataset=dataset)
    latest = snapshots[-1] if snapshots else None
    if latest is None and corrections.num_rows == 0:
        return None
    if latest is not None and latest.sha256 == sha256:
        return None
    sequence = 1 if latest is None else latest.sequence + 1
    snapshot = Snapshot(
        snapshot_id=f"{sequence:06d}-{sha256[:12]}",
        sequence=sequence,
        parent_snapshot_id=None if latest is None else latest.snapshot_id,
        created_at=datetime.now(tz=UTC),
        row_count=corrections.num_rows,
        size=len(content),
        sha256=sha256,
    )
    connection.execute(
        f"""
        insert into {SYNC_SCHEMA}.snapshot
            (dataset, sequence, snapshot_id, parent_snapshot_id, created_at,
             row_count, size, sha256, content)
        values (%(dataset)s, %(sequence)s, %(snapshot_id)s, %(parent_snapshot_id)s,
                %(created_at)s, %(row_count)s, %(size)s, %(sha256)s, %(content)s)
        """,
        {
            "dataset": dataset,
            "sequence": snapshot.sequence,
            "snapshot_id": snapshot.snapshot_id,
            "parent_snapshot_id": snapshot.parent_snapshot_id,
            "created_at": snapshot.created_at,
            "row_count": snapshot.row_count,
            "size": snapshot.size,
            "sha256": snapshot.sha256,
            "content": content,
        },
    )
    return snapshot


def list_snapshots(connection: psycopg.Connection, dataset: str) -> list[Snapshot]:
    rows = connection.execute(
        f"""
        select snapshot_id, sequence, parent_snapshot_id, created_at, row_count,
               size, sha256
        from {SYNC_SCHEMA}.snapshot
        where dataset = %(dataset)s
        order by sequence
        """,
        {"dataset": dataset},
    ).fetchall()
    return [
        Snapshot(
            snapshot_id=row[0],
            sequence=row[1],
            parent_snapshot_id=row[2],
            created_at=row[3],
            row_count=row[4],
            size=row[5],
            sha256=row[6],
        )
        for row in rows
    ]


def read_snapshot_content(
    *, connection: psycopg.Connection, dataset: str, snapshot_id: str
) -> bytes | None:
    row = connection.execute(
        f"""
        select content from {SYNC_SCHEMA}.snapshot
        where dataset = %(dataset)s and snapshot_id = %(snapshot_id)s
        """,
        {"dataset": dataset, "snapshot_id": snapshot_id},
    ).fetchone()
    return None if row is None else bytes(row[0])


def _serialize_corrections(corrections: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table=corrections, where=sink)
    return sink.getvalue().to_pybytes()


# Every object detection on an image of the dataset, as LightlyStudio holds it now. The
# annotation collection does not matter: the GUI puts a new box into the collection
# that the curator has selected.
_CORRECTION_QUERY = f"""
with studio_box as (
    select base.sample_id as box_id, base.parent_sample_id as image_id,
           label.annotation_label_name as label,
           detection.x, detection.y, detection.width, detection.height
    from annotation_base base
    join object_detection_annotation detection on detection.sample_id = base.sample_id
    join annotation_label label
        on label.annotation_label_id = base.annotation_label_id
    join {SYNC_SCHEMA}.loaded_image image on image.image_id = base.parent_sample_id
    where image.dataset = %(dataset)s
),
loaded as (
    select * from {SYNC_SCHEMA}.loaded_box where dataset = %(dataset)s
),
correction as (
    select loaded.box_id, loaded.image_id, 'update' as action,
           case when studio_box.label <> loaded.label then studio_box.label end
               as label_name,
           (studio_box.x, studio_box.y, studio_box.width, studio_box.height)
               is distinct from (loaded.x, loaded.y, loaded.width, loaded.height)
               as is_moved,
           studio_box.x, studio_box.y, studio_box.width, studio_box.height
    from loaded
    join studio_box on studio_box.box_id = loaded.box_id
    where (studio_box.label, studio_box.x, studio_box.y, studio_box.width,
           studio_box.height)
          is distinct from
          (loaded.label, loaded.x, loaded.y, loaded.width, loaded.height)
    union all
    select loaded.box_id, loaded.image_id, 'delete', null, false,
           null, null, null, null
    from loaded
    left join studio_box on studio_box.box_id = loaded.box_id
    where studio_box.box_id is null
    union all
    select studio_box.box_id, studio_box.image_id, 'add', studio_box.label, true,
           studio_box.x, studio_box.y, studio_box.width, studio_box.height
    from studio_box
    left join {SYNC_SCHEMA}.loaded_box loaded on loaded.box_id = studio_box.box_id
    where loaded.box_id is null
)
select image.dataset, image.split, image.file_name, correction.box_id::text,
       correction.action, correction.label_name,
       case when correction.is_moved then correction.x::float8 end,
       case when correction.is_moved then correction.y::float8 end,
       case when correction.is_moved then correction.width::float8 end,
       case when correction.is_moved then correction.height::float8 end
from correction
join {SYNC_SCHEMA}.loaded_image image on image.image_id = correction.image_id
order by correction.box_id
"""
