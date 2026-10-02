#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Copy gold pages into temporary tables, so the merge runs inside Postgres.

A row never passes through an insert statement. The scalar tables go in as CSV, and the
vectors go in as binary rows.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from uuid import UUID

import numpy as np
import psycopg
import pyarrow as pa
import pyarrow.csv as pa_csv
from pgvector.psycopg import register_vector
from psycopg import sql
from sqlalchemy import text
from sqlmodel import Session

# Every table goes away with the transaction, so a failed run leaves nothing behind.
_CREATE_STAGING_TABLES = (
    "create temp table stage_label (name text, label_id uuid) on commit drop",
    "create temp table stage_tag (name text, tag_id uuid) on commit drop",
    """
    create temp table stage_image (
        image_id uuid, file_name text, width integer, height integer,
        image_path text, role text, split text
    ) on commit drop
    """,
    """
    create temp table stage_box (
        box_id uuid, image_id uuid, label text, confidence double precision,
        x integer, y integer, width integer, height integer
    ) on commit drop
    """,
    """
    create temp table stage_embedding (sample_id uuid, embedding vector)
    on commit drop
    """,
)


def create_staging_tables(session: Session) -> None:
    for statement in _CREATE_STAGING_TABLES:
        session.execute(text(statement))


def stage_names(session: Session, table: str, ids: dict[str, UUID]) -> None:
    """Fill `stage_label` or `stage_tag`, which map a name onto its id."""
    column = "label_id" if table == "stage_label" else "tag_id"
    for name, value in ids.items():
        session.execute(
            text(f"insert into {table} (name, {column}) values (:name, :value)"),
            {"name": name, "value": value},
        )


def stage_images(session: Session, page: pa.Table) -> None:
    _copy_as_csv(
        session=session,
        table="stage_image",
        columns={
            "image_id": page.column("image_id"),
            "file_name": page.column("file_name"),
            "width": page.column("width"),
            "height": page.column("height"),
            "image_path": page.column("image_path"),
            "role": page.column("role"),
            "split": page.column("split"),
        },
    )


def stage_boxes(session: Session, page: pa.Table) -> None:
    _copy_as_csv(
        session=session,
        table="stage_box",
        columns={
            "box_id": page.column("box_id"),
            "image_id": page.column("image_id"),
            "label": page.column("class_name"),
            "confidence": page.column("confidence"),
            # LightlyStudio stores whole pixels. Gold keeps the exact float.
            "x": _round_to_pixels(page.column("x")),
            "y": _round_to_pixels(page.column("y")),
            "width": _round_to_pixels(page.column("w")),
            "height": _round_to_pixels(page.column("h")),
        },
    )


def stage_embeddings(*, session: Session, ids: Sequence[str], page: pa.Table) -> None:
    """Copy one page of vectors, each onto the sample id beside it."""
    vectors = (
        page.column("embedding")
        .combine_chunks()
        .values.to_numpy()
        .reshape(len(ids), -1)
    )
    connection = _get_driver_connection(session)
    register_vector(connection)
    with (
        connection.cursor() as cursor,
        cursor.copy(
            "copy stage_embedding (sample_id, embedding) from stdin (format binary)"
        ) as copy,
    ):
        copy.set_types(["uuid", "vector"])
        for sample_id, vector in zip(ids, vectors, strict=True):
            copy.write_row((UUID(sample_id), vector))


def _round_to_pixels(column: pa.ChunkedArray) -> pa.Array:
    """Round half up, as the DuckDB loader of silver does."""
    return pa.array(np.floor(column.to_numpy() + 0.5).astype(np.int32))


def _copy_as_csv(
    *, session: Session, table: str, columns: dict[str, pa.ChunkedArray | pa.Array]
) -> None:
    sink = io.BytesIO()
    pa_csv.write_csv(
        data=pa.table(columns),
        output_file=sink,
        write_options=pa_csv.WriteOptions(include_header=False),
    )
    with (
        _get_driver_connection(session).cursor() as cursor,
        cursor.copy(
            sql.SQL("copy {} ({}) from stdin (format csv)").format(
                sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, columns))
            )
        ) as copy,
    ):
        copy.write(sink.getvalue())


def _get_driver_connection(session: Session) -> psycopg.Connection:
    """Return the psycopg connection of the open transaction, which has `COPY`."""
    connection = session.connection().connection.driver_connection
    assert isinstance(connection, psycopg.Connection)
    return connection
