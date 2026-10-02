#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The tables in which the sync records what it wrote into LightlyStudio.

LightlyStudio tracks no change: an annotation has no update time, and a delete leaves
no row. `loaded_box` holds every box as the sync last wrote it, so a later run can
tell a box that a curator changed from a box that only gold changed.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session

SYNC_SCHEMA = "lakehouse_sync"

_STATEMENTS = (
    f"create schema if not exists {SYNC_SCHEMA}",
    f"""
    create table if not exists {SYNC_SCHEMA}.synced_dataset (
        dataset text primary key,
        gold_version integer not null,
        gold_built_at timestamptz not null
    )
    """,
    f"""
    create table if not exists {SYNC_SCHEMA}.loaded_box (
        box_id uuid primary key,
        dataset text not null,
        label text not null,
        x integer not null,
        y integer not null,
        width integer not null,
        height integer not null
    )
    """,
    f"""
    create index if not exists loaded_box_dataset
    on {SYNC_SCHEMA}.loaded_box (dataset)
    """,
)


def create_sync_schema(session: Session) -> None:
    for statement in _STATEMENTS:
        session.execute(text(statement))
    session.commit()
