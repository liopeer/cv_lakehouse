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
    # How a gold dataset maps onto Studio datasets. A row is written once, so a later
    # count never moves an image into another Studio dataset.
    f"""
    create table if not exists {SYNC_SCHEMA}.studio_dataset (
        studio_dataset text primary key,
        dataset text not null,
        split text,
        shard integer,
        num_shards integer
    )
    """,
    # Before shards, a Studio dataset had the name of its gold dataset.
    f"""
    do $$ begin
        if exists (
            select from information_schema.columns
            where table_schema = '{SYNC_SCHEMA}' and table_name = 'synced_dataset'
              and column_name = 'dataset'
        ) then
            alter table {SYNC_SCHEMA}.synced_dataset
                rename column dataset to studio_dataset;
        end if;
    end $$
    """,
    f"""
    create table if not exists {SYNC_SCHEMA}.synced_dataset (
        studio_dataset text primary key,
        gold_version integer not null,
        gold_built_at timestamptz not null
    )
    """,
    f"""
    create table if not exists {SYNC_SCHEMA}.loaded_image (
        image_id uuid primary key,
        dataset text not null,
        split text not null,
        file_name text not null
    )
    """,
    f"""
    create table if not exists {SYNC_SCHEMA}.loaded_box (
        box_id uuid primary key,
        image_id uuid not null,
        dataset text not null,
        studio_dataset text not null,
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
    f"""
    alter table {SYNC_SCHEMA}.loaded_box add column if not exists studio_dataset text
    """,
    f"""
    update {SYNC_SCHEMA}.loaded_box set studio_dataset = dataset
    where studio_dataset is null
    """,
    f"alter table {SYNC_SCHEMA}.loaded_box alter column studio_dataset set not null",
    f"""
    create index if not exists loaded_box_studio_dataset
    on {SYNC_SCHEMA}.loaded_box (studio_dataset)
    """,
    # The export publishes the corrections of a dataset as snapshots. A row never
    # changes, so the bytes that the lake pinned stay the bytes that this serves.
    f"""
    create table if not exists {SYNC_SCHEMA}.snapshot (
        dataset text not null,
        sequence integer not null,
        snapshot_id text not null,
        parent_snapshot_id text,
        created_at timestamptz not null,
        row_count integer not null,
        size integer not null,
        sha256 text not null,
        content bytea not null,
        primary key (dataset, sequence)
    )
    """,
)


def create_sync_schema(session: Session) -> None:
    for statement in _STATEMENTS:
        session.execute(text(statement))
    session.commit()
