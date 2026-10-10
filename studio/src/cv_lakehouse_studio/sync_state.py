#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The tables and the triggers that the sync and the export add to LightlyStudio.

LightlyStudio records no change. So triggers log every write of a curator to a box in
`edit_log`, and the export publishes the logged boxes as event files. The sync marks its
own writes with `lakehouse.origin`, and the triggers skip them.

The triggers also guard a box that a curator edited after the last edit that gold
holds. A write of the sync to such a box does nothing, also when the curator edits
while the sync runs.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlmodel import Session

SYNC_SCHEMA = "lakehouse_sync"

# The setting that marks a write of the sync, and the setting that names the last log
# entry that gold holds. Both hold for one transaction.
ORIGIN_SETTING = "lakehouse.origin"
SYNC_ORIGIN = "sync"
GOLD_LOG_SEQUENCE_SETTING = "lakehouse.gold_log_sequence"

# The tables of LightlyStudio that hold a box. A relabel writes `annotation_base`, and a
# move writes `object_detection_annotation`.
BOX_TABLES = ("annotation_base", "object_detection_annotation")

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
    # The gold identity of every image that the sync wrote. The export names the image
    # of an event with it.
    f"""
    create table if not exists {SYNC_SCHEMA}.loaded_image (
        image_id uuid primary key,
        dataset text not null,
        split text not null,
        file_name text not null
    )
    """,
    # The record of what the sync last wrote, and of the gold it last read. A full
    # overwrite needs neither.
    f"drop table if exists {SYNC_SCHEMA}.loaded_box",
    f"drop table if exists {SYNC_SCHEMA}.synced_dataset",
    # The snapshots of ADR 0009. The first event file of a dataset takes over the boxes
    # of its latest snapshot, and then no code reads them.
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
    # One row per database. A new database starts a new chain of event files.
    f"""
    create table if not exists {SYNC_SCHEMA}.event_chain (
        chain_id text primary key,
        created_at timestamptz not null default now()
    )
    """,
    f"""
    insert into {SYNC_SCHEMA}.event_chain (chain_id)
    select gen_random_uuid()::text
    where not exists (select from {SYNC_SCHEMA}.event_chain)
    """,
    # One row per write of a curator to a box. `image_id` is the image of the box, when
    # the trigger can tell it.
    f"""
    create table if not exists {SYNC_SCHEMA}.edit_log (
        sequence bigserial primary key,
        sample_id uuid not null,
        image_id uuid,
        logged_at timestamptz not null default now()
    )
    """,
    f"""
    create index if not exists edit_log_sample_id
    on {SYNC_SCHEMA}.edit_log (sample_id, sequence)
    """,
    # An event file never changes. The bytes that the lake pinned stay the bytes that
    # the export serves.
    f"""
    create table if not exists {SYNC_SCHEMA}.event_file (
        dataset text not null,
        sequence integer not null,
        event_file_id text not null,
        parent_event_file_id text,
        after_log_sequence bigint not null,
        last_log_sequence bigint not null,
        created_at timestamptz not null,
        row_count integer not null,
        size integer not null,
        sha256 text not null,
        content bytea not null,
        primary key (dataset, sequence)
    )
    """,
    # A curator write: log it. A box that a curator moves loses its vector, as the
    # vector shows the old crop. The sync loads the new vector once gold holds the move.
    f"""
    create or replace function {SYNC_SCHEMA}.log_curator_edit() returns trigger
    language plpgsql as $$
    declare
        edited record;
        image uuid;
    begin
        if tg_op = 'DELETE' then edited := old; else edited := new; end if;
        if tg_table_name = 'annotation_base' then
            image := edited.parent_sample_id;
        else
            select parent_sample_id into image from annotation_base
            where sample_id = edited.sample_id;
        end if;
        insert into {SYNC_SCHEMA}.edit_log (sample_id, image_id)
        values (edited.sample_id, image);
        -- PL/pgSQL resolves every field of a condition, so the table comes first.
        if tg_table_name = 'object_detection_annotation' and tg_op = 'UPDATE' then
            if (old.x, old.y, old.width, old.height)
                is distinct from (new.x, new.y, new.width, new.height)
            then
                delete from sample_embedding where sample_id = new.sample_id;
            end if;
        end if;
        return null;
    end $$
    """,
    # A sync write to a box that a curator edited after the gold of the sync: skip it.
    # Each query of a trigger takes a new snapshot, so it sees an edit that committed
    # while the statement of the sync waited for the row.
    f"""
    create or replace function {SYNC_SCHEMA}.guard_curator_edit() returns trigger
    language plpgsql as $$
    declare
        edited record;
    begin
        if tg_op = 'DELETE' then edited := old; else edited := new; end if;
        if exists (
            select from {SYNC_SCHEMA}.edit_log
            where sample_id = edited.sample_id
              and sequence > current_setting('{GOLD_LOG_SEQUENCE_SETTING}')::bigint
        ) then
            return null;
        end if;
        return edited;
    end $$
    """,
    *(
        statement
        for table in BOX_TABLES
        for statement in (
            f"""
            create or replace trigger lakehouse_log_curator_edit
            after insert or update or delete on {table}
            for each row
            when (current_setting('{ORIGIN_SETTING}', true)
                  is distinct from '{SYNC_ORIGIN}')
            execute function {SYNC_SCHEMA}.log_curator_edit()
            """,
            f"""
            create or replace trigger lakehouse_guard_curator_edit
            before update or delete on {table}
            for each row
            when (current_setting('{ORIGIN_SETTING}', true) = '{SYNC_ORIGIN}')
            execute function {SYNC_SCHEMA}.guard_curator_edit()
            """,
        )
    ),
)


def create_sync_schema(session: Session) -> None:
    for statement in _STATEMENTS:
        session.execute(text(statement))
    session.commit()


def read_chain_id(session: Session) -> str:
    return session.execute(
        text(f"select chain_id from {SYNC_SCHEMA}.event_chain")
    ).scalar_one()
