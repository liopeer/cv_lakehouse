#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Sync every gold dataset into LightlyStudio, then sleep, then again.

`sync_service` starts this loop beside the export. The LightlyStudio server owns the
schema: it runs the migrations when it starts. This loop only writes while the database
is at the migration that its own LightlyStudio build expects.
"""

from __future__ import annotations

import logging
import threading
import time

import psycopg
from lightly_studio.database import db_manager, db_migrations

from cv_lakehouse_studio.gold_client import GoldClient, HttpGoldClient
from cv_lakehouse_studio.studio_datasets import (
    DEFAULT_MAX_IMAGES_PER_DATASET,
    read_or_plan_studio_datasets,
)
from cv_lakehouse_studio.studio_settings import StudioSettings
from cv_lakehouse_studio.sync import SyncReport, read_synced_gold, sync_dataset
from cv_lakehouse_studio.sync_state import create_sync_schema

logger = logging.getLogger(__name__)


class SchemaMismatchError(RuntimeError):
    """The database is not at the migration that this LightlyStudio build expects."""


def reject_schema_mismatch(database_url: str) -> None:
    expected = db_migrations.get_head_revision()
    try:
        # psycopg takes a libpq URL, which names no SQLAlchemy driver.
        libpq_url = database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(libpq_url) as connection:
            row = connection.execute(
                "select version_num from alembic_version"
            ).fetchone()
    except psycopg.Error as error:
        raise SchemaMismatchError(
            f"The LightlyStudio server has not migrated the database yet: {error}"
        ) from error
    found = None if row is None else row[0]
    if found != expected:
        raise SchemaMismatchError(
            f"The database is at migration {found}, and this build expects {expected}."
        )


def sync_every_dataset(
    *,
    client: GoldClient,
    image_base: str,
    max_images: int = DEFAULT_MAX_IMAGES_PER_DATASET,
) -> list[SyncReport]:
    """Sync each Studio dataset whose gold changed since its last sync.

    The caller connects `db_manager` first.
    """
    session = db_manager.persistent_session()
    create_sync_schema(session)
    meta = client.read_meta()
    class_names = client.read_class_names()
    reports: list[SyncReport] = []
    for dataset in meta.datasets:
        studio_datasets = read_or_plan_studio_datasets(
            session=session, client=client, dataset=dataset, max_images=max_images
        )
        for studio_dataset in studio_datasets:
            synced = read_synced_gold(
                session=session, studio_dataset=studio_dataset.name
            )
            if synced == (meta.version, meta.built_at):
                continue
            report = sync_dataset(
                client=client,
                meta=meta,
                dataset=dataset,
                studio_dataset=studio_dataset,
                class_names=class_names,
                image_base=image_base,
            )
            logger.info(f"Synced {report}")
            reports.append(report)
    return reports


def run_sync_loop(settings: StudioSettings, sync_lock: threading.Lock) -> None:
    """Run forever. Hold `sync_lock` during a run, so the export waits for its end."""
    client = HttpGoldClient(
        base_url=settings.gold_api_url, timeout_seconds=settings.request_timeout_seconds
    )
    connected = False
    while True:
        try:
            reject_schema_mismatch(settings.database_url)
            with sync_lock:
                if not connected:
                    db_manager.connect(db_url=settings.database_url)
                    connected = True
                sync_every_dataset(
                    client=client,
                    image_base=settings.image_base,
                    max_images=settings.max_images_per_dataset,
                )
        except Exception:
            # A failed run changes nothing, and the next run starts from gold again.
            logger.exception("The sync failed. The next run tries again.")
            if connected:
                db_manager.persistent_session().rollback()
        time.sleep(settings.sync_interval_seconds)
