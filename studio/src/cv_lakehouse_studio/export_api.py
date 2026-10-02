#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Serve the correction snapshots to the lakehouse.

The asset `bronze/<dataset>_corrections` reads the listing, and then each file that it
does not hold. A listing first looks for new corrections, so a run of that asset takes
what the curators changed up to that moment.

Nothing here authenticates. Keep it on a private network.
"""

from __future__ import annotations

import threading

import psycopg
from fastapi import FastAPI, HTTPException, Response

from cv_lakehouse_studio.corrections import (
    CORRECTIONS_FILE,
    Snapshot,
    list_snapshots,
    publish_snapshot_if_changed,
    read_snapshot_content,
)

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


def create_export_app(*, database_url: str, sync_lock: threading.Lock) -> FastAPI:
    """Build the app. `sync_lock` is held by the sync while it writes.

    A sync run changes LightlyStudio and its own record in more than one transaction.
    A look at the two in between would read a half written run as a correction.
    """
    app = FastAPI(title="cv_lakehouse studio export")
    # psycopg takes a libpq URL, which names no SQLAlchemy driver.
    libpq_url = database_url.replace("postgresql+psycopg://", "postgresql://")

    def connect() -> psycopg.Connection:
        try:
            return psycopg.connect(libpq_url)
        except psycopg.Error as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/healthz")
    def check_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/datasets/{dataset}/snapshots")
    def list_dataset_snapshots(dataset: str) -> list[dict]:
        with sync_lock, connect() as connection:
            try:
                publish_snapshot_if_changed(connection=connection, dataset=dataset)
                snapshots = list_snapshots(connection=connection, dataset=dataset)
            except (
                psycopg.errors.UndefinedTable,
                psycopg.errors.InvalidSchemaName,
            ) as error:
                raise HTTPException(
                    status_code=503, detail="The sync has not run yet."
                ) from error
        return [_describe_snapshot(dataset=dataset, snapshot=s) for s in snapshots]

    @app.get("/v1/datasets/{dataset}/snapshots/{snapshot_id}/" + CORRECTIONS_FILE)
    def read_snapshot_file(dataset: str, snapshot_id: str) -> Response:
        with connect() as connection:
            content = read_snapshot_content(
                connection=connection, dataset=dataset, snapshot_id=snapshot_id
            )
        if content is None:
            raise HTTPException(status_code=404, detail="No such snapshot.")
        return Response(content=content, media_type=PARQUET_MEDIA_TYPE)

    return app


def _describe_snapshot(dataset: str, snapshot: Snapshot) -> dict:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "sequence": snapshot.sequence,
        "parent_snapshot_id": snapshot.parent_snapshot_id,
        "created_at": snapshot.created_at.isoformat(),
        "row_count": snapshot.row_count,
        # Relative to the URL of this service, which the reader knows.
        "path": (
            f"v1/datasets/{dataset}/snapshots/{snapshot.snapshot_id}/{CORRECTIONS_FILE}"
        ),
        "size": snapshot.size,
        "sha256": snapshot.sha256,
    }
