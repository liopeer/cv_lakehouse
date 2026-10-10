#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Serve the event files of the curator edits to the lakehouse.

The asset `bronze/<dataset>_corrections` reads the listing, and then each file that it
does not hold. A listing first publishes the edits that the log holds after the last
file, so a run of that asset takes what the curators changed up to that moment.

Nothing here authenticates. Keep it on a private network.
"""

from __future__ import annotations

import psycopg
from fastapi import FastAPI, HTTPException, Response

from cv_lakehouse_studio.edit_events import (
    EVENTS_FILE,
    EventFile,
    list_event_files,
    publish_event_file_if_new,
    read_chain_id,
    read_event_file_content,
)

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


def create_export_app(*, database_url: str) -> FastAPI:
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

    @app.get("/v1/datasets/{dataset}/events")
    def list_dataset_events(dataset: str) -> dict:
        with connect() as connection:
            try:
                publish_event_file_if_new(connection=connection, dataset=dataset)
                chain_id = read_chain_id(connection)
                event_files = list_event_files(connection=connection, dataset=dataset)
            except (
                psycopg.errors.UndefinedTable,
                psycopg.errors.InvalidSchemaName,
            ) as error:
                raise HTTPException(
                    status_code=503, detail="The sync has not run yet."
                ) from error
        return {
            "chain_id": chain_id,
            "event_files": [
                _describe_event_file(dataset=dataset, event_file=event_file)
                for event_file in event_files
            ],
        }

    @app.get("/v1/datasets/{dataset}/events/{event_file_id}/" + EVENTS_FILE)
    def read_event_file(dataset: str, event_file_id: str) -> Response:
        with connect() as connection:
            content = read_event_file_content(
                connection=connection, dataset=dataset, event_file_id=event_file_id
            )
        if content is None:
            raise HTTPException(status_code=404, detail="No such event file.")
        return Response(content=content, media_type=PARQUET_MEDIA_TYPE)

    return app


def _describe_event_file(dataset: str, event_file: EventFile) -> dict:
    return {
        "event_file_id": event_file.event_file_id,
        "sequence": event_file.sequence,
        "parent_event_file_id": event_file.parent_event_file_id,
        "after_log_sequence": event_file.after_log_sequence,
        "last_log_sequence": event_file.last_log_sequence,
        "created_at": event_file.created_at.isoformat(),
        "row_count": event_file.row_count,
        # Relative to the URL of this service, which the reader knows.
        "path": (
            f"v1/datasets/{dataset}/events/{event_file.event_file_id}/{EVENTS_FILE}"
        ),
        "size": event_file.size,
        "sha256": event_file.sha256,
    }
