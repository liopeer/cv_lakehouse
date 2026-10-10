#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The sync image: the sync loop, and the export of the corrections beside it.

Start it with `python -m cv_lakehouse_studio.sync_service`. The loop runs in a thread
of its own, and the export answers on port 8002 meanwhile.
"""

import logging
import threading

import uvicorn

from cv_lakehouse_studio.export_api import create_export_app
from cv_lakehouse_studio.studio_settings import StudioSettings
from cv_lakehouse_studio.sync_loop import run_sync_loop


def start_sync_service() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    # Alembic sets the root logger to WARNING when LightlyStudio connects. A level on
    # this package stands, so the report of each sync run reaches the container log.
    logging.getLogger("cv_lakehouse_studio").setLevel(logging.INFO)
    settings = StudioSettings.model_validate({})
    threading.Thread(
        target=run_sync_loop, kwargs={"settings": settings}, daemon=True
    ).start()
    uvicorn.run(
        create_export_app(database_url=settings.database_url),
        host="0.0.0.0",
        port=settings.export_port,
    )


if __name__ == "__main__":
    start_sync_service()
