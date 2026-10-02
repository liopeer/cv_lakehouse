#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Serve the LightlyStudio GUI and API against LIGHTLY_STUDIO_DATABASE_URL.

`lightly-studio gui` does not work as a service entrypoint. It refuses to start while
the database holds no dataset, and it moves to a random port when its port is taken.
This starts the same FastAPI app on a fixed address, on a database that can be empty.
The overview page then shows that no dataset is loaded yet.

Run one process with one worker. The app keeps one database session per process.
"""

import os

import uvicorn
from lightly_studio.api.app import app
from lightly_studio.database import db_manager
from lightly_studio.dataset import env


def start_gui_server() -> None:
    db_url = os.environ.get("LIGHTLY_STUDIO_DATABASE_URL")
    if not db_url:
        raise SystemExit("LIGHTLY_STUDIO_DATABASE_URL is not set.")
    # This creates the database, enables pgvector and runs the migrations.
    db_manager.connect(db_url=db_url)
    # The settings of `lightly_studio.api.server.Server`.
    uvicorn.run(
        app,
        host=env.LIGHTLY_STUDIO_HOST,
        port=env.LIGHTLY_STUDIO_PORT,
        http="h11",
        limit_concurrency=128,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=30,
        access_log=env.LIGHTLY_STUDIO_DEBUG,
    )


if __name__ == "__main__":
    start_gui_server()
