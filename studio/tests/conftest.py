#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A Postgres with pgvector for the tests, and one fresh database per test.

Set CV_LAKEHOUSE_STUDIO_TEST_POSTGRES_URL to a server, such as
postgresql://postgres:test@localhost:5432, to use it. Otherwise the tests start the
image that LightlyStudio tests against, with Docker.
"""

import os
import subprocess
import time
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest
from lightly_studio.database import db_manager

POSTGRES_IMAGE = "pgvector/pgvector:0.8.1-pg18-bookworm"
STARTUP_TIMEOUT_SECONDS = 60


@pytest.fixture(scope="session")
def postgres_server() -> Iterator[str]:
    configured = os.environ.get("CV_LAKEHOUSE_STUDIO_TEST_POSTGRES_URL")
    if configured:
        yield configured.rstrip("/")
        return
    container = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--env",
            "POSTGRES_PASSWORD=test",
            "--publish",
            "127.0.0.1::5432",
            POSTGRES_IMAGE,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        port = (
            subprocess.run(
                ["docker", "port", container, "5432/tcp"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .rsplit(":", maxsplit=1)[1]
        )
        server = f"postgresql://postgres:test@127.0.0.1:{port}"
        _wait_for_server(server)
        yield server
    finally:
        subprocess.run(["docker", "stop", container], check=False, capture_output=True)


def _wait_for_server(server: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while True:
        try:
            # Over TCP with a query. The bootstrap server of the image listens on the
            # Unix socket only, so this passes when the real server is up.
            with psycopg.connect(f"{server}/postgres") as connection:
                connection.execute("select 1")
            return
        except psycopg.OperationalError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


@pytest.fixture
def empty_database_url(postgres_server: str) -> str:
    """A database that exists and that no LightlyStudio has migrated."""
    name = f"test_{uuid4().hex}"
    with psycopg.connect(f"{postgres_server}/postgres", autocommit=True) as connection:
        connection.execute(f'create database "{name}"')
    return f"{postgres_server}/{name}"


@pytest.fixture
def database_url(postgres_server: str) -> Iterator[str]:
    """A migrated LightlyStudio database, with `db_manager` connected to it."""
    url = f"{postgres_server}/test_{uuid4().hex}"
    db_manager.close()
    db_manager.connect(db_url=url)
    yield url
    db_manager.close()
