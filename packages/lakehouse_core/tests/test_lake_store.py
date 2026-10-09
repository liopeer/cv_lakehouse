#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A DuckDB connection of the lake takes its limits from the settings."""

from pathlib import Path

from lakehouse_core.lake_store import LakeStore


def test_a_connection_takes_the_threads_and_the_memory_limit(tmp_path: Path) -> None:
    store = LakeStore(
        root=str(tmp_path),
        storage_options={},
        duckdb_threads=3,
        duckdb_memory_limit="1GB",
    )

    with store.duckdb() as connection:
        row = connection.execute(
            "select current_setting('threads'), current_setting('memory_limit')"
        ).fetchone()

    assert row is not None
    threads, memory_limit = row
    assert threads == 3
    # DuckDB shows 1 GB, which is 10^9 bytes, in GiB.
    assert memory_limit == "953.6 MiB"
