#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from lakehouse_core.lake_resource import LakeResource


def test_store_applies_the_duckdb_limits(tmp_path: Path) -> None:
    lake = LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=1,
        request_timeout_seconds=5.0,
        duckdb_threads=2,
        duckdb_memory_limit="1GB",
    )

    with lake.store.duckdb() as connection:
        threads, memory_limit = connection.execute(
            "select current_setting('threads'), current_setting('memory_limit')"
        ).fetchone() or (None, None)

    assert threads == 2
    assert memory_limit == "953.6 MiB"
