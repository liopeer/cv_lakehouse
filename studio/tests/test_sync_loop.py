#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import pytest

from cv_lakehouse_studio.sync_loop import SchemaMismatchError, reject_schema_mismatch


def test_a_migrated_database_passes(database_url: str) -> None:
    reject_schema_mismatch(database_url)
    reject_schema_mismatch(
        database_url.replace("postgresql://", "postgresql+psycopg://")
    )


def test_a_database_that_no_server_migrated_is_rejected(
    empty_database_url: str,
) -> None:
    with pytest.raises(expected_exception=SchemaMismatchError, match="not migrated"):
        reject_schema_mismatch(empty_database_url)


def test_a_database_at_another_migration_is_rejected(
    database_url: str, empty_database_url: str
) -> None:
    import psycopg

    with psycopg.connect(database_url, autocommit=True) as connection:
        connection.execute("update alembic_version set version_num = 'older'")
    with pytest.raises(expected_exception=SchemaMismatchError, match="older"):
        reject_schema_mismatch(database_url)
