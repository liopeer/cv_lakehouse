#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LakeSettings(BaseSettings):
    """Read the settings that every domain shares from the environment.

    A domain subclasses this and sets its own `env_prefix`, so two domains on one
    machine read different variables.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # A local directory, or a URL such as s3://lake/cv. See `LakeStore`.
    root: str = "data"
    # The fsspec options of the root, as JSON, such as the endpoint and the keys of S3.
    storage_options: dict[str, Any] = Field(default_factory=dict)
    download_workers: int = Field(default=16, ge=1, le=64)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    # The threads and the memory of one DuckDB connection, such as 4 and "2GB". Unset
    # means the DuckDB default: every core of the machine, and 80% of its memory.
    duckdb_threads: int | None = Field(default=None, ge=1)
    duckdb_memory_limit: str | None = None
    # The lock file of the disk lease, such as /var/lib/lakehouse/disk.lock. Set it only
    # for a disk that suffers from contention, such as an HDD. See ADR 0012.
    disk_lease_path: str | None = None
