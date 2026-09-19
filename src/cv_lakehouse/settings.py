#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Settings and path layout for the lakehouse."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read the configuration from the environment.

    Every field maps to an environment variable with the prefix CV_LAKEHOUSE_. For
    example, CV_LAKEHOUSE_ROOT sets the root.
    """

    model_config = SettingsConfigDict(
        env_prefix="CV_LAKEHOUSE_",
        env_file=".env",
        extra="ignore",
    )

    root: Path = Path("data")
    download_workers: int = Field(default=16, ge=1, le=64)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    # `host:port` of the MobileCLIP gRPC server in `triton/`, such as localhost:8011.
    # Unset means silver writes no embedding, which is what lets a machine with no
    # server still build the layer.
    triton_url: str | None = None


class LakePaths:
    """Resolve the directory of a dataset in each layer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def bronze_dir(self, name: str) -> Path:
        return self.root / "bronze" / name

    def silver_dir(self, name: str) -> Path:
        return self.root / "silver" / name

    def studio_database(self, name: str) -> Path:
        """A LightlyStudio DuckDB file, built from Parquet on demand.

        Not a layer. The DuckDB file is a cache: its schema is whatever the installed
        LightlyStudio creates, it holds an exclusive write lock, and DuckDB offers it no
        migration. Delete it whenever; `tools/studio.py` rebuilds it in seconds.
        """
        return self.root / ".studio" / f"{name}.db"
