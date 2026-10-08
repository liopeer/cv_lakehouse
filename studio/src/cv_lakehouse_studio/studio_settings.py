#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Settings of the sync, read from the environment."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cv_lakehouse_studio.studio_datasets import DEFAULT_MAX_IMAGES_PER_DATASET


class StudioSettings(BaseSettings):
    """Every field but the database maps to CV_LAKEHOUSE_STUDIO_<FIELD>.

    The database is the one that the LightlyStudio server reads, under the same name.
    """

    model_config = SettingsConfigDict(
        env_prefix="CV_LAKEHOUSE_STUDIO_", env_file=".env", extra="ignore"
    )

    database_url: str = Field(validation_alias="LIGHTLY_STUDIO_DATABASE_URL")
    # The gold API of the lakehouse, such as http://lakehouse:8000.
    gold_api_url: str
    # Where the lake root is from this service: a directory that holds the mount, or an
    # fsspec URL such as s3://bucket/lake. Gold names a pixel relative to it.
    image_base: str
    sync_interval_seconds: float = Field(default=3600.0, gt=0)
    export_port: int = 8002
    # LightlyStudio holds about 1M images per dataset. A larger gold dataset is split.
    max_images_per_dataset: int = Field(default=DEFAULT_MAX_IMAGES_PER_DATASET, gt=0)
    request_timeout_seconds: float = Field(default=300.0, gt=0)
    # The gold API rejects a page of more than 100000 rows. An embedding row holds a
    # vector. A large page of embeddings runs the gold API out of memory.
    rows_per_page: int = Field(default=50_000, gt=0, le=100_000)
    embedding_rows_per_page: int = Field(default=5_000, gt=0, le=100_000)
