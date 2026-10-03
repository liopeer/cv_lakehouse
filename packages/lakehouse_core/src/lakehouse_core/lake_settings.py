#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LakeSettings(BaseSettings):
    """Read the settings that every domain shares from the environment.

    A domain subclasses this and sets its own `env_prefix`, so two domains on one
    machine read different variables.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    root: Path = Path("data")
    download_workers: int = Field(default=16, ge=1, le=64)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
