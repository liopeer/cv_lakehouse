#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The resource that gives every asset the lakehouse root."""

from pathlib import Path
from typing import Self

import dagster as dg

from cv_lakehouse.settings import LakePaths, Settings


class LakeResource(dg.ConfigurableResource):
    """Expose the settings to the assets, and let the UI override them per run."""

    root: str
    download_workers: int
    request_timeout_seconds: float
    triton_url: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        settings = Settings()
        return cls(
            root=str(settings.root),
            download_workers=settings.download_workers,
            request_timeout_seconds=settings.request_timeout_seconds,
            triton_url=settings.triton_url,
        )

    @property
    def paths(self) -> LakePaths:
        return LakePaths(Path(self.root))


defs = dg.Definitions(resources={"lake": LakeResource.from_env()})
