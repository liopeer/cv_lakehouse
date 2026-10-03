#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The resource that gives every CV asset the lake root and its services."""

from pathlib import Path
from typing import Self

import dagster as dg

from lakehouse_core.lake_resource import LakeResource
from lakehouse_cv.settings import CvLakePaths, CvSettings


class CvLakeResource(LakeResource):
    triton_url: str | None = None
    studio_export_url: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        settings = CvSettings()
        return cls(
            root=str(settings.root),
            download_workers=settings.download_workers,
            request_timeout_seconds=settings.request_timeout_seconds,
            triton_url=settings.triton_url,
            studio_export_url=settings.studio_export_url,
        )

    @property
    def paths(self) -> CvLakePaths:
        return CvLakePaths(Path(self.root))


defs = dg.Definitions(resources={"lake": CvLakeResource.from_env()})
