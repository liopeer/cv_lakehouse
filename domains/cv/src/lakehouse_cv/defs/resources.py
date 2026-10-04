#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The resource that gives every CV asset the lake root and its services."""

import os
from typing import Self

import dagster as dg

from lakehouse_core.lake_resource import LakeResource
from lakehouse_cv.settings import CvLakePaths, CvSettings

STORAGE_OPTIONS_VARIABLE = "CV_LAKEHOUSE_STORAGE_OPTIONS"


class CvLakeResource(LakeResource):
    triton_url: str | None = None
    studio_export_url: str | None = None

    @classmethod
    def from_env(cls) -> Self:
        settings = CvSettings()
        return cls(
            root=settings.root,
            # The value holds keys, so it reaches the resource as an EnvVar, which the
            # UI shows by name only.
            storage_options=(
                dg.EnvVar(STORAGE_OPTIONS_VARIABLE)
                if STORAGE_OPTIONS_VARIABLE in os.environ
                else "{}"
            ),
            download_workers=settings.download_workers,
            request_timeout_seconds=settings.request_timeout_seconds,
            triton_url=settings.triton_url,
            studio_export_url=settings.studio_export_url,
        )

    @property
    def paths(self) -> CvLakePaths:
        return CvLakePaths(self.store.root)


defs = dg.Definitions(resources={"lake": CvLakeResource.from_env()})
