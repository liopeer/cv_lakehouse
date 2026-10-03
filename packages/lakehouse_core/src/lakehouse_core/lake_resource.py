#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The resource that gives every asset the lake root."""

from pathlib import Path

import dagster as dg

from lakehouse_core.lake_paths import LakePaths


class LakeResource(dg.ConfigurableResource):
    """Expose the settings to the assets, and let the UI override them per run."""

    root: str
    download_workers: int
    request_timeout_seconds: float

    @property
    def paths(self) -> LakePaths:
        return LakePaths(Path(self.root))
