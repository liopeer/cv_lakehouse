#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The resource that gives every asset the lake root."""

import json

import dagster as dg

from lakehouse_core.lake_paths import LakePaths
from lakehouse_core.lake_store import LakeStore


class LakeResource(dg.ConfigurableResource):
    """Expose the settings to the assets, and let the UI override them per run."""

    root: str
    # JSON. It holds keys, so a domain sets it from a `dg.EnvVar`, which the UI does not
    # show.
    storage_options: str = "{}"
    download_workers: int
    request_timeout_seconds: float

    @property
    def store(self) -> LakeStore:
        return LakeStore(
            root=self.root, storage_options=json.loads(self.storage_options)
        )

    @property
    def paths(self) -> LakePaths:
        return LakePaths(self.store.root)
