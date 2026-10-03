#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Settings and path layout for the CV lake."""

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from lakehouse_core.lake_paths import LakePaths
from lakehouse_core.lake_settings import LakeSettings


class CvSettings(LakeSettings):
    """Read the configuration from the environment.

    Every field maps to an environment variable with the prefix CV_LAKEHOUSE_. For
    example, CV_LAKEHOUSE_ROOT sets the root.
    """

    model_config = SettingsConfigDict(
        env_prefix="CV_LAKEHOUSE_",
        env_file=".env",
        extra="ignore",
    )

    # `host:port` of the MobileCLIP gRPC server in `triton/`, such as localhost:8011.
    # Unset means silver writes no embedding, which is what lets a machine with no
    # server still build the layer.
    triton_url: str | None = None
    # The export of `studio/`, such as http://studio-sync:8002. Unset means that the
    # corrections assets fetch nothing, so a lake with no LightlyStudio still builds.
    studio_export_url: str | None = None


class CvLakePaths(LakePaths):
    """Add the directories that only the CV lake has."""

    def corrections_dir(self, name: str) -> Path:
        return self.root / "bronze" / f"{name}_corrections"

    def gold_version_dir(self, version: int) -> Path:
        return self.gold_dir() / "versions" / str(version)

    def gold_release_dir(self, release: int) -> Path:
        return self.gold_dir() / "releases" / f"{release:04d}"
