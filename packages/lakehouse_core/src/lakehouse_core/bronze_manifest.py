#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The manifest that bronze writes next to a dataset."""

from enum import StrEnum

from pydantic import BaseModel, Field

from lakehouse_core.published_files import PublishedFile

BRONZE_MANIFEST = "_bronze.json"


class BronzeMode(StrEnum):
    """How a bronze dataset got onto disk."""

    DOWNLOAD = "download"
    LINK = "link"


class BronzeManifest(BaseModel):
    """What every domain records about a bronze copy. A domain subclasses it."""

    dataset: str
    mode: BronzeMode
    homepage: str
    license: str
    commercial_use: bool
    path: str
    # Every file the publisher publishes, with its URL and its checksum. A linked copy
    # lists them too: they define what a complete copy is.
    published_files: list[PublishedFile] = Field(default_factory=list)

    def materialization_metadata(self) -> dict[str, str | int | float | list[str]]:
        """Return the domain fields that the bronze asset reports to Dagster."""
        return {}
