#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The contract between a domain's source and the bronze asset of core."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lakehouse_core.bronze_manifest import BronzeManifest
from lakehouse_core.published_files import PublishedFile


@dataclass(frozen=True)
class Publication:
    """What a publisher distributes, and under which terms."""

    name: str
    homepage: str
    license: str
    commercial_use: bool
    description: str
    published_files: tuple[PublishedFile, ...]


class PublishedSource[M: BronzeManifest](Protocol):
    """A dataset that bronze downloads or links, and then describes for its domain."""

    @property
    def publication(self) -> Publication: ...

    def describe_bronze_copy(self, *, bronze_dir: Path, base: BronzeManifest) -> M:
        """Add the domain fields to `base`. Raise if the copy holds nothing usable."""
        ...
