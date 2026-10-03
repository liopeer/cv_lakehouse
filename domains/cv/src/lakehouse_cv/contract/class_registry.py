#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The canonical class vocabulary that every layer shares.

These are our classes, not any dataset's. A source maps its own categories onto these
names, and a source category that no name covers falls to the source's `default_class`.
See `DatasetSpec.category_map`.

This module is the source of truth. An id is written here once and never shifts. To add
a class, append a member with the next free value. Bronze holds every annotation a
dataset publishes, so a wider vocabulary costs a silver rebuild, not a download.
"""

from enum import IntEnum, unique
from typing import Self

from lakehouse_core.fingerprints import sha256_fingerprint


@unique
class CanonicalClass(IntEnum):
    """One class of the shared vocabulary. The value is the class id silver writes."""

    FACE = 0
    LICENSE_PLATE = 1
    HEAD = 2
    PERSON = 3
    TEXT = 4
    OTHER = 5

    @property
    def class_name(self) -> str:
        return self.name.lower()

    @classmethod
    def from_class_name(cls, class_name: str) -> Self:
        return cls[class_name.upper()]

    @classmethod
    def all_class_names(cls) -> frozenset[str]:
        return frozenset(member.class_name for member in cls)


def class_registry_sha256_fingerprint() -> str:
    return sha256_fingerprint(
        [f"{member.value}={member.class_name}" for member in CanonicalClass]
    )
