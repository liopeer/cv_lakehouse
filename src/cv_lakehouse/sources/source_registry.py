#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The registry of every bronze source."""

from __future__ import annotations

from cv_lakehouse.sources.base import BronzeSource, DatasetSpec
from cv_lakehouse.sources.open_images import OpenImagesSource
from cv_lakehouse.sources.pp4av import PP4AVSource
from cv_lakehouse.sources.wider_face import WiderFaceSource

SOURCE_BY_NAME: dict[str, BronzeSource] = {
    source.spec.name: source
    for source in (OpenImagesSource(), WiderFaceSource(), PP4AVSource())
}

SPEC_BY_NAME: dict[str, DatasetSpec] = {
    name: source.spec for name, source in SOURCE_BY_NAME.items()
}
