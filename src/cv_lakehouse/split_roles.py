#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""What a split is for. A source names its splits as its publisher does."""

from enum import StrEnum


class SplitRole(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
