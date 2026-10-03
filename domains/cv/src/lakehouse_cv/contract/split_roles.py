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

    @property
    def is_eval(self) -> bool:
        """Whether a result on this role is compared over time."""
        return self is not SplitRole.TRAIN
