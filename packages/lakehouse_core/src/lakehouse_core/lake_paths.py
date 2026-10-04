#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from upath import UPath


class LakePaths:
    """Resolve the directory of a dataset in each layer."""

    def __init__(self, root: UPath) -> None:
        self.root = root

    def bronze_dir(self, name: str) -> UPath:
        return self.root / "bronze" / name

    def silver_dir(self, name: str) -> UPath:
        return self.root / "silver" / name

    def gold_dir(self) -> UPath:
        return self.root / "gold"
