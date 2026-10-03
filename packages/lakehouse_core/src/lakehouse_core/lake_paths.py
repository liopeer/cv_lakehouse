#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path


class LakePaths:
    """Resolve the directory of a dataset in each layer."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def bronze_dir(self, name: str) -> Path:
        return self.root / "bronze" / name

    def silver_dir(self, name: str) -> Path:
        return self.root / "silver" / name

    def gold_dir(self) -> Path:
        return self.root / "gold"
