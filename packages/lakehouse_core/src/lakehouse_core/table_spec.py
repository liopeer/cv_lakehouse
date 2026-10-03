#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The description of one table that a domain writes to a layer."""

from dataclasses import dataclass
from enum import StrEnum

import pyarrow as pa


class Layer(StrEnum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass(frozen=True)
class TableSpec:
    """Name a table, its layer, its schema and the columns that identify a row."""

    name: str
    layer: Layer
    schema: pa.Schema
    key: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError(f"{self.name} needs at least one key column")
        missing = [column for column in self.key if column not in self.schema.names]
        if missing:
            raise ValueError(f"{self.name} has no key columns {missing}")
        nullable = [c for c in self.key if self.schema.field(c).nullable]
        if nullable:
            raise ValueError(f"{self.name} has nullable key columns {nullable}")
