#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

from collections.abc import Mapping
from dataclasses import dataclass

from lakehouse_core.fingerprints import sha256_fingerprint
from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.split_roles import SplitRole


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset is, and how its categories map onto the class registry.

    In category_map the key is the name the source gives a category. The value is a
    canonical class name. A source category that the map omits falls to default_class,
    so no box is ever lost to a gap in the map.

    Point default_class at `other` on a source that boxes more than the class registry
    names.
    Silver then keeps the box and loses the label, and its consumer decides what the
    class is worth.

    split_roles gives every split its role in gold. Two splits can share a role.
    """

    name: str
    homepage: str
    license: str
    commercial_use: bool
    splits: tuple[str, ...]
    split_roles: Mapping[str, SplitRole]
    category_map: Mapping[str, str]
    default_class: str
    notes: str = ""

    def __post_init__(self) -> None:
        targets = set(self.category_map.values()) | {self.default_class}
        unknown = sorted(targets - CanonicalClass.all_class_names())
        if unknown:
            raise ValueError(f"{self.name} maps onto unknown classes {unknown}")
        if set(self.split_roles) != set(self.splits):
            raise ValueError(f"{self.name} needs one role for each of {self.splits}")

    @property
    def category_map_sha256_fingerprint(self) -> str:
        """A map change marks this dataset's silver asset stale."""
        pairs = sorted(f"{key}={value}" for key, value in self.category_map.items())
        return sha256_fingerprint([self.name, self.default_class, *pairs])

    @property
    def split_roles_sha256_fingerprint(self) -> str:
        """A role change marks gold stale."""
        pairs = sorted(f"{split}={role}" for split, role in self.split_roles.items())
        return sha256_fingerprint([self.name, *pairs])
