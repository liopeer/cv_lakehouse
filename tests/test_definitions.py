#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from dagster import Definitions

from cv_lakehouse.definitions import defs


def test_defs_load() -> None:
    assert isinstance(defs(), Definitions)
