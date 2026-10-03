#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Bronze assets: one per CV dataset, built by the bronze asset of core."""

import dagster as dg

from lakehouse_core.bronze_asset import build_bronze_asset
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME

defs = dg.Definitions(
    assets=[build_bronze_asset(source) for source in SOURCE_BY_NAME.values()]
)
