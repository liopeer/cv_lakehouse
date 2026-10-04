#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Materialize assets on a fixture lake. Nothing downloads."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import dagster as dg
from upath import UPath

from lakehouse_core.bronze_asset import build_bronze_asset
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME

DATASETS = ("open_images", "wider_face", "pp4av")


def materialize_bronze_links(
    lake: CvLakeResource, sources: Mapping[str, Path | UPath]
) -> None:
    for name in DATASETS:
        asset = build_bronze_asset(SOURCE_BY_NAME[name])
        result = dg.materialize(
            assets=[asset],
            resources={"lake": lake},
            run_config=dg.RunConfig(
                ops={asset.op.name: {"config": {"source_dir": str(sources[name])}}}
            ),
        )
        assert result.success


def materialize_assets(
    lake: CvLakeResource,
    assets: Sequence[dg.AssetsDefinition | dg.AssetChecksDefinition],
) -> dg.ExecuteInProcessResult:
    result = dg.materialize(assets=assets, resources={"lake": lake})
    assert result.success
    return result
