#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Materialize assets on a fixture lake. Nothing downloads."""

from collections.abc import Sequence
from pathlib import Path

import dagster as dg

from cv_lakehouse.defs.bronze import build_bronze_asset
from cv_lakehouse.defs.resources import LakeResource

DATASETS = ("open_images", "wider_face", "pp4av")


def materialize_bronze_links(lake: LakeResource, sources: dict[str, Path]) -> None:
    for name in DATASETS:
        asset = build_bronze_asset(name)
        result = dg.materialize(
            assets=[asset],
            resources={"lake": lake},
            run_config=dg.RunConfig(
                ops={asset.op.name: {"config": {"source_dir": str(sources[name])}}}
            ),
        )
        assert result.success


def materialize_assets(
    lake: LakeResource, assets: Sequence[dg.AssetsDefinition | dg.AssetChecksDefinition]
) -> dg.ExecuteInProcessResult:
    result = dg.materialize(assets=assets, resources={"lake": lake})
    assert result.success
    return result
