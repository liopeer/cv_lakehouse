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
from lakehouse_cv.transforms.layer_builds import (
    embeddings_build_dir,
    read_embeddings_manifest,
    read_silver_manifest,
    silver_build_dir,
)

DATASETS = ("open_images", "wider_face", "pp4av")

# One instance per test, so a run sees what an earlier run of the test materialized.
# The fixture `dagster_instance` sets it.
_instance: dg.DagsterInstance | None = None


def use_instance(instance: dg.DagsterInstance | None) -> None:
    global _instance
    _instance = instance


def run_assets(
    *,
    lake: CvLakeResource,
    assets: Sequence[dg.AssetsDefinition | dg.AssetChecksDefinition],
    run_config: dg.RunConfig | None = None,
    raise_on_error: bool = True,
) -> dg.ExecuteInProcessResult:
    return dg.materialize(
        assets=assets,
        resources={"lake": lake},
        run_config=run_config,
        raise_on_error=raise_on_error,
        instance=_instance,
    )


def materialize_bronze_links(
    lake: CvLakeResource, sources: Mapping[str, Path | UPath]
) -> None:
    for name in DATASETS:
        asset = build_bronze_asset(SOURCE_BY_NAME[name])
        result = run_assets(
            lake=lake,
            assets=[asset],
            run_config=dg.RunConfig(
                ops={asset.op.name: {"config": {"source_dir": str(sources[name])}}}
            ),
        )
        assert result.success


def materialize_assets(
    lake: CvLakeResource,
    assets: Sequence[dg.AssetsDefinition | dg.AssetChecksDefinition],
) -> dg.ExecuteInProcessResult:
    result = run_assets(lake=lake, assets=assets)
    assert result.success
    return result


def find_silver_files(lake: CvLakeResource, name: str) -> UPath:
    """Return the directory of the silver build that the manifest names."""
    return silver_build_dir(
        paths=lake.paths, manifest=read_silver_manifest(paths=lake.paths, name=name)
    )


def find_embedding_files(lake: CvLakeResource, name: str) -> UPath:
    manifest = read_embeddings_manifest(paths=lake.paths, name=name)
    assert manifest is not None
    return embeddings_build_dir(paths=lake.paths, manifest=manifest)


def read_silver_build_ids(lake: CvLakeResource, names: Sequence[str]) -> dict[str, str]:
    """Return the build that each silver manifest names, as Dagster would pass it."""
    return {
        name: read_silver_manifest(paths=lake.paths, name=name).build_id
        for name in names
    }
