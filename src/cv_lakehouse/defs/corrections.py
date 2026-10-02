#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Bronze assets for the corrections that curators make in LightlyStudio.

LightlyStudio is outside the lakehouse, so this asset has no upstream asset, as a
download has none. The graph stays acyclic, although the corrections are made on gold.
"""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

import dagster as dg
import httpx

from cv_lakehouse.defs.resources import LakeResource
from cv_lakehouse.manifests import (
    CORRECTIONS_MANIFEST,
    CorrectionsManifest,
    read_manifest,
    write_manifest,
)
from cv_lakehouse.settings import LakePaths
from cv_lakehouse.sources.correction_snapshots import fetch_new_snapshots
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME

NO_SNAPSHOT_DATA_VERSION = "none"


def build_corrections_key(name: str) -> dg.AssetKey:
    return dg.AssetKey(["bronze", f"{name}_corrections"])


def read_corrections_manifest(paths: LakePaths, name: str) -> CorrectionsManifest:
    """Return the manifest, or an empty one before the asset ran."""
    path = paths.corrections_dir(name) / CORRECTIONS_MANIFEST
    if not path.exists():
        return CorrectionsManifest(dataset=name)
    return read_manifest(path=path, model=CorrectionsManifest)


def open_export_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def build_corrections_asset(name: str) -> dg.AssetsDefinition:
    @dg.asset(
        key=build_corrections_key(name),
        group_name="bronze",
        kinds={"file"},
        description=(
            f"The corrections that curators made to {name} in LightlyStudio, as "
            "immutable snapshots. Silver applies the latest one."
        ),
    )
    def _corrections(
        context: dg.AssetExecutionContext, lake: LakeResource
    ) -> dg.MaterializeResult:
        manifest = read_corrections_manifest(paths=lake.paths, name=name)
        if lake.studio_export_url is None:
            context.log.warning(
                "CV_LAKEHOUSE_STUDIO_EXPORT_URL is unset, so this run fetches no "
                "correction."
            )
        else:
            with open_export_client(lake.request_timeout_seconds) as client:
                manifest = CorrectionsManifest(
                    dataset=name,
                    snapshots=fetch_new_snapshots(
                        client=client,
                        export_url=lake.studio_export_url,
                        dataset=name,
                        corrections_dir=lake.paths.corrections_dir(name),
                        stored_snapshots=manifest.snapshots,
                        log=context.log,
                    ),
                )
        write_manifest(
            path=lake.paths.corrections_dir(name) / CORRECTIONS_MANIFEST,
            manifest=manifest,
        )
        latest = manifest.snapshots[-1] if manifest.snapshots else None
        return dg.MaterializeResult(
            # Silver is stale when a new snapshot lands, and not on every run.
            data_version=dg.DataVersion(
                NO_SNAPSHOT_DATA_VERSION if latest is None else latest.snapshot_id
            ),
            metadata={
                "num_snapshots": len(manifest.snapshots),
                "latest_snapshot": "none" if latest is None else latest.snapshot_id,
                "dagster/row_count": 0 if latest is None else latest.row_count,
            },
        )

    return _corrections


defs = dg.Definitions(assets=[build_corrections_asset(name) for name in SOURCE_BY_NAME])
