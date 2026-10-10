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
from upath import UPath

from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_cv.contract.manifests import (
    CORRECTIONS_MANIFEST,
    CorrectionsManifest,
    EventMarker,
)
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.settings import CvLakePaths
from lakehouse_cv.sources.correction_events import fetch_new_event_files
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME

NO_EVENT_DATA_VERSION = "none"


def build_corrections_key(name: str) -> dg.AssetKey:
    return dg.AssetKey(["bronze", f"{name}_corrections"])


def read_corrections_manifest(paths: CvLakePaths, name: str) -> CorrectionsManifest:
    """Return the manifest, or an empty one before the asset ran."""
    path = paths.corrections_dir(name) / CORRECTIONS_MANIFEST
    if not path.exists():
        return CorrectionsManifest(dataset=name)
    return read_manifest(path=path, model=CorrectionsManifest)


def list_event_paths(
    *, paths: CvLakePaths, manifest: CorrectionsManifest, last_event: EventMarker | None
) -> list[UPath]:
    """Return the event files in the order of bronze, up to the one of `last_event`."""
    if last_event is None:
        return []
    files: list[UPath] = []
    for event_file in manifest.event_files:
        files.append(paths.corrections_dir(manifest.dataset) / event_file.file.path)
        if (event_file.chain_id, event_file.last_log_sequence) == (
            last_event.chain_id,
            last_event.log_sequence,
        ):
            return files
    raise ValueError(f"Bronze holds no event file that ends at {last_event}.")


def open_export_client(timeout: float) -> httpx.Client:
    return httpx.Client(timeout=timeout, follow_redirects=True)


def build_corrections_asset(name: str) -> dg.AssetsDefinition:
    @dg.asset(
        key=build_corrections_key(name),
        group_name="bronze",
        kinds={"file"},
        description=(
            f"The edits that curators made to {name} in LightlyStudio, as immutable "
            "event files. Silver folds every one of them."
        ),
    )
    def _corrections(
        context: dg.AssetExecutionContext, lake: CvLakeResource
    ) -> dg.MaterializeResult:
        manifest = read_corrections_manifest(paths=lake.paths, name=name)
        if lake.studio_export_url is None:
            context.log.warning(
                "CV_LAKEHOUSE_STUDIO_EXPORT_URL is unset, so this run fetches no "
                "correction."
            )
        else:
            with open_export_client(lake.request_timeout_seconds) as client:
                manifest = manifest.model_copy(
                    update={
                        "event_files": fetch_new_event_files(
                            client=client,
                            export_url=lake.studio_export_url,
                            dataset=name,
                            corrections_dir=lake.paths.corrections_dir(name),
                            stored_files=manifest.event_files,
                            log=context.log,
                        )
                    }
                )
        write_manifest(
            path=lake.paths.corrections_dir(name) / CORRECTIONS_MANIFEST,
            manifest=manifest,
        )
        latest = manifest.event_files[-1] if manifest.event_files else None
        return dg.MaterializeResult(
            # Silver is stale when a new event file lands, and not on every run.
            data_version=dg.DataVersion(
                NO_EVENT_DATA_VERSION if latest is None else latest.event_file_id
            ),
            metadata={
                "num_event_files": len(manifest.event_files),
                "latest_event_file": "none" if latest is None else latest.event_file_id,
                "dagster/row_count": sum(
                    event_file.row_count for event_file in manifest.event_files
                ),
            },
        )

    return _corrections


defs = dg.Definitions(assets=[build_corrections_asset(name) for name in SOURCE_BY_NAME])
