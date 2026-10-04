#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The bronze asset of one source: the dataset, downloaded or linked to a full copy."""

# Dagster resolves the config and resource annotations at runtime, so this module
# must not postpone its annotations.

from pathlib import Path

import dagster as dg

from lakehouse_core.bronze_manifest import BRONZE_MANIFEST, BronzeManifest, BronzeMode
from lakehouse_core.lake_resource import LakeResource
from lakehouse_core.lake_store import require_local
from lakehouse_core.manifest_files import write_manifest
from lakehouse_core.published_files import (
    download_published_files,
    reject_incomplete_copy,
)
from lakehouse_core.published_source import PublishedSource


class BronzeConfig(dg.Config):
    """Set source_dir to link a complete copy instead of downloading it."""

    source_dir: str | None = None


def _create_symlink(source_dir: Path, target: Path) -> None:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"{target} exists and is not a symlink. Remove it first.")
    target.symlink_to(target=source_dir, target_is_directory=True)


def build_bronze_asset(source: PublishedSource[BronzeManifest]) -> dg.AssetsDefinition:
    publication = source.publication
    name = publication.name

    @dg.asset(
        key=dg.AssetKey(["bronze", name]),
        group_name="bronze",
        kinds={"file"},
        description=publication.description or f"Raw {name}.",
        metadata={
            "license": publication.license,
            "commercial_use": publication.commercial_use,
            "homepage": dg.MetadataValue.url(publication.homepage),
        },
    )
    def _bronze(
        context: dg.AssetExecutionContext,
        config: BronzeConfig,
        lake: LakeResource,
    ) -> dg.MaterializeResult:
        bronze_dir = lake.paths.bronze_dir(name)
        local_dir = require_local(path=bronze_dir, purpose="Bronze")
        if config.source_dir is not None:
            source_dir = Path(config.source_dir).expanduser().resolve()
            context.log.info(f"Mapping {name} to {source_dir}")
            _create_symlink(source_dir=source_dir, target=local_dir)
            mode = BronzeMode.LINK
        else:
            local_dir.mkdir(parents=True, exist_ok=True)
            download_published_files(
                published_files=publication.published_files,
                bronze_dir=local_dir,
                workers=lake.download_workers,
                timeout=lake.request_timeout_seconds,
                log=context.log,
            )
            mode = BronzeMode.DOWNLOAD
        reject_incomplete_copy(
            bronze_dir=local_dir, published_files=publication.published_files
        )

        base = BronzeManifest(
            dataset=name,
            mode=mode,
            homepage=publication.homepage,
            license=publication.license,
            commercial_use=publication.commercial_use,
            path=lake.store.location(bronze_dir),
            published_files=list(publication.published_files),
        )
        manifest = source.describe_bronze_copy(bronze_dir=bronze_dir, base=base)
        write_manifest(path=bronze_dir / BRONZE_MANIFEST, manifest=manifest)
        return dg.MaterializeResult(
            metadata={
                "mode": mode.value,
                "path": str(bronze_dir),
                **manifest.materialization_metadata(),
                "license": publication.license,
                "commercial_use": publication.commercial_use,
            }
        )

    return _bronze
