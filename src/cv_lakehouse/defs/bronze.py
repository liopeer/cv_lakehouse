#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Bronze assets: the raw dataset, downloaded or linked to a complete copy."""

# Dagster resolves the config and resource annotations at runtime, so this module
# must not postpone its annotations.

from pathlib import Path

import dagster as dg

from cv_lakehouse.defs.resources import LakeResource
from cv_lakehouse.manifests import (
    BRONZE_MANIFEST,
    BronzeManifest,
    BronzeMode,
    write_manifest,
)
from cv_lakehouse.sources.base import BronzeSource
from cv_lakehouse.sources.published_files import (
    download_published_files,
    reject_incomplete_copy,
)
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME


class BronzeConfig(dg.Config):
    """Set source_dir to link a complete copy instead of downloading it."""

    source_dir: str | None = None


def detect_materialized_splits(
    source: BronzeSource, bronze_dir: Path
) -> tuple[str, ...]:
    """Report the splits that are really on disk, not the splits the source can make."""
    found = []
    for split in source.spec.splits:
        try:
            root = source.image_root(bronze_dir=bronze_dir, split=split)
        except FileNotFoundError:
            continue
        if root.is_dir():
            found.append(split)
    return tuple(found)


def _create_symlink(source_dir: Path, target: Path) -> None:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {source_dir}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        raise FileExistsError(f"{target} exists and is not a symlink. Remove it first.")
    target.symlink_to(target=source_dir, target_is_directory=True)


def build_bronze_asset(name: str) -> dg.AssetsDefinition:
    source = SOURCE_BY_NAME[name]
    spec = source.spec

    @dg.asset(
        key=dg.AssetKey(["bronze", name]),
        group_name="bronze",
        kinds={"file"},
        description=spec.notes or f"Raw {name}.",
        metadata={
            "license": spec.license,
            "commercial_use": spec.commercial_use,
            "homepage": dg.MetadataValue.url(spec.homepage),
        },
    )
    def _bronze(
        context: dg.AssetExecutionContext,
        config: BronzeConfig,
        lake: LakeResource,
    ) -> dg.MaterializeResult:
        bronze_dir = lake.paths.bronze_dir(name)
        if config.source_dir is not None:
            # Symlink an existing copy of the dataset to the bronze directory.
            source_dir = Path(config.source_dir).expanduser().resolve()
            context.log.info(f"Mapping {name} to {source_dir}")
            _create_symlink(source_dir=source_dir, target=bronze_dir)
            mode = BronzeMode.LINK
        else:
            bronze_dir.mkdir(parents=True, exist_ok=True)
            download_published_files(
                published_files=source.published_files,
                bronze_dir=bronze_dir,
                workers=lake.download_workers,
                timeout=lake.request_timeout_seconds,
                log=context.log,
            )
            mode = BronzeMode.DOWNLOAD
        reject_incomplete_copy(
            bronze_dir=bronze_dir, published_files=source.published_files
        )

        # Probe the bronze directory to see which splits were materialized.
        splits = detect_materialized_splits(source=source, bronze_dir=bronze_dir)
        if not splits:
            raise RuntimeError(f"No split materialized for {name} in {bronze_dir}")

        image_roots = {
            split: str(source.image_root(bronze_dir=bronze_dir, split=split))
            for split in splits
        }
        manifest = BronzeManifest(
            dataset=name,
            mode=mode,
            homepage=spec.homepage,
            license=spec.license,
            commercial_use=spec.commercial_use,
            path=str(bronze_dir),
            splits=list(splits),
            image_roots=image_roots,
            published_files=list(source.published_files),
        )
        write_manifest(path=bronze_dir / BRONZE_MANIFEST, manifest=manifest)
        return dg.MaterializeResult(
            metadata={
                "mode": mode.value,
                "path": str(bronze_dir),
                "splits": list(splits),
                "license": spec.license,
                "commercial_use": spec.commercial_use,
            }
        )

    return _bronze


defs = dg.Definitions(assets=[build_bronze_asset(name) for name in SOURCE_BY_NAME])
