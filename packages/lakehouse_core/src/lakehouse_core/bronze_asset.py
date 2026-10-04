#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The bronze asset of one source: the dataset, downloaded or linked to a full copy.

A linked copy stays where it is, on a local disk or in an object store. The manifest in
the lake records its location, and every layer above reads bronze from there.
"""

# Dagster resolves the config and resource annotations at runtime, so this module
# must not postpone its annotations.

from pathlib import Path

import dagster as dg
from fsspec.core import split_protocol
from upath import UPath

from lakehouse_core.bronze_manifest import BRONZE_MANIFEST, BronzeManifest, BronzeMode
from lakehouse_core.lake_resource import LakeResource
from lakehouse_core.lake_store import LakeStore, is_local
from lakehouse_core.manifest_files import write_manifest
from lakehouse_core.published_files import (
    download_published_files,
    reject_incomplete_copy,
)
from lakehouse_core.published_source import PublishedSource


class BronzeConfig(dg.Config):
    """Set source_dir to link a complete copy instead of downloading it.

    It is a local directory, or a URL such as s3://datasets/coco2017.
    """

    source_dir: str | None = None


def resolve_copy_dir(store: LakeStore, source_dir: str) -> UPath:
    """Return the directory of a linked copy. A relative local path is from the cwd."""
    protocol, path = split_protocol(source_dir)
    if protocol is None:
        source_dir = str(Path(path).expanduser().resolve())
    copy_dir = store.resolve(source_dir)
    if not copy_dir.is_dir():
        raise NotADirectoryError(f"source_dir is not a directory: {copy_dir}")
    return copy_dir


def _remove_old_symlink(bronze_dir: UPath) -> None:
    """Remove the symlink by which a lake before linked a copy."""
    if is_local(bronze_dir) and Path(str(bronze_dir)).is_symlink():
        Path(str(bronze_dir)).unlink()


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
        _remove_old_symlink(bronze_dir)
        if config.source_dir is not None:
            copy_dir = resolve_copy_dir(store=lake.store, source_dir=config.source_dir)
            context.log.info(f"Linking {name} to {copy_dir}")
            mode = BronzeMode.LINK
        else:
            copy_dir = bronze_dir
            copy_dir.mkdir(parents=True, exist_ok=True)
            download_published_files(
                published_files=publication.published_files,
                bronze_dir=copy_dir,
                workers=lake.download_workers,
                timeout=lake.request_timeout_seconds,
                log=context.log,
            )
            mode = BronzeMode.DOWNLOAD
        reject_incomplete_copy(
            bronze_dir=copy_dir, published_files=publication.published_files
        )

        base = BronzeManifest(
            dataset=name,
            mode=mode,
            homepage=publication.homepage,
            license=publication.license,
            commercial_use=publication.commercial_use,
            path=lake.store.location(copy_dir),
            published_files=list(publication.published_files),
        )
        manifest = source.describe_bronze_copy(bronze_dir=copy_dir, base=base)
        write_manifest(path=bronze_dir / BRONZE_MANIFEST, manifest=manifest)
        return dg.MaterializeResult(
            metadata={
                "mode": mode.value,
                "path": str(copy_dir),
                **manifest.materialization_metadata(),
                "license": publication.license,
                "commercial_use": publication.commercial_use,
            }
        )

    return _bronze
