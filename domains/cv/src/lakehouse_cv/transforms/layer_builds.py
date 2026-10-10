#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Write each run of a layer to a new directory, and then point the manifest at it.

An object store has no atomic rename of a directory. So a run writes its files under
`builds/<random id>/`, and replaces the manifest only when the files are whole and
checked. A reader takes the build that the manifest names, and never sees a half
written run. A run that fails leaves the last good build in place.
"""

from __future__ import annotations

import json
import uuid

from upath import UPath

from lakehouse_core.lake_files import remove_tree, replace_file
from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_cv.contract.manifests import (
    EMBEDDINGS_MANIFEST,
    SILVER_MANIFEST,
    BuildManifest,
    EmbeddingsManifest,
    SilverManifest,
    build_dir,
    builds_dir,
)
from lakehouse_cv.settings import CvLakePaths


def create_build_id() -> str:
    return uuid.uuid4().hex


def read_build_manifest[T: BuildManifest](path: UPath, model: type[T]) -> T | None:
    return read_manifest(path=path, model=model) if path.exists() else None


def read_silver_manifest(paths: CvLakePaths, name: str) -> SilverManifest:
    path = paths.silver_dir(name) / SILVER_MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"Silver {name} is not materialized: no {path}")
    return read_manifest(path=path, model=SilverManifest)


def silver_build_dir(paths: CvLakePaths, manifest: SilverManifest) -> UPath:
    return build_dir(
        layer_dir=paths.silver_dir(manifest.dataset), build_id=manifest.build_id
    )


def read_embeddings_manifest(
    paths: CvLakePaths, name: str
) -> EmbeddingsManifest | None:
    return read_build_manifest(
        path=paths.embeddings_dir(name) / EMBEDDINGS_MANIFEST, model=EmbeddingsManifest
    )


def embeddings_build_dir(paths: CvLakePaths, manifest: EmbeddingsManifest) -> UPath:
    return build_dir(
        layer_dir=paths.embeddings_dir(manifest.dataset), build_id=manifest.build_id
    )


def publish_build(*, manifest_path: UPath, manifest: BuildManifest) -> None:
    """Replace the manifest, and remove every build but this one and the one before.

    A reader that read the old manifest just before the swap still finds its files.
    """
    previous = _read_build_id(manifest_path)
    staged = manifest_path.with_name(manifest_path.name + ".staged")
    write_manifest(path=staged, manifest=manifest)
    replace_file(source=staged, target=manifest_path)
    kept = {manifest.build_id} | (set() if previous is None else {previous})
    remove_builds(layer_dir=manifest_path.parent, kept=kept)


def remove_builds(*, layer_dir: UPath, kept: set[str]) -> None:
    directory = builds_dir(layer_dir)
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.name not in kept:
            remove_tree(path)


def discard_build(*, layer_dir: UPath, build_id: str) -> None:
    remove_tree(build_dir(layer_dir=layer_dir, build_id=build_id))


def _read_build_id(manifest_path: UPath) -> str | None:
    """Return the build that a manifest names, or None for a manifest with no build.

    A manifest of the layout before builds names none, and its run is replaced anyway.
    """
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text()).get("build_id")
