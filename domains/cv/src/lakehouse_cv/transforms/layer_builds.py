#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Write each run of a layer to a new directory, and then point the manifest at it.

An object store has no atomic rename of a directory. So a run writes its files under
`builds/<id>/`, and replaces the manifest only when the files are whole and checked. A
reader takes the build that the manifest names, and never sees a half written run. A
run that fails leaves the last good build in place.

The id is a digest of the code version and the inputs. A run whose build is already
whole writes nothing, and points the manifest at it. A whole build is never rewritten,
so no run writes into the files of a reader.

The manifest beside `builds/` is for a reader outside Dagster, such as the gold API. A
downstream asset reads the build that Dagster recorded for its upstream, by its id, from
the copy of the manifest inside the build.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from upath import UPath

from lakehouse_core.lake_files import remove_tree, replace_file
from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_cv.contract.manifests import (
    BUILD_MANIFEST,
    EMBEDDINGS_MANIFEST,
    SILVER_MANIFEST,
    BuildManifest,
    EmbeddingsManifest,
    GoldManifest,
    SilverManifest,
    build_dir,
    builds_dir,
)
from lakehouse_cv.settings import CvLakePaths


def derive_build_id(parts: Sequence[str]) -> str:
    """Digest the code version and the ids of the inputs of a build."""
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]


def read_whole_build[T: BuildManifest](
    *, layer_dir: UPath, build_id: str, model: type[T]
) -> T | None:
    """Return the manifest of a build that a run finished, or None."""
    return read_build_manifest(
        path=build_dir(layer_dir=layer_dir, build_id=build_id) / BUILD_MANIFEST,
        model=model,
    )


def reuse_whole_build[T: BuildManifest](
    *, manifest_path: UPath, build_id: str, model: type[T]
) -> T | None:
    """Point the manifest at a whole build of this id, and return it. None if none.

    A run that failed before it finished leaves no copy of the manifest, so its files
    go, and the run writes them again.
    """
    layer_dir = manifest_path.parent
    whole = read_whole_build(layer_dir=layer_dir, build_id=build_id, model=model)
    if whole is None:
        discard_build(layer_dir=layer_dir, build_id=build_id)
        return None
    publish_build(manifest_path=manifest_path, manifest=whole)
    return whole


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


def read_silver_build(paths: CvLakePaths, name: str, build_id: str) -> SilverManifest:
    manifest = read_whole_build(
        layer_dir=paths.silver_dir(name), build_id=build_id, model=SilverManifest
    )
    if manifest is None:
        raise _build_missing_error(description=f"silver {name}", build_id=build_id)
    return manifest


def read_gold_build(paths: CvLakePaths, build_id: str) -> GoldManifest:
    manifest = read_whole_build(
        layer_dir=paths.gold_dir(), build_id=build_id, model=GoldManifest
    )
    if manifest is None:
        raise _build_missing_error(description="gold", build_id=build_id)
    return manifest


def _build_missing_error(*, description: str, build_id: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"The lake holds no whole build {build_id} of {description}. A later run "
        "removed it, or a run before builds wrote it. Materialize it again."
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
    """Mark the build whole, replace the manifest, and remove every other build but
    the one before.

    A reader that read the old manifest just before the swap still finds its files.
    """
    write_manifest(
        path=build_dir(layer_dir=manifest_path.parent, build_id=manifest.build_id)
        / BUILD_MANIFEST,
        manifest=manifest,
    )
    previous = _read_build_id(manifest_path)
    if previous == manifest.build_id:
        # The manifest names this build already. The build before it stays.
        return
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
