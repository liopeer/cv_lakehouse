#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Freeze the val and test rows of gold as a numbered release.

Gold changes with every correction, and a benchmark must not. A release is a copy of
the val and test files of one gold build, under a number that never moves. The
curators keep working on the gold above it, and their fixes reach the next release.

A release holds no vector. The vectors stay in silver, and silver moves on.
"""

from __future__ import annotations

import hashlib
import stat
from datetime import datetime

from upath import UPath

from lakehouse_core.lake_files import copy_file, move_tree, remove_tree
from lakehouse_core.lake_store import is_local
from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_cv.contract.gold_tables import gold_boxes_file, gold_images_file
from lakehouse_cv.contract.manifests import (
    RELEASE_MANIFEST,
    GoldDataset,
    ReleaseFile,
    ReleaseManifest,
)
from lakehouse_cv.settings import CvLakePaths
from lakehouse_cv.transforms.layer_builds import read_gold_build

READ_CHUNK_SIZE = 1024 * 1024
_READ_ONLY = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH


class IdenticalReleaseError(RuntimeError):
    pass


def list_release_numbers(paths: CvLakePaths) -> list[int]:
    """List every release that is complete, lowest first."""
    releases_dir = paths.gold_release_dir(0).parent
    if not releases_dir.exists():
        return []
    return sorted(
        int(path.name)
        for path in releases_dir.iterdir()
        # A failed move leaves a staging directory with a manifest.
        if path.name.isdigit() and (path / RELEASE_MANIFEST).exists()
    )


def read_release_manifest(paths: CvLakePaths, release: int) -> ReleaseManifest:
    return read_manifest(
        path=paths.gold_release_dir(release) / RELEASE_MANIFEST, model=ReleaseManifest
    )


def write_eval_release(
    *, paths: CvLakePaths, gold_build_id: str, created_at: datetime
) -> ReleaseManifest:
    """Copy the val and test files of one gold build into the next release."""
    gold = read_gold_build(paths=paths, build_id=gold_build_id)
    numbers = list_release_numbers(paths)
    release = numbers[-1] + 1 if numbers else 1
    release_dir = paths.gold_release_dir(release)
    # A release takes its final name when it is complete, so a run that fails leaves
    # no release behind.
    staging_dir = release_dir.with_name(f".{release_dir.name}.staging")
    remove_tree(staging_dir)

    gold_dir = paths.gold_build_dir(gold.build_id)
    datasets: list[GoldDataset] = []
    files: list[ReleaseFile] = []
    for dataset in gold.datasets:
        eval_splits = [split for split in dataset.splits if split.role.is_eval]
        if not eval_splits:
            continue
        datasets.append(dataset.model_copy(update={"splits": eval_splits}))
        for split in eval_splits:
            for gold_file in (gold_images_file, gold_boxes_file):
                names = {"dataset": dataset.dataset, "split": split.split}
                source = gold_file(build_dir=gold_dir, **names)
                target = gold_file(build_dir=staging_dir, **names)
                copy_file(source=source, target=target)
                # An object store has no file modes.
                if is_local(target):
                    target.chmod(_READ_ONLY)
                files.append(
                    ReleaseFile(
                        path=target.relative_to(staging_dir).as_posix(),
                        size=target.stat().st_size,
                        sha256=compute_sha256(target),
                    )
                )
    if not files:
        remove_tree(staging_dir)
        raise RuntimeError("Gold holds no val and no test split to release.")
    if numbers and _pin_files(files) == _pin_files(
        read_release_manifest(paths=paths, release=numbers[-1]).files
    ):
        remove_tree(staging_dir)
        raise IdenticalReleaseError(
            f"The val and test rows are those of release {numbers[-1]}. A new release "
            "would hold the same files."
        )

    manifest = ReleaseManifest(
        release=release,
        created_at=created_at,
        gold_build_id=gold.build_id,
        gold_code_version=gold.code_version,
        datasets=datasets,
        files=files,
    )
    write_manifest(path=staging_dir / RELEASE_MANIFEST, manifest=manifest)
    move_tree(source=staging_dir, target=release_dir)
    return manifest


def find_changed_release_files(paths: CvLakePaths) -> list[str]:
    """Return every file of every release that is not what its manifest pins."""
    changed: list[str] = []
    for release in list_release_numbers(paths):
        release_dir = paths.gold_release_dir(release)
        for pinned in read_release_manifest(paths=paths, release=release).files:
            path = release_dir / pinned.path
            if not path.exists() or compute_sha256(path) != pinned.sha256:
                changed.append(f"{release:04d}/{pinned.path}")
    return changed


def compute_sha256(path: UPath) -> str:
    digest = hashlib.sha256()
    with path.open(mode="rb") as handle:
        while chunk := handle.read(READ_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _pin_files(files: list[ReleaseFile]) -> dict[str, str]:
    return {pinned.path: pinned.sha256 for pinned in files}
