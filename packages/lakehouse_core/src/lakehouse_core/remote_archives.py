#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Unpack an archive on an object store into the same store, member by member.

Nothing goes to local disk. Each member streams from the archive to its own key.

- A zip is read through range requests, since its index is at its end.
- A tar is read as one stream, twice. The first pass lists the names, so that the
  unpack knows whether the archive wraps everything in one directory before it writes
  a single member. An object store has no rename to undo that later, and a stream is
  cheaper than a copy of every member.

An object store has no rename either for the staging directory of a local unpack. A
marker beside the archive, written last, tells a whole tree from part of one.
"""

import posixpath
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from typing import IO

from upath import UPath

from lakehouse_core.lake_files import remove_tree

COPY_CHUNK_SIZE = 8 * 1024 * 1024
# `data_filter` judges a member by where it would land under this directory. Nothing is
# written there.
_FILTER_ROOT = "/unpacked"


def unpack_remote_archive(
    *, archive: UPath, unpacked: UPath, skipped_prefix: str
) -> UPath:
    """Unpack `archive` into `unpacked`, unless the marker says that it is done."""
    marker = unpacked_marker(archive=archive, unpacked=unpacked)
    if marker.exists():
        return unpacked
    # A run that died left part of a tree.
    remove_tree(unpacked)
    if archive.name.endswith(".zip"):
        _unzip(archive=archive, unpacked=unpacked, skipped_prefix=skipped_prefix)
    else:
        _untar(archive=archive, unpacked=unpacked)
    marker.write_bytes(b"")
    return unpacked


def unpacked_marker(*, archive: UPath, unpacked: UPath) -> UPath:
    return archive.with_name(f".{unpacked.name}.unpacked")


def wrapped_prefix(*, names: Iterable[str], unpacked_name: str) -> str:
    """Return the one directory that wraps every member and is named `unpacked_name`.

    A desktop unarchiver does not double that directory, and neither does a local
    unpack. Return "" when the archive holds anything beside it.
    """
    tops = set()
    nested = False
    for name in names:
        normalized = posixpath.normpath(name)
        if normalized == ".":
            continue
        top, _, rest = normalized.partition("/")
        tops.add(top)
        nested = nested or bool(rest) or name.endswith("/")
    return f"{unpacked_name}/" if tops == {unpacked_name} and nested else ""


def _unzip(*, archive: UPath, unpacked: UPath, skipped_prefix: str) -> None:
    with archive.open(mode="rb") as handle, zipfile.ZipFile(handle) as zipped:
        members = [
            info
            for info in zipped.infolist()
            if not info.filename.startswith(skipped_prefix)
        ]
        prefix = wrapped_prefix(
            names=(info.filename for info in members), unpacked_name=unpacked.name
        )
        for info in members:
            if info.is_dir():
                continue
            target = _member_target(
                unpacked=unpacked, name=info.filename, prefix=prefix
            )
            with zipped.open(info) as source:
                _copy(source=source, target=target)


def _untar(*, archive: UPath, unpacked: UPath) -> None:
    with (
        archive.open(mode="rb") as handle,
        tarfile.open(fileobj=handle, mode="r|*") as tar,
    ):
        prefix = wrapped_prefix(
            names=(member.name for member in tar), unpacked_name=unpacked.name
        )
    with (
        archive.open(mode="rb") as handle,
        tarfile.open(fileobj=handle, mode="r|*") as tar,
    ):
        for member in tar:
            tarfile.data_filter(member, _FILTER_ROOT)
            if member.isdir():
                continue
            if not member.isreg():
                raise NotImplementedError(
                    f"{archive} holds {member.name}, which is not a regular file. An "
                    "object store holds no links and no devices."
                )
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"{archive} gives no bytes for {member.name}")
            target = _member_target(unpacked=unpacked, name=member.name, prefix=prefix)
            with source:
                _copy(source=source, target=target)


def _member_target(*, unpacked: UPath, name: str, prefix: str) -> UPath:
    relative = posixpath.normpath(name).removeprefix(prefix)
    if relative.startswith("../") or posixpath.isabs(relative):
        raise ValueError(f"{name} would unpack outside of {unpacked}")
    return unpacked / relative


def _copy(*, source: IO[bytes], target: UPath) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open(mode="wb") as writer:
        shutil.copyfileobj(source, writer, length=COPY_CHUNK_SIZE)
