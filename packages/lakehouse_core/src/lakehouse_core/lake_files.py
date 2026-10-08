#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The file operations that `UPath` does not offer on an object store.

An object store has no rename. There, a move is a copy and a delete, and a single
object appears whole or not at all, so `replace_file` stays atomic for a reader.
"""

import pathlib
import shutil

from upath import UPath

from lakehouse_core.lake_store import is_local


def replace_file(source: UPath, target: UPath) -> None:
    if is_local(source) and is_local(target):
        pathlib.Path(str(source)).replace(pathlib.Path(str(target)))
        return
    target.fs.mv(source.path, target.path)


def move_tree(source: UPath, target: UPath) -> None:
    if is_local(source) and is_local(target):
        pathlib.Path(str(source)).rename(pathlib.Path(str(target)))
        return
    # A recursive `mv` also copies each directory. On S3 a directory is no object, so
    # that copy fails with NoSuchKey. Copy the files only.
    files = [str(file) for file in source.fs.find(source.path, withdirs=False)]
    target.fs.copy(
        files, [target.path + file.removeprefix(source.path) for file in files]
    )
    source.fs.rm(source.path, recursive=True)


def remove_tree(path: UPath) -> None:
    """Remove a directory and everything under it. A missing directory is fine."""
    if is_local(path):
        shutil.rmtree(pathlib.Path(str(path)), ignore_errors=True)
        return
    if path.fs.exists(path.path):
        path.fs.rm(path.path, recursive=True)


def copy_file(source: UPath, target: UPath) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if is_local(source) and is_local(target):
        shutil.copyfile(src=pathlib.Path(str(source)), dst=pathlib.Path(str(target)))
    elif source.fs is target.fs:
        target.fs.cp_file(source.path, target.path)
    else:
        with source.open(mode="rb") as reader, target.open(mode="wb") as writer:
            shutil.copyfileobj(reader, writer)
