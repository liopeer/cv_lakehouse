#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import pytest
from upath import UPath

from lakehouse_core.lake_files import move_tree

TREE = {
    "_release.json": b"{}",
    "boxes/open_images/test.parquet": b"boxes",
    "images/open_images/test.parquet": b"images",
}


@pytest.fixture(params=["local", "memory", "s3"])
def lake_dir(
    request: pytest.FixtureRequest, tmp_path: UPath, memory_dir: UPath
) -> UPath:
    if request.param == "local":
        return UPath(tmp_path)
    if request.param == "memory":
        return memory_dir
    return request.getfixturevalue("s3_dir")


def test_move_tree_moves_nested_files_and_removes_the_source(lake_dir: UPath) -> None:
    source = lake_dir / ".0001.staging"
    target = lake_dir / "0001"
    for name, content in TREE.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(content)

    move_tree(source=source, target=target)

    assert {name: (target / name).read_bytes() for name in TREE} == TREE
    assert not source.exists()
