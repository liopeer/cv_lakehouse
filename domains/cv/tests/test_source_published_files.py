#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

import pytest

from lakehouse_core.published_files import list_required_paths
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME


@pytest.mark.parametrize("name", sorted(SOURCE_BY_NAME))
def test_every_source_publishes_unique_relative_paths(name: str) -> None:
    files = SOURCE_BY_NAME[name].published_files
    paths = [published_file.path for published_file in files]
    assert len(set(paths)) == len(paths)
    assert all(not Path(path).is_absolute() and ".." not in path for path in paths)
    required = list_required_paths(files)
    assert len(set(required)) == len(required)
