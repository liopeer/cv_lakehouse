#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

import pytest

from cv_lakehouse.defs.resources import LakeResource
from tests.fixtures import make_all


@pytest.fixture
def lake(tmp_path: Path) -> LakeResource:
    return LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def bronze_sources(tmp_path: Path) -> dict[str, Path]:
    return make_all(tmp_path / "external")
