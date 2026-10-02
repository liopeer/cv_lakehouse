#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

import pytest

from cv_lakehouse.defs import silver as silver_defs
from cv_lakehouse.defs.resources import LakeResource
from tests.fakes import FakeEmbedder
from tests.fixtures import make_all


@pytest.fixture
def lake(tmp_path: Path) -> LakeResource:
    return LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def embedding_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LakeResource:
    """A lake with a Triton server that is a fake, so no test needs a GPU."""
    monkeypatch.setattr(
        target=silver_defs, name="TritonEmbedder", value=lambda url: FakeEmbedder()
    )
    return LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
        triton_url="fake:0",
    )


@pytest.fixture
def bronze_sources(tmp_path: Path) -> dict[str, Path]:
    return make_all(tmp_path / "external")
