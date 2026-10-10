#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

import pytest
from upath import UPath

from lakehouse_cv.defs import silver_embeddings as embeddings_defs
from lakehouse_cv.defs.resources import CvLakeResource
from tests.fakes import FakeEmbedder
from tests.fixtures import make_all


@pytest.fixture
def lake(tmp_path: Path) -> CvLakeResource:
    return CvLakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
    )


@pytest.fixture
def embedding_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CvLakeResource:
    """A lake with a Triton server that is a fake, so no test needs a GPU."""
    monkeypatch.setattr(
        target=embeddings_defs, name="TritonEmbedder", value=lambda url: FakeEmbedder()
    )
    return CvLakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
        triton_url="fake:0",
    )


@pytest.fixture
def bronze_sources(tmp_path: Path) -> dict[str, UPath]:
    return make_all(tmp_path / "external")
