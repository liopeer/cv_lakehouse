#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The Triton client, against a fake server that records what it was sent."""

from __future__ import annotations

import numpy as np
import pytest

from lakehouse_cv.transforms import embeddings
from lakehouse_cv.transforms.embeddings import (
    EMBEDDING_DIMENSION,
    ITEMS_PER_REQUEST,
    Crop,
    TritonEmbedder,
)
from tests.fakes import FakeTritonClient


def _embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TritonEmbedder, FakeTritonClient]:
    client = FakeTritonClient()
    monkeypatch.setattr(
        target=embeddings.grpcclient,
        name="InferenceServerClient",
        value=lambda url: client,
    )
    return TritonEmbedder("fake:0"), client


def test_embed_images_sends_one_bytes_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder, client = _embedder(monkeypatch)

    embeddings = embedder.embed_images(["/lake/a.jpg", "/lake/b.jpg"])

    assert embeddings.shape == (2, EMBEDDING_DIMENSION)
    assert embeddings.dtype == np.float32
    assert client.model_names == ["mobileclip_s0"]
    assert client.output_names == [["EMBEDDING"]]
    (request,) = client.requests
    assert list(request) == ["IMAGE_PATH"]
    assert request["IMAGE_PATH"] == [b"/lake/a.jpg", b"/lake/b.jpg"]


def test_embed_crops_sends_the_four_crop_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder, client = _embedder(monkeypatch)

    embedder.embed_crops([Crop(path="/lake/a.jpg", x=1, y=2, width=3, height=4)])

    (request,) = client.requests
    assert request["IMAGE_PATH"] == [b"/lake/a.jpg"]
    assert request["CROP_X"] == [1]
    assert request["CROP_Y"] == [2]
    assert request["CROP_WIDTH"] == [3]
    assert request["CROP_HEIGHT"] == [4]


def test_more_items_than_a_chunk_become_several_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder, client = _embedder(monkeypatch)
    count = ITEMS_PER_REQUEST + 3

    embeddings = embedder.embed_images([f"/lake/{index}.jpg" for index in range(count)])

    assert embeddings.shape == (count, EMBEDDING_DIMENSION)
    assert [len(request["IMAGE_PATH"]) for request in client.requests] == [
        ITEMS_PER_REQUEST,
        3,
    ]


def test_nothing_to_embed_sends_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder, client = _embedder(monkeypatch)

    embeddings = embedder.embed_images([])

    assert embeddings.shape == (0, EMBEDDING_DIMENSION)
    assert client.requests == []


def test_a_wrong_embedding_count_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    embedder, client = _embedder(monkeypatch)
    client.drop_last = True

    with pytest.raises(expected_exception=ValueError, match="shape"):
        embedder.embed_images(["/lake/a.jpg", "/lake/b.jpg"])
