#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The Triton client, against a fake server that records what it was sent."""

from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
import pytest

from cv_lakehouse import embeddings
from cv_lakehouse.embeddings import (
    EMBEDDING_DIMENSION,
    ITEMS_PER_REQUEST,
    Crop,
    TritonEmbedder,
    write_crop_embeddings,
    write_image_embeddings,
)
from cv_lakehouse.silver_schema import (
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_SCHEMA,
    SilverBox,
    SilverImage,
    crop_embeddings_file,
    embeddings_file,
    write_split,
)
from tests.fakes import FakeEmbedder, FakeTritonClient


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


def test_a_crop_rounds_onto_the_pixel_grid() -> None:
    crop = Crop.rounded_to_pixels(path="/lake/a.jpg", x=10.4, y=10.6, w=5.5, h=0.2)

    # A box under half a pixel wide still names one pixel, so the GPU crop has an area.
    assert crop == Crop(path="/lake/a.jpg", x=10, y=11, width=6, height=1)


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


def _write_silver(tmp_path):
    write_split(
        silver_dir=tmp_path,
        dataset="d",
        split="train",
        images=[
            SilverImage(
                file_name="a.jpg",
                width=100,
                height=100,
                boxes=(
                    SilverBox(
                        box_index=0,
                        class_id=0,
                        class_name="face",
                        source_class="face",
                        x=1.4,
                        y=2.6,
                        w=10.0,
                        h=20.0,
                    ),
                    SilverBox(
                        box_index=1,
                        class_id=1,
                        class_name="license_plate",
                        source_class="plate",
                        x=3.0,
                        y=4.0,
                        w=5.0,
                        h=6.0,
                    ),
                ),
            ),
            SilverImage(file_name="b.jpg", width=10, height=10),
        ],
    )


def test_writing_embeddings_matches_the_schema_and_the_row_order(tmp_path) -> None:
    _write_silver(tmp_path)
    embedder = FakeEmbedder()

    images = write_image_embeddings(
        embedder=embedder,
        silver_dir=tmp_path,
        dataset="d",
        split="train",
        image_root="/lake/images/",
    )
    crops = write_crop_embeddings(
        embedder=embedder,
        silver_dir=tmp_path,
        dataset="d",
        split="train",
        image_root="/lake/images/",
    )

    assert (images, crops) == (2, 2)
    image_table = pq.read_table(embeddings_file(silver_dir=tmp_path, split="train"))
    crop_table = pq.read_table(crop_embeddings_file(silver_dir=tmp_path, split="train"))
    assert image_table.schema == EMBEDDING_SCHEMA
    assert crop_table.schema == CROP_EMBEDDING_SCHEMA
    assert image_table.column("file_name").to_pylist() == ["a.jpg", "b.jpg"]
    assert crop_table.column("box_index").to_pylist() == [0, 1]
    assert len(image_table.column("embedding")[0]) == EMBEDDING_DIMENSION


def test_writing_embeddings_sends_absolute_paths_and_rounded_boxes(tmp_path) -> None:
    _write_silver(tmp_path)
    embedder = FakeEmbedder()

    write_image_embeddings(
        embedder=embedder,
        silver_dir=tmp_path,
        dataset="d",
        split="train",
        image_root="/lake/images/",
    )
    write_crop_embeddings(
        embedder=embedder,
        silver_dir=tmp_path,
        dataset="d",
        split="train",
        image_root="/lake/images/",
    )

    assert embedder.paths == ["/lake/images/a.jpg", "/lake/images/b.jpg"]
    assert embedder.crops == [
        Crop(path="/lake/images/a.jpg", x=1, y=3, width=10, height=20),
        Crop(path="/lake/images/a.jpg", x=3, y=4, width=5, height=6),
    ]
