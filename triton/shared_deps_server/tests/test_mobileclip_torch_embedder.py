#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import io

import numpy as np
import pytest
import torch
from PIL import Image

from shared_deps.mobileclip_torch_embedder import (
    EMBEDDING_DIMENSION,
    FULL_IMAGE,
    MobileCLIPTorchEmbedder,
)

_BATCH_SIZE = 4


def _encode_image(*, seed: int, width: int = 320, height: int = 240) -> bytes:
    rng = np.random.default_rng(seed=seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def embedder():
    torch.manual_seed(0)
    embedder = MobileCLIPTorchEmbedder(
        model_name="mobileclip_s0",
        checkpoint_path=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=_BATCH_SIZE,
        workers=2,
        fetch_timeout_seconds=5,
        compile_encoders=False,
    )
    # Only a checkpoint fills the text projection, which starts as torch.empty.
    torch.nn.init.normal_(embedder._text_encoder.projection_layer)
    yield embedder
    embedder.close()


class TestEmbedImages:
    def test_returns_one_unit_vector_per_image_across_batches(self, embedder):
        encoded = [_encode_image(seed=seed) for seed in range(_BATCH_SIZE + 1)]

        embeddings = embedder.embed_images(
            sources=encoded, crop_boxes=[FULL_IMAGE] * len(encoded)
        )

        assert embeddings.shape == (_BATCH_SIZE + 1, EMBEDDING_DIMENSION)
        assert embeddings.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5)

    def test_keeps_the_order_of_the_sources(self, embedder):
        first, second = _encode_image(seed=1), _encode_image(seed=2)

        forward = embedder.embed_images(
            sources=[first, second], crop_boxes=[FULL_IMAGE] * 2
        )
        backward = embedder.embed_images(
            sources=[second, first], crop_boxes=[FULL_IMAGE] * 2
        )

        np.testing.assert_allclose(forward, backward[::-1], atol=1e-5)
        assert not np.allclose(forward[0], forward[1])

    def test_embeds_a_path_like_its_bytes(self, embedder, tmp_path):
        encoded = _encode_image(seed=3)
        path = tmp_path / "image.png"
        path.write_bytes(encoded)

        from_path, from_bytes = embedder.embed_images(
            sources=[str(path), encoded], crop_boxes=[FULL_IMAGE] * 2
        )

        np.testing.assert_allclose(from_path, from_bytes, atol=1e-5)

    def test_embeds_each_crop_of_a_shared_path(self, embedder, tmp_path):
        encoded = _encode_image(seed=4)
        path = tmp_path / "image.png"
        path.write_bytes(encoded)
        crops = [(0, 0, 160, 240), (160, 0, 160, 240), FULL_IMAGE]

        shared = embedder.embed_images(sources=[str(path)] * 3, crop_boxes=crops)
        apart = np.concatenate(
            [
                embedder.embed_images(sources=[encoded], crop_boxes=[crop])
                for crop in crops
            ]
        )

        np.testing.assert_allclose(shared, apart, atol=1e-5)

    def test_rejects_a_crop_box_count_that_differs(self, embedder):
        with pytest.raises(ValueError, match="1 images came with 2 crop boxes"):
            embedder.embed_images(
                sources=[_encode_image(seed=5)], crop_boxes=[FULL_IMAGE] * 2
            )

    def test_raises_for_a_missing_path(self, embedder, tmp_path):
        with pytest.raises(FileNotFoundError):
            embedder.embed_images(
                sources=[str(tmp_path / "missing.png")], crop_boxes=[FULL_IMAGE]
            )


class TestEmbedTexts:
    def test_returns_one_unit_vector_per_text_across_batches(self, embedder):
        texts = [f"a photo of {index} cats" for index in range(_BATCH_SIZE + 1)]

        embeddings = embedder.embed_texts(texts)

        assert embeddings.shape == (_BATCH_SIZE + 1, EMBEDDING_DIMENSION)
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5)

    def test_embeds_a_text_alike_in_any_batch(self, embedder):
        alone = embedder.embed_texts(["a dog"])
        among_others = embedder.embed_texts(["a cat", "a dog", "a bird"])

        np.testing.assert_allclose(alone[0], among_others[1], atol=1e-5)


def test_warm_up_runs_both_encoders(embedder):
    embedder.warm_up()
