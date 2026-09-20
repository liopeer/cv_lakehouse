#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import io

import numpy as np
import pytest
import torch
from PIL import Image

from shared_deps.mobileclip_image_encoder import MobileCLIPPreprocessor
from shared_deps.mobileclip_rocm_preprocessing import (
    IMAGE_SIZE,
    NO_CROP,
    MobileCLIPImageDecoder,
    crop_image,
    preprocess_images,
)

_FULL_IMAGE = (NO_CROP,) * 4


def _encode_image(
    *, width: int, height: int, image_format: str, progressive: bool = False
) -> bytes:
    rng = np.random.default_rng(seed=0)
    # A smooth gradient with little noise, so that JPEG loses little.
    y, x = np.mgrid[0:height, 0:width]
    pixels = np.stack([x * 255 // width, y * 255 // height, (x + y) * 127 // (width + height)], axis=-1)
    pixels = np.clip(pixels + rng.integers(-3, 4, size=pixels.shape), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format=image_format, progressive=progressive)
    return buffer.getvalue()


@pytest.fixture
def cpu_decoder() -> MobileCLIPImageDecoder:
    return MobileCLIPImageDecoder(device=torch.device("cpu"), decode_threads=4)


class TestMobileCLIPImageDecoder:
    @pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
    def test_decodes_to_chw_uint8(self, cpu_decoder, image_format):
        encoded = _encode_image(width=40, height=30, image_format=image_format)

        (image,) = cpu_decoder.decode_images([encoded])

        assert image.shape == (3, 30, 40)
        assert image.dtype == torch.uint8

    def test_keeps_the_order_of_mixed_formats(self, cpu_decoder):
        encoded = [
            _encode_image(width=10, height=20, image_format="JPEG"),
            _encode_image(width=30, height=40, image_format="PNG"),
            _encode_image(width=50, height=60, image_format="JPEG"),
        ]

        images = cpu_decoder.decode_images(encoded)

        assert [image.shape for image in images] == [(3, 20, 10), (3, 40, 30), (3, 60, 50)]

    def test_decodes_a_progressive_jpeg(self, cpu_decoder):
        encoded = _encode_image(
            width=40, height=30, image_format="JPEG", progressive=True
        )

        (image,) = cpu_decoder.decode_images([encoded])

        assert image.shape == (3, 30, 40)

    def test_keeps_the_order_of_a_large_batch(self, cpu_decoder):
        # A batch of this size faulted the GPU decoder that this path replaced.
        encoded = [
            _encode_image(
                width=8 + index,
                height=16 + index,
                image_format="JPEG" if index % 2 else "PNG",
            )
            for index in range(128)
        ]

        images = cpu_decoder.decode_images(encoded)

        assert [image.shape for image in images] == [
            (3, 16 + index, 8 + index) for index in range(128)
        ]


class TestCropImage:
    def test_no_crop_returns_the_full_image(self):
        image = torch.zeros(3, 20, 30)

        assert crop_image(image=image, crop_box=_FULL_IMAGE).shape == (3, 20, 30)

    def test_crops_x_y_width_height(self):
        image = torch.arange(20 * 30).reshape(1, 20, 30)

        cropped = crop_image(image=image, crop_box=(5, 2, 10, 4))

        assert cropped.shape == (1, 4, 10)
        assert cropped[0, 0, 0] == 2 * 30 + 5

    def test_trims_a_crop_that_leaves_the_image(self):
        image = torch.zeros(3, 20, 30)

        assert crop_image(image=image, crop_box=(25, 15, 10, 10)).shape == (3, 5, 5)
        assert crop_image(image=image, crop_box=(-5, -5, 10, 10)).shape == (3, 5, 5)


class TestPreprocessImages:
    def test_batches_images_of_different_sizes(self):
        images = [
            torch.randint(0, 256, (3, 300, 400), dtype=torch.uint8),
            torch.randint(0, 256, (3, 500, 260), dtype=torch.uint8),
        ]

        batch = preprocess_images(images=images, crop_boxes=[_FULL_IMAGE] * 2)

        assert batch.shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch.dtype == torch.float32
        assert 0.0 <= batch.min() and batch.max() <= 1.0

    @pytest.mark.parametrize("crop_box", [_FULL_IMAGE, (40, 30, 300, 200)])
    def test_matches_the_pil_preprocessor(self, tmp_path, cpu_decoder, crop_box):
        encoded = _encode_image(width=640, height=480, image_format="PNG")
        path = tmp_path / "image.png"
        path.write_bytes(encoded)
        reference = MobileCLIPPreprocessor(image_size=IMAGE_SIZE)(
            str(path), crop_box=None if crop_box == _FULL_IMAGE else crop_box
        )

        batch = preprocess_images(
            images=cpu_decoder.decode_images([encoded]), crop_boxes=[crop_box]
        )

        assert (batch[0] - reference).abs().mean() < 2 / 255
