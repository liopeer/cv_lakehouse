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
    MobileCLIPImagePreprocessor,
    crop_image,
    reduce_jpeg_decoding,
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
def preprocessor() -> MobileCLIPImagePreprocessor:
    return MobileCLIPImagePreprocessor(decode_threads=4)


def _encode_solid_image(*, width: int, height: int, value: int, image_format: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (value,) * 3).save(buffer, format=image_format)
    return buffer.getvalue()


class TestMobileCLIPImagePreprocessor:
    @pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
    def test_preprocesses_to_a_float32_batch(self, preprocessor, image_format):
        encoded = _encode_image(width=400, height=300, image_format=image_format)

        batch = preprocessor.preprocess_images(
            encoded_images=[encoded], crop_boxes=[_FULL_IMAGE]
        )

        assert batch.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch.dtype == torch.float32
        assert 0.0 <= batch.min() and batch.max() <= 1.0

    def test_preprocesses_a_progressive_jpeg(self, preprocessor):
        encoded = _encode_image(
            width=40, height=30, image_format="JPEG", progressive=True
        )

        batch = preprocessor.preprocess_images(
            encoded_images=[encoded], crop_boxes=[_FULL_IMAGE]
        )

        assert batch.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_keeps_the_order_of_a_large_batch(self, preprocessor):
        # A batch of this size faulted the GPU decoder that this path replaced.
        encoded = [
            _encode_solid_image(
                width=8 + index,
                height=16 + index,
                value=index * 2,
                image_format="JPEG" if index % 2 else "PNG",
            )
            for index in range(128)
        ]

        batch = preprocessor.preprocess_images(
            encoded_images=encoded, crop_boxes=[_FULL_IMAGE] * 128
        )

        means = (batch.mean(dim=(1, 2, 3)) * 255).round()
        assert (means - torch.arange(128) * 2).abs().max() <= 2

    def test_rejects_a_crop_box_count_that_differs(self, preprocessor):
        encoded = _encode_image(width=40, height=30, image_format="PNG")

        with pytest.raises(ValueError, match="1 images came with 2 crop boxes"):
            preprocessor.preprocess_images(
                encoded_images=[encoded], crop_boxes=[_FULL_IMAGE] * 2
            )


class TestReduceJpegDecoding:
    def test_reduces_a_large_jpeg_to_the_smallest_scale_above_the_image_size(self):
        encoded = _encode_image(width=2048, height=1536, image_format="JPEG")

        with Image.open(io.BytesIO(encoded)) as image:
            crop_box = reduce_jpeg_decoding(image=image, crop_box=_FULL_IMAGE)

            assert image.size == (512, 384)
        assert crop_box == _FULL_IMAGE

    def test_scales_the_crop_box_with_the_image(self):
        encoded = _encode_image(width=2048, height=1536, image_format="JPEG")

        with Image.open(io.BytesIO(encoded)) as image:
            crop_box = reduce_jpeg_decoding(image=image, crop_box=(400, 300, 1200, 800))

            assert image.size == (1024, 768)
        assert crop_box == (200, 150, 600, 400)

    def test_keeps_a_crop_smaller_than_the_image_size_at_full_scale(self):
        encoded = _encode_image(width=2048, height=1536, image_format="JPEG")

        with Image.open(io.BytesIO(encoded)) as image:
            crop_box = reduce_jpeg_decoding(image=image, crop_box=(10, 20, 300, 200))

            assert image.size == (2048, 1536)
        assert crop_box == (10, 20, 300, 200)

    def test_keeps_a_png_at_full_scale(self):
        encoded = _encode_image(width=2048, height=1536, image_format="PNG")

        with Image.open(io.BytesIO(encoded)) as image:
            reduce_jpeg_decoding(image=image, crop_box=_FULL_IMAGE)

            assert image.size == (2048, 1536)


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


class TestMatchesThePilPreprocessor:
    @pytest.mark.parametrize(
        ("width", "height", "image_format", "crop_box"),
        [
            (640, 480, "PNG", _FULL_IMAGE),
            (640, 480, "PNG", (40, 30, 300, 200)),
            (2048, 1536, "JPEG", _FULL_IMAGE),
            (2048, 1536, "JPEG", (400, 300, 1200, 800)),
        ],
    )
    def test_matches_the_pil_preprocessor(
        self, tmp_path, preprocessor, width, height, image_format, crop_box
    ):
        encoded = _encode_image(width=width, height=height, image_format=image_format)
        path = tmp_path / f"image.{image_format.lower()}"
        path.write_bytes(encoded)
        reference = MobileCLIPPreprocessor(image_size=IMAGE_SIZE)(
            str(path), crop_box=None if crop_box == _FULL_IMAGE else crop_box
        )

        batch = preprocessor.preprocess_images(
            encoded_images=[encoded], crop_boxes=[crop_box]
        )

        assert (batch[0] - reference).abs().mean() < 2 / 255
