#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# Image preprocessing for the ROCm image, where DALI does not run. It reproduces the
# DALI pipeline in `triton/dali_pipeline.py`:
#   encoded image -> optional crop -> resize-shorter-side(256) -> center-crop(256)
#   -> /255 -> CHW float32
from __future__ import annotations

import math
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGE_SIZE = 256
NO_CROP = -1

CropBox = tuple[int, int, int, int]


class MobileCLIPImagePreprocessor:
    """Preprocesses images on the CPU, over `decode_threads` threads.

    A worker returns each image at its final size, so a batch holds at most one full
    image per thread. rocJPEG decoded on the GPU before, and it faulted the process on
    a large batch.
    """

    def __init__(self, *, decode_threads: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=decode_threads)

    def preprocess_images(
        self, *, encoded_images: Sequence[bytes], crop_boxes: Sequence[CropBox]
    ) -> torch.Tensor:
        """Crops, resizes and normalises encoded images into one float32 batch."""
        if len(encoded_images) != len(crop_boxes):
            raise ValueError(
                f"{len(encoded_images)} images came with {len(crop_boxes)} crop boxes."
            )
        resized = list(self._pool.map(_preprocess_image, encoded_images, crop_boxes))
        return torch.stack(resized).float().div_(255.0)

    def close(self) -> None:
        self._pool.shutdown()


def _preprocess_image(encoded: bytes, crop_box: CropBox) -> torch.Tensor:
    """Decodes, crops, resizes and center-crops one image to CHW uint8."""
    with Image.open(BytesIO(encoded)) as image:
        crop_box = reduce_jpeg_decoding(image=image, crop_box=crop_box)
        pixels = np.asarray(image.convert("RGB"))
    decoded = torch.from_numpy(pixels).permute(2, 0, 1)
    return F.center_crop(
        F.resize(
            crop_image(image=decoded, crop_box=crop_box),
            size=[IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        output_size=[IMAGE_SIZE, IMAGE_SIZE],
    )


def reduce_jpeg_decoding(*, image: Image.Image, crop_box: CropBox) -> CropBox:
    """Lets a JPEG decode at 1/2, 1/4 or 1/8 scale, and scales the crop box with it.

    The scale keeps the shorter side of the crop at IMAGE_SIZE pixels or more. Other
    formats decode at full size.
    """
    full_width, full_height = image.size
    left, top, right, bottom = _crop_edges(
        crop_box=crop_box, width=full_width, height=full_height
    )
    scale = IMAGE_SIZE / max(min(right - left, bottom - top), 1)
    if scale >= 1.0:
        return crop_box
    image.draft(
        "RGB", (math.ceil(full_width * scale), math.ceil(full_height * scale))
    )
    width, height = image.size
    if crop_box[2] < 0 or (width, height) == (full_width, full_height):
        return crop_box
    x_scale = width / full_width
    y_scale = height / full_height
    reduced_left = math.floor(left * x_scale)
    reduced_top = math.floor(top * y_scale)
    return (
        reduced_left,
        reduced_top,
        math.ceil(right * x_scale) - reduced_left,
        math.ceil(bottom * y_scale) - reduced_top,
    )


def crop_image(*, image: torch.Tensor, crop_box: CropBox) -> torch.Tensor:
    """Crops `(x, y, width, height)` from a CHW image, trimmed to the image.

    A negative width means the full image, as `NO_CROP` does in the DALI pipeline.
    """
    image_height, image_width = image.shape[-2:]
    left, top, right, bottom = _crop_edges(
        crop_box=crop_box, width=image_width, height=image_height
    )
    return image[..., top:bottom, left:right]


def _crop_edges(
    *, crop_box: CropBox, width: int, height: int
) -> tuple[int, int, int, int]:
    """Returns the edges `(left, top, right, bottom)` of the crop inside the image."""
    x, y, crop_width, crop_height = crop_box
    if crop_width < 0:
        return 0, 0, width, height
    left = min(max(x, 0), width)
    top = min(max(y, 0), height)
    right = min(max(x + crop_width, left), width)
    bottom = min(max(y + crop_height, top), height)
    return left, top, right, bottom
