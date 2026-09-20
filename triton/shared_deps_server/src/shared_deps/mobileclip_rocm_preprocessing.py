#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# GPU image preprocessing for the ROCm image, where DALI does not run. It reproduces
# the DALI pipeline in `triton/dali_pipeline.py`:
#   encoded image -> optional crop -> resize-shorter-side(256) -> center-crop(256)
#   -> /255 -> CHW float32
from __future__ import annotations

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


class MobileCLIPImageDecoder:
    """Decodes images to CHW uint8 tensors on `device`.

    Pillow decodes every image on the CPU, over `decode_threads` threads. rocJPEG
    decoded on the GPU before, and it faulted the process on a large batch.
    """

    def __init__(self, *, device: torch.device, decode_threads: int) -> None:
        self._device = device
        self._pool = ThreadPoolExecutor(max_workers=decode_threads)

    def decode_images(self, encoded_images: Sequence[bytes]) -> list[torch.Tensor]:
        # A worker returns a CPU tensor, so every GPU call stays on this thread.
        return [
            image.to(self._device)
            for image in self._pool.map(_decode_image, encoded_images)
        ]

    def close(self) -> None:
        self._pool.shutdown()


def _decode_image(encoded: bytes) -> torch.Tensor:
    with Image.open(BytesIO(encoded)) as image:
        pixels = np.asarray(image.convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1)


def preprocess_images(
    *, images: Sequence[torch.Tensor], crop_boxes: Sequence[CropBox]
) -> torch.Tensor:
    """Crops, resizes and normalises CHW uint8 images into one float32 batch."""
    resized = [
        F.center_crop(
            F.resize(
                crop_image(image=image, crop_box=crop_box),
                size=[IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            output_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        for image, crop_box in zip(images, crop_boxes, strict=True)
    ]
    return torch.stack(resized).float().div_(255.0)


def crop_image(*, image: torch.Tensor, crop_box: CropBox) -> torch.Tensor:
    """Crops `(x, y, width, height)` from a CHW image, trimmed to the image.

    A negative width means the full image, as `NO_CROP` does in the DALI pipeline.
    """
    x, y, width, height = crop_box
    if width < 0:
        return image
    image_height, image_width = image.shape[-2:]
    left = min(max(x, 0), image_width)
    top = min(max(y, 0), image_height)
    right = min(max(x + width, left), image_width)
    bottom = min(max(y + height, top), image_height)
    return image[..., top:bottom, left:right]
