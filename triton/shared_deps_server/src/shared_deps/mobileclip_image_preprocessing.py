#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# Image preprocessing on the CPU, for the ROCm image, where DALI does not run. It
# reproduces the DALI pipeline in `triton/dali_pipeline.py`:
#   encoded image -> optional crop -> resize-shorter-side(256) -> center-crop(256)
from __future__ import annotations

import math
from collections.abc import Sequence
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGE_SIZE = 256
NO_CROP = -1

CropBox = tuple[int, int, int, int]


def preprocess_crops(
    encoded: bytes, crop_boxes: Sequence[CropBox]
) -> list[torch.Tensor]:
    """Decodes one image once, and returns each crop as a CHW uint8 tensor of 256x256.

    A worker returns each crop at its final size, so it holds at most one full image.
    """
    with Image.open(BytesIO(encoded)) as image:
        crop_boxes = reduce_jpeg_decoding(image=image, crop_boxes=crop_boxes)
        pixels = np.array(image.convert("RGB"))
    decoded = torch.from_numpy(pixels).permute(2, 0, 1)
    return [
        F.center_crop(
            F.resize(
                crop_image(image=decoded, crop_box=crop_box),
                size=[IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            output_size=[IMAGE_SIZE, IMAGE_SIZE],
        )
        for crop_box in crop_boxes
    ]


def reduce_jpeg_decoding(
    *, image: Image.Image, crop_boxes: Sequence[CropBox]
) -> list[CropBox]:
    """Lets a JPEG decode at 1/2, 1/4 or 1/8 scale, and scales the crop boxes with it.

    The scale keeps the shorter side of every crop at IMAGE_SIZE pixels or more. Other
    formats decode at full size.
    """
    full_width, full_height = image.size
    edges = [
        _crop_edges(crop_box=crop_box, width=full_width, height=full_height)
        for crop_box in crop_boxes
    ]
    shortest_side = min(
        min(right - left, bottom - top) for left, top, right, bottom in edges
    )
    scale = IMAGE_SIZE / max(shortest_side, 1)
    if scale >= 1.0:
        return list(crop_boxes)
    image.draft(
        "RGB", (math.ceil(full_width * scale), math.ceil(full_height * scale))
    )
    width, height = image.size
    if (width, height) == (full_width, full_height):
        return list(crop_boxes)
    return [
        crop_box
        if crop_box[2] < 0
        else _scale_crop_edges(
            edges=crop_edges, x_scale=width / full_width, y_scale=height / full_height
        )
        for crop_box, crop_edges in zip(crop_boxes, edges, strict=True)
    ]


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


def _scale_crop_edges(
    *, edges: tuple[int, int, int, int], x_scale: float, y_scale: float
) -> CropBox:
    left, top, right, bottom = edges
    reduced_left = math.floor(left * x_scale)
    reduced_top = math.floor(top * y_scale)
    return (
        reduced_left,
        reduced_top,
        math.ceil(right * x_scale) - reduced_left,
        math.ceil(bottom * y_scale) - reduced_top,
    )
