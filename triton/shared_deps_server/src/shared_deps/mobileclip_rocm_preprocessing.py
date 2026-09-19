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

import torch
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as F

IMAGE_SIZE = 256
NO_CROP = -1

_JPEG_START = b"\xff\xd8"
_JPEG_START_OF_SCAN = 0xDA
# The start-of-frame markers of the baseline and the extended sequential Huffman JPEG.
_SEQUENTIAL_JPEG_FRAMES = (0xC0, 0xC1)
_START_OF_FRAME_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}

CropBox = tuple[int, int, int, int]


class MobileCLIPImageDecoder:
    """Decodes images to CHW uint8 tensors on `device`.

    Sequential JPEGs go to rocJPEG on the GPU when `use_rocjpeg` is set. Other images,
    such as a PNG or a progressive JPEG, and the JPEGs that rocJPEG rejects fall back to
    torchvision on the CPU.
    """

    def __init__(self, *, device: torch.device, use_rocjpeg: bool) -> None:
        self._device = device
        self._rocjpeg_decoder = None
        if use_rocjpeg:
            import pyRocJpegDecode.decoder as rocjpeg  # type: ignore[import-not-found]

            device_index = device.index or 0
            _, ready = rocjpeg.initialize_hip(device_index, False)
            if not ready:
                raise RuntimeError(f"rocJPEG found no GPU at index {device_index}.")
            self._rocjpeg_decoder = rocjpeg.decoder(device_id=device_index)

    def decode_images(self, encoded_images: Sequence[bytes]) -> list[torch.Tensor]:
        images: list[torch.Tensor | None] = [None] * len(encoded_images)
        sequential_jpeg_indices = [
            index
            for index, encoded in enumerate(encoded_images)
            if self._rocjpeg_decoder is not None and is_sequential_jpeg(encoded)
        ]
        for index, image in zip(
            sequential_jpeg_indices,
            self._decode_jpegs_on_gpu([encoded_images[i] for i in sequential_jpeg_indices]),
        ):
            images[index] = image
        return [
            image if image is not None else self._decode_on_cpu(encoded)
            for image, encoded in zip(images, encoded_images)
        ]

    def _decode_jpegs_on_gpu(
        self, encoded_jpegs: Sequence[bytes]
    ) -> list[torch.Tensor | None]:
        if not encoded_jpegs:
            return []
        # A batch skips an image that rocJPEG cannot decode, without saying which
        # one. Only a complete batch keeps its order, so otherwise decode one by one.
        try:
            _, decoded = self._rocjpeg_decoder.decode(list(encoded_jpegs))
        except RuntimeError:
            decoded = []
        if len(decoded) == len(encoded_jpegs):
            return [_convert_rocjpeg_image(image) for image in decoded]
        return [self._decode_jpeg_on_gpu(encoded) for encoded in encoded_jpegs]

    def _decode_jpeg_on_gpu(self, encoded_jpeg: bytes) -> torch.Tensor | None:
        try:
            _, decoded = self._rocjpeg_decoder.decode(encoded_jpeg)
        except RuntimeError:
            return None
        if decoded is None or decoded.width == 0:
            return None
        return _convert_rocjpeg_image(decoded)

    def _decode_on_cpu(self, encoded: bytes) -> torch.Tensor:
        data = torch.frombuffer(bytearray(encoded), dtype=torch.uint8)
        return decode_image(data, mode=ImageReadMode.RGB).to(self._device)


def is_sequential_jpeg(encoded: bytes) -> bool:
    """Tells whether the first frame of a JPEG is sequential, which rocJPEG decodes.

    rocJPEG fails a whole batch on a progressive JPEG, so these never reach it.
    """
    if not encoded.startswith(_JPEG_START):
        return False
    offset = len(_JPEG_START)
    while offset + 4 <= len(encoded):
        if encoded[offset] != 0xFF:
            return False
        marker = encoded[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker in _START_OF_FRAME_MARKERS:
            return marker in _SEQUENTIAL_JPEG_FRAMES
        if marker == _JPEG_START_OF_SCAN:
            return False
        segment_length = int.from_bytes(encoded[offset + 2 : offset + 4], "big")
        offset += 2 + segment_length
    return False


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


def _convert_rocjpeg_image(image) -> torch.Tensor:
    # The interleaved RGB buffer is HWC and can carry row padding, which the DLPack
    # strides describe. The permute is a view, so resize reads the buffer in place.
    return torch.from_dlpack(image).permute(2, 0, 1)
