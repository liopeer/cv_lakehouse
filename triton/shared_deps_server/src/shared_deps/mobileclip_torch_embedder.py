#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# The MobileCLIP encoders in torch, for the ROCm image. One process reads, preprocesses
# and embeds a whole request, so no image passes between processes.
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional

from shared_deps.image_sources import fetch_url, is_url
from shared_deps.mobileclip.modules.common.transformer import LayerNormFP32
from shared_deps.mobileclip_image_preprocessing import (
    IMAGE_SIZE,
    NO_CROP,
    CropBox,
    preprocess_crops,
)

EMBEDDING_DIMENSION = 512
TEXT_CONTEXT_LENGTH = 77
FULL_IMAGE: CropBox = (NO_CROP,) * 4

# An image source is a local path, an http or https URL, or the encoded bytes.
ImageSource = str | bytes


class MobileCLIPTorchEmbedder:
    """Embeds images and texts with MobileCLIP, in fixed-size batches on one device.

    Every batch has `batch_size` rows, and the last one is padded. So a compiled
    encoder compiles once.
    """

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str | None,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        workers: int,
        fetch_timeout_seconds: float,
        compile_encoders: bool,
    ) -> None:
        from shared_deps import mobileclip

        model, _, _ = mobileclip.create_model_and_transforms(
            model_name=model_name, pretrained=checkpoint_path, reparameterize=True
        )
        self._tokenizer = mobileclip.get_tokenizer(model_name)
        self._device = device
        self._dtype = dtype
        self._batch_size = batch_size
        self._fetch_timeout_seconds = fetch_timeout_seconds
        self._pool = ThreadPoolExecutor(max_workers=workers)
        image_encoder = _to_inference_dtype(encoder=model.image_encoder, dtype=dtype)
        text_encoder = _to_inference_dtype(encoder=model.text_encoder, dtype=dtype)
        self._image_encoder = image_encoder.to(device)
        self._text_encoder = text_encoder.to(device)
        self._encode_images: Callable[[torch.Tensor], torch.Tensor] = self._embed_pixels
        self._encode_tokens: Callable[[torch.Tensor], torch.Tensor] = self._embed_tokens
        if compile_encoders:
            self._encode_images = torch.compile(self._embed_pixels, dynamic=False)
            self._encode_tokens = torch.compile(self._embed_tokens, dynamic=False)

    def warm_up(self) -> None:
        """Runs one batch through each encoder, so the compilation happens now."""
        self.embed_texts([""])
        pixels = torch.zeros(
            self._batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8
        )
        with torch.inference_mode():
            self._encode_images(pixels.to(self._device))

    def embed_images(
        self, *, sources: Sequence[ImageSource], crop_boxes: Sequence[CropBox]
    ) -> np.ndarray:
        """Embeds one crop per source, in the order given.

        Each distinct path or URL is read once, and decoded once for all its crops.
        Workers preprocess while the device embeds the crops that are ready.
        """
        if len(sources) != len(crop_boxes):
            raise ValueError(
                f"{len(sources)} images came with {len(crop_boxes)} crop boxes."
            )
        embeddings = np.empty((len(sources), EMBEDDING_DIMENSION), dtype=np.float32)
        future_indices = {
            self._pool.submit(
                self._read_and_preprocess,
                source,
                [crop_boxes[index] for index in indices],
            ): indices
            for source, indices in _group_by_source(sources)
        }
        ready_indices: list[int] = []
        ready_crops: list[torch.Tensor] = []
        try:
            for future in as_completed(future_indices):
                ready_indices.extend(future_indices[future])
                ready_crops.extend(future.result())
                while len(ready_crops) >= self._batch_size:
                    self._embed_crops(
                        crops=ready_crops[: self._batch_size],
                        indices=ready_indices[: self._batch_size],
                        embeddings=embeddings,
                    )
                    del ready_crops[: self._batch_size], ready_indices[: self._batch_size]
        except BaseException:
            for future in future_indices:
                future.cancel()
            raise
        if ready_crops:
            self._embed_crops(
                crops=ready_crops, indices=ready_indices, embeddings=embeddings
            )
        return embeddings

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embeds each text, in the order given."""
        embeddings = np.empty((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            tokens = torch.zeros(
                self._batch_size, TEXT_CONTEXT_LENGTH, dtype=torch.long
            )
            tokens[: len(chunk)] = self._tokenizer(list(chunk))
            with torch.inference_mode():
                batch = self._encode_tokens(tokens.to(self._device))
            embeddings[start : start + len(chunk)] = batch[: len(chunk)].cpu().numpy()
        return embeddings

    def close(self) -> None:
        self._pool.shutdown(cancel_futures=True)

    def _read_and_preprocess(
        self, source: ImageSource, crop_boxes: list[CropBox]
    ) -> list[torch.Tensor]:
        return preprocess_crops(self._read_source(source), crop_boxes)

    def _read_source(self, source: ImageSource) -> bytes:
        if isinstance(source, bytes):
            return source
        if is_url(source):
            return fetch_url(url=source, timeout_seconds=self._fetch_timeout_seconds)
        return Path(source).read_bytes()

    def _embed_crops(
        self,
        *,
        crops: list[torch.Tensor],
        indices: list[int],
        embeddings: np.ndarray,
    ) -> None:
        pixels = torch.zeros(
            self._batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8
        )
        torch.stack(crops, out=pixels[: len(crops)])
        with torch.inference_mode():
            batch = self._encode_images(pixels.to(self._device))
        embeddings[indices] = batch[: len(crops)].cpu().numpy()

    def _embed_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        images = pixels.to(self._dtype).div(255.0)
        output = self._image_encoder(images)
        if isinstance(output, dict):
            output = output["logits"]
        return _normalize(output.float())

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return _normalize(self._text_encoder(tokens).float())


def _group_by_source(
    sources: Sequence[ImageSource],
) -> list[tuple[ImageSource, list[int]]]:
    """Pairs each distinct path or URL with the indices that name it.

    Bytes stay one source per index, because hashing them costs more than it saves.
    """
    groups: dict[str, list[int]] = {}
    ungrouped: list[tuple[ImageSource, list[int]]] = []
    for index, source in enumerate(sources):
        if isinstance(source, bytes):
            ungrouped.append((source, [index]))
        else:
            groups.setdefault(source, []).append(index)
    return [*groups.items(), *ungrouped]


def _to_inference_dtype(*, encoder: nn.Module, dtype: torch.dtype) -> nn.Module:
    # LayerNormFP32 computes in fp32, so it keeps fp32 weights.
    encoder = encoder.to(dtype).eval()
    for module in encoder.modules():
        if isinstance(module, LayerNormFP32):
            module.float()
    return encoder


def _normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(
        embeddings, dim=-1, eps=torch.finfo(torch.float32).eps
    )
