#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import torch
import torch.nn as nn

from shared_deps import mobileclip_compile


class MobileCLIPTextPreprocessor:
    """Tokenizes a UTF-8 string into a fixed-length int64 token array (context_length=77)."""

    def __init__(self, model_name: str) -> None:
        from shared_deps import mobileclip

        self._tokenizer = mobileclip.get_tokenizer(model_name)

    def __call__(self, text: str) -> np.ndarray:
        """
        Args:
            text: UTF-8 string.

        Returns:
            int64 numpy array of shape [77].
        """
        tokens = self._tokenizer(text)  # [1, 77] or [77] int64
        return tokens.squeeze(0).numpy().astype(np.int64)


class MobileCLIPTextBackend(nn.Module):
    """MobileCLIP text encoder — receives token ids, returns raw embeddings.

    Mirrors MobileCLIPTorchBackend but keeps model.text_encoder instead of
    model.image_encoder. Both backends load the checkpoint independently so
    each retains only its respective encoder in GPU memory.
    """

    def __init__(self, model_name: str, checkpoint_path: str) -> None:
        super().__init__()
        from shared_deps import mobileclip

        model, _, _ = mobileclip.create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint_path,
            reparameterize=True,
        )
        # Keep the encoder in fp16 internally while preserving fp32 API outputs.
        self._encoder = mobileclip_compile.compile_encoder_for_inference(
            model.text_encoder.half()
        )  # TextTransformer

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: int64 tensor of shape [B, 77].

        Returns:
            float32 tensor of shape [B, embed_dim] — NOT L2-normalized.
        """
        return self._encoder(tokens).float()
