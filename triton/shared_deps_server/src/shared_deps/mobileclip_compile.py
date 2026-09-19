#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import torch
import torch.nn as nn


def compile_encoder_for_inference(encoder: nn.Module) -> nn.Module:
    """Compiles an encoder for inference with torch.compile."""
    return torch.compile(encoder, mode="reduce-overhead")
