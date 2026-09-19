#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional

from shared_deps.export_mixins import Precision, TensorRTMixin
from shared_deps.mobileclip.modules.common.transformer import LayerNormFP32
from shared_deps.mobileclip_image_encoder import MOBILECLIP_CONFIGS

EncoderName = Literal["image", "text"]

_TEXT_CONTEXT_LENGTH = 77


class MobileCLIPImageExportWrapper(TensorRTMixin, nn.Module):
    """Image export wrapper preserving the existing FP32 public API."""

    onnx_input_name = "images"

    def __init__(
        self,
        encoder: nn.Module,
        image_size: int,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = _convert_to_half(encoder)
        self.image_size = image_size
        self.normalize_embeddings = normalize_embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.encoder(images.to(dtype=torch.float16))
        if isinstance(out, dict):
            out = out["logits"]
        embeddings = out.float()
        if self.normalize_embeddings:
            return _normalize_embeddings(embeddings)
        return embeddings

    def create_onnx_example_input(self) -> torch.Tensor:
        return torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.float32)


class MobileCLIPTextExportWrapper(TensorRTMixin, nn.Module):
    """Text export wrapper preserving the existing INT64 input / FP32 output API."""

    onnx_input_name = "tokens"

    def __init__(self, encoder: nn.Module, normalize_embeddings: bool = False) -> None:
        super().__init__()
        self.encoder = _convert_to_half(encoder)
        self.normalize_embeddings = normalize_embeddings

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(tokens).float()
        if self.normalize_embeddings:
            return _normalize_embeddings(embeddings)
        return embeddings

    def create_onnx_example_input(self) -> torch.Tensor:
        # torch.export specializes the dynamic batch dim when traced with an
        # example batch size of exactly 1 (it treats 1 as broadcastable), which
        # for this model's EOT-embedding gather produces a self-contradictory
        # generalization guard. Tracing with batch_size >= 2 avoids that trap;
        # the exported graph still generalizes down to batch size 1 at runtime.
        return torch.zeros(2, _TEXT_CONTEXT_LENGTH, dtype=torch.long)


MobileCLIPExportWrapper = MobileCLIPImageExportWrapper | MobileCLIPTextExportWrapper


def load_mobileclip_export_model(
    *,
    encoder: EncoderName,
    model_name: str,
    checkpoint_path: str | Path,
    precision: Precision = "fp16",
    normalize_embeddings: bool = False,
) -> MobileCLIPExportWrapper:
    from shared_deps import mobileclip

    if model_name not in MOBILECLIP_CONFIGS:
        raise ValueError(f"Unsupported MobileCLIP model: {model_name}")

    model, _, _ = mobileclip.create_model_and_transforms(
        model_name=model_name,
        pretrained=str(checkpoint_path),
        reparameterize=True,
    )
    if encoder == "image":
        export_model: MobileCLIPExportWrapper = MobileCLIPImageExportWrapper(
            encoder=model.image_encoder,
            image_size=MOBILECLIP_CONFIGS[model_name].image_size,
            normalize_embeddings=normalize_embeddings,
        )
    else:
        export_model = MobileCLIPTextExportWrapper(
            encoder=model.text_encoder,
            normalize_embeddings=normalize_embeddings,
        )
    if precision == "fp32":
        export_model.float()
    return export_model.eval()


def main_export_onnx() -> None:
    parser = _create_common_parser()
    parser.add_argument("--opset-version", type=int, default=18)
    args = _parse_args_with_suffix(parser=parser, output_suffix=".onnx")
    export_model = _load_export_model_from_args(args)
    export_model.export_onnx(
        out=args.out,
        max_batch_size=args.max_batch_size,
        opset_version=args.opset_version,
    )


def main_export_tensorrt() -> None:
    parser = _create_common_parser()
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--opt-batch-size", type=int, default=64)
    parser.add_argument("--workspace-size-gib", type=int, default=1)
    parser.add_argument("--onnx-out", type=Path)
    args = _parse_args_with_suffix(parser=parser, output_suffix=".plan")
    export_model = _load_export_model_from_args(args)
    export_model.export_tensorrt(
        out=args.out,
        onnx_out=args.onnx_out,
        precision=args.precision,
        min_batch_size=args.min_batch_size,
        opt_batch_size=args.opt_batch_size,
        max_batch_size=args.max_batch_size,
        workspace_size_gib=args.workspace_size_gib,
    )


def _convert_to_half(encoder: nn.Module) -> nn.Module:
    # LayerNormFP32 casts its input to FP32, so FP16 weights give the ONNX
    # LayerNormalization node mixed types, which ONNX Runtime rejects.
    encoder = encoder.half().eval()
    for module in encoder.modules():
        if isinstance(module, LayerNormFP32):
            module.float()
    return encoder


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(
        embeddings,
        dim=-1,
        eps=torch.finfo(torch.float32).eps,
    )


def _create_common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("image", "text"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-name", default="mobileclip_s0")
    parser.add_argument(
        "--checkpoint-path", type=Path, default=Path("/checkpoints/mobileclip_s0.pt")
    )
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-batch-size", type=int, default=256)
    parser.add_argument("--normalize-embeddings", action="store_true")
    return parser


def _parse_args_with_suffix(
    *, parser: argparse.ArgumentParser, output_suffix: str
) -> argparse.Namespace:
    args = parser.parse_args()
    if args.out.suffix != output_suffix:
        args.out = args.out.with_suffix(output_suffix)
    return args


def _load_export_model_from_args(args: argparse.Namespace) -> MobileCLIPExportWrapper:
    return load_mobileclip_export_model(
        encoder=args.encoder,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        precision=args.precision,
        normalize_embeddings=args.normalize_embeddings,
    )
