#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional

from shared_deps.mobileclip_image_encoder import MOBILECLIP_CONFIGS

logger = logging.getLogger(__name__)

EncoderName = Literal["image", "text"]
Precision = Literal["fp16", "fp32"]

_TEXT_CONTEXT_LENGTH = 77


class MobileCLIPImageExportWrapper(nn.Module):
    """Image export wrapper preserving the existing FP32 public API."""

    def __init__(self, encoder: nn.Module, normalize_embeddings: bool = False) -> None:
        super().__init__()
        self.encoder = encoder.half().eval()
        self.normalize_embeddings = normalize_embeddings

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        out = self.encoder(images.to(dtype=torch.float16))
        if isinstance(out, dict):
            out = out["logits"]
        embeddings = out.float()
        if self.normalize_embeddings:
            return _normalize_embeddings(embeddings)
        return embeddings


class MobileCLIPTextExportWrapper(nn.Module):
    """Text export wrapper preserving the existing INT64 input / FP32 output API."""

    def __init__(self, encoder: nn.Module, normalize_embeddings: bool = False) -> None:
        super().__init__()
        self.encoder = encoder.half().eval()
        self.normalize_embeddings = normalize_embeddings

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(tokens).float()
        if self.normalize_embeddings:
            return _normalize_embeddings(embeddings)
        return embeddings


def export_mobileclip_image_onnx(
    *,
    out: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    batch_size: int = 1,
    max_batch_size: int = 256,
    precision: Precision = "fp16",
    opset_version: int = 18,
    normalize_embeddings: bool = False,
) -> None:
    """Export a MobileCLIP image encoder to ONNX using torch dynamo."""
    if model_name not in MOBILECLIP_CONFIGS:
        raise ValueError(f"Unsupported MobileCLIP model: {model_name}")

    config = MOBILECLIP_CONFIGS[model_name]
    model = _load_mobileclip_export_model(
        encoder_name="image",
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        precision=precision,
        normalize_embeddings=normalize_embeddings,
    )
    example_images = torch.zeros(
        batch_size,
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    _export_onnx(
        model=model,
        example_input=example_images,
        out=out,
        input_name="images",
        max_batch_size=max_batch_size,
        opset_version=opset_version,
    )


def export_mobileclip_text_onnx(
    *,
    out: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    batch_size: int = 2,
    max_batch_size: int = 256,
    precision: Precision = "fp16",
    opset_version: int = 18,
    normalize_embeddings: bool = False,
) -> None:
    """Export a MobileCLIP text encoder to ONNX using torch dynamo."""
    model = _load_mobileclip_export_model(
        encoder_name="text",
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        precision=precision,
        normalize_embeddings=normalize_embeddings,
    )
    # torch.export specializes the dynamic batch dim when traced with an
    # example batch size of exactly 1 (it treats 1 as broadcastable), which
    # for this model's EOT-embedding gather produces a self-contradictory
    # generalization guard. Tracing with batch_size >= 2 avoids that trap;
    # the exported graph still generalizes down to batch size 1 at runtime.
    example_tokens = torch.zeros(batch_size, _TEXT_CONTEXT_LENGTH, dtype=torch.long)
    _export_onnx(
        model=model,
        example_input=example_tokens,
        out=out,
        input_name="tokens",
        max_batch_size=max_batch_size,
        opset_version=opset_version,
    )


def export_mobileclip_image_tensorrt(
    *,
    out: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    precision: Precision = "fp16",
    min_batch_size: int = 1,
    opt_batch_size: int = 64,
    max_batch_size: int = 256,
    workspace_size_gib: int = 1,
    onnx_out: str | Path | None = None,
    normalize_embeddings: bool = False,
) -> None:
    """Export a MobileCLIP image encoder to a TensorRT plan."""
    out = Path(out)
    onnx_out = Path(onnx_out) if onnx_out is not None else out.with_suffix(".onnx")
    export_mobileclip_image_onnx(
        out=onnx_out,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        max_batch_size=max_batch_size,
        precision=precision,
        normalize_embeddings=normalize_embeddings,
    )
    _build_tensorrt_engine(
        onnx_path=onnx_out,
        out=out,
        input_name="images",
        precision=precision,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        workspace_size_gib=workspace_size_gib,
    )


def export_mobileclip_text_tensorrt(
    *,
    out: str | Path,
    model_name: str,
    checkpoint_path: str | Path,
    precision: Precision = "fp16",
    min_batch_size: int = 1,
    opt_batch_size: int = 64,
    max_batch_size: int = 256,
    workspace_size_gib: int = 1,
    onnx_out: str | Path | None = None,
    normalize_embeddings: bool = False,
) -> None:
    """Export a MobileCLIP text encoder to a TensorRT plan."""
    out = Path(out)
    onnx_out = Path(onnx_out) if onnx_out is not None else out.with_suffix(".onnx")
    export_mobileclip_text_onnx(
        out=onnx_out,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        max_batch_size=max_batch_size,
        precision=precision,
        normalize_embeddings=normalize_embeddings,
    )
    _build_tensorrt_engine(
        onnx_path=onnx_out,
        out=out,
        input_name="tokens",
        precision=precision,
        min_batch_size=min_batch_size,
        opt_batch_size=opt_batch_size,
        max_batch_size=max_batch_size,
        workspace_size_gib=workspace_size_gib,
    )


def main_export_onnx() -> None:
    args = _parse_common_args(output_suffix=".onnx")
    kwargs = vars(args)
    encoder = kwargs.pop("encoder")
    if encoder == "image":
        export_mobileclip_image_onnx(**kwargs)
    else:
        export_mobileclip_text_onnx(**kwargs)


def main_export_tensorrt() -> None:
    args = _parse_common_args(output_suffix=".plan", tensorrt=True)
    kwargs = vars(args)
    encoder = kwargs.pop("encoder")
    if encoder == "image":
        export_mobileclip_image_tensorrt(**kwargs)
    else:
        export_mobileclip_text_tensorrt(**kwargs)


def _load_mobileclip_export_model(
    *,
    encoder_name: EncoderName,
    model_name: str,
    checkpoint_path: str | Path,
    precision: Precision,
    normalize_embeddings: bool,
) -> nn.Module:
    from shared_deps import mobileclip

    model, _, _ = mobileclip.create_model_and_transforms(
        model_name=model_name,
        pretrained=str(checkpoint_path),
        reparameterize=True,
    )
    if encoder_name == "image":
        export_model: nn.Module = MobileCLIPImageExportWrapper(
            model.image_encoder,
            normalize_embeddings=normalize_embeddings,
        )
    else:
        export_model = MobileCLIPTextExportWrapper(
            model.text_encoder,
            normalize_embeddings=normalize_embeddings,
        )
    if precision == "fp32":
        export_model.float()
    return export_model.eval()


def _normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(
        embeddings,
        dim=-1,
        eps=torch.finfo(torch.float32).eps,
    )


@torch.no_grad()
def _export_onnx(
    *,
    model: nn.Module,
    example_input: torch.Tensor,
    out: str | Path,
    input_name: str,
    max_batch_size: int,
    opset_version: int,
) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    batch_dim = torch.export.Dim("batch", min=1, max=max_batch_size)
    model.eval()
    torch.onnx.export(
        model,
        (example_input,),
        out,
        input_names=[input_name],
        output_names=["embeddings"],
        dynamo=True,
        dynamic_shapes=({0: batch_dim},),
        opset_version=opset_version,
        external_data=False,
    )


def _build_tensorrt_engine(
    *,
    onnx_path: Path,
    out: Path,
    input_name: str,
    precision: Precision,
    min_batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    workspace_size_gib: int = 1,
) -> None:
    try:
        import tensorrt as trt  # type: ignore[import-not-found,import-untyped]
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "TensorRT is required for TensorRT export. Install TensorRT in a CUDA "
            "environment or run this exporter inside the Triton TensorRT container."
        ) from e

    if not (min_batch_size <= opt_batch_size <= max_batch_size):
        raise ValueError("Batch sizes must satisfy: min <= opt <= max")
    if workspace_size_gib < 1:
        raise ValueError("Workspace size must be at least 1 GiB")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error_index in range(parser.num_errors):
                logger.error(parser.get_error(error_index))
            raise RuntimeError(f"Failed to parse ONNX file: {onnx_path}")

    model_input = _get_tensorrt_input(network=network, input_name=input_name)
    input_shape = tuple(model_input.shape)
    static_shape = input_shape[1:]
    if any(dim == -1 for dim in static_shape):
        raise ValueError("Only the batch dimension may be dynamic for TensorRT export.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_size_gib << 30,
    )
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            if hasattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            elif hasattr(trt.BuilderFlag, "PREFER_PRECISION_CONSTRAINTS"):
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        else:
            logger.warning("FP16 is not supported on this platform; building FP32.")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch_size, *static_shape),
        opt=(opt_batch_size, *static_shape),
        max=(max_batch_size, *static_shape),
    )
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(engine)


def _get_tensorrt_input(network, input_name: str):
    for input_index in range(network.num_inputs):
        model_input = network.get_input(input_index)
        if model_input.name == input_name:
            return model_input
    raise RuntimeError(f"Could not find {input_name!r} input in ONNX network.")


def _parse_common_args(
    *, output_suffix: str, tensorrt: bool = False
) -> argparse.Namespace:
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
    if tensorrt:
        parser.add_argument("--min-batch-size", type=int, default=1)
        parser.add_argument("--opt-batch-size", type=int, default=64)
        parser.add_argument("--workspace-size-gib", type=int, default=1)
        parser.add_argument("--onnx-out", type=Path)
    else:
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--opset-version", type=int, default=18)
    args = parser.parse_args()
    if args.out.suffix != output_suffix:
        args.out = args.out.with_suffix(output_suffix)
    return args
