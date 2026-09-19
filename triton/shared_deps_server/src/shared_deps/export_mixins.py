#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

Precision = Literal["fp16", "fp32"]


class ONNXExportMixin(ABC):
    """Exports an ``nn.Module`` with one input and a dynamic batch dimension to ONNX."""

    onnx_input_name: ClassVar[str]
    onnx_output_name: ClassVar[str] = "embeddings"

    @abstractmethod
    def create_onnx_example_input(self) -> torch.Tensor: ...

    @torch.no_grad()
    def export_onnx(
        self,
        *,
        out: str | Path,
        max_batch_size: int,
        opset_version: int = 18,
    ) -> None:
        if not isinstance(self, nn.Module):
            raise TypeError(f"{type(self).__name__} must be an nn.Module to export.")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        batch_dim = torch.export.Dim("batch", min=1, max=max_batch_size)
        self.eval()
        torch.onnx.export(
            self,
            (self.create_onnx_example_input(),),
            out,
            input_names=[self.onnx_input_name],
            output_names=[self.onnx_output_name],
            dynamo=True,
            dynamic_shapes=({0: batch_dim},),
            opset_version=opset_version,
            external_data=False,
        )


class TensorRTMixin(ONNXExportMixin):
    """Builds a TensorRT plan from the ONNX that ``export_onnx`` writes."""

    def export_tensorrt(
        self,
        *,
        out: str | Path,
        onnx_out: str | Path | None = None,
        precision: Precision = "fp16",
        min_batch_size: int = 1,
        opt_batch_size: int = 64,
        max_batch_size: int = 256,
        workspace_size_gib: int = 1,
    ) -> None:
        out = Path(out)
        onnx_out = Path(onnx_out) if onnx_out is not None else out.with_suffix(".onnx")
        self.export_onnx(out=onnx_out, max_batch_size=max_batch_size)
        build_tensorrt_engine(
            onnx_path=onnx_out,
            out=out,
            input_name=self.onnx_input_name,
            precision=precision,
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            workspace_size_gib=workspace_size_gib,
        )


def build_tensorrt_engine(
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
    # Imported here, so that an image without TensorRT still exports ONNX.
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

    model_input = _find_tensorrt_input(network=network, input_name=input_name)
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


def _find_tensorrt_input(*, network, input_name: str):
    for input_index in range(network.num_inputs):
        model_input = network.get_input(input_index)
        if model_input.name == input_name:
            return model_input
    raise RuntimeError(f"Could not find {input_name!r} input in ONNX network.")
