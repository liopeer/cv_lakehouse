#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from unittest import mock

import pytest
import torch

from shared_deps import export_mixins, mobileclip_export


class _ImageEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert images.dtype == torch.float16
        return images.mean(dim=(2, 3)) * self.scale


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens[:, 0])


class TestMobileCLIPExportWrappers:
    def test_image_wrapper_casts_input_to_fp16_and_returns_fp32(self):
        wrapper = mobileclip_export.MobileCLIPImageExportWrapper(_ImageEncoder(), image_size=4)

        output = wrapper(torch.ones(2, 3, 4, 4, dtype=torch.float32))

        assert next(wrapper.encoder.parameters()).dtype == torch.float16
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, torch.ones(2, 3))

    def test_text_wrapper_keeps_tokens_int64_and_returns_fp32(self):
        wrapper = mobileclip_export.MobileCLIPTextExportWrapper(_TextEncoder())
        with torch.no_grad():
            wrapper.encoder.embedding.weight.copy_(
                torch.arange(30, dtype=torch.float16).reshape(10, 3)
            )

        output = wrapper(torch.tensor([[1, 2], [5, 6]], dtype=torch.long))

        assert next(wrapper.encoder.parameters()).dtype == torch.float16
        assert output.dtype == torch.float32
        torch.testing.assert_close(
            output, torch.tensor([[3.0, 4.0, 5.0], [15.0, 16.0, 17.0]])
        )

    def test_image_wrapper_optionally_normalizes_embeddings(self):
        wrapper = mobileclip_export.MobileCLIPImageExportWrapper(
            _ImageEncoder(),
            image_size=4,
            normalize_embeddings=True,
        )

        output = wrapper(torch.ones(2, 3, 4, 4, dtype=torch.float32))

        assert output.dtype == torch.float32
        torch.testing.assert_close(
            torch.linalg.norm(output, dim=-1),
            torch.ones(2),
        )

    def test_text_wrapper_optionally_normalizes_embeddings(self):
        wrapper = mobileclip_export.MobileCLIPTextExportWrapper(
            _TextEncoder(),
            normalize_embeddings=True,
        )
        with torch.no_grad():
            wrapper.encoder.embedding.weight.copy_(
                torch.arange(30, dtype=torch.float16).reshape(10, 3)
            )

        output = wrapper(torch.tensor([[1, 2], [5, 6]], dtype=torch.long))

        assert output.dtype == torch.float32
        torch.testing.assert_close(
            torch.linalg.norm(output, dim=-1),
            torch.ones(2),
        )


class TestMobileCLIPONNXExport:
    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_image_export_writes_onnx_with_dynamo(self, tmp_path, normalize_embeddings):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "image.onnx"
        model = mobileclip_export.MobileCLIPImageExportWrapper(
            _ImageEncoder(),
            image_size=8,
            normalize_embeddings=normalize_embeddings,
        )

        model.export_onnx(out=out, max_batch_size=8)

        assert out.is_file()

    @pytest.mark.parametrize("normalize_embeddings", [False, True])
    def test_text_export_writes_onnx_with_dynamo(self, tmp_path, normalize_embeddings):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "text.onnx"
        model = mobileclip_export.MobileCLIPTextExportWrapper(
            _TextEncoder(),
            normalize_embeddings=normalize_embeddings,
        )

        model.export_onnx(out=out, max_batch_size=8)

        assert out.is_file()

    def test_exported_onnx_accepts_any_batch_size(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "image.onnx"
        model = mobileclip_export.MobileCLIPImageExportWrapper(_ImageEncoder(), image_size=8)

        model.export_onnx(out=out, max_batch_size=8)

        graph_input = onnx.load(out).graph.input[0]
        assert graph_input.name == "images"
        assert graph_input.type.tensor_type.shape.dim[0].dim_param


class TestMobileCLIPTensorRTExport:
    def test_export_tensorrt_writes_onnx_then_builds_engine(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        onnx_out = tmp_path / "text.onnx"
        plan_out = tmp_path / "text.plan"
        model = mobileclip_export.MobileCLIPTextExportWrapper(_TextEncoder())

        def assert_onnx_written(**kwargs):
            assert kwargs["onnx_path"].is_file()

        with mock.patch.object(
            export_mixins, "build_tensorrt_engine", side_effect=assert_onnx_written
        ) as build:
            model.export_tensorrt(out=plan_out, onnx_out=onnx_out, max_batch_size=4)

        build.assert_called_once()
        assert build.call_args.kwargs["input_name"] == "tokens"
        assert build.call_args.kwargs["out"] == plan_out

    def test_image_export_builds_tensorrt_plan(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        pytest.importorskip("tensorrt")
        plan_out = tmp_path / "image.plan"
        model = mobileclip_export.MobileCLIPImageExportWrapper(
            _ImageEncoder(),
            image_size=8,
            normalize_embeddings=True,
        )

        model.export_tensorrt(
            out=plan_out, opt_batch_size=2, max_batch_size=4
        )

        assert plan_out.stat().st_size > 0
        assert plan_out.with_suffix(".onnx").is_file()

    def test_text_export_builds_tensorrt_plan(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        pytest.importorskip("tensorrt")
        plan_out = tmp_path / "text.plan"
        model = mobileclip_export.MobileCLIPTextExportWrapper(_TextEncoder())

        model.export_tensorrt(
            out=plan_out, opt_batch_size=2, max_batch_size=4
        )

        assert plan_out.stat().st_size > 0
