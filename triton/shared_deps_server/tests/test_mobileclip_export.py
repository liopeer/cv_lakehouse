#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import pytest
import torch

from shared_deps import mobileclip_export


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
        wrapper = mobileclip_export.MobileCLIPImageExportWrapper(_ImageEncoder())

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
    def test_image_export_writes_onnx_with_dynamo(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "image.onnx"
        model = mobileclip_export.MobileCLIPImageExportWrapper(_ImageEncoder())

        mobileclip_export._export_onnx(
            model=model,
            example_input=torch.zeros(2, 3, 8, 8, dtype=torch.float32),
            out=out,
            input_name="images",
            max_batch_size=8,
            opset_version=18,
        )

        assert out.is_file()

    def test_text_export_writes_onnx_with_dynamo(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "text.onnx"
        model = mobileclip_export.MobileCLIPTextExportWrapper(_TextEncoder())

        mobileclip_export._export_onnx(
            model=model,
            example_input=torch.zeros(2, 77, dtype=torch.long),
            out=out,
            input_name="tokens",
            max_batch_size=8,
            opset_version=18,
        )

        assert out.is_file()

    def test_normalized_image_export_writes_onnx_with_dynamo(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "normalized_image.onnx"
        model = mobileclip_export.MobileCLIPImageExportWrapper(
            _ImageEncoder(),
            normalize_embeddings=True,
        )

        mobileclip_export._export_onnx(
            model=model,
            example_input=torch.zeros(2, 3, 8, 8, dtype=torch.float32),
            out=out,
            input_name="images",
            max_batch_size=8,
            opset_version=18,
        )

        assert out.is_file()

    def test_normalized_text_export_writes_onnx_with_dynamo(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        out = tmp_path / "normalized_text.onnx"
        model = mobileclip_export.MobileCLIPTextExportWrapper(
            _TextEncoder(),
            normalize_embeddings=True,
        )

        mobileclip_export._export_onnx(
            model=model,
            example_input=torch.zeros(2, 77, dtype=torch.long),
            out=out,
            input_name="tokens",
            max_batch_size=8,
            opset_version=18,
        )

        assert out.is_file()


class TestMobileCLIPTensorRTExport:
    def test_image_onnx_builds_tensorrt_plan(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        pytest.importorskip("tensorrt")
        onnx_out = tmp_path / "image.onnx"
        plan_out = tmp_path / "image.plan"

        mobileclip_export._export_onnx(
            model=mobileclip_export.MobileCLIPImageExportWrapper(
                _ImageEncoder(),
                normalize_embeddings=True,
            ),
            example_input=torch.zeros(1, 3, 8, 8, dtype=torch.float32),
            out=onnx_out,
            input_name="images",
            max_batch_size=4,
            opset_version=18,
        )
        mobileclip_export._build_tensorrt_engine(
            onnx_path=onnx_out,
            out=plan_out,
            input_name="images",
            precision="fp16",
            min_batch_size=1,
            opt_batch_size=2,
            max_batch_size=4,
        )

        assert plan_out.is_file()
        assert plan_out.stat().st_size > 0

    def test_text_onnx_builds_tensorrt_plan(self, tmp_path):
        pytest.importorskip("onnx")
        pytest.importorskip("onnxscript")
        pytest.importorskip("tensorrt")
        onnx_out = tmp_path / "text.onnx"
        plan_out = tmp_path / "text.plan"

        mobileclip_export._export_onnx(
            model=mobileclip_export.MobileCLIPTextExportWrapper(_TextEncoder()),
            example_input=torch.zeros(1, 77, dtype=torch.long),
            out=onnx_out,
            input_name="tokens",
            max_batch_size=4,
            opset_version=18,
        )
        mobileclip_export._build_tensorrt_engine(
            onnx_path=onnx_out,
            out=plan_out,
            input_name="tokens",
            precision="fp16",
            min_batch_size=1,
            opt_batch_size=2,
            max_batch_size=4,
        )

        assert plan_out.is_file()
        assert plan_out.stat().st_size > 0
