#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import pytest
import torch
from PIL import Image

from shared_deps import mobileclip
from shared_deps import mobileclip_compile
from shared_deps.mobileclip_image_encoder import (
    MOBILECLIP_CONFIGS,
    MobileCLIPPreprocessor,
    MobileCLIPTorchBackend,
)
from shared_deps.mobileclip_text_encoder import MobileCLIPTextBackend


def _make_image_path(tmp_path, width: int = 64, height: int = 64):
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    path = tmp_path / "image.jpg"
    img.save(path, format="JPEG")
    return path


class TestMobileCLIPConfigs:
    def test_all_variants_present(self):
        assert set(MOBILECLIP_CONFIGS.keys()) == {
            "mobileclip_s0",
            "mobileclip_s1",
            "mobileclip_s2",
            "mobileclip_b",
        }

    def test_embed_dim(self):
        for cfg in MOBILECLIP_CONFIGS.values():
            assert cfg.embed_dim == 512

    def test_s0_image_size(self):
        assert MOBILECLIP_CONFIGS["mobileclip_s0"].image_size == 256

    def test_other_variants_image_size(self):
        for name in ("mobileclip_s1", "mobileclip_s2", "mobileclip_b"):
            assert MOBILECLIP_CONFIGS[name].image_size == 224


class TestMobileCLIPPreprocessor:
    @pytest.mark.parametrize("image_size", [224, 256])
    def test_output_shape(self, tmp_path, image_size: int):
        preprocessor = MobileCLIPPreprocessor(image_size=image_size)
        image_path = _make_image_path(tmp_path)
        tensor = preprocessor(str(image_path))
        assert tensor.shape == (3, image_size, image_size)

    def test_output_dtype(self, tmp_path):
        preprocessor = MobileCLIPPreprocessor(image_size=224)
        tensor = preprocessor(str(_make_image_path(tmp_path)))
        assert tensor.dtype == torch.float32

    def test_output_range(self, tmp_path):
        preprocessor = MobileCLIPPreprocessor(image_size=224)
        tensor = preprocessor(str(_make_image_path(tmp_path)))
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_full_image_crop_matches_uncropped(self, tmp_path):
        image_path = _make_image_path(tmp_path, width=80, height=60)
        preprocessor = MobileCLIPPreprocessor(image_size=224)

        uncropped = preprocessor(str(image_path))
        cropped = preprocessor(str(image_path), crop_box=(0, 0, 80, 60))

        assert torch.allclose(cropped, uncropped)


class TestTorchCompile:
    def test_compiled_encoder_forward_pass(self):
        class AddOne(torch.nn.Module):
            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return inputs + 1

        encoder = mobileclip_compile.compile_encoder_for_inference(AddOne())

        output = encoder(torch.zeros(2, 3))

        torch.testing.assert_close(output, torch.ones(2, 3))

    def test_image_backend_compiled_encoder_forward_pass(self, monkeypatch):
        class ImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, images: torch.Tensor) -> torch.Tensor:
                assert images.dtype == torch.float16
                return images.mean(dim=(2, 3)) * self.scale

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.image_encoder = ImageEncoder()

        def create_model_and_transforms(**kwargs):
            assert kwargs["reparameterize"] is True
            return Model(), None, None

        monkeypatch.setattr(
            mobileclip, "create_model_and_transforms", create_model_and_transforms
        )
        backend = MobileCLIPTorchBackend(
            model_name="mobileclip_s0", checkpoint_path="checkpoint.pt"
        )

        output = backend(torch.ones(2, 3, 4, 4))

        assert next(backend._encoder.parameters()).dtype == torch.float16
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, torch.ones(2, 3))

    def test_text_backend_compiled_encoder_forward_pass(self, monkeypatch):
        class TextEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(9, 3)

            def forward(self, tokens: torch.Tensor) -> torch.Tensor:
                return self.embedding(tokens[:, 0])

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.text_encoder = TextEncoder()

        def create_model_and_transforms(**kwargs):
            assert kwargs["reparameterize"] is True
            return Model(), None, None

        monkeypatch.setattr(
            mobileclip, "create_model_and_transforms", create_model_and_transforms
        )
        backend = MobileCLIPTextBackend(
            model_name="mobileclip_s0", checkpoint_path="checkpoint.pt"
        )

        with torch.no_grad():
            backend._encoder._orig_mod.embedding.weight.copy_(
                torch.arange(27, dtype=torch.float16).reshape(9, 3)
            )

        output = backend(torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))

        assert next(backend._encoder.parameters()).dtype == torch.float16
        assert output.dtype == torch.float32
        torch.testing.assert_close(
            output, torch.tensor([[3.0, 4.0, 5.0], [15.0, 16.0, 17.0]])
        )


class TestMobileCLIPImport:
    def test_create_model_and_transforms_importable(self):
        assert callable(mobileclip.create_model_and_transforms)

    def test_configs_loadable(self):
        for model_name in (
            "mobileclip_s0",
            "mobileclip_s1",
            "mobileclip_s2",
            "mobileclip_b",
        ):
            import json
            import os

            configs_dir = os.path.join(os.path.dirname(mobileclip.__file__), "configs")
            cfg = json.load(open(os.path.join(configs_dir, f"{model_name}.json")))
            assert "image_cfg" in cfg

    def test_no_open_clip_on_import(self):
        import sys

        assert "open_clip" not in sys.modules, (
            "open_clip must not be imported at module load time"
        )
