#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from dataclasses import dataclass

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Resize,
    ToTensor,
)

from shared_deps import mobileclip_compile


# All MobileCLIP variants share embed_dim=512; image_size differs only for s0.
@dataclass(frozen=True)
class MobileCLIPVariantConfig:
    model_name: str
    image_size: int
    embed_dim: int = 512


MOBILECLIP_CONFIGS: dict[str, MobileCLIPVariantConfig] = {
    "mobileclip_s0": MobileCLIPVariantConfig("mobileclip_s0", image_size=256),
    "mobileclip_s1": MobileCLIPVariantConfig("mobileclip_s1", image_size=224),
    "mobileclip_s2": MobileCLIPVariantConfig("mobileclip_s2", image_size=224),
    "mobileclip_b": MobileCLIPVariantConfig("mobileclip_b", image_size=224),
}


class MobileCLIPPreprocessor:
    """Loads an image path into a preprocessed image tensor.

    Applies the same Resize + CenterCrop + ToTensor pipeline used during
    MobileCLIP training, matching mobileclip.create_model_and_transforms().
    """

    def __init__(self, image_size: int) -> None:
        self._transform = Compose(
            [
                Resize(image_size, interpolation=InterpolationMode.BILINEAR),
                CenterCrop(image_size),
                ToTensor(),
            ]
        )

    def __call__(
        self,
        image_path: str,
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            image_path: Path to an image file readable by the Triton server.
            crop_box: Optional crop box as ``(x, y, width, height)`` in source
                image pixels.

        Returns:
            Float32 tensor of shape [3, image_size, image_size] in [0, 1].
        """
        pil_image = Image.open(image_path).convert("RGB")
        if crop_box is not None:
            x, y, width, height = crop_box
            pil_image = pil_image.crop((x, y, x + width, y + height))
        return self._transform(pil_image)


class MobileCLIPTorchBackend(nn.Module):
    """MobileCLIP image encoder — receives preprocessed tensors, returns raw embeddings.

    The full CLIP model is loaded and reparameterized for fast inference, then only
    the image encoder is retained to avoid carrying the text encoder in GPU memory.
    """

    def __init__(self, model_name: str, checkpoint_path: str) -> None:
        super().__init__()
        from shared_deps import mobileclip

        model, _, _ = mobileclip.create_model_and_transforms(
            model_name=model_name,
            pretrained=checkpoint_path,
            reparameterize=True,
        )
        # Detach from the full CLIP graph; text encoder and logit_scale are not needed.
        # Keep the encoder in fp16 internally while preserving fp32 API outputs.
        self._encoder = mobileclip_compile.compile_encoder_for_inference(
            model.image_encoder.half()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Float32 tensor of shape [B, 3, H, W] in [0, 1].

        Returns:
            Float32 tensor of shape [B, embed_dim] — NOT L2-normalized.
            Normalization is the proxy's responsibility so the backend output
            can be cached or reused before normalization if needed.
        """
        images = images.to(dtype=torch.float16)
        out = self._encoder(images)
        # MCi may return a dict (e.g. when output_dict=True is propagated).
        if isinstance(out, dict):
            return out["logits"].float()
        return out.float()
