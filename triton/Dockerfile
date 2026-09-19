##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Includes the TensorRT, DALI, and Python backends used by the ensemble.
# Image preprocessing runs on GPU via the bundled DALI backend, so no PIL/
# libjpeg toolchain is needed here -- the BLS model only needs numpy. Text
# preprocessing (tokenization) has no DALI equivalent and reuses
# shared_deps.mobileclip's tokenizer, which pulls in torch/torchvision/
# open-clip-torch/timm as a side effect of module import; shared_deps is
# installed with --no-deps to skip its unrelated pillow-simd dependency
# (only used by the image encoder module, which the text path never imports).
#
# The image holds no weights and no engine. `entrypoint.sh` builds both engines and
# both DALI pipelines on the first start, for the GPU it finds, and caches them in
# /cache. An engine only runs on the GPU architecture and the TensorRT version that
# built it, so it cannot be built in CI, which has no GPU.
FROM nvcr.io/nvidia/tritonserver:25.03-py3

WORKDIR /opt/mobileclip

# tensorrt-cu12 10.9 matches the TensorRT that Triton 25.03 runs the engines with.
COPY shared_deps_server/ /opt/mobileclip/shared_deps_server/
RUN pip install --no-cache-dir --no-deps ./shared_deps_server/ \
    && pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu126 \
    "numpy>=1.24" \
    "open-clip-torch>=2.20" \
    "timm>=0.9.5" \
    "torch>=2.5,<3" \
    "onnx>=1.16" \
    "onnxscript>=0.1" \
    "tensorrt-cu12==10.9.0.34" \
    "huggingface_hub>=0.34"

COPY model_repository/ /models/
COPY dali_pipeline.py entrypoint.sh version.txt /opt/mobileclip/

# The base image's entrypoint sets up the CUDA environment, then runs ours.
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/mobileclip/entrypoint.sh"]
