#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Build the engines and the DALI pipelines for this GPU on the first start, cache them,
# link them into the model repository and start Triton. The arguments go to tritonserver.
#
# An engine is tied to the GPU architecture and to the TensorRT version, and the image
# version fixes the TensorRT version. So the cache key is the image version and the
# compute capability, and a new release or a new GPU builds again.

set -euo pipefail

CACHE_ROOT=/cache
MODEL_NAME=mobileclip_s0
DALI_MAX_BATCH_SIZE=256

version=$(cat /opt/mobileclip/version.txt)
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0)
cache="${CACHE_ROOT}/${version}/sm_${compute_cap/./}"

build_cache() {
    # Build into a temporary directory and rename it at the end, so an interrupted
    # build leaves nothing that looks complete.
    mkdir -p "${CACHE_ROOT}/${version}"
    local build
    build=$(mktemp -d "${cache}.build.XXXXXX")
    trap 'rm -rf "${build}"' EXIT

    echo "Building the engines for compute capability ${compute_cap} into ${cache}"
    hf download "apple/MobileCLIP-S0" "${MODEL_NAME}.pt" --local-dir "${build}/checkpoint"

    for encoder in image text; do
        mobileclip-export-tensorrt \
            --encoder "${encoder}" \
            --model-name "${MODEL_NAME}" \
            --checkpoint-path "${build}/checkpoint/${MODEL_NAME}.pt" \
            --precision fp16 \
            --min-batch-size 1 \
            --opt-batch-size 256 \
            --max-batch-size 1024 \
            --workspace-size-gib 8 \
            --normalize-embeddings \
            --out "${build}/${encoder}.plan" \
            --onnx-out "${build}/${encoder}.onnx"
    done

    for input_kind in path bytes; do
        python3 /opt/mobileclip/dali_pipeline.py \
            --out "${build}/image_${input_kind}.dali" \
            --max-batch-size "${DALI_MAX_BATCH_SIZE}" \
            --input-kind "${input_kind}"
    done

    # The engines are all that Triton loads. The weights and the ONNX go.
    rm -rf "${build}/checkpoint" "${build}"/*.onnx
    mv "${build}" "${cache}"
    trap - EXIT
}

link() {
    mkdir -p "/models/$2/1"
    ln -sfn "${cache}/$1" "/models/$2/1/$3"
}

if [[ ! -d "${cache}" ]]; then
    build_cache
fi

link image.plan _mobileclip_s0_image_backend_trt model.plan
link text.plan _mobileclip_s0_text_backend_trt model.plan
link image_path.dali _mobileclip_s0_image_preprocessing model.dali
link image_bytes.dali _mobileclip_s0_image_bytes_preprocessing model.dali

exec tritonserver --model-repository=/models "$@"
