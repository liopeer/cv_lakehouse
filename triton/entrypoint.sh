#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Build the models for this GPU on the first start, cache them, link them into the
# model repository and start Triton. The arguments go to tritonserver.
#
# TRITON_PLATFORM comes from the Dockerfile.
# - cuda: TensorRT engines and DALI pipelines. An engine is tied to the GPU architecture
#   and to the TensorRT version, and the image version fixes the TensorRT version.
# - rocm: ONNX models. ONNX Runtime compiles them with MIGraphX when Triton loads them,
#   and keeps the compiled programs in the cache too. They are tied to the GPU
#   architecture and to the MIGraphX version.
#
# So the cache key is the image version, the platform and the GPU architecture. A new
# release or a new GPU builds again.

set -euo pipefail
trap 'echo "entrypoint.sh failed at line ${LINENO}" >&2' ERR

CACHE_ROOT=/cache
MODEL_NAME=mobileclip_s0
MAX_BATCH_SIZE=256

version=$(cat /opt/mobileclip/version.txt)
case "${TRITON_PLATFORM}" in
    cuda)
        compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0)
        gpu_arch="sm_${compute_cap/./}"
        ;;
    rocm)
        # Read the whole output first. `grep -m1` in the pipe stops rocminfo with
        # SIGPIPE, and pipefail then fails the script.
        agents=$(rocminfo)
        gpu_arch=$(grep -m1 -oE 'gfx[0-9a-f]+' <<<"${agents}")
        ;;
    *)
        echo "Unknown TRITON_PLATFORM: ${TRITON_PLATFORM}" >&2
        exit 1
        ;;
esac
cache="${CACHE_ROOT}/${version}/${TRITON_PLATFORM}/${gpu_arch}"

build_cuda_models() {
    local build=$1
    for encoder in image text; do
        mobileclip-export-tensorrt \
            --encoder "${encoder}" \
            --model-name "${MODEL_NAME}" \
            --checkpoint-path "${build}/checkpoint/${MODEL_NAME}.pt" \
            --precision fp16 \
            --min-batch-size 1 \
            --opt-batch-size "${MAX_BATCH_SIZE}" \
            --max-batch-size 1024 \
            --workspace-size-gib 8 \
            --normalize-embeddings \
            --out "${build}/${encoder}.plan" \
            --onnx-out "${build}/${encoder}.onnx"
    done
    rm "${build}"/*.onnx

    for input_kind in path bytes; do
        python3 /opt/mobileclip/dali_pipeline.py \
            --out "${build}/image_${input_kind}.dali" \
            --max-batch-size "${MAX_BATCH_SIZE}" \
            --input-kind "${input_kind}"
    done
}

build_rocm_models() {
    local build=$1
    for encoder in image text; do
        mobileclip-export-onnx \
            --encoder "${encoder}" \
            --model-name "${MODEL_NAME}" \
            --checkpoint-path "${build}/checkpoint/${MODEL_NAME}.pt" \
            --precision fp16 \
            --max-batch-size "${MAX_BATCH_SIZE}" \
            --normalize-embeddings \
            --out "${build}/${encoder}.onnx"
    done
}

build_cache() {
    # Build into a temporary directory and rename it at the end, so an interrupted
    # build leaves nothing that looks complete.
    mkdir -p "$(dirname "${cache}")"
    local build
    build=$(mktemp -d "${cache}.build.XXXXXX")
    trap 'rm -rf "${build}"' EXIT

    echo "Building the ${TRITON_PLATFORM} models for ${gpu_arch} into ${cache}"
    hf download "apple/MobileCLIP-S0" "${MODEL_NAME}.pt" --local-dir "${build}/checkpoint"
    "build_${TRITON_PLATFORM}_models" "${build}"

    # The models are all that Triton loads. The weights go.
    rm -rf "${build}/checkpoint"
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

if [[ "${TRITON_PLATFORM}" == cuda ]]; then
    link image.plan _mobileclip_s0_image_backend model.plan
    link text.plan _mobileclip_s0_text_backend model.plan
    link image_path.dali _mobileclip_s0_image_preprocessing model.dali
    link image_bytes.dali _mobileclip_s0_image_bytes_preprocessing model.dali
else
    link image.onnx _mobileclip_s0_image_backend model.onnx
    link text.onnx _mobileclip_s0_text_backend model.onnx
    # The first load compiles each model for every batch size up to the maximum,
    # which takes minutes. The next start reads the compiled programs from here.
    mkdir -p "${cache}/migraphx"
    export ORT_MIGRAPHX_MODEL_CACHE_PATH="${cache}/migraphx"
    export ORT_MIGRAPHX_CACHE_PATH="${cache}/migraphx"
fi

exec tritonserver --model-repository=/models "$@"
