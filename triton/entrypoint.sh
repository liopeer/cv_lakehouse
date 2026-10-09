#!/usr/bin/env bash
##
## SPDX-License-Identifier: MIT
## Copyright (c) 2025–2026 Lionel Peer
##

# Build the models for this GPU on the first start, cache them, link them into the
# model repository and start Triton. The arguments go to tritonserver.
#
# TRITON_PLATFORM comes from the base image.
# - cuda: TensorRT engines and DALI pipelines. An engine is tied to the GPU architecture
#   and to the TensorRT version, and the image version fixes the TensorRT version.
# - rocm: the checkpoint. torch.compile keeps the kernels that it compiles in the cache
#   too. They are tied to the GPU architecture and to the torch version.
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
        gpu_arch=$(python3 -c \
            "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")
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

    # The engines are all that Triton loads. The weights go.
    rm -rf "${build}/checkpoint"
}

# torch loads the checkpoint itself.
build_rocm_models() {
    :
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
    link checkpoint/mobileclip_s0.pt mobileclip_s0 mobileclip_s0.pt
    export TORCHINDUCTOR_CACHE_DIR="${cache}/inductor"
    export TRITON_CACHE_DIR="${cache}/triton"
fi

# A restarted container keeps the /dev/shm of its pod, and with it the shared-memory
# regions of the Python backend that a killed Triton never removed. They fill /dev/shm,
# and every request that needs a new region fails. No Triton runs in this container
# yet, so none of them is in use. A /dev/shm shared with another Triton, as with
# --ipc=host, would lose that Triton's regions too.
rm -f /dev/shm/triton_python_backend_shm_region_*

exec tritonserver --model-repository=/models "$@"
