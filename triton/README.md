<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# MobileCLIP-S0 on Triton

The embedding server that silver calls. It serves the MobileCLIP-S0 image and text
encoders as TensorRT FP16 engines, and it does every preprocessing step on the GPU.

```text
IMAGE_PATH  -> DALI: decode, crop, resize, normalise -> TensorRT image encoder -> EMBEDDING
IMAGE_BYTES -> DALI: decode, resize, normalise       -> TensorRT image encoder -> EMBEDDING
TEXT        -> CPU tokenisation (Python backend)     -> TensorRT text encoder  -> EMBEDDING
```

Text tokenisation has no DALI equivalent, so it runs on the CPU before the text engine.

This directory is vendored from `github.com/liopeer/lightly-studio`, at
`tritonserver/server_trt` and `tritonserver/server/shared_deps_server`. `shared_deps_server`
vendors Apple's `mobileclip` package, under the licence beside it. The two directories are
merged here, so the Dockerfile builds from this directory and the Makefile exports with the
local project.

## Run

The server needs a Linux host with docker and the NVIDIA container runtime.

```bash
export CV_LAKEHOUSE_ROOT=/absolute/path/to/the/lake
make up                                  # build this checkout and start it
TRITON_VERSION=0.1.0 docker compose up -d   # or start a released image
```

The server listens on `8010` HTTP, `8011` gRPC and `8012` metrics. From the repository
root, `make triton-up` does the same as `make up`.

## Engines

The image holds no weights and no engine. The first start downloads the MobileCLIP-S0
checkpoint, builds the image and text TensorRT engines, and serializes the two DALI
pipelines. That takes several minutes. Then it deletes the checkpoint and the ONNX.

The engines go to the `/cache` volume, under the image version and the GPU compute
capability. A restart reuses them. A new release or a different GPU builds again. An
engine only runs on the GPU architecture and the TensorRT version that built it, so CI
cannot build one.

Silver sends an absolute image path, so compose mounts `$CV_LAKEHOUSE_ROOT` read only at
the same path inside the container. If a bronze directory is a symlink to a path outside
the root, add a mount for that path to `docker-compose.yml`.

## The model interface

- Model: `mobileclip_s0`
- Input: exactly one of `IMAGE_PATH`, `IMAGE_BYTES` or `TEXT`, each a `BYTES` tensor of
  shape `[N]`
- Optional crop inputs, with `IMAGE_PATH` only: `CROP_X`, `CROP_Y`, `CROP_WIDTH`,
  `CROP_HEIGHT`, each an `INT64` tensor of shape `[N]`. `-1` means the full image.
- Output: `EMBEDDING`, float32 `[N, 512]`, L2 normalised

One request carries many items. The entry model fans them out as concurrent sub-requests,
so the dynamic batchers on the TensorRT and DALI models coalesce them into one execution.
`cv_lakehouse.embeddings` is the client.

`mobileclip_s0` is the only model to call. Every other model in `model_repository` starts
with an underscore, because it is a step that `mobileclip_s0` reaches through. Triton
loads and serves them all, so the underscore is a convention, not a rule it enforces.
