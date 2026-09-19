<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# MobileCLIP-S0 on Triton

The embedding server that silver calls. It serves the MobileCLIP-S0 image and text
encoders, and it does the image preprocessing on the GPU. It comes as two images with the
same interface: one for NVIDIA GPUs and one for AMD GPUs.

The CUDA image runs TensorRT FP16 engines and DALI:

```text
IMAGE_PATH  -> DALI: decode, crop, resize, normalise -> TensorRT image encoder -> EMBEDDING
IMAGE_BYTES -> DALI: decode, resize, normalise       -> TensorRT image encoder -> EMBEDDING
TEXT        -> CPU tokenisation (Python backend)     -> TensorRT text encoder  -> EMBEDDING
```

The ROCm image runs ONNX Runtime with its MIGraphX execution provider. It has no DALI, so
a Python model decodes a JPEG with rocJPEG and resizes it with torchvision on the GPU:

```text
IMAGE_PATH  -> rocJPEG + torchvision: decode, crop, resize, normalise -> ONNX image encoder -> EMBEDDING
IMAGE_BYTES -> rocJPEG + torchvision: decode, resize, normalise       -> ONNX image encoder -> EMBEDDING
TEXT        -> CPU tokenisation (Python backend)                      -> ONNX text encoder  -> EMBEDDING
```

rocJPEG decodes sequential JPEG only. The ROCm image decodes every other image on the
CPU, such as a PNG or a progressive JPEG. The two images resize with different code, so
their embeddings differ slightly.

Text tokenisation has no DALI equivalent, so it runs on the CPU before the text encoder.

This directory is vendored from `github.com/liopeer/lightly-studio`, at
`tritonserver/server_trt` and `tritonserver/server/shared_deps_server`. `shared_deps_server`
vendors Apple's `mobileclip` package, under the licence beside it. The two directories are
merged here, so the Dockerfiles build from this directory and the Makefile exports with the
local project.

## Run

The server needs a Linux host with docker, and an NVIDIA GPU with the NVIDIA container
runtime or an AMD GPU with ROCm. The CUDA image needs a GPU of compute capability 7.5 or
newer, and a driver that supports CUDA 13.1, which is R580 or newer.

```bash
export CV_LAKEHOUSE_ROOT=/absolute/path/to/the/lake
make up                        # build this checkout, and start it
make up TRITON_VERSION=0.1.0   # or pull that release, and start it
```

`make up` starts the ROCm image on a host with `/dev/kfd`, and the CUDA image otherwise.
`make up PLATFORM=cuda` overrides the choice. `make down`, `make logs` and `make image`
take the same variable. In production, pin `TRITON_VERSION` to an X.Y.Z release.

The ROCm container joins the render group of the host, which owns `/dev/kfd`. `make up`
passes the ID of that group as `RENDER_GID`. A direct `docker compose` call passes it too:

```bash
RENDER_GID=$(getent group render | cut -d: -f3) docker compose --profile rocm up -d
```

The server listens on `8010` HTTP, `8011` gRPC and `8012` metrics. From the repository
root, `make triton-up` does the same as `make up`.

## Images

A release publishes both images to `ghcr.io/liopeer/cv_lakehouse-triton`. The platform is
a tag suffix: `X.Y.Z-cuda` and `X.Y.Z-rocm`, and `X.Y-` and `X-` tags move with them.

Each image starts from a base image in `base/`, pinned by digest. The base image holds
Triton, built from source, and the large packages, such as torch. `base/README.md` tells
how to build it. To try a local base image, build with
`--build-arg BASE_IMAGE=ghcr.io/liopeer/cv_lakehouse-triton-base:rocm-dev`.

Each image installs its other Python packages from a lock file, `requirements-cuda.txt` or
`requirements-rocm.txt`. The lock files pin every package with its hash. To change a
package, edit the `.in` file beside the lock file and run `make lock`. `make lock` keeps
every other pin. `make lock UPGRADE=--upgrade` moves every pin to its newest version.

## Models

The images hold no weights. The first start downloads the MobileCLIP-S0 checkpoint and
builds the models. Then it deletes the checkpoint.

- The CUDA image builds the image and text TensorRT engines, and serializes the two DALI
  pipelines. That takes several minutes.
- The ROCm image exports the two encoders to ONNX. Then Triton loads them, and MIGraphX
  compiles a program for each batch size of 1, 2, 4 and so on up to 64. That takes about
  30 minutes on a Radeon RX 7900 XTX. A restart then takes seconds.

The models go to the `/cache` volume, under the image version, the platform and the GPU
architecture. The ROCm image also keeps the compiled MIGraphX programs there. A restart
reuses them. A new release or a different GPU builds again. A model only runs on the GPU
architecture and the library versions that built it, so CI cannot build one.

Silver sends an absolute image path, so compose mounts `$CV_LAKEHOUSE_ROOT` read only at
the same path inside the container. If a bronze directory is a symlink to a path outside
the root, add a mount for that path to `docker-compose.yml`.

Create the lake root before `make up`. Compose refuses to create it, because the daemon
creates a directory that belongs to root, and the lake belongs to the host user.

## Export

`shared_deps.export_mixins` holds two mixins for an `nn.Module` with one input and a dynamic
batch dimension. `ONNXExportMixin.export_onnx` writes ONNX.
`TensorRTMixin.export_tensorrt` calls `export_onnx`, then builds a TensorRT engine from
the ONNX. The MobileCLIP export wrappers inherit `TensorRTMixin`. The commands
`mobileclip-export-onnx` and `mobileclip-export-tensorrt` run them.

`make test` runs the tests of `shared_deps_server`.

## The model interface

- Model: `mobileclip_s0`
- Input: exactly one of `IMAGE_PATH`, `IMAGE_BYTES` or `TEXT`, each a `BYTES` tensor of
  shape `[N]`
- Optional crop inputs, with `IMAGE_PATH` only: `CROP_X`, `CROP_Y`, `CROP_WIDTH`,
  `CROP_HEIGHT`, each an `INT64` tensor of shape `[N]`. `-1` means the full image.
- Output: `EMBEDDING`, float32 `[N, 512]`, L2 normalised

One request carries many items. The entry model fans them out as concurrent sub-requests,
so the dynamic batchers on the encoder and preprocessing models coalesce them into one
execution. `cv_lakehouse.embeddings` is the client.

`mobileclip_s0` is the only model to call. Every other model in `model_repository` starts
with an underscore, because it is a step that `mobileclip_s0` reaches through. Triton
loads and serves them all, so the underscore is a convention, not a rule it enforces.

`model_repository/common` holds the models that both images share. `model_repository/cuda`
and `model_repository/rocm` hold the encoders and the image preprocessing of each image.
