<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# MobileCLIP-S0 on Triton

The embedding server that silver calls. It serves the MobileCLIP-S0 image and text
encoders. It comes as two images with the same interface: one for NVIDIA GPUs and one for
AMD GPUs.

The CUDA image runs TensorRT FP16 engines and DALI:

```text
IMAGE_PATH  -> DALI: decode, crop, resize, normalise -> TensorRT image encoder -> EMBEDDING
IMAGE_BYTES -> DALI: decode, resize, normalise       -> TensorRT image encoder -> EMBEDDING
TEXT        -> CPU tokenisation (Python backend)     -> TensorRT text encoder  -> EMBEDDING
```

The ROCm image runs one Python model in torch. Triton does not see the GPU. torch in the
Python stub drives it:

```text
IMAGE_PATH  -> read or fetch, Pillow + torchvision on CPU threads: decode, crop, resize -> torch.compile image encoder -> EMBEDDING
IMAGE_BYTES -> Pillow + torchvision on CPU threads: decode, crop, resize                 -> torch.compile image encoder -> EMBEDDING
TEXT        -> tokenisation                                                             -> torch.compile text encoder  -> EMBEDDING
```

The model reads each distinct path or URL once, and decodes it once for all its crops. A
large JPEG decodes at 1/2, 1/4 or 1/8 scale, so each crop keeps 256 pixels or more. The
GPU embeds batches of 64 while the threads preprocess the next crops. So no image passes
between processes, and a request holds at most one full image per thread.

The parameters of `mobileclip_s0` tune it: `batch_size`, 64 by default, `workers`, 16 by
default, and `fetch_timeout_seconds`, 60 by default. The two images resize with different
code, so their embeddings differ slightly.

Text tokenisation has no DALI equivalent, so it runs on the CPU before the text encoder.

This directory is vendored from `github.com/liopeer/lightly-studio`, at
`tritonserver/server_trt` and `tritonserver/server/shared_deps_server`. `shared_deps_server`
vendors Apple's `mobileclip` package, under the licence beside it. The two directories are
merged here, so the Dockerfiles build from this directory and the Makefile exports with the
local project.

## Build

The server needs a Linux host with docker, and an NVIDIA GPU with the NVIDIA container
runtime or an AMD GPU with ROCm. The CUDA image needs a GPU of compute capability 7.5 or
newer, and a driver that supports CUDA 13.1, which is R580 or newer.

```bash
make image                  # build this checkout as :dev-rocm or :dev-cuda
make image PLATFORM=cuda    # or build the other platform
```

`make image` builds the ROCm image on a host with `/dev/kfd`, and the CUDA image
otherwise.

The deployment stack runs the server. It holds the compose file, and it pins
`TRITON_VERSION` to an X.Y.Z release. This directory builds and tests the image, and it
starts nothing.

A container needs the lake bound at the same absolute path inside and outside, because a
bronze manifest holds absolute paths. The ROCm container also needs `/dev/kfd`,
`/dev/dri`, the render group of the host, `seccomp=unconfined` and a 2 GB `/dev/shm`.

The server listens on `8000` HTTP, `8001` gRPC and `8002` metrics. The deployment stack
publishes them on `8010`, `8011` and `8012`.

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
builds the models. The CUDA image then deletes the checkpoint.

- The CUDA image builds the image and text TensorRT engines, and serializes the two DALI
  pipelines. That takes several minutes.
- The ROCm image keeps the checkpoint, which torch loads. Each instance compiles both
  encoders with torch.compile when it loads. That takes under a minute on a Radeon RX
  7900 XTX, and a restart reuses the compiled kernels.

The models go to the `/cache` volume, under the image version, the platform and the GPU
architecture. The ROCm image also keeps the kernels of torch.compile there. A restart
reuses them. A new release or a different GPU builds again. A model only runs on the GPU
architecture and the library versions that built it, so CI cannot build one.

A local image path is read inside the container. For a lake on a local disk, the stack
mounts `$CV_LAKEHOUSE_ROOT` read only at the same path inside the container. A lake in
an object store needs no mount, since silver sends presigned URLs. If bronze links a
local copy outside the root, add a mount for that path to the compose file of the stack.

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
- `IMAGE_PATH` is a local path inside the container, or an `http` or `https` URL. The
  server holds no credentials, so a URL into private storage, such as S3, is presigned.
  The entry model fetches each distinct URL of a request once, then sends the bytes
  through the bytes pipeline. Any other scheme, such as `s3://`, is refused. An error
  names a URL without its query, so no signature reaches a log.
- Optional crop inputs, with `IMAGE_PATH` or `IMAGE_BYTES`: `CROP_X`, `CROP_Y`,
  `CROP_WIDTH`, `CROP_HEIGHT`, each an `INT64` tensor of shape `[N]`. `-1` means the
  full image.
- Output: `EMBEDDING`, float32 `[N, 512]`, L2 normalised

One request carries many items. The entry model fans them out as concurrent sub-requests,
so the dynamic batchers on the encoder and preprocessing models coalesce them into one
execution. `cv_lakehouse.embeddings` is the client.

`mobileclip_s0` is the only model to call. In the CUDA image, every other model starts
with an underscore, because it is a step that `mobileclip_s0` reaches through. Triton
loads and serves them all, so the underscore is a convention, not a rule it enforces.

Two parameters of the CUDA `mobileclip_s0` tune the fetch: `fetch_workers`, 32 by
default, and `fetch_timeout_seconds`, 60 by default.

`model_repository/cuda` and `model_repository/rocm` hold the models of each image.
