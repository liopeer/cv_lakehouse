<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# Triton base images

The images that the Triton server in `triton/` builds on. Each one holds Triton, its
backends, the GPU runtime and the heavy Python packages. They change rarely, so they build
on a local machine and not in CI.

| Image | Holds |
| --- | --- |
| `cv_lakehouse-triton-base:cuda-r25.12-<date>` | Triton 25.12 with the ensemble, Python, TensorRT and DALI backends. CUDA 13.1, TensorRT 10.14, DALI 1.52, CPU torch 2.14. |
| `cv_lakehouse-triton-base:rocm-r25.12-<date>` | Triton 25.12 with the Python backend, built without GPU support. torch 2.14 for ROCm 7.2, with the hipBLASLt and rocBLAS kernels of gfx1100 only. |

The images are `ghcr.io/liopeer/cv_lakehouse-triton-base`, public. The Dockerfiles in
`triton/` pin a base image by digest.

## Build and push

```bash
make -C triton/base build PLATFORM=rocm   # or PLATFORM=cuda
make -C triton/base push PLATFORM=rocm    # tags rocm-r25.12-<today>, prints the digest
```

`make push` needs `docker login ghcr.io` with a token that has the `write:packages` scope.
Then pin the printed digest in `triton/Dockerfile.rocm` or `triton/Dockerfile.cuda`.

A build takes the whole machine for up to an hour, and it needs about 60 GB of free disk.
`make clean` removes the build cache.

## Pins

`versions.env` holds every input of both images, and the Makefile passes each line as a
build argument:

- the base images, by digest;
- the Ubuntu archive, as one `snapshot.ubuntu.com` time stamp;
- the Triton repositories, by commit;
- TensorRT, by apt version;
- boost, by URL and checksum.

The Python packages come from `requirements-*.txt`, with a hash for every package. To
change one, edit its `.in` file and run `make lock`. `make lock UPGRADE=--upgrade` moves
every pin to its newest version.

`build.py` clones by branch or tag, not by commit. So `fetch_sources.sh` fetches each
repository at its commit into a local repository, and git sends every fetch of a Triton
repository there. A branch that is not pinned fails the build.

## What differs from upstream

- The NVIDIA Deep Learning Container License governs every NVIDIA container image,
  `nvidia/cuda` too. So only the CUDA builder uses `nvidia/cuda`. The CUDA runtime stage
  starts from `ubuntu:24.04`, and it installs the CUDA runtime, CUPTI and TensorRT from
  NVIDIA's apt repository.
- The CUDA image has no `cuda-compat`, so the host driver supports CUDA 13.1 itself.
- The DALI backend builds against the DALI of `requirements-cuda.txt`, not against its own
  download. The backend and the pipelines that `triton/dali_pipeline.py` serializes use one
  DALI.
- There are no GPU metrics, because those need DCGM. CPU metrics stay.
- The ROCm image builds upstream Triton without GPU support, with the Python backend only.
  torch in the Python stub drives the GPU. The PyTorch builds for ROCm bundle the ROCm
  libraries, so the image installs no ROCm package.
- The ROCm image keeps the hipBLASLt and rocBLAS kernels of `GPU_ARCHS` only, gfx1100 by
  default. That saves 5.1 GB. Another GPU needs a build with its architecture.

## First build

Nobody has built these images yet. The first build can fail on a detail that only a
build shows. These are the likely spots:

- `collect_licences.sh` fails if a component has no licence file. Its error names the
  component.
- `check_libraries.sh` fails if a library is missing in the runtime image. Its error names
  the file and the library.

## Licences

Each image holds the licence and notice files of every component in `/opt/licences/`, one
directory per component. `collect_licences.sh` copies them from the pinned sources during
the build. Every apt package keeps its own licence under `/usr/share/doc/`, and every
Python package in its `dist-info` directory.

- Triton and its backends are BSD-3-Clause. A binary distribution must include their
  licence, which `/opt/licences/triton-*` does.
- gRPC and abseil are Apache-2.0, and their NOTICE files are in
  `/opt/licences/triton-linked-libraries`.
- The CUDA image redistributes the CUDA runtime and the TensorRT runtime libraries. The
  CUDA EULA (Attachment A) and the TensorRT license (supplement, section 8.2) allow that for
  the runtime libraries, in an application with material additional functionality. Triton
  is that application. The image holds no headers and no developer tools.
- The ROCm image holds the ROCm libraries inside the torch wheel, as PyTorch publishes
  it.
- MobileCLIP is not in a base image. The root `NOTICE` covers it.
