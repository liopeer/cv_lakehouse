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
| `cv_lakehouse-triton-base:rocm-r25.12-<date>` | Triton 25.12 from AMD's fork with the ensemble, Python and ONNX Runtime backends. ROCm 7.2.4, ONNX Runtime 1.23.2 with MIGraphX, rocJPEG, rocPyDecode, torch 2.11. |

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

- the four base images, by digest;
- the Ubuntu archive, as one `snapshot.ubuntu.com` time stamp;
- the Triton repositories and AMD's forks, by commit;
- TensorRT, by apt version, and ROCm, by its versioned apt repository;
- ONNX Runtime and boost, by URL and checksum.

The Python packages come from `requirements-*.txt`, with a hash for every package. To
change one, edit its `.in` file and run `make lock`. `make lock UPGRADE=--upgrade` moves
every pin to its newest version.

`build.py` clones by branch or tag, not by commit. So `fetch_sources.sh` fetches each
repository at its commit into a local repository, and git sends every fetch of a Triton
repository there. A branch that is not pinned fails the build.

## What differs from upstream

- The CUDA image derives from the public `nvidia/cuda` images, not from `nvcr.io`. So the
  NVIDIA Deep Learning Container License does not apply.
- The DALI backend builds against the DALI of `requirements-cuda.txt`, not against its own
  download. The backend and the pipelines that `triton/dali_pipeline.py` serializes use one
  DALI.
- There are no GPU metrics, because those need DCGM. CPU metrics stay.
- AMD's fork leaves the ensemble scheduler out of the ROCm build.
  `patches/rocm-core-hipify-ensemble.patch` is AMD's own fix from a later branch.
- The ROCm image links AMD's prebuilt ONNX Runtime. It builds neither ONNX Runtime nor
  MIGraphX.
- The ROCm image keeps the GPU kernels of RDNA3 and RDNA4 only (gfx110x, gfx120x). Another
  GPU runs, but without tuned kernels.

## First build

Nobody has built these images yet. The first build can fail on a detail that only a
build shows. These are the likely spots:

- `build.py --no-container-build` with `--enable-rocm` is untested upstream.
- `collect_licences.sh` fails if a component has no licence file. Its error names the
  component.
- `check_libraries.sh` fails if a library is missing in the runtime image. Its error names
  the file and the library.
- The ROCm image keeps `rocm-llvm` (about 2.2 GB), because `migraphx` depends on it through
  `hip-dev`. MIGraphX compiles with hipRTC, so the image can possibly drop it. Test that
  with an end-to-end run.

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
- MobileCLIP is not in a base image. The root `NOTICE` covers it.
