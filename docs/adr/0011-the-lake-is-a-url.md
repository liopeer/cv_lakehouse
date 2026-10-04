<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0011: The lake root is a URL, read through fsspec

- Status: proposed
- Date: 2026-10-04

## Context

The lake is a directory on one disk. Every layer reads and writes it with `pathlib`.
The bronze manifests hold absolute paths, and silver sends absolute paths to Triton.
So the code server and Triton mount the lake at the same path, on the same host.

The lakehouse moves to Kubernetes. There, the lake goes to S3 on SeaweedFS. Other
deployments use GCS or Azure. A test wants a lake in memory.

lightly-lakehouse and lightly-modelrepo solve the same problem with fsspec. One store is
rooted at a URL, such as `s3://lakehouse`, `gs://bucket`, `file:///data` or `memory://`.
Every object has a key relative to the root.

## Decision

The lake root is a URL, and the lake is read and written through fsspec.

- `CV_LAKEHOUSE_ROOT` takes a URL: `s3://`, `gs://`, `az://`, `file://` or `memory://`.
  A plain path still means a local directory.
- `CV_LAKEHOUSE_STORAGE_OPTIONS` is a JSON object, such as the S3 endpoint and keys. The
  settings parse it and pass it on every call. A broken value then stops the process,
  and fsspec never falls back to real AWS.
- `LakeStore` in core is the one way to the lake. It replaces `Path` under the root,
  and addresses an object by its key. A domain never imports fsspec.
- `s3fs`, `gcsfs` and `adlfs` are dependencies of core, so a scheme never fails as an
  `ImportError` in a run.
- Every path in a manifest or a table is a key relative to the root. No absolute path is
  stored. Gold already does this.
- pyarrow reads and writes Parquet through the fsspec filesystem of the store. DuckDB
  reads it through `register_filesystem`.
- Bronze downloads, verifies and unpacks in a local scratch directory,
  `CV_LAKEHOUSE_SCRATCH_DIR`. It then uploads each file, and the archive, to its key. A
  download resumes from scratch. ADR 0008 holds as before.
- A linked copy stays a symlink on a local root. On an object store, bronze copies each
  published file from the copy and verifies it. Readers see bronze at the same keys
  either way.
- Silver sends the image bytes to Triton as `IMAGE_BYTES`, not a path. Triton then needs
  no lake, no mount and no credentials. The bytes pipeline of Triton gains the `CROP_*`
  inputs that the path pipeline has.

## Consequences

- One code path serves a local disk, S3, GCS, Azure and a test in memory.
- The code server and Triton no longer share a mount, so they can run on any node.
- The existing lake is not migrated. It holds 16 GB of bronze. A deployment on a new
  root materializes bronze again.
- An object store keeps no file dates. A file unpacked from an archive keeps its bytes,
  but not its date.
- A bronze unpack needs local scratch space of the size of its largest archive.
- A crop sends the whole image to Triton, once per box. An image with many boxes is sent
  many times.
- Triton needs a release before silver can use `IMAGE_BYTES` with crops.
- Studio reads images by `image_base` and `image_path`. It needs a way to read an image
  from the store. That is open, and comes last.
