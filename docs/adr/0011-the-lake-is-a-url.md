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

### The store

- `CV_LAKEHOUSE_ROOT` takes a URL: `s3://`, `gs://`, `az://`, `file://` or `memory://`.
  A plain path still means a local directory.
- `CV_LAKEHOUSE_STORAGE_OPTIONS` is a JSON object, such as the S3 endpoint and keys. The
  settings parse it and pass it on every call. A broken value then stops the process,
  and fsspec never falls back to real AWS.
- `LakeStore` in core is the one way to the lake. A domain never imports fsspec.
- `s3fs`, `gcsfs` and `adlfs` are dependencies of core, so a scheme never fails as an
  `ImportError` in a run.
- A location is a key relative to the root, or a full URL for a linked copy outside it.
  No manifest or table stores an absolute local path. `LakeStore` resolves both kinds.
- pyarrow reads and writes Parquet through the fsspec filesystem of the store. DuckDB
  reads it through `register_filesystem`.

### Bronze

A dataset can be terabytes. Bronze never stages a dataset on local disk, and never
copies a linked copy.

- A download streams from the publisher into the store. Large files go as a multipart
  upload. The checksum is computed on the stream, and a file that fails it is deleted.
- An archive unpacks from the store into the store. A tar archive is read as a stream.
  A zip archive is read by range requests, because its index is at the end. Each member
  streams to its own key. The archive stays, as ADR 0008 requires.
- A linked copy stays where it is. Its manifest records its URL, such as
  `s3://datasets/coco2017` or `file:///mnt/8_tb/datasets/coco2017`. Bronze verifies that
  the copy holds every published part, as before. Silver and gold store locations under
  that URL.
- On a local root, a download still resumes inside a file. On an object store, a file
  that fails during download restarts from its first byte. A completed file is never
  downloaded again.

### Triton

Triton reads an image in three ways. Silver picks one per lake.

- `IMAGE_PATH` with a local path, as today. Triton then mounts the lake at that path.
- `IMAGE_PATH` with an `https://` URL. Triton fetches it, and holds no credentials. On an
  object store, silver sends a presigned URL with a short expiry.
- `IMAGE_BYTES` with the encoded image.

The `CROP_*` inputs work with all three. A URL is fetched to bytes, and then takes the
bytes pipeline, which gains the crop inputs. Triton refuses any other scheme.

### Studio

LightlyStudio reads every image through `fsspec.url_to_fs`, so it reads `s3://`,
`gs://` and `az://` paths. The sync writes the resolved URL of each image as its
`file_path_abs`. The studio image installs `s3fs`, `gcsfs` and `adlfs`. The deployment
gives it read access to the lake through the fsspec environment variables, such as
`FSSPEC_S3_ENDPOINT_URL` and the `AWS_*` keys.

## Consequences

- One code path serves a local disk, S3, GCS, Azure and a test in memory.
- On an object store, the code server and Triton share no mount, so they can run on any
  node.
- No run needs local disk of the size of a dataset or an archive.
- A linked copy costs no space in the lake. The lake then depends on the copy: if the
  copy moves, bronze of that dataset must link it again.
- An object store keeps no file dates. A file unpacked from an archive keeps its bytes,
  but not its date.
- A file that fails during download on an object store loses its progress. For a very
  large file, a resumable multipart upload is future work.
- An unpack from an object store reads every byte of the archive back once.
- Presigning needs credentials that can sign. On GCS that is a service account key, and
  on Azure an account key or a SAS.
- Triton needs a release: the URL fetch and the crop inputs on the bytes pipeline.
- The existing lake holds 16 GB of bronze in two datasets. A deployment on a new root
  downloads them again. A larger lake later moves by copying its objects, since every
  location is relative to the root.
