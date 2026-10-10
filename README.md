<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# cv_lakehouse

[![build](https://github.com/liopeer/cv_lakehouse/actions/workflows/build.yml/badge.svg)](https://github.com/liopeer/cv_lakehouse/actions/workflows/build.yml)
[![release](https://github.com/liopeer/cv_lakehouse/actions/workflows/release.yml/badge.svg)](https://github.com/liopeer/cv_lakehouse/actions/workflows/release.yml)

A medallion lakehouse on Dagster. Computer vision is its first domain, and each domain
is one package with its own code location. See [docs/adr](docs/adr/README.md) for the
decisions behind the layout.

The first pipeline covers PII redaction: detecting human faces and vehicle licence
plates, so that they can be blurred.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Then run:

```bash
make sync
```

## Development

```bash
make dev     # start the Dagster UI on http://localhost:3000
make check   # format, lint, typecheck, test, validate definitions
make api     # serve gold on http://localhost:8000
make help    # list every target
```

## Releases and images

The app, the Triton server and the LightlyStudio images release independently. For each
one, release-please keeps a release PR open. Merging that PR tags the release and pushes the image. Any other merge
publishes nothing.

| Component | Tag | Image |
| --- | --- | --- |
| app | `vX.Y.Z` | `ghcr.io/liopeer/cv_lakehouse` |
| triton | `triton-vX.Y.Z` | `ghcr.io/liopeer/cv_lakehouse-triton` |
| studio | `studio-vX.Y.Z` | `ghcr.io/liopeer/cv_lakehouse-studio` and `-studio-sync` |

Each image is tagged `X.Y.Z`, `X.Y` and `X`. Pin `X.Y.Z`, because the other two move. The
app image is a Dagster code location on port 4000. The same image serves the
[gold API](#the-gold-api) on port 8000 when it starts with the uvicorn command.

No image holds model weights. The Triton image downloads the checkpoint and builds its
engines on the first start. See [triton/README.md](triton/README.md).

## The medallion model

| Layer | What it is | How it materializes |
| --- | --- | --- |
| bronze | the raw dataset, complete, as published | download it, or link a complete copy |
| silver | one dataset, normalised | two Parquet files per split, on the class registry, and two of vectors |
| gold | every dataset in one table set | two Parquet files per dataset and split, with a role |

Bronze holds every file that a dataset publishes, byte for byte, in the layout of a
manual download. See [Bronze](#bronze) below. Silver keeps
every class the registry knows. A consumer picks the classes it wants, so a new class
costs a silver rebuild, not a download. A silver rebuild reads no network.

Gold is the union of every silver dataset on disk. See [Gold](#gold) below.

### Why silver is Parquet

labelformat's COCO writer emits `image_id`, `category_id` and `bbox` per annotation, and
its reader drops anything else, so not even COCO's own `area` or `iscrowd` survives a
round trip. That cost us real signal: WIDER FACE grades every face on five axes and flags
the ones its annotators rejected, and Open Images marks a crowd box and a depiction.

Silver now writes one row per image and one row per box, so a box carries its grades, its
flags and the class name its source used. The layer is queryable with no code of ours:

```bash
build=$(jq -r .build_id "$CV_LAKEHOUSE_ROOT/silver/wider_face/_silver.json")
duckdb -c "select class_name, count(*), avg(w * h)
           from read_parquet('$CV_LAKEHOUSE_ROOT/silver/wider_face/builds/$build/boxes/*.parquet')
           group by 1"
```

A run writes a new build under `builds/`, checks every box, and only then replaces
`_silver.json`, which names the build. A run that fails leaves the last good build in
place. Silver keeps the current build and the one before it.

Every dataset writes the same columns and leaves the ones it knows nothing about null, so
a scan across datasets needs no `union_by_name`. An `attr_*` column holds a per box
attribute:

| Dataset | Columns it fills |
| --- | --- |
| wider_face | `attr_blur`, `attr_expression`, `attr_illumination`, `attr_occlusion`, `attr_pose`, `attr_invalid` |
| open_images | `attr_is_group_of`, `attr_is_depiction` |
| pp4av | none |

A box is keyed by `box_id`, an md5 of the dataset, the split, the file name and the box's
position in the source, written as a UUID. A box that normalisation drops leaves every
other id as it was, so `box_id` names a box across layers and in LightlyStudio.
`box_index` only numbers the boxes that one image kept.

LightlyStudio cannot display these yet: its `CreateObjectDetection` carries a class name,
a confidence and a box, and its public `Annotation` exposes no metadata. They are there
for a query and for training.

### Corrections in silver

Silver applies the latest [correction snapshot](#corrections) of its dataset on top of
the normalised source. Bronze stays as published.

| A curator | Silver |
| --- | --- |
| relabels a box | takes the class. The box keeps its exact float coordinates |
| moves a box | takes the whole pixels of LightlyStudio, clipped to the image |
| deletes a box | drops it, and counts it under `dropped_reasons` as `studio_deleted` |
| draws a box | adds it, under the id that LightlyStudio gave it |

- Three columns say what happened to a box: `origin` is `source` or `studio`, and
  `is_class_corrected` and `is_geometry_corrected` mark a change.
- A corrected box keeps its `attr_*` grades and its `source_class`. A drawn box has
  neither.
- A label that the class registry does not name becomes `other`, and `source_class`
  carries the label. To make it a class, add it to the registry.
- The check `corrections_are_applied` lists such labels, and the corrections that found
  no box. It warns and never blocks.
- `_silver.json` names the snapshot that silver applied.

A new snapshot marks silver stale. Rebuild silver and gold by hand.

### Flagged boxes

Silver keeps every box a source publishes, including the ones a flag disqualifies, and
**leaves the judgement to whoever reads the layer**. A crowd box over a dozen faces is a
poor positive, but dropping it at silver labels those faces background and teaches a
detector not to fire there. Kept and flagged, a consumer can use it as an ignore region
instead. The same goes for a depiction and for a WIDER FACE region the annotators
rejected.

So silver's box count is larger than what a training set would use. Normalisation
genuinely throws away only one thing, a box that clipping leaves with no area, and the
run logs it under `dropped_reasons`.

### Embeddings

The asset `silver/<dataset>_embeddings` embeds the current silver build. Every image gets
one MobileCLIP-S0 vector, and so does every box, cropped to its own pixels. The vectors
are 512 float32 and L2 normalised, so a cosine similarity is a dot product.

They are a separate asset because a vector is 2 KB next to a box row of a few dozen
bytes, and most queries over silver want the boxes. Silver then means the same on a
machine with no server. The keys are the keys of the silver files, so a join needs
nothing else. A file is sorted by `image_id` or `box_id`, so a query on a few ids reads a
few row groups. `_embeddings.json` names the current build, and the silver build that
it covers.

A [Triton](triton/README.md) server does the work. It is in `triton/`, it runs the two
encoders as TensorRT engines, and it crops, resizes and normalises on the GPU, so this
repository sends a path and a box and never opens an image. The deployment stack starts
it, and `triton/` builds the image.

```bash
export CV_LAKEHOUSE_TRITON_URL=localhost:8011
```

The embeddings asset fails when the URL is unset or the server does not answer. It
never deletes a vector.

The server resolves the path itself, so the stack mounts `$CV_LAKEHOUSE_ROOT` read only
at the same path inside the container. If bronze links a copy outside the root, add a
mount for that path to the compose file of the stack.

A run embeds only what has no vector yet, on any code. An image keeps its vector,
because bronze pins the pixels. A box keeps its vector while its crop on the pixel grid
is the same, so a relabelled box is not embedded again, and a moved or a drawn box is.
The run metadata counts the vectors it reused.

The footer of an embedding file names the model and `EMBEDDING_VERSION`. A file of
another model or version is embedded again. A new model behind the same name is not
detected: bump `EMBEDDING_VERSION`. See
[ADR 0013](docs/adr/0013-silver-vectors-are-sorted-and-kept.md).

A run writes new vectors to closed parts of 65536 vectors, under `parts/`. If a run
stops, the next run reuses every closed part. The run log reports the progress of each
step once a minute, with the rate and the time left.

Before this asset, silver kept the vectors in `silver/<dataset>/embeddings/` and
`crop_embeddings/`. The first run of the asset reuses them. Delete them after that run.

### The class registry

`CanonicalClass` in `domains/cv/src/lakehouse_cv/contract/class_registry.py` holds the vocabulary that
silver writes. These are our classes, not any dataset's:

| id | name |
| --- | --- |
| 0 | face |
| 1 | license_plate |
| 2 | head |
| 3 | person |
| 4 | text |
| 5 | other |

The enum is the source of truth, so no id ever shifts. To add a class, append a member
with the next free value.

A source maps its own category names onto these names in its `DatasetSpec`. A category
that the map omits falls to `default_class`, which every source sets, so a gap in a map
never loses a box. Open Images boxes 601 classes and names four of them, so the rest
land as `other`: silver keeps the box and loses the label, and `source_class` still
carries what the source called it. What `other` is worth is a question for a training
pipeline, not for silver.

### The datasets

| Dataset | Role | Licence | Commercial |
| --- | --- | --- | --- |
| open_images | backbone, both classes | CC-BY-4.0 and CC-BY-2.0 | yes |
| wider_face | face recall at small scale | CC-BY-NC-ND-4.0 | no |
| pp4av | road driving evaluation only | CC-BY-NC-ND-4.0 | no |

`SilverManifest.commercial_use` carries each dataset's answer, so a consumer that mixes
them can enforce its own licence rule.

### Gold

Gold joins the silver datasets into one table set. A consumer reads gold and no silver.

- A row carries `image_id` or `box_id`, the dataset, the split and a role.
- A role is `train`, `val` or `test`. Each source maps its own split names onto the
  roles in `DatasetSpec.split_roles`.
- Gold holds no flagged box. A crowd box, a depiction and a region the annotators
  rejected stay in silver.
- `image_path` is relative to the lake root, so gold names a pixel on any machine. A pixel
  of a copy that bronze links from outside the lake keeps its absolute path or URL.
- `changed_at` is the build that last changed the row. A row that a rebuild leaves
  equal keeps its time.
- Gold copies no embedding. The vectors stay in the silver files.

| Dataset | Split | Role |
| --- | --- | --- |
| open_images | train, validation, test | train, val, test |
| wider_face | train, val | train, val |
| pp4av | test, fisheye | test, test |

A build writes a new directory under `gold/versions/`, and then replaces `_gold.json`,
which names the current version. A reader takes the version from `_gold.json`, so it
never sees a half written build. Gold keeps the current version and the one before it.

```bash
duckdb -c "select dataset, role, class_name, count(*)
           from read_parquet('$CV_LAKEHOUSE_ROOT/gold/versions/1/boxes/*/*.parquet')
           group by all"
```

A dataset with no silver on disk is left out, and the run logs its name.

### Eval releases

Gold changes with every correction, and a benchmark must not. The asset
`gold/eval_release` freezes the val and test rows of every dataset under a number.

- Each run copies the val and test files of the current gold version to
  `gold/releases/<number>/`, and writes `_release.json` with the checksum of every file.
- A release never changes. The check `releases_are_unchanged` verifies every checksum.
- The curators keep working on the gold above it. Their fixes to a val or a test row
  reach the next release, and no earlier one.
- The run fails when the val and test rows are those of the last release.
- One number covers every dataset, so a benchmark names one release.
- `_release.json` names the gold version, the silver code version and the correction
  snapshot of each dataset.
- A release holds no vector.

Run the asset by hand, when a round of corrections is done.

### The gold API

A read only HTTP API serves gold, so a consumer needs no access to the lake files.

```bash
make api
curl 'localhost:8000/v1/boxes?dataset=wider_face&role=val&class_name=face&release=1'
```

| Endpoint | Rows |
| --- | --- |
| `/v1/meta` | the gold manifest: the version, the datasets, the splits and their roles |
| `/v1/releases`, `/v1/releases/{n}` | the eval releases, each with its files |
| `/v1/classes` | the class registry, also before gold exists |
| `/v1/images` | one per image |
| `/v1/images/count` | none: `{"count": n}`, the number of images that the filter selects |
| `/v1/boxes` | one per box |
| `/v1/embeddings` | `image_id` and the MobileCLIP vector of the image |
| `/v1/crop_embeddings` | `box_id` and the MobileCLIP vector of the box crop |

| Parameter | Selects |
| --- | --- |
| `dataset`, `split`, `role` | rows of these datasets, splits or roles |
| `release` | which val and test rows: the number of a release, or `draft` |
| `class_name` | boxes of these classes. Not on `/v1/images` and `/v1/embeddings` |
| `commercial_use` | rows of the datasets with this licence answer |
| `changed_since` | rows that a gold build changed after this time |
| `shard`, `num_shards` | the images of one shard, and their boxes and vectors |
| `limit`, `after` | one page. `limit` is 1000 by default and at most 100000 |

A request that can return val or test rows must name a `release`. A number gives the
frozen rows of that release, and `draft` gives the gold that the curators work on. So a
benchmark never moves to other rows without a change on its side. Train rows are always
those of the current gold, and a request with only `role=train` needs no release. The
vector endpoints serve the draft only.

The shard of an image is the first 32 bits of its `image_id`, modulo `num_shards`. It
never changes, so a consumer that keeps `num_shards` keeps its images. See
[ADR 0011](docs/adr/0011-gold-serves-stable-shards.md).

A parameter that is given more than once matches any of its values. The rows come in
the order of their id. A full page carries `next_after`, and the next request sends it
as `after`. A page of vectors can hold fewer rows than `limit` and still carry it, as an
id can lack a vector. Read until a page has no `next_after`.

The answer is JSON. With `Accept: application/vnd.apache.arrow.stream` it is an Arrow
IPC stream in the gold schema, and the cursor is in the `X-Next-After` header.

The API authenticates nothing. Keep it on a private network.

## Storage

Every path derives from one root. The root is a local directory, or a URL that fsspec
reads: `s3://`, `gs://`, `az://` or `memory://`. Set it with environment variables:

```bash
export CV_LAKEHOUSE_ROOT=/Volumes/data/cv_lakehouse   # defaults to ./data
export CV_LAKEHOUSE_ROOT=s3://lake/cv
export CV_LAKEHOUSE_STORAGE_OPTIONS='{"key": "...", "secret": "...", "endpoint_url": "http://seaweedfs:8333"}'
```

`CV_LAKEHOUSE_STORAGE_OPTIONS` is JSON, and passes to fsspec as the options of the
root. DuckDB reads S3 with its httpfs extension, from the same options. A manifest
stores every location relative to the root, so the lake moves as one tree.

DuckDB uses every core of the machine, also in a container with a CPU limit. Its memory
limit leaves out the memory of Python and Arrow. In a container, set both:

```bash
export CV_LAKEHOUSE_DUCKDB_THREADS=4
export CV_LAKEHOUSE_DUCKDB_MEMORY_LIMIT=2GB
```

Every layer runs on any root. On a local root, the embeddings send Triton image paths, and
the container mounts the lake. On an object store, they send presigned URLs, valid for an
hour, so Triton needs no mount and no credentials. That needs Triton 0.3.0 or newer.

Bronze stages nothing on local disk. On S3 a download is a multipart upload, and a run
that dies resumes after the last part that S3 holds. On another object store a download
that dies starts again. An archive unpacks from the store into the store, member by
member. A linked copy stays where it is, local or remote, and the manifest records its
location: link one with `source_dir`, such as `s3://datasets/coco2017`.

On a local HDD, set a lock file for the disk lease:

```bash
export CV_LAKEHOUSE_DISK_LEASE_PATH=/var/lib/lakehouse/disk.lock
```

Then only one unit of heavy I/O uses the disk at a time: one unpack, one checksum, or
one silver split. A run that waits logs the work that holds the disk. Leave it unset
for S3 or an SSD. Two domains on one disk set the same path. See ADR 0012.

Set `LAKEHOUSE_TEST_S3_ENDPOINT` to run the S3 tests against a real S3, such as
SeaweedFS, instead of moto.

```
$CV_LAKEHOUSE_ROOT/
  bronze/<dataset>/            a manual download, or only the manifest of a linked copy
    _bronze.json                 every published file, with its URL and checksum
  bronze/<dataset>_corrections/  the corrections that curators made in LightlyStudio
    _corrections.json            every snapshot, with its checksum
    snapshots/<id>/corrections.parquet   one immutable snapshot
  silver/<dataset>/
    _silver.json                     the current build
    builds/<id>/images/<split>.parquet   one row per image
    builds/<id>/boxes/<split>.parquet    one row per box, XYWH pixels, exact floats
  silver/<dataset>_embeddings/
    _embeddings.json                 the current build, and the silver build it covers
    builds/<id>/embeddings/<split>.parquet       one vector per image, by image_id
    builds/<id>/crop_embeddings/<split>.parquet  one vector per box crop, by box_id
    parts/                           the vectors of a run that stopped
  gold/
    _gold.json                       the current version, and its datasets
    versions/<n>/images/<dataset>/<split>.parquet   one row per image
    versions/<n>/boxes/<dataset>/<split>.parquet    one row per box
    releases/<number>/               the frozen val and test files of one release
```

Only bronze holds pixels. Silver holds annotations, so it costs almost no disk.

## Curating gold in LightlyStudio

[studio/](studio/README.md) holds a LightlyStudio server on Postgres, and a sync that
loads gold into it through the gold API. Its start page lists every dataset. A curator's
edit survives every later sync.

The loop, each step by hand:

1. A curator relabels, moves, deletes or draws boxes in LightlyStudio.
2. Materialize `bronze/<dataset>_corrections`. It fetches a snapshot of the corrections.
3. Materialize `silver/<dataset>`, and then `gold/current`.
4. The sync sees the new gold version on its next run. The corrected boxes stay as the
   curator left them.
5. When a round of corrections is done, materialize `gold/eval_release`.

## Bronze

Bronze holds each dataset complete and raw. Many use cases read it, so it decides
nothing about the content.

- Every file that the publisher publishes is there: a split without labels, an
  evaluation kit, a dataset card, an annotation that no reader uses yet.
- Every file keeps its bytes and its published name. A file only gets its final name
  after its size and checksum match the pinned ones, so a final name means a verified
  file. A partial download resumes.
- Every archive stays, and is unpacked beside itself into a folder of its name, as a
  desktop unarchiver does. An archive that already wraps its content in that folder is
  not doubled. The unpacked files keep the dates the archive stores. A zip made on a
  Mac carries a `__MACOSX` folder of Finder metadata, which is not unpacked. The kept
  archive still holds it.

The result is what a person gets who downloads every file by hand and unpacks it. So a
copy that already exists links in place of a download:

| Dataset | Source of the files | Layout |
| --- | --- | --- |
| wider_face | the official page, which links to Hugging Face for the images | `WIDER_{train,val,test}/`, `wider_face_split/`, `eval_tools/`, `Submission_example/` |
| pp4av | its Hugging Face repository, at a pinned commit | `README.md`, `pp4av.py`, `data/{images,annotations,soiling_annotations}/` |
| open_images | every link on the V7 download page, and the CVDF image tars | `train_0/` to `train_f/`, `validation/`, `test/`, `challenge2018/`, and the CSVs |

PP4AV publishes two label sets. Its dataset card calls `annotations` the curated one and
`soiling_annotations` the raw one, before filtering. Bronze holds both, and the reader
takes the curated one.

Open Images is about 740 GB: 560 GB of image tars, 130 GB of narrative voice recordings,
and the rest annotations.

### Corrections

A curator fixes annotations in LightlyStudio. The export of [studio/](studio/README.md)
publishes the fixes of a dataset as snapshots, and the asset
`bronze/<dataset>_corrections` fetches them.

```bash
export CV_LAKEHOUSE_STUDIO_EXPORT_URL=http://studio-sync:8002
```

- A snapshot is one Parquet file that never changes. It holds every correction that is
  live when it is made, so the latest snapshot alone says what to change.
- Bronze verifies the size and the checksum of a snapshot, and pins them in
  `_corrections.json`.
- Each snapshot names the one before it. The run fails when the chain breaks, or when
  the export lists fewer snapshots than bronze holds. A LightlyStudio database that was
  lost then cannot silently drop every correction.
- The asset has no upstream asset, so the graph stays acyclic.
- When the variable is unset, the asset fetches nothing and still succeeds.

Materialize the asset by hand in the UI. Nothing schedules it.

## Running it

```bash
make dev   # then materialize bronze, silver and gold in the UI
```

To link a dataset that you already have, set `source_dir` in the run config of its
bronze asset. Nothing downloads. The copy must hold every file and every unpacked
archive that the source publishes, or the run fails and names what is missing. The
archives themselves are optional.

```yaml
ops:
  bronze__pp4av:
    config:
      source_dir: /Volumes/data/pp4av
```

## Adding a dataset

1. Write a source in `domains/cv/src/lakehouse_cv/sources/` that inherits `BronzeSource`. It lists every file the
   dataset publishes in `published_files`, with a size and a checksum, reports its
   image root, and returns a `labelformat` reader over the raw labels. It holds no
   download code: bronze downloads, verifies and unpacks the files.
2. Give it a `DatasetSpec` (`contract/dataset_spec.py`) that maps its category names onto canonical class names.
3. Register it in `sources/source_registry.py`. Its bronze and silver assets then
   appear on their own.

If the dataset publishes something per box that labelformat cannot carry, also inherit
`BoxAttributeSource`, implement `read_raw_images`, and add the `attr_*` columns to
`contract/silver_tables.ATTRIBUTE_COLUMNS`. A source without it is adapted from its labelformat
reader and simply writes no attributes.

## Layout

```
pyproject.toml       the uv workspace and the dg workspace, and the development tools
uv.lock              one lock for every member
packages/lakehouse_core/src/lakehouse_core/
  published_files.py   download, verify and unpack the files a publisher distributes
  published_source.py  the protocol between a source of a domain and bronze
  bronze_asset.py      the bronze asset of any source: download or link, then verify
  bronze_manifest.py   what bronze writes beside a dataset
  lake_store.py        the root as a UPath, and DuckDB and pyarrow on it
  lake_files.py        move, copy and remove, also on an object store
  parquet_files.py     read and write Parquet at any path of the lake
  lake_paths.py        the bronze, silver and gold directories under a root
  lake_settings.py     the settings every domain shares
  lake_resource.py     the Dagster resource with the lake root
  manifest_files.py    write and read a manifest
  table_spec.py        the name, layer, schema and key of a table
  fingerprints.py      the digest behind the code versions
domains/cv/src/lakehouse_cv/
  contract/          what each layer holds: schemas, tables, manifests, classes, ids
  sources/           one module per dataset, the registry, and the corrections export
  transforms/        normalise, apply corrections, embed, write silver, build gold
  gold_api/          the HTTP API over gold
  defs/              the Dagster assets and checks
  settings.py        pydantic-settings, prefix CV_LAKEHOUSE_
domains/cv/Dockerfile  the image of the CV code location
docs/adr/            the architecture decision records
tests/               the import rules between the packages
triton/              the MobileCLIP server, vendored, with its own lock
studio/              the LightlyStudio images: the server, and the sync from gold
```

Dagster discovers everything in `defs/` automatically. Every other module is plain
Python, so the tests need no Dagster and no network.

## Tooling

uv for dependencies, ruff for lint and format, pyrefly for types, pytest for tests.
See [AGENTS.md](AGENTS.md) for the conventions.
