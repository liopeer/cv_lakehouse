<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# cv_lakehouse

[![build](https://github.com/liopeer/cv_lakehouse/actions/workflows/build.yml/badge.svg)](https://github.com/liopeer/cv_lakehouse/actions/workflows/build.yml)
[![release](https://github.com/liopeer/cv_lakehouse/actions/workflows/release.yml/badge.svg)](https://github.com/liopeer/cv_lakehouse/actions/workflows/release.yml)

A medallion lakehouse for computer vision datasets, built on Dagster.

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

The app and the Triton server release independently. For each one, release-please keeps a
release PR open. Merging that PR tags the release and pushes the image. Any other merge
publishes nothing.

| Component | Tag | Image |
| --- | --- | --- |
| app | `vX.Y.Z` | `ghcr.io/liopeer/cv_lakehouse` |
| triton | `triton-vX.Y.Z` | `ghcr.io/liopeer/cv_lakehouse-triton` |

Each image is tagged `X.Y.Z`, `X.Y` and `X`. Pin `X.Y.Z`, because the other two move. The
app image is a Dagster code location on port 4000. The same image serves the
[gold API](#the-gold-api) on port 8000 when it starts with the uvicorn command.

No image holds model weights. The Triton image downloads the checkpoint and builds its
engines on the first start. See [triton/README.md](triton/README.md).

## The medallion model

| Layer | What it is | How it materializes |
| --- | --- | --- |
| bronze | the raw dataset, complete, as published | download it, or link a complete copy |
| silver | one dataset, normalised | four Parquet files per split, on the class registry |
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
duckdb -c "select class_name, count(*), avg(w * h)
           from read_parquet('$CV_LAKEHOUSE_ROOT/silver/*/boxes/*.parquet')
           group by 1"
```

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

Silver also embeds, in the same run that normalises. Every image gets one MobileCLIP-S0
vector, and so does every box, cropped to its own pixels. The vectors are 512 float32 and
L2 normalised, so a cosine similarity is a dot product.

They are separate files because a vector is 2 KB next to a box row of a few dozen bytes,
and most queries over silver want the boxes. The keys are the keys of the other two
files, so a join needs nothing else:

```bash
duckdb -c "select b.class_name, count(*)
           from read_parquet('$CV_LAKEHOUSE_ROOT/silver/*/boxes/*.parquet') b
           join read_parquet('$CV_LAKEHOUSE_ROOT/silver/*/crop_embeddings/*.parquet') e
             using (box_id)
           where list_dot_product(e.embedding, (
                 select embedding
                 from read_parquet('$CV_LAKEHOUSE_ROOT/silver/*/crop_embeddings/*.parquet')
                 limit 1)) > 0.9
           group by 1"
```

A [Triton](triton/README.md) server does the work. It is in `triton/`, it runs the two
encoders as TensorRT engines, and it crops, resizes and normalises on the GPU, so this
repository sends a path and a box and never opens an image. The deployment stack starts
it, and `triton/` builds the image.

```bash
export CV_LAKEHOUSE_TRITON_URL=localhost:8011
```

Silver reads the URL and skips the embedding when it is unset, so a machine with no
server still builds the layer. The two counts land in the asset metadata either way.

The server resolves the path itself, so the stack mounts `$CV_LAKEHOUSE_ROOT` read only
at the same path inside the container. If a bronze directory is a symlink to a path
outside the root, add a mount for that path to the compose file of the stack.

Nothing is cached. A silver rematerialisation embeds the dataset again, which is the
price of leaving the versioning out. The model name is part of the silver `code_version`,
so a new model marks every silver asset stale.

### The class registry

`CanonicalClass` in `src/cv_lakehouse/class_registry.py` holds the vocabulary that
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
- `image_path` is relative to the lake root, so gold names a pixel on any machine.
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

### The gold API

A read only HTTP API serves gold, so a consumer needs no access to the lake files.

```bash
make api
curl 'localhost:8000/v1/boxes?dataset=wider_face&role=val&class_name=face'
```

| Endpoint | Rows |
| --- | --- |
| `/v1/meta` | the gold manifest: the version, the datasets, the splits and their roles |
| `/v1/classes` | the class registry, also before gold exists |
| `/v1/images` | one per image |
| `/v1/boxes` | one per box |
| `/v1/embeddings` | `image_id` and the MobileCLIP vector of the image |
| `/v1/crop_embeddings` | `box_id` and the MobileCLIP vector of the box crop |

| Parameter | Selects |
| --- | --- |
| `dataset`, `split`, `role` | rows of these datasets, splits or roles |
| `class_name` | boxes of these classes. Not on `/v1/images` and `/v1/embeddings` |
| `commercial_use` | rows of the datasets with this licence answer |
| `changed_since` | rows that a gold build changed after this time |
| `limit`, `after` | one page. `limit` is 1000 by default and at most 100000 |

A parameter that is given more than once matches any of its values. The rows come in
the order of their id. A full page carries `next_after`, and the next request sends it
as `after`.

The answer is JSON. With `Accept: application/vnd.apache.arrow.stream` it is an Arrow
IPC stream in the gold schema, and the cursor is in the `X-Next-After` header.

The API authenticates nothing. Keep it on a private network.

## Storage

Every path derives from one root. Set it with an environment variable:

```bash
export CV_LAKEHOUSE_ROOT=/Volumes/data/cv_lakehouse   # defaults to ./data
```

```
$CV_LAKEHOUSE_ROOT/
  bronze/<dataset>/            a manual download, or a symlink to a complete copy
    _bronze.json                 every published file, with its URL and checksum
  silver/<dataset>/
    images/<split>.parquet           one row per image
    boxes/<split>.parquet            one row per box, XYWH pixels, exact floats
    embeddings/<split>.parquet       one MobileCLIP vector per image
    crop_embeddings/<split>.parquet  one MobileCLIP vector per box crop
  gold/
    _gold.json                       the current version, and its datasets
    versions/<n>/images/<dataset>/<split>.parquet   one row per image
    versions/<n>/boxes/<dataset>/<split>.parquet    one row per box
  .studio/<name>.db            a LightlyStudio cache, safe to delete
```

Only bronze holds pixels. Silver holds annotations, so it costs almost no disk.

`.studio` is not a layer. See below.

## Browsing silver in LightlyStudio

```bash
make studio DATASETS="wider_face pp4av"
```

That builds a LightlyStudio database from the Parquet and opens the GUI. Around 100k
images and 1M boxes load in about seven seconds, because every row goes in through
`INSERT ... SELECT ... FROM read_parquet(...)` that DuckDB runs server side. Nothing
iterates a row in Python.

The database is a cache, never an artifact of the lake. DuckDB gives it no migration path,
it holds an exclusive write lock, and its storage format is tied to a DuckDB range, so
Parquet stays the thing we keep and the database gets rebuilt. Delete
`$CV_LAKEHOUSE_ROOT/.studio` whenever.

Every id LightlyStudio needs is derived during the load rather than stored in silver: an
md5 of `(dataset, split, file_name)` and of the box index on it. Keys are unique across
datasets by construction, so loading several datasets into one database is a
concatenation, and a reload is idempotent. `file_path_abs` is derived too, since an
absolute path in silver would not survive a move to another machine. The same load will
serve a training mix later, which is the other reason silver keeps no database of its
own.

Silver's embeddings load the same way, onto the sample of the image and the sample of
the box. LightlyStudio therefore embeds nothing itself: the GUI's plot and its similarity
search read what silver already wrote. A silver built with no Triton server loads too,
without the plot.

`tools/studio.py` is a script and not part of the package: browsing is neither a layer
nor a step in producing one. `uv run` reads its dependencies from its own header, and it
depends on the checkout beside it, so the Parquet paths, the manifest and the class
registry it reads are the ones silver wrote. `cv_lakehouse` therefore depends on no GUI.

LightlyStudio pulls torch, which is why the script routes torch and torchvision to the
PyTorch CPU channel: nothing here trains or embeds, the Triton server in `triton/` holds
the weights. On Linux the CPU channel removes 2.8 GB of wheels, the CUDA stack included,
and the torch stack lands at 198 MB. The three are also in this project's `dev` group,
for the test that loads silver through the script.

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

1. Write a source in `sources/` that inherits `BronzeSource`. It lists every file the
   dataset publishes in `published_files`, with a size and a checksum, reports its
   image root, and returns a `labelformat` reader over the raw labels. It holds no
   download code: bronze downloads, verifies and unpacks the files.
2. Give it a `DatasetSpec` that maps its category names onto canonical class names.
3. Register it in `sources/source_registry.py`. Its bronze and silver assets then
   appear on their own.

If the dataset publishes something per box that labelformat cannot carry, also inherit
`BoxAttributeSource`, implement `read_raw_images`, and add the `attr_*` columns to
`silver_schema.ATTRIBUTE_COLUMNS`. A source without it is adapted from its labelformat
reader and simply writes no attributes.

## Layout

```
src/cv_lakehouse/
  class_registry.py  the canonical classes every layer shares
  settings.py        pydantic-settings, prefix CV_LAKEHOUSE_
  normalization.py   normalise one dataset onto the canonical classes
  silver_schema.py   the Parquet schema silver writes, and the streaming writer
  box_identity.py    the ids of an image and of a box
  split_roles.py     the roles a split can have in gold
  gold_schema.py     the Parquet schema gold writes
  gold_build.py      build one gold version from the silver datasets on disk
  gold_api/          the HTTP API over gold
  embeddings.py      the MobileCLIP client, and the embedding Parquet it writes
  manifests.py       what each layer writes beside its output
  sources/           one module per dataset, the source registry, and the
                     downloader that fetches, verifies and unpacks bronze
  defs/              the Dagster assets and checks
tests/
tools/
  studio.py          load silver into LightlyStudio. A uv script, not in the package
triton/              the MobileCLIP server, vendored
```

Dagster discovers everything in `defs/` automatically. Every other module is plain
Python, so the tests need no Dagster and no network.

## Tooling

uv for dependencies, ruff for lint and format, pyrefly for types, pytest for tests.
See [AGENTS.md](AGENTS.md) for the conventions.
