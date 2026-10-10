<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# studio

[LightlyStudio](https://github.com/lightly-ai/lightly-studio) (Apache-2.0) as a shared
service on Postgres, kept in step with the gold layer of the lakehouse.

Two images come from one Dockerfile:

| Image | What it runs |
| --- | --- |
| `ghcr.io/liopeer/cv_lakehouse-studio` | the LightlyStudio server and GUI, on port 8001 |
| `ghcr.io/liopeer/cv_lakehouse-studio-sync` | the sync from gold, and the export of the curator edits, on port 8002 |

| File | What it is |
| --- | --- |
| `Dockerfile` | Upstream at `LIGHTLY_STUDIO_REF`, `patches/` applied, GUI built, CPU torch. |
| `patches/` | Our changes to upstream, applied with `git apply` in name order. |
| `src/cv_lakehouse_studio/` | The server entrypoint, the sync and the export. |

## The patch

`0001-dataset-overview-page.patch` changes the start page. Upstream redirects `/` to the
newest dataset, and fails when there is none. The patched page lists every dataset with
its sample count, from the `GET /api/collections/overview` route that upstream already
has. The logo in the header leads back to the list.

## The sync

The sync reads the [gold API](../README.md#the-gold-api) and writes Postgres. It shares
no code with `cv_lakehouse`.

- A lakehouse dataset of at most 500k images is one LightlyStudio dataset, with the
  same name. A larger one is split. See [Large datasets](#large-datasets).
- An image carries the tags `role/<role>` and `split/<split>`.
- The boxes are in the annotation collection `lakehouse`.
- The id of an annotation is its `box_id` in gold, and the id of an image its `image_id`.
- The MobileCLIP vectors of silver load too, so LightlyStudio embeds nothing.
- The val and test rows are the draft of gold, not a frozen release, so a curator can
  fix them. The fix reaches the next release.

Every run writes every image and every box of gold, and keeps no record of what it
wrote. It updates a row in place by its id, so a run of the same gold changes nothing.

The lake owns the annotations. A box that gold does not hold is deleted. The lake also
owns the tags `role/` and `split/`. A tag that a curator adds or removes stays in
LightlyStudio, and the lake never reads it.

A curator's edit wins until gold holds it:

- Triggers log every write of a curator to a box, in `lakehouse_sync.edit_log`. The
  sync marks its own writes with the setting `lakehouse.origin`, and the triggers skip
  them.
- `/v1/meta` names the last log entry that gold holds, for each dataset. The sync leaves
  a box with a later entry: it neither updates nor deletes it, and gold does not bring
  back a box that a curator deleted.
- A trigger checks the same rule on every write of the sync, so an edit that lands
  while the sync runs also stays.
- Gold of another database holds no edit of this one.

A box that a curator moves loses its vector, as the vector shows the old crop. The sync
reads the vectors of a dataset only when a sample lacks one, and then writes every one.

The LightlyStudio server owns the schema, and migrates it on start. The sync writes only
while the database is at the migration that its own LightlyStudio build expects. Deploy
both images from one release.

## Large datasets

LightlyStudio holds about 1M images per dataset. A lakehouse dataset over
`CV_LAKEHOUSE_STUDIO_MAX_IMAGES_PER_DATASET` is one LightlyStudio dataset per split,
such as `open_images.validation`. A split over the cap is one per shard of the gold API,
such as `open_images.train.2-of-4`.

The sync counts the images when it first sees a dataset, and stores the plan in
`lakehouse_sync.studio_dataset`. It never plans that dataset again, so an image never
moves to another LightlyStudio dataset. The corrections of every part come under the
lakehouse dataset.

## The export

The sync image also serves the edits of the curators, for the lakehouse asset
`bronze/<dataset>_corrections`. See
[ADR 0014](../docs/adr/0014-curator-edits-are-bronze-events.md).

| Endpoint | Answer |
| --- | --- |
| `GET /v1/datasets/{dataset}/events` | the chain of this database, and every event file with its size and SHA-256 |
| `GET /v1/datasets/{dataset}/events/{id}/events.parquet` | one event file |

- An event file holds the boxes of the log entries after the last file: each box as
  LightlyStudio holds it now, or a tombstone. A box edited twice is one row.
- A listing first publishes the new log entries, and stores a file when there are any.
  A stored file never changes.
- Each file names the one before it, and the range of the log that it covers. The
  listing locks the log, so a file never skips an entry that commits late.
- The boxes are whole pixels, as LightlyStudio stores them. The lake compares an event
  with the source box rounded to whole pixels, so a box that was only relabelled keeps
  its exact coordinates.
- The first file of a dataset also holds the boxes of its latest snapshot of ADR 0009,
  as LightlyStudio holds them now.

The files are in Postgres, in `lakehouse_sync.event_file`. A new database starts a new
chain under a new id, and the lake keeps the files of the old one. Back up Postgres all
the same: an edit that the lake did not fetch yet is lost with the database.

## Settings

| Variable | Read by | Meaning |
| --- | --- | --- |
| `LIGHTLY_STUDIO_DATABASE_URL` | both | `postgresql://user:password@host:5432/database` |
| `CV_LAKEHOUSE_STUDIO_GOLD_API_URL` | sync | the gold API, such as `http://lakehouse:8000` |
| `CV_LAKEHOUSE_STUDIO_IMAGE_BASE` | sync | where the lake root is for the server |
| `CV_LAKEHOUSE_STUDIO_SYNC_INTERVAL_SECONDS` | sync | the pause between two runs. 3600 by default |
| `CV_LAKEHOUSE_STUDIO_MAX_IMAGES_PER_DATASET` | sync | the cap of a LightlyStudio dataset. 500000 by default |
| `CV_LAKEHOUSE_STUDIO_EXPORT_PORT` | sync | the port of the export. 8002 by default |
| `CV_LAKEHOUSE_STUDIO_ROWS_PER_PAGE` | sync | the rows of one request for images and boxes. 50000 by default |
| `CV_LAKEHOUSE_STUDIO_EMBEDDING_ROWS_PER_PAGE` | sync | the rows of one request for embeddings. 5000 by default |
| `CV_LAKEHOUSE_STUDIO_STATEMENT_TIMEOUT_SECONDS` | sync | the longest that one SQL statement of a sync runs. 1800 by default |

Gold names a pixel relative to the lake root. The sync puts `IMAGE_BASE` in front, and
LightlyStudio reads the result through fsspec. A pixel of a copy that bronze links from
outside the lake has an absolute path or a URL in gold, and the sync keeps it.

- A directory: mount the lake read only at that path in the server container. Also mount
  every linked copy outside of it.
- `s3://bucket/prefix`, `gs://...` or `az://...`: give the server read access through the
  environment, such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
  `AWS_ENDPOINT_URL`. With a custom endpoint, such as SeaweedFS, s3fs addresses the
  bucket in the path, so no bucket subdomain is needed. The image installs the fsspec
  backends of S3, GCS and Azure.

A sync run with a new `IMAGE_BASE` moves every path.

## Deploying

- Postgres needs pgvector. Upstream tests against `pgvector/pgvector:0.8.1-pg18-bookworm`.
- The server creates the database and runs `CREATE EXTENSION vector`, so its user needs
  the rights for that.
- Run one server replica with one worker. The app holds one database session per process.
- Nothing authenticates. Every route is open, dataset deletion included. Keep the server
  and the export on a private network, and terminate TLS in front of port 8001.
- Point `CV_LAKEHOUSE_STUDIO_EXPORT_URL` of the lakehouse at port 8002 of the sync.
- The GUI uses relative URLs, so serve it at the root of its host name.
- Analytics are off.

## Development

```bash
make test    # needs Docker, for a Postgres with pgvector
make image   # build both images
```

## A new upstream version

1. Clone the new tag, run `git apply patches/*.patch`, and fix what does not apply.
2. Regenerate the patch with `git add -N . && git diff`.
3. Set `LIGHTLY_STUDIO_REF` in the Dockerfile and the `lightly-studio` pin in
   `pyproject.toml` to the new version. Run `make lock` at the root of the repository.
4. Run `make test`. The tests write the tables of the new version.
