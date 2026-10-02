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
| `ghcr.io/liopeer/cv_lakehouse-studio-sync` | the sync that loads gold into LightlyStudio |

| File | What it is |
| --- | --- |
| `Dockerfile` | Upstream at `LIGHTLY_STUDIO_REF`, `patches/` applied, GUI built, CPU torch. |
| `patches/` | Our changes to upstream, applied with `git apply` in name order. |
| `src/cv_lakehouse_studio/` | The server entrypoint and the sync. |

## The patch

`0001-dataset-overview-page.patch` changes the start page. Upstream redirects `/` to the
newest dataset, and fails when there is none. The patched page lists every dataset with
its sample count, from the `GET /api/collections/overview` route that upstream already
has. The logo in the header leads back to the list.

## The sync

The sync reads the [gold API](../README.md#the-gold-api) and writes Postgres. It shares
no code with `cv_lakehouse`.

- One LightlyStudio dataset per lakehouse dataset, with the same name.
- An image carries the tags `role/<role>` and `split/<split>`.
- The boxes are in the annotation collection `lakehouse`.
- The id of an annotation is its `box_id` in gold, and the id of an image its `image_id`.
- The MobileCLIP vectors of silver load too, so LightlyStudio embeds nothing.

A run does nothing for a dataset whose gold version it already loaded. For a new gold
version it reads every image and box, and only the vectors that changed.

The sync records every box as it last wrote it, in the table `lakehouse_sync.loaded_box`.
That record decides who owns a box:

| LightlyStudio holds | Gold holds | The sync |
| --- | --- | --- |
| what the sync wrote | another value | updates the box |
| what the sync wrote | no such box | deletes the box |
| another value, or no box | anything | leaves it. A curator changed it |
| no record | a new box | inserts the box |

So a curator's edit is never overwritten, and a box that a curator deleted never comes
back.

The LightlyStudio server owns the schema, and migrates it on start. The sync writes only
while the database is at the migration that its own LightlyStudio build expects. Deploy
both images from one release.

## Settings

| Variable | Read by | Meaning |
| --- | --- | --- |
| `LIGHTLY_STUDIO_DATABASE_URL` | both | `postgresql://user:password@host:5432/database` |
| `CV_LAKEHOUSE_STUDIO_GOLD_API_URL` | sync | the gold API, such as `http://lakehouse:8000` |
| `CV_LAKEHOUSE_STUDIO_IMAGE_BASE` | sync | where the lake root is for the server |
| `CV_LAKEHOUSE_STUDIO_SYNC_INTERVAL_SECONDS` | sync | the pause between two runs. 3600 by default |

Gold names a pixel relative to the lake root. The sync puts `IMAGE_BASE` in front, and
LightlyStudio reads the result through fsspec:

- A directory: mount the lake read only at that path in the server container. Also mount
  the target of a bronze directory that is a symlink.
- `s3://bucket/prefix`, `gs://...` or `az://...`: give the server the credentials, such
  as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_ENDPOINT_URL`.

A sync run with a new `IMAGE_BASE` moves every path.

## Deploying

- Postgres needs pgvector. Upstream tests against `pgvector/pgvector:0.8.1-pg18-bookworm`.
- The server creates the database and runs `CREATE EXTENSION vector`, so its user needs
  the rights for that.
- Run one server replica with one worker. The app holds one database session per process.
- Nothing authenticates. Every route is open, dataset deletion included. Keep the server
  on a private network, and terminate TLS in front of port 8001.
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
   `pyproject.toml` to the new version. Run `make lock`.
4. Run `make test`. The tests write the tables of the new version.
