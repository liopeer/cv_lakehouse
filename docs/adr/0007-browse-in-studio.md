<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0007: Browse the lake in studio, not in a local script

- Status: accepted
- Date: 2026-10-03

## Context

`tools/studio.py` loaded silver into a local LightlyStudio DuckDB file under
`.studio/`. It needed LightlyStudio and torch in the development dependencies of the
app. `studio/` now serves LightlyStudio on Postgres, kept in step with gold and with the
corrections of the curators.

## Decision

The local script, `make studio`, its test and the `.studio` cache are removed. A curator
and a developer browse the lake in `studio/`.

## Consequences

- The workspace root needs no LightlyStudio and no torch.
- Silver of one dataset is no longer viewable without studio. A developer starts studio
  with Docker, or queries the Parquet files with DuckDB.
