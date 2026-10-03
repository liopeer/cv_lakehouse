<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0009: Corrections are bronze snapshots

- Status: accepted
- Date: 2026-10-03

## Context

Curators fix annotations in LightlyStudio. Silver applies the fixes. A fix that lives
only in the LightlyStudio database is lost with that database, and silver cannot be
rebuilt from bronze.

## Decision

The export of `studio/` publishes the corrections of a dataset as snapshots. A snapshot
is one Parquet file that never changes. It holds every correction that is live when it
is made, and it names the snapshot before it.

The corrections asset copies the snapshots into bronze. This is the one exception to a
static file list, because the snapshots are not known before the run. The bronze rules
still apply:

- The size and the checksum of each file come from the listing, and the download
  verifies them.
- The manifest pins them. A later listing that disagrees with a pin fails the run.
- Bronze never changes or deletes a snapshot that it holds.
- A snapshot that does not continue the chain fails the run.

## Consequences

- Silver applies the latest snapshot, and a rebuild from bronze keeps every correction.
- A lost LightlyStudio database starts a new chain. Bronze rejects it, so silver never
  drops the corrections made so far without notice.
