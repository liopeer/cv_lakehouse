<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0014: Curator edits are bronze events

- Status: proposed
- Date: 2026-10-10

## Context

ADR 0009 publishes the corrections as cumulative snapshots. The export finds a
correction as the difference between LightlyStudio and `loaded_box`, the record of what
the sync last wrote.

- `loaded_box` is mutable state of the sync, and every correction depends on it.
- A sync that overwrites LightlyStudio from gold makes that difference empty. The next
  snapshot then holds no correction, and silver drops every correction without notice.
- So the sync runs a three-way merge against `loaded_box`. That merge is the slowest
  part of a sync.
- LightlyStudio records no change. `sample.updated_at` keeps the time of the insert,
  and a delete leaves no row.

A source box changes only with a new release of the source. Bronze keeps each release
as published.

## Decision

A curator edit is an event in bronze. Silver folds every event over the untouched
source.

1. Triggers in the LightlyStudio database log every write to `annotation_base` and
   `object_detection_annotation`. A log row holds a sequence number and the
   `sample_id`.
2. The sync sets `lakehouse.origin` to `sync` in its transactions. The triggers skip
   those writes.
3. The export reads the log after its last cursor. For each logged box, it writes the
   box as LightlyStudio holds it now, or a tombstone if the box is gone.
4. An event file never changes. It names the sequence range that it covers and the file
   before it. Bronze verifies and pins the files, as ADR 0009 does for a snapshot.
5. Silver keeps the latest event of each box, in file order and then sequence order.
   It compares that event with the source box.
   - A tombstone removes the box.
   - A label that differs from the source relabels the box.
   - A geometry that differs from the source, rounded to whole pixels, moves the box.
     Otherwise the box keeps the float coordinates of the source.
   - An event for a box that the source does not hold adds the box.
6. The fold gives the corrections that `CorrectionOverlay` reads today. The overlay
   does not change.
7. The gold manifest records the last event that gold holds. The sync overwrites
   LightlyStudio from gold, but it skips a box with a later event.
8. The edit of a curator wins over the source.

## Consequences

- `loaded_box` goes away, and so does the three-way merge of the sync.
- A lost LightlyStudio database loses no correction, because bronze holds every event.
  A new database starts a new chain of event files, and silver folds both chains.
- A curator who undoes an edit produces an event that equals the source. The fold then
  changes nothing.
- An edit that gold does not hold yet stays in LightlyStudio until gold catches up.
- Silver reads every event file, not only the latest snapshot. The files hold one row per
  edited box, so the cost grows with the edits and not with the dataset.
- A new release of a source is a separate decision. The box ids of that release decide
  whether the events of the old release still apply.
- The migration turns the latest snapshot of each dataset into a first event file.
- This record supersedes ADR 0009 when it is accepted.
