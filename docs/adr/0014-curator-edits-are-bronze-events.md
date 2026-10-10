<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0014: Curator edits are bronze events

- Status: accepted
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
source. The sync overwrites LightlyStudio from gold on every run.

### The log

1. Triggers in the LightlyStudio database log every write to `annotation_base` and
   `object_detection_annotation`. A log row holds a sequence number, the `sample_id`,
   and the image of the box when the trigger can tell it.
2. The sync sets `lakehouse.origin` to `sync` in its transactions. The log triggers
   skip those writes.
3. A database has a chain id. A new database starts a new chain.

### The event files

4. The export reads the log after the last file of the dataset. For each logged box, it
   writes the box as LightlyStudio holds it now, or a tombstone if the box is gone. A box
   edited twice is one row.
5. The export locks the log while it writes a file. Every entry up to the last one has
   then committed, so no file skips an entry that commits late.
6. An event file never changes. It names its chain, the range of the log that it
   covers, and the file before it. Bronze verifies and pins the files, as ADR 0009 does
   for a snapshot. A stored file of another chain stays in bronze.

### The fold

7. Silver keeps the latest event of each box, in file order and then sequence order.
   It compares that event with the source box.
   - A tombstone removes the box.
   - A label that differs from the source relabels the box.
   - A geometry that differs from the source, rounded to whole pixels, moves the box.
     Otherwise the box keeps the float coordinates of the source.
   - An event for a box that the source does not hold adds the box.
   - An event that equals the source changes nothing.
8. The fold gives the corrections that `CorrectionOverlay` applied before. The overlay
   now holds the events, and folds each one when it meets its box.
9. The edit of a curator wins over the source.

### The sync

10. Silver records its last event: the chain and the log sequence. Gold copies it for
    each dataset, and the gold API serves it.
11. On every run, the sync writes every gold image and box, by `image_id` and `box_id`.
    It updates a row in place, so a rerun changes nothing.
12. The sync leaves a box with a log entry after the one that gold holds. It does not
    update it, delete it, or insert it again. Gold of another chain holds no entry of
    this database.
13. A trigger checks rule 12 on every write of the sync to a box. Each query of a
    trigger takes a new snapshot, so an edit that commits while the sync waits for the
    row also stays.
14. A box in LightlyStudio that gold does not hold, and that rule 12 does not keep, is
    deleted.
15. The lake owns the annotations, and its own tags `role/` and `split/`. A tag that a
    curator changes is not exported.
16. A curator move deletes the vector of the box, as the vector shows the old crop. The
    sync reads the vectors only when a sample lacks one.

## Consequences

- `loaded_box`, `synced_dataset`, the three-way merge and `changed_since` go away.
- A lost LightlyStudio database loses no correction that bronze fetched. A new database
  starts a new chain, and silver folds both chains.
- A curator who undoes an edit produces an event that equals the source. The fold then
  changes nothing.
- An edit that gold does not hold yet stays in LightlyStudio until gold catches up.
- Every change to a gold row reaches LightlyStudio: a curator event, a new silver logic,
  a change to the class registry, and a new source release.
- Silver reads every event file. The files hold one row per edited box, so the cost
  grows with the edits and not with the dataset.
- A sync writes every box on every run. Time one run on Open Images.
- A new release of a source is a separate decision. The box ids of that release decide
  whether the events of the old release still apply.
- The first event file of a dataset also holds the boxes of its latest snapshot, as
  LightlyStudio holds them then. No correction of ADR 0009 is lost.
- A deleted LightlyStudio dataset logs a tombstone for each of its boxes.
- This record supersedes ADR 0009.
