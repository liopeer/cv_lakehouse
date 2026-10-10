<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# Architecture decision records

An ADR records one decision about the architecture, and the reason for it.

- Write an ADR when a decision changes how packages, layers or deployments fit together.
- Number the records in order. Do not reuse a number.
- Do not edit a record after it is accepted. To change a decision, write a new record
  that supersedes it, and set the status of the old record to `superseded by NNNN`.
- Follow the prose rules of `AGENTS.md`.

0008 to 0010 record rules that existed before the first ADR.

## Template

```markdown
# ADR NNNN: Title

- Status: proposed | accepted | superseded by NNNN
- Date: YYYY-MM-DD

## Context

## Decision

## Consequences
```

## Records

| Number | Title |
| --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions |
| [0002](0002-one-uv-workspace.md) | One uv workspace, and triton outside it |
| [0003](0003-a-core-package-and-one-package-per-domain.md) | A core package, and one package per domain |
| [0004](0004-the-contract-of-a-domain.md) | The contract of a domain lives in `contract/` |
| [0005](0005-one-code-location-per-domain.md) | One Dagster code location per domain |
| [0006](0006-a-lake-root-per-domain.md) | A lake root per domain |
| [0007](0007-browse-in-studio.md) | Browse the lake in studio, not in a local script |
| [0008](0008-bronze-keeps-the-published-bytes.md) | Bronze keeps the bytes as the publisher distributes them |
| [0009](0009-corrections-are-bronze-snapshots.md) | Corrections are bronze snapshots |
| [0010](0010-eval-splits-are-numbered-releases.md) | Eval splits are frozen as numbered releases |
| [0011](0011-gold-serves-stable-shards.md) | The gold API serves stable shards of the images |
| [0012](0012-heavy-io-on-a-local-disk-takes-a-lease.md) | Heavy I/O on a local disk takes a lease |
| [0013](0013-silver-vectors-are-sorted-and-kept.md) | Silver vectors are sorted by their id, and kept across runs |
| [0014](0014-curator-edits-are-bronze-events.md) | Curator edits are bronze events |
