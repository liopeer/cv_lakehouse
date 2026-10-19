<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0017: An eval release with the files of the last one is skipped

- Status: accepted
- Date: 2026-10-19

## Context

ADR 0010 failed a run of the eval release asset that would copy the files of the last
release. A materialization of every asset then failed until a correction reached the
val or the test rows. The failure hid the real failures of the same run.

## Decision

The eval release asset copies the val and test files of the current gold version into
`gold/releases/NNNN`, under the next number.

- The files are read only, and the release manifest pins each one with a size and a
  sha256.
- A release takes its final name only when it is complete.
- A run that would copy the files of the last release writes nothing. The step
  succeeds with no materialization, and the log names the last release.
- A run never changes an earlier release.
- A release holds no vector. The vectors stay in silver.

## Consequences

- A benchmark names the release it ran on.
- The curators keep working on gold, and their fixes reach the next release.
- The gold API serves a release by its number.
- A failed run of the eval release asset reports a real error.
- This record supersedes ADR 0010.
