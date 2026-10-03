<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0010: Eval splits are frozen as numbered releases

- Status: accepted
- Date: 2026-10-03

## Context

Gold changes with every correction. A benchmark must not change, or two results on it
do not compare.

## Decision

The eval release asset copies the val and test files of the current gold version into
`gold/releases/NNNN`, under the next number.

- The files are read only, and the release manifest pins each one with a size and a
  sha256.
- A release takes its final name only when it is complete.
- A run that would copy the same files as the last release fails.
- A run never changes an earlier release.
- A release holds no vector. The vectors stay in silver.

## Consequences

- A benchmark names the release it ran on.
- The curators keep working on gold, and their fixes reach the next release.
- The gold API serves a release by its number.
