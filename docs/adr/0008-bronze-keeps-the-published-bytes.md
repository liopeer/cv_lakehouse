<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0008: Bronze keeps the bytes as the publisher distributes them

- Status: accepted
- Date: 2026-10-03

## Context

Many use cases read one dataset. A decision about the content in bronze serves one use
case and hides data from the others. A file that changes without notice changes every
layer above it.

## Decision

Bronze holds a dataset as its publisher distributes it.

- Bronze downloads every published file, also a split without labels or an evaluation
  kit.
- Bronze does not rewrite, filter, re-encode or rename a file.
- Each file is at its published name, and each archive is unpacked beside itself. The
  archive stays.
- Each file is pinned to a size and a checksum, and the download verifies them.
- A linked copy holds every published part, or the link fails.
- A source lists its files in `published_files`, and holds no download code.

Every decision about the content is in silver.

## Consequences

- A new use case rebuilds silver, and never downloads again.
- The bronze asset of core enforces the rules for every domain. A domain adds only the
  fields of its manifest.
- A publisher that changes a file fails the checksum, and the run stops.
