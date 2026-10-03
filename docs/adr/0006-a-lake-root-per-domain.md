<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0006: A lake root per domain

- Status: accepted
- Date: 2026-10-03

## Context

The CV lake has bronze, silver and gold under `CV_LAKEHOUSE_ROOT`. Gold and every eval
release store image paths relative to that root. A second domain needs a place for its
layers.

## Decision

Each domain has its own lake root, which its own environment variable sets. The CV
domain keeps `CV_LAKEHOUSE_ROOT` and every directory under it. `LakePaths` in core
resolves the layer directories under a root. A domain subclasses it for directories
that only it has, such as the gold versions and releases of CV.

## Consequences

- Nothing on disk moves, and every path in gold and in the releases stays valid.
- Two domains can share one disk. Each points its root at its own directory.
- A query across two domains opens two roots.
