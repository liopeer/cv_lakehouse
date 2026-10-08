<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0012: Silver vectors are sorted by their id, and kept across runs

- Status: accepted
- Date: 2026-10-08

## Context

The silver embedding files of Open Images train hold 14.6M crop vectors, which is 39 GB.
Their rows were in the order of the boxes file.

- The gold API reads one page of vectors by a range of ids. DuckDB read the whole file
  for every page, and gold-api ran out of memory.
- A rebuild reused a vector only when the whole silver code was unchanged. A change to
  the class rules embedded every image again, which took 14 hours.
- A run that stopped lost every vector that it had made.

## Decision

- An embedding file is sorted by its id, in row groups of 1024 vectors. The image file
  holds `image_id` for that.
- A crop embedding row holds the crop that the server cut: the box on the pixel grid.
- The footer of a file names the model and `EMBEDDING_VERSION`.
- A vector serves a row while its model, its version and its key are the same. The key
  of a crop is the box id and the crop. No other code decides it.
- A run writes new vectors to closed parts beside the file. The next run reuses them.
- The gold API selects a page of ids first, and then reads the vectors of those ids.

## Consequences

- A page of vectors reads a few row groups, whatever the size of the split.
- A change to silver that keeps the pixels and the crops embeds nothing again.
- New weights behind the same model name need a bump of `EMBEDDING_VERSION`.
- A file of the old layout is moved once with `lakehouse_cv.maintenance.sort_silver_embeddings`.
  The command sends no request to the server.
