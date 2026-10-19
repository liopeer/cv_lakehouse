<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0018: Gold holds only the target classes

- Status: accepted
- Date: 2026-10-19

## Context

Every source maps a category that its map does not name to the class `other`. Gold
held those boxes, and the images that held only them. On Open Images, most boxes are
`other`. The curators, the gold API and the eval releases do not use them.

## Decision

Gold is the union of the silver datasets, with the role columns. It holds a box only
when the box has a target class and no flag. It holds an image only when the image
holds such a box.

- `TARGET_CLASS_NAMES` in `contract/gold_tables.py` names the target classes.
- Silver keeps every class, with the class that the source gave it.
- This record supersedes rule 1 of ADR 0015. The other rules of ADR 0015 stay.

## Consequences

- A change to the target classes costs a gold rebuild, not a silver rebuild.
- A curator sees no `other` box in LightlyStudio. A box that a curator relabels to
  `other` leaves gold.
- A curator adds no box to an image with no target box, as that image is not in gold.
- An image that leaves gold stays in LightlyStudio until the sync removes it.
- The next eval release holds new files, as the val and test rows change.
