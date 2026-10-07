<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0011: The gold API serves stable shards of the images

- Status: accepted
- Date: 2026-10-07

## Context

LightlyStudio holds about 1M images per dataset. Open Images has about 1.9M. A consumer
needs a part of a dataset that stays the same part over time. The lake needs no
partitions at this scale.

## Decision

The gold API takes `shard` and `num_shards`, and serves `/v1/images/count`.

- The shard of an image is the first 32 bits of its `image_id`, modulo `num_shards`.
- A box, an embedding and a crop embedding come in the shard of their image.
- Shard k of N of an image never changes. A change of the rule is a breaking change.
- The count takes the same selection as `/v1/images`, without `limit` and `after`.

A page by position is not used. An image that enters or leaves gold moves every later
image. A range of ids is not used. It puts the format of the id into the contract.

## Consequences

- A consumer picks `num_shards` from the count, once, and keeps it.
- A shard of N is not a part of a shard of 2N. A consumer that splits again doubles N.
- DuckDB's `hash()` is not stable across versions, so the rule does not use it.
