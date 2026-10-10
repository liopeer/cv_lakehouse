<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0015: Every asset is a pure function of its inputs

- Status: accepted
- Date: 2026-10-10

## Context

Functional data engineering asks that a rebuild from bronze gives the same lake. Bronze,
the correction files and the eval releases follow that rule. Silver and gold did not:

- Gold joined each row onto the previous gold version for `changed_at`, and counted
  its versions. A rebuild from bronze set every `changed_at` to the time of the rebuild.
- Silver wrote its files in place. A run that failed left a mix of new and old splits,
  and the blocking check ran after the overwrite.
- Silver deleted the embeddings when no Triton server was set. One code version and one
  input gave two outputs.
- Gold left out a dataset with no silver on disk. It held what happened to be
  materialized, not its declared inputs.

## Decision

An asset is a pure function of its declared inputs and its code version. It reads no
earlier output of its own, no clock and no environment.

1. Gold is the union of the silver datasets, with the role columns, without the flagged
   boxes. `changed_at` and `changed_since` go away. The manifest keeps `built_at` as a
   fact about the run.
2. Gold has no versions. An eval release is the only frozen copy of gold. It names the
   gold code version and the silver build of each dataset.
3. Silver, the embeddings and gold write each run to `builds/<random id>/`, check it,
   and then replace the manifest that names the build. A run that fails leaves the last
   good build. A layer keeps the current build and the one before it, for a reader that
   read the manifest just before the swap.
4. The vectors are the asset `silver/<dataset>_embeddings`. It reads the current silver
   build, and fails when the Triton server is unset or unreachable. It never deletes a
   vector. The server URL stays a resource: it says where the server is, not what the
   asset does.
5. The gold API serves a crop vector only for the crop of the gold box, so a box that
   moved after its vector was made gets none.
6. Gold joins every registered dataset, and fails on a dataset with no silver. A lake
   that holds only some datasets names them in the config of the gold asset.

## Consequences

- A rebuild from bronze gives the same silver and gold rows.
- Silver means the same on any machine. A machine with no server builds silver and
  gold, and fails the embeddings.
- A change to the class rules reruns silver. The embeddings then reuse every vector
  whose pixels and crop are the same, as ADR 0013 decides.
- The sync of LightlyStudio can no longer pull only the changed rows. ADR 0014 makes
  it a full overwrite.
- A reader of the files reads the manifest first, to find the build.
- The first run of the embeddings reuses the vectors that silver kept before. The tool
  that moved a file of the layout before ADR 0013 goes away.
- A dev lake with only some datasets needs the `datasets` config to build gold.
