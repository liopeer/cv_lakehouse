<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0002: One uv workspace, and triton outside it

- Status: accepted
- Date: 2026-10-03

## Context

The app, `studio/` and `triton/` each had a lock file. The app and studio share many
dependencies, such as pyarrow, httpx and pydantic. Two locks let their versions drift
apart. The next domains add more packages, and each one needs the same tools.

## Decision

The repository root is a uv workspace with one `uv.lock`. Its members are
`packages/lakehouse_core`, every package under `domains/`, and `studio/`. The root
holds the development tools and their configuration.

`triton/` stays outside the workspace, with its own lock. It replaces pillow with
pillow-simd and takes torch for CUDA or ROCm. One resolution cannot hold both, and
triton shares no code with the members.

## Consequences

- `make check` runs over every member with one set of tools.
- The workspace resolves for Python 3.12 only, because studio requires `<3.13`.
- A dependency change of studio changes the root `uv.lock`. release-please then lists
  that commit in the app changelog as well.
- The studio image still builds from the constraints of upstream LightlyStudio, not
  from this lock. The lock decides only the versions that the studio tests run on.
- The app version is in `version.txt`, because the root has no `[project]`.
  release-please writes it into the pyproject files of the members and into `uv.lock`.
