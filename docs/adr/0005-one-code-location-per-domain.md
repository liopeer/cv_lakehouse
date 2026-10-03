<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0005: One Dagster code location per domain

- Status: accepted
- Date: 2026-10-03

## Context

A Dagster code location is a process that loads definitions and runs them. One
location for every domain installs every dependency in one image. A broken import in
one domain then stops the assets of every domain.

## Decision

Each domain is one Dagster project and one code location, with its own image. The root
`pyproject.toml` is a `dg` workspace that lists the projects. The CV location keeps the
name `cv_lakehouse`, because Dagster keys the state of schedules and sensors by it.

## Consequences

- `make dev` loads every location of the workspace in one UI.
- `make defs` and `make dev` pass `--use-active-venv`, because the workspace has one
  virtualenv at its root.
- Each domain has a Dockerfile under `domains/<name>/`. Its build context is the root
  of the repository, because the lock belongs to the workspace.
