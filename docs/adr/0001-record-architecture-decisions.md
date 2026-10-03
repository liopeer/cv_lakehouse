<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0001: Record architecture decisions

- Status: accepted
- Date: 2026-10-03

## Context

The repository grows from one computer vision pipeline into a lakehouse for several
domains. A reader of the code sees what the code does. The reader does not see why a
boundary is where it is, or which options were rejected.

## Decision

Each decision about the architecture is a numbered record in `docs/adr/`. A record has a
context, a decision and its consequences. A record does not change after it is
accepted. A new record supersedes it instead.

## Consequences

A change that moves a boundary comes with a record in the same pull request. A reviewer
reads the record before the diff. An agent that works on the code reads the records
that `AGENTS.md` names.
