<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0003: A core package, and one package per domain

- Status: accepted
- Date: 2026-10-03

## Context

The lakehouse adds a second domain: coding agent trajectories. It has pull requests,
diff statistics, complexity metrics and agent runs. It shares no table with computer
vision, and it needs other dependencies. Both domains need the same bronze rules, the
same layer directories and the same manifests.

## Decision

The code is in two kinds of packages.

- `packages/lakehouse_core` knows how the lake works: the published files and their
  checksums, the bronze asset, the layer directories, the manifests and `TableSpec`.
  It names no domain.
- Each domain is one package under `domains/`. It holds the contract, the sources, the
  transforms and the Dagster definitions of that domain.

The imports follow four rules:

- Core imports no domain.
- A domain imports itself and core, and nothing else of the lakehouse.
- No domain imports another domain.
- Only `defs/` and `definitions.py` import `defs/`.

`tests/test_import_direction.py` enforces the rules in `make check`.

Code moves into core when a second domain uses it with the same meaning. A guess at
what is generic stays in the domain until then.

## Consequences

- A domain image installs core and one domain, and none of the dependencies of
  another domain.
- Core reaches a domain only through a protocol, such as `PublishedSource`. Core
  never names a type of a domain.
- Two domains that need the same table join in a third place, such as a consumer. A
  domain does not import the other to join.
