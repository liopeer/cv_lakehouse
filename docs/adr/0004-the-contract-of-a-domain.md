<!--
SPDX-License-Identifier: MIT
Copyright (c) 2025–2026 Lionel Peer
-->
# ADR 0004: The contract of a domain lives in `contract/`

- Status: accepted
- Date: 2026-10-03

## Context

A reader of a layer needs its schema, its keys and its manifests. A reader does not
need the code that fills the layer. In the CV domain, that code needs Dagster, PIL,
labelformat and a Triton client. Schemas were next to writers, so a reader imported
both.

## Decision

Each domain has a `contract/` package. It holds what each layer of the domain holds:

- the Arrow schemas, the file paths and the row types,
- a `TableSpec` for each table, with its layer, its key and a description,
- the manifests, the vocabularies and the ids.

`contract/` imports only the standard library, pyarrow, pydantic, core and itself.
The logic of a domain is in `sources/` and `transforms/`, and both import the
contract. `tests/test_import_direction.py` enforces the rule.

## Consequences

- A consumer reads a layer with the contract, and installs neither Dagster nor a model
  client.
- A schema change is one change in `contract/`, and every writer and reader follows it.
- A file that holds a schema and its writer splits in two. `silver_tables.py` holds the
  silver schema, and `transforms/silver_writer.py` writes it.
