#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Read and write Parquet at a path in the lake, local or remote."""

from collections.abc import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from lakehouse_core.lake_store import arrow_location


def open_parquet_writer(path: UPath, schema: pa.Schema) -> pq.ParquetWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    where, filesystem = arrow_location(path)
    return pq.ParquetWriter(where=where, schema=schema, filesystem=filesystem)


def open_parquet_file(path: UPath) -> pq.ParquetFile:
    where, filesystem = arrow_location(path)
    return pq.ParquetFile(where, filesystem=filesystem)


def read_parquet_table(path: UPath, columns: Sequence[str] | None = None) -> pa.Table:
    where, filesystem = arrow_location(path)
    return pq.read_table(
        where, columns=None if columns is None else list(columns), filesystem=filesystem
    )


def read_parquet_schema(path: UPath) -> pa.Schema:
    where, filesystem = arrow_location(path)
    return pq.read_schema(where, filesystem=filesystem)
