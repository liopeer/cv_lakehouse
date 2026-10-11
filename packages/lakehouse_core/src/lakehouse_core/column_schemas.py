#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The Parquet schemas of an asset, as the Dagster UI shows them."""

from collections.abc import Sequence

import dagster as dg
import pyarrow as pa

from lakehouse_core.table_spec import TableSpec


def build_column_schema(schema: pa.Schema, key: Sequence[str] = ()) -> dg.TableSchema:
    key = tuple(key)
    return dg.TableSchema(
        columns=[
            dg.TableColumn(
                name=field.name,
                type=str(field.type),
                constraints=dg.TableColumnConstraints(
                    nullable=field.nullable, unique=key == (field.name,)
                ),
            )
            for field in schema
        ],
        # A key of one column is a unique column. Only a composite key needs a note.
        constraints=(
            dg.TableConstraints(other=[f"unique key: {', '.join(key)}"])
            if len(key) > 1
            else None
        ),
    )


def build_table_schema_metadata(
    tables: Sequence[TableSpec],
) -> dict[str, dg.TableSchemaMetadataValue]:
    return {
        f"{table.name}_schema": dg.MetadataValue.table_schema(
            build_column_schema(schema=table.schema, key=table.key)
        )
        for table in tables
    }
