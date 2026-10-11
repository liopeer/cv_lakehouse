#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import dagster as dg
import pyarrow as pa

from lakehouse_core.column_schemas import (
    build_column_schema,
    build_table_schema_metadata,
)
from lakehouse_core.table_spec import Layer, TableSpec

SCHEMA = pa.schema(
    [
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="row_id", type=pa.string(), nullable=False),
        pa.field(name="vector", type=pa.list_(pa.float32())),
    ]
)


def test_a_column_keeps_its_type_and_nullability() -> None:
    columns = {column.name: column for column in build_column_schema(SCHEMA).columns}
    assert columns["vector"].type == "list<item: float>"
    assert columns["vector"].constraints.nullable
    assert not columns["split"].constraints.nullable


def test_a_key_of_one_column_marks_the_column_unique() -> None:
    schema = build_column_schema(schema=SCHEMA, key=("row_id",))
    assert [column.constraints.unique for column in schema.columns] == [
        False,
        True,
        False,
    ]
    assert schema.constraints.other == []


def test_a_composite_key_is_a_table_constraint() -> None:
    schema = build_column_schema(schema=SCHEMA, key=("split", "row_id"))
    assert not any(column.constraints.unique for column in schema.columns)
    assert schema.constraints.other == ["unique key: split, row_id"]


def test_every_table_gets_one_schema_entry() -> None:
    table = TableSpec(
        name="rows", layer=Layer.SILVER, schema=SCHEMA, key=("row_id",), description=""
    )
    metadata = build_table_schema_metadata([table])
    assert list(metadata) == ["rows_schema"]
    assert isinstance(metadata["rows_schema"], dg.TableSchemaMetadataValue)
