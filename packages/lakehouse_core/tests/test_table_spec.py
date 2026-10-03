#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import pyarrow as pa
import pytest

from lakehouse_core.table_spec import Layer, TableSpec

SCHEMA = pa.schema(
    [
        pa.field(name="row_id", type=pa.string(), nullable=False),
        pa.field(name="note", type=pa.string()),
    ]
)


def test_a_key_on_a_required_column_is_accepted() -> None:
    spec = TableSpec(
        name="rows", layer=Layer.SILVER, schema=SCHEMA, key=("row_id",), description=""
    )
    assert spec.key == ("row_id",)


@pytest.mark.parametrize(
    ("key", "message"),
    [((), "at least one"), (("missing",), "no key columns"), (("note",), "nullable")],
)
def test_a_bad_key_is_rejected(key: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TableSpec(
            name="rows", layer=Layer.SILVER, schema=SCHEMA, key=key, description=""
        )
