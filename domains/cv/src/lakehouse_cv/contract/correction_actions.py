#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The corrections that the LightlyStudio export publishes, one row per box."""

import pyarrow as pa


class CorrectionAction:
    # The class, the box, or both changed. A null field is unchanged.
    UPDATE = "update"
    DELETE = "delete"
    # A box that a curator drew. Its id is the id LightlyStudio gave it.
    ADD = "add"


# One row per corrected box. `label_name` is the label as LightlyStudio holds it. The
# box is in whole pixels, as LightlyStudio stores it, and all four are null together.
CORRECTION_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="action", type=pa.string(), nullable=False),
        pa.field(name="label_name", type=pa.string()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
    ]
)
