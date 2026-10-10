#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The edits of the curators: the events that bronze holds, and the corrections of
silver.

The LightlyStudio export publishes an event per edited box: the box as LightlyStudio
holds it, or a tombstone. Silver folds the events over the source into corrections.
"""

import pyarrow as pa


class CorrectionAction:
    # The class, the box, or both changed. A None field is unchanged.
    UPDATE = "update"
    DELETE = "delete"
    # A box that a curator drew. Its id is the id LightlyStudio gave it.
    ADD = "add"


# One row per edited box, in the order of the log. `log_sequence` is the position of the
# last edit in the log of the LightlyStudio database. A tombstone has no label and no
# box. Otherwise the box is in whole pixels, as LightlyStudio stores it.
EVENT_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="log_sequence", type=pa.int64(), nullable=False),
        pa.field(name="is_deleted", type=pa.bool_(), nullable=False),
        pa.field(name="label_name", type=pa.string()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
    ]
)
