#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The Parquet schema that gold writes.

Gold is the union of every silver dataset, one images and one boxes file per dataset
and split. A row carries its ids, its role and its licence, so one scan over every file
answers a question across datasets.

Gold holds no flagged box. Silver keeps a crowd box, a depiction and a region the
annotators rejected, and gold is where they drop out.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from cv_lakehouse.silver_schema import ATTRIBUTE_COLUMNS

# A box with one of these set is not in gold, so gold has no column for them.
FLAG_COLUMNS = ("attr_is_group_of", "attr_is_depiction", "attr_invalid")

CHANGED_AT_COLUMN = "changed_at"
_CHANGED_AT_FIELD = pa.field(
    name=CHANGED_AT_COLUMN, type=pa.timestamp("us", tz="UTC"), nullable=False
)

GOLD_IMAGE_SCHEMA = pa.schema(
    [
        pa.field(name="image_id", type=pa.string(), nullable=False),
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="role", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="width", type=pa.int32(), nullable=False),
        pa.field(name="height", type=pa.int32(), nullable=False),
        # Relative to the lake root.
        pa.field(name="image_path", type=pa.string(), nullable=False),
        pa.field(name="license", type=pa.string(), nullable=False),
        pa.field(name="commercial_use", type=pa.bool_(), nullable=False),
        _CHANGED_AT_FIELD,
    ]
)

GOLD_BOX_SCHEMA = pa.schema(
    [
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="image_id", type=pa.string(), nullable=False),
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="role", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="class_id", type=pa.int32(), nullable=False),
        pa.field(name="class_name", type=pa.string(), nullable=False),
        pa.field(name="source_class", type=pa.string()),
        pa.field(name="x", type=pa.float64(), nullable=False),
        pa.field(name="y", type=pa.float64(), nullable=False),
        pa.field(name="w", type=pa.float64(), nullable=False),
        pa.field(name="h", type=pa.float64(), nullable=False),
        pa.field(name="confidence", type=pa.float64()),
        *(field for field in ATTRIBUTE_COLUMNS if field.name not in FLAG_COLUMNS),
        pa.field(name="commercial_use", type=pa.bool_(), nullable=False),
        _CHANGED_AT_FIELD,
    ]
)


def gold_images_file(*, version_dir: Path, dataset: str, split: str) -> Path:
    return version_dir / "images" / dataset / f"{split}.parquet"


def gold_boxes_file(*, version_dir: Path, dataset: str, split: str) -> Path:
    return version_dir / "boxes" / dataset / f"{split}.parquet"
