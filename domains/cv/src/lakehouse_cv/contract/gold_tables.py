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

import pyarrow as pa
from upath import UPath

from lakehouse_core.table_spec import Layer, TableSpec
from lakehouse_cv.contract.silver_tables import ATTRIBUTE_COLUMNS, EMBEDDING_COLUMN

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
        pa.field(name="origin", type=pa.string(), nullable=False),
        pa.field(name="is_class_corrected", type=pa.bool_(), nullable=False),
        pa.field(name="is_geometry_corrected", type=pa.bool_(), nullable=False),
        pa.field(name="commercial_use", type=pa.bool_(), nullable=False),
        _CHANGED_AT_FIELD,
    ]
)

# What the gold API serves for a vector. Gold stores none: the rows come from the
# silver embedding files, on the id of the gold row they belong to.
_EMBEDDING_FIELD = pa.field(
    name=EMBEDDING_COLUMN, type=pa.list_(pa.float32()), nullable=False
)
GOLD_EMBEDDING_SCHEMA = pa.schema(
    [pa.field(name="image_id", type=pa.string(), nullable=False), _EMBEDDING_FIELD]
)
GOLD_CROP_EMBEDDING_SCHEMA = pa.schema(
    [pa.field(name="box_id", type=pa.string(), nullable=False), _EMBEDDING_FIELD]
)


GOLD_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        name="images",
        layer=Layer.GOLD,
        schema=GOLD_IMAGE_SCHEMA,
        key=("image_id",),
        description="One row per image of every dataset, with its role and licence.",
    ),
    TableSpec(
        name="boxes",
        layer=Layer.GOLD,
        schema=GOLD_BOX_SCHEMA,
        key=("box_id",),
        description="One row per box that gold keeps. No flagged box is here.",
    ),
    TableSpec(
        name="embeddings",
        layer=Layer.GOLD,
        schema=GOLD_EMBEDDING_SCHEMA,
        key=("image_id",),
        description="One vector per image, served from the silver files.",
    ),
    TableSpec(
        name="crop_embeddings",
        layer=Layer.GOLD,
        schema=GOLD_CROP_EMBEDDING_SCHEMA,
        key=("box_id",),
        description="One vector per box crop, served from the silver files.",
    ),
)


def gold_images_file(*, version_dir: UPath, dataset: str, split: str) -> UPath:
    return version_dir / "images" / dataset / f"{split}.parquet"


def gold_boxes_file(*, version_dir: UPath, dataset: str, split: str) -> UPath:
    return version_dir / "boxes" / dataset / f"{split}.parquet"
