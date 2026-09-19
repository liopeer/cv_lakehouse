#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The Parquet schema that silver writes, and the streaming writer that fills it.

Two files per split. One row per image, and one row per box that joins back to it on
`(dataset, split, file_name)`. A row per box alone would lose an image that carries no
box, and a detector needs those negatives.

Two more files hold the MobileCLIP embeddings, one per image and one per box crop, on
the same keys. They are separate files because a vector is 2 KB next to a box row of a
few dozen bytes, and most queries over silver want the boxes and not the vectors.

Every dataset writes the same columns. A source that knows nothing about `attr_blur`
leaves it null, which costs almost nothing in Parquet and keeps a multi dataset
`read_parquet([...])` free of `union_by_name`. DuckDB refuses a strict scan over files
whose schemas disagree, so one column set is what keeps a merge across datasets
simple.

The schema here is the write time validation. The COCO path had none.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Per box attributes, keyed by column name. A source declares the ones it fills in its
# `DatasetSpec`. Which ones a dataset really filled is a question for its Parquet, and
# the manifest does not answer it.
#
# LightlyStudio cannot show these yet: its `CreateObjectDetection` carries only a class
# name, a confidence and a box, and its public `Annotation` exposes no metadata. They
# are here for queries and for training, not for the GUI.
ATTRIBUTE_COLUMNS: tuple[pa.Field, ...] = (
    # WIDER FACE grades every face on five axes and flags the ones it rejected.
    pa.field(name="attr_blur", type=pa.int32()),
    pa.field(name="attr_expression", type=pa.int32()),
    pa.field(name="attr_illumination", type=pa.int32()),
    pa.field(name="attr_occlusion", type=pa.int32()),
    pa.field(name="attr_pose", type=pa.int32()),
    pa.field(name="attr_invalid", type=pa.bool_()),
    # Open Images marks a box that covers a crowd, and one that bounds a depiction.
    pa.field(name="attr_is_group_of", type=pa.bool_()),
    pa.field(name="attr_is_depiction", type=pa.bool_()),
)

IMAGE_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="width", type=pa.int32(), nullable=False),
        pa.field(name="height", type=pa.int32(), nullable=False),
    ]
)

BOX_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_index", type=pa.int32(), nullable=False),
        pa.field(name="class_id", type=pa.int32(), nullable=False),
        pa.field(name="class_name", type=pa.string(), nullable=False),
        pa.field(name="source_class", type=pa.string()),
        pa.field(name="x", type=pa.float64(), nullable=False),
        pa.field(name="y", type=pa.float64(), nullable=False),
        pa.field(name="w", type=pa.float64(), nullable=False),
        pa.field(name="h", type=pa.float64(), nullable=False),
        pa.field(name="confidence", type=pa.float64()),
        *ATTRIBUTE_COLUMNS,
    ]
)

# A variable length list, not a fixed size one. DuckDB reads a fixed size list as
# `FLOAT[512]`, and LightlyStudio's `sample_embedding.embedding` column is `FLOAT[]`, so
# a fixed size list would need a cast on every insert.
EMBEDDING_COLUMN = "embedding"
_EMBEDDING_FIELD = pa.field(
    name=EMBEDDING_COLUMN, type=pa.list_(pa.float32()), nullable=False
)

EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        _EMBEDDING_FIELD,
    ]
)

CROP_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_index", type=pa.int32(), nullable=False),
        _EMBEDDING_FIELD,
    ]
)

# Rows buffered before a row group is written. A large split never lands in memory.
ROWS_PER_ROW_GROUP = 8192


def images_file(silver_dir: Path, split: str) -> Path:
    return silver_dir / "images" / f"{split}.parquet"


def boxes_file(silver_dir: Path, split: str) -> Path:
    return silver_dir / "boxes" / f"{split}.parquet"


def embeddings_file(silver_dir: Path, split: str) -> Path:
    return silver_dir / "embeddings" / f"{split}.parquet"


def crop_embeddings_file(silver_dir: Path, split: str) -> Path:
    return silver_dir / "crop_embeddings" / f"{split}.parquet"


@dataclass(frozen=True)
class SilverBox:
    """One box on a canonical class, in pixel XYWH, clipped to its image."""

    box_index: int
    class_id: int
    class_name: str
    source_class: str
    x: float
    y: float
    w: float
    h: float
    confidence: float | None = None
    attrs: Mapping[str, int | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class SilverImage:
    """One image and the boxes that survived normalisation."""

    file_name: str
    width: int
    height: int
    boxes: tuple[SilverBox, ...] = ()


class _BatchWriter:
    """Buffer dict rows and flush them as one row group per batch."""

    def __init__(self, path: Path, schema: pa.Schema) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._writer = pq.ParquetWriter(where=path, schema=schema)
        self._rows: list[dict[str, object]] = []
        self.count = 0

    def add(self, row: Mapping[str, object]) -> None:
        self._rows.append(dict(row))
        self.count += 1
        if len(self._rows) >= ROWS_PER_ROW_GROUP:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        # from_pylist validates every value against the schema, and fills a column the
        # row omits with null. That is how a source skips an attribute it knows nothing
        # about.
        self._writer.write_batch(
            pa.RecordBatch.from_pylist(mapping=self._rows, schema=self._schema)
        )
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        self._writer.close()


def write_split(
    *,
    silver_dir: Path,
    dataset: str,
    split: str,
    images: Iterable[SilverImage],
) -> tuple[int, int]:
    """Write one split's two Parquet files. Return the image and the box count.

    An empty split still writes both files, so a reader never has to tell a missing
    split from an empty one.
    """
    image_writer = _BatchWriter(
        path=images_file(silver_dir=silver_dir, split=split), schema=IMAGE_SCHEMA
    )
    box_writer = _BatchWriter(
        path=boxes_file(silver_dir=silver_dir, split=split), schema=BOX_SCHEMA
    )
    try:
        for image in images:
            key = {"dataset": dataset, "split": split, "file_name": image.file_name}
            image_writer.add({**key, "width": image.width, "height": image.height})
            for box in image.boxes:
                box_writer.add(
                    {
                        **key,
                        "box_index": box.box_index,
                        "class_id": box.class_id,
                        "class_name": box.class_name,
                        "source_class": box.source_class,
                        "x": box.x,
                        "y": box.y,
                        "w": box.w,
                        "h": box.h,
                        "confidence": box.confidence,
                        **box.attrs,
                    }
                )
    finally:
        image_writer.close()
        box_writer.close()
    return image_writer.count, box_writer.count
