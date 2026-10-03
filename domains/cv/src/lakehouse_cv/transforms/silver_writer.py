#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The streaming writer that fills the silver images and boxes files of one split."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse_cv.contract.silver_tables import (
    BOX_SCHEMA,
    IMAGE_SCHEMA,
    ROWS_PER_ROW_GROUP,
    SilverImage,
    boxes_file,
    images_file,
)


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
                        "box_id": box.box_id,
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
                        "origin": box.origin,
                        "is_class_corrected": box.is_class_corrected,
                        "is_geometry_corrected": box.is_geometry_corrected,
                    }
                )
    finally:
        image_writer.close()
        box_writer.close()
    return image_writer.count, box_writer.count
