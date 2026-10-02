#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A gold that lives in memory, so a sync test needs no lakehouse."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import md5
from uuid import UUID

import numpy as np
import pyarrow as pa

from cv_lakehouse_studio.gold_client import (
    GoldClient,
    GoldDataset,
    GoldMeta,
    GoldSplit,
)

DATASET = "faces"
CLASS_NAMES = ["face", "license_plate", "other"]
EMBEDDING_DIMENSION = 4
FIRST_BUILD = datetime(2026, 1, 1, tzinfo=UTC)


def derive_id(key: str) -> str:
    return str(UUID(hex=md5(key.encode("utf-8")).hexdigest()))


IMAGE_A = derive_id("a.jpg")
IMAGE_B = derive_id("b.jpg")
BOX_1 = derive_id("a.jpg#0")
BOX_2 = derive_id("a.jpg#1")


def _make_default_boxes() -> list[dict]:
    return [
        make_box(box_id=BOX_1, class_name="face", x=10.4, y=20.5, w=30.0, h=40.0),
        make_box(box_id=BOX_2, class_name="license_plate", x=1.0, y=2.0, w=3.0, h=4.0),
    ]


def make_box(
    *,
    box_id: str,
    class_name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    origin: str = "source",
) -> dict:
    return {
        "box_id": box_id,
        "image_id": IMAGE_A,
        "class_name": class_name,
        "confidence": None,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "origin": origin,
    }


@dataclass
class FakeGoldClient(GoldClient):
    """One dataset of two images. `a.jpg` holds the boxes, and `b.jpg` holds none.

    A test edits `boxes` and calls `publish_new_version`, as a gold build does.
    """

    boxes: list[dict] = field(default_factory=_make_default_boxes)
    embedding_model: str | None = "mobileclip_s0"
    version: int = 1
    requested_changed_since: list[datetime | None] = field(default_factory=list)

    def publish_new_version(self) -> None:
        self.version += 1

    def read_meta(self) -> GoldMeta:
        return GoldMeta(
            version=self.version,
            built_at=FIRST_BUILD + timedelta(days=self.version),
            datasets=[
                GoldDataset(
                    dataset=DATASET,
                    embedding_model=self.embedding_model,
                    splits=[
                        GoldSplit(split="train", role="train"),
                        GoldSplit(split="validation", role="val"),
                    ],
                )
            ],
        )

    def read_class_names(self) -> list[str]:
        return CLASS_NAMES

    def iter_pages(
        self, *, table: str, dataset: str, changed_since: datetime | None = None
    ) -> Iterator[pa.Table]:
        assert dataset == DATASET
        if table == "images":
            yield pa.Table.from_pylist(
                [
                    {
                        "image_id": IMAGE_A,
                        "file_name": "a.jpg",
                        "width": 100,
                        "height": 80,
                        "image_path": "bronze/faces/train/a.jpg",
                        "role": "train",
                        "split": "train",
                    },
                    {
                        "image_id": IMAGE_B,
                        "file_name": "b.jpg",
                        "width": 64,
                        "height": 64,
                        "image_path": "bronze/faces/validation/b.jpg",
                        "role": "val",
                        "split": "validation",
                    },
                ]
            )
        elif table == "boxes":
            yield pa.Table.from_pylist(self.boxes, schema=_BOX_SCHEMA)
        else:
            self.requested_changed_since.append(changed_since)
            key = "image_id" if table == "embeddings" else "box_id"
            ids = (
                [IMAGE_A, IMAGE_B]
                if table == "embeddings"
                else [box["box_id"] for box in self.boxes]
            )
            yield _make_embedding_page(key=key, ids=ids)


_BOX_SCHEMA = pa.schema(
    [
        pa.field(name="box_id", type=pa.string()),
        pa.field(name="image_id", type=pa.string()),
        pa.field(name="class_name", type=pa.string()),
        pa.field(name="confidence", type=pa.float64()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
        pa.field(name="origin", type=pa.string()),
    ]
)


def _make_embedding_page(key: str, ids: list[str]) -> pa.Table:
    vectors = np.ones(shape=(len(ids), EMBEDDING_DIMENSION), dtype=np.float32)
    return pa.table(
        {
            key: pa.array(ids, type=pa.string()),
            "embedding": pa.array(vectors.tolist(), type=pa.list_(pa.float32())),
        }
    )
