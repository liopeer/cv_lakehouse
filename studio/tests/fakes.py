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
    GoldSlice,
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


def make_image(*, file_name: str, split: str, image_path: str) -> dict:
    return {
        "image_id": derive_id(file_name),
        "file_name": file_name,
        "width": 64,
        "height": 64,
        "image_path": image_path,
        "role": "train" if split == "train" else "val",
        "split": split,
    }


def make_box(
    *,
    box_id: str,
    class_name: str,
    x: float,
    y: float,
    w: float,
    h: float,
    origin: str = "source",
    image_id: str = IMAGE_A,
) -> dict:
    return {
        "box_id": box_id,
        "image_id": image_id,
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
    # Relative to the lake root, unless bronze links a copy outside of it.
    image_paths: tuple[str, str] = (
        "bronze/faces/train/a.jpg",
        "bronze/faces/validation/b.jpg",
    )
    # Images beyond `a.jpg` and `b.jpg`, such as for a test of shards.
    more_images: list[dict] = field(default_factory=list)
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

    def count_images(self, gold_slice: GoldSlice) -> int:
        return len(self._select_images(gold_slice))

    def iter_pages(
        self,
        *,
        table: str,
        gold_slice: GoldSlice,
        changed_since: datetime | None = None,
    ) -> Iterator[pa.Table]:
        images = self._select_images(gold_slice)
        image_ids = {image["image_id"] for image in images}
        boxes = [box for box in self.boxes if box["image_id"] in image_ids]
        if table == "images":
            yield pa.Table.from_pylist(images, schema=_IMAGE_SCHEMA)
        elif table == "boxes":
            yield pa.Table.from_pylist(boxes, schema=_BOX_SCHEMA)
        else:
            self.requested_changed_since.append(changed_since)
            key = "image_id" if table == "embeddings" else "box_id"
            ids = (
                [image["image_id"] for image in images]
                if table == "embeddings"
                else [box["box_id"] for box in boxes]
            )
            yield _make_embedding_page(key=key, ids=ids)

    def _select_images(self, gold_slice: GoldSlice) -> list[dict]:
        assert gold_slice.dataset == DATASET
        images = [
            {
                "image_id": IMAGE_A,
                "file_name": "a.jpg",
                "width": 100,
                "height": 80,
                "image_path": self.image_paths[0],
                "role": "train",
                "split": "train",
            },
            {
                "image_id": IMAGE_B,
                "file_name": "b.jpg",
                "width": 64,
                "height": 64,
                "image_path": self.image_paths[1],
                "role": "val",
                "split": "validation",
            },
            *self.more_images,
        ]
        return [
            image
            for image in images
            if (gold_slice.split is None or image["split"] == gold_slice.split)
            and (
                gold_slice.shard is None
                or gold_slice.num_shards is None
                or select_shard(
                    image_id=image["image_id"], num_shards=gold_slice.num_shards
                )
                == gold_slice.shard
            )
        ]


def select_shard(*, image_id: str, num_shards: int) -> int:
    """The rule of the gold API."""
    return int(image_id[:8], 16) % num_shards


_IMAGE_SCHEMA = pa.schema(
    [
        pa.field(name="image_id", type=pa.string()),
        pa.field(name="file_name", type=pa.string()),
        pa.field(name="width", type=pa.int64()),
        pa.field(name="height", type=pa.int64()),
        pa.field(name="image_path", type=pa.string()),
        pa.field(name="role", type=pa.string()),
        pa.field(name="split", type=pa.string()),
    ]
)

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
