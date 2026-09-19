#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Open Images V7, every file that its download page links to.

This is the only large box source for both faces and licence plates whose licence allows
commercial use. Annotations are CC-BY-4.0 and the images are CC-BY-2.0.

Bronze holds every annotation file byte for byte and every image tar, unpacked. The
readers here use the box CSVs and the images. Everything else waits in bronze for a
consumer.
"""

from __future__ import annotations

import csv
from argparse import ArgumentParser
from collections.abc import Iterable, Iterator
from pathlib import Path

from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)

from cv_lakehouse.sources.base import (
    BoxAttributeSource,
    BronzeSource,
    DatasetSpec,
    RawBox,
    RawImage,
    read_image_size,
)
from cv_lakehouse.sources.open_images_files import OPEN_IMAGES_FILES

# Open Images labels a box with a Freebase id. These are the ids the class registry
# names.
# Every other class falls to the default, so bronze keeps it and silver keeps the box.
FREEBASE_MID_TO_CLASS_NAME = {
    "/m/0dzct": "face",
    "/m/01jfm_": "license_plate",
    "/m/04hgtk": "head",
    "/m/01g317": "person",
}


def freebase_category(mid: str) -> Category:
    """Name a bronze category by its Freebase id.

    Silver resolves a category by name, so the id here never reaches a file.
    """
    return Category(id=0, name=mid)


BOX_CSV_NAME = {
    "train": "oidv6-train-annotations-bbox.csv",
    "validation": "validation-annotations-bbox.csv",
    "test": "test-annotations-bbox.csv",
}
# The CVDF tars shard the train images by the first hex digit of the image id, into
# train_0 to train_f. Validation and test each unpack into one folder.
TRAIN_SHARD_PREFIX = "train_"

SPEC = DatasetSpec(
    name="open_images",
    homepage="https://storage.googleapis.com/openimages/web/index.html",
    license="CC-BY-4.0 annotations, CC-BY-2.0 images",
    commercial_use=True,
    splits=("train", "validation", "test"),
    category_map=FREEBASE_MID_TO_CLASS_NAME,
    default_class="other",
    notes="Web photography. The only commercially usable source for both classes.",
)


class OpenImagesSource(BronzeSource, BoxAttributeSource):
    spec = SPEC
    published_files = OPEN_IMAGES_FILES

    def image_root(self, bronze_dir: Path, split: str) -> Path:
        _check_split(split)
        return bronze_dir if split == "train" else bronze_dir / split

    def open_labelformat_reader(
        self, bronze_dir: Path, split: str
    ) -> ObjectDetectionInput:
        """The labelformat view, which keeps only the single instance boxes.

        labelformat carries no per box flag, so a reader of this cannot tell a crowd
        box from one face. `read_raw_images` keeps every row and reports both flags.
        Silver reads raw images.
        """
        return OpenImagesObjectDetectionInput(
            image_root=self.image_root(bronze_dir=bronze_dir, split=split),
            annotation_file=_annotation_file(bronze_dir=bronze_dir, split=split),
            split=split,
        )

    def read_raw_images(self, bronze_dir: Path, split: str) -> Iterator[RawImage]:
        """Read every row, with IsGroupOf and IsDepiction alongside the box.

        `is_instance_box` says a consumer decides about these. It now can: silver keeps
        the box and the flag, and its consumer applies the rule.
        """
        image_root = self.image_root(bronze_dir=bronze_dir, split=split)
        for image_id, rows in _rows_by_image(
            _annotation_file(bronze_dir=bronze_dir, split=split)
        ).items():
            file_name = build_image_file_name(split=split, image_id=image_id)
            path = image_root / file_name
            if not path.exists():
                continue
            width, height = read_image_size(path)
            yield RawImage(
                file_name=file_name,
                width=width,
                height=height,
                boxes=tuple(
                    _build_raw_box(row=row, width=width, height=height) for row in rows
                ),
            )


class OpenImagesObjectDetectionInput(ObjectDetectionInput):
    """Read an Open Images box CSV against the images that downloaded.

    Coordinates in the CSV are normalised to the image, so every box needs the real
    width and height. Open Images does not publish them, so they come from the file.
    """

    def __init__(self, *, image_root: Path, annotation_file: Path, split: str) -> None:
        self.image_root = image_root
        self.annotation_file = annotation_file
        self.split = split

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError

    def get_categories(self) -> Iterable[Category]:
        return (freebase_category(mid) for mid in FREEBASE_MID_TO_CLASS_NAME)

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        by_image = _rows_by_image(self.annotation_file)
        index = 0
        for image_id, rows in by_image.items():
            file_name = build_image_file_name(split=self.split, image_id=image_id)
            path = self.image_root / file_name
            if not path.exists():
                continue
            width, height = read_image_size(path)
            image = Image(id=index, filename=file_name, width=width, height=height)
            index += 1
            yield ImageObjectDetection(
                image=image,
                objects=[
                    SingleObjectDetection(
                        category=freebase_category(row["LabelName"]),
                        box=BoundingBox.from_format(
                            bbox=[
                                float(row["XMin"]) * width,
                                float(row["YMin"]) * height,
                                float(row["XMax"]) * width,
                                float(row["YMax"]) * height,
                            ],
                            format=BoundingBoxFormat.XYXY,
                        ),
                    )
                    for row in rows
                    if is_instance_box(row)
                ],
            )


def _build_raw_box(row: dict[str, str], width: int, height: int) -> RawBox:
    """Scale one CSV row onto the image. Coordinates in the file are normalised."""
    return RawBox(
        source_class=row["LabelName"],
        xmin=float(row["XMin"]) * width,
        ymin=float(row["YMin"]) * height,
        xmax=float(row["XMax"]) * width,
        ymax=float(row["YMax"]) * height,
        attrs={
            "attr_is_group_of": row["IsGroupOf"] == "1",
            "attr_is_depiction": row["IsDepiction"] == "1",
        },
    )


def _rows_by_image(annotation_file: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with annotation_file.open(newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["ImageID"], []).append(row)
    return grouped


def build_image_file_name(split: str, image_id: str) -> str:
    """Name an image relative to the image root of its split."""
    if split == "train":
        return f"{TRAIN_SHARD_PREFIX}{image_id[0]}/{image_id}.jpg"
    return f"{image_id}.jpg"


def is_instance_box(row: dict[str, str]) -> bool:
    """Decide whether one CSV row bounds a single real instance.

    A group box covers many instances at once. A depiction is a drawing or a statue of
    the thing. Both change what the box means, so a consumer decides about them. Bronze
    keeps both flags, so a silver rebuild reverses this rule without a download.
    """
    return row["IsGroupOf"] != "1" and row["IsDepiction"] != "1"


def _annotation_file(bronze_dir: Path, split: str) -> Path:
    return bronze_dir / BOX_CSV_NAME[split]


def _check_split(split: str) -> None:
    if split not in BOX_CSV_NAME:
        raise ValueError(f"Unknown Open Images split: {split}")
