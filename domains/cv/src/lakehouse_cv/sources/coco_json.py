#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A streaming reader for the COCO JSON layout, which COCO, LVIS and COCO-WholeBody use.

A file holds every image and every annotation of a split in one object. Polygons and
keypoints make most of it, up to 3 GB for COCO-WholeBody train. `json.load` holds the
text and the objects at once, so it peaks at twice the file size. This reader streams
each list, keeps the fields that it is asked for, and peaks at a few hundred MB.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import ijson
from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)
from upath import UPath

from lakehouse_cv.sources.base import RawBox, RawImage

type CocoObject = dict[str, Any]


@dataclass(frozen=True)
class CocoJson:
    class_name_by_id: dict[int, str]
    images: list[CocoObject]
    annotations_by_image_id: dict[int, list[CocoObject]]

    def iter_images_with_annotations(
        self,
    ) -> Iterator[tuple[CocoObject, list[CocoObject]]]:
        """Yield every image in file order, also an image without an annotation."""
        for image in self.images:
            yield image, self.annotations_by_image_id.get(image["id"], [])


def read_coco_json(
    annotation_file: UPath, *, annotation_fields: Sequence[str]
) -> CocoJson:
    """Read the file three times, once for each list, and keep `annotation_fields`."""
    categories = _read_items(path=annotation_file, prefix="categories.item")
    images = _read_items(path=annotation_file, prefix="images.item")
    annotations_by_image_id: dict[int, list[CocoObject]] = defaultdict(list)
    for annotation in _iter_items(path=annotation_file, prefix="annotations.item"):
        kept = {field: annotation[field] for field in annotation_fields}
        annotations_by_image_id[annotation["image_id"]].append(kept)
    return CocoJson(
        class_name_by_id={category["id"]: category["name"] for category in categories},
        images=images,
        annotations_by_image_id=dict(annotations_by_image_id),
    )


def coco_image_path(image: CocoObject) -> str:
    """Name an image relative to the bronze of COCO, such as `val2017/000000397133.jpg`.

    LVIS gives no file name, and mixes train2017 and val2017 images in one split. The
    folder and the file name of `coco_url` locate the image in either case.
    """
    folder, file_name = image["coco_url"].rsplit("/", maxsplit=2)[-2:]
    return f"{folder}/{file_name}"


def build_crowd_flagged_box(*, annotation: CocoObject, source_class: str) -> RawBox:
    """Turn a COCO `bbox` into a box. A crowd region covers many instances."""
    return build_raw_box(
        xywh=annotation["bbox"],
        source_class=source_class,
        attrs={"attr_is_group_of": bool(annotation["iscrowd"])},
    )


def build_raw_box(
    *,
    xywh: Sequence[int | float],
    source_class: str,
    attrs: Mapping[str, int | bool | None] | None = None,
) -> RawBox:
    x, y, width, height = (float(value) for value in xywh)
    return RawBox(
        source_class=source_class,
        xmin=x,
        ymin=y,
        xmax=x + width,
        ymax=y + height,
        attrs=attrs or {},
    )


def _read_items(*, path: UPath, prefix: str) -> list[CocoObject]:
    return list(_iter_items(path=path, prefix=prefix))


def _iter_items(*, path: UPath, prefix: str) -> Iterator[CocoObject]:
    with path.open(mode="rb") as handle:
        yield from ijson.items(handle, prefix, use_float=True)


class RawImageObjectDetectionInput(ObjectDetectionInput):
    """The labelformat view of raw images, which keeps only the single instance boxes.

    labelformat carries no per box flag, so this view drops a group box. Silver reads
    the raw images, flags included.
    """

    def __init__(self, *, read_images: Callable[[], Iterable[RawImage]]) -> None:
        self.read_images = read_images

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError

    def get_categories(self) -> Iterable[Category]:
        names = sorted(
            {box.source_class for image in self.read_images() for box in image.boxes}
        )
        return [Category(id=index, name=name) for index, name in enumerate(names)]

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        categories = {category.name: category for category in self.get_categories()}
        for index, image in enumerate(self.read_images()):
            yield ImageObjectDetection(
                image=Image(
                    id=index,
                    filename=image.file_name,
                    width=image.width,
                    height=image.height,
                ),
                objects=[
                    SingleObjectDetection(
                        category=categories[box.source_class],
                        box=BoundingBox.from_format(
                            bbox=[box.xmin, box.ymin, box.xmax, box.ymax],
                            format=BoundingBoxFormat.XYXY,
                        ),
                    )
                    for box in image.boxes
                    if not box.attrs.get("attr_is_group_of")
                ],
            )
