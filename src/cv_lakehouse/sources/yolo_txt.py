#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""A reader for a dataset that ships YOLO text labels beside its images.

Several redaction datasets use this layout. The label file holds one box per line as
`class cx cy w h`, normalised to the image. Boxes come back in pixel coordinates.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections.abc import Iterable, Sequence
from pathlib import Path

from labelformat.model.bounding_box import BoundingBox, BoundingBoxFormat
from labelformat.model.category import Category
from labelformat.model.image import Image
from labelformat.model.object_detection import (
    ImageObjectDetection,
    ObjectDetectionInput,
    SingleObjectDetection,
)

from cv_lakehouse.sources.base import iter_image_paths, read_image_size


class YoloTxtObjectDetectionInput(ObjectDetectionInput):
    """Pair every image under image_root with a label file under label_root."""

    def __init__(
        self,
        image_root: Path,
        label_root: Path,
        class_names: Sequence[str],
        require_label_file: bool = True,
        subdirs: Sequence[str] | None = None,
    ) -> None:
        self.image_root = image_root
        self.label_root = label_root
        self._categories = [
            Category(id=index, name=name) for index, name in enumerate(class_names)
        ]
        self._require_label_file = require_label_file
        self._paths = sorted(iter_image_paths(image_root))
        if subdirs is not None:
            allowed = set(subdirs)
            self._paths = [
                path
                for path in self._paths
                if path.relative_to(image_root).parts[0] in allowed
            ]

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError

    def get_categories(self) -> Iterable[Category]:
        return iter(self._categories)

    def get_images(self) -> Iterable[Image]:
        for index, path in enumerate(self._paths):
            width, height = read_image_size(path)
            yield Image(
                id=index,
                filename=path.relative_to(self.image_root).as_posix(),
                width=width,
                height=height,
            )

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        category_by_id = {category.id: category for category in self._categories}
        for image in self.get_images():
            label_file = (self.label_root / image.filename).with_suffix(".txt")
            if not label_file.exists():
                if self._require_label_file:
                    raise FileNotFoundError(f"No label file for {image.filename}")
                yield ImageObjectDetection(image=image, objects=[])
                continue
            yield ImageObjectDetection(
                image=image,
                objects=list(
                    _read_detections_from_label_file(
                        label_file=label_file,
                        image=image,
                        category_by_id=category_by_id,
                    )
                ),
            )


def _read_detections_from_label_file(
    label_file: Path, image: Image, category_by_id: dict[int, Category]
) -> Iterable[SingleObjectDetection]:
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id not in category_by_id:
            continue
        cx, cy, width, height = (float(value) for value in parts[1:5])
        yield SingleObjectDetection(
            category=category_by_id[class_id],
            box=BoundingBox.from_format(
                bbox=[
                    cx * image.width,
                    cy * image.height,
                    width * image.width,
                    height * image.height,
                ],
                format=BoundingBoxFormat.CXCYWH,
            ),
        )
