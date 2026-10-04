#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The protocol that every CV bronze source implements, and its readers."""

from __future__ import annotations

import posixpath
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from labelformat.model.category import Category
from labelformat.model.object_detection import ObjectDetectionInput
from PIL import Image as PILImage
from upath import UPath

from lakehouse_core.bronze_manifest import BronzeManifest
from lakehouse_core.published_files import PublishedFile
from lakehouse_core.published_source import Publication, PublishedSource
from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.dataset_spec import DatasetSpec
from lakehouse_cv.contract.manifests import CvBronzeManifest

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

LABELFORMAT_CATEGORIES: list[Category] = [
    Category(id=member.value, name=member.class_name) for member in CanonicalClass
]


@dataclass(frozen=True)
class RawBox:
    """One box as its source publishes it, before any remap or clip.

    Coordinates are XYXY pixels, matching labelformat's `BoundingBox`. `attrs` holds the
    per box columns of `silver_tables.ATTRIBUTE_COLUMNS` that this source knows about. A
    column the dict omits lands as null.
    """

    source_class: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: float | None = None
    attrs: Mapping[str, int | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RawImage:
    """One image, and every box its source publishes for it, flagged ones included."""

    file_name: str
    width: int
    height: int
    boxes: tuple[RawBox, ...] = ()


@runtime_checkable
class BronzeSource(PublishedSource[CvBronzeManifest], Protocol):
    """List the files a dataset publishes, describe its layout, and read its labels.

    Bronze downloads `published_files` and nothing else, so a source holds no download
    code.
    """

    spec: DatasetSpec
    published_files: tuple[PublishedFile, ...]

    def image_root(self, bronze_dir: UPath, split: str) -> UPath: ...

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput: ...

    @property
    def publication(self) -> Publication:
        return Publication(
            name=self.spec.name,
            homepage=self.spec.homepage,
            license=self.spec.license,
            commercial_use=self.spec.commercial_use,
            description=self.spec.notes,
            published_files=self.published_files,
        )

    def describe_bronze_copy(
        self, *, bronze_dir: UPath, base: BronzeManifest
    ) -> CvBronzeManifest:
        splits = detect_materialized_splits(source=self, bronze_dir=bronze_dir)
        if not splits:
            raise RuntimeError(
                f"No split materialized for {base.dataset} in {bronze_dir}"
            )
        return CvBronzeManifest(
            **base.model_dump(),
            splits=list(splits),
            # Under the location of bronze, so the roots move with the lake.
            image_roots={
                split: posixpath.join(
                    base.path,
                    self.image_root(bronze_dir=bronze_dir, split=split)
                    .relative_to(bronze_dir)
                    .as_posix(),
                )
                for split in splits
            },
        )


def detect_materialized_splits(
    source: BronzeSource, bronze_dir: UPath
) -> tuple[str, ...]:
    """Report the splits that are really on disk, not the splits the source can make."""
    found = []
    for split in source.spec.splits:
        try:
            root = source.image_root(bronze_dir=bronze_dir, split=split)
        except FileNotFoundError:
            continue
        if root.is_dir():
            found.append(split)
    return tuple(found)


@runtime_checkable
class BoxAttributeSource(Protocol):
    """A source that publishes more per box than labelformat's model can carry.

    labelformat's `SingleObjectDetection` holds a category, a box and a confidence, so a
    source with per box attributes needs a second way out. Inherit this and silver
    reads it instead of `read`. A source without attributes inherits nothing extra.
    """

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterable[RawImage]: ...


def read_raw_images(
    source: BronzeSource, bronze_dir: UPath, split: str
) -> Iterable[RawImage]:
    """Read one split as raw images, preferring the attribute carrying path.

    A source that implements `read_raw_images` decides for itself what reaches
    silver, flags included. Every other source is adapted from its labelformat
    reader, with no attrs.
    """
    if isinstance(source, BoxAttributeSource):
        return source.read_raw_images(bronze_dir=bronze_dir, split=split)
    return _read_raw_images_from_labelformat(
        source.open_labelformat_reader(bronze_dir=bronze_dir, split=split)
    )


def _read_raw_images_from_labelformat(
    reader: ObjectDetectionInput,
) -> Iterator[RawImage]:
    for label in reader.get_labels():
        image = label.image
        yield RawImage(
            file_name=str(image.filename),
            width=image.width,
            height=image.height,
            boxes=tuple(
                RawBox(
                    source_class=obj.category.name,
                    xmin=obj.box.xmin,
                    ymin=obj.box.ymin,
                    xmax=obj.box.xmax,
                    ymax=obj.box.ymax,
                    confidence=obj.confidence,
                )
                for obj in label.objects
            ),
        )


def read_image_size(path: UPath) -> tuple[int, int]:
    """Read the width and the height from the header, without decoding the pixels."""
    with path.open(mode="rb") as handle, PILImage.open(handle) as image:
        return image.width, image.height


def iter_image_paths(root: UPath) -> Iterable[UPath]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path
