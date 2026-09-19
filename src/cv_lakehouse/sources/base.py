#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The contract that every bronze source implements."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from labelformat.model.object_detection import ObjectDetectionInput
from PIL import Image as PILImage

from cv_lakehouse.class_registry import CanonicalClass, sha256_fingerprint
from cv_lakehouse.sources.published_files import PublishedFile

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset is, and how its categories map onto the class registry.

    In category_map the key is the name the source gives a category. The value is a
    canonical class name. A source category that the map omits falls to default_class,
    so no box is ever lost to a gap in the map.

    Point default_class at `other` on a source that boxes more than the class registry
    names.
    Silver then keeps the box and loses the label, and its consumer decides what the
    class is worth.
    """

    name: str
    homepage: str
    license: str
    commercial_use: bool
    splits: tuple[str, ...]
    category_map: Mapping[str, str]
    default_class: str
    notes: str = ""

    def __post_init__(self) -> None:
        targets = set(self.category_map.values()) | {self.default_class}
        unknown = sorted(targets - CanonicalClass.all_class_names())
        if unknown:
            raise ValueError(f"{self.name} maps onto unknown classes {unknown}")

    @property
    def category_map_sha256_fingerprint(self) -> str:
        """A map change marks this dataset's silver asset stale."""
        pairs = sorted(f"{key}={value}" for key, value in self.category_map.items())
        return sha256_fingerprint([self.name, self.default_class, *pairs])


@dataclass(frozen=True)
class RawBox:
    """One box as its source publishes it, before any remap or clip.

    Coordinates are XYXY pixels, matching labelformat's `BoundingBox`. `attrs` holds the
    per box columns of `silver_schema.ATTRIBUTE_COLUMNS` that this source knows about. A
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
class BronzeSource(Protocol):
    """List the files a dataset publishes, describe its layout, and read its labels.

    Bronze downloads `published_files` and nothing else, so a source holds no download
    code.
    """

    spec: DatasetSpec
    published_files: tuple[PublishedFile, ...]

    def image_root(self, bronze_dir: Path, split: str) -> Path: ...

    def open_labelformat_reader(
        self, bronze_dir: Path, split: str
    ) -> ObjectDetectionInput: ...


@runtime_checkable
class BoxAttributeSource(Protocol):
    """A source that publishes more per box than labelformat's model can carry.

    labelformat's `SingleObjectDetection` holds a category, a box and a confidence, so a
    source with per box attributes needs a second way out. Inherit this and silver
    reads it instead of `read`. A source without attributes inherits nothing extra.
    """

    def read_raw_images(self, bronze_dir: Path, split: str) -> Iterable[RawImage]: ...


def read_raw_images(
    source: BronzeSource, bronze_dir: Path, split: str
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


def read_image_size(path: Path) -> tuple[int, int]:
    """Read the width and the height from the header, without decoding the pixels."""
    with PILImage.open(path) as image:
        return image.width, image.height


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            yield path
