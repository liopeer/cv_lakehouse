#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""WIDER FACE: the standard face detection benchmark.

The dataset is event photography, not street scenes, so it teaches recall on small and
occluded faces rather than the road driving domain. Its licence forbids commercial use.
"""

from __future__ import annotations

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
from cv_lakehouse.sources.published_files import (
    Checksum,
    ChecksumKind,
    PublishedFile,
)

# The official page links its image archives to this Hugging Face repository, and
# hosts every other file itself.
HF_RESOLVE_URL = (
    "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/"
    "db171f1b7fedf4d3453e81297ff02f9915356d19/data/{name}"
)
SITE_URL = "http://shuoyang1213.me/WIDERFACE/support/{folder}/{name}"
FACE = Category(id=0, name="face")

# A box line is `x1 y1 w h blur expression illumination invalid occlusion pose`. The
# four coordinates come first; these are the grades that follow, by position.
BOX_COORDINATE_COUNT = 4
ATTRIBUTE_LINE_POSITIONS = {
    "attr_blur": 4,
    "attr_expression": 5,
    "attr_illumination": 6,
    "attr_invalid": 7,
    "attr_occlusion": 8,
    "attr_pose": 9,
}
INVALID_LINE_POSITION = ATTRIBUTE_LINE_POSITIONS["attr_invalid"]

# The test split ships no public ground truth, so silver never reads it. Bronze holds
# its images all the same.
SPLITS = ("train", "val")


def _published_file(*, name: str, url: str, size: int, sha256: str) -> PublishedFile:
    return PublishedFile(
        path=name,
        url=url,
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


PUBLISHED_FILES = (
    _published_file(
        name="WIDER_train.zip",
        url=HF_RESOLVE_URL.format(name="WIDER_train.zip"),
        size=1465602149,
        sha256="e23b76129c825cafae8be944f65310b2e1ba1c76885afe732f179c41e5ed6d59",
    ),
    _published_file(
        name="WIDER_val.zip",
        url=HF_RESOLVE_URL.format(name="WIDER_val.zip"),
        size=362752168,
        sha256="f9efbd09f28c5d2d884be8c0eaef3967158c866a593fc36ab0413e4b2a58a17a",
    ),
    _published_file(
        name="WIDER_test.zip",
        url=HF_RESOLVE_URL.format(name="WIDER_test.zip"),
        size=1844140520,
        sha256="3b0313e11ea292ec58894b47ac4c0503b230e12540330845d70a7798241f88d3",
    ),
    _published_file(
        name="wider_face_split.zip",
        url=SITE_URL.format(folder="bbx_annotation", name="wider_face_split.zip"),
        size=3591642,
        sha256="c7561e4f5e7a118c249e0a5c5c902b0de90bbf120d7da9fa28d99041f68a8a5c",
    ),
    _published_file(
        name="eval_tools.zip",
        url=SITE_URL.format(folder="eval_script", name="eval_tools.zip"),
        size=8447003,
        sha256="1cf49c8243fa8a1632efe7f6aa09fd7a3994ac838c9008bdbc9dd1d547bb234a",
    ),
    _published_file(
        name="Submission_example.zip",
        url=SITE_URL.format(folder="example", name="Submission_example.zip"),
        size=624,
        sha256="7b62f3466a53b29efcbe86c7bc38422f9290f3433a421309d47d8745ea824df7",
    ),
)

SPEC = DatasetSpec(
    name="wider_face",
    homepage="http://shuoyang1213.me/WIDERFACE/",
    license="CC-BY-NC-ND-4.0",
    commercial_use=False,
    splits=SPLITS,
    category_map={FACE.name: "face"},
    default_class="other",
    notes="Event photography. Strong on small and occluded faces, weak on road scenes.",
)


class WiderFaceSource(BronzeSource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES

    def image_root(self, bronze_dir: Path, split: str) -> Path:
        _check_split(split)
        return bronze_dir / f"WIDER_{split}" / "images"

    def open_labelformat_reader(
        self, bronze_dir: Path, split: str
    ) -> ObjectDetectionInput:
        """The labelformat view, which drops the boxes the annotators rejected.

        labelformat carries no per box flag, so a reader of this cannot tell a
        rejected region from a face. `read_raw_images` keeps every box and reports the
        flag instead. Silver reads raw images.
        """
        _check_split(split)
        return WiderFaceObjectDetectionInput(
            image_root=self.image_root(bronze_dir=bronze_dir, split=split),
            annotation_file=_annotation_file(bronze_dir=bronze_dir, split=split),
        )

    def read_raw_images(self, bronze_dir: Path, split: str) -> Iterator[RawImage]:
        """Read every box, with the five grades and the reject flag alongside it.

        A rejected region is face-like but unusable as a positive. Dropping it labels
        it background, which teaches a detector not to fire there. Silver keeps it with
        the flag, and its consumer decides.
        """
        _check_split(split)
        image_root = self.image_root(bronze_dir=bronze_dir, split=split)
        annotation_file = _annotation_file(bronze_dir=bronze_dir, split=split)
        for filename, rows in _read_annotation_blocks(annotation_file):
            width, height = read_image_size(image_root / filename)
            yield RawImage(
                file_name=filename,
                width=width,
                height=height,
                boxes=tuple(_build_raw_box(row) for row in rows),
            )


class WiderFaceObjectDetectionInput(ObjectDetectionInput):
    """Parse a wider_face_*_bbx_gt.txt file.

    The file repeats three blocks: a filename, a box count, then that many box lines of
    `x1 y1 w h blur expression illumination invalid occlusion pose`.
    """

    def __init__(self, image_root: Path, annotation_file: Path) -> None:
        self.image_root = image_root
        self.annotation_file = annotation_file

    @staticmethod
    def add_cli_arguments(parser: ArgumentParser) -> None:
        raise NotImplementedError

    def get_categories(self) -> Iterable[Category]:
        return iter([FACE])

    def get_images(self) -> Iterable[Image]:
        for label in self.get_labels():
            yield label.image

    def get_labels(self) -> Iterable[ImageObjectDetection]:
        for index, (filename, rows) in enumerate(
            _read_annotation_blocks(self.annotation_file)
        ):
            width, height = read_image_size(self.image_root / filename)
            image = Image(id=index, filename=filename, width=width, height=height)
            yield ImageObjectDetection(
                image=image,
                objects=[
                    SingleObjectDetection(
                        category=FACE,
                        box=BoundingBox.from_format(
                            bbox=[float(value) for value in row[:4]],
                            format=BoundingBoxFormat.XYWH,
                        ),
                    )
                    for row in rows
                    if row[INVALID_LINE_POSITION] == 0
                ],
            )


def _build_raw_box(row: list[int]) -> RawBox:
    x, y, width, height = (float(value) for value in row[:BOX_COORDINATE_COUNT])
    return RawBox(
        source_class=FACE.name,
        xmin=x,
        ymin=y,
        xmax=x + width,
        ymax=y + height,
        attrs={
            name: bool(row[index]) if name == "attr_invalid" else row[index]
            for name, index in ATTRIBUTE_LINE_POSITIONS.items()
            if index < len(row)
        },
    )


def _read_annotation_blocks(
    annotation_file: Path,
) -> Iterable[tuple[str, list[list[int]]]]:
    lines = annotation_file.read_text().splitlines()
    position = 0
    while position < len(lines):
        filename = lines[position].strip()
        position += 1
        if not filename:
            continue
        count = int(lines[position].strip())
        position += 1
        # A count of zero is still followed by one all zero line. Skip it.
        taken = max(count, 1)
        rows = [
            [int(float(value)) for value in lines[position + offset].split()]
            for offset in range(taken)
        ]
        position += taken
        yield filename, rows if count else []


def _annotation_file(bronze_dir: Path, split: str) -> Path:
    return bronze_dir / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown WIDER FACE split: {split}")
