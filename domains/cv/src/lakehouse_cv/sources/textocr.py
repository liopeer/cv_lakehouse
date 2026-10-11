#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""TextOCR v0.1: every word on about 25k Open Images images.

The image zips are the TextVQA zips. Their files are byte for byte the files of Open
Images, so a file name here is the Open Images id. The annotations are CC-BY-4.0, and
the images CC-BY-2.0.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import Any

import ijson
from labelformat.model.object_detection import ObjectDetectionInput
from upath import UPath

from lakehouse_core.published_files import (
    Checksum,
    ChecksumKind,
    PublishedFile,
)
from lakehouse_cv.contract.dataset_spec import DatasetSpec
from lakehouse_cv.contract.split_roles import SplitRole
from lakehouse_cv.sources.base import BoxAttributeSource, BronzeSource, RawImage
from lakehouse_cv.sources.coco_json import (
    RawImageObjectDetectionInput,
    build_raw_box,
)

DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/textvqa/{path}"

# The test file lists images only, so silver never reads it.
SPLITS = ("train", "val")
WORD = "word"
# TextOCR writes "." for a word that no one can read.
ILLEGIBLE_TEXT = "."


def _published_file(*, path: str, size: int, sha256: str) -> PublishedFile:
    return PublishedFile(
        path=path.rsplit("/", maxsplit=1)[-1],
        url=DOWNLOAD_URL.format(path=path),
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


# The ETags of these files sum parts of 16 MiB, and `S3_ETAG` sums parts of 8 MiB.
PUBLISHED_FILES = (
    _published_file(
        path="data/textocr/TextOCR_0.1_train.json",
        size=279887401,
        sha256="5bac8303c5999a3ddf630efde8ac9743b4461be6022b193a1075df90fbd2ce0c",
    ),
    _published_file(
        path="data/textocr/TextOCR_0.1_val.json",
        size=40028022,
        sha256="0d04c1c9f5ba99e90be5b0df9ab34f6237b3fbc55443f263133d5c49b6435918",
    ),
    _published_file(
        path="data/textocr/TextOCR_0.1_test.json",
        size=436594,
        sha256="b16eea55c3cc2f9089df89c5167eec37f7522fdfc3f1b76b41e031748ac8ad5b",
    ),
    _published_file(
        path="images/train_val_images.zip",
        size=7072297970,
        sha256="ecf35005640d0708eae185aab1c0a10f89b2db7420b29185a1ed92a8f4290498",
    ),
    _published_file(
        path="images/test_images.zip",
        size=970296721,
        sha256="1276b908994c444c46484fb21e9e15fcda1be9c675f6ad727489e52eea68cbcd",
    ),
)

SPEC = DatasetSpec(
    name="textocr",
    homepage="https://textvqa.org/textocr/",
    license="CC-BY-4.0 annotations, CC-BY-2.0 images",
    commercial_use=True,
    splits=SPLITS,
    split_roles={"train": SplitRole.TRAIN, "val": SplitRole.VAL},
    category_map={WORD: "text"},
    default_class="other",
    notes="Every word on Open Images photos, also a word that no one can read.",
)


class TextOcrSource(BronzeSource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES

    def image_root(self, bronze_dir: UPath, split: str) -> UPath:
        _check_split(split)
        return bronze_dir / "train_val_images" / "train_images"

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput:
        return RawImageObjectDetectionInput(
            read_images=partial(
                self.read_raw_images, bronze_dir=bronze_dir, split=split
            )
        )

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterator[RawImage]:
        """Read every image of the split and every word on it.

        A file name in the JSON is `train/<id>.jpg` for both splits, but the zip holds
        the images in `train_images/`, so a file name here is `<id>.jpg`.
        """
        annotation_file = _annotation_file(bronze_dir=bronze_dir, split=split)
        # 1M words in train, so each keeps only its box and its flag.
        words_by_image_id: dict[str, list[tuple[list[float], bool]]] = {}
        for _, word in _iter_values(path=annotation_file, prefix="anns"):
            words_by_image_id.setdefault(word["image_id"], []).append(
                (word["bbox"], word["utf8_string"] == ILLEGIBLE_TEXT)
            )
        for image_id, image in _iter_values(path=annotation_file, prefix="imgs"):
            yield RawImage(
                file_name=f"{image_id}.jpg",
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=tuple(
                    build_raw_box(
                        xywh=xywh,
                        source_class=WORD,
                        attrs={"attr_is_illegible": is_illegible},
                    )
                    for xywh, is_illegible in words_by_image_id.get(image_id, [])
                ),
            )


def _iter_values(*, path: UPath, prefix: str) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(mode="rb") as handle:
        yield from ijson.kvitems(handle, prefix, use_float=True)


def _annotation_file(bronze_dir: UPath, split: str) -> UPath:
    _check_split(split)
    return bronze_dir / f"TextOCR_0.1_{split}.json"


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown TextOCR split: {split}")
