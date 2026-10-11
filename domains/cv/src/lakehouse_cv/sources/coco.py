#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""COCO 2017, every file that its download page links to for the 2017 release.

The annotations are CC-BY-4.0. The images are Flickr photos, each under its own licence,
and most of them forbid commercial use. The `license` field of each image names it.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

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
    build_crowd_flagged_box,
    read_coco_json,
)

# The files are on S3, and the ETag of each one is its checksum.
DOWNLOAD_URL = "http://images.cocodataset.org/{folder}/{name}"

# test2017 and unlabeled2017 ship no public boxes, so silver never reads them. Bronze
# holds their images all the same.
SPLITS = ("train", "val")


def _published_file(*, folder: str, name: str, size: int, etag: str) -> PublishedFile:
    return PublishedFile(
        path=name,
        url=DOWNLOAD_URL.format(folder=folder, name=name),
        size=size,
        checksum=Checksum(kind=ChecksumKind.S3_ETAG, value=etag),
    )


PUBLISHED_FILES = (
    _published_file(
        folder="zips",
        name="train2017.zip",
        size=19336861798,
        etag="62ff7d7fbcc7e0c0604cbb0f9047ce77-2306",
    ),
    _published_file(
        folder="zips",
        name="val2017.zip",
        size=815585330,
        etag="d366be60d3dc737327160d62453e3973-98",
    ),
    _published_file(
        folder="zips",
        name="test2017.zip",
        size=6646970404,
        etag="1d4d2301bdee781394c21718428764d2-793",
    ),
    _published_file(
        folder="zips",
        name="unlabeled2017.zip",
        size=20126613414,
        etag="9a34e756dff70baa95970d77d1f9ccb7-2400",
    ),
    _published_file(
        folder="annotations",
        name="annotations_trainval2017.zip",
        size=252907541,
        etag="f4bbac642086de4f52a3fdda2de5fa2c",
    ),
    _published_file(
        folder="annotations",
        name="stuff_annotations_trainval2017.zip",
        size=1148688564,
        etag="2a27c15a2dfcbd2e1c9276dc23cac101",
    ),
    _published_file(
        folder="annotations",
        name="panoptic_annotations_trainval2017.zip",
        size=860725834,
        etag="4170db65fc022c9c296af880dbca6055",
    ),
    _published_file(
        folder="annotations",
        name="image_info_test2017.zip",
        size=1144034,
        etag="85da7065e5e600ebfee8af1edb634eb5",
    ),
    _published_file(
        folder="annotations",
        name="image_info_unlabeled2017.zip",
        size=4902013,
        etag="ede38355d5c3e5251bb7f8b68e2c068f",
    ),
)

SPEC = DatasetSpec(
    name="coco",
    homepage="https://cocodataset.org/",
    license="CC-BY-4.0 annotations, Flickr images under a licence per image",
    commercial_use=False,
    splits=SPLITS,
    split_roles={"train": SplitRole.TRAIN, "val": SplitRole.VAL},
    category_map={"person": "person"},
    default_class="other",
    notes="Everyday scenes. Many persons, at every scale, in crowds and in context.",
)


class CocoSource(BronzeSource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES

    def image_root(self, bronze_dir: UPath, split: str) -> UPath:
        _check_split(split)
        return bronze_dir / f"{split}2017"

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput:
        return RawImageObjectDetectionInput(
            read_images=partial(
                self.read_raw_images, bronze_dir=bronze_dir, split=split
            )
        )

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterator[RawImage]:
        """Read every image and every box, with `iscrowd` as the group flag."""
        coco_json = read_coco_json(
            _annotation_file(bronze_dir=bronze_dir, split=split),
            annotation_fields=("category_id", "bbox", "iscrowd"),
        )
        for image, image_annotations in coco_json.iter_images_with_annotations():
            yield RawImage(
                file_name=image["file_name"],
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=tuple(
                    build_crowd_flagged_box(
                        annotation=annotation,
                        source_class=coco_json.class_name_by_id[
                            annotation["category_id"]
                        ],
                    )
                    for annotation in image_annotations
                ),
            )


def _annotation_file(bronze_dir: UPath, split: str) -> UPath:
    _check_split(split)
    return (
        bronze_dir
        / "annotations_trainval2017"
        / "annotations"
        / f"instances_{split}2017.json"
    )


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown COCO split: {split}")
