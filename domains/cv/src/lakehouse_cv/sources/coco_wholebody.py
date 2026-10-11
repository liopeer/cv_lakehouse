#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""COCO-WholeBody v1.0: face, hand and body boxes for the persons of COCO 2017.

COCO-WholeBody publishes annotations only, so silver reads the images in the bronze of
COCO. A person box is the box of COCO. The face and hand boxes are new.

A face box exists only where `face_valid` is set. A face under about 10 px has none, and
neither has some larger face, so the faces are not labelled completely.

The annotations are CC-BY-NC-4.0 and for research only. The images keep the Flickr
licences of COCO.
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
from lakehouse_cv.sources.base import (
    AnnotationOnlySource,
    BoxAttributeSource,
    RawBox,
    RawImage,
)
from lakehouse_cv.sources.coco import SPEC as COCO_SPEC
from lakehouse_cv.sources.coco_json import (
    CocoObject,
    RawImageObjectDetectionInput,
    build_crowd_flagged_box,
    build_raw_box,
    coco_image_path,
    read_coco_json,
)

# The README links each file on Google Drive. This URL serves the file itself.
DRIVE_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

SPLITS = ("train", "val")

# Each part box, with the flag that says it is annotated, and its source class.
PART_BOXES = (
    ("face_box", "face_valid", "face"),
    ("lefthand_box", "lefthand_valid", "left_hand"),
    ("righthand_box", "righthand_valid", "right_hand"),
)
ANNOTATION_FIELDS = (
    "category_id",
    "bbox",
    "iscrowd",
    *(field for part_box in PART_BOXES for field in part_box[:2]),
)


def _published_file(
    *, name: str, file_id: str, size: int, sha256: str
) -> PublishedFile:
    return PublishedFile(
        path=name,
        url=DRIVE_URL.format(file_id=file_id),
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


PUBLISHED_FILES = (
    _published_file(
        name="coco_wholebody_train_v1.0.json",
        file_id="1thErEToRbmM9uLNi1JXXfOsaS5VK2FXf",
        size=3158585395,
        sha256="9e9d2bf973319f9fd989b7422281ee7de027705ae149d69bc164ff43e5e5206e",
    ),
    _published_file(
        name="coco_wholebody_val_v1.0.json",
        file_id="1N6VgwKnj8DeyGXCvp1eYgNbRmw6jdfrb",
        size=132970317,
        sha256="f8272e9c12f3a42457033ebc75da1167546edf1be5e2ffaf586d6ee97541ff6e",
    ),
)

SPEC = DatasetSpec(
    name="coco_wholebody",
    homepage="https://github.com/jin-s13/COCO-WholeBody",
    license="CC-BY-NC-4.0 annotations, COCO images under a Flickr licence per image",
    commercial_use=False,
    splits=SPLITS,
    split_roles={"train": SplitRole.TRAIN, "val": SplitRole.VAL},
    category_map={"person": "person", "face": "face"},
    default_class="other",
    notes=(
        "Face and hand boxes for the persons of COCO. Silver reads the images in "
        "bronze/coco.\n\n"
        "Google Drive hosts the files. If Drive refuses the download, download both "
        "files from the links in the README of the homepage. On S3, upload each file "
        "to bronze/coco_wholebody/ under its published name. Beside it, upload an "
        "empty file of the same name with the suffix `.unverified`. On a local disk, "
        "put each file there with the suffix `.partial`. Then materialize the asset. "
        "It verifies the checksum of each file."
    ),
)


class CocoWholeBodySource(AnnotationOnlySource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES
    image_source_name = COCO_SPEC.name

    def annotation_file(self, bronze_dir: UPath, split: str) -> UPath:
        _check_split(split)
        return bronze_dir / f"coco_wholebody_{split}_v1.0.json"

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput:
        return RawImageObjectDetectionInput(
            read_images=partial(
                self.read_raw_images, bronze_dir=bronze_dir, split=split
            )
        )

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterator[RawImage]:
        """Read each person box, and each part box that is annotated."""
        coco_json = read_coco_json(
            self.annotation_file(bronze_dir=bronze_dir, split=split),
            annotation_fields=ANNOTATION_FIELDS,
        )
        for image, image_annotations in coco_json.iter_images_with_annotations():
            yield RawImage(
                file_name=coco_image_path(image),
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=tuple(
                    box
                    for annotation in image_annotations
                    for box in _build_person_boxes(
                        annotation=annotation,
                        person_class=coco_json.class_name_by_id[
                            annotation["category_id"]
                        ],
                    )
                ),
            )


def _build_person_boxes(
    *, annotation: CocoObject, person_class: str
) -> Iterator[RawBox]:
    yield build_crowd_flagged_box(annotation=annotation, source_class=person_class)
    for box_field, valid_field, source_class in PART_BOXES:
        if annotation[valid_field]:
            yield build_raw_box(xywh=annotation[box_field], source_class=source_class)


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown COCO-WholeBody split: {split}")
