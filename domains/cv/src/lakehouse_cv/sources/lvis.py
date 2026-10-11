#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""LVIS v1: 1203 categories on the images of COCO 2017.

LVIS publishes annotations only, so silver reads the images in the bronze of COCO. Its
categories include `license_plate` and `person`, and signs, screens and ID cards.

The annotations are CC-BY-4.0, and the images keep the Flickr licences of COCO.
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
    RawImage,
)
from lakehouse_cv.sources.coco import SPEC as COCO_SPEC
from lakehouse_cv.sources.coco_json import (
    RawImageObjectDetectionInput,
    build_raw_box,
    coco_image_path,
    read_coco_json,
)

DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/LVIS/{name}"

# The two test files list images only, so silver never reads them.
SPLITS = ("train", "val")


def _published_file(*, name: str, size: int, sha256: str) -> PublishedFile:
    return PublishedFile(
        path=name,
        url=DOWNLOAD_URL.format(name=name),
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


# The ETags of these files sum parts of 16 MiB, and `S3_ETAG` sums parts of 8 MiB.
PUBLISHED_FILES = (
    _published_file(
        name="lvis_v1_train.json.zip",
        size=350264821,
        sha256="334a4caa374030a7817cf050364525e910f7960f9b6968cef47cffbf3893f8ba",
    ),
    _published_file(
        name="lvis_v1_val.json.zip",
        size=64026968,
        sha256="5cae9a3c79aadb667550c2b5dcf7f4d86e059a41ec91ef690225b667e28e9ba5",
    ),
    _published_file(
        name="lvis_v1_image_info_test_dev.json.zip",
        size=384629,
        sha256="bbe4880cef768f3c14ea903d31dcce9db24916cf9e9f1f76d1a022120f47254b",
    ),
    _published_file(
        name="lvis_v1_image_info_test_challenge.json.zip",
        size=384573,
        sha256="8339a1d15c58845790e747960dfda837da41df3b34a2a437dac1cf783854a622",
    ),
)

SPEC = DatasetSpec(
    name="lvis",
    homepage="https://www.lvisdataset.org/",
    license="CC-BY-4.0 annotations, COCO images under a Flickr licence per image",
    commercial_use=False,
    splits=SPLITS,
    split_roles={"train": SplitRole.TRAIN, "val": SplitRole.VAL},
    category_map={"license_plate": "license_plate", "person": "person"},
    default_class="other",
    notes=(
        "Licence plates, persons, signs and screens on the COCO images. Silver reads "
        "the images in bronze/coco."
    ),
)


class LvisSource(AnnotationOnlySource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES
    image_source_name = COCO_SPEC.name

    def annotation_file(self, bronze_dir: UPath, split: str) -> UPath:
        _check_split(split)
        return bronze_dir / f"lvis_v1_{split}.json" / f"lvis_v1_{split}.json"

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput:
        return RawImageObjectDetectionInput(
            read_images=partial(
                self.read_raw_images, bronze_dir=bronze_dir, split=split
            )
        )

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterator[RawImage]:
        """Read every image and every box. LVIS has no crowd box.

        An image of the val split can be in train2017 of COCO, so each file name keeps
        its COCO folder.
        """
        coco_json = read_coco_json(
            self.annotation_file(bronze_dir=bronze_dir, split=split),
            annotation_fields=("category_id", "bbox"),
        )
        for image, image_annotations in coco_json.iter_images_with_annotations():
            yield RawImage(
                file_name=coco_image_path(image),
                width=int(image["width"]),
                height=int(image["height"]),
                boxes=tuple(
                    build_raw_box(
                        xywh=annotation["bbox"],
                        source_class=coco_json.class_name_by_id[
                            annotation["category_id"]
                        ],
                    )
                    for annotation in image_annotations
                ),
            )


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown LVIS split: {split}")
