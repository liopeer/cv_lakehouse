#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""PP4AV: faces and licence plates on European road driving footage.

The dataset is an evaluation benchmark. Its licence forbids commercial use, so it
belongs in a test split only.
"""

from __future__ import annotations

from pathlib import Path

from labelformat.model.object_detection import ObjectDetectionInput

from cv_lakehouse.sources.base import BronzeSource, DatasetSpec
from cv_lakehouse.sources.published_files import (
    Checksum,
    ChecksumKind,
    PublishedFile,
)
from cv_lakehouse.sources.yolo_txt import YoloTxtObjectDetectionInput

# The Hugging Face repository is the official distribution, so bronze holds all of
# it, at this commit.
RESOLVE_URL = (
    "https://huggingface.co/datasets/khaclinh/pp4av/resolve/"
    "bcb26e69554574d87cc8286ed42b028183d0fc55/{path}"
)
CLASS_NAMES = ("face", "license_plate")
CITIES = (
    "netherlands_day",
    "netherlands_night",
    "paris",
    "strasbourg",
    "stuttgart",
    "switzerland",
    "zurich",
)
SPLIT_SUBDIRS = {"test": CITIES, "fisheye": ("fisheye",)}

# The dataset card calls `annotations` the curated labels and `soiling_annotations`
# the raw ones, before filtering. Bronze holds both. The reader takes the curated.
LABEL_DIR = "data/annotations"
IMAGE_DIR = "data/images"


def _published_file(*, path: str, size: int, sha256: str) -> PublishedFile:
    return PublishedFile(
        path=path,
        url=RESOLVE_URL.format(path=path),
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


PUBLISHED_FILES = (
    _published_file(
        path=".gitattributes",
        size=2126,
        sha256="d6cf0a7f5c480678e9330d780fb911c414602b5869d669ad0a369f45a56356ca",
    ),
    _published_file(
        path="README.md",
        size=12897,
        sha256="4d7e37fc70c662b6450ecc4b93f452bea121458f6548ab832aeea1799523a96b",
    ),
    _published_file(
        path="pp4av.py",
        size=5682,
        sha256="3a3c670d21b2609d4427111d778dfec3438529f196a2287b778541809e357c54",
    ),
    _published_file(
        path="requirements.txt",
        size=45,
        sha256="75d5618340023c3bab178f665513bee627f116ad66ba3a13963d7fe798330588",
    ),
    _published_file(
        path="data/.keep",
        size=1,
        sha256="01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    ),
    _published_file(
        path="data/images.zip",
        size=4512838147,
        sha256="486d98266f56405840eda965c6d4e54008013e11456ed2e878abb9070cdc5286",
    ),
    _published_file(
        path="data/annotations.zip",
        size=982986,
        sha256="aca1fed529a6f00598db63a1c685c3915d33d420430bad252a2822bac4b6609c",
    ),
    _published_file(
        path="data/soiling_annotations.zip",
        size=1005826,
        sha256="10731fc73eead247d280ce895d342f71b934eefdec1b7ab3622c92d7a3ddf516",
    ),
)

SPEC = DatasetSpec(
    name="pp4av",
    homepage="https://github.com/khaclinh/pp4av",
    license="CC-BY-NC-ND-4.0",
    commercial_use=False,
    splits=("test", "fisheye"),
    category_map={CLASS_NAMES[0]: "face", CLASS_NAMES[1]: "license_plate"},
    default_class="other",
    notes="Evaluation benchmark on road driving footage. Never use it for training.",
)


class PP4AVSource(BronzeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES

    def image_root(self, bronze_dir: Path, split: str) -> Path:
        _check_split(split)
        return bronze_dir / IMAGE_DIR

    def open_labelformat_reader(
        self, bronze_dir: Path, split: str
    ) -> ObjectDetectionInput:
        _check_split(split)
        return YoloTxtObjectDetectionInput(
            image_root=self.image_root(bronze_dir=bronze_dir, split=split),
            label_root=bronze_dir / LABEL_DIR,
            class_names=CLASS_NAMES,
            subdirs=SPLIT_SUBDIRS[split],
        )


def _check_split(split: str) -> None:
    if split not in SPLIT_SUBDIRS:
        raise ValueError(f"Unknown PP4AV split: {split}")
