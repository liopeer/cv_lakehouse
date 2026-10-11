#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""HierText v1.0: every word, line and paragraph on about 11.6k Open Images images.

The GitHub repository holds the annotations, and CVDF hosts the images on S3. The images
are new exports of Open Images, with the long side up to 1600 px, so they are not the
files of the Open Images bronze. The annotations are CC-BY-SA-4.0, so a set derived from
them is CC-BY-SA-4.0 too. The images are CC-BY-2.0.
"""

from __future__ import annotations

import gzip
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
from lakehouse_cv.sources.base import BoxAttributeSource, BronzeSource, RawBox, RawImage
from lakehouse_cv.sources.coco_json import RawImageObjectDetectionInput

# The repository is the official distribution, so bronze holds all of it, at this
# commit.
REPO_URL = (
    "https://raw.githubusercontent.com/google-research-datasets/hiertext/"
    "70b6620b2b112597d8219e11eee9773a1403827c/{path}"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/ocr/{name}"

SPLITS = ("train", "validation", "test")
WORD = "word"


def _repo_file(*, path: str, size: int, sha256: str) -> PublishedFile:
    return PublishedFile(
        path=path,
        url=REPO_URL.format(path=path),
        size=size,
        checksum=Checksum(kind=ChecksumKind.SHA256, value=sha256),
    )


def _image_archive(*, name: str, size: int, etag: str) -> PublishedFile:
    return PublishedFile(
        path=name,
        url=IMAGE_URL.format(name=name),
        size=size,
        checksum=Checksum(kind=ChecksumKind.S3_ETAG, value=etag),
    )


PUBLISHED_FILES = (
    _repo_file(
        path="HierText_Visualization.ipynb",
        size=1331383,
        sha256="42a359aa2b5ae4dbc27ff4422a3163077dfb8113c385aabcbc717f265298f202",
    ),
    _repo_file(
        path="LICENSE",
        size=20139,
        sha256="71a43a169b52bb1b39510b65ec5b43b4bbb59196d449a835fe58b50af6a72742",
    ),
    _repo_file(
        path="README.md",
        size=11848,
        sha256="0726df8cbae6be2e587836ed7d8cc96377e3cc2d873ade2830cc3c33e3274e57",
    ),
    _repo_file(
        path="docs/images/ICDAR_2023_Competition_on_Hierarchical_Text_Detection_and_Recognition_result_and_ranking_.pdf",
        size=80248,
        sha256="fb7baa546f6dc41389d3186989f9b396f344acdd46fb67869310c0ce86304e66",
    ),
    _repo_file(
        path="docs/images/dataset.png",
        size=1973316,
        sha256="49492df550e77094b2cbccfe9931c7c2343f817edb14085b5fd89b7d52bcc03a",
    ),
    _repo_file(
        path="docs/images/vertices.png",
        size=75723,
        sha256="194de89c22f247cdc4b5fe57b5f7c364523bc01818dd995f3841dc69ded5696b",
    ),
    _repo_file(
        path="eval.py",
        size=12770,
        sha256="4667860c6f5b48ee794c7adc41176a5b420e41bbeca8e28702de6c895fb28859",
    ),
    _repo_file(
        path="evaluator/__init__.py",
        size=24,
        sha256="f64ae3797d575a116a019dd490bedb4443a7049db3810fb0b11e7134a7477255",
    ),
    _repo_file(
        path="evaluator/evaluator.py",
        size=9615,
        sha256="40e810a9a49d8f67c569b95bb15f4dd8f3c03a932f8e8a735e4897a2eb4bd6f5",
    ),
    _repo_file(
        path="evaluator/np_box_list.py",
        size=4781,
        sha256="5f21333ca5b4170e1083a4dae61f267c93bf698f9113a11ae7a697fe4fcafb1c",
    ),
    _repo_file(
        path="evaluator/np_box_list_ops.py",
        size=21001,
        sha256="1b59a90ebf6183c063999c2003a159bde01e576f3ee52ec50aae39eb408e3cc6",
    ),
    _repo_file(
        path="evaluator/np_box_mask_list.py",
        size=2645,
        sha256="adfab5d1e5e8fd9e38a23fcd7cd3f6a7e603b52636b34209eb7a17b5bd179f00",
    ),
    _repo_file(
        path="evaluator/np_box_mask_list_ops.py",
        size=15707,
        sha256="750994a2386aff0d67ee18798d70c81571a2af26f34f69599ee6d842d087d0e0",
    ),
    _repo_file(
        path="evaluator/np_box_ops.py",
        size=3556,
        sha256="ddf867c67936be2a110a01595561b31d6113a36501ce890d3d29189c7169d144",
    ),
    _repo_file(
        path="evaluator/np_mask_ops.py",
        size=4461,
        sha256="0993e66685df9e6b70ffd56703cd768a4ed1ef7e92501cee7a66312340eb7978",
    ),
    _repo_file(
        path="evaluator/polygon_list.py",
        size=2877,
        sha256="6b6b391bfea76935f10790893f50d55590c0a2bfcf2bf0fab74f6331506d657d",
    ),
    _repo_file(
        path="evaluator/polygon_ops.py",
        size=4508,
        sha256="99cd5dd3977c7fad984acc90737d8e91ca7be85eba22e090c091a9a1a2e8f4a0",
    ),
    _repo_file(
        path="gt/test.jsonl.gz",
        size=11342161,
        sha256="8a8f6cb6dd706b457b822634e9f19b7f51cc9813729d57401a3bb6ea45abd438",
    ),
    _repo_file(
        path="gt/train.jsonl.gz",
        size=59214216,
        sha256="bf7a271c7a016e66ad87556078f0371f8fb12beaa2e2e60fc33458ad57c3e4fd",
    ),
    _repo_file(
        path="gt/validation.jsonl.gz",
        size=12274736,
        sha256="ce7085d8bf24c2f13df852715c34494e6748cafcabb4bfc7a6e2bb320f58d23a",
    ),
    _repo_file(
        path="requirements.txt",
        size=109,
        sha256="12c926e6024728d52488c4a4d619a34765000f79f5b295157a1dc8e735e74067",
    ),
    _repo_file(
        path="sample_eval_scores.txt",
        size=552,
        sha256="cbb93ed0783df7ac59d6331a9c22b71493b93295d91c129f86f305f49e56d3f6",
    ),
    _repo_file(
        path="sample_output.jsonl.gz",
        size=20357682,
        sha256="627d9d00e45afc4285aee81a30a034bd12f8f430bb515fcc771d18131f0d6755",
    ),
    _image_archive(
        name="train.tgz",
        size=2798458845,
        etag="94338c1bee0085f169a22da64b6bece0-334",
    ),
    _image_archive(
        name="validation.tgz",
        size=577088394,
        etag="73d869b27676152b870d243eaa34fdea-69",
    ),
    _image_archive(
        name="test.tgz",
        size=537066598,
        etag="623f5f85a8ece90bf11af5f79087b9dc-65",
    ),
)

SPEC = DatasetSpec(
    name="hiertext",
    homepage="https://github.com/google-research-datasets/hiertext",
    license="CC-BY-SA-4.0 annotations, CC-BY-2.0 images",
    commercial_use=True,
    splits=SPLITS,
    split_roles={
        "train": SplitRole.TRAIN,
        "validation": SplitRole.VAL,
        "test": SplitRole.TEST,
    },
    category_map={WORD: "text"},
    default_class="other",
    notes=(
        "Dense text on Open Images photos at up to 1600 px. A derived label set is "
        "CC-BY-SA-4.0."
    ),
)


class HierTextSource(BronzeSource, BoxAttributeSource):
    spec = SPEC
    published_files = PUBLISHED_FILES

    def image_root(self, bronze_dir: UPath, split: str) -> UPath:
        _check_split(split)
        return bronze_dir / split

    def open_labelformat_reader(
        self, bronze_dir: UPath, split: str
    ) -> ObjectDetectionInput:
        return RawImageObjectDetectionInput(
            read_images=partial(
                self.read_raw_images, bronze_dir=bronze_dir, split=split
            )
        )

    def read_raw_images(self, bronze_dir: UPath, split: str) -> Iterator[RawImage]:
        """Read every word as the box around its polygon.

        A line and a paragraph are unions of words, so silver keeps the words only.
        Bronze keeps the lines and the paragraphs.
        """
        _check_split(split)
        with (
            (bronze_dir / "gt" / f"{split}.jsonl.gz").open(mode="rb") as handle,
            gzip.open(handle) as unzipped,
        ):
            for image in ijson.items(unzipped, "annotations.item"):
                yield RawImage(
                    file_name=f"{image['image_id']}.jpg",
                    width=int(image["image_width"]),
                    height=int(image["image_height"]),
                    boxes=tuple(
                        _build_word_box(word)
                        for paragraph in image["paragraphs"]
                        for line in paragraph["lines"]
                        for word in line["words"]
                    ),
                )


def _build_word_box(word: dict[str, Any]) -> RawBox:
    xs = [vertex[0] for vertex in word["vertices"]]
    ys = [vertex[1] for vertex in word["vertices"]]
    return RawBox(
        source_class=WORD,
        xmin=float(min(xs)),
        ymin=float(min(ys)),
        xmax=float(max(xs)),
        ymax=float(max(ys)),
        attrs={
            "attr_is_illegible": not word["legible"],
            "attr_is_handwritten": word["handwritten"],
            "attr_is_vertical": word["vertical"],
        },
    )


def _check_split(split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"Unknown HierText split: {split}")
