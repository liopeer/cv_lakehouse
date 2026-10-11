#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from lakehouse_core.published_files import reject_incomplete_copy
from lakehouse_cv.sources.coco_wholebody import CocoWholeBodySource
from tests.fixtures import make_coco_wholebody as _fixture


def test_the_fixture_is_a_complete_copy(tmp_path: Path) -> None:
    reject_incomplete_copy(
        bronze_dir=Path(str(_fixture(tmp_path))),
        published_files=CocoWholeBodySource().published_files,
    )


def test_reads_a_face_only_where_it_is_annotated(tmp_path: Path) -> None:
    (image,) = CocoWholeBodySource().read_raw_images(
        bronze_dir=_fixture(tmp_path), split="train"
    )

    assert image.file_name == "train2017/000000000001.jpg"
    assert [box.source_class for box in image.boxes] == ["person", "face", "person"]
    face = image.boxes[1]
    assert (face.xmin, face.ymin, face.xmax, face.ymax) == (15, 22, 23, 30)
    assert face.attrs == {}
