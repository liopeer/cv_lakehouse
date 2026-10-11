#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

import pytest

from lakehouse_core.published_files import reject_incomplete_copy
from lakehouse_cv.sources.lvis import LvisSource
from tests.fixtures import make_lvis as _fixture


def test_the_fixture_is_a_complete_copy(tmp_path: Path) -> None:
    reject_incomplete_copy(
        bronze_dir=Path(str(_fixture(tmp_path))),
        published_files=LvisSource().published_files,
    )


def test_a_file_name_keeps_its_coco_folder(tmp_path: Path) -> None:
    """An image of the LVIS val split can be in train2017 of COCO."""
    images = list(
        LvisSource().read_raw_images(bronze_dir=_fixture(tmp_path), split="val")
    )

    assert [image.file_name for image in images] == ["train2017/000000000002.jpg"]
    assert images[0].boxes == ()


def test_reads_the_plate_and_the_person(tmp_path: Path) -> None:
    (image,) = LvisSource().read_raw_images(
        bronze_dir=_fixture(tmp_path), split="train"
    )

    plate, person = image.boxes
    assert plate.source_class == "license_plate"
    assert (plate.xmin, plate.ymin, plate.xmax, plate.ymax) == (50, 50, 70, 60)
    assert person.source_class == "person"


def test_the_images_are_in_another_bronze(tmp_path: Path) -> None:
    source = LvisSource()
    assert source.image_source_name == "coco"
    with pytest.raises(TypeError, match="bronze/coco"):
        source.image_root(bronze_dir=_fixture(tmp_path), split="train")
