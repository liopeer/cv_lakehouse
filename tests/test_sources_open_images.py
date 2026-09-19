#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.normalization import RawImageNormalizer
from cv_lakehouse.sources.base import read_raw_images
from cv_lakehouse.sources.open_images import (
    OpenImagesSource,
    build_image_file_name,
    is_instance_box,
)
from tests.fixtures import make_open_images as _fixture


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "LabelName": "/m/0dzct",
        "IsGroupOf": "0",
        "IsDepiction": "0",
    }
    return row | overrides


def test_a_group_box_and_a_depiction_are_not_instance_boxes() -> None:
    assert is_instance_box(_row())
    assert not is_instance_box(_row(IsGroupOf="1"))
    assert not is_instance_box(_row(IsDepiction="1"))


def test_the_reader_drops_a_group_box_and_a_depiction(tmp_path: Path) -> None:
    labels = list(
        OpenImagesSource()
        .open_labelformat_reader(bronze_dir=_fixture(tmp_path), split="validation")
        .get_labels()
    )
    mids = [obj.category.name for obj in labels[0].objects]
    # The fixture holds two more face rows, one group and one depiction.
    assert mids.count("/m/0dzct") == 1


def test_scales_normalized_boxes_to_pixels(tmp_path: Path) -> None:
    labels = list(
        OpenImagesSource()
        .open_labelformat_reader(bronze_dir=_fixture(tmp_path), split="validation")
        .get_labels()
    )
    assert len(labels) == 1
    assert labels[0].image.filename == "aaa.jpg"
    face, plate, _cat = labels[0].objects
    assert (face.box.xmin, face.box.xmax) == (10.0, 20.0)
    assert (face.box.ymin, face.box.ymax) == (50.0, 100.0)
    assert (plate.box.xmax, plate.box.ymax) == (100.0, 200.0)


def test_skips_an_image_that_did_not_download(tmp_path: Path) -> None:
    reader = OpenImagesSource().open_labelformat_reader(
        bronze_dir=_fixture(tmp_path), split="validation"
    )
    assert [image.filename for image in reader.get_images()] == ["aaa.jpg"]


def test_read_rows_keeps_the_group_box_and_the_depiction(tmp_path: Path) -> None:
    source = OpenImagesSource()
    images = list(
        source.read_raw_images(bronze_dir=_fixture(tmp_path), split="validation")
    )
    assert [image.file_name for image in images] == ["aaa.jpg"]

    flags = [
        (box.attrs["attr_is_group_of"], box.attrs["attr_is_depiction"])
        for box in images[0].boxes
    ]
    assert flags == [
        (False, False),
        (False, False),
        (False, False),
        (True, False),
        (False, True),
    ]


def test_silver_keeps_an_unnamed_class_as_other(tmp_path: Path) -> None:
    """Bronze holds a cat. Silver keeps the box and loses the label."""
    source = OpenImagesSource()
    normalizer = RawImageNormalizer(
        category_map=source.spec.category_map, default_class=source.spec.default_class
    )
    images = list(
        normalizer.normalize_to_silver_images(
            read_raw_images(
                source=source, bronze_dir=_fixture(tmp_path), split="validation"
            )
        )
    )
    names = [
        CanonicalClass(box.class_id).class_name
        for image in images
        for box in image.boxes
    ]
    # The crowd box and the depiction are faces too. Gold is where they drop out.
    assert names == ["face", "license_plate", "other", "face", "face"]


def test_a_train_image_lives_in_the_shard_of_its_first_digit(tmp_path: Path) -> None:
    """The CVDF tars unpack train into train_0 to train_f."""
    images = list(
        OpenImagesSource().read_raw_images(bronze_dir=_fixture(tmp_path), split="train")
    )
    assert [image.file_name for image in images] == ["train_d/d1e.jpg"]
    assert build_image_file_name(split="validation", image_id="d1e") == "d1e.jpg"
