#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from labelformat.model.bounding_box import BoundingBoxFormat

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.normalization import RawImageNormalizer
from cv_lakehouse.sources.base import read_raw_images
from cv_lakehouse.sources.pp4av import PP4AVSource
from cv_lakehouse.sources.published_files import reject_incomplete_copy
from tests.fixtures import make_pp4av as _fixture


def test_reads_the_test_split(tmp_path: Path) -> None:
    source = PP4AVSource()
    bronze = _fixture(tmp_path)
    reject_incomplete_copy(bronze_dir=bronze, published_files=source.published_files)

    labels = list(
        source.open_labelformat_reader(bronze_dir=bronze, split="test").get_labels()
    )
    assert len(labels) == 1
    assert labels[0].image.filename == "zurich/a.png"
    face, plate = labels[0].objects
    assert face.category.name == "face"
    assert face.box.to_format(BoundingBoxFormat.XYWH) == [90.0, 40.0, 20.0, 20.0]
    assert plate.category.name == "license_plate"


def test_bronze_holds_both_label_sets_and_reads_the_curated_one(
    tmp_path: Path,
) -> None:
    """The unfiltered labels of the fixture hold one more face than the curated."""
    source = PP4AVSource()
    bronze = _fixture(tmp_path)
    assert (bronze / "data" / "soiling_annotations" / "zurich" / "a.txt").exists()

    labels = list(
        source.open_labelformat_reader(bronze_dir=bronze, split="test").get_labels()
    )
    assert [obj.category.name for obj in labels[0].objects] == [
        "face",
        "license_plate",
    ]


def test_the_fisheye_split_is_separate(tmp_path: Path) -> None:
    source = PP4AVSource()
    bronze = _fixture(tmp_path)
    filenames = [
        image.filename
        for image in source.open_labelformat_reader(
            bronze_dir=bronze, split="fisheye"
        ).get_images()
    ]
    assert filenames == ["fisheye/a.png"]


def test_normalizes_onto_the_canonical_classes(tmp_path: Path) -> None:
    """PP4AV has no per box attribute, so `read_raw_images` adapts its reader."""
    source = PP4AVSource()
    bronze = _fixture(tmp_path)
    normalizer = RawImageNormalizer(
        category_map=source.spec.category_map, default_class=source.spec.default_class
    )
    images = list(
        normalizer.normalize_to_silver_images(
            read_raw_images(source=source, bronze_dir=bronze, split="test")
        )
    )
    boxes = [box for image in images for box in image.boxes]
    assert [CanonicalClass(box.class_id).class_name for box in boxes] == [
        "face",
        "license_plate",
    ]
    assert all(box.attrs == {} for box in boxes)
