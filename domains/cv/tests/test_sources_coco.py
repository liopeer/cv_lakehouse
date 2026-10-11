#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from labelformat.model.bounding_box import BoundingBoxFormat

from lakehouse_core.published_files import reject_incomplete_copy
from lakehouse_cv.sources.base import read_raw_images
from lakehouse_cv.sources.coco import CocoSource
from lakehouse_cv.transforms.normalization import RawImageNormalizer
from tests.fixtures import make_coco as _fixture


def test_the_fixture_is_a_complete_copy(tmp_path: Path) -> None:
    reject_incomplete_copy(
        bronze_dir=Path(str(_fixture(tmp_path))),
        published_files=CocoSource().published_files,
    )


def test_labelformat_view_drops_the_crowd_box(tmp_path: Path) -> None:
    labels = list(
        CocoSource()
        .open_labelformat_reader(bronze_dir=_fixture(tmp_path), split="train")
        .get_labels()
    )

    assert [label.image.filename for label in labels] == [
        "000000000001.jpg",
        "000000000002.jpg",
    ]
    person, cat = labels[0].objects
    assert person.category.name == "person"
    assert person.box.to_format(BoundingBoxFormat.XYWH) == [10.0, 20.0, 30.0, 40.0]
    assert cat.category.name == "cat"
    assert labels[1].objects == []


def test_raw_images_keep_the_crowd_box_and_flag_it(tmp_path: Path) -> None:
    images = list(
        CocoSource().read_raw_images(bronze_dir=_fixture(tmp_path), split="train")
    )

    person, cat, crowd = images[0].boxes
    assert (images[0].width, images[0].height) == (100, 80)
    assert (person.xmin, person.ymin, person.xmax, person.ymax) == (10, 20, 40, 60)
    assert person.attrs == {"attr_is_group_of": False}
    assert cat.source_class == "cat"
    assert crowd.attrs == {"attr_is_group_of": True}
    assert images[1].boxes == ()


def test_normalization_maps_person_and_the_rest_to_other(tmp_path: Path) -> None:
    source = CocoSource()
    normalizer = RawImageNormalizer(
        dataset=source.spec.name,
        split="train",
        category_map=source.spec.category_map,
        default_class=source.spec.default_class,
    )
    images = list(
        normalizer.normalize_to_silver_images(
            read_raw_images(source=source, bronze_dir=_fixture(tmp_path), split="train")
        )
    )

    assert [box.class_name for box in images[0].boxes] == ["person", "other", "person"]
