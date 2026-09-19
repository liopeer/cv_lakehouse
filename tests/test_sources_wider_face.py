#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from cv_lakehouse.normalization import RawImageNormalizer
from cv_lakehouse.sources.base import read_raw_images
from cv_lakehouse.sources.wider_face import WiderFaceSource
from tests.fixtures import make_wider_face as _fixture


def test_parses_blocks_and_drops_invalid_boxes(tmp_path: Path) -> None:
    """`read` is the labelformat view, which cannot carry the reject flag."""
    labels = list(
        WiderFaceSource()
        .open_labelformat_reader(bronze_dir=_fixture(tmp_path), split="train")
        .get_labels()
    )

    assert [label.image.filename for label in labels] == [
        "0--Parade/a.jpg",
        "1--Handshaking/b.jpg",
        "2--Demonstration/c.jpg",
    ]
    assert len(labels[0].objects) == 1
    assert labels[1].objects == []
    assert len(labels[2].objects) == 1


def test_read_rows_keeps_the_rejected_box_and_flags_it(tmp_path: Path) -> None:
    source = WiderFaceSource()
    images = {
        image.file_name: image
        for image in source.read_raw_images(
            bronze_dir=_fixture(tmp_path), split="train"
        )
    }

    kept, rejected = images["0--Parade/a.jpg"].boxes
    assert kept.attrs["attr_invalid"] is False
    assert rejected.attrs["attr_invalid"] is True
    # `50 60 5 5 0 0 0 1 0 0`: the flag is the eighth field, the grades surround it.
    assert (rejected.xmin, rejected.ymin) == (50.0, 60.0)
    assert (rejected.xmax, rejected.ymax) == (55.0, 65.0)
    assert rejected.attrs["attr_occlusion"] == 0
    assert set(kept.attrs) == {
        "attr_blur",
        "attr_expression",
        "attr_illumination",
        "attr_invalid",
        "attr_occlusion",
        "attr_pose",
    }


def test_normalization_drops_the_zero_area_box(tmp_path: Path) -> None:
    source = WiderFaceSource()
    normalizer = RawImageNormalizer(
        category_map=source.spec.category_map, default_class=source.spec.default_class
    )
    bronze = _fixture(tmp_path)
    counts = [
        len(image.boxes)
        for image in normalizer.normalize_to_silver_images(
            read_raw_images(source=source, bronze_dir=bronze, split="train")
        )
    ]
    # a.jpg keeps both of its boxes now, one of them flagged. c.jpg holds an all zero
    # line, which clipping cannot rescue.
    assert counts == [2, 0, 0]
    assert normalizer.dropped_box_reasons == {"degenerate_box": 1}


def test_image_ids_are_unique_and_match_the_labels(tmp_path: Path) -> None:
    reader = WiderFaceSource().open_labelformat_reader(
        bronze_dir=_fixture(tmp_path), split="train"
    )
    ids = [image.id for image in reader.get_images()]
    assert ids == [0, 1, 2]
    assert [label.image.id for label in reader.get_labels()] == ids
