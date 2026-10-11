#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from lakehouse_core.published_files import reject_incomplete_copy
from lakehouse_cv.sources.hiertext import HierTextSource
from tests.fixtures import make_hiertext as _fixture


def test_the_fixture_is_a_complete_copy(tmp_path: Path) -> None:
    reject_incomplete_copy(
        bronze_dir=Path(str(_fixture(tmp_path))),
        published_files=HierTextSource().published_files,
    )


def test_a_word_is_the_box_around_its_polygon(tmp_path: Path) -> None:
    (image,) = HierTextSource().read_raw_images(
        bronze_dir=_fixture(tmp_path), split="train"
    )

    assert image.file_name == "ddd.jpg"
    printed, handwritten = image.boxes
    assert (printed.xmin, printed.ymin, printed.xmax, printed.ymax) == (10, 10, 30, 20)
    assert printed.attrs == {
        "attr_is_illegible": False,
        "attr_is_handwritten": False,
        "attr_is_vertical": False,
    }
    assert handwritten.attrs["attr_is_illegible"] is True
    assert handwritten.attrs["attr_is_handwritten"] is True


def test_reads_the_test_split_and_the_vertical_flag(tmp_path: Path) -> None:
    (image,) = HierTextSource().read_raw_images(
        bronze_dir=_fixture(tmp_path), split="test"
    )

    (word,) = image.boxes
    assert word.attrs["attr_is_vertical"] is True
