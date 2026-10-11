#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from pathlib import Path

from lakehouse_core.published_files import reject_incomplete_copy
from lakehouse_cv.sources.textocr import TextOcrSource
from tests.fixtures import make_textocr as _fixture


def test_the_fixture_is_a_complete_copy(tmp_path: Path) -> None:
    reject_incomplete_copy(
        bronze_dir=Path(str(_fixture(tmp_path))),
        published_files=TextOcrSource().published_files,
    )


def test_reads_every_word_and_flags_the_illegible_one(tmp_path: Path) -> None:
    source = TextOcrSource()
    bronze = _fixture(tmp_path)
    images = list(source.read_raw_images(bronze_dir=bronze, split="train"))

    assert [image.file_name for image in images] == ["aaa.jpg", "bbb.jpg"]
    printed, illegible = images[0].boxes
    assert (printed.xmin, printed.ymin, printed.xmax, printed.ymax) == (10, 10, 30, 20)
    assert printed.attrs == {"attr_is_illegible": False}
    assert illegible.attrs == {"attr_is_illegible": True}
    assert images[1].boxes == ()
    root = source.image_root(bronze_dir=bronze, split="train")
    assert all((root / image.file_name).is_file() for image in images)


def test_a_val_image_is_in_the_train_folder_of_the_zip(tmp_path: Path) -> None:
    source = TextOcrSource()
    bronze = _fixture(tmp_path)
    (image,) = source.read_raw_images(bronze_dir=bronze, split="val")

    root = source.image_root(bronze_dir=bronze, split="val")
    assert (root / image.file_name).is_file()
