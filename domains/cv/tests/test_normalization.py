#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import duckdb
from labelformat.model.category import Category

from lakehouse_cv.contract.box_identity import derive_box_id, derive_image_id
from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.silver_tables import SilverImage
from lakehouse_cv.sources.base import RawBox, RawImage
from lakehouse_cv.transforms.normalization import RawImageNormalizer

SRC_FACE = Category(id=7, name="Human face")
SRC_PLATE = Category(id=9, name="Vehicle registration plate")
SRC_CAT = Category(id=11, name="Cat")

CATEGORY_MAP = {
    "human face": "face",
    "vehicle registration plate": "license_plate",
}

FACE_ID = CanonicalClass.FACE.value
PLATE_ID = CanonicalClass.LICENSE_PLATE.value
OTHER_ID = CanonicalClass.OTHER.value


def _raw(**attrs: int | bool | None) -> list[RawImage]:
    """One image with four boxes, as the rows a source publishes."""
    return [
        RawImage(
            file_name="a.jpg",
            width=100,
            height=80,
            boxes=(
                RawBox(
                    source_class=SRC_FACE.name,
                    xmin=10,
                    ymin=10,
                    xmax=30,
                    ymax=40,
                    attrs=dict(attrs),
                ),
                RawBox(source_class=SRC_CAT.name, xmin=0, ymin=0, xmax=50, ymax=50),
                RawBox(
                    source_class=SRC_PLATE.name, xmin=90, ymin=60, xmax=140, ymax=120
                ),
                RawBox(source_class=SRC_FACE.name, xmin=20, ymin=20, xmax=20, ymax=45),
            ),
        ),
        RawImage(file_name="b.jpg", width=64, height=64),
    ]


def _build_normalizer() -> RawImageNormalizer:
    return RawImageNormalizer(
        dataset="d", split="train", category_map=CATEGORY_MAP, default_class="other"
    )


def _derive_a_box_id(source_box_index: int) -> str:
    return derive_box_id(
        dataset="d", split="train", file_name="a.jpg", source_box_index=source_box_index
    )


def _normalized() -> tuple[RawImageNormalizer, list[SilverImage]]:
    normalizer = _build_normalizer()
    return normalizer, list(normalizer.normalize_to_silver_images(_raw()))


def test_normalize_remaps_clips_and_drops() -> None:
    normalizer, images = _normalized()

    first = images[0].boxes
    # The cat is a class the map omits, so it keeps its box under `other`.
    assert [box.class_id for box in first] == [FACE_ID, OTHER_ID, PLATE_ID]
    assert first[0].x == 10
    assert first[1].source_class == SRC_CAT.name
    # The plate runs past the right and the bottom edge, so clipping pulls it back.
    assert (first[2].x + first[2].w, first[2].y + first[2].h) == (100, 80)
    assert images[1].boxes == ()
    assert normalizer.dropped_box_reasons == {"degenerate_box": 1}


def test_normalize_keeps_the_source_class_and_the_registry_name() -> None:
    _, images = _normalized()
    face = images[0].boxes[0]
    assert face.source_class == SRC_FACE.name
    assert face.class_name == "face"


def test_normalize_numbers_the_boxes_it_keeps() -> None:
    """A dropped box leaves no gap, so the index identifies a row in the Parquet."""
    _, images = _normalized()
    assert [box.box_index for box in images[0].boxes] == [0, 1, 2]


def test_normalize_carries_the_attributes_through() -> None:
    normalizer = _build_normalizer()
    images = list(
        normalizer.normalize_to_silver_images(_raw(attr_blur=2, attr_invalid=False))
    )
    assert images[0].boxes[0].attrs == {"attr_blur": 2, "attr_invalid": False}
    assert images[0].boxes[1].attrs == {}


def test_normalize_keys_a_box_by_its_position_in_the_source() -> None:
    """A dropped box shifts `box_index` but no id."""
    raw = RawImage(
        file_name="a.jpg",
        width=100,
        height=80,
        boxes=(
            RawBox(source_class=SRC_FACE.name, xmin=20, ymin=20, xmax=20, ymax=45),
            RawBox(source_class=SRC_FACE.name, xmin=10, ymin=10, xmax=30, ymax=40),
        ),
    )
    (image,) = _build_normalizer().normalize_to_silver_images([raw])
    (box,) = image.boxes
    assert box.box_index == 0
    assert box.box_id == _derive_a_box_id(1)


def test_normalize_gives_every_kept_box_its_source_id() -> None:
    _, images = _normalized()
    assert [box.box_id for box in images[0].boxes] == [
        _derive_a_box_id(index) for index in (0, 1, 2)
    ]


def test_derived_ids_match_the_duckdb_expression() -> None:
    """LightlyStudio and gold derive the same ids in SQL."""
    with duckdb.connect() as connection:
        row = connection.execute(
            query=(
                "select md5('d/train/a.jpg')::UUID::VARCHAR, "
                "md5('d/train/a.jpg#3')::UUID::VARCHAR"
            )
        ).fetchone()
    assert row == (
        derive_image_id(dataset="d", split="train", file_name="a.jpg"),
        _derive_a_box_id(3),
    )
