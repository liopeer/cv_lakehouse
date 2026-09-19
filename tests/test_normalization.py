#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
from labelformat.model.category import Category

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.normalization import RawImageNormalizer
from cv_lakehouse.silver_schema import SilverImage
from cv_lakehouse.sources.base import RawBox, RawImage

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


def _normalized() -> tuple[RawImageNormalizer, list[SilverImage]]:
    normalizer = RawImageNormalizer(category_map=CATEGORY_MAP, default_class="other")
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
    normalizer = RawImageNormalizer(category_map=CATEGORY_MAP, default_class="other")
    images = list(
        normalizer.normalize_to_silver_images(_raw(attr_blur=2, attr_invalid=False))
    )
    assert images[0].boxes[0].attrs == {"attr_blur": 2, "attr_invalid": False}
    assert images[0].boxes[1].attrs == {}
