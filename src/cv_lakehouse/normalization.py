#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Normalise a source's raw images onto the canonical classes.

`RawImageNormalizer` maps every box onto a `CanonicalClass` and clips it to its
image. `silver_schema.write_split` then serialises what it yields.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Mapping

from labelformat.model.bounding_box import BoundingBox

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.silver_schema import SilverBox, SilverImage
from cv_lakehouse.sources.base import RawImage

MIN_BOX_SIDE_PIXELS = 1e-6


class RawImageNormalizer:
    """Map a source's raw images onto the canonical classes.

    It remaps the category names, clips every box to its image, and drops a box that
    clipping leaves no area. A category the map omits is not a drop: it falls to
    `default_class`, and `source_class` still carries what the source called it. The
    file names and the image dimensions pass through.

    A flagged box is not a dropped box. Silver keeps whatever a source publishes, with
    the flag alongside it, in the `attr_*` columns, and leaves the judgement to whoever
    reads the layer.

    `dropped_box_reasons` fills while `normalize_to_silver_images` runs. Read it
    after the write.
    """

    def __init__(self, category_map: Mapping[str, str], default_class: str) -> None:
        self._category_map = {
            name.lower(): CanonicalClass.from_class_name(target)
            for name, target in category_map.items()
        }
        self._default_class = CanonicalClass.from_class_name(default_class)
        self.dropped_box_reasons = Counter[str]()

    def normalize_to_silver_images(
        self, images: Iterable[RawImage]
    ) -> Iterator[SilverImage]:
        self.dropped_box_reasons.clear()
        for image in images:
            yield SilverImage(
                file_name=image.file_name,
                width=image.width,
                height=image.height,
                boxes=tuple(self._iter_silver_boxes(image)),
            )

    def _iter_silver_boxes(self, image: RawImage) -> Iterator[SilverBox]:
        box_index = 0
        for raw in image.boxes:
            canonical_class = self._category_map.get(
                raw.source_class.lower(), self._default_class
            )
            box = _clip_to_image(
                box=BoundingBox(
                    xmin=raw.xmin, ymin=raw.ymin, xmax=raw.xmax, ymax=raw.ymax
                ),
                width=image.width,
                height=image.height,
            )
            if box is None:
                self.dropped_box_reasons["degenerate_box"] += 1
                continue
            yield SilverBox(
                box_index=box_index,
                class_id=canonical_class.value,
                class_name=canonical_class.class_name,
                source_class=raw.source_class,
                x=box.xmin,
                y=box.ymin,
                w=box.xmax - box.xmin,
                h=box.ymax - box.ymin,
                confidence=raw.confidence,
                attrs=dict(raw.attrs),
            )
            box_index += 1


def _clip_to_image(box: BoundingBox, width: int, height: int) -> BoundingBox | None:
    xmin = min(max(box.xmin, 0.0), float(width))
    ymin = min(max(box.ymin, 0.0), float(height))
    xmax = min(max(box.xmax, 0.0), float(width))
    ymax = min(max(box.ymax, 0.0), float(height))
    if xmax - xmin < MIN_BOX_SIDE_PIXELS or ymax - ymin < MIN_BOX_SIDE_PIXELS:
        return None
    return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
