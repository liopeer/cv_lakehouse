#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Apply the corrections of a curator on top of the normalised source.

Bronze keeps the dataset as published, and the corrections as snapshots beside it.
Silver is where the two meet: the overlay takes the images that the normaliser yields,
and changes, removes and adds boxes as the latest snapshot says.

A correction wins over the source, field by field. A box that a curator only relabelled
keeps the exact float coordinates of the source, because LightlyStudio holds whole
pixels and the snapshot leaves an unmoved box null.
"""

from __future__ import annotations

import dataclasses
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self

from labelformat.model.bounding_box import BoundingBox
from upath import UPath

from lakehouse_core.parquet_files import read_parquet_table
from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.correction_actions import CorrectionAction
from lakehouse_cv.contract.silver_tables import BoxOrigin, SilverBox, SilverImage
from lakehouse_cv.transforms.normalization import clip_to_image

# A label that the class registry does not name lands here, as an unmapped source
# category does. `source_class` then carries the label.
UNKNOWN_LABEL_CLASS = CanonicalClass.OTHER


@dataclass(frozen=True)
class Correction:
    split: str
    file_name: str
    box_id: str
    action: str
    label_name: str | None
    # XYWH in pixels, or None when the box did not move.
    geometry: tuple[float, float, float, float] | None


class CorrectionOverlay:
    """The corrections of one snapshot, applied to one dataset.

    `dropped_box_reasons` and `num_applied` fill while `apply_to_silver_images` runs.
    Read them after the write.
    """

    def __init__(self, corrections: Iterable[Correction]) -> None:
        self._changes: dict[str, Correction] = {}
        self._additions: dict[tuple[str, str], list[Correction]] = defaultdict(list)
        for correction in corrections:
            if correction.action == CorrectionAction.ADD:
                key = (correction.split, correction.file_name)
                self._additions[key].append(correction)
            else:
                self._changes[correction.box_id] = correction
        self.dropped_box_reasons = Counter[str]()
        self.num_applied = 0

    @classmethod
    def from_snapshot(cls, path: UPath | None) -> Self:
        """Read a snapshot file. No file gives an overlay that changes nothing."""
        if path is None:
            return cls([])
        return cls(
            Correction(
                split=row["split"],
                file_name=row["file_name"],
                box_id=row["box_id"],
                action=row["action"],
                label_name=row["label_name"],
                geometry=(
                    None
                    if row["x"] is None
                    else (row["x"], row["y"], row["w"], row["h"])
                ),
            )
            for row in read_parquet_table(path).to_pylist()
        )

    def apply_to_silver_images(
        self, split: str, images: Iterable[SilverImage]
    ) -> Iterator[SilverImage]:
        for image in images:
            boxes = [
                corrected
                for box in image.boxes
                if (corrected := self._correct_box(box=box, image=image)) is not None
            ]
            for addition in self._additions.get((split, image.file_name), []):
                added = self._build_added_box(addition=addition, image=image)
                if added is not None:
                    boxes.append(added)
            yield dataclasses.replace(
                image,
                boxes=tuple(
                    dataclasses.replace(box, box_index=index)
                    for index, box in enumerate(boxes)
                ),
            )

    def _correct_box(self, box: SilverBox, image: SilverImage) -> SilverBox | None:
        correction = self._changes.get(box.box_id)
        if correction is None:
            return box
        self.num_applied += 1
        if correction.action == CorrectionAction.DELETE:
            self.dropped_box_reasons["studio_deleted"] += 1
            return None
        if correction.label_name is not None:
            canonical_class, source_class = _resolve_label(
                label_name=correction.label_name, source_class=box.source_class
            )
            box = dataclasses.replace(
                box,
                class_id=canonical_class.value,
                class_name=canonical_class.class_name,
                source_class=source_class,
                is_class_corrected=True,
            )
        if correction.geometry is not None:
            clipped = self._clip_geometry(geometry=correction.geometry, image=image)
            if clipped is None:
                return None
            box = dataclasses.replace(
                box,
                x=clipped.xmin,
                y=clipped.ymin,
                w=clipped.xmax - clipped.xmin,
                h=clipped.ymax - clipped.ymin,
                is_geometry_corrected=True,
            )
        return box

    def _build_added_box(
        self, addition: Correction, image: SilverImage
    ) -> SilverBox | None:
        self.num_applied += 1
        # The export always gives an added box its label and its geometry.
        if addition.label_name is None or addition.geometry is None:
            self.dropped_box_reasons["incomplete_addition"] += 1
            return None
        clipped = self._clip_geometry(geometry=addition.geometry, image=image)
        if clipped is None:
            return None
        canonical_class, source_class = _resolve_label(
            label_name=addition.label_name, source_class=addition.label_name
        )
        return SilverBox(
            box_id=addition.box_id,
            # The caller renumbers every box of the image.
            box_index=0,
            class_id=canonical_class.value,
            class_name=canonical_class.class_name,
            source_class=source_class,
            x=clipped.xmin,
            y=clipped.ymin,
            w=clipped.xmax - clipped.xmin,
            h=clipped.ymax - clipped.ymin,
            origin=BoxOrigin.STUDIO,
        )

    def _clip_geometry(
        self, geometry: tuple[float, float, float, float], image: SilverImage
    ) -> BoundingBox | None:
        x, y, w, h = geometry
        box = clip_to_image(
            box=BoundingBox(xmin=x, ymin=y, xmax=x + w, ymax=y + h),
            width=image.width,
            height=image.height,
        )
        if box is None:
            self.dropped_box_reasons["degenerate_box"] += 1
        return box


def _resolve_label(label_name: str, source_class: str) -> tuple[CanonicalClass, str]:
    """Map a LightlyStudio label onto the class registry.

    Return the class, and what `source_class` holds. A known label keeps what the
    source called the box. An unknown label replaces it, so the label of the curator
    is not lost.
    """
    if label_name.lower() in CanonicalClass.all_class_names():
        return CanonicalClass.from_class_name(label_name), source_class
    return UNKNOWN_LABEL_CLASS, label_name
