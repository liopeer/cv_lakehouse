#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fold the edits of the curators over the normalised source.

Bronze keeps the dataset as published, and the edits as event files beside it. Silver
is where the two meet: the overlay takes the images that the normaliser yields, and
changes, removes and adds boxes as the latest event of each box says.

An event holds the box as LightlyStudio held it, or a tombstone. Folded over the source
box, it gives a correction, and the correction wins over the source, field by field. A
box that a curator only relabelled keeps the exact float coordinates of the source,
because LightlyStudio holds whole pixels.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
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


Geometry = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxEvent:
    """The last edit of one box: the box as LightlyStudio held it, or a tombstone."""

    split: str
    file_name: str
    box_id: str
    is_deleted: bool
    # None for a tombstone.
    label_name: str | None
    # XYWH in whole pixels, or None for a tombstone.
    geometry: Geometry | None


@dataclass(frozen=True)
class Correction:
    split: str
    file_name: str
    box_id: str
    action: str
    label_name: str | None
    # XYWH in pixels, or None when the box did not move.
    geometry: Geometry | None


def round_to_pixels(box: SilverBox) -> Geometry:
    """Return the box as LightlyStudio stores it: each value rounded half up."""
    x, y, w, h = (math.floor(value + 0.5) for value in (box.x, box.y, box.w, box.h))
    return float(x), float(y), float(w), float(h)


def fold_event(event: BoxEvent, source: SilverBox | None) -> Correction | None:
    """Return the correction that an event makes to its source box, or None.

    `source` is the box of the untouched source, or None when the source holds no box
    of that id, such as a box that a curator drew.
    """
    keys = {"split": event.split, "file_name": event.file_name, "box_id": event.box_id}
    if event.is_deleted:
        if source is None:
            return None
        return Correction(
            **keys, action=CorrectionAction.DELETE, label_name=None, geometry=None
        )
    if source is None:
        return Correction(
            **keys,
            action=CorrectionAction.ADD,
            label_name=event.label_name,
            geometry=event.geometry,
        )
    label_name = event.label_name if event.label_name != source.class_name else None
    geometry = event.geometry if event.geometry != round_to_pixels(source) else None
    if label_name is None and geometry is None:
        return None
    return Correction(
        **keys, action=CorrectionAction.UPDATE, label_name=label_name, geometry=geometry
    )


class CorrectionOverlay:
    """The latest edit of each box, folded over one dataset.

    `dropped_box_reasons` and `num_applied` fill while `apply_to_silver_images` runs.
    Read them after the write.
    """

    def __init__(self, events: Iterable[BoxEvent]) -> None:
        # A later event of a box replaces an earlier one.
        self._events = {event.box_id: event for event in events}
        self._events_by_image: dict[tuple[str, str], list[BoxEvent]] = defaultdict(list)
        for event in self._events.values():
            self._events_by_image[(event.split, event.file_name)].append(event)
        self.dropped_box_reasons = Counter[str]()
        self.num_applied = 0

    @classmethod
    def from_event_files(cls, paths: Sequence[UPath]) -> Self:
        """Read the event files in order, and each in the order of its log."""
        return cls(
            BoxEvent(
                split=row["split"],
                file_name=row["file_name"],
                box_id=row["box_id"],
                is_deleted=row["is_deleted"],
                label_name=row["label_name"],
                geometry=(
                    None
                    if row["x"] is None
                    else (row["x"], row["y"], row["w"], row["h"])
                ),
            )
            for path in paths
            for row in read_parquet_table(path).sort_by("log_sequence").to_pylist()
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
            source_ids = {box.box_id for box in image.boxes}
            for event in self._events_by_image.get((split, image.file_name), []):
                if event.box_id in source_ids:
                    continue
                addition = fold_event(event=event, source=None)
                if addition is None or addition.action != CorrectionAction.ADD:
                    continue
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
        event = self._events.get(box.box_id)
        correction = None if event is None else fold_event(event=event, source=box)
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
        # An event that is no tombstone always holds a label and a geometry.
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
