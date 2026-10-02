#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Edit LightlyStudio as the GUI does, through the services behind its routes."""

from uuid import UUID

from lightly_studio.database import db_manager
from lightly_studio.models.annotation.annotation_base import AnnotationType
from lightly_studio.resolvers import annotation_label_resolver, collection_resolver
from lightly_studio.resolvers.annotation_resolver.update_bounding_box import (
    BoundingBoxCoordinates,
)
from lightly_studio.services import annotations_service
from lightly_studio.services.annotations_service.create_annotation import (
    AnnotationCreateParams,
)

from tests.fakes import DATASET


def move_box(*, box_id: str, x: int, y: int, width: int, height: int) -> None:
    annotations_service.update_annotation_bounding_box(
        session=db_manager.persistent_session(),
        annotation_id=UUID(box_id),
        bounding_box=BoundingBoxCoordinates(x=x, y=y, width=width, height=height),
    )


def relabel_box(box_id: str, label_name: str) -> None:
    annotations_service.update_annotation_label(
        session=db_manager.persistent_session(),
        annotation_id=UUID(box_id),
        label_name=label_name,
    )


def delete_box(box_id: str) -> None:
    annotations_service.delete_annotation(
        session=db_manager.persistent_session(), annotation_id=UUID(box_id)
    )


def draw_box(
    *, image_id: str, label_name: str, x: int, y: int, width: int, height: int
) -> str:
    """Draw a box with no annotation source selected. Return the id it got."""
    session = db_manager.persistent_session()
    collection_id = collection_resolver.get_by_name(
        session=session, name=DATASET, parent_collection_id=None
    )
    assert collection_id is not None
    collection = collection_resolver.get_by_id(
        session=session, collection_id=collection_id
    )
    assert collection is not None
    label = annotation_label_resolver.get_by_label_name(
        session=session, dataset_id=collection.dataset_id, label_name=label_name
    )
    assert label is not None
    annotation = annotations_service.create_annotation(
        session=session,
        annotation=AnnotationCreateParams(
            annotation_label_id=label.annotation_label_id,
            annotation_type=AnnotationType.OBJECT_DETECTION,
            collection_id=collection_id,
            parent_sample_id=UUID(image_id),
            x=x,
            y=y,
            width=width,
            height=height,
        ),
    )
    return str(annotation.sample_id)
