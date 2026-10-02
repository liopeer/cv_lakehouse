#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Create the few rows a dataset needs before its samples, through LightlyStudio.

These are a handful of rows per dataset, so they go through the resolvers. The rows are
then what the installed LightlyStudio expects, whatever its schema is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from lightly_studio.core.image.image_dataset import ImageDataset
from lightly_studio.models.annotation_label import AnnotationLabelCreate
from lightly_studio.models.collection import SampleType
from lightly_studio.models.embedding_model import EmbeddingModelCreate
from lightly_studio.resolvers import (
    annotation_label_resolver,
    collection_embedding_model_resolver,
    collection_resolver,
    embedding_model_resolver,
    tag_resolver,
)
from sqlmodel import Session

# The annotation collection that holds the boxes of the lake. LightlyStudio shows it as
# the annotation source.
LAKE_ANNOTATION_COLLECTION = "lakehouse"


@dataclass(frozen=True)
class DatasetScaffold:
    session: Session
    dataset_id: UUID
    # The root collection, which holds the images.
    collection_id: UUID
    annotation_collection_id: UUID
    label_ids: dict[str, UUID]
    tag_ids: dict[str, UUID]


def get_or_create_dataset_scaffold(
    *, dataset: str, class_names: Sequence[str], tag_names: Sequence[str]
) -> DatasetScaffold:
    image_dataset = ImageDataset.load_or_create(dataset)
    session = image_dataset.session
    annotation_collection_id = collection_resolver.get_or_create_child_collection(
        session=session,
        collection_id=image_dataset.collection_id,
        sample_type=SampleType.ANNOTATION,
        name=LAKE_ANNOTATION_COLLECTION,
    )
    scaffold = DatasetScaffold(
        session=session,
        dataset_id=image_dataset.dataset_id,
        collection_id=image_dataset.collection_id,
        annotation_collection_id=annotation_collection_id,
        label_ids={
            name: _get_or_create_label(
                session=session, dataset_id=image_dataset.dataset_id, name=name
            )
            for name in class_names
        },
        tag_ids={
            name: tag_resolver.get_or_create_sample_tag_by_name(
                session=session,
                collection_id=image_dataset.collection_id,
                tag_name=name,
            ).tag_id
            for name in tag_names
        },
    )
    session.commit()
    return scaffold


def _get_or_create_label(session: Session, dataset_id: UUID, name: str) -> UUID:
    label = annotation_label_resolver.get_by_label_name(
        session=session, dataset_id=dataset_id, label_name=name
    )
    if label is None:
        label = annotation_label_resolver.create(
            session=session,
            label=AnnotationLabelCreate(
                dataset_id=dataset_id, annotation_label_name=name
            ),
        )
    return label.annotation_label_id


def get_or_create_default_embedding_model(
    *, scaffold: DatasetScaffold, name: str, dimension: int
) -> UUID:
    """Register the model of the lake vectors, and make it the default.

    The GUI reads the plot and the similarity search from the default model of a
    collection, so a model with no link is invisible.
    """
    model = embedding_model_resolver.get_or_create(
        session=scaffold.session,
        embedding_model=EmbeddingModelCreate(
            name=name, embedding_dimension=dimension, dataset_id=scaffold.dataset_id
        ),
    )
    for collection_id in (scaffold.collection_id, scaffold.annotation_collection_id):
        collection_embedding_model_resolver.get_or_add_collection_model(
            session=scaffold.session,
            collection_id=collection_id,
            embedding_model_id=model.embedding_model_id,
        )
        collection_embedding_model_resolver.set_default(
            session=scaffold.session,
            collection_id=collection_id,
            embedding_model_id=model.embedding_model_id,
        )
    return model.embedding_model_id
