#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Bring one Studio dataset in step with its slice of gold.

The rule that lets a curator and the lake both write: the sync records every box as it
last wrote it, in `loaded_box`. A box that differs from its record, or that is gone,
belongs to the curator, and the sync never writes it again. Every other box follows
gold.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID

import pyarrow as pa
from lightly_studio.resolvers import annotation_resolver
from sqlalchemy import CursorResult, text
from sqlmodel import Session

from cv_lakehouse_studio.gold_client import (
    GoldClient,
    GoldDataset,
    GoldMeta,
    GoldSlice,
)
from cv_lakehouse_studio.scaffolding import (
    DatasetScaffold,
    get_or_create_dataset_scaffold,
    get_or_create_default_embedding_model,
)
from cv_lakehouse_studio.staging import (
    create_staging_tables,
    stage_boxes,
    stage_embeddings,
    stage_images,
    stage_names,
)
from cv_lakehouse_studio.studio_datasets import StudioDataset
from cv_lakehouse_studio.sync_state import SYNC_SCHEMA

# The `origin` that gold gives a box that a curator drew.
STUDIO_ORIGIN = "studio"


@dataclass(frozen=True)
class SyncReport:
    studio_dataset: str
    num_images: int
    num_new_boxes: int
    num_updated_boxes: int
    num_removed_boxes: int
    num_embeddings: int


def read_synced_gold(
    session: Session, studio_dataset: str
) -> tuple[int, datetime] | None:
    """Return the gold version and build time that this dataset was last synced to."""
    row = session.execute(
        text(
            f"select gold_version, gold_built_at from {SYNC_SCHEMA}.synced_dataset "
            "where studio_dataset = :studio_dataset"
        ),
        {"studio_dataset": studio_dataset},
    ).one_or_none()
    return None if row is None else (row[0], row[1])


def sync_dataset(
    *,
    client: GoldClient,
    meta: GoldMeta,
    dataset: GoldDataset,
    studio_dataset: StudioDataset,
    class_names: Sequence[str],
    image_base: str,
) -> SyncReport:
    name = studio_dataset.name
    gold_slice = studio_dataset.gold_slice
    splits = [
        split
        for split in dataset.splits
        if gold_slice.split is None or split.split == gold_slice.split
    ]
    scaffold = get_or_create_dataset_scaffold(
        dataset=name,
        class_names=class_names,
        tag_names=sorted(
            {f"role/{split.role}" for split in splits}
            | {f"split/{split.split}" for split in splits}
        ),
    )
    session = scaffold.session
    synced = read_synced_gold(session=session, studio_dataset=name)

    # The first page tells the dimension, which the model row needs. The resolvers
    # commit, and a commit drops the staging tables, so every resolver call comes
    # before the first staging statement.
    embedding_pages = _iter_embedding_pages(
        client=client,
        dataset=dataset,
        gold_slice=gold_slice,
        changed_since=None if synced is None else synced[1],
    )
    first_embedding_page = next(embedding_pages, None)
    embedding_model_id = _get_or_create_embedding_model(
        scaffold=scaffold, dataset=dataset, first_page=first_embedding_page
    )

    create_staging_tables(session)
    stage_names(session=session, table="stage_label", ids=scaffold.label_ids)
    stage_names(session=session, table="stage_tag", ids=scaffold.tag_ids)
    for page in client.iter_pages(table="images", gold_slice=gold_slice):
        stage_images(session=session, page=page)
    for page in client.iter_pages(table="boxes", gold_slice=gold_slice):
        stage_boxes(session=session, page=page)
    if first_embedding_page is not None:
        embedding_pages = itertools.chain([first_embedding_page], embedding_pages)
    for ids, page in embedding_pages:
        stage_embeddings(session=session, ids=ids, page=page)

    parameters = {
        "dataset": dataset.dataset,
        "studio_dataset": name,
        "image_base": image_base.rstrip("/"),
        "collection": scaffold.collection_id,
        "annotations": scaffold.annotation_collection_id,
        "model": embedding_model_id,
        "now": datetime.now(tz=UTC).replace(tzinfo=None),
        "gold_version": meta.version,
        "gold_built_at": meta.built_at,
    }
    counts = {
        key: cast(CursorResult, session.execute(text(statement), parameters)).rowcount
        for key, statement in _MERGE_STATEMENTS
    }
    removed_box_ids = [
        row[0] for row in session.execute(text(_REMOVED_BOX_QUERY), parameters)
    ]
    session.commit()

    _delete_boxes(session=session, box_ids=removed_box_ids)
    return SyncReport(
        studio_dataset=name,
        num_images=counts["images"],
        num_new_boxes=counts["new_boxes"],
        num_updated_boxes=counts["updated_boxes"],
        num_removed_boxes=len(removed_box_ids),
        num_embeddings=counts["embeddings"],
    )


def _iter_embedding_pages(
    *,
    client: GoldClient,
    dataset: GoldDataset,
    gold_slice: GoldSlice,
    changed_since: datetime | None,
) -> Iterator[tuple[list[str], pa.Table]]:
    """Yield the ids and the page of every page that holds a vector.

    An image id and a box id are both the id of a LightlyStudio sample, so the two
    tables go into one staging table.
    """
    if dataset.embedding_model is None:
        return
    for table, key in (("embeddings", "image_id"), ("crop_embeddings", "box_id")):
        for page in client.iter_pages(
            table=table, gold_slice=gold_slice, changed_since=changed_since
        ):
            if page.num_rows:
                yield page.column(key).to_pylist(), page


def _get_or_create_embedding_model(
    *,
    scaffold: DatasetScaffold,
    dataset: GoldDataset,
    first_page: tuple[list[str], pa.Table] | None,
) -> UUID | None:
    if first_page is None or dataset.embedding_model is None:
        return None
    return get_or_create_default_embedding_model(
        scaffold=scaffold,
        name=dataset.embedding_model,
        dimension=len(first_page[1].column("embedding")[0]),
    )


def _delete_boxes(session: Session, box_ids: Sequence[UUID]) -> None:
    """Delete the boxes that gold dropped, and that no curator touched.

    The resolver also removes what hangs on an annotation, such as its tags and its
    evaluation results. A box leaves gold seldom, so one call per box is cheap enough.
    """
    for box_id in box_ids:
        annotation_resolver.delete_annotation(session=session, annotation_id=box_id)
        session.execute(
            text(f"delete from {SYNC_SCHEMA}.loaded_box where box_id = :box_id"),
            {"box_id": box_id},
        )
    session.commit()


# LightlyStudio as it is now equals the record of the last write.
_STUDIO_EQUALS_LOADED = """
(label.annotation_label_name, detection.x, detection.y, detection.width,
 detection.height)
is not distinct from (loaded.label, loaded.x, loaded.y, loaded.width, loaded.height)
"""

_STUDIO_BOX_JOINS = """
join annotation_base base on base.sample_id = loaded.box_id
join object_detection_annotation detection on detection.sample_id = loaded.box_id
join annotation_label label on label.annotation_label_id = base.annotation_label_id
"""

# Each statement with the count that the report takes from it. They run in this order,
# in one transaction.
_MERGE_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "image_samples",
        """
        insert into sample (collection_id, sample_id, created_at, updated_at)
        select :collection, image_id, :now, :now from stage_image
        on conflict (sample_id) do nothing
        """,
    ),
    (
        # The path follows the image base, so a lake that moves needs no reload. A
        # linked copy outside the lake has an absolute path or a URL in gold, and keeps
        # it.
        "images",
        """
        insert into image
            (file_name, width, height, file_path_abs, sample_id, created_at, updated_at)
        select file_name, width, height,
               case
                   when image_path like '/%' or image_path ~ '^[a-z][a-z0-9+.-]*://'
                       then image_path
                   else :image_base || '/' || image_path
               end,
               image_id, :now, :now
        from stage_image
        on conflict (sample_id) do update
            set file_path_abs = excluded.file_path_abs, updated_at = excluded.updated_at
            where image.file_path_abs <> excluded.file_path_abs
        """,
    ),
    (
        "image_tags",
        """
        insert into sampletaglinktable (sample_id, tag_id)
        select image.image_id, tag.tag_id
        from stage_image image
        join stage_tag tag
            on tag.name in ('role/' || image.role, 'split/' || image.split)
        on conflict do nothing
        """,
    ),
    (
        # An image with no box is covered too: the lake says that it holds none.
        "coverage",
        """
        insert into annotation_collection_coverage
            (annotation_collection_id, parent_sample_id)
        select :annotations, image_id from stage_image
        on conflict do nothing
        """,
    ),
    (
        "loaded_images",
        f"""
        insert into {SYNC_SCHEMA}.loaded_image (image_id, dataset, split, file_name)
        select image_id, :dataset, split, file_name from stage_image
        on conflict (image_id) do nothing
        """,
    ),
    (
        # A box that a curator drew comes back from gold under the id that
        # LightlyStudio gave it. It stays the curator's: the sync neither inserts it
        # nor records it, so the export keeps it as a correction.
        "new_box_table",
        f"""
        create temp table new_box on commit drop as
        select staged.* from stage_box staged
        left join {SYNC_SCHEMA}.loaded_box loaded on loaded.box_id = staged.box_id
        where loaded.box_id is null and staged.origin <> '{STUDIO_ORIGIN}'
        """,
    ),
    (
        "box_samples",
        """
        insert into sample (collection_id, sample_id, created_at, updated_at)
        select :annotations, box_id, :now, :now from new_box
        on conflict (sample_id) do nothing
        """,
    ),
    (
        "box_annotations",
        """
        insert into annotation_base
            (created_at, sample_id, annotation_type, annotation_label_id, confidence,
             parent_sample_id, object_track_id)
        select :now, new_box.box_id, 'OBJECT_DETECTION', stage_label.label_id,
               new_box.confidence, new_box.image_id, null
        from new_box
        join stage_label on stage_label.name = new_box.label
        on conflict (sample_id) do nothing
        """,
    ),
    (
        "box_detections",
        """
        insert into object_detection_annotation (sample_id, x, y, width, height)
        select box_id, x, y, width, height from new_box
        on conflict (sample_id) do nothing
        """,
    ),
    (
        "new_boxes",
        f"""
        insert into {SYNC_SCHEMA}.loaded_box
            (box_id, image_id, dataset, studio_dataset, label, x, y, width, height)
        select box_id, image_id, :dataset, :studio_dataset, label, x, y, width, height
        from new_box
        """,
    ),
    (
        # Gold changed the box, and LightlyStudio still holds what the sync wrote.
        "changed_box_table",
        f"""
        create temp table changed_box on commit drop as
        select staged.* from stage_box staged
        join {SYNC_SCHEMA}.loaded_box loaded on loaded.box_id = staged.box_id
        {_STUDIO_BOX_JOINS}
        where (staged.label, staged.x, staged.y, staged.width, staged.height)
              is distinct from
              (loaded.label, loaded.x, loaded.y, loaded.width, loaded.height)
          and {_STUDIO_EQUALS_LOADED}
        """,
    ),
    (
        "changed_detections",
        """
        update object_detection_annotation detection
        set x = changed.x, y = changed.y, width = changed.width, height = changed.height
        from changed_box changed
        where detection.sample_id = changed.box_id
        """,
    ),
    (
        "changed_labels",
        """
        update annotation_base base
        set annotation_label_id = stage_label.label_id
        from changed_box changed
        join stage_label on stage_label.name = changed.label
        where base.sample_id = changed.box_id
        """,
    ),
    (
        "updated_boxes",
        f"""
        update {SYNC_SCHEMA}.loaded_box loaded
        set label = changed.label, x = changed.x, y = changed.y,
            width = changed.width, height = changed.height
        from changed_box changed
        where loaded.box_id = changed.box_id
        """,
    ),
    (
        # A vector of the lake replaces the one in place. A sample that LightlyStudio
        # does not hold, such as a box that a curator deleted, gets none.
        "embeddings",
        """
        insert into sample_embedding (sample_id, embedding_model_id, embedding)
        select staged.sample_id, :model, staged.embedding
        from stage_embedding staged
        join sample on sample.sample_id = staged.sample_id
        on conflict (sample_id, embedding_model_id) do update
            set embedding = excluded.embedding
        """,
    ),
    (
        "synced_dataset",
        f"""
        insert into {SYNC_SCHEMA}.synced_dataset
            (studio_dataset, gold_version, gold_built_at)
        values (:studio_dataset, :gold_version, :gold_built_at)
        on conflict (studio_dataset) do update
            set gold_version = excluded.gold_version,
                gold_built_at = excluded.gold_built_at
        """,
    ),
)

# Gold dropped the box, and LightlyStudio still holds what the sync wrote.
_REMOVED_BOX_QUERY = f"""
select loaded.box_id from {SYNC_SCHEMA}.loaded_box loaded
left join stage_box staged on staged.box_id = loaded.box_id
{_STUDIO_BOX_JOINS}
where loaded.studio_dataset = :studio_dataset
  and staged.box_id is null
  and {_STUDIO_EQUALS_LOADED}
"""
