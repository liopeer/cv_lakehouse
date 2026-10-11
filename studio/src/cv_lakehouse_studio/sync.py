#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Overwrite one Studio dataset with its slice of gold.

Every run writes every image and every box of gold, by the ids of gold. A row that
exists is updated in place, so a rerun changes nothing. The lake owns the annotations,
so a box that gold does not hold is deleted. The sync keeps no record of what it wrote.

An edit of a curator wins until gold holds it. The triggers of `sync_state` log every
edit, and the sync leaves a box that has an edit after the last one that gold holds.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import pyarrow as pa
from sqlalchemy import CursorResult, text
from sqlmodel import Session

from cv_lakehouse_studio.gold_client import (
    EventMarker,
    GoldClient,
    GoldDataset,
    GoldSlice,
)
from cv_lakehouse_studio.scaffolding import (
    DatasetScaffold,
    get_or_create_dataset_scaffold,
    get_or_create_default_embedding_model,
)
from cv_lakehouse_studio.staging import (
    create_embedding_staging_table,
    create_staging_tables,
    stage_boxes,
    stage_embeddings,
    stage_images,
    stage_names,
)
from cv_lakehouse_studio.studio_datasets import StudioDataset
from cv_lakehouse_studio.sync_state import (
    GOLD_LOG_SEQUENCE_SETTING,
    ORIGIN_SETTING,
    SYNC_ORIGIN,
    SYNC_SCHEMA,
    read_chain_id,
)

# The largest statement of a healthy first sync runs for minutes. A bad plan runs for
# hours.
DEFAULT_STATEMENT_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True)
class SyncReport:
    studio_dataset: str
    num_images: int
    num_new_boxes: int
    num_updated_boxes: int
    num_removed_boxes: int
    num_embeddings: int


def find_gold_log_sequence(session: Session, last_event: EventMarker | None) -> int:
    """Return the last log entry of this database that gold holds, or 0 for none.

    Gold of another database holds no edit of this one.
    """
    if last_event is None or last_event.chain_id != read_chain_id(session):
        return 0
    return last_event.log_sequence


def sync_dataset(
    *,
    client: GoldClient,
    dataset: GoldDataset,
    studio_dataset: StudioDataset,
    class_names: Sequence[str],
    image_base: str,
    statement_timeout_seconds: float,
) -> SyncReport:
    name = studio_dataset.name
    gold_slice = studio_dataset.gold_slice
    splits = [
        split
        for split in dataset.splits
        if gold_slice.split is None or split.split == gold_slice.split
    ]
    # The resolvers commit, and a commit drops the staging tables, so every resolver
    # call comes before the first staging statement.
    scaffold = get_or_create_dataset_scaffold(
        dataset=name,
        class_names=class_names,
        tag_names=sorted(
            {f"role/{split.role}" for split in splits}
            | {f"split/{split.split}" for split in splits}
        ),
    )
    session = scaffold.session
    gold_log_sequence = find_gold_log_sequence(
        session=session, last_event=dataset.last_event
    )

    _begin_sync_transaction(
        session=session,
        statement_timeout_seconds=statement_timeout_seconds,
        gold_log_sequence=gold_log_sequence,
    )
    create_staging_tables(session)
    stage_names(session=session, table="stage_label", ids=scaffold.label_ids)
    stage_names(session=session, table="stage_tag", ids=scaffold.tag_ids)
    for page in client.iter_pages(table="images", gold_slice=gold_slice):
        stage_images(session=session, page=page)
    for page in client.iter_pages(table="boxes", gold_slice=gold_slice):
        stage_boxes(session=session, page=page)
    # Autovacuum never analyzes a temp table. Without statistics, the planner guesses
    # the size of each staging table.
    session.execute(text("analyze stage_label, stage_tag, stage_image, stage_box"))

    parameters = {
        "dataset": dataset.dataset,
        "image_base": image_base.rstrip("/"),
        "collection": scaffold.collection_id,
        "annotations": scaffold.annotation_collection_id,
        "now": datetime.now(tz=UTC).replace(tzinfo=None),
        "model_name": dataset.embedding_model,
        "dataset_id": scaffold.dataset_id,
    }
    counts = {
        key: cast(CursorResult, session.execute(text(statement), parameters)).rowcount
        for key, statement in _MERGE_STATEMENTS
    }
    lacks_vectors = (
        dataset.embedding_model is not None
        and session.execute(text(_LACKS_VECTORS_QUERY), parameters).scalar_one()
    )
    session.commit()

    num_embeddings = (
        _load_vectors(
            client=client,
            scaffold=scaffold,
            dataset=dataset,
            gold_slice=gold_slice,
            statement_timeout_seconds=statement_timeout_seconds,
            gold_log_sequence=gold_log_sequence,
        )
        if lacks_vectors
        else 0
    )
    return SyncReport(
        studio_dataset=name,
        num_images=counts["images"],
        num_new_boxes=counts["new_boxes"],
        num_updated_boxes=counts["changed_box_table"],
        num_removed_boxes=counts["removed_box_samples"],
        num_embeddings=num_embeddings,
    )


def _begin_sync_transaction(
    *, session: Session, statement_timeout_seconds: float, gold_log_sequence: int
) -> None:
    """Mark the transaction as the sync. The settings end with the transaction."""
    session.execute(
        text(
            "select set_config('statement_timeout', :timeout, true), "
            "set_config(:origin_setting, :origin, true), "
            "set_config(:gold_setting, :gold_log_sequence, true)"
        ),
        {
            "timeout": f"{statement_timeout_seconds * 1000:.0f}",
            "origin_setting": ORIGIN_SETTING,
            "origin": SYNC_ORIGIN,
            "gold_setting": GOLD_LOG_SEQUENCE_SETTING,
            "gold_log_sequence": str(gold_log_sequence),
        },
    )


def _load_vectors(
    *,
    client: GoldClient,
    scaffold: DatasetScaffold,
    dataset: GoldDataset,
    gold_slice: GoldSlice,
    statement_timeout_seconds: float,
    gold_log_sequence: int,
) -> int:
    """Load every vector of the slice, and return how many samples took one.

    The sync runs this only when a sample lacks a vector, as it reads every vector of
    the slice. A vector replaces the one in place.
    """
    pages = _iter_embedding_pages(client=client, gold_slice=gold_slice)
    first_page = next(pages, None)
    if first_page is None or dataset.embedding_model is None:
        return 0
    # The first page tells the dimension, which the model row needs. The resolver
    # commits, so it comes before the staging table.
    model_id = get_or_create_default_embedding_model(
        scaffold=scaffold,
        name=dataset.embedding_model,
        dimension=len(first_page[1].column("embedding")[0]),
    )
    session = scaffold.session
    _begin_sync_transaction(
        session=session,
        statement_timeout_seconds=statement_timeout_seconds,
        gold_log_sequence=gold_log_sequence,
    )
    create_embedding_staging_table(session)
    for ids, page in itertools.chain([first_page], pages):
        stage_embeddings(session=session, ids=ids, page=page)
    session.execute(text("analyze stage_embedding"))
    count = cast(
        CursorResult,
        session.execute(
            text(_LOAD_VECTORS_STATEMENT),
            {"model": model_id, "gold_log_sequence": gold_log_sequence},
        ),
    ).rowcount
    session.commit()
    return count


def _iter_embedding_pages(
    *, client: GoldClient, gold_slice: GoldSlice
) -> Iterator[tuple[list[str], pa.Table]]:
    """Yield the ids and the page of every page that holds a vector.

    An image id and a box id are both the id of a LightlyStudio sample, so the two
    tables go into one staging table.
    """
    for table, key in (("embeddings", "image_id"), ("crop_embeddings", "box_id")):
        for page in client.iter_pages(table=table, gold_slice=gold_slice):
            if page.num_rows:
                yield page.column(key).to_pylist(), page


# A box with a log entry after the last one that gold holds. A curator edited it, and
# gold does not hold the edit yet.
_LATER_EDIT = f"""
exists (
    select from {SYNC_SCHEMA}.edit_log edit
    where edit.sample_id = {{box_id}}
      and edit.sequence > current_setting('{GOLD_LOG_SEQUENCE_SETTING}')::bigint
)
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
        # The lake writes its own tags. A tag that a curator changes stays in
        # LightlyStudio, and the lake never reads it.
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
        # A box that LightlyStudio holds with another label or another geometry.
        "changed_box_table",
        f"""
        create temp table changed_box on commit drop as
        select staged.*,
               (detection.x, detection.y, detection.width, detection.height)
                   is distinct from (staged.x, staged.y, staged.width, staged.height)
                   as is_moved
        from stage_box staged
        join annotation_base base on base.sample_id = staged.box_id
        join object_detection_annotation detection
            on detection.sample_id = staged.box_id
        join annotation_label label
            on label.annotation_label_id = base.annotation_label_id
        where (label.annotation_label_name, detection.x, detection.y, detection.width,
               detection.height)
              is distinct from
              (staged.label, staged.x, staged.y, staged.width, staged.height)
          and not {_LATER_EDIT.format(box_id="staged.box_id")}
        """,
    ),
    (
        "changed_detections",
        """
        update object_detection_annotation detection
        set x = changed.x, y = changed.y, width = changed.width, height = changed.height
        from changed_box changed
        where detection.sample_id = changed.box_id and changed.is_moved
        """,
    ),
    (
        # The vector of a moved box shows the old crop. The sync loads the new one.
        "moved_vectors",
        """
        delete from sample_embedding
        where sample_id in (select box_id from changed_box where is_moved)
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
          and base.annotation_label_id <> stage_label.label_id
        """,
    ),
    (
        # A box that a curator deleted, and that gold still holds, stays deleted.
        "new_box_table",
        f"""
        create temp table new_box on commit drop as
        select staged.* from stage_box staged
        where not exists (
                select from annotation_base base where base.sample_id = staged.box_id
            )
          and not {_LATER_EDIT.format(box_id="staged.box_id")}
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
        "new_boxes",
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
        # A box on an image of the slice that gold does not hold, and that no curator
        # edited after gold.
        "removed_box_table",
        f"""
        create temp table removed_box on commit drop as
        select base.sample_id as box_id, base.parent_sample_id as image_id
        from annotation_base base
        join object_detection_annotation detection using (sample_id)
        join stage_image image on image.image_id = base.parent_sample_id
        where not exists (
                select from stage_box staged where staged.box_id = base.sample_id
            )
          and not {_LATER_EDIT.format(box_id="base.sample_id")}
        """,
    ),
    (
        # The evaluation results of a box go with it, as LightlyStudio removes them.
        "removed_evaluation_runs",
        """
        create temp table removed_evaluation_run on commit drop as
        select distinct evaluation_run_id from evaluation_annotation_metric
        where pred_annotation_id in (select box_id from removed_box)
           or gt_annotation_id in (select box_id from removed_box)
        """,
    ),
    (
        "removed_annotation_metrics",
        """
        delete from evaluation_annotation_metric
        where pred_annotation_id in (select box_id from removed_box)
           or gt_annotation_id in (select box_id from removed_box)
        """,
    ),
    (
        "removed_sample_metrics",
        """
        delete from evaluation_sample_metric
        where evaluation_run_id in (
                select evaluation_run_id from removed_evaluation_run
            )
          and sample_id in (select image_id from removed_box)
        """,
    ),
    (
        "removed_detections",
        """
        delete from object_detection_annotation
        where sample_id in (select box_id from removed_box)
        """,
    ),
    (
        # The guard of `sync_state` keeps a box that a curator edited meanwhile. The
        # rest of the deletion follows the boxes that went.
        "removed_annotations",
        """
        with removed as (
            delete from annotation_base
            where sample_id in (select box_id from removed_box)
            returning sample_id
        )
        delete from removed_box
        where box_id not in (select sample_id from removed)
        """,
    ),
    (
        "removed_box_tags",
        """
        delete from sampletaglinktable
        where sample_id in (select box_id from removed_box)
        """,
    ),
    (
        "removed_box_vectors",
        """
        delete from sample_embedding
        where sample_id in (select box_id from removed_box)
        """,
    ),
    (
        "removed_box_samples",
        """
        delete from sample where sample_id in (select box_id from removed_box)
        """,
    ),
)

# A sample of the slice that LightlyStudio holds, with no vector of the gold model.
_LACKS_VECTORS_QUERY = """
select exists (
    select from (
        select image_id as sample_id from stage_image
        union all
        select box_id from stage_box
    ) staged
    join sample on sample.sample_id = staged.sample_id
    where not exists (
        select from sample_embedding embedding
        join embedding_model model using (embedding_model_id)
        where embedding.sample_id = staged.sample_id
          and model.name = :model_name
          and model.dataset_id = :dataset_id
    )
)
"""

# A vector of the lake replaces the one in place. A sample that LightlyStudio does not
# hold, such as a box that a curator deleted, gets none. Nor does a box that a curator
# edited after gold: its vector shows the crop of gold.
_LOAD_VECTORS_STATEMENT = f"""
insert into sample_embedding (sample_id, embedding_model_id, embedding)
select staged.sample_id, :model, staged.embedding
from stage_embedding staged
join sample on sample.sample_id = staged.sample_id
where not exists (
    select from {SYNC_SCHEMA}.edit_log edit
    where edit.sample_id = staged.sample_id
      and edit.sequence > :gold_log_sequence
)
on conflict (sample_id, embedding_model_id) do update
    set embedding = excluded.embedding
"""
