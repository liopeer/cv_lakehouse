#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Sync an in memory gold into a real LightlyStudio database on Postgres."""

from uuid import UUID

import pyarrow as pa
import pytest
from lightly_studio.database import db_manager
from lightly_studio.resolvers import annotation_resolver
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from cv_lakehouse_studio import sync
from cv_lakehouse_studio.sync_loop import sync_every_dataset
from cv_lakehouse_studio.sync_state import create_sync_schema
from tests.curator import relabel_box
from tests.fakes import (
    BOX_1,
    BOX_2,
    DATASET,
    IMAGE_A,
    IMAGE_B,
    FakeGoldClient,
    derive_id,
    make_box,
    make_image,
    select_shard,
)

IMAGE_BASE = "/lake"

pytestmark = pytest.mark.usefixtures("database_url")


def _sync(client: FakeGoldClient, image_base: str = IMAGE_BASE):
    return sync_every_dataset(client=client, image_base=image_base)


def _fetch(query: str) -> list[tuple]:
    session = db_manager.persistent_session()
    rows = [tuple(row) for row in session.execute(text(query))]
    session.commit()
    return rows


def _fetch_boxes() -> dict[str, tuple]:
    """Every box in LightlyStudio as (label, x, y, width, height), by its id."""
    rows = _fetch(
        """
        select base.sample_id, label.annotation_label_name,
               detection.x, detection.y, detection.width, detection.height
        from annotation_base base
        join object_detection_annotation detection using (sample_id)
        join annotation_label label using (annotation_label_id)
        """
    )
    return {str(row[0]): row[1:] for row in rows}


def _move_box(box_id: str, x: int) -> None:
    """Move a box as the GUI does."""
    session = db_manager.persistent_session()
    session.execute(
        text("update object_detection_annotation set x = :x where sample_id = :id"),
        {"x": x, "id": UUID(box_id)},
    )
    session.commit()


def _delete_box(box_id: str) -> None:
    annotation_resolver.delete_annotation(
        session=db_manager.persistent_session(), annotation_id=UUID(box_id)
    )


def test_the_first_sync_loads_gold() -> None:
    (report,) = _sync(FakeGoldClient())

    assert (report.num_images, report.num_new_boxes) == (2, 2)
    # LightlyStudio stores whole pixels, so 10.4 and 20.5 round.
    assert _fetch_boxes() == {
        BOX_1: ("face", 10, 21, 30, 40),
        BOX_2: ("license_plate", 1, 2, 3, 4),
    }
    assert _fetch("select name from collection where parent_collection_id is null") == [
        (DATASET,)
    ]
    assert sorted(_fetch("select file_name, file_path_abs from image")) == [
        ("a.jpg", "/lake/bronze/faces/train/a.jpg"),
        ("b.jpg", "/lake/bronze/faces/validation/b.jpg"),
    ]
    # Two images and two boxes, one vector each.
    assert _fetch("select count(*) from sample_embedding") == [(4,)]


def test_an_image_carries_its_role_and_its_split_as_tags() -> None:
    _sync(FakeGoldClient())
    tags = _fetch(
        """
        select image.file_name, tag.name from sampletaglinktable link
        join tag using (tag_id)
        join image using (sample_id)
        order by 1, 2
        """
    )
    assert tags == [
        ("a.jpg", "role/train"),
        ("a.jpg", "split/train"),
        ("b.jpg", "role/val"),
        ("b.jpg", "split/validation"),
    ]


def test_a_sync_with_no_new_gold_does_nothing() -> None:
    client = FakeGoldClient()
    _sync(client)
    assert _sync(client) == []


def test_a_new_gold_with_equal_rows_writes_no_box() -> None:
    client = FakeGoldClient()
    _sync(client)
    client.publish_new_version()

    (report,) = _sync(client)

    assert (report.num_new_boxes, report.num_updated_boxes) == (0, 0)
    assert report.num_removed_boxes == 0
    assert _fetch("select count(*) from sample") == [(4,)]
    # The second run asks only for the vectors that changed since the first.
    assert client.requested_changed_since[:2] == [None, None]
    assert all(since is not None for since in client.requested_changed_since[2:])


def test_a_box_that_gold_changed_follows_gold() -> None:
    client = FakeGoldClient()
    _sync(client)
    client.boxes[0] = make_box(
        box_id=BOX_1, class_name="other", x=50.0, y=20.0, w=30.0, h=40.0
    )
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_updated_boxes == 1
    assert _fetch_boxes()[BOX_1] == ("other", 50, 20, 30, 40)


def test_a_box_that_a_curator_moved_is_never_overwritten() -> None:
    client = FakeGoldClient()
    _sync(client)
    _move_box(box_id=BOX_1, x=77)
    client.boxes[0] = make_box(
        box_id=BOX_1, class_name="other", x=50.0, y=20.0, w=30.0, h=40.0
    )
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_updated_boxes == 0
    assert _fetch_boxes()[BOX_1] == ("face", 77, 21, 30, 40)


def test_a_box_that_a_curator_relabelled_is_never_overwritten() -> None:
    client = FakeGoldClient()
    _sync(client)
    relabel_box(box_id=BOX_1, label_name="other")
    client.boxes[0] = make_box(
        box_id=BOX_1, class_name="face", x=50.0, y=20.0, w=30.0, h=40.0
    )
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_updated_boxes == 0
    assert _fetch_boxes()[BOX_1] == ("other", 10, 21, 30, 40)


def test_the_first_sync_updates_no_box() -> None:
    (report,) = _sync(FakeGoldClient())

    assert report.num_updated_boxes == 0


def test_a_sync_over_its_statement_timeout_fails_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stage_boxes_slowly(*, session: Session, page: pa.Table) -> None:
        session.execute(text("select pg_sleep(0.1)"))

    monkeypatch.setattr(sync, "stage_boxes", stage_boxes_slowly)
    with pytest.raises(OperationalError, match="statement timeout"):
        sync_every_dataset(
            client=FakeGoldClient(),
            image_base=IMAGE_BASE,
            statement_timeout_seconds=0.01,
        )
    db_manager.persistent_session().rollback()

    assert _fetch("select count(*) from lakehouse_sync.loaded_box") == [(0,)]


def test_a_box_that_a_curator_deleted_is_never_loaded_again() -> None:
    client = FakeGoldClient()
    _sync(client)
    _delete_box(BOX_1)
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_new_boxes == 0
    assert set(_fetch_boxes()) == {BOX_2}


def test_a_box_that_gold_dropped_leaves_lightly_studio() -> None:
    client = FakeGoldClient()
    _sync(client)
    del client.boxes[1]
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_removed_boxes == 1
    assert set(_fetch_boxes()) == {BOX_1}
    assert _fetch("select count(*) from lakehouse_sync.loaded_box") == [(1,)]


def test_a_box_that_gold_dropped_stays_when_a_curator_moved_it() -> None:
    client = FakeGoldClient()
    _sync(client)
    _move_box(box_id=BOX_2, x=9)
    del client.boxes[1]
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_removed_boxes == 0
    assert set(_fetch_boxes()) == {BOX_1, BOX_2}


def test_a_new_gold_box_is_added() -> None:
    client = FakeGoldClient()
    _sync(client)
    new_box = derive_id("a.jpg#2")
    client.boxes.append(
        make_box(box_id=new_box, class_name="face", x=5.0, y=5.0, w=5.0, h=5.0)
    )
    client.publish_new_version()

    (report,) = _sync(client)

    assert report.num_new_boxes == 1
    assert _fetch_boxes()[new_box] == ("face", 5, 5, 5, 5)
    assert _fetch(
        f"select parent_sample_id from annotation_base where sample_id = '{new_box}'"
    ) == [(UUID(IMAGE_A),)]


def test_a_new_image_base_moves_every_path() -> None:
    client = FakeGoldClient()
    _sync(client)
    client.publish_new_version()

    _sync(client=client, image_base="s3://bucket/lake/")

    assert sorted(_fetch("select file_path_abs from image")) == [
        ("s3://bucket/lake/bronze/faces/train/a.jpg",),
        ("s3://bucket/lake/bronze/faces/validation/b.jpg",),
    ]


def test_a_path_of_a_linked_copy_keeps_its_location() -> None:
    client = FakeGoldClient(
        image_paths=(
            "s3://datasets/faces/train/a.jpg",
            "/mnt/datasets/faces/validation/b.jpg",
        )
    )

    _sync(client=client, image_base="s3://bucket/lake")

    assert sorted(_fetch("select file_path_abs from image")) == [
        ("/mnt/datasets/faces/validation/b.jpg",),
        ("s3://datasets/faces/train/a.jpg",),
    ]


def test_a_gold_with_no_embedding_model_loads_no_vector() -> None:
    _sync(FakeGoldClient(embedding_model=None))
    assert _fetch("select count(*) from sample_embedding") == [(0,)]
    assert _fetch("select count(*) from embedding_model") == [(0,)]


def _make_sharded_client() -> FakeGoldClient:
    """Train holds seven images, so a cap of two gives four shards."""
    more_images = [
        make_image(file_name=f"{n}.jpg", split="train", image_path=f"t/{n}.jpg")
        for n in range(6)
    ]
    boxes = [
        make_box(
            box_id=derive_id(f"{image['file_name']}#0"),
            image_id=image["image_id"],
            class_name="face",
            x=1.0,
            y=1.0,
            w=2.0,
            h=2.0,
        )
        for image in more_images
    ]
    return FakeGoldClient(more_images=more_images, boxes=[*_DEFAULT_BOXES, *boxes])


_DEFAULT_BOXES = FakeGoldClient().boxes


def _fetch_images_by_studio_dataset() -> dict[str, set[str]]:
    rows = _fetch(
        """
        select collection.name, image.sample_id
        from image
        join sample using (sample_id)
        join collection using (collection_id)
        """
    )
    images: dict[str, set[str]] = {}
    for name, image_id in rows:
        images.setdefault(name, set()).add(str(image_id))
    return images


def test_a_dataset_over_the_cap_syncs_into_one_studio_dataset_per_shard() -> None:
    client = _make_sharded_client()
    reports = sync_every_dataset(client=client, image_base=IMAGE_BASE, max_images=2)

    names = [f"faces.train.{k}-of-4" for k in range(1, 5)] + ["faces.validation"]
    assert sorted(report.studio_dataset for report in reports) == names
    images = _fetch_images_by_studio_dataset()
    assert images["faces.validation"] == {IMAGE_B}
    for shard in range(4):
        assert all(
            select_shard(image_id=image_id, num_shards=4) == shard
            for image_id in images.get(f"faces.train.{shard + 1}-of-4", set())
        )
    assert len(_fetch_boxes()) == 8


def test_a_shard_keeps_the_boxes_of_another_shard() -> None:
    client = _make_sharded_client()
    sync_every_dataset(client=client, image_base=IMAGE_BASE, max_images=2)
    client.publish_new_version()

    reports = sync_every_dataset(client=client, image_base=IMAGE_BASE, max_images=2)

    assert all(report.num_removed_boxes == 0 for report in reports)
    assert len(_fetch_boxes()) == 8


def test_a_stored_plan_stays_when_gold_grows() -> None:
    client = _make_sharded_client()
    sync_every_dataset(client=client, image_base=IMAGE_BASE, max_images=2)
    client.more_images.extend(
        make_image(file_name=f"new{n}.jpg", split="train", image_path=f"t/new{n}.jpg")
        for n in range(4)
    )
    client.publish_new_version()

    sync_every_dataset(client=client, image_base=IMAGE_BASE, max_images=2)

    names = _fetch("select name from collection where parent_collection_id is null")
    assert sorted(name for (name,) in names) == [
        f"faces.train.{k}-of-4" for k in range(1, 5)
    ] + ["faces.validation"]
    images = _fetch_images_by_studio_dataset()
    assert sum(len(ids) for ids in images.values()) == 12


def test_the_state_of_a_sync_before_shards_carries_over() -> None:
    session = db_manager.persistent_session()
    for statement in (
        "create schema lakehouse_sync",
        """
        create table lakehouse_sync.synced_dataset (
            dataset text primary key,
            gold_version integer not null,
            gold_built_at timestamptz not null
        )
        """,
        """
        create table lakehouse_sync.loaded_box (
            box_id uuid primary key, image_id uuid not null, dataset text not null,
            label text not null, x integer not null, y integer not null,
            width integer not null, height integer not null
        )
        """,
        "insert into lakehouse_sync.synced_dataset values ('faces', 1, now())",
        f"""
        insert into lakehouse_sync.loaded_box
        values ('{BOX_1}', '{IMAGE_A}', 'faces', 'face', 1, 2, 3, 4)
        """,
    ):
        session.execute(text(statement))
    session.commit()

    create_sync_schema(session)

    assert _fetch("select studio_dataset from lakehouse_sync.synced_dataset") == [
        ("faces",)
    ]
    assert _fetch("select studio_dataset from lakehouse_sync.loaded_box") == [
        ("faces",)
    ]
