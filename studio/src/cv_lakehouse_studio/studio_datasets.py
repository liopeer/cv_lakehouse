#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Map a gold dataset onto the Studio datasets that hold it.

LightlyStudio holds about 1M images per dataset. A gold dataset at or under the cap is
one Studio dataset, under its own name. A larger one is a Studio dataset per split, and
a split over the cap is one per shard, such as `open_images.train.2-of-4`.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlmodel import Session

from cv_lakehouse_studio.gold_client import GoldClient, GoldDataset, GoldSlice
from cv_lakehouse_studio.sync_state import SYNC_SCHEMA

DEFAULT_MAX_IMAGES_PER_DATASET = 500_000


class StudioDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    gold_slice: GoldSlice


def read_or_plan_studio_datasets(
    *, session: Session, client: GoldClient, dataset: GoldDataset, max_images: int
) -> list[StudioDataset]:
    """Return the stored plan of the dataset. Plan only what it lacks, and store it."""
    stored = _read_studio_datasets(session=session, dataset=dataset.dataset)
    if not stored:
        planned = plan_studio_datasets(
            client=client, dataset=dataset, max_images=max_images
        )
    elif any(studio.gold_slice.split is None for studio in stored):
        return stored
    else:
        # A split that gold added after the plan.
        planned_splits = {studio.gold_slice.split for studio in stored}
        planned = [
            studio
            for split in dataset.splits
            if split.split not in planned_splits
            for studio in _plan_split(
                client=client,
                dataset=dataset.dataset,
                split=split.split,
                max_images=max_images,
            )
        ]
    _store_studio_datasets(session=session, studio_datasets=planned)
    return stored + planned


def plan_studio_datasets(
    *, client: GoldClient, dataset: GoldDataset, max_images: int
) -> list[StudioDataset]:
    whole = GoldSlice(dataset=dataset.dataset)
    if client.count_images(whole) <= max_images:
        return [StudioDataset(name=dataset.dataset, gold_slice=whole)]
    return [
        studio
        for split in dataset.splits
        for studio in _plan_split(
            client=client,
            dataset=dataset.dataset,
            split=split.split,
            max_images=max_images,
        )
    ]


def _plan_split(
    *, client: GoldClient, dataset: str, split: str, max_images: int
) -> list[StudioDataset]:
    count = client.count_images(GoldSlice(dataset=dataset, split=split))
    num_shards = max(1, math.ceil(count / max_images))
    if num_shards == 1:
        return [
            StudioDataset(
                name=f"{dataset}.{split}",
                gold_slice=GoldSlice(dataset=dataset, split=split),
            )
        ]
    return [
        StudioDataset(
            name=f"{dataset}.{split}.{shard + 1}-of-{num_shards}",
            gold_slice=GoldSlice(
                dataset=dataset, split=split, shard=shard, num_shards=num_shards
            ),
        )
        for shard in range(num_shards)
    ]


def _read_studio_datasets(session: Session, dataset: str) -> list[StudioDataset]:
    rows = session.execute(
        text(
            f"""
            select studio_dataset, split, shard, num_shards
            from {SYNC_SCHEMA}.studio_dataset
            where dataset = :dataset
            order by studio_dataset
            """
        ),
        {"dataset": dataset},
    )
    return [
        StudioDataset(
            name=name,
            gold_slice=GoldSlice(
                dataset=dataset, split=split, shard=shard, num_shards=num_shards
            ),
        )
        for name, split, shard, num_shards in rows
    ]


def _store_studio_datasets(
    session: Session, studio_datasets: list[StudioDataset]
) -> None:
    for studio in studio_datasets:
        session.execute(
            text(
                f"""
                insert into {SYNC_SCHEMA}.studio_dataset
                    (studio_dataset, dataset, split, shard, num_shards)
                values (:name, :dataset, :split, :shard, :num_shards)
                """
            ),
            {"name": studio.name, **studio.gold_slice.model_dump()},
        )
    session.commit()
