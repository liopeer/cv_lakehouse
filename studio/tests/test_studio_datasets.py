#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

from cv_lakehouse_studio.gold_client import GoldSlice
from cv_lakehouse_studio.studio_datasets import StudioDataset, plan_studio_datasets
from tests.fakes import DATASET, FakeGoldClient, make_image


def _plan(client: FakeGoldClient, max_images: int) -> list[StudioDataset]:
    (dataset,) = client.read_meta().datasets
    return plan_studio_datasets(client=client, dataset=dataset, max_images=max_images)


def test_a_dataset_at_the_cap_stays_one_studio_dataset() -> None:
    assert _plan(client=FakeGoldClient(), max_images=2) == [
        StudioDataset(name=DATASET, gold_slice=GoldSlice(dataset=DATASET))
    ]


def test_a_dataset_over_the_cap_is_split_by_split() -> None:
    assert [studio.name for studio in _plan(FakeGoldClient(), max_images=1)] == [
        "faces.train",
        "faces.validation",
    ]


def test_a_split_over_the_cap_is_split_into_shards() -> None:
    client = FakeGoldClient(
        more_images=[
            make_image(file_name=f"{n}.jpg", split="train", image_path=f"t/{n}.jpg")
            for n in range(4)
        ]
    )
    plan = _plan(client=client, max_images=2)

    assert [studio.name for studio in plan] == [
        "faces.train.1-of-3",
        "faces.train.2-of-3",
        "faces.train.3-of-3",
        "faces.validation",
    ]
    assert plan[1].gold_slice == GoldSlice(
        dataset=DATASET, split="train", shard=1, num_shards=3
    )
