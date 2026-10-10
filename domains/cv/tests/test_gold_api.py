#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Query gold over HTTP on the fixture lake."""

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.gold_tables import (
    GOLD_BOX_SCHEMA,
    GOLD_CROP_EMBEDDING_SCHEMA,
)
from lakehouse_cv.contract.silver_tables import boxes_file
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs import silver_embeddings as embeddings_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.gold_api.app import create_app
from lakehouse_cv.gold_api.arrow_responses import (
    ARROW_STREAM_MEDIA_TYPE,
    NEXT_AFTER_HEADER,
)
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.embeddings import EMBEDDING_DIMENSION
from lakehouse_cv.transforms.gold_build import build_gold
from tests.lake_runs import (
    DATASETS,
    find_silver_files,
    materialize_assets,
    materialize_bronze_links,
    read_silver_build_ids,
)

SPECS = [source.spec for source in SOURCE_BY_NAME.values()]
FIRST_BUILD = datetime(2026, 1, 1, tzinfo=UTC)
ARROW = {"Accept": ARROW_STREAM_MEDIA_TYPE}


@pytest.fixture
def client(
    embedding_lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> TestClient:
    materialize_bronze_links(lake=embedding_lake, sources=bronze_sources)
    materialize_assets(
        lake=embedding_lake,
        assets=[
            *(silver_defs.build_silver_asset(name) for name in DATASETS),
            *(embeddings_defs.build_embeddings_asset(name) for name in DATASETS),
        ],
    )
    build_gold(
        store=embedding_lake.store,
        paths=embedding_lake.paths,
        specs=SPECS,
        silver_build_ids=read_silver_build_ids(
            lake=embedding_lake, names=[spec.name for spec in SPECS]
        ),
        code_version="test",
        built_at=FIRST_BUILD,
    )
    return TestClient(create_app(embedding_lake.store))


def _rows(client: TestClient, path: str, **params: object) -> list[dict]:
    """Read one page of the gold that the curators work on."""
    response = client.get(url=path, params={"release": "draft", **params})
    assert response.status_code == 200, response.text
    return response.json()["rows"]


def test_the_api_answers_503_before_gold_exists(lake: CvLakeResource) -> None:
    client = TestClient(create_app(lake.store))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/v1/meta").status_code == 503
    assert client.get("/v1/boxes").status_code == 503


def test_meta_is_the_gold_manifest(client: TestClient) -> None:
    meta = client.get("/v1/meta").json()
    assert "version" not in meta
    assert {dataset["dataset"] for dataset in meta["datasets"]} == set(DATASETS)
    assert {dataset["embedding_model"] for dataset in meta["datasets"]} == {
        "mobileclip_s0"
    }


def test_a_parameter_of_the_past_is_rejected(client: TestClient) -> None:
    """Gold holds no history, so it cannot tell what changed since a time."""
    response = client.get(
        url="/v1/boxes",
        params={"release": "draft", "changed_since": "2026-01-01T00:00:00Z"},
    )
    assert response.status_code == 422


def test_classes_are_the_class_registry(lake: CvLakeResource) -> None:
    """A consumer that creates labels needs every class, also one with no box yet."""
    classes = TestClient(create_app(lake.store)).get("/v1/classes").json()
    assert classes[0] == {"class_id": 0, "class_name": "face"}
    assert len(classes) == len(CanonicalClass)


def test_images_and_boxes_come_unfiltered(client: TestClient) -> None:
    assert len(_rows(client=client, path="/v1/images")) == 9
    assert len(_rows(client=client, path="/v1/boxes")) == 10


@pytest.mark.parametrize(
    argnames=("params", "expected"),
    argvalues=[
        ({"dataset": "pp4av"}, 3),
        ({"dataset": ["pp4av", "wider_face"]}, 5),
        ({"role": "test"}, 4),
        ({"role": ["val", "test"]}, 8),
        ({"split": "fisheye"}, 1),
        ({"dataset": "open_images", "role": "val"}, 3),
        ({"class_name": "license_plate"}, 3),
        ({"class_name": ["face", "other"]}, 7),
        ({"commercial_use": "true"}, 5),
        ({"dataset": "no_such_dataset"}, 0),
    ],
)
def test_boxes_filter(client: TestClient, params: dict, expected: int) -> None:
    assert len(_rows(client=client, path="/v1/boxes", **params)) == expected


def test_a_parameter_the_table_lacks_is_rejected(client: TestClient) -> None:
    assert (
        client.get(url="/v1/images", params={"class_name": "face"}).status_code == 422
    )
    assert client.get(url="/v1/boxes", params={"limit": 0}).status_code == 422


def test_paging_returns_every_box_once(client: TestClient) -> None:
    seen: list[str] = []
    after: str | None = None
    while True:
        params: dict[str, object] = {"release": "draft", "limit": 3}
        if after is not None:
            params["after"] = after
        body = client.get(url="/v1/boxes", params=params).json()
        seen.extend(row["box_id"] for row in body["rows"])
        after = body["next_after"]
        if after is None:
            break
    assert len(seen) == len(set(seen)) == 10
    assert seen == sorted(seen)


def test_boxes_come_as_an_arrow_stream_on_request(client: TestClient) -> None:
    response = client.get(
        url="/v1/boxes", params={"release": "draft", "limit": 4}, headers=ARROW
    )
    assert response.headers["content-type"] == ARROW_STREAM_MEDIA_TYPE
    table = pa.ipc.open_stream(response.content).read_all()
    assert table.schema == GOLD_BOX_SCHEMA
    assert table.num_rows == 4
    assert response.headers[NEXT_AFTER_HEADER] == table.column("box_id")[-1].as_py()


def test_embeddings_join_the_silver_vectors_onto_gold_ids(client: TestClient) -> None:
    images = _rows(client=client, path="/v1/embeddings")
    assert {row["image_id"] for row in images} == {
        row["image_id"] for row in _rows(client=client, path="/v1/images")
    }
    assert len(images[0]["embedding"]) == EMBEDDING_DIMENSION

    response = client.get(
        url="/v1/crop_embeddings", params={"release": "draft"}, headers=ARROW
    )
    crops = pa.ipc.open_stream(response.content).read_all()
    assert crops.schema == GOLD_CROP_EMBEDDING_SCHEMA
    # One vector per gold box. A flagged box has a vector in silver and none here.
    assert set(crops.column("box_id").to_pylist()) == {
        row["box_id"] for row in _rows(client=client, path="/v1/boxes")
    }


def test_a_box_with_no_vector_does_not_end_the_paging(
    client: TestClient, embedding_lake: CvLakeResource
) -> None:
    """A page of vectors can be short, so its cursor comes from the page of ids."""
    boxes = [row["box_id"] for row in _rows(client=client, path="/v1/boxes")]
    first = min(boxes)
    for path in embedding_lake.store.root.glob(
        "silver/*_embeddings/builds/*/crop_embeddings/*.parquet"
    ):
        table = pq.read_table(path)
        kept = [box_id != first for box_id in table.column("box_id").to_pylist()]
        pq.write_table(table.filter(pa.array(kept)), path)

    seen: list[str] = []
    after: str | None = None
    while True:
        params: dict[str, object] = {"release": "draft", "limit": 1}
        if after is not None:
            params["after"] = after
        body = client.get(url="/v1/crop_embeddings", params=params).json()
        seen.extend(row["box_id"] for row in body["rows"])
        after = body["next_after"]
        if after is None:
            break
    assert seen == sorted(set(boxes) - {first})


def test_a_box_that_moved_after_its_vector_gets_none(
    client: TestClient, embedding_lake: CvLakeResource
) -> None:
    """Silver moved on, and the embeddings did not run yet."""
    path = boxes_file(
        build_dir=find_silver_files(lake=embedding_lake, name="wider_face"),
        split="val",
    )
    table = pq.read_table(path)
    (moved,) = table.column("box_id").to_pylist()
    pq.write_table(
        table.set_column(
            table.schema.get_field_index("x"),
            table.schema.field("x"),
            pa.array([1.0], type=pa.float64()),
        ),
        path,
    )
    build_gold(
        store=embedding_lake.store,
        paths=embedding_lake.paths,
        specs=SPECS,
        silver_build_ids=read_silver_build_ids(
            lake=embedding_lake, names=[spec.name for spec in SPECS]
        ),
        code_version="moved",
        built_at=FIRST_BUILD,
    )

    crops = _rows(client=client, path="/v1/crop_embeddings", dataset="wider_face")

    assert moved not in {row["box_id"] for row in crops}
    assert len(crops) == 1


def test_crop_embeddings_filter_by_class(client: TestClient) -> None:
    crops = _rows(client=client, path="/v1/crop_embeddings", class_name="face")
    assert len(crops) == 6


def test_shards_split_the_images_by_a_fixed_rule(client: TestClient) -> None:
    every_id = {row["image_id"] for row in _rows(client=client, path="/v1/images")}
    seen: set[str] = set()
    for shard in range(4):
        ids = {
            row["image_id"]
            for row in _rows(
                client=client, path="/v1/images", shard=shard, num_shards=4
            )
        }
        # The rule is part of the API. A change of it moves images between shards.
        assert all(int(image_id[:8], 16) % 4 == shard for image_id in ids)
        assert not ids & seen
        seen |= ids
    assert seen == every_id


@pytest.mark.parametrize(
    argnames="path", argvalues=["/v1/boxes", "/v1/crop_embeddings"]
)
def test_a_box_comes_in_the_shard_of_its_image(client: TestClient, path: str) -> None:
    box_image = {
        row["box_id"]: row["image_id"] for row in _rows(client=client, path="/v1/boxes")
    }
    seen: set[str] = set()
    for shard in range(3):
        image_ids = {
            row["image_id"]
            for row in _rows(
                client=client, path="/v1/images", shard=shard, num_shards=3
            )
        }
        box_ids = {
            row["box_id"]
            for row in _rows(client=client, path=path, shard=shard, num_shards=3)
        }
        assert all(box_image[box_id] in image_ids for box_id in box_ids)
        seen |= box_ids
    assert seen == set(box_image)


@pytest.mark.parametrize(
    argnames="params",
    argvalues=[{"shard": 0}, {"num_shards": 2}, {"shard": 4, "num_shards": 4}],
)
def test_an_incomplete_shard_is_rejected(client: TestClient, params: dict) -> None:
    for path in ("/v1/images", "/v1/images/count"):
        response = client.get(url=path, params={"release": "draft", **params})
        assert response.status_code == 422


@pytest.mark.parametrize(
    argnames="params",
    argvalues=[
        {},
        {"dataset": "open_images"},
        {"role": "val"},
        {"shard": 1, "num_shards": 2},
        {"dataset": "no_such_dataset"},
    ],
)
def test_the_count_equals_the_rows(client: TestClient, params: dict) -> None:
    response = client.get(url="/v1/images/count", params={"release": "draft", **params})
    assert response.status_code == 200, response.text
    rows = _rows(client=client, path="/v1/images", **params)
    assert response.json() == {"count": len(rows)}


def test_a_count_of_eval_rows_names_a_release(client: TestClient) -> None:
    assert client.get("/v1/images/count").status_code == 400
    assert client.get("/v1/images/count", params={"role": "train"}).status_code == 200
