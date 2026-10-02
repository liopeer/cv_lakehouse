#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Query gold over HTTP on the fixture lake."""

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.defs import silver as silver_defs
from cv_lakehouse.defs.resources import LakeResource
from cv_lakehouse.embeddings import EMBEDDING_DIMENSION
from cv_lakehouse.gold_api.app import create_app
from cv_lakehouse.gold_api.arrow_responses import (
    ARROW_STREAM_MEDIA_TYPE,
    NEXT_AFTER_HEADER,
)
from cv_lakehouse.gold_build import build_gold_version
from cv_lakehouse.gold_schema import GOLD_BOX_SCHEMA, GOLD_CROP_EMBEDDING_SCHEMA
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME
from tests.lake_runs import DATASETS, materialize_assets, materialize_bronze_links

SPECS = [source.spec for source in SOURCE_BY_NAME.values()]
FIRST_BUILD = datetime(2026, 1, 1, tzinfo=UTC)
ARROW = {"Accept": ARROW_STREAM_MEDIA_TYPE}


@pytest.fixture
def client(embedding_lake: LakeResource, bronze_sources: dict[str, Path]) -> TestClient:
    materialize_bronze_links(lake=embedding_lake, sources=bronze_sources)
    materialize_assets(
        lake=embedding_lake,
        assets=[silver_defs.build_silver_asset(name) for name in DATASETS],
    )
    build_gold_version(
        paths=embedding_lake.paths,
        specs=SPECS,
        code_version="test",
        built_at=FIRST_BUILD,
    )
    return TestClient(create_app(embedding_lake.paths))


def _rows(client: TestClient, path: str, **params: object) -> list[dict]:
    """Read one page of the gold that the curators work on."""
    response = client.get(url=path, params={"release": "draft", **params})
    assert response.status_code == 200, response.text
    return response.json()["rows"]


def test_the_api_answers_503_before_gold_exists(lake: LakeResource) -> None:
    client = TestClient(create_app(lake.paths))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/v1/meta").status_code == 503
    assert client.get("/v1/boxes").status_code == 503


def test_meta_is_the_gold_manifest(client: TestClient) -> None:
    meta = client.get("/v1/meta").json()
    assert meta["version"] == 1
    assert {dataset["dataset"] for dataset in meta["datasets"]} == set(DATASETS)


def test_classes_are_the_class_registry(lake: LakeResource) -> None:
    """A consumer that creates labels needs every class, also one with no box yet."""
    classes = TestClient(create_app(lake.paths)).get("/v1/classes").json()
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
        ({"changed_since": "2025-12-31T00:00:00Z"}, 10),
        ({"changed_since": "2026-01-01T00:00:00Z"}, 0),
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


def test_crop_embeddings_filter_by_class(client: TestClient) -> None:
    crops = _rows(client=client, path="/v1/crop_embeddings", class_name="face")
    assert len(crops) == 6
