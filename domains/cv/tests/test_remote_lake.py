#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Build every layer and the gold API on a lake that is not on a local disk.

Bronze links copies that sit beside the lake, in the same store. Every layer then runs
on `memory://`, and on S3 against a moto server, with DuckDB on httpfs.
"""

import contextlib
import json
import os
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from fastapi.testclient import TestClient
from moto.server import ThreadedMotoServer
from upath import UPath

from lakehouse_cv.contract.manifests import CvBronzeManifest
from lakehouse_cv.defs import gold as gold_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.gold_api.app import create_app
from tests.lake_runs import DATASETS, materialize_assets, materialize_bronze_links

BUCKET = "lake"


@pytest.fixture(scope="module")
def s3_endpoint() -> Iterator[str]:
    """A moto server, or the S3 at LAKEHOUSE_TEST_S3_ENDPOINT, such as SeaweedFS."""
    endpoint = os.environ.get("LAKEHOUSE_TEST_S3_ENDPOINT")
    if endpoint is not None:
        _create_bucket(endpoint)
        yield endpoint
        return
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    _create_bucket(endpoint)
    yield endpoint
    server.stop()


def _create_bucket(endpoint: str) -> None:
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="key",
        aws_secret_access_key="secret",
        region_name="us-east-1",
    )
    with contextlib.suppress(client.exceptions.BucketAlreadyOwnedByYou):
        client.create_bucket(Bucket=BUCKET)


@pytest.fixture(params=["memory", "s3"])
def remote_lake(request: pytest.FixtureRequest) -> CvLakeResource:
    prefix = uuid.uuid4().hex
    if request.param == "memory":
        return CvLakeResource(
            root=f"memory://{prefix}/lake",
            download_workers=2,
            request_timeout_seconds=5.0,
        )
    endpoint = request.getfixturevalue("s3_endpoint")
    return CvLakeResource(
        root=f"s3://{BUCKET}/{prefix}/lake",
        storage_options=json.dumps(
            {"key": "key", "secret": "secret", "endpoint_url": endpoint}
        ),
        download_workers=2,
        request_timeout_seconds=5.0,
    )


def test_silver_and_gold_build_on_a_remote_lake(
    lake: CvLakeResource,
    remote_lake: CvLakeResource,
    bronze_sources: dict[str, UPath],
) -> None:
    materialize_bronze_links(lake=lake, sources=bronze_sources)
    remote_sources = {
        name: remote_lake.store.root.parent / "external" / name for name in DATASETS
    }
    for name, copy_dir in remote_sources.items():
        _upload_tree(source=Path(str(bronze_sources[name])), target=copy_dir)
    materialize_bronze_links(lake=remote_lake, sources=remote_sources)

    for run_lake in (lake, remote_lake):
        materialize_assets(
            lake=run_lake,
            assets=[
                *(silver_defs.build_silver_asset(name) for name in DATASETS),
                *(
                    check
                    for name in DATASETS
                    for check in silver_defs.build_silver_checks(name)
                ),
            ],
        )
        materialize_assets(
            lake=run_lake,
            assets=[gold_defs.build_gold_asset(), *gold_defs.build_gold_checks()],
        )

    for name in DATASETS:
        bronze = CvBronzeManifest.model_validate_json(
            (remote_lake.paths.bronze_dir(name) / "_bronze.json").read_text()
        )
        assert bronze.path == str(remote_sources[name])
        for root in bronze.image_roots.values():
            assert root.startswith(str(remote_sources[name]))
            # SeaweedFS refuses a key with a "." segment, and moto takes it.
            assert "/./" not in f"{root}/" and not root.endswith("/.")

    local_boxes = _read_boxes(lake)
    assert local_boxes
    assert _read_boxes(remote_lake) == local_boxes


def _read_boxes(lake: CvLakeResource) -> list[dict]:
    client = TestClient(create_app(lake.store))
    response = client.get("/v1/boxes", params={"release": "draft", "limit": 10_000})
    assert response.status_code == 200, response.text
    return sorted(
        (
            {key: value for key, value in row.items() if key != "changed_at"}
            for row in response.json()["rows"]
        ),
        key=lambda row: row["box_id"],
    )


def _upload_tree(source: Path, target: UPath) -> None:
    """Upload a tree. An object store has no empty directory, so one gets a file."""
    for directory, subdirectories, files in os.walk(source):
        target_dir = target.joinpath(*Path(directory).relative_to(source).parts)
        if not files and not subdirectories:
            (target_dir / ".keep").write_bytes(b"")
        for file_name in files:
            (target_dir / file_name).write_bytes(
                (Path(directory) / file_name).read_bytes()
            )


def test_a_location_is_a_key_under_the_root(remote_lake: CvLakeResource) -> None:
    store = remote_lake.store
    path = store.resolve("bronze/wider_face/WIDER_train")
    assert str(path) == f"{store.root}/bronze/wider_face/WIDER_train"
    assert store.location(path) == "bronze/wider_face/WIDER_train"


def test_a_location_outside_the_root_stays_whole(
    remote_lake: CvLakeResource, tmp_path: Path
) -> None:
    store = remote_lake.store
    assert store.location(store.resolve(str(tmp_path))) == str(tmp_path)
