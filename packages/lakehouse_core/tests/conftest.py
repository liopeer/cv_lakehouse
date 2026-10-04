#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import contextlib
import os
import socket
import uuid
from collections.abc import Iterator

import boto3
import pytest
from moto.server import ThreadedMotoServer
from upath import UPath

BUCKET = "lake"
S3_KEYS = {"key": "key", "secret": "secret"}


@pytest.fixture(scope="session")
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
        aws_access_key_id=S3_KEYS["key"],
        aws_secret_access_key=S3_KEYS["secret"],
        region_name="us-east-1",
    )
    with contextlib.suppress(client.exceptions.BucketAlreadyOwnedByYou):
        client.create_bucket(Bucket=BUCKET)


@pytest.fixture
def s3_dir(s3_endpoint: str) -> UPath:
    return UPath(
        f"s3://{BUCKET}/{uuid.uuid4().hex}", endpoint_url=s3_endpoint, **S3_KEYS
    )


@pytest.fixture
def memory_dir() -> UPath:
    return UPath(f"memory://{uuid.uuid4().hex}")
