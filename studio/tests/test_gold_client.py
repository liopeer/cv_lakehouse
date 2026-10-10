#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import httpx
import pyarrow as pa

from cv_lakehouse_studio.gold_client import (
    ARROW_STREAM_MEDIA_TYPE,
    NEXT_AFTER_HEADER,
    GoldSlice,
    HttpGoldClient,
)


def _serialize(ids: list[str]) -> bytes:
    table = pa.table({"box_id": ids})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink=sink, schema=table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def test_iter_pages_follows_the_cursor_to_the_last_page() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "after" not in request.url.params:
            return httpx.Response(
                status_code=200,
                content=_serialize(["a", "b"]),
                headers={NEXT_AFTER_HEADER: "b"},
            )
        return httpx.Response(status_code=200, content=_serialize(["c"]))

    client = HttpGoldClient(
        base_url="http://gold",
        timeout_seconds=1.0,
        rows_per_page=10,
        embedding_rows_per_page=2,
        transport=httpx.MockTransport(respond),
    )
    pages = list(
        client.iter_pages(table="boxes", gold_slice=GoldSlice(dataset="faces"))
    )

    assert [page.column("box_id").to_pylist() for page in pages] == [["a", "b"], ["c"]]
    assert [request.url.params.get("after") for request in requests] == [None, "b"]
    assert all(
        request.headers["Accept"] == ARROW_STREAM_MEDIA_TYPE
        and request.url.params["dataset"] == "faces"
        and request.url.params["release"] == "draft"
        for request in requests
    )


def test_read_meta_and_class_names_parse_the_api_answers() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/classes":
            return httpx.Response(
                status_code=200, json=[{"class_id": 0, "class_name": "face"}]
            )
        return httpx.Response(
            status_code=200,
            json={
                "build_id": "b1",
                "built_at": "2026-01-01T00:00:00Z",
                "code_version": "abc",
                "datasets": [
                    {
                        "dataset": "faces",
                        "license": "MIT",
                        "commercial_use": True,
                        "silver_code_version": "x",
                        "silver_build_id": "s1",
                        "last_event": {"chain_id": "c1", "log_sequence": 7},
                        "embedding_model": None,
                        "splits": [
                            {"split": "train", "role": "train", "image_root": "r"}
                        ],
                    }
                ],
            },
        )

    client = HttpGoldClient(
        base_url="http://gold",
        timeout_seconds=1.0,
        rows_per_page=10,
        embedding_rows_per_page=2,
        transport=httpx.MockTransport(respond),
    )
    assert client.read_class_names() == ["face"]
    meta = client.read_meta()
    assert (meta.build_id, meta.datasets[0].splits[0].role) == ("b1", "train")
    last_event = meta.datasets[0].last_event
    assert last_event is not None
    assert (last_event.chain_id, last_event.log_sequence) == ("c1", 7)


def test_a_slice_reaches_the_api_as_parameters() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/images/count":
            return httpx.Response(status_code=200, json={"count": 7})
        return httpx.Response(status_code=200, content=_serialize(["a"]))

    client = HttpGoldClient(
        base_url="http://gold",
        timeout_seconds=1.0,
        rows_per_page=10,
        embedding_rows_per_page=2,
        transport=httpx.MockTransport(respond),
    )
    gold_slice = GoldSlice(dataset="faces", split="train", shard=1, num_shards=4)
    assert client.count_images(GoldSlice(dataset="faces")) == 7
    list(client.iter_pages(table="images", gold_slice=gold_slice))

    assert dict(requests[0].url.params) == {"dataset": "faces", "release": "draft"}
    assert {
        key: requests[1].url.params[key]
        for key in ("dataset", "split", "shard", "num_shards")
    } == {"dataset": "faces", "split": "train", "shard": "1", "num_shards": "4"}


def test_a_page_of_embeddings_holds_fewer_rows() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code=200, content=_serialize(["a"]))

    client = HttpGoldClient(
        base_url="http://gold",
        timeout_seconds=1.0,
        rows_per_page=10,
        embedding_rows_per_page=2,
        transport=httpx.MockTransport(respond),
    )
    for table in ("boxes", "embeddings", "crop_embeddings"):
        list(client.iter_pages(table=table, gold_slice=GoldSlice(dataset="faces")))

    assert [request.url.params["limit"] for request in requests] == ["10", "2", "2"]
