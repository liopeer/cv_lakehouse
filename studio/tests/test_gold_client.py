#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import httpx
import pyarrow as pa

from cv_lakehouse_studio.gold_client import (
    ARROW_STREAM_MEDIA_TYPE,
    NEXT_AFTER_HEADER,
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
        transport=httpx.MockTransport(respond),
    )
    pages = list(client.iter_pages(table="boxes", dataset="faces"))

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
                "version": 3,
                "built_at": "2026-01-01T00:00:00Z",
                "code_version": "abc",
                "datasets": [
                    {
                        "dataset": "faces",
                        "license": "MIT",
                        "commercial_use": True,
                        "silver_code_version": "x",
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
        transport=httpx.MockTransport(respond),
    )
    assert client.read_class_names() == ["face"]
    meta = client.read_meta()
    assert (meta.version, meta.datasets[0].splits[0].role) == (3, "train")
