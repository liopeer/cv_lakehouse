#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import threading
from collections import Counter
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from shared_deps.image_sources import (
    ImageFetchError,
    fetch_urls,
    is_url,
    redact_url,
)

BODIES = {"/a.jpg": b"jpeg a", "/b.png": b"png b"}


class _Handler(BaseHTTPRequestHandler):
    requests: Counter[str] = Counter()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        _Handler.requests[path] += 1
        body = BODIES.get(path)
        if body is None:
            self.send_error(403, "SignatureDoesNotMatch")
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def base_url() -> Iterator[str]:
    _Handler.requests.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_a_path_is_not_a_url() -> None:
    assert not is_url("/mnt/lake/bronze/a.jpg")
    assert is_url("https://s3.example.com/lake/a.jpg?X-Amz-Signature=abc")


@pytest.mark.parametrize("path", ["s3://lake/a.jpg", "file:///lake/a.jpg"])
def test_any_other_scheme_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="Presign"):
        is_url(path)


def test_each_url_is_fetched_once(base_url: str) -> None:
    signed = f"{base_url}/a.jpg?X-Amz-Signature=abc"
    urls = [signed, f"{base_url}/b.png", signed, signed]

    bodies = fetch_urls(urls, timeout_seconds=5, max_workers=4)

    assert [bodies[url] for url in urls] == [b"jpeg a", b"png b", b"jpeg a", b"jpeg a"]
    assert _Handler.requests == {"/a.jpg": 1, "/b.png": 1}


def test_a_refused_url_fails_without_its_signature(base_url: str) -> None:
    url = f"{base_url}/missing.jpg?X-Amz-Signature=secret"
    with pytest.raises(ImageFetchError, match="403") as error:
        fetch_urls([url], timeout_seconds=5, max_workers=1)
    assert "secret" not in str(error.value)


def test_redacting_keeps_the_object() -> None:
    assert (
        redact_url("https://s3.example.com/lake/a.jpg?X-Amz-Signature=abc#x")
        == "https://s3.example.com/lake/a.jpg"
    )
