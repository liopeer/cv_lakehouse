#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import base64
import hashlib
import io
import logging
import os
import tarfile
import time
import zipfile
from pathlib import Path

import httpx
import pytest

from cv_lakehouse.sources import published_files
from cv_lakehouse.sources.published_files import (
    Checksum,
    ChecksumKind,
    ChecksumMismatchError,
    PublishedFile,
    compute_checksum,
    download_published_file,
    list_required_paths,
    reject_incomplete_copy,
    unpack_archive_beside,
)
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME

LOG = logging.getLogger(__name__)
CONTENT = b"0123456789" * 100
URL = "https://example.com/data/file.bin"


def _published_file(
    content: bytes = CONTENT, path: str = "data/file.bin"
) -> PublishedFile:
    return PublishedFile(
        path=path,
        url=URL,
        size=len(content),
        checksum=Checksum(
            kind=ChecksumKind.SHA256, value=hashlib.sha256(content).hexdigest()
        ),
    )


class _Server:
    """Serve one body, honour a Range header, and record every request."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.ranges: list[str | None] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get("Range")
        self.ranges.append(header)
        # A stream, not content, so the client reads the raw bytes as it does on a
        # real connection.
        if header is None:
            return httpx.Response(status_code=200, stream=httpx.ByteStream(self.body))
        start = int(header.removeprefix("bytes=").removesuffix("-"))
        return httpx.Response(
            status_code=206, stream=httpx.ByteStream(self.body[start:])
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


def test_downloads_and_verifies_a_file(tmp_path: Path) -> None:
    server = _Server(CONTENT)
    path = download_published_file(
        published_file=_published_file(),
        bronze_dir=tmp_path,
        client=server.client(),
        log=LOG,
    )
    assert path == tmp_path / "data" / "file.bin"
    assert path.read_bytes() == CONTENT
    assert not (tmp_path / "data" / "file.bin.partial").exists()


def test_resumes_a_partial_download(tmp_path: Path) -> None:
    partial = tmp_path / "data" / "file.bin.partial"
    partial.parent.mkdir()
    partial.write_bytes(CONTENT[:300])
    server = _Server(CONTENT)
    path = download_published_file(
        published_file=_published_file(),
        bronze_dir=tmp_path,
        client=server.client(),
        log=LOG,
    )
    assert server.ranges == ["bytes=300-"]
    assert path.read_bytes() == CONTENT


def test_skips_a_file_that_is_there(tmp_path: Path) -> None:
    target = tmp_path / "data" / "file.bin"
    target.parent.mkdir()
    target.write_bytes(CONTENT)
    server = _Server(CONTENT)
    download_published_file(
        published_file=_published_file(),
        bronze_dir=tmp_path,
        client=server.client(),
        log=LOG,
    )
    assert server.ranges == []


def test_rejects_a_checksum_mismatch_and_keeps_no_file(tmp_path: Path) -> None:
    tampered = CONTENT[:-1] + b"X"
    with pytest.raises(ChecksumMismatchError):
        download_published_file(
            published_file=_published_file(),
            bronze_dir=tmp_path,
            client=_Server(tampered).client(),
            log=LOG,
        )
    assert not (tmp_path / "data" / "file.bin").exists()
    assert not (tmp_path / "data" / "file.bin.partial").exists()


def test_computes_every_checksum_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(CONTENT)
    md5 = hashlib.md5(CONTENT)
    assert compute_checksum(path=path, kind=ChecksumKind.SHA256) == (
        hashlib.sha256(CONTENT).hexdigest()
    )
    assert compute_checksum(path=path, kind=ChecksumKind.MD5_BASE64) == (
        base64.b64encode(md5.digest()).decode()
    )
    assert compute_checksum(path=path, kind=ChecksumKind.S3_ETAG) == md5.hexdigest()

    monkeypatch.setattr(target=published_files, name="S3_PART_SIZE", value=400)
    parts = [CONTENT[0:400], CONTENT[400:800], CONTENT[800:]]
    combined = hashlib.md5(b"".join(hashlib.md5(part).digest() for part in parts))
    assert compute_checksum(path=path, kind=ChecksumKind.S3_ETAG) == (
        f"{combined.hexdigest()}-3"
    )


def _write_zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(file=path, mode="w") as zipped:
        for name in names:
            info = zipfile.ZipInfo(filename=name, date_time=(2017, 3, 31, 14, 46, 0))
            zipped.writestr(
                zinfo_or_arcname=info,
                data=b"" if name.endswith("/") else name.encode(),
            )


def test_a_wrapped_zip_is_not_doubled(tmp_path: Path) -> None:
    archive = tmp_path / "WIDER_val.zip"
    _write_zip(path=archive, names=["WIDER_val/", "WIDER_val/images/a.jpg"])
    unpacked = unpack_archive_beside(archive)
    assert unpacked == tmp_path / "WIDER_val"
    assert (tmp_path / "WIDER_val" / "images" / "a.jpg").is_file()
    assert archive.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "WIDER_val",
        "WIDER_val.zip",
    ]


def test_an_unwrapped_zip_unpacks_into_its_stem(tmp_path: Path) -> None:
    archive = tmp_path / "images.zip"
    _write_zip(path=archive, names=["paris/a.png", "zurich/b.png"])
    unpack_archive_beside(archive)
    assert (tmp_path / "images" / "paris" / "a.png").is_file()
    assert (tmp_path / "images" / "zurich" / "b.png").is_file()


def test_a_zip_made_on_a_mac_is_not_doubled(tmp_path: Path) -> None:
    archive = tmp_path / "eval_tools.zip"
    _write_zip(
        path=archive,
        names=["eval_tools/", "eval_tools/a.m", "__MACOSX/eval_tools/._a.m"],
    )
    unpack_archive_beside(archive)
    assert sorted(path.name for path in (tmp_path / "eval_tools").iterdir()) == ["a.m"]
    assert not (tmp_path / "__MACOSX").exists()


def test_unzip_keeps_the_dates_in_the_zip(tmp_path: Path) -> None:
    archive = tmp_path / "example.zip"
    _write_zip(path=archive, names=["example/", "example/a/", "example/a/b.txt"])
    unpack_archive_beside(archive)
    expected = time.mktime((2017, 3, 31, 14, 46, 0, 0, 0, -1))
    for path in ("example", "example/a", "example/a/b.txt"):
        assert os.stat(tmp_path / path).st_mtime == expected


def test_a_tar_unpacks_beside_itself(tmp_path: Path) -> None:
    archive = tmp_path / "validation.tar.gz"
    with tarfile.open(name=archive, mode="w:gz") as tar:
        data = b"jpeg"
        info = tarfile.TarInfo(name="validation/aaa.jpg")
        info.size = len(data)
        tar.addfile(tarinfo=info, fileobj=io.BytesIO(data))
    unpack_archive_beside(archive)
    assert (tmp_path / "validation" / "aaa.jpg").read_bytes() == b"jpeg"


def test_a_failed_unpack_leaves_no_tree(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        unpack_archive_beside(archive)
    assert not (tmp_path / "broken").exists()


def test_a_copy_without_a_required_path_is_rejected(tmp_path: Path) -> None:
    files = [
        _published_file(path="WIDER_test.zip"),
        _published_file(path="README.md"),
    ]
    assert list_required_paths(files) == ["WIDER_test", "README.md"]
    (tmp_path / "README.md").touch()
    with pytest.raises(FileNotFoundError, match="Missing: WIDER_test$"):
        reject_incomplete_copy(bronze_dir=tmp_path, published_files=files)
    (tmp_path / "WIDER_test").mkdir()
    reject_incomplete_copy(bronze_dir=tmp_path, published_files=files)


@pytest.mark.parametrize("name", sorted(SOURCE_BY_NAME))
def test_every_source_publishes_unique_relative_paths(name: str) -> None:
    files = SOURCE_BY_NAME[name].published_files
    paths = [published_file.path for published_file in files]
    assert len(set(paths)) == len(paths)
    assert all(not Path(path).is_absolute() and ".." not in path for path in paths)
    required = list_required_paths(files)
    assert len(set(required)) == len(required)
