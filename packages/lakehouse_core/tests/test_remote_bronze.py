#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Download and unpack into an object store: memory, and S3 on a moto server."""

import hashlib
import io
import logging
import os
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from upath import UPath

from lakehouse_core import published_files
from lakehouse_core.published_files import (
    UNVERIFIED_SUFFIX,
    Checksum,
    ChecksumKind,
    ChecksumMismatchError,
    PublishedFile,
    download_published_file,
    unpack_archive_beside,
)
from lakehouse_core.remote_archives import unpacked_marker
from lakehouse_core.s3_uploads import MIB, choose_part_size

LOG = logging.getLogger(__name__)
URL = "https://example.com/data/big.bin"
# moto refuses a part under 5 MiB, unless it is the last.
PART_SIZE = 5 * MIB
CONTENT = os.urandom(2 * PART_SIZE + 123)


def _published_file(content: bytes = CONTENT) -> PublishedFile:
    return PublishedFile(
        path="data/big.bin",
        url=URL,
        size=len(content),
        checksum=Checksum(
            kind=ChecksumKind.SHA256, value=hashlib.sha256(content).hexdigest()
        ),
    )


class _BrokenStream(httpx.SyncByteStream):
    """Send `body` in chunks, and drop the connection after `fail_after` bytes."""

    def __init__(self, body: bytes, fail_after: int | None) -> None:
        self._body = body
        self._fail_after = fail_after

    def __iter__(self) -> Iterator[bytes]:
        for start in range(0, len(self._body), MIB):
            if self._fail_after is not None and start >= self._fail_after:
                raise httpx.ReadError("connection dropped")
            yield self._body[start : start + MIB]


class _Server:
    """Serve one body, honour a Range header unless told not to, and record requests."""

    def __init__(
        self, body: bytes, fail_after: int | None = None, honours_range: bool = True
    ) -> None:
        self.body = body
        self.fail_after = fail_after
        self.honours_range = honours_range
        self.ranges: list[str | None] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get("Range")
        self.ranges.append(header)
        if header is None or not self.honours_range:
            return httpx.Response(
                status_code=200, stream=_BrokenStream(self.body, self.fail_after)
            )
        start = int(header.removeprefix("bytes=").removesuffix("-"))
        return httpx.Response(
            status_code=206, stream=_BrokenStream(self.body[start:], None)
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))


@pytest.fixture(autouse=True)
def small_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        target=published_files, name="choose_part_size", value=lambda size: PART_SIZE
    )


def _download(bronze_dir: UPath, server: _Server, content: bytes = CONTENT) -> UPath:
    path = download_published_file(
        published_file=_published_file(content),
        bronze_dir=bronze_dir,
        client=server.client(),
        log=LOG,
    )
    assert isinstance(path, UPath)
    return path


def _open_uploads(s3_dir: UPath) -> list[str]:
    bucket, _, prefix = s3_dir.path.partition("/")
    response = s3_dir.fs.call_s3("list_multipart_uploads", Bucket=bucket, Prefix=prefix)
    return [upload["Key"] for upload in response.get("Uploads", [])]


def test_the_part_size_keeps_a_file_under_the_part_limit() -> None:
    assert choose_part_size(1) == 64 * MIB
    terabyte = 1024**4
    part_size = choose_part_size(terabyte)
    assert part_size % MIB == 0
    assert -(-terabyte // part_size) < 10_000


@pytest.mark.parametrize("store", ["memory_dir", "s3_dir"])
def test_a_download_streams_into_the_store(
    store: str, request: pytest.FixtureRequest
) -> None:
    bronze_dir: UPath = request.getfixturevalue(store)
    server = _Server(CONTENT)
    path = _download(bronze_dir=bronze_dir, server=server)
    assert path.read_bytes() == CONTENT
    assert server.ranges == [None]

    _download(bronze_dir=bronze_dir, server=server)
    assert server.ranges == [None]


@pytest.mark.parametrize("store", ["memory_dir", "s3_dir"])
def test_a_download_with_the_wrong_bytes_leaves_nothing(
    store: str, request: pytest.FixtureRequest
) -> None:
    bronze_dir: UPath = request.getfixturevalue(store)
    tampered = CONTENT[:-1] + bytes([CONTENT[-1] ^ 1])
    with pytest.raises(ChecksumMismatchError):
        _download(bronze_dir=bronze_dir, server=_Server(tampered))
    assert not (bronze_dir / "data" / "big.bin").exists()


def test_s3_resumes_after_the_last_whole_part(s3_dir: UPath) -> None:
    server = _Server(CONTENT, fail_after=2 * PART_SIZE)
    with pytest.raises(httpx.ReadError):
        _download(bronze_dir=s3_dir, server=server)
    assert not (s3_dir / "data" / "big.bin").exists()
    assert len(_open_uploads(s3_dir)) == 1

    path = _download(bronze_dir=s3_dir, server=server)

    assert server.ranges == [None, f"bytes={2 * PART_SIZE}-"]
    assert path.read_bytes() == CONTENT
    assert not path.with_name(path.name + UNVERIFIED_SUFFIX).exists()
    assert _open_uploads(s3_dir) == []


def test_s3_deletes_a_resumed_file_that_fails_its_checksum(s3_dir: UPath) -> None:
    with pytest.raises(httpx.ReadError):
        _download(bronze_dir=s3_dir, server=_Server(CONTENT, fail_after=PART_SIZE))
    changed = CONTENT[:-1] + bytes([CONTENT[-1] ^ 1])

    with pytest.raises(ChecksumMismatchError):
        _download(bronze_dir=s3_dir, server=_Server(changed), content=CONTENT)

    target = s3_dir / "data" / "big.bin"
    assert not target.exists()
    assert not target.with_name(target.name + UNVERIFIED_SUFFIX).exists()


def test_s3_starts_over_when_the_server_ignores_the_range(s3_dir: UPath) -> None:
    with pytest.raises(httpx.ReadError):
        _download(bronze_dir=s3_dir, server=_Server(CONTENT, fail_after=PART_SIZE))
    server = _Server(CONTENT, honours_range=False)

    path = _download(bronze_dir=s3_dir, server=server)

    assert server.ranges == [f"bytes={PART_SIZE}-"]
    assert path.read_bytes() == CONTENT


def test_s3_verifies_a_file_whose_read_back_never_ran(s3_dir: UPath) -> None:
    target = s3_dir / "data" / "big.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(CONTENT[:-1] + b"X")
    target.with_name(target.name + UNVERIFIED_SUFFIX).write_bytes(b"")

    with pytest.raises(ChecksumMismatchError):
        _download(bronze_dir=s3_dir, server=_Server(CONTENT))
    assert not target.exists()

    _download(bronze_dir=s3_dir, server=_Server(CONTENT))
    assert target.read_bytes() == CONTENT


def _make_archives(directory: Path) -> list[Path]:
    """One archive of each kind and layout that bronze meets."""
    directory.mkdir()
    wrapped_zip = directory / "WIDER_val.zip"
    with zipfile.ZipFile(wrapped_zip, mode="w") as zipped:
        zipped.writestr("WIDER_val/", b"")
        zipped.writestr("WIDER_val/images/a.jpg", b"a")
        zipped.writestr("__MACOSX/WIDER_val/._a.jpg", b"fork")
    flat_zip = directory / "images.zip"
    with zipfile.ZipFile(flat_zip, mode="w") as zipped:
        zipped.writestr("paris/a.png", b"p")
        zipped.writestr("zurich/b.png", b"z")
    wrapped_tar = directory / "validation.tar.gz"
    flat_tar = directory / "train_0.tar.gz"
    for archive, names in (
        (wrapped_tar, ["validation/aaa.jpg", "validation/sub/bbb.jpg"]),
        (flat_tar, ["./x.jpg", "./y/z.jpg"]),
    ):
        with tarfile.open(name=archive, mode="w:gz") as tar:
            for name in names:
                info = tarfile.TarInfo(name=name)
                info.size = len(name)
                tar.addfile(tarinfo=info, fileobj=io.BytesIO(name.encode()))
    return [wrapped_zip, flat_zip, wrapped_tar, flat_tar]


def _list_files(root: Path | UPath) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("store", ["memory_dir", "s3_dir"])
def test_an_archive_unpacks_in_the_store_as_it_does_on_disk(
    store: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    bronze_dir: UPath = request.getfixturevalue(store)
    for archive in _make_archives(tmp_path / "local"):
        remote_archive = bronze_dir / archive.name
        remote_archive.write_bytes(archive.read_bytes())

        local = unpack_archive_beside(archive)
        remote = unpack_archive_beside(remote_archive)

        assert isinstance(remote, UPath)
        assert remote.name == local.name
        assert _list_files(remote) == _list_files(local)
        assert unpacked_marker(archive=remote_archive, unpacked=remote).exists()


def test_an_unpack_without_its_marker_starts_over(
    memory_dir: UPath, tmp_path: Path
) -> None:
    archive = _make_archives(tmp_path / "local")[2]
    remote_archive = memory_dir / archive.name
    remote_archive.write_bytes(archive.read_bytes())
    leftover = memory_dir / "validation" / "half_written.jpg"
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_bytes(b"half")

    unpack_archive_beside(remote_archive)

    assert not leftover.exists()
    assert (memory_dir / "validation" / "aaa.jpg").exists()


def test_a_tar_with_a_link_is_refused_on_an_object_store(memory_dir: UPath) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        link = tarfile.TarInfo(name="links/latest")
        link.type = tarfile.SYMTYPE
        link.linkname = "a.jpg"
        tar.addfile(link)
    archive = memory_dir / "links.tar.gz"
    archive.write_bytes(buffer.getvalue())

    with pytest.raises(NotImplementedError, match="not a regular file"):
        unpack_archive_beside(archive)
