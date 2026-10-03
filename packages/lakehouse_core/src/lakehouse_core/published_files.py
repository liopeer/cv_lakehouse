#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Download the files that a publisher publishes, verify them, and unpack them.

Bronze is a manual download: every file at its published name, every archive unpacked
beside itself. A file only takes its final name after its checksum matches, so a
final name always means a verified file.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import shutil
import tarfile
import time
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

ARCHIVE_SUFFIXES = (".tar.gz", ".zip")
PARTIAL_SUFFIX = ".partial"
# Every Open Images tar was uploaded in parts of this size. The part count in each
# ETag confirms it.
S3_PART_SIZE = 8 * 1024 * 1024
READ_CHUNK_SIZE = 1024 * 1024
# A zip made on a Mac carries Finder resource forks here. They are not the dataset,
# macOS never unpacks them, and the kept archive still holds them.
MACOS_METADATA_DIR = "__MACOSX/"


class ChecksumKind(StrEnum):
    SHA256 = "sha256"
    # The form that the `x-goog-hash` header of Google Cloud Storage gives.
    MD5_BASE64 = "md5_base64"
    # The ETag of an S3 object: an md5, or the md5 of the part md5s with `-<parts>`.
    S3_ETAG = "s3_etag"


class Checksum(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: ChecksumKind
    value: str


class PublishedFile(BaseModel):
    """One file as its publisher publishes it. `path` is relative to bronze."""

    model_config = ConfigDict(frozen=True)

    path: str
    url: str
    size: int
    checksum: Checksum

    @property
    def unpacked_path(self) -> str | None:
        """The directory that the archive unpacks to, or None for a plain file."""
        for suffix in ARCHIVE_SUFFIXES:
            if self.path.endswith(suffix):
                return self.path.removesuffix(suffix)
        return None


class ChecksumMismatchError(ValueError):
    pass


def list_required_paths(published_files: Iterable[PublishedFile]) -> list[str]:
    """List what a complete copy holds: every unpacked archive and every plain file.

    The archives themselves are not listed, so a copy that deleted them still links.
    """
    return [
        published_file.unpacked_path or published_file.path
        for published_file in published_files
    ]


def reject_incomplete_copy(
    bronze_dir: Path, published_files: Iterable[PublishedFile]
) -> None:
    missing = [
        path
        for path in list_required_paths(published_files)
        if not (bronze_dir / path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{bronze_dir} is not a complete copy. Missing: {', '.join(missing)}"
        )


def compute_checksum(path: Path, kind: ChecksumKind) -> str:
    if kind is ChecksumKind.S3_ETAG:
        return _compute_s3_etag(path)
    digest = hashlib.sha256() if kind is ChecksumKind.SHA256 else hashlib.md5()
    for chunk in _read_chunks(path=path, size=READ_CHUNK_SIZE):
        digest.update(chunk)
    if kind is ChecksumKind.SHA256:
        return digest.hexdigest()
    return base64.b64encode(digest.digest()).decode()


def download_published_files(
    *,
    published_files: Sequence[PublishedFile],
    bronze_dir: Path,
    workers: int,
    timeout: float,
    log: logging.Logger,
) -> None:
    """Download, verify and unpack every file, several at a time."""
    with (
        httpx.Client(timeout=timeout, follow_redirects=True) as client,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        results = pool.map(
            lambda published_file: _download_and_unpack(
                published_file=published_file,
                bronze_dir=bronze_dir,
                client=client,
                log=log,
            ),
            published_files,
        )
        # Consume the results, so the first failure raises here.
        list(results)


def download_published_file(
    *,
    published_file: PublishedFile,
    bronze_dir: Path,
    client: httpx.Client,
    log: logging.Logger,
) -> Path:
    """Download one file, resuming a partial download, and verify it."""
    target = bronze_dir / published_file.path
    if target.is_file() and target.stat().st_size == published_file.size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > published_file.size:
        partial.unlink()
        offset = 0
    if offset < published_file.size:
        log.info(f"Downloading {published_file.url} from byte {offset}")
        _stream_to_file(
            url=published_file.url, partial=partial, offset=offset, client=client
        )
    _verify(path=partial, published_file=published_file)
    partial.rename(target)
    return target


def unpack_archive_beside(archive: Path) -> Path:
    """Unpack an archive into a directory beside it, named after the archive.

    An archive that wraps everything in one directory of that name is not doubled,
    as a desktop unarchiver does. The archive unpacks into a staging directory that
    takes the final name at the end, so a crash leaves no half tree behind.
    """
    unpacked = _unpacked_dir(archive)
    if unpacked.exists():
        return unpacked
    staging = archive.parent / f".{unpacked.name}.unpacking"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    if archive.name.endswith(".zip"):
        _unzip_keeping_dates(archive=archive, dest=staging)
    else:
        with tarfile.open(archive) as tar:
            tar.extractall(path=staging, filter="data")
    entries = list(staging.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == unpacked.name:
        entries[0].rename(unpacked)
        staging.rmdir()
    else:
        staging.rename(unpacked)
    return unpacked


def _download_and_unpack(
    *,
    published_file: PublishedFile,
    bronze_dir: Path,
    client: httpx.Client,
    log: logging.Logger,
) -> None:
    path = download_published_file(
        published_file=published_file, bronze_dir=bronze_dir, client=client, log=log
    )
    unpacked_path = published_file.unpacked_path
    if unpacked_path is not None and not (bronze_dir / unpacked_path).exists():
        log.info(f"Unpacking {published_file.path}")
        unpack_archive_beside(path)


def _stream_to_file(
    *, url: str, partial: Path, offset: int, client: httpx.Client
) -> None:
    # Ask for the bytes as stored, so no transfer encoding changes the checksum.
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with client.stream(method="GET", url=url, headers=headers) as response:
        response.raise_for_status()
        resumed = offset > 0 and response.status_code == httpx.codes.PARTIAL_CONTENT
        with partial.open(mode="ab" if resumed else "wb") as handle:
            for chunk in response.iter_raw():
                handle.write(chunk)


def _verify(path: Path, published_file: PublishedFile) -> None:
    size = path.stat().st_size
    if size != published_file.size:
        # A short file stays, so the next run resumes it.
        if size > published_file.size:
            path.unlink()
        raise ChecksumMismatchError(
            f"{published_file.url} gave {size} bytes, expected {published_file.size}"
        )
    kind = published_file.checksum.kind
    actual = compute_checksum(path=path, kind=kind)
    if actual != published_file.checksum.value:
        path.unlink()
        raise ChecksumMismatchError(
            f"{published_file.url} has {kind} {actual}, "
            f"expected {published_file.checksum.value}"
        )


def _read_chunks(path: Path, size: int) -> Iterator[bytes]:
    with path.open(mode="rb") as handle:
        while chunk := handle.read(size):
            yield chunk


def _compute_s3_etag(path: Path) -> str:
    part_digests = [
        hashlib.md5(part).digest()
        for part in _read_chunks(path=path, size=S3_PART_SIZE)
    ]
    if len(part_digests) == 1:
        return part_digests[0].hex()
    combined = hashlib.md5(b"".join(part_digests)).hexdigest()
    return f"{combined}-{len(part_digests)}"


def _unpacked_dir(archive: Path) -> Path:
    for suffix in ARCHIVE_SUFFIXES:
        if archive.name.endswith(suffix):
            return archive.with_name(archive.name.removesuffix(suffix))
    raise ValueError(f"Not an archive: {archive}")


def _unzip_keeping_dates(archive: Path, dest: Path) -> None:
    """Unzip, then give every entry the date that the zip stores for it.

    `zipfile` sets no date. A directory gets its date last, deepest first, because
    writing a file into it changes it.
    """
    with zipfile.ZipFile(archive) as zipped:
        members = [
            info
            for info in zipped.infolist()
            if not info.filename.startswith(MACOS_METADATA_DIR)
        ]
        zipped.extractall(path=dest, members=members)
        for info in sorted(
            members,
            key=lambda info: (info.is_dir(), -info.filename.count("/")),
        ):
            stamp = time.mktime((*info.date_time, 0, 0, -1))
            os.utime(path=dest / info.filename, times=(stamp, stamp))
