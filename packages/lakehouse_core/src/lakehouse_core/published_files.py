#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Download the files that a publisher publishes, verify them, and unpack them.

Bronze is a manual download: every file at its published name, every archive unpacked
beside itself. A file only takes its final name after its checksum matches, so a
final name always means a verified file.

On a local disk a download goes to `<name>.partial` first. On S3 it is a multipart
upload that a later run resumes, see `s3_uploads`. On any other object store it
streams to its final name, and a failure deletes it. No dataset is staged on local disk.
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
from upath import UPath

from lakehouse_core.disk_lease import DiskLease
from lakehouse_core.lake_store import is_local
from lakehouse_core.remote_archives import unpack_remote_archive
from lakehouse_core.s3_uploads import ResumableUpload, choose_part_size

ARCHIVE_SUFFIXES = (".tar.gz", ".zip")
PARTIAL_SUFFIX = ".partial"
# Beside an object on S3 whose upload resumed, until a read back verified it.
UNVERIFIED_SUFFIX = ".unverified"
S3_PROTOCOLS = ("s3", "s3a")
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
    bronze_dir: Path | UPath, published_files: Iterable[PublishedFile]
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


class StreamingChecksum:
    """Compute a checksum of any kind over chunks, as they arrive."""

    def __init__(self, kind: ChecksumKind) -> None:
        self._kind = kind
        self._digest = (
            hashlib.sha256() if kind is ChecksumKind.SHA256 else hashlib.md5()
        )
        self._part_digests: list[bytes] = []
        self._part_size = 0

    def update(self, chunk: bytes) -> None:
        if self._kind is not ChecksumKind.S3_ETAG:
            self._digest.update(chunk)
            return
        view = memoryview(chunk)
        while view:
            taken = view[: S3_PART_SIZE - self._part_size]
            self._digest.update(taken)
            self._part_size += len(taken)
            view = view[len(taken) :]
            if self._part_size == S3_PART_SIZE:
                self._part_digests.append(self._digest.digest())
                self._digest = hashlib.md5()
                self._part_size = 0

    def value(self) -> str:
        if self._kind is ChecksumKind.SHA256:
            return self._digest.hexdigest()
        if self._kind is ChecksumKind.MD5_BASE64:
            return base64.b64encode(self._digest.digest()).decode()
        part_digests = list(self._part_digests)
        if self._part_size or not part_digests:
            part_digests.append(self._digest.digest())
        if len(part_digests) == 1:
            return part_digests[0].hex()
        combined = hashlib.md5(b"".join(part_digests)).hexdigest()
        return f"{combined}-{len(part_digests)}"


def compute_checksum(path: Path | UPath, kind: ChecksumKind) -> str:
    checksum = StreamingChecksum(kind)
    for chunk in _read_chunks(path=path, size=READ_CHUNK_SIZE):
        checksum.update(chunk)
    return checksum.value()


def download_published_files(
    *,
    published_files: Sequence[PublishedFile],
    bronze_dir: Path | UPath,
    workers: int,
    timeout: float,
    disk_lease: DiskLease,
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
                disk_lease=disk_lease,
                log=log,
            ),
            published_files,
        )
        # Consume the results, so the first failure raises here.
        list(results)


def download_published_file(
    *,
    published_file: PublishedFile,
    bronze_dir: Path | UPath,
    client: httpx.Client,
    disk_lease: DiskLease,
    log: logging.Logger,
) -> Path | UPath:
    """Download one file, resuming a partial download, and verify it."""
    if isinstance(bronze_dir, Path) or is_local(bronze_dir):
        return _download_to_disk(
            published_file=published_file,
            bronze_dir=Path(str(bronze_dir)),
            client=client,
            disk_lease=disk_lease,
            log=log,
        )
    target = bronze_dir / published_file.path
    if target.protocol in S3_PROTOCOLS:
        _download_to_s3(
            published_file=published_file, target=target, client=client, log=log
        )
    else:
        _download_to_object(
            published_file=published_file, target=target, client=client, log=log
        )
    return target


def _download_to_disk(
    *,
    published_file: PublishedFile,
    bronze_dir: Path,
    client: httpx.Client,
    disk_lease: DiskLease,
    log: logging.Logger,
) -> Path:
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
    with disk_lease.hold(
        target=partial, reason=f"Verifying {published_file.path}", log=log
    ):
        _verify(path=partial, published_file=published_file)
    partial.rename(target)
    return target


def unpack_archive_beside(archive: Path | UPath) -> Path | UPath:
    """Unpack an archive into a directory beside it, named after the archive.

    An archive that wraps everything in one directory of that name is not doubled,
    as a desktop unarchiver does. The archive unpacks into a staging directory that
    takes the final name at the end, so a crash leaves no half tree behind. On an
    object store, which has no rename, a marker beside the archive says the same.
    """
    if not isinstance(archive, Path) and not is_local(archive):
        return unpack_remote_archive(
            archive=archive,
            unpacked=archive.with_name(_unpacked_name(archive.name)),
            skipped_prefix=MACOS_METADATA_DIR,
        )
    archive = Path(str(archive))
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
    bronze_dir: Path | UPath,
    client: httpx.Client,
    disk_lease: DiskLease,
    log: logging.Logger,
) -> None:
    path = download_published_file(
        published_file=published_file,
        bronze_dir=bronze_dir,
        client=client,
        disk_lease=disk_lease,
        log=log,
    )
    if published_file.unpacked_path is not None:
        reason = f"Unpacking {published_file.path}"
        log.info(f"{reason}, unless it is unpacked")
        with disk_lease.hold(target=path, reason=reason, log=log):
            unpack_archive_beside(path)


def _download_to_s3(
    *,
    published_file: PublishedFile,
    target: UPath,
    client: httpx.Client,
    log: logging.Logger,
) -> None:
    """Stream into a multipart upload that resumes, and verify before it completes.

    A resumed upload holds bytes that this process never saw, so its checksum cannot be
    computed on the stream. It completes beside a marker, and a read back verifies it.
    A run that dies before the read back finds the marker, and verifies then.
    """
    marker = target.with_name(target.name + UNVERIFIED_SUFFIX)
    if target.exists():
        if marker.exists():
            _verify_object(path=target, marker=marker, published_file=published_file)
            return
        if target.stat().st_size == published_file.size:
            return
        target.unlink()
    upload = ResumableUpload(
        target=target, part_size=choose_part_size(published_file.size)
    )
    offset = upload.resume_or_start()
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    log.info(f"Downloading {published_file.url} from byte {offset}")
    with client.stream(
        method="GET", url=published_file.url, headers=headers
    ) as response:
        response.raise_for_status()
        if offset and response.status_code != httpx.codes.PARTIAL_CONTENT:
            # The server sends the whole file. The parts already sent are useless.
            offset = upload.restart()
        checksum = None if offset else StreamingChecksum(published_file.checksum.kind)
        buffer = bytearray()
        for chunk in response.iter_raw():
            if checksum is not None:
                checksum.update(chunk)
            buffer += chunk
            while len(buffer) >= upload.part_size:
                upload.upload_part(bytes(buffer[: upload.part_size]))
                del buffer[: upload.part_size]
    size = upload.offset + len(buffer)
    if size > published_file.size:
        upload.abort()
    if size != published_file.size:
        # A short stream keeps its whole parts, and the next run resumes after them.
        raise ChecksumMismatchError(
            f"{published_file.url} gave {size} bytes, expected {published_file.size}"
        )
    if buffer:
        upload.upload_part(bytes(buffer))
    if checksum is not None:
        actual = checksum.value()
        if actual != published_file.checksum.value:
            upload.abort()
            _raise_mismatch(published_file=published_file, actual=actual)
        upload.complete()
        return
    marker.write_bytes(b"")
    upload.complete()
    _verify_object(path=target, marker=marker, published_file=published_file)


def _download_to_object(
    *,
    published_file: PublishedFile,
    target: UPath,
    client: httpx.Client,
    log: logging.Logger,
) -> None:
    """Stream to the final name, and delete it unless it verifies. Nothing resumes."""
    if target.exists() and target.stat().st_size == published_file.size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum = StreamingChecksum(published_file.checksum.kind)
    size = 0
    log.info(f"Downloading {published_file.url}")
    try:
        with (
            client.stream(
                method="GET",
                url=published_file.url,
                headers={"Accept-Encoding": "identity"},
            ) as response,
            target.open(mode="wb") as handle,
        ):
            response.raise_for_status()
            for chunk in response.iter_raw():
                checksum.update(chunk)
                handle.write(chunk)
                size += len(chunk)
        if size != published_file.size:
            raise ChecksumMismatchError(
                f"{published_file.url} gave {size} bytes, "
                f"expected {published_file.size}"
            )
        actual = checksum.value()
        if actual != published_file.checksum.value:
            _raise_mismatch(published_file=published_file, actual=actual)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _verify_object(path: UPath, marker: UPath, published_file: PublishedFile) -> None:
    actual = compute_checksum(path=path, kind=published_file.checksum.kind)
    if actual != published_file.checksum.value:
        path.unlink()
        marker.unlink()
        _raise_mismatch(published_file=published_file, actual=actual)
    marker.unlink()


def _raise_mismatch(published_file: PublishedFile, actual: str) -> None:
    raise ChecksumMismatchError(
        f"{published_file.url} has {published_file.checksum.kind} {actual}, "
        f"expected {published_file.checksum.value}"
    )


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


def _read_chunks(path: Path | UPath, size: int) -> Iterator[bytes]:
    with path.open(mode="rb") as handle:
        while chunk := handle.read(size):
            yield chunk


def _unpacked_dir(archive: Path) -> Path:
    return archive.with_name(_unpacked_name(archive.name))


def _unpacked_name(archive_name: str) -> str:
    for suffix in ARCHIVE_SUFFIXES:
        if archive_name.endswith(suffix):
            return archive_name.removesuffix(suffix)
    raise ValueError(f"Not an archive: {archive_name}")


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
