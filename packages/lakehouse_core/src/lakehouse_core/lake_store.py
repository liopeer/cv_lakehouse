#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The lake: a root on a local disk or in an object store, and the way to reach it.

Every path in the lake is a `UPath`, so one code path serves a local directory,
`s3://`, `gs://`, `az://` and `memory://`. A local root is a `PosixUPath`, which is a
`pathlib.Path`, so the standard library and every library that takes a path still work
on it.
"""

import os
import pathlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import duckdb
import pyarrow.fs as pafs
from fsspec.core import split_protocol
from upath import UPath

LOCAL_PROTOCOLS = (None, "file", "local")


class LakeStore:
    """Resolve the locations in the lake, and open its Parquet in pyarrow and DuckDB."""

    def __init__(self, *, root: str, storage_options: Mapping[str, Any]) -> None:
        self.storage_options = dict(storage_options)
        self.root = to_upath(location=root, storage_options=self.storage_options)

    @property
    def protocol(self) -> str | None:
        return None if is_local(self.root) else self.root.protocol

    def resolve(self, location: str) -> UPath:
        """Return the path of a location.

        A location is a key relative to the root, an absolute local path, or a URL. A
        URL on the protocol of the root reaches it with the storage options of the root.
        """
        protocol, path = split_protocol(location)
        if protocol is None and not pathlib.PurePosixPath(path).is_absolute():
            return self.root / location
        same_protocol = protocol == self.protocol or (
            protocol in LOCAL_PROTOCOLS and self.protocol is None
        )
        return to_upath(
            location=location,
            storage_options=self.storage_options if same_protocol else {},
        )

    def location(self, path: UPath) -> str:
        """Return the key of a path under the root, or the location of one outside."""
        try:
            return path.relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    @contextmanager
    def duckdb(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Open a DuckDB connection that reads the lake by `str(path)`.

        DuckDB reads S3 with its own httpfs extension, which is much faster than through
        fsspec. Every other remote protocol goes through the fsspec filesystem of the
        root.
        """
        with duckdb.connect(config=_duckdb_config()) as connection:
            if self.protocol in ("s3", "s3a"):
                _configure_s3(
                    connection=connection, storage_options=self.storage_options
                )
            elif self.protocol is not None:
                connection.register_filesystem(self.root.fs)
            yield connection


def _duckdb_config() -> dict[str, bool | float | int | list[str] | str]:
    # An image installs the extensions here, so a run never downloads them, whatever
    # the user that it runs as.
    directory = os.environ.get("DUCKDB_EXTENSION_DIRECTORY")
    return {} if directory is None else {"extension_directory": directory}


def build_reader_locator(root: UPath, *, expires_seconds: int) -> Callable[[str], str]:
    """Map a name under `root` to what a server with no credentials can read.

    That is the local path for a local root, and a presigned URL for a remote one. The
    signature is computed locally, with no request to the store.
    """
    if is_local(root):
        base = str(root).rstrip("/")
        return lambda name: f"{base}/{name}"
    filesystem = root.fs
    base = root.path.rstrip("/")
    return lambda name: filesystem.sign(f"{base}/{name}", expiration=expires_seconds)


def to_upath(*, location: str, storage_options: Mapping[str, Any]) -> UPath:
    """Return a local location as a resolved `PosixUPath`, and any other as `UPath`."""
    protocol, path = split_protocol(location)
    if protocol in LOCAL_PROTOCOLS:
        return UPath(pathlib.Path(path).expanduser().resolve())
    return UPath(location, **storage_options)


def is_local(path: UPath) -> bool:
    """Tell a path that the standard library can open: a `LocalPath`, or any `Path`."""
    return isinstance(path, pathlib.Path)


def arrow_location(path: UPath) -> tuple[str, pafs.FileSystem | None]:
    """Return what pyarrow takes as `where` and `filesystem` for a path."""
    if is_local(path):
        return str(path), None
    return path.path, pafs.PyFileSystem(pafs.FSSpecHandler(path.fs))


def _configure_s3(
    *, connection: duckdb.DuckDBPyConnection, storage_options: Mapping[str, Any]
) -> None:
    client_kwargs = storage_options.get("client_kwargs", {})
    endpoint_url = storage_options.get("endpoint_url") or client_kwargs.get(
        "endpoint_url"
    )
    parameters: dict[str, str] = {
        "region": client_kwargs.get("region_name") or "us-east-1",
    }
    if storage_options.get("key"):
        parameters["key_id"] = storage_options["key"]
        parameters["secret"] = storage_options["secret"]
    if endpoint_url:
        endpoint = urlparse(endpoint_url)
        parameters["endpoint"] = endpoint.netloc
        parameters["use_ssl"] = str(endpoint.scheme == "https").lower()
        # SeaweedFS and MinIO serve no bucket subdomains.
        parameters["url_style"] = "path"
    # A no-op when the extension is installed. Elsewhere this downloads it once.
    connection.install_extension("httpfs")
    connection.load_extension("httpfs")
    options = ", ".join(
        f"{name} '{value.replace("'", "''")}'" for name, value in parameters.items()
    )
    connection.execute(f"create secret lake (type s3, {options})")
