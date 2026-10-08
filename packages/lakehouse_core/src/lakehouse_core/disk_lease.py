#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Let one unit of heavy I/O at a time use a local disk. See ADR 0012.

The lease is a `flock` on a file that every run on the host opens. The kernel releases
it when a process dies. A lock belongs to one open file, so two threads of one run
also wait for each other.
"""

import fcntl
import logging
import os
import socket
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from upath import UPath

from lakehouse_core.lake_store import is_local

POLL_SECONDS = 0.5
WAIT_LOG_SECONDS = 60.0


class DiskLease:
    """Hold the disk for one unit of work, such as one archive or one split.

    A lease with no path does nothing. A deployment sets the path only for a disk that
    suffers from contention, such as an HDD. S3 and an SSD set none.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path

    @contextmanager
    def hold(
        self, *, target: Path | UPath, reason: str, log: logging.Logger
    ) -> Iterator[None]:
        """Hold the lease while the work reads or writes `target`.

        A remote target takes no lease, because an object store handles concurrent
        requests.
        """
        if self.path is None or not (isinstance(target, Path) or is_local(target)):
            yield
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open(mode="a+") as handle:
            _wait_for_lock(handle=handle, path=self.path, reason=reason, log=log)
            handle.truncate(0)
            handle.write(_describe_holder(reason))
            handle.flush()
            try:
                yield
            finally:
                handle.truncate(0)
                fcntl.flock(handle, fcntl.LOCK_UN)


NO_DISK_LEASE = DiskLease(None)


def _wait_for_lock(
    *, handle: TextIO, path: Path, reason: str, log: logging.Logger
) -> None:
    last_log: float | None = None
    while True:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            now = time.monotonic()
            if last_log is None or now - last_log >= WAIT_LOG_SECONDS:
                holder = path.read_text().strip() or "an unknown holder"
                log.info(f"{reason} waits for the disk, held by {holder}")
                last_log = now
            time.sleep(POLL_SECONDS)


def _describe_holder(reason: str) -> str:
    since = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return f"{socket.gethostname()} pid {os.getpid()} since {since}: {reason}\n"
