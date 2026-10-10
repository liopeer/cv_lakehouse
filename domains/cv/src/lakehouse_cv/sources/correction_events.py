#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fetch the event files of curator edits that the LightlyStudio export publishes.

The LightlyStudio database logs every edit of a curator. The export of `studio/` writes
the edited boxes of a part of that log into one event file, which never changes. Each
file names the range of the log that it covers, and the file before it.

The file list is not known before the run, as it is for a dataset. The listing gives
each file a size and a checksum. Bronze verifies them at the download and then pins
them in its manifest. A later listing that disagrees with a pin fails the run.

The files of one database form a chain. A lost database starts a new chain under a new
id. Bronze keeps the old chain, and silver folds both, so no edit is lost.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime

import httpx
from pydantic import BaseModel
from upath import UPath

from lakehouse_core.disk_lease import NO_DISK_LEASE
from lakehouse_core.published_files import (
    Checksum,
    ChecksumKind,
    PublishedFile,
    compute_checksum,
    download_published_file,
)
from lakehouse_cv.contract.manifests import CorrectionEventFile

EVENTS_FILE = "events.parquet"


class PublishedEventFile(BaseModel):
    """One entry of the listing that the export serves."""

    event_file_id: str
    sequence: int
    parent_event_file_id: str | None
    after_log_sequence: int
    last_log_sequence: int
    created_at: datetime
    row_count: int
    # The path of the file under the export URL.
    path: str
    size: int
    sha256: str


class EventListing(BaseModel):
    """The event files of the chain of the database that the export reads now."""

    chain_id: str
    event_files: list[PublishedEventFile]


class EventChainError(RuntimeError):
    pass


def read_event_listing(
    *, client: httpx.Client, export_url: str, dataset: str
) -> EventListing:
    response = client.get(f"{export_url.rstrip('/')}/v1/datasets/{dataset}/events")
    response.raise_for_status()
    listing = EventListing.model_validate(response.json())
    listing.event_files.sort(key=lambda published: published.sequence)
    return listing


def fetch_new_event_files(
    *,
    client: httpx.Client,
    export_url: str,
    dataset: str,
    corrections_dir: UPath,
    stored_files: Sequence[CorrectionEventFile],
    log: logging.Logger,
) -> list[CorrectionEventFile]:
    """Download every event file that bronze does not hold. Return all of them."""
    listing = read_event_listing(client=client, export_url=export_url, dataset=dataset)
    reject_changed_event_files(
        corrections_dir=corrections_dir, stored_files=stored_files, listing=listing
    )
    event_files = list(stored_files)
    chain = [stored for stored in stored_files if stored.chain_id == listing.chain_id]
    for event_file in listing.event_files[len(chain) :]:
        reject_broken_event_chain(
            event_file=event_file, previous=chain[-1] if chain else None
        )
        published_file = PublishedFile(
            path=f"events/{listing.chain_id}/{event_file.event_file_id}/{EVENTS_FILE}",
            url=f"{export_url.rstrip('/')}/{event_file.path.lstrip('/')}",
            size=event_file.size,
            checksum=Checksum(kind=ChecksumKind.SHA256, value=event_file.sha256),
        )
        download_published_file(
            published_file=published_file,
            bronze_dir=corrections_dir,
            client=client,
            # An event file is one small Parquet file, so its checksum needs no lease.
            disk_lease=NO_DISK_LEASE,
            log=log,
        )
        stored = CorrectionEventFile(
            chain_id=listing.chain_id,
            event_file_id=event_file.event_file_id,
            sequence=event_file.sequence,
            parent_event_file_id=event_file.parent_event_file_id,
            after_log_sequence=event_file.after_log_sequence,
            last_log_sequence=event_file.last_log_sequence,
            created_at=event_file.created_at,
            row_count=event_file.row_count,
            file=published_file,
        )
        chain.append(stored)
        event_files.append(stored)
    return event_files


def reject_changed_event_files(
    *,
    corrections_dir: UPath,
    stored_files: Sequence[CorrectionEventFile],
    listing: EventListing,
) -> None:
    """Fail when a stored file is not what the export and the disk still hold.

    A stored file of another chain comes from a database that is gone. The export no
    longer lists it, and bronze keeps it.
    """
    chain = [stored for stored in stored_files if stored.chain_id == listing.chain_id]
    if len(listing.event_files) < len(chain):
        raise EventChainError(
            f"The export lists {len(listing.event_files)} event files, and bronze "
            f"holds {len(chain)} of the same chain. The database lost its log."
        )
    for stored, published in zip(chain, listing.event_files, strict=False):
        pin = stored.file.checksum.value
        if (published.event_file_id, published.sha256) != (stored.event_file_id, pin):
            raise EventChainError(
                f"Event file {stored.sequence} is {stored.event_file_id} in bronze, "
                f"and the export now lists {published.event_file_id}."
            )
    for stored in stored_files:
        path = corrections_dir / stored.file.path
        if (
            compute_checksum(path=path, kind=ChecksumKind.SHA256)
            != stored.file.checksum.value
        ):
            raise EventChainError(f"{path} changed after bronze pinned it.")


def reject_broken_event_chain(
    *, event_file: PublishedEventFile, previous: CorrectionEventFile | None
) -> None:
    expected_parent = None if previous is None else previous.event_file_id
    expected_sequence = 1 if previous is None else previous.sequence + 1
    expected_after = None if previous is None else previous.last_log_sequence
    if (event_file.parent_event_file_id, event_file.sequence) != (
        expected_parent,
        expected_sequence,
    ) or (
        expected_after is not None and event_file.after_log_sequence != expected_after
    ):
        raise EventChainError(
            f"Event file {event_file.event_file_id} has sequence "
            f"{event_file.sequence}, parent {event_file.parent_event_file_id} and "
            "starts after log entry "
            f"{event_file.after_log_sequence}. Bronze expects sequence "
            f"{expected_sequence}, parent {expected_parent}, and log entry "
            f"{expected_after}."
        )
