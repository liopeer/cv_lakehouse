#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fetch the correction snapshots that the LightlyStudio export publishes.

A curator fixes annotations in LightlyStudio. The export of `studio/` publishes the
fixes of one dataset as snapshots. A snapshot is one Parquet file, it never changes,
and it holds every correction that is live when it is made. So the latest snapshot
alone says what to change, and the older ones are history.

The file list is not known before the run, as it is for a dataset. The listing gives
each file a size and a checksum. Bronze verifies them at the download and then pins
them in its manifest. A later listing that disagrees with a pin fails the run.

Each snapshot names the one before it. A LightlyStudio database that was lost starts a
new chain, and bronze rejects it. Otherwise a silver rebuild would silently drop every
correction made so far.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import httpx
import pyarrow as pa
from pydantic import BaseModel

from cv_lakehouse.manifests import CorrectionSnapshot
from cv_lakehouse.sources.published_files import (
    Checksum,
    ChecksumKind,
    PublishedFile,
    compute_checksum,
    download_published_file,
)

CORRECTIONS_FILE = "corrections.parquet"


class CorrectionAction:
    # The class, the box, or both changed. A null field is unchanged.
    UPDATE = "update"
    DELETE = "delete"
    # A box that a curator drew. Its id is the id LightlyStudio gave it.
    ADD = "add"


# One row per corrected box. `label_name` is the label as LightlyStudio holds it. The
# box is in whole pixels, as LightlyStudio stores it, and all four are null together.
CORRECTION_SCHEMA = pa.schema(
    [
        pa.field(name="dataset", type=pa.string(), nullable=False),
        pa.field(name="split", type=pa.string(), nullable=False),
        pa.field(name="file_name", type=pa.string(), nullable=False),
        pa.field(name="box_id", type=pa.string(), nullable=False),
        pa.field(name="action", type=pa.string(), nullable=False),
        pa.field(name="label_name", type=pa.string()),
        pa.field(name="x", type=pa.float64()),
        pa.field(name="y", type=pa.float64()),
        pa.field(name="w", type=pa.float64()),
        pa.field(name="h", type=pa.float64()),
    ]
)


class PublishedSnapshot(BaseModel):
    """One entry of the listing that the export serves."""

    snapshot_id: str
    sequence: int
    parent_snapshot_id: str | None
    created_at: datetime
    row_count: int
    # The path of the file under the export URL.
    path: str
    size: int
    sha256: str


class CorrectionChainError(RuntimeError):
    pass


def list_published_snapshots(
    *, client: httpx.Client, export_url: str, dataset: str
) -> list[PublishedSnapshot]:
    response = client.get(f"{export_url.rstrip('/')}/v1/datasets/{dataset}/snapshots")
    response.raise_for_status()
    snapshots = [PublishedSnapshot.model_validate(entry) for entry in response.json()]
    return sorted(snapshots, key=lambda snapshot: snapshot.sequence)


def fetch_new_snapshots(
    *,
    client: httpx.Client,
    export_url: str,
    dataset: str,
    corrections_dir: Path,
    stored_snapshots: Sequence[CorrectionSnapshot],
    log: logging.Logger,
) -> list[CorrectionSnapshot]:
    """Download every snapshot that bronze does not hold. Return all of them."""
    published = list_published_snapshots(
        client=client, export_url=export_url, dataset=dataset
    )
    reject_changed_snapshots(
        corrections_dir=corrections_dir,
        stored_snapshots=stored_snapshots,
        published_snapshots=published,
    )
    snapshots = list(stored_snapshots)
    for snapshot in published[len(snapshots) :]:
        reject_broken_snapshot_chain(
            snapshot=snapshot, previous=snapshots[-1] if snapshots else None
        )
        published_file = PublishedFile(
            path=f"snapshots/{snapshot.snapshot_id}/{CORRECTIONS_FILE}",
            url=f"{export_url.rstrip('/')}/{snapshot.path.lstrip('/')}",
            size=snapshot.size,
            checksum=Checksum(kind=ChecksumKind.SHA256, value=snapshot.sha256),
        )
        download_published_file(
            published_file=published_file,
            bronze_dir=corrections_dir,
            client=client,
            log=log,
        )
        snapshots.append(
            CorrectionSnapshot(
                snapshot_id=snapshot.snapshot_id,
                sequence=snapshot.sequence,
                parent_snapshot_id=snapshot.parent_snapshot_id,
                created_at=snapshot.created_at,
                row_count=snapshot.row_count,
                file=published_file,
            )
        )
    return snapshots


def reject_changed_snapshots(
    *,
    corrections_dir: Path,
    stored_snapshots: Sequence[CorrectionSnapshot],
    published_snapshots: Sequence[PublishedSnapshot],
) -> None:
    """Fail when a stored snapshot is not what the export and the disk still hold."""
    if len(published_snapshots) < len(stored_snapshots):
        raise CorrectionChainError(
            f"The export lists {len(published_snapshots)} snapshots, and bronze holds "
            f"{len(stored_snapshots)}. The LightlyStudio database lost its history."
        )
    for stored, published in zip(stored_snapshots, published_snapshots, strict=False):
        pin = stored.file.checksum.value
        if (published.snapshot_id, published.sha256) != (stored.snapshot_id, pin):
            raise CorrectionChainError(
                f"Snapshot {stored.sequence} is {stored.snapshot_id} in bronze, and "
                f"the export now lists {published.snapshot_id}."
            )
        path = corrections_dir / stored.file.path
        if compute_checksum(path=path, kind=ChecksumKind.SHA256) != pin:
            raise CorrectionChainError(f"{path} changed after bronze pinned it.")


def reject_broken_snapshot_chain(
    snapshot: PublishedSnapshot, previous: CorrectionSnapshot | None
) -> None:
    expected_parent = None if previous is None else previous.snapshot_id
    expected_sequence = 1 if previous is None else previous.sequence + 1
    if (snapshot.parent_snapshot_id, snapshot.sequence) != (
        expected_parent,
        expected_sequence,
    ):
        raise CorrectionChainError(
            f"Snapshot {snapshot.snapshot_id} has sequence {snapshot.sequence} and "
            f"parent {snapshot.parent_snapshot_id}. Bronze expects sequence "
            f"{expected_sequence} and parent {expected_parent}."
        )
