#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The export of `studio/`, in memory."""

import hashlib
from datetime import UTC, datetime

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from cv_lakehouse.sources.correction_snapshots import CORRECTION_SCHEMA

EXPORT_URL = "http://studio-sync:8002"


def serialize_corrections(rows: list[dict]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table=pa.Table.from_pylist(mapping=rows, schema=CORRECTION_SCHEMA), where=sink
    )
    return sink.getvalue().to_pybytes()


class FakeExportServer:
    """Publish snapshots as the export does, and serve them to an httpx client."""

    def __init__(self) -> None:
        self.listings: dict[str, list[dict]] = {}
        self.files: dict[str, bytes] = {}

    def publish(self, dataset: str, rows: list[dict]) -> str:
        """Add one snapshot that holds these rows. Return its id."""
        listing = self.listings.setdefault(dataset, [])
        content = serialize_corrections(rows)
        sha256 = hashlib.sha256(content).hexdigest()
        sequence = len(listing) + 1
        snapshot_id = f"{sequence:06d}-{sha256[:12]}"
        path = f"v1/datasets/{dataset}/snapshots/{snapshot_id}/corrections.parquet"
        listing.append(
            {
                "snapshot_id": snapshot_id,
                "sequence": sequence,
                "parent_snapshot_id": listing[-1]["snapshot_id"] if listing else None,
                "created_at": datetime(2026, 1, sequence, tzinfo=UTC).isoformat(),
                "row_count": len(rows),
                "path": path,
                "size": len(content),
                "sha256": sha256,
            }
        )
        self.files[f"/{path}"] = content
        return snapshot_id

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/snapshots"):
            dataset = path.split("/")[3]
            return httpx.Response(status_code=200, json=self.listings.get(dataset, []))
        return httpx.Response(
            status_code=200, stream=httpx.ByteStream(self.files[path])
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))
