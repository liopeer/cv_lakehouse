#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The export of `studio/`, in memory."""

import hashlib
import uuid
from datetime import UTC, datetime

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from lakehouse_cv.contract.correction_actions import EVENT_SCHEMA

EXPORT_URL = "http://studio-sync:8002"


def serialize_events(rows: list[dict]) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table=pa.Table.from_pylist(mapping=rows, schema=EVENT_SCHEMA), where=sink
    )
    return sink.getvalue().to_pybytes()


class FakeExportServer:
    """Publish event files as the export does, and serve them to an httpx client.

    A row that names no `log_sequence` gets the next entry of the log.
    """

    def __init__(self) -> None:
        self.chain_id = uuid.uuid4().hex
        self.listings: dict[str, list[dict]] = {}
        self.files: dict[str, bytes] = {}
        self.log_sequence = 0

    def publish(self, dataset: str, rows: list[dict]) -> str:
        """Add one event file that holds these rows. Return its id."""
        listing = self.listings.setdefault(dataset, [])
        after = listing[-1]["last_log_sequence"] if listing else 0
        logged = []
        for row in rows:
            if "log_sequence" not in row:
                self.log_sequence += 1
                row = {**row, "log_sequence": self.log_sequence}
            logged.append(row)
        content = serialize_events(logged)
        sha256 = hashlib.sha256(content).hexdigest()
        sequence = len(listing) + 1
        event_file_id = f"{sequence:06d}-{sha256[:12]}"
        path = f"v1/datasets/{dataset}/events/{event_file_id}/events.parquet"
        listing.append(
            {
                "event_file_id": event_file_id,
                "sequence": sequence,
                "parent_event_file_id": (
                    listing[-1]["event_file_id"] if listing else None
                ),
                "after_log_sequence": after,
                "last_log_sequence": max(
                    [after, self.log_sequence, *(row["log_sequence"] for row in logged)]
                ),
                "created_at": datetime(2026, 1, sequence, tzinfo=UTC).isoformat(),
                "row_count": len(logged),
                "path": path,
                "size": len(content),
                "sha256": sha256,
            }
        )
        self.files[f"/{path}"] = content
        return event_file_id

    def start_new_chain(self) -> None:
        """Act as a new LightlyStudio database, with a log of its own."""
        self.chain_id = uuid.uuid4().hex
        self.listings.clear()
        self.log_sequence = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/events"):
            dataset = path.split("/")[3]
            return httpx.Response(
                status_code=200,
                json={
                    "chain_id": self.chain_id,
                    "event_files": self.listings.get(dataset, []),
                },
            )
        return httpx.Response(
            status_code=200, stream=httpx.ByteStream(self.files[path])
        )

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handle))
