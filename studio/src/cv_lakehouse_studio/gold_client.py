#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Read gold over the HTTP API of the lakehouse.

This package shares no code with `cv_lakehouse`. It reads what the API serves, as any
consumer does, so the two images release independently.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Protocol

import httpx
import pyarrow as pa
from pydantic import BaseModel

ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
NEXT_AFTER_HEADER = "X-Next-After"
ROWS_PER_PAGE = 50_000


class GoldSplit(BaseModel):
    split: str
    role: str


class GoldDataset(BaseModel):
    dataset: str
    embedding_model: str | None
    splits: list[GoldSplit]


class GoldMeta(BaseModel):
    """The part of the gold manifest that the sync reads."""

    version: int
    built_at: datetime
    datasets: list[GoldDataset]


class GoldClient(Protocol):
    """What the sync needs from gold. A test substitutes its own."""

    def read_meta(self) -> GoldMeta: ...

    def read_class_names(self) -> list[str]: ...

    def iter_pages(
        self, *, table: str, dataset: str, changed_since: datetime | None = None
    ) -> Iterator[pa.Table]: ...


class HttpGoldClient(GoldClient):
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, transport=transport
        )

    def read_meta(self) -> GoldMeta:
        return GoldMeta.model_validate(self._get_json("/v1/meta"))

    def read_class_names(self) -> list[str]:
        return [entry["class_name"] for entry in self._get_json("/v1/classes")]

    def iter_pages(
        self, *, table: str, dataset: str, changed_since: datetime | None = None
    ) -> Iterator[pa.Table]:
        # The curators work on the gold of now, not on a frozen release.
        params: dict[str, str | int] = {
            "dataset": dataset,
            "release": "draft",
            "limit": ROWS_PER_PAGE,
        }
        if changed_since is not None:
            params["changed_since"] = changed_since.isoformat()
        while True:
            response = self._client.get(
                url=f"/v1/{table}",
                params=params,
                headers={"Accept": ARROW_STREAM_MEDIA_TYPE},
            )
            response.raise_for_status()
            yield pa.ipc.open_stream(response.content).read_all()
            after = response.headers.get(NEXT_AFTER_HEADER)
            if after is None:
                return
            params["after"] = after

    def _get_json(self, path: str):
        response = self._client.get(path)
        response.raise_for_status()
        return response.json()
