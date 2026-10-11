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
from typing import Protocol

import httpx
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
NEXT_AFTER_HEADER = "X-Next-After"
EMBEDDING_TABLES = frozenset({"embeddings", "crop_embeddings"})


class GoldSplit(BaseModel):
    split: str
    role: str


class EventMarker(BaseModel):
    """The last log entry of a LightlyStudio database that gold holds."""

    chain_id: str
    log_sequence: int


class GoldDataset(BaseModel):
    dataset: str
    embedding_model: str | None
    splits: list[GoldSplit]
    last_event: EventMarker | None = None


class GoldSlice(BaseModel):
    """The rows of one dataset, or of one split of it, or of one shard of that split."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    split: str | None = None
    shard: int | None = None
    num_shards: int | None = None

    def to_params(self) -> dict[str, str | int]:
        return {
            key: value for key, value in self.model_dump().items() if value is not None
        }


class GoldMeta(BaseModel):
    """The part of the gold manifest that the sync reads."""

    build_id: str
    datasets: list[GoldDataset]


class GoldClient(Protocol):
    """What the sync needs from gold. A test substitutes its own."""

    def read_meta(self) -> GoldMeta: ...

    def read_class_names(self) -> list[str]: ...

    def count_images(self, gold_slice: GoldSlice) -> int: ...

    def iter_pages(
        self, *, table: str, gold_slice: GoldSlice
    ) -> Iterator[pa.Table]: ...


class HttpGoldClient(GoldClient):
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        rows_per_page: int,
        embedding_rows_per_page: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout_seconds, transport=transport
        )
        self._rows_per_page = rows_per_page
        self._embedding_rows_per_page = embedding_rows_per_page

    def read_meta(self) -> GoldMeta:
        return GoldMeta.model_validate(self._get_json("/v1/meta"))

    def read_class_names(self) -> list[str]:
        return [entry["class_name"] for entry in self._get_json("/v1/classes")]

    def count_images(self, gold_slice: GoldSlice) -> int:
        response = self._client.get(
            url="/v1/images/count",
            params={**gold_slice.to_params(), "release": "draft"},
        )
        response.raise_for_status()
        return response.json()["count"]

    def iter_pages(self, *, table: str, gold_slice: GoldSlice) -> Iterator[pa.Table]:
        # The curators work on the gold of now, not on a frozen release.
        params: dict[str, str | int] = {
            **gold_slice.to_params(),
            "release": "draft",
            "limit": self._embedding_rows_per_page
            if table in EMBEDDING_TABLES
            else self._rows_per_page,
        }
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
