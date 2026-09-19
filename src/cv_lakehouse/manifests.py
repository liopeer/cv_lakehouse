#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The manifest that every layer writes next to its output."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cv_lakehouse.sources.published_files import PublishedFile

BRONZE_MANIFEST = "_bronze.json"
SILVER_MANIFEST = "_silver.json"


class BronzeMode(StrEnum):
    """How a bronze dataset got onto disk."""

    DOWNLOAD = "download"
    LINK = "link"


class BronzeManifest(BaseModel):
    dataset: str
    mode: BronzeMode
    homepage: str
    license: str
    commercial_use: bool
    path: str
    splits: list[str]
    image_roots: dict[str, str] = Field(default_factory=dict)
    # Every file the publisher publishes, with its URL and its checksum. A linked copy
    # lists them too: they define what a complete copy is.
    published_files: list[PublishedFile] = Field(default_factory=list)


class SilverManifest(BaseModel):
    """One normalized dataset.

    A manifest describes the layer, and does not report on the data in it. Anything
    counted or derived from the Parquet -- a class tally, a size histogram, which
    `attr_*` columns a dataset fills -- belongs to whoever reads the layer, and is one
    DuckDB query away.
    """

    dataset: str
    code_version: str
    license: str
    commercial_use: bool
    # Keyed by the source's own split names, which are also the Parquet file stems.
    image_roots: dict[str, str]
    splits: list[str]
    # The model behind the embedding files, or None when the run wrote none.
    embedding_model: str | None = None


def write_manifest(path: Path, manifest: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate(json.loads(path.read_text()))
