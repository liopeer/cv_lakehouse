#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The manifest that every layer writes next to its output."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from cv_lakehouse.sources.published_files import PublishedFile
from cv_lakehouse.split_roles import SplitRole

BRONZE_MANIFEST = "_bronze.json"
SILVER_MANIFEST = "_silver.json"
GOLD_MANIFEST = "_gold.json"
CORRECTIONS_MANIFEST = "_corrections.json"
RELEASE_MANIFEST = "_release.json"


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


class CorrectionSnapshot(BaseModel):
    snapshot_id: str
    sequence: int
    parent_snapshot_id: str | None
    created_at: datetime
    row_count: int
    # The pin: the size and the checksum that the download verified.
    file: PublishedFile


class CorrectionsManifest(BaseModel):
    """The correction snapshots of one dataset, oldest first.

    Silver applies the last one. The others are history.
    """

    dataset: str
    snapshots: list[CorrectionSnapshot] = Field(default_factory=list)


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
    # The correction snapshot that this silver applied, or None for the bare source.
    correction_snapshot_id: str | None = None


class GoldSplit(BaseModel):
    split: str
    role: SplitRole
    # Relative to the lake root, so gold names a pixel on any machine and in any store.
    image_root: str


class GoldDataset(BaseModel):
    dataset: str
    license: str
    commercial_use: bool
    silver_code_version: str
    # Gold copies no vector. A reader takes them from the silver files of this dataset.
    embedding_model: str | None
    # The correction snapshot that the silver of this dataset applied.
    correction_snapshot_id: str | None = None
    splits: list[GoldSplit]


class GoldManifest(BaseModel):
    """The pointer to the gold version that a reader takes.

    A build writes a whole new version directory and then replaces this file, so a
    reader never sees a version that is half written.
    """

    version: int
    built_at: datetime
    code_version: str
    datasets: list[GoldDataset]


class ReleaseFile(BaseModel):
    # Relative to the release directory.
    path: str
    size: int
    sha256: str


class ReleaseManifest(BaseModel):
    """One frozen copy of the val and test rows of every dataset.

    A benchmark names the release it ran on, so two results on one release compare.
    The manifest says what the release was built from, and pins every file.
    """

    release: int
    created_at: datetime
    gold_version: int
    gold_code_version: str
    # The datasets as gold held them, with only their val and test splits.
    datasets: list[GoldDataset]
    files: list[ReleaseFile]


def write_manifest(path: Path, manifest: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate(json.loads(path.read_text()))
