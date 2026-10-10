#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The manifests that the CV layers write next to their output."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field
from upath import UPath

from lakehouse_core.bronze_manifest import BronzeManifest
from lakehouse_core.published_files import PublishedFile
from lakehouse_cv.contract.split_roles import SplitRole

SILVER_MANIFEST = "_silver.json"
EMBEDDINGS_MANIFEST = "_embeddings.json"
GOLD_MANIFEST = "_gold.json"
CORRECTIONS_MANIFEST = "_corrections.json"
RELEASE_MANIFEST = "_release.json"


class CvBronzeManifest(BronzeManifest):
    """A bronze copy, and the splits and image roots that are on disk."""

    splits: list[str]
    image_roots: dict[str, str] = Field(default_factory=dict)

    def materialization_metadata(self) -> dict[str, str | int | float | list[str]]:
        return {"splits": list(self.splits)}


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


def builds_dir(layer_dir: UPath) -> UPath:
    return layer_dir / "builds"


def build_dir(*, layer_dir: UPath, build_id: str) -> UPath:
    return builds_dir(layer_dir) / build_id


class BuildManifest(BaseModel):
    """The manifest of a layer that writes each run to a new directory."""

    # The directory under `builds/` that holds the files of this run.
    build_id: str


class SilverManifest(BuildManifest):
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
    # The correction snapshot that this silver applied, or None for the bare source.
    correction_snapshot_id: str | None = None


class EmbeddingsManifest(BuildManifest):
    """The vectors of one silver build: one per image, and one per box crop."""

    dataset: str
    code_version: str
    embedding_model: str
    embedding_version: str
    # The silver build whose rows the vectors cover.
    silver_build_id: str
    splits: list[str]


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
    silver_build_id: str
    # The correction snapshot that the silver of this dataset applied.
    correction_snapshot_id: str | None = None
    splits: list[GoldSplit]


class GoldManifest(BuildManifest):
    """The pointer to the gold build that a reader takes.

    Gold has no versions. It is the latest pure function of silver, and a rebuild from
    the same silver gives the same rows.
    """

    # When the run happened. A fact about the run, and not about the data.
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
    The manifest says what the release was built from, and pins every file. An eval
    release is the only frozen copy of gold.
    """

    release: int
    created_at: datetime
    gold_code_version: str
    # The datasets as gold held them, with only their val and test splits. Each names
    # the silver build that it came from.
    datasets: list[GoldDataset]
    files: list[ReleaseFile]
