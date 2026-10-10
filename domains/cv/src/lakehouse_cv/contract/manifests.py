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


class CorrectionEventFile(BaseModel):
    # The LightlyStudio database that logged the events. A new database starts a new
    # chain.
    chain_id: str
    event_file_id: str
    # The position of the file in its chain, from 1.
    sequence: int
    parent_event_file_id: str | None
    # The file covers the log entries after the one and up to the other.
    after_log_sequence: int
    last_log_sequence: int
    created_at: datetime
    row_count: int
    # The pin: the size and the checksum that the download verified.
    file: PublishedFile


class EventMarker(BaseModel):
    """The last log entry of a LightlyStudio database that a layer holds."""

    chain_id: str
    log_sequence: int


class CorrectionsManifest(BaseModel):
    """The curator edits of one dataset, in the order that bronze took them.

    Silver folds every event file. The snapshots of ADR 0009 are history, and silver
    reads them no longer.
    """

    dataset: str
    snapshots: list[CorrectionSnapshot] = Field(default_factory=list)
    event_files: list[CorrectionEventFile] = Field(default_factory=list)

    @property
    def last_event(self) -> EventMarker | None:
        if not self.event_files:
            return None
        last = self.event_files[-1]
        return EventMarker(chain_id=last.chain_id, log_sequence=last.last_log_sequence)


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
    # The last curator edit that this silver folded, or None for the bare source.
    last_event: EventMarker | None = None


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
    # The last curator edit that gold holds. The sync leaves a box with a later edit.
    last_event: EventMarker | None = None
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
