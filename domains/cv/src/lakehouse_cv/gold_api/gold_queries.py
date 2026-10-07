#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Read one page of gold rows with DuckDB.

The manifest names the role of every file, so a filter on the dataset, the split or
the role selects files and never opens the others. Every value that a caller sends is
a bound parameter.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import pyarrow as pa
from upath import UPath

from lakehouse_core.lake_store import LakeStore
from lakehouse_cv.contract.gold_tables import (
    GOLD_BOX_SCHEMA,
    GOLD_CROP_EMBEDDING_SCHEMA,
    GOLD_EMBEDDING_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    gold_boxes_file,
    gold_images_file,
)
from lakehouse_cv.contract.manifests import (
    GoldDataset,
    GoldManifest,
    GoldSplit,
    ReleaseManifest,
)
from lakehouse_cv.contract.silver_tables import crop_embeddings_file, embeddings_file
from lakehouse_cv.gold_api.row_filters import BoxFilter, ImageFilter, ImageSelection
from lakehouse_cv.settings import CvLakePaths


class GoldTable(StrEnum):
    IMAGES = "images"
    BOXES = "boxes"
    EMBEDDINGS = "embeddings"
    CROP_EMBEDDINGS = "crop_embeddings"


@dataclass(frozen=True)
class _TableLayout:
    schema: pa.Schema
    key: str
    # Whether the gold rows are the boxes. Otherwise they are the images.
    reads_boxes: bool
    # The silver embedding rows join onto the gold rows on these columns.
    embedding_join: tuple[str, ...] | None = None


_LAYOUTS = {
    GoldTable.IMAGES: _TableLayout(
        schema=GOLD_IMAGE_SCHEMA, key="image_id", reads_boxes=False
    ),
    GoldTable.BOXES: _TableLayout(
        schema=GOLD_BOX_SCHEMA, key="box_id", reads_boxes=True
    ),
    GoldTable.EMBEDDINGS: _TableLayout(
        schema=GOLD_EMBEDDING_SCHEMA,
        key="image_id",
        reads_boxes=False,
        embedding_join=("dataset", "split", "file_name"),
    ),
    GoldTable.CROP_EMBEDDINGS: _TableLayout(
        schema=GOLD_CROP_EMBEDDING_SCHEMA,
        key="box_id",
        reads_boxes=True,
        embedding_join=("box_id",),
    ),
}


@dataclass(frozen=True)
class GoldPage:
    rows: pa.Table
    # The `after` of the next page, or None on the last page.
    next_after: str | None


def read_gold_page(
    *,
    store: LakeStore,
    paths: CvLakePaths,
    manifest: GoldManifest,
    release: ReleaseManifest | None,
    table: GoldTable,
    row_filter: ImageFilter,
) -> GoldPage:
    """Read at most `row_filter.limit` rows, in the order of the key of the table.

    With a release, the val and test rows are the frozen ones. Without one, they are
    those of the current gold. The train rows are always those of the current gold.
    """
    layout = _LAYOUTS[table]
    rows = _read_rows(
        store=store,
        paths=paths,
        manifest=manifest,
        release=release,
        layout=layout,
        row_filter=row_filter,
    )
    # A full page can have a successor. A short page is the last one.
    is_full = rows.num_rows == row_filter.limit
    return GoldPage(
        rows=rows,
        next_after=rows.column(layout.key)[-1].as_py() if is_full else None,
    )


def _read_rows(
    *,
    store: LakeStore,
    paths: CvLakePaths,
    manifest: GoldManifest,
    release: ReleaseManifest | None,
    layout: _TableLayout,
    row_filter: ImageFilter,
) -> pa.Table:
    gold_files, embedding_files = _select_files(
        paths=paths,
        manifest=manifest,
        release=release,
        layout=layout,
        row_filter=row_filter,
    )
    if not gold_files or (layout.embedding_join is not None and not embedding_files):
        return layout.schema.empty_table()

    parameters: dict[str, object] = {"gold": gold_files, "limit": row_filter.limit}
    conditions = _build_selection_conditions(
        selection=row_filter, parameters=parameters
    )
    if row_filter.after is not None:
        conditions.append(f"g.{layout.key} > cast($after as varchar)")
        parameters["after"] = row_filter.after
    if isinstance(row_filter, BoxFilter) and row_filter.class_name:
        conditions.append("list_contains($class_names, g.class_name)")
        parameters["class_names"] = row_filter.class_name

    if layout.embedding_join is None:
        selected, joined = "g.*", ""
    else:
        selected = f"g.{layout.key}, e.embedding"
        join_columns = ", ".join(layout.embedding_join)
        joined = f"join read_parquet($embeddings) e using ({join_columns})"
        parameters["embeddings"] = embedding_files
    where = f"where {' and '.join(conditions)}" if conditions else ""
    query = f"""
    select {selected}
    from read_parquet($gold) g
    {joined}
    {where}
    order by g.{layout.key}
    limit $limit
    """
    with store.duckdb() as connection:
        # A timestamp then reaches Arrow in UTC, whatever the machine is set to.
        connection.execute("set TimeZone = 'UTC'")
        rows = connection.execute(query=query, parameters=parameters).to_arrow_table()
    return rows.cast(layout.schema)


def count_gold_images(
    *,
    store: LakeStore,
    paths: CvLakePaths,
    manifest: GoldManifest,
    release: ReleaseManifest | None,
    selection: ImageSelection,
) -> int:
    gold_files, _ = _select_files(
        paths=paths,
        manifest=manifest,
        release=release,
        layout=_LAYOUTS[GoldTable.IMAGES],
        row_filter=selection,
    )
    if not gold_files:
        return 0
    parameters: dict[str, object] = {"gold": gold_files}
    conditions = _build_selection_conditions(selection=selection, parameters=parameters)
    where = f"where {' and '.join(conditions)}" if conditions else ""
    query = f"select count(*) from read_parquet($gold) g {where}"
    with store.duckdb() as connection:
        connection.execute("set TimeZone = 'UTC'")
        row = connection.execute(query=query, parameters=parameters).fetchone()
    return 0 if row is None else row[0]


def _build_selection_conditions(
    *, selection: ImageSelection, parameters: dict[str, object]
) -> list[str]:
    """Return the conditions on the gold rows, and add their values to `parameters`."""
    conditions: list[str] = []
    if selection.changed_since is not None:
        conditions.append("g.changed_at > cast($changed_since as timestamptz)")
        parameters["changed_since"] = selection.changed_since
    if selection.commercial_use is not None:
        conditions.append("g.commercial_use = cast($commercial_use as boolean)")
        parameters["commercial_use"] = selection.commercial_use
    if selection.shard is not None and selection.num_shards is not None:
        # The image id is an MD5, so its first 32 bits spread evenly. A box follows
        # its image. DuckDB's hash() is not stable across versions, so it is not used.
        conditions.append(
            "('0x' || left(g.image_id, 8))::ubigint % $num_shards = $shard"
        )
        parameters["num_shards"] = selection.num_shards
        parameters["shard"] = selection.shard
    return conditions


def _select_files(
    *,
    paths: CvLakePaths,
    manifest: GoldManifest,
    release: ReleaseManifest | None,
    layout: _TableLayout,
    row_filter: ImageSelection,
) -> tuple[list[str], list[str]]:
    """Return the gold files and the silver embedding files of the selected splits."""
    gold_file = gold_boxes_file if layout.reads_boxes else gold_images_file
    embedding_file = crop_embeddings_file if layout.reads_boxes else embeddings_file
    gold_files: list[UPath] = []
    embedding_files: list[UPath] = []
    for directory, dataset, split in _iter_splits(
        paths=paths, manifest=manifest, release=release
    ):
        if row_filter.dataset and dataset.dataset not in row_filter.dataset:
            continue
        if row_filter.split and split.split not in row_filter.split:
            continue
        if row_filter.role and split.role not in row_filter.role:
            continue
        gold_files.append(
            gold_file(version_dir=directory, dataset=dataset.dataset, split=split.split)
        )
        if dataset.embedding_model is not None:
            embedding_files.append(
                embedding_file(
                    silver_dir=paths.silver_dir(dataset.dataset), split=split.split
                )
            )
    return [str(path) for path in gold_files], [str(path) for path in embedding_files]


def _iter_splits(
    *, paths: CvLakePaths, manifest: GoldManifest, release: ReleaseManifest | None
) -> Iterator[tuple[UPath, GoldDataset, GoldSplit]]:
    """Yield every split with the directory that holds its files."""
    version_dir = paths.gold_version_dir(manifest.version)
    for dataset in manifest.datasets:
        for split in dataset.splits:
            if release is None or not split.role.is_eval:
                yield version_dir, dataset, split
    if release is not None:
        release_dir = paths.gold_release_dir(release.release)
        for dataset in release.datasets:
            for split in dataset.splits:
                yield release_dir, dataset, split
