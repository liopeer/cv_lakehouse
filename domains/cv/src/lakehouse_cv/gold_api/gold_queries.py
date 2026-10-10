#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Read one page of gold rows with DuckDB.

The manifest names the role of every file, so a filter on the dataset, the split or
the role selects files and never opens the others. Every value that a caller sends is
a bound parameter.

A page of vectors selects its ids from gold first, and then reads their vectors. The
embeddings assets sort the vectors by the id, so DuckDB reads only the row groups of the
page. A crop vector serves a gold box only when its crop is the crop of that box, so a
box that moved after its vector was made gets none.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

import duckdb
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
from lakehouse_cv.gold_api.row_filters import BoxFilter, ImageFilter, ImageSelection
from lakehouse_cv.settings import CvLakePaths
from lakehouse_cv.transforms.embedding_files import (
    CROP_SQL,
    CROP_VECTORS,
    IMAGE_VECTORS,
    VectorTable,
)
from lakehouse_cv.transforms.layer_builds import (
    embeddings_build_dir,
    read_embeddings_manifest,
)


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
    # The vectors of the gold rows, or None for the gold rows themselves.
    vectors: VectorTable | None = None
    # The columns beyond the key that a vector matches, from a gold row.
    match_sql: str | None = None

    @property
    def reads_vectors(self) -> bool:
        return self.vectors is not None


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
        vectors=IMAGE_VECTORS,
    ),
    GoldTable.CROP_EMBEDDINGS: _TableLayout(
        schema=GOLD_CROP_EMBEDDING_SCHEMA,
        key="box_id",
        reads_boxes=True,
        vectors=CROP_VECTORS,
        match_sql=CROP_SQL,
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
    gold_files, embedding_files = _select_files(
        paths=paths,
        manifest=manifest,
        release=release,
        layout=layout,
        row_filter=row_filter,
    )
    if not gold_files or (layout.reads_vectors and not embedding_files):
        return GoldPage(rows=layout.schema.empty_table(), next_after=None)

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
    where = f"where {' and '.join(conditions)}" if conditions else ""
    selected = (
        "g.*"
        if not layout.reads_vectors
        else ", ".join(
            [f"g.{layout.key}", *([layout.match_sql] if layout.match_sql else [])]
        )
    )
    query = f"""
    select {selected}
    from read_parquet($gold) g
    {where}
    order by g.{layout.key}
    limit $limit
    """
    with store.duckdb() as connection:
        rows = connection.execute(query=query, parameters=parameters).to_arrow_table()
        # A full page can have a successor. A short page is the last one. The page of
        # ids decides, because an id can lack a vector.
        next_after = (
            rows.column(layout.key)[-1].as_py()
            if rows.num_rows == row_filter.limit
            else None
        )
        if layout.vectors is not None:
            rows = _read_vectors(
                connection=connection,
                vectors=layout.vectors,
                page=rows,
                embedding_files=embedding_files,
            )
    return GoldPage(rows=rows.cast(layout.schema), next_after=next_after)


def _read_vectors(
    *,
    connection: duckdb.DuckDBPyConnection,
    vectors: VectorTable,
    page: pa.Table,
    embedding_files: list[str],
) -> pa.Table:
    connection.register(view_name="page", python_object=page)
    key = vectors.key
    return connection.execute(
        query=f"""
        select page.{key}, e.embedding
        from page
        join read_parquet($embeddings) e using ({", ".join(vectors.match_columns)})
        order by page.{key}
        """,
        parameters={"embeddings": embedding_files},
    ).to_arrow_table()


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
        row = connection.execute(query=query, parameters=parameters).fetchone()
    return 0 if row is None else row[0]


def _build_selection_conditions(
    *, selection: ImageSelection, parameters: dict[str, object]
) -> list[str]:
    """Return the conditions on the gold rows, and add their values to `parameters`."""
    conditions: list[str] = []
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
    """Return the gold files and the embedding files of the selected splits."""
    gold_file = gold_boxes_file if layout.reads_boxes else gold_images_file
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
            gold_file(build_dir=directory, dataset=dataset.dataset, split=split.split)
        )
        if layout.vectors is None:
            continue
        embeddings = read_embeddings_manifest(paths=paths, name=dataset.dataset)
        if embeddings is not None and split.split in embeddings.splits:
            embedding_files.append(
                layout.vectors.target_file(
                    build_dir=embeddings_build_dir(paths=paths, manifest=embeddings),
                    split=split.split,
                )
            )
    return [str(path) for path in gold_files], [str(path) for path in embedding_files]


def _iter_splits(
    *, paths: CvLakePaths, manifest: GoldManifest, release: ReleaseManifest | None
) -> Iterator[tuple[UPath, GoldDataset, GoldSplit]]:
    """Yield every split with the directory that holds its files."""
    gold_dir = paths.gold_build_dir(manifest.build_id)
    for dataset in manifest.datasets:
        for split in dataset.splits:
            if release is None or not split.role.is_eval:
                yield gold_dir, dataset, split
    if release is not None:
        release_dir = paths.gold_release_dir(release.release)
        for dataset in release.datasets:
            for split in dataset.splits:
                yield release_dir, dataset, split
