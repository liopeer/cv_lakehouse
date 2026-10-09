#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Write the embedding files of one silver split, and reuse every vector that exists.

A vector depends on the pixels, the crop and the model. Bronze pins the pixels, and a
crop embedding row holds its crop. So a vector of an earlier run stays right while its
file names the same model, whatever else in silver changed since.

The new vectors go into parts beside the target, one closed file at a time. A run that
stops keeps every closed part, and the next run reuses it. The target is written last,
one range of ids at a time, so a split of any size fits in memory.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from upath import UPath

from lakehouse_core.lake_files import remove_tree, replace_file
from lakehouse_core.lake_store import LakeStore
from lakehouse_core.parquet_files import open_parquet_writer, read_parquet_schema
from lakehouse_cv.contract.box_identity import IMAGE_ID_SQL
from lakehouse_cv.contract.silver_tables import (
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_COLUMN,
    EMBEDDING_MODEL_METADATA_KEY,
    EMBEDDING_ROWS_PER_ROW_GROUP,
    EMBEDDING_SCHEMA,
    EMBEDDING_VERSION_METADATA_KEY,
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
    images_file,
)
from lakehouse_cv.transforms.embeddings import (
    EMBEDDING_MODEL,
    ITEMS_PER_REQUEST,
    Crop,
    Embedder,
)
from lakehouse_cv.transforms.progress_log import ProgressLog

# Bump this when a vector changes under the same model name, such as with new weights
# or a new crop. Every vector is then embedded again.
EMBEDDING_VERSION = "1"
# A closed part holds at most this many new vectors, which is 128 MB.
ROWS_PER_PART = 65_536
# The target is written in ranges of ids of about this many rows, which is 128 MB.
ROWS_PER_RANGE = 65_536

_PARTS_SUFFIX = ".parts"
_PART_SUFFIX = ".parquet"
_NEXT_SUFFIX = ".next"


@dataclass(frozen=True)
class VectorTable:
    """One embedding file: the rows that need a vector, and the way to embed them."""

    label: str
    schema: pa.Schema
    # The id that sorts the file.
    key: str
    # A vector serves a row when these columns are equal.
    match_columns: tuple[str, ...]
    rows_file: Callable[..., UPath]
    target_file: Callable[..., UPath]
    # Select the rows of the silver file `$rows`, in the columns of the schema less the
    # vector.
    rows_query: str
    embed: Callable[
        [Embedder, pa.RecordBatch, Callable[[str], str]], NDArray[np.float32]
    ]

    @property
    def row_columns(self) -> list[str]:
        return [name for name in self.schema.names if name != EMBEDDING_COLUMN]


def _embed_images(
    embedder: Embedder,
    rows: pa.RecordBatch,
    locate_image: Callable[[str], str],
) -> NDArray[np.float32]:
    names: list[str] = rows.column("file_name").to_pylist()
    return embedder.embed_images([locate_image(name) for name in names])


def _embed_crops(
    embedder: Embedder,
    rows: pa.RecordBatch,
    locate_image: Callable[[str], str],
) -> NDArray[np.float32]:
    columns = rows.to_pydict()
    return embedder.embed_crops(
        [
            Crop(path=locate_image(name), x=x, y=y, width=width, height=height)
            for name, x, y, width, height in zip(
                columns["file_name"],
                columns["crop_x"],
                columns["crop_y"],
                columns["crop_width"],
                columns["crop_height"],
                strict=True,
            )
        ]
    )


IMAGE_VECTORS = VectorTable(
    label="image embeddings",
    schema=EMBEDDING_SCHEMA,
    key="image_id",
    match_columns=("image_id",),
    rows_file=images_file,
    target_file=embeddings_file,
    rows_query=f"""
    select dataset, split, file_name, {IMAGE_ID_SQL} as image_id
    from read_parquet($rows)
    """,
    embed=_embed_images,
)

# `round_even` rounds half to even, as Python's `round` does. A box under half a pixel
# still names one pixel, so the crop on the server has an area.
CROP_VECTORS = VectorTable(
    label="crop embeddings",
    schema=CROP_EMBEDDING_SCHEMA,
    key="box_id",
    match_columns=("box_id", "crop_x", "crop_y", "crop_width", "crop_height"),
    rows_file=boxes_file,
    target_file=crop_embeddings_file,
    rows_query="""
    select dataset, split, file_name, box_id, box_index,
           round_even(x, 0)::integer as crop_x,
           round_even(y, 0)::integer as crop_y,
           greatest(1, round_even(w, 0))::integer as crop_width,
           greatest(1, round_even(h, 0))::integer as crop_height
    from read_parquet($rows)
    """,
    embed=_embed_crops,
)

VECTOR_TABLES = (IMAGE_VECTORS, CROP_VECTORS)


@dataclass(frozen=True)
class VectorCount:
    written: int
    embedded: int

    @property
    def reused(self) -> int:
        return self.written - self.embedded


def write_vectors(
    *,
    store: LakeStore,
    embedder: Embedder,
    table: VectorTable,
    silver_dir: UPath,
    split: str,
    locate_image: Callable[[str], str],
    log: Callable[[str], None],
) -> VectorCount:
    """Write the embedding file of one split, and embed only a row with no vector."""
    target = table.target_file(silver_dir=silver_dir, split=split)
    parts_dir = parts_dir_of(target)
    _remove_unusable_parts(table=table, parts_dir=parts_dir)
    with store.duckdb() as connection:
        select_wanted_rows(
            connection=connection,
            table=table,
            rows=table.rows_file(silver_dir=silver_dir, split=split),
        )
        embedded = _embed_missing_rows(
            connection=connection,
            embedder=embedder,
            table=table,
            pool=list_vector_files(table=table, target=target),
            parts_dir=parts_dir,
            locate_image=locate_image,
            progress_label=f"{split} {table.label} to embed",
            log=log,
        )
        written = write_sorted_vectors(
            connection=connection,
            table=table,
            pool=list_vector_files(table=table, target=target),
            target=target,
            progress_label=f"{split} {table.label} to write",
            log=log,
        )
    remove_tree(parts_dir)
    return VectorCount(written=written, embedded=embedded)


def clear_vectors(silver_dir: UPath, split: str) -> None:
    """Remove the embedding files of one split, and their parts.

    A rematerialisation with no server rewrites the images and the boxes, so a vector
    left behind describes pixels that a query no longer has a row for.
    """
    for table in VECTOR_TABLES:
        target = table.target_file(silver_dir=silver_dir, split=split)
        target.unlink(missing_ok=True)
        remove_tree(parts_dir_of(target))


def holds_vectors_of_this_model(*, table: VectorTable, path: UPath) -> bool:
    """Tell an embedding file of this layout whose footer names the model of this code.

    A file of the layout before ADR 0013 has no such footer, and fails the test.
    """
    try:
        schema = read_parquet_schema(path)
    except (OSError, pa.ArrowInvalid):
        return False
    metadata = schema.metadata or {}
    footer = {key: metadata.get(key) for key in _build_footer()}
    return footer == _build_footer() and schema.remove_metadata().equals(table.schema)


def open_vector_writer(*, table: VectorTable, path: UPath) -> pq.ParquetWriter:
    return open_parquet_writer(
        path=path, schema=table.schema.with_metadata(_build_footer())
    )


def _build_footer() -> dict[bytes, bytes]:
    return {
        EMBEDDING_MODEL_METADATA_KEY: EMBEDDING_MODEL.encode(),
        EMBEDDING_VERSION_METADATA_KEY: EMBEDDING_VERSION.encode(),
    }


def select_wanted_rows(
    *, connection: duckdb.DuckDBPyConnection, table: VectorTable, rows: UPath
) -> None:
    """Fill the temporary table `wanted` with the rows of a silver file, by the id."""
    # Every range of the target reads the parts again. The footers stay in memory.
    connection.execute("set parquet_metadata_cache = true")
    connection.execute(
        query=f"create temp table wanted as {table.rows_query} order by {table.key}",
        parameters={"rows": str(rows)},
    )


def _embed_missing_rows(
    *,
    connection: duckdb.DuckDBPyConnection,
    embedder: Embedder,
    table: VectorTable,
    pool: Sequence[UPath],
    parts_dir: UPath,
    locate_image: Callable[[str], str],
    progress_label: str,
    log: Callable[[str], None],
) -> int:
    match = ", ".join(table.match_columns)
    if pool:
        connection.execute(
            query=f"""
            create temp table missing as
            select w.* from wanted w
            anti join (select {match} from read_parquet($pool)) p using ({match})
            """,
            parameters={"pool": [str(path) for path in pool]},
        )
    else:
        connection.execute("create temp view missing as select * from wanted")
    total = _count_rows(connection=connection, table_name="missing")
    progress = ProgressLog(log=log, label=progress_label, total=total)
    writer = _PartWriter(table=table, parts_dir=parts_dir)
    try:
        # In the order of the id, so that every part covers one narrow range of ids.
        for rows in connection.execute(
            f"select * from missing order by {table.key}"
        ).to_arrow_reader(ITEMS_PER_REQUEST):
            writer.add(
                build_vector_batch(
                    table=table,
                    rows=rows,
                    vectors=table.embed(embedder, rows, locate_image),
                )
            )
            progress.advance(rows.num_rows)
    finally:
        # A part closes on an error too, so every vector already paid for stays.
        writer.close()
    progress.finish()
    return total


def write_sorted_vectors(
    *,
    connection: duckdb.DuckDBPyConnection,
    table: VectorTable,
    pool: Sequence[UPath],
    target: UPath,
    progress_label: str,
    log: Callable[[str], None],
) -> int:
    """Write a vector for every row of `wanted` to the target, sorted by the id.

    Every file of the pool is sorted by the id, or holds one range of the ids, so a
    range reads only a few row groups. Return the row count.
    """
    total = _count_rows(connection=connection, table_name="wanted")
    progress = ProgressLog(log=log, label=progress_label, total=total)
    following = target.with_name(target.name + _NEXT_SUFFIX)
    written = 0
    with open_vector_writer(table=table, path=following) as writer:
        for lower, upper in split_key_ranges(total) if pool and total else ():
            rows = _read_vector_range(
                connection=connection,
                table=table,
                pool=pool,
                lower=lower,
                upper=upper,
            )
            writer.write_table(rows, row_group_size=EMBEDDING_ROWS_PER_ROW_GROUP)
            written += rows.num_rows
            progress.advance(rows.num_rows)
    if written != total:
        following.unlink()
        raise ValueError(
            f"{target}: {total - written} of {total} rows have no vector to write."
        )
    replace_file(source=following, target=target)
    progress.finish()
    return written


def _read_vector_range(
    *,
    connection: duckdb.DuckDBPyConnection,
    table: VectorTable,
    pool: Sequence[UPath],
    lower: str | None,
    upper: str | None,
) -> pa.Table:
    parameters: dict[str, object] = {"pool": [str(path) for path in pool]}
    conditions: list[str] = []
    for alias in ("w", "v"):
        if lower is not None:
            conditions.append(f"{alias}.{table.key} >= $lower")
        if upper is not None:
            conditions.append(f"{alias}.{table.key} < $upper")
    if lower is not None:
        parameters["lower"] = lower
    if upper is not None:
        parameters["upper"] = upper
    where = f"where {' and '.join(conditions)}" if conditions else ""
    columns = ", ".join(f"w.{name}" for name in table.row_columns)
    # A run that stopped after the target was replaced and before the parts were
    # removed leaves a vector twice. Both copies are equal, so either one serves.
    query = f"""
    select {columns}, v.{EMBEDDING_COLUMN}
    from wanted w
    join read_parquet($pool) v using ({", ".join(table.match_columns)})
    {where}
    qualify row_number() over (partition by w.{table.key}) = 1
    order by w.{table.key}
    """
    return (
        connection.execute(query=query, parameters=parameters)
        .to_arrow_table()
        .cast(table.schema)
    )


def split_key_ranges(row_count: int) -> list[tuple[str | None, str | None]]:
    """Split the ids into ranges of about `ROWS_PER_RANGE` rows, by the leading digits.

    An id is an MD5 written as a UUID, so its hex digits spread evenly. The first and
    the last range are open, so every string is in exactly one range.
    """
    digits = 1
    while 16**digits * ROWS_PER_RANGE < row_count:
        digits += 1
    bounds: list[str | None] = [f"{i:0{digits}x}" for i in range(1, 16**digits)]
    return list(zip([None, *bounds], [*bounds, None], strict=True))


def build_vector_batch(
    *, table: VectorTable, rows: pa.RecordBatch, vectors: NDArray[np.float32]
) -> pa.RecordBatch:
    columns = [
        rows.column(field.name).cast(field.type)
        for field in table.schema
        if field.name != EMBEDDING_COLUMN
    ]
    return pa.RecordBatch.from_arrays(
        arrays=[*columns, _wrap_vectors_as_list_array(vectors)], schema=table.schema
    )


def _wrap_vectors_as_list_array(vectors: NDArray[np.float32]) -> pa.ListArray:
    """Wrap an (N, 512) array as a list column, with no Python round trip."""
    count, dimension = vectors.shape
    offsets = pa.array(np.arange(count + 1, dtype=np.int32) * dimension)
    return pa.ListArray.from_arrays(
        offsets=offsets, values=pa.array(vectors.reshape(-1))
    )


def _count_rows(*, connection: duckdb.DuckDBPyConnection, table_name: str) -> int:
    row = connection.execute(f"select count(*) from {table_name}").fetchone()
    return 0 if row is None else int(row[0])


def parts_dir_of(target: UPath) -> UPath:
    return target.with_name(target.name + _PARTS_SUFFIX)


def list_vector_files(*, table: VectorTable, target: UPath) -> list[UPath]:
    """Return the target and the closed parts that hold vectors of this model."""
    parts_dir = parts_dir_of(target)
    parts = sorted(parts_dir.glob(f"*{_PART_SUFFIX}")) if parts_dir.exists() else []
    return [
        path
        for path in [target, *parts]
        if path.exists() and holds_vectors_of_this_model(table=table, path=path)
    ]


def _remove_unusable_parts(*, table: VectorTable, parts_dir: UPath) -> None:
    """Remove a part that a stopped run left open, or that holds another model."""
    if not parts_dir.exists():
        return
    for path in parts_dir.iterdir():
        if not path.name.endswith(_PART_SUFFIX) or not holds_vectors_of_this_model(
            table=table, path=path
        ):
            path.unlink()


class _PartWriter:
    """Write vectors into parts of at most `ROWS_PER_PART` rows.

    A part is written under a temporary name, and gets its own name once it is closed,
    so every file that ends in `.parquet` is whole.
    """

    def __init__(self, *, table: VectorTable, parts_dir: UPath) -> None:
        self._table = table
        self._parts_dir = parts_dir
        self._writer: pq.ParquetWriter | None = None
        self._writing: UPath | None = None
        self._rows = 0

    def add(self, batch: pa.RecordBatch) -> None:
        if batch.num_rows == 0:
            return
        if self._writer is None:
            self._writing = self._parts_dir / f"part-{uuid.uuid4().hex}.writing"
            self._writer = open_vector_writer(table=self._table, path=self._writing)
        self._writer.write_batch(batch, row_group_size=EMBEDDING_ROWS_PER_ROW_GROUP)
        self._rows += batch.num_rows
        if self._rows >= ROWS_PER_PART:
            self.close()

    def close(self) -> None:
        if self._writer is None or self._writing is None:
            return
        self._writer.close()
        replace_file(
            source=self._writing,
            target=self._writing.with_name(self._writing.stem + _PART_SUFFIX),
        )
        self._writer, self._writing, self._rows = None, None, 0
