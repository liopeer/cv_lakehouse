#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Move the silver embedding files to the layout of ADR 0013, with no new embedding.

    python -m lakehouse_cv.maintenance.sort_silver_embeddings [DATASET ...]

The command reads the lake from the environment, as the code location does. It leaves
a file of the new layout as it is, so it can run again after it stopped.

A file of the old layout holds one vector for every row of the images or the boxes file
beside it, because the same run wrote both. The command adds the id and the crop of
each row from that file, sorts the vectors by the id, and names the model in the footer.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Sequence

import duckdb
import numpy as np
import pyarrow as pa
from upath import UPath

from lakehouse_core.lake_files import remove_tree
from lakehouse_core.lake_store import LakeStore
from lakehouse_core.manifest_files import read_manifest
from lakehouse_core.parquet_files import open_parquet_file, open_parquet_writer
from lakehouse_cv.contract.box_identity import IMAGE_ID_SQL
from lakehouse_cv.contract.manifests import SILVER_MANIFEST, SilverManifest
from lakehouse_cv.contract.silver_tables import (
    EMBEDDING_COLUMN,
    EMBEDDING_ROWS_PER_ROW_GROUP,
)
from lakehouse_cv.settings import CvLakePaths, CvSettings
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.embedding_files import (
    IMAGE_VECTORS,
    VECTOR_TABLES,
    VectorTable,
    holds_vectors_of_this_model,
    list_vector_files,
    open_vector_writer,
    parts_dir_of,
    select_wanted_rows,
    write_sorted_vectors,
)
from lakehouse_cv.transforms.embeddings import EMBEDDING_MODEL
from lakehouse_cv.transforms.progress_log import ProgressLog

# A bucket is sorted in memory. Its Arrow table, sorted, takes about twice its size.
BYTES_PER_BUCKET = 1 << 30
# Each bucket holds an open upload, which buffers up to 50 MB. A power of two, so that
# the buckets split the first two hex digits evenly.
MAX_BUCKETS = 32
_ROWS_PER_SCATTER_BATCH = 8192
_SORTING_SUFFIX = ".sorting"

_LOG = logging.getLogger(__name__)


def sort_silver_embeddings(
    *, store: LakeStore, datasets: Sequence[str], log: Callable[[str], None]
) -> None:
    paths = CvLakePaths(store.root)
    for dataset in datasets:
        silver_dir = paths.silver_dir(dataset)
        manifest_path = silver_dir / SILVER_MANIFEST
        if not manifest_path.exists():
            log(f"{dataset}: no silver, skipped")
            continue
        manifest = read_manifest(path=manifest_path, model=SilverManifest)
        if manifest.embedding_model != EMBEDDING_MODEL:
            log(f"{dataset}: no vector of {EMBEDDING_MODEL}, skipped")
            continue
        for split in manifest.splits:
            for table in VECTOR_TABLES:
                sort_vector_file(
                    store=store,
                    table=table,
                    silver_dir=silver_dir,
                    split=split,
                    log=log,
                )


def sort_vector_file(
    *,
    store: LakeStore,
    table: VectorTable,
    silver_dir: UPath,
    split: str,
    log: Callable[[str], None],
) -> None:
    target = table.target_file(silver_dir=silver_dir, split=split)
    label = f"{silver_dir.name} {split} {table.label}"
    if not target.exists() or holds_vectors_of_this_model(table=table, path=target):
        log(f"{label}: nothing to sort")
        return
    sorting_dir = target.with_name(target.name + _SORTING_SUFFIX)
    parts_dir = parts_dir_of(target)
    remove_tree(sorting_dir)
    remove_tree(parts_dir)
    with store.duckdb() as connection:
        select_wanted_rows(
            connection=connection,
            table=table,
            rows=table.rows_file(silver_dir=silver_dir, split=split),
        )
        num_buckets = _count_buckets(target)
        _scatter_into_buckets(
            connection=connection,
            table=table,
            source=target,
            sorting_dir=sorting_dir,
            num_buckets=num_buckets,
            progress=ProgressLog(log=log, label=f"{label} read"),
        )
        _sort_buckets_into_parts(
            connection=connection,
            table=table,
            sorting_dir=sorting_dir,
            parts_dir=parts_dir,
            num_buckets=num_buckets,
            progress=ProgressLog(log=log, label=f"{label} buckets", total=num_buckets),
        )
        write_sorted_vectors(
            connection=connection,
            table=table,
            pool=list_vector_files(table=table, target=target),
            target=target,
            progress_label=f"{label} write",
            log=log,
        )
    remove_tree(parts_dir)
    remove_tree(sorting_dir)


def _count_buckets(source: UPath) -> int:
    size = source.stat().st_size
    num_buckets = 1
    while num_buckets < MAX_BUCKETS and num_buckets * BYTES_PER_BUCKET < size:
        num_buckets *= 2
    return num_buckets


def _split_bucket_bounds(num_buckets: int) -> list[str]:
    """Return the first id of every bucket but the first.

    The first and the last bucket are open, so every string is in exactly one bucket.
    """
    return [f"{index * 256 // num_buckets:02x}" for index in range(1, num_buckets)]


def _bucket_range(*, bucket: int, bounds: list[str]) -> tuple[str | None, str | None]:
    lowers: list[str | None] = [None, *bounds]
    uppers: list[str | None] = [*bounds, None]
    return lowers[bucket], uppers[bucket]


def _scatter_into_buckets(
    *,
    connection: duckdb.DuckDBPyConnection,
    table: VectorTable,
    source: UPath,
    sorting_dir: UPath,
    num_buckets: int,
    progress: ProgressLog,
) -> None:
    """Read the source once, and write each vector to the bucket of its id."""
    key_sql = IMAGE_ID_SQL if table is IMAGE_VECTORS else table.key
    # The bounds are hex digits that this module makes, so they go in as literals.
    whens = "".join(
        f"when key < '{bound}' then {index} "
        for index, bound in enumerate(_split_bucket_bounds(num_buckets))
    )
    bucket_sql = f"case {whens}else {num_buckets - 1} end" if whens else "0"
    query = f"""
    select key, {EMBEDDING_COLUMN}, {bucket_sql} as bucket
    from (select {key_sql} as key, {EMBEDDING_COLUMN} from read_parquet($source))
    """
    schema = pa.schema(
        [
            pa.field(name=table.key, type=pa.string(), nullable=False),
            table.schema.field(EMBEDDING_COLUMN),
        ]
    )
    writers = [
        open_parquet_writer(path=_bucket_file(sorting_dir, bucket), schema=schema)
        for bucket in range(num_buckets)
    ]
    try:
        for batch in connection.execute(
            query=query,
            parameters={"source": str(source)},
        ).to_arrow_reader(_ROWS_PER_SCATTER_BATCH):
            buckets = batch.column("bucket").to_numpy()
            vectors = pa.RecordBatch.from_arrays(
                arrays=[
                    batch.column("key"),
                    batch.column(EMBEDDING_COLUMN).cast(
                        table.schema.field(EMBEDDING_COLUMN).type
                    ),
                ],
                schema=schema,
            )
            for bucket in np.unique(buckets).tolist():
                writers[bucket].write_batch(vectors.filter(pa.array(buckets == bucket)))
            progress.advance(batch.num_rows)
    finally:
        for writer in writers:
            writer.close()
    progress.finish()


def _sort_buckets_into_parts(
    *,
    connection: duckdb.DuckDBPyConnection,
    table: VectorTable,
    sorting_dir: UPath,
    parts_dir: UPath,
    num_buckets: int,
    progress: ProgressLog,
) -> None:
    """Sort each bucket by the id, add the columns of its rows, and write it as a part.

    The rows of a bucket and its vectors have the same ids, one to one. Otherwise the
    old file did not match its silver file, and the command stops.
    """
    bounds = _split_bucket_bounds(num_buckets)
    for bucket in range(num_buckets):
        path = _bucket_file(sorting_dir, bucket)
        vectors = open_parquet_file(path).read()
        vectors = vectors.sort_by(table.key)
        rows = _read_bucket_rows(
            connection=connection,
            table=table,
            bucket_range=_bucket_range(bucket=bucket, bounds=bounds),
        )
        if not rows.column(table.key).equals(vectors.column(table.key)):
            raise ValueError(
                f"{path}: the vectors do not match the rows of their silver file. "
                "Materialize silver to embed them again."
            )
        sorted_vectors = pa.Table.from_arrays(
            arrays=[
                *(rows.column(name) for name in table.row_columns),
                vectors.column(EMBEDDING_COLUMN),
            ],
            names=table.schema.names,
        ).cast(table.schema)
        with open_vector_writer(
            table=table, path=parts_dir / f"bucket-{bucket:02d}.parquet"
        ) as writer:
            writer.write_table(
                sorted_vectors, row_group_size=EMBEDDING_ROWS_PER_ROW_GROUP
            )
        path.unlink()
        progress.advance(1)
    progress.finish()


def _read_bucket_rows(
    *,
    connection: duckdb.DuckDBPyConnection,
    table: VectorTable,
    bucket_range: tuple[str | None, str | None],
) -> pa.Table:
    lower, upper = bucket_range
    conditions: list[str] = []
    parameters: dict[str, str] = {}
    if lower is not None:
        conditions.append(f"{table.key} >= $lower")
        parameters["lower"] = lower
    if upper is not None:
        conditions.append(f"{table.key} < $upper")
        parameters["upper"] = upper
    where = f"where {' and '.join(conditions)}" if conditions else ""
    rows = connection.execute(
        query=f"select * from wanted {where} order by {table.key}",
        parameters=parameters,
    ).to_arrow_table()
    return rows.combine_chunks()


def _bucket_file(sorting_dir: UPath, bucket: int) -> UPath:
    return sorting_dir / f"bucket-{bucket:02d}.parquet"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sort the silver embedding files, with no new embedding."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(SOURCE_BY_NAME),
        help="The datasets to sort. All of them by default.",
    )
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = CvSettings()
    sort_silver_embeddings(
        store=LakeStore(
            root=settings.root,
            storage_options=settings.storage_options,
            duckdb_threads=settings.duckdb_threads,
            duckdb_memory_limit=settings.duckdb_memory_limit,
        ),
        datasets=arguments.datasets,
        log=_LOG.info,
    )


if __name__ == "__main__":
    main()
