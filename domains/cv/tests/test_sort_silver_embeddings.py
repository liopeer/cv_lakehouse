#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Sort the embedding files of the old layout into the new one, with no embedding."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.maintenance import sort_silver_embeddings as sorting
from lakehouse_cv.maintenance.sort_silver_embeddings import sort_silver_embeddings
from lakehouse_cv.transforms.embedding_files import VECTOR_TABLES
from tests.lake_runs import materialize_assets, materialize_bronze_links

DATASET = "wider_face"
# The columns that a file of the old layout lacks.
NEW_COLUMNS = {"image_id", "crop_x", "crop_y", "crop_width", "crop_height"}


def _vector_files(lake: CvLakeResource) -> list[Path]:
    silver_dir = lake.paths.silver_dir(DATASET)
    return [
        Path(str(table.target_file(silver_dir=silver_dir, split=split)))
        for table in VECTOR_TABLES
        for split in ("train", "val")
    ]


def _write_old_layout(path: Path) -> None:
    """Drop the new columns and the footer, and turn the order of the rows around."""
    table = pq.read_table(path)
    old = table.drop_columns(
        [name for name in table.column_names if name in NEW_COLUMNS]
    )
    old = old.take(pa.array(range(old.num_rows - 1, -1, -1))).replace_schema_metadata()
    pq.write_table(old, path)


@pytest.fixture
def silver_lake(
    embedding_lake: CvLakeResource, bronze_sources: dict[str, Path]
) -> CvLakeResource:
    materialize_bronze_links(lake=embedding_lake, sources=bronze_sources)
    materialize_assets(
        lake=embedding_lake, assets=[silver_defs.build_silver_asset(DATASET)]
    )
    return embedding_lake


def test_the_old_layout_becomes_the_file_that_silver_writes(
    silver_lake: CvLakeResource, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Many buckets for a few rows, so that the rows cross every bucket.
    monkeypatch.setattr(target=sorting, name="BYTES_PER_BUCKET", value=1)
    paths = _vector_files(silver_lake)
    expected = {path: pq.read_table(path) for path in paths}
    for path in paths:
        _write_old_layout(path)

    sort_silver_embeddings(
        store=silver_lake.store, datasets=[DATASET], log=lambda message: None
    )

    for path in paths:
        table = pq.read_table(path)
        assert table == expected[path]
        assert table.schema.metadata == expected[path].schema.metadata
    files = {path.name for path in paths[0].parent.parent.rglob("*") if path.is_file()}
    assert files == {"_silver.json", "train.parquet", "val.parquet"}


def test_a_file_of_the_new_layout_is_left_as_it_is(
    silver_lake: CvLakeResource,
) -> None:
    paths = _vector_files(silver_lake)
    modified = {path: path.stat().st_mtime_ns for path in paths}

    sort_silver_embeddings(
        store=silver_lake.store, datasets=[DATASET], log=lambda message: None
    )

    assert {path: path.stat().st_mtime_ns for path in paths} == modified


def test_vectors_that_do_not_match_their_rows_stop_the_command(
    silver_lake: CvLakeResource,
) -> None:
    path = _vector_files(silver_lake)[0]
    _write_old_layout(path)
    table = pq.read_table(path)
    pq.write_table(table.slice(length=table.num_rows - 1), path)

    with pytest.raises(expected_exception=ValueError, match="do not match"):
        sort_silver_embeddings(
            store=silver_lake.store, datasets=[DATASET], log=lambda message: None
        )
