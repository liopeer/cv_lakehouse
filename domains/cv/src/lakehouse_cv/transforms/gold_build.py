#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Build one gold version from the silver datasets on disk.

A build never touches the version a reader has open. It writes a new directory, then
replaces the manifest that points at it, then removes the versions nothing points at.

`changed_at` is the build that last changed a row. A row that is equal to its row in
the previous version keeps the previous time, so a reader can ask what changed since.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import duckdb
import pyarrow as pa
from upath import UPath

from lakehouse_core.lake_files import remove_tree, replace_file
from lakehouse_core.lake_store import LakeStore
from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_core.parquet_files import open_parquet_writer, read_parquet_schema
from lakehouse_cv.contract.box_identity import IMAGE_ID_SQL
from lakehouse_cv.contract.dataset_spec import DatasetSpec
from lakehouse_cv.contract.gold_tables import (
    CHANGED_AT_COLUMN,
    FLAG_COLUMNS,
    GOLD_BOX_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    gold_boxes_file,
    gold_images_file,
)
from lakehouse_cv.contract.manifests import (
    GOLD_MANIFEST,
    SILVER_MANIFEST,
    GoldDataset,
    GoldManifest,
    GoldSplit,
    SilverManifest,
)
from lakehouse_cv.contract.silver_tables import (
    ROWS_PER_ROW_GROUP,
    boxes_file,
    images_file,
)
from lakehouse_cv.settings import CvLakePaths

# The current version and the one before it. A reader that opened the manifest just
# before a build still finds its files.
GOLD_VERSIONS_KEPT = 2

_IMAGE_ROWS = f"""
select
    {IMAGE_ID_SQL} as image_id,
    $image_root || '/' || file_name as image_path,
    cast($role as varchar) as role,
    cast($license as varchar) as license,
    cast($commercial_use as boolean) as commercial_use,
    *
from read_parquet($silver)
"""

_BOX_ROWS = f"""
select
    {IMAGE_ID_SQL} as image_id,
    cast($role as varchar) as role,
    cast($commercial_use as boolean) as commercial_use,
    *
from read_parquet($silver)
where not ({" or ".join(f"coalesce({name}, false)" for name in FLAG_COLUMNS)})
"""


@dataclass(frozen=True)
class GoldBuild:
    manifest: GoldManifest
    num_images: int
    num_boxes: int
    # A registered dataset with no silver on disk. Gold holds what is materialized.
    skipped_datasets: tuple[str, ...]


def read_gold_manifest(paths: CvLakePaths) -> GoldManifest | None:
    path = paths.gold_dir() / GOLD_MANIFEST
    if not path.exists():
        return None
    return read_manifest(path=path, model=GoldManifest)


def build_gold_version(
    *,
    store: LakeStore,
    paths: CvLakePaths,
    specs: Sequence[DatasetSpec],
    code_version: str,
    built_at: datetime,
) -> GoldBuild:
    previous = read_gold_manifest(paths)
    version = 1 if previous is None else previous.version + 1
    version_dir = paths.gold_version_dir(version)
    # A build that failed leaves a directory that no manifest points at.
    remove_tree(version_dir)
    previous_dir = (
        None if previous is None else paths.gold_version_dir(previous.version)
    )

    datasets: list[GoldDataset] = []
    skipped: list[str] = []
    num_images = num_boxes = 0
    with store.duckdb() as connection:
        # A timestamp then reaches Arrow in UTC, whatever the machine is set to.
        connection.execute("set TimeZone = 'UTC'")
        for spec in specs:
            silver_dir = paths.silver_dir(spec.name)
            if not (silver_dir / SILVER_MANIFEST).exists():
                skipped.append(spec.name)
                continue
            silver = read_manifest(
                path=silver_dir / SILVER_MANIFEST, model=SilverManifest
            )
            splits: list[GoldSplit] = []
            for split in silver.splits:
                gold_split = GoldSplit(
                    split=split,
                    role=spec.split_roles[split],
                    image_root=store.location(store.resolve(silver.image_roots[split])),
                )
                splits.append(gold_split)
                constants = {
                    "role": gold_split.role.value,
                    "commercial_use": silver.commercial_use,
                    "built_at": built_at,
                }
                num_images += _write_rows(
                    connection=connection,
                    rows=_IMAGE_ROWS,
                    parameters={
                        **constants,
                        "silver": str(images_file(silver_dir=silver_dir, split=split)),
                        "image_root": gold_split.image_root,
                        "license": silver.license,
                    },
                    schema=GOLD_IMAGE_SCHEMA,
                    key="image_id",
                    path=gold_images_file(
                        version_dir=version_dir, dataset=spec.name, split=split
                    ),
                    previous_path=_find_previous_file(
                        previous_dir=previous_dir,
                        gold_file=gold_images_file,
                        schema=GOLD_IMAGE_SCHEMA,
                        dataset=spec.name,
                        split=split,
                    ),
                )
                num_boxes += _write_rows(
                    connection=connection,
                    rows=_BOX_ROWS,
                    parameters={
                        **constants,
                        "silver": str(boxes_file(silver_dir=silver_dir, split=split)),
                    },
                    schema=GOLD_BOX_SCHEMA,
                    key="box_id",
                    path=gold_boxes_file(
                        version_dir=version_dir, dataset=spec.name, split=split
                    ),
                    previous_path=_find_previous_file(
                        previous_dir=previous_dir,
                        gold_file=gold_boxes_file,
                        schema=GOLD_BOX_SCHEMA,
                        dataset=spec.name,
                        split=split,
                    ),
                )
            datasets.append(
                GoldDataset(
                    dataset=spec.name,
                    license=silver.license,
                    commercial_use=silver.commercial_use,
                    silver_code_version=silver.code_version,
                    embedding_model=silver.embedding_model,
                    correction_snapshot_id=silver.correction_snapshot_id,
                    splits=splits,
                )
            )
    if not datasets:
        raise RuntimeError(f"No silver dataset is materialized under {paths.root}")

    manifest = GoldManifest(
        version=version, built_at=built_at, code_version=code_version, datasets=datasets
    )
    _replace_gold_manifest(paths=paths, manifest=manifest)
    _prune_gold_versions(paths=paths, current_version=version)
    return GoldBuild(
        manifest=manifest,
        num_images=num_images,
        num_boxes=num_boxes,
        skipped_datasets=tuple(skipped),
    )


def _find_previous_file(
    *,
    previous_dir: UPath | None,
    gold_file: Callable[..., UPath],
    schema: pa.Schema,
    dataset: str,
    split: str,
) -> UPath | None:
    if previous_dir is None:
        return None
    path = gold_file(version_dir=previous_dir, dataset=dataset, split=split)
    if not path.exists():
        return None
    # A version that an older schema wrote has nothing to compare a new column with.
    return path if read_parquet_schema(path) == schema else None


def _write_rows(
    *,
    connection: duckdb.DuckDBPyConnection,
    rows: str,
    parameters: dict[str, object],
    schema: pa.Schema,
    key: str,
    path: UPath,
    previous_path: UPath | None,
) -> int:
    """Write the rows of one split, in the columns and the types of the schema."""
    if previous_path is not None:
        parameters = {**parameters, "previous": str(previous_path)}
    reader = connection.execute(
        query=_build_changed_at_query(
            rows=rows, schema=schema, key=key, has_previous=previous_path is not None
        ),
        parameters=parameters,
    ).to_arrow_reader(ROWS_PER_ROW_GROUP)
    count = 0
    with open_parquet_writer(path=path, schema=schema) as writer:
        for batch in reader:
            # The cast is the write time validation: it rejects a null in a column
            # that the schema declares not null.
            writer.write_batch(batch.cast(schema))
            count += batch.num_rows
    return count


def _build_changed_at_query(
    *, rows: str, schema: pa.Schema, key: str, has_previous: bool
) -> str:
    columns = [name for name in schema.names if name != CHANGED_AT_COLUMN]
    selected = ", ".join(f"c.{name}" for name in columns)
    built_at = "cast($built_at as timestamptz)"
    if not has_previous:
        return (
            f"with c as ({rows}) "
            f"select {selected}, {built_at} as {CHANGED_AT_COLUMN} from c"
        )
    unchanged = " and ".join(
        f"c.{name} is not distinct from p.{name}" for name in columns
    )
    return f"""
    with c as ({rows})
    select {selected},
           case when {unchanged} then p.{CHANGED_AT_COLUMN} else {built_at} end
               as {CHANGED_AT_COLUMN}
    from c
    left join read_parquet($previous) p on p.{key} = c.{key}
    """


def _replace_gold_manifest(paths: CvLakePaths, manifest: GoldManifest) -> None:
    path = paths.gold_dir() / GOLD_MANIFEST
    staged = path.with_suffix(".json.staged")
    write_manifest(path=staged, manifest=manifest)
    replace_file(source=staged, target=path)


def _prune_gold_versions(paths: CvLakePaths, current_version: int) -> None:
    for version_dir in paths.gold_version_dir(current_version).parent.iterdir():
        if int(version_dir.name) <= current_version - GOLD_VERSIONS_KEPT:
            remove_tree(version_dir)
