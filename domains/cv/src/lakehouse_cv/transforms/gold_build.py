#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Build gold from the current silver build of each dataset.

Gold is the union of the silver datasets, with the role columns, and without the
flagged boxes. Nothing else: a build reads no earlier gold, so a rebuild from the same
silver gives the same rows.

A build never touches the build a reader has open. It writes a new directory, then
replaces the manifest that points at it, then removes the builds nothing points at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import duckdb
import pyarrow as pa
from upath import UPath

from lakehouse_core.lake_store import LakeStore
from lakehouse_core.manifest_files import read_manifest
from lakehouse_core.parquet_files import open_parquet_writer
from lakehouse_cv.contract.box_identity import IMAGE_ID_SQL
from lakehouse_cv.contract.dataset_spec import DatasetSpec
from lakehouse_cv.contract.gold_tables import (
    FLAG_COLUMNS,
    GOLD_BOX_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    gold_boxes_file,
    gold_images_file,
)
from lakehouse_cv.contract.manifests import (
    GOLD_MANIFEST,
    GoldDataset,
    GoldManifest,
    GoldSplit,
)
from lakehouse_cv.contract.silver_tables import (
    ROWS_PER_ROW_GROUP,
    boxes_file,
    images_file,
)
from lakehouse_cv.settings import CvLakePaths
from lakehouse_cv.transforms.layer_builds import (
    create_build_id,
    publish_build,
    read_silver_manifest,
    silver_build_dir,
)

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


def read_gold_manifest(paths: CvLakePaths) -> GoldManifest | None:
    path = paths.gold_dir() / GOLD_MANIFEST
    if not path.exists():
        return None
    return read_manifest(path=path, model=GoldManifest)


def build_gold(
    *,
    store: LakeStore,
    paths: CvLakePaths,
    specs: Sequence[DatasetSpec],
    code_version: str,
    built_at: datetime,
) -> GoldBuild:
    """Join the current silver of every dataset in `specs`.

    A dataset with no silver fails the build. Gold holds its declared inputs, not what
    happens to be on disk.
    """
    if not specs:
        raise ValueError("Gold needs at least one dataset.")
    build_id = create_build_id()
    gold_dir = paths.gold_build_dir(build_id)

    datasets: list[GoldDataset] = []
    num_images = num_boxes = 0
    with store.duckdb() as connection:
        for spec in specs:
            silver = read_silver_manifest(paths=paths, name=spec.name)
            silver_dir = silver_build_dir(paths=paths, manifest=silver)
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
                }
                num_images += _write_rows(
                    connection=connection,
                    rows=_IMAGE_ROWS,
                    parameters={
                        **constants,
                        "silver": str(images_file(build_dir=silver_dir, split=split)),
                        "image_root": gold_split.image_root,
                        "license": silver.license,
                    },
                    schema=GOLD_IMAGE_SCHEMA,
                    path=gold_images_file(
                        build_dir=gold_dir, dataset=spec.name, split=split
                    ),
                )
                num_boxes += _write_rows(
                    connection=connection,
                    rows=_BOX_ROWS,
                    parameters={
                        **constants,
                        "silver": str(boxes_file(build_dir=silver_dir, split=split)),
                    },
                    schema=GOLD_BOX_SCHEMA,
                    path=gold_boxes_file(
                        build_dir=gold_dir, dataset=spec.name, split=split
                    ),
                )
            datasets.append(
                GoldDataset(
                    dataset=spec.name,
                    license=silver.license,
                    commercial_use=silver.commercial_use,
                    silver_code_version=silver.code_version,
                    silver_build_id=silver.build_id,
                    correction_snapshot_id=silver.correction_snapshot_id,
                    splits=splits,
                )
            )

    manifest = GoldManifest(
        build_id=build_id,
        built_at=built_at,
        code_version=code_version,
        datasets=datasets,
    )
    publish_build(manifest_path=paths.gold_dir() / GOLD_MANIFEST, manifest=manifest)
    return GoldBuild(manifest=manifest, num_images=num_images, num_boxes=num_boxes)


def _write_rows(
    *,
    connection: duckdb.DuckDBPyConnection,
    rows: str,
    parameters: dict[str, object],
    schema: pa.Schema,
    path: UPath,
) -> int:
    """Write the rows of one split, in the columns and the types of the schema."""
    selected = ", ".join(schema.names)
    reader = connection.execute(
        query=f"select {selected} from ({rows})", parameters=parameters
    ).to_arrow_reader(ROWS_PER_ROW_GROUP)
    count = 0
    with open_parquet_writer(path=path, schema=schema) as writer:
        for batch in reader:
            # The cast is the write time validation: it rejects a null in a column
            # that the schema declares not null.
            writer.write_batch(batch.cast(schema))
            count += batch.num_rows
    return count
