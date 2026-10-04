#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The gold asset: every silver dataset in one table set, with no flagged box."""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

from datetime import UTC, datetime

import dagster as dg

from lakehouse_core.fingerprints import sha256_fingerprint
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.gold_build import build_gold_version, read_gold_manifest

GOLD_KEY = dg.AssetKey(["gold", "current"])

# Bump this when the gold rules change, such as which boxes drop out.
GOLD_LOGIC_VERSION = "2"


def build_gold_asset() -> dg.AssetsDefinition:
    specs = [source.spec for source in SOURCE_BY_NAME.values()]
    code_version = sha256_fingerprint(
        [GOLD_LOGIC_VERSION, *(spec.split_roles_sha256_fingerprint for spec in specs)]
    )

    @dg.asset(
        key=GOLD_KEY,
        deps=[dg.AssetKey(["silver", spec.name]) for spec in specs],
        group_name="gold",
        kinds={"file"},
        code_version=code_version,
        description=(
            "Every materialized silver dataset as one images and one boxes Parquet "
            "per dataset and split. A row carries its role, and no box is flagged."
        ),
    )
    def _gold(
        context: dg.AssetExecutionContext, lake: CvLakeResource
    ) -> dg.MaterializeResult:
        build = build_gold_version(
            store=lake.store,
            paths=lake.paths,
            specs=specs,
            code_version=code_version,
            built_at=datetime.now(tz=UTC),
        )
        if build.skipped_datasets:
            context.log.warning(
                f"No silver on disk for {list(build.skipped_datasets)}, so gold "
                "leaves them out."
            )
        return dg.MaterializeResult(
            metadata={
                "version": build.manifest.version,
                "datasets": [dataset.dataset for dataset in build.manifest.datasets],
                "dagster/row_count": build.num_images,
                "num_boxes": build.num_boxes,
            }
        )

    return _gold


def build_gold_checks() -> list[dg.AssetChecksDefinition]:
    @dg.asset_check(
        asset=GOLD_KEY,
        name="keys_are_unique",
        description="Every image and every box has one row, and every box an image.",
        blocking=True,
    )
    def _keys_are_unique(lake: CvLakeResource) -> dg.AssetCheckResult:
        manifest = read_gold_manifest(lake.paths)
        if manifest is None:
            return dg.AssetCheckResult(passed=False, metadata={"problem": "no gold"})
        version_dir = lake.paths.gold_version_dir(manifest.version)
        with lake.store.duckdb() as connection:
            problems = connection.execute(
                query=_KEY_PROBLEM_QUERY,
                parameters={
                    "images": str(version_dir / "images" / "*" / "*.parquet"),
                    "boxes": str(version_dir / "boxes" / "*" / "*.parquet"),
                },
            ).fetchall()
        return dg.AssetCheckResult(
            passed=not problems,
            metadata={
                "num_problems": len(problems),
                "examples": [f"{key}: {problem}" for key, problem in problems[:10]],
            },
        )

    return [_keys_are_unique]


_KEY_PROBLEM_QUERY = """
select image_id, 'repeated image id' from read_parquet($images)
group by image_id having count(*) > 1
union all
select box_id, 'repeated box id' from read_parquet($boxes)
group by box_id having count(*) > 1
union all
select box_id, 'no image row' from read_parquet($boxes)
where image_id not in (select image_id from read_parquet($images))
"""


defs = dg.Definitions(assets=[build_gold_asset()], asset_checks=build_gold_checks())
