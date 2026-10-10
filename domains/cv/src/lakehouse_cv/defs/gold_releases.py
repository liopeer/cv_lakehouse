#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The asset that publishes the val and test rows of gold as a numbered release."""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

from datetime import UTC, datetime

import dagster as dg

from lakehouse_cv.defs.gold import GOLD_KEY
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.transforms.gold_release import (
    IdenticalReleaseError,
    find_changed_release_files,
    list_release_numbers,
    write_eval_release,
)

EVAL_RELEASE_KEY = dg.AssetKey(["gold", "eval_release"])


def build_eval_release_asset() -> dg.AssetsDefinition:
    @dg.asset(
        key=EVAL_RELEASE_KEY,
        deps=[GOLD_KEY],
        group_name="gold",
        kinds={"file"},
        description=(
            "A frozen copy of the val and test rows of gold. Every run publishes the "
            "next number, and no run changes an earlier one. Run it by hand, when a "
            "round of corrections is done."
        ),
    )
    def _eval_release(lake: CvLakeResource) -> dg.MaterializeResult:
        try:
            manifest = write_eval_release(
                paths=lake.paths, created_at=datetime.now(tz=UTC)
            )
        except IdenticalReleaseError as error:
            raise dg.Failure(description=str(error)) from error
        return dg.MaterializeResult(
            metadata={
                "release": manifest.release,
                "gold_code_version": manifest.gold_code_version,
                "datasets": [dataset.dataset for dataset in manifest.datasets],
                "num_files": len(manifest.files),
            }
        )

    return _eval_release


def build_eval_release_checks() -> list[dg.AssetChecksDefinition]:
    @dg.asset_check(
        asset=EVAL_RELEASE_KEY,
        name="releases_are_unchanged",
        description="Every file of every release has the checksum its manifest pins.",
        blocking=True,
    )
    def _releases_are_unchanged(lake: CvLakeResource) -> dg.AssetCheckResult:
        changed = find_changed_release_files(lake.paths)
        return dg.AssetCheckResult(
            passed=not changed,
            metadata={
                "num_releases": len(list_release_numbers(lake.paths)),
                "changed_files": changed[:10],
            },
        )

    return [_releases_are_unchanged]


defs = dg.Definitions(
    assets=[build_eval_release_asset()], asset_checks=build_eval_release_checks()
)
