#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Silver assets: one dataset, normalised onto the class registry, written as Parquet.

A run writes a new build, checks every box, and only then replaces the manifest. A run
that fails leaves the last good silver as it was.
"""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

from collections import Counter
from collections.abc import Mapping, Sequence

import dagster as dg
from upath import UPath

from lakehouse_core.bronze_manifest import BRONZE_MANIFEST
from lakehouse_core.fingerprints import sha256_fingerprint
from lakehouse_core.lake_store import LakeStore
from lakehouse_core.manifest_files import read_manifest
from lakehouse_cv.contract.class_registry import (
    CanonicalClass,
    class_registry_sha256_fingerprint,
)
from lakehouse_cv.contract.manifests import (
    SILVER_MANIFEST,
    CvBronzeManifest,
    SilverManifest,
    build_dir,
)
from lakehouse_cv.contract.silver_tables import boxes_file, images_file
from lakehouse_cv.defs.corrections import (
    build_corrections_key,
    list_event_paths,
    read_corrections_manifest,
)
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.base import read_raw_images
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.correction_overlay import CorrectionOverlay
from lakehouse_cv.transforms.layer_builds import (
    derive_build_id,
    discard_build,
    publish_build,
    read_silver_manifest,
    reuse_whole_build,
    silver_build_dir,
)
from lakehouse_cv.transforms.normalization import RawImageNormalizer
from lakehouse_cv.transforms.progress_log import ProgressLog, iter_with_progress
from lakehouse_cv.transforms.silver_writer import write_split

# Clipping writes a coordinate back as a float, so a box that ends exactly on the edge
# can land a hair either side of it. Only a real overflow should fail the check.
EDGE_TOLERANCE_PIXELS = 1e-6


# Bump this when the normalisation rules change, such as clipping or the drop policy.
SILVER_LOGIC_VERSION = "6"


def build_silver_asset(name: str) -> dg.AssetsDefinition:
    source = SOURCE_BY_NAME[name]
    spec = source.spec
    code_version = sha256_fingerprint(
        [
            SILVER_LOGIC_VERSION,
            class_registry_sha256_fingerprint(),
            spec.category_map_sha256_fingerprint,
        ]
    )

    @dg.asset(
        key=dg.AssetKey(["silver", name]),
        deps=[dg.AssetKey(["bronze", name]), build_corrections_key(name)],
        group_name="silver",
        kinds={"file"},
        code_version=code_version,
        description=(
            f"{name} on the class registry, with the corrections of the curators "
            "applied, as one images and one boxes Parquet per split. Per box "
            "`attr_*` columns carry what the source published; "
            "LightlyStudio cannot display them yet."
        ),
        metadata={"license": spec.license, "commercial_use": spec.commercial_use},
    )
    def _silver(
        context: dg.AssetExecutionContext, lake: CvLakeResource
    ) -> dg.MaterializeResult:
        bronze_path = lake.paths.bronze_dir(name) / BRONZE_MANIFEST
        bronze = read_manifest(path=bronze_path, model=CvBronzeManifest)
        bronze_dir = lake.store.resolve(bronze.path)
        silver_dir = lake.paths.silver_dir(name)
        corrections = read_corrections_manifest(paths=lake.paths, name=name)
        # Bronze pins every file it names, and an event file never changes, so the
        # manifests name the inputs.
        build_id = derive_build_id(
            [
                code_version,
                bronze_path.read_text(),
                *(
                    f"{event_file.chain_id}/{event_file.event_file_id}"
                    for event_file in corrections.event_files
                ),
            ]
        )
        if reuse_whole_build(
            manifest_path=silver_dir / SILVER_MANIFEST,
            build_id=build_id,
            model=SilverManifest,
        ):
            context.log.info(f"Build {build_id} is whole, so this run writes nothing.")
            return dg.MaterializeResult(
                data_version=dg.DataVersion(build_id),
                metadata={"build_id": build_id, "is_reused": True},
            )
        staged_dir = build_dir(layer_dir=silver_dir, build_id=build_id)
        overlay = CorrectionOverlay.from_event_files(
            list_event_paths(
                paths=lake.paths,
                manifest=corrections,
                last_event=corrections.last_event,
            )
        )

        # Counted by the writer as it streams, not read back off the Parquet. These
        # go to the run log and the Dagster metadata, and no further: a materialisation
        # reports what it did, and the manifest describes what the layer is.
        num_images = num_boxes = 0
        dropped_box_reasons = Counter[str]()
        for split in bronze.splits:
            normalizer = RawImageNormalizer(
                dataset=name,
                split=split,
                category_map=spec.category_map,
                default_class=spec.default_class,
            )
            with lake.disk_lease.hold(
                target=bronze_dir, reason=f"Silver {name} {split}", log=context.log
            ):
                written_images, written_boxes = write_split(
                    build_dir=staged_dir,
                    dataset=name,
                    split=split,
                    images=iter_with_progress(
                        items=overlay.apply_to_silver_images(
                            split=split,
                            images=normalizer.normalize_to_silver_images(
                                read_raw_images(
                                    source=source, bronze_dir=bronze_dir, split=split
                                )
                            ),
                        ),
                        progress=ProgressLog(
                            log=context.log.info, label=f"{split} images normalised"
                        ),
                    ),
                )
            num_images += written_images
            num_boxes += written_boxes
            dropped_box_reasons.update(normalizer.dropped_box_reasons)
            context.log.info(
                f"{split}: {written_images} images, {written_boxes} boxes, "
                f"{sum(normalizer.dropped_box_reasons.values())} dropped"
            )
        # The overlay counts across the splits, so its tally joins once.
        dropped_box_reasons.update(overlay.dropped_box_reasons)

        problems = find_box_problems(
            store=lake.store, build_dir=staged_dir, splits=bronze.splits
        )
        if problems:
            discard_build(layer_dir=silver_dir, build_id=build_id)
            raise dg.Failure(
                description=f"{len(problems)} boxes are not valid, so silver keeps "
                "its last build.",
                metadata={"examples": problems[:10]},
            )
        publish_build(
            manifest_path=silver_dir / SILVER_MANIFEST,
            manifest=SilverManifest(
                dataset=name,
                code_version=code_version,
                build_id=build_id,
                license=spec.license,
                commercial_use=spec.commercial_use,
                image_roots=bronze.image_roots,
                splits=list(bronze.splits),
                last_event=corrections.last_event,
            ),
        )
        return dg.MaterializeResult(
            data_version=dg.DataVersion(build_id),
            metadata=_build_run_metadata(
                build_id=build_id,
                license_name=spec.license,
                num_images=num_images,
                num_boxes=num_boxes,
                dropped_box_reasons=dropped_box_reasons,
                num_event_files=len(corrections.event_files),
                num_corrections_applied=overlay.num_applied,
            ),
        )

    return _silver


def find_box_problems(
    *, store: LakeStore, build_dir: UPath, splits: Sequence[str]
) -> list[str]:
    """Check every box with one query per split, rather than a sample."""
    problems: list[str] = []
    for split in splits:
        with store.duckdb() as connection:
            rows = connection.execute(
                query=_BOX_PROBLEM_QUERY,
                parameters={
                    "boxes": str(boxes_file(build_dir=build_dir, split=split)),
                    "images": str(images_file(build_dir=build_dir, split=split)),
                    "ids": [member.value for member in CanonicalClass],
                    "tolerance": EDGE_TOLERANCE_PIXELS,
                },
            ).fetchall()
        problems.extend(
            f"{split}:{file_name}: {problem}" for file_name, problem in rows
        )
    return problems


def _build_run_metadata(
    *,
    build_id: str,
    license_name: str,
    num_images: int,
    num_boxes: int,
    dropped_box_reasons: Mapping[str, int],
    num_event_files: int,
    num_corrections_applied: int,
) -> dict:
    """What this run did, for the Dagster UI. Not persisted beside the data."""
    return {
        "build_id": build_id,
        "is_reused": False,
        "dagster/row_count": num_images,
        "num_boxes": num_boxes,
        "license": license_name,
        "dropped_reasons": dg.MetadataValue.json(dict(dropped_box_reasons)),
        "num_event_files": num_event_files,
        "num_corrections_applied": num_corrections_applied,
    }


def build_silver_checks(name: str) -> list[dg.AssetChecksDefinition]:
    key = dg.AssetKey(["silver", name])

    @dg.asset_check(
        asset=key,
        name="boxes_are_valid",
        description=(
            "Every box has a unique id, is inside its image, has an area, and a "
            "known class. The asset runs the same check before it publishes a build."
        ),
        blocking=True,
    )
    def _boxes_are_valid(lake: CvLakeResource) -> dg.AssetCheckResult:
        manifest = read_silver_manifest(paths=lake.paths, name=name)
        problems = find_box_problems(
            store=lake.store,
            build_dir=silver_build_dir(paths=lake.paths, manifest=manifest),
            splits=manifest.splits,
        )
        return dg.AssetCheckResult(
            passed=not problems,
            metadata={"num_problems": len(problems), "examples": problems[:10]},
        )

    @dg.asset_check(
        asset=key,
        name="images_exist",
        description="Every image the annotations name is on disk.",
    )
    def _images_exist(lake: CvLakeResource) -> dg.AssetCheckResult:
        """Read one column and stat what it names.

        Reading a single Parquet column is cheap enough to check every image, so this no
        longer samples.
        """
        manifest = read_silver_manifest(paths=lake.paths, name=name)
        current_dir = silver_build_dir(paths=lake.paths, manifest=manifest)
        missing: list[str] = []
        checked = 0
        for split in manifest.splits:
            root = lake.store.resolve(manifest.image_roots[split])
            with lake.store.duckdb() as connection:
                names = connection.execute(
                    query=_IMAGE_NAME_QUERY,
                    parameters={
                        "path": str(images_file(build_dir=current_dir, split=split))
                    },
                ).fetchall()
            for (file_name,) in names:
                checked += 1
                if not (root / file_name).exists():
                    missing.append(f"{split}:{file_name}")
        return dg.AssetCheckResult(
            passed=not missing,
            metadata={
                "num_checked": checked,
                "num_missing": len(missing),
                "examples": missing[:10],
            },
        )

    @dg.asset_check(
        asset=key,
        name="corrections_are_applied",
        description=(
            "Every edited box that is no tombstone is in silver, and every label of a "
            "curator is a class of the registry."
        ),
    )
    def _corrections_are_applied(lake: CvLakeResource) -> dg.AssetCheckResult:
        """Report what the overlay could not apply as the curator meant it.

        Neither case fails the run. A box with an unknown label is in silver as
        `other`, and an event with no box changes nothing.
        """
        manifest = read_silver_manifest(paths=lake.paths, name=name)
        folded = list_event_paths(
            paths=lake.paths,
            manifest=read_corrections_manifest(paths=lake.paths, name=name),
            last_event=manifest.last_event,
        )
        if not folded:
            return dg.AssetCheckResult(passed=True, metadata={"num_event_files": 0})
        parameters = {
            "events": [str(path) for path in folded],
            "boxes": str(
                silver_build_dir(paths=lake.paths, manifest=manifest)
                / "boxes"
                / "*.parquet"
            ),
        }
        with lake.store.duckdb() as connection:
            orphans = connection.execute(
                query=_ORPHAN_EVENT_QUERY, parameters=parameters
            ).fetchall()
            unknown_labels = connection.execute(
                query=_UNKNOWN_LABEL_QUERY,
                parameters={
                    "events": parameters["events"],
                    "names": sorted(CanonicalClass.all_class_names()),
                },
            ).fetchall()
        return dg.AssetCheckResult(
            passed=not orphans and not unknown_labels,
            severity=dg.AssetCheckSeverity.WARN,
            metadata={
                "num_event_files": len(folded),
                "num_orphans": len(orphans),
                "orphans": [box_id for (box_id,) in orphans[:10]],
                "unknown_labels": [label for (label,) in unknown_labels],
            },
        )

    return [_boxes_are_valid, _images_exist, _corrections_are_applied]


_IMAGE_NAME_QUERY = "select file_name from read_parquet($path) order by file_name"


# One pass over the join. The left join also catches a box whose image is not listed,
# which the COCO reader could never report because it indexed labels by image.
_BOX_PROBLEM_QUERY = """
with joined as (
    select
        b.file_name,
        count(*) over (partition by b.box_id) as box_id_count,
        b.class_id,
        b.x,
        b.y,
        b.w,
        b.h,
        i.width,
        i.height
    from read_parquet($boxes) b
    left join read_parquet($images) i using (dataset, split, file_name)
),
judged as (
    select
        file_name,
        case
            when width is null then 'no image row'
            when box_id_count > 1 then 'repeated box id'
            when not list_contains($ids, class_id) then 'unknown class ' || class_id
            when w <= 0 or h <= 0 then 'zero area box'
            when x < -$tolerance or y < -$tolerance
                then 'box starts outside the image'
            when x + w > width + $tolerance or y + h > height + $tolerance
                then 'box ends outside the image'
        end as problem
    from joined
)
select file_name, problem from judged where problem is not null
"""


# The latest event of each box, in the order of the files and then of the log.
_LATEST_EVENTS = """
select * from read_parquet($events, filename = true)
qualify row_number() over (
    partition by box_id order by list_position($events, filename) desc,
        log_sequence desc
) = 1
"""

# An edited box that is no tombstone and that silver does not hold. Its image is not in
# the split, or the box has no area inside the image. A tombstone leaves no trace either
# way, so this cannot report it.
_ORPHAN_EVENT_QUERY = f"""
with latest as ({_LATEST_EVENTS})
select e.box_id
from latest e
left join read_parquet($boxes) b on b.box_id = e.box_id
where not e.is_deleted and b.box_id is null
order by e.box_id
"""

_UNKNOWN_LABEL_QUERY = """
select distinct label_name
from read_parquet($events)
where label_name is not null and not list_contains($names, lower(label_name))
order by label_name
"""


_names = list(SOURCE_BY_NAME)
defs = dg.Definitions(
    assets=[build_silver_asset(name) for name in _names],
    asset_checks=[check for name in _names for check in build_silver_checks(name)],
)
