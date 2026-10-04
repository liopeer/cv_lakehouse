#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Silver assets: one dataset, normalised onto the class registry, written as Parquet.

When `CV_LAKEHOUSE_TRITON_URL` names a server, the same run also embeds every image
and every box crop with MobileCLIP. A run on the same code keeps the vector of an
image, and of a box that did not move, from the run before.
"""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

from collections import Counter
from collections.abc import Mapping

import dagster as dg
import duckdb
from upath import UPath

from lakehouse_core.bronze_manifest import BRONZE_MANIFEST
from lakehouse_core.fingerprints import sha256_fingerprint
from lakehouse_core.lake_store import LakeStore, is_local
from lakehouse_core.manifest_files import read_manifest, write_manifest
from lakehouse_cv.contract.class_registry import (
    CanonicalClass,
    class_registry_sha256_fingerprint,
)
from lakehouse_cv.contract.manifests import (
    SILVER_MANIFEST,
    CvBronzeManifest,
    SilverManifest,
)
from lakehouse_cv.contract.silver_tables import boxes_file, images_file
from lakehouse_cv.defs.corrections import (
    build_corrections_key,
    read_corrections_manifest,
)
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.base import read_raw_images
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.correction_overlay import CorrectionOverlay
from lakehouse_cv.transforms.embeddings import (
    EMBEDDING_MODEL,
    Embedder,
    PreviousSplit,
    TritonEmbedder,
    clear_embeddings,
    discard_previous_split,
    reuse_or_embed_crops,
    reuse_or_embed_images,
    stash_previous_split,
    write_crop_embeddings,
    write_image_embeddings,
)
from lakehouse_cv.transforms.normalization import RawImageNormalizer
from lakehouse_cv.transforms.silver_writer import write_split

# Clipping writes a coordinate back as a float, so a box that ends exactly on the edge
# can land a hair either side of it. Only a real overflow should fail the check.
EDGE_TOLERANCE_PIXELS = 1e-6


# Bump this when the normalisation rules change, such as clipping or the drop policy.
SILVER_LOGIC_VERSION = "6"


def build_silver_asset(name: str) -> dg.AssetsDefinition:
    source = SOURCE_BY_NAME[name]
    spec = source.spec
    # The model name belongs in the version and the server URL does not: moving the
    # server leaves the weights, and therefore the vectors, exactly as they were.
    code_version = sha256_fingerprint(
        [
            SILVER_LOGIC_VERSION,
            class_registry_sha256_fingerprint(),
            spec.category_map_sha256_fingerprint,
            EMBEDDING_MODEL,
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
            "LightlyStudio cannot display them yet. A configured Triton server adds "
            f"one {EMBEDDING_MODEL} embedding per image and per box crop."
        ),
        metadata={"license": spec.license, "commercial_use": spec.commercial_use},
    )
    def _silver(
        context: dg.AssetExecutionContext, lake: CvLakeResource
    ) -> dg.MaterializeResult:
        bronze_dir = lake.paths.bronze_dir(name)
        bronze = read_manifest(
            path=bronze_dir / BRONZE_MANIFEST, model=CvBronzeManifest
        )
        silver_dir = lake.paths.silver_dir(name)
        corrections = read_corrections_manifest(paths=lake.paths, name=name)
        snapshot = corrections.snapshots[-1] if corrections.snapshots else None
        overlay = CorrectionOverlay.from_snapshot(
            None
            if snapshot is None
            else lake.paths.corrections_dir(name) / snapshot.file.path
        )

        embedder: Embedder | None = (
            TritonEmbedder(lake.triton_url) if lake.triton_url is not None else None
        )
        if embedder is None:
            context.log.warning(
                "CV_LAKEHOUSE_TRITON_URL is unset, so this run writes no embedding."
            )

        # A vector of the run before is still right when only the corrections changed
        # since: the same code, on the same model, over the pixels that bronze pins.
        previous_manifest_path = silver_dir / SILVER_MANIFEST
        reuses_embeddings = (
            embedder is not None
            and previous_manifest_path.exists()
            and _describes_this_code(
                manifest=read_manifest(
                    path=previous_manifest_path, model=SilverManifest
                ),
                code_version=code_version,
            )
        )

        # Counted by the writer as it streams, not read back off the Parquet. These
        # go to the run log and the Dagster metadata, and no further: a materialisation
        # reports what it did, and the manifest describes what the layer is.
        num_images = num_boxes = num_embeddings = num_crop_embeddings = 0
        num_reused_embeddings = 0
        dropped_box_reasons = Counter[str]()
        for split in bronze.splits:
            normalizer = RawImageNormalizer(
                dataset=name,
                split=split,
                category_map=spec.category_map,
                default_class=spec.default_class,
            )
            previous = (
                stash_previous_split(silver_dir=silver_dir, split=split)
                if reuses_embeddings
                else None
            )
            written_images, written_boxes = write_split(
                silver_dir=silver_dir,
                dataset=name,
                split=split,
                images=overlay.apply_to_silver_images(
                    split=split,
                    images=normalizer.normalize_to_silver_images(
                        read_raw_images(
                            source=source, bronze_dir=bronze_dir, split=split
                        )
                    ),
                ),
            )
            if embedder is None:
                clear_embeddings(silver_dir=silver_dir, split=split)
                written_vectors, written_crops, reused = 0, 0, 0
            else:
                written_vectors, written_crops, reused = _embed_split(
                    store=lake.store,
                    embedder=embedder,
                    silver_dir=silver_dir,
                    dataset=name,
                    split=split,
                    image_root=bronze.image_roots[split],
                    previous=previous,
                )
            discard_previous_split(silver_dir=silver_dir, split=split)
            num_reused_embeddings += reused
            num_images += written_images
            num_boxes += written_boxes
            num_embeddings += written_vectors
            num_crop_embeddings += written_crops
            dropped_box_reasons.update(normalizer.dropped_box_reasons)
            context.log.info(
                f"{split}: {written_images} images, {written_boxes} boxes, "
                f"{sum(normalizer.dropped_box_reasons.values())} dropped, "
                f"{written_vectors + written_crops} embeddings, {reused} of them reused"
            )
        # The overlay counts across the splits, so its tally joins once.
        dropped_box_reasons.update(overlay.dropped_box_reasons)

        write_manifest(
            path=silver_dir / SILVER_MANIFEST,
            manifest=SilverManifest(
                dataset=name,
                code_version=code_version,
                license=spec.license,
                commercial_use=spec.commercial_use,
                image_roots=bronze.image_roots,
                splits=list(bronze.splits),
                embedding_model=EMBEDDING_MODEL if embedder is not None else None,
                correction_snapshot_id=(
                    None if snapshot is None else snapshot.snapshot_id
                ),
            ),
        )
        return dg.MaterializeResult(
            metadata=_build_run_metadata(
                license_name=spec.license,
                num_images=num_images,
                num_boxes=num_boxes,
                num_embeddings=num_embeddings,
                num_crop_embeddings=num_crop_embeddings,
                num_reused_embeddings=num_reused_embeddings,
                dropped_box_reasons=dropped_box_reasons,
                embedded=embedder is not None,
                correction_snapshot_id=(
                    "none" if snapshot is None else snapshot.snapshot_id
                ),
                num_corrections_applied=overlay.num_applied,
            )
        )

    return _silver


def _describes_this_code(manifest: SilverManifest, code_version: str) -> bool:
    return (
        manifest.code_version == code_version
        and manifest.embedding_model == EMBEDDING_MODEL
    )


def _embed_split(
    *,
    store: LakeStore,
    embedder: Embedder,
    silver_dir: UPath,
    dataset: str,
    split: str,
    image_root: str,
    previous: PreviousSplit | None,
) -> tuple[int, int, int]:
    """Embed the images and the box crops of one split.

    Return the image count, the crop count, and how many vectors of the two came from
    the run before.
    """
    root = store.resolve(image_root)
    if not is_local(root):
        raise NotImplementedError(
            f"Triton reads images only from a local disk, and {root} is not on one."
        )
    target = {
        "embedder": embedder,
        "silver_dir": silver_dir,
        "dataset": dataset,
        "split": split,
        "image_root": str(root),
    }
    if previous is None:
        return write_image_embeddings(**target), write_crop_embeddings(**target), 0
    num_embeddings, reused_images = reuse_or_embed_images(
        **target, store=store, previous=previous
    )
    num_crops, reused_crops = reuse_or_embed_crops(
        **target, store=store, previous=previous
    )
    return num_embeddings, num_crops, reused_images + reused_crops


def _build_run_metadata(
    *,
    license_name: str,
    num_images: int,
    num_boxes: int,
    num_embeddings: int,
    num_crop_embeddings: int,
    num_reused_embeddings: int,
    dropped_box_reasons: Mapping[str, int],
    embedded: bool,
    correction_snapshot_id: str,
    num_corrections_applied: int,
) -> dict:
    """What this run did, for the Dagster UI. Not persisted beside the data."""
    return {
        "dagster/row_count": num_images,
        "num_boxes": num_boxes,
        "num_embeddings": num_embeddings,
        "num_crop_embeddings": num_crop_embeddings,
        "num_reused_embeddings": num_reused_embeddings,
        "embedding_model": EMBEDDING_MODEL if embedded else "none",
        "license": license_name,
        "dropped_reasons": dg.MetadataValue.json(dict(dropped_box_reasons)),
        "correction_snapshot": correction_snapshot_id,
        "num_corrections_applied": num_corrections_applied,
    }


def build_silver_checks(name: str) -> list[dg.AssetChecksDefinition]:
    key = dg.AssetKey(["silver", name])

    @dg.asset_check(
        asset=key,
        name="boxes_are_valid",
        description=(
            "Every box has a unique id, is inside its image, has an area, and a "
            "known class."
        ),
        blocking=True,
    )
    def _boxes_are_valid(lake: CvLakeResource) -> dg.AssetCheckResult:
        """Check every box with one query per split, rather than a sample."""
        silver_dir = lake.paths.silver_dir(name)
        problems: list[str] = []
        checked = 0
        for split in _read_manifest_splits(lake=lake, name=name):
            with lake.store.duckdb() as connection:
                rows = connection.execute(
                    query=_BOX_PROBLEM_QUERY,
                    parameters={
                        "boxes": str(boxes_file(silver_dir=silver_dir, split=split)),
                        "images": str(images_file(silver_dir=silver_dir, split=split)),
                        "ids": [member.value for member in CanonicalClass],
                        "tolerance": EDGE_TOLERANCE_PIXELS,
                    },
                ).fetchall()
                checked += _count_boxes(
                    connection=connection,
                    path=boxes_file(silver_dir=silver_dir, split=split),
                )
            problems.extend(
                f"{split}:{file_name}: {problem}" for file_name, problem in rows
            )
        return dg.AssetCheckResult(
            passed=not problems,
            metadata={
                "num_boxes": checked,
                "num_problems": len(problems),
                "examples": problems[:10],
            },
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
        manifest = read_manifest(
            path=lake.paths.silver_dir(name) / SILVER_MANIFEST, model=SilverManifest
        )
        silver_dir = lake.paths.silver_dir(name)
        missing: list[str] = []
        checked = 0
        for split in manifest.splits:
            root = lake.store.resolve(manifest.image_roots[split])
            with lake.store.duckdb() as connection:
                names = connection.execute(
                    query=_IMAGE_NAME_QUERY,
                    parameters={
                        "path": str(images_file(silver_dir=silver_dir, split=split))
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
            "Every correction found its box, and every label of a curator is a class "
            "of the registry."
        ),
    )
    def _corrections_are_applied(lake: CvLakeResource) -> dg.AssetCheckResult:
        """Report what the overlay could not apply as the curator meant it.

        Neither case fails the run. A box with an unknown label is in silver as
        `other`, and a correction with no box changes nothing.
        """
        silver_dir = lake.paths.silver_dir(name)
        manifest = read_manifest(
            path=silver_dir / SILVER_MANIFEST, model=SilverManifest
        )
        snapshot_path = next(
            (
                lake.paths.corrections_dir(name) / snapshot.file.path
                for snapshot in read_corrections_manifest(
                    paths=lake.paths, name=name
                ).snapshots
                if snapshot.snapshot_id == manifest.correction_snapshot_id
            ),
            None,
        )
        if snapshot_path is None:
            return dg.AssetCheckResult(passed=True, metadata={"snapshot": "none"})
        parameters = {
            "snapshot": str(snapshot_path),
            "boxes": str(silver_dir / "boxes" / "*.parquet"),
        }
        with lake.store.duckdb() as connection:
            orphans = connection.execute(
                query=_ORPHAN_CORRECTION_QUERY, parameters=parameters
            ).fetchall()
            unknown_labels = connection.execute(
                query=_UNKNOWN_LABEL_QUERY,
                parameters={
                    "snapshot": str(snapshot_path),
                    "names": sorted(CanonicalClass.all_class_names()),
                },
            ).fetchall()
        return dg.AssetCheckResult(
            passed=not orphans and not unknown_labels,
            severity=dg.AssetCheckSeverity.WARN,
            metadata={
                "snapshot": manifest.correction_snapshot_id,
                "num_orphans": len(orphans),
                "orphans": [f"{action} {box_id}" for box_id, action in orphans[:10]],
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


# A changed or an added box that silver does not hold. Its source box is gone, its
# image is not in the split, or the corrected box has no area inside the image. A
# deleted box leaves no trace either way, so this cannot report it.
_ORPHAN_CORRECTION_QUERY = """
select c.box_id, c.action
from read_parquet($snapshot) c
left join read_parquet($boxes) b on b.box_id = c.box_id
where c.action <> 'delete' and b.box_id is null
order by c.box_id
"""

_UNKNOWN_LABEL_QUERY = """
select distinct label_name
from read_parquet($snapshot)
where label_name is not null and not list_contains($names, lower(label_name))
order by label_name
"""


def _count_boxes(connection: duckdb.DuckDBPyConnection, path: UPath) -> int:
    row = connection.execute(
        query="select count(*) from read_parquet($path)",
        parameters={"path": str(path)},
    ).fetchone()
    return 0 if row is None else int(row[0])


def _read_manifest_splits(lake: CvLakeResource, name: str):
    manifest = read_manifest(
        path=lake.paths.silver_dir(name) / SILVER_MANIFEST, model=SilverManifest
    )
    return manifest.splits


_names = list(SOURCE_BY_NAME)
defs = dg.Definitions(
    assets=[build_silver_asset(name) for name in _names],
    asset_checks=[check for name in _names for check in build_silver_checks(name)],
)
