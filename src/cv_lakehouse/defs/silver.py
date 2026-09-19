#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Silver assets: one dataset, normalised onto the class registry, written as Parquet.

When `CV_LAKEHOUSE_TRITON_URL` names a server, the same run also embeds every image
and every box crop with MobileCLIP. No embedding is cached, so a rematerialisation
embeds the dataset again.
"""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import dagster as dg
import duckdb

from cv_lakehouse.class_registry import (
    CanonicalClass,
    class_registry_sha256_fingerprint,
    sha256_fingerprint,
)
from cv_lakehouse.defs.resources import LakeResource
from cv_lakehouse.embeddings import (
    EMBEDDING_MODEL,
    Embedder,
    TritonEmbedder,
    clear_embeddings,
    write_crop_embeddings,
    write_image_embeddings,
)
from cv_lakehouse.manifests import (
    BRONZE_MANIFEST,
    SILVER_MANIFEST,
    BronzeManifest,
    SilverManifest,
    read_manifest,
    write_manifest,
)
from cv_lakehouse.normalization import RawImageNormalizer
from cv_lakehouse.silver_schema import boxes_file, images_file, write_split
from cv_lakehouse.sources.base import read_raw_images
from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME

# Clipping writes a coordinate back as a float, so a box that ends exactly on the edge
# can land a hair either side of it. Only a real overflow should fail the check.
EDGE_TOLERANCE_PIXELS = 1e-6


# Bump this when the normalisation rules change, such as clipping or the drop policy.
SILVER_LOGIC_VERSION = "4"


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
        deps=[dg.AssetKey(["bronze", name])],
        group_name="silver",
        kinds={"file"},
        code_version=code_version,
        description=(
            f"{name} on the class registry, as one images and one boxes Parquet per "
            "split. Per box `attr_*` columns carry what the source published; "
            "LightlyStudio cannot display them yet. A configured Triton server adds "
            f"one {EMBEDDING_MODEL} embedding per image and per box crop."
        ),
        metadata={"license": spec.license, "commercial_use": spec.commercial_use},
    )
    def _silver(
        context: dg.AssetExecutionContext, lake: LakeResource
    ) -> dg.MaterializeResult:
        bronze_dir = lake.paths.bronze_dir(name)
        bronze = read_manifest(path=bronze_dir / BRONZE_MANIFEST, model=BronzeManifest)
        silver_dir = lake.paths.silver_dir(name)

        embedder: Embedder | None = (
            TritonEmbedder(lake.triton_url) if lake.triton_url is not None else None
        )
        if embedder is None:
            context.log.warning(
                "CV_LAKEHOUSE_TRITON_URL is unset, so this run writes no embedding."
            )

        # Counted by the writer as it streams, not read back off the Parquet. These
        # go to the run log and the Dagster metadata, and no further: a materialisation
        # reports what it did, and the manifest describes what the layer is.
        num_images = num_boxes = num_embeddings = num_crop_embeddings = 0
        dropped_box_reasons = Counter[str]()
        for split in bronze.splits:
            normalizer = RawImageNormalizer(
                category_map=spec.category_map, default_class=spec.default_class
            )
            written_images, written_boxes = write_split(
                silver_dir=silver_dir,
                dataset=name,
                split=split,
                images=normalizer.normalize_to_silver_images(
                    read_raw_images(source=source, bronze_dir=bronze_dir, split=split)
                ),
            )
            if embedder is None:
                clear_embeddings(silver_dir=silver_dir, split=split)
                written_vectors, written_crops = 0, 0
            else:
                written_vectors, written_crops = _embed_split(
                    embedder=embedder,
                    silver_dir=silver_dir,
                    dataset=name,
                    split=split,
                    image_root=bronze.image_roots[split],
                )
            num_images += written_images
            num_boxes += written_boxes
            num_embeddings += written_vectors
            num_crop_embeddings += written_crops
            dropped_box_reasons.update(normalizer.dropped_box_reasons)
            context.log.info(
                f"{split}: {written_images} images, {written_boxes} boxes, "
                f"{sum(normalizer.dropped_box_reasons.values())} dropped, "
                f"{written_vectors + written_crops} embeddings"
            )

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
            ),
        )
        return dg.MaterializeResult(
            metadata=_build_run_metadata(
                license_name=spec.license,
                num_images=num_images,
                num_boxes=num_boxes,
                num_embeddings=num_embeddings,
                num_crop_embeddings=num_crop_embeddings,
                dropped_box_reasons=dropped_box_reasons,
                embedded=embedder is not None,
            )
        )

    return _silver


def _embed_split(
    *,
    embedder: Embedder,
    silver_dir: Path,
    dataset: str,
    split: str,
    image_root: str,
) -> tuple[int, int]:
    """Embed one split's images and box crops. Return the image and the crop count."""
    num_embeddings = write_image_embeddings(
        embedder=embedder,
        silver_dir=silver_dir,
        dataset=dataset,
        split=split,
        image_root=image_root,
    )
    num_crop_embeddings = write_crop_embeddings(
        embedder=embedder,
        silver_dir=silver_dir,
        dataset=dataset,
        split=split,
        image_root=image_root,
    )
    return num_embeddings, num_crop_embeddings


def _build_run_metadata(
    *,
    license_name: str,
    num_images: int,
    num_boxes: int,
    num_embeddings: int,
    num_crop_embeddings: int,
    dropped_box_reasons: Mapping[str, int],
    embedded: bool,
) -> dict:
    """What this run did, for the Dagster UI. Not persisted beside the data."""
    return {
        "dagster/row_count": num_images,
        "num_boxes": num_boxes,
        "num_embeddings": num_embeddings,
        "num_crop_embeddings": num_crop_embeddings,
        "embedding_model": EMBEDDING_MODEL if embedded else "none",
        "license": license_name,
        "dropped_reasons": dg.MetadataValue.json(dict(dropped_box_reasons)),
    }


def build_silver_checks(name: str) -> list[dg.AssetChecksDefinition]:
    key = dg.AssetKey(["silver", name])

    @dg.asset_check(
        asset=key,
        name="boxes_are_valid",
        description="Every box is inside its image, has an area, and a known class.",
        blocking=True,
    )
    def _boxes_are_valid(lake: LakeResource) -> dg.AssetCheckResult:
        """Check every box with one query per split, rather than a sample."""
        silver_dir = lake.paths.silver_dir(name)
        problems: list[str] = []
        checked = 0
        for split in _read_manifest_splits(lake=lake, name=name):
            with duckdb.connect() as connection:
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
    def _images_exist(lake: LakeResource) -> dg.AssetCheckResult:
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
            root = Path(manifest.image_roots[split])
            with duckdb.connect() as connection:
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

    return [_boxes_are_valid, _images_exist]


_IMAGE_NAME_QUERY = "select file_name from read_parquet($path) order by file_name"


# One pass over the join. The left join also catches a box whose image is not listed,
# which the COCO reader could never report because it indexed labels by image.
_BOX_PROBLEM_QUERY = """
with joined as (
    select
        b.file_name,
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


def _count_boxes(connection: duckdb.DuckDBPyConnection, path: Path) -> int:
    row = connection.execute(
        query="select count(*) from read_parquet($path)",
        parameters={"path": str(path)},
    ).fetchone()
    return 0 if row is None else int(row[0])


def _read_manifest_splits(lake: LakeResource, name: str):
    manifest = read_manifest(
        path=lake.paths.silver_dir(name) / SILVER_MANIFEST, model=SilverManifest
    )
    return manifest.splits


_names = list(SOURCE_BY_NAME)
defs = dg.Definitions(
    assets=[build_silver_asset(name) for name in _names],
    asset_checks=[check for name in _names for check in build_silver_checks(name)],
)
