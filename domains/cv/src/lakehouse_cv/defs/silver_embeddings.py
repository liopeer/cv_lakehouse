#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Embedding assets: one MobileCLIP vector per image and per box crop of silver.

The asset reads the silver build that Dagster recorded, and fails when no Triton server
is set. The vectors come from a cache of the model, keyed by the pixels and the crop. A
run embeds only a row that the cache lacks, so a change to the class rules embeds no
crop again.
"""

# Dagster resolves the resource annotations at runtime, so this module must not
# postpone its annotations.

import dagster as dg

from lakehouse_core.bronze_manifest import BRONZE_MANIFEST
from lakehouse_core.column_schemas import build_table_schema_metadata
from lakehouse_core.fingerprints import sha256_fingerprint
from lakehouse_core.lake_store import build_reader_locator
from lakehouse_core.manifest_files import read_manifest
from lakehouse_cv.contract.manifests import (
    EMBEDDINGS_MANIFEST,
    CvBronzeManifest,
    EmbeddingsManifest,
    build_dir,
)
from lakehouse_cv.contract.silver_tables import SILVER_TABLES
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.defs.upstream_builds import read_upstream_build_id
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.embedding_files import (
    EMBEDDING_VERSION,
    VECTOR_TABLES,
    SplitVectorFiles,
    write_vectors,
)
from lakehouse_cv.transforms.embeddings import EMBEDDING_MODEL, TritonEmbedder
from lakehouse_cv.transforms.layer_builds import (
    derive_build_id,
    publish_build,
    read_silver_build,
    reuse_whole_build,
    silver_build_dir,
)

# Long enough for a request that waits in the queue of Triton.
PRESIGNED_URL_SECONDS = 3600

VECTOR_TABLE_NAMES = ("embeddings", "crop_embeddings")


def build_embeddings_key(name: str) -> dg.AssetKey:
    return dg.AssetKey(["silver", f"{name}_embeddings"])


def build_embeddings_asset(name: str) -> dg.AssetsDefinition:
    # The model name belongs in the version and the server URL does not: moving the
    # server leaves the weights, and therefore the vectors, exactly as they were.
    code_version = sha256_fingerprint([EMBEDDING_MODEL, EMBEDDING_VERSION])

    @dg.asset(
        key=build_embeddings_key(name),
        deps=[dg.AssetKey(["silver", name])],
        group_name="embeddings",
        kinds={"silver", "parquet"},
        code_version=code_version,
        description=(
            f"One {EMBEDDING_MODEL} embedding per image and per box crop of silver "
            f"{name}, sorted by the id. Needs CV_LAKEHOUSE_TRITON_URL."
        ),
        metadata=build_table_schema_metadata(
            [table for table in SILVER_TABLES if table.name in VECTOR_TABLE_NAMES]
        ),
    )
    def _embeddings(
        context: dg.AssetExecutionContext, lake: CvLakeResource
    ) -> dg.MaterializeResult:
        if lake.triton_url is None:
            raise dg.Failure(
                description="CV_LAKEHOUSE_TRITON_URL is unset, so nothing can embed."
            )
        embedder = TritonEmbedder(lake.triton_url)
        silver = read_silver_build(
            paths=lake.paths,
            name=name,
            build_id=read_upstream_build_id(
                context=context, key=dg.AssetKey(["silver", name])
            ),
        )
        rows_dir = silver_build_dir(paths=lake.paths, manifest=silver)
        bronze = read_manifest(
            path=lake.paths.bronze_dir(name) / BRONZE_MANIFEST, model=CvBronzeManifest
        )
        embeddings_dir = lake.paths.embeddings_dir(name)
        build_id = derive_build_id([code_version, silver.build_id])
        if reuse_whole_build(
            manifest_path=embeddings_dir / EMBEDDINGS_MANIFEST,
            build_id=build_id,
            model=EmbeddingsManifest,
        ):
            return dg.MaterializeResult(
                data_version=dg.DataVersion(build_id),
                metadata={"build_id": build_id, "is_reused": True},
            )
        staged_dir = build_dir(layer_dir=embeddings_dir, build_id=build_id)
        cache_dir = embeddings_dir / "cache" / code_version

        written = {table.label: 0 for table in VECTOR_TABLES}
        num_reused = 0
        for split in silver.splits:
            # A lake on an object store reaches Triton as presigned URLs, made batch by
            # batch, just before each request.
            locate_image = build_reader_locator(
                lake.store.resolve(silver.image_roots[split]),
                expires_seconds=PRESIGNED_URL_SECONDS,
            )
            # Triton reads the images by path.
            with lake.disk_lease.hold(
                target=lake.store.resolve(bronze.path),
                reason=f"Embeddings {name} {split}",
                log=context.log,
            ):
                for table in VECTOR_TABLES:
                    target = table.target_file(build_dir=staged_dir, split=split)
                    count = write_vectors(
                        store=lake.store,
                        embedder=embedder,
                        table=table,
                        files=SplitVectorFiles(
                            rows=table.rows_file(build_dir=rows_dir, split=split),
                            target=target,
                            cache_dir=cache_dir / target.parent.name / split,
                        ),
                        locate_image=locate_image,
                        log=context.log.info,
                    )
                    written[table.label] += count.written
                    num_reused += count.reused
                    context.log.info(
                        f"{split}: {count.written} {table.label}, "
                        f"{count.reused} of them reused"
                    )

        publish_build(
            manifest_path=embeddings_dir / EMBEDDINGS_MANIFEST,
            manifest=EmbeddingsManifest(
                dataset=name,
                code_version=code_version,
                build_id=build_id,
                embedding_model=EMBEDDING_MODEL,
                embedding_version=EMBEDDING_VERSION,
                silver_build_id=silver.build_id,
                splits=list(silver.splits),
            ),
        )
        return dg.MaterializeResult(
            data_version=dg.DataVersion(build_id),
            metadata={
                "build_id": build_id,
                "is_reused": False,
                **{f"num_{label.replace(' ', '_')}": n for label, n in written.items()},
                "num_reused_embeddings": num_reused,
                "embedding_model": EMBEDDING_MODEL,
            },
        )

    return _embeddings


defs = dg.Definitions(assets=[build_embeddings_asset(name) for name in SOURCE_BY_NAME])
