#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The read only HTTP API over gold.

Start it with `make api`, or in the app image with
`uvicorn --factory lakehouse_cv.gold_api.app:create_app_from_env`. Nothing here
authenticates, so keep it on a private network.

Every request reads `_gold.json` again, so a gold build shows up with no restart.
"""

from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Response

from lakehouse_core.lake_store import LakeStore
from lakehouse_cv.contract.class_registry import CanonicalClass
from lakehouse_cv.contract.manifests import GoldManifest, ReleaseManifest
from lakehouse_cv.gold_api.arrow_responses import build_rows_response
from lakehouse_cv.gold_api.gold_queries import (
    GoldTable,
    count_gold_images,
    read_gold_page,
)
from lakehouse_cv.gold_api.row_filters import (
    DRAFT_RELEASE,
    BoxFilter,
    ImageFilter,
    ImageSelection,
)
from lakehouse_cv.settings import CvLakePaths, CvSettings
from lakehouse_cv.transforms.gold_build import read_gold_manifest
from lakehouse_cv.transforms.gold_release import (
    list_release_numbers,
    read_release_manifest,
)

VECTOR_TABLES = (GoldTable.EMBEDDINGS, GoldTable.CROP_EMBEDDINGS)


def create_app_from_env() -> FastAPI:
    settings = CvSettings()
    return create_app(
        LakeStore(
            root=settings.root,
            storage_options=settings.storage_options,
            duckdb_threads=settings.duckdb_threads,
            duckdb_memory_limit=settings.duckdb_memory_limit,
        )
    )


def create_app(store: LakeStore) -> FastAPI:
    app = FastAPI(title="cv_lakehouse gold")
    paths = CvLakePaths(store.root)

    def read_current_manifest() -> GoldManifest:
        manifest = read_gold_manifest(paths)
        if manifest is None:
            raise HTTPException(status_code=503, detail="Gold is not materialized.")
        return manifest

    def read_release(release: int) -> ReleaseManifest:
        if release not in list_release_numbers(paths):
            raise HTTPException(status_code=404, detail=f"No release {release}.")
        return read_release_manifest(paths=paths, release=release)

    def resolve_release(row_filter: ImageSelection) -> ReleaseManifest | None:
        """Return the release that holds the val and test rows, or None for gold."""
        reads_eval_rows = not row_filter.role or any(
            role.is_eval for role in row_filter.role
        )
        if not reads_eval_rows or row_filter.release == DRAFT_RELEASE:
            return None
        if row_filter.release is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This request can return val and test rows. Name their release "
                    "with `release`: a number, or `draft`."
                ),
            )
        return read_release(int(row_filter.release))

    def respond_with_rows(
        *, table: GoldTable, row_filter: ImageFilter, accept: str | None
    ) -> Response:
        manifest = read_current_manifest()
        release = resolve_release(row_filter)
        if release is not None and table in VECTOR_TABLES:
            raise HTTPException(
                status_code=400,
                detail="A release holds no vector. Ask for `release=draft`.",
            )
        page = read_gold_page(
            store=store,
            paths=paths,
            manifest=manifest,
            release=release,
            table=table,
            row_filter=row_filter,
        )
        return build_rows_response(
            rows=page.rows, next_after=page.next_after, accept=accept
        )

    @app.get("/healthz")
    def check_health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/meta")
    def read_meta() -> GoldManifest:
        return read_current_manifest()

    @app.get("/v1/releases")
    def list_releases() -> list[ReleaseManifest]:
        return [
            read_release_manifest(paths=paths, release=release)
            for release in list_release_numbers(paths)
        ]

    @app.get("/v1/releases/{release}")
    def read_one_release(release: int) -> ReleaseManifest:
        return read_release(release)

    @app.get("/v1/classes")
    def list_classes() -> list[dict[str, int | str]]:
        return [
            {"class_id": member.value, "class_name": member.class_name}
            for member in CanonicalClass
        ]

    @app.get("/v1/images")
    def list_images(
        row_filter: Annotated[ImageFilter, Query()],
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        return respond_with_rows(
            table=GoldTable.IMAGES, row_filter=row_filter, accept=accept
        )

    @app.get("/v1/images/count")
    def count_images(
        selection: Annotated[ImageSelection, Query()],
    ) -> dict[str, int]:
        count = count_gold_images(
            store=store,
            paths=paths,
            manifest=read_current_manifest(),
            release=resolve_release(selection),
            selection=selection,
        )
        return {"count": count}

    @app.get("/v1/boxes")
    def list_boxes(
        row_filter: Annotated[BoxFilter, Query()],
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        return respond_with_rows(
            table=GoldTable.BOXES, row_filter=row_filter, accept=accept
        )

    @app.get("/v1/embeddings")
    def list_embeddings(
        row_filter: Annotated[ImageFilter, Query()],
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        return respond_with_rows(
            table=GoldTable.EMBEDDINGS, row_filter=row_filter, accept=accept
        )

    @app.get("/v1/crop_embeddings")
    def list_crop_embeddings(
        row_filter: Annotated[BoxFilter, Query()],
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        return respond_with_rows(
            table=GoldTable.CROP_EMBEDDINGS, row_filter=row_filter, accept=accept
        )

    return app
