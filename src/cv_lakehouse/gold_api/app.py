#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The read only HTTP API over gold.

Start it with `make api`, or in the app image with
`uvicorn --factory cv_lakehouse.gold_api.app:create_app_from_env`. Nothing here
authenticates, so keep it on a private network.

Every request reads `_gold.json` again, so a gold build shows up with no restart.
"""

from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Query, Response

from cv_lakehouse.gold_api.arrow_responses import build_rows_response
from cv_lakehouse.gold_api.gold_queries import GoldTable, read_gold_page
from cv_lakehouse.gold_api.row_filters import BoxFilter, ImageFilter
from cv_lakehouse.gold_build import read_gold_manifest
from cv_lakehouse.manifests import GoldManifest
from cv_lakehouse.settings import LakePaths, Settings


def create_app_from_env() -> FastAPI:
    return create_app(LakePaths(Settings().root))


def create_app(paths: LakePaths) -> FastAPI:
    app = FastAPI(title="cv_lakehouse gold")

    def read_current_manifest() -> GoldManifest:
        manifest = read_gold_manifest(paths)
        if manifest is None:
            raise HTTPException(status_code=503, detail="Gold is not materialized.")
        return manifest

    def respond_with_rows(
        *, table: GoldTable, row_filter: ImageFilter, accept: str | None
    ) -> Response:
        page = read_gold_page(
            paths=paths,
            manifest=read_current_manifest(),
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

    @app.get("/v1/images")
    def list_images(
        row_filter: Annotated[ImageFilter, Query()],
        accept: Annotated[str | None, Header()] = None,
    ) -> Response:
        return respond_with_rows(
            table=GoldTable.IMAGES, row_filter=row_filter, accept=accept
        )

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
