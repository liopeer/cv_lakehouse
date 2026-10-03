#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import json

from lakehouse_cv.contract.gold_tables import (
    GOLD_BOX_SCHEMA,
    GOLD_IMAGE_SCHEMA,
    GOLD_TABLES,
)
from lakehouse_cv.contract.manifests import CvBronzeManifest
from lakehouse_cv.contract.silver_tables import BOX_SCHEMA, IMAGE_SCHEMA, SILVER_TABLES

# A manifest as bronze wrote it before core took over the bronze asset.
BRONZE_MANIFEST_OF_RELEASE_0_2 = {
    "dataset": "pp4av",
    "mode": "link",
    "homepage": "https://huggingface.co/datasets/khaclinh/pp4av",
    "license": "CC-BY-NC-ND-4.0",
    "commercial_use": False,
    "path": "/lake/bronze/pp4av",
    "splits": ["fisheye", "normal"],
    "image_roots": {
        "fisheye": "/lake/bronze/pp4av/fisheye",
        "normal": "/lake/bronze/pp4av/normal",
    },
    "published_files": [],
}


def test_a_bronze_manifest_of_release_0_2_still_reads() -> None:
    manifest = CvBronzeManifest.model_validate(BRONZE_MANIFEST_OF_RELEASE_0_2)
    assert manifest.splits == ["fisheye", "normal"]
    assert json.loads(manifest.model_dump_json()) == BRONZE_MANIFEST_OF_RELEASE_0_2


def test_every_layer_names_its_tables_once() -> None:
    for tables in (SILVER_TABLES, GOLD_TABLES):
        names = [table.name for table in tables]
        assert len(set(names)) == len(names)


def test_the_specs_hold_the_schemas_that_the_layers_write() -> None:
    silver = {table.name: table.schema for table in SILVER_TABLES}
    gold = {table.name: table.schema for table in GOLD_TABLES}
    assert silver["images"] is IMAGE_SCHEMA
    assert silver["boxes"] is BOX_SCHEMA
    assert gold["images"] is GOLD_IMAGE_SCHEMA
    assert gold["boxes"] is GOLD_BOX_SCHEMA
