#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import hashlib
from pathlib import Path

import dagster as dg
import pytest

from lakehouse_core.bronze_asset import build_bronze_asset
from lakehouse_core.bronze_manifest import BRONZE_MANIFEST, BronzeManifest, BronzeMode
from lakehouse_core.lake_resource import LakeResource
from lakehouse_core.manifest_files import read_manifest
from lakehouse_core.published_files import Checksum, ChecksumKind, PublishedFile
from lakehouse_core.published_source import Publication, PublishedSource

CONTENT = b"one row per line\n"


class _RowsManifest(BronzeManifest):
    row_files: list[str]

    def materialization_metadata(self) -> dict[str, str | int | float | list[str]]:
        return {"row_files": list(self.row_files)}


class _RowsSource(PublishedSource[_RowsManifest]):
    publication = Publication(
        name="rows",
        homepage="https://example.com/rows",
        license="CC0-1.0",
        commercial_use=True,
        description="",
        published_files=(
            PublishedFile(
                path="rows.txt",
                url="https://example.com/rows.txt",
                size=len(CONTENT),
                checksum=Checksum(
                    kind=ChecksumKind.SHA256, value=hashlib.sha256(CONTENT).hexdigest()
                ),
            ),
        ),
    )

    def describe_bronze_copy(
        self, *, bronze_dir: Path, base: BronzeManifest
    ) -> _RowsManifest:
        row_files = sorted(path.name for path in bronze_dir.glob("*.txt"))
        if not row_files:
            raise RuntimeError(f"No rows in {bronze_dir}")
        return _RowsManifest(**base.model_dump(), row_files=row_files)


def _lake(tmp_path: Path) -> LakeResource:
    return LakeResource(
        root=str(tmp_path / "lake"), download_workers=1, request_timeout_seconds=5.0
    )


def _materialize(
    lake: LakeResource, source_dir: Path | None = None
) -> dg.ExecuteInProcessResult:
    asset = build_bronze_asset(_RowsSource())
    config = {"source_dir": None if source_dir is None else str(source_dir)}
    return dg.materialize(
        assets=[asset],
        resources={"lake": lake},
        run_config=dg.RunConfig(ops={asset.op.name: {"config": config}}),
    )


def test_the_asset_is_named_after_the_publication() -> None:
    asset = build_bronze_asset(_RowsSource())
    spec = asset.get_asset_spec()
    assert spec.key == dg.AssetKey(["bronze", "rows"])
    assert spec.group_name == "bronze"
    assert spec.description == "Raw rows."


def test_a_link_writes_the_manifest_of_the_domain(tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()
    (copy / "rows.txt").write_bytes(CONTENT)
    lake = _lake(tmp_path)

    result = _materialize(lake=lake, source_dir=copy)

    bronze_dir = lake.paths.bronze_dir("rows")
    assert bronze_dir.resolve() == copy
    manifest = read_manifest(path=bronze_dir / BRONZE_MANIFEST, model=_RowsManifest)
    assert manifest.mode == BronzeMode.LINK
    assert manifest.row_files == ["rows.txt"]
    assert [file.path for file in manifest.published_files] == ["rows.txt"]
    metadata = result.asset_materializations_for_node("bronze__rows")[0].metadata
    assert metadata["row_files"].value == ["rows.txt"]


def test_a_download_skips_a_file_that_is_complete(tmp_path: Path) -> None:
    lake = _lake(tmp_path)
    bronze_dir = lake.paths.bronze_dir("rows")
    bronze_dir.mkdir(parents=True)
    (bronze_dir / "rows.txt").write_bytes(CONTENT)

    _materialize(lake=lake)

    manifest = read_manifest(path=bronze_dir / BRONZE_MANIFEST, model=_RowsManifest)
    assert manifest.mode == BronzeMode.DOWNLOAD


def test_a_link_to_an_incomplete_copy_fails(tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    copy.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing: rows.txt"):
        _materialize(lake=_lake(tmp_path), source_dir=copy)
