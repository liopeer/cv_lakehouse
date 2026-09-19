#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Load silver Parquet into a LightlyStudio database."""

from pathlib import Path

import dagster as dg
import pytest

from cv_lakehouse.defs import silver as silver_defs
from cv_lakehouse.defs.bronze import build_bronze_asset
from cv_lakehouse.defs.resources import LakeResource
from tests.fakes import FakeEmbedder
from tests.fixtures import make_all

DATASETS = ("wider_face", "pp4av")


@pytest.fixture
def silver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LakeResource:
    """A materialized silver lake holding two datasets, embeddings included."""
    monkeypatch.setattr(
        target=silver_defs, name="TritonEmbedder", value=lambda url: FakeEmbedder()
    )
    lake = LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
        triton_url="fake:0",
    )
    sources = make_all(tmp_path / "external")
    for name in DATASETS:
        asset = build_bronze_asset(name)
        assert dg.materialize(
            assets=[asset],
            resources={"lake": lake},
            run_config=dg.RunConfig(
                ops={asset.op.name: {"config": {"source_dir": str(sources[name])}}}
            ),
        ).success
    assert dg.materialize(
        assets=[silver_defs.build_silver_asset(name) for name in DATASETS],
        resources={"lake": lake},
    ).success
    return lake


def _counts(db_path: Path) -> dict[str, int]:
    import duckdb

    tables = (
        "sample",
        "image",
        "annotation_base",
        "object_detection_annotation",
        "annotation_collection_coverage",
        "sample_embedding",
        "embedding_model",
        "collection_embedding_model",
    )
    with duckdb.connect(database=str(db_path), read_only=True) as connection:
        counts = {}
        for table in tables:
            row = connection.execute(f"select count(*) from {table}").fetchone()
            assert row is not None
            counts[table] = row[0]
    return counts


def test_loads_every_image_and_box(silver: LakeResource, tmp_path: Path) -> None:
    from studio import build_studio_database

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )

    counts = _counts(db_path)
    # wider_face train has 3 images and val 1; pp4av test and fisheye have 1 each.
    assert counts["image"] == 6
    # Two of the wider_face train boxes, one in val, two in pp4av test, one in fisheye.
    # The flagged box is loaded too: looking at it is the point of keeping it.
    assert counts["object_detection_annotation"] == 6
    assert counts["annotation_base"] == 6
    # One sample per image plus one per box.
    assert counts["sample"] == 12
    assert counts["annotation_collection_coverage"] > 0
    # One embedding per sample, so the GUI embeds nothing itself.
    assert counts["sample_embedding"] == 12
    assert counts["embedding_model"] == 1
    # The root image collection, plus one annotation collection per dataset.
    assert counts["collection_embedding_model"] == 3


def test_a_box_keeps_its_class_and_rounds_to_integer_pixels(
    silver: LakeResource, tmp_path: Path
) -> None:
    import duckdb

    from studio import build_studio_database

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )

    with duckdb.connect(database=str(db_path), read_only=True) as connection:
        rows = connection.execute(
            """
            select l.annotation_label_name, o.x, o.y, o.width, o.height, i.file_name
            from object_detection_annotation o
            join annotation_base a using (sample_id)
            join annotation_label l using (annotation_label_id)
            join image i on i.sample_id = a.parent_sample_id
            order by i.file_name, o.x
            """
        ).fetchall()

    on_a = [row[:5] for row in rows if row[5] == "0--Parade/a.jpg"]
    # Both boxes, the flagged one included, at the pixels the fixture gives them.
    assert on_a == [
        ("face", 10, 20, 30, 40),
        ("face", 50, 60, 5, 5),
    ]
    assert all(isinstance(value, int) for value in on_a[0][1:])


def test_the_image_path_resolves_on_disk(silver: LakeResource, tmp_path: Path) -> None:
    """file_path_abs is derived at load time, so the GUI finds the bronze pixels."""
    import duckdb

    from studio import build_studio_database

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )

    with duckdb.connect(database=str(db_path), read_only=True) as connection:
        paths = [
            row[0]
            for row in connection.execute("select file_path_abs from image").fetchall()
        ]
    assert paths
    assert all(Path(path).exists() for path in paths)


def test_an_embedding_lands_on_its_own_sample(
    silver: LakeResource, tmp_path: Path
) -> None:
    """A crop embedding attaches to the box, and an image embedding to the image."""
    import duckdb

    from studio import build_studio_database

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )

    with duckdb.connect(database=str(db_path), read_only=True) as connection:
        images = connection.execute(
            """
            select count(*), min(len(e.embedding))
            from sample_embedding e join image i using (sample_id)
            """
        ).fetchone()
        boxes = connection.execute(
            """
            select count(*)
            from sample_embedding e join object_detection_annotation o using (sample_id)
            """
        ).fetchone()
    assert images == (6, 512)
    assert boxes == (6,)


def test_silver_without_embeddings_still_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GUI opens a lake built with no Triton server, minus the plot."""
    import duckdb

    from studio import build_studio_database

    lake = LakeResource(
        root=str(tmp_path / "lake"),
        download_workers=2,
        request_timeout_seconds=5.0,
    )
    sources = make_all(tmp_path / "external")
    asset = build_bronze_asset("wider_face")
    assert dg.materialize(
        assets=[asset],
        resources={"lake": lake},
        run_config=dg.RunConfig(
            ops={asset.op.name: {"config": {"source_dir": str(sources["wider_face"])}}}
        ),
    ).success
    assert dg.materialize(
        assets=[silver_defs.build_silver_asset("wider_face")], resources={"lake": lake}
    ).success

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=["wider_face"],
        db_path=db_path,
        paths=lake.paths,
        database_name="silver",
    )

    with duckdb.connect(database=str(db_path), read_only=True) as connection:
        row = connection.execute("select count(*) from sample_embedding").fetchone()
    assert row == (0,)


def test_reloading_is_idempotent(silver: LakeResource, tmp_path: Path) -> None:
    """An id is a function of the data, so a second load collides, not doubles."""
    from studio import build_studio_database

    db_path = tmp_path / "studio.db"
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )
    first = _counts(db_path)
    build_studio_database(
        dataset_names=DATASETS,
        db_path=db_path,
        paths=silver.paths,
        database_name="silver",
    )
    assert _counts(db_path) == first
