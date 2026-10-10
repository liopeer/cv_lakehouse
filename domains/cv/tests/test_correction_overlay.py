#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Apply correction snapshots in silver, on the fixture lake."""

from datetime import UTC, datetime
from pathlib import Path

import dagster as dg
import pyarrow.parquet as pq
import pytest

from lakehouse_core.manifest_files import read_manifest
from lakehouse_cv.contract.box_identity import derive_box_id
from lakehouse_cv.contract.correction_actions import CorrectionAction
from lakehouse_cv.contract.gold_tables import gold_boxes_file
from lakehouse_cv.contract.manifests import SILVER_MANIFEST, SilverManifest
from lakehouse_cv.contract.silver_tables import (
    BOX_SCHEMA,
    BoxOrigin,
    SilverBox,
    SilverImage,
    boxes_file,
)
from lakehouse_cv.defs import corrections as corrections_defs
from lakehouse_cv.defs import silver as silver_defs
from lakehouse_cv.defs.corrections import build_corrections_asset
from lakehouse_cv.defs.resources import CvLakeResource
from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME
from lakehouse_cv.transforms.correction_overlay import Correction, CorrectionOverlay
from lakehouse_cv.transforms.gold_build import build_gold
from tests.export_fakes import EXPORT_URL, FakeExportServer
from tests.lake_runs import (
    find_silver_files,
    materialize_assets,
    materialize_bronze_links,
)

DATASET = "wider_face"
PARADE = "0--Parade/a.jpg"
HANDSHAKING = "1--Handshaking/b.jpg"
# The face of a.jpg, at 10 20 30 40 in the fixture. Its second box is the region the
# annotators rejected.
FACE = derive_box_id(
    dataset=DATASET, split="train", file_name=PARADE, source_box_index=0
)
DRAWN = "aaaaaaaa-0000-0000-0000-000000000001"


def _correction(box_id: str, action: str, file_name: str = PARADE, **fields) -> dict:
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": file_name,
        "box_id": box_id,
        "action": action,
        **fields,
    }


@pytest.fixture
def export_lake(
    lake: CvLakeResource,
    bronze_sources: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CvLakeResource, FakeExportServer]:
    """A lake with bronze linked, and a fake export that a test publishes to."""
    server = FakeExportServer()
    monkeypatch.setattr(
        target=corrections_defs,
        name="open_export_client",
        value=lambda timeout: server.client(),
    )
    export_lake = CvLakeResource(
        root=lake.root,
        download_workers=lake.download_workers,
        request_timeout_seconds=lake.request_timeout_seconds,
        studio_export_url=EXPORT_URL,
    )
    materialize_bronze_links(lake=export_lake, sources=bronze_sources)
    return export_lake, server


def _build_silver(
    export_lake: tuple[CvLakeResource, FakeExportServer], rows: list[dict] | None
) -> dg.ExecuteInProcessResult:
    """Publish the rows as a snapshot, fetch it, and build silver with its checks."""
    lake, server = export_lake
    if rows is not None:
        server.publish(dataset=DATASET, rows=rows)
    return materialize_assets(
        lake=lake,
        assets=[
            build_corrections_asset(DATASET),
            silver_defs.build_silver_asset(DATASET),
            *silver_defs.build_silver_checks(DATASET),
        ],
    )


def _read_train_boxes(lake: CvLakeResource) -> dict[str, dict]:
    table = pq.read_table(
        boxes_file(build_dir=find_silver_files(lake=lake, name=DATASET), split="train")
    )
    assert table.schema == BOX_SCHEMA
    return {row["box_id"]: row for row in table.to_pylist()}


def _read_check(result: dg.ExecuteInProcessResult) -> dg.AssetCheckEvaluation:
    (evaluation,) = (
        evaluation
        for evaluation in result.get_asset_check_evaluations()
        if evaluation.check_name == "corrections_are_applied"
    )
    return evaluation


def test_silver_with_no_snapshot_is_the_bare_source(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    result = _build_silver(export_lake=export_lake, rows=None)

    boxes = _read_train_boxes(export_lake[0])
    assert len(boxes) == 2
    assert {box["origin"] for box in boxes.values()} == {BoxOrigin.SOURCE}
    assert not any(box["is_class_corrected"] for box in boxes.values())
    assert _read_check(result).passed
    manifest = read_manifest(
        path=export_lake[0].paths.silver_dir(DATASET) / SILVER_MANIFEST,
        model=SilverManifest,
    )
    assert manifest.correction_snapshot_id is None


def test_a_relabelled_box_keeps_its_exact_coordinates(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [_correction(box_id=FACE, action=CorrectionAction.UPDATE, label_name="head")]
    result = _build_silver(export_lake=export_lake, rows=rows)

    face = _read_train_boxes(export_lake[0])[FACE]
    assert (face["class_name"], face["source_class"]) == ("head", "face")
    assert (face["x"], face["y"], face["w"], face["h"]) == (10.0, 20.0, 30.0, 40.0)
    assert (face["is_class_corrected"], face["is_geometry_corrected"]) == (True, False)
    assert face["origin"] == BoxOrigin.SOURCE
    assert _read_check(result).passed


def test_a_moved_box_takes_the_pixels_of_the_curator(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [
        _correction(
            box_id=FACE, action=CorrectionAction.UPDATE, x=12.0, y=22.0, w=33.0, h=44.0
        )
    ]
    _build_silver(export_lake=export_lake, rows=rows)

    face = _read_train_boxes(export_lake[0])[FACE]
    assert (face["x"], face["y"], face["w"], face["h"]) == (12.0, 22.0, 33.0, 44.0)
    assert (face["is_class_corrected"], face["is_geometry_corrected"]) == (False, True)
    assert face["class_name"] == "face"
    # WIDER FACE grades the face, and a move does not lose the grades.
    assert face["attr_invalid"] is False


def test_a_deleted_box_leaves_silver_and_is_counted(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [_correction(box_id=FACE, action=CorrectionAction.DELETE)]
    result = _build_silver(export_lake=export_lake, rows=rows)

    boxes = _read_train_boxes(export_lake[0])
    assert FACE not in boxes
    # The box that is left is the only one of its image, so it is number 0 now.
    assert [box["box_index"] for box in boxes.values()] == [0]
    materialization = result.asset_materializations_for_node("silver__wider_face")[0]
    dropped_reasons = materialization.metadata["dropped_reasons"].value
    assert isinstance(dropped_reasons, dict)
    assert dropped_reasons["studio_deleted"] == 1
    assert materialization.metadata["num_corrections_applied"].value == 1


def test_a_drawn_box_joins_its_image_under_the_id_of_lightly_studio(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [
        _correction(
            box_id=DRAWN,
            action=CorrectionAction.ADD,
            file_name=HANDSHAKING,
            label_name="license_plate",
            x=190.0,
            y=5.0,
            w=50.0,
            h=10.0,
        )
    ]
    result = _build_silver(export_lake=export_lake, rows=rows)

    drawn = _read_train_boxes(export_lake[0])[DRAWN]
    assert drawn["file_name"] == HANDSHAKING
    assert (drawn["class_name"], drawn["origin"]) == ("license_plate", BoxOrigin.STUDIO)
    # The image is 200 wide, so the box is clipped as a source box is.
    assert (drawn["x"], drawn["w"]) == (190.0, 10.0)
    assert drawn["attr_invalid"] is None
    assert _read_check(result).passed


def test_an_unknown_label_lands_as_other_and_the_check_names_it(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [_correction(box_id=FACE, action=CorrectionAction.UPDATE, label_name="Dog")]
    result = _build_silver(export_lake=export_lake, rows=rows)

    face = _read_train_boxes(export_lake[0])[FACE]
    assert (face["class_name"], face["source_class"]) == ("other", "Dog")
    check = _read_check(result)
    assert not check.passed
    assert check.metadata["unknown_labels"].value == ["Dog"]


def test_a_correction_with_no_box_changes_nothing_and_the_check_names_it(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    orphan = "bbbbbbbb-0000-0000-0000-000000000002"
    rows = [
        _correction(box_id=orphan, action=CorrectionAction.UPDATE, label_name="head")
    ]
    result = _build_silver(export_lake=export_lake, rows=rows)

    assert len(_read_train_boxes(export_lake[0])) == 2
    check = _read_check(result)
    assert not check.passed
    assert check.metadata["orphans"].value == [f"update {orphan}"]


def test_only_the_latest_snapshot_applies(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    """A snapshot holds every live correction, so a reverted one is just absent."""
    first = [_correction(box_id=FACE, action=CorrectionAction.DELETE)]
    _build_silver(export_lake=export_lake, rows=first)
    second = [
        _correction(box_id=FACE, action=CorrectionAction.UPDATE, label_name="head")
    ]
    _build_silver(export_lake=export_lake, rows=second)

    assert _read_train_boxes(export_lake[0])[FACE]["class_name"] == "head"
    manifest = read_manifest(
        path=export_lake[0].paths.silver_dir(DATASET) / SILVER_MANIFEST,
        model=SilverManifest,
    )
    assert manifest.correction_snapshot_id is not None
    assert manifest.correction_snapshot_id.startswith("000002-")


def test_gold_carries_what_a_curator_changed(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [
        _correction(box_id=FACE, action=CorrectionAction.UPDATE, label_name="head"),
        _correction(
            box_id=DRAWN,
            action=CorrectionAction.ADD,
            file_name=HANDSHAKING,
            label_name="face",
            x=1.0,
            y=1.0,
            w=5.0,
            h=5.0,
        ),
    ]
    _build_silver(export_lake=export_lake, rows=rows)
    lake = export_lake[0]
    build = build_gold(
        store=lake.store,
        paths=lake.paths,
        specs=[SOURCE_BY_NAME[DATASET].spec],
        code_version="test",
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    gold = pq.read_table(
        gold_boxes_file(
            build_dir=lake.paths.gold_build_dir(build.manifest.build_id),
            dataset=DATASET,
            split="train",
        )
    ).to_pylist()
    by_id = {row["box_id"]: row for row in gold}
    assert (by_id[FACE]["class_name"], by_id[FACE]["is_class_corrected"]) == (
        "head",
        True,
    )
    assert by_id[DRAWN]["origin"] == BoxOrigin.STUDIO


def test_the_overlay_drops_a_box_that_a_move_leaves_no_area() -> None:
    overlay = CorrectionOverlay(
        [
            Correction(
                split="train",
                file_name="a.jpg",
                box_id="box",
                action=CorrectionAction.UPDATE,
                label_name=None,
                geometry=(500.0, 500.0, 10.0, 10.0),
            )
        ]
    )
    image = SilverImage(
        file_name="a.jpg",
        width=100,
        height=100,
        boxes=(
            SilverBox(
                box_id="box",
                box_index=0,
                class_id=0,
                class_name="face",
                source_class="face",
                x=1.0,
                y=1.0,
                w=5.0,
                h=5.0,
            ),
        ),
    )
    (corrected,) = overlay.apply_to_silver_images(split="train", images=[image])

    assert corrected.boxes == ()
    assert overlay.dropped_box_reasons == {"degenerate_box": 1}
