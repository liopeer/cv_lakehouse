#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Fold the event files of the curators in silver, on the fixture lake."""

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
from lakehouse_cv.transforms.correction_overlay import (
    BoxEvent,
    Correction,
    CorrectionOverlay,
    fold_event,
)
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
# The face as LightlyStudio holds it, in whole pixels.
FACE_PIXELS = (10.0, 20.0, 30.0, 40.0)


def _event(
    box_id: str,
    *,
    file_name: str = PARADE,
    label_name: str = "face",
    geometry: tuple[float, float, float, float] = FACE_PIXELS,
) -> dict:
    """The box as a curator left it in LightlyStudio."""
    x, y, w, h = geometry
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": file_name,
        "box_id": box_id,
        "is_deleted": False,
        "label_name": label_name,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
    }


def _tombstone(box_id: str) -> dict:
    return {
        "dataset": DATASET,
        "split": "train",
        "file_name": PARADE,
        "box_id": box_id,
        "is_deleted": True,
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
    """Publish the rows as an event file, fetch it, and build silver with its checks."""
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


def test_silver_with_no_event_is_the_bare_source(
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
    assert manifest.last_event is None


def test_a_relabelled_box_keeps_its_exact_coordinates(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [_event(FACE, label_name="head")]
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
    rows = [_event(FACE, geometry=(12.0, 22.0, 33.0, 44.0))]
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
    rows = [_tombstone(FACE)]
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
        _event(
            DRAWN,
            file_name=HANDSHAKING,
            label_name="license_plate",
            geometry=(190.0, 5.0, 50.0, 10.0),
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
    rows = [_event(FACE, label_name="Dog")]
    result = _build_silver(export_lake=export_lake, rows=rows)

    face = _read_train_boxes(export_lake[0])[FACE]
    assert (face["class_name"], face["source_class"]) == ("other", "Dog")
    check = _read_check(result)
    assert not check.passed
    assert check.metadata["unknown_labels"].value == ["Dog"]


def test_an_event_with_no_image_changes_nothing_and_the_check_names_it(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    orphan = "bbbbbbbb-0000-0000-0000-000000000002"
    rows = [_event(orphan, file_name="9--Nowhere/z.jpg")]
    result = _build_silver(export_lake=export_lake, rows=rows)

    assert len(_read_train_boxes(export_lake[0])) == 2
    check = _read_check(result)
    assert not check.passed
    assert check.metadata["orphans"].value == [orphan]


def test_the_latest_event_of_a_box_wins(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    _build_silver(export_lake=export_lake, rows=[_tombstone(FACE)])
    _build_silver(export_lake=export_lake, rows=[_event(FACE, label_name="head")])

    assert _read_train_boxes(export_lake[0])[FACE]["class_name"] == "head"
    manifest = read_manifest(
        path=export_lake[0].paths.silver_dir(DATASET) / SILVER_MANIFEST,
        model=SilverManifest,
    )
    assert manifest.last_event is not None
    assert manifest.last_event.log_sequence == 2


def test_an_event_that_equals_the_source_changes_nothing(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    """A curator who undoes an edit leaves the box as the source has it."""
    _build_silver(export_lake=export_lake, rows=[_event(FACE, label_name="head")])
    result = _build_silver(export_lake=export_lake, rows=[_event(FACE)])

    face = _read_train_boxes(export_lake[0])[FACE]
    assert (face["class_name"], face["is_class_corrected"]) == ("face", False)
    materialization = result.asset_materializations_for_node("silver__wider_face")[0]
    assert materialization.metadata["num_corrections_applied"].value == 0


def test_a_new_database_keeps_the_edits_of_the_old_one(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    lake, server = export_lake
    _build_silver(export_lake=export_lake, rows=[_tombstone(FACE)])
    server.start_new_chain()
    drawn = _event(DRAWN, file_name=HANDSHAKING, geometry=(1.0, 2.0, 3.0, 4.0))

    _build_silver(export_lake=export_lake, rows=[drawn])

    boxes = _read_train_boxes(lake)
    assert FACE not in boxes
    assert DRAWN in boxes
    manifest = read_manifest(
        path=lake.paths.silver_dir(DATASET) / SILVER_MANIFEST, model=SilverManifest
    )
    assert manifest.last_event is not None
    assert manifest.last_event.chain_id == server.chain_id


def test_gold_carries_what_a_curator_changed(
    export_lake: tuple[CvLakeResource, FakeExportServer],
) -> None:
    rows = [
        _event(FACE, label_name="head"),
        _event(DRAWN, file_name=HANDSHAKING, geometry=(1.0, 1.0, 5.0, 5.0)),
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
            BoxEvent(
                split="train",
                file_name="a.jpg",
                box_id="box",
                is_deleted=False,
                label_name="face",
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


# A source box off the pixel grid. LightlyStudio holds it as 10 21 30 40.
_SOURCE = SilverBox(
    box_id="box",
    box_index=0,
    class_id=0,
    class_name="face",
    source_class="face",
    x=10.4,
    y=20.5,
    w=29.6,
    h=40.0,
)
_KEYS = {"split": "train", "file_name": "a.jpg", "box_id": "box"}


def _box_event(
    *, label_name: str = "face", geometry: tuple[float, float, float, float]
) -> BoxEvent:
    return BoxEvent(**_KEYS, is_deleted=False, label_name=label_name, geometry=geometry)


_TOMBSTONE = BoxEvent(**_KEYS, is_deleted=True, label_name=None, geometry=None)


@pytest.mark.parametrize(
    argnames=("event", "source", "expected"),
    argvalues=[
        pytest.param(
            _box_event(geometry=(10.0, 21.0, 30.0, 40.0)),
            _SOURCE,
            None,
            id="the source as LightlyStudio rounds it",
        ),
        pytest.param(
            _box_event(label_name="other", geometry=(10.0, 21.0, 30.0, 40.0)),
            _SOURCE,
            Correction(
                **_KEYS,
                action=CorrectionAction.UPDATE,
                label_name="other",
                geometry=None,
            ),
            id="a relabel keeps the exact source",
        ),
        pytest.param(
            _box_event(geometry=(11.0, 21.0, 30.0, 40.0)),
            _SOURCE,
            Correction(
                **_KEYS,
                action=CorrectionAction.UPDATE,
                label_name=None,
                geometry=(11.0, 21.0, 30.0, 40.0),
            ),
            id="a move",
        ),
        pytest.param(
            _box_event(label_name="other", geometry=(11.0, 21.0, 30.0, 40.0)),
            _SOURCE,
            Correction(
                **_KEYS,
                action=CorrectionAction.UPDATE,
                label_name="other",
                geometry=(11.0, 21.0, 30.0, 40.0),
            ),
            id="a relabel and a move",
        ),
        pytest.param(
            _TOMBSTONE,
            _SOURCE,
            Correction(
                **_KEYS, action=CorrectionAction.DELETE, label_name=None, geometry=None
            ),
            id="a tombstone of a source box",
        ),
        pytest.param(
            _box_event(geometry=(1.0, 2.0, 3.0, 4.0)),
            None,
            Correction(
                **_KEYS,
                action=CorrectionAction.ADD,
                label_name="face",
                geometry=(1.0, 2.0, 3.0, 4.0),
            ),
            id="a box that a curator drew",
        ),
        pytest.param(
            _TOMBSTONE, None, None, id="a drawn box that a curator deleted again"
        ),
    ],
)
def test_an_event_folds_over_its_source_box(
    event: BoxEvent, source: SilverBox | None, expected: Correction | None
) -> None:
    assert fold_event(event=event, source=source) == expected
