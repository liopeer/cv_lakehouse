#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Miniature copies of every bronze dataset, so a test never downloads anything.

Each copy has the layout of a real download, archives left out as a link allows. Every
required path that holds no test data is an empty stand-in, so each copy is complete.
"""

import json
from pathlib import Path

from PIL import Image as PILImage
from upath import UPath

from lakehouse_cv.sources.source_registry import SOURCE_BY_NAME

WIDER_GT_TRAIN = """0--Parade/a.jpg
2
10 20 30 40 0 0 0 0 0 0
50 60 5 5 0 0 0 1 0 0
1--Handshaking/b.jpg
0
0 0 0 0 0 0 0 0 0 0
2--Demonstration/c.jpg
1
0 0 0 0 0 0 0 0 0 0
"""

WIDER_GT_VAL = """3--Riot/d.jpg
1
5 5 20 20 0 0 0 0 0 0
"""

# The header that Open Images publishes for validation and test. The train file adds
# the XClick columns, which no reader uses.
OPEN_IMAGES_HEADER = (
    "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,"
    "IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside"
)


def _image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new(mode="RGB", size=size).save(path)


def _open_images_row(
    *, image_id: str, mid: str, box: str, group_of: str = "0", depiction: str = "0"
) -> str:
    """One CSV row. `box` is `XMin,XMax,YMin,YMax`."""
    return f"{image_id},xclick,{mid},1,{box},0,0,{group_of},{depiction},0"


def complete_with_stand_ins(root: Path, name: str) -> Path:
    """Create every required path of a source that the test data left out."""
    for published_file in SOURCE_BY_NAME[name].published_files:
        path = root / (published_file.unpacked_path or published_file.path)
        if path.exists():
            continue
        if published_file.unpacked_path is not None:
            path.mkdir(parents=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


def make_pp4av(root: Path) -> UPath:
    """PP4AV, as its Hugging Face repository unpacks."""
    for city in ("zurich", "fisheye"):
        _image(path=root / "data" / "images" / city / "a.png", size=(200, 100))
    labels = root / "data" / "annotations"
    (labels / "zurich").mkdir(parents=True)
    (labels / "fisheye").mkdir(parents=True)
    (labels / "zurich" / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n1 0.25 0.5 0.5 0.5\n")
    (labels / "fisheye" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    # The unfiltered labels hold one more face. No reader takes them.
    soiling = root / "data" / "soiling_annotations" / "zurich"
    soiling.mkdir(parents=True)
    (soiling / "a.txt").write_text(
        "0 0.5 0.5 0.1 0.2\n0 0.1 0.1 0.1 0.1\n1 0.25 0.5 0.5 0.5\n"
    )
    return UPath(complete_with_stand_ins(root=root, name="pp4av"))


def make_wider_face(root: Path) -> UPath:
    """WIDER FACE, with the zero count block and an annotator rejected box."""
    for event, name in (
        ("0--Parade", "a.jpg"),
        ("1--Handshaking", "b.jpg"),
        ("2--Demonstration", "c.jpg"),
    ):
        _image(path=root / "WIDER_train" / "images" / event / name, size=(200, 150))
    _image(path=root / "WIDER_val" / "images" / "3--Riot" / "d.jpg", size=(100, 100))

    split = root / "wider_face_split"
    split.mkdir(parents=True)
    (split / "wider_face_train_bbx_gt.txt").write_text(WIDER_GT_TRAIN)
    (split / "wider_face_val_bbx_gt.txt").write_text(WIDER_GT_VAL)
    return UPath(complete_with_stand_ins(root=root, name="wider_face"))


def make_coco(root: Path) -> UPath:
    """COCO, with a crowd box, a cat and an image without a box."""
    _image(path=root / "train2017" / "000000000001.jpg", size=(100, 80))
    _image(path=root / "train2017" / "000000000002.jpg", size=(60, 60))
    _image(path=root / "val2017" / "000000000003.jpg", size=(50, 50))
    annotations = root / "annotations_trainval2017" / "annotations"
    annotations.mkdir(parents=True)
    categories = [{"id": 1, "name": "person"}, {"id": 17, "name": "cat"}]
    train = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "000000000002.jpg", "width": 60, "height": 60},
        ],
        "annotations": [
            _coco_box(box_id=10, image_id=1, category_id=1, bbox=[10, 20, 30, 40]),
            _coco_box(box_id=11, image_id=1, category_id=17, bbox=[0, 0, 5, 5]),
            _coco_box(
                box_id=12, image_id=1, category_id=1, bbox=[0, 0, 100, 80], iscrowd=1
            ),
        ],
        "categories": categories,
    }
    val = {
        "images": [
            {"id": 3, "file_name": "000000000003.jpg", "width": 50, "height": 50}
        ],
        "annotations": [
            _coco_box(box_id=13, image_id=3, category_id=1, bbox=[5, 5, 20, 20])
        ],
        "categories": categories,
    }
    (annotations / "instances_train2017.json").write_text(json.dumps(train))
    (annotations / "instances_val2017.json").write_text(json.dumps(val))
    return UPath(complete_with_stand_ins(root=root, name="coco"))


def make_lvis(root: Path) -> UPath:
    """LVIS on the images of the COCO fixture. Its val split takes a train2017 image."""
    categories = [{"id": 642, "name": "license_plate"}, {"id": 793, "name": "person"}]
    train = {
        "images": [_lvis_image(image_id=1, folder="train2017", size=(100, 80))],
        "annotations": [
            _lvis_box(box_id=1, image_id=1, category_id=642, bbox=[50, 50, 20, 10]),
            _lvis_box(box_id=2, image_id=1, category_id=793, bbox=[10, 20, 30, 40]),
        ],
        "categories": categories,
    }
    val = {
        "images": [_lvis_image(image_id=2, folder="train2017", size=(60, 60))],
        "annotations": [],
        "categories": categories,
    }
    for split, content in (("train", train), ("val", val)):
        folder = root / f"lvis_v1_{split}.json"
        folder.mkdir(parents=True)
        (folder / f"lvis_v1_{split}.json").write_text(json.dumps(content))
    return UPath(complete_with_stand_ins(root=root, name="lvis"))


def make_coco_wholebody(root: Path) -> UPath:
    """COCO-WholeBody on the COCO fixture: one face, one face that is not annotated."""
    root.mkdir(parents=True, exist_ok=True)
    train = {
        "images": [
            _coco_wholebody_image(image_id=1, folder="train2017", size=(100, 80))
        ],
        "annotations": [
            _coco_box(box_id=10, image_id=1, category_id=1, bbox=[10, 20, 30, 40])
            | _parts(face_box=[15, 22, 8, 8], face_valid=True),
            _coco_box(box_id=12, image_id=1, category_id=1, bbox=[60, 5, 30, 40])
            | _parts(face_box=[0, 0, 0, 0], face_valid=False),
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    val = {
        "images": [_coco_wholebody_image(image_id=3, folder="val2017", size=(50, 50))],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    (root / "coco_wholebody_train_v1.0.json").write_text(json.dumps(train))
    (root / "coco_wholebody_val_v1.0.json").write_text(json.dumps(val))
    return UPath(complete_with_stand_ins(root=root, name="coco_wholebody"))


def _coco_url(*, image_id: int, folder: str) -> str:
    return f"http://images.cocodataset.org/{folder}/{image_id:012d}.jpg"


def _lvis_image(
    *, image_id: int, folder: str, size: tuple[int, int]
) -> dict[str, object]:
    """An LVIS image names no file, only its COCO URL."""
    return {
        "id": image_id,
        "coco_url": _coco_url(image_id=image_id, folder=folder),
        "width": size[0],
        "height": size[1],
        "neg_category_ids": [],
        "not_exhaustive_category_ids": [],
    }


def _coco_wholebody_image(
    *, image_id: int, folder: str, size: tuple[int, int]
) -> dict[str, object]:
    return {
        "id": image_id,
        "file_name": f"{image_id:012d}.jpg",
        "coco_url": _coco_url(image_id=image_id, folder=folder),
        "width": size[0],
        "height": size[1],
    }


def _lvis_box(
    *, box_id: int, image_id: int, category_id: int, bbox: list[int]
) -> dict[str, object]:
    return {
        "id": box_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "segmentation": [[0, 0, 1, 0, 1, 1]],
    }


def _parts(*, face_box: list[int], face_valid: bool) -> dict[str, object]:
    """The part boxes of one person. The hands are never annotated here."""
    return {
        "face_box": face_box,
        "face_valid": face_valid,
        "lefthand_box": [0, 0, 0, 0],
        "lefthand_valid": False,
        "righthand_box": [0, 0, 0, 0],
        "righthand_valid": False,
        "keypoints": [0] * 51,
    }


def _coco_box(
    *, box_id: int, image_id: int, category_id: int, bbox: list[int], iscrowd: int = 0
) -> dict[str, object]:
    return {
        "id": box_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "iscrowd": iscrowd,
        "segmentation": [[0, 0, 1, 0, 1, 1]],
    }


def make_open_images(root: Path) -> UPath:
    """Open Images. Bronze keeps every class, so the CSV holds a non PII row too."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "validation-annotations-bbox.csv").write_text(
        "\n".join(
            [
                OPEN_IMAGES_HEADER,
                _open_images_row(
                    image_id="aaa", mid="/m/0dzct", box="0.1,0.2,0.25,0.5"
                ),
                _open_images_row(
                    image_id="aaa", mid="/m/01jfm_", box="0.5,1.0,0.5,1.0"
                ),
                # A cat. Silver keeps it, the PII view drops it.
                _open_images_row(image_id="aaa", mid="/m/01yrx", box="0.0,0.3,0.0,0.3"),
                # A crowd of faces, then a drawing of one. The reader drops both.
                _open_images_row(
                    image_id="aaa", mid="/m/0dzct", box="0.0,1.0,0.0,1.0", group_of="1"
                ),
                _open_images_row(
                    image_id="aaa", mid="/m/0dzct", box="0.0,1.0,0.0,1.0", depiction="1"
                ),
                # No image. The reader skips it.
                _open_images_row(image_id="bbb", mid="/m/0dzct", box="0.0,1.0,0.0,1.0"),
            ]
        )
        + "\n"
    )
    (root / "test-annotations-bbox.csv").write_text(
        "\n".join(
            [
                OPEN_IMAGES_HEADER,
                _open_images_row(
                    image_id="ccc", mid="/m/01jfm_", box="0.2,0.4,0.2,0.4"
                ),
            ]
        )
        + "\n"
    )
    (root / "oidv6-train-annotations-bbox.csv").write_text(
        "\n".join(
            [
                OPEN_IMAGES_HEADER,
                _open_images_row(image_id="d1e", mid="/m/0dzct", box="0.1,0.5,0.1,0.5"),
            ]
        )
        + "\n"
    )
    _image(path=root / "validation" / "aaa.jpg", size=(100, 200))
    _image(path=root / "test" / "ccc.jpg", size=(50, 50))
    _image(path=root / "train_d" / "d1e.jpg", size=(40, 40))
    return UPath(complete_with_stand_ins(root=root, name="open_images"))


BUILDERS = {
    "open_images": make_open_images,
    "wider_face": make_wider_face,
    "pp4av": make_pp4av,
    "coco": make_coco,
    "lvis": make_lvis,
    "coco_wholebody": make_coco_wholebody,
}


def make_all(root: Path) -> dict[str, UPath]:
    """Build every dataset under its own directory and return the paths."""
    return {name: build(root / name) for name, build in BUILDERS.items()}
