#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Miniature copies of every bronze dataset, so a test never downloads anything.

Each copy has the layout of a real download, archives left out as a link allows. Every
required path that holds no test data is an empty stand-in, so each copy is complete.
"""

from pathlib import Path

from PIL import Image as PILImage

from cv_lakehouse.sources.source_registry import SOURCE_BY_NAME

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


def make_pp4av(root: Path) -> Path:
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
    return complete_with_stand_ins(root=root, name="pp4av")


def make_wider_face(root: Path) -> Path:
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
    return complete_with_stand_ins(root=root, name="wider_face")


def make_open_images(root: Path) -> Path:
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
    return complete_with_stand_ins(root=root, name="open_images")


BUILDERS = {
    "open_images": make_open_images,
    "wider_face": make_wider_face,
    "pp4av": make_pp4av,
}


def make_all(root: Path) -> dict[str, Path]:
    """Build every dataset under its own directory and return the paths."""
    return {name: build(root / name) for name, build in BUILDERS.items()}
