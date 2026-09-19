#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "cv-lakehouse",
#     "lightly-studio>=1.1.1",
#     "sqlalchemy>=2.0",
#     "torch>=2.6.0",
#     "torchvision>=0.20.0",
# ]
#
# # The lakehouse itself, from the checkout beside this script, so the schema and the
# # manifest this reads are the ones silver writes. Nothing is vendored or restated.
# [tool.uv.sources]
# cv-lakehouse = { path = "../", editable = true }
# torch = [{ index = "pytorch-cpu" }]
# torchvision = [{ index = "pytorch-cpu" }]
#
# # LightlyStudio pulls torch. Nothing here trains, so take the CPU wheels and save
# # 2.8 GB on Linux.
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Load silver Parquet into a LightlyStudio DuckDB, without iterating rows.

Run it with `uv run tools/studio.py <dataset>...`, or `make studio`. This is a
script and not part of `cv_lakehouse`: browsing is neither a layer nor a step in
producing one, and keeping it out means the package depends on no GUI and no torch.
It depends on the package, so the Parquet paths, the manifest and the class
registry it reads are the ones silver wrote, and cannot drift from them.

Silver is Parquet and only Parquet. This builds a LightlyStudio database from it on
demand, so visualising a dataset costs one bulk load rather than a second copy of the
annotations on disk.

Nothing here sends a row through Python. The O(1) scaffolding -- the dataset, the
annotation collection, one label per registry class -- goes through LightlyStudio's own
API, so those rows are exactly what the installed version expects. Every O(N) table is
filled by `INSERT ... SELECT ... FROM read_parquet(...)`, which DuckDB runs server side
and streams.

Every id LightlyStudio needs is derived here rather than stored in silver: an md5 of the
semantic key, whose 32 hex characters cast straight into the UUID columns it uses for a
primary key. Keys are unique across datasets by construction, so loading several silver
datasets into one database is a concatenation with no collision handling, and a reload
is idempotent.

The same bulk load serves a training mix later, which is why silver keeps no
database file of its own.

Silver's MobileCLIP embeddings load the same way, onto the sample of the image and the
sample of the box they belong to. LightlyStudio therefore embeds nothing itself: the
GUI's plot and its similarity search read what silver already wrote.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlmodel import Session

from cv_lakehouse.class_registry import CanonicalClass
from cv_lakehouse.embeddings import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from cv_lakehouse.manifests import SILVER_MANIFEST, SilverManifest, read_manifest
from cv_lakehouse.settings import LakePaths, Settings
from cv_lakehouse.silver_schema import (
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
    images_file,
)

logger = logging.getLogger(__name__)

# `annotation_base.annotation_type`. DuckDB makes the column
# `ENUM('CLASSIFICATION', 'SEGMENTATION_MASK', 'OBJECT_DETECTION')`, because SQLAlchemy
# stores a Python enum by member name, not by value. DuckDB casts a string literal into
# an enum but not a bind parameter, so this belongs in the statement.
OBJECT_DETECTION = "'OBJECT_DETECTION'"

# The semantic key of an image, and of one box on it. An md5 is 32 hex characters, which
# DuckDB casts straight into the UUID column LightlyStudio uses for a primary key. Both
# Parquet files carry these three columns, so the image key is the same expression on
# either side of the join.
#
# A `::` cast is safe here but never on a bind parameter: SQLAlchemy's bind regex
# mis-parses `:name::TYPE`, so a parameter is cast with `cast(:name as TYPE)` instead.
IMAGE_ID = "md5(dataset || '/' || split || '/' || file_name)::UUID"
BOX_ID = "md5(dataset || '/' || split || '/' || file_name || '#' || box_index)::UUID"


def build_studio_database(
    *,
    dataset_names: Sequence[str],
    db_path: Path,
    paths: LakePaths,
    database_name: str | None = None,
) -> Path:
    """Build a LightlyStudio database holding every named silver dataset.

    Flagged boxes are loaded too. A training mix would exclude them, but being able to
    look at them is the whole point of keeping them.
    """
    from lightly_studio.core.image.image_dataset import ImageDataset
    from lightly_studio.database import db_manager

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # The engine is a process wide singleton and `set_engine` refuses to replace one, so
    # drop any open engine first. That is also what lets this be called twice.
    db_manager.close()
    db_manager.connect(db_file=str(db_path), cleanup_existing=True)

    image_dataset = ImageDataset.create(database_name or "_".join(dataset_names))
    session = image_dataset.session
    _create_annotation_labels(session=session, dataset_id=image_dataset.dataset_id)
    model_id = _register_embedding_model(
        session=session,
        dataset_id=image_dataset.dataset_id,
        collection_id=image_dataset.collection_id,
    )

    for dataset_name in dataset_names:
        silver_dir = paths.silver_dir(dataset_name)
        manifest = read_manifest(
            path=silver_dir / SILVER_MANIFEST, model=SilverManifest
        )
        collection_id = _get_or_create_annotation_collection(
            session=session,
            root_collection_id=image_dataset.collection_id,
            name=dataset_name,
        )
        _link_embedding_model(
            session=session, collection_id=collection_id, embedding_model_id=model_id
        )
        for split in manifest.splits:
            _load_split(
                session=session,
                images_collection_id=image_dataset.collection_id,
                annotation_collection_id=collection_id,
                embedding_model_id=model_id,
                embedded=manifest.embedding_model is not None,
                images_path=images_file(silver_dir=silver_dir, split=split),
                boxes_path=boxes_file(silver_dir=silver_dir, split=split),
                embeddings_path=embeddings_file(silver_dir=silver_dir, split=split),
                crop_embeddings_path=crop_embeddings_file(
                    silver_dir=silver_dir, split=split
                ),
                image_root=manifest.image_roots[split],
            )
            logger.info(f"Loaded {dataset_name}/{split}")
    session.commit()
    # DuckDB holds an exclusive write lock, so hand the file back before returning.
    # Anything that wants to read it, the GUI included, opens it itself.
    db_manager.close()
    return db_path


def _create_annotation_labels(session: Session, dataset_id: UUID) -> None:
    """Create one annotation label per registry class, through LightlyStudio.

    A handful of rows, so the API is the right tool. A temporary table then maps a class
    name onto its label id inside the bulk inserts.
    """
    from lightly_studio.models.annotation_label import AnnotationLabelCreate
    from lightly_studio.resolvers import annotation_label_resolver

    session.execute(
        text(
            "create or replace temporary table _label_ids "
            "(class_name varchar, label_id varchar)"
        )
    )
    for canonical_class in CanonicalClass:
        label = annotation_label_resolver.create(
            session=session,
            label=AnnotationLabelCreate(
                dataset_id=dataset_id, annotation_label_name=canonical_class.class_name
            ),
        )
        session.execute(
            text("insert into _label_ids values (:name, :label_id)").bindparams(
                name=canonical_class.class_name,
                label_id=str(label.annotation_label_id),
            )
        )
    session.flush()


def _register_embedding_model(
    session: Session, dataset_id: UUID, collection_id: UUID
) -> UUID:
    """Register the model silver embedded with, and make it the collection's default.

    The link table is what the 2D projection and the similarity search read, so a model
    with no link is invisible to the GUI.
    """
    from lightly_studio.models.embedding_model import EmbeddingModelCreate
    from lightly_studio.resolvers import embedding_model_resolver

    model = embedding_model_resolver.get_or_create(
        session=session,
        embedding_model=EmbeddingModelCreate(
            name=EMBEDDING_MODEL,
            embedding_dimension=EMBEDDING_DIMENSION,
            dataset_id=dataset_id,
        ),
    )
    _link_embedding_model(
        session=session,
        collection_id=collection_id,
        embedding_model_id=model.embedding_model_id,
    )
    return model.embedding_model_id


def _link_embedding_model(
    session: Session, collection_id: UUID, embedding_model_id: UUID
) -> None:
    from lightly_studio.resolvers import collection_embedding_model_resolver

    collection_embedding_model_resolver.get_or_add_collection_model(
        session=session,
        collection_id=collection_id,
        embedding_model_id=embedding_model_id,
    )
    collection_embedding_model_resolver.set_default(
        session=session,
        collection_id=collection_id,
        embedding_model_id=embedding_model_id,
    )


def _get_or_create_annotation_collection(
    session: Session, root_collection_id: UUID, name: str
) -> UUID:
    from lightly_studio.models.collection import SampleType
    from lightly_studio.resolvers import collection_resolver

    return collection_resolver.get_or_create_child_collection(
        session=session,
        collection_id=root_collection_id,
        sample_type=SampleType.ANNOTATION,
        name=name,
    )


def _load_split(
    *,
    session: Session,
    images_collection_id: UUID,
    annotation_collection_id: UUID,
    embedding_model_id: UUID,
    embedded: bool,
    images_path: Path,
    boxes_path: Path,
    embeddings_path: Path,
    crop_embeddings_path: Path,
    image_root: str,
) -> None:
    """Fill every per row table for one split with server side statements."""
    root = image_root.rstrip("/")
    binds = {
        "images": str(images_path),
        "boxes": str(boxes_path),
        "image_embeddings": str(embeddings_path),
        "crop_embeddings": str(crop_embeddings_path),
        "collection": _bind_uuid(images_collection_id),
        "annotations": _bind_uuid(annotation_collection_id),
        "model": _bind_uuid(embedding_model_id),
        "root": root,
    }
    statements = list(_LOAD_STATEMENTS)
    # A silver built with no Triton server has no embedding file. The manifest says so,
    # and it is the layer's contract.
    if embedded:
        statements.extend(_EMBEDDING_STATEMENTS)
    for statement in statements:
        used = {name: binds[name] for name in _BIND_NAME_PATTERN.findall(statement)}
        session.execute(statement=text(statement), params=used)


# A bind parameter in a text clause, but not a `::type` cast.
_BIND_NAME_PATTERN = re.compile(r"(?<!:):(\w+)")


def _bind_uuid(value: UUID) -> str:
    """Bind a UUID as its canonical text, which DuckDB casts back to a UUID."""
    return str(value)


# `or ignore` makes a reload idempotent: an id is a function of the data, so a row
# already present collides and is skipped rather than duplicated.
_LOAD_STATEMENTS = (
    # One sample per image, in the root collection.
    f"""
    insert or ignore into sample (collection_id, sample_id, created_at, updated_at)
    select cast(:collection as UUID), {IMAGE_ID}, now()::timestamp, now()::timestamp
    from read_parquet(:images)
    """,
    # The image itself. file_path_abs is built here, not stored: an absolute path would
    # make silver unreadable on another machine.
    f"""
    insert or ignore into image
        (file_name, width, height, file_path_abs, sample_id, created_at, updated_at)
    select file_name, width, height, :root || '/' || file_name, {IMAGE_ID},
           now()::timestamp, now()::timestamp
    from read_parquet(:images)
    """,
    # One sample per box, in the annotation collection.
    f"""
    insert or ignore into sample (collection_id, sample_id, created_at, updated_at)
    select cast(:annotations as UUID), {BOX_ID}, now()::timestamp, now()::timestamp
    from read_parquet(:boxes)
    """,
    # The annotation, joined onto its label by class name.
    f"""
    insert or ignore into annotation_base
        (created_at, sample_id, annotation_type, annotation_label_id, confidence,
         parent_sample_id, object_track_id)
    select now()::timestamp, {BOX_ID}, {OBJECT_DETECTION}, cast(l.label_id as UUID),
           b.confidence, {IMAGE_ID}, null
    from read_parquet(:boxes) b
    join _label_ids l on l.class_name = b.class_name
    """,
    # The box. LightlyStudio stores pixels as integers, so the rounding happens here and
    # silver keeps the exact float.
    f"""
    insert or ignore into object_detection_annotation (sample_id, x, y, width, height)
    select {BOX_ID}, round(x)::int, round(y)::int, round(w)::int, round(h)::int
    from read_parquet(:boxes)
    """,
    # Which images this annotation collection covers, so the GUI can filter by source.
    f"""
    insert or ignore into annotation_collection_coverage
        (annotation_collection_id, parent_sample_id)
    select distinct cast(:annotations as UUID), {IMAGE_ID}
    from read_parquet(:boxes)
    """,
)


# The vectors, onto the sample of the image and the sample of the box. DuckDB stores
# `sample_embedding.embedding` as `FLOAT[]`, which is what the Parquet list column reads
# back as, so neither statement casts.
_EMBEDDING_STATEMENTS = (
    f"""
    insert or ignore into sample_embedding (sample_id, embedding_model_id, embedding)
    select {IMAGE_ID}, cast(:model as UUID), embedding
    from read_parquet(:image_embeddings)
    """,
    f"""
    insert or ignore into sample_embedding (sample_id, embedding_model_id, embedding)
    select {BOX_ID}, cast(:model as UUID), embedding
    from read_parquet(:crop_embeddings)
    """,
)


def open_gui(dataset_names: Iterable[str]) -> None:
    """Build the database for these datasets and start the LightlyStudio GUI."""
    from lightly_studio import start_gui
    from lightly_studio.database import db_manager

    names = list(dataset_names)
    paths = LakePaths(Settings().root)
    database_name = "_".join(names)
    db_path = paths.studio_database(database_name)
    logger.info(f"Building {db_path}")
    build_studio_database(
        dataset_names=names, db_path=db_path, paths=paths, database_name=database_name
    )
    db_manager.connect(db_file=str(db_path), must_exist=True)
    start_gui()


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", help="silver dataset names")
    open_gui(parser.parse_args().datasets)


if __name__ == "__main__":
    main()
