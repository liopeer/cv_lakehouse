#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The MobileCLIP client for the Triton server in `triton/`.

The server decodes, crops, resizes and normalises on the GPU, so this sends a path and
a box and never opens an image. `triton/README.md` documents the model interface.

One request carries many items. The entry model fans them out as concurrent sub-requests
that the dynamic batchers coalesce into one execution, so a request per image throws the
speed away.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import duckdb
import numpy as np
import pyarrow as pa
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray
from upath import UPath

from lakehouse_core.lake_store import LakeStore
from lakehouse_core.parquet_files import open_parquet_file, open_parquet_writer
from lakehouse_cv.contract.silver_tables import (
    CROP_EMBEDDING_SCHEMA,
    EMBEDDING_SCHEMA,
    boxes_file,
    crop_embeddings_file,
    embeddings_file,
    images_file,
)

EMBEDDING_MODEL = "mobileclip_s0"
EMBEDDING_DIMENSION = 512

# Items per request. The entry model caps its concurrent sub-requests at the same
# number, so a larger request buys no more batching.
ITEMS_PER_REQUEST = 1024

_IMAGE_PATH_INPUT = "IMAGE_PATH"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"
_EMBEDDING_OUTPUT = "EMBEDDING"


@dataclass(frozen=True)
class Crop:
    """One region of one image, in whole pixels."""

    path: str
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def rounded_to_pixels(
        cls, path: str, x: float, y: float, w: float, h: float
    ) -> Self:
        """Round a silver box onto the pixel grid the GPU crop needs.

        Silver keeps the exact float. A box under half a pixel wide still has to name
        at least one pixel, so the size rounds up to one rather than to zero.
        """
        return cls(
            path=path,
            x=round(x),
            y=round(y),
            width=max(1, round(w)),
            height=max(1, round(h)),
        )


class Embedder(Protocol):
    """What the write functions need. A test substitutes its own."""

    def embed_images(self, paths: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_crops(self, crops: Sequence[Crop]) -> NDArray[np.float32]: ...


class InferResult(Protocol):
    """The part of a Triton result that this module reads."""

    def as_numpy(self, name: str) -> NDArray[np.float32] | None: ...


class InferenceClient(Protocol):
    """The part of the Triton gRPC client that this module calls.

    tritonclient ships no types, so this names the call. A test substitutes its own.
    """

    # The real client takes `model_version` between `inputs` and `outputs`, so a
    # caller that matches this passes them by keyword.
    def infer(
        self,
        *,
        model_name: str,
        inputs: list[grpcclient.InferInput],
        outputs: list[grpcclient.InferRequestedOutput],
    ) -> InferResult: ...


class TritonEmbedder(Embedder):
    """Embed images and box crops with MobileCLIP, over gRPC."""

    def __init__(self, url: str, model_name: str = EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._client: InferenceClient = grpcclient.InferenceServerClient(url=url)

    def embed_images(self, paths: Sequence[str]) -> NDArray[np.float32]:
        """Embed whole images, in the order given.

        A path is a local path, which names the same file inside the container of the
        server, or an http URL, such as a presigned one, which the server fetches.
        """
        return _concatenate_embeddings(
            self._infer(
                inputs=[_bytes_input(name=_IMAGE_PATH_INPUT, values=chunk)],
                count=len(chunk),
            )
            for chunk in _split_into_request_chunks(paths)
        )

    def embed_crops(self, crops: Sequence[Crop]) -> NDArray[np.float32]:
        """Embed one region of one image per crop, in the order given."""
        return _concatenate_embeddings(
            self._infer(inputs=_crop_inputs(chunk), count=len(chunk))
            for chunk in _split_into_request_chunks(crops)
        )

    def _infer(
        self, inputs: list[grpcclient.InferInput], count: int
    ) -> NDArray[np.float32]:
        result = self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=[grpcclient.InferRequestedOutput(_EMBEDDING_OUTPUT)],
        )
        output = result.as_numpy(_EMBEDDING_OUTPUT)
        if output is None:
            raise ValueError(f"Triton returned no output '{_EMBEDDING_OUTPUT}'.")
        embeddings = np.asarray(a=output, dtype=np.float32)
        expected = (count, EMBEDDING_DIMENSION)
        if embeddings.shape != expected:
            raise ValueError(
                f"Triton returned embeddings of shape {embeddings.shape}, "
                f"and this request expected {expected}."
            )
        return embeddings


def _crop_inputs(crops: Sequence[Crop]) -> list[grpcclient.InferInput]:
    return [
        _bytes_input(name=_IMAGE_PATH_INPUT, values=[crop.path for crop in crops]),
        _int64_input(name=_CROP_X_INPUT, values=[crop.x for crop in crops]),
        _int64_input(name=_CROP_Y_INPUT, values=[crop.y for crop in crops]),
        _int64_input(name=_CROP_WIDTH_INPUT, values=[crop.width for crop in crops]),
        _int64_input(name=_CROP_HEIGHT_INPUT, values=[crop.height for crop in crops]),
    ]


def _bytes_input(name: str, values: Sequence[str]) -> grpcclient.InferInput:
    encoded = [value.encode("utf-8") for value in values]
    infer_input = grpcclient.InferInput(
        name=name, shape=[len(encoded)], datatype="BYTES"
    )
    infer_input.set_data_from_numpy(np.asarray(a=encoded, dtype=object))
    return infer_input


def _int64_input(name: str, values: Sequence[int]) -> grpcclient.InferInput:
    infer_input = grpcclient.InferInput(
        name=name, shape=[len(values)], datatype="INT64"
    )
    infer_input.set_data_from_numpy(np.asarray(a=values, dtype=np.int64))
    return infer_input


def _split_into_request_chunks[T](items: Sequence[T]) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), ITEMS_PER_REQUEST):
        yield items[start : start + ITEMS_PER_REQUEST]


def _concatenate_embeddings(
    chunks: Iterator[NDArray[np.float32]],
) -> NDArray[np.float32]:
    arrays = list(chunks)
    if not arrays:
        return np.empty(shape=(0, EMBEDDING_DIMENSION), dtype=np.float32)
    return np.concatenate(arrays, axis=0)


@dataclass(frozen=True)
class PreviousSplit:
    """The files of one split as the run before wrote them, set aside for reuse."""

    boxes: UPath
    embeddings: UPath
    crop_embeddings: UPath


def _previous_file(path: UPath) -> UPath:
    # Not `.parquet`, so a `*.parquet` scan over silver never reads it.
    return path.with_name(path.name + ".previous")


def stash_previous_split(silver_dir: UPath, split: str) -> PreviousSplit | None:
    """Set the boxes and the vectors of a split aside, before the run rewrites them.

    Return None when one of the three is missing. The run then embeds everything.
    """
    discard_previous_split(silver_dir=silver_dir, split=split)
    paths = (
        boxes_file(silver_dir=silver_dir, split=split),
        embeddings_file(silver_dir=silver_dir, split=split),
        crop_embeddings_file(silver_dir=silver_dir, split=split),
    )
    if not all(path.exists() for path in paths):
        return None
    for path in paths:
        path.rename(_previous_file(path))
    return PreviousSplit(*(_previous_file(path) for path in paths))


def discard_previous_split(silver_dir: UPath, split: str) -> None:
    for path in (
        boxes_file(silver_dir=silver_dir, split=split),
        embeddings_file(silver_dir=silver_dir, split=split),
        crop_embeddings_file(silver_dir=silver_dir, split=split),
    ):
        _previous_file(path).unlink(missing_ok=True)


def clear_embeddings(silver_dir: UPath, split: str) -> None:
    """Remove one split's embedding files.

    A rematerialisation with no server rewrites the images and the boxes, so a vector
    left behind describes pixels that a query no longer has a row for.
    """
    embeddings_file(silver_dir=silver_dir, split=split).unlink(missing_ok=True)
    crop_embeddings_file(silver_dir=silver_dir, split=split).unlink(missing_ok=True)


def write_image_embeddings(
    *,
    embedder: Embedder,
    silver_dir: UPath,
    dataset: str,
    split: str,
    locate_image: Callable[[str], str],
) -> int:
    """Embed every image of one split, and write the Parquet. Return the row count.

    The rows keep the order of the images file, so a reader joins the two on the key
    and never has to sort.
    """
    writer = _EmbeddingWriter(
        path=embeddings_file(silver_dir=silver_dir, split=split),
        schema=EMBEDDING_SCHEMA,
        dataset=dataset,
        split=split,
    )
    try:
        for batch in _read_parquet_batches(
            path=images_file(silver_dir=silver_dir, split=split), columns=["file_name"]
        ):
            names: list[str] = batch.column("file_name").to_pylist()
            writer.add(
                file_names=names,
                box_keys=None,
                vectors=embedder.embed_images([locate_image(name) for name in names]),
            )
    finally:
        writer.close()
    return writer.count


def write_crop_embeddings(
    *,
    embedder: Embedder,
    silver_dir: UPath,
    dataset: str,
    split: str,
    locate_image: Callable[[str], str],
) -> int:
    """Embed every box of one split as a crop, and write the Parquet.

    Flagged boxes are embedded too. Silver keeps them, and its consumer decides.
    """
    columns = ["file_name", "box_id", "box_index", "x", "y", "w", "h"]
    writer = _EmbeddingWriter(
        path=crop_embeddings_file(silver_dir=silver_dir, split=split),
        schema=CROP_EMBEDDING_SCHEMA,
        dataset=dataset,
        split=split,
    )
    try:
        for batch in _read_parquet_batches(
            path=boxes_file(silver_dir=silver_dir, split=split), columns=columns
        ):
            rows = batch.to_pydict()
            writer.add(
                file_names=rows["file_name"],
                box_keys=BoxKeys(ids=rows["box_id"], indices=rows["box_index"]),
                vectors=embedder.embed_crops(
                    _build_crops(locate_image=locate_image, rows=rows)
                ),
            )
    finally:
        writer.close()
    return writer.count


@dataclass(frozen=True)
class BoxKeys:
    """The two box columns of a crop embedding row, one entry per vector."""

    ids: list[str]
    indices: list[int]


def reuse_or_embed_images(
    *,
    store: LakeStore,
    embedder: Embedder,
    silver_dir: UPath,
    dataset: str,
    split: str,
    locate_image: Callable[[str], str],
    previous: PreviousSplit,
) -> tuple[int, int]:
    """Write the image vectors, and embed only an image that had none.

    Bronze pins the pixels, so a vector of the run before is still the vector of its
    image. Return the row count, and how many of the rows were reused.
    """
    target = embeddings_file(silver_dir=silver_dir, split=split)
    fresh = target.with_name(target.name + ".fresh")
    parameters = {
        "rows": str(images_file(silver_dir=silver_dir, split=split)),
        "previous": str(previous.embeddings),
    }
    with store.duckdb() as connection:
        writer = _EmbeddingWriter(
            path=fresh, schema=EMBEDDING_SCHEMA, dataset=dataset, split=split
        )
        try:
            for batch in connection.execute(
                query=_IMAGES_WITH_NO_VECTOR, parameters=parameters
            ).to_arrow_reader(ITEMS_PER_REQUEST):
                names: list[str] = batch.column("file_name").to_pylist()
                writer.add(
                    file_names=names,
                    box_keys=None,
                    vectors=embedder.embed_images(
                        [locate_image(name) for name in names]
                    ),
                )
        finally:
            writer.close()
        total = _write_query_rows(
            connection=connection,
            query=_IMAGE_VECTORS,
            parameters={**parameters, "fresh": str(fresh)},
            path=target,
            schema=EMBEDDING_SCHEMA,
        )
    fresh.unlink()
    return total, total - writer.count


def reuse_or_embed_crops(
    *,
    store: LakeStore,
    embedder: Embedder,
    silver_dir: UPath,
    dataset: str,
    split: str,
    locate_image: Callable[[str], str],
    previous: PreviousSplit,
) -> tuple[int, int]:
    """Write the crop vectors, and embed only a box that is new or that moved.

    A box keeps its vector when its four coordinates are equal to the run before. A
    relabelled box is such a box. Return the row count, and how many were reused.
    """
    target = crop_embeddings_file(silver_dir=silver_dir, split=split)
    fresh = target.with_name(target.name + ".fresh")
    parameters = {
        "rows": str(boxes_file(silver_dir=silver_dir, split=split)),
        "previous": str(previous.crop_embeddings),
        "previous_boxes": str(previous.boxes),
    }
    with store.duckdb() as connection:
        writer = _EmbeddingWriter(
            path=fresh, schema=CROP_EMBEDDING_SCHEMA, dataset=dataset, split=split
        )
        try:
            for batch in connection.execute(
                query=_CROPS_WITH_NO_VECTOR, parameters=parameters
            ).to_arrow_reader(ITEMS_PER_REQUEST):
                rows = batch.to_pydict()
                writer.add(
                    file_names=rows["file_name"],
                    box_keys=BoxKeys(ids=rows["box_id"], indices=rows["box_index"]),
                    vectors=embedder.embed_crops(
                        _build_crops(locate_image=locate_image, rows=rows)
                    ),
                )
        finally:
            writer.close()
        total = _write_query_rows(
            connection=connection,
            query=_CROP_VECTORS,
            parameters={**parameters, "fresh": str(fresh)},
            path=target,
            schema=CROP_EMBEDDING_SCHEMA,
        )
    fresh.unlink()
    return total, total - writer.count


# `file_row_number` keeps every query in the order of the images or the boxes file,
# which is the order the embedding files promise.
_IMAGES_WITH_NO_VECTOR = """
select n.file_name
from read_parquet($rows, file_row_number = true) n
anti join read_parquet($previous) p using (dataset, split, file_name)
order by n.file_row_number
"""

_IMAGE_VECTORS = """
select n.dataset, n.split, n.file_name,
       coalesce(f.embedding, p.embedding) as embedding
from read_parquet($rows, file_row_number = true) n
left join read_parquet($previous) p using (dataset, split, file_name)
left join read_parquet($fresh) f using (dataset, split, file_name)
order by n.file_row_number
"""

# A vector of the run before, for each box whose coordinates did not change.
_REUSABLE_CROPS = """
select p.box_id, p.embedding
from read_parquet($previous) p
join read_parquet($previous_boxes) pb using (box_id)
join read_parquet($rows) n using (box_id)
where pb.x = n.x and pb.y = n.y and pb.w = n.w and pb.h = n.h
"""

_CROPS_WITH_NO_VECTOR = f"""
with reusable as ({_REUSABLE_CROPS})
select n.file_name, n.box_id, n.box_index, n.x, n.y, n.w, n.h
from read_parquet($rows, file_row_number = true) n
anti join reusable using (box_id)
order by n.file_row_number
"""

_CROP_VECTORS = f"""
with reusable as ({_REUSABLE_CROPS})
select n.dataset, n.split, n.file_name, n.box_id, n.box_index,
       coalesce(f.embedding, r.embedding) as embedding
from read_parquet($rows, file_row_number = true) n
left join reusable r using (box_id)
left join read_parquet($fresh) f using (box_id)
order by n.file_row_number
"""


def _write_query_rows(
    *,
    connection: duckdb.DuckDBPyConnection,
    query: str,
    parameters: dict[str, str],
    path: UPath,
    schema: pa.Schema,
) -> int:
    count = 0
    with open_parquet_writer(path=path, schema=schema) as writer:
        for batch in connection.execute(
            query=query, parameters=parameters
        ).to_arrow_reader(ITEMS_PER_REQUEST):
            writer.write_batch(batch.cast(schema))
            count += batch.num_rows
    return count


def _build_crops(
    locate_image: Callable[[str], str], rows: dict[str, list]
) -> list[Crop]:
    return [
        Crop.rounded_to_pixels(path=locate_image(name), x=x, y=y, w=w, h=h)
        for name, x, y, w, h in zip(
            rows["file_name"], rows["x"], rows["y"], rows["w"], rows["h"], strict=True
        )
    ]


class _EmbeddingWriter:
    """Write embedding rows column wise, one row group per call to `add`.

    A 512 float vector per dict row costs far more to convert than the request that
    produced it, so this builds the Arrow arrays directly.
    """

    def __init__(
        self, path: UPath, schema: pa.Schema, dataset: str, split: str
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._writer = open_parquet_writer(path=path, schema=schema)
        self._dataset = dataset
        self._split = split
        self.count = 0

    def add(
        self,
        *,
        file_names: list[str],
        box_keys: BoxKeys | None,
        vectors: NDArray[np.float32],
    ) -> None:
        count = len(file_names)
        if count == 0:
            return
        columns: list[pa.Array] = [
            pa.array(obj=[self._dataset] * count, type=pa.string()),
            pa.array(obj=[self._split] * count, type=pa.string()),
            pa.array(obj=file_names, type=pa.string()),
        ]
        if box_keys is not None:
            columns.append(pa.array(obj=box_keys.ids, type=pa.string()))
            columns.append(pa.array(obj=box_keys.indices, type=pa.int32()))
        columns.append(_wrap_vectors_as_list_array(vectors))
        self._writer.write_batch(
            pa.RecordBatch.from_arrays(arrays=columns, schema=self._schema)
        )
        self.count += count

    def close(self) -> None:
        self._writer.close()


def _wrap_vectors_as_list_array(vectors: NDArray[np.float32]) -> pa.ListArray:
    """Wrap an (N, 512) array as a list column, with no Python round trip."""
    count, dimension = vectors.shape
    offsets = pa.array(np.arange(count + 1, dtype=np.int32) * dimension)
    return pa.ListArray.from_arrays(
        offsets=offsets, values=pa.array(vectors.reshape(-1))
    )


def _read_parquet_batches(path: UPath, columns: list[str]) -> Iterator[pa.RecordBatch]:
    return open_parquet_file(path).iter_batches(
        batch_size=ITEMS_PER_REQUEST, columns=columns
    )
