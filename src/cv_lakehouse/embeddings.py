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

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray

from cv_lakehouse.silver_schema import (
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

        The path is resolved by the server, so it has to name the same file inside the
        container.
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


def clear_embeddings(silver_dir: Path, split: str) -> None:
    """Remove one split's embedding files.

    A rematerialisation with no server rewrites the images and the boxes, so a vector
    left behind describes pixels that a query no longer has a row for.
    """
    embeddings_file(silver_dir=silver_dir, split=split).unlink(missing_ok=True)
    crop_embeddings_file(silver_dir=silver_dir, split=split).unlink(missing_ok=True)


def write_image_embeddings(
    *,
    embedder: Embedder,
    silver_dir: Path,
    dataset: str,
    split: str,
    image_root: str,
) -> int:
    """Embed every image of one split, and write the Parquet. Return the row count.

    The rows keep the order of the images file, so a reader joins the two on the key
    and never has to sort.
    """
    root = image_root.rstrip("/")
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
                box_indices=None,
                vectors=embedder.embed_images([f"{root}/{name}" for name in names]),
            )
    finally:
        writer.close()
    return writer.count


def write_crop_embeddings(
    *,
    embedder: Embedder,
    silver_dir: Path,
    dataset: str,
    split: str,
    image_root: str,
) -> int:
    """Embed every box of one split as a crop, and write the Parquet.

    Flagged boxes are embedded too. Silver keeps them, and its consumer decides.
    """
    root = image_root.rstrip("/")
    columns = ["file_name", "box_index", "x", "y", "w", "h"]
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
            crops = [
                Crop.rounded_to_pixels(path=f"{root}/{name}", x=x, y=y, w=w, h=h)
                for name, x, y, w, h in zip(
                    rows["file_name"],
                    rows["x"],
                    rows["y"],
                    rows["w"],
                    rows["h"],
                    strict=True,
                )
            ]
            writer.add(
                file_names=rows["file_name"],
                box_indices=rows["box_index"],
                vectors=embedder.embed_crops(crops),
            )
    finally:
        writer.close()
    return writer.count


class _EmbeddingWriter:
    """Write embedding rows column wise, one row group per call to `add`.

    A 512 float vector per dict row costs far more to convert than the request that
    produced it, so this builds the Arrow arrays directly.
    """

    def __init__(self, path: Path, schema: pa.Schema, dataset: str, split: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._schema = schema
        self._writer = pq.ParquetWriter(where=path, schema=schema)
        self._dataset = dataset
        self._split = split
        self.count = 0

    def add(
        self,
        *,
        file_names: list[str],
        box_indices: list[int] | None,
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
        if box_indices is not None:
            columns.append(pa.array(obj=box_indices, type=pa.int32()))
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


def _read_parquet_batches(path: Path, columns: list[str]) -> Iterator[pa.RecordBatch]:
    return pq.ParquetFile(path).iter_batches(
        batch_size=ITEMS_PER_REQUEST, columns=columns
    )
