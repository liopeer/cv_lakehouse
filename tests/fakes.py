#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Stand-ins for the Triton server, so no test needs one.

`FakeTritonClient` replaces the gRPC client and decodes the tensors the real one would
put on the wire. `FakeEmbedder` replaces the whole embedder, for a test that cares about
the Parquet rather than the wire.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from numpy.typing import NDArray
from tritonclient.utils import deserialize_bytes_tensor

from cv_lakehouse.embeddings import (
    EMBEDDING_DIMENSION,
    Crop,
    Embedder,
    InferenceClient,
    InferResult,
)


def unit_vectors(count: int) -> NDArray[np.float32]:
    """One distinct unit vector per row, so a test can tell the rows apart."""
    vectors = np.zeros(shape=(count, EMBEDDING_DIMENSION), dtype=np.float32)
    for index in range(count):
        vectors[index, index % EMBEDDING_DIMENSION] = 1.0
    return vectors


class _FakeResult(InferResult):
    def __init__(self, embeddings: NDArray[np.float32]) -> None:
        self._embeddings = embeddings

    def as_numpy(self, name: str) -> NDArray[np.float32] | None:
        return self._embeddings if name == "EMBEDDING" else None


class FakeTritonClient(InferenceClient):
    """Record every request, and answer with one unit vector per item."""

    def __init__(self) -> None:
        self.requests: list[dict[str, list[Any]]] = []
        self.model_names: list[str] = []
        self.output_names: list[list[str]] = []
        # Set this to answer with one embedding too few, which the client must reject.
        self.drop_last = False

    def infer(
        self,
        model_name: str,
        inputs: list[grpcclient.InferInput],
        outputs: list[grpcclient.InferRequestedOutput],
    ) -> _FakeResult:
        request = {
            infer_input.name(): _decode_infer_input(infer_input)
            for infer_input in inputs
        }
        self.requests.append(request)
        self.model_names.append(model_name)
        self.output_names.append([output.name() for output in outputs])
        count = len(next(iter(request.values())))
        return _FakeResult(unit_vectors(count - 1 if self.drop_last else count))


def _decode_infer_input(infer_input: grpcclient.InferInput) -> list[Any]:
    """Read a tensor back off the wire, which is what the server sees."""
    # `_get_content` is the only accessor tritonclient offers for a serialised tensor.
    content = infer_input._get_content()  # noqa: SLF001
    assert content is not None
    if infer_input.datatype() == "BYTES":
        return list(
            deserialize_bytes_tensor(np.frombuffer(buffer=content, dtype=np.uint8))
        )
    return list(np.frombuffer(buffer=content, dtype=np.int64))


class FakeEmbedder(Embedder):
    """Answer like the embedder, and remember what it was asked to embed."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.crops: list[Crop] = []

    def embed_images(self, paths: Sequence[str]) -> NDArray[np.float32]:
        self.paths.extend(paths)
        return unit_vectors(len(paths))

    def embed_crops(self, crops: Sequence[Crop]) -> NDArray[np.float32]:
        self.crops.extend(crops)
        return unit_vectors(len(crops))
