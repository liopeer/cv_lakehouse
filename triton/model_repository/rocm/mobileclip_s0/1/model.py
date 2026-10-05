#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# MobileCLIP on ROCm, in torch. The model takes the same inputs as the CUDA ensemble
# behind `mobileclip_s0`: IMAGE_PATH, IMAGE_BYTES or TEXT, with optional CROP_* inputs.
# `entrypoint.sh` links the checkpoint into the version directory.
import json
import os

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from shared_deps.image_sources import ImageFetchError
from shared_deps.mobileclip_torch_embedder import FULL_IMAGE, MobileCLIPTorchEmbedder

_MODEL_NAME = "mobileclip_s0"
_IMAGE_PATH_INPUT = "IMAGE_PATH"
_IMAGE_BYTES_INPUT = "IMAGE_BYTES"
_TEXT_INPUT = "TEXT"
_CROP_NAMES = ("CROP_X", "CROP_Y", "CROP_WIDTH", "CROP_HEIGHT")
_EMBEDDING_OUTPUT = "EMBEDDING"


class TritonPythonModel:
    def initialize(self, args):
        parameters = json.loads(args["model_config"])["parameters"]
        checkpoint_path = os.path.join(
            args["model_repository"], args["model_version"], f"{_MODEL_NAME}.pt"
        )
        self._embedder = MobileCLIPTorchEmbedder(
            model_name=_MODEL_NAME,
            checkpoint_path=checkpoint_path,
            device=torch.device("cuda", 0),
            dtype=torch.float16,
            batch_size=_get_int_parameter(parameters=parameters, name="batch_size"),
            workers=_get_int_parameter(parameters=parameters, name="workers"),
            fetch_timeout_seconds=_get_int_parameter(
                parameters=parameters, name="fetch_timeout_seconds"
            ),
            compile_encoders=True,
        )
        self._embedder.warm_up()

    def execute(self, requests):
        return [self._execute_one(request) for request in requests]

    def finalize(self):
        self._embedder.close()

    def _execute_one(self, request):
        try:
            embeddings = self._embed(request)
        except (ValueError, ImageFetchError, OSError) as error:
            return pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error)))
        return pb_utils.InferenceResponse([pb_utils.Tensor(_EMBEDDING_OUTPUT, embeddings)])

    def _embed(self, request):
        image_paths = pb_utils.get_input_tensor_by_name(request, _IMAGE_PATH_INPUT)
        image_bytes = pb_utils.get_input_tensor_by_name(request, _IMAGE_BYTES_INPUT)
        texts = pb_utils.get_input_tensor_by_name(request, _TEXT_INPUT)
        primary_inputs = (image_paths, image_bytes, texts)
        if sum(tensor is not None for tensor in primary_inputs) != 1:
            raise ValueError(
                f"Provide exactly one of {_IMAGE_PATH_INPUT}, {_IMAGE_BYTES_INPUT}, "
                f"or {_TEXT_INPUT}."
            )
        crop_tensors = [pb_utils.get_input_tensor_by_name(request, name) for name in _CROP_NAMES]
        if texts is not None:
            if any(tensor is not None for tensor in crop_tensors):
                raise ValueError(
                    f"Crop inputs go with {_IMAGE_PATH_INPUT} or {_IMAGE_BYTES_INPUT}, "
                    f"not {_TEXT_INPUT}."
                )
            return self._embedder.embed_texts(_decode_strings(texts.as_numpy()))
        if image_paths is not None:
            sources = _decode_strings(image_paths.as_numpy())
        else:
            sources = [bytes(value) for value in image_bytes.as_numpy().reshape(-1)]
        return self._embedder.embed_images(
            sources=sources,
            crop_boxes=_read_crop_boxes(crop_tensors=crop_tensors, count=len(sources)),
        )


def _decode_strings(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def _read_crop_boxes(*, crop_tensors, count):
    present = [tensor is not None for tensor in crop_tensors]
    if not any(present):
        return [FULL_IMAGE] * count
    if not all(present):
        missing = [name for name, is_present in zip(_CROP_NAMES, present) if not is_present]
        raise ValueError(f"Missing crop inputs: {', '.join(missing)}.")
    columns = [tensor.as_numpy().reshape(-1) for tensor in crop_tensors]
    if any(len(column) != count for column in columns):
        raise ValueError(f"Every crop input needs {count} values.")
    return [tuple(int(column[index]) for column in columns) for index in range(count)]


def _get_int_parameter(*, parameters, name):
    value = int(parameters[name]["string_value"])
    if value < 1:
        raise pb_utils.TritonModelException(f"{name} must be positive.")
    return value
