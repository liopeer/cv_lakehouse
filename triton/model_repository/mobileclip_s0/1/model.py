#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import asyncio
import json

import numpy as np
import triton_python_backend_utils as pb_utils

_IMAGE_PATH_INPUT = "IMAGE_PATH"
_IMAGE_BYTES_INPUT = "IMAGE_BYTES"
_TEXT_INPUT = "TEXT"
_CROP_X_INPUT = "CROP_X"
_CROP_Y_INPUT = "CROP_Y"
_CROP_WIDTH_INPUT = "CROP_WIDTH"
_CROP_HEIGHT_INPUT = "CROP_HEIGHT"
_EMBEDDING_OUTPUT = "EMBEDDING"
_INTERNAL_EMBEDDING_OUTPUT = "embeddings"
_MAX_CONCURRENT_SUB_REQUESTS_PARAMETER = "max_concurrent_sub_requests"

# The DALI preprocessing model's CROP_* inputs are required (not optional),
# since a single ragged-length UINT8 IMAGE_PATH row can't be batched together
# with others the way TYPE_STRING can -- each image is sent to the ensemble as
# its own single-image request (see _infer_one_image), and this sentinel means
# "no crop requested", resolved against the full image inside the DALI graph.
_NO_CROP = -1

# Caps concurrent BLS sub-requests per model instance. Each in-flight sub-request
# owns a python-backend shared-memory region, so an unbounded asyncio.gather over
# a large request (e.g. thousands of image paths in one call) can exhaust
# /dev/shm regardless of its configured size. 256 matches the TRT/DALI backends'
# preferred_batch_size, which is the most concurrency that actually helps.
_DEFAULT_MAX_CONCURRENT_SUB_REQUESTS = 256


class TritonPythonModel:
    def initialize(self, args):
        max_concurrent_sub_requests = _get_model_parameter_int(
            args=args,
            name=_MAX_CONCURRENT_SUB_REQUESTS_PARAMETER,
            default=_DEFAULT_MAX_CONCURRENT_SUB_REQUESTS,
        )
        self._semaphore = asyncio.Semaphore(max_concurrent_sub_requests)

    async def execute(self, requests):
        return list(await asyncio.gather(*(self._execute_one(request) for request in requests)))

    async def _execute_one(self, request):
        image_path_input = pb_utils.get_input_tensor_by_name(request, _IMAGE_PATH_INPUT)
        image_bytes_input = pb_utils.get_input_tensor_by_name(request, _IMAGE_BYTES_INPUT)
        text_input = pb_utils.get_input_tensor_by_name(request, _TEXT_INPUT)

        primary_inputs = (image_path_input, image_bytes_input, text_input)
        if sum(input_tensor is not None for input_tensor in primary_inputs) > 1:
            raise pb_utils.TritonModelException(
                f"Provide exactly one of {_IMAGE_PATH_INPUT}, {_IMAGE_BYTES_INPUT}, or {_TEXT_INPUT}."
            )
        if not any(input_tensor is not None for input_tensor in primary_inputs):
            raise pb_utils.TritonModelException(
                f"Provide {_IMAGE_PATH_INPUT}, {_IMAGE_BYTES_INPUT}, or {_TEXT_INPUT}."
            )

        if image_path_input is not None:
            image_paths = _decode_string_array(image_path_input.as_numpy())
            crop_boxes = _get_crop_boxes(request=request, count=len(image_paths))

            # Fan the images in this request out as concurrent BLS sub-requests
            # (rather than one call carrying all of them) so Triton's dynamic
            # batcher can still coalesce them into a single batched DALI/TensorRT
            # execution -- a single ragged-per-row UINT8 IMAGE_PATH tensor can't
            # represent images of different path lengths the way TYPE_STRING did.
            # The semaphore bounds how many of these are in flight at once.
            embeddings = await asyncio.gather(
                *(
                    self._infer_one_image(image_path=path, crop_box=crop_box)
                    for path, crop_box in zip(image_paths, crop_boxes)
                )
            )
        elif image_bytes_input is not None:
            _raise_if_crop_inputs_present(request=request)
            image_bytes = _get_bytes_values(image_bytes_input.as_numpy())
            embeddings = await asyncio.gather(
                *(self._infer_one_image_bytes(value=value) for value in image_bytes)
            )
        else:
            _raise_if_crop_inputs_present(request=request)
            texts = _decode_string_array(text_input.as_numpy())
            # Fan the texts out the same way as images, so the dynamic batcher
            # on _mobileclip_s0_text_backend_trt can coalesce concurrent
            # single-string sub-requests into one batched TensorRT execution.
            embeddings = await asyncio.gather(
                *(self._infer_one_text(text=text) for text in texts)
            )

        stacked = np.stack(embeddings, axis=0).astype(np.float32)
        return pb_utils.InferenceResponse([pb_utils.Tensor(_EMBEDDING_OUTPUT, stacked)])

    async def _infer_one_image(self, image_path, crop_box):
        x, y, width, height = crop_box if crop_box is not None else (_NO_CROP,) * 4
        inputs = [
            pb_utils.Tensor(_IMAGE_PATH_INPUT, _path_to_bytes(image_path)),
            pb_utils.Tensor(_CROP_X_INPUT, _scalar_int64(x)),
            pb_utils.Tensor(_CROP_Y_INPUT, _scalar_int64(y)),
            pb_utils.Tensor(_CROP_WIDTH_INPUT, _scalar_int64(width)),
            pb_utils.Tensor(_CROP_HEIGHT_INPUT, _scalar_int64(height)),
        ]
        infer_req = pb_utils.InferenceRequest(
            model_name="_mobileclip_s0_image_pipeline",
            requested_output_names=[_INTERNAL_EMBEDDING_OUTPUT],
            inputs=inputs,
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU,
                0,
            ),
        )
        async with self._semaphore:
            result = await infer_req.async_exec()
        if result.has_error():
            raise pb_utils.TritonModelException(result.error().message())
        embedding = pb_utils.get_output_tensor_by_name(
            result, _INTERNAL_EMBEDDING_OUTPUT
        ).as_numpy()
        return embedding[0]

    async def _infer_one_image_bytes(self, value):
        infer_req = pb_utils.InferenceRequest(
            model_name="_mobileclip_s0_image_bytes_pipeline",
            requested_output_names=[_INTERNAL_EMBEDDING_OUTPUT],
            inputs=[pb_utils.Tensor(_IMAGE_BYTES_INPUT, _bytes_to_tensor(value))],
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU,
                0,
            ),
        )
        async with self._semaphore:
            result = await infer_req.async_exec()
        if result.has_error():
            raise pb_utils.TritonModelException(result.error().message())
        embedding = pb_utils.get_output_tensor_by_name(
            result, _INTERNAL_EMBEDDING_OUTPUT
        ).as_numpy()
        return embedding[0]

    async def _infer_one_text(self, text):
        infer_req = pb_utils.InferenceRequest(
            model_name="_mobileclip_s0_text_pipeline",
            requested_output_names=[_INTERNAL_EMBEDDING_OUTPUT],
            inputs=[pb_utils.Tensor("text", _text_to_string_tensor(text))],
            preferred_memory=pb_utils.PreferredMemory(
                pb_utils.TRITONSERVER_MEMORY_CPU,
                0,
            ),
        )
        async with self._semaphore:
            result = await infer_req.async_exec()
        if result.has_error():
            raise pb_utils.TritonModelException(result.error().message())
        embedding = pb_utils.get_output_tensor_by_name(
            result, _INTERNAL_EMBEDDING_OUTPUT
        ).as_numpy()
        return embedding[0]


def _path_to_bytes(path):
    return np.frombuffer(path.encode("utf-8"), dtype=np.uint8).reshape(1, -1)


def _bytes_to_tensor(value):
    return np.frombuffer(value, dtype=np.uint8).reshape(1, -1)


def _text_to_string_tensor(text):
    return np.array([[text.encode("utf-8")]], dtype=object)


def _scalar_int64(value):
    return np.array([[value]], dtype=np.int64)


def _decode_string_array(values):
    flat = np.asarray(values).reshape(-1)
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in flat]


def _get_bytes_values(values):
    return [bytes(value) for value in np.asarray(values).reshape(-1)]


def _get_model_parameter_int(args, name, default):
    model_config = json.loads(args["model_config"])
    parameter = model_config.get("parameters", {}).get(name)
    if parameter is None:
        return default

    value = int(parameter["string_value"])
    if value < 1:
        raise pb_utils.TritonModelException(f"{name} must be positive.")
    return value


def _get_crop_boxes(request, count):
    crop_names = (_CROP_X_INPUT, _CROP_Y_INPUT, _CROP_WIDTH_INPUT, _CROP_HEIGHT_INPUT)
    crop_inputs = {name: pb_utils.get_input_tensor_by_name(request, name) for name in crop_names}
    present_names = {name for name, value in crop_inputs.items() if value is not None}

    if not present_names:
        return [None] * count
    if present_names != set(crop_names):
        missing = ", ".join(sorted(set(crop_names) - present_names))
        raise pb_utils.TritonModelException(f"Missing crop inputs: {missing}.")

    xs, ys, widths, heights = (
        np.asarray(crop_inputs[name].as_numpy()).reshape(-1) for name in crop_names
    )
    return [
        (int(xs[i]), int(ys[i]), int(widths[i]), int(heights[i])) for i in range(count)
    ]


def _raise_if_crop_inputs_present(request):
    crop_names = (_CROP_X_INPUT, _CROP_Y_INPUT, _CROP_WIDTH_INPUT, _CROP_HEIGHT_INPUT)
    if any(pb_utils.get_input_tensor_by_name(request, name) is not None for name in crop_names):
        raise pb_utils.TritonModelException(
            f"Crop inputs are only supported with {_IMAGE_PATH_INPUT}."
        )
