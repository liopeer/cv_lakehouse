#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# GPU image preprocessing for the ROCm image, in place of the DALI models. The
# `input_kind` parameter picks IMAGE_PATH with the CROP_* inputs, or IMAGE_BYTES.
# `_mobileclip_s0_image_bytes_preprocessing` links to this file.
#
# The output goes back as a CPU tensor, and the ORT backend copies it to the GPU.
import json

import torch
import triton_python_backend_utils as pb_utils

from shared_deps.mobileclip_rocm_preprocessing import (
    NO_CROP,
    MobileCLIPImageDecoder,
    preprocess_images,
)

_INPUT_KIND_PARAMETER = "input_kind"
_DECODE_THREADS_PARAMETER = "decode_threads"
_CROP_NAMES = ("CROP_X", "CROP_Y", "CROP_WIDTH", "CROP_HEIGHT")
_OUTPUT_NAME = "images"


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        parameters = model_config["parameters"]
        self._input_kind = parameters[_INPUT_KIND_PARAMETER]["string_value"]
        if self._input_kind not in ("path", "bytes"):
            raise pb_utils.TritonModelException(
                f"{_INPUT_KIND_PARAMETER} must be 'path' or 'bytes'."
            )
        self._device = torch.device("cuda", int(args["model_instance_device_id"]))
        self._decoder = MobileCLIPImageDecoder(
            device=self._device,
            decode_threads=int(parameters[_DECODE_THREADS_PARAMETER]["string_value"]),
        )

    def execute(self, requests):
        encoded_images = []
        crop_boxes = []
        request_sizes = []
        for request in requests:
            request_encoded, request_crop_boxes = self._read_request(request)
            encoded_images.extend(request_encoded)
            crop_boxes.extend(request_crop_boxes)
            request_sizes.append(len(request_encoded))

        with torch.inference_mode():
            images = self._decoder.decode_images(encoded_images)
            batch = preprocess_images(images=images, crop_boxes=crop_boxes).cpu().numpy()

        responses = []
        offset = 0
        for size in request_sizes:
            output = pb_utils.Tensor(_OUTPUT_NAME, batch[offset : offset + size])
            responses.append(pb_utils.InferenceResponse([output]))
            offset += size
        return responses

    def finalize(self):
        self._decoder.close()

    def _read_request(self, request):
        if self._input_kind == "bytes":
            rows = pb_utils.get_input_tensor_by_name(request, "IMAGE_BYTES").as_numpy()
            return [row.tobytes() for row in rows], [(NO_CROP,) * 4] * len(rows)

        rows = pb_utils.get_input_tensor_by_name(request, "IMAGE_PATH").as_numpy()
        encoded = [_read_file(row.tobytes().decode("utf-8")) for row in rows]
        crops = [
            pb_utils.get_input_tensor_by_name(request, name).as_numpy().reshape(-1)
            for name in _CROP_NAMES
        ]
        return encoded, [tuple(int(values[i]) for values in crops) for i in range(len(rows))]


def _read_file(path):
    with open(path, "rb") as f:
        return f.read()
