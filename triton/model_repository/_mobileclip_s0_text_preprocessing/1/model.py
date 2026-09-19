#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import numpy as np
import triton_python_backend_utils as pb_utils

from shared_deps.mobileclip_text_encoder import MobileCLIPTextPreprocessor

VARIANT_NAME = "mobileclip_s0"


class TritonPythonModel:
    def initialize(self, args):
        self.preprocessor = MobileCLIPTextPreprocessor(VARIANT_NAME)

    def execute(self, requests):
        responses = []
        for request in requests:
            text_np = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            text_str = text_np[0].decode("utf-8")
            tokens = self.preprocessor(text_str)  # [77] int64
            out_tensor = pb_utils.Tensor("tokens", tokens)
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
