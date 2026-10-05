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
            texts = pb_utils.get_input_tensor_by_name(request, "text").as_numpy().reshape(-1)
            tokens = np.stack([self.preprocessor(text.decode("utf-8")) for text in texts])
            out_tensor = pb_utils.Tensor("tokens", tokens)  # [N, 77] int64
            responses.append(pb_utils.InferenceResponse([out_tensor]))
        return responses
