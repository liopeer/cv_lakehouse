#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# The ROCm build of Triton has no ensemble scheduler, so this Python model does what the
# ensembles of the CUDA image do. It sends a request to the preprocessing model, and
# sends that output to the encoder. The `steps` parameter lists each model with its
# output, as `model:output,model:output`. The output of a step is the input of the next
# step with the same name.
#
# The text and image bytes pipelines link to this file.
import asyncio
import json

import triton_python_backend_utils as pb_utils

_STEPS_PARAMETER = "steps"


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        steps = model_config["parameters"][_STEPS_PARAMETER]["string_value"]
        self._steps = [step.split(":") for step in steps.split(",")]

    async def execute(self, requests):
        return list(await asyncio.gather(*(self._run_steps(request) for request in requests)))

    async def _run_steps(self, request):
        tensors = request.inputs()
        for model_name, output_name in self._steps:
            infer_request = pb_utils.InferenceRequest(
                model_name=model_name,
                requested_output_names=[output_name],
                inputs=tensors,
                preferred_memory=pb_utils.PreferredMemory(pb_utils.TRITONSERVER_MEMORY_CPU, 0),
            )
            result = await infer_request.async_exec()
            if result.has_error():
                return pb_utils.InferenceResponse(error=pb_utils.TritonError(result.error().message()))
            tensors = result.output_tensors()
        return pb_utils.InferenceResponse(output_tensors=tensors)

