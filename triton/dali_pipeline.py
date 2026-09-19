#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#

# GPU preprocessing pipeline for _mobileclip_s0_image_preprocessing, run through
# NVIDIA DALI instead of the CPU/PIL Python backend. Reproduces:
#   IMAGE_PATH or IMAGE_BYTES -> optional crop -> resize-shorter-side(256) -> center-crop(256)
#   -> /255 -> CHW float32
# Serialized to model.dali by entrypoint.sh on the first start; the Triton DALI
# backend loads the serialized plan directly. The script runs inside the tritonserver
# image, so the DALI that serializes the pipeline is the one that later loads it.

import argparse

from nvidia.dali import fn, pipeline_def, types

IMAGE_SIZE = 256
MAX_BATCH_SIZE = 64


@pipeline_def
def mobileclip_preprocessing_pipeline(input_kind: str):
    if input_kind == "path":
        paths = fn.external_source(name="IMAGE_PATH", dtype=types.UINT8, ndim=1)
        encoded = fn.io.file.read(paths)
        crop_x = fn.external_source(name="CROP_X", dtype=types.INT64, ndim=1)
        crop_y = fn.external_source(name="CROP_Y", dtype=types.INT64, ndim=1)
        crop_w = fn.external_source(name="CROP_WIDTH", dtype=types.INT64, ndim=1)
        crop_h = fn.external_source(name="CROP_HEIGHT", dtype=types.INT64, ndim=1)
    else:
        encoded = fn.external_source(name="IMAGE_BYTES", dtype=types.UINT8, ndim=1)

    image = fn.decoders.image(encoded, device="mixed", output_type=types.RGB)

    if input_kind == "bytes":
        resized = fn.resize(image, resize_shorter=IMAGE_SIZE, interp_type=types.INTERP_LINEAR)
        return fn.crop_mirror_normalize(
            resized,
            crop=(IMAGE_SIZE, IMAGE_SIZE),
            mean=0.0,
            std=255.0,
            output_layout="CHW",
            dtype=types.FLOAT,
        )

    # -1 in CROP_* means "no crop requested" -> fall back to the full image,
    # resolved per-sample via cheap header-only shape peeking + arithmetic
    # select (mask * requested + (1 - mask) * full_image), since a static
    # DALI graph cannot branch on this at trace time.
    shape = fn.peek_image_shape(encoded)  # [H, W, C]
    img_h = fn.cast(shape[0], dtype=types.INT64)
    img_w = fn.cast(shape[1], dtype=types.INT64)

    mask = fn.cast(crop_w >= 0, dtype=types.INT64)
    not_mask = 1 - mask
    eff_x = mask * crop_x
    eff_y = mask * crop_y
    eff_w = mask * crop_w + not_mask * img_w
    eff_h = mask * crop_h + not_mask * img_h

    cropped = fn.slice(
        image,
        fn.stack(eff_y, eff_x).gpu(),
        fn.stack(eff_h, eff_w).gpu(),
        axes=[0, 1],
        out_of_bounds_policy="trim_to_shape",
    )

    resized = fn.resize(cropped, resize_shorter=IMAGE_SIZE, interp_type=types.INTERP_LINEAR)
    return fn.crop_mirror_normalize(
        resized,
        crop=(IMAGE_SIZE, IMAGE_SIZE),
        mean=0.0,
        std=255.0,
        output_layout="CHW",
        dtype=types.FLOAT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-batch-size", type=int, default=MAX_BATCH_SIZE)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--input-kind", choices=("path", "bytes"), required=True)
    args = parser.parse_args()

    pipe = mobileclip_preprocessing_pipeline(
        batch_size=args.max_batch_size,
        num_threads=args.num_threads,
        device_id=args.device_id,
        input_kind=args.input_kind,
    )
    pipe.serialize(filename=args.out)


if __name__ == "__main__":
    main()
