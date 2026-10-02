#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Send one page of rows as JSON, or as an Arrow IPC stream on request."""

from __future__ import annotations

import pyarrow as pa
from fastapi import Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
# An Arrow stream has no place for the cursor, so it travels in a header.
NEXT_AFTER_HEADER = "X-Next-After"


def build_rows_response(
    *, rows: pa.Table, next_after: str | None, accept: str | None
) -> Response:
    if accept is not None and ARROW_STREAM_MEDIA_TYPE in accept:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink=sink, schema=rows.schema) as writer:
            writer.write_table(rows)
        headers = {} if next_after is None else {NEXT_AFTER_HEADER: next_after}
        return Response(
            content=sink.getvalue().to_pybytes(),
            media_type=ARROW_STREAM_MEDIA_TYPE,
            headers=headers,
        )
    return JSONResponse(
        content=jsonable_encoder({"rows": rows.to_pylist(), "next_after": next_after})
    )
