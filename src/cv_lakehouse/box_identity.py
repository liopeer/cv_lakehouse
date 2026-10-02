#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The ids that name an image and a box everywhere outside its own silver file.

An id is an md5 of the semantic key, written as a UUID. DuckDB derives the same text
with `md5(...)::UUID`, and LightlyStudio takes it as a primary key.

A box is keyed by its position in the source, before normalisation drops anything.
Bronze pins the bytes, so a change to the drop policy leaves every id as it was.
"""

from __future__ import annotations

from hashlib import md5
from uuid import UUID


def derive_image_id(*, dataset: str, split: str, file_name: str) -> str:
    return _render_md5_as_uuid(f"{dataset}/{split}/{file_name}")


def derive_box_id(
    *, dataset: str, split: str, file_name: str, source_box_index: int
) -> str:
    return _render_md5_as_uuid(f"{dataset}/{split}/{file_name}#{source_box_index}")


def _render_md5_as_uuid(key: str) -> str:
    return str(UUID(hex=md5(key.encode("utf-8")).hexdigest()))
