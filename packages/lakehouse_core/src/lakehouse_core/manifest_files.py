#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Write and read the manifest that every layer keeps next to its output."""

import json

from pydantic import BaseModel
from upath import UPath


def write_manifest(path: UPath, manifest: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest[T: BaseModel](path: UPath, model: type[T]) -> T:
    return model.model_validate(json.loads(path.read_text()))
