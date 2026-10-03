#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Write and read the manifest that every layer keeps next to its output."""

import json
from pathlib import Path

from pydantic import BaseModel


def write_manifest(path: Path, manifest: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest[T: BaseModel](path: Path, model: type[T]) -> T:
    return model.model_validate(json.loads(path.read_text()))
