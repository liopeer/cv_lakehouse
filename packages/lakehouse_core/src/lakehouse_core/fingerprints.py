#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import hashlib
from collections.abc import Sequence

FINGERPRINT_CHARACTERS = 12


def sha256_fingerprint(parts: Sequence[str]) -> str:
    """Digest the parts that decide what an asset writes."""
    joined = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:FINGERPRINT_CHARACTERS]
