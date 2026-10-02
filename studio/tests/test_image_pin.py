#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import re
from importlib.metadata import version
from pathlib import Path


def test_the_image_builds_the_lightly_studio_that_the_tests_run() -> None:
    """The sync writes LightlyStudio tables, so both must be one schema."""
    dockerfile = (Path(__file__).parents[1] / "Dockerfile").read_text()
    match = re.search(
        pattern=r"^ARG LIGHTLY_STUDIO_REF=v(.+)$", string=dockerfile, flags=re.M
    )
    assert match is not None
    assert match.group(1) == version("lightly-studio")
