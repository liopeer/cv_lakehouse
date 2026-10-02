#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""The query parameters that select gold rows."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from cv_lakehouse.split_roles import SplitRole

DRAFT_RELEASE = "draft"
DEFAULT_ROW_LIMIT = 1_000
MAX_ROW_LIMIT = 100_000


class ImageFilter(BaseModel):
    """Select images. A parameter that is given more than once matches any value."""

    model_config = ConfigDict(extra="forbid")

    dataset: list[str] = Field(default_factory=list)
    role: list[SplitRole] = Field(default_factory=list)
    split: list[str] = Field(default_factory=list)
    # Which val and test rows: the number of a release, or `draft` for the gold that
    # the curators work on. A request that can return such rows must name one, so no
    # benchmark moves to other rows without a change on its side. Train rows are
    # always those of the current gold.
    release: str | None = Field(default=None, pattern=r"^(draft|[0-9]+)$")
    # Rows that a gold build changed after this time.
    changed_since: datetime | None = None
    commercial_use: bool | None = None
    limit: int = Field(default=DEFAULT_ROW_LIMIT, ge=1, le=MAX_ROW_LIMIT)
    # The `next_after` of the page before. Rows come in the order of their id.
    after: str | None = None


class BoxFilter(ImageFilter):
    class_name: list[str] = Field(default_factory=list)
