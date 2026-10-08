#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
"""Log the progress of a long loop, at most once per interval."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from datetime import timedelta

DEFAULT_INTERVAL_SECONDS = 60.0


class ProgressLog:
    """Count the items of one step, and log the count, the rate and the time left."""

    def __init__(
        self,
        *,
        log: Callable[[str], None],
        label: str,
        total: int | None = None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._log = log
        self._label = label
        self._total = total
        self._interval_seconds = interval_seconds
        self._clock = clock
        self._started_at = self._last_logged_at = clock()
        self.done = 0

    def advance(self, count: int) -> None:
        self.done += count
        now = self._clock()
        if now - self._last_logged_at >= self._interval_seconds:
            self._last_logged_at = now
            self._log(self._describe(now))

    def finish(self) -> None:
        elapsed = self._clock() - self._started_at
        self._log(f"{self._label}: {self.done:,} done in {_format_duration(elapsed)}")

    def _describe(self, now: float) -> str:
        elapsed = now - self._started_at
        rate = self.done / elapsed if elapsed > 0 else 0.0
        if self._total is None:
            return f"{self._label}: {self.done:,}, {rate:,.0f}/s"
        share = self.done / self._total if self._total else 1.0
        left = (self._total - self.done) / rate if rate > 0 else None
        return (
            f"{self._label}: {self.done:,} of {self._total:,} ({share:.1%}), "
            f"{rate:,.0f}/s, {_format_duration(left)} left"
        )


def iter_with_progress[T](items: Iterable[T], progress: ProgressLog) -> Iterator[T]:
    for item in items:
        yield item
        progress.advance(1)
    progress.finish()


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown time"
    return str(timedelta(seconds=round(seconds)))
