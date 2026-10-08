#
# SPDX-License-Identifier: MIT
# Copyright (c) 2025–2026 Lionel Peer
#
import logging
import threading
from pathlib import Path

import pytest
from upath import UPath

from lakehouse_core import disk_lease as disk_lease_module
from lakehouse_core.disk_lease import DiskLease

LOG = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _poll_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(target=disk_lease_module, name="POLL_SECONDS", value=0.01)


def test_a_second_holder_waits_and_logs_the_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease = DiskLease(tmp_path / "disk.lock")
    held = threading.Event()
    release = threading.Event()
    order: list[str] = []

    def hold_first() -> None:
        with lease.hold(target=tmp_path, reason="Unpacking train_0.tar.gz", log=LOG):
            held.set()
            release.wait(timeout=5)
            order.append("first")

    def hold_second() -> None:
        with lease.hold(target=tmp_path, reason="Silver pp4av test", log=LOG):
            order.append("second")

    first = threading.Thread(target=hold_first)
    first.start()
    held.wait(timeout=5)
    with caplog.at_level(logging.INFO):
        second = threading.Thread(target=hold_second)
        second.start()
        second.join(timeout=0.2)
        assert second.is_alive()
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)

    assert order == ["first", "second"]
    assert "Silver pp4av test waits for the disk" in caplog.text
    assert "Unpacking train_0.tar.gz" in caplog.text


def test_a_lease_with_no_path_does_nothing(tmp_path: Path) -> None:
    lease = DiskLease(None)
    with (
        lease.hold(target=tmp_path, reason="one", log=LOG),
        lease.hold(target=tmp_path, reason="two", log=LOG),
    ):
        pass
    assert list(tmp_path.iterdir()) == []


def test_a_remote_target_takes_no_lease(tmp_path: Path) -> None:
    lease = DiskLease(tmp_path / "disk.lock")
    remote = UPath("memory://lake/bronze")
    with (
        lease.hold(target=remote, reason="one", log=LOG),
        lease.hold(target=remote, reason="two", log=LOG),
    ):
        pass
    assert not (tmp_path / "disk.lock").exists()


def test_the_lease_is_free_after_a_failure(tmp_path: Path) -> None:
    lease = DiskLease(tmp_path / "disk.lock")
    with (
        pytest.raises(RuntimeError),
        lease.hold(target=tmp_path, reason="one", log=LOG),
    ):
        raise RuntimeError("unpack failed")
    with lease.hold(target=tmp_path, reason="two", log=LOG):
        pass
