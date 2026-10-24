"""Single-server flock on ``<socket>.lock``.

``connect()`` cannot tell a stale socket from a live one with a full listen
backlog (macOS reports ECONNREFUSED for both), so liveness is decided by a lock
the kernel drops when the owner dies. Clients never touch it.
"""

from __future__ import annotations

import errno
import os
import shutil
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from stt_server import __main__ as cli
from stt_server.backend import EchoBackend
from stt_server.server import (
    ServerConfig,
    TranscriptionServer,
    acquire_socket_lock,
    socket_lock_held,
)


@pytest.fixture
def short_home(monkeypatch):
    """Private short $HOME: the socket-dir ancestor walk is rooted at it."""
    home = Path(tempfile.mkdtemp(prefix="stt.", dir="/tmp"))
    monkeypatch.setenv("HOME", str(home))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def test_second_acquire_fails_until_first_released(tmp_path):
    sock = tmp_path / "a.sock"
    fd = acquire_socket_lock(sock)
    with pytest.raises(OSError) as exc:
        acquire_socket_lock(sock)
    assert exc.value.errno == errno.EADDRINUSE
    os.close(fd)  # what process death does
    os.close(acquire_socket_lock(sock))


def test_lock_file_is_private_and_left_in_place(tmp_path):
    sock = tmp_path / "a.sock"
    os.close(acquire_socket_lock(sock))
    lock = Path(f"{sock}.lock")
    assert lock.is_file() and (lock.stat().st_mode & 0o777) == 0o600


def test_lock_refuses_symlink(tmp_path):
    sock = tmp_path / "a.sock"
    target = tmp_path / "elsewhere"
    target.write_text("x")
    Path(f"{sock}.lock").symlink_to(target)
    with pytest.raises(OSError):
        acquire_socket_lock(sock)


def test_probe_never_creates_the_lock_file(tmp_path):
    sock = tmp_path / "a.sock"
    assert socket_lock_held(sock) is False
    assert not Path(f"{sock}.lock").exists()


def test_probe_reports_held_then_free(tmp_path):
    sock = tmp_path / "a.sock"
    fd = acquire_socket_lock(sock)
    assert socket_lock_held(sock) is True
    os.close(fd)
    assert socket_lock_held(sock) is False


def test_preflight_held_lock_beats_econnrefused(tmp_path, monkeypatch):
    """A live server with a full backlog looks stale to connect(); the lock wins."""
    sock = tmp_path / "live.sock"
    sock.write_bytes(b"socket inode sentinel")
    inode = sock.stat().st_ino
    real_is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda self: self == sock or real_is_socket(self))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise ConnectionRefusedError(errno.ECONNREFUSED, "backlog full")

    monkeypatch.setattr(socket, "socket", lambda *_a, **_k: Probe())
    args = SimpleNamespace(socket_path=str(sock), host=None, port=None)

    cli._preflight_listener(args)  # no lock holder: ECONNREFUSED is stale, as before
    fd = acquire_socket_lock(sock)
    try:
        with pytest.raises(OSError) as exc:
            cli._preflight_listener(args)
    finally:
        os.close(fd)
    assert exc.value.errno == errno.EADDRINUSE
    assert sock.stat().st_ino == inode


async def test_second_server_fails_before_backend_start_and_keeps_socket(short_home):
    sock = short_home / "s" / "stt.sock"
    first = TranscriptionServer(EchoBackend(), ServerConfig(socket_path=str(sock)))
    await first.start()
    try:
        inode = sock.stat().st_ino
        second_backend = EchoBackend()
        started = []

        async def _start():
            started.append(True)

        second_backend.start = _start
        second = TranscriptionServer(second_backend, ServerConfig(socket_path=str(sock)))
        with pytest.raises(OSError) as exc:
            await second.start()
        assert exc.value.errno == errno.EADDRINUSE
        assert not started  # failed before any model load
        assert sock.stat().st_ino == inode
    finally:
        await first.shutdown()


async def test_shutdown_releases_lock_for_restart(short_home):
    sock = short_home / "s" / "stt.sock"
    for _ in range(2):
        srv = TranscriptionServer(EchoBackend(), ServerConfig(socket_path=str(sock)))
        await srv.start()
        assert socket_lock_held(sock) is True
        await srv.shutdown()
        assert socket_lock_held(sock) is False


async def test_failed_start_releases_lock(short_home):
    sock = short_home / "s" / "stt.sock"
    backend = EchoBackend()

    async def _boom():
        raise RuntimeError("load failed")

    backend.start = _boom
    srv = TranscriptionServer(backend, ServerConfig(socket_path=str(sock)))
    with pytest.raises(RuntimeError):
        await srv.start()
    assert socket_lock_held(sock) is False


async def test_abort_sync_releases_lock(short_home):
    sock = short_home / "s" / "stt.sock"
    srv = TranscriptionServer(EchoBackend(), ServerConfig(socket_path=str(sock)))
    await srv.start()
    srv.abort_sync()
    assert socket_lock_held(sock) is False
    await srv.shutdown()
