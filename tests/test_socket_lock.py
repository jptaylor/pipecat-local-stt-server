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


async def test_abort_sync_keeps_lock_until_process_exit(short_home):
    """``abort_sync`` precedes ``os._exit``; the kernel drops the lock with the
    process. Releasing it early would let a new server start loading while this
    one (and its model thread) still exists."""
    sock = short_home / "s" / "stt.sock"
    srv = TranscriptionServer(EchoBackend(), ServerConfig(socket_path=str(sock)))
    await srv.start()
    srv.abort_sync()
    assert socket_lock_held(sock) is True
    await srv.shutdown()
    assert socket_lock_held(sock) is False


def _hold(sock, kind):
    import fcntl

    fd = os.open(f"{sock}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, kind)
    return fd


@pytest.mark.parametrize("holder", [None, "shared", "exclusive"])
def test_probe_state_space(tmp_path, holder):
    """Probe vs. {no holder, another probe (shared), a server (exclusive)}: only
    a server reads as held; a concurrent probe does not."""
    import fcntl

    sock = tmp_path / "a.sock"
    os.close(acquire_socket_lock(sock))  # lock file exists, unheld
    fd = None
    if holder == "shared":
        fd = _hold(sock, fcntl.LOCK_SH)
    elif holder == "exclusive":
        fd = _hold(sock, fcntl.LOCK_EX)
    try:
        assert socket_lock_held(sock) is (holder == "exclusive")
    finally:
        if fd is not None:
            os.close(fd)


def test_acquire_survives_a_concurrent_probe(tmp_path):
    """A preflight probe briefly holds the lock (shared). A real starter racing
    it must not fail with EADDRINUSE when no server is running."""
    import fcntl
    import threading

    sock = tmp_path / "a.sock"
    os.close(acquire_socket_lock(sock))
    fd = _hold(sock, fcntl.LOCK_SH)  # the probe, mid-check
    threading.Timer(0.05, os.close, args=(fd,)).start()
    os.close(acquire_socket_lock(sock))


@pytest.mark.parametrize("mode", [0o666, 0o644, 0o620, 0o604])
def test_owned_lock_file_with_unsafe_mode_is_tightened(tmp_path, mode):
    sock = tmp_path / "a.sock"
    lock = Path(f"{sock}.lock")
    lock.write_text("")
    os.chmod(lock, mode)
    os.close(acquire_socket_lock(sock))
    assert (lock.stat().st_mode & 0o777) == 0o600


class _AbortCounting(EchoBackend):
    def __init__(self, start_exc=None):
        self.aborts = 0
        self.closes = 0
        self._start_exc = start_exc

    async def start(self):
        if self._start_exc is not None:
            raise self._start_exc

    def abort(self):
        self.aborts += 1

    async def close(self):
        self.closes += 1


async def test_failed_start_runs_backend_abort_in_every_failure_cell(short_home):
    """``close()`` never runs for a server that never started, so every failed
    start must run the backend's ``abort()`` (parakeet/nemotron temp dir)."""
    sock = short_home / "s" / "stt.sock"

    # Cell 1: the lock is held by a live server (fast EADDRINUSE path).
    first = TranscriptionServer(EchoBackend(), ServerConfig(socket_path=str(sock)))
    await first.start()
    try:
        b = _AbortCounting()
        with pytest.raises(OSError):
            await TranscriptionServer(b, ServerConfig(socket_path=str(sock))).start()
        assert b.aborts == 1
    finally:
        await first.shutdown()

    # Cell 2: backend.start() fails (e.g. missing extra).
    b = _AbortCounting(start_exc=ModuleNotFoundError("no extra"))
    srv = TranscriptionServer(b, ServerConfig(socket_path=str(sock)))
    with pytest.raises(ModuleNotFoundError):
        await srv.start()
    assert b.aborts == 1 and socket_lock_held(sock) is False

    # Cell 3: the socket directory fails enforcement (outside the trusted root).
    b = _AbortCounting()
    with pytest.raises(ValueError):
        await TranscriptionServer(b, ServerConfig(socket_path="/tmp/x/stt.sock")).start()
    assert b.aborts == 1

    # Cell 4: the TCP bind fails (port already taken).
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    try:
        b = _AbortCounting()
        port = taken.getsockname()[1]
        with pytest.raises(OSError):
            await TranscriptionServer(b, ServerConfig(host="127.0.0.1", port=port)).start()
        assert b.aborts == 1
    finally:
        taken.close()

    # Control: a successful start does not abort; shutdown closes.
    b = _AbortCounting()
    srv = TranscriptionServer(b, ServerConfig(socket_path=str(sock)))
    await srv.start()
    assert b.aborts == 0
    await srv.shutdown()
    assert b.closes == 1


async def test_serve_bad_config_aborts_backend():
    from stt_server.server import serve

    b = _AbortCounting()
    with pytest.raises(ValueError):
        await serve(b, host="10.0.0.1", port=1, install_signal_handlers=False)
    assert b.aborts == 1


async def test_failed_backend_close_still_releases_lock(short_home):
    sock = short_home / "s" / "stt.sock"

    class BadClose(EchoBackend):
        async def close(self):
            raise RuntimeError("metal teardown failed")

    srv = TranscriptionServer(BadClose(), ServerConfig(socket_path=str(sock)))
    await srv.start()
    with pytest.raises(RuntimeError):
        await srv.shutdown()
    assert socket_lock_held(sock) is False
    assert srv._started is False
    await srv.shutdown()  # a second call returns instead of spinning


async def test_cancelled_shutdown_still_releases_lock(short_home):
    import asyncio

    sock = short_home / "s" / "stt.sock"
    gate = asyncio.Event()

    class SlowClose(EchoBackend):
        async def close(self):
            await gate.wait()

    srv = TranscriptionServer(SlowClose(), ServerConfig(socket_path=str(sock)))
    await srv.start()
    task = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.wait({task})
    assert socket_lock_held(sock) is False and srv._started is False


async def test_shutdown_is_bounded_when_a_task_swallows_cancellation(monkeypatch):
    """``wait_for(gather(...))`` waited for cancellation to finish; a session
    task that swallows ``CancelledError`` must not hold shutdown forever."""
    import asyncio

    from stt_server import server as S

    monkeypatch.setattr(S, "_CANCEL_GRACE_SECONDS", 0.2)
    cfg = ServerConfig(host="127.0.0.1", port=0, drain_timeout_seconds=0.1)
    srv = TranscriptionServer(EchoBackend(), cfg)
    await srv.start()
    released = False

    async def stubborn():
        while True:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                if released:
                    raise

    task = asyncio.ensure_future(stubborn())
    srv._active_handlers.add(task)
    try:
        await asyncio.wait_for(srv.shutdown(), timeout=3)
    finally:
        released = True
        task.cancel()
        await asyncio.wait({task}, timeout=1)
