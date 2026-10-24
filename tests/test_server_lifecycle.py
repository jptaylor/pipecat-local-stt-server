"""``TranscriptionServer`` lifecycle state machine and single-server lock.

States: new -> starting -> running -> stopping -> stopped (``state``). Events:
start, shutdown, a concurrent start / shutdown, cancel during start, cancel
during the drain, a stuck session task, and a backend decode thread that
outlives its cancelled task. Invariants under test:

1. The lock is held iff the state is starting, running, or stopping with live
   work (``holds_socket_lock`` and the kernel view ``socket_lock_held`` agree).
2. It is released once, only when no start, drain, session task or backend
   worker thread can still run.
3. ``shutdown()`` is idempotent, safe before ``start()``, concurrent-safe and
   cancellation-safe; a shut-down server is single-use (``start()`` raises).
4. A started backend is closed on every exit from ``start()`` short of running.

Every wait is an event or a bounded ``asyncio.wait_for``: no sleeps decide an
outcome, so a regression fails instead of hanging.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import pytest

from stt_server import server as S
from stt_server.backend import EchoBackend, IdleAwareBackend
from stt_server.server import ServerConfig, TranscriptionServer, socket_lock_held

T = 5  # seconds: upper bound on any single wait in this file


@pytest.fixture
def sock(monkeypatch):
    home = Path(tempfile.mkdtemp(prefix="stt.", dir="/tmp"))
    monkeypatch.setenv("HOME", str(home))
    try:
        yield home / "s" / "stt.sock"
    finally:
        shutil.rmtree(home, ignore_errors=True)


class Gated(EchoBackend):
    """``start()`` blocks on ``gate``; counts close/abort; optional ``idle()``."""

    def __init__(self):
        self.entered = asyncio.Event()
        self.gate = asyncio.Event()
        self.gate.set()
        self.closes = 0
        self.aborts = 0
        self.starts = 0

    async def start(self):
        self.starts += 1
        self.entered.set()
        await self.gate.wait()

    async def close(self):
        self.closes += 1

    def abort(self):
        self.aborts += 1


class ThreadedDecodes(Gated):
    """A backend whose decode thread outlives its task (the ``idle()`` hook)."""

    def __init__(self):
        super().__init__()
        self.thread_done: asyncio.Future | None = None

    def idle(self):
        if self.thread_done is None:
            self.thread_done = asyncio.get_running_loop().create_future()
        return self.thread_done


def _lock_view(srv: TranscriptionServer, sock: Path) -> bool:
    """The server's own view and the kernel's must agree (invariant 1)."""
    held = socket_lock_held(sock)
    assert srv.holds_socket_lock is held
    return held


def _server(backend, sock, **cfg) -> TranscriptionServer:
    return TranscriptionServer(backend, ServerConfig(socket_path=str(sock), **cfg))


# --- shutdown before start / single-use ---------------------------------------


async def test_shutdown_before_start_is_terminal_and_never_locks(sock):
    b = Gated()
    srv = _server(b, sock)
    await srv.shutdown()
    assert srv.state == "stopped" and not _lock_view(srv, sock)
    await srv.shutdown()  # idempotent
    with pytest.raises(RuntimeError, match="single-use"):
        await srv.start()
    assert b.starts == 0 and not sock.exists()


async def test_start_after_shutdown_is_refused(sock):
    srv = _server(Gated(), sock)
    await srv.start()
    await srv.shutdown()
    with pytest.raises(RuntimeError, match="single-use"):
        await srv.start()
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_failed_start_is_retryable(sock):
    """A failed start returns to ``new`` (lock released); a retry may run."""
    b = Gated()
    b.gate.clear()
    srv = _server(b, sock)
    task = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(b.entered.wait(), T)
    task.cancel()  # cancel during start, no shutdown
    with pytest.raises(asyncio.CancelledError):
        await task
    assert srv.state == "new" and not _lock_view(srv, sock)
    assert b.closes == 0 and b.aborts == 1  # backend.start() never returned
    b.gate.set()
    await srv.start()
    assert srv.state == "running" and _lock_view(srv, sock)
    await srv.shutdown()
    assert not _lock_view(srv, sock)


# --- concurrent start ----------------------------------------------------------


async def test_concurrent_start_never_releases_the_active_starters_lock(sock):
    b = Gated()
    b.gate.clear()
    srv = _server(b, sock)
    first = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(b.entered.wait(), T)
    assert srv.state == "starting" and _lock_view(srv, sock)
    with pytest.raises(RuntimeError, match="already in progress"):
        await srv.start()
    assert srv.state == "starting" and _lock_view(srv, sock)  # still the first's
    b.gate.set()
    await asyncio.wait_for(first, T)
    assert srv.state == "running" and _lock_view(srv, sock) and sock.exists()
    await srv.start()  # running: a no-op
    await srv.shutdown()
    assert srv.state == "stopped" and not _lock_view(srv, sock)


# --- shutdown during start -----------------------------------------------------


@pytest.mark.parametrize("where", ["backend_start", "bind"])
async def test_shutdown_during_start_closes_a_started_backend(sock, monkeypatch, where):
    b = Gated()
    srv = _server(b, sock)
    if where == "backend_start":
        b.gate.clear()
    else:
        real_bind = S.ws_unix_serve

        async def bind_then_shutdown(*a, **k):
            listener = await real_bind(*a, **k)
            await srv.shutdown()
            return listener

        monkeypatch.setattr(S, "ws_unix_serve", bind_then_shutdown)
    task = asyncio.ensure_future(srv.start())
    if where == "backend_start":
        await asyncio.wait_for(b.entered.wait(), T)
        await srv.shutdown()
        assert srv.state == "stopping" and _lock_view(srv, sock)  # the start owns it
        await srv.shutdown()  # a concurrent / repeated shutdown returns
        b.gate.set()
    with pytest.raises(RuntimeError, match="during startup"):
        await asyncio.wait_for(task, T)
    assert b.closes == 1, "a started backend must be closed when start() unwinds"
    assert b.aborts == 1
    assert srv.state == "stopped" and not _lock_view(srv, sock)
    assert srv.sockets_bound == []
    assert not sock.exists()


# --- concurrent / cancelled shutdown -------------------------------------------


async def test_concurrent_shutdown_waits_for_the_first_drain(sock):
    gate = asyncio.Event()

    class SlowClose(Gated):
        async def close(self):
            self.closes += 1
            await gate.wait()

    b = SlowClose()
    srv = _server(b, sock)
    await srv.start()
    first = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0)
    second = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0)
    assert not second.done() and srv.state == "stopping" and _lock_view(srv, sock)
    gate.set()
    await asyncio.wait_for(asyncio.gather(first, second), T)
    assert b.closes == 1 and srv.state == "stopped" and not _lock_view(srv, sock)


async def test_cancel_during_drain_keeps_the_lock_while_sessions_run(sock):
    """Codex HIGH: a cancel in the first drain wait released the lock while
    session tasks that were never cancelled kept running."""
    srv = _server(Gated(), sock, drain_timeout_seconds=60)
    await srv.start()
    entered, release = asyncio.Event(), asyncio.Event()

    async def session():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()  # unwinds slowly (sends a close frame, ...)
            raise

    task = asyncio.ensure_future(session())
    srv._active_handlers.add(task)
    await asyncio.wait_for(entered.wait(), T)
    try:
        stopping = asyncio.ensure_future(srv.shutdown())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert srv.state == "stopping" and not task.done()
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert task.cancelling(), "an abandoned drain must cancel the sessions it left"
        assert not task.done() and srv.stuck_task_count == 1
        assert srv.state == "stopping" and _lock_view(srv, sock)
        release.set()
        await asyncio.wait({task}, timeout=T)
        await asyncio.sleep(0)  # done callbacks
        assert srv.state == "stopped" and not _lock_view(srv, sock)
    finally:
        release.set()
        task.cancel()
        await asyncio.wait({task}, timeout=T)


async def test_cancel_during_backend_close_releases_when_nothing_runs(sock):
    class HangingClose(Gated):
        async def close(self):
            await asyncio.Event().wait()

    srv = _server(HangingClose(), sock)
    await srv.start()
    stopping = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0.05)
    stopping.cancel()
    await asyncio.wait({stopping}, timeout=T)
    assert srv.state == "stopped" and not _lock_view(srv, sock)


# --- backend worker thread behind a cancelled decode task ----------------------


async def test_backend_decode_thread_keeps_the_lock(sock):
    b = ThreadedDecodes()
    assert isinstance(b, IdleAwareBackend)
    srv = _server(b, sock)
    await srv.start()
    b.idle()  # a decode thread is in flight (its task already cancelled/done)
    await srv.shutdown()
    assert srv.stuck_task_count == 0  # no asyncio task is left ...
    assert srv.state == "stopping" and _lock_view(srv, sock)  # ... but the thread is
    b.thread_done.set_result(None)
    await asyncio.sleep(0)
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_decode_guard_idle_tracks_the_thread():
    """``DecodeGuardMixin.idle()``: done at once when idle; otherwise completes
    when the in-flight count returns to zero (from the decode thread)."""
    import threading

    from stt_server.backends._thread_util import DecodeGuardMixin

    class G(DecodeGuardMixin):
        def __init__(self):
            self._init_decode_guards()

    g = G()
    assert g.idle().done()
    g._mark_inflight_start()
    fut = g.idle()
    await asyncio.sleep(0.05)
    assert not fut.done()
    threading.Thread(target=g._mark_inflight_end).start()
    await asyncio.wait_for(fut, T)


# --- state x event matrix -------------------------------------------------------


async def _drive(state: str, sock: Path):
    """A server (and its gated backend) parked in ``state``."""
    b = Gated()
    srv = _server(b, sock)
    pending = None
    if state == "starting":
        b.gate.clear()
        pending = asyncio.ensure_future(srv.start())
        await asyncio.wait_for(b.entered.wait(), T)
    elif state == "running":
        await srv.start()
    elif state == "stopping":
        b = ThreadedDecodes()
        srv = _server(b, sock)
        await srv.start()
        b.idle()
        await srv.shutdown()
    elif state == "stopped":
        await srv.start()
        await srv.shutdown()
    assert srv.state == state
    return srv, b, pending


async def _settle(srv, b, pending):
    b.gate.set()
    if isinstance(b, ThreadedDecodes) and b.thread_done and not b.thread_done.done():
        b.thread_done.set_result(None)
    if pending is not None:
        await asyncio.wait({pending}, timeout=T)
        if not pending.cancelled():
            pending.exception()
    await srv.shutdown()
    await asyncio.sleep(0)


# expected: (outcome of start(), state after) / (state after shutdown()).
_START = {
    "new": (None, "running"),
    "starting": ("already in progress", "starting"),
    "running": (None, "running"),
    "stopping": ("single-use", "stopping"),
    "stopped": ("single-use", "stopped"),
}
_SHUTDOWN = {
    "new": "stopped",
    "starting": "stopping",  # the in-flight start unwinds and finishes the stop
    "running": "stopped",
    "stopping": "stopping",  # still waiting on the backend thread
    "stopped": "stopped",
}
_LOCKED = {"new": False, "starting": True, "running": True, "stopping": True, "stopped": False}


@pytest.mark.parametrize("state", list(_START))
async def test_matrix_start(sock, state):
    srv, b, pending = await _drive(state, sock)
    try:
        assert _lock_view(srv, sock) is _LOCKED[state]
        err, after = _START[state]
        if err is None:
            await asyncio.wait_for(srv.start(), T)
        else:
            with pytest.raises(RuntimeError, match=err):
                await asyncio.wait_for(srv.start(), T)
        assert srv.state == after
        assert _lock_view(srv, sock) is _LOCKED[after]
    finally:
        await _settle(srv, b, pending)
    assert srv.state == "stopped" and not _lock_view(srv, sock)


@pytest.mark.parametrize("state", list(_SHUTDOWN))
async def test_matrix_shutdown(sock, state):
    srv, b, pending = await _drive(state, sock)
    try:
        await asyncio.wait_for(srv.shutdown(), T)
        assert srv.state == _SHUTDOWN[state]
        assert _lock_view(srv, sock) is _LOCKED[_SHUTDOWN[state]]
    finally:
        await _settle(srv, b, pending)
    assert srv.state == "stopped" and not _lock_view(srv, sock)
    assert not sock.exists()


# --- the CLI's stuck-task hard exit must not swallow a failure ------------------


def test_cli_stuck_task_exit_still_reports_the_failure():
    """``serve(hard_exit_on_stop=True)`` ``os._exit``s when a session task
    outlives shutdown, skipping ``_cmd_serve``'s handlers: a failing
    ``backend.close()`` must still reach stderr and the supervised ``error``
    event, with exit status 1."""
    import json
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import asyncio, functools, sys
        from stt_server import __main__ as cli, server as S
        from stt_server.backend import EchoBackend

        S._CANCEL_GRACE_SECONDS = 0.1
        S.ServerConfig = functools.partial(S.ServerConfig, drain_timeout_seconds=0.1)

        class BadClose(EchoBackend):
            async def close(self):
                raise RuntimeError("metal teardown failed")

        async def stubborn():
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    pass

        async def ready(server):
            server._active_handlers.add(asyncio.ensure_future(stubborn()))

        cli._make_backend = lambda name, model: BadClose()
        cli._prepare_ready = ready
        sys.argv = ["stt_server", "serve", "--supervised", "--host", "127.0.0.1", "--port", "0"]
        cli.main()
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        import select

        assert proc.stdout is not None and proc.stdin is not None
        readable, _, _ = select.select([proc.stdout], [], [], 20)
        assert readable
        assert json.loads(proc.stdout.readline())["event"] == "ready"
        proc.stdin.close()  # supervisor gone -> stop
        proc.wait(timeout=20)
        out, err = proc.stdout.read(), proc.stderr.read()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 1
    assert b"outlived shutdown" in err
    assert b"metal teardown failed" in err
    events = [json.loads(line) for line in out.splitlines() if line.startswith(b"{")]
    assert events and events[-1]["event"] == "error"
    assert events[-1]["code"] == "BackendStartError"
    assert "metal teardown failed" in events[-1]["message"]
