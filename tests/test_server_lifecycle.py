"""``TranscriptionServer`` lifecycle state machine and single-server lock.

States: new -> starting -> running -> stopping -> stopped (``state``). Events:
start, shutdown, a concurrent start / shutdown, cancel during start (once or
twice, mid-bind), a session accepted by a start that then fails, cancel during
the drain or the backend close, a hung close, a stuck session task, a readiness
step that ignores cancellation, and a backend decode thread that outlives its
cancelled task. Invariants under test:

1. The lock is never held in new or stopped (``holds_socket_lock`` and the
   kernel view ``socket_lock_held`` agree).
2. ``_transition`` into new / stopped is the only release, refused while the
   live-work predicate (listener, sessions, start/drain/close/bind/readiness
   steps, backend worker thread) is non-empty.
3. ``shutdown()`` is idempotent, safe before ``start()``, concurrent-safe and
   cancellation-safe (every caller waits for the one drain, close included).
   Single-use: once ``shutdown()`` was called, or a failed start left work
   running (stopping), ``start()`` raises; a failed start that returned to new
   may be retried, from a clean attempt (every per-attempt slot reset).
4. A started backend is closed on every exit from ``start()`` short of running;
   every wait on foreign code is bounded, and what outlives it stays in the one
   live-work ledger, marked stuck: ``stuck_task_count`` and the lock are views
   of the same ledger. ``wait_stopped()`` is the public "lock is free" wait.

Every wait is an event or a bounded ``asyncio.wait_for``: no sleeps decide an
outcome, so a regression fails instead of hanging.
"""

from __future__ import annotations

import asyncio
import json
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


class CloseFailsOnce(Gated):
    async def close(self):
        self.closes += 1
        if self.closes == 1:
            raise RuntimeError("first close failed")


@pytest.mark.parametrize("first_close_fails", [False, True])
async def test_retry_after_a_failed_start_closes_the_backend_again(
    sock, monkeypatch, first_close_fails
):
    """Codex HIGH: the failed start's finished ``_close_task`` survived the
    return to ``new``; the retry's shutdown reused it and never closed the
    backend (starts=2, closes=1), or re-raised the first close's failure."""
    b = CloseFailsOnce() if first_close_fails else Gated()
    real_bind = S.ws_unix_serve
    failures = [OSError("bind failed")]

    async def bind(*a, **k):
        if failures:
            raise failures.pop()
        return await real_bind(*a, **k)

    monkeypatch.setattr(S, "ws_unix_serve", bind)
    srv = _server(b, sock)
    with pytest.raises(OSError, match="bind failed"):
        await srv.start()
    assert srv.state == "new" and not _lock_view(srv, sock)
    assert (b.starts, b.closes) == (1, 1)
    await srv.start()
    await asyncio.wait_for(srv.shutdown(), T)  # never the first attempt's close error
    assert (b.starts, b.closes) == (2, 2), "every started backend must be closed"
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_failed_start_resets_every_per_attempt_slot(sock, monkeypatch):
    """Security minor: the bound-socket inode and ``_server`` (and the close
    task, the shutdown event, the session sets) survived a failed start's
    return to ``new``, so a retry's drain could act on attempt 1's inode."""
    monkeypatch.setattr(S, "_CANCEL_GRACE_SECONDS", 0.05)
    srv = _server(Gated(), sock, drain_timeout_seconds=0.05)

    async def handler():
        await asyncio.Event().wait()

    async def accept_a_session(listener):
        srv._track(asyncio.ensure_future(handler()), "session")

    gate, bound, _ = _slow_tail_bind(monkeypatch, accept_a_session)
    task = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(bound.wait(), T)
    task.cancel()  # one cancel: the bind finishes and its listener is adopted
    await asyncio.sleep(0)
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, T)
    assert srv.state == "new" and not _lock_view(srv, sock)
    assert srv._server is None
    assert srv._bound_socket_id is None
    assert srv._close_task is None
    assert srv._stop_task is None
    assert not srv._shutdown_event.is_set()
    assert not srv._live_work
    assert srv.sockets_bound == []


async def test_wait_stopped_is_the_public_lock_free_wait(sock):
    """``shutdown()`` from ``starting`` returns with the lock still held;
    ``wait_stopped()`` returns only once it is free (and at once when it
    already is: ``new``, or a failed start back in ``new``)."""
    b = Gated()
    srv = _server(b, sock)
    await asyncio.wait_for(srv.wait_stopped(), T)  # new: nothing held
    b.gate.clear()
    task = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(b.entered.wait(), T)
    await srv.shutdown()
    assert srv.state == "stopping" and _lock_view(srv, sock)
    waiter = asyncio.ensure_future(srv.wait_stopped())
    await asyncio.sleep(0)
    assert not waiter.done(), "wait_stopped() must not return while the lock is held"
    b.gate.set()
    with pytest.raises(RuntimeError, match="during startup"):
        await asyncio.wait_for(task, T)
    await asyncio.wait_for(waiter, T)
    assert srv.state == "stopped" and not _lock_view(srv, sock)

    b2 = Gated()
    b2.gate.clear()
    srv2 = _server(b2, sock)
    task = asyncio.ensure_future(srv2.start())
    await asyncio.wait_for(b2.entered.wait(), T)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(srv2.wait_stopped(), T)  # back in new: lock free
    assert srv2.state == "new" and not _lock_view(srv2, sock)


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
    srv._track(task, "session")
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
        # The drain was never abandoned: it finishes (closing the backend) and
        # only then releases the lock.
        await asyncio.wait_for(srv.wait_stopped(), T)
        assert srv.state == "stopped" and not _lock_view(srv, sock)
        assert srv.backend.closes == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.wait({task}, timeout=T)


class HangingClose(Gated):
    """``close()`` blocks until ``release`` (counting entries and exits)."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()
        self.closed = 0

    async def close(self):
        self.closes += 1
        await self.release.wait()
        self.closed += 1


async def test_hung_backend_close_is_bounded_and_keeps_the_lock(sock):
    """Codex medium: ``close()`` was awaited with no bound, so a hung close
    held shutdown (and the CLI's hard exit) forever."""
    b = HangingClose()
    srv = _server(b, sock, drain_timeout_seconds=0.1)
    await srv.start()
    await asyncio.wait_for(srv.shutdown(), T)  # returns at the close bound
    assert b.closes == 1 and b.closed == 0
    assert srv.stuck_task_count == 1  # the close: serve's hard exit sees it
    assert srv.state == "stopping" and _lock_view(srv, sock)
    await asyncio.wait_for(srv.shutdown(), T)  # a later caller: same drain
    assert b.closes == 1
    b.release.set()
    await asyncio.wait_for(srv.wait_stopped(), T)
    assert b.closed == 1 and srv.stuck_task_count == 0
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_cancelled_first_shutdown_still_finishes_the_close(sock):
    """A cancelled first ``shutdown()`` marked the stop done: later callers
    returned while ``close()`` had not finished (Metal drain / ``abort()``
    skipped). Now every caller waits for the one server-owned drain."""
    b = HangingClose()
    srv = _server(b, sock)
    await srv.start()
    first = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0.05)
    assert b.closes == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert srv.state == "stopping" and _lock_view(srv, sock)
    second = asyncio.ensure_future(srv.shutdown())
    await asyncio.sleep(0.05)
    assert not second.done(), "a later caller must wait for the unfinished close"
    b.release.set()
    await asyncio.wait_for(second, T)
    assert b.closes == 1 and b.closed == 1
    assert srv.state == "stopped" and not _lock_view(srv, sock)


# --- failed start: the same drain as shutdown -----------------------------------


def _slow_tail_bind(monkeypatch, on_bound):
    """Patch the UDS bind: bind for real, call ``on_bound(listener)``, then
    wait on the returned gate (the await after the synchronous bind)."""
    real_bind = S.ws_unix_serve
    gate, bound, listeners = asyncio.Event(), asyncio.Event(), []

    async def bind(*a, **k):
        listener = await real_bind(*a, **k)
        listeners.append(listener)
        await on_bound(listener)
        bound.set()
        await gate.wait()
        return listener

    monkeypatch.setattr(S, "ws_unix_serve", bind)
    return gate, bound, listeners


async def test_second_cancel_during_bind_never_orphans_the_listener(sock, monkeypatch):
    """Codex high: the cleanup wait after the first cancel was not shielded; a
    second cancel left the bind running, start() went back to ``new`` and
    released the lock, and the bind then returned a listener nobody owned."""

    async def nothing(listener):
        return None

    gate, bound, listeners = _slow_tail_bind(monkeypatch, nothing)
    srv = _server(Gated(), sock)
    task = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(bound.wait(), T)
    task.cancel()
    await asyncio.sleep(0)  # start() is now waiting for the bind to finish
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, T)
    # The bind is still running: it is live work, so the lock stays held, and
    # it outlived its start, so it is stuck (one ledger: the CLI's hard exit
    # sees what holds the lock).
    assert srv.state == "stopping" and _lock_view(srv, sock)
    assert srv.stuck_task_count == 1
    # Security minor: ``Server.close()`` only schedules the listening
    # socket's close; the lock must not be released before it really closed.
    serving_at_release = []
    real_release = srv._release_lock

    def spy_release():
        if srv._lock_fd is not None:
            serving_at_release.append(listeners[0].is_serving())
        real_release()

    srv._release_lock = spy_release
    gate.set()
    await asyncio.wait_for(srv.wait_stopped(), T)
    assert serving_at_release == [False], "lock released while the orphan listener served"
    assert srv.stuck_task_count == 0
    (listener,) = listeners
    assert not listener.is_serving()
    assert not sock.exists()
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_orphan_bind_keeps_the_lock_until_its_cleanup_ran(sock, monkeypatch):
    """Codex high: the ledger dropped a future the moment it was done, before
    its done-callbacks ran. Another entry ending in the same loop tick then
    released the lock (and reset the attempt) before the orphan bind's cleanup
    had closed its listener; the cleanup later unlinked with no lock held.

    Deterministic: the bind finishes a second tracked future right before it
    returns, so both futures' callbacks are queued in one tick, the other
    one's first."""
    real_bind = S.ws_unix_serve
    gate, bound, listeners = asyncio.Event(), asyncio.Event(), []
    other: asyncio.Future = asyncio.get_running_loop().create_future()

    async def bind(*a, **k):
        listener = await real_bind(*a, **k)
        listeners.append(listener)
        bound.set()
        await gate.wait()
        other.set_result(None)  # queued before the bind task's own callbacks
        return listener

    monkeypatch.setattr(S, "ws_unix_serve", bind)
    srv = _server(Gated(), sock)
    task = asyncio.ensure_future(srv.start())
    await asyncio.wait_for(bound.wait(), T)
    task.cancel()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()  # second cancel: the bind is orphaned
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, T)
    srv._linger(other, "close")  # e.g. a backend close past its bound
    assert srv.state == "stopping" and _lock_view(srv, sock)
    at_release = []
    real_release = srv._release_lock

    def spy_release():
        if srv._lock_fd is not None:
            at_release.append((listeners[0].is_serving(), sock.exists()))
        real_release()

    srv._release_lock = spy_release
    gate.set()
    await asyncio.wait_for(srv.wait_stopped(), T)
    assert at_release == [(False, False)], (
        "lock released before the orphan listener was closed and unlinked"
    )
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_every_server_task_is_live_work_until_it_completes(sock):
    """Codex medium: the listener's ``wait_closed`` task (and before it, the
    normal bind task) was created outside the ledger, so the lock could be
    released while websockets' close (and the handshakes it waits for) still
    ran. Every task the server creates must be in ``_live_work`` from creation
    until it completes."""
    import os
    import sys

    from websockets.asyncio.client import unix_connect

    loop = asyncio.get_running_loop()
    asyncio_dir = os.path.dirname(asyncio.__file__)
    created: list[str] = []
    escaped: list[str] = []
    srv = _server(EchoBackend(), sock)

    def factory(loop, coro, **kw):
        task = asyncio.Task(coro, loop=loop, **kw)
        frame = sys._getframe(1)
        while frame is not None and frame.f_code.co_filename.startswith(asyncio_dir):
            frame = frame.f_back
        if frame is not None and frame.f_code.co_filename == S.__file__:
            where = frame.f_code.co_name
            created.append(where)

            def check(t, where=where):  # runs before the ledger's own callback
                if t not in srv._live_work:
                    escaped.append(where)

            task.add_done_callback(check)
        return task

    loop.set_task_factory(factory)
    try:
        await srv.start()
        async with unix_connect(str(sock)) as ws:
            await asyncio.wait_for(ws.recv(), T)  # server.hello
            await ws.send(b"\x00\x00" * 1600)
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            while True:
                ev = json.loads(await asyncio.wait_for(ws.recv(), T))
                if ev["type"] == "conversation.item.input_audio_transcription.completed":
                    break
            await asyncio.wait_for(srv.shutdown(), T)
        await asyncio.wait_for(srv.wait_stopped(), T)
    finally:
        loop.set_task_factory(None)
    # bind, decode, the drain, the listener close and the backend close.
    assert {"_bind", "_on_commit", "shutdown", "_begin_listener_close"} <= set(created), created
    assert escaped == [], f"tasks that left the ledger before completing: {escaped}"
    assert srv.state == "stopped" and not _lock_view(srv, sock)


@pytest.mark.parametrize("trigger", ["shutdown", "cancel"])
@pytest.mark.parametrize("session", ["prompt", "stubborn"])
async def test_failed_start_drains_the_sessions_it_accepted(sock, monkeypatch, trigger, session):
    """Codex high: the listener accepts before ``running``; the failed-start
    unwind closed it but never drained the sessions, so the lock was released
    while one still used the backend."""
    monkeypatch.setattr(S, "_CANCEL_GRACE_SECONDS", 0.05)
    srv = _server(Gated(), sock, drain_timeout_seconds=0.05)
    release, sessions = asyncio.Event(), []

    async def handler():
        while True:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if session == "prompt":
                    raise
                await release.wait()  # outlives the drain's bounds
                raise

    async def accept_a_session(listener):
        t = asyncio.ensure_future(handler())
        sessions.append(t)
        srv._track(t, "session")
        if trigger == "shutdown":
            await srv.shutdown()

    gate, bound, _ = _slow_tail_bind(monkeypatch, accept_a_session)
    if trigger == "shutdown":
        gate.set()
    task = asyncio.ensure_future(srv.start())
    try:
        await asyncio.wait_for(bound.wait(), T)
        if trigger == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            gate.set()
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await asyncio.wait_for(task, T)
        (t,) = sessions
        assert t.cancelling() or t.cancelled(), "the unwind must cancel the session"
        if session == "stubborn":
            assert not t.done() and srv.stuck_task_count == 1
            assert srv.state == "stopping" and _lock_view(srv, sock)
            # Single-use, and the error names the real cause.
            cause = "after shutdown" if trigger == "shutdown" else "failed start"
            with pytest.raises(RuntimeError, match=cause):
                await srv.start()
            release.set()
            await asyncio.wait_for(srv.wait_stopped(), T)
            assert srv.state == "stopped"
        else:
            assert t.done()
            # Nothing left running: a failed start without shutdown may retry.
            assert srv.state == ("stopped" if trigger == "shutdown" else "new")
        assert not _lock_view(srv, sock) and not sock.exists()
    finally:
        release.set()
        for t in sessions:
            t.cancel()
        await asyncio.wait(sessions or {task}, timeout=T)


async def test_release_is_refused_while_live_work_remains(sock):
    """Invariant 3: ``_transition`` is the one place the lock is released, and
    it refuses while the live-work predicate is non-empty."""
    srv = _server(Gated(), sock)
    await srv.start()
    srv._transition(S._State.STOPPING)  # bypass the drain on purpose
    assert srv._live_reasons() == ["listener bound"]
    with pytest.raises(RuntimeError, match="live work remains: listener bound"):
        srv._transition(S._State.STOPPED)
    assert srv.state == "stopping" and _lock_view(srv, sock)
    closer = srv._close_listener()
    # The bound listener is retired only by registering its close: never a
    # moment with neither in the ledger.
    assert srv._live_reasons() == ["1 listener close"]
    assert srv.state == "stopping" and _lock_view(srv, sock)
    await asyncio.wait_for(closer, T)
    await asyncio.wait_for(srv.wait_stopped(), T)
    assert srv.state == "stopped" and not _lock_view(srv, sock)


async def test_a_session_is_live_work_from_creation(sock):
    """Architecture minor: a live session counted only through a side set no
    callback re-settled; it must be in the one ledger from creation, so its
    end alone moves ``stopping`` to ``stopped``."""
    from websockets.asyncio.client import unix_connect

    srv = _server(Gated(), sock)
    await srv.start()
    async with unix_connect(str(sock)) as ws:
        await asyncio.wait_for(ws.recv(), T)  # server.hello: the handler runs
        assert "1 session/decode tasks" in srv._live_reasons()
        srv._transition(S._State.STOPPING)  # bypass the drain on purpose
        srv._close_listener()
        # The listener's close waits for the handler too (websockets).
        assert sorted(srv._live_reasons()) == ["1 listener close", "1 session/decode tasks"]
        assert srv.state == "stopping" and _lock_view(srv, sock)
    await asyncio.wait_for(srv.wait_stopped(), T)  # the session's end settles it
    assert srv.state == "stopped" and not _lock_view(srv, sock)


# --- serve(): a readiness step that ignores cancellation --------------------------


async def test_soft_stop_keeps_an_uncancellable_readiness_step(sock, monkeypatch, caplog):
    """Codex medium: after its bounded wait, a soft stop dropped a readiness
    task that ignored cancellation; serve returned, the lock was released, and
    the task later failed unobserved (and could still touch the backend)."""
    import functools
    import os
    import signal

    monkeypatch.setattr(
        S, "ServerConfig", functools.partial(S.ServerConfig, drain_timeout_seconds=0.05)
    )
    entered, gate, seen = asyncio.Event(), asyncio.Event(), []

    async def ready(server):
        seen.append(server)
        entered.set()
        while not gate.is_set():
            try:
                await gate.wait()
            except asyncio.CancelledError:
                pass  # ignores the stop's cancel
        raise RuntimeError("warm-up failed late")

    serving = asyncio.ensure_future(S.serve(Gated(), socket_path=str(sock), ready=ready))
    try:
        await asyncio.wait_for(entered.wait(), T)
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(serving, T)
        (srv,) = seen
        assert srv.stuck_task_count == 1
        assert srv.state == "stopping" and _lock_view(srv, sock)
        gate.set()
        await asyncio.wait_for(srv.wait_stopped(), T)
        assert not _lock_view(srv, sock)
        assert "warm-up failed late" in caplog.text
    finally:
        gate.set()
        asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
        asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)


async def test_serve_rejects_on_fatal_without_hard_exit():
    b = Gated()
    with pytest.raises(S.ConfigError, match="hard_exit_on_stop"):
        await asyncio.wait_for(
            S.serve(
                b,
                host="127.0.0.1",
                port=0,
                install_signal_handlers=False,
                on_fatal=lambda exc: None,
            ),
            T,
        )
    assert b.aborts == 1 and b.starts == 0


# --- backend worker thread behind a cancelled decode task ----------------------


async def test_backend_decode_thread_keeps_the_lock(sock):
    b = ThreadedDecodes()
    assert isinstance(b, IdleAwareBackend)
    srv = _server(b, sock)
    await srv.start()
    b.idle()  # a decode thread is in flight (its task already cancelled/done)
    await srv.shutdown()
    # One ledger: the thread holds the lock AND counts as stuck, so the CLI's
    # hard exit and the lock agree.
    assert srv.stuck_task_count == 1
    assert srv.state == "stopping" and _lock_view(srv, sock)
    b.thread_done.set_result(None)
    await asyncio.wait_for(srv.wait_stopped(), T)
    assert srv.state == "stopped" and not _lock_view(srv, sock)
    assert srv.stuck_task_count == 0


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
            server._track(asyncio.ensure_future(stubborn()), "session")

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
