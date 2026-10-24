"""``serve --supervised``: the contract a managing app relies on (echo backend, lean).

- stdout carries exactly one ``{"event": "ready", ...}`` JSON line once the
  listener is bound and warm, and nothing else (logs stay on stderr);
- closing stdin stops the server cleanly (exit 0), like SIGTERM;
- a busy TCP port or a live Unix socket fails fast with ``code: EADDRINUSE``;
- a missing backend extra fails with ``code: ModuleNotFoundError``;
- without ``--supervised`` stdout stays empty (behaviour unchanged).
"""

from __future__ import annotations

import errno
import json
import os
import select
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from stt_server import supervise
from stt_server import __main__ as cli


def _env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("STT_WS_", "PIPECAT_STT_", "KODA_STT_"))
    }
    env.update(extra)
    return env


def _serve(*args: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "stt_server", "serve", *args, "--supervised"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env or _env(),
    )


def _read_event(proc: subprocess.Popen, timeout: float = 20.0) -> dict:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], timeout)
    if not ready:
        proc.kill()
        raise AssertionError("no stdout line in time; stderr:\n" + proc.stderr.read().decode())
    return json.loads(proc.stdout.readline())


def _finish(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def test_ready_line_then_stdin_eof_stops():
    proc = _serve("--backend", "echo", "--host", "127.0.0.1", "--port", "0")
    try:
        event = _read_event(proc)
        assert event["event"] == "ready"
        assert event["backend"] == "echo" and event["model"] is None
        assert event["pid"] == proc.pid
        assert event["rate"] == 16000
        assert event["host"] == "127.0.0.1" and event["socket_path"] is None
        port = event["port"]
        assert isinstance(port, int) and port > 0  # --port 0 reports the real port
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            pass
        proc.stdin.close()  # the supervisor goes away
        assert proc.wait(timeout=15) == 0
        assert proc.stdout.read() == b""  # nothing after the ready line
    finally:
        _finish(proc)


@pytest.mark.parametrize("stop_kind", ["stdin_eof", "sigterm"])
def test_supervised_stop_during_blocked_warmup_exits_without_ready(stop_kind):
    script = textwrap.dedent(
        """
        import asyncio
        import os
        import threading

        from stt_server.backend import EchoBackend
        from stt_server.server import TranscriptionServer, serve

        async def start_without_listener(self):
            return None

        async def shutdown_without_listener(self):
            return None

        TranscriptionServer.start = start_without_listener
        TranscriptionServer.shutdown = shutdown_without_listener

        async def blocked_ready(server):
            os.write(2, b"warmup-blocked\\n")
            await asyncio.to_thread(threading.Event().wait)
            os.write(1, b'{"event":"ready"}\\n')

        asyncio.run(
            serve(
                EchoBackend(),
                host="127.0.0.1",
                port=0,
                ready=blocked_ready,
                stop_on_stdin_eof=True,
            )
        )
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )
    try:
        assert proc.stderr is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            ready, _, _ = select.select([proc.stderr], [], [], 0.25)
            if ready and proc.stderr.readline() == b"warmup-blocked\n":
                break
        else:
            raise AssertionError("warm-up did not block in time")

        started = time.monotonic()
        if stop_kind == "stdin_eof":
            proc.stdin.close()
        else:
            proc.terminate()
        assert proc.wait(timeout=3) == 0
        assert time.monotonic() - started < 3
        assert proc.stdout.read() == b""
    finally:
        _finish(proc)


@pytest.fixture
def short_home():
    """A private, short $HOME so the socket-dir ancestor walk passes and the
    socket path stays under the AF_UNIX length limit."""
    home = Path(tempfile.mkdtemp(prefix="stt.", dir="/tmp"))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def test_ready_line_reports_socket_path(short_home):
    sock = short_home / "s" / "stt.sock"
    proc = _serve("--backend", "echo", "--socket-path", str(sock), env=_env(HOME=str(short_home)))
    try:
        event = _read_event(proc)
        assert event["event"] == "ready"
        assert event["socket_path"] == str(sock)
        assert event["port"] is None and event["host"] is None
        proc.stdin.close()
        assert proc.wait(timeout=15) == 0
    finally:
        _finish(proc)


def test_live_unix_socket_reports_eaddrinuse(short_home):
    sock = short_home / "s" / "stt.sock"
    env = _env(HOME=str(short_home))
    first = _serve("--backend", "echo", "--socket-path", str(sock), env=env)
    try:
        assert _read_event(first)["event"] == "ready"
        second = _serve("--backend", "echo", "--socket-path", str(sock), env=env)
        try:
            event = _read_event(second)
            assert second.wait(timeout=15) == 1
        finally:
            _finish(second)
        assert event["event"] == "error" and event["code"] == "EADDRINUSE"
        # The first server still owns its socket (it was not replaced).
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.connect(str(sock))
    finally:
        _finish(first)


@pytest.mark.parametrize(
    "probe_error",
    [TimeoutError("timed out"), BlockingIOError(errno.EAGAIN, "backlog full")],
)
def test_unix_socket_ambiguous_probe_error_preserves_inode(short_home, monkeypatch, probe_error):
    path = short_home / "listener.sock"
    path.write_bytes(b"socket inode sentinel")
    inode = path.stat().st_ino
    real_is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda self: self == path or real_is_socket(self))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise probe_error

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: Probe())
    with pytest.raises(OSError) as exc:
        cli._preflight_listener(SimpleNamespace(socket_path=str(path), host=None, port=None))

    assert exc.value.errno == errno.EADDRINUSE
    assert path.stat().st_ino == inode


@pytest.mark.parametrize("stale_errno", [errno.ECONNREFUSED, errno.ENOENT])
def test_unix_socket_narrow_stale_probe_errors_are_allowed(short_home, monkeypatch, stale_errno):
    path = short_home / "listener.sock"
    path.write_bytes(b"socket inode sentinel")
    real_is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda self: self == path or real_is_socket(self))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise OSError(stale_errno, "stale")

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: Probe())
    cli._preflight_listener(SimpleNamespace(socket_path=str(path), host=None, port=None))


@pytest.mark.parametrize("fatal_errno", [errno.EACCES, errno.EPERM, errno.EIO])
def test_unix_socket_fatal_probe_error_preserves_inode(short_home, monkeypatch, fatal_errno):
    path = short_home / "listener.sock"
    path.write_bytes(b"socket inode sentinel")
    inode = path.stat().st_ino
    real_is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda self: self == path or real_is_socket(self))

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def settimeout(self, _timeout):
            pass

        def connect(self, _path):
            raise OSError(fatal_errno, "fatal")

    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: Probe())
    with pytest.raises(OSError) as exc:
        cli._preflight_listener(SimpleNamespace(socket_path=str(path), host=None, port=None))

    assert exc.value.errno == fatal_errno
    assert path.stat().st_ino == inode


def test_busy_tcp_port_reports_eaddrinuse_fast():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        started = time.monotonic()
        proc = _serve("--backend", "echo", "--host", "127.0.0.1", "--port", str(port))
        try:
            event = _read_event(proc)
            assert proc.wait(timeout=15) == 1
        finally:
            _finish(proc)
        assert event["event"] == "error" and event["code"] == "EADDRINUSE"
        assert time.monotonic() - started < 10


def test_missing_extra_reports_module_not_found(tmp_path):
    # A stand-in ``moshi_mlx`` that fails like an absent extra, whatever is
    # installed in this environment.
    fake = tmp_path / "moshi_mlx"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'moshi_mlx'\", name='moshi_mlx')\n"
    )
    proc = _serve(
        "--backend",
        "kyutai",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        env=_env(PYTHONPATH=str(tmp_path)),
    )
    try:
        event = _read_event(proc, timeout=60)
        assert proc.wait(timeout=15) == 1
    finally:
        _finish(proc)
    assert event["event"] == "error"
    assert event["code"] == "ModuleNotFoundError"
    assert "uv sync --extra kyutai" in event["message"]


def test_unsupervised_stdout_stays_empty():
    proc = subprocess.Popen(
        [sys.executable, "-m", "stt_server", "serve", "--backend", "echo"]
        + ["--host", "127.0.0.1", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )
    try:
        time.sleep(1.5)
        proc.terminate()
        out, _ = proc.communicate(timeout=15)
    finally:
        _finish(proc)
    assert out == b""


class _HubError(Exception):
    pass


_HubError.__module__ = "huggingface_hub.errors"


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (
            ModuleNotFoundError("No module named 'moshi_mlx'", name="moshi_mlx"),
            "ModuleNotFoundError",
        ),
        (OSError(errno.EADDRINUSE, "Address already in use"), "EADDRINUSE"),
        (OSError(errno.EACCES, "Permission denied"), "EACCES"),
        (_HubError("404"), "ModelDownloadError"),
        (ConnectionResetError("reset"), "ModelDownloadError"),
        (ValueError("V1 only permits loopback binds"), "ConfigError"),
        (RuntimeError("decoder exploded"), "BackendStartError"),
    ],
)
def test_classify_startup_error(exc, code):
    assert supervise.classify_startup_error(exc)[0] == code


def test_classify_follows_cause_chain():
    try:
        try:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        except OSError as inner:
            raise RuntimeError("bind failed") from inner
    except RuntimeError as outer:
        assert supervise.classify_startup_error(outer)[0] == "EADDRINUSE"


def test_json_channel_routes_prints_to_stderr(capsys):
    with supervise.json_channel():
        print("library noise")
        supervise.emit({"event": "x", "n": 1})
    out, err = capsys.readouterr()
    assert out == '{"event":"x","n":1}\n'
    assert "library noise" in err
