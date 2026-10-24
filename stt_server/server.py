"""WebSocket server runtime for the local transcription service.

Lifecycle summary:

- TCP or Unix-socket listener, accepting one WebSocket per transcription
  session.
- On connect: send ``server.hello``, mint ``session_id``, send
  ``session.created``.
- Text messages parsed as JSON control events.
- Binary messages treated as raw PCM16LE audio append (V1 default path).
- ``input_audio_buffer.commit`` drains the uncommitted buffer and runs one
  backend decode to completion, streaming delta/completed events back.
- ``session.close`` drains any in-flight decode, sends ``session.closed``,
  then closes the socket. ``session.cancel`` discards uncommitted audio,
  cancels the in-flight decode, sends ``session.closed`` and closes.
- ``serve()`` installs SIGINT/SIGTERM handlers for a bounded graceful drain
  that covers both connection handlers and their decode tasks.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import resource
import signal
import socket
import stat as _stat
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets
from websockets.asyncio.server import (
    Server,
    ServerConnection,
)
from websockets.asyncio.server import (
    serve as ws_serve,
)
from websockets.asyncio.server import (
    unix_serve as ws_unix_serve,
)

from . import protocol as P
from ._peercred import peer_uid
from .backend import BackendStream, EchoBackend, TranscriptionBackend, abort_backend
from .supervise import ConfigError

# OpenAI Realtime groups errors into a coarse "type" field (e.g.
# invalid_request_error, server_error, authentication_error) alongside the
# narrower "code". Map our ErrorCode enum into that taxonomy so strict
# OpenAI-shaped clients can branch on error.type without parsing strings.
_ERROR_TYPE_FOR_CODE: dict[P.ErrorCode, str] = {
    P.ErrorCode.INVALID_JSON: "invalid_request_error",
    P.ErrorCode.INVALID_EVENT: "invalid_request_error",
    P.ErrorCode.UNSUPPORTED_EVENT: "invalid_request_error",
    P.ErrorCode.INVALID_CONFIG: "invalid_request_error",
    P.ErrorCode.BUFFER_EMPTY: "invalid_request_error",
    P.ErrorCode.BUFFER_OVERFLOW: "invalid_request_error",
    P.ErrorCode.PAYLOAD_TOO_LARGE: "invalid_request_error",
    P.ErrorCode.UNAUTHORIZED: "authentication_error",
    P.ErrorCode.BACKEND_ERROR: "server_error",
    P.ErrorCode.INTERNAL_ERROR: "server_error",
}

logger = logging.getLogger("stt_server")


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


def _process_peak_rss_bytes() -> int:
    # ru_maxrss units differ by platform: bytes on macOS, kibibytes on
    # Linux/BSD. Normalize so the wire value is always bytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _session_id() -> str:
    return f"session_{uuid.uuid4().hex[:16]}"


def _item_id() -> str:
    return f"item_{uuid.uuid4().hex[:16]}"


def _enforce_socket_dir_secure(path: Path, trusted_root: Path) -> Path:
    """Refuse to bind unless the socket's directory chain cannot be hijacked.

    The UDS file mode (``0o600``) stops a foreign uid from ``connect()``-ing,
    but it does NOT stop a plant/swap: anyone with write on an ancestor
    directory can ``unlink()`` our socket and ``bind()`` their own at the same
    path, after which a client connects to the attacker's socket. The only
    defense is denying others write on every directory that could replace the
    socket. We therefore walk every component from the socket's bind directory
    up to AND INCLUDING ``trusted_root`` and require each to be owned by the
    running uid (``st_uid == os.geteuid()``) and not group/other-writable
    (``st_mode & 0o022 == 0``); sticky-bit directories are the explicit
    exception (the sticky bit already prevents non-owners from removing our
    inode, which is the property we need).

    Missing socket directories are created ``0o700`` and then re-``stat``-ed and
    verified rather than trusted: the ``mkdir`` mode is masked by the process
    umask, so we confirm the result instead of assuming it. A pre-existing
    directory we do not own is never silently ``chmod``/``chown``-ed — we raise
    instead, naming the offending component and condition.

    Raises ``ValueError`` on any failure; the serve entrypoint turns that into
    ``stt_server: <msg>`` + ``SystemExit(1)`` rather than a bare traceback.

    Returns the verified socket path (the literal ``path``). We verify the
    LITERAL lexical chain that clients and the LaunchAgent actually traverse —
    NOT a ``resolve()``-d one — and reject any symlink component (``lstat``).
    The kernel resolves the literal socket path at connect/bind time, so the
    directories that could replace the socket are the literal lexical ancestors.
    Resolving here would verify a *different* chain than clients use: a symlinked
    but group/other-writable lexical ancestor would be skipped, then repointed
    after startup to make clients connect to an attacker's socket while the
    server stayed bound to the safe resolved target. By rejecting symlinks the
    verified chain, the bound path, and the client-visible path are identical.
    """
    euid = os.geteuid()
    bind_dir = path.parent

    # Verify (and bind) the literal path, so we never resolve past a symlink the
    # client will still traverse. An absolute, ``..``-free path is required: with
    # no ``resolve()`` the containment check below is purely lexical, and a
    # ``..`` segment could let a path that escapes the trusted root pass it.
    if not bind_dir.is_absolute():
        raise ConfigError(
            f"socket directory {bind_dir} must be an absolute path under {trusted_root}"
        )
    if ".." in bind_dir.parts:
        raise ConfigError(
            f"socket directory {bind_dir} must not contain '..' components; "
            f"point the socket at a direct path under {trusted_root}"
        )

    # The socket MUST live under the trusted root. A path outside it (e.g. a
    # custom socket path pointing at a world-writable location) has no
    # owner-only chain we can vouch for, and walking to the filesystem root
    # would only verify root-owned system directories that we do not own. Refuse
    # rather than pretend the path is safe.
    if bind_dir != trusted_root and trusted_root not in bind_dir.parents:
        raise ConfigError(
            f"socket directory {bind_dir} is not under the trusted root "
            f"{trusted_root}; point the socket at a path under {trusted_root}"
        )

    # Create every missing component owner-only. Each is created with its own
    # ``mkdir(mode=0o700)`` rather than ``parents=True``: the latter creates
    # intermediate parents at the umask default (not ``mode``), which would
    # leave a group/other-readable directory in the chain. ``mkdir``'s mode is
    # itself umask-masked, but ``0o700`` carries no group/other bits for any
    # umask to widen, and the walk below re-stats and verifies regardless.
    to_create: list[Path] = []
    cursor = bind_dir
    while not cursor.exists():
        to_create.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    for missing in reversed(to_create):
        missing.mkdir(mode=0o700)

    component = bind_dir
    while True:
        # ``lstat`` (not ``stat``) so a symlinked component is detected, not
        # silently followed. As we walk up, every component eventually becomes
        # the final path element, so a symlink anywhere in the chain is caught.
        st = os.lstat(component)
        if _stat.S_ISLNK(st.st_mode):
            raise ConfigError(
                f"socket directory component {component} is a symlink; refusing "
                "to bind (a symlinked ancestor can be repointed by another user "
                "to hijack the client-visible socket path)"
            )
        if st.st_uid != euid:
            raise ConfigError(
                f"socket directory component {component} is owned by uid "
                f"{st.st_uid}, not the server uid {euid}; refusing to bind "
                "(another user could replace the socket at this path)"
            )
        sticky = bool(st.st_mode & 0o1000)
        if not sticky and (st.st_mode & 0o022):
            raise ConfigError(
                f"socket directory component {component} is group/other-writable "
                f"(mode {oct(st.st_mode & 0o7777)}); refusing to bind "
                "(another user could replace the socket at this path)"
            )
        if component == trusted_root:
            break
        component = component.parent

    # Bind on the literal, now-verified path: the chain clients traverse is
    # exactly the chain we vouched for (no symlinks), so verified == bound ==
    # client-visible.
    return path


def _lock_path(socket_path: str | os.PathLike) -> str:
    return f"{socket_path}.lock"


# After the drain timeout, how long cancelled session tasks get to unwind
# before shutdown proceeds regardless.
_CANCEL_GRACE_SECONDS = 2.0

# How long ``acquire_socket_lock`` retries a contended lock before reporting
# EADDRINUSE. A live server holds it for its whole lifetime; a preflight probe
# (``socket_lock_held``) holds it for microseconds, and must not make a
# concurrent real starter fail.
_LOCK_RETRY_SECONDS = 0.25
_LOCK_RETRY_INTERVAL = 0.01


def acquire_socket_lock(socket_path: str | os.PathLike) -> int:
    """Take the single-server lock for ``socket_path``; return the held fd.

    ``connect()`` cannot tell a stale socket from a live one whose listen
    backlog is full (macOS returns ECONNREFUSED for both), so liveness is
    decided by an ``flock`` on ``<socket>.lock`` instead. The kernel drops the
    lock when the owning process dies, so it can never go stale; clients never
    touch it, so any number of them share one server. The lock file is left in
    place on exit (unlinking it would race a concurrent starter). It is created
    0600; a pre-existing one we own with group/other bits is tightened to 0600
    (another uid able to open it could hold the lock and block every future
    server), and one we do not own is refused. The mode is set with ``fchmod``
    on the open fd, so it never depends on the process umask (a ``0o777`` umask
    would otherwise create a mode-000 file no later start could open). Raises
    ``OSError(EADDRINUSE)`` when another process holds it for longer than a
    brief probe would, and ``OSError(EACCES)`` naming the remedy when an
    existing lock file cannot be opened.
    """
    import errno
    import fcntl

    path = _lock_path(socket_path)
    try:
        # O_NONBLOCK: a FIFO planted at the path must not block the open.
        fd = os.open(
            path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
            0o600,
        )
    except PermissionError as exc:
        raise OSError(
            errno.EACCES,
            f"cannot open the single-server lock file {path} ({exc.strerror}); "
            "if it is yours, chmod 600 it (or remove it while no server runs)",
        ) from exc
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise OSError(errno.EACCES, f"{path} is not a regular file owned by this user")
        if _stat.S_IMODE(st.st_mode) != 0o600:
            os.fchmod(fd, 0o600)
        deadline = time.monotonic() + _LOCK_RETRY_SECONDS
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() >= deadline:
                    raise OSError(
                        errno.EADDRINUSE, f"a server is already running for {socket_path}"
                    ) from exc
                time.sleep(_LOCK_RETRY_INTERVAL)
    except BaseException:
        os.close(fd)
        raise
    return fd


def socket_lock_held(socket_path: str | os.PathLike) -> bool:
    """True when another process holds the lock for ``socket_path``.

    Read-only probe for preflight: never creates the lock file, never follows a
    symlink, and treats anything that is not a plain owned file as "not held"
    (``start()`` will report it properly when it tries to acquire). Probes with
    a *shared* lock: that conflicts with a server's exclusive lock (so a live
    server reads as held) but not with another probe. A shared lock still
    conflicts with a concurrent starter's exclusive one, so the probe holds it
    only for an instant and ``acquire_socket_lock`` retries briefly.
    """
    import errno
    import fcntl

    try:
        # O_NONBLOCK: preflight runs before the socket-directory trust check, so
        # the path may be in a directory another user writes; a FIFO planted
        # there must not hang the open.
        fd = os.open(
            _lock_path(socket_path),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        )
    except OSError:
        return False
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            return False  # not a plain owned file: start() reports it
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            return exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


@dataclass
class ServerConfig:
    """Transport and policy configuration for ``TranscriptionServer``."""

    socket_path: str | None = None
    host: str | None = None
    port: int | None = None
    auth_token: str | None = None
    reject_browser_origins: bool = True
    max_append_bytes: int = P.MAX_APPEND_BYTES
    max_uncommitted_bytes: int = P.MAX_UNCOMMITTED_BYTES
    # Bounded write-buffer per session: if the socket's pending-send buffer
    # exceeds this many bytes, the server closes the session with
    # ``send_queue_overflow`` rather than blocking the decode loop on a slow
    # consumer.
    send_queue_high_water_bytes: int = P.SEND_QUEUE_HIGH_WATER_BYTES
    drain_timeout_seconds: float = P.SHUTDOWN_DRAIN_TIMEOUT_SECONDS
    # chmod applied to UDS after bind. 0o600 restricts connect to the owning
    # user; set to None to inherit the process umask (not recommended on
    # shared hosts — the UDS is the V1 trust boundary).
    unix_socket_mode: int | None = 0o600

    def __post_init__(self) -> None:
        if self.socket_path is None and (self.host is None or self.port is None):
            raise ConfigError("ServerConfig requires socket_path or host+port")
        if self.host is not None and self.host not in ("127.0.0.1", "::1", "localhost"):
            raise ConfigError("V1 only permits loopback binds (127.0.0.1/::1/localhost)")


@dataclass
class _SessionState:
    session_id: str
    config: dict = field(default_factory=lambda: {"turn_detection": None})
    buffer: bytearray = field(default_factory=bytearray)
    # ``asyncio.Task`` here reflects the V1 in-process topology. A future
    # worker-process backend would replace this with a cross-process handle.
    in_flight_task: asyncio.Task | None = None
    current_stream: BackendStream | None = None
    closed: bool = False
    started_monotonic: float = field(default_factory=time.monotonic)


class TranscriptionServer:
    """Owns the listener and lifecycle for in-process WebSocket sessions."""

    def __init__(self, backend: TranscriptionBackend, config: ServerConfig) -> None:
        self._backend = backend
        self._config = config
        self._server: Server | None = None
        self._active_handlers: set[asyncio.Task] = set()
        self._active_decodes: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        self._started = False
        # (st_dev, st_ino) of the Unix socket file THIS process bound, recorded
        # right after bind so ``abort_sync`` only ever unlinks our own inode.
        self._bound_socket_id: tuple[int, int] | None = None
        self._socket_lock_fd: int | None = None
        # Lifecycle contract between ``start()`` and ``shutdown()``:
        # ``_closing`` is set (for good) the moment ``shutdown()`` is entered;
        # a ``start()`` still in flight then never binds a listener or sets
        # ``_started`` -- it fails at its next checkpoint, closing anything it
        # bound and releasing the lock itself (``_starting`` tells
        # ``shutdown()`` the lock is still that start's to release).
        self._closing = False
        self._starting = False
        # Session tasks that did not unwind within the cancel grace of the last
        # ``shutdown()`` (see ``stuck_task_count``).
        self._stuck_tasks: set[asyncio.Task] = set()

    # --- lifecycle ---
    async def start(self) -> None:
        if self._started:
            return
        if self._closing:
            raise RuntimeError("stt_server: start() after shutdown()")
        self._starting = True
        try:
            await self._start_unlocked()
        except BaseException:
            # A server that never started is never ``close()``d, so close any
            # listener this start bound, release the lock and run the backend's
            # cheap ``abort()`` hook here (parakeet / nemotron would otherwise
            # leak their private temp dir on every failed start, e.g. a launchd
            # crash-loop against a held lock).
            self._close_unstarted_listener()
            self._release_socket_lock()
            abort_backend(self._backend, logger)
            raise
        finally:
            self._starting = False

    def _check_not_closing(self) -> None:
        if self._closing:
            raise RuntimeError("stt_server: shutdown() was called during startup")

    async def _bind(self, coro: Awaitable[Server]) -> Server:
        """Await a listener bind; never leak the listener if cancelled mid-bind.

        The bind itself is synchronous inside ``create_server`` but is followed
        by an await, so a cancellation landing there would drop a bound,
        serving listener. Shield it; on cancel, record the finished listener in
        ``_server`` (``start()``'s failure path closes it) and re-raise.
        """
        task = asyncio.ensure_future(coro)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.wait({task})
            if not task.cancelled() and task.exception() is None:
                self._server = task.result()
                self._record_bound_socket()
            raise

    def _record_bound_socket(self) -> None:
        """Record the (st_dev, st_ino) of the Unix socket this process bound.

        Called right after bind, while this start holds the single-server lock,
        so the inode at the path is ours."""
        if not self._config.socket_path:
            return
        try:
            st = os.stat(self._config.socket_path)
            if _stat.S_ISSOCK(st.st_mode):
                self._bound_socket_id = (st.st_dev, st.st_ino)
        except OSError:
            pass

    def _unlink_own_socket(self) -> None:
        """Remove the Unix socket file only if it is still the inode we bound."""
        bound = self._bound_socket_id
        if bound is not None and self._config.socket_path:
            path = str(self._config.socket_path)
            try:
                st = os.lstat(path)
                if _stat.S_ISSOCK(st.st_mode) and (st.st_dev, st.st_ino) == bound:
                    os.unlink(path)
            except OSError:
                pass

    def _close_unstarted_listener(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
            self._unlink_own_socket()

    async def _start_unlocked(self) -> None:
        verified_socket: Path | None = None
        if self._config.socket_path:
            socket_path = Path(self._config.socket_path)
            # Trusted root for the ancestor walk: the user's home directory. The
            # default socket lives under ~/Library/Caches/pipecat-stt (see
            # scripts/install_stt_agent.sh), so on stock macOS every component
            # from the cache dir up through home is owner-owned and 0700. Enforce
            # an owner-only, non-group/other-writable chain from the socket
            # directory up to and including home BEFORE bind so the socket cannot
            # be planted/swapped by another local uid. This replaces a bare
            # mkdir that trusted the path blindly and verifies the full ancestor
            # walk, not just the immediate parent. The enforcement rejects any
            # symlink component and returns the literal verified path, so the
            # chain we vouched for is exactly the one clients traverse.
            verified_socket = _enforce_socket_dir_secure(socket_path, Path.home())
            # Single-server lock BEFORE the model load: a second instance fails
            # in milliseconds instead of after a multi-GB load, and the socket
            # file below is only ever replaced by the lock holder.
            self._socket_lock_fd = acquire_socket_lock(verified_socket)
        await self._start_locked(verified_socket)

    def _release_socket_lock(self) -> None:
        fd, self._socket_lock_fd = self._socket_lock_fd, None
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass

    async def _start_locked(self, verified_socket: Path | None) -> None:
        await self._backend.start()
        # Checkpoint: a ``shutdown()`` that ran while the backend started must
        # not be followed by a bind (nobody would ever close that listener).
        self._check_not_closing()
        if verified_socket is not None:
            # Restrict the socket file to owner-only before bind so the UDS
            # trust boundary actually holds on multi-user hosts.
            prior_umask = os.umask(0o077)
            try:
                self._server = await self._bind(
                    ws_unix_serve(
                        self._handle_connection,
                        path=str(verified_socket),
                        max_size=self._config.max_append_bytes,
                        process_request=self._process_request,
                    )
                )
            finally:
                os.umask(prior_umask)
            self._record_bound_socket()
            if self._config.unix_socket_mode is not None:
                try:
                    os.chmod(verified_socket, self._config.unix_socket_mode)
                except OSError as exc:
                    logger.warning("stt_server: chmod on UDS failed: %s", exc)
        else:
            self._server = await self._bind(
                ws_serve(
                    self._handle_connection,
                    host=self._config.host,
                    port=self._config.port,
                    max_size=self._config.max_append_bytes,
                    process_request=self._process_request,
                )
            )
        # Checkpoint: ``shutdown()`` ran during the bind. ``start()``'s failure
        # path closes the listener we just bound and unlinks our socket inode.
        self._check_not_closing()
        self._started = True
        logger.info(
            "stt_server listening on %s",
            self._config.socket_path or f"{self._config.host}:{self._config.port}",
        )
        # TCP without a token accepts unauthed clients. UDS mode is protected
        # by the 0o600 file permission and the local trust boundary; TCP has
        # neither. Make the gap loud so operators notice before shipping.
        if self._config.socket_path is None and not self._config.auth_token:
            logger.warning(
                "stt_server: TCP listener is running without --auth-token; "
                "any local process that can reach %s:%s can connect. "
                "Configure PIPECAT_STT_AUTH_TOKEN (or the deprecated "
                "KODA_STT_AUTH_TOKEN, or --auth-token-file) for "
                "any non-experimental deployment.",
                self._config.host,
                self._config.port,
            )

    @property
    def sockets_bound(self) -> list:
        if self._server is None:
            return []
        return list(self._server.sockets or [])

    def listening_port(self) -> int | None:
        """For TCP mode, return the actual port (helpful with port=0)."""
        for s in self.sockets_bound:
            try:
                name = s.getsockname()
            except Exception:
                continue
            # A Unix socket's name is a path string, not a (host, port) tuple.
            if isinstance(name, tuple) and len(name) >= 2:
                return name[1]
        return None

    @property
    def backend(self) -> TranscriptionBackend:
        return self._backend

    @property
    def config(self) -> ServerConfig:
        return self._config

    async def wait_closed(self) -> None:
        if self._server is not None:
            await self._server.wait_closed()

    def abort_sync(self) -> None:
        """Best-effort, synchronous cleanup for a hard process exit.

        Used by ``serve`` when a stop arrives while startup/warm-up is still
        running and the process must ``os._exit`` (model-load threads cannot be
        cancelled), which skips ``shutdown()`` and ``backend.close()``. Removes
        the Unix socket file only if it is still the inode this process bound
        (never a socket someone else created at the same path), and calls the
        backend's optional ``abort()`` hook (cheap, non-blocking: e.g. removing
        a private temp dir -- no thread joins, no Metal drain).

        The single-server lock is deliberately NOT released: the caller is about
        to ``os._exit``, and the kernel drops the lock with the process. Releasing
        it here would let a new server acquire it and start loading while this
        process (and its model-load thread) still exists.
        """
        self._unlink_own_socket()
        abort_backend(self._backend, logger)

    @property
    def stuck_task_count(self) -> int:
        """Session tasks from the last ``shutdown()`` still alive after its
        cancel grace (they swallowed or outlived cancellation)."""
        return sum(1 for t in self._stuck_tasks if not t.done())

    async def shutdown(self) -> None:
        """Stop accepting new connections and drain active sessions.

        Drains both connection handlers and their spawned decode tasks under
        a single bounded timeout. Force-cancels anything still running so the
        process never wedges on MLX/Metal teardown.

        Contract:

        - Once entered, no ``start()`` in flight (or later) binds a listener or
          marks the server started; an in-flight start releases the
          single-server lock itself when it unwinds.
        - Returns within the drain timeout plus the cancel grace even when a
          session task swallows cancellation. Such a task is left running
          (``stuck_task_count``) and the single-server lock stays held until
          the last of them finishes, so no new server can take over the socket
          while this one's sessions still run. The CLI (``serve`` with
          ``hard_exit_on_stop=True``) then exits the process; an embedding
          caller owns them (``asyncio.run`` teardown would wait on them).
        """
        self._closing = True
        if not self._started:
            if not self._starting:
                self._release_socket_lock()
            return
        # launchd sends SIGTERM once, but defensive double-shutdown is cheap:
        # if a second caller races in while the first is still draining, let
        # it wait for the first to finish and return. Without this the second
        # call would re-gather already-awaited tasks and call
        # ``backend.close()`` twice.
        if self._shutdown_event.is_set():
            while self._started:
                await asyncio.sleep(0)
            return
        self._shutdown_event.set()
        try:
            await self._drain_and_close()
        finally:
            # Even when the drain or ``backend.close()`` raises (or this task is
            # cancelled), release the single-server lock (deferred while a stuck
            # session task lives) and mark the server stopped: a concurrent
            # second ``shutdown()`` spins on ``_started``.
            stuck = {t for t in self._stuck_tasks if not t.done()}
            if stuck:
                self._release_socket_lock_after(stuck)
            else:
                self._release_socket_lock()
            self._started = False

    def _release_socket_lock_after(self, tasks: set[asyncio.Task]) -> None:
        """Hand the lock fd to ``tasks``: closed when the last one finishes."""
        fd, self._socket_lock_fd = self._socket_lock_fd, None
        if fd is None:
            return
        remaining = set(tasks)

        def _done(task: asyncio.Task) -> None:
            remaining.discard(task)
            if not remaining:
                try:
                    os.close(fd)
                except OSError:
                    pass

        for t in tasks:
            t.add_done_callback(_done)

    async def _drain_and_close(self) -> None:
        if self._server is not None:
            # ``close_connections=True`` sends a close frame to every open
            # socket so handler recv loops exit instead of blocking forever
            # on an idle client.
            self._server.close(close_connections=True)
        pending = list(self._active_handlers | self._active_decodes)
        if pending:
            # ``asyncio.wait`` (not ``wait_for(gather(...))``): ``wait_for``
            # waits for the cancelled gather to finish unwinding, so a task
            # that delays or swallows cancellation (the session handlers catch
            # ``CancelledError`` around in-flight decodes and then send/close)
            # would hold shutdown with no bound.
            _done, still = await asyncio.wait(pending, timeout=self._config.drain_timeout_seconds)
            if still:
                logger.warning(
                    "stt_server: drain timeout expired with %d tasks in flight", len(still)
                )
                for t in still:
                    t.cancel()
                # Give the cancellations a short, bounded window to unwind
                # before backend.close() touches shared MLX/Metal state.
                _done, stuck = await asyncio.wait(still, timeout=_CANCEL_GRACE_SECONDS)
                self._stuck_tasks = set(stuck)
                if stuck:
                    logger.warning(
                        "stt_server: %d tasks did not unwind within %.1fs of cancel; "
                        "closing anyway",
                        len(stuck),
                        _CANCEL_GRACE_SECONDS,
                    )
        for t in pending:
            # Retrieve outcomes, as ``gather(return_exceptions=True)`` did, so a
            # failed task is not reported as "exception never retrieved".
            if t.done() and not t.cancelled():
                t.exception()
        if self._server is not None:
            # ``wait_closed`` also waits for connection handlers; bound it the
            # same way so a stuck handler cannot hold shutdown.
            closer = asyncio.ensure_future(self._server.wait_closed())
            _done, stuck = await asyncio.wait({closer}, timeout=_CANCEL_GRACE_SECONDS)
            if stuck:
                closer.cancel()
        await self._backend.close()

    # --- connection handling ---
    async def _process_request(self, connection, request):
        # On a Unix-domain socket the only legitimate peer is a process owned
        # by the same effective uid as this server. Enforce that BEFORE (and
        # independently of) bearer auth so a foreign uid is rejected even when
        # no token is configured. Fail closed: any inability to resolve the
        # peer's uid is a 403, not a pass. TCP listeners cannot answer
        # SO_PEERCRED-style queries, so this gate is UDS-only.
        if self._config.socket_path is not None:
            transport = connection.transport
            if transport is None:
                # websockets 16 sets ``transport`` before ``process_request``
                # runs; guard defensively so a future/edge None fails closed
                # rather than raising AttributeError.
                logger.warning("stt_server: UDS connection has no transport; rejecting")
                return connection.respond(403, "peer not permitted\n")
            sock = transport.get_extra_info("socket")
            if sock is None:
                logger.warning("stt_server: UDS connection has no underlying socket; rejecting")
                return connection.respond(403, "peer not permitted\n")
            if sock.family != socket.AF_UNIX:
                logger.warning(
                    "stt_server: UDS connection on unexpected socket family %r; rejecting",
                    sock.family,
                )
                return connection.respond(403, "peer not permitted\n")
            try:
                uid = peer_uid(sock)
            except Exception as exc:
                logger.warning("stt_server: peer uid resolution failed (%s); rejecting", exc)
                return connection.respond(403, "peer not permitted\n")
            if uid is None or uid != os.geteuid():
                logger.warning(
                    "stt_server: rejecting UDS peer uid %r (expected %d)",
                    uid,
                    os.geteuid(),
                )
                return connection.respond(403, "peer not permitted\n")

        # Reject unexpected browser Origin headers for non-browser-focused V1,
        # and enforce optional bearer auth in one place.
        headers = request.headers
        origin = headers.get("Origin")
        if self._config.reject_browser_origins and origin:
            return connection.respond(403, "origin not permitted\n")
        if self._config.auth_token:
            provided = headers.get("Authorization", "") or ""
            expected = f"Bearer {self._config.auth_token}"
            # Constant-time compare so a loopback attacker cannot recover the
            # token byte-by-byte via response-timing signals.
            if not hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
                return connection.respond(401, "unauthorized\n")
        return None

    async def _handle_connection(self, ws: ServerConnection) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._active_handlers.add(task)
        state = _SessionState(session_id=_session_id())
        try:
            await self._send(
                ws,
                state,
                {
                    "type": P.EVT_SERVER_HELLO,
                    "event_id": _event_id(),
                    "protocol_version": P.PROTOCOL_VERSION,
                    "capabilities": {
                        "binary_audio": True,
                        "base64_audio_append": True,
                        "server_vad": False,
                    },
                    "audio": {
                        "format": P.AUDIO_FORMAT,
                        "rate": P.AUDIO_SAMPLE_RATE_HZ,
                        "channels": P.AUDIO_CHANNELS,
                    },
                    # Backend identity — lets a client verify which ASR is
                    # actually behind this socket instead of trusting the
                    # socket path. Additive optional field; old readers ignore.
                    "backend": {
                        "name": self._backend.backend_name,
                        "model": self._backend.model,
                    },
                },
            )
            session_snapshot = _session_snapshot(state)
            # Emit both the legacy name (existing clients) and the OpenAI
            # transcription-mode name so strict OpenAI clients see the event
            # they expect. Same ``session`` payload under both.
            for evt_type in (P.EVT_SESSION_CREATED, P.EVT_TRANSCRIPTION_SESSION_CREATED):
                await self._send(
                    ws,
                    state,
                    {
                        "type": evt_type,
                        "event_id": _event_id(),
                        "session": session_snapshot,
                    },
                )
            async for raw in ws:
                if state.closed:
                    break
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        await self._handle_binary_audio(ws, state, bytes(raw))
                    else:
                        await self._handle_text(ws, state, raw)
                except Exception:
                    # Don't echo raw exception text (paths, tokens in
                    # nested RuntimeError(...) args, etc.) to the peer.
                    logger.exception("stt_server: error handling message")
                    await self._error(ws, state, P.ErrorCode.INTERNAL_ERROR, "internal error")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._teardown_session(ws, state)
            self._active_handlers.discard(task)

    # --- message handlers ---
    async def _handle_text(self, ws: ServerConnection, state: _SessionState, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError as exc:
            await self._error(ws, state, P.ErrorCode.INVALID_JSON, str(exc))
            return
        if not isinstance(msg, dict) or "type" not in msg:
            await self._error(ws, state, P.ErrorCode.INVALID_EVENT, "missing type")
            return
        t = msg["type"]
        client_event_id = msg.get("event_id") if isinstance(msg.get("event_id"), str) else None
        if t not in P.CLIENT_EVENT_TYPES:
            await self._error(
                ws,
                state,
                P.ErrorCode.UNSUPPORTED_EVENT,
                f"unknown event: {t}",
                client_event_id=client_event_id,
            )
            return

        if t in (P.EVT_SESSION_UPDATE, P.EVT_TRANSCRIPTION_SESSION_UPDATE):
            await self._on_session_update(ws, state, msg, client_event_id)
        elif t == P.EVT_AUDIO_APPEND:
            await self._on_audio_append_json(ws, state, msg)
        elif t == P.EVT_AUDIO_COMMIT:
            await self._on_commit(ws, state, client_event_id)
        elif t == P.EVT_AUDIO_CLEAR:
            await self._on_clear(ws, state, client_event_id)
        elif t == P.EVT_SERVER_STATUS_REQ:
            await self._on_status(ws, state)
        elif t == P.EVT_SESSION_CLOSE:
            await self._on_close(ws, state)
        elif t == P.EVT_SESSION_CANCEL:
            await self._on_cancel(ws, state)

    async def _handle_binary_audio(
        self, ws: ServerConnection, state: _SessionState, data: bytes
    ) -> None:
        # Reject appends while a decode is in flight rather than silently
        # merging new audio into the next turn's buffer, which would lose the
        # commit boundary the client just signalled.
        if state.in_flight_task is not None and not state.in_flight_task.done():
            await self._error(
                ws,
                state,
                P.ErrorCode.INVALID_EVENT,
                "audio append while previous decode in flight",
            )
            return
        if len(data) > self._config.max_append_bytes:
            await self._error(
                ws, state, P.ErrorCode.PAYLOAD_TOO_LARGE, "binary audio exceeds max_append_bytes"
            )
            return
        if len(data) % P.AUDIO_SAMPLE_WIDTH_BYTES != 0:
            # PCM16 samples are 2 bytes; an odd-length append would misalign
            # every subsequent sample (silent wrong-result transcript, not a
            # crash). Reject at the boundary so the client notices.
            await self._error(
                ws,
                state,
                P.ErrorCode.INVALID_EVENT,
                f"audio length {len(data)} is not a multiple of {P.AUDIO_SAMPLE_WIDTH_BYTES}",
            )
            return
        if len(state.buffer) + len(data) > self._config.max_uncommitted_bytes:
            await self._error(
                ws,
                state,
                P.ErrorCode.BUFFER_OVERFLOW,
                "uncommitted audio exceeds per-session cap",
            )
            return
        state.buffer.extend(data)

    async def _on_session_update(
        self,
        ws: ServerConnection,
        state: _SessionState,
        msg: dict,
        client_event_id: str | None,
    ) -> None:
        session = msg.get("session") or {}
        audio_in = (session.get("audio") or {}).get("input") or {}

        # ``input_audio_format`` — accept OpenAI's enum string at session
        # root (``"pcm16"``) or the legacy nested dict. Only pcm16/16 kHz/
        # mono is on the wire regardless.
        fmt_root = session.get("input_audio_format")
        if isinstance(fmt_root, str):
            if fmt_root != P.AUDIO_FORMAT:
                await self._error(
                    ws,
                    state,
                    P.ErrorCode.INVALID_CONFIG,
                    f"only {P.AUDIO_FORMAT!r} input_audio_format is supported",
                    client_event_id=client_event_id,
                )
                return
        fmt = audio_in.get("format") or {}
        if fmt:
            if fmt.get("encoding", P.AUDIO_FORMAT) != P.AUDIO_FORMAT:
                await self._error(
                    ws,
                    state,
                    P.ErrorCode.INVALID_CONFIG,
                    "only pcm16 encoding is supported",
                    client_event_id=client_event_id,
                )
                return
            if fmt.get("rate", P.AUDIO_SAMPLE_RATE_HZ) != P.AUDIO_SAMPLE_RATE_HZ:
                await self._error(
                    ws,
                    state,
                    P.ErrorCode.INVALID_CONFIG,
                    "only 16000 Hz sample rate is supported",
                    client_event_id=client_event_id,
                )
                return
            if fmt.get("channels", P.AUDIO_CHANNELS) != P.AUDIO_CHANNELS:
                await self._error(
                    ws,
                    state,
                    P.ErrorCode.INVALID_CONFIG,
                    "only mono audio is supported",
                    client_event_id=client_event_id,
                )
                return

        # ``turn_detection`` — accept at the OpenAI session root OR nested
        # under ``session.audio.input``. Either must be null (server VAD
        # unimplemented).
        for td_container in (session, audio_in):
            if "turn_detection" in td_container:
                td = td_container["turn_detection"]
                if td is not None:
                    await self._error(
                        ws,
                        state,
                        P.ErrorCode.INVALID_CONFIG,
                        "server VAD is not implemented; use turn_detection: null",
                        client_event_id=client_event_id,
                    )
                    return
                state.config["turn_detection"] = None

        # Language may be supplied either nested (legacy) or at the session
        # root (pragmatic; OpenAI keeps it under input_audio_transcription).
        if "language" in audio_in:
            state.config["language"] = audio_in["language"]
        transcription = session.get("input_audio_transcription")
        if isinstance(transcription, dict) and "language" in transcription:
            state.config["language"] = transcription["language"]

        session_snapshot = _session_snapshot(state)
        for evt_type in (P.EVT_SESSION_UPDATED, P.EVT_TRANSCRIPTION_SESSION_UPDATED):
            await self._send(
                ws,
                state,
                {
                    "type": evt_type,
                    "event_id": _event_id(),
                    "session": session_snapshot,
                },
            )

    async def _on_audio_append_json(
        self, ws: ServerConnection, state: _SessionState, msg: dict
    ) -> None:
        import base64

        audio_b64 = msg.get("audio")
        if not isinstance(audio_b64, str):
            await self._error(
                ws, state, P.ErrorCode.INVALID_EVENT, "input_audio_buffer.append missing audio"
            )
            return
        try:
            data = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            await self._error(
                ws, state, P.ErrorCode.INVALID_EVENT, f"audio base64 decode failed: {exc}"
            )
            return
        await self._handle_binary_audio(ws, state, data)

    async def _on_commit(
        self,
        ws: ServerConnection,
        state: _SessionState,
        client_event_id: str | None,
    ) -> None:
        if len(state.buffer) == 0:
            await self._error(
                ws,
                state,
                P.ErrorCode.BUFFER_EMPTY,
                "commit on empty buffer",
                client_event_id=client_event_id,
            )
            return
        if state.in_flight_task is not None and not state.in_flight_task.done():
            await self._error(
                ws,
                state,
                P.ErrorCode.INVALID_EVENT,
                "commit while previous decode in flight",
                client_event_id=client_event_id,
            )
            return

        audio = bytes(state.buffer)
        state.buffer.clear()
        item = _item_id()
        payload: dict[str, Any] = {
            "type": P.EVT_AUDIO_COMMITTED,
            "event_id": _event_id(),
            "item_id": item,
        }
        if client_event_id:
            # Echo the client's event_id so request/response correlation works
            # without making clients track server-minted ids.
            payload["previous_event_id"] = client_event_id
        await self._send(ws, state, payload)
        decode_task = asyncio.create_task(self._run_decode(ws, state, audio, item))
        state.in_flight_task = decode_task
        self._active_decodes.add(decode_task)
        decode_task.add_done_callback(self._active_decodes.discard)

    async def _on_clear(
        self,
        ws: ServerConnection,
        state: _SessionState,
        client_event_id: str | None,
    ) -> None:
        # OpenAI ``input_audio_buffer.clear`` — drop uncommitted audio
        # without touching session lifecycle. Different from
        # ``session.cancel``, which also closes the session.
        if state.in_flight_task is not None and not state.in_flight_task.done():
            await self._error(
                ws,
                state,
                P.ErrorCode.INVALID_EVENT,
                "clear while decode in flight; use session.cancel to abort",
                client_event_id=client_event_id,
            )
            return
        state.buffer.clear()
        payload: dict[str, Any] = {
            "type": P.EVT_AUDIO_CLEARED,
            "event_id": _event_id(),
        }
        if client_event_id:
            payload["previous_event_id"] = client_event_id
        await self._send(ws, state, payload)

    async def _run_decode(
        self,
        ws: ServerConnection,
        state: _SessionState,
        audio: bytes,
        item_id: str,
    ) -> None:
        stream: BackendStream | None = None
        try:
            # The client ``language`` is passed through verbatim; each backend
            # translates it into its own idiom. Accepted values: ``None`` /
            # ``"auto"`` (detect) or an ISO/BCP-47 code. Per-backend handling:
            # mlx_whisper recasts ``"auto"``/blank -> ``None`` (auto-detect),
            # parakeet ignores it (model is language-pinned), nemotron forwards
            # it (``"auto"`` is a real LID prompt key; ``None`` -> its ``"auto"``
            # default). A new backend MUST decide how it maps ``"auto"``.
            stream = await self._backend.open_stream(language=state.config.get("language"))
            state.current_stream = stream
            await stream.feed(audio)
            await stream.end()
            final_text = ""
            async for ev in stream.events():
                if ev.kind == "delta":
                    await self._send(
                        ws,
                        state,
                        {
                            "type": P.EVT_TRANSCRIPT_DELTA,
                            "event_id": _event_id(),
                            "item_id": item_id,
                            "content_index": 0,
                            "delta": ev.text,
                        },
                    )
                elif ev.kind == "completed":
                    final_text = ev.text
            await self._send(
                ws,
                state,
                {
                    "type": P.EVT_TRANSCRIPT_COMPLETED,
                    "event_id": _event_id(),
                    "item_id": item_id,
                    "content_index": 0,
                    "transcript": final_text,
                },
            )
        except asyncio.CancelledError:
            if stream is not None:
                try:
                    await stream.cancel()
                except Exception:
                    pass
            raise
        except Exception:
            # Log full detail server-side; send a generic message to the
            # client (exception text can leak filesystem paths, model names,
            # or misconfigured backend secrets).
            logger.exception("stt_server: backend decode failed")
            await self._send(
                ws,
                state,
                {
                    "type": P.EVT_TRANSCRIPT_FAILED,
                    "event_id": _event_id(),
                    "item_id": item_id,
                    "content_index": 0,
                    "error": {
                        "type": _ERROR_TYPE_FOR_CODE[P.ErrorCode.BACKEND_ERROR],
                        "code": P.ErrorCode.BACKEND_ERROR.value,
                        "message": "backend decode failed",
                    },
                },
                force=True,
            )
        finally:
            state.current_stream = None

    async def _on_status(self, ws: ServerConnection, state: _SessionState) -> None:
        await self._send(
            ws,
            state,
            {
                "type": P.EVT_SERVER_STATUS,
                "event_id": _event_id(),
                "session_id": state.session_id,
                # Backend identity, mirrored from server.hello — so a status
                # probe alone is enough to confirm which ASR is running.
                "backend": {
                    "name": self._backend.backend_name,
                    "model": self._backend.model,
                },
                "queue_depth": 1 if state.in_flight_task and not state.in_flight_task.done() else 0,
                "uncommitted_bytes": len(state.buffer),
                "uptime_seconds": time.monotonic() - state.started_monotonic,
                "pid": os.getpid(),
                # Peak RSS from getrusage — stdlib, no psutil dep. Sufficient
                # for soak leak-detection (peak only climbs when a leak is
                # actually growing). ru_maxrss is bytes on macOS, kibibytes
                # on Linux/BSD — normalize to bytes so the wire contract is
                # platform-independent.
                "rss_bytes": _process_peak_rss_bytes(),
            },
        )

    async def _on_close(self, ws: ServerConnection, state: _SessionState) -> None:
        if state.closed:
            return
        if state.in_flight_task and not state.in_flight_task.done():
            # If shutdown() is already running it owns the drain budget; don't
            # double-spend it here or a client sending session.close during
            # shutdown doubles the worst-case wait. Cancel immediately and
            # let shutdown's gather clean up.
            if self._shutdown_event.is_set():
                state.in_flight_task.cancel()
            else:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(state.in_flight_task),
                        timeout=self._config.drain_timeout_seconds,
                    )
                except TimeoutError:
                    # Force-cancel and wait for unwind so the decode can't
                    # outlive the session.
                    state.in_flight_task.cancel()
                    try:
                        await state.in_flight_task
                    except (asyncio.CancelledError, Exception):
                        pass
        state.closed = True
        await self._send(
            ws,
            state,
            {
                "type": P.EVT_SESSION_CLOSED,
                "event_id": _event_id(),
                "session_id": state.session_id,
                "reason": "client_close",
            },
            force=True,
        )
        await ws.close()

    async def _on_cancel(self, ws: ServerConnection, state: _SessionState) -> None:
        if state.closed:
            return
        state.buffer.clear()
        if state.current_stream is not None:
            try:
                await state.current_stream.cancel()
            except Exception:
                pass
        if state.in_flight_task and not state.in_flight_task.done():
            state.in_flight_task.cancel()
            try:
                await state.in_flight_task
            except (asyncio.CancelledError, Exception):
                pass
        state.closed = True
        await self._send(
            ws,
            state,
            {
                "type": P.EVT_SESSION_CLOSED,
                "event_id": _event_id(),
                "session_id": state.session_id,
                "reason": "client_cancel",
            },
            force=True,
        )
        await ws.close()

    async def _teardown_session(self, ws: ServerConnection, state: _SessionState) -> None:
        if state.in_flight_task and not state.in_flight_task.done():
            state.in_flight_task.cancel()
            try:
                await state.in_flight_task
            except (asyncio.CancelledError, Exception):
                pass
        state.closed = True

    # --- send helpers ---
    async def _send(
        self,
        ws: ServerConnection,
        state: _SessionState,
        payload: dict,
        *,
        force: bool = False,
    ) -> None:
        # Backpressure: if the socket's pending write buffer is larger than
        # the per-session high-water mark, the consumer isn't keeping up. We
        # surface an explicit error and close the socket rather than block
        # the decode loop indefinitely. ``force=True`` bypasses the check for
        # terminal events (session.closed, error) so teardown can still flush.
        if not force and not state.closed:
            pending = _pending_write_bytes(ws)
            if pending is not None and pending > self._config.send_queue_high_water_bytes:
                logger.warning(
                    "stt_server: send-queue overflow (%d bytes pending), closing session %s",
                    pending,
                    state.session_id,
                )
                state.closed = True
                try:
                    await ws.close(code=1011, reason="send_queue_overflow")
                except Exception:
                    pass
                return
        try:
            await ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _error(
        self,
        ws: ServerConnection,
        state: _SessionState,
        code: P.ErrorCode,
        message: str,
        *,
        item_id: str | None = None,
        client_event_id: str | None = None,
    ) -> None:
        error_obj: dict[str, Any] = {
            "type": _ERROR_TYPE_FOR_CODE.get(code, "server_error"),
            "code": code.value,
            "message": message,
        }
        if client_event_id:
            error_obj["event_id"] = client_event_id
        payload: dict[str, Any] = {
            "type": P.EVT_ERROR,
            "event_id": _event_id(),
            "error": error_obj,
        }
        if item_id:
            payload["item_id"] = item_id
        await self._send(ws, state, payload, force=True)


def _session_snapshot(state: _SessionState) -> dict[str, Any]:
    """Build a ``session`` payload that names both legacy nested fields
    and the OpenAI transcription-mode flat fields, so clients of either
    shape get the values they expect."""
    return {
        "id": state.session_id,
        "object": "realtime.transcription_session",
        "type": "transcription",
        "input_audio_format": P.AUDIO_FORMAT,
        "turn_detection": state.config.get("turn_detection"),
        "input_audio_transcription": {"language": state.config.get("language")},
        "audio": {
            "input": {
                "format": {
                    "encoding": P.AUDIO_FORMAT,
                    "rate": P.AUDIO_SAMPLE_RATE_HZ,
                    "channels": P.AUDIO_CHANNELS,
                },
                "turn_detection": state.config.get("turn_detection"),
                "language": state.config.get("language"),
            }
        },
    }


def _pending_write_bytes(ws: ServerConnection) -> int | None:
    """Best-effort read of the socket's pending outbound byte count.

    websockets 15.x exposes the underlying asyncio transport; we look up the
    write-buffer size through it. Returns None if we cannot determine it,
    which disables the backpressure guard for that send.
    """
    try:
        transport = ws.transport  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if transport is None:
        return None
    try:
        return transport.get_write_buffer_size()
    except Exception:
        return None


async def serve(
    backend: TranscriptionBackend | None = None,
    *,
    socket_path: str | None = None,
    host: str | None = None,
    port: int | None = None,
    auth_token: str | None = None,
    install_signal_handlers: bool = True,
    ready: Callable[[TranscriptionServer], Awaitable[None]] | None = None,
    publish_ready: Callable[[TranscriptionServer], None] | None = None,
    stop_on_stdin_eof: bool = False,
    hard_exit_on_stop: bool = False,
) -> None:
    """Start the server, wait for shutdown signal, then drain and exit.

    The public entrypoint behind ``python -m stt_server``; tests drive
    ``TranscriptionServer`` directly.

    ``ready`` performs asynchronous readiness preparation such as model warm-up;
    ``publish_ready`` performs the final synchronous notification after a stop
    check. ``stop_on_stdin_eof`` (``serve --supervised``) treats stdin EOF like
    SIGTERM.

    A stop (signal, or stdin EOF) that arrives while startup or readiness
    preparation is pending:

    - ``hard_exit_on_stop=False`` (default, for in-process callers): the pending
      step is cancelled and awaited for at most the drain timeout
      (``ServerConfig.drain_timeout_seconds``), then ``serve`` returns after the
      normal ``shutdown()``. A step that outlives that bound can no longer bind
      a listener or mark the server started (see ``shutdown()``); a model load
      already running in a worker thread keeps running until it finishes
      (threads cannot be cancelled). Such leftovers are the caller's.
    - ``hard_exit_on_stop=True`` (the CLI, which owns its process): best-effort
      cleanup (``abort_sync``) and an immediate ``os._exit(0)``, so SIGTERM
      during a multi-GB load is honoured at once. Hugging Face downloads resume
      from their ``.incomplete`` blobs next time.

    After a normal ``shutdown()``, session tasks that swallowed cancellation
    (``stuck_task_count``) would make ``asyncio.run``'s teardown wait on them
    without a bound. With ``hard_exit_on_stop=True`` the process exits there
    (``os._exit``: 0 after a requested stop, 1 if ``serve`` was failing); with
    ``False`` they are the caller's, and the single-server lock stays held
    until they finish.
    """
    try:
        cfg = ServerConfig(socket_path=socket_path, host=host, port=port, auth_token=auth_token)
    except BaseException:
        # No server will ever ``start()``/``close()`` this backend.
        if backend is not None:
            abort_backend(backend, logger)
        raise
    server = TranscriptionServer(backend or EchoBackend(), cfg)
    loop = asyncio.get_running_loop()
    stop = loop.create_future()

    def _request_stop() -> None:
        if not stop.done():
            stop.set_result(None)

    def _hard_exit(stage: str) -> None:
        logger.warning("stt_server: stop requested during %s; exiting", stage)
        server.abort_sync()
        logging.shutdown()
        os._exit(0)

    async def _run_until_stop(coro: Awaitable[None], stage: str) -> bool:
        """Await ``coro`` unless a stop arrives first; True when it completed
        and no stop is pending."""
        task = asyncio.ensure_future(coro)
        await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        if stop.done() and not task.done():
            if hard_exit_on_stop:
                _hard_exit(stage)
            logger.info("stt_server: stop requested during %s; cancelling it", stage)
            task.cancel()
            # Bounded: a step that swallows cancellation cannot hold the stop.
            await asyncio.wait({task}, timeout=cfg.drain_timeout_seconds)
            if task.done() and not task.cancelled():
                task.exception()  # retrieved; we are stopping regardless
            return False
        await task
        return not stop.done()

    if install_signal_handlers:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_stop)
            except NotImplementedError:
                pass
    if stop_on_stdin_eof:
        from .supervise import watch_stdin_eof

        watch_stdin_eof(loop, _request_stop)
    failed = False
    try:
        # Race startup against stop in every mode: the signal handlers above are
        # installed before ``start()``, so awaiting it bare would defer a
        # SIGTERM/SIGINT until the backend import finished (``main`` installed
        # them after ``start()``, leaving the default terminate-now action).
        if not await _run_until_stop(server.start(), "startup"):
            return
        if ready is not None and not await _run_until_stop(ready(server), "readiness preparation"):
            return
        if publish_ready is not None:
            publish_ready(server)
        await stop
    except BaseException:
        failed = True
        raise
    finally:
        try:
            await server.shutdown()
        except BaseException:
            failed = True
            raise
        finally:
            if hard_exit_on_stop and server.stuck_task_count:
                logger.warning(
                    "stt_server: %d session tasks outlived shutdown; exiting",
                    server.stuck_task_count,
                )
                logging.shutdown()
                os._exit(1 if failed else 0)
