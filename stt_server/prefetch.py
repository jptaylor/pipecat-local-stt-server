"""``python -m stt_server prefetch`` — fetch a backend's weights without serving.

The ``mlx``, ``parakeet`` and ``nemotron`` backends download their checkpoint
on the first decode, so a fresh server answers its first request only after a
silent multi-GB download. ``prefetch`` does that download up front, with
optional JSON-lines progress for a supervising app, and ``--check`` answers
"are the weights already cached?" without touching the network.

Each backend module declares what it needs with a module-level
``prefetch_specs(model) -> list[SnapshotSpec]`` next to its default model, so a
model change stays a one-file change.

JSON events (one object per stdout line; the same schema as
``tts_server prefetch`` in pipecat-local-tts-server):

- fetch: ``start{repo,revision,kind}`` -> ``progress{repo,downloaded_bytes,
  total_bytes,files_done,files_total}``* -> ``done{repo,path,size_bytes}`` per
  snapshot, then ``complete{backend,model,cached,size_bytes}``. Progress counts
  only what this run has to download (files already in the cache are excluded,
  like ``tts_server prefetch``): ``total_bytes``/``files_total`` are the
  not-yet-cached files, ``downloaded_bytes`` includes partial ``.incomplete``
  blobs of an interrupted earlier run, and a last progress event with
  ``downloaded_bytes == total_bytes`` and ``files_done == files_total`` is
  emitted before ``done``;
- ``--check``: ``check{repo,revision,kind,required,cached,path,size_bytes}`` per
  snapshot, then ``summary{backend,model,cached,size_bytes}``;
- failure: ``error{code,message,repo}`` and exit status 1.

Progress is measured from the Hugging Face cache on disk (bytes landing in the
repo's ``blobs/``), so it works with every ``huggingface_hub`` version the
backends pin; Kyutai's ``moshi_mlx`` requires ``huggingface-hub<0.29``.

Lean at import: ``huggingface_hub`` is imported only inside the seams.
"""

from __future__ import annotations

import contextlib
import fnmatch
import importlib
import json
import logging
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger("stt_server.prefetch")

_PROGRESS_INTERVAL_S = 0.3


@dataclass(frozen=True)
class SnapshotSpec:
    """One Hugging Face snapshot a backend needs before it can decode offline."""

    repo_id: str
    revision: str | None = None  # None = the repo's main branch, as the loaders use
    #: ``None`` downloads the whole repo (what ``mlx_whisper`` does).
    allow_patterns: tuple[str, ...] | None = None
    #: Glob patterns that must each match a file for the snapshot to count as
    #: cached. Checked explicitly because older ``huggingface_hub`` releases
    #: return a partial snapshot folder from ``local_files_only=True``.
    required: tuple[str, ...] = ()
    kind: str = "model"
    #: Set only for an explicit local model reference. It bypasses Hub calls but
    #: still uses this backend-owned spec's ``required`` completeness patterns.
    local_path: str | None = None
    #: For repos whose weight filenames live in their own ``config.json`` (Kyutai:
    #: ``moshi_name`` may be a ``.q4``/``.q8`` variant). Called with the parsed
    #: config, it returns the exact filenames to fetch and require, replacing
    #: ``allow_patterns``/``required``. ``config.json`` is fetched first.
    resolve: Callable[[dict], tuple[str, ...]] | None = None


def _resolved(spec: SnapshotSpec, path: str | Path) -> SnapshotSpec:
    """``spec`` with its filenames taken from the snapshot's ``config.json``.

    Raises ``FileNotFoundError``/``ValueError``/``KeyError`` when the config is
    missing or does not name what the loader needs.
    """
    if spec.resolve is None:
        return spec
    config = json.loads((Path(path) / "config.json").read_text())
    names = tuple(spec.resolve(config))
    return replace(spec, allow_patterns=names, required=names, resolve=None)


def is_local_model(model: str) -> bool:
    """True when ``model`` names a local path rather than a hub repo id."""
    return model.startswith(("/", "./", "../", "~")) or Path(model).expanduser().exists()


_BACKEND_MODULES = {
    "mlx": "stt_server.backends.mlx_whisper",
    "parakeet": "stt_server.backends.parakeet",
    "nemotron": "stt_server.backends.nemotron",
    "kyutai": "stt_server.backends.kyutai",
}


def specs_for(backend: str, model: str | None) -> list[SnapshotSpec]:
    """Every snapshot ``backend`` needs to start ``model`` without the network."""
    if backend == "echo":
        return []
    try:
        module = _BACKEND_MODULES[backend]
    except KeyError:
        raise ValueError(f"unknown backend: {backend!r}") from None
    return list(importlib.import_module(module).prefetch_specs(model))


def local_specs_for(backend: str, model: str) -> list[SnapshotSpec]:
    """Attach an explicit local directory to the backend's structural specs."""
    if backend == "kyutai":
        raise ValueError("the kyutai backend does not support local model paths")
    path = Path(model).expanduser()
    return [replace(spec, local_path=str(path)) for spec in specs_for(backend, model)]


# --- patchable seams -------------------------------------------------------------


def _snapshot_download(spec: SnapshotSpec, *, local_files_only: bool) -> str:
    from huggingface_hub import snapshot_download  # type: ignore

    return str(
        snapshot_download(
            spec.repo_id,
            revision=spec.revision,
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
            local_files_only=local_files_only,
        )
    )


def _remote_files(spec: SnapshotSpec) -> list[tuple[str, int, str | None]]:
    """``(path, size, blob_id)`` of the files the snapshot will hold (for progress).

    ``blob_id`` is the file's name under the cache's ``blobs/`` dir (the LFS
    sha256, else the git blob id), or ``None`` when the hub did not report one.
    """
    from huggingface_hub import HfApi  # type: ignore

    files = []
    for entry in HfApi().list_repo_tree(spec.repo_id, revision=spec.revision, recursive=True):
        size = getattr(entry, "size", None)
        if size is None:
            continue  # a folder
        lfs = getattr(entry, "lfs", None)
        size = _lfs_field(lfs, "size") or size
        blob_id = _lfs_field(lfs, "sha256") or getattr(entry, "blob_id", None)
        if spec.allow_patterns and not any(
            fnmatch.fnmatch(entry.path, p) for p in spec.allow_patterns
        ):
            continue
        files.append((entry.path, int(size), blob_id))
    return files


def _lfs_field(lfs: Any, name: str) -> Any:
    """``lfs.<name>`` for both the object and the TypedDict forms of ``BlobLfsInfo``."""
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get(name)
    return getattr(lfs, name, None)


def _repo_cache_dir(repo_id: str) -> Path:
    from huggingface_hub import constants  # type: ignore

    return Path(constants.HF_HUB_CACHE) / ("models--" + repo_id.replace("/", "--"))


def _snapshot_size(path: str | Path) -> int:
    """Bytes in a snapshot folder (follows its symlinks into ``blobs/``)."""
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _complete(path: str | Path, required: tuple[str, ...]) -> bool:
    names = [str(p.relative_to(path)) for p in Path(path).rglob("*") if p.is_file()]
    return all(any(fnmatch.fnmatch(n, pattern) for n in names) for pattern in required)


# --- check / fetch ---------------------------------------------------------------


def check_spec(spec: SnapshotSpec) -> dict[str, Any]:
    """Offline cache probe for one snapshot. ``ImportError`` propagates."""
    row: dict[str, Any] = {
        "event": "check",
        "repo": spec.repo_id,
        "revision": spec.revision,
        "kind": spec.kind,
        "required": True,
        "cached": False,
        "path": None,
        "size_bytes": 0,
    }
    if spec.local_path is not None:
        path = spec.local_path
        if not Path(path).is_dir():
            return row
    else:
        try:
            path = _snapshot_download(spec, local_files_only=True)
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001 - any miss (incl. incomplete snapshot) = not cached
            logger.debug("prefetch: %s not cached (%s)", spec.repo_id, exc)
            return row
    try:
        spec = _resolved(spec, path)
    except (OSError, ValueError, KeyError) as exc:  # no/odd config.json = incomplete snapshot
        logger.debug("prefetch: %s config unreadable (%s)", spec.repo_id, exc)
        return row
    if not _complete(path, spec.required):
        return row
    row.update(cached=True, path=path, size_bytes=_snapshot_size(path))
    return row


def fetch_spec(spec: SnapshotSpec, emit: Callable[[dict], None] | None) -> dict[str, Any]:
    """Download one snapshot (resumable). ``emit`` set = JSON progress events."""
    if emit is not None:
        emit({"event": "start", "repo": spec.repo_id, "revision": spec.revision, "kind": spec.kind})
    if spec.local_path is not None:
        path = spec.local_path
        if not Path(path).is_dir():
            raise FileNotFoundError(f"local model directory does not exist: {path}")
        spec = _resolved(spec, path)
        if not _complete(path, spec.required):
            raise FileNotFoundError(f"{path}: local model is missing {', '.join(spec.required)}")
        result = {
            "event": "done",
            "repo": spec.repo_id,
            "path": path,
            "size_bytes": _snapshot_size(path),
        }
        if emit is not None:
            emit(result)
        return result
    if spec.resolve is not None:
        # Fetch the (tiny) config first to learn which weight files this repo
        # names, then fetch exactly those.
        config_only = replace(spec, allow_patterns=("config.json",), required=(), resolve=None)
        spec = _resolved(spec, _snapshot_download(config_only, local_files_only=False))
    reporter: threading.Thread | None = None
    done = threading.Event()
    if emit is not None:
        blobs = _repo_cache_dir(spec.repo_id) / "blobs"
        # Only files not yet in the cache count: a finished blob is named by its
        # id, a partial one ``<id>.incomplete``.
        pending = [
            (int(f[1]), f[2] if len(f) > 2 else None)
            for f in _remote_files(spec)
            if not ((f[2] if len(f) > 2 else None) and (blobs / f[2]).is_file())
        ]
        total = sum(size for size, _ in pending)
        files_total = len(pending)

        def measure() -> tuple[int, int]:
            got = n_done = 0
            for size, blob_id in pending:
                if blob_id is None:
                    continue  # unknown blob: counted only by the final event
                if (blobs / blob_id).is_file():
                    got += size
                    n_done += 1
                    continue
                with contextlib.suppress(OSError):
                    got += min(size, (blobs / f"{blob_id}.incomplete").stat().st_size)
            return got, n_done

        def progress(got: int, n_done: int) -> dict:
            return {
                "event": "progress",
                "repo": spec.repo_id,
                "downloaded_bytes": got,
                "total_bytes": total,
                "files_done": n_done,
                "files_total": files_total,
            }

        def report() -> None:
            while not done.wait(_PROGRESS_INTERVAL_S):
                emit(progress(*measure()))

        reporter = threading.Thread(target=report, name="prefetch-progress", daemon=True)
        reporter.start()
    try:
        path = _snapshot_download(spec, local_files_only=False)
    finally:
        done.set()
        if reporter is not None:
            reporter.join(timeout=1)
    if not _complete(path, spec.required):
        raise FileNotFoundError(f"{spec.repo_id}: snapshot is missing {', '.join(spec.required)}")
    if emit is not None:
        emit(progress(total, files_total))  # the download finished: report 100%
    result = {
        "event": "done",
        "repo": spec.repo_id,
        "path": path,
        "size_bytes": _snapshot_size(path),
    }
    if emit is not None:
        emit(result)
    return result


def run_prefetch(backend: str, model: str | None, *, check: bool, json_mode: bool) -> int:
    """CLI body; returns the process exit status."""
    from . import supervise

    if json_mode:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    ctx = supervise.json_channel() if json_mode else contextlib.nullcontext()
    with ctx:
        emit = supervise.emit if json_mode else None
        try:
            specs = (
                local_specs_for(backend, model)
                if model and is_local_model(model)
                else specs_for(backend, model)
            )
            return (
                _run_check(backend, model, specs, emit)
                if check
                else _run_fetch(backend, model, specs, emit)
            )
        except Exception as exc:  # noqa: BLE001 - CLI boundary: report, exit 1
            code, message = supervise.classify_startup_error(exc)
            if code == supervise.CODE_MODULE_NOT_FOUND:
                message = (
                    f"{message} (prefetch needs huggingface_hub, which every model extra installs)"
                )
            repo = getattr(exc, "_prefetch_repo", None)
            if emit is not None:
                emit({"event": "error", "code": code, "message": message, "repo": repo})
            else:
                print(f"stt_server: prefetch failed: {message}", file=sys.stderr)
            return 1


def _run_check(backend: str, model: str | None, specs: list[SnapshotSpec], emit) -> int:
    rows = [check_spec(s) for s in specs]
    summary = {
        "event": "summary",
        "backend": backend,
        "model": model,
        "cached": all(r["cached"] for r in rows),
        "size_bytes": sum(r["size_bytes"] for r in rows),
    }
    if emit is not None:
        for r in rows:
            emit(r)
        emit(summary)
        return 0
    for r in rows:
        print(
            f"{r['kind']:<6} {r['repo']}@{r['revision'] or 'main'}: {'cached' if r['cached'] else 'missing'}"
        )
    print(
        f"{backend}: {'cached' if summary['cached'] else 'not cached'}"
        f" ({summary['size_bytes'] / 1e6:.1f} MB on disk)"
    )
    return 0


def _run_fetch(backend: str, model: str | None, specs: list[SnapshotSpec], emit) -> int:
    total = 0
    for spec in specs:
        if emit is None:
            print(f"prefetch: {spec.kind} {spec.repo_id}@{spec.revision or 'main'}")
        try:
            total += fetch_spec(spec, emit)["size_bytes"]
        except ImportError:
            raise
        except Exception as exc:
            exc._prefetch_repo = spec.repo_id  # type: ignore[attr-defined]
            from . import supervise

            supervise.mark_download_phase(exc)  # a fetch failure is a download failure
            raise
    if emit is not None:
        emit(
            {
                "event": "complete",
                "backend": backend,
                "model": model,
                "cached": True,
                "size_bytes": total,
            }
        )
    else:
        print(f"prefetch: {backend} ready ({total / 1e6:.1f} MB)")
    return 0
