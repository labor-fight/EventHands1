"""Frozen-hash helper for the finished S45-line studies.

The S45-line tool chains (``tools/s45tr``, ``s45rt``, ``s45at``, ``s45ct``, ``s45px``, ``s45loc``, ``s45rf``,
``s45grade``, ``tools/c1sf``, ``tools/s45sm/common.py``) freeze the SHA256 of core source files in their protocols /
manifests and refuse to run when a file differs. The 2026-11-01 core cleanup removed the dead opt-in features of the
core training / model code, which necessarily changed those files. The finished studies' JSON protocols are history and
are never rewritten; instead ``tools/frozen_amendments.json`` lists, per changed or deleted file, the old digests that
the cleanup supersedes and the digest of the file after the cleanup:

    {"files": {"<repo-relative posix path>": {"old": ["<sha256 frozen in the protocols>", ...],
                                               "new": "<sha256 after the cleanup>" | null}}}

``new: null`` means the file was deleted by the cleanup.

``accepts(path, frozen_digest)`` is the one comparison to use wherever a frozen digest is checked:

* True when the file's current SHA256 equals ``frozen_digest`` (the unchanged-file case, as before);
* True when the amendments list ``frozen_digest`` among the old digests of that path and the current state of the file
  equals the amended ``new`` digest (``new`` null: the file must be absent).

Anything else (a file edited after the cleanup, an unlisted old digest, a deleted file that is not listed) is False.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Union

ROOT = Path(__file__).resolve().parents[1]
AMENDMENTS_PATH = Path(__file__).with_name("frozen_amendments.json")
#: directory name of the repository, used to re-root absolute paths recorded by another checkout of it
_REPO_NAME = "EventHands1"

PathLike = Union[str, Path]
_CACHE = {"mtime": None, "files": {}}


def sha256(path: PathLike) -> Optional[str]:
    """SHA256 hex digest of a file, or None when it does not exist."""
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: PathLike) -> str:
    """Repository-relative posix path of ``path`` (the key used in ``frozen_amendments.json``)."""
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        parts = p.parts
        if _REPO_NAME in parts:
            return Path(*parts[len(parts) - parts[::-1].index(_REPO_NAME):]).as_posix()
        return p.as_posix()


def amendments() -> dict:
    """The ``files`` mapping of ``frozen_amendments.json`` (empty when the file is absent); re-read on change."""
    try:
        mtime = AMENDMENTS_PATH.stat().st_mtime_ns
    except OSError:
        return {}
    if _CACHE["mtime"] != mtime:
        data = json.loads(AMENDMENTS_PATH.read_text())
        _CACHE["files"] = dict(data.get("files", {}))
        _CACHE["mtime"] = mtime
    return _CACHE["files"]


def accepts(path: PathLike, frozen_digest: Optional[str]) -> bool:
    """True when ``path`` is the file frozen as ``frozen_digest`` or its listed, amended successor (see module doc)."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    current = sha256(p)
    if frozen_digest is not None and current == frozen_digest:
        return True
    entry = amendments().get(relpath(p))
    if entry is None or frozen_digest is None or frozen_digest not in entry.get("old", ()):
        return False
    return current == entry.get("new")


__all__ = ["accepts", "amendments", "relpath", "sha256"]
