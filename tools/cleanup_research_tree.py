#!/usr/bin/env python3
"""Audit and optionally remove generated/retired EventHands research artifacts.

Dry-run is the default. Destructive source-code deletion is intentionally hard:
a .py file must be passed explicitly with --failed-file, --allow-source-delete
must be set, and git-grep must find no references outside the file itself.
"""

from __future__ import print_function

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple


AUTO_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
AUTO_FILE_NAMES = {"nohup.out"}
AUTO_FILE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".swp"}
PROTECTED_TOP_LEVEL = {
    ".git",
    "data",
    "real_eval_data",
    "synth_eval_data",
}
PROTECTED_BRANCHES = {"main", "master"}


def _run(args: Sequence[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(
        list(args),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_state(root: Path) -> Dict[str, object]:
    branch_rc, branch, branch_err = _run(
        ["git", "branch", "--show-current"], root
    )
    head_rc, head, head_err = _run(["git", "rev-parse", "HEAD"], root)
    status_rc, status, status_err = _run(
        ["git", "status", "--short"], root
    )
    return {
        "is_git_repo": branch_rc == 0 and head_rc == 0,
        "branch": branch if branch_rc == 0 else None,
        "head": head if head_rc == 0 else None,
        "status_short": status.splitlines() if status_rc == 0 and status else [],
        "errors": [
            item for item in (branch_err, head_err, status_err) if item
        ],
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _has_protected_top_level(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return bool(parts and parts[0] in PROTECTED_TOP_LEVEL)


def _validate_explicit(
    raw_path: str, root: Path, expected: str
) -> Tuple[Optional[Path], Optional[str]]:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate

    # Resolve the parent, but do not follow the final component before checking it.
    parent = candidate.parent.resolve()
    candidate = parent / candidate.name

    if not _is_within(candidate, root):
        return None, "outside repository root"
    if candidate == root:
        return None, "refusing repository root"
    if _has_protected_top_level(candidate, root):
        return None, "protected top-level path"
    if not candidate.exists() and not candidate.is_symlink():
        return None, "path does not exist"
    if candidate.is_symlink():
        return None, "symlink deletion is refused"

    if expected == "dir" and not candidate.is_dir():
        return None, "expected directory"
    if expected == "file" and not candidate.is_file():
        return None, "expected file"
    return candidate, None


def _walk_generated(
    root: Path, include_logs: bool
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    candidates: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    for current, dirnames, filenames in os.walk(str(root), topdown=True):
        current_path = Path(current)

        kept_dirs: List[str] = []
        for dirname in dirnames:
            path = current_path / dirname
            if current_path == root and dirname in PROTECTED_TOP_LEVEL:
                skipped.append(
                    {
                        "path": _relative(path, root),
                        "reason": "protected top-level path",
                    }
                )
                continue
            if path.is_symlink():
                skipped.append(
                    {
                        "path": _relative(path, root),
                        "reason": "symlink directory",
                    }
                )
                continue
            if dirname in AUTO_DIR_NAMES:
                candidates.append(
                    {
                        "path": _relative(path, root),
                        "kind": "directory",
                        "reason": "generated cache directory",
                    }
                )
                # Do not descend into a directory that will be removed.
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                skipped.append(
                    {
                        "path": _relative(path, root),
                        "reason": "symlink file",
                    }
                )
                continue
            suffix = path.suffix.lower()
            reason: Optional[str] = None
            if filename in AUTO_FILE_NAMES:
                reason = "generated runtime file"
            elif suffix in AUTO_FILE_SUFFIXES or filename.endswith("~"):
                reason = "generated temporary/cache file"
            elif include_logs and suffix == ".log":
                reason = "explicitly included log file"
            if reason:
                candidates.append(
                    {
                        "path": _relative(path, root),
                        "kind": "file",
                        "reason": reason,
                    }
                )
    return candidates, skipped


def _git_references(root: Path, target: Path) -> List[str]:
    """Return textual references to a Python source outside the file itself."""
    rel = _relative(target, root)
    stem = target.stem
    module_path = ".".join(Path(rel).with_suffix("").parts)
    patterns = sorted(
        {
            target.name,
            stem,
            module_path,
            "from {0} import".format(module_path),
            "import {0}".format(module_path),
        }
    )

    references = set()
    for pattern in patterns:
        rc, out, _ = _run(
            ["git", "grep", "-n", "-F", "--", pattern], root
        )
        if rc not in (0, 1):
            continue
        for line in out.splitlines():
            if not line:
                continue
            referenced_path = line.split(":", 1)[0]
            if referenced_path != rel:
                references.add(line)
    return sorted(references)


def _deduplicate(
    candidates: List[Dict[str, str]], root: Path
) -> List[Dict[str, str]]:
    # Explicit parent directories subsume child files/caches.
    unique: Dict[str, Dict[str, str]] = {}
    for item in candidates:
        unique[item["path"]] = item

    items = list(unique.values())
    dir_paths = sorted(
        [
            root / item["path"]
            for item in items
            if item["kind"] == "directory"
        ],
        key=lambda path: len(path.parts),
    )

    result: List[Dict[str, str]] = []
    for item in items:
        path = root / item["path"]
        if any(
            parent != path and _is_within(path, parent)
            for parent in dir_paths
        ):
            continue
        result.append(item)

    # Delete deepest paths first.
    return sorted(
        result,
        key=lambda item: len((root / item["path"]).parts),
        reverse=True,
    )


def _delete_candidate(root: Path, item: Dict[str, str]) -> Optional[str]:
    path = root / item["path"]
    try:
        if path.is_symlink():
            return "symlink encountered at apply time"
        if item["kind"] == "directory":
            if path.exists():
                shutil.rmtree(str(path))
        else:
            if path.exists():
                path.unlink()
        return None
    except Exception as exc:  # pragma: no cover - defensive reporting
        return "{0}: {1}".format(type(exc).__name__, exc)


def _write_report(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit generated artifacts and explicitly named failed experiment "
            "paths. Dry-run by default."
        )
    )
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--report",
        default="artifacts/cleanup/cleanup_audit.json",
        help="JSON audit report path, relative to root unless absolute.",
    )
    parser.add_argument(
        "--failed-dir",
        action="append",
        default=[],
        help="Explicit retired experiment directory; repeat as needed.",
    )
    parser.add_argument(
        "--failed-file",
        action="append",
        default=[],
        help="Explicit retired file; repeat as needed.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include every .log file in the candidate list.",
    )
    parser.add_argument(
        "--allow-source-delete",
        action="store_true",
        help=(
            "Allow an explicitly named .py file to be deleted, but only if "
            "git-grep finds no external references."
        ),
    )
    parser.add_argument(
        "--allow-protected-branch",
        action="store_true",
        help="Allow --apply on main/master. Strongly discouraged.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete candidates. Without this flag, only audit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(
            "ERROR: root is not a directory: {0}".format(root),
            file=sys.stderr,
        )
        return 2

    git_state = _git_state(root)
    if not git_state["is_git_repo"]:
        print(
            "ERROR: root is not a Git repository: {0}".format(root),
            file=sys.stderr,
        )
        return 2

    branch = git_state.get("branch")
    if (
        args.apply
        and branch in PROTECTED_BRANCHES
        and not args.allow_protected_branch
    ):
        print(
            "ERROR: refusing --apply on protected branch {0!r}; "
            "use a research branch or pass --allow-protected-branch.".format(
                branch
            ),
            file=sys.stderr,
        )
        return 2

    candidates, skipped = _walk_generated(root, args.include_logs)
    errors: List[Dict[str, object]] = []

    for raw in args.failed_dir:
        path, error = _validate_explicit(raw, root, "dir")
        if error:
            errors.append({"path": raw, "error": error})
            continue
        assert path is not None
        candidates.append(
            {
                "path": _relative(path, root),
                "kind": "directory",
                "reason": "explicit retired experiment directory",
            }
        )

    for raw in args.failed_file:
        path, error = _validate_explicit(raw, root, "file")
        if error:
            errors.append({"path": raw, "error": error})
            continue
        assert path is not None

        if path.suffix.lower() == ".py":
            if not args.allow_source_delete:
                errors.append(
                    {
                        "path": _relative(path, root),
                        "error": "Python source requires --allow-source-delete",
                    }
                )
                continue
            refs = _git_references(root, path)
            if refs:
                errors.append(
                    {
                        "path": _relative(path, root),
                        "error": "source still has git-grep references",
                        "references": refs[:100],
                    }
                )
                continue

        candidates.append(
            {
                "path": _relative(path, root),
                "kind": "file",
                "reason": "explicit retired file",
            }
        )

    candidates = _deduplicate(candidates, root)

    deleted: List[Dict[str, str]] = []
    if args.apply:
        for item in candidates:
            error = _delete_candidate(root, item)
            if error:
                errors.append({"path": item["path"], "error": error})
            else:
                deleted.append(item)

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path
    if not _is_within(report_path.resolve().parent, root):
        print(
            "ERROR: report path must be inside repository root",
            file=sys.stderr,
        )
        return 2

    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "root": str(root),
        "mode": "apply" if args.apply else "dry-run",
        "git": git_state,
        "options": {
            "include_logs": bool(args.include_logs),
            "allow_source_delete": bool(args.allow_source_delete),
            "allow_protected_branch": bool(args.allow_protected_branch),
        },
        "candidates": candidates,
        "deleted": deleted,
        "skipped": skipped,
        "errors": errors,
        "summary": {
            "candidate_count": len(candidates),
            "deleted_count": len(deleted),
            "skipped_count": len(skipped),
            "error_count": len(errors),
        },
    }
    _write_report(report_path, payload)

    print(
        "{mode}: {candidates} candidates, {deleted} deleted, "
        "{errors} errors; report={report}".format(
            mode=payload["mode"],
            candidates=len(candidates),
            deleted=len(deleted),
            errors=len(errors),
            report=_relative(report_path, root),
        )
    )

    # Explicit path mistakes/references are a failed cleanup audit.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
