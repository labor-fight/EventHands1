#!/usr/bin/env python3
"""Give borrowed PMR trainings enough file descriptors, then use the original attempt wrapper.

The system service's default soft limit is 1024. Closed-loop rol5 data loading
needs more descriptors. Raising this process's soft limit within its existing
hard limit needs no change to the system helper or to the training recipe.
"""
from __future__ import annotations

from pathlib import Path
import resource
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NOFILE_TARGET = 65536


def raise_nofile_limit() -> tuple[int, int]:
    """Raise only the soft descriptor limit and verify the effective limits."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = NOFILE_TARGET if hard == resource.RLIM_INFINITY else min(NOFILE_TARGET, hard)
    requested = soft if soft == resource.RLIM_INFINITY else max(soft, target)
    if requested != soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (requested, hard))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"borrowed PMR training cannot raise RLIMIT_NOFILE soft limit "
                f"from {soft} to {requested} (hard={hard}): {exc}"
            ) from exc
    actual_soft, actual_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if (actual_hard != hard or
            actual_soft != resource.RLIM_INFINITY and actual_soft < requested):
        raise RuntimeError(
            f"borrowed PMR training has insufficient RLIMIT_NOFILE: "
            f"expected soft>={requested}, hard={hard}; "
            f"got soft={actual_soft}, hard={actual_hard}"
        )
    return actual_soft, actual_hard


def main(argv=None):
    raise_nofile_limit()
    from tools.pmr import borrow_job

    return borrow_job.main(argv)


if __name__ == "__main__":
    main()
