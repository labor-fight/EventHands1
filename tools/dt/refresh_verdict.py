#!/usr/bin/env python3
"""Refresh generated regions inside the consolidated DT verdict section.

    python tools/dt/refresh_verdict.py TABLES="python tools/dt/verdict_tables.py" GATES="cat outputs/dt/reports/screen_6k.md"
Each argument is NAME=command; the command's stdout replaces the region.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.experiment_records import demote_headings, get_section, replace_section

DOC = REPO / "docs" / "EXPERIMENT_RECORDS.md"


def main() -> None:
    if len(sys.argv) == 1:
        print("refreshed", [])
        return
    snapshot = get_section(DOC, "DT-VERDICT")
    text = str(snapshot)
    for arg in sys.argv[1:]:
        name, cmd = arg.split("=", 1)
        b, e = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
        if text.count(b) != 1 or text.count(e) != 1:
            raise ValueError(f"markers for {name} must occur exactly once in DT-VERDICT")
        i, j = text.index(b) + len(b), text.index(e)
        if j < i:
            raise ValueError(f"END marker for {name} precedes its BEGIN marker")
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO)
        if out.returncode != 0:
            raise RuntimeError(out.stderr or f"command for {name} failed: {out.returncode}")
        generated = demote_headings(out.stdout).strip("\n")
        text = text[:i] + "\n" + generated + "\n" + text[j:]
        if text.count(b) != 1 or text.count(e) != 1:
            raise ValueError(f"generated content duplicated markers for {name}")
    replace_section(DOC, "DT-VERDICT", text, expected=snapshot)
    print("refreshed", [a.split("=", 1)[0] for a in sys.argv[1:]])


if __name__ == "__main__":
    main()
