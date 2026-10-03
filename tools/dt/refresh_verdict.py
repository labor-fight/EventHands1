#!/usr/bin/env python3
"""Replace the generated regions of docs/DT_RENDER_TRACK_VERDICT.md: <!-- NAME:BEGIN --> ... <!-- NAME:END -->.

    python tools/dt/refresh_verdict.py TABLES="python tools/dt/verdict_tables.py" GATES="cat outputs/dt/reports/screen_6k.md"
Each argument is NAME=command; the command's stdout replaces the region.
"""
import subprocess
import sys
from pathlib import Path

DOC = Path(__file__).resolve().parents[2] / "docs" / "DT_RENDER_TRACK_VERDICT.md"
text = DOC.read_text()
for arg in sys.argv[1:]:
    name, cmd = arg.split("=", 1)
    b, e = f"<!-- {name}:BEGIN -->", f"<!-- {name}:END -->"
    assert b in text and e in text, f"markers for {name} missing"
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=DOC.parents[1])
    assert out.returncode == 0, out.stderr
    i, j = text.index(b) + len(b), text.index(e)
    text = text[:i] + "\n" + out.stdout.strip("\n") + "\n" + text[j:]
DOC.write_text(text)
print("refreshed", [a.split("=", 1)[0] for a in sys.argv[1:]])
