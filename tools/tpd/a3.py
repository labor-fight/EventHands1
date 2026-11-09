#!/usr/bin/env python3
"""TPD amendment-3 entry (docs/EXPERIMENT_RECORDS.md#tpd-amend3-20261105): run a frozen TPD entry unchanged inside
tools/tpd/ns_amend3.py.

    python tools/tpd/a3.py train_rol --config <amend3 config>        tools/tpd/train_rol.py
    python tools/tpd/a3.py train_final --config <amend3 config>      tools/tpd/train_final.py
    python tools/tpd/a3.py run_rol | run_final | run_rolc | analyze_rol | analyze_syn <args>
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tpd import ns_amend3  # noqa: E402

ENTRIES = ("train_rol", "train_final", "run_rol", "run_final", "run_rolc", "analyze_rol", "analyze_syn")

def entry_of(argv):
    """The entry and its arguments. The training entries reset sys.argv to [a3.py, --config, <path>] and Lightning
    re-executes that for DDP rank 1, so `--config` alone dispatches by the config (a FINAL block: train_final)."""
    if argv[1] == "--config":
        import yaml
        cfg = yaml.safe_load(Path(argv[2]).read_text())
        return ("train_final" if "FINAL" in cfg else "train_rol"), argv[1:]
    return argv[1], argv[2:]


if __name__ == "__main__":
    ns_amend3.enter()
    which, rest = entry_of(sys.argv)
    if which not in ENTRIES:
        raise SystemExit("unknown entry " + which)
    sys.argv = [sys.argv[0], *rest] if which.startswith("train_") else sys.argv
    import importlib
    mod = importlib.import_module(f"tools.tpd.{which}")
    if which.startswith("train_"):
        mod.main()
    else:
        mod.main(rest)
