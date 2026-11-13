#!/usr/bin/env python3
"""ABL gates (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): CPU jobs of PMR driver 7 that wait for the RDR screen and
the RDR trunk reading and open exactly one ablation path. Exit 0 = open, exit 3 = closed by the written rule.

    python -m tools.pmr.abl wait-routers                      v2 and v3 routers finished (fails on a traceback)
    python -m tools.pmr.abl gate-router --router v2|v3         open iff the screen selected this router
    python -m tools.pmr.abl gate-path --path 1 --router v2|v3  open iff that router was selected AND its trunk is TRUNK_BETTER
    python -m tools.pmr.abl gate-path --path 2                 open iff no router was selected, or the selected router's
                                                               trunk is not TRUNK_BETTER (or its chain failed)
    python -m tools.pmr.abl nop1                               open unless path 1 has been opened (late check before every
                                                               stage of a speculative path-2 chain)
Written rules: the screen (#rdr-prereg-20261110, tools/rdr/decide.py, decisions/screen_a.json); TRUNK_BETTER
(#rdr-trunk-prereg-20261110) = against EvHandTrack seed 3407 in the streaming form, RA down >= 0.5 and MPJPE-L, MPJPE-G
not worse by more than 0.3, on `d1rv<k>_rol5`'s streaming e1 reading.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.pmr.common import OUT, now, read  # noqa: E402

RDR = ROOT / "outputs/rdr/20261110"
SCREEN = RDR / "decisions/screen_a.json"
STATUS = OUT / "driver7/status.json"
DECISIONS = OUT / "decisions"
REF = OUT / "readings/pmr_d1r_rol5_s3407__str/e1/report.json"
CLOSED = 3
TERMINAL_BAD = ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED")
RA_DROP, LG_TOL = 0.5, 0.3


def five(report):
    s = report["per_sequence"]
    return [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], report["overall"]["mpjpe_ra_mm"]]


def trunk_better(new, ref):
    """The written TRUNK_BETTER rule (pure). new / ref: five metrics [L, G, VL, VG, RA]."""
    d = [a - b for a, b in zip(new, ref)]
    return bool(d[4] <= -RA_DROP and d[0] <= LG_TOL and d[1] <= LG_TOL), d


def wait_for(path, timeout_s=6 * 3600, poll=20):
    t0 = time.time()
    while not Path(path).exists():
        if time.time() - t0 > timeout_s:
            raise TimeoutError(str(path))
        time.sleep(poll)


def wait_routers():
    t0 = time.time()
    while True:
        done = [(RDR / f"routers/{v}/{v}_final.json").exists() for v in ("v2", "v3")]
        for v in ("v2", "v3"):
            log = RDR / f"logs/{v}_formal.log"
            if log.exists() and "Traceback" in log.read_text(errors="replace"):
                print("router training failed:", v, flush=True)
                sys.exit(1)
        if all(done):
            print("routers done", now(), flush=True)
            return
        if time.time() - t0 > 4 * 3600:
            sys.exit(1)
        time.sleep(20)


def selected():
    wait_for(SCREEN)
    return read(SCREEN)["selected"]


def trunk_result(router):
    """(better, deltas, reason) once the selected router's trunk chain has its streaming reading or has failed."""
    key = f"t:fread:d1rv{router[1:]}_rol5:str:e1"
    prefix = f"t:"
    while True:
        st = read(STATUS)["jobs"] if STATUS.exists() else {}
        mine = {k: v for k, v in st.items() if k.startswith(prefix) and f"d1rv{router[1:]}" in k}
        if any(v["status"] in TERMINAL_BAD for v in mine.values()):
            return False, None, "trunk chain failed or skipped: " + ", ".join(k for k, v in mine.items()
                                                                               if v["status"] in TERMINAL_BAD)
        if st.get(key, {}).get("status") == "EXIT_0":
            new = five(read(OUT / f"readings/pmr_d1rv{router[1:]}_rol5_s3407__str/e1/report.json"))
            better, d = trunk_better(new, five(read(REF)))
            return better, {"new": new, "delta": d}, "TRUNK_BETTER" if better else "not TRUNK_BETTER"
        time.sleep(30)


def record(name, obj):
    DECISIONS.mkdir(parents=True, exist_ok=True)
    p = DECISIONS / name
    if not p.exists():
        p.write_text(json.dumps({**obj, "written_utc": now()}, indent=1) + "\n")


def gate_router(router):
    sel = selected()
    print("screen selected", sel, flush=True)
    sys.exit(0 if sel == router else CLOSED)


def gate_path(path, router=None):
    sel = selected()
    if path == 1:
        if sel != router:
            record(f"abl_gate_p1_{router}.json", {"open": False, "selected": sel, "reason": "router not selected"})
            sys.exit(CLOSED)
        better, nums, reason = trunk_result(router)
        record(f"abl_gate_p1_{router}.json", {"open": better, "selected": sel, "reason": reason, "numbers": nums})
        print("path 1", router, reason, nums, flush=True)
        sys.exit(0 if better else CLOSED)
    if sel is None:
        record("abl_gate_p2.json", {"open": True, "selected": None, "reason": "no router selected"})
        print("path 2: no router selected", flush=True)
        sys.exit(0)
    better, nums, reason = trunk_result(sel)
    record("abl_gate_p2.json", {"open": not better, "selected": sel, "reason": reason, "numbers": nums})
    print("path 2", "closed" if better else "open", reason, nums, flush=True)
    sys.exit(CLOSED if better else 0)


def nop1():
    opened = [p.name for p in DECISIONS.glob("abl_gate_p1_v*.json") if read(p).get("open")]
    print("path 1 opened" if opened else "path 1 not opened", opened, flush=True)
    sys.exit(CLOSED if opened else 0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("wait-routers")
    s = sub.add_parser("gate-router")
    s.add_argument("--router", required=True, choices=("v2", "v3"))
    s = sub.add_parser("gate-path")
    s.add_argument("--path", type=int, required=True, choices=(1, 2))
    s.add_argument("--router", choices=("v2", "v3"))
    sub.add_parser("nop1")
    a = p.parse_args(argv)
    if a.verb == "wait-routers":
        wait_routers()
    elif a.verb == "gate-router":
        gate_router(a.router)
    elif a.verb == "gate-path":
        if a.path == 1 and not a.router:
            p.error("--router is required for path 1")
        gate_path(a.path, a.router)
    else:
        nop1()


if __name__ == "__main__":
    main()
