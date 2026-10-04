#!/usr/bin/env python3
"""DT2: the "deployment form" main-table row (docs/DT2_PREREG.md section 6).

Accuracy cells = the adaptive-filter row (outputs/semkine/dt2_dz_l3_dt2afilt_main_row.json); the deployment form is
bit-identical to it (outputs/dt2/reports/deploy_check.json: DeployTracker steps == the recorded filter_eval steps, every
seed). Latency cell = the full deployment loop of tools/dt/bench_loop.py variant v5 on an idle GPU (events in host memory ->
pinned buffer -> one H2D -> GPU LNES -> forward -> GPU AdaptiveFilter, all in one CUDA graph replay -> one D2H) at the
deployment cadence (`--mode hz20`: 20 Hz, 45 ms idle before each packet; `b2b` = back to back), the larger of the global /
local p50, raw milliseconds, NOT scaled by the 1.75 ms anchor. The value is written into
the `latency_ms_scaled_full1p75` key only because tools/report_table.py reads that key; `latency_kind` says what it is.

    python tools/dt/deploy_row.py --bench outputs/dt2/reports/D_bench_idle.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "semkine"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(REPO / "outputs/dt2/reports/D_bench_idle.json"))
    ap.add_argument("--check", default=str(REPO / "outputs/dt2/reports/deploy_check.json"))
    ap.add_argument("--src", default="dt2_dz_l3_dt2afilt")
    ap.add_argument("--arm", default="dt2_dz_l3_deploy")
    ap.add_argument("--variant", default="v5")
    ap.add_argument("--mode", default="hz20", choices=("hz20", "b2b"))
    a = ap.parse_args()
    row = json.loads((OUT / f"{a.src}_main_row.json").read_text())
    bench = json.loads(Path(a.bench).read_text())
    check = json.loads(Path(a.check).read_text())
    bad = {k: v for k, v in check.items() if v["steps_bit_identical"] != v["steps"]}
    if bad or sorted(check) != sorted(Path(v["ckpt"]).parent.name for v in row["per_seed"].values()):
        raise SystemExit(f"deploy_check does not cover the row's seeds bit for bit: {sorted(check)} / {bad}")
    t = bench["table"][a.variant]
    p50 = max(t["global"][a.mode]["p50"], t["local"][a.mode]["p50"])
    row.update({
        "arm": a.arm, "source_row": a.src, "latency_kind": f"deployment loop {a.variant}, idle GPU, {a.mode}, max(global p50, local p50), "
        "raw ms (not scaled)", "latency_ms_scaled_full1p75": p50,
        "deploy_latency": {k: {m: t[k][m] for m in t[k]} for k in t}, "deploy_bench": str(a.bench),
        "deploy_check": check,
    })
    (OUT / f"{a.arm}_main_row.json").write_text(json.dumps(row, indent=1))
    print(json.dumps({k: row[k] for k in ("arm", "latency_kind", "latency_ms_scaled_full1p75")}, indent=1))


if __name__ == "__main__":
    main()
