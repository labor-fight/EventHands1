#!/usr/bin/env python3
"""PFO2 stage-1 written rules (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103 stage-1 structure, applied by
#pfo2-s1-prereg-20261103); CPU only, read-only on the readings.

    python -m tools.pfo2.decide_s1 --plan tools/pfo2/plan_s1.json [--final]

The plan names the base and seeds and the stage-1 arms by family:
  {"base": "fl1", "seeds": [3407, 3408],
   "gnn":  {family: {"full": arm, "self": arm, "rewire": arm}},
   "main": {family: arm},
   "router": {family: {"router": name, "seeds": [3407, 3408, 3409]}}}     (zero-training swaps into the base)
Readings (step 6000): E1 five metrics, E3 overall RA, pert local R15 / F15 retention (median).
  load-bearing (gnn family): full better than self and than rewire in RA by >= 0.3 mm (paired-seed means) with every seed
                             better than both, and full's local R15 and F15 (seed means) both >= 0.05 below the base's
  kill (main / router):      every seed's local R15 > 0.35, or the seed-mean E3 RA worse than the base's by > 0.3 mm
  promote:                   not killed, seed-mean RA better than the base's by >= 0.5 mm and every seed better
                             (gnn families additionally need load-bearing to enter the merge)
Writes decisions/s1.json and analysis/s1.md (once; `--final` after every reading, otherwise a dry status to stdout).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import OUT, S43_ENGINE, now, read, require, save  # noqa: E402

LOAD_MARGIN, RETENTION_DROP = 0.3, 0.05
KILL_R15, KILL_E3 = 0.35, 0.3
PROMOTE = 0.5


def rep(tag, task):
    p = OUT / "readings" / tag / task / "report.json"
    return read(p) if p.exists() else None


def metrics(tag):
    e1, e3, pt = rep(tag, "e1"), rep(tag, "e3"), rep(tag, "pert")
    if e1 is None or e3 is None or pt is None:
        return None
    s = e1["per_sequence"]
    loc = pt["per_sequence"]["zgz_local"]
    return {"five": [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
                     s["zgz_global"]["mpvpe_ra_mm"], e1["overall"]["mpjpe_ra_mm"]],
            "e3": e3["overall"]["mpjpe_ra_mm"], "R15": loc["R15"]["rho_median"], "F15": loc["F15"]["rho_median"]}


def gap(five):
    return float(sum(max(0.0, a - b) for a, b in zip(five, S43_ENGINE)))


def tag(arm, seed, router=None):
    return f"pfo2_{arm}_s{seed}" + (f"+{router}" if router else "")


def judge(base, arm_tags, seeds):
    """Kill / promote of an arm against the base, paired seeds."""
    rows = {s: (metrics(tag(base, s)), metrics(arm_tags[s])) for s in seeds}
    if any(b is None or a is None for b, a in rows.values()):
        return {"status": "pending"}
    gain = {s: rows[s][0]["five"][4] - rows[s][1]["five"][4] for s in seeds}
    e3 = float(np.mean([rows[s][1]["e3"] - rows[s][0]["e3"] for s in seeds]))
    killed = all(rows[s][1]["R15"] > KILL_R15 for s in seeds) or e3 > KILL_E3
    mean_gain = float(np.mean(list(gain.values())))
    promoted = (not killed) and mean_gain >= PROMOTE and all(g > 0 for g in gain.values())
    arm_mean = np.mean([rows[s][1]["five"] for s in seeds], 0)
    base_mean = np.mean([rows[s][0]["five"] for s in seeds], 0)
    return {"status": "complete", "ra_gain": gain, "mean_ra_gain": mean_gain, "e3_change": e3,
            "R15": {s: rows[s][1]["R15"] for s in seeds}, "F15": {s: rows[s][1]["F15"] for s in seeds},
            "base_R15": {s: rows[s][0]["R15"] for s in seeds}, "base_F15": {s: rows[s][0]["F15"] for s in seeds},
            "five_mean": arm_mean.tolist(), "base_five_mean": base_mean.tolist(),
            "gap_sum": gap(arm_mean), "base_gap_sum": gap(base_mean), "killed": bool(killed), "promoted": bool(promoted)}


def load_bearing(base, fam, seeds):
    m = {k: {s: metrics(tag(a, s)) for s in seeds} for k, a in fam.items()}
    b = {s: metrics(tag(base, s)) for s in seeds}
    if any(v is None for d in m.values() for v in d.values()) or any(v is None for v in b.values()):
        return {"status": "pending"}
    ra = {k: {s: m[k][s]["five"][4] for s in seeds} for k in m}
    over = {c: {s: ra[c][s] - ra["full"][s] for s in seeds} for c in ("self", "rewire")}
    d_r15 = float(np.mean([b[s]["R15"] for s in seeds]) - np.mean([m["full"][s]["R15"] for s in seeds]))
    d_f15 = float(np.mean([b[s]["F15"] for s in seeds]) - np.mean([m["full"][s]["F15"] for s in seeds]))
    ok = all(np.mean(list(over[c].values())) >= LOAD_MARGIN and all(v > 0 for v in over[c].values())
             for c in over) and d_r15 >= RETENTION_DROP and d_f15 >= RETENTION_DROP
    return {"status": "complete", "full_minus_controls": over, "R15_drop": d_r15, "F15_drop": d_f15,
            "load_bearing": bool(ok)}


def run(plan, final):
    base, seeds = plan["base"], plan["seeds"]
    out = {"plan": plan, "gnn": {}, "main": {}, "router": {}}
    for fam, arms in plan.get("gnn", {}).items():
        out["gnn"][fam] = {"load": load_bearing(base, arms, seeds),
                           **{k: judge(base, {s: tag(a, s) for s in seeds}, seeds) for k, a in arms.items()}}
    for fam, arm in plan.get("main", {}).items():
        out["main"][fam] = judge(base, {s: tag(arm, s) for s in seeds}, seeds)
    for fam, r in plan.get("router", {}).items():
        out["router"][fam] = judge(base, {s: tag(base, s, r["router"]) for s in r["seeds"]}, r["seeds"])
    merge = [f"main:{f}" for f, j in out["main"].items() if j.get("promoted")]
    merge += [f"router:{f}" for f, j in out["router"].items() if j.get("promoted")]
    merge += [f"gnn:{f}" for f, j in out["gnn"].items() if j["load"].get("load_bearing") and j["full"].get("promoted")]
    pending = [k for sec in ("gnn", "main", "router") for k, v in out[sec].items()
               if (v.get("status") if sec != "gnn" else v["load"].get("status")) == "pending"]
    out.update(merge=merge, pending=pending, stop_ii=(not merge and not pending), decided_at_utc=now())
    if not final:
        print({k: out[k] for k in ("merge", "pending", "stop_ii")})
        return out
    require(not pending, f"readings pending: {pending}")
    save(OUT / "decisions" / "s1.json", out, exclusive=True)
    fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    lines = ["## 第 1 阶段判读（两种子配对，对 B2）", "", "| 臂 | 五项均值 | RA 增益（逐种子） | E3 变化 | local R15 / F15 | 杀死 | 晋级 |",
             "|---|---|---|---|---|---|---|"]
    for sec in ("gnn", "main", "router"):
        for fam, v in out[sec].items():
            items = [(f"{fam}:{k}", j) for k, j in v.items() if k != "load"] if sec == "gnn" else [(fam, v)]
            for name, j in items:
                lines.append(f"| {sec}:{name} | {fmt(j['five_mean'])} | {', '.join(f'{g:+.2f}' for g in j['ra_gain'].values())} | "
                             f"{j['e3_change']:+.2f} | {np.mean(list(j['R15'].values())):.2f} / {np.mean(list(j['F15'].values())):.2f} | "
                             f"{j['killed']} | {j['promoted']} |")
    for fam, v in out["gnn"].items():
        L = v["load"]
        lines.append(f"\n承重 {fam}：full − self / rewire = {L['full_minus_controls']}，R15 / F15 下降 {L['R15_drop']:.3f} / "
                     f"{L['F15_drop']:.3f} → {L['load_bearing']}")
    lines.append(f"\n合并集：{merge or '空（停止点 (ii)）'}")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "s1.md").write_text("\n".join(lines) + "\n")
    print(merge, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--final", action="store_true")
    a = p.parse_args(argv)
    run(read(a.plan), a.final)


if __name__ == "__main__":
    main()
