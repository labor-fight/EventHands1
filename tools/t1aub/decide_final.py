#!/usr/bin/env python3
"""T1A-UB final paired comparison (CPU only; once; the rule written in docs/EXPERIMENT_RECORDS.md#t1aub-r3-prereg-20261102
before any reading of either replicate).

    python tools/t1aub/decide_final.py --x-legacy NET --x-dedicated NET --block B
        -> outputs/t1aub/20261102/summary_final.json, report_final.md

Candidate X and T1a each trained on two sampling streams (legacy_none, dedicated). Per stream s: Delta_s = T1a_s - X_s on the
main row (local / global / RA; X's rows and T1a's (`t1aref_<B>`) from block B, the replicate t1arep's from the block it was
measured in) and on the in-sample medians (three checkpoints). REPRODUCED: on both streams local better >= 5 mm and RA better >= 3.1 mm and X's C <= 0.8;
DIRECTIONAL: same sign on both streams but at least one within the noise floor; NOT_REPRODUCED: opposite signs. Also the
mean over the two streams and half their difference (descriptive), and the deployment costs of X against T1a in block B.

Also (written in docs/EXPERIMENT_RECORDS.md#t1aub-r5-prereg-20261102): the best deployable configuration (X on the legacy
stream when all three of its costs are within budget in block B; otherwise the in-budget network with C <= 0.8 and the
lowest main-row RA over rounds 1-5, label-assisted networks excluded), and the descriptive decomposition of X's remaining
gap with its trunk fixed: router accuracy (X - the oracle cell `x5g`, label routes at test) and router capacity (X - the same
trunk with T0P's routes), on zgz E1 at step 6000 and the in-sample medians.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402

TH = {"local": 5.0, "ra": 3.1, "C": 0.8}


def reading(net, block):
    """decide.py's readings; the T1a replicate's row is read in the block it was measured in (accuracy does not depend on
    the block; the budget is always checked against T1a re-measured in block B)."""
    if net == "t1arep":
        own = C.read(C.OUT / "evaluations" / net / "row_step6000/report.json")["block"]
        return {"zgz": D.zgz(net), "insample": D.insample(net), "fixed_point": D.fixed_point(net),
                "single_step": D.single_step(net), "retention": D.retention(net), "row": D.main_row(net, own),
                "row_block": own}
    return D.readings(net, block)


def label_at_test(net):
    """True for label-assisted networks (label routes or label features at test time): never 'deployable'."""
    from tools.t1aub import r5
    if r5.is_r5(net):
        return True
    arms = C.registry()["arms"]
    if net in arms:
        s = arms[net]
        return bool(s.get("gtfeat")) or (s.get("route") or {}).get("router") == "gt"
    return False


def best_deployable(cmp, block, extra):
    """X (legacy) if within budget in block B, else the lowest main-row RA among in-budget networks with C <= 0.8."""
    leg = cmp["streams"]["legacy_none"]
    if leg["budget_ok"]:
        return {"network": leg["candidate"], "rule": "X within all three budgets in block " + block}
    pool = {}
    s1 = C.read(C.OUT / "summary_r1.json")
    for n, v in s1["networks"].items():
        if n != "t1a" and not label_at_test(n):
            pool[n] = {"ra": v["row"]["ra"], "C": v["C"], "budget_ok": v["budget_ok"], "block": s1["block"]}
    s2 = C.read(C.OUT / "summary_r2.json")
    for n, v in s2["cells"].items():
        pool[n] = {"ra": v["row"]["ra"], "C": v["C"], "budget_ok": v["budget_ok"], "block": s2["block"]}
    s4 = C.read(C.OUT / "summary_r4.json")
    for m, v in s4["mechanisms"].items():
        pool[v["network"]] = {"ra": v["row"]["ra"], "C": v["C"], "budget_ok": v["budget_ok"], "block": s4["block"]}
    ref_lat = D.main_row("t1a", block)[1]["latency_ms"]
    for n in extra:
        row, cost = D.main_row(n, block)
        pool[n] = {"ra": D.row_metrics(row)["ra"], "C": D.fixed_point(n)["C"], "budget_ok": D.budget(cost, ref_lat)[0],
                   "block": block}
    ok = {n: v for n, v in pool.items() if v["budget_ok"] and v["C"] <= TH["C"]}
    best = min(ok, key=lambda n: (ok[n]["ra"], ok[n]["C"])) if ok else None
    return {"network": best, "rule": "X over budget: lowest RA in budget with C <= 0.8 over rounds 1-5", "pool": pool}


def gap(x, oracle, t0p_net):
    """X's remaining gap with its trunk fixed (descriptive): zgz E1 step 6000 and in-sample medians, positive = X worse."""
    out = {}
    X = {"e1": D.zgz(x)["e1"][6000], "ins": {s: D.med(D.insample(x)[s]) for s in C.STEPS}}
    for name, net in (("router_accuracy", oracle), ("router_capacity", t0p_net)):
        if net is None:
            continue
        R = {"e1": D.zgz(net)["e1"][6000], "ins": {s: D.med(D.insample(net)[s]) for s in C.STEPS}}
        out[name] = {"against": net, "zgz_e1_6000": {d: X["e1"][d] - R["e1"][d] for d in ("local", "global")},
                     "insample": {s: X["ins"][s] - R["ins"][s] for s in C.STEPS},
                     "against_zgz_e1_6000": R["e1"], "against_insample": R["ins"]}
    out["x_zgz_e1_6000"], out["x_insample"] = X["e1"], X["ins"]
    return out


def compare(x_legacy, x_dedicated, block):
    pairs = {"legacy_none": ("t1a", x_legacy), "dedicated": ("t1arep", x_dedicated)}
    out = {"block": block, "streams": {}}
    signs = []
    for stream, (ref, x) in pairs.items():
        R, X = reading(ref, block), reading(x, block)
        rr, rx = D.row_metrics(R["row"][0]), D.row_metrics(X["row"][0])
        delta = {k: rr[k] - rx[k] for k in ("local", "global", "ra")}
        ins = {s: D.med(R["insample"][s]) - D.med(X["insample"][s]) for s in C.STEPS}
        ok, parts = D.budget(X["row"][1], R["row"][1]["latency_ms"] if ref == "t1a" else D.main_row("t1a", block)[1]["latency_ms"])
        out["streams"][stream] = {"reference": ref, "candidate": x, "row_ref": rr, "row_x": rx, "delta_row": delta,
                                  "delta_insample": ins, "C_x": X["fixed_point"]["C"], "C_ref": R["fixed_point"]["C"],
                                  "budget_ok": ok, "budget_parts": parts,
                                  "cost_x": {k: X["row"][1][k] for k in ("params_total", "macs_total", "latency_ms")}}
        signs.append((delta["local"] > 0, delta["ra"] > 0))
    S = out["streams"].values()
    strong = all(v["delta_row"]["local"] >= TH["local"] and v["delta_row"]["ra"] >= TH["ra"] and v["C_x"] <= TH["C"] for v in S)
    same = len({s for s in signs}) == 1
    out["verdict"] = "REPRODUCED" if strong else ("DIRECTIONAL" if same else "NOT_REPRODUCED")
    better = [v["delta_row"]["local"] > 0 and v["delta_row"]["ra"] > 0 for v in S]
    worse = [v["delta_row"]["local"] < 0 and v["delta_row"]["ra"] < 0 for v in S]
    out["direction"] = "X_better" if all(better) else ("X_worse" if all(worse) else "mixed")
    ds = [v["delta_row"] for v in out["streams"].values()]
    out["mean_delta_row"] = {k: (ds[0][k] + ds[1][k]) / 2 for k in ds[0]}
    out["half_spread_delta_row"] = {k: abs(ds[0][k] - ds[1][k]) / 2 for k in ds[0]}
    out["computed_at_utc"] = C.now()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--x-legacy", required=True)
    p.add_argument("--x-dedicated", required=True)
    p.add_argument("--block", required=True)
    a = p.parse_args(argv)
    from tools.t1aub import r5
    r5.patch_common()
    C.verify_protocol()
    s = compare(a.x_legacy, a.x_dedicated, a.block)
    plan = C.read(C.OUT / "final_plan.json")
    leg = plan["x"]["legacy_trunk"]
    t0p_net = leg if C.arm_spec(leg)["route"]["router"] == "t0p" else None
    oracle = "x5g" if "x5g" in plan["registry5"]["cells"] else None
    extra = [n for n in (a.x_legacy, a.x_dedicated, "a5_srdwa") if (C.OUT / "evaluations" / n / "row_step6000").is_dir()]
    s["best_deployable"] = best_deployable(s, a.block, extra)
    s["gap"] = gap(a.x_legacy, oracle, t0p_net)
    C.save(C.OUT / "summary_final.json", s, exclusive=True)
    lines = [f"verdict: {s['verdict']} (direction: {s['direction']})", f"best deployable: {s['best_deployable']['network']} ({s['best_deployable']['rule']})", "",
             "| stream | T1a row (local / global / RA) | X row | delta | C_x | budget |",
             "|---|---|---|---|---|---|"]
    for st, v in s["streams"].items():
        r, x, d = v["row_ref"], v["row_x"], v["delta_row"]
        lines.append(f"| {st} | {r['local']:.2f} / {r['global']:.2f} / {r['ra']:.2f} | {x['local']:.2f} / {x['global']:.2f} / "
                     f"{x['ra']:.2f} | {d['local']:+.2f} / {d['global']:+.2f} / {d['ra']:+.2f} | {v['C_x']:.2f} | "
                     f"{'OK' if v['budget_ok'] else 'over'} |")
    lines += ["", "gap (X minus, zgz E1 6000 local / global; in-sample 5000 / 5500 / 6000):"]
    for k, v in s["gap"].items():
        if isinstance(v, dict) and "against" in v:
            lines.append(f"- {k} vs {v['against']}: {v['zgz_e1_6000']['local']:+.2f} / {v['zgz_e1_6000']['global']:+.2f}; "
                         + " / ".join(f"{v['insample'][st]:+.2f}" for st in C.STEPS))
    md = C.OUT / "report_final.md"
    C.require(not md.exists(), "report exists")
    md.write_text("\n".join(lines) + "\n")
    print(md)


if __name__ == "__main__":
    main()
