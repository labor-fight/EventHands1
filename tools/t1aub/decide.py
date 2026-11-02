#!/usr/bin/env python3
"""T1A-UB round summary (CPU only; run once per round after every registered reading; single seed 3407, descriptive).

    python tools/t1aub/decide.py --round K --block B     -> outputs/t1aub/20261102/summary_r<K>.json, report_r<K>.md

Implements the written rules of the round's pre-registration (docs/EXPERIMENT_RECORDS.md#t1aub-r1-prereg-20261102):
readings per network (T1a from S45-TR / S45-GRADE; arms and cells from this study), the basic criteria (in-sample
detected, convergence C <= 0.8, the XCL single-seed zgz rule), the layer attributions of round 1, the classes of the
in-budget candidates and the written next step. Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.t1aub import common as C  # noqa: E402

GRADE = ROOT / "outputs/s45grade/20261030/insample/readouts"
TH = {"insample": 2.5, "local": 5.0, "global": 8.0, "ra": 3.1, "C": 0.8, "recs": 7, "row_match": 0.05,
      "feat_hi": 0.6, "feat_lo": 0.15}
TL = ("ch_local", "lfz_local", "lpc_local", "lr_local", "ly_local", "lyh_local", "lyq_local", "ycy_local", "ylf_local")


# --------------------------------------------------------------------------- readings
def _eval_dir(net):
    return C.S45TR / "evaluations/t1a" if net == "t1a" else C.OUT / "evaluations" / net


def _readout_dir(net, step=6000):
    if net == "t1a":
        return C.S45TR / "readouts/t1a" if step == 6000 else GRADE / f"t1a_{step}"
    return C.OUT / "readouts" / (net if step == 6000 else f"{net}_{step}")


def trace(directory):
    m = C.read(Path(directory) / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def zgz(net):
    ev = _eval_dir(net)
    return {"e1": {s: trace(ev / f"e1_step{s}") for s in C.STEPS}, "e2": trace(ev / "e2_step6000"),
            "e3": trace(ev / "e3_step6000")}


def per_recording(path):
    with np.load(path, allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})
        return {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}


def insample(net):
    out = {}
    for s in C.STEPS:
        per = per_recording(_readout_dir(net, s) / "tl_e1/states.npz")
        C.require(sorted(per) == sorted(TL), "nine training local recordings expected")
        out[s] = per
    return out


def fixed_point(net):
    """C (mean start: RA@20 / RA@1), the GT start (correct-state keeping) and the own-loop start; medians over TL."""
    from semkine import eval_track as ET
    from tools.s45loc.analyze import FK
    fk = FK()
    with np.load(_readout_dir(net) / "tl_f1/states.npz", allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})

        def ra(q, key):
            a, b = fk.joints(q, z[f"{q}/{key}"]), fk.joints(q, z[f"{q}/target"])
            return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean())
        out = {}
        for start in ("mean", "gt", "loop"):
            r1 = float(np.median([ra(q, f"{start}_iter1") for q in recs]))
            r20 = float(np.median([ra(q, f"{start}_iter20") for q in recs]))
            out[start] = {"ra1": r1, "ra20": r20, "ratio": r20 / r1}
    out["C"] = out["mean"]["ratio"]
    return out


def single_step(net):
    tl = per_recording(_readout_dir(net) / "tl_e3/states.npz")
    zl = per_recording(_readout_dir(net) / "zl_e3/states.npz")
    return {"tl_e3": float(np.median(list(tl.values()))), "zl_e3": zl["zgz_local"]}


def retention(net):
    """Train-local one-step survival rho_param of the T10 / R5 / Fs perturbations (S45-LOC's formula, S45-TR analyze)."""
    from scipy.spatial.transform import Rotation as R
    d = _readout_dir(net)
    out = {}
    for kind in ("T10", "R5", "Fs"):
        per = {}
        with np.load(d / f"tl_{kind}/states.npz", allow_pickle=False) as zk, \
                np.load(d / "tl_e3/states.npz", allow_pickle=False) as z3:
            for q in TL:
                pd_, p0 = zk[f"{q}/prediction"].astype(np.float64), z3[f"{q}/prediction"].astype(np.float64)
                used, truth = zk[f"{q}/incoming_used"].astype(np.float64), zk[f"{q}/gt_previous"].astype(np.float64)
                if kind == "T10":
                    din, dout = used[:, 0:3] - truth[:, 0:3], pd_[:, 0:3] - p0[:, 0:3]
                elif kind == "R5":
                    din = (R.from_rotvec(used[:, 3:6]) * R.from_rotvec(truth[:, 3:6]).inv()).as_rotvec()
                    dout = (R.from_rotvec(pd_[:, 3:6]) * R.from_rotvec(p0[:, 3:6]).inv()).as_rotvec()
                else:
                    din, dout = used[:, 6:51] - truth[:, 6:51], pd_[:, 6:51] - p0[:, 6:51]
                per[q] = float((din * dout).sum() / max((din * din).sum(), 1e-18))
        out[kind] = float(np.median(list(per.values())))
    return out


def main_row(net, block):
    """(row JSON of tools/make_s36_row.py, cost dict) of a network's deployment-form row in block B."""
    name = f"t1aref_{block}" if net == "t1a" else net
    r = C.read(C.OUT / "evaluations" / name / "row_step6000/report.json")
    row = C.read(r["main_row"]["path"])
    C.require(C.sha(r["main_row"]["path"]) == r["main_row"]["sha256"], "main-row file changed")
    C.require(r["block"] == block, "row measured in another block: " + name)
    return row, r["cost"]


def readings(net, block):
    return {"zgz": zgz(net), "insample": insample(net), "fixed_point": fixed_point(net), "single_step": single_step(net),
            "retention": retention(net), "row": main_row(net, block)}


# --------------------------------------------------------------------------- criteria
def med(per):
    return float(np.median(list(per.values())))


def insample_gain(ref, net):
    """{step: median(ref) - median(net)} and, at 6000, how many recordings are better for net."""
    gains = {s: med(ref["insample"][s]) - med(net["insample"][s]) for s in C.STEPS}
    better = sum(net["insample"][6000][q] < ref["insample"][6000][q] for q in TL)
    return gains, better


def insample_detected(ref, net):
    gains, better = insample_gain(ref, net)
    return all(g >= TH["insample"] for g in gains.values()) and better >= TH["recs"]


def zgz_support(ref, net, dom="local"):
    """XCL single-seed rule on the zgz domain `dom` (local 5 / global 8 mm)."""
    th = TH[dom]
    d1 = ref["zgz"]["e1"][6000][dom] - net["zgz"]["e1"][6000][dom]
    d2 = ref["zgz"]["e2"][dom] - net["zgz"]["e2"][dom]
    d3 = ref["zgz"]["e3"][dom] - net["zgz"]["e3"][dom]
    late = [ref["zgz"]["e1"][s][dom] - net["zgz"]["e1"][s][dom] for s in C.STEPS]
    same = np.sign(d1) == np.sign(d2) == np.sign(d3)
    two = sum(np.sign(x) == np.sign(d1) for x in late) >= 2
    return bool(abs(d1) >= th and same and two), d1


def three_point(diffs, th=TH["insample"]):
    """Classes of a three-checkpoint difference (positive = the first network better)."""
    v = list(diffs.values())
    if all(x >= th for x in v):
        return "HELPS"
    if all(x <= -th for x in v):
        return "HURTS"
    if all(abs(x) < th for x in v):
        return "NULL"
    return "MIXED"


def diffs(a, b):
    """{step: median_insample(b) - median_insample(a)}: positive = a better."""
    return {s: med(b["insample"][s]) - med(a["insample"][s]) for s in C.STEPS}


def budget(cost, ref_latency):
    ok = {"params": cost["params_total"] <= C.BUDGET["params"], "macs": cost["macs_total"] <= C.BUDGET["macs"],
          "latency": cost["latency_ms"] <= ref_latency}
    return all(ok.values()), ok


def row_metrics(row):
    m = row["two_seed_mean"]
    return {"local": m["local"]["mpjpe_ra_mm"], "global": m["global"]["mpjpe_ra_mm"], "ra": m["overall"]["mpjpe_ra_mm"],
            "local_v": m["local"]["mpvpe_ra_mm"], "global_v": m["global"]["mpvpe_ra_mm"]}


# --------------------------------------------------------------------------- round 1
def round1(block):
    reg = C.registry()
    nets = ["t1a"] + [a for a, s in reg["arms"].items() if s["round"] == 1 and "formal" in C.phases(a)] + \
           [c for c, s in reg["cells"].items() if s["round"] == 1]
    R = {n: readings(n, block) for n in nets}
    ref_row, ref_cost = R["t1a"]["row"]
    ref_lat = ref_cost["latency_ms"]
    t1a = R["t1a"]
    out = {"round": 1, "block": block, "t1aref_latency_ms": ref_lat, "networks": {}}
    for n in nets:
        r = R[n]
        row, cost = r["row"]
        rm = row_metrics(row)
        ok, parts = budget(cost, ref_lat)
        gains, better = insample_gain(t1a, r) if n != "t1a" else ({s: 0.0 for s in C.STEPS}, 0)
        e1 = r["zgz"]["e1"][6000]
        out["networks"][n] = {
            "row": rm, "row_vs_e1": {d: rm[d] - e1[d] for d in ("local", "global")},
            "cost": {k: cost[k] for k in ("params_total", "params_model", "params_router", "macs_total", "latency_ms",
                                          "latency_raw_ms")},
            "tail_ms": cost["tail_scaled_ms"]["per_packet_min_over_passes_ms"],
            "budget_ok": ok, "budget_parts": parts, "zgz": r["zgz"], "insample_median": {s: med(r["insample"][s]) for s in C.STEPS},
            "insample_gain_vs_t1a": gains, "recordings_better_6000": better,
            "insample_detected_vs_t1a": n != "t1a" and insample_detected(t1a, r),
            "zgz_support_local": n != "t1a" and zgz_support(t1a, r, "local")[0],
            "zgz_support_global": n != "t1a" and zgz_support(t1a, r, "global")[0],
            "C": r["fixed_point"]["C"], "converges": r["fixed_point"]["C"] <= TH["C"],
            "keep_correct_gt_ra20": r["fixed_point"]["gt"]["ra20"], "keep_correct_gt_ra1": r["fixed_point"]["gt"]["ra1"],
            "loop_start": r["fixed_point"]["loop"], "single_step": r["single_step"], "retention": r["retention"]}
    N = out["networks"]

    def ratio(num_net, den_net):
        res = {}
        for s in C.STEPS:
            den = med(t1a["insample"][s]) - med(R[den_net]["insample"][s])
            res[s] = None if den <= 0 else (med(t1a["insample"][s]) - med(R[num_net]["insample"][s])) / den
        return res
    layers = {}
    layers["assoc_headroom"] = {"row": {k: N["t1a"]["row"][k] - N["gt"]["row"][k] for k in ("local", "global", "ra")},
                                "insample": N["gt"]["insample_gain_vs_t1a"]}
    rho = ratio("gtf", "gt")
    if any(v is None for v in rho.values()):
        feat = "NO_HEADROOM"
    elif N["gtf"]["converges"] and all(v >= TH["feat_hi"] for v in rho.values()):
        feat = "FEATURE_SUFFICES"
    elif not N["gtf"]["converges"] and all(v <= TH["feat_lo"] for v in rho.values()):
        feat = "ASSOCIATION_NECESSARY"
    else:
        feat = "PARTIAL"
    layers["representation"] = {"rho_feat": rho, "class": feat}
    layers["same_input_router"] = {"R_t0p": ratio("t0p", "gt"),
                                   "router_cost_row": {k: N["t0p"]["row"][k] - N["gt"]["row"][k] for k in ("local", "global", "ra")}}
    dz_det = N["dz"]["insample_detected_vs_t1a"]
    dz_worse = all(g <= -TH["insample"] for g in N["dz"]["insample_gain_vs_t1a"].values())
    layers["observation_dz"] = {"class": "DETECTED" if dz_det else ("WORSE" if dz_worse else "NOT_DETECTED"),
                                "zgz_local": N["dz"]["zgz_support_local"]}
    eg = three_point(diffs(R["t0p"], R["t0pnoeg"]))
    layers["event_graph"] = {"diffs_t0pnoeg_minus_t0p": diffs(R["t0p"], R["t0pnoeg"]), "class": "EG_" + eg}

    def slim(full, slim_):
        d = {s: med(R[slim_]["insample"][s]) - med(R[full]["insample"][s]) for s in C.STEPS}
        if all(x < TH["insample"] for x in d.values()):
            return d, "SLIM_OK"
        if all(x >= TH["insample"] for x in d.values()):
            return d, "SLIM_COSTS"
        return d, "MIXED"
    layers["readout_slim_eg"] = dict(zip(("diffs", "class"), slim("t0p", "t0ps16")))
    layers["readout_slim_noeg"] = dict(zip(("diffs", "class"), slim("t0pnoeg", "t0pnoeg20")))
    out["layers"] = layers
    cands = {}
    for name in ("t0ps16_srdwa", "t0pnoeg20_srdwa"):
        n = N[name]
        if not n["budget_ok"]:
            cls = "OVER_BUDGET"
        elif (N["t1a"]["row"]["local"] - n["row"]["local"] >= TH["local"] and N["t1a"]["row"]["ra"] - n["row"]["ra"] >= TH["ra"]
              and n["converges"]):
            cls = "IMPROVED"
        elif n["insample_detected_vs_t1a"]:
            cls = "IMPROVED_INSAMPLE"
        else:
            cls = "NOT_IMPROVED"
        cands[name] = cls
    out["candidates"] = cands
    inb = [c for c in cands if cands[c] != "OVER_BUDGET"]
    if inb:
        base = min(inb, key=lambda c: (N[c]["row"]["ra"], N[c]["C"]))
        nxt = {"base": base}
    else:
        over = sorted({k for c in cands for k, v in N[c]["budget_parts"].items() if not v})
        nxt = {"base": None, "fix_first": over}
    nxt["feature_arm"] = feat == "FEATURE_SUFFICES"
    nxt["event_graph"] = "keep" if eg == "HELPS" else "budget_to_router_or_readout"
    out["next"] = nxt
    out["computed_at_utc"] = C.now()
    return out


def fmt_row(label, rm, cost):
    return (f"| {label} | {rm['local']:.2f} | {rm['global']:.2f} | {rm['local_v']:.2f} | {rm['global_v']:.2f} | {rm['ra']:.2f} "
            f"| {cost['latency_ms']:.2f} ms | {cost['macs_total'] / 1e9:.3f} G | {cost['params_total'] / 1e6:.2f} M |")


def report(s):
    N = s["networks"]
    lines = ["| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |",
             "|---|---|---|---|---|---|---|---|---|",
             "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"]
    for n, v in N.items():
        lines.append(fmt_row("S45-T1a（基线，同块重测）" if n == "t1a" else f"T1A-UB {n}", v["row"], v["cost"]))
    lines += ["", "| 网络 | E3 local / global | 样本内 5000 / 5500 / 6000 | C（均值起点） | GT 起点 RA@1 → @20 | 闭环起点 RA@1 → @20 "
              "| 保留率 T10 / R5 / Fs | 参数（模型 + 路由器） | MACs | 时延 p50 / p99 | 预算 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in N.items():
        e3 = v["zgz"]["e3"]
        ins = " / ".join(f"{v['insample_median'][st]:.2f}" for st in C.STEPS)
        r = v["retention"]
        t = v["tail_ms"]
        lines.append(f"| {n} | {e3['local']:.2f} / {e3['global']:.2f} | {ins} | {v['C']:.2f} | {v['keep_correct_gt_ra1']:.2f} → "
                     f"{v['keep_correct_gt_ra20']:.2f} | {v['loop_start']['ra1']:.2f} → {v['loop_start']['ra20']:.2f} "
                     f"| {r['T10']:.2f} / {r['R5']:.2f} / {r['Fs']:.2f} | {v['cost']['params_model']:,} + {v['cost']['params_router']:,} "
                     f"| {v['cost']['macs_total'] / 1e9:.3f} G | {t['p50']:.2f} / {t['p99']:.2f} ms "
                     f"| {'OK' if v['budget_ok'] else 'over: ' + ','.join(k for k, ok in v['budget_parts'].items() if not ok)} |")
    lines += ["", "```", __import__("json").dumps({"layers": s["layers"], "candidates": s["candidates"], "next": s["next"]},
                                                     indent=1, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--block", required=True)
    a = p.parse_args(argv)
    C.verify_protocol()
    C.require(a.round == 1, "round-specific rules: only round 1 is registered in this tool")
    s = round1(a.block)
    C.save(C.OUT / f"summary_r{a.round}.json", s, exclusive=True)
    md = C.OUT / f"report_r{a.round}.md"
    C.require(not md.exists(), "report exists")
    md.write_text(report(s))
    print(md)


if __name__ == "__main__":
    main()
