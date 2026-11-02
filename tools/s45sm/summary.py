"""Round summaries (CPU only, run once per round after every registered reading; single seed 3407, descriptive).

Writes <results>/summary_r<k>.json and report_r<k>.md (exclusive). Three schema levels, chosen by the round:
  1      slow-path main row (`row_step6000`) against one S45 re-measurement (`s45ref`), the arms of the round only
  2-4    latency-path main row (`row_fast_step6000`) against S45 re-measured in the same exclusive block
         (`s45ref_<block>`); the round's arms plus every arm that has a latency-path row
  >= 5   as 2-4, plus: arms registered `"conditional": "select_r4"` are summarised only if the round-4 selection chose
         srkd (and must have no formal training otherwise); the label-route oracles of the round (their own slow-path
         row, PFA ceiling class AT / NEAR / CAPPED, latency not interpreted); the gap of the in-budget arm to the oracle
         split into router and association; and the written next step (`next`)
Per arm: zgz E1 local / global (steps 5000 / 5500 / 6000), E2 / E3 (6000); training-local E1 median per step and
R = (S45 - arm) / (S45 - T1d), R_dz = (S45 - arm) / (S45 - t1ddz); C = RA after 20 feeds of the same packet from the
mean hand / after 1 (training local, step 6000); the main row with the router's parameters and MACs added; the latency
against S45 re-measured with it. Classes (first that applies):
  AT_TARGET     main-row local / global / RA each <= t1ddz + 1.5 mm, C <= 0.8, within budget
  NEAR_TARGET   main-row local <= t1ddz + 3.5, global <= + 2.75, RA <= + 3.1 (S45 sampling-stream noise), C <= 0.8
  MECHANISM     C <= 0.8 and R >= 0.6 at all three steps, zgz not near the target
  WEAK          any R >= 0.15
  NULL          otherwise
The arm set of the latency-path schemas depends on which rows exist when the summary is computed; `as_of_round` = K keeps
only arms registered in rounds <= K (and recomputes `next` on them), which reproduces the summaries as they were recorded
in round K.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import (BUDGET, NOISE, OUT, S45AT, S45CT, S45TR, S45_EVAL, STEPS, TARGET, TH, now, read, require, results_dir,
                    save, sha)
from .protocol import verify_protocol
from .registry import arm_spec, oracles, registry
from .routing import register_kinds
from .selection import selected as select_r4_selected

SPLIT_ARM = "pfasrdz"
REFERENCES = ("s45", "t1d", "t1ddz")


# --------------------------------------------------------------------------- readings of one network
def trace(directory):
    m = read(Path(directory) / "report.json")["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def zgz(net):
    if net == "s45":
        e1 = {s: trace(S45_EVAL / f"e1_step{s}_repeat1_none") for s in STEPS}
        return {"e1": e1, "e2": e1[6000], "e3": trace(S45_EVAL / "e3_step6000_repeat1_none")}
    ev = (S45AT if net == "t1ddz" else results_dir()) / "evaluations" / net
    e2 = trace(ev / "e2_step6000") if (ev / "e2_step6000").is_dir() else None
    return {"e1": {s: trace(ev / f"e1_step{s}") for s in STEPS}, "e2": e2, "e3": trace(ev / "e3_step6000")}


def insample_path(net, step):
    if net in ("s45", "t1d"):
        return (S45TR / "readouts" / net if step == 6000 else S45CT / "readouts" / f"{net}_{step}") / "tl_e1/states.npz"
    root = S45AT if net == "t1ddz" else results_dir()
    return root / "readouts" / (net if step == 6000 else f"{net}_{step}") / "tl_e1/states.npz"


def insample(net, step):
    with np.load(insample_path(net, step), allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})
        per = {r: float(z[f"{r}/mpjpe_ra_mm"].mean()) for r in recs}
    require(len(per) == 9, "nine training local recordings expected")
    return per


def convergence(net):
    from semkine import eval_track as ET
    from tools.s45loc.analyze import FK
    root = S45TR if net in ("s45", "t1d") else (S45AT if net == "t1ddz" else results_dir())
    path = root / "readouts" / net / "tl_f1/states.npz"
    fk = FK()
    with np.load(path, allow_pickle=False) as z:
        recs = sorted({k.split("/")[0] for k in z.files})

        def ra(q, k):
            a, b = fk.joints(q, z[f"{q}/mean_iter{k}"]), fk.joints(q, z[f"{q}/target"])
            return float(np.linalg.norm(ET.root_align(a) - ET.root_align(b), axis=-1).mean())
        r1, r20 = float(np.median([ra(q, 1) for q in recs])), float(np.median([ra(q, 20) for q in recs]))
    return {"ra_feed1_mm": r1, "ra_feed20_mm": r20, "C": r20 / r1}


def router_cost(routing):
    """Parameters and MACs (thop, standard layers) of the router for one packet of 2048 nodes."""
    import torch
    from thop import profile
    from semkine import s45sm as SM
    from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y
    net = SM._load_router(routing["router"], routing["checkpoint"])
    n = 2048
    ev = torch.zeros(n, 5)
    ev[:, EV_BATCH] = 0
    ev[:, EV_X] = torch.arange(n) % 240
    ev[:, EV_Y] = torch.arange(n) % 180
    ev[:, EV_T] = torch.linspace(0, 0.05, n)
    ev[:, EV_P] = (torch.arange(n) % 2).float()
    macs, _ = profile(net, inputs=(ev, torch.tensor([0, n]), torch.tensor([0.05])), verbose=False)
    return {"params": SM.n_params(net), "macs": float(macs)}


def _pointed_row(path):
    r = read(path)
    pointer = r["main_row"]
    require(sha(pointer["path"]) == pointer["sha256"], "main-row file changed: " + str(path))
    return r, read(pointer["path"])


def row(net):
    """The slow-path main-row JSON written by the original generator (via the report's pointer, SHA-checked) and the routing."""
    r, data = _pointed_row(results_dir() / "evaluations" / net / "row_step6000" / "report.json")
    data.update({k: data["two_seed_mean"][k] for k in ("overall", "local", "global")})
    return data, r.get("routing")


def fast_row(net):
    """(row JSON, routing, block) of the latency-path main row."""
    r, data = _pointed_row(results_dir() / "evaluations" / net / "row_fast_step6000" / "report.json")
    data.update({k: data["two_seed_mean"][k] for k in ("overall", "local", "global")})
    return data, r.get("routing"), r["block"]


def oracle_row(net):
    r, data = _pointed_row(results_dir() / "evaluations" / net / "row_step6000" / "report.json")
    m = data["two_seed_mean"]
    got = {"local": m["local"]["mpjpe_ra_mm"], "global": m["global"]["mpjpe_ra_mm"], "ra": m["overall"]["mpjpe_ra_mm"],
           "mpvpe_local": m["local"]["mpvpe_ra_mm"], "mpvpe_global": m["global"]["mpvpe_ra_mm"]}
    return got, data


def references():
    return {n: {s: float(np.median(list(insample(n, s).values()))) for s in STEPS} for n in REFERENCES}


# --------------------------------------------------------------------------- classes
def classify_gaps(conv_ok, budget_ok, gap, R):
    if conv_ok and budget_ok and all(gap[k] <= TH["at_target_mm"] for k in gap):
        return "AT_TARGET"
    if conv_ok and all(gap[k] <= NOISE[k] for k in gap):
        return "NEAR_TARGET"
    if conv_ok and min(R.values()) >= TH["R_high"]:
        return "MECHANISM"
    if max(R.values()) >= TH["R_null"]:
        return "WEAK"
    return "NULL"


def _common(arm, ref):
    med = {s: float(np.median(list(insample(arm, s).values()))) for s in STEPS}
    R = {s: (ref["s45"][s] - med[s]) / (ref["s45"][s] - ref["t1d"][s]) for s in STEPS}
    Rdz = {s: (ref["s45"][s] - med[s]) / (ref["s45"][s] - ref["t1ddz"][s]) for s in STEPS}
    return med, R, Rdz, convergence(arm), zgz(arm)


def classify(arm, ref, fast=True):
    """The summary entry of one arm: latency-path main row against the block's S45 (`fast`), or the slow-path main row
    against the single S45 re-measurement (round 1)."""
    med, R, Rdz, conv, z = _common(arm, ref)
    if fast:
        main, routing, block = fast_row(arm)
        s45ref, _, ref_block = fast_row("s45ref_" + block)
        require(ref_block == block, "S45 reference measured in another block")
    else:
        main, routing = row(arm)
        s45ref, _ = row("s45ref")
    cost = router_cost(routing)
    params_total = int(main["params_total"]) + cost["params"]
    latency_extra = float(main["latency_ms_scaled_full1p75"]) - float(s45ref["latency_ms_scaled_full1p75"])
    budget_ok = params_total <= BUDGET["params_total_max"] and latency_extra <= BUDGET["latency_scaled_extra_ms_max"]
    got = {"local": main["local"]["mpjpe_ra_mm"], "global": main["global"]["mpjpe_ra_mm"], "ra": main["overall"]["mpjpe_ra_mm"]}
    if fast:
        got.update(mpvpe_local=main["local"]["mpvpe_ra_mm"], mpvpe_global=main["global"]["mpvpe_ra_mm"])
    gap = {k: got[k] - TARGET[k] for k in TARGET}
    cls = classify_gaps(conv["C"] <= TH["C"], budget_ok, gap, R)
    out = {"insample_median_by_step": {str(s): v for s, v in med.items()}, "R": {str(s): v for s, v in R.items()},
           "R_dz": {str(s): v for s, v in Rdz.items()}, "convergence": conv,
           "zgz": {"e1": {str(s): v for s, v in z["e1"].items()}, "e2": z["e2"], "e3": z["e3"]},
           "main_row": got, "gap_to_t1ddz": gap, "router": cost, "routing": routing,
           "params_total_with_router": params_total, "latency_scaled_ms": main["latency_ms_scaled_full1p75"]}
    if fast:
        out.update(latency_block=block, s45ref_latency_scaled_ms=s45ref["latency_ms_scaled_full1p75"])
    out.update(latency_extra_vs_s45ref_ms=latency_extra,
               macs_total_with_router=main["macs_forward_packet"] + cost["macs"], budget_ok=budget_ok, **{"class": cls})
    return out


# --------------------------------------------------------------------------- oracles, split and the written next step
def condition_holds(spec):
    cond = spec.get("conditional")
    if cond is None:
        return True
    require(cond == "select_r4", "unknown condition " + repr(cond))
    return select_r4_selected()


def ceiling_class(gap, C):
    if C <= TH["C"] and all(gap[k] <= TH["at_target_mm"] for k in TARGET):
        return "AT"
    if C <= TH["C"] and all(gap[k] <= NOISE[k] for k in TARGET):
        return "NEAR"
    return "CAPPED"


def summarize_oracle(net, ref):
    med, R, Rdz, conv, z = _common(net, ref)
    got, data = oracle_row(net)
    gap = {k: got[k] - TARGET[k] for k in TARGET}
    return {"insample_median_by_step": {str(s): v for s, v in med.items()}, "R": {str(s): v for s, v in R.items()},
            "R_dz": {str(s): v for s, v in Rdz.items()}, "convergence": conv,
            "zgz": {"e1": {str(s): v for s, v in z["e1"].items()}, "e2": z["e2"], "e3": z["e3"]},
            "main_row": got, "gap_to_t1ddz": gap, "params_total": int(data["params_total"]),
            "macs_forward_packet": float(data["macs_forward_packet"]),
            "latency_scaled_ms_not_interpreted": float(data["latency_ms_scaled_full1p75"]),
            "deployable": False, "class": ceiling_class(gap, conv["C"])}


def split(arm_row, oracle_row_):
    return {k: {"router": arm_row[k] - oracle_row_[k], "association": oracle_row_[k] - TARGET[k],
                "total": arm_row[k] - TARGET[k]} for k in TARGET}


def next_step(arms, ceilings):
    ok = {a: v for a, v in arms.items() if v["budget_ok"] and v["class"] in ("AT_TARGET", "NEAR_TARGET")}
    repeat = min(ok, key=lambda a: ok[a]["main_row"]["ra"]) if ok else None
    if any(v["class"] == "AT_TARGET" for v in ok.values()):
        line = "ROBUSTNESS"
    elif any(c == "CAPPED" for c in ceilings.values()):
        line = "ASSOCIATION"
    else:
        line = "ROUTER"
    return {"repeat": repeat, "line": line}


def schema_of(round_):
    return 1 if round_ == 1 else (3 if round_ <= 4 else 5)


# --------------------------------------------------------------------------- the summary
def _has_fast_row(arm):
    return (results_dir() / "evaluations" / arm / "row_fast_step6000" / "report.json").is_file()


def _header(round_, s45ref_ms=None, block_note=True):
    if block_note:
        budget = (f"预算：总参数 ≤ {BUDGET['params_total_max']}，时延比同一独占测量块里重测的 S45 多 ≤ "
                  f"{BUDGET['latency_scaled_extra_ms_max']} ms")
        return [f"# S45-SM 第 {round_} 轮汇总（自动生成；单 seed 3407，主行固定 step 6000，时延用快速实现）", "",
                f"目标（t1ddz，oracle）：{TARGET['local']} / {TARGET['global']} / RA {TARGET['ra']}；" + budget, ""]
    return [f"# S45-SM 第 {round_} 轮（自动生成；单 seed 3407，主行固定 step 6000）", "",
            f"目标（t1ddz，oracle）：{TARGET['local']} / {TARGET['global']} / RA {TARGET['ra']}；"
            f"预算：总参数 ≤ {BUDGET['params_total_max']}，时延比同期重测的 S45（{s45ref_ms:.2f} ms）"
            f"多 ≤ {BUDGET['latency_scaled_extra_ms_max']} ms", ""]


def _table_rows(arms, reg, round_, with_block):
    cols = ("| 臂 | 轮 | 主行 local / global / RA | 差距（对 t1ddz） | R@5000 / 5500 / 6000 | C | 参数（含路由器） | 时延增量（块） | 类别 |"
            "\n|---|---|---|---|---|---|---|---|---|")
    lines = cols.split("\n") if with_block else [
        "| 臂 | 主行 local / global / RA | 差距（对 t1ddz） | R@5000 / 5500 / 6000 | C | 参数（含路由器） | 时延增量 | 类别 |",
        "|---|---|---|---|---|---|---|---|"]
    for arm, v in arms.items():
        m, g, R, c = v["main_row"], v["gap_to_t1ddz"], v["R"], v["convergence"]
        if with_block:
            lines.append(f"| {arm} | {reg[arm]['round']} | {m['local']:.2f} / {m['global']:.2f} / {m['ra']:.2f} | "
                         f"{g['local']:+.2f} / {g['global']:+.2f} / {g['ra']:+.2f} | {R['5000']:.2f} / {R['5500']:.2f} / "
                         f"{R['6000']:.2f} | {c['C']:.2f} | {v['params_total_with_router'] / 1e6:.3f} M | "
                         f"{v['latency_extra_vs_s45ref_ms']:+.2f} ms（{v['latency_block']}） | {v['class']} |")
        else:
            lines.append(f"| {arm} | {m['local']:.2f} / {m['global']:.2f} / {m['ra']:.2f} | {g['local']:+.2f} / "
                         f"{g['global']:+.2f} / {g['ra']:+.2f} | {R['5000']:.2f} / {R['5500']:.2f} / {R['6000']:.2f} | "
                         f"{c['C']:.2f} | {v['params_total_with_router'] / 1e6:.3f} M | "
                         f"{v['latency_extra_vs_s45ref_ms']:+.2f} ms | {v['class']} |")
    return lines


def summarize(round_, as_of_round=None):
    """Compute and write the summary of `round_` (see the module docstring). `as_of_round` keeps arms of rounds <= K."""
    verify_protocol()
    register_kinds()
    require(round_ in registry()["rounds"], "round not registered")
    target = results_dir() / f"summary_r{round_}.json"
    require(not target.exists(), "the round summary is read once")
    schema = schema_of(round_)
    ref = references()
    reg = registry()["arms"]
    limit = as_of_round if as_of_round is not None else max(registry()["rounds"])

    def in_scope(spec):
        return spec["round"] <= limit
    if schema == 1:
        arms = {arm: classify(arm, ref, fast=False) for arm, spec in reg.items() if spec["round"] == round_}
        s45ref, _ = row("s45ref")
        summary = {"computed_utc": now(), "round": round_, "thresholds": TH, "noise": NOISE, "target": TARGET,
                   "budget": BUDGET, "references": {k: {str(s): v for s, v in d.items()} for k, d in ref.items()},
                   "reference_convergence": {n: convergence(n) for n in REFERENCES},
                   "s45ref_latency_scaled_ms": s45ref["latency_ms_scaled_full1p75"], "arms": arms,
                   "single_seed_descriptive": True}
        save(target, summary, exclusive=True)
        lines = _header(round_, s45ref["latency_ms_scaled_full1p75"], block_note=False) + _table_rows(arms, reg, round_, False)
    elif schema == 3:
        arms = {arm: classify(arm, ref) for arm, spec in reg.items()
                if (spec["round"] == round_ or _has_fast_row(arm)) and in_scope(spec)}
        summary = {"computed_utc": now(), "round": round_, "thresholds": TH, "noise": NOISE, "target": TARGET,
                   "budget": BUDGET, "references": {k: {str(s): v for s, v in d.items()} for k, d in ref.items()},
                   "arms": arms, "single_seed_descriptive": True}
        save(target, summary, exclusive=True)
        lines = _header(round_) + _table_rows(arms, reg, round_, True)
    else:
        arms, untrained = {}, {}
        for arm, spec in reg.items():
            has_row = _has_fast_row(arm)
            if spec["round"] == round_ and not condition_holds(spec):
                require(not (results_dir() / "receipts" / f"formal_{arm}.json").exists() and not has_row,
                        "an arm whose condition failed was trained: " + arm)
                untrained[arm] = {"status": "NOT_TRAINED_BY_RULE", "conditional": spec["conditional"]}
                continue
            if (spec["round"] == round_ or has_row) and in_scope(spec):
                arms[arm] = classify(arm, ref)
        orc = {net: summarize_oracle(net, ref) for net, spec in oracles().items() if spec["round"] == round_}
        require(orc or arms, "nothing to summarise")
        base = fast_row(SPLIT_ARM)[0]
        base_row = {"local": base["local"]["mpjpe_ra_mm"], "global": base["global"]["mpjpe_ra_mm"],
                    "ra": base["overall"]["mpjpe_ra_mm"]}
        splits = {net: split(base_row, v["main_row"]) for net, v in orc.items()}
        nxt = next_step(arms, {net: v["class"] for net, v in orc.items()})
        summary = {"computed_utc": now(), "round": round_, "thresholds": TH, "noise": NOISE, "target": TARGET,
                   "budget": BUDGET, "references": {k: {str(s): v for s, v in d.items()} for k, d in ref.items()},
                   "arms": arms, "untrained": untrained, "oracles": orc, "split_of": SPLIT_ARM, "splits": splits,
                   "next": nxt, "single_seed_descriptive": True}
        save(target, summary, exclusive=True)
        lines = _header(round_) + _table_rows(arms, reg, round_, True)
        for arm in untrained:
            lines.append(f"| {arm} | {round_} | 未训练（第 4 轮选择未选 srkd） | | | | | | |")
        if orc:
            lines += ["", "标签面路由 + PFA（不可部署上界；时延不作解读）：", "",
                      "| 臂 | 主行 local / global / RA | 差距（对 t1ddz） | R@5000 / 5500 / 6000 | C | E3 local / global | 上界类别 |",
                      "|---|---|---|---|---|---|---|"]
            for net, v in orc.items():
                m, g, R, c, e3 = v["main_row"], v["gap_to_t1ddz"], v["R"], v["convergence"], v["zgz"]["e3"]
                lines.append(f"| {net} | {m['local']:.2f} / {m['global']:.2f} / {m['ra']:.2f} | {g['local']:+.2f} / "
                             f"{g['global']:+.2f} / {g['ra']:+.2f} | {R['5000']:.2f} / {R['5500']:.2f} / {R['6000']:.2f} | "
                             f"{c['C']:.2f} | {e3['local']:.2f} / {e3['global']:.2f} | {v['class']} |")
            for net, s in splits.items():
                lines.append("")
                lines.append(f"{SPLIT_ARM} − t1ddz 的拆分（{net}，描述）：" + "；".join(
                    f"{k} 路由 {s[k]['router']:+.2f} + 关联 {s[k]['association']:+.2f} = {s[k]['total']:+.2f}" for k in TARGET))
        lines += ["", f"写定的下一步：line = {nxt['line']}，repeat = {nxt['repeat']}"]
    (results_dir() / f"report_r{round_}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return target
