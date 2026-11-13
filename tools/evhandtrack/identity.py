#!/usr/bin/env python3
r"""EvHandTrack: identity of the converged method (docs/EXPERIMENT_RECORDS.md#evhandtrack-d1-20261110; AGENTS.md naming).

The user's EvHandTrack is the kinematic-tree design whose GNN was shown to work in the three-seed comparison
(2026-11-10: "我之前让你命名的EvHandTrack是那个利用运动学树的那个实验", then "EvHandTrack我说的是那个GNN确定能发挥效果的那一个，
我昨天进行对比实验的那个"): PMR design D1 (`d1r`), whose propagation P passed S2 (RA(R-noprop) - RA(R) = 2.75 / 4.28 /
3.80 mm, #pmr-s2-results-20261109), in its final form R + rol5 (S3a). Earlier identities stay as records:
B2 + rol5 (tools/evhandtrack/identity_b2rol5.py, outputs/evhandtrack/b2rol5/) and PMR ghand + rol5 (`--arm d1ghand`
below, outputs/evhandtrack/ghand/).

EvHandTrack = PMR `d1r` + rol5, three seeds (PMR runs `pmr_d1r_rol5_s3407` / `s3408` / `s3409`):
    input   as B2: r_s2d router + PFA association -> event-vertex relation encoding -> event-additive per-part statistics
            and learned channels for 16 part nodes and one hand node
    trunk   kinematic-tree propagation P (semkine/pmr_trunk.py `PMRNet`, variant r): 3 rounds of gated messages over the
            62 typed edges of the 16-part tree + hand node (down / up / hand->part / part->hand); the gates come from the
            prior state's FK joint geometry, shared by the rounds; local readout, no bypass: joint j's increment from
            node j only, root and translation from the hand node; 51-D additive state update
    recipe  d1r from scratch, 6000 steps, B2's recipe and data stream; rol5 = TPD ROL fine-tune from d1r's step-6000
            checkpoint of the same seed: K = 5 consecutive 50 ms packets, gradient through the previous state, equal
            weights, 1500 steps, peak lr 1e-3, 100 warm-up, cosine to 0
Code, config keys, checkpoints and run names keep their identifiers; nothing is retrained.

Primary form = streaming (semkine/tpd_str.py; user 2026-11-09 "主行最好用流式形式"): each event's node selection and
token depend only on itself and the stream before it, the pixel time surface persists across updates, nothing is cut;
the router r_s2d stays per update interval (exempt). The readings are PMR's zero-training `__str` readings (stage S3c,
tools/pmr/run_forms.py); the ordinary packet form (not asynchronous) and the causal form are descriptive.
Async receipts: <root>/receipts/ (tools/evhandtrack/asynccheck.py). Cost: <root>/costs/ (tools/evhandtrack/cost2.py, one
block on one GPU, default flags, S43's executor in the same block; the trunk on semkine/pmr_fast.py, the same function in
fewer kernels, checked against the reference; the streaming reference implementation is recorded as well).

    python -m tools.evhandtrack.identity write [--arm d1r|d1ghand] [--provisional]
        -> <root>/identity.json and the rows outputs/semkine/<rows>_main_row.json (streaming), <rows>_packet_main_row.json,
           <rows>_s43_main_row.json (S43, same cost block); --provisional writes the identity without the cost block
           and the async receipts (both pending, no rows), so readers of the identity (CMP) see the right network
           before they run
    python -m tools.evhandtrack.identity verify [--arm d1r|d1ghand]   -> re-hash every file identity.json names
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evhandtrack.identity_b2rol5 import KEYS, S43_ROW, SEMKINE, SEEDS, blocks, five, rel, sha  # noqa: E402
from tools.pfo.common import now, read, require, save  # noqa: E402

PMR = ROOT / "outputs/pmr/20261109"
EHT = ROOT / "outputs/evhandtrack"
ROUTER = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
ARMS = {
    "d1r": {"name": "EvHandTrack", "root": EHT, "rows": "evhandtrack",
            "trunk": {"design": "d1", "variant": "r", "module": "semkine.pmr_trunk"},
            "label": "EvHandTrack（运动学树 D1 + rol5，三种子，流式形式）",
            "packet_label": "EvHandTrack 普通分包形式（不异步，描述）",
            "definition": "PMR d1r (kinematic-tree propagation P over 16 part nodes + hand node, gates from the prior FK "
                          "geometry, local readout; P established by the three-seed S2 comparison; from scratch 6000 "
                          "steps, B2's recipe) + rol5 (TPD ROL, K = 5, gradient through the previous state, 1500 steps), "
                          "deployed router r_s2d + PFA; seeds 3407 / 3408 / 3409"},
    "d1ghand": {"name": "PMR ghand + rol5", "root": EHT / "ghand", "rows": "ghandrol5",
                "trunk": {"design": "ghand", "variant": "ghand", "module": "semkine.pmr_trunk6"},
                "label": "PMR ghand + rol5（运动学树，门在手系，三种子，流式形式；描述）",
                "packet_label": "PMR ghand + rol5 普通分包形式（描述）",
                "definition": "PMR d1ghand (D1 with the gate geometry in the hand frame, R_0^T v / 2 cm; from scratch 6000 "
                              "steps) + rol5; EvHandTrack from 2026-11-10 02:20 to 02:50 UTC, superseded by d1r"},
}
FORMS = {"streaming": "__str", "packet": "", "causal": "__causal"}
PRIMARY = "streaming"


def paths(arm):
    root = ARMS[arm]["root"]
    return {"identity": root / "identity.json", "receipts": root / "receipts", "sampler": root / "logs/gpu7_sampler.log",
            "fast_check": root / "costs/evhandtrack_s3407__fast_check.json",
            "costs": {"streaming": root / "costs/evhandtrack_s3407__stream__pmrfast__default",
                      "packet": root / "costs/evhandtrack_s3407__packet__pmrfast__default",
                      "streaming_ref": root / "costs/evhandtrack_s3407__stream__ref__default",
                      "s43": root / "costs/s43_s3407__exec__default"}}


def seed_identity(arm, seed, files, provisional=False):
    run, base_run = f"pmr_{arm}_rol5_s{seed}", f"pmr_{arm}_s{seed}"
    receipt = read(PMR / "receipts" / f"{run}.json")
    require(receipt["status"] == "PASS", run + ": rol5 acceptance")
    ckpt = PMR / "runs" / run / f"{run}-step=1500.ckpt"
    h = sha(ckpt)
    require(h == receipt["checkpoint_sha256_by_step"]["1500"], run + ": checkpoint changed since acceptance")
    launch = read(PMR / "runs" / run / "pmr_launch.json")
    cfg = Path(launch["config"])
    require(sha(cfg) == launch["config_sha256"], run + ": training config changed since launch")
    require(launch["trunk"] == ARMS[arm]["trunk"] and launch["rol"]["K"] == 5 and launch["rol"]["GRAD"],
            run + ": not " + arm + " + rol5")
    base = PMR / "runs" / base_run / f"{base_run}-step=6000.ckpt"
    base_receipt = read(PMR / "receipts" / f"{base_run}.json")
    hb = sha(base)
    require(launch["init_from"] == str(base) and launch["init_from_sha256"] == hb,
            run + ": INIT_FROM is not " + base_run + " step 6000")
    require(base_receipt["status"] == "PASS" and hb == base_receipt["checkpoint_sha256_by_step"]["6000"],
            base_run + ": checkpoint changed since acceptance")
    forms = {}
    for form, suffix in FORMS.items():
        rp = PMR / "readings" / f"{run}{suffix}" / "e1" / "report.json"
        r = read(rp)
        require(r["status"] == "COMPLETE" and r["checkpoint_sha256"] == h, f"{run}{suffix}: reading of another file")
        ecfg = Path(r["identity"]["config"])
        require(sha(ecfg) == r["config_sha256"], f"{run}{suffix}: evaluation config changed since the reading")
        forms[form] = {"reading": rel(rp), "eval_config": rel(ecfg), **blocks(r)}
        files[rel(ecfg)] = r["config_sha256"]
        files[rel(rp)] = sha(rp)
    files.update({rel(ckpt): h, rel(cfg): launch["config_sha256"], rel(base): hb})
    ap = paths(arm)["receipts"] / f"async_evhandtrack_s{seed}.json"
    if not provisional:
        a = read(ap)
        require(a["status"] == "PASS" and a["form"] == "stream" and a["checkpoint_sha256"] == h,
                f"{run}: streaming async check")
        files[rel(ap)] = sha(ap)
    return {"seed": seed, "run": run, "checkpoint": rel(ckpt), "checkpoint_sha256": h, "train_config": rel(cfg),
            "base": {"run": base_run, "checkpoint": rel(base), "checkpoint_sha256": hb},
            "receipt": rel(PMR / "receipts" / f"{run}.json"), "async_streaming": None if provisional else rel(ap),
            "forms": forms}


def cost_fields(c, d):
    return {"latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
            "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
            "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": rel(d)}


def cost_block(arm, seeds, files):
    P = paths(arm)
    cost = {k: read(d / "report.json") for k, d in P["costs"].items()}
    for k, d in P["costs"].items():
        require(cost[k]["status"] == "COMPLETE" and cost[k]["policy"] == "default",
                k + ": the table's cost comes from a complete PyTorch-default-flag reading")
        files[rel(d / "report.json")] = sha(d / "report.json")
    for k in ("streaming", "packet", "streaming_ref"):
        require(cost[k]["checkpoint_sha256"] == seeds[0]["checkpoint_sha256"], k + ": cost of seed 3407")
    check = read(P["fast_check"])
    require(check["run"] == seeds[0]["run"] and check["max_abs_diff"] < 1e-3, "fast path differs from the reference")
    files[rel(P["fast_check"])] = sha(P["fast_check"])
    foreign = set()
    for line in P["sampler"].read_text().splitlines():
        foreign |= {p for p in line.split()[1:] if p.isdigit()}
    require(len(foreign) == len(P["costs"]), "the block's GPU carried a process other than its four runs: " + str(foreign))
    loads = [float(x.split("load")[1]) for x in P["sampler"].read_text().splitlines() if "load" in x]

    def ratios(k):
        c, c43 = cost[k], cost["s43"]
        return {"params": c["params_total"] / c43["params_total"],
                "macs": c["macs_layers"]["median"] / c43["macs_layers"]["median"],
                "latency_row": c["latency_row"]["scaled_ms"] / c43["latency_row"]["scaled_ms"],
                "latency_e2e_p50": c["latency_e2e_ms"]["p50"] / c43["latency_e2e_ms"]["p50"],
                "latency_e2e_p99": c["latency_e2e_ms"]["p99"] / c43["latency_e2e_ms"]["p99"]}
    block = {"readings": {k: rel(d) for k, d in P["costs"].items()}, "fast_check": check,
             "ratios_vs_s43_same_block": {k: ratios(k) for k in ("streaming", "packet", "streaming_ref")},
             "gpu_compute_pids_seen": sorted(foreign), "host_loadavg_1min": [min(loads), max(loads)],
             "note": "one block on one GPU (no other process on it); the main row's cost is the fast path (pmr_fast, max "
                     f"|difference| {check['max_abs_diff']:.1e} over 100 packets), the reference implementation is recorded"}
    return cost, block


def write(arm="d1r", provisional=False):
    A, P = ARMS[arm], paths(arm)
    files = {}
    seeds = [seed_identity(arm, s, files, provisional) for s in SEEDS]
    files[rel(ROUTER)] = sha(ROUTER)
    means = {f: {b: {k: float(np.mean([s["forms"][f][b][k] for s in seeds])) for k in KEYS}
                 for b in ("local", "global", "overall")} for f in FORMS}
    s43_row = read(S43_ROW)
    s43 = five(s43_row["two_seed_mean"])
    cost, block = ((None, {"status": "pending: async receipts and the tools/evhandtrack/cost2.py block"}) if provisional
                   else cost_block(arm, seeds, files))
    ident = {"name": A["name"], "written_utc": now(), "provisional": provisional, "definition": A["definition"],
             "trunk": A["trunk"], "primary_form": PRIMARY, "seeds": seeds,
             "router": {"checkpoint": rel(ROUTER), "sha256": files[rel(ROUTER)]},
             "five_order": ["MPJPE-local", "MPJPE-global", "MPVPE-local", "MPVPE-global", "RA-MPJPE"],
             "three_seed_five": {f: five(means[f]) for f in FORMS},
             "s43_engine_five": s43,
             "minus_s43": {f: [a - b for a, b in zip(five(means[f]), s43)] for f in FORMS},
             "cost": block,
             "records": {"b2rol5": rel(EHT / "b2rol5/identity.json"), "d1ghand": rel(EHT / "ghand/identity.json")},
             "files": files}
    save(P["identity"], ident)                              # atomic replace: CMP's tools read outputs/evhandtrack/identity.json
    print("wrote", P["identity"], "(provisional)" if provisional else "")
    for f in FORMS:
        print(f, " / ".join(f"{x:.2f}" for x in ident["three_seed_five"][f]))
    if provisional:
        return
    note = (f"{A['name']} identity ({rel(P['identity'])}); PFO engine E1, deterministic flags; cost from "
            "tools/evhandtrack/cost2.py's block (default flags, fast path, S43 in the same block); 'two_seed_mean' is the "
            "three-seed mean")
    rows = {A["rows"]: ("streaming", A["label"]), f"{A['rows']}_packet": ("packet", A["packet_label"])}
    for name, (form, label) in rows.items():
        per = {str(i): {"reading": s["forms"][form]["reading"], "checkpoint": s["checkpoint"],
                        **{b: s["forms"][form][b] for b in ("local", "global", "overall")}} for i, s in enumerate(seeds)}
        save(SEMKINE / f"{name}_main_row.json",
             {"arm": name, "label": label, "split": "val_core", "n_seeds": len(seeds), "form": form,
              "two_seed_mean": means[form], "per_seed": per, **cost_fields(cost[form], P["costs"][form]),
              "written_utc": now(), "note": note})
    save(SEMKINE / f"{A['rows']}_s43_main_row.json",
         {**{k: v for k, v in s43_row.items() if k in ("split", "n_seeds", "two_seed_mean", "per_seed")},
          "arm": f"{A['rows']}_s43", "label": f"S43（W50 原始，引擎三种子；{A['name']} 同块成本）",
          **cost_fields(cost["s43"], P["costs"]["s43"]), "written_utc": now(),
          "note": "accuracy = " + rel(S43_ROW) + f"; cost = S43's sparse executor in {A['name']}'s cost block"})
    for k in ("streaming", "packet", "streaming_ref"):
        print(k, {a: round(v, 2) for a, v in block["ratios_vs_s43_same_block"][k].items()})
    print("pids", block["gpu_compute_pids_seen"], "load", block["host_loadavg_1min"], "fast check",
          block["fast_check"]["max_abs_diff"])


def verify(arm="d1r"):
    ident = read(paths(arm)["identity"])
    bad = [p for p, h in ident["files"].items() if sha(ROOT / p) != h]
    require(not bad, "changed: " + ", ".join(bad))
    print("OK", len(ident["files"]), "files", "(provisional)" if ident.get("provisional") else "")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("write", "verify"))
    p.add_argument("--arm", default="d1r", choices=tuple(ARMS))
    p.add_argument("--provisional", action="store_true")
    a = p.parse_args(argv)
    if a.verb == "write":
        return write(a.arm, a.provisional)
    require(not a.provisional, "--provisional is for write")
    return verify(a.arm)


if __name__ == "__main__":
    main()
