#!/usr/bin/env python3
r"""EvHandTrack: identity of the converged method (docs/EXPERIMENT_RECORDS.md#evhandtrack-tree-20261110; AGENTS.md
naming). The user's EvHandTrack is the kinematic-tree method (2026-11-10 "我之前让你命名的EvHandTrack是那个利用运动学树的
那个实验"); the first identity, B2 + rol5, is kept as a comparison row (tools/evhandtrack/identity_b2rol5.py,
outputs/evhandtrack/b2rol5/).

EvHandTrack = PMR `d1ghand` + rol5, three seeds (PMR runs `pmr_d1ghand_rol5_s3407` / `s3408` / `s3409`), the best
three-seed row of the PMR study (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109 onward):
    input   as B2: r_s2d router + PFA association -> event-vertex relation encoding -> event-additive per-part statistics
            and learned channels for 16 part nodes and one hand node
    trunk   kinematic-tree propagation P (semkine/pmr_trunk.py `PMRNet`, ghand gates semkine/pmr_trunk6.py): 3 rounds of
            gated messages over the 62 typed edges of the 16-part tree + hand node (down / up / hand->part / part->hand);
            the gates come from the prior state's FK joint vectors in the hand frame (R_0^T v / 2 cm, length, edge type),
            shared by the rounds; local readout, no bypass: joint j's increment from node j only, root and translation
            from the hand node; 51-D additive state update
    recipe  d1ghand from scratch, 6000 steps, B2's recipe and data stream; rol5 = TPD ROL fine-tune from d1ghand's
            step-6000 checkpoint of the same seed: K = 5 consecutive 50 ms packets, gradient through the previous state,
            equal weights, 1500 steps, peak lr 1e-3, 100 warm-up, cosine to 0
Code, config keys, checkpoints and run names keep their identifiers; nothing is retrained.

Primary form = streaming (semkine/tpd_str.py; user 2026-11-09 "主行最好用流式形式"): each event's node selection and
token depend only on itself and the stream before it, the pixel time surface persists across updates, nothing is cut;
the router r_s2d stays per update interval (exempt). The readings are PMR's zero-training `__str` readings (stage S3c,
tools/pmr/run_forms.py); the ordinary packet form (not asynchronous) and the causal form are descriptive.
Async receipts: outputs/evhandtrack/receipts/ (tools/evhandtrack/asynccheck.py). Cost: outputs/evhandtrack/costs/
(tools/evhandtrack/cost2.py, one block on GPU 7, default flags, S43's executor in the same block; the trunk on
semkine/pmr_fast.py, the same function in fewer kernels, checked against the reference; the streaming reference
implementation is recorded as well).

    python -m tools.evhandtrack.identity write    -> outputs/evhandtrack/identity.json; outputs/semkine/
                                                     evhandtrack_main_row.json (streaming), evhandtrack_packet_main_row.json,
                                                     evhandtrack_s43_main_row.json (S43, same cost block)
    python -m tools.evhandtrack.identity verify   -> re-hash every file identity.json names
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evhandtrack.identity_b2rol5 import KEYS, S43_ROW, SEMKINE, SEEDS, blocks, five, rel, sha  # noqa: E402
from tools.pfo.common import now, read, require, save  # noqa: E402

NAME = "EvHandTrack"
PMR = ROOT / "outputs/pmr/20261109"
EHT = ROOT / "outputs/evhandtrack"
ROUTER = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
TRUNK = {"design": "ghand", "variant": "ghand", "module": "semkine.pmr_trunk6"}
COSTS = {"streaming": EHT / "costs/evhandtrack_s3407__stream__pmrfast__default",
         "packet": EHT / "costs/evhandtrack_s3407__packet__pmrfast__default",
         "streaming_ref": EHT / "costs/evhandtrack_s3407__stream__ref__default",
         "s43": EHT / "costs/s43_s3407__exec__default"}
FAST_CHECK = EHT / "costs/evhandtrack_s3407__fast_check.json"
SAMPLER = EHT / "logs/gpu7_sampler.log"
IDENTITY = EHT / "identity.json"
SUPERSEDED = EHT / "b2rol5/identity.json"
FORMS = {"streaming": "__str", "packet": "", "causal": "__causal"}
PRIMARY = "streaming"
LABEL = "EvHandTrack（运动学树 ghand + rol5，三种子，流式形式）"


def seed_identity(seed, files):
    run, base_run = f"pmr_d1ghand_rol5_s{seed}", f"pmr_d1ghand_s{seed}"
    receipt = read(PMR / "receipts" / f"{run}.json")
    require(receipt["status"] == "PASS", run + ": rol5 acceptance")
    ckpt = PMR / "runs" / run / f"{run}-step=1500.ckpt"
    h = sha(ckpt)
    require(h == receipt["checkpoint_sha256_by_step"]["1500"], run + ": checkpoint changed since acceptance")
    launch = read(PMR / "runs" / run / "pmr_launch.json")
    cfg = Path(launch["config"])
    require(sha(cfg) == launch["config_sha256"], run + ": training config changed since launch")
    require(launch["trunk"] == TRUNK and launch["rol"]["K"] == 5 and launch["rol"]["GRAD"], run + ": not ghand + rol5")
    base = PMR / "runs" / base_run / f"{base_run}-step=6000.ckpt"
    base_receipt = read(PMR / "receipts" / f"{base_run}.json")
    hb = sha(base)
    require(launch["init_from"] == str(base) and launch["init_from_sha256"] == hb,
            run + ": INIT_FROM is not d1ghand's step 6000 of the same seed")
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
    ap = EHT / "receipts" / f"async_evhandtrack_s{seed}.json"
    a = read(ap)
    require(a["status"] == "PASS" and a["form"] == "stream" and a["checkpoint_sha256"] == h,
            f"{run}: streaming async check")
    files.update({rel(ckpt): h, rel(cfg): launch["config_sha256"], rel(base): hb, rel(ap): sha(ap)})
    return {"seed": seed, "run": run, "checkpoint": rel(ckpt), "checkpoint_sha256": h, "train_config": rel(cfg),
            "base": {"run": base_run, "checkpoint": rel(base), "checkpoint_sha256": hb},
            "receipt": rel(PMR / "receipts" / f"{run}.json"), "async_streaming": rel(ap), "forms": forms}


def cost_fields(c, d):
    return {"latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
            "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
            "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": rel(d)}


def write():
    files = {}
    seeds = [seed_identity(s, files) for s in SEEDS]
    files[rel(ROUTER)] = sha(ROUTER)
    cost = {k: read(d / "report.json") for k, d in COSTS.items()}
    for k, d in COSTS.items():
        require(cost[k]["status"] == "COMPLETE" and cost[k]["policy"] == "default",
                k + ": the table's cost comes from a complete PyTorch-default-flag reading")
        files[rel(d / "report.json")] = sha(d / "report.json")
    for k in ("streaming", "packet", "streaming_ref"):
        require(cost[k]["checkpoint_sha256"] == seeds[0]["checkpoint_sha256"], k + ": cost of EvHandTrack 3407")
    check = read(FAST_CHECK)
    require(check["max_abs_diff"] < 1e-3, "fast path differs from the reference")
    files[rel(FAST_CHECK)] = sha(FAST_CHECK)
    foreign = set()
    for line in SAMPLER.read_text().splitlines():
        foreign |= {p for p in line.split()[1:] if p.isdigit()}
    require(len(foreign) == len(COSTS), "GPU 7 carried a process other than the block's four runs: " + str(foreign))
    loads = [float(x.split("load")[1]) for x in SAMPLER.read_text().splitlines() if "load" in x]
    means = {f: {b: {k: float(np.mean([s["forms"][f][b][k] for s in seeds])) for k in KEYS}
                 for b in ("local", "global", "overall")} for f in FORMS}
    s43_row = read(S43_ROW)
    s43 = five(s43_row["two_seed_mean"])

    def ratios(k):
        c, c43 = cost[k], cost["s43"]
        return {"params": c["params_total"] / c43["params_total"],
                "macs": c["macs_layers"]["median"] / c43["macs_layers"]["median"],
                "latency_row": c["latency_row"]["scaled_ms"] / c43["latency_row"]["scaled_ms"],
                "latency_e2e_p50": c["latency_e2e_ms"]["p50"] / c43["latency_e2e_ms"]["p50"],
                "latency_e2e_p99": c["latency_e2e_ms"]["p99"] / c43["latency_e2e_ms"]["p99"]}
    ident = {"name": NAME, "written_utc": now(),
             "definition": "PMR d1ghand (kinematic-tree propagation P over 16 part nodes + hand node, gates from the "
                           "prior FK geometry in the hand frame, local readout; from scratch 6000 steps, B2's recipe) + "
                           "rol5 (TPD ROL, K = 5, gradient through the previous state, 1500 steps), deployed router r_s2d "
                           "+ PFA; seeds 3407 / 3408 / 3409",
             "trunk": TRUNK, "primary_form": PRIMARY, "seeds": seeds,
             "router": {"checkpoint": rel(ROUTER), "sha256": files[rel(ROUTER)]},
             "five_order": ["MPJPE-local", "MPJPE-global", "MPVPE-local", "MPVPE-global", "RA-MPJPE"],
             "three_seed_five": {f: five(means[f]) for f in FORMS},
             "s43_engine_five": s43,
             "minus_s43": {f: [a - b for a, b in zip(five(means[f]), s43)] for f in FORMS},
             "cost": {"readings": {k: rel(d) for k, d in COSTS.items()}, "fast_check": check,
                      "ratios_vs_s43_same_block": {k: ratios(k) for k in ("streaming", "packet", "streaming_ref")},
                      "gpu7_compute_pids_seen": sorted(foreign), "host_loadavg_1min": [min(loads), max(loads)],
                      "note": "one block on GPU 7 (no other process on it) while CMP's three EventHands trainings ran "
                              "on GPUs 4-6; the main row's cost is the fast path (pmr_fast, max |difference| "
                              f"{check['max_abs_diff']:.1e} over 100 packets), the reference implementation is recorded"},
             "superseded": {"identity": rel(SUPERSEDED), "name": "B2 + rol5 (EvHandTrack 2026-11-09, renamed 2026-11-10)"},
             "files": files}
    save(IDENTITY, ident)                                   # atomic replace: CMP's tools read this path
    note = ("EvHandTrack identity (outputs/evhandtrack/identity.json); PFO engine E1, deterministic flags; cost from "
            "tools/evhandtrack/cost2.py's block (GPU 7, default flags, fast path, S43 in the same block); 'two_seed_mean' "
            "is the three-seed mean")
    rows = {"evhandtrack": ("streaming", LABEL),
            "evhandtrack_packet": ("packet", "EvHandTrack 普通分包形式（不异步，描述）")}
    for arm, (form, label) in rows.items():
        per = {str(i): {"reading": s["forms"][form]["reading"], "checkpoint": s["checkpoint"],
                        **{b: s["forms"][form][b] for b in ("local", "global", "overall")}} for i, s in enumerate(seeds)}
        save(SEMKINE / f"{arm}_main_row.json",
             {"arm": arm, "label": label, "split": "val_core", "n_seeds": len(seeds), "form": form,
              "two_seed_mean": means[form], "per_seed": per, **cost_fields(cost[form], COSTS[form]),
              "written_utc": now(), "note": note})
    save(SEMKINE / "evhandtrack_s43_main_row.json",
         {**{k: v for k, v in s43_row.items() if k in ("split", "n_seeds", "two_seed_mean", "per_seed")},
          "arm": "evhandtrack_s43", "label": "S43（W50 原始，引擎三种子；EvHandTrack 同块成本）",
          **cost_fields(cost["s43"], COSTS["s43"]), "written_utc": now(),
          "note": "accuracy = " + rel(S43_ROW) + "; cost = S43's sparse executor in EvHandTrack's cost block"})
    print("wrote", IDENTITY)
    for f in FORMS:
        print(f, " / ".join(f"{x:.2f}" for x in ident["three_seed_five"][f]))
    for k in ("streaming", "packet", "streaming_ref"):
        print(k, {a: round(v, 2) for a, v in ident["cost"]["ratios_vs_s43_same_block"][k].items()})
    print("gpu7 pids", sorted(foreign), "load", min(loads), max(loads), "fast check", check["max_abs_diff"])


def verify():
    ident = read(IDENTITY)
    bad = [p for p, h in ident["files"].items() if sha(ROOT / p) != h]
    require(not bad, "changed: " + ", ".join(bad))
    print("OK", len(ident["files"]), "files")


if __name__ == "__main__":
    {"write": write, "verify": verify}[sys.argv[1]]()
