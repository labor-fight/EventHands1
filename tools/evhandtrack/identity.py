#!/usr/bin/env python3
r"""EvHandTrack: identity of the converged method (docs/EXPERIMENT_RECORDS.md#evhandtrack-20261109,
#evhandtrack-str-20261109; AGENTS.md naming).

EvHandTrack = B2 + rol5, three seeds (TPD runs `tpd_rol5_s3407` / `s3408` / `s3409`):
    B2    PFO2 `fgs` = F + gshare: r_s2d router + PFA association -> event-vertex relation encoding -> one-round gated
          MeshGNN -> LBS slot pooling -> 51-D additive state update; B2 recipe, from scratch, 6000 steps
    rol5  TPD ROL fine-tune from B2's step-6000 checkpoint of the same seed: K = 5 consecutive 50 ms packets, gradient
          through the previous state, equal weights, 1500 steps, peak lr 1e-3, 100 warm-up, cosine to 0
Code, config keys, checkpoints and run names keep their identifiers; nothing is retrained.

Primary form = streaming (semkine/tpd_str.py; user 2026-11-09 "主行最好用流式形式"): each event's node selection and
token depend only on itself and the stream before it, the pixel time surface persists across updates, nothing is cut;
the router r_s2d stays per update interval (exempt). The weights are rol5's, read in this form with zero training
(TPD's `__str` readings). The ordinary packet form (not asynchronous) and the causal form are kept as descriptive rows.
Cost: outputs/evhandtrack/costs/ (tools/evhandtrack/cost.py, one block on GPU 4, S43's executor in the same block).

    python -m tools.evhandtrack.identity write    -> outputs/evhandtrack/identity.json; outputs/semkine/
                                                     evhandtrack_main_row.json (streaming), evhandtrack_packet_main_row.json,
                                                     evhandtrack_s43_main_row.json (S43, same cost block)
    python -m tools.evhandtrack.identity verify   -> re-hash every file identity.json names
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import now, read, require, save  # noqa: E402

NAME = "EvHandTrack"
SEEDS = (3407, 3408, 3409)
TPD = ROOT / "outputs/tpd/20261105"
PFO2 = ROOT / "outputs/pfo2/20261103"
EHT = ROOT / "outputs/evhandtrack"
ROUTER = ROOT / "outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt"
COSTS = {"streaming": EHT / "costs/evhandtrack_s3407__stream__default",
         "packet": EHT / "costs/evhandtrack_s3407__packet__default",
         "s43": EHT / "costs/s43_s3407__exec__default"}
SAMPLER = EHT / "logs/gpu4_sampler.log"
S43_ROW = ROOT / "outputs/semkine/evf_s43_main_row.json"
IDENTITY = EHT / "identity.json"
SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
FIVE = (("local", "mpjpe_ra_mm"), ("global", "mpjpe_ra_mm"), ("local", "mpvpe_ra_mm"), ("global", "mpvpe_ra_mm"),
        ("overall", "mpjpe_ra_mm"))
FORMS = {"streaming": "__str", "packet": "", "causal": "__causal"}
PRIMARY = "streaming"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def blocks(report):
    s = report["per_sequence"]
    return {"local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
            "overall": {k: report["overall"][k] for k in KEYS}}


def five(b):
    return [b[blk][k] for blk, k in FIVE]


def seed_identity(seed, files):
    run = f"tpd_rol5_s{seed}"
    receipt = read(TPD / "receipts" / f"{run}.json")
    require(receipt["status"] == "PASS", run + ": rol5 acceptance")
    ckpt = TPD / "runs" / run / f"{run}-step=1500.ckpt"
    h = sha(ckpt)
    require(h == receipt["checkpoint_sha256_by_step"]["1500"], run + ": checkpoint changed since acceptance")
    launch = read(TPD / "runs" / run / "tpd_launch.json")
    cfg = Path(launch["config"])
    require(sha(cfg) == launch["config_sha256"], run + ": training config changed since launch")
    base_run = f"pfo2_fgs_s{seed}"
    base = PFO2 / "runs" / base_run / f"{base_run}-step=6000.ckpt"
    require(str(base) in cfg.read_text(), run + ": INIT_FROM is not B2's step 6000 of the same seed")
    base_receipt = read(PFO2 / "receipts" / f"{base_run}.json")
    hb = sha(base)
    require(base_receipt["status"] == "PASS" and hb == base_receipt["checkpoint_sha256_by_step"]["6000"],
            base_run + ": B2 checkpoint changed since acceptance")
    forms = {}
    for form, suffix in FORMS.items():
        rp = TPD / "readings" / f"{run}{suffix}" / "e1" / "report.json"
        r = read(rp)
        require(r["status"] == "COMPLETE" and r["checkpoint_sha256"] == h, f"{run}{suffix}: reading of another file")
        ecfg = Path(r["identity"]["config"])
        require(sha(ecfg) == r["config_sha256"], f"{run}{suffix}: evaluation config changed since the reading")
        forms[form] = {"reading": rel(rp), "eval_config": rel(ecfg), **blocks(r)}
        files[rel(ecfg)] = r["config_sha256"]
        files[rel(rp)] = sha(rp)
    ap = EHT / "receipts" / f"async_evhandtrack_s{seed}.json"
    a = read(ap)
    require(a["status"] == "PASS" and a["form"] == "stream", f"{run}: streaming async check")
    files.update({rel(ckpt): h, rel(cfg): launch["config_sha256"], rel(base): hb, rel(ap): sha(ap)})
    return {"seed": seed, "run": run, "checkpoint": rel(ckpt), "checkpoint_sha256": h, "train_config": rel(cfg),
            "base": {"run": base_run, "checkpoint": rel(base), "checkpoint_sha256": hb},
            "receipt": rel(TPD / "receipts" / f"{run}.json"), "async_streaming": rel(ap), "forms": forms}


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
        require(cost[k]["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
        files[rel(d / "report.json")] = sha(d / "report.json")
    for k in ("streaming", "packet"):
        require(cost[k]["checkpoint_sha256"] == seeds[0]["checkpoint_sha256"], k + ": cost of EvHandTrack 3407")
    foreign = set()
    for line in SAMPLER.read_text().splitlines():
        foreign |= {p for p in line.split()[1:] if p.isdigit()}
    require(len(foreign) == len(COSTS), "GPU 4 carried a process other than the block's three runs: " + str(foreign))
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
             "definition": "B2 (PFO2 fgs, from scratch 6000 steps) + rol5 (TPD ROL, K = 5, gradient through the previous "
                           "state, 1500 steps), deployed router r_s2d + PFA; seeds 3407 / 3408 / 3409",
             "primary_form": PRIMARY, "seeds": seeds, "router": {"checkpoint": rel(ROUTER), "sha256": files[rel(ROUTER)]},
             "five_order": ["MPJPE-local", "MPJPE-global", "MPVPE-local", "MPVPE-global", "RA-MPJPE"],
             "three_seed_five": {f: five(means[f]) for f in FORMS},
             "s43_engine_five": s43,
             "minus_s43": {f: [a - b for a, b in zip(five(means[f]), s43)] for f in FORMS},
             "cost": {"readings": {k: rel(d) for k, d in COSTS.items()},
                      "ratios_vs_s43_same_block": {k: ratios(k) for k in ("streaming", "packet")},
                      "gpu4_compute_pids_seen": sorted(foreign), "host_loadavg_1min": [min(loads), max(loads)],
                      "note": "one block on GPU 4 (no other process on it); the host carried other users' jobs, so the "
                              "absolute latencies are above EVF's quiet block, the same-block ratios are the comparison"},
             "files": files}
    save(IDENTITY, ident)
    note = ("EvHandTrack identity (outputs/evhandtrack/identity.json); PFO engine E1, deterministic flags; cost from "
            "tools/evhandtrack/cost.py's block (GPU 4, default flags, S43 in the same block); 'two_seed_mean' is the "
            "three-seed mean")
    rows = {"evhandtrack": ("streaming", "EvHandTrack（B2 + rol5，三种子，流式形式）"),
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
    for k in ("streaming", "packet"):
        print(k, {a: round(v, 2) for a, v in ident["cost"]["ratios_vs_s43_same_block"][k].items()})
    print("gpu4 pids", sorted(foreign), "load", min(loads), max(loads))


def verify():
    ident = read(IDENTITY)
    bad = [p for p, h in ident["files"].items() if sha(ROOT / p) != h]
    require(not bad, "changed: " + ", ".join(bad))
    print("OK", len(ident["files"]), "files")


if __name__ == "__main__":
    {"write": write, "verify": verify}[sys.argv[1]]()
