#!/usr/bin/env python3
"""SBA round 3, E1 / E3 single-window landscape (docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022).

P0 / Q0's packets (zgz from outputs/xdl/packet_bank.json; the 576 training-subject 'check' packets), the E0 hypotheses
(outputs/sba/r3/e0/states_<domain>.npz), each scored on the packet's real events with the P4 emission model and the
nuisances profiled per hypothesis (tools/sba/r3_lik.py), at every sigma of the gauge grid; donors (wrong content,
count-matched) on the same features. Profile log-likelihoods are stored per hypothesis x sigma; the sigma arms
(P = profiled per hypothesis, R = recording-level cross-fit, C = GT-optimal per packet) are formed in `summarize`.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python tools/sba/r3_e1.py selftest
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python tools/sba/r3_e1.py run --domain check --shard 0 --shards 6
    python tools/sba/r3_e1.py summarize --domains check && python tools/sba/r3_e1.py calib_gate
    python tools/sba/r3_e1.py summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import C  # noqa: E402
from sba.p0_landscape import packets  # noqa: E402
from sba import r3_lik as L  # noqa: E402
from sba import r3_probe as PR  # noqa: E402

R3 = CM.OUT / "r3"
OUT = R3 / "e1"
E0 = R3 / "e0"
PROTOCOL = R3 / "protocol.json"
HYP = PR.HYP
H_IDX = {h: i for i, h in enumerate(HYP)}
DONOR_H = ["clean", "pip+10", "pip-10", "mcpf+10", "mcpf-10", "mcpa+10", "mcpa-10", "synrc+5", "synrc-5", "root+10",
           "root-10", "fsw"]
VARIANTS = ("mg", "m1", "geo")
FIT_SUBJ = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SEL_SUBJ = ("ycy", "ylf")
#: Q0 V3 GT-preference of the twist probe (outputs/sba/q0/result.json) -- the implementation regression (G0.5)
Q0_TW = {"zgz_local": 0.5699, "zgz_global": 0.6444, "check_fit_local": 0.5897, "check_fit_global": 0.7701,
         "check_sel_local": 0.6016, "check_sel_global": 0.7422}

# ------------------------------------------------------------------------------------------------ frozen gates
CELLS = {"F5": ["pip+5", "pip-5", "mcpf+5", "mcpf-5"], "F10": ["pip+10", "pip-10", "mcpf+10", "mcpf-10"],
         "S5": ["synrc+5", "synrc-5"]}
PASS_THR = {"F5": 0.70, "F10": 0.75, "S5": 0.70}
FALS_THR = {"F10": 0.58, "S5": 0.58}
CI_SLACK = 0.04
VIS_MIN_AREA = 40.0
DONOR_BAND = (0.40, 0.60)
DONOR_VALIDITY = {"F10": ["pip+10", "pip-10", "mcpf+10", "mcpf-10"], "S5": ["synrc+5", "synrc-5"]}
DONOR_FAMILIES = {"pip10": ["pip+10", "pip-10"], "mcpf10": ["mcpf+10", "mcpf-10"], "mcpa10": ["mcpa+10", "mcpa-10"],
                  "synrc5": ["synrc+5", "synrc-5"], "root10": ["root+10", "root-10"], "fsw": ["fsw"]}
FAIL_MAX = 0.02
EDGE_MAX = 0.05
TIER_REAL = (0.15, 0.90)
G0_ROOT5 = 0.80
G0_GAUGE_R = 0.75
G0_Q0_TOL = 0.03
G0_DOSE = 0.03
G0_SEL_GAP = 0.10
N_BOOT = 2000
BOOT_SEED = 20261022
DESCRIPTIVE = {
    "S10": ["synrc+10", "synrc-10"], "syn3": ["syn+3", "syn-3"], "syn5": ["syn+5", "syn-5"], "syn10": ["syn+10", "syn-10"],
    "pip5": ["pip+5", "pip-5"], "pip10": ["pip+10", "pip-10"], "mcpf5": ["mcpf+5", "mcpf-5"],
    "mcpf10": ["mcpf+10", "mcpf-10"], "mcpa5": ["mcpa+5", "mcpa-5"], "mcpa10": ["mcpa+10", "mcpa-10"],
    "thm5": ["thm+5", "thm-5"], "thm10": ["thm+10", "thm-10"], "root5": ["root+5", "root-5"],
    "root10": ["root+10", "root-10"], "trans5": ["trans+5", "trans-5"], "trans10": ["trans+10", "trans-10"],
    "tw20": ["tw+20", "tw-20"], "synM5": ["synM+5", "synM-5"], "synM10": ["synM+10", "synM-10"], "fsw": ["fsw"],
    "gauge": ["gauge0.97", "gauge1.03"], "sm25": ["sm25"]}


def frozen_constants() -> dict:
    return {"HYP": HYP, "DONOR_H": DONOR_H, "SIG": L.SIG, "CELLS": CELLS, "PASS_THR": PASS_THR, "FALS_THR": FALS_THR,
            "CI_SLACK": CI_SLACK, "VIS_MIN_AREA": VIS_MIN_AREA, "DONOR_BAND": DONOR_BAND,
            "DONOR_VALIDITY": DONOR_VALIDITY, "DONOR_FAMILIES": DONOR_FAMILIES, "FAIL_MAX": FAIL_MAX,
            "EDGE_MAX": EDGE_MAX, "TIER_REAL": TIER_REAL, "G0_ROOT5": G0_ROOT5, "G0_GAUGE_R": G0_GAUGE_R,
            "G0_Q0_TOL": G0_Q0_TOL, "Q0_TW": Q0_TW, "G0_DOSE": G0_DOSE, "G0_SEL_GAP": G0_SEL_GAP, "N_BOOT": N_BOOT,
            "BOOT_SEED": BOOT_SEED, "FIT_SUBJ": FIT_SUBJ, "SEL_SUBJ": SEL_SUBJ, "E0_VALID": PR.VALID}


#: files whose SHA-256 is frozen before any GPU reading (relative to the repository)
FROZEN_FILES = (
    "tools/sba/r3_probe.py", "tools/sba/r3_lik.py", "tools/sba/r3_e1.py", "tools/sba/r3_e8.py", "tests/test_sba_r3.py",
    "outputs/sba/run_r3.sh", "tools/sba/q0_landscape.py", "tools/sba/p0_landscape.py", "tools/sba/p0_visibility.py",
    "tools/sba/common.py", "tools/xvm/common.py", "semkine/sba.py", "semkine/sba_emission.py",
    "outputs/xdl/packet_bank.json", "outputs/xde/e2/protocol.json", "outputs/sba/p4/fit_raw_b1.json",
    "outputs/sba/p4/patches.npy", "outputs/sba/s0/calibration.json", "outputs/sba/r3/e0/result.json",
    "outputs/sba/r3/e0/states_zgz_local.npz", "outputs/sba/r3/e0/states_zgz_global.npz",
    "outputs/sba/r3/e0/states_check.npz", "outputs/sba/r3/e0/probes_zgz_local.json",
    "outputs/sba/r3/e0/probes_zgz_global.json", "outputs/sba/r3/e0/probes_check.json")


def freeze():
    """Write outputs/sba/r3/protocol.json: SHA-256 of the frozen files and every frozen constant (E0 / E1 / E8)."""
    from sba import r3_e8 as E8
    e0 = json.loads((E0 / "result.json").read_text())
    if not e0["validity"]["pass"]:
        raise SystemExit("E0 validity failed: nothing to freeze")
    p = {"prereg": "docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022", "gpus": L.GPUS,
         "sha256": {f: C.sha(CM.REPO / f) for f in FROZEN_FILES},
         "constants": {"E0": {"TIERS_F": PR.TIERS_F, "TIERS_W": PR.TIERS_W, "TIERS_RC": PR.TIERS_RC,
                              "SYN_RATIO": PR.SYN_RATIO, "THUMB_LIFT": PR.THUMB_LIFT, "HI_DEG": PR.HI_DEG,
                              "ITERS": PR.ITERS, "TOL": PR.TOL, "GAUGE": PR.GAUGE, "FSW_MIN_MM": PR.FSW_MIN_MM,
                              "VALID": PR.VALID, "TWIST_HELD": PR.TWIST_HELD},
                       "E1": frozen_constants(), "E8": E8.frozen_constants(),
                       "newton": {"iters": 40, "tol": 1e-12, "CONV_TOL": L.CONV_TOL}},
         "e0_validity": e0["validity"]}
    PROTOCOL.write_text(json.dumps(p, indent=1))
    print(f"frozen {len(p['sha256'])} files -> {PROTOCOL}")


def require_protocol():
    """GPU stages refuse to start unless every frozen file still has its frozen SHA-256."""
    if not PROTOCOL.exists():
        raise SystemExit(f"{PROTOCOL} missing: freeze the protocol before any GPU reading")
    p = json.loads(PROTOCOL.read_text())
    bad = [f for f, h in p["sha256"].items() if C.sha(CM.REPO / f) != h]
    if bad:
        raise SystemExit(f"protocol SHA mismatch: {bad}")


# ------------------------------------------------------------------------------------------------ run
def find_donor(ctx, sd, rows, index: int, prev: int, end: int, x0, x1, n_target: int, betas, max_try: int = 60):
    """Wrong-content control: events of the first row from index + n/2 on whose GT end differs by >= 40 mm (wrist)
    or >= 15 deg (root), re-timed onto (prev, end]; in-crop count >= max(100, n_target / 2), thinned to n_target if
    above 2 n_target."""
    n = len(rows)
    tgt = sd.gt(end)
    wt = ctx.wrist(tgt, betas)[0]
    tried = 0
    for k in range(n):
        r = rows[(index + n // 2 + k) % n]
        g = sd.gt(int(r["end_ms"]))
        far = float((ctx.wrist(g, betas)[0] - wt).norm()) * 1000 >= 40.0 or CM.geo_deg(tgt[3:6], g[3:6]) >= 15.0
        if not far:
            continue
        tried += 1
        if tried > max_try:
            break
        evms = CM.retime(sd.events(int(r["previous_end_ms"]), int(r["end_ms"])), int(r["previous_end_ms"]), prev, end)
        if len(evms[1]) == 0:
            continue
        wd = CM.window(ctx, sd, prev, end, x0, x1, events=evms)
        if wd.n_crop < max(CM.MIN_EVENTS, 0.5 * n_target):
            continue
        thinned = False
        if wd.n_crop > 2 * n_target:
            evms = L.thin(evms, n_target / wd.n_crop, L.seed_of(sd.name, index, "donor"))
            wd = CM.window(ctx, sd, prev, end, x0, x1, events=evms)
            thinned = True
        return wd, {"end": int(r["end_ms"]), "index": int(r["index"]), "n_crop": wd.n_crop, "thinned": thinned}
    return None, None


def score_packet(ctx, lik, sd, row, rows, index, X):
    """All stored readings of one informative packet. X (H, 2, 51) hypothesis knots."""
    prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
    x0, x1 = sd.gt(prev), sd.gt(end)
    win = CM.window(ctx, sd, prev, end, x0, x1)
    rec = {"seq": sd.name, "index": index, "end": end, "run_id": int(row["run_id"]), "n_events": win.n_events,
           "n_crop": win.n_crop}
    if win.n_events < CM.MIN_EVENTS:
        return rec
    wd, dinfo = find_donor(ctx, sd, rows, index, prev, end, x0, x1, win.n_crop, sd.betas)
    didx = [H_IDX[h] for h in DONOR_H]
    ll = {v: np.full((len(HYP), len(L.SIG)), np.nan) for v in VARIANTS}
    dl = np.full((len(DONOR_H), len(L.SIG)), np.nan)
    kl = twin = None
    for si, sg in enumerate(L.SIG):
        Gs, Gg = [], []
        for hi, h in enumerate(HYP):
            g, gg = lik.groups(sd, win, X[hi, 0], X[hi, 1], sg * PR.GAUGE.get(h, 1.0))
            Gs.append(g)
            Gg.append(gg)
        G, Gg = torch.stack(Gs), torch.stack(Gg)
        v_mg, w_mg = L.newton(*L.problem(G, win, "mg"), nan_unconverged=True)
        ll["mg"][:, si] = v_mg.cpu().numpy()
        ll["m1"][:, si] = L.newton(*L.problem(G, win, "m1"), nan_unconverged=True)[0].cpu().numpy()
        ll["geo"][:, si] = L.newton(*L.problem(Gg, win, "mg"), nan_unconverged=True)[0].cpu().numpy()
        if wd is not None:
            dl[:, si] = L.newton(*L.problem(G[didx], wd, "mg"), nan_unconverged=True)[0].cpu().numpy()
        if sg == 1.0:
            lam = L.dense_rate(G, win, w_mg[0])
            kl = L.kl_poisson(lam[0], lam).cpu().numpy()
            gen = torch.Generator(device=lam.device).manual_seed(L.seed_of(sd.name, index, "twin"))
            Nt = torch.poisson(lam[0].double().clamp(min=0), generator=gen).float()
            twin = L.newton(*L.problem(G, win, "mg", N=Nt), nan_unconverged=True)[0].cpu().numpy()
        del G, Gg, Gs
    rec.update({"ll": {v: ll[v].tolist() for v in VARIANTS}, "donor": dinfo,
                "D": dl.tolist() if wd is not None else None, "kl": kl.tolist(), "twin": twin.tolist()})
    return rec


def load_states(domain):
    z = np.load(E0 / f"states_{domain}.npz")
    return {k: i for i, k in enumerate(z["keys"].tolist())}, z["X"]


def run(domain, shard, shards, limit, out_dir=OUT):
    dev = C.require_gpu(allowed=L.GPUS)
    if out_dir == OUT:
        require_protocol()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    lik = L.Lik(ctx)
    keys, Xall = load_states(domain)
    items = packets(domain)[shard::shards]
    if limit:
        items = items[:limit]
    cache, recs = {}, []
    t0 = time.time()
    for seq, d, index, row, rows in items:
        if seq not in cache:
            cache = {seq: CM.SeqData(seq, d, dev)}
        sd = cache[seq]
        X = torch.from_numpy(Xall[keys[f"{seq}|{index}"]]).to(dev)
        recs.append(score_packet(ctx, lik, sd, row, rows, index, X))
        if len(recs) % 25 == 0:
            print(json.dumps({"domain": domain, "shard": shard, "done": len(recs), "of": len(items),
                              "s": round(time.time() - t0)}), flush=True)
    path = out_dir / "raw" / f"{domain}__{shard:02d}of{shards:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hyp": HYP, "donor_h": DONOR_H, "sig": L.SIG, "records": recs,
                                "wall_s": time.time() - t0}))
    CM.receipt(out_dir / "raw" / f"{domain}__{shard:02d}", {"probe": "SBA r3 E1", "domain": domain, "shard": shard},
               scripts=(__file__, L.__file__, PR.__file__, CM.REPO / "semkine" / "sba_emission.py"))


def selftest():
    """Engineering check on 2 training-subject packets (no zgz): batched Newton == Q0.profile, and the 'mg' sigma = 1
    log-likelihood of the clean / twist hypotheses == Q0.scores()['V2'] (the same states)."""
    from sba.q0_landscape import Q0, profile, hyp_states
    dev = C.require_gpu(allowed=L.GPUS)
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    lik = L.Lik(ctx)
    q = Q0(ctx, json.loads((L.P4 / "fit_raw_b1.json").read_text()), json.loads(CM.CALIB.read_text()))
    keys, Xall = load_states("check")
    out = []
    for seq, d, index, row, rows in packets("check")[:2]:
        sd = CM.SeqData(seq, d, dev)
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        x0, x1 = sd.gt(prev), sd.gt(end)
        win = CM.window(ctx, sd, prev, end, x0, x1)
        X = torch.from_numpy(Xall[keys[f"{seq}|{index}"]]).to(dev)
        for h, qn in (("clean", "clean"), ("tw+20", "f+20"), ("tw-20", "f-20")):
            a, b = hyp_states(ctx, sd, x0, x1, qn, index % 15)
            dx = float(torch.stack([(a - X[H_IDX[h], 0]).abs().max(), (b - X[H_IDX[h], 1]).abs().max()]).max())
            g, _ = lik.groups(sd, win, X[H_IDX[h], 0], X[H_IDX[h], 1], 1.0)
            mine = float(L.newton(*L.problem(g[None], win, "mg"))[0][0])
            ref = q.scores(sd, win, a, b, 1.0)["V2"]
            Xp, Xt, B, Bt, N = L.problem(g[None], win, "mg")
            ref_prof = profile(Xp[0], Xt[0], B, Bt, N)
            out.append({"packet": f"{seq}|{index}", "hyp": h, "state_maxdiff": dx, "mine": mine, "q0_V2": ref,
                        "q0_profile_same_problem": ref_prof, "diff_V2": mine - ref, "diff_profile": mine - ref_prof})
    ok = all(abs(o["diff_V2"]) < 1e-3 and abs(o["diff_profile"]) < 1e-6 * max(1.0, abs(o["mine"])) and
             o["state_maxdiff"] < 1e-5 for o in out)
    print(json.dumps({"ok": ok, "rows": out}, indent=1))
    return ok


# ------------------------------------------------------------------------------------------------ summarize
def base_of(seq: str) -> str:
    for v in ("_v2", "_v3", "_v4"):
        seq = seq.replace(v, "")
    return seq


def subject_of(seq: str) -> str:
    return seq.split("_")[0]


def load(domains):
    recs = []
    for domain in domains:
        for f in sorted((OUT / "raw").glob(f"{domain}__*of*.json")):
            recs += json.loads(f.read_text())["records"]
    return recs


def e0_index():
    out = {}
    for domain in PR.DOMAINS:
        p = E0 / f"probes_{domain}.json"
        if p.exists():
            for r in json.loads(p.read_text())["records"]:
                out[f"{r['seq']}|{r['index']}"] = r
    return out


def sigma_r_crossfit(recs):
    """Recording-level sigma_r per packet from the clean 'mg' likelihood summed over the other half of the recording
    (zgz: alternating 20-packet blocks in time order; check: alternating base packets (end ms) of the recording)."""
    by = defaultdict(list)
    for r in recs:
        if "ll" in r:
            by[base_of(r["seq"])].append(r)
    out, full = {}, {}
    for b, rs in by.items():
        if b in CM.ZGZ:
            order = sorted(rs, key=lambda r: r["index"])
            par = {id(r): (k // 20) % 2 for k, r in enumerate(order)}
        else:
            ends = sorted({r["end"] for r in rs})
            rank = {e: k % 2 for k, e in enumerate(ends)}
            par = {id(r): rank[r["end"]] for r in rs}
        clean = {id(r): np.asarray(r["ll"]["mg"][0]) for r in rs}
        tot = {p: sum(clean[id(r)] for r in rs if par[id(r)] == p) for p in (0, 1)}
        sig = {p: L.refine_max(tot[1 - p])[1] if np.ndim(tot[1 - p]) else 1.0 for p in (0, 1)}
        for r in rs:
            out[f"{r['seq']}|{r['index']}"] = sig[par[id(r)]]
        full[b] = L.refine_max(sum(clean.values()))[1]
    return out, full


def packet_scores(r, sig_r, arm: str, variant: str = "mg", donor: bool = False):
    """{hyp: LL at the arm's sigma} of one record (P: refined max; R: recording cross-fit; C: clean optimum;
    Q0: max over Q0's 8 sigma points without refinement)."""
    names = DONOR_H if donor else HYP
    M = np.asarray(r["D"] if donor else r["ll"][variant], np.float64)
    clean_mg = np.asarray(r["ll"]["mg"][0], np.float64)
    if arm == "C":
        s = L.refine_max(clean_mg)[1]
    elif arm == "R":
        s = sig_r[f"{r['seq']}|{r['index']}"]
    out = {}
    for i, h in enumerate(names):
        y = M[i]
        if not np.all(np.isfinite(y)):
            out[h] = np.nan
        elif arm == "P":
            out[h] = L.refine_max(y)[0]
        elif arm == "Q0":
            out[h] = float(np.max(y[list(L.Q0_GRID)]))
        else:
            out[h] = L.interp(y, s)
    return out


def pref_of(sc: dict, hyps, strict: bool = False) -> float:
    """GT preference of one packet over hypotheses `hyps` (ties / undefined = 0.5; strict: ties = 0, as Q0)."""
    v = []
    for h in hyps:
        a, b = sc["clean"], sc[h]
        if not (np.isfinite(a) and np.isfinite(b)):
            v.append(0.5)
        elif strict:
            v.append(1.0 if a > b else 0.0)
        else:
            v.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return float(np.mean(v))


def block_of(r) -> str:
    if r["seq"] == "zgz_local":
        return f"run{r['run_id']}"
    if r["seq"] == "zgz_global":
        return f"b{r['index'] // 20}"
    return f"{base_of(r['seq'])}|{r['end']}"


def boot(scores, blocks, n=N_BOOT, seed=BOOT_SEED):
    """(point, lo95, hi95) of the packet mean with a block bootstrap."""
    s = np.asarray(scores, np.float64)
    if len(s) == 0:
        return float("nan"), float("nan"), float("nan")
    ub = sorted(set(blocks))
    bi = np.array([ub.index(b) for b in blocks])
    sums = np.bincount(bi, weights=s, minlength=len(ub))
    cnts = np.bincount(bi, minlength=len(ub)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ub), size=(n, len(ub)))
    bs = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(s.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def table(rs, sig_r, arm, cells, variant="mg", donor=False, strict=False):
    out = {}
    scs = [packet_scores(r, sig_r, arm, variant, donor) for r in rs]
    blocks = [block_of(r) for r in rs]
    for name, hyps in cells.items():
        if donor and not all(h in DONOR_H for h in hyps):
            continue
        p = [pref_of(sc, hyps, strict) for sc in scs]
        out[name] = dict(zip(("pref", "lo", "hi"), boot(p, blocks)))
        out[name]["n"] = len(p)
    return out


def twin_kl_table(rs):
    """Poisson-twin ceiling (events drawn from the model at sigma = 1) and median KL per descriptive family."""
    out = {}
    for name, hyps in {**CELLS, **DESCRIPTIVE}.items():
        tw = [pref_of({h: r["twin"][H_IDX[h]] for h in ["clean"] + hyps}, hyps) for r in rs]
        kl = [np.mean([r["kl"][H_IDX[h]] for h in hyps]) for r in rs]
        out[name] = {"twin_pref": float(np.mean(tw)), "kl_median": float(np.median(kl))}
    return out


def domain_summary(rs, sig_r, e0):
    inf = [r for r in rs if "ll" in r]
    cells = {**CELLS, **DESCRIPTIVE}
    d = {"n_packets": len(rs), "n_informative": len(inf)}
    for arm in ("P", "R", "C"):
        d[arm] = table(inf, sig_r, arm, cells)
    d["Q0rep_strict"] = table(inf, sig_r, "Q0", {"tw20": ["tw+20", "tw-20"]}, strict=True)
    for v in ("m1", "geo"):
        d[f"P_{v}"] = table(inf, sig_r, "P", cells, variant=v)
    dr = [r for r in inf if r.get("D")]
    d["donor_P"] = table(dr, sig_r, "P", {**DONOR_FAMILIES, **DONOR_VALIDITY}, donor=True)
    d["donor_n"] = len(dr)
    d["twin_kl"] = twin_kl_table(inf)
    fails = sum(1 for r in inf for v in r["ll"]["mg"] if not np.all(np.isfinite(v)))
    d["fail_rate"] = fails / max(1, len(inf) * len(HYP))
    d["edge_rate_clean"] = float(np.mean([L.refine_max(r["ll"]["mg"][0])[2] for r in inf])) if inf else float("nan")
    # visibility of the falsifiable cells (E0 areas on the informative packets)
    for c in ("F10", "S5"):
        d[f"area_median_{c}"] = float(np.median([e0[f"{r['seq']}|{r['index']}"]["hyps"][h]["area"]
                                                 for r in inf for h in CELLS[c]])) if inf else float("nan")
    # preference by E0 visible-area bin (P arm), and the all-packet preference with uninformative packets as 0.5
    bins = (0, 5, 10, 20, 40, 80, 160, 320, 1e9)
    byb = {}
    for name in ("F5", "F10", "S5", "mcpa5", "mcpa10", "thm5", "syn5", "tw20"):
        hyps = cells[name]
        vals = defaultdict(list)
        for r in inf:
            sc = packet_scores(r, sig_r, "P")
            for h in hyps:
                a = e0[f"{r['seq']}|{r['index']}"]["hyps"][h]["area"]
                k = int(np.searchsorted(bins, a, side="right") - 1)
                vals[f"[{bins[k]:g},{bins[k + 1]:g})"].append(pref_of(sc, [h]))
        byb[name] = {k: {"pref": float(np.mean(v)), "n": len(v)} for k, v in sorted(vals.items())}
    d["P_by_area"] = byb
    allp = {}
    for c, hyps in CELLS.items():
        sc = [pref_of(packet_scores(r, sig_r, "P"), hyps) if "ll" in r else 0.5 for r in rs]
        allp[c] = float(np.mean(sc))
    d["P_all_packets"] = allp
    # sigma diagnostics: sigma* under each cell hypothesis minus sigma* under GT (P arm)
    ds = {}
    for c, hyps in {**CELLS, "gauge": DESCRIPTIVE["gauge"], "syn5": DESCRIPTIVE["syn5"]}.items():
        diffs = [L.refine_max(r["ll"]["mg"][H_IDX[h]])[1] - L.refine_max(r["ll"]["mg"][0])[1] for r in inf
                 for h in hyps]
        ds[c] = {"median": float(np.median(diffs)), "mean": float(np.mean(diffs))}
    d["dsigma_star"] = ds
    # per-run rates (runs with >= 20 informative packets) and leave-largest-run-out (zgz_local)
    if rs and rs[0]["seq"] == "zgz_local":
        runs = defaultdict(list)
        for r in inf:
            runs[r["run_id"]].append(r)
        pr = {}
        for c, hyps in CELLS.items():
            rates = [np.mean([pref_of(packet_scores(r, sig_r, "P"), hyps) for r in v]) for v in runs.values()
                     if len(v) >= 20]
            big = max(runs, key=lambda k: len(runs[k]))
            rest = [r for r in inf if r["run_id"] != big]
            pr[c] = {"median_of_run_rates": float(np.median(rates)) if rates else float("nan"),
                     "leave_largest_run_out": float(np.mean([pref_of(packet_scores(r, sig_r, "P"), hyps)
                                                             for r in rest]))}
        d["runs"] = pr
    return d


def cell_verdict(dsum, cell, area_ok=True):
    """'PASS' / 'FALSIFIED' / 'PROBE-LIMITED' / 'PARTIAL' for one cell from the arm tables."""
    thr = PASS_THR[cell]
    pas = all(dsum[a][cell]["pref"] >= thr and dsum[a][cell]["lo"] >= thr - CI_SLACK for a in ("P", "R"))
    if pas:
        return "PASS"
    if cell in FALS_THR:
        tf = FALS_THR[cell]
        fal = all(dsum[a][cell]["pref"] < tf and dsum[a][cell]["hi"] < tf + CI_SLACK for a in ("P", "R", "C"))
        if fal:
            return "FALSIFIED" if area_ok else "PROBE-LIMITED"
    return "PARTIAL"


def verdict(dsum):
    cv = {"F5": cell_verdict(dsum, "F5"),
          "F10": cell_verdict(dsum, "F10", area_ok=dsum["area_median_F10"] >= VIS_MIN_AREA),
          "S5": cell_verdict(dsum, "S5")}
    don = {c: dsum["donor_P"][c]["pref"] for c in DONOR_VALIDITY if c in dsum["donor_P"]}
    valid = bool(len(don) == len(DONOR_VALIDITY) and all(DONOR_BAND[0] <= v <= DONOR_BAND[1] for v in don.values())
                 and dsum["fail_rate"] < FAIL_MAX)
    if valid and all(v == "PASS" for v in cv.values()):
        v = "PASS"
    elif cv["F10"] == "FALSIFIED" or cv["S5"] == "FALSIFIED":
        v = "FALSIFIED"
    else:
        v = "PARTIAL"
    return {"verdict": v, "cells": cv, "valid": valid, "donor_validity": don, "fail_rate": dsum["fail_rate"]}


def pregate(dom, e0_all):
    """Training-subject pre-gate on check_fit_{local,global} (G0.1-G0.5 stop; G0.6-G0.7 flags)."""
    g = {}
    for lg in ("local", "global"):
        f = dom[f"check_fit_{lg}"]
        s = dom.get(f"check_sel_{lg}")
        don = {k: f["donor_P"][k]["pref"] for k in DONOR_FAMILIES if k in f["donor_P"]}
        g[lg] = {
            "G0.1_root5": f["P"]["root5"]["pref"],
            "G0.1": f["P"]["root5"]["pref"] >= G0_ROOT5,
            "G0.2_donors": don,
            "G0.2": len(don) == len(DONOR_FAMILIES) and all(DONOR_BAND[0] <= v <= DONOR_BAND[1] for v in don.values()),
            "G0.3_gauge": {"P": f["P"]["gauge"]["pref"], "R": f["R"]["gauge"]["pref"]},
            "G0.3": DONOR_BAND[0] <= f["P"]["gauge"]["pref"] <= DONOR_BAND[1] and f["R"]["gauge"]["pref"] >= G0_GAUGE_R,
            "G0.4_numerics": {"fail_rate": f["fail_rate"], "edge_rate": f["edge_rate_clean"]},
            "G0.4": f["fail_rate"] < FAIL_MAX and f["edge_rate_clean"] < EDGE_MAX,
            "G0.5_q0rep": {"fit": f["Q0rep_strict"]["tw20"]["pref"],
                           "sel": s["Q0rep_strict"]["tw20"]["pref"] if s else None},
            "G0.5": abs(f["Q0rep_strict"]["tw20"]["pref"] - Q0_TW[f"check_fit_{lg}"]) <= G0_Q0_TOL
            and (s is None or abs(s["Q0rep_strict"]["tw20"]["pref"] - Q0_TW[f"check_sel_{lg}"]) <= G0_Q0_TOL),
            "G0.6_flag": not (f["P"]["F10"]["pref"] >= f["P"]["F5"]["pref"] + G0_DOSE
                              and f["P"]["syn10"]["pref"] >= f["P"]["syn5"]["pref"] + G0_DOSE),
            "G0.7_flag": s is not None and any(f["P"][c]["pref"] - s["P"][c]["pref"] > G0_SEL_GAP for c in CELLS),
            "cells": verdict(f),
        }
    # tier realisation at the start knot (E0, all check packets): |e(x0) / target - 1| <= 0.15 for >= 90 %
    rr = [abs(m["e_metric_x0"] / m["target"] - 1.0) <= TIER_REAL[0] for r in e0_all.values()
          if r["seq"] not in CM.ZGZ for h, m in r["hyps"].items() if PR.split(h)[0] in PR.BISECT]
    tier = float(np.mean(rr))
    stop_ok = all(g[lg][k] for lg in ("local", "global") for k in ("G0.1", "G0.2", "G0.3", "G0.4", "G0.5"))
    return {"by_domain": g, "tier_realisation_x0": tier, "G0.4_tier": tier >= TIER_REAL[1],
            "pass": bool(stop_ok and tier >= TIER_REAL[1])}


def summarize(domains):
    recs = load(domains)
    sig_r, sig_full = sigma_r_crossfit(recs)
    e0 = e0_index()
    groups = defaultdict(list)
    for r in recs:
        if r["seq"] in CM.ZGZ:
            groups[r["seq"]].append(r)
        else:
            who = "fit" if subject_of(r["seq"]) in FIT_SUBJ else "sel"
            lg = "local" if "_local" in r["seq"] else "global"
            groups[f"check_{who}_{lg}"].append(r)
    res = {"domains": {k: domain_summary(v, sig_r, e0) for k, v in sorted(groups.items())}}
    res["sigma_r_recording"] = sig_full
    res["e3"] = {b: {"sigma_r": s, "depth_median": float(np.median([e0[f"{r['seq']}|{r['index']}"]["depth"]
                                                                    for r in recs if base_of(r["seq"]) == b])),
                     } for b, s in sig_full.items()}
    for b, v in res["e3"].items():
        v["z_pred_mm"] = v["depth_median"] * (1.0 / v["sigma_r"] - 1.0) * 1000.0
    if any(k.startswith("check_fit") for k in res["domains"]):
        res["pregate"] = pregate(res["domains"], {k: v for k, v in e0.items() if not k.startswith("zgz")})
    res["verdicts"] = {k: verdict(v) for k, v in res["domains"].items()}
    tag = "_".join(domains)
    (OUT / f"result_{tag}.json").write_text(json.dumps(res, indent=1))
    (OUT / f"sigma_r_{tag}.json").write_text(json.dumps(sig_r))
    CM.receipt(OUT / f"summary_{tag}", {"probe": "SBA r3 E1 summary", "domains": list(domains)},
               scripts=(__file__, L.__file__, PR.__file__))
    print(json.dumps({"verdicts": res["verdicts"], "pregate": res.get("pregate", {}).get("pass")}, indent=1))
    return res


def calib_gate():
    """Training pre-gate -> which zgz domains may be read (outputs/sba/r3/e1/zgz_allowed.json)."""
    res = json.loads((OUT / "result_check.json").read_text())
    pg = res["pregate"]
    allowed = {}
    for lg in ("local", "global"):
        cells = pg["by_domain"][lg]["cells"]
        allowed[lg] = bool(pg["pass"] and cells["verdict"] != "FALSIFIED")
    status = "PREGATE_FAIL" if not pg["pass"] else ("TRAINING_FALSIFIED" if not all(allowed.values()) else "OK")
    out = {"local": allowed["local"], "global": allowed["global"], "status": status}
    (OUT / "zgz_allowed.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize", "calib_gate", "selftest", "freeze"))
    ap.add_argument("--domain", choices=PR.DOMAINS)
    ap.add_argument("--domains", nargs="+", default=list(PR.DOMAINS))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="scratch output dir (engineering smoke only; skips the SHA check)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.domain, a.shard, a.shards, a.limit, Path(a.out) if a.out else OUT)
    elif a.cmd == "summarize":
        summarize(a.domains)
    elif a.cmd == "calib_gate":
        calib_gate()
    elif a.cmd == "freeze":
        freeze()
    else:
        sys.exit(0 if selftest() else 1)


if __name__ == "__main__":
    main()
