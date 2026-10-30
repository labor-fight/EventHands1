#!/usr/bin/env python3
"""SBA round 3, E2 / E8 chained-window accumulation (docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022).

For a terminal packet t (informative, with three predecessors in the same run), knots at the ends of bank rows
t-3 .. t (every 50 ms) and windows (k_j, k_{j+1}], j = 0..3, each event in exactly one window. K = 1 / 2 / 4 windows
ending at t (50 / 100 / 200 ms) are scored jointly: event cells concatenated, nuisances (m_g, eta) shared, sigma shared
on a 5-point grid around the recording-level sigma_r of E1 (R arm). Hypotheses (E0 specs of packet t):

    persistent   clean, pip +-5, mcpf +-5 (F5), synrc +-5, root +-5 -- the same offset at every knot
    end only     mcpf +-5 at the last knot (mcpfE): earlier windows identical to clean -> structural null

Conditions: natural (equal time) K = 1, 2, 4; equal unique events K = 2, 4 (uniform Bernoulli thinning of the K-window
events to the terminal window's in-crop count, 5 seeded draws); time shuffle K = 4 (timestamps permuted among the
200 ms events); donor chain K = 1, 4 (events of a far chain, re-timed window by window, count-matched).

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python tools/sba/r3_e8.py run --domain check --shard 0 --shards 6
    python tools/sba/r3_e8.py summarize
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
from sba import r3_e1 as E1  # noqa: E402

OUT = E1.R3 / "e8"
PERSIST = ["clean", "pip+5", "pip-5", "mcpf+5", "mcpf-5", "synrc+5", "synrc-5", "root+5", "root-5"]
ENDONLY = {"mcpfE+5": "mcpf+5", "mcpfE-5": "mcpf-5"}
HYP8 = PERSIST + list(ENDONLY)
DONOR8 = ["clean", "pip+5", "pip-5", "mcpf+5", "mcpf-5", "synrc+5", "synrc-5"]
OFFS = (-0.04, -0.02, 0.0, 0.02, 0.04)
N_THIN = 5
ZGZ_STRIDE = 2
F5 = ["pip+5", "pip-5", "mcpf+5", "mcpf-5"]
FAM8 = {"F5": F5, "synrc5": ["synrc+5", "synrc-5"], "root5": ["root+5", "root-5"], "endonly": list(ENDONLY)}
TIER_MM = 5.0
# frozen E8 gate
G_PASS, R_PASS, P4_PASS = 0.06, 2.0, 0.70
G_FALS, G_FALS_HI = 0.02, 0.04
ATTR_EVENTS, ATTR_SHUFFLE, END_NULL, END_FLAG = 0.5, 0.80, 0.03, 0.05


def frozen_constants() -> dict:
    return {"HYP8": HYP8, "DONOR8": DONOR8, "OFFS": OFFS, "N_THIN": N_THIN, "ZGZ_STRIDE": ZGZ_STRIDE,
            "G_PASS": G_PASS, "R_PASS": R_PASS, "P4_PASS": P4_PASS, "G_FALS": G_FALS, "G_FALS_HI": G_FALS_HI,
            "ATTR_EVENTS": ATTR_EVENTS, "ATTR_SHUFFLE": ATTR_SHUFFLE, "END_NULL": END_NULL, "END_FLAG": END_FLAG}


def chain_rows(rows, index: int):
    """Bank rows index-3 .. index if they form one contiguous run, else None."""
    if index < 3:
        return None
    rs = [rows[index - k] for k in (3, 2, 1, 0)]
    if any(r["index"] != index - k for r, k in zip(rs, (3, 2, 1, 0))):
        return None
    if len({r["run_id"] for r in rs}) != 1:
        return None
    if any(int(rs[j + 1]["previous_end_ms"]) != int(rs[j]["end_ms"]) for j in range(3)):
        return None
    return rs


def chain_items(domain):
    """Terminal packets with a complete 4-chain (zgz: every second packet of a run)."""
    out = []
    for seq, d, index, row, rows in packets(domain):
        rs = chain_rows(rows, index)
        if rs is None:
            continue
        if domain in CM.ZGZ:
            first = min(r["index"] for r in rows if r["run_id"] == row["run_id"])
            if (index - first) % ZGZ_STRIDE != 1:
                continue
        out.append((seq, d, index, row, rows, rs))
    return out


def sigma_r_all():
    out = {}
    for f in sorted(E1.OUT.glob("sigma_r_*.json")):
        out.update(json.loads(f.read_text()))
    return out


def e0_specs(domain):
    recs = json.loads((E1.E0 / f"probes_{domain}.json").read_text())["records"]
    return {f"{r['seq']}|{r['index']}": {h: r["hyps"][h]["spec"] for h in r["hyps"]} for r in recs}


def knot_states(cpu_ctx, fr, sd, kms, spec_t):
    """(H8, 5, 51) knot states of every E8 hypothesis at the knot ms `kms`; perturbations on the CPU context, as E0
    computed them, then moved to the scoring device."""
    Xg = torch.stack([sd.gt(m) for m in kms]).cpu()
    betas = sd.betas.cpu()
    out = []
    for h in HYP8:
        if h == "clean":
            out.append(Xg.clone())
        elif h in ENDONLY:
            X = Xg.clone()
            X[-1] = PR.apply(cpu_ctx, fr, PR.Spec(**spec_t[ENDONLY[h]]), Xg[-1:], betas)[0]
            out.append(X)
        else:
            out.append(PR.apply(cpu_ctx, fr, PR.Spec(**spec_t[h]), Xg, betas))
    return torch.stack(out).to(sd.betas.device)


def windows_from(ctx, sd, kms, Xg, ev_ms):
    """Windows (k_j, k_{j+1}] built from GT knots (crop from the GT trajectory) with the given events."""
    return [CM.window(ctx, sd, kms[j], kms[j + 1], Xg[j], Xg[j + 1], events=L.take(ev_ms, kms[j], kms[j + 1]))
            for j in range(4)]


def find_donor_chain(ctx, sd, rows, index, kms, Xg, T, betas, max_try=60):
    """A far chain (terminal GT >= 40 mm / 15 deg away), its 4 windows re-timed onto ours; terminal in-crop count
    >= max(100, T / 2), all windows thinned by the same factor if it exceeds 2 T."""
    n = len(rows)
    wt = ctx.wrist(Xg[-1], betas)[0]
    tried = 0
    for k in range(n):
        di = (index + n // 2 + k) % n
        drs = chain_rows(rows, di)
        if drs is None:
            continue
        g = sd.gt(int(drs[-1]["end_ms"]))
        far = float((ctx.wrist(g, betas)[0] - wt).norm()) * 1000 >= 40.0 or CM.geo_deg(Xg[-1][3:6], g[3:6]) >= 15.0
        if not far:
            continue
        tried += 1
        if tried > max_try:
            break
        evs, mss = [], []
        for j in range(4):
            e, m = CM.retime(sd.events(int(drs[j]["previous_end_ms"]), int(drs[j]["end_ms"])),
                             int(drs[j]["previous_end_ms"]), kms[j], kms[j + 1])
            evs.append(e)
            mss.append(m)
        ev_ms = (np.concatenate(evs), np.concatenate(mss))
        wins = windows_from(ctx, sd, kms, Xg, ev_ms)
        if wins[-1].n_crop < max(CM.MIN_EVENTS, 0.5 * T):
            continue
        if wins[-1].n_crop > 2 * T:
            ev_ms = L.thin(ev_ms, T / wins[-1].n_crop, L.seed_of(sd.name, index, "donor8"))
            wins = windows_from(ctx, sd, kms, Xg, ev_ms)
        return wins, {"index": int(drs[-1]["index"]), "n_crop_terminal": wins[-1].n_crop}
    return None, None


def score_chain(ctx, cpu_ctx, lik, fr, sd, rows, index, rs, spec_t, sig_r):
    kms = [int(rs[0]["previous_end_ms"])] + [int(r["end_ms"]) for r in rs]
    Xg = torch.stack([sd.gt(m) for m in kms])
    ev_all = sd.events(kms[0], kms[-1])
    nat = windows_from(ctx, sd, kms, Xg, ev_all)
    rec = {"seq": sd.name, "index": index, "end": kms[-1], "run_id": int(rs[-1]["run_id"]), "kms": kms,
           "n_events": [w.n_events for w in nat], "n_crop": [w.n_crop for w in nat]}
    if nat[-1].n_events < CM.MIN_EVENTS:
        return rec
    T = nat[-1].n_crop
    XS = knot_states(cpu_ctx, fr, sd, kms, spec_t)                             # (H8, 5, 51)
    sig = [round(sig_r + o, 4) for o in OFFS]
    conds = {"nat": nat}
    for K in (2, 4):
        lo = kms[4 - K]
        sub = L.take(ev_all, lo, kms[-1])
        n_k = sum(w.n_crop for w in nat[4 - K:])
        for dr in range(N_THIN):
            conds[f"thin{K}_{dr}"] = windows_from(ctx, sd, kms, Xg, L.thin(sub, T / max(n_k, 1),
                                                                         L.seed_of(sd.name, index, K, dr)))
    conds["shuf4"] = windows_from(ctx, sd, kms, Xg, L.shuffle_times(ev_all, L.seed_of(sd.name, index, "shuf")))
    dwins, dinfo = find_donor_chain(ctx, sd, rows, index, kms, Xg, T, sd.betas)
    rec["donor"] = dinfo
    di = [HYP8.index(h) for h in DONOR8]
    ll = defaultdict(lambda: np.full((len(HYP8), len(sig)), np.nan))
    dl = {1: np.full((len(DONOR8), len(sig)), np.nan), 4: np.full((len(DONOR8), len(sig)), np.nan)}
    for si, sg in enumerate(sig):
        G = []                                                                  # per window: (H8, 6, M_j)
        for j in range(4):
            Gj = []
            for hi, h in enumerate(HYP8):
                if h in ENDONLY and j < 3:
                    Gj.append(None)
                else:
                    Gj.append(lik.groups(sd, nat[j], XS[hi, j], XS[hi, j + 1], sg)[0])
            clean_j = Gj[0]
            G.append(torch.stack([g if g is not None else clean_j for g in Gj]))
        for name, wins in conds.items():
            Ks = (1, 2, 4) if name == "nat" else ((int(name[4]),) if name.startswith("thin") else (4,))
            for K in Ks:
                key = f"nat{K}" if name == "nat" else name
                ll[key][:, si] = L.newton(*L.chain_problem(G[4 - K:], wins[4 - K:]),
                                          nan_unconverged=True)[0].cpu().numpy()
        if dwins is not None:
            for K in (1, 4):
                dl[K][:, si] = L.newton(*L.chain_problem([g[di] for g in G[4 - K:]], dwins[4 - K:]),
                                        nan_unconverged=True)[0].cpu().numpy()
        del G
    rec.update({"sig": sig, "ll": {k: v.tolist() for k, v in ll.items()},
                "D": {str(K): v.tolist() for K, v in dl.items()} if dwins is not None else None})
    return rec


def run(domain, shard, shards, limit, out_dir=OUT):
    dev = C.require_gpu(allowed=L.GPUS)
    if out_dir == OUT:
        E1.require_protocol()
    torch.set_grad_enabled(False)
    ctx = CM.Ctx(dev)
    cpu_ctx = CM.Ctx(torch.device("cpu"))
    lik = L.Lik(ctx)
    specs = e0_specs(domain)
    sig_r = sigma_r_all()
    items = chain_items(domain)[shard::shards]
    if limit:
        items = items[:limit]
    cache, frames, recs = {}, {}, []
    t0 = time.time()
    for seq, d, index, row, rows, rs in items:
        if seq not in cache:
            cache = {seq: CM.SeqData(seq, d, dev)}
            frames = {seq: PR.make_frame(cpu_ctx, cache[seq].betas.cpu())}
        key = f"{seq}|{index}"
        if key not in sig_r and out_dir != OUT:
            sig_r[key] = 1.0                                                   # engineering smoke only
        if key not in sig_r:
            recs.append({"seq": seq, "index": index, "skipped": "no sigma_r (terminal packet not informative in E1)"})
            continue
        recs.append(score_chain(ctx, cpu_ctx, lik, frames[seq], cache[seq], rows, index, rs, specs[key], sig_r[key]))
        if len(recs) % 25 == 0:
            print(json.dumps({"domain": domain, "shard": shard, "done": len(recs), "of": len(items),
                              "s": round(time.time() - t0)}), flush=True)
    path = out_dir / "raw" / f"{domain}__{shard:02d}of{shards:02d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hyp": HYP8, "donor_h": DONOR8, "records": recs, "wall_s": time.time() - t0}))
    CM.receipt(out_dir / "raw" / f"{domain}__{shard:02d}", {"probe": "SBA r3 E8", "domain": domain, "shard": shard},
               scripts=(__file__, L.__file__, PR.__file__, E1.__file__, CM.REPO / "semkine" / "sba_emission.py"))


# ------------------------------------------------------------------------------------------------ summarize
def chain_scores(r, cond: str, arm: str, donor_K=None):
    names = DONOR8 if donor_K else HYP8
    M = np.asarray(r["D"][str(donor_K)] if donor_K else r["ll"][cond], np.float64)
    grid = r["sig"]
    out = {}
    for i, h in enumerate(names):
        y = M[i]
        if not np.all(np.isfinite(y)):
            out[h] = np.nan
        elif arm == "P":
            out[h] = L.refine_max(y, grid)[0]
        else:
            out[h] = float(y[len(grid) // 2])                                  # R: sigma_r (grid centre)
    return out


def info(sc, fam) -> float:
    """Local information along a family: (dLL_+ + dLL_-) / d^2 averaged over its +- pairs (nats / mm^2)."""
    pairs = [(fam[i], fam[i + 1]) for i in range(0, len(fam), 2)]
    v = [((sc["clean"] - sc[a]) + (sc["clean"] - sc[b])) / TIER_MM ** 2 for a, b in pairs]
    return float(np.nanmean(v))


def cond_pref(r, cond, fam, arm):
    if cond.startswith("thin"):
        return float(np.mean([E1.pref_of(chain_scores(r, f"{cond}_{d}", arm), fam) for d in range(N_THIN)]))
    return E1.pref_of(chain_scores(r, cond, arm), fam)


def domain_summary(rs):
    inf = [r for r in rs if "ll" in r]
    blocks = [E1.block_of(r) for r in inf]
    d = {"n_chains": len(rs), "n_informative": len(inf)}
    conds = ("nat1", "nat2", "nat4", "thin2", "thin4", "shuf4")
    for arm in ("P", "R"):
        t = {}
        for fam_name, fam in FAM8.items():
            per = {c: [cond_pref(r, c, fam, arm) for r in inf] for c in conds}
            t[fam_name] = {c: dict(zip(("pref", "lo", "hi"), E1.boot(v, blocks))) for c, v in per.items()}
            for c in ("nat2", "nat4", "thin2", "thin4", "shuf4"):
                diff = np.asarray(per[c]) - np.asarray(per["nat1"])
                t[fam_name][f"g_{c}"] = dict(zip(("pref", "lo", "hi"), E1.boot(diff, blocks)))
            if fam_name != "endonly":
                I = {c: [info(chain_scores(r, c, arm), fam) for r in inf] for c in ("nat1", "nat2", "nat4")}
                t[fam_name]["info_median"] = {c: float(np.median(v)) for c, v in I.items()}
                t[fam_name]["r200"] = float(np.median(I["nat4"]) / np.median(I["nat1"])) \
                    if np.median(I["nat1"]) > 0 else float("nan")
        d[arm] = t
    dr = [r for r in inf if r.get("D")]
    d["donor_n"] = len(dr)
    d["donor_P"] = {f"K{K}": {fam_name: float(np.mean([E1.pref_of(chain_scores(r, None, "P", donor_K=K), fam)
                                                       for r in dr])) if dr else float("nan")
                              for fam_name, fam in (("F5", F5), ("synrc5", ["synrc+5", "synrc-5"]))}
                    for K in (1, 4)}
    return d


def e8_verdict(d):
    t = d["P"]["F5"]
    g = t["g_nat4"]
    r200 = t["r200"]
    p4 = t["nat4"]["pref"]
    don = d["donor_P"]["K4"]["F5"]
    if g["pref"] >= G_PASS and r200 >= R_PASS and p4 >= P4_PASS and E1.DONOR_BAND[0] <= don <= E1.DONOR_BAND[1]:
        v = "PASS"
    elif g["pref"] <= G_FALS and g["hi"] <= G_FALS_HI:
        v = "FALSIFIED"
    else:
        v = "PARTIAL"
    gN, gS = t["g_thin4"]["pref"], t["g_shuf4"]["pref"]
    ge = d["P"]["endonly"]["g_nat4"]["pref"]
    attr = {"events_share": gN / g["pref"] if g["pref"] > 0 else float("nan"),
            "sweep_diversity": bool(g["pref"] > 0 and gN >= ATTR_EVENTS * g["pref"]),
            "shuffle_retained": gS / g["pref"] if g["pref"] > 0 else float("nan"),
            "not_temporal_order": bool(g["pref"] > 0 and gS >= ATTR_SHUFFLE * g["pref"]),
            "endonly_gain": ge, "endonly_null_ok": abs(ge) <= END_NULL, "endonly_flag": ge >= END_FLAG}
    return {"verdict": v, "g200": g, "r200": r200, "pref200": p4, "donor_K4": don, "attribution": attr}


def summarize():
    recs = []
    for f in sorted((OUT / "raw").glob("*of*.json")):
        recs += json.loads(f.read_text())["records"]
    groups = defaultdict(list)
    for r in recs:
        if r["seq"] in CM.ZGZ:
            groups[r["seq"]].append(r)
        else:
            who = "fit" if E1.subject_of(r["seq"]) in E1.FIT_SUBJ else "sel"
            groups[f"check_{who}_{'local' if '_local' in r['seq'] else 'global'}"].append(r)
    res = {"domains": {k: domain_summary(v) for k, v in sorted(groups.items())}}
    res["verdicts"] = {k: e8_verdict(v) for k, v in res["domains"].items() if v["n_informative"]}
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    CM.receipt(OUT / "summary", {"probe": "SBA r3 E8 summary"}, scripts=(__file__, E1.__file__, L.__file__))
    print(json.dumps({k: {"verdict": v["verdict"], "g200": round(v["g200"]["pref"], 3), "r200": round(v["r200"], 2),
                          "pref200": round(v["pref200"], 3)} for k, v in res["verdicts"].items()}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize", "count"))
    ap.add_argument("--domain", choices=PR.DOMAINS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="scratch output dir (engineering smoke only; skips the SHA check)")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.domain, a.shard, a.shards, a.limit, Path(a.out) if a.out else OUT)
    elif a.cmd == "count":
        print({dm: len(chain_items(dm)) for dm in PR.DOMAINS})
    else:
        summarize()


if __name__ == "__main__":
    main()
