#!/usr/bin/env python3
"""OBS-A OA1 (clock discrimination) and OA3 (label-timing floor) of docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026.

OA1, per stream (all 74): the v2e emulator stamps the last common time layer of every interpolation interval exactly at
the interpolated frame time (float32 `linspace(t_prev + d/m, t_frame, m)`), so the source frame times of the clock the
stream was generated on carry event layers. Hit rate h_C = share of interior source frames n (active under clock C:
>= 1 event in (c_{n-1}, c_n]) with a layer within tol(c_n) = 3 us + 2 ulp_float32(c_n) of c_n, for C = H (header) and
U (average frame rate); null h_C^0 = the same grid shifted by +-113 / 227 / 341 / 457 us. Verdict H / U / undecided by the
pre-registered rule. The U clock is a family c^U(p) = n D / p over v2e's possible last interpolation positions p
(`common.pmax_candidates`): h_U is the best over p, and the U null takes the same best-over-p per shift. Validity: (i)
synthetic layers on either clock (9 per interval, Bernoulli(0.3) intervals dropped, plus as many uniform random layers as
the real stream has; the synthetic U clock uses v2e's formula with a random last-batch factor m in 2..30 unknown to the
judge) must be judged as their clock, (ii) the real layers mapped to the other clock by source-frame position (U with the
selected p, or p = N - 1 for an H stream) must flip the verdict, (iii) the real stream's null rates <= 0.10.

OA3, per base stream (20): GT(t) against GT(t + e(t)) every 5 valid ms, RA-MPJPE and absolute MPJPE (the evaluator's
`evalx.mano_fk`), mean and p95, and the |e| distribution.

    python tools/obs/oa_clock.py clock --out outputs/obs/oa1 [--workers 16]
    python tools/obs/oa_clock.py floor --out outputs/obs/oa3 [--workers 10]
    python tools/obs/oa_clock.py smoke
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402

SHIFTS_US = (113.0, -113.0, 227.0, -227.0, 341.0, -341.0, 457.0, -457.0)
H_MIN, RATIO_MIN, NULL_MAX = 0.5, 10.0, 0.10
SYN_SUB, SYN_DROP = 9, 0.30
FLOOR_EVERY_MS = 5


# ----------------------------------------------------------------------------------------------- OA1
def hit_rates(L: np.ndarray, c: np.ndarray) -> dict:
    """Hit rates of the interior frame times c[1:-1] against the sorted layer times L (all frames, active frames)."""
    inner = c[1:-1]
    hit = C.nearest_dist(L, inner) <= C.tol_us(inner)
    lo = np.searchsorted(L, c[:-2], side="right")
    hi = np.searchsorted(L, c[1:-1], side="right")
    active = hi > lo
    return {"h_active": float(hit[active].mean()) if active.any() else float("nan"), "h_all": float(hit.mean()),
            "n_active": int(active.sum()), "n_interior": int(len(inner))}


def null_rate(L: np.ndarray, c: np.ndarray) -> float:
    return float(np.mean([hit_rates(L, c + d)["h_active"] for d in SHIFTS_US]))


def u_family(c_h: np.ndarray):
    return [(p, C.clock_u(c_h, p)) for p in C.pmax_candidates(len(c_h))]


def best_u(L: np.ndarray, fam) -> tuple:
    """(rates, pmax, clock) of the U-family member with the highest active hit rate."""
    best = None
    for p, c in fam:
        r = hit_rates(L, c)
        if best is None or r["h_active"] > best[0]["h_active"]:
            best = (r, p, c)
    return best


def null_u(L: np.ndarray, fam) -> float:
    """U null with the same best-over-p selection, per shift."""
    return float(np.mean([max(hit_rates(L, c + d)["h_active"] for _, c in fam) for d in SHIFTS_US]))


def judge(L: np.ndarray, c_h: np.ndarray, fam) -> dict:
    rh = hit_rates(L, c_h)
    ru, p_star, _ = best_u(L, fam)
    nh, nu = null_rate(L, c_h), null_u(L, fam)
    hh, hu = rh["h_active"], ru["h_active"]
    if hh >= H_MIN and hh >= RATIO_MIN * max(nh, hu, 1e-12):
        v = "H"
    elif hu >= H_MIN and hu >= RATIO_MIN * max(nu, hh, 1e-12):
        v = "U"
    else:
        v = "undecided"
    return {"H": rh, "U": ru, "U_pmax": p_star, "U_pmax_minus_n1": p_star - (len(c_h) - 1.0),
            "null_H": nh, "null_U": nu, "verdict": v}


def synthetic_layers(c: np.ndarray, n_random: int, rng: np.random.Generator) -> np.ndarray:
    """Layers on the clock c: SYN_SUB per source interval (the last at the next frame time), SYN_DROP of the intervals
    without any, plus n_random uniform random layers over the stream."""
    a, b = c[:-1], c[1:]
    keep = rng.random(len(a)) >= SYN_DROP
    j = np.arange(1, SYN_SUB + 1, dtype=np.float64) / SYN_SUB
    lat = (a[keep, None] + (b[keep] - a[keep])[:, None] * j[None, :]).ravel()
    rnd = rng.uniform(0.0, float(c[-1]), size=int(n_random))
    return np.unique(np.rint(np.concatenate([lat, rnd])).astype(np.int64))


def clock_job(item) -> dict:
    seq, d = item
    t0 = time.time()
    try:
        aux = C.load_aux(seq, d)
        c_h, c_u = C.clocks_us(aux)
        t = C.event_times_us(seq, d)
        L = C.layers_us(t)
        out = {"seq": seq, "kind": C.stream_kind(seq), "n_frames": int(len(c_h)), "n_events": int(len(t)),
               "n_layers": int(len(L)), "duration_s": float(c_h[-1] / 1e6),
               "e_abs_ms": {"median": float(np.median(np.abs(c_h - c_u)) / 1e3),
                            "p95": float(np.percentile(np.abs(c_h - c_u), 95) / 1e3),
                            "max": float(np.abs(c_h - c_u).max() / 1e3)},
               "sorted_events": bool(np.all(t[1:] >= t[:-1])) if len(t) > 1 else True}
        fam = u_family(c_h)
        out["real"] = judge(L, c_h, fam)
        rng = np.random.default_rng(stable_seed(seq))
        sh = synthetic_layers(c_h, len(L), rng)
        m_true = int(rng.integers(2, 31))
        c_u_true = C.clock_u(c_h, len(c_h) - 2.0 + (m_true - 1.0) / m_true)
        su = synthetic_layers(c_u_true, len(L), rng)
        out["syn_H"] = judge(sh, c_h, fam)
        out["syn_U"] = judge(su, c_h, fam)
        out["syn_U_m_true"] = m_true
        v = out["real"]["verdict"]
        if v in ("H", "U"):
            c_u_sel = C.clock_u(c_h, out["real"]["U_pmax"]) if v == "U" else c_u
            mapped = C.h_to_u(L, c_h, c_u_sel) if v == "H" else C.u_to_h(L, c_h, c_u_sel)
            out["mapped"] = judge(np.unique(np.rint(mapped).astype(np.int64)), c_h, fam)
            out["mapped_flips"] = out["mapped"]["verdict"] == ("U" if v == "H" else "H")
        out["validity"] = {"syn_H_ok": out["syn_H"]["verdict"] == "H", "syn_U_ok": out["syn_U"]["verdict"] == "U",
                           "null_ok": max(out["real"]["null_H"], out["real"]["null_U"]) <= NULL_MAX,
                           "mapped_ok": out.get("mapped_flips", None)}
        out["wall_s"] = time.time() - t0
        return out
    except Exception as ex:                                                     # noqa: BLE001
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}


def stable_seed(seq: str) -> int:
    """Per-stream seed of the synthetic controls (crc32: reproducible across processes, unlike hash())."""
    import zlib
    return zlib.crc32(f"obs-oa1:{seq}".encode())


def aggregate_clock(rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    base = [r for r in ok if r["kind"] in ("base_train", "zgz")]
    var = [r for r in ok if r["kind"] == "variant"]
    count = lambda rs: {v: sum(1 for r in rs if r["real"]["verdict"] == v) for v in ("H", "U", "undecided")}  # noqa: E731
    valid = all(r["validity"]["syn_H_ok"] and r["validity"]["syn_U_ok"] and r["validity"]["null_ok"]
                and r["validity"]["mapped_ok"] in (True, None) for r in ok) and len(ok) == len(rows)
    vb = count(base)
    if not valid:
        verdict = "tool_defect"
    elif vb["U"] > 0:
        verdict = "data_defect_U_clock_streams"
    elif vb["H"] == len(base) == 20:
        verdict = "all_base_streams_on_header_clock"
    else:
        verdict = "undecided_streams"
    return {"n_streams": len(rows), "n_ok": len(ok), "errors": [r for r in rows if "error" in r],
            "base_counts": vb, "variant_counts": count(var), "validity_all": bool(valid),
            "U_streams": [r["seq"] for r in ok if r["real"]["verdict"] == "U"],
            "undecided_streams": [r["seq"] for r in ok if r["real"]["verdict"] == "undecided"],
            "verdict": verdict}


# ----------------------------------------------------------------------------------------------- OA3
def floor_job(seq: str) -> dict:
    import torch
    torch.set_num_threads(2)
    try:
        sys.path.insert(0, str(C.REPO / "tools" / "tracking"))
        import evalx as EX
        from mano_layer import ManoLayer
        from config import load_config
        cfg = load_config(C.REPO / "outputs" / "semkine" / "dt_dz_l3_s3407" / "config_resolved.yaml")
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).eval()
        events, offsets, aux, pos51 = C.MC.load_seq(seq)
        c_h, c_u = C.clocks_us(aux)
        valid = np.asarray(aux["valid_ms"], dtype=bool)
        n_ms = len(valid)
        t = np.arange(0, n_ms, FLOOR_EVERY_MS, dtype=np.int64)
        e = C.e_ms_at(t.astype(np.float64), c_h, c_u)
        t2 = np.rint(t + e).astype(np.int64)
        ok = valid[t] & (t2 >= 0) & (t2 < n_ms)
        ok[ok] &= valid[t2[ok]]
        t, t2, e = t[ok], t2[ok], e[ok]
        betas = np.asarray(aux["betas"], np.float32).reshape(-1)
        P1 = np.stack([np.asarray(pos51[int(i)], np.float32) for i in t])
        P2 = np.stack([np.asarray(pos51[int(i)], np.float32) for i in t2])
        dev = torch.device("cpu")
        bt = torch.tensor(betas, dtype=torch.float32).view(1, -1)
        j1, _ = EX.mano_fk(mano, P1, bt, dev)
        j2, _ = EX.mano_fk(mano, P2, bt, dev)
        ra = lambda x: x - x[..., :1, :]                                        # noqa: E731
        d_ra = np.linalg.norm(ra(j1) - ra(j2), axis=-1).mean(-1) * 1000
        d_abs = np.linalg.norm(j1 - j2, axis=-1).mean(-1) * 1000
        return {"seq": seq, "n": int(len(t)),
                "ra_mm": {"mean": float(d_ra.mean()), "p95": float(np.percentile(d_ra, 95))},
                "abs_mm": {"mean": float(d_abs.mean()), "p95": float(np.percentile(d_abs, 95))},
                "e_abs_ms": {"mean": float(np.abs(e).mean()), "p95": float(np.percentile(np.abs(e), 95)),
                             "max": float(np.abs(e).max())}}
    except Exception as ex:                                                     # noqa: BLE001
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}


# ----------------------------------------------------------------------------------------------- main
def smoke():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # synthetic clock pair: header frames with jitter and one dropped frame
    n = 400
    dt = 16734.0 + rng.normal(0, 40, n - 1)
    dt[200] *= 2
    c_h = np.concatenate([[0.0], np.cumsum(dt)])
    c_u = np.arange(n) * (c_h[-1] / (n - 1))
    fam = u_family(c_h)
    c_u_v2e = C.clock_u(c_h, n - 2.0 + (7 - 1.0) / 7)
    for name, c in (("H", c_h), ("U (v2e formula, m = 7)", c_u_v2e)):
        L = synthetic_layers(c, 4000, rng)
        print(f"[smoke] synthetic on {name}: verdict {judge(L, c_h, fam)['verdict']}")
    seqs = C.all_streams()
    print(f"[smoke] {len(seqs)} streams; kinds", {k: sum(1 for s, _ in seqs if C.stream_kind(s) == k)
                                                  for k in ("base_train", "variant", "zgz")})
    s, d = seqs[0]
    aux = C.load_aux(s, d)
    c_h, c_u = C.clocks_us(aux)
    t = C.event_times_us(s, d)
    print(f"[smoke] {s}: frames {len(c_h)}, events {len(t)} (shapes only), {time.time() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("clock", "floor", "smoke"))
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    C.force_cpu()
    if a.stage == "smoke":
        return smoke()
    if not a.out:
        raise SystemExit("--out required")
    C.verify_protocol()
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    t0 = time.time()
    ctx = mp.get_context("spawn")
    if a.stage == "clock":
        items = C.all_streams()
        with ctx.Pool(a.workers) as pool:
            rows = pool.map(clock_job, items, chunksize=1)
        res = {"probe": "OA1", "rows": rows, "decision": aggregate_clock(rows), "wall_s": time.time() - t0}
    else:
        with ctx.Pool(a.workers) as pool:
            rows = pool.map(floor_job, list(C.BASE), chunksize=1)
        res = {"probe": "OA3", "rows": rows, "wall_s": time.time() - t0}
    C.save_json(out / "result.json", res)
    C.receipt(out, {"probe": res["probe"]}, scripts=["tools/obs/oa_clock.py", "tools/obs/common.py"])
    print(f"[{res['probe']}] done in {time.time() - t0:.0f} s -> {out}/result.json")


if __name__ == "__main__":
    main()
