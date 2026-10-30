#!/usr/bin/env python3
"""OBS-A OA1-R2 (redesign of the clock discrimination; docs/EXPERIMENT_RECORDS.md#obs-a1r2-prereg-20261026).

OA1 asked for an event layer exactly at each source frame time. That layer exists only while the pixel with the largest
threshold-crossing count of the last interpolation interval survives the hand-box event filter, so its hit rate varies
from stream to stream for reasons unrelated to the clock; OA1's U-family null was also inflated by selecting p. OA1-R2
uses the phase structure instead. On the clock a stream was generated on, every event layer sits at a rational position
of its source interval (interpolated frame k/U of the batch's upsampling factor U, then the emulator's common sub-steps
j/m), so the layers' phases phi = (tau - c_n) / (c_{n+1} - c_n) pile up on a few hundred values; on the other clock the
offset e(n) (tens of ms, varying) smears them out.

Statistic, per stream and clock c (K = 1000 phase bins; intervals in three folds by n mod 3):
  * D(c) = cross-fitted log-likelihood ratio of the phases against uniform: the fold-1 histogram (add-1 smoothed) scores
    the fold-2 layers and vice versa, mean log(K p(bin)) per layer, averaged (uniform phases give D slightly below 0);
  * U only: p (`common.pmax_candidates`) maximises the same cross-fitted D inside fold 0 (n mod 6 = 0 against 3);
  * null: D of 20 sets of as many uniform random layers on the header clock -> mean mu0, SD sd0; z(c) = (D(c) - mu0) / sd0.
Verdict H: z_H >= 10 and z_U <= z_H / 3; U: z_U >= 10 and z_H <= z_U / 3; else undecided.
Validity: (i) synthetic layers built like v2e's under harsh conditions (batch factor U in 9..30 per 96 intervals; per
interpolated interval m in 1..20 with P(m) ~ 1 / m and layer j kept with probability 0.9 (1 - (j - 1) / m)^2, the last
layer (the frame time) with 0.1; 30 % of the intervals empty; plus uniform random layers half as many as the lattice ones)
on the header clock and on U(p_true) (m_last in 2..30, unknown to the judge) must be judged H and U for every stream;
(ii) the real layers mapped to the other clock (U with the selected p, or p = N - 1 for an H stream) must flip the
verdict; (iii) one more uniform replicate stays within |z| <= 4.

    python tools/obs/oa_phase.py --out outputs/obs/oa1r2 [--workers 24]
    python tools/obs/oa_phase.py --smoke
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402

K_BINS = 1000
Z_MIN, RATIO, NULL_Z_MAX, N_NULL = 10.0, 3.0, 4.0, 20


def phases(L: np.ndarray, c: np.ndarray):
    """(phase, interval index) of the layers inside [c_0, c_{N-1})."""
    L = np.asarray(L, dtype=np.float64)
    sel = (L >= c[0]) & (L < c[-1])
    t = L[sel]
    n = np.searchsorted(c, t, side="right") - 1
    n = np.clip(n, 0, len(c) - 2)
    ph = (t - c[n]) / (c[n + 1] - c[n])
    return np.clip(ph, 0.0, 1.0 - 1e-12), n


def hist(ph: np.ndarray) -> np.ndarray:
    return np.bincount(np.minimum((ph * K_BINS).astype(np.int64), K_BINS - 1), minlength=K_BINS).astype(np.float64)


def llr_folds(ph: np.ndarray, sel_a, sel_b) -> float:
    """Cross-fitted mean log-likelihood ratio (nats per layer) of two phase folds against the uniform density."""
    ha, hb = hist(ph[sel_a]), hist(ph[sel_b])
    pa = (ha + 1.0) / (ha.sum() + K_BINS)
    pb = (hb + 1.0) / (hb.sum() + K_BINS)
    dab = float((hb * np.log(K_BINS * pa)).sum() / max(hb.sum(), 1.0))
    dba = float((ha * np.log(K_BINS * pb)).sum() / max(ha.sum(), 1.0))
    return 0.5 * (dab + dba)


def d_stat(L: np.ndarray, c: np.ndarray) -> float:
    ph, n = phases(L, c)
    return llr_folds(ph, n % 3 == 1, n % 3 == 2)


def d_fold0(L: np.ndarray, c: np.ndarray) -> float:
    ph, n = phases(L, c)
    return llr_folds(ph, n % 6 == 0, n % 6 == 3)


def select_p(L: np.ndarray, c_h: np.ndarray):
    return max(C.pmax_candidates(len(c_h)), key=lambda p: d_fold0(L, C.clock_u(c_h, p)))


def uniform_layers(c: np.ndarray, n: int, rng) -> np.ndarray:
    return np.unique(np.rint(rng.uniform(float(c[0]), float(c[-1]), size=int(n))).astype(np.int64))


def judge(L: np.ndarray, c_h: np.ndarray, rng) -> dict:
    p_sel = select_p(L, c_h)
    c_u = C.clock_u(c_h, p_sel)
    d_h, d_u = d_stat(L, c_h), d_stat(L, c_u)
    nul = [d_stat(uniform_layers(c_h, len(L), rng), c_h) for _ in range(N_NULL)]
    mu0, sd0 = float(np.mean(nul)), float(np.std(nul, ddof=1))
    z_h, z_u = (d_h - mu0) / sd0, (d_u - mu0) / sd0
    z_extra = (d_stat(uniform_layers(c_h, len(L), rng), c_h) - mu0) / sd0
    if z_h >= Z_MIN and z_u <= z_h / RATIO:
        v = "H"
    elif z_u >= Z_MIN and z_h <= z_u / RATIO:
        v = "U"
    else:
        v = "undecided"
    return {"D_H": d_h, "D_U": d_u, "null_mu": mu0, "null_sd": sd0, "z_H": z_h, "z_U": z_u, "z_extra_uniform": z_extra,
            "U_pmax": p_sel, "U_pmax_minus_n1": p_sel - (len(c_h) - 1.0), "verdict": v}


def v2e_like_layers(c: np.ndarray, n_random_frac: float, rng, u_rng=(9, 31), m_max=20, p_last=0.1) -> np.ndarray:
    """Layers placed like v2e's on the clock c under harsh conditions (see the module docstring)."""
    out = []
    n_int = len(c) - 1
    ms = np.arange(1, m_max + 1)
    pm = (1.0 / ms) / (1.0 / ms).sum()
    for b0 in range(0, n_int, 96):
        U = int(rng.integers(*u_rng))
        for n in range(b0, min(b0 + 96, n_int)):
            if rng.random() < 0.30:
                continue
            a, d = c[n], (c[n + 1] - c[n]) / U
            m_k = rng.choice(ms, size=U, p=pm)
            for k in range(U):
                m = int(m_k[k])
                j = np.arange(1, m + 1)
                keep_p = 0.9 * (1.0 - (j - 1) / m) ** 2
                keep_p[-1] = p_last
                keep = rng.random(m) < keep_p
                out.append((a + k * d + d * j / m)[keep])
    lat = np.concatenate(out) if out else np.zeros(0)
    rnd = rng.uniform(float(c[0]), float(c[-1]), size=int(len(lat) * n_random_frac))
    return np.unique(np.rint(np.concatenate([lat, rnd])).astype(np.int64))


def stream_job(item) -> dict:
    seq, d = item
    t0 = time.time()
    try:
        rng = np.random.default_rng(zlib.crc32(f"obs-oa1r2:{seq}".encode()))
        aux = C.load_aux(seq, d)
        c_h, c_u1 = C.clocks_us(aux)
        L = C.layers_us(C.event_times_us(seq, d))
        out = {"seq": seq, "kind": C.stream_kind(seq), "n_layers": int(len(L)), "n_frames": int(len(c_h))}
        out["real"] = judge(L, c_h, rng)
        syn_h = v2e_like_layers(c_h, 0.5, rng)
        m_true = int(rng.integers(2, 31))
        c_u_true = C.clock_u(c_h, len(c_h) - 2.0 + (m_true - 1.0) / m_true)
        syn_u = v2e_like_layers(c_u_true, 0.5, rng)
        out["syn_H"] = judge(syn_h, c_h, rng)
        out["syn_U"] = judge(syn_u, c_h, rng)
        out["syn_U_m_true"] = m_true
        v = out["real"]["verdict"]
        if v in ("H", "U"):
            c_sel = C.clock_u(c_h, out["real"]["U_pmax"]) if v == "U" else c_u1
            mapped = C.h_to_u(L, c_h, c_sel) if v == "H" else C.u_to_h(L, c_h, c_sel)
            out["mapped"] = judge(np.unique(np.rint(mapped).astype(np.int64)), c_h, rng)
            out["mapped_flips"] = out["mapped"]["verdict"] == ("U" if v == "H" else "H")
        out["validity"] = {"syn_H_ok": out["syn_H"]["verdict"] == "H", "syn_U_ok": out["syn_U"]["verdict"] == "U",
                           "null_ok": abs(out["real"]["z_extra_uniform"]) <= NULL_Z_MAX,
                           "mapped_ok": out.get("mapped_flips", None)}
        out["wall_s"] = time.time() - t0
        return out
    except Exception as ex:                                                     # noqa: BLE001
        import traceback
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()}


def aggregate(rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    base = [r for r in ok if r["kind"] in ("base_train", "zgz")]
    var = [r for r in ok if r["kind"] == "variant"]
    count = lambda rs: {v: sum(1 for r in rs if r["real"]["verdict"] == v) for v in ("H", "U", "undecided")}  # noqa: E731
    valid = len(ok) == len(rows) and all(r["validity"]["syn_H_ok"] and r["validity"]["syn_U_ok"]
                                         and r["validity"]["null_ok"] and r["validity"]["mapped_ok"] in (True, None)
                                         for r in ok)
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
            "validity_failures": {k: [r["seq"] for r in ok if r["validity"][k] is False]
                                  for k in ("syn_H_ok", "syn_U_ok", "null_ok", "mapped_ok")},
            "U_streams": [r["seq"] for r in ok if r["real"]["verdict"] == "U"],
            "undecided_streams": [r["seq"] for r in ok if r["real"]["verdict"] == "undecided"],
            "verdict": verdict}


def smoke():
    t0 = time.time()
    rng = np.random.default_rng(0)
    n = 1200
    dt = 16760.0 + rng.normal(0, 60, n - 1) + 120 * np.sin(np.arange(n - 1) / 300.0)
    c_h = np.concatenate([[0.0], np.cumsum(dt)])
    c_u7 = C.clock_u(c_h, n - 2.0 + 6.0 / 7.0)
    for name, c in (("H", c_h), ("U (v2e formula, m = 7)", c_u7)):
        j = judge(v2e_like_layers(c, 0.5, rng), c_h, rng)
        print(f"[smoke] synthetic harsh v2e-like on {name}: {j['verdict']} (z_H {j['z_H']:.1f}, z_U {j['z_U']:.1f})")
    j = judge(uniform_layers(c_h, 40000, rng), c_h, rng)
    print(f"[smoke] uniform random layers: {j['verdict']} (z_H {j['z_H']:.1f}, z_U {j['z_U']:.1f}); {time.time() - t0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    C.force_cpu()
    if a.smoke:
        return smoke()
    if not a.out:
        raise SystemExit("--out required")
    C.verify_extra("obs-a1r2-prereg-20261026")
    out = Path(a.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(stream_job, C.all_streams(), chunksize=1)
    res = {"probe": "OA1-R2", "rows": rows, "decision": aggregate(rows), "wall_s": time.time() - t0}
    C.save_json(out / "result.json", res)
    C.receipt(out, {"probe": "OA1-R2"}, scripts=["tools/obs/oa_phase.py", "tools/obs/common.py"])
    print(f"[OA1-R2] done in {time.time() - t0:.0f} s -> {out}/result.json ({res['decision']['verdict']})")


if __name__ == "__main__":
    main()
