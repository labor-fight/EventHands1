#!/usr/bin/env python3
"""KA step 1, synthetic part (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028).

Contexts: 24 base hand states (finger articulation, root orientation, betas, K) drawn with a fixed seed from the GT of
the 18 training base recordings. Motion classes: 5 fingers x {flex, extend}; the finger's three joints rotate about the
finger's flexion axis (template frame, fixed in the parent frame) in the ratio 1 : 1 : 0.67, amplitude found by
bisection so that the fingertip moves exactly 8 mm, linearly over 50 ms; root static. Camera nuisance settings (applied
about the hand's vertex centroid; hand centred with a small per-(context, setting) image offset <= 10 px):

  0 ref (z 0.55)   1-2 depth z 0.42 / 0.70   3-4 roll +-30 deg   5-6 yaw +-35 deg   7-8 pitch +-25 deg
  9 combined (z 0.70, yaw +30, roll -25)     10 combined (z 0.42, pitch -20, roll +25)

Rendering: textured MANO (tools/s43_br/geometry.py rasterizer, texture seed 7, background 0.15, 2x supersampling) on
the 240 x 180 event grid every 2 ms; events from log-intensity crossings with per-pixel thresholds 0.15 (1 + 0.03 N(0,1))
(realisation seed), reference reset by whole multiples. The A/A control re-renders the reference setting with a second
threshold realisation. Representations get the exact previous state (t = 0) and, descriptively, a perturbed one.

    python tools/kin_atlas/synth.py run --out outputs/kin_atlas/synth [--workers 48]
    python tools/kin_atlas/synth.py analyze --out outputs/kin_atlas/synth
    python tools/kin_atlas/synth.py smoke
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.kin_atlas import common as C  # noqa: E402

N_CONTEXTS = 24
TIP_MOTION_M = 0.008
FRAME_MS = 2
WINDOW_MS = 50
THRESH, THRESH_SD = 0.15, 0.03
Z_REF = 0.55
SETTINGS = [  # (name, family, z, yaw_deg, pitch_deg, roll_deg)
    ("ref", "ref", 0.55, 0, 0, 0),
    ("depth_near", "depth", 0.42, 0, 0, 0), ("depth_far", "depth", 0.70, 0, 0, 0),
    ("roll_p", "roll", 0.55, 0, 0, 30), ("roll_n", "roll", 0.55, 0, 0, -30),
    ("yaw_p", "oop", 0.55, 35, 0, 0), ("yaw_n", "oop", 0.55, -35, 0, 0),
    ("pitch_p", "oop", 0.55, 0, 25, 0), ("pitch_n", "oop", 0.55, 0, -25, 0),
    ("comb1", "combined", 0.70, 30, 0, -25), ("comb2", "combined", 0.42, 0, -20, 25),
]
MOTIONS = [(f, s) for f in C.FINGERS for s in (+1, -1)]
SEED = 20261028


def _seed(*parts) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode())


def _rot(yaw, pitch, roll):
    from scipy.spatial.transform import Rotation as R
    return (R.from_rotvec([0, 0, np.radians(roll)]) * R.from_rotvec([np.radians(pitch), 0, 0]) *
            R.from_rotvec([0, np.radians(yaw), 0]))


def place(state, betas, setting, offset_px):
    """The base state rotated about its vertex centroid by the setting's camera rotation and centred at depth z with an
    image offset (pixels on the event grid, converted with f_ev = 226 px / m at that depth)."""
    from scipy.spatial.transform import Rotation as R
    name, fam, z, yaw, pitch, roll = setting
    s = np.asarray(state, np.float64).copy()
    s[3:6] = (_rot(yaw, pitch, roll) * R.from_rotvec(s[3:6])).as_rotvec()
    s[:3] = 0.0
    v, _ = C.fk(s[None], betas)
    c0 = v[0].mean(0)
    target = np.array([offset_px[0] * z / 226.0, offset_px[1] * z / 226.0, z])
    s[:3] = target - c0
    return s


def flex(state, finger, theta):
    """Rotate the finger's three joints by theta * (1, 1, 0.67) about its flexion axis, fixed in the parent frame."""
    from scipy.spatial.transform import Rotation as R
    s = np.asarray(state, np.float64).copy()
    mean = C.mano_npz()["hands_mean"].reshape(15, 3).astype(np.float64)
    ax = C.flex_axes()[finger]
    for j, r in zip(C.FINGER_JOINTS[finger], (1.0, 1.0, 0.67)):
        k = j - 1
        full = R.from_rotvec(s[6 + 3 * k: 9 + 3 * k] + mean[k])
        s[6 + 3 * k: 9 + 3 * k] = (R.from_rotvec(ax * theta * r) * full).as_rotvec() - mean[k]
    return s


def amplitude(state, betas, finger, sign):
    """theta (rad, signed) such that the fingertip moves TIP_MOTION_M (bisection, 30 steps on [0, 80 deg])."""
    tip = C.OP_TIPS[C.FINGERS.index(finger)]
    _, j0 = C.fk(state[None], betas)
    lo, hi = 0.0, np.radians(80)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        _, j1 = C.fk(flex(state, finger, sign * mid)[None], betas)
        if np.linalg.norm(j1[0, tip] - j0[0, tip]) < TIP_MOTION_M:
            lo = mid
        else:
            hi = mid
    return sign * 0.5 * (lo + hi)


def render_events(states, betas, K_cam, realisation_seed):
    """Event counts (x, y, p) of the trajectory `states` (frames every FRAME_MS)."""
    from tools.s43_br import geometry as G
    m = G._mano()
    faces = m.f.cpu().numpy()
    alb = G._vertex_albedo(m, 7, 0.7)
    K = C.event_K(K_cam)
    verts, _ = C.fk(states, betas)
    L = [np.log(G.rasterize(v, faces, K, C.EV_HW, alb, 0.15, supersample=2)) for v in verts]
    rng = np.random.default_rng(realisation_seed)
    thr = np.clip(THRESH * (1 + THRESH_SD * rng.normal(size=C.EV_HW)), 0.05, None)
    ref = L[0].copy()
    xs, ys, ps = [], [], []
    for Lk in L[1:]:
        d = Lk - ref
        n_on = np.floor(np.clip(d, 0, None) / thr).astype(np.int64)
        n_off = np.floor(np.clip(-d, 0, None) / thr).astype(np.int64)
        ref += (n_on - n_off) * thr
        for n, pol in ((n_on, 1), (n_off, 0)):
            yy, xx = np.nonzero(n)
            if len(xx):
                rep = n[yy, xx]
                xs.append(np.repeat(xx, rep))
                ys.append(np.repeat(yy, rep))
                ps.append(np.full(int(rep.sum()), pol))
    if not xs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def contexts():
    """24 base states (seeded) from the 18 training base recordings: (recording, ms, state, betas, K). A context is
    rejected (next draw) when the hand leaves the image margin (10 px) in any setting."""
    seqs = list(C.base_seqs())
    rng = np.random.default_rng(SEED)
    out, tries = [], 0
    cache = {}
    while len(out) < N_CONTEXTS and tries < 2000:
        tries += 1
        seq = seqs[rng.integers(len(seqs))]
        if seq not in cache:
            ev, off, aux, pos = C.load_seq(seq)
            cache[seq] = (aux, pos)
        aux, pos = cache[seq]
        runs = np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)
        a, b = runs[rng.integers(len(runs))]
        if b - a < 10:
            continue
        ms = int(rng.integers(a, b))
        st = np.asarray(pos[ms], np.float64)
        betas = np.asarray(aux["betas"], np.float32).reshape(-1)[:10]
        K = np.asarray(aux["camera_K"], np.float64).reshape(3, 3)
        ok = True
        for si, s in enumerate(SETTINGS):
            off_px = np.random.default_rng(_seed("off", len(out), si)).uniform(-10, 10, 2)
            v, _ = C.fk(place(st, betas, s, off_px)[None], betas)
            uv = C.project(v[0], C.event_K(K))
            if uv.min() < 10 or uv[:, 0].max() > 230 or uv[:, 1].max() > 170:
                ok = False
                break
        if ok:
            out.append((seq, ms, st, betas, K))
    return out


def task(item):
    ci, (seq, ms, st, betas, K), mi = item
    finger, sign = MOTIONS[mi]
    t0 = time.time()
    try:
        base_ref = place(st, betas, SETTINGS[0], (0, 0))
        theta = amplitude(base_ref, betas, finger, sign)
        rows = []
        for si, setting in enumerate(SETTINGS):
            off_px = np.random.default_rng(_seed("off", ci, si)).uniform(-10, 10, 2)
            s0 = place(st, betas, setting, off_px)
            nf = WINDOW_MS // FRAME_MS + 1
            traj = np.stack([flex(s0, finger, theta * k / (nf - 1)) for k in range(nf)])
            for real in ((1, 2) if si == 0 else (1,)):
                ex, ey, ep = render_events(traj, betas, K, _seed("thr", ci, mi, si, real))
                feats = C.representations(ex, ey, ep, s0, betas, K, blur=True)
                raw = C.representations(ex, ey, ep, s0, betas, K, blur=False)
                pert = C.representations(ex, ey, ep, C.perturb_prev(s0, betas, np.random.default_rng(_seed("pp", ci, mi, si, real))),
                                         betas, K, blur=True)
                rows.append({"context": ci, "motion": mi, "setting": si, "real": real,
                             "feats": {k: v for k, v in feats.items() if not k.startswith("_")},
                             "raw": {k: v for k, v in raw.items() if not k.startswith("_")},
                             "pert": {k: v for k, v in pert.items() if not k.startswith("_")},
                             "n_events": int(feats["_n_events"][0]), "offband_kin": float(feats["_offband_kin"][0]),
                             "offband_rigid": float(feats["_offband_rigid"][0])})
        return {"context": ci, "motion": mi, "theta_deg": float(np.degrees(theta)), "rows": rows, "wall_s": time.time() - t0}
    except Exception as ex:                                                     # noqa: BLE001
        import traceback
        return {"context": ci, "motion": mi, "error": f"{type(ex).__name__}: {ex}", "tb": traceback.format_exc()}


def run(out: Path, workers: int):
    C.verify_protocol()
    if (out / "features.npz").exists():
        raise SystemExit(f"{out}/features.npz exists; preserve it")
    t0 = time.time()
    ctx = contexts()
    items = [(ci, c, mi) for ci, c in enumerate(ctx) for mi in range(len(MOTIONS))]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(task, items, chunksize=1)
    bad = [r for r in res if "error" in r]
    if bad:
        print(bad[0]["tb"])
        raise SystemExit(f"{len(bad)} synthetic tasks failed")
    rows = [row for r in res for row in r["rows"]]
    arrays = {"context": np.array([r["context"] for r in rows]), "motion": np.array([r["motion"] for r in rows]),
              "setting": np.array([r["setting"] for r in rows]), "real": np.array([r["real"] for r in rows]),
              "n_events": np.array([r["n_events"] for r in rows]), "offband_kin": np.array([r["offband_kin"] for r in rows]),
              "offband_rigid": np.array([r["offband_rigid"] for r in rows])}
    for kind in ("feats", "raw", "pert"):
        for rep in C.REPS:
            arrays[f"{kind}|{rep}"] = np.stack([r[kind][rep] for r in rows]).astype(np.float32)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", **arrays)
    C.save_json(out / "run_manifest.json", {
        "contexts": [{"seq": c[0], "ms": c[1]} for c in ctx], "n_rows": len(rows),
        "theta_deg": {f"{r['context']}|{r['motion']}": r["theta_deg"] for r in res}, "wall_s": time.time() - t0})
    print(f"[KA synth] {len(rows)} renders in {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------- analysis
def family_auc(A, feats, family, shuffle_rng=None, aa=False):
    """Per-context mean AUC: positive = same motion, reference vs a setting of `family`; negatives = other motions at
    that same setting. aa=True: reference vs its second realisation. shuffle_rng: motion labels of the comparison set
    permuted within (context, setting)."""
    ctx, mot, st, real = A["context"], A["motion"], A["setting"], A["real"]
    fam_settings = [i for i, s in enumerate(SETTINGS) if s[1] == family] if not aa else [0]
    per_ctx = {}
    for c in np.unique(ctx):
        vals = []
        for si in fam_settings:
            comp_real = 2 if aa else 1
            ref_idx = {int(m): int(np.flatnonzero((ctx == c) & (mot == m) & (st == 0) & (real == 1))[0]) for m in range(len(MOTIONS))}
            cmp_idx = {int(m): int(np.flatnonzero((ctx == c) & (mot == m) & (st == si) & (real == comp_real))[0]) for m in range(len(MOTIONS))}
            labels = list(range(len(MOTIONS)))
            if shuffle_rng is not None:
                perm = shuffle_rng.permutation(len(MOTIONS))
                cmp_idx = {m: cmp_idx[int(perm[m])] for m in labels}
            for m in labels:
                a = feats[ref_idx[m]]
                pos = [float(a @ feats[cmp_idx[m]])]
                neg = [float(a @ feats[cmp_idx[m2]]) for m2 in labels if m2 != m]
                vals.append(C.auc_from(pos, neg))
        per_ctx[int(c)] = float(np.mean(vals))
    return per_ctx


def cross_context_auc(A, feats):
    ctx, mot, st, real = A["context"], A["motion"], A["setting"], A["real"]
    sel = (st == 0) & (real == 1)
    idx = {(int(c), int(m)): int(np.flatnonzero(sel & (ctx == c) & (mot == m))[0]) for c in np.unique(ctx) for m in range(len(MOTIONS))}
    per_ctx = {}
    cs = sorted(np.unique(ctx).tolist())
    for c in cs:
        vals = []
        for c2 in cs:
            if c2 == c:
                continue
            for m in range(len(MOTIONS)):
                a = feats[idx[(c, m)]]
                pos = [float(a @ feats[idx[(c2, m)]])]
                neg = [float(a @ feats[idx[(c2, m2)]]) for m2 in range(len(MOTIONS)) if m2 != m]
                vals.append(C.auc_from(pos, neg))
        per_ctx[c] = float(np.mean(vals))
    return per_ctx


def analyze(out: Path):
    C.verify_protocol()
    if (out / "result.json").exists():
        raise SystemExit(f"{out}/result.json exists; preserve it")
    A = dict(np.load(out / "features.npz"))
    res = {"families": {}, "controls": {}, "descriptive": {}}
    fams = ("depth", "roll", "oop", "combined")
    for kind in ("feats", "raw", "pert"):
        dst = res["families"] if kind == "feats" else res["descriptive"].setdefault(kind, {})
        for rep in C.REPS:
            F = A[f"{kind}|{rep}"]
            d = {f: family_auc(A, F, f) for f in fams}
            d["cross_context"] = cross_context_auc(A, F)
            dst[rep] = d
    for rep in C.REPS:
        F = A[f"feats|{rep}"]
        res["controls"][rep] = {"aa": family_auc(A, F, "ref", aa=True),
                                "shuffled_oop": family_auc(A, F, "oop", shuffle_rng=np.random.default_rng(SEED))}
    res["events"] = {"n_events_median": float(np.median(A["n_events"])), "n_events_p05": float(np.percentile(A["n_events"], 5)),
                     "zero_event_renders": int((A["n_events"] == 0).sum()),
                     "offband_kin_median": float(np.median(A["offband_kin"])),
                     "offband_rigid_median": float(np.median(A["offband_rigid"]))}
    C.save_json(out / "result.json", res)
    print(f"[KA synth] analysed -> {out}/result.json")


def smoke():
    t0 = time.time()
    tab = C.atlas_table()
    print(f"[smoke] atlas cells occupied {len(np.unique(tab))}/512")
    rng = np.random.default_rng(0)
    st = np.zeros(51)
    st[3:6] = [2.2, 0.1, -0.2]
    betas = np.zeros(10, np.float32)
    K = np.array([[603.45, 0, 325.09], [0, 602.96, 242.10], [0, 0, 1.0]])
    s0 = place(st, betas, SETTINGS[5], (3, -2))
    th = amplitude(s0, betas, "index", +1)
    traj = np.stack([flex(s0, "index", th * k / 25) for k in range(26)])
    ex, ey, ep = render_events(traj, betas, K, 1)
    f = C.representations(ex, ey, ep, s0, betas, K)
    print(f"[smoke] one render: theta {np.degrees(th):.1f} deg, {len(ex)} events, dims "
          f"{ {k: v.shape for k, v in f.items() if not k.startswith('_')} }, offband {f['_offband_kin'][0]:.2f}; "
          f"{time.time() - t0:.1f} s (shapes / counts only)")
    _ = rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("run", "analyze", "smoke"))
    ap.add_argument("--out", default=str(C.OUT / "synth"))
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    C.force_cpu()
    if a.stage == "smoke":
        return smoke()
    if a.stage == "run":
        return run(Path(a.out), a.workers)
    return analyze(Path(a.out))


if __name__ == "__main__":
    main()
