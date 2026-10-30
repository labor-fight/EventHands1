#!/usr/bin/env python3
"""K-star KS3 (PC'): the per-vertex measurement quality r* that local fingers need when vertex errors are spatially
correlated (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025).

The XVM P3n oracle ladder (`tools/xvm/ladder.py`: GT contour-normal displacement on the observed contour set + noise
-> 3 Gauss-Newton steps, E3 single step) with the noise correlated across vertices: corr(i, j) = rho(hop(i, j)), where
rho(h) = rho1^h (rho1 in {0, 0.7}) or the hop profile of the frozen patch head's residual measured by S2
(`outputs/kstar/s2_stale/result.json`, hops 1-4, 0 beyond). Two solvers: `ind` treats the noise as independent
(mass-weighted, exactly the ladder's weights; what XVM / PI did) and `gls` uses the true correlation (what C1's
correlated-error model could at best recover). r* = the normal correlation c at which R_art crosses 0.85 (linear
interpolation, the ladder's rule). FIT + SELECT base sequences, every `--stride`-th endpoint, local / global separately.

Validity: V0 rho1 = 0 with the `ind` solver on zgz must reproduce the ledger's P3n r* (local 0.825 +- 0.02);
V1 `gls` at rho1 = 0 equals `ind` (same weights).

    python tools/kstar/corr_rstar.py --out outputs/kstar/s3_corr_rstar
    python tools/kstar/corr_rstar.py --validate-zgz --out outputs/kstar/s3_v0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402
from xvm import common as C  # noqa: E402
from xvm.common import EF  # noqa: E402
from xvm import ladder as LD  # noqa: E402

CORRS = (0.99, 0.97, 0.95, 0.9, 0.8, 0.68, 0.5, 0.42, 0.3, 0.22)
MAX_HOP = 4
NOISE_SEED = 0


def r_star(cs, ra):
    order = np.argsort(cs)
    cs, ra = np.asarray(cs)[order], np.asarray(ra)[order]
    if ra[-1] > 0.85:
        return f">{cs[-1]}"
    if ra[0] <= 0.85:
        return f"<={cs[0]}"
    k = int(np.argmax(ra <= 0.85))
    c0, c1, r0, r1 = cs[k - 1], cs[k], ra[k - 1], ra[k]
    return float(c0 + (0.85 - r0) * (c1 - c0) / (r1 - r0))


def r_star_gain(cs, gain, line=-0.2):
    """Correlation at which the E3 RA gain over identity reaches `line` mm (gain falls as c rises)."""
    order = np.argsort(cs)
    cs, gain = np.asarray(cs)[order], np.asarray(gain)[order]
    if gain[-1] > line:
        return f">{cs[-1]}"
    if gain[0] <= line:
        return f"<={cs[0]}"
    k = int(np.argmax(gain <= line))
    c0, c1, g0, g1 = cs[k - 1], cs[k], gain[k - 1], gain[k]
    return float(c0 + (line - g0) * (c1 - c0) / (g1 - g0))


def corr_matrix(vid, D, prof):
    """Correlation among `vid` from the hop profile prof[h] (h = 0..MAX_HOP, prof[0] = 1), projected to PSD."""
    h = D[vid][:, vid].clamp(max=MAX_HOP + 1)
    table = torch.tensor(list(prof) + [0.0] * (MAX_HOP + 2 - len(prof)), dtype=torch.float64)
    R = table[h]
    w, V = torch.linalg.eigh(R)
    R = (V * w.clamp(min=1e-8)) @ V.T
    d = R.diagonal().sqrt()
    return R / (d[:, None] * d[None])


def gn(ctx, x0, betas, intr, spec, m_obs, sig, Rinv, mode):
    """3 GN steps on the 1-D normal measurements; W = M^1/2 R^-1 M^1/2 / sig^2 (gls) or M / sig^2 (ind)."""
    vid, mass, e1 = spec
    x = x0.clone()
    if len(vid) == 0:
        return x
    v0, dX0, uv0, z0, J20, Jp0 = C.fd_jacobian(ctx.mano, x0, betas, intr)
    s2 = max(float(sig), C.SIGMA_2D) ** 2
    sq_m = mass.sqrt()
    W = (sq_m[:, None] * Rinv * sq_m[None]) / s2 if mode == "gls" else torch.diag(mass) / s2
    for it in range(C.GN_ITERS):
        if it == 0:
            uv, J2 = uv0, J20
        else:
            _, _, uv, _, J2, _ = C.fd_jacobian(ctx.mano, x, betas, intr)
        A = (e1[:, None, :] @ J2[0, vid]).squeeze(1)                        # (n, 51)
        y = (e1 * (uv[0, vid] - uv0[0, vid])).sum(-1)                         # predicted normal displacement
        r = y - m_obs
        H = A.T @ W @ A + C.LAM_P * ctx.P
        g = A.T @ W @ r + C.LAM_P * (ctx.P @ (x[0] - x0[0]))
        x[0] += -torch.linalg.solve(H, g)
    return x


def run(seqs, stride, profiles, solvers, out_tag, dev, quiet=False):
    ctx = LD.Ctx(dev)
    from semkine.mrf import hop_matrix
    D = hop_matrix(ctx.faces.cpu(), 778, MAX_HOP)
    res = {}
    for s, d in seqs:
        sq, rows = LD.seq_arrays(s, d)
        rows = rows[::stride]
        intr = sq.intrinsics()
        betas = LD.t64(sq.betas, dev)[None]
        specs, mtrue, x0s = [], [], []
        for r in rows:
            x0 = LD.t64(sq.pos51[r[2]], dev)[None]
            ge, gv, guv = LD.gt_geometry(ctx, sq, [r[1]], betas, intr)
            v0, dX0, uv0, z0, J20, Jp0 = C.fd_jacobian(ctx.mano, x0, betas, intr)
            vid, mass = C.observed(sq, r[1], ctx.faces)
            _, cos, e1a = C.projected_normals(v0, ctx.faces, Jp0)
            keep = cos[0, vid] < C.CONTOUR_COS
            vid, mass = vid[keep], mass[keep]
            e1 = e1a[0, vid]
            specs.append((vid, mass, e1))
            mtrue.append((e1 * (guv[0, vid] - uv0[0, vid])).sum(-1))
            x0s.append(x0)
        sd = float(torch.cat(mtrue).std())
        X0 = torch.cat(x0s)
        GE = LD.t64(sq.pos51[[r[1] for r in rows]], dev)
        ra_id = C.score_ra(ctx.m32, X0.cpu().numpy(), GE.cpu().numpy(), sq.betas, dev)["mpjpe_ra_mm"]
        ent = {"n_endpoints": len(rows), "true_normal_disp_std_px": sd, "ra_identity": float(ra_id.mean())}
        gains = {}
        for pname, prof in profiles.items():
            Rinvs, Ls = [], []
            for vid, _, _ in specs:
                if len(vid) == 0:
                    Rinvs.append(None); Ls.append(None); continue
                R = corr_matrix(vid.cpu(), D, prof).to(dev)
                Rinvs.append(torch.linalg.inv(R))
                Ls.append(torch.linalg.cholesky(R))
            rng = np.random.default_rng(NOISE_SEED)
            for c in CORRS:
                sig = sd * np.sqrt(1.0 / c ** 2 - 1.0)
                noise = []
                for (vid, _, _), L in zip(specs, Ls):
                    z = torch.as_tensor(rng.standard_normal(len(vid)), dtype=torch.float64, device=dev)
                    noise.append((L @ z) * sig if L is not None else z)
                for mode in solvers:
                    X = torch.cat([gn(ctx, x0, betas, intr, sp, mt + nz, sig, Ri, mode)
                                   for x0, sp, mt, nz, Ri in zip(x0s, specs, mtrue, noise, Rinvs)])
                    ea, ez, _, _ = C.r_art_terms(ctx.mano, X0, X, GE, betas)
                    ent.setdefault(pname, {}).setdefault(mode, {})[str(c)] = float(ea.sum() / ez.sum())
                    ra = C.score_ra(ctx.m32, X.cpu().numpy(), GE.cpu().numpy(), sq.betas, dev)["mpjpe_ra_mm"]
                    gains.setdefault((pname, mode), {})[c] = float(ra.mean() - ra_id.mean())
            for mode in solvers:
                lv = {k: v for k, v in ent[pname][mode].items()}
                ent[pname][mode]["r_star"] = r_star([float(k) for k in lv], list(lv.values()))
                gm = gains[(pname, mode)]
                ent[pname][mode]["ra_gain"] = {str(c): g for c, g in gm.items()}
                ent[pname][mode]["r_star_e3"] = r_star_gain(list(gm), list(gm.values()))
        res[s] = ent
        if not quiet:
            print(out_tag, s, json.dumps({p: {m: [ent[p][m]["r_star"], ent[p][m]["r_star_e3"]] for m in solvers}
                                          for p in profiles}), flush=True)
    return res


def measured_profile():
    p = KC.OUT / "s2_stale" / "result.json"
    if not p.exists():
        return None
    R = json.loads(p.read_text())["readings"]
    out = {}
    for dom in ("local", "global"):
        prof = [1.0] + [max(R.get(f"all|{dom}|hop{h}|residual", {}).get("r", 0.0), 0.0) for h in range(1, MAX_HOP + 1)]
        out[f"measured_{dom}"] = prof
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--validate-zgz", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(8)
    dev = torch.device("cpu")
    out = Path(a.out)
    if not a.smoke:
        KC.verify_protocol()
        if (out / "result.json").exists():
            raise SystemExit(f"{out}/result.json exists; preserve it")
    t0 = time.time()
    if a.validate_zgz:
        res = run(list(C.zgz_seqs()), 1, {"rho0": [1.0]}, ("ind", "gls"), "V0", dev)
        v0 = res["zgz_local"]["rho0"]["ind"]["r_star"]
        v1 = max(abs(res[s]["rho0"]["ind"][str(k)] - res[s]["rho0"]["gls"][str(k)]) for s in res for k in CORRS)
        res["validity"] = {"V0_local_r_star": v0, "V0_ok": isinstance(v0, float) and abs(v0 - 0.825) <= 0.02,
                           "V1_max_abs_R_art_diff": v1, "V1_ok": v1 < 1e-6}
        print("VALIDITY", res["validity"])
    else:
        seqs = [x for x in C.subject_seqs(KC.FIT_SUBJ + KC.SELECT_SUBJ, base_only=True)]
        if a.smoke:
            seqs = seqs[:1]
        profiles = {"rho0": [1.0], "rho0.7": [0.7 ** h for h in range(MAX_HOP + 1)]}
        meas = measured_profile()
        if meas:
            profiles.update(meas)
        res = run(seqs, a.stride if not a.smoke else 200, profiles, ("ind", "gls"), "PC", dev, quiet=a.smoke)
        res["profiles"] = profiles
    res["wall_s"] = time.time() - t0
    if a.smoke:
        print("SMOKE ok", f"{res['wall_s']:.0f} s")
        return
    KC.save_json(out / "result.json", res)
    KC.receipt(out, {"probe": "KS3 correlated r*", "validate_zgz": a.validate_zgz},
               scripts=(__file__, KC.REPO / "tools/xvm/ladder.py"))


if __name__ == "__main__":
    main()
