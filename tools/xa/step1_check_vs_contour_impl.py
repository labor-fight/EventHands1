#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 side check: the opt-in production rule semkine.xyz_mesh.associate_rays(mode="contour") against the
P1 reimplementations `alt` (no-hit fallback = unfiltered near layer) and `altf` (no-hit fallback = back-face-filtered
near layer), sampled events of evenly spaced windows, regimes R3 / R0 / R1s / R1l.  Read-only."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_assoc_regimes as A                                    # noqa: E402
import step1_common as C                                           # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
from semkine.xyz_mesh import associate_rays, stride_sample, vertex_normals as prod_normals   # noqa: E402

n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 16
res = {}
for seq, legacy in C.sequences():
    z = np.load(C.OUT_DIR / f"labels_{seq}.npz")
    lab = {k: z[k] for k in ("ev_x", "ev_y", "ms", "step_ev_start", "samp_local", "samp_start", "step_end_ms")}
    A._init(seq, legacy, "sampled", lab, np.random.default_rng(0).standard_normal((len(lab["step_end_ms"]), 51)).astype(np.float32))
    g = A._G
    steps = np.unique(np.linspace(0, len(g["ends"]) - 1, n_per).astype(int))
    tot = {}
    def bump(k, v):
        tot[k] = tot.get(k, 0) + int(v)
    for s in steps:
        end = int(g["ends"][s])
        ev5 = C._window_events(g["events"], g["offsets"], g["tsub"], end, 50)
        n = len(ev5)
        src, mk = stride_sample(torch.tensor([0, n], dtype=torch.int64), C.CAP)
        samp = src[0][mk[0]].numpy()
        px, py = torch.from_numpy(ev5[:, 1].copy()).float(), torch.from_numpy(ev5[:, 2].copy()).float()
        d_all = torch.stack([(px - g["cx"]) / g["fx"], (py - g["cy"]) / g["fy"], torch.ones_like(px)], -1)
        verts = A._regime_grids(int(s))
        vn = A.vertex_normals(verts, g["faces"])
        ms_rel = np.repeat(np.arange(50), np.diff(g["offsets"][end - 49:end + 2]).astype(np.int64))
        for rg in A.REGIMES:
            dp, mask, vb, nb, band, front, take = A._batch(samp, ms_rel, d_all, rg, verts, vn)
            m = len(samp)
            prod = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN,
                                  chunk=A.CHUNK, mode="contour", normals=prod_normals(vb, g["faces"]), gap=0.01)
            hf = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN, chunk=A.CHUNK)
            cur, alt, altf = A.associate_dual(dp, mask, vb, g["faces"], g["vert_faces"], nb, band, front, C.Z_TOL, C.KNN, 0.01, A.CHUNK)
            P = {k: take(v) for k, v in prod.items()}
            hf = {k: take(v) for k, v in hf.items()}
            cur = {k: take(v) for k, v in cur.items()}
            alt = {k: take(v) for k, v in alt.items()}
            altf = {k: take(v) for k, v in altf.items()}
            bump(f"{rg}/n", m)
            # default mode unchanged vs the P1 library copy
            bump(f"{rg}/default_mode_vs_cur_mismatch", sum(int((hf[k] != cur[k]).reshape(m, -1).any(1).sum()) for k in ("vid", "A", "q", "z", "valid", "hit")))
            w_p = P["q"].unsqueeze(1) * P["A"]
            for nm, ref in (("alt", alt), ("altf", altf)):
                bad = ((P["vid"] != ref["vid"]).any(1) | ((w_p - ref["w"]).abs().max(1).values > 1e-6) | ((P["z"] - ref["zr"]).abs().max(1).values > 1e-5)
                       | (P["valid"] != ref["valid"]) | (P["hit"] != ref["hit"]) | (P["ctr"].any(1) != ref["ctr"]))
                bump(f"{rg}/{nm}/events_differing_from_production", bad.sum())
            bump(f"{rg}/production_ctr_events", P["ctr"].any(1).sum())
    res[seq] = tot
    print(seq, json.dumps(tot, indent=0, sort_keys=True), flush=True)
(C.OUT_DIR / "check_vs_contour_impl.json").write_text(json.dumps(res, indent=1, sort_keys=True))
