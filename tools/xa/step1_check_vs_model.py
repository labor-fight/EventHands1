#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 independent check: the association inputs / outputs of step1_assoc_regimes.py (R0) against the
production path model/model.py::_xyz_mesh_forward (randomly initialised C1 model on CPU, `xyz_keep_aux`; the
association has no parameters).  Compares sampled event set, rays, FK grid, and every field of the association,
plus the model's own route_stats (valid / hit / resid mass).  Usage: step1_check_vs_model.py [n_steps_per_seq]"""
import os
import sys
import json
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_assoc_regimes as A                                    # noqa: E402
import step1_common as C                                           # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

os.chdir(C.REPO)
from config import load_config                                     # noqa: E402
from model import MNISTModel                                       # noqa: E402
from semkine.eval_track import make_eval_packet                    # noqa: E402
from semkine.xyz_mesh import stride_sample, associate_rays         # noqa: E402

n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 12
cfg = load_config(C.REPO / "configs/semkine/s37_xyz_c1_full_s3407.yaml")
torch.manual_seed(0)
model = MNISTModel(cfg).eval()
model.xyz_keep_aux = True
res = {}
for seq, legacy in C.sequences():
    z = np.load(C.OUT_DIR / f"labels_{seq}.npz")
    lab = {k: z[k] for k in ("ev_x", "ev_y", "ms", "step_ev_start", "samp_local", "samp_start", "src_kind", "src_part", "src_depth", "fingertip_in_front", "step_end_ms")}
    A._init(seq, legacy, "both", lab, np.zeros((len(lab["step_end_ms"]), 51), np.float32))
    g = A._G
    S = len(g["ends"])
    steps = np.unique(np.linspace(0, S - 1, n_per).astype(int))
    betas, K = g["betas"], torch.tensor(np.asarray(C.load_seq_bundle(seq, legacy)[6]), dtype=torch.float32).view(1, 3, 3)
    model.set_hand_context(betas, K)
    bad = {}
    mx = {}
    nev = 0
    for s in steps:
        end, prev_ms = int(g["ends"][s]), int(g["prev"][s])
        ev5 = C._window_events(g["events"], g["offsets"], g["tsub"], end, 50)
        n = len(ev5)
        prev = torch.from_numpy(g["pos51"][prev_ms].astype(np.float32)).view(1, -1)
        pk = make_eval_packet(ev5, prev, betas, K, 50, torch.device("cpu"))
        with torch.no_grad():
            model.forward_packet(pk)
        xl = model.xyz_last
        # ---- my pipeline, R0
        src, mk = stride_sample(torch.tensor([0, n], dtype=torch.int64), C.CAP)
        samp = src[0][mk[0]].numpy()
        x, y = ev5[:, 1], ev5[:, 2]
        px, py = torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).float()
        d_all = torch.stack([(px - g["cx"]) / g["fx"], (py - g["cy"]) / g["fy"], torch.ones_like(px)], dim=-1)
        verts = A._regime_grids(int(s))
        vn = A.vertex_normals(verts, g["faces"])
        ms_rel = np.repeat(np.arange(50), np.diff(g["offsets"][end - 49:end + 2]).astype(np.int64))
        dp, mask, vb, nb, band, front, take = A._batch(samp, ms_rel, d_all, "R0", verts, vn)
        lib = associate_rays(dp, mask, vb, g["faces"], g["vert_faces"], band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN, chunk=A.CHUNK)
        m = len(samp)
        nev += m
        dv = float((xl["verts"] - vb).abs().max())
        mx["verts"] = max(mx.get("verts", 0.0), dv)
        chk = {"mask": bool(xl["mask"].shape[1] == m and xl["mask"].all()),
               "d": torch.equal(xl["d"][0, :m], dp[0]), "verts_1e-6": dv < 1e-6}
        for k_ in ("vid", "valid", "hit"):
            chk[k_] = torch.equal(xl["assoc"][k_][0, :m], lib[k_][0])
        for k_, tol in (("A", 1e-3), ("q", 1e-3), ("z", 1e-4)):
            dd_ = float((xl["assoc"][k_][0, :m] - lib[k_][0]).abs().max())
            mx[k_] = max(mx.get(k_, 0.0), dd_)
            chk[f"{k_}_maxdiff<{tol}"] = dd_ < tol
        rs = model.route_stats
        w = lib["q"] * 1.0
        chk["stats_valid"] = abs(float(rs["xyz_valid"]) - float(lib["valid"].float().mean())) < 1e-7
        chk["stats_hit"] = abs(float(rs["xyz_hit"]) - float(lib["hit"].float().mean())) < 1e-7
        chk["stats_resid"] = abs(float(rs["xyz_resid_mass"]) - float((1.0 - lib["q"]).mean())) < 1e-6
        # labels aligned with the sampled events
        e0 = int(lab["step_ev_start"][s])
        chk["labels_xy"] = bool((lab["ev_x"][e0:e0 + n] == x.astype(np.int16)).all() and (lab["ev_y"][e0:e0 + n] == y.astype(np.int16)).all())
        for k_, v_ in chk.items():
            if not v_:
                bad[k_] = bad.get(k_, 0) + 1
    res[seq] = {"steps_checked": int(len(steps)), "events_checked": int(nev), "failed_checks": bad, "max_abs_diff": mx}
    print(seq, res[seq], flush=True)
(C.OUT_DIR / "check_vs_model.json").write_text(json.dumps(res, indent=1))
print("all identical" if not any(r["failed_checks"] for r in res.values()) else "MISMATCH")
