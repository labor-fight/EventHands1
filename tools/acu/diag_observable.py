#!/usr/bin/env python3
"""ACU D4 (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): do the trackers' persistent errors live in directions the
packet's evidence barely constrains? Zero training, CPU, read-only on existing PFO-engine readings.

    python -m tools.acu.diag_observable run --seq zgz_local --part 0 --nparts 4
    python -m tools.acu.diag_observable table

Per packet (every 4th zgz endpoint): the information matrix of exact correspondences (XAG labels, point-to-point, each
event against the GT mesh at its own ms, Jacobian at the mid-packet GT state), H = sum J^T J / sig^2 (sig = 3 px),
whitened by the prior scales S (1 cm, 0.1 rad): Hw = S H S, eigenbasis U, eigenvalues lam. "Weak" directions: lam < 1
(the packet constrains them less than a 1 cm / 0.1 rad prior), "strong": lam >= 10. For each network reading, the state
errors at that endpoint (E1 incoming - GT previous = the loop's own prior error; E3 prediction - target = the single-step
error from the GT start) are whitened (root as a rotation vector), projected on U, and reported as
  (a) the share of whitened error energy in weak / strong directions, against the share of dimensions (isotropic error);
  (b) the root-aligned joint error (mm per joint, linearised, translation excluded) carried by the weak component.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu.diag_update import NETS, READ_ROOTS  # noqa: E402

OUT = ROOT / "outputs/acu/20261104/diag/d4"
STRIDE = 4
SIG = 3.0
SCALE = np.array([0.01] * 3 + [0.1] * 48)
WEAK, STRONG = 1.0, 10.0
READ = ("B2 3407", "B2 3408", "B2-gt label", "B2 3407 +label test", "S43 3407")


def run(seq, part, nparts):
    import torch
    from scipy.spatial.transform import Rotation
    from semkine import eval_track as ET
    from semkine.xyz_mesh import stride_sample
    from tools.acu import diag_interface as D
    from tools.pfo import engine as E
    from tools.pfo.nets import sequence
    cfg, mano, SM = D.setup()
    dev = torch.device("cpu")
    native = SM._NATIVE["sample_sources"]
    root, seqs = E.sequences(cfg, (seq,))
    ctx = sequence(root, seqs[0][1], seq, dev)
    rows = E.endpoint_rows(ctx.aux)
    betas = ctx.betas.double()
    K = ctx.K[0].double()
    s = 0.375
    fx, fy, cx, cy = float(K[0, 0] * s), float(K[1, 1] * s), float(K[0, 2] * s), float(K[1, 2] * s)
    faces = mano.f.long()
    jacV = torch.func.vmap(torch.func.jacfwd(lambda p: D.fk(mano, p.view(1, 51), betas)[0][0]))

    def ra_joints(p):
        j = D.fk(mano, p.view(1, 51), betas)[1][0]
        return j - j[:1]
    jacJ = torch.func.jacfwd(ra_joints)
    reads = {}
    for name in READ:
        r, tag = NETS[name]
        reads[name] = {t: np.load(READ_ROOTS[r] / tag / t / "states.npz") for t in ("e1", "e3")
                       if (READ_ROOTS[r] / tag / t / "states.npz").exists()}
    sel = list(range(0, len(rows), STRIDE))[part::nparts]
    out = []
    with torch.no_grad():
        for i in sel:
            _, end, prev_ms, _ = rows[i]
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) < 50:
                continue
            b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, 2048)
            lab = native(b, src, mask)
            m = mask[0]
            F_ = faces.shape[0]
            lv = lab["xag_valid"][0][m] & (lab["xag_face"][0][m] >= 0) & (lab["xag_face"][0][m] < F_)
            lb = lab["xag_bary"][0][m].double()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            idx = lv.nonzero().flatten()
            if idx.numel() < 20:
                continue
            lf = lab["xag_face"][0][m].clamp(0, F_ - 1).long()[idx]
            lb = (lb / lb.sum(-1, keepdim=True).clamp_min(1e-12))[idx]
            mid = torch.tensor(np.asarray(ctx.gt[int(end) - 25], np.float64))
            JV = jacV(mid.view(1, 51))[0]                                              # (778, 3, 51)
            V, _ = D.fk(mano, mid.view(1, 51), betas)
            tri = faces[lf]
            X = (lb.unsqueeze(-1) * V[0][tri]).sum(1)
            dX = (lb[:, :, None, None] * JV[tri]).sum(1)
            J = D.proj_jac(X, fx, fy) @ dX                                             # (n, 2, 51)
            H = torch.einsum("nkp,nkq->pq", J, J).numpy() / SIG ** 2
            Hw = SCALE[:, None] * H * SCALE[None, :]
            lam, U = np.linalg.eigh(Hw)
            weak, strong = lam < WEAK, lam >= STRONG
            JJ = jacJ(mid).reshape(-1, 51).numpy() * 1000.0                             # (63, 51) mm per unit
            JJw = JJ * SCALE[None, :]
            rec = {"end": int(end), "n": int(idx.numel()), "dim_weak": int(weak.sum()), "dim_strong": int(strong.sum()),
                   "lam": lam.tolist()}
            for name, rd in reads.items():
                for task in rd:
                    z = rd[task]
                    ends = z[f"{seq}/end_ms"]
                    k = np.nonzero(ends == int(end))[0]
                    if not len(k):
                        continue
                    k = int(k[0])
                    if task == "e1":
                        a, g = z[f"{seq}/incoming"][k].astype(np.float64), z[f"{seq}/gt_previous"][k].astype(np.float64)
                    else:
                        a, g = z[f"{seq}/prediction"][k].astype(np.float64), z[f"{seq}/target"][k].astype(np.float64)
                    e = a - g
                    e[3:6] = (Rotation.from_rotvec(a[3:6]) * Rotation.from_rotvec(g[3:6]).inv()).as_rotvec()
                    u = e / SCALE
                    c = U.T @ u
                    en = c ** 2
                    tot = float(en.sum())
                    pose = np.ones(51, bool)
                    pose[:3] = False                                                   # RA ignores translation
                    uw = U[:, weak] @ c[weak]
                    us = U[:, ~weak] @ c[~weak]
                    jw = JJw[:, pose] @ uw[pose]
                    js = JJw[:, pose] @ us[pose]
                    rec[f"{name}|{task}"] = {"share_weak": float(en[weak].sum() / max(tot, 1e-12)),
                                             "share_strong": float(en[strong].sum() / max(tot, 1e-12)),
                                             "ra_weak_mm": float(np.sqrt((jw.reshape(-1, 3) ** 2).sum(-1).mean())),
                                             "ra_rest_mm": float(np.sqrt((js.reshape(-1, 3) ** 2).sum(-1).mean()))}
            out.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{seq}_p{part}of{nparts}.json").write_text(json.dumps(out))
    print(seq, part, len(out), flush=True)


def table():
    agg = {}
    for f in sorted(OUT.glob("*_p*of*.json")):
        agg.setdefault(f.name.split("_p")[0], []).extend(json.loads(f.read_text()))
    L = ["# ACU D4: where the trackers' errors live relative to the packet's evidence (exact correspondences, sig 3 px)", "",
         "Weak = whitened information < 1 (packet constrains less than a 1 cm / 0.1 rad prior), strong = >= 10. Shares are of "
         "whitened error energy (isotropic error would give the dimension shares). RA mm: linearised root-aligned joint error "
         "(per joint RMS, translation excluded) of the weak component / of the rest.", ""]
    for seq, recs in agg.items():
        dw = np.mean([r["dim_weak"] for r in recs]) / 51
        ds = np.mean([r["dim_strong"] for r in recs]) / 51
        L += [f"## {seq} ({len(recs)} packets): dimension share weak {dw:.2f}, strong {ds:.2f}", "",
              "| network | reading | share weak | share strong | RA mm weak / rest |", "|---|---|---|---|---|"]
        for name in READ:
            for task in ("e1", "e3"):
                key = f"{name}|{task}"
                v = [r[key] for r in recs if key in r]
                if not v:
                    continue
                sw = np.mean([x["share_weak"] for x in v])
                ss = np.mean([x["share_strong"] for x in v])
                rw = np.sqrt(np.mean([x["ra_weak_mm"] ** 2 for x in v]))
                rr = np.sqrt(np.mean([x["ra_rest_mm"] ** 2 for x in v]))
                L.append(f"| {name} | {'E1 prior error' if task == 'e1' else 'E3 error'} | {sw:.2f} | {ss:.2f} | {rw:.2f} / {rr:.2f} |")
        L.append("")
    (OUT.parent / "d4.md").write_text("\n".join(L))
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seq", required=True)
    r.add_argument("--part", type=int, default=0)
    r.add_argument("--nparts", type=int, default=1)
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "run":
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
            raise SystemExit("CPU only: set CUDA_VISIBLE_DEVICES=''")
        run(a.seq, a.part, a.nparts)
    else:
        table()


if __name__ == "__main__":
    main()
