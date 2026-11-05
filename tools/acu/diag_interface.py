#!/usr/bin/env python3
"""ACU D2 (docs/EXPERIMENT_RECORDS.md#acu-diag-20261104): what pose information each observation interface carries under
the deployed router's real errors. Zero training, CPU, descriptive; not an update rule (one fixed-damping MAP step is the
measuring instrument, the same for every interface).

    python -m tools.acu.diag_interface run --seq zgz_local --part 0 --nparts 4
    python -m tools.acu.diag_interface table

Per packet (every 4th zgz endpoint): S45's stride-sampled nodes, their XAG pseudo-labels and r_s2d's routes. Each event is
compared with the GT mesh *at its own millisecond* (within-packet motion removed by construction), perturbed by one
packet-wide error d (none, or the engine's `pert` draws R15 / F15 / T30 / N). Interfaces (residual r_i in pixels):
  PP-lab      point-to-point, label face + barycentrics          r = pi(X_lab) - p                  (2-D)
  PP-rt       point-to-point, r_s2d face + barycentrics (on-hand) r = pi(X_rt) - p                   (2-D)
  PL-lab      label point, image normal of its face              r = nu^T (pi(X_lab) - p)           (1-D)
  PL-rt       r_s2d point, image normal of its face              r = nu^T (pi(X_rt) - p)            (1-D)
  CA-lab16    label part identity; nearest rim vertex of that part on the (perturbed) mesh, its image normal
  CA-rt16     r_s2d part identity, same association               r = nu_c^T (pi(V_c) - p)          (1-D)
  CA-rt6      r_s2d finger-group identity (palm + 5 fingers), rim vertices of the group
Rim vertex = |n . v/|v|| < RIM_COS (grazing). Step: delta = -(H / sig^2 + diag(s^-2))^-1 g / sig^2, H = sum w J^T J,
g = sum w J^T r, Huber weight w at HUBER px, s = 1 cm / 0.1 rad. Scores (mm / deg, joint space): from GT the injected
change per group; from each perturbation the joint-space retention per group and the RA-MPJPE after the step.
Information (GT start, w = 1, sig = 1 px): posterior std per group.
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

OUT = ROOT / "outputs/acu/20261104/diag/d2"
STRIDE = 4
KINDS = ("GT", "R15", "F15", "T30", "N")
IFACES = ("PP-lab", "PP-rt", "PL-lab", "PL-rt", "CA-lab16", "CA-rt16", "CA-lab6", "CA-rt6")
#: router-minus-label pairs: the same events (label-valid and routed), only the router-supplied part differs
PAIRS = (("PP-rt", "PP-lab"), ("PL-rt", "PL-lab"), ("CA-rt16", "CA-lab16"), ("CA-rt6", "CA-lab6"))
SETTINGS = {"s2h5": (2.0, 5.0), "s4h10": (4.0, 10.0)}
SCALE = np.array([0.01] * 3 + [0.1] * 48)
RIM_COS = 0.35
GROUP_OF_PART = np.array([0, 2, 2, 2, 3, 3, 3, 5, 5, 5, 4, 4, 4, 1, 1, 1])
THREADS = 8
MANO_F = {}


def setup():
    import torch
    import yaml
    from mano_layer import ManoLayer
    from semkine import pfo_router as PR, s45sm as SM
    from tools.pfo.common import XAG_SOURCE_ROOT
    from tools.pfo.router_train import checkpoint
    torch.set_num_threads(THREADS)
    cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    PR.register()
    SM.install("pfr", str(checkpoint("r_s2d")), "pfa", 0.5)
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval().double()
    return cfg, mano, SM


def fk(mano, params, betas):
    from pose_repr import decode_to_mano_inputs
    dec = decode_to_mano_inputs(params, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
    return mano(betas.expand(params.shape[0], -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])


def perturb_fn(kind, truth, rng, prior):
    """The engine's injected error of `kind` at the mid-packet truth, as a map applied to every ms's GT state."""
    from scipy.spatial.transform import Rotation
    from tools.pfo.engine import inject
    if kind == "GT":
        return lambda s: s.copy(), None
    x, delta, _ = inject(truth, kind, rng, prior)
    if kind.startswith("R"):
        Rd = Rotation.from_rotvec(x[3:6].astype(np.float64)) * Rotation.from_rotvec(truth[3:6].astype(np.float64)).inv()

        def f(s):
            o = np.array(s, np.float64, copy=True)
            o[3:6] = (Rd * Rotation.from_rotvec(o[3:6])).as_rotvec()
            return o
        return f, x
    return (lambda s: np.asarray(s, np.float64) + delta), x


def image_normal(n, X, fx, fy):
    """(.., 2) unit image direction of a 3-D surface normal n at point X (camera frame)."""
    import torch
    Z = X[..., 2].clamp_min(1e-6)
    e = torch.stack([fx * (n[..., 0] - X[..., 0] * n[..., 2] / Z), fy * (n[..., 1] - X[..., 1] * n[..., 2] / Z)], -1) / Z[..., None]
    return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def proj(X, fx, fy, cx, cy):
    import torch
    Z = X[..., 2].clamp_min(1e-6)
    return torch.stack([fx * X[..., 0] / Z + cx, fy * X[..., 1] / Z + cy], -1)


def proj_jac(X, fx, fy):
    """(.., 2, 3) d pi / d X."""
    import torch
    Z = X[..., 2].clamp_min(1e-6)
    J = torch.zeros(*X.shape[:-1], 2, 3, dtype=X.dtype)
    J[..., 0, 0] = fx / Z
    J[..., 0, 2] = -fx * X[..., 0] / Z ** 2
    J[..., 1, 1] = fy / Z
    J[..., 1, 2] = -fy * X[..., 1] / Z ** 2
    return J


def run(seq, part, nparts):
    import torch
    from scipy.spatial.transform import Rotation
    from semkine import eval_track as ET
    from semkine.xyz_mesh import stride_sample, vertex_normals
    from tools.pfo import engine as E
    from tools.pfo.nets import sequence
    from tools.acu.diag_update import hand_joints
    cfg, mano, SM = setup()
    from mano_layer import ManoLayer
    MANO_F["m"] = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    dev = torch.device("cpu")
    net = SM._net(dev)
    native = SM._NATIVE["sample_sources"]
    root, seqs = E.sequences(cfg, (seq,))
    _, directory = seqs[0]
    ctx = sequence(root, directory, seq, dev)
    rows = E.endpoint_rows(ctx.aux)
    prior = E.prior_noise_fn(cfg)
    betas = ctx.betas.double()
    K = ctx.K[0].double()
    s = 0.375
    fx, fy, cx, cy = float(K[0, 0] * s), float(K[1, 1] * s), float(K[0, 2] * s), float(K[1, 2] * s)
    faces = mano.f.long() if hasattr(mano, "f") else torch.as_tensor(np.asarray(np.load(ROOT / "assets/mano_right.npz")["f"]), dtype=torch.long)
    vpart = torch.as_tensor(np.asarray(mano.weights.detach().argmax(1)), dtype=torch.long)
    vgroup = torch.as_tensor(GROUP_OF_PART)[vpart]
    sel = list(range(0, len(rows), STRIDE))[part::nparts]
    out = {"rows": [], "inj": {}, "ret": {}, "ra": {}, "info": {}}
    del Rotation
    jac = torch.func.vmap(torch.func.jacfwd(lambda p: fk(mano, p.view(1, 51), betas)[0][0]))
    with torch.no_grad():
        for i in sel:
            _, end, prev_ms, _ = rows[i]
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) < 50:
                continue
            b = ET.make_eval_packet(ev5, torch.zeros(1, 51), ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, 2048)
            ro, routed, rface, rbary, _ = SM.route_nodes(net, SM._STATE["router"], b.events, b.ptr, b.delta_t_s, 0.5)
            lab = native(b, src, mask)
            m = mask[0]
            ev = b.events[src[0][m]].double()
            px = torch.stack([ev[:, 1], ev[:, 2]], -1)
            ms = (end - 49 + (ev[:, 3] * 1000.0).floor().long().clamp(0, 49)).long()
            F_ = faces.shape[0]
            lv = (lab["xag_valid"][0][m] & (lab["xag_face"][0][m] >= 0) & (lab["xag_face"][0][m] < F_))
            lb = lab["xag_bary"][0][m].double()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lf = lab["xag_face"][0][m].clamp(0, F_ - 1).long()
            lb = lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)
            lpart = lab["xag_part"][0][m].long()
            rv = routed[0][m]
            rf = rface[0][m].long()
            rb = rbary[0][m].double()
            rpart = ro["part"][0][m].float().argmax(-1).long()
            gts = np.asarray(ctx.gt[int(end) - 49:int(end) + 1], np.float64)
            mid = gts[24].astype(np.float32)
            starts = {}
            for k, kind in enumerate(KINDS):
                salt = 101 + list(E.PERT_KINDS).index(kind) if kind != "GT" else None
                f, x_mid = perturb_fn(kind, mid, E.rng_for(seq, int(end), salt) if kind != "GT" else None, prior)
                starts[kind] = (np.stack([f(g) for g in gts]), f(mid.astype(np.float64)))
            Sx = torch.tensor(np.stack([starts[k][0] for k in KINDS]))                       # (S, 50, 51)
            V, _ = fk(mano, Sx.view(-1, 51), betas)
            V = V.view(len(KINDS), 50, -1, 3)
            Xmid = torch.tensor(np.stack([starts[k][1] for k in KINDS]))                     # (S, 51)
            JV = jac(Xmid)                                                                 # (S, 778, 3, 51)
            nV = vertex_normals(V.view(-1, V.shape[2], 3).float(), faces).double().view_as(V)
            res = {}
            common = lv & rv & (lpart >= 0)
            for iface in IFACES:
                if iface in ("PP-lab", "PL-lab"):
                    fsel, bsel = lf, lb
                elif iface in ("PP-rt", "PL-rt"):
                    fsel, bsel = rf, rb
                idx = common.nonzero().flatten()
                if idx.numel() < 10:
                    continue
                Hs, gs = [], []
                for si in range(len(KINDS)):
                    Vm = V[si, ms[idx] - (end - 49)]                                       # (n, 778, 3)
                    if iface.startswith(("PP", "PL")):
                        tri = faces[fsel[idx]]                                             # (n, 3)
                        Xv = torch.gather(Vm, 1, tri.unsqueeze(-1).expand(-1, -1, 3))       # (n, 3, 3)
                        X = (bsel[idx].unsqueeze(-1) * Xv).sum(1)
                        dX = (bsel[idx][:, :, None, None] * JV[si][tri]).sum(1)            # (n, 3, 51)
                        Jp = proj_jac(X, fx, fy) @ dX                                      # (n, 2, 51)
                        r2 = proj(X, fx, fy, cx, cy) - px[idx]
                        if iface.startswith("PP"):
                            J, r = Jp, r2
                        else:
                            n3 = torch.cross(Xv[:, 1] - Xv[:, 0], Xv[:, 2] - Xv[:, 0], dim=-1)
                            nu = image_normal(n3 / n3.norm(dim=-1, keepdim=True).clamp_min(1e-12), X, fx, fy)
                            J, r = (nu.unsqueeze(1) @ Jp), (nu * r2).sum(-1, keepdim=True)
                    else:
                        ident = lpart[idx] if iface.startswith("CA-lab") else rpart[idx]
                        six = iface in ("CA-lab6", "CA-rt6")
                        vmatch = vgroup if six else vpart
                        ident = torch.as_tensor(GROUP_OF_PART)[ident] if six else ident
                        nm = nV[si, ms[idx] - (end - 49)]
                        cosv = (nm * Vm).sum(-1).abs() / Vm.norm(dim=-1).clamp_min(1e-9)
                        rim = (cosv < RIM_COS) & (vmatch.unsqueeze(0) == ident.unsqueeze(1))
                        pv = proj(Vm, fx, fy, cx, cy)
                        d2 = ((pv - px[idx].unsqueeze(1)) ** 2).sum(-1).masked_fill(~rim, float("inf"))
                        c = d2.argmin(1)
                        ok = torch.isfinite(d2.gather(1, c[:, None]).squeeze(1))
                        Xc = Vm[torch.arange(len(c)), c]
                        nu = image_normal(nm[torch.arange(len(c)), c], Xc, fx, fy)
                        Jp = proj_jac(Xc, fx, fy) @ JV[si][c]
                        J = (nu.unsqueeze(1) @ Jp) * ok[:, None, None]
                        r = (nu * (proj(Xc, fx, fy, cx, cy) - px[idx])).sum(-1, keepdim=True) * ok[:, None]
                    Hs.append(J)
                    gs.append(r)
                res[iface] = (Hs, gs)
            # ---- MAP steps and scores (all states of the packet through one FK)
            truth_mid = gts[24]
            from scipy.spatial.transform import Rotation
            row = {"seq": seq, "end": int(end), "n_lab": int(lv.sum()), "n_rt": int(rv.sum()), "n_common": int(common.sum()),
                   "rt_part_acc": float((rpart[common] == lpart[common]).double().mean()) if common.any() else float("nan"),
                   "rt_group_acc": float((torch.as_tensor(GROUP_OF_PART)[rpart[common]] ==
                                          torch.as_tensor(GROUP_OF_PART)[lpart[common]]).double().mean()) if common.any() else float("nan")}
            keys, x0s, x1s = [], [], []
            for iface, (Js, rs) in res.items():
                for sname, (sig, hub) in SETTINGS.items():
                    for si, kind in enumerate(KINDS):
                        J, r = Js[si], rs[si]
                        rn = r.norm(dim=-1)
                        w = (hub / rn.clamp_min(1e-9)).clamp(max=1.0)
                        H = torch.einsum("n,nkp,nkq->pq", w, J, J) / sig ** 2 + torch.diag(torch.tensor(SCALE ** -2.0))
                        g = torch.einsum("n,nkp,nk->p", w, J, r) / sig ** 2
                        delta = -torch.linalg.solve(H, g).numpy()
                        out.setdefault("delta", {}).setdefault(f"{iface}|{sname}|{kind}", []).append(delta.tolist())
                        keys.append((iface, sname, kind))
                        x0s.append(starts[kind][1])
                        x1s.append(starts[kind][1] + delta)
                J = Js[0]
                H = torch.einsum("nkp,nkq->pq", J, J) + torch.diag(torch.tensor(SCALE ** -2.0))
                P = torch.linalg.inv(H).numpy()
                out["info"].setdefault(iface, []).append([float(np.sqrt(np.trace(P[:3, :3]))) * 1000,
                                                         float(np.rad2deg(np.sqrt(np.trace(P[3:6, 3:6])))),
                                                         float(np.sqrt(np.trace(P[6:, 6:])))])
            if not keys:
                continue
            allx = np.concatenate([np.stack(x0s), np.stack(x1s), truth_mid[None]]).astype(np.float32)
            jh = hand_joints(allx, seq)
            jfull, _ = ET.mano_fk(MANO_F["m"], allx, ctx.betas.float(), dev)
            jfull = ET.root_align(jfull.astype(np.float64)) * 1000.0
            n = len(keys)
            jt, jft = jh[-1], jfull[-1]
            Rt = Rotation.from_rotvec(truth_mid[3:6])
            for q, (iface, sname, kind) in enumerate(keys):
                x0, x1 = x0s[q], x1s[q]
                key = f"{iface}|{sname}|{kind}"
                out["ra"].setdefault(key, []).append(float(np.linalg.norm(jfull[n + q] - jft, axis=-1).mean()))
                rot0 = (Rotation.from_rotvec(x0[3:6]) * Rt.inv()).as_rotvec()
                rot1 = (Rotation.from_rotvec(x1[3:6]) * Rt.inv()).as_rotvec()
                dT0, dT1 = (x0[:3] - truth_mid[:3]) * 1000, (x1[:3] - truth_mid[:3]) * 1000
                dF0, dF1 = (jh[q] - jt).reshape(-1), (jh[n + q] - jt).reshape(-1)
                if kind == "GT":
                    out["inj"].setdefault(key, []).append([float(np.linalg.norm(dT1)), float(np.rad2deg(np.linalg.norm(rot1))),
                                                            float(np.sqrt((dF1 ** 2).sum() / 21))])
                else:
                    ret = []
                    for a0, a1 in ((dT0, dT1), (rot0, rot1), (dF0, dF1)):
                        den = float((a0 * a0).sum())
                        ret.append(float((a1 * a0).sum() / den) if den > 1e-12 else float("nan"))
                    out["ret"].setdefault(key, []).append(ret)
            out["rows"].append(row)
            if len(out["rows"]) % 20 == 0:
                print(seq, part, len(out["rows"]), "/", len(sel), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{seq}_p{part}of{nparts}.json").write_text(json.dumps(out))


def table():
    files = sorted(OUT.glob("*_p*of*.json"))
    agg = {}
    for f in files:
        seq = f.name.split("_p")[0]
        d = json.loads(f.read_text())
        a = agg.setdefault(seq, {"rows": 0, "inj": {}, "ret": {}, "ra": {}, "info": {}, "delta": {}, "acc": []})
        a["rows"] += len(d["rows"])
        a["acc"].extend([(r["rt_part_acc"], r["rt_group_acc"], r["n_common"]) for r in d["rows"]])
        for k in ("inj", "ret", "ra", "info", "delta"):
            for key, v in d[k].items():
                a[k].setdefault(key, []).extend(v)
    L = ["# ACU D2: observation interfaces under r_s2d's real errors (one fixed MAP step as the instrument)", "",
         "GT start: injected change T mm / R deg / F mm-per-joint (RMS); perturbed starts: joint-space retention of the "
         "perturbed group (1 = not corrected, 0 = fully corrected; median); hand-frame MPJPE after the step (mean, mm).", ""]
    for seq, a in agg.items():
        L += [f"## {seq} ({a['rows']} packets)", ""]
        for sname in SETTINGS:
            L += [f"### setting {sname}", "", "| interface | GT inj T / R / F | R15: ret R / F-ra | F15: ret F / F-ra | T30: ret T | N: ret T / R / F | GT F-ra |",
                  "|---|---|---|---|---|---|---|"]
            for iface in IFACES:
                k = lambda kind: f"{iface}|{sname}|{kind}"  # noqa: E731
                if k("GT") not in a["inj"]:
                    continue
                inj = np.sqrt(np.mean(np.square(np.array(a["inj"][k("GT")])), 0))
                med = lambda kind, j: float(np.nanmedian(np.array(a["ret"][k(kind)])[:, j]))  # noqa: E731
                ra = lambda kind: float(np.mean(a["ra"][k(kind)]))  # noqa: E731
                L.append(f"| {iface} | {inj[0]:.2f} / {inj[1]:.2f} / {inj[2]:.2f} | {med('R15', 1):.2f} / {ra('R15'):.2f} | "
                         f"{med('F15', 2):.2f} / {ra('F15'):.2f} | {med('T30', 0):.2f} | {med('N', 0):.2f} / {med('N', 1):.2f} / {med('N', 2):.2f} | {ra('GT'):.2f} |")
            L.append("")
        acc = np.array(a["acc"], np.float64)
        L += [f"r_s2d on the common events: part accuracy {np.nanmean(acc[:, 0]):.3f}, finger-group accuracy {np.nanmean(acc[:, 1]):.3f},"
              f" mean common events per packet {acc[:, 2].mean():.0f}", "",
              "Router-minus-label step difference from GT (same events; joint space via the step's own FK is not needed: "
              "R = root axis-angle deg, F = finger axis-angle RMS rad, T mm), RMS over packets", "",
              "| pair | setting | T | R | F |", "|---|---|---|---|---|"]
        for rt, lb_ in PAIRS:
            for sname in SETTINGS:
                ka, kb = f"{rt}|{sname}|GT", f"{lb_}|{sname}|GT"
                if ka in a["delta"] and kb in a["delta"]:
                    dd = np.array(a["delta"][ka]) - np.array(a["delta"][kb])
                    L.append(f"| {rt} - {lb_} | {sname} | {np.sqrt((dd[:, :3] ** 2).sum(1).mean()) * 1000:.2f} | "
                             f"{np.rad2deg(np.sqrt((dd[:, 3:6] ** 2).sum(1).mean())):.2f} | {np.sqrt((dd[:, 6:] ** 2).mean()):.4f} |")
        L.append("")
        L += ["Information at GT (iid 1 px noise, prior 1 cm / 0.1 rad): posterior std T mm / R deg / finger axis-angle rad (prior 17.3 / 9.9 / 0.67)", "",
              "| interface | T | R | F |", "|---|---|---|---|"]
        for iface in IFACES:
            if iface in a["info"]:
                v = np.median(np.array(a["info"][iface]), 0)
                L.append(f"| {iface} | {v[0]:.2f} | {v[1]:.2f} | {v[2]:.3f} |")
        L.append("")
    (OUT.parent / "d2.md").write_text("\n".join(L))
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
