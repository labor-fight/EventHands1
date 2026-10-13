#!/usr/bin/env python3
r"""[XYZ-ASSOC] step4: frozen-weight probes on a trained S37-XYZ checkpoint.

D  confidence scaling: the association (vid / A / valid / hit) and the event features are kept, only the
   per-event confidence q is rescaled (or some events' q is zeroed) in a forward pre-hook on `model.xyz`.
E1 geometry-channel ablation by forward pre-hooks on `model.xyz.rel` (columns of the 13-d relation input)
   and `model.xyz.edge` (edge vectors); out-of-distribution inputs, so only sensitivity.
E2 prev-state perturbation recovery (teacher-forced, then k-step recursion with real events and own prev).
E3 event perturbation response (shift +4 px, events of another packet, empty packet).

Read-only on the repository and the checkpoint. All hooks live in `Rig`; with `rig.spec = None` every hook
returns None (identity), which the closed-loop check proves bit for bit against the P2 record.

Reuses `tools/xa/probe_xa_assoc.py` (checkpoint loading, step enumeration, TF packets, update errors).

    CUDA_VISIBLE_DEVICES=2 python tools/xa/probe_xa_conf.py --part tfD1,clD ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import probe_xa_assoc as P                                         # noqa: E402  (reused, unchanged)
from semkine.events import EventPacketBatch                        # noqa: E402

ET, XM = P.ET, P.XM
REPO = P.REPO
ST = P.ST
RAD2DEG = P.RAD2DEG
OUT_DEFAULT = "outputs/xa/step4/conf"
P2_JSON = "outputs/xa/step2/p2a_closedloop.json"
BLK = ("t_mm", "rot_deg", "fing_rad", "mcp_rad", "fing_other_rad")
TB = {"t": 0, "rot": 1, "fing": 2, "mcp": 3}
log = P.log


# ============================================================================================ statistics
class Est:
    def __init__(self, point, boot, n):
        self.point, self.boot, self.n = float(point), np.asarray(boot, np.float64), int(n)

    def ci(self):
        b = self.boot[np.isfinite(self.boot)]
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))] if len(b) > 20 else [float("nan")] * 2

    def d(self):
        return {"mean": self.point, "ci95": self.ci(), "n": self.n}

    def __sub__(self, o):
        return Est(self.point - o.point, self.boot - o.boot, self.n)


def ratio(a: Est, b: Est, minus_one=True):
    """a / b - 1 (paired bootstrap)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        pt, bt = a.point / b.point, a.boot / b.boot
    return Est(pt - 1 if minus_one else pt, bt - 1 if minus_one else bt, a.n)


class Boot:
    """Moving-block bootstrap indices (block length L steps) per sequence, shared by every estimate (paired)."""

    def __init__(self, ns: dict, B=1000, L=20, seed=7):
        self.B, self.L, self.idx, self.ns = B, L, {}, dict(ns)
        for k, (s, n) in enumerate(ns.items()):
            n = int(n)
            if n < 1:
                self.idx[s] = np.zeros((B, 0), np.int64)
                continue
            rng = np.random.default_rng([seed, k])
            Lk = max(1, min(L, n))
            nb = int(math.ceil(n / Lk))
            st = rng.integers(0, n - Lk + 1, (B, nb))
            self.idx[s] = (st[:, :, None] + np.arange(Lk)[None, None, :]).reshape(B, -1)[:, :n]

    def est(self, x: dict, seqs, sel: dict | None = None) -> Est:
        """Pooled mean of x over the selected finite packets of `seqs`; ratio-of-sums bootstrap."""
        num, den = 0.0, 0.0
        nb = np.zeros(self.B)
        db = np.zeros(self.B)
        for s in seqs:
            v = np.asarray(x[s], np.float64)
            m = np.isfinite(v)
            if sel is not None:
                m &= np.asarray(sel[s], bool)
            if not len(v):
                continue
            vz = np.where(m, v, 0.0)
            mf = m.astype(np.float64)
            num += vz.sum()
            den += mf.sum()
            ii = self.idx[s]
            nb += vz[ii].sum(1)
            db += mf[ii].sum(1)
        pt = num / den if den > 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            bt = np.where(db > 0, nb / np.where(db > 0, db, 1), np.nan)
        return Est(pt, bt, int(den))


def groups_for(seqs, fit):
    """name -> (sequence list, per-sequence selection or None)"""
    g = {s: ([s], None) for s in seqs}
    g["overall"] = (list(seqs), None)
    loc = [s for s in seqs if P.LOCAL(s)]
    if loc and fit is not None:
        g["local_fit_true"] = (loc, {s: fit[s] for s in loc})
        g["local_fit_false"] = (loc, {s: ~fit[s] for s in loc})
    return g


# ============================================================================================ hooks
class Rig:
    """Every hook of this probe. `spec` (dict) selects what the hooks do; `None` = identity.

    q spec     {"name", "q": "scale"|"keep"|"drop", "s", "target": all|near|hit, "frac", "rng"}  (xyz pre-hook)
    rel spec   {"rel_zero": (cols,), "z_median": True}                                           (rel pre-hook)
    edge spec  {"edge_zero": True}                                                               (edge pre-hook)
    Counters (`cnt`, keyed "<spec name>/<what>") count how often and how much each hook changed something."""

    def __init__(self, model):
        self.model, x = model, model.xyz
        self.spec = None
        self.capture = False
        self.cap = {}
        self.cnt = defaultdict(float)
        self.stats_log = None
        self._zmed = None
        self.ref = None                      # baseline capture of the same packet (D1b path-isolation variants)
        self.h = [x.register_forward_pre_hook(self._xyz_pre), x.register_forward_hook(self._xyz_post),
                  x.vin.register_forward_pre_hook(self._vin_pre), x.vin.register_forward_hook(self._vin_post),
                  x.rel.register_forward_pre_hook(self._rel_pre), x.edge.register_forward_pre_hook(self._edge_pre),
                  x.slot_proj.register_forward_pre_hook(self._slot_pre)]

    def remove(self):
        for h in self.h:
            h.remove()

    # ---- q rewriting -----------------------------------------------------------------------------
    def _new_q(self, q, mask, hit, sp):
        live = mask.bool()
        cand = live & (q > 0)
        kind = sp["q"]
        if kind == "scale":
            tgt = sp.get("target", "all")
            sel = cand if tgt == "all" else (cand & hit if tgt == "hit" else cand & ~hit)
            return torch.where(sel, (q * float(sp["s"])).clamp(max=1.0), q), int(sel.sum())
        flat = np.flatnonzero(cand.reshape(-1).cpu().numpy())
        n = len(flat)
        rng = sp["rng"]
        perm = rng.permutation(n)
        frac = float(sp["frac"])
        if kind == "keep":
            k = min(n, max(1, int(round(frac * n)))) if n else 0
            zero = flat[perm[k:]]
        else:                                                      # "drop": zero a fraction of the candidates
            k = int(round(frac * n))
            zero = flat[perm[:k]]
        qn = q.clone()
        if len(zero):
            qn.view(-1)[torch.from_numpy(zero).to(q.device)] = 0.0
        return qn, int(len(zero))

    def _xyz_pre(self, mod, args):
        self.cnt["xyz_calls"] += 1
        self.cap = {}
        self._zmed = None
        sp = self.spec
        tok, d, mask, verts, assoc, prev48 = args
        self.cap["mask"] = mask
        if sp is None:
            return None
        nm = sp["name"]
        out_assoc = None
        if sp.get("q"):
            q = assoc["q"]
            qn, n_tgt = self._new_q(q, mask, assoc["hit"], sp)
            n_ch = int((qn != q).sum())
            self.cnt[f"{nm}/q_hook_calls"] += 1
            self.cnt[f"{nm}/q_events_targeted"] += n_tgt
            self.cnt[f"{nm}/q_events_changed"] += n_ch
            self.cnt[f"{nm}/q_events_total_pos"] += int((q > 0).sum())
            self.cap["q0"], self.cap["q1"] = q, qn
            if n_ch:
                out_assoc = dict(assoc)
                out_assoc["q"] = qn
        if sp.get("z_median"):
            sel = mask.bool() & assoc["valid"]
            zm = torch.quantile(assoc["z"][sel].float(), 0.5) if bool(sel.any()) else torch.tensor(XM.Z_REF, device=mask.device)
            self._zmed = (zm - XM.Z_REF) / XM.Z_SCALE
            self.cnt[f"{nm}/zmed_packets"] += 1
            self.cnt[f"{nm}/zmed_value_sum_m"] += float(zm)
        return None if out_assoc is None else (tok, d, mask, verts, out_assoc, prev48)

    def _xyz_post(self, mod, args, out):
        delta, stats, aux = out
        if self.stats_log is not None:
            self.stats_log.append({k: float(v) for k, v in stats.items()})
        if self.capture:
            self.cap.update(slots=aux["slots"], O=aux["O"], S=aux["S"], w=aux["w"], r=aux["r"],
                            stats={k: float(v) for k, v in stats.items()})

    def _vin_pre(self, mod, args):
        x = args[0]
        C = (x.shape[-1] - 1) // 2
        sp = self.spec
        out = None
        if sp is not None and sp.get("lm_fix") and self.ref is not None and "lm" in self.ref:
            # D1b: keep the scaled-q run but give the net the BASELINE log1p(mass) channel (isolates the residual path)
            x = x.clone()
            nm = sp["name"]
            self.cnt[f"{nm}/lm_fix_calls"] += 1
            self.cnt[f"{nm}/lm_fix_elems_changed"] += int((x[..., 2 * C] != self.ref["lm"]).sum())
            x[..., 2 * C] = self.ref["lm"]
            out = (x,)
        if self.capture:
            self.cap.update(mean=x[..., :C], peak=x[..., C:2 * C], lm=x[..., 2 * C])
        return out

    def _slot_pre(self, mod, args):
        sp = self.spec
        if sp is None or not sp.get("res_fix") or self.ref is None or "slots" not in self.ref:
            return None
        # D1b: keep the scaled-q run but give the readout the BASELINE residual slot (isolates the mass path)
        x = args[0].clone()
        B = x.shape[0]
        v = x.view(B, XM.XYZMeshNet.N_SLOTS, -1)
        nm = sp["name"]
        self.cnt[f"{nm}/res_fix_calls"] += 1
        self.cnt[f"{nm}/res_fix_elems_changed"] += int((v[:, 16, :] != self.ref["slots"][:, 16, :].to(v.dtype)).sum())
        v[:, 16, :] = self.ref["slots"][:, 16, :].to(v.dtype)
        return (x,)

    def _vin_post(self, mod, args, out):
        if self.capture:
            self.cap["H"] = out

    def _rel_pre(self, mod, args):
        sp = self.spec
        if sp is None or not (sp.get("rel_zero") or sp.get("z_median")):
            return None
        x = args[0]
        assert x.shape[1] == XM.REL_IN, x.shape                      # 7 token + 3 dXYZ + zn + valid + hit
        nm = sp["name"]
        x = x.clone()
        self.cnt[f"{nm}/rel_calls"] += 1
        cols = list(sp.get("rel_zero") or ())
        if cols:
            self.cnt[f"{nm}/rel_nonzero_zeroed"] += int((x[:, cols] != 0).sum())
            x[:, cols] = 0
        if sp.get("z_median"):
            matched = x[:, 11] > 0.5                                  # matched-record call (residual rows have valid = 0)
            self.cnt[f"{nm}/zmed_rows"] += int(matched.sum())
            self.cnt[f"{nm}/zmed_rows_changed"] += int((x[matched, 10] != self._zmed.to(x.dtype)).sum())
            x[matched, 10] = self._zmed.to(x.dtype)
        return (x,)

    def _edge_pre(self, mod, args):
        sp = self.spec
        if sp is None or not sp.get("edge_zero"):
            return None
        g = args[0]
        nm = sp["name"]
        self.cnt[f"{nm}/edge_calls"] += 1
        self.cnt[f"{nm}/edge_nonzero_zeroed"] += int((g != 0).sum())
        return (torch.zeros_like(g),)

    def run(self, batch):
        self.capture, self.cap = True, {}
        with torch.no_grad():
            pred = self.model.forward_packet(batch)
        cap, self.cap, self.capture = self.cap, {}, False
        return pred, cap

    def counters(self, name):
        pre = name + "/"
        return {k[len(pre):]: v for k, v in self.cnt.items() if k.startswith(pre)}


# ============================================================================================ packet records
FEAT_KEYS = ("H_fro", "H_obs_norm", "H_rel", "mean_rel", "peak_rel", "peak_rel_common", "lm_rel", "frac_vert_kept",
             "slot_rel_joint", "slot_rel_resid", "verts_direct", "verts_prop", "resid_mass", "valid", "hit", "n_rec",
             "mass_sum", "r_sum")
ERR_KEYS = ("e_t_mm", "e_rot_deg", "e_rot_geo_deg", "e_fing_rad", "d_t_mm", "d_rot_deg", "d_fing_rad",
            "dd_t_mm", "dd_rot_deg", "dd_fing_rad")
NAN = float("nan")


def _rel(a, b):
    nb = float(b.float().norm())
    return float((a.float() - b.float()).norm()) / nb if nb > 0 else NAN


def feature_metrics(cap, cap0):
    r = {k: NAN for k in FEAT_KEYS}
    if "H" not in cap:
        return r
    H, O = cap["H"][0].float(), cap["O"][0]
    r["H_fro"] = float(H.norm())
    r["H_obs_norm"] = float(H[O > 0].norm(dim=-1).mean()) if bool((O > 0).any()) else NAN
    st = cap["stats"]
    r.update(verts_direct=st["xyz_verts_direct"], verts_prop=st["xyz_verts_prop"], resid_mass=st["xyz_resid_mass"],
             valid=st["xyz_valid"], hit=st["xyz_hit"], n_rec=float((cap["w"] > 0).sum()), mass_sum=float(cap["w"].sum()),
             r_sum=float(cap["r"].sum()))
    if cap0 and "H" in cap0:
        O0 = cap0["O"][0]
        r["H_rel"] = _rel(H, cap0["H"][0])
        r["mean_rel"] = _rel(cap["mean"][0], cap0["mean"][0])
        r["peak_rel"] = _rel(cap["peak"][0], cap0["peak"][0])
        r["lm_rel"] = _rel(cap["lm"][0], cap0["lm"][0])
        both = (O > 0) & (O0 > 0)
        r["peak_rel_common"] = _rel(cap["peak"][0][both], cap0["peak"][0][both]) if bool(both.any()) else NAN
        r["frac_vert_kept"] = float((O > 0).sum()) / max(float((O0 > 0).sum()), 1.0)
        r["slot_rel_joint"] = _rel(cap["slots"][0, :16], cap0["slots"][0, :16])
        r["slot_rel_resid"] = _rel(cap["slots"][0, 16], cap0["slots"][0, 16])
    return r


def err_record(p, p0, prev, gt):
    e = P.update_errors(p, prev, gt)
    prev = np.asarray(prev, np.float64)
    dl, d0 = p - prev, p0 - prev
    n = np.linalg.norm
    return {"e_t_mm": e["e_t_mm"], "e_rot_deg": e["e_rot_deg"], "e_rot_geo_deg": e["e_rot_geo_deg"], "e_fing_rad": e["e_fing_rad"],
            "d_t_mm": n(dl[0:3]) * 1000, "d_rot_deg": n(dl[3:6]) * RAD2DEG, "d_fing_rad": n(dl[6:51]),
            "dd_t_mm": n(dl[0:3] - d0[0:3]) * 1000, "dd_rot_deg": n(dl[3:6] - d0[3:6]) * RAD2DEG, "dd_fing_rad": n(dl[6:51] - d0[6:51])}


def ra_frames(mano, betas, preds, gts, device):
    pj, _ = ET.mano_fk(mano, np.asarray(preds, np.float32), betas, device)
    gj, _ = ET.mano_fk(mano, np.asarray(gts), betas, device)
    return np.linalg.norm(ET.root_align(pj) - ET.root_align(gj), axis=-1).mean(-1) * 1000


def block_err(s, g):
    """(..., 51) vs (..., 51) -> (..., 5) [t mm, root geodesic deg, fingers rad, index-MCP rad, other fingers rad]"""
    s = np.asarray(s, np.float64)
    g = np.asarray(g, np.float64)
    sh = np.broadcast_shapes(s.shape, g.shape)
    s, g = np.broadcast_to(s, sh), np.broadcast_to(g, sh)
    lead = sh[:-1]
    t = np.linalg.norm(s[..., 0:3] - g[..., 0:3], axis=-1) * 1000.0
    rs = Rotation.from_rotvec(np.array(s[..., 3:6], dtype=np.float64).reshape(-1, 3))
    rg = Rotation.from_rotvec(np.array(g[..., 3:6], dtype=np.float64).reshape(-1, 3))
    rot = ((rs.inv() * rg).magnitude() * RAD2DEG).reshape(lead)
    fing = np.linalg.norm(s[..., 6:51] - g[..., 6:51], axis=-1)
    mcp = np.linalg.norm(s[..., 6:9] - g[..., 6:9], axis=-1)
    oth = np.linalg.norm(s[..., 9:51] - g[..., 9:51], axis=-1)
    return np.stack([t, rot, fing, mcp, oth], axis=-1)


# ============================================================================================ env
def make_env(args):
    cfg, cfg_path, device, mano, root, seqs, model, ckpt = P.load_everything(args)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    env = dict(cfg=cfg, cfg_path=cfg_path, device=device, mano=mano, root=root, seqs=seqs, model=model, ckpt=ckpt,
               cl=P.build_ctx_labels(root, seqs, REPO / args.labels_dir))
    env["rig"] = Rig(model)
    env["tag"] = Path(args.run_dir).name
    env["names"] = [s for s, _ in seqs]
    env["n"] = {s: len(env["cl"][s][0].steps) for s in env["names"]}
    env["fit"] = {s: env["cl"][s][1].fit[:env["n"][s]].astype(bool) for s in env["names"]}
    env["boot"] = Boot(env["n"])
    return env


def meta(args, env, extra=None):
    m = {"script": "tools/xa/probe_xa_conf.py", "argv": sys.argv[1:], "date": time.strftime("%F %T"),
         "run_dir": args.run_dir, "checkpoint_tag": env["tag"], "config": str(env["cfg_path"]),
         "ckpt": str(env["ckpt"]), "ckpt_mtime": time.strftime("%F %T", time.localtime(Path(env["ckpt"]).stat().st_mtime)),
         "device": str(env["device"]), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "gpu_name": torch.cuda.get_device_name(0) if env["device"].type == "cuda" else None, "torch": torch.__version__,
         "max_steps": args.max_steps, "n_packets": env["n"], "bootstrap": {"block": 20, "n": env["boot"].B},
         "protocol": "TF: prev = GT at previous step end (run start for the first step), no noise; stride_sample(2048) events; "
                     "closed loop: rng(0), 50 ms, val_core, noise scale 1, one rng across the sequences"}
    m.update(extra or {})
    return m


def save_cols(out_dir, name, cols):
    p = Path(out_dir) / "cols" / f"{name}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **cols)


# ============================================================================================ TF groups (D, E1)
def d1_variants():
    v = []
    for s in (0.0, 0.1, 0.25, 0.5, 1.5):
        nm = f"D1_s{s:g}" + ("_trunc1" if s > 1 else "")
        v.append(dict(name=nm, spec=lambda st, ctxi, nm=nm, s=s: dict(name=nm, q="scale", s=s, target="all"), s=s, kind="scale", target="all"))
    v.append(dict(name="D1_s1_check", first=60, spec=lambda st, ctxi: dict(name="D1_s1_check", q="scale", s=1.0, target="all"),
                  s=1.0, kind="scale", target="all"))
    return v


def d1b_variants():
    """Path isolation for q <- s q: the net sees scaled q, but (lmfix) the baseline log1p(mass) channel, or
    (resfix) the baseline residual slot, or both. mean / peak are s-invariant, so `both` must give the baseline back."""
    v = []
    for s in (0.25, 0.5):
        for tag, fl in (("lmfix", dict(lm_fix=True)), ("resfix", dict(res_fix=True)), ("bothfix", dict(lm_fix=True, res_fix=True))):
            nm = f"D1b_s{s:g}_{tag}"
            v.append(dict(name=nm, spec=lambda st, ctxi, nm=nm, s=s, fl=fl: dict(name=nm, q="scale", s=s, target="all", **fl),
                          s=s, kind="scale", target="all"))
    return v


def d23_variants():
    v = []
    for tgt, tag in (("near", "D2_near"), ("hit", "D3_hit")):
        for s in (0.0, 0.5):
            nm = f"{tag}_s{s:g}"
            v.append(dict(name=nm, spec=lambda st, ctxi, nm=nm, s=s, tgt=tgt: dict(name=nm, q="scale", s=s, target=tgt),
                          s=s, kind="scale", target=tgt))
    return v


def d4_variants(reps):
    v = []
    for kind, tag in (("keep", "D4_keep5"), ("drop", "D4_drop5")):
        for rep in range(reps):
            nm = f"{tag}_r{rep}"
            v.append(dict(name=nm, spec=lambda st, ctxi, nm=nm, kind=kind, rep=rep: dict(
                name=nm, q=kind, frac=0.05, rng=np.random.default_rng([rep, ctxi, st.j, 1 if kind == "keep" else 2])), kind=kind))
    return v


E1_FLAGS = {"E1_no_dxyz": dict(rel_zero=(7, 8, 9)), "E1_no_z": dict(rel_zero=(10,)), "E1_no_edge": dict(edge_zero=True),
            "E1_no_geom": dict(rel_zero=(7, 8, 9, 10), edge_zero=True), "E1_z_median": dict(z_median=True)}


def e1_variants():
    return [dict(name=nm, spec=lambda st, ctxi, nm=nm, fl=fl: dict(name=nm, **fl), kind="geom") for nm, fl in E1_FLAGS.items()]


def verify_variant(v, sp, pred, pred0, cap, cap0, rig):
    """Per-packet assertions that the hook did what it claims."""
    nm = sp["name"]
    if "mask" not in cap:                                  # empty packet: the model never reached xyz
        return
    if sp.get("q"):
        q0, q1, mask = cap["q0"], cap["q1"], cap["mask"].float()
        # residual accounting: r rises by exactly sum(q0 - q1); the vertex mass falls with the weights
        dr = float(cap["r"].sum() - cap0["r"].sum())
        dq = float(q0.sum() - q1.sum())
        assert abs(dr - dq) < 5e-3 + 1e-3 * abs(dq), (nm, dr, dq)
        if sp["q"] == "scale" and sp["s"] == 1.0:
            assert torch.equal(pred, pred0), f"{nm}: s=1 hook changed the output"
            assert torch.equal(q0, q1)
        if sp["q"] == "scale" and sp["s"] == 0.0 and sp.get("target", "all") == "all":
            assert float(cap["w"].abs().sum()) == 0.0 and int((cap["w"] > 0).sum()) == 0, f"{nm}: s=0 left vertex weights"
            assert torch.equal(cap["r"], mask), f"{nm}: s=0 did not put every live event in the residual slot"
            assert float(cap["O"].sum()) == 0.0 and float(cap["H"].abs().sum()) == 0.0
    rig.cnt[f"{nm}/packets_verified"] += 1
    rig.cnt[f"{nm}/pred_changed"] += int(not torch.equal(pred, pred0))
    rig.cnt[f"{nm}/pred_maxabs_diff_max"] = max(rig.cnt[f"{nm}/pred_maxabs_diff_max"], float((pred - pred0).abs().max()))


def run_tf_group(args, env, group, variants, out_dir):
    model, rig, device, mano = env["model"], env["rig"], env["device"], env["mano"]
    t0 = time.time()
    names = ["base"] + [v["name"] for v in variants]
    cols = {nm: {s: defaultdict(list) for s in env["names"]} for nm in names}
    preds = {nm: {s: [] for s in env["names"]} for nm in names}
    gts, betas_of = {}, {}
    rig.cnt.clear()
    for si, (seq, d) in enumerate(env["seqs"]):
        ctx, lab = env["cl"][seq]
        P.set_context(model, ctx, device)
        betas_of[seq] = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        gts[seq] = np.stack([ctx.pos51[s.end] for s in ctx.steps])
        ts = time.time()
        for st, ev5, prev, gt, batch in P.tf_packets(ctx, device):
            rig.spec = None
            pred0, cap0 = rig.run(batch)
            p0 = pred0.double().cpu().numpy()[0]
            r = err_record(p0, p0, prev, gt)
            r.update(feature_metrics(cap0, None))
            for k, val in r.items():
                cols["base"][seq][k].append(val)
            preds["base"][seq].append(pred0.cpu().numpy()[0])
            for v in variants:
                if "first" in v and st.j >= v["first"]:
                    continue
                sp = v["spec"](st, si)
                rig.spec = sp
                rig.ref = cap0
                pred, cap = rig.run(batch)
                rig.spec = None
                rig.ref = None
                if st.n_ev:
                    verify_variant(v, sp, pred, pred0, cap, cap0, rig)
                p = pred.double().cpu().numpy()[0]
                r = err_record(p, p0, prev, gt)
                r.update(feature_metrics(cap, cap0))
                for k, val in r.items():
                    cols[v["name"]][seq][k].append(val)
                preds[v["name"]][seq].append(pred.cpu().numpy()[0])
            if (st.j + 1) % 200 == 0:
                log(f"TF {group} {seq}: {st.j + 1}/{len(ctx.steps)} ({time.time() - ts:.0f}s)")
        for nm in names:
            n_have = len(preds[nm][seq])
            if n_have == len(ctx.steps):
                cols[nm][seq]["ra_mm"] = ra_frames(mano, betas_of[seq], np.stack(preds[nm][seq]), gts[seq], device)
        log(f"TF {group} {seq} done ({time.time() - ts:.0f}s)")
    return names, cols, preds, rig.cnt


def summarize_tf(env, names, cols, metrics, rel_metrics, extra_seq_arrays=None):
    """Per variant x group: pooled mean + CI for every metric; relative change vs 'base' for `rel_metrics`."""
    boot, seqs = env["boot"], env["names"]
    grp = groups_for(seqs, env["fit"])
    out = {}
    for nm in names:
        out[nm] = {}
        for gname, (gs, gsel) in grp.items():
            e = {}
            for mt in metrics:
                if not all(mt in cols[nm][s] and len(cols[nm][s][mt]) == env["n"][s] for s in gs):
                    continue
                x = {s: np.asarray(cols[nm][s][mt], np.float64) for s in gs}
                e[mt] = boot.est(x, gs, gsel).d()
            if nm != "base":
                rel = {}
                for mt in rel_metrics:
                    if not all(mt in cols[nm][s] and len(cols[nm][s][mt]) == env["n"][s] and mt in cols["base"][s] for s in gs):
                        continue
                    xv = {s: np.asarray(cols[nm][s][mt], np.float64) for s in gs}
                    x0 = {s: np.asarray(cols["base"][s][mt], np.float64) for s in gs}
                    # variants that run on a prefix of the packets only (checks) are skipped above by the length test
                    ev, e0 = boot.est(xv, gs, gsel), boot.est(x0, gs, gsel)
                    rel[mt] = {"rel_change": ratio(ev, e0).d(), "abs_change": (ev - e0).d()}
                e["_rel_vs_base"] = rel
            out[nm][gname] = e
    return out


D_METRICS = ("ra_mm", "e_t_mm", "e_rot_deg", "e_rot_geo_deg", "e_fing_rad", "d_t_mm", "d_rot_deg", "d_fing_rad",
             "dd_t_mm", "dd_rot_deg", "dd_fing_rad") + FEAT_KEYS
D_REL = ("ra_mm", "e_t_mm", "e_rot_deg", "e_fing_rad", "d_t_mm", "d_rot_deg", "d_fing_rad", "H_fro", "H_obs_norm",
         "verts_direct", "verts_prop", "resid_mass", "n_rec", "mass_sum", "r_sum")


def part_tf(args, env, part, out_dir):
    if part == "tfD1":
        group, variants, fname = "D1", d1_variants(), f"D_tf_D1_{env['tag']}"
    elif part == "tfD1b":
        group, variants, fname = "D1b", d1b_variants(), f"D_tf_D1b_{env['tag']}"
    elif part == "tfD23":
        group, variants, fname = "D23", d23_variants(), f"D_tf_D23_{env['tag']}"
    elif part == "tfD4":
        group, variants, fname = "D4", d4_variants(args.reps), f"D_tf_D4_{env['tag']}"
    else:
        group, variants, fname = "E1", e1_variants(), f"E_tf_E1_{env['tag']}"
    t0 = time.time()
    names, cols, preds, cnt = run_tf_group(args, env, group, variants, out_dir)
    extra = {}
    if group == "E1":
        extra = e1_flag_check(args, env, variants)
    if group == "D1":
        extra = d1_zero_check(args, env)
    if env["tag"] == "xa_c1_s3407" and not args.max_steps:
        p2b = json.loads((REPO / "outputs/xa/step2/p2b_tf_corr.json").read_text())["sequences"]
        extra["base_vs_P2b_columns_maxabs"] = {
            s: {k: float(np.abs(np.asarray(cols["base"][s][k]) - np.asarray(p2b[s]["columns"][k])).max()) for k in ("e_rot_deg", "e_fing_rad", "e_t_mm")}
            for s in env["names"]}
    summ = summarize_tf(env, names, cols, D_METRICS, D_REL)
    rig = env["rig"]
    out = {"meta": meta(args, env, {"seconds": time.time() - t0, "group": group, "gpu_max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20
                                    if env["device"].type == "cuda" else None}),
           "variants": {v["name"]: {k: (val if not callable(val) else None) for k, val in v.items() if k in ("s", "kind", "target", "first")}
                        for v in variants},
           "hook_counters": {nm: rig.counters(nm) for nm in names if nm != "base"},
           "checks": extra, "summary": summ}
    P.dump(out_dir / f"{fname}.json", out)
    flat = {f"{nm}|{s}|{k}": np.asarray(v) for nm in names for s in env["names"] for k, v in cols[nm][s].items()}
    save_cols(out_dir, fname, flat)
    return out


def d1_zero_check(args, env):
    """s=0 via the q pre-hook == the P2 wrapper `associate_rays` that zeroes q of every live event (bitwise), first 40 packets."""
    model, rig, device = env["model"], env["rig"], env["device"]
    seq = env["names"][-1]
    ctx, _ = env["cl"][seq]
    P.set_context(model, ctx, device)
    patch = P.AssocPatch()
    n_eq = n = 0
    with patch:
        for st, ev5, prev, gt, batch in P.tf_packets(ctx, device):
            if st.j >= 40:
                break
            if not st.n_ev:
                continue
            rig.spec = dict(name="zero_check", q="scale", s=0.0, target="all")
            a, _ = rig.run(batch)
            rig.spec = None
            idx = P.sampled_index(st.n_ev)
            patch.sel = torch.arange(len(idx), device=device)
            b, _ = rig.run(batch)
            patch.sel = None
            n += 1
            n_eq += int(torch.equal(a, b))
    assert patch.calls > 0 and XM.associate_rays is patch.orig
    assert n_eq == n, f"s=0 hook differs from the associate_rays wrapper in {n - n_eq}/{n} packets"
    return {"s0_hook_vs_associate_wrapper_bitwise_equal": f"{n_eq}/{n}", "wrapper_calls": patch.calls, "seq": seq}


def e1_flag_check(args, env, variants):
    """no_geom hooks == `model.xyz.geom = False` (the C0 input) bitwise, first 40 packets; no_edge/no_dxyz/no_z differ from base."""
    model, rig, device = env["model"], env["rig"], env["device"]
    seq = env["names"][-1]
    ctx, _ = env["cl"][seq]
    P.set_context(model, ctx, device)
    n_eq = n = 0
    for st, ev5, prev, gt, batch in P.tf_packets(ctx, device):
        if st.j >= 40:
            break
        if not st.n_ev:
            continue
        rig.spec = dict(name="geom_check", **E1_FLAGS["E1_no_geom"])
        a, _ = rig.run(batch)
        rig.spec = None
        model.xyz.geom = False
        b, _ = rig.run(batch)
        model.xyz.geom = True
        n += 1
        n_eq += int(torch.equal(a, b))
    assert n_eq == n, f"no_geom hooks differ from xyz.geom=False in {n - n_eq}/{n} packets"
    return {"no_geom_hooks_vs_geom_False_bitwise_equal": f"{n_eq}/{n}", "seq": seq, "counters": rig.counters("geom_check")}


# ============================================================================================ closed loop
class CollectHook:
    def __init__(self):
        self.preds = []

    def __call__(self, kind, prev_t, pred):
        if kind == "step":
            self.preds.append(pred.cpu().numpy()[0])
        return pred


def run_closed_loop(env, spec, name):
    model, rig, mano, device, cfg, root = env["model"], env["rig"], env["mano"], env["device"], env["cfg"], env["root"]
    rig.spec = spec
    rig.stats_log = []
    rng = np.random.default_rng(0)                       # one rng across the sequences, as select_checkpoint / P2-a
    res = {"per_seq": {}, "per_step_ra": {}, "route": {}}
    for seq, d in env["seqs"]:
        ctx, _ = env["cl"][seq]
        n0 = len(rig.stats_log)
        hook = CollectHook()
        r = ET.track_sequence(model, mano, cfg, root, d, seq, ST, device, rng, 1.0, 1.0, None, state_hook=hook)
        assert len(hook.preds) == r["n_frames"] == len(ctx.steps)
        betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        gts = np.stack([ctx.pos51[s.end] for s in ctx.steps])
        ra = ra_frames(mano, betas, np.stack(hook.preds), gts, device)
        assert abs(float(ra.mean()) - r["mpjpe_ra_mm"]) < 1e-9 * max(1.0, abs(r["mpjpe_ra_mm"])) + 1e-9, (ra.mean(), r["mpjpe_ra_mm"])
        res["per_seq"][seq] = {"ra": r["mpjpe_ra_mm"], "mpvpe_ra_mm": r["mpvpe_ra_mm"], "mpjpe_abs_mm": r["mpjpe_abs_mm"], "n": r["n_frames"]}
        res["per_step_ra"][seq] = ra
        st_ = rig.stats_log[n0:]
        res["route"][seq] = {k: float(np.mean([x[k] for x in st_])) for k in st_[0]} if st_ else {}
        res["preds_" + seq] = np.stack(hook.preds)
    n_tot = sum(v["n"] for v in res["per_seq"].values())
    res["overall"] = sum(v["ra"] * v["n"] for v in res["per_seq"].values()) / n_tot
    rig.spec = None
    rig.stats_log = None
    log(f"CL {env['tag']} {name}: overall {res['overall']:.4f} " + " ".join(f"{s}={v['ra']:.4f}" for s, v in res["per_seq"].items()))
    return res


def ref_check(args, env, base):
    """Baseline (hooks installed but inert) vs the P2 record (xa_c1_s3407: exact) or the selection JSON (others: 0.05 mm gate)."""
    out = {"hooks_inert_spec": None}
    if args.max_steps:
        return {"skipped": "self-test"}
    if env["tag"] == "xa_c1_s3407":
        ref = json.loads((REPO / P2_JSON).read_text())["ra"]
        exact = (base["overall"] == ref["overall"]) and all(base["per_seq"][s]["ra"] == ref["per_seq"][s] for s in ref["per_seq"])
        out.update(reference=P2_JSON, ref=ref, this={"overall": base["overall"], "per_seq": {s: v["ra"] for s, v in base["per_seq"].items()}},
                   bitwise_equal=bool(exact),
                   task_numbers={"overall": 22.9343, "global": 14.0670, "local": 33.1421})
        if not exact and not args.no_assert_ref:
            raise AssertionError(f"hooks-off closed loop differs from the P2 record: {out}")
    else:
        ref = P.find_ra_reference(REPO / args.run_dir)
        d = {s: abs(base["per_seq"][s]["ra"] - ref["per_seq"][s]) for s in base["per_seq"]} if ref else {}
        out.update(reference=ref["source"] if ref else None, ref=ref, abs_diff_per_seq=d,
                   abs_diff_overall=abs(base["overall"] - ref["overall"]) if ref else None,
                   within_0p05mm=bool(ref and all(v < P.RA_REPRO_TOL_MM for v in d.values()) and abs(base["overall"] - ref["overall"]) < P.RA_REPRO_TOL_MM),
                   bitwise_equal=bool(ref and base["overall"] == ref["overall"] and all(base["per_seq"][s]["ra"] == ref["per_seq"][s] for s in ref["per_seq"])))
    return out


def part_cl(args, env, part, out_dir):
    rig = env["rig"]
    t0 = time.time()
    if part == "clD":
        runs = [("base", None)] + [(f"D5_s{s:g}", dict(name=f"D5_s{s:g}", q="scale", s=s, target="all")) for s in (1.0, 0.0, 0.25, 0.5)]
        runs += [(f"D5x_{tg}_s0", dict(name=f"D5x_{tg}_s0", q="scale", s=0.0, target=tg)) for tg in ("near", "hit")] if args.cl_extra else []
        fname = f"D_cl_{env['tag']}"
    else:
        runs = [("base", None)] + [(nm, dict(name=nm, **fl)) for nm, fl in E1_FLAGS.items()]
        fname = f"E_cl_E1_{env['tag']}"
    res = {}
    rig.cnt.clear()
    for nm, sp in runs:
        res[nm] = run_closed_loop(env, sp, nm)
        if nm == "base":
            chk = ref_check(args, env, res[nm])
            log("reference check:", {k: v for k, v in chk.items() if k not in ("ref",)})
    boot, seqs = env["boot"], env["names"]
    summ = {}
    for nm in res:
        summ[nm] = {}
        for gname, (gs, gsel) in groups_for(seqs, env["fit"]).items():
            x = {s: res[nm]["per_step_ra"][s] for s in gs}
            x0 = {s: res["base"]["per_step_ra"][s] for s in gs}
            e = {"ra_mm": boot.est(x, gs, gsel).d()}
            if nm != "base":
                e["ra_rel_change"] = ratio(boot.est(x, gs, gsel), boot.est(x0, gs, gsel)).d()
                e["ra_abs_change"] = (boot.est(x, gs, gsel) - boot.est(x0, gs, gsel)).d()
            summ[nm][gname] = e
        summ[nm]["route_mean"] = res[nm]["route"]
        summ[nm]["track_sequence_ra"] = {"overall": res[nm]["overall"], "per_seq": {s: v["ra"] for s, v in res[nm]["per_seq"].items()}}
    out = {"meta": meta(args, env, {"seconds": time.time() - t0, "group": part}), "reference_check": chk,
           "hook_counters": {nm: rig.counters(sp["name"]) for nm, sp in runs if sp}, "summary": summ}
    P.dump(out_dir / f"{fname}.json", out)
    save_cols(out_dir, fname, {f"{nm}|{s}|ra": res[nm]["per_step_ra"][s] for nm in res for s in seqs})
    return out


# ============================================================================================ E2
def pert_list():
    L = []
    for ax, a in enumerate("xyz"):
        for sg in (1, -1):
            L.append(dict(name=f"t{a}{'+' if sg > 0 else '-'}", kind="t", ax=ax, sg=sg, target="t", fam=f"t{a}"))
    for ax, a in enumerate("xyz"):
        for sg in (1, -1):
            L.append(dict(name=f"r{a}{'+' if sg > 0 else '-'}", kind="r", ax=ax, sg=sg, target="rot", fam=f"r{a}"))
    L.append(dict(name="finger_index_mcp", kind="fmcp", target="mcp", fam="f_mcp"))
    L.append(dict(name="fingers_all", kind="fall", target="fing", fam="f_all"))
    return L


def apply_pert(p, s):
    s = np.asarray(s, np.float64).copy()
    k = p["kind"]
    if k == "t":
        s[p["ax"]] += p["sg"] * 0.020
    elif k == "r":
        e = np.zeros(3)
        e[p["ax"]] = 1.0
        s[3:6] = (Rotation.from_rotvec(p["sg"] * math.radians(10.0) * e) * Rotation.from_rotvec(s[3:6])).as_rotvec()
    elif k == "fmcp":
        s[6:9] += 0.3
    else:
        s[6:51] += 0.1
    return s.astype(np.float32)


def make_batch(ev5, prevs, betas, K, device):
    B, n = prevs.shape[0], ev5.shape[0]
    if n:
        ev = np.tile(ev5, (B, 1))
        ev[:, 0] = np.repeat(np.arange(B, dtype=np.float32), n)
    else:
        ev = np.zeros((0, ev5.shape[1]), np.float32)
    z = lambda dt: torch.zeros(B, dtype=dt, device=device)   # noqa: E731
    return EventPacketBatch(
        events=torch.from_numpy(ev).to(device), ptr=(torch.arange(B + 1, dtype=torch.int64) * n).to(device),
        sequence_id=z(torch.int64), t_start_us=z(torch.int64), t_end_us=torch.full((B,), ST * 1000, dtype=torch.int64, device=device),
        delta_t_s=torch.full((B,), ST * 1e-3, dtype=torch.float32, device=device),
        is_sequence_start=z(torch.bool), is_sequence_end=z(torch.bool), target=torch.zeros(B, 51, device=device),
        prev_state=torch.from_numpy(np.ascontiguousarray(prevs, dtype=np.float32)).to(device),
        betas=betas.expand(B, -1).contiguous(), camera_K=K.expand(B, -1, -1).contiguous(), lnes=None)


def part_e2(args, env, out_dir):
    model, rig, device = env["model"], env["rig"], env["device"]
    perts = pert_list()
    NP = len(perts) + 1
    KMAX = args.kmax
    stride = args.stride
    t0 = time.time()
    data = {}
    chk = {"b1_vs_batch_maxabs": 0.0, "b1_checked": 0}
    for seq, d in env["seqs"]:
        ctx, lab = env["cl"][seq]
        P.set_context(model, ctx, device)
        betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(ctx.aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        n = len(ctx.steps)
        runid = np.array([s.run_id for s in ctx.steps])
        err0 = np.zeros((n, NP, 5))
        errb = np.zeros((n, NP, 5))
        err1 = np.zeros((n, NP, 5))
        dist1 = np.zeros((n, NP - 1, 5))
        roll_j = [j for j in range(n) if stride and j % stride == 0]
        errk = np.full((len(roll_j), KMAX, NP, 5), np.nan)
        distk = np.full((len(roll_j), KMAX, NP - 1, 5), np.nan)
        rpos = {j: i for i, j in enumerate(roll_j)}
        ts = time.time()
        for j, st in enumerate(ctx.steps):
            gprev = ctx.pos51[st.prev_idx].astype(np.float32)
            S = np.stack([gprev] + [apply_pert(p, gprev) for p in perts])
            err0[j] = block_err(S, gprev)
            avail = 1
            if j in rpos:
                while avail < KMAX and j + avail < n and runid[j + avail] == st.run_id:
                    avail += 1
            for i in range(avail):
                sti = ctx.steps[j + i]
                ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, sti.end, ST)
                batch = make_batch(ev5, S, betas, K, device)
                with torch.no_grad():
                    pred = model.forward_packet(batch).cpu().numpy()
                gt_i = ctx.pos51[sti.end]
                e = block_err(pred, gt_i)
                ds = block_err(pred[1:], pred[0])
                if i == 0:
                    errb[j] = block_err(S, gt_i)
                    err1[j] = e
                    dist1[j] = ds
                    if j < 25 and st.n_ev:                      # batch-vs-single check on the control and one perturbed sample
                        for sm in (0, 5):
                            b1 = ET.make_eval_packet(ev5, torch.from_numpy(S[sm]).view(1, -1).to(device), betas, K, ST, device)
                            with torch.no_grad():
                                q1 = model.forward_packet(b1).cpu().numpy()[0]
                            chk["b1_vs_batch_maxabs"] = max(chk["b1_vs_batch_maxabs"], float(np.abs(q1 - pred[sm]).max()))
                            chk["b1_checked"] += 1
                if j in rpos:
                    errk[rpos[j], i] = e
                    distk[rpos[j], i] = ds
                S = pred
            if (j + 1) % 200 == 0:
                log(f"E2 {seq}: {j + 1}/{n} ({time.time() - ts:.0f}s) max mem {torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else 0:.0f} MB")
        data[seq] = dict(err0=err0, errb=errb, err1=err1, dist1=dist1, roll_j=np.array(roll_j), errk=errk, distk=distk)
        log(f"E2 {seq} done ({time.time() - ts:.0f}s)")
    out = {"meta": meta(args, env, {"seconds": time.time() - t0, "group": "E2", "stride": stride, "kmax": KMAX,
                                    "gpu_max_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None,
                                    "perturbations": [{k: v for k, v in p.items()} for p in perts]}),
           "checks": {**chk, "note": "batch of 15 (control + 14 perturbations, same events) vs single-sample forward"},
           "summary": summarize_e2(args, env, data, perts)}
    fname = f"E_e2_{env['tag']}"
    P.dump(out_dir / f"{fname}.json", out)
    save_cols(out_dir, fname, {f"{s}|{k}": v for s, dd in data.items() for k, v in dd.items()})
    return out


def summarize_e2(args, env, data, perts):
    seqs = env["names"]
    stride = max(1, args.stride)
    boot_all = env["boot"]
    boot_roll = Boot({s: len(data[s]["roll_j"]) for s in seqs}, L=max(1, int(math.ceil(20 / stride))), seed=11)
    fit_all = env["fit"]
    fit_roll = {s: fit_all[s][data[s]["roll_j"]] if len(data[s]["roll_j"]) else np.zeros(0, bool) for s in seqs}
    fams = {}
    for i, p in enumerate(perts):
        fams.setdefault(p["fam"], dict(target=p["target"], members=[], names=[]))["members"].append(i + 1)
        fams[p["fam"]]["names"].append(p["name"])
    others = {"t": ["rot_deg", "fing_rad"], "rot": ["t_mm", "fing_rad"], "mcp": ["t_mm", "rot_deg", "fing_other_rad"], "fing": ["t_mm", "rot_deg"]}
    bidx = {n: i for i, n in enumerate(BLK)}
    tfield = {"t": "t_mm", "rot": "rot_deg", "fing": "fing_rad", "mcp": "mcp_rad"}
    ks = [k for k in (1, 3, 5, 10) if k <= args.kmax]
    out = {"families": {f: v for f, v in fams.items()}, "block_names": BLK, "k_list": ks, "groups": {}}
    gall = groups_for(seqs, fit_all)
    groll = groups_for(seqs, fit_roll)

    for gname in gall:
        gs, gsel = gall[gname]
        gr, grsel = groll[gname]
        G = {}
        for fam, F in fams.items():
            tb, mem = F["target"], F["members"]
            tbn = tfield[tb]
            m0 = [0]
            rec = {"target_block": tbn, "members": F["names"]}
            # ---- TF (k = 1), all packets: arrays (n, members)
            def avg(key, ids, blk, shift=0):
                return {s: data[s][key][:, [i - shift for i in ids], bidx[blk]].mean(1) for s in gs}
            after = boot_all.est(avg("err1", mem, tbn), gs, gsel)
            before = boot_all.est(avg("errb", mem, tbn), gs, gsel)
            c_after = boot_all.est({s: data[s]["err1"][:, 0, bidx[tbn]] for s in gs}, gs, gsel)
            c_before = boot_all.est({s: data[s]["errb"][:, 0, bidx[tbn]] for s in gs}, gs, gsel)
            dist = boot_all.est(avg("dist1", mem, tbn, shift=1), gs, gsel)
            delta = boot_all.est(avg("err0", mem, tbn), gs, gsel)
            one_minus = lambda a, b: Est(1 - a.point / b.point, 1 - a.boot / b.boot, a.n)       # noqa: E731
            rec["tf"] = {"err_before": before.d(), "err_after": after.d(), "ctrl_err_before": c_before.d(), "ctrl_err_after": c_after.d(),
                         "perturbation_size": delta.d(), "pred_distance_to_ctrl": dist.d(),
                         "recovery_ratio": one_minus(after, before).d(),
                         "recovery_ratio_ctrl_same_def": one_minus(c_after, c_before).d(),
                         "recovery_ratio_excess": one_minus(dist, delta).d()}
            pm = {}
            for s_ in gs:
                a_ = avg("err1", mem, tbn)[s_]
                b_ = avg("errb", mem, tbn)[s_]
                with np.errstate(divide="ignore", invalid="ignore"):
                    pm[s_] = np.where(b_ > 1e-9, 1 - a_ / b_, np.nan)
            rec["tf"]["recovery_ratio_per_packet_mean"] = boot_all.est(pm, gs, gsel).d()
            rec["tf"]["crosstalk"] = {o: (boot_all.est(avg("err1", mem, o), gs, gsel) - boot_all.est({s: data[s]["err1"][:, 0, bidx[o]] for s in gs}, gs, gsel)).d()
                                      for o in others[tb]}
            rec["tf"]["crosstalk_before"] = {o: (boot_all.est(avg("errb", mem, o), gs, gsel) - boot_all.est({s: data[s]["errb"][:, 0, bidx[o]] for s in gs}, gs, gsel)).d()
                                             for o in others[tb]}
            # ---- k-step, stride subsample
            rk = {}
            if args.stride:
                for k in ks:
                    def avgk(key, ids, blk, shift=0):
                        return {s: data[s][key][:, k - 1][:, [i - shift for i in ids], bidx[blk]].mean(1) for s in gr}
                    ek_arr = avgk("errk", mem, tbn)
                    ok = {s: np.isfinite(ek_arr[s]) for s in gr}                 # starts whose run still has k steps
                    mk = lambda a: {s: np.where(ok[s], a[s], np.nan) for s in gr}     # noqa: E731
                    e0 = boot_roll.est(mk({s: data[s]["err0"][data[s]["roll_j"]][:, mem, bidx[tbn]].mean(1) for s in gr}), gr, grsel)
                    ek = boot_roll.est(ek_arr, gr, grsel)
                    ck = boot_roll.est(mk({s: data[s]["errk"][:, k - 1, 0, bidx[tbn]] for s in gr}), gr, grsel)
                    dk = boot_roll.est(avgk("distk", mem, tbn, shift=1), gr, grsel)
                    rk[str(k)] = {"err_k": ek.d(), "err_0": e0.d(), "err_k_over_err_0": ratio(ek, e0, minus_one=False).d(),
                                  "ctrl_err_k_over_err_0": ratio(ck, e0, minus_one=False).d(),
                                  "residual_perturbation_ratio": ratio(dk, e0, minus_one=False).d(),
                                  "crosstalk": {o: (boot_roll.est(avgk("errk", mem, o), gr, grsel) -
                                                    boot_roll.est({s: data[s]["errk"][:, k - 1, 0, bidx[o]] for s in gr}, gr, grsel)).d()
                                                for o in others[tb]}}
            rec["k_step"] = rk
            G[fam] = rec
        # per-member (signed) detail for the translation / rotation families, TF only
        det = {}
        for i, p in enumerate(perts):
            tbn = tfield[p["target"]]
            a_ = boot_all.est({s: data[s]["err1"][:, i + 1, bidx[tbn]] for s in gs}, gs, gsel)
            b_ = boot_all.est({s: data[s]["errb"][:, i + 1, bidx[tbn]] for s in gs}, gs, gsel)
            det[p["name"]] = {"err_before": b_.d(), "err_after": a_.d(), "recovery_ratio": Est(1 - a_.point / b_.point, 1 - a_.boot / b_.boot, a_.n).d()}
        G["_signed_detail_tf"] = det
        ctrl = {}
        for blk in BLK:
            ctrl[blk] = {"err_before_ctrl": boot_all.est({s: data[s]["errb"][:, 0, bidx[blk]] for s in gs}, gs, gsel).d(),
                         "err_after_ctrl": boot_all.est({s: data[s]["err1"][:, 0, bidx[blk]] for s in gs}, gs, gsel).d()}
        G["_control_tf"] = ctrl
        out["groups"][gname] = G
    return out


# ============================================================================================ E3
def e3_keys(nm):
    ks = ["avail", "maxabs_pred_minus_prev", "n_events"]
    for blk in ("t", "rot", "fing"):
        ks += [f"d_{blk}", f"d_ratio_{blk}", f"dd_{blk}", f"cos0_{blk}", f"cosgt_{blk}"]
    if nm == "a_shift4px":
        ks += ["cos_shift_x", "shift_mm", "shift_over_expected", "shift_x_over_expected", "expected_mm"]
    return ks


def part_e3(args, env, out_dir):
    model, rig, device, mano = env["model"], env["rig"], env["device"], env["mano"]
    t0 = time.time()
    names = ["a_shift4px", "b_other_packet", "c_empty"]
    cols = {nm: {s: defaultdict(list) for s in env["names"]} for nm in names}
    cols["base"] = {s: defaultdict(list) for s in env["names"]}
    n = np.linalg.norm

    def cos(a, b):
        na, nb = n(a), n(b)
        return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else NAN

    xhat = np.array([1.0, 0.0, 0.0])
    sl = {"t": slice(0, 3), "rot": slice(3, 6), "fing": slice(6, 51)}
    unit = {"t": 1000.0, "rot": RAD2DEG, "fing": 1.0}
    for seq, d in env["seqs"]:
        ctx, lab = env["cl"][seq]
        P.set_context(model, ctx, device)
        betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(ctx.aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        fx_r = float(K[0, 0, 0]) * float(model.render_scale)
        prevs = np.stack([ctx.pos51[s.prev_idx] for s in ctx.steps]).astype(np.float32)
        jprev, _ = ET.mano_fk(mano, prevs, betas, device)
        zroot = jprev[:, 0, 2]
        runid = np.array([s.run_id for s in ctx.steps])
        n_steps = len(ctx.steps)
        ts = time.time()
        for st, ev5, prev, gt, batch in P.tf_packets(ctx, device):
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            with torch.no_grad():
                p0 = model.forward_packet(batch).double().cpu().numpy()[0]
            d0 = p0 - prev
            dg = gt.astype(np.float64) - prev
            variants = {}
            ea = ev5.copy()
            ea[:, 1] += 4.0
            variants["a_shift4px"] = ea
            j2 = st.j + 20 if (st.j + 20 < n_steps and runid[st.j + 20] == st.run_id) else \
                (st.j - 20 if (st.j - 20 >= 0 and runid[st.j - 20] == st.run_id) else None)
            variants["b_other_packet"] = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, ctx.steps[j2].end, ST) if j2 is not None else None
            variants["c_empty"] = np.zeros((0, ev5.shape[1]), np.float32)
            for blk in sl:
                cols["base"][seq][f"d_{blk}"].append(n(d0[sl[blk]]) * unit[blk])
                cols["base"][seq][f"cosgt_{blk}"].append(cos(d0[sl[blk]], dg[sl[blk]]))
            for nm, evv in variants.items():
                r = {}
                if evv is None:                                 # packet (b) not available: NaN for every key, so lengths stay n
                    r = {k: NAN for k in e3_keys(nm)}
                    r["avail"] = 0.0
                    pv = None
                else:
                    with torch.no_grad():
                        pv = model.forward_packet(ET.make_eval_packet(evv, prev_t, betas, K, ST, device)).double().cpu().numpy()[0]
                    dv = pv - prev
                    r["avail"] = 1.0
                    for blk in sl:
                        a, b = dv[sl[blk]], d0[sl[blk]]
                        r[f"d_{blk}"] = n(a) * unit[blk]
                        r[f"d_ratio_{blk}"] = n(a) / n(b) if n(b) > 1e-12 else NAN
                        r[f"dd_{blk}"] = n(a - b) * unit[blk]
                        r[f"cos0_{blk}"] = cos(a, b)
                        r[f"cosgt_{blk}"] = cos(a, dg[sl[blk]])
                    r["maxabs_pred_minus_prev"] = float(np.abs(dv).max())
                    r["n_events"] = float(len(evv))
                    if nm == "a_shift4px":
                        shift = dv[0:3] - d0[0:3]
                        r["cos_shift_x"] = cos(shift, xhat)
                        exp_mm = 4.0 / fx_r * float(zroot[st.j]) * 1000.0
                        r["shift_mm"] = n(shift) * 1000.0
                        r["shift_over_expected"] = n(shift) * 1000.0 / exp_mm
                        r["shift_x_over_expected"] = float(shift[0]) * 1000.0 / exp_mm
                        r["expected_mm"] = exp_mm
                for k, v in r.items():
                    cols[nm][seq][k].append(v)
            if (st.j + 1) % 300 == 0:
                log(f"E3 {seq}: {st.j + 1}/{n_steps} ({time.time() - ts:.0f}s)")
    boot, seqs = env["boot"], env["names"]
    grp = groups_for(seqs, env["fit"])
    summ = {}
    for nm in ["base"] + names:
        summ[nm] = {}
        keys = sorted({k for s in seqs for k in cols[nm][s]})
        for gname, (gs, gsel) in grp.items():
            e = {}
            for k in keys:
                x = {s: np.asarray(cols[nm][s][k], np.float64) for s in gs}
                if all(len(x[s]) == env["n"][s] for s in gs):
                    e[k] = boot.est(x, gs, gsel).d()
            summ[nm][gname] = e
    out = {"meta": meta(args, env, {"seconds": time.time() - t0, "group": "E3",
                                    "definitions": {"d_*": "|Delta| per block (t mm, rot deg, fing rad), Delta = pred - prev",
                                                    "d_ratio": "|Delta_v| / |Delta_0|", "dd": "|Delta_v - Delta_0|",
                                                    "cos0": "cos(Delta_v, Delta_0) per block", "cosgt": "cos(Delta_v, GT update)",
                                                    "cos_shift_x": "cos(Delta_a[t] - Delta_0[t], +x): the truth direction of +4 px",
                                                    "expected_mm": "4 px / fx_render * z_wrist(prev) in mm"}}),
           "summary": summ}
    fname = f"E_e3_{env['tag']}"
    P.dump(out_dir / f"{fname}.json", out)
    save_cols(out_dir, fname, {f"{nm}|{s}|{k}": np.asarray(v) for nm in cols for s in seqs for k, v in cols[nm][s].items()})
    return out



# ============================================================================================ report
def _ld(out_dir, name):
    p = Path(out_dir) / name
    return json.loads(p.read_text()) if p.exists() else None


def fe(d, nd=2, sci=False, pct=False):
    """{'mean', 'ci95'} -> 'm [lo, hi]'"""
    if d is None or d.get("mean") is None:
        return "n/a"
    m, (lo, hi) = d["mean"], d["ci95"]
    sc = 100.0 if pct else 1.0
    fmt = (lambda v: "n/a" if v is None else f"{sc * v:.{nd}e}") if sci else (lambda v: "n/a" if v is None else f"{sc * v:.{nd}f}")
    return f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]" + ("%" if pct else "")


def fp(d, nd=1):
    """relative change as signed percent with CI"""
    if d is None or d.get("mean") is None:
        return "n/a"
    m, (lo, hi) = d["mean"], d["ci95"]
    g = lambda v: "n/a" if v is None else f"{100 * v:+.{nd}f}"        # noqa: E731
    return f"{g(m)}% [{g(lo)}, {g(hi)}]"


def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


TAGS = ("xa_c1_s3407", "xa_c1_s3408", "xa_c1r_s3407")
SEQN = {"overall": "全部(global+local)", "zgz_global": "zgz_global", "zgz_local": "zgz_local",
        "local_fit_true": "local & fingertip_in_front=真", "local_fit_false": "local & fingertip_in_front=假"}


def judge_j1(D1):
    """pre-registered J1 on the overall group of a D_tf_D1 json"""
    M = ("d_t_mm", "d_rot_deg", "d_fing_rad", "ra_mm", "e_t_mm", "e_rot_deg", "e_fing_rad")
    S = D1["summary"]

    def rc(nm, k):
        return S[nm]["overall"]["_rel_vs_base"][k]["rel_change"]["mean"]

    r25 = {k: abs(rc("D1_s0.25", k)) for k in M}
    r0 = {k: abs(rc("D1_s0", k)) for k in M}
    A = all(v < 0.05 for v in r25.values())
    kmax25 = max(r25, key=r25.get)
    B = any(r0[k] >= 0.05 and r0[k] >= 3 * r25[k] for k in M)
    if A and B:
        txt = "A 成立且 B 成立：符合“置信度只通过 mass/残差起作用、未被当作可靠性”的读数"
    elif not A:
        txt = f"A 不成立：s=0.25 时 {kmax25} 已变化 {100 * r25[kmax25]:.1f}%（>=5%），该读数不成立"
    else:
        txt = "A 成立但 B 不成立：s=0 也未骤变，该读数不适用"
    return {"A": A, "B": B, "r25": r25, "r0": r0, "text": txt, "max25": (kmax25, r25[kmax25])}


def judge_j2(D4):
    S = D4["summary"]
    keep = [nm for nm in S if nm.startswith("D4_keep5")]
    pk = [S[nm]["overall"]["peak_rel"]["mean"] for nm in keep]
    pc = [S[nm]["overall"]["peak_rel_common"]["mean"] for nm in keep]
    m = float(np.mean(pk))
    txt = (f"keep5 的 peak 相对变化（{len(keep)} 次随机抽样平均）= {100 * m:.1f}%："
           + ("< 10%，“max 通道让少量事件决定特征”读数成立" if m < 0.10 else ">= 10%，该读数不成立"))
    return {"peak_rel": pk, "peak_rel_mean": m, "peak_rel_common": pc, "text": txt, "holds": m < 0.10}



def _rc(S, nm, g, k):
    try:
        return S[nm][g]["_rel_vs_base"][k]["rel_change"]
    except (KeyError, TypeError):
        return None


def _cl(Dc, nm, g):
    try:
        return Dc["summary"][nm][g]["ra_rel_change"]
    except (KeyError, TypeError):
        return None


def _mean_rc(S, nms, k):
    v = [_rc(S, nm, "overall", k) for nm in nms]
    return {"mean": float(np.mean([x["mean"] for x in v])), "ci95": [float(np.mean([x["ci95"][0] for x in v])), float(np.mean([x["ci95"][1] for x in v]))]}


def _na(f, *a, **k):
    try:
        return f(*a, **k)
    except (KeyError, TypeError, IndexError, ValueError):
        return "n/a"


def summary_tables(T, A, L):
    """Section 3: one D table and one E table block (primary checkpoint), plus the two seed-robustness tables."""
    t0 = "xa_c1_s3407"
    P_ = T[t0]
    D1, D1b, D23, D4, Dc = P_["D1"], P_["D1b"], P_["D23"], P_["D4"], P_["Dcl"]
    A("## 3. 汇总表（D 一张、E 一块；主检查点 `xa_c1_s3407`，其余两个检查点见各表后的种子稳健性表）\n")
    A("### 3.0 D 汇总表\n")
    A("全部单元格为相对基线（q 未改动、hook 关闭）的变化，形如 `变化% [95% 区间]`。TF = 真值前态单步，全部 2590 包（global 1386 + local 1204）；闭环 = `track_sequence`，global 1386 / local 1204 / 全部 2590 步。"
      "“特征层读数”：H_rel = ‖H−H₁‖/‖H₁‖（vin 输出），peak_rel = peak 通道相对变化，resid = 残差质量（每 live 事件）。基线行为绝对值（mm / ° / rad，均值 [区间]）。\n")
    ks = [("d_t_mm", "TF |Δ|_t(mm)", 2), ("d_rot_deg", "TF |Δ|_rot(°)", 2), ("d_fing_rad", "TF |Δ|_finger(rad)", 3), ("ra_mm", "TF RA(mm)", 2), ("e_rot_deg", "TF e_rot(°)", 2)]
    head = ["设定"] + [k[1] for k in ks] + ["闭环 overall RA", "闭环 global RA", "闭环 local RA", "特征层读数"]
    rows = []
    if D1:
        S1 = D1["summary"]
        base_row = ["基线（绝对值）"] + [fe(S1["base"]["overall"].get(k), nd) for k, _, nd in ks]
        base_row += [_na(lambda g=g: fe(Dc["summary"]["base"][g]["ra_mm"], 2)) for g in ("overall", "zgz_global", "zgz_local")]
        base_row += ["resid " + _na(lambda: f"{S1['base']['overall']['resid_mass']['mean']:.3f}")]
        rows.append(base_row)

        def add(S, nm, lab, clname, feat):
            r = [lab] + [fp(_rc(S, nm, "overall", k)) for k, _, _ in ks]
            r += [fp(_cl(Dc, clname, g)) if clname else "—" for g in ("overall", "zgz_global", "zgz_local")]
            r += [feat]
            rows.append(r)
        for nm, lab, cl in (("D1_s0.1", "D1 s=0.1", None), ("D1_s0.25", "D1 s=0.25", "D5_s0.25"), ("D1_s0.5", "D1 s=0.5", "D5_s0.5"), ("D1_s0", "D1 s=0", "D5_s0"),
                            ("D1_s1.5_trunc1", "D1 s=1.5（min(1,1.5q)）", None)):
            o = S1[nm]["overall"]
            add(S1, nm, lab, cl, _na(lambda o=o: f"H_rel {o['H_rel']['mean']:.3f}; peak_rel {o['peak_rel']['mean']:.3f}; resid {o['resid_mass']['mean']:.3f}"))
    if D23:
        S = D23["summary"]
        for nm, lab in (("D2_near_s0", "D2 near s=0"), ("D2_near_s0.5", "D2 near s=0.5"), ("D3_hit_s0", "D3 hit s=0"), ("D3_hit_s0.5", "D3 hit s=0.5")):
            o = S[nm]["overall"]
            rows.append([lab] + [fp(_rc(S, nm, "overall", k)) for k, _, _ in ks] + ["—"] * 3 +
                        [_na(lambda o=o: f"H_rel {o['H_rel']['mean']:.3f}; peak_rel {o['peak_rel']['mean']:.3f}; resid {o['resid_mass']['mean']:.3f}")])
    if D4:
        S = D4["summary"]
        for nm in S:
            if nm == "base":
                continue
            o = S[nm]["overall"]
            rows.append([nm.replace("D4_", "D4 ")] + [fp(_rc(S, nm, "overall", k)) for k, _, _ in ks] + ["—"] * 3 +
                        [_na(lambda o=o: f"peak_rel {o['peak_rel']['mean']:.3f} (共同顶点 {o['peak_rel_common']['mean']:.3f}); 顶点保留 {o['frac_vert_kept']['mean']:.3f}; resid {o['resid_mass']['mean']:.3f}")])
    if D1b:
        S = D1b["summary"]
        for nm, lab in (("D1b_s0.25_lmfix", "D1b s=0.25 仅残差路径"), ("D1b_s0.25_resfix", "D1b s=0.25 仅 mass 路径"), ("D1b_s0.25_bothfix", "D1b s=0.25 两路径都还原")):
            c = D1b["hook_counters"][nm]
            rows.append([lab] + [fp(_rc(S, nm, "overall", k)) for k, _, _ in ks] + ["—"] * 3 + [f"输出改变 {int(c.get('pred_changed', 0))}/{int(c.get('packets_verified', 0))} 包"])
    L += md_table(head, rows)
    A("")
    if D1 and D4:
        j1, j2 = judge_j1(D1), judge_j2(D4)
        A(f"**读数（按第 2 节预先写定的口径）**：J1：{j1['text']}（A={'成立' if j1['A'] else '不成立'}，B={'成立' if j1['B'] else '不成立'}）。J2：{j2['text']}。\n")
    # seeds D
    A("**D 种子稳健性**（三个检查点同表；全部单元格为相对基线的变化；TF 全部 2590 包，闭环 2590 步）：\n")
    ks2 = [("d_rot_deg", "TF |Δ|_rot"), ("ra_mm", "TF RA"), ("e_rot_deg", "TF e_rot")]
    rows = []
    for t in TAGS:
        d1, dc, d4 = T[t]["D1"], T[t]["Dcl"], T[t]["D4"]
        if not d1:
            continue
        S1 = d1["summary"]
        for nm, lab, cl in (("D1_s0.25", "s=0.25", "D5_s0.25"), ("D1_s0.5", "s=0.5", "D5_s0.5"), ("D1_s0", "s=0", "D5_s0")):
            rows.append([t, lab] + [fp(_rc(S1, nm, "overall", k)) for k, _ in ks2] + [fp(_cl(dc, cl, g)) if dc else "—" for g in ("overall", "zgz_global", "zgz_local")] +
                        [_na(lambda: f"|Δ|_t {fp(_rc(S1, nm, 'overall', 'd_t_mm'))}")])
        if d4:
            S4 = d4["summary"]
            kk = [nm for nm in S4 if nm.startswith("D4_keep5")]
            pk = np.mean([S4[nm]["overall"]["peak_rel"]["mean"] for nm in kk])
            rows.append([t, "D4 keep5（peak_rel %.1f%%）" % (100 * pk)] + [fp(_mean_rc(S4, kk, k)) for k, _ in ks2] + ["—"] * 3 + ["（%d 次抽样平均）" % len(kk)])
        j1 = _na(lambda: judge_j1(d1))
        if j1 != "n/a":
            rows.append([t, "J1 读数"] + [""] * 6 + [j1["text"]])
    L += md_table(["检查点", "设定"] + [k[1] for k in ks2] + ["闭环 overall RA", "闭环 global RA", "闭环 local RA", "备注"], rows)
    A("")

    # ---------------------------------------------------------------- E
    A("### 3.0b E 汇总表\n")
    E1, E1c, E2, E3 = P_["E1"], P_["E1cl"], P_["E2"], P_["E3"]
    A("**E1 冻结几何消融**（分布外输入，只说明敏感性）：相对基线的变化；TF 全部 2590 包，闭环 2590 步。\n")
    ks = [("ra_mm", "TF RA"), ("e_t_mm", "TF e_t"), ("e_rot_deg", "TF e_rot"), ("e_fing_rad", "TF e_finger"), ("d_t_mm", "TF |Δ|_t"), ("d_rot_deg", "TF |Δ|_rot")]
    rows = []
    if E1:
        S = E1["summary"]
        rows.append(["基线（绝对值）"] + [fe(S["base"]["overall"].get(k), 2 if k != "e_fing_rad" else 3) for k, _ in ks] +
                    [_na(lambda g=g: fe(E1c["summary"]["base"][g]["ra_mm"], 2)) for g in ("overall", "zgz_global", "zgz_local")])
        for nm in S:
            if nm != "base":
                rows.append([nm.replace("E1_", "")] + [fp(_rc(S, nm, "overall", k)) for k, _ in ks] +
                            [fp(E1c["summary"][nm][g].get("ra_rel_change")) if E1c else "—" for g in ("overall", "zgz_global", "zgz_local")])
    L += md_table(["变体"] + [k[1] for k in ks] + ["闭环 overall RA", "闭环 global RA", "闭环 local RA"], rows)
    A("")
    A("**E1 种子稳健性**（TF RA 与闭环 overall RA 的相对变化）：\n")
    rows = []
    for t in TAGS:
        e1, e1c = T[t]["E1"], T[t]["E1cl"]
        if not e1:
            continue
        for nm in e1["summary"]:
            if nm == "base":
                continue
            rows.append([t, nm.replace("E1_", ""), fp(_rc(e1["summary"], nm, "overall", "ra_mm")), fp(_rc(e1["summary"], nm, "overall", "e_rot_deg")),
                         fp(e1c["summary"][nm]["overall"].get("ra_rel_change")) if e1c else "—", fp(e1c["summary"][nm]["zgz_global"].get("ra_rel_change")) if e1c else "—",
                         fp(e1c["summary"][nm]["zgz_local"].get("ra_rel_change")) if e1c else "—"])
    L += md_table(["检查点", "变体", "TF RA", "TF e_rot", "闭环 overall RA", "闭环 global RA", "闭环 local RA"], rows)
    A("")
    if E2:
        S = E2["summary"]
        A(f"**E2 前态扰动恢复**（TF k=1，全部 2590 包；扰动族内 ± 合并）：R_excess = 1 − mean‖pred_扰动 − pred_无扰动‖/‖扰动‖（越大越恢复，0=原样保留，负=放大）；R = 1 − err_after/err_before（相对窗末真值，含模型自身误差底）；"
          f"k 步 = 起点每 {E2['meta']['stride']} 包取 1 个，残余扰动比 = mean‖pred_扰动−pred_对照‖/‖扰动‖ 在第 k 步；串扰 = 其他块误差相对无扰动的增量（TF，k=1，global / local）。\n")
        fams = [f for f in S["groups"]["overall"] if not f.startswith("_")]
        rows = []
        for f_ in fams:
            row = [f_]
            for g in ("zgz_global", "zgz_local"):
                row.append(fe(S["groups"][g][f_]["tf"]["recovery_ratio_excess"], 2))
            for g in ("zgz_global", "zgz_local"):
                row.append(fe(S["groups"][g][f_]["tf"]["recovery_ratio"], 2))
            for g in ("zgz_global", "zgz_local"):
                ks_ = S["k_list"]
                row.append(_na(lambda g=g: " / ".join(f"{S['groups'][g][f_]['k_step'][str(k)]['residual_perturbation_ratio']['mean']:.2f}" for k in ks_)))
            for g in ("zgz_global", "zgz_local"):
                ct = S["groups"][g][f_]["tf"]["crosstalk"]
                row.append("；".join(f"{k.replace('_mm', ' mm').replace('_deg', '°').replace('_rad', ' rad')} {v['mean']:+.2f}" for k, v in ct.items()))
            rows.append(row)
        ksl = S["k_list"] if E2["meta"]["stride"] else []
        L += md_table(["扰动族", "R_excess global", "R_excess local", "R global", "R local"] +
                      [f"残余扰动比 k={'/'.join(map(str, ksl))} global", f"残余扰动比 k={'/'.join(map(str, ksl))} local", "串扰 global", "串扰 local"], rows)
        A("")
        A("**E2 种子稳健性**（TF k=1 的 R_excess，global / local；全部 2590 包）：\n")
        rows = []
        for f_ in fams:
            row = [f_]
            for t in TAGS:
                e2 = T[t]["E2"]
                for g in ("zgz_global", "zgz_local"):
                    row.append(_na(lambda e2=e2, g=g: fe(e2["summary"]["groups"][g][f_]["tf"]["recovery_ratio_excess"], 2)))
            rows.append(row)
        L += md_table(["扰动族"] + [f"{t.replace('xa_', '')} {g}" for t in TAGS for g in ("global", "local")], rows)
        A("")
    if E3:
        S = E3["summary"]
        A("**E3 事件扰动响应**（TF，全部 2590 包；d_ratio = |Δ_v|/|Δ_0|，cos0 = cos(Δ_v, Δ_0)，t / rot / fing 三块）：\n")
        rows = []
        for nm, lab_ in (("a_shift4px", "(a) 事件 +4 px (x)"), ("b_other_packet", "(b) 同 run 1 s 外另一包事件"), ("c_empty", "(c) 空包")):
            o = S[nm]["overall"]
            g = lambda k, nd=2: fe(o.get(k), nd)       # noqa: E731
            rows.append([lab_, g("avail", 3), f"{g('d_ratio_t')} / {g('d_ratio_rot')} / {g('d_ratio_fing')}", f"{g('cos0_t')} / {g('cos0_rot')} / {g('cos0_fing')}",
                         (f"cos(ΔΔt,+x̂) {g('cos_shift_x')}；‖ΔΔt‖/期望 {g('shift_over_expected')}；x 分量/期望 {g('shift_x_over_expected')}" if nm == "a_shift4px" else "")])
        L += md_table(["变体", "可用比例", "d_ratio t / rot / fing", "cos0 t / rot / fing", "(a) 与真值位移方向比较（只看平移块）"], rows)
        A("")


def make_report(args, out_dir):
    out_dir = Path(out_dir)
    L = []
    A = L.append
    T = {t: {k: _ld(out_dir, f"{pre}_{t}.json") for k, pre in (
        ("D1", "D_tf_D1"), ("D1b", "D_tf_D1b"), ("D23", "D_tf_D23"), ("D4", "D_tf_D4"), ("Dcl", "D_cl"), ("E1", "E_tf_E1"), ("E1cl", "E_cl_E1"),
        ("E2", "E_e2"), ("E3", "E_e3"))} for t in TAGS}
    prim = T["xa_c1_s3407"]
    mm = next(v for v in prim.values() if v)["meta"]
    A("# [XYZ-ASSOC] step4 conf 报告：D 置信度缩放 与 E 几何通道 / 前态扰动恢复（冻结权重探针）\n")
    A(f"- 脚本：`tools/xa/probe_xa_conf.py`（复用 `tools/xa/probe_xa_assoc.py`，两者均为新增诊断脚本，未改仓库已有文件）；产物：`outputs/xa/step4/conf/`（`D_*.json`、`E_*.json`、`cols/*.npz` 逐包列、`logs/`、`gpu_choice.txt`、`PREREG_criteria.txt`）。")
    for t in TAGS:
        d = next((v for v in T[t].values() if v), None)
        if d:
            A(f"- 检查点 `{t}`：`{d['meta']['ckpt']}`（mtime {d['meta']['ckpt_mtime']}）；配置 `{d['meta']['config']}`；包数 {d['meta']['n_packets']}。")
    A(f"- 设备：{mm['device']}（CUDA_VISIBLE_DEVICES={mm['cuda_visible_devices']}，{mm['gpu_name']}），torch {mm['torch']}；选卡依据见 `gpu_choice.txt`。")
    A("- 模式：TF = 真值前态单步（prev = 上一步末真值，run 首步为 run 起点真值，不加噪，`stride_sample(2048)` 事件，与 P2-b 同口径）；闭环 = `eval_track.track_sequence`（rng(0)、50 ms、val_core、噪声 1.0，两序列共用一个 rng，与 P2-a / select_checkpoint 同）。")
    A("- 区间：滑动块 bootstrap，块长 20 步，1000 次；基线与变体共用重采样下标（配对）；“全部”= zgz_global 与 zgz_local 池化（帧加权，与 22.9343 的口径一致）；相对变化 = 池化均值之比 − 1。")
    A("- 每个表头标注：检查点 / 模式 / 序列 / 包数；单元格形如 `均值 [95% 区间]`。\n")

    # ------------------------------------------------------------------ code facts
    A("## 1. 代码事实（行号对应仓库当前文件）\n")
    A("`semkine/xyz_mesh.py`（`XYZMeshNet.forward`，421–521 行）与 `model/model.py::_xyz_mesh_forward`（2531–2608 行）：\n")
    A("- **权重**：`w = q·A·mask`（433 行），残差 `r = (1−q)·mask`（434 行）；每个 live 事件总质量 = Σ_r w + r = 1（invalid 事件 q=0，整条进残差）。`associate_rays` 中 hit 事件 q=1（310 行），near 事件 q = exp(−½(ang₀/σ)²) < 1（288、310 行），none 事件 valid=0、q=0（215–221 行的零初始化 + `ok` 掩码，309–313 行）。")
    A("- **零权重记录被丢弃**：只有 `w>0` 的 (事件, 顶点) 记录进入 φ 与聚合（`(w > 0).nonzero`，439–441 行）。hit 事件的第 4 个记录 A=0（259 行）、near 层未选中的槽 kw=0（285 行）、q=0 的事件都因此不产生记录。无任何 w>0 的顶点：`has = mass>0` 为假（470 行），mean、peak 均乘 `has` 置 0（471、474 行），`log1p(mass)=0`，顶点特征 x≡0；`vin` 无 bias（394 行），故 H≡0，且 `XYZMeshLayer` 无 bias（327–328 行说明）使全零场保持为零。")
    A("- **均值通道对 q 的均匀缩放不变**：`mass = Σ w`（467 行），`num = Σ w·φ`（468–469 行），`mean = num / mass.clamp_min(1e-12)`（471 行；eval 时累加在 float64，431 行 `acc`）。q→s·q 时 num 与 mass 同乘 s，mean 不变（仅 float 舍入）；改变相对权重（如只缩放 near 或 hit，或截断 min(1,1.5q)）才改变 mean。")
    A("- **max 通道不加权**：`peak = scatter_reduce(φ, amax)`（472–473 行）对每个顶点的全部 w>0 记录取逐通道最大，不乘 w；只要 w>0，低权重事件完整进入 max。q 的均匀缩放不改变 w>0 的记录集合，故 peak 与 s 无关（s>0 时逐位不变）；q 置 0 才使该事件离开 peak。")
    A("- **mass 通道**：`x = [mean, peak, log1p(mass)]`（475 行），mass 随 s 线性缩放，`log1p` 使 H 在 s<1 时改变（H = vin(x)，477 行）。因 `rel` 是含 ReLU 的 MLP，φ 本身不依赖 q；q 只通过 w（均值权、mass、记录有无）与 r（残差）起作用。")
    A("- **残差槽**：`eb,en = (r>0)`（488 行），`phi_r = rel([token7, 0…])`（489–491 行，无几何、valid=0）；`rs = Σ r`，`rmean = Σ r·φ_r / rs`（492–497 行），`rpeak = amax φ_r`（498–499 行，同样不加权，范围是所有 r>0 的事件）；`bslot = [rmean, rpeak, rs/live, rhas]`（500–502 行）是第 17 个槽。q 缩放到 s<1 时，**所有**事件（含 hit 的 q=1 者）r=1−s·q>0，因此残差槽的 rmean/rpeak/(rs/live) 全部改变，这是 s<1 时除 mass 外的第二条路径；s=0 时 w≡0、r≡mask，16 个关节槽全零，只剩残差槽（事件 token 袋）。")
    A("- **读出**：`slot_proj`（分组 1×1 卷积，17 槽）→ ReLU →拼 `prev48` → `fc1` → ReLU → `fc2`（505–507 行）→ delta；`forward_packet` 里 `out = delta + prev`（`model/model.py` 2820–2825 行，PREVPOS_EMBED=false，零事件包 delta 置 0）。因此 prev[3:51] 也直接进读出，与网格无关。")
    A("- **无事件**：`associate_rays` 在 `flat.numel()==0` 时直接返回全零（xyz_mesh.py 224–226 行）；整包无事件时 `_xyz_mesh_forward` 返回 `prev.new_zeros(B,51)` 且不进入 `self.xyz`（model.py 2540–2542 行），`out = 0 + prev`，即逐位返回 prev。\n")

    # ------------------------------------------------------------------ criteria
    A("## 2. 判读口径（写定于任何 D/E 数字产生之前，`PREREG_criteria.txt` 原文）\n")
    pr = out_dir / "PREREG_criteria.txt"
    A("```")
    A(pr.read_text().rstrip() if pr.exists() else "(PREREG_criteria.txt 缺失)")
    A("```\n")

    summary_tables(T, A, L)

    # ------------------------------------------------------------------ D
    A("## 4. D 详表：置信度缩放（TF 单步，全部包；闭环只对关键设定）\n")
    for t in TAGS:
        D1 = T[t]["D1"]
        if not D1:
            continue
        S = D1["summary"]
        n = D1["meta"]["n_packets"]
        order = ["base", "D1_s0", "D1_s0.1", "D1_s0.25", "D1_s0.5", "D1_s1.5_trunc1"]
        lab = {"base": "1.0（基线）", "D1_s0": "0", "D1_s0.1": "0.1", "D1_s0.25": "0.25", "D1_s0.5": "0.5", "D1_s1.5_trunc1": "1.5（min(1,1.5q)）"}
        A(f"### 4.1 D1 均匀缩放 q←s·q（检查点 `{t}`；TF；全部包 {sum(n.values())}（global {n['zgz_global']} + local {n['zgz_local']}）；vid / A / valid / hit 与事件特征不变）\n")
        A("顶点特征与路由统计（逐包量，对包池化平均；H_rel = ‖H−H₁‖_F/‖H₁‖_F，mean_rel / peak_rel / lm_rel 同理，分别是 vin 输入中的 mean、peak、log1p(mass) 通道相对 s=1 的变化；H_obs_norm = 被观测顶点 H 行范数均值）：\n")
        rows = []
        for nm in order:
            o = S[nm]["overall"]
            rows.append([lab[nm], fe(o.get("resid_mass"), 3), fe(o.get("verts_direct"), 1), fe(o.get("H_obs_norm"), 3),
                         fe(o.get("H_rel"), 3) if nm != "base" else "—", fe(o.get("mean_rel"), 2, sci=True) if nm != "base" else "—",
                         fe(o.get("peak_rel"), 3) if nm != "base" else "—", fe(o.get("lm_rel"), 3) if nm != "base" else "—",
                         fe(o.get("slot_rel_joint"), 3) if nm != "base" else "—", fe(o.get("slot_rel_resid"), 3) if nm != "base" else "—"])
        L += md_table(["s", "残差质量 resid_mass", "xyz_verts_direct", "H_obs_norm", "H_rel", "mean_rel", "peak_rel", "lm_rel(log1p mass)", "关节槽相对变化", "残差槽相对变化"], rows)
        A("")
        A("输出（TF 单步）：|Δ| = ‖pred−prev‖ 各块；TF RA = 窗末根对齐 MPJPE；更新误差 = ‖(pred−prev)−(gt−prev)‖（根平移 mm、根旋转 ° 为轴角向量差范数（P2 口径）、手指 rad）：\n")
        cols = [("d_t_mm", "|Δ|_t mm", 2), ("d_rot_deg", "|Δ|_rot °", 2), ("d_fing_rad", "|Δ|_finger rad", 3), ("ra_mm", "TF RA mm", 2),
                ("e_t_mm", "根平移更新误差 mm", 2), ("e_rot_deg", "根旋转更新误差 °", 2), ("e_fing_rad", "手指更新误差 rad", 3)]
        rows = [[lab[nm]] + [fe(S[nm]["overall"].get(k), nd) for k, _, nd in cols] for nm in order]
        L += md_table(["s"] + [c[1] for c in cols], rows)
        A("")
        A("相对 s=1 的变化（配对 bootstrap 区间）：\n")
        rows = [[lab[nm]] + [fp(S[nm]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _, _ in cols] for nm in order[1:]]
        L += md_table(["s"] + [c[1] for c in cols], rows)
        A("")
        A("分序列的相对变化（TF RA / 根旋转更新误差 / |Δ|_rot / |Δ|_t）：\n")
        rows = []
        for g in ("zgz_global", "zgz_local", "local_fit_true", "local_fit_false"):
            for nm in order[1:]:
                r_ = S[nm][g]["_rel_vs_base"]
                rows.append([SEQN[g], lab[nm], fp(r_["ra_mm"]["rel_change"]), fp(r_["e_rot_deg"]["rel_change"]), fp(r_["d_rot_deg"]["rel_change"]), fp(r_["d_t_mm"]["rel_change"])])
        L += md_table(["序列/层", "s", "TF RA", "e_rot", "|Δ|_rot", "|Δ|_t"], rows)
        A("")
        j1 = judge_j1(D1)
        A(f"**J1 读数（`{t}`）**：{j1['text']}。s=0.25 各指标 |相对变化| = " + "，".join(f"{k} {100 * v:.2f}%" for k, v in j1["r25"].items()) +
          "；s=0 各指标 = " + "，".join(f"{k} {100 * v:.2f}%" for k, v in j1["r0"].items()) + "。\n")
        ch = D1.get("checks", {})
        A(f"自检（`{t}`）：s=0 的 hook 与 P2 的 `associate_rays` 包装器（对所有 live 事件 q:=0）逐位相等 {ch.get('s0_hook_vs_associate_wrapper_bitwise_equal')}；"
          f"s=1 hook 激活但输出逐位等于基线（每包断言，`D1_s1_check` 前 60 包）；每包断言 s=0 时 w≡0、r≡mask、H≡0，且残差增量 = Σ(q−q′)。\n")

    # D1b path isolation
    for t in TAGS:
        D1b, D1 = T[t]["D1b"], T[t]["D1"]
        if not D1b or not D1:
            continue
        S, S1 = D1b["summary"], D1["summary"]
        n = D1b["meta"]["n_packets"]
        A(f"### 4.1b D1 的路径隔离（检查点 `{t}`；TF；全部包 {sum(n.values())}）\n")
        A("q←s·q 时，网络通过两条路径看到变化：(i) 顶点 mass 通道 log1p(mass)，(ii) 残差槽（r=1−s·q 把所有事件部分地送进残差，rmean/rpeak/rs·live⁻¹ 变）。mean、peak 通道在 s∈{0.25,0.5}（2 的幂，浮点缩放精确）时逐位不变。"
          "隔离：`lmfix` = 缩放 q，但把 vin 输入的 log1p(mass) 通道换回基线值（只剩残差路径）；`resfix` = 缩放 q，但把读出前第 17 槽（残差槽）换回基线值（只剩 mass 路径）；`bothfix` = 两者都换回，应逐位还原基线。\n")
        cols = [("d_t_mm", "|Δ|_t"), ("d_rot_deg", "|Δ|_rot"), ("d_fing_rad", "|Δ|_finger"), ("ra_mm", "TF RA"), ("e_t_mm", "e_t"), ("e_rot_deg", "e_rot"), ("e_fing_rad", "e_finger")]
        rows = []
        for sv in (0.25, 0.5):
            rows.append([f"s={sv:g} 完整（D1）"] + [fp(S1[f"D1_s{sv:g}"]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _ in cols] + ["—"])
            for tag, lab_ in (("lmfix", "仅残差路径（lmfix）"), ("resfix", "仅 mass 路径（resfix）"), ("bothfix", "两条都还原（bothfix）")):
                nm = f"D1b_s{sv:g}_{tag}"
                c = D1b["hook_counters"][nm]
                rows.append([f"s={sv:g} {lab_}"] + [fp(S[nm]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _ in cols] +
                            [f"输出改变 {int(c.get('pred_changed', 0))}/{int(c.get('packets_verified', 0))} 包，max|Δpred|={c.get('pred_maxabs_diff_max', 0):.2e}"])
        L += md_table(["设定"] + [c[1] for c in cols] + ["与基线逐位比较"], rows)
        A("")

    # D2 / D3
    for t in TAGS:
        D23 = T[t]["D23"]
        if not D23:
            continue
        S = D23["summary"]
        n = D23["meta"]["n_packets"]
        A(f"### 4.2 D2（只缩放 near，hit 保持 q=1）与 D3（只缩放 hit，near 不变）（检查点 `{t}`；TF；全部包 {sum(n.values())}）\n")
        nm_lab = {"D2_near_s0": "D2 near s=0", "D2_near_s0.5": "D2 near s=0.5", "D3_hit_s0": "D3 hit s=0", "D3_hit_s0.5": "D3 hit s=0.5"}
        cols = [("resid_mass", "resid_mass", 3), ("H_rel", "H_rel", 3), ("mean_rel", "mean_rel", 3), ("peak_rel", "peak_rel", 3), ("lm_rel", "lm_rel", 3),
                ("d_t_mm", "|Δ|_t mm", 2), ("d_rot_deg", "|Δ|_rot °", 2), ("d_fing_rad", "|Δ|_finger rad", 3), ("ra_mm", "TF RA mm", 2), ("e_rot_deg", "e_rot °", 2)]
        rows = [[nm_lab[nm]] + [fe(S[nm]["overall"].get(k), nd) for k, _, nd in cols] for nm in nm_lab]
        L += md_table(["变体"] + [c[1] for c in cols], rows)
        A("")
        cols2 = [("d_t_mm", "|Δ|_t"), ("d_rot_deg", "|Δ|_rot"), ("d_fing_rad", "|Δ|_finger"), ("ra_mm", "TF RA"), ("e_t_mm", "e_t"), ("e_rot_deg", "e_rot"), ("e_fing_rad", "e_finger")]
        A("相对基线（s=1）的变化：\n")
        rows = [[nm_lab[nm]] + [fp(S[nm]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _ in cols2] for nm in nm_lab]
        L += md_table(["变体"] + [c[1] for c in cols2], rows)
        A("")

    # D4
    for t in TAGS:
        D4 = T[t]["D4"]
        if not D4:
            continue
        S = D4["summary"]
        n = D4["meta"]["n_packets"]
        A(f"### 4.3 D4 少量事件形成强特征（检查点 `{t}`；TF；全部包 {sum(n.values())}；每包随机 5% 的 q>0 事件，两次随机抽样 r0/r1）\n")
        A("keep5 = 随机保留 5% 的 q>0 事件、其余 95% 的 q 置 0（进残差槽，保留者 q 不变）；drop5 = 只把随机 5% 的 q>0 事件 q 置 0。peak_rel 为整张 (778,128) peak 张量的相对变化；peak_rel_common 只在基线与变体都被观测的顶点上；frac_vert_kept = 变体被观测顶点数 / 基线被观测顶点数。\n")
        cols = [("n_rec", "w>0 记录数", 0), ("frac_vert_kept", "顶点保留率", 3), ("peak_rel", "peak_rel", 3), ("peak_rel_common", "peak_rel_common", 3), ("mean_rel", "mean_rel", 3),
                ("lm_rel", "lm_rel", 3), ("H_rel", "H_rel", 3), ("resid_mass", "resid_mass", 3)]
        rows = [["基线"] + [fe(S["base"]["overall"].get(k), nd) if k in ("n_rec", "resid_mass") else "—" for k, _, nd in cols]]
        for nm in S:
            if nm == "base":
                continue
            rows.append([nm] + [fe(S[nm]["overall"].get(k), nd) for k, _, nd in cols])
        L += md_table(["变体"] + [c[1] for c in cols], rows)
        A("")
        cols2 = [("d_t_mm", "|Δ|_t"), ("d_rot_deg", "|Δ|_rot"), ("d_fing_rad", "|Δ|_finger"), ("ra_mm", "TF RA"), ("e_t_mm", "e_t"), ("e_rot_deg", "e_rot"), ("e_fing_rad", "e_finger")]
        A("输出相对基线的变化，以及 |Δ| 与基线 Δ 的差 ‖Δ_v−Δ_0‖（dd）：\n")
        rows = [[nm] + [fp(S[nm]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _ in cols2] +
                [fe(S[nm]["overall"].get("dd_t_mm"), 2), fe(S[nm]["overall"].get("dd_rot_deg"), 2), fe(S[nm]["overall"].get("dd_fing_rad"), 3)] for nm in S if nm != "base"]
        L += md_table(["变体"] + [c[1] for c in cols2] + ["dd_t mm", "dd_rot °", "dd_finger rad"], rows)
        A("")
        j2 = judge_j2(D4)
        A(f"**J2 读数（`{t}`）**：{j2['text']}（逐次：" + "，".join(f"{100 * v:.1f}%" for v in j2["peak_rel"]) + f"；peak_rel_common：" + "，".join(f"{100 * v:.1f}%" for v in j2["peak_rel_common"]) + "）。\n")

    # D5
    A("### 4.4 D5 闭环（模式：闭环 `track_sequence`，val_core，rng(0)；序列 global 1386 / local 1204 / 全部 2590 步）\n")
    for t in TAGS:
        Dc = T[t]["Dcl"]
        if not Dc:
            continue
        S = Dc["summary"]
        A(f"检查点 `{t}`：闭环 RA（mm）与相对 s=1 的变化；区间同上（块 bootstrap 配对）。\n")
        rows = []
        order = [nm for nm in ("base", "D5_s1", "D5_s0", "D5_s0.25", "D5_s0.5", "D5x_near_s0", "D5x_hit_s0") if nm in S]
        for nm in order:
            r_ = S[nm]
            row = [nm if nm != "base" else "基线（hook 关闭）"]
            for g in ("overall", "zgz_global", "zgz_local"):
                row.append(fe(r_[g]["ra_mm"], 3) + ("" if nm == "base" else f"；{fp(r_[g].get('ra_rel_change'))}"))
            rm = r_["route_mean"]
            row += [" / ".join(f"{rm[sq].get('xyz_resid_mass', float('nan')):.3f}" for sq in rm), " / ".join(f"{rm[sq].get('xyz_verts_direct', float('nan')):.1f}" for sq in rm)]
            rows.append(row)
        L += md_table(["设定", "overall RA", "global RA", "local RA", "闭环 resid_mass 均值 (global / local)", "verts_direct 均值 (global / local)"], rows)
        A("")
        ck = Dc.get("reference_check", {})
        A(f"hook 关闭自检（`{t}`）：{json.dumps({k: v for k, v in ck.items() if k != 'ref'}, ensure_ascii=False)}\n")

    # ------------------------------------------------------------------ E
    A("## 5. E 详表：几何通道依赖与前态扰动恢复\n")
    A("### 5.1 E1 冻结消融（分布外输入，只说明敏感性）\n")
    A("实现：`no_dxyz` = `rel` 输入第 7:10 列置 0；`no_z` = 第 10 列（z_prior）置 0；`no_edge` = `edge` 层输入（边向量）置 0；`no_geom` = 三者全零（逐位等于 `xyz.geom=False` 的 C0 输入，见自检）；`z_median` = 匹配记录的第 10 列换成该包 valid 事件 z 的中位数常量。残差记录的几何本来为 0，不受影响。\n")
    for t in TAGS:
        E1 = T[t]["E1"]
        if not E1:
            continue
        S = E1["summary"]
        n = E1["meta"]["n_packets"]
        A(f"检查点 `{t}`；TF；全部包 {sum(n.values())}：\n")
        cols = [("ra_mm", "TF RA", 2), ("e_t_mm", "e_t mm", 2), ("e_rot_deg", "e_rot °", 2), ("e_fing_rad", "e_finger rad", 3), ("d_t_mm", "|Δ|_t mm", 2), ("d_rot_deg", "|Δ|_rot °", 2), ("d_fing_rad", "|Δ|_finger rad", 3), ("H_rel", "H_rel", 3)]
        rows = [["基线"] + [fe(S["base"]["overall"].get(k), nd) if k != "H_rel" else "—" for k, _, nd in cols]]
        for nm in S:
            if nm != "base":
                rows.append([nm] + [fe(S[nm]["overall"].get(k), nd) for k, _, nd in cols])
        L += md_table(["变体"] + [c[1] for c in cols], rows)
        A("")
        A("相对基线的变化：\n")
        rows = [[nm] + [fp(S[nm]["overall"]["_rel_vs_base"].get(k, {}).get("rel_change")) for k, _, _ in cols[:-1]] for nm in S if nm != "base"]
        L += md_table(["变体"] + [c[1] for c in cols[:-1]], rows)
        A("")
        A("分序列（TF RA 相对变化 / 根旋转更新误差相对变化）：\n")
        rows = []
        for g in ("zgz_global", "zgz_local", "local_fit_true", "local_fit_false"):
            for nm in S:
                if nm != "base":
                    r_ = S[nm][g]["_rel_vs_base"]
                    rows.append([SEQN[g], nm, fp(r_["ra_mm"]["rel_change"]), fp(r_["e_rot_deg"]["rel_change"]), fp(r_["e_t_mm"]["rel_change"]), fp(r_["e_fing_rad"]["rel_change"])])
        L += md_table(["序列/层", "变体", "TF RA", "e_rot", "e_t", "e_finger"], rows)
        A("")
        ck = E1.get("checks", {})
        A(f"自检（`{t}`）：`no_geom` hook 与 `xyz.geom=False` 逐位相等 {ck.get('no_geom_hooks_vs_geom_False_bitwise_equal')}。计数器见 JSON `hook_counters`。\n")
    A("闭环（模式：闭环；global 1386 / local 1204 / 全部 2590 步）：\n")
    for t in TAGS:
        Ec = T[t]["E1cl"]
        if not Ec:
            continue
        S = Ec["summary"]
        A(f"检查点 `{t}`：\n")
        rows = []
        for nm in S:
            row = ["基线（hook 关闭）" if nm == "base" else nm]
            for g in ("overall", "zgz_global", "zgz_local"):
                row.append(fe(S[nm][g]["ra_mm"], 3) + ("" if nm == "base" else f"；{fp(S[nm][g].get('ra_rel_change'))}"))
            rows.append(row)
        L += md_table(["变体", "overall RA", "global RA", "local RA"], rows)
        A("")
        A(f"hook 关闭自检：{json.dumps({k: v for k, v in Ec.get('reference_check', {}).items() if k != 'ref'}, ensure_ascii=False)}\n")

    # ---------------- E2
    A("### 5.2 E2 前态扰动恢复（TF：真值前态 + 确定性扰动，事件不变；k 步：真实事件、自身 prev）\n")
    A("扰动：平移 ±20 mm（x/y/z 各自）、根旋转 ±10°（绕相机 x/y/z，左乘 Exp(±10°·e_axis)·R_prev，轴角上合成）、单指（食指 MCP 的 3 个轴角分量各 +0.3 rad，范数 0.520 rad）、全指（45 维各 +0.1 rad，范数 0.671 rad）。表中同族的 ± 两个方向合并（逐包取均值），分号方向明细在 JSON `_signed_detail_tf`。"
      "误差块：平移 mm、根旋转测地线 °、手指 rad（单指为该 MCP 3 维，串扰里的手指为其余 42 维）。\n")
    for t in TAGS:
        E2 = T[t]["E2"]
        if not E2:
            continue
        S = E2["summary"]
        n = E2["meta"]["n_packets"]
        A(f"#### 检查点 `{t}`（TF 全部包 {sum(n.values())}；k 步起点每 {E2['meta']['stride']} 包取 1 个）\n")
        A(f"自检：B=15 批与单样本 forward 的最大绝对差 = {E2['checks']['b1_vs_batch_maxabs']:.2e}（{E2['checks']['b1_checked']} 次比较）。\n")
        fams = [f for f in S["groups"]["overall"] if not f.startswith("_")]
        for g in ("zgz_global", "zgz_local", "overall"):
            G = S["groups"][g]
            A(f"TF 恢复（k=1）：{SEQN[g]}（包数 {G[fams[0]]['tf']['err_after']['n']}）。恢复比 R = 1 − mean err_after / mean err_before（相对窗末真值）；R_excess = 1 − mean‖pred_扰动 − pred_无扰动‖ / 扰动大小；串扰 = 其他块上 [扰动 − 无扰动] 的误差增量：\n")
            rows = []
            for f_ in fams:
                r_ = G[f_]["tf"]
                ct = "；".join(f"{k.replace('_mm','').replace('_deg','°').replace('_rad','')} {fe(v, 2 if 'mm' in k or 'deg' in k else 3)}" for k, v in r_["crosstalk"].items())
                rows.append([f_, G[f_]["target_block"], fe(r_["perturbation_size"], 2), fe(r_["err_before"], 2), fe(r_["err_after"], 2), fe(r_["recovery_ratio"], 2),
                             fe(r_["recovery_ratio_excess"], 2), ct])
            L += md_table(["扰动族", "目标块", "扰动大小", "err_before", "err_after", "恢复比 R", "R_excess", "串扰（其他块误差增量）"], rows)
            ctrl = G["_control_tf"]
            A(f"\n无扰动对照（同口径，相对窗末真值）：" + "；".join(f"{b}：前态 {fe(ctrl[b]['err_before_ctrl'], 2)} → 输出 {fe(ctrl[b]['err_after_ctrl'], 2)}" for b in ("t_mm", "rot_deg", "fing_rad")) + "。\n")
        A("按 fingertip_in_front 分层（仅 zgz_local；TF 恢复比 R、R_excess，以及 k 步残余扰动比 k=" + "/".join(map(str, S["k_list"])) + "）：\n")
        rows = []
        for f_ in fams:
            row = [f_]
            for g in ("local_fit_true", "local_fit_false"):
                row += [fe(S["groups"][g][f_]["tf"][k], 2) for k in ("recovery_ratio", "recovery_ratio_excess")]
                row.append(_na(lambda g=g: " / ".join(f"{S['groups'][g][f_]['k_step'][str(k)]['residual_perturbation_ratio']['mean']:.2f}" for k in S["k_list"])) if E2["meta"]["stride"] else "—")
            rows.append(row)
        L += md_table(["扰动族", "fit=真 R", "fit=真 R_excess", "fit=真 残余扰动比", "fit=假 R", "fit=假 R_excess", "fit=假 残余扰动比"], rows)
        A(f"（fingertip_in_front=真 包数 {S['groups']['local_fit_true'][fams[0]]['tf']['err_after']['n']}，假 {S['groups']['local_fit_false'][fams[0]]['tf']['err_after']['n']}）\n")
        if E2["meta"]["stride"]:
            ks = S["k_list"]
            for g in ("zgz_global", "zgz_local"):
                G = S["groups"][g]
                A(f"k 步递推：{SEQN[g]}。err_k/err_0（err_0 = 扰动大小；err_k = 第 k 步输出相对该步窗末真值的目标块误差），括号内为无扰动对照的 err_k/err_0；残余扰动比 = mean‖pred_扰动−pred_对照‖/err_0（越小说明扰动被抹掉得越多）：\n")
                rows = []
                for f_ in fams:
                    row = [f_]
                    for k in ks:
                        r_ = G[f_]["k_step"][str(k)]
                        row.append(f"{fe(r_['err_k_over_err_0'], 2)}（对照 {r_['ctrl_err_k_over_err_0']['mean']:.2f}）")
                    for k in ks:
                        row.append(fe(G[f_]["k_step"][str(k)]["residual_perturbation_ratio"], 2))
                    row.append(str(G[f_]["k_step"][str(ks[-1])]["err_k"]["n"]) + "/" + str(G[f_]["k_step"][str(ks[0])]["err_k"]["n"]))
                    rows.append(row)
                L += md_table(["扰动族"] + [f"err_{k}/err_0" for k in ks] + [f"残余扰动比 k={k}" for k in ks] + [f"起点数 k={ks[-1]}/k={ks[0]}"], rows)
                A("")
            A("k 步串扰（其他块误差相对对照的增量；zgz_local）：\n")
            G = S["groups"]["zgz_local"]
            rows = []
            for f_ in fams:
                for k in ks:
                    ct = G[f_]["k_step"][str(k)]["crosstalk"]
                    rows.append([f_, k] + [f"{o}: {fe(v, 2 if 'mm' in o or 'deg' in o else 3)}" for o, v in ct.items()])
            L += md_table(["扰动族", "k", "串扰 1", "串扰 2", "串扰 3"], [r + [""] * (5 - len(r)) for r in rows])
            A("")

    # ---------------- E3
    A("### 5.3 E3 事件扰动响应（TF；全部包）\n")
    A("(a) 事件整体 x 平移 +4 px（不裁剪）；(b) 事件换成同 run 内 ≥1 s 外另一包（优先 +20 步，否则 −20 步，两者都不在同 run 内则该包不可用）；(c) 空包。Δ = pred − prev。d_ratio = |Δ_v|/|Δ_0|；dd = ‖Δ_v−Δ_0‖；cos0 = cos(Δ_v, Δ_0)；cosgt = cos(Δ_v, 真值更新 gt−prev)；(a) 的 cos_shift_x = cos(Δ_a[t]−Δ_0[t], +x̂)（+4 px 的真值位移方向），shift_over_expected = ‖Δ_a[t]−Δ_0[t]‖ / (4 px / fx_render × z_wrist(prev))（fx_render = fx×0.375）。\n")
    for t in TAGS:
        E3 = T[t]["E3"]
        if not E3:
            continue
        S = E3["summary"]
        n = E3["meta"]["n_packets"]
        A(f"检查点 `{t}`；TF；全部包 {sum(n.values())}：\n")
        rows = []
        for nm, lab_ in (("a_shift4px", "(a) +4 px"), ("b_other_packet", "(b) 1 s 外另一包"), ("c_empty", "(c) 空包")):
            o = S[nm]["overall"]
            g = lambda k, nd=2: fe(o.get(k), nd)       # noqa: E731
            rows.append([lab_, g("avail", 3), f"{g('d_ratio_t')} / {g('d_ratio_rot')} / {g('d_ratio_fing')}", f"{g('dd_t')} / {g('dd_rot')} / {g('dd_fing', 3)}",
                         f"{g('cos0_t')} / {g('cos0_rot')} / {g('cos0_fing')}", f"{g('cosgt_t')} / {g('cosgt_rot')} / {g('cosgt_fing')}"])
        o = S["base"]["overall"]
        rows.append(["基线 Δ_0", "—", "—", f"|Δ| {fe(o.get('d_t'), 2)} mm / {fe(o.get('d_rot'), 2)}° / {fe(o.get('d_fing'), 3)} rad", "—",
                     f"{fe(o.get('cosgt_t'), 2)} / {fe(o.get('cosgt_rot'), 2)} / {fe(o.get('cosgt_fing'), 2)}"])
        L += md_table(["变体", "可用比例", "d_ratio t / rot / fing", "dd t mm / rot ° / fing rad", "cos0 t / rot / fing", "cosgt t / rot / fing"], rows)
        a = S["a_shift4px"]["overall"]
        A(f"\n(a) +4 px：cos_shift_x = {fe(a.get('cos_shift_x'), 2)}；位移幅度 {fe(a.get('shift_mm'), 2)} mm，期望 {fe(a.get('expected_mm'), 2)} mm，shift_over_expected = {fe(a.get('shift_over_expected'), 2)}，x 分量/期望 = {fe(a.get('shift_x_over_expected'), 2)}。\n")
        bb = S["b_other_packet"]["overall"]
        cc = S["c_empty"]["overall"]
        A(f"(c) 空包：max|pred−prev| = {fe(cc.get('maxabs_pred_minus_prev'), 2, sci=True)}（模型对空包逐位返回 prev）。\n")

    # ------------------------------------------------------------------ self-checks
    A("## 6. 自检与猴补/hook 生效证明\n")
    for t in TAGS:
        for key, nm in (("Dcl", "D 闭环"), ("E1cl", "E1 闭环")):
            d = T[t][key]
            if d:
                ck = d.get("reference_check", {})
                A(f"- `{t}` {nm}：hook 全部关闭（hook 已装但 spec=None）时的闭环 RA："
                  f"{json.dumps({k: v for k, v in ck.items() if k not in ('ref',)}, ensure_ascii=False)}")
    A("")
    A("hook 计数器（节选；全量在各 JSON 的 `hook_counters`）。`q_events_changed / q_events_total_pos` = 被 hook 改写 q 的事件数 / q>0 的事件数（TF 全部包累计）；`pred_changed / packets_verified` = 输出与基线不同的包数 / 经逐包断言的包数：\n")
    rows = []
    for t in TAGS:
        for key in ("D1", "D23", "D4", "E1"):
            d = T[t][key]
            if not d:
                continue
            for nm, c in d["hook_counters"].items():
                if nm.endswith("_check") and "D1_s1" not in nm:
                    continue
                rows.append([t, nm, int(c.get("packets_verified", 0)), int(c.get("pred_changed", 0)), int(c.get("q_events_changed", 0)), int(c.get("q_events_total_pos", 0)),
                             int(c.get("rel_nonzero_zeroed", 0)), int(c.get("edge_nonzero_zeroed", 0)), int(c.get("zmed_rows_changed", 0))])
    L += md_table(["检查点", "变体", "断言包数", "输出改变包数", "q 改写事件", "q>0 事件", "rel 输入被清零元素", "edge 输入被清零元素", "z_median 改写行"], rows)
    A("")
    for t in TAGS:
        d = T[t]["D1"]
        if d and d.get("checks", {}).get("base_vs_P2b_columns_maxabs"):
            A(f"- `{t}` 基线 TF 与 P2-b 同包列的最大绝对差（e_rot_deg / e_fing_rad / e_t_mm）：{json.dumps(d['checks']['base_vs_P2b_columns_maxabs'], ensure_ascii=False)}")
    for t in TAGS:
        pe2, pd1 = out_dir / "cols" / f"E_e2_{t}.npz", out_dir / "cols" / f"D_tf_D1_{t}.npz"
        if pe2.exists() and pd1.exists():
            e2, d1 = np.load(pe2), np.load(pd1)
            parts_ = []
            for sq in ("zgz_global", "zgz_local"):
                e1_ = e2[f"{sq}|err1"][:, 0]
                dt = np.abs(e1_[:, 0] - d1[f"base|{sq}|e_t_mm"])
                dr = np.abs(e1_[:, 1] - d1[f"base|{sq}|e_rot_geo_deg"])
                df = np.abs(e1_[:, 2] - d1[f"base|{sq}|e_fing_rad"])
                parts_.append(f"{sq}：平移 mm 均值/最大 {dt.mean():.4f}/{dt.max():.3f}，根旋转 ° {dr.mean():.4f}/{dr.max():.3f}，手指 rad {df.mean():.5f}/{df.max():.4f}")
            A(f"- `{t}` E2 的 B=15 批内无扰动对照样本 vs D 的 B=1 基线（同包，相对窗末真值的块误差之差，全部 2590 包）：" + "；".join(parts_))
    A("- 每个 q hook 的每包断言：残差质量增量 = Σ(q_原 − q_新)；s=1 的 hook 激活时输出与基线逐位相同；s=0 时 w≡0、r≡mask、H≡0（见 4.1）。")
    A("- 每个几何 hook：`rel_nonzero_zeroed` / `edge_nonzero_zeroed` 为被清零的非零元素数（>0 即证明生效）；`no_geom` 与 `xyz.geom=False` 逐位相等。")
    A("- E2/E3 不使用 hook（spec=None，hook 为恒等）；E2 的 B=15 批与单样本 forward 的差见 5.2。\n")

    # ------------------------------------------------------------------ deviations / limitations
    A("## 7. 偏离、口径选择与局限\n")
    A("- **分布外输入（单列）**：E1 的三种几何清零（dXYZ、z_prior、边向量）以及 D 中 s<1 / s>1 的 q 缩放，都是训练从未见过的输入分布（训练时 q 由关联规则给出；C0 对照是另一组权重训练出来的，不等于“冻结权重 + 清零”）。这些结果只说明冻结模型对该输入通道的**敏感性**，不能当作“该通道的因果贡献”或“重训后的消融结果”。`z_median` 同理（训练时 z_prior 是逐事件值）。D4 的 95% 事件置零亦是极端输入。")
    A("- **E2 扰动幅度与训练噪声**：训练时 prev 噪声（`TRACK.PREV_NOISE_*`）小档 sd = 5 mm / 0.05 rad / 0.05 rad（平移/根旋转/手指），大档 50 mm / 0.3 / 0.3；本探针的 ±20 mm、±10°(0.175 rad)、手指 +0.3×3 / +0.1×45 是确定性单向偏移，不同于零均值高斯噪声（相关偏移，且全指 45 维同向），因此不能直接对应训练分布。")
    A("- **恢复比的定义局限**：主恢复比 err_before 取“扰动后的前态相对窗末真值”，含真实运动量（50 ms 内手本身会动），所以对照（无扰动）的“恢复比”没有意义，只作参考；补充的 R_excess（扰动差）抵消真实运动，但依赖 pred_无扰动，是线性化读数。两者在表中并列。恢复比为池化均值之比，不是逐包比值的均值（逐包均值版在 JSON `recovery_ratio_per_packet_mean`）。")
    A("- **k 步递推**：起点每 2 包取 1 个（stride=2），k 步要求该 run 内至少还有 k 步，故 k=10 的起点数少于 k=1；bootstrap 块长按起点间隔折算为 10 个起点（≈20 步）。err_k 是相对该步窗末真值的块误差，包含无扰动对照也会有的跟踪误差，所以另报对照与残余扰动比。")
    A("- **B=15 批**：E2 把“对照 + 14 种扰动”放进同一 forward（事件相同、prev 不同）。批内逐样本结果与单样本 forward 的差见 5.2 自检（~1e-4 量级，浮点归约顺序），不影响度量到 0.01° / 0.01 mm 的读数，但不是逐位一致；D、E1、E3 与闭环全部是 B=1（主检查点的 TF 基线与 P2-b 同包列逐位一致，hook 全关的闭环与 P2-a 逐位一致，见第 6 节）。")
    A("- **D4 的候选集**：“95% 事件置零”取自 q>0 的事件（q=0 者本来就在残差槽里），保留数 = max(1, round(5%·n_q>0))；两次随机抽样（r0/r1，种子 [rep, 序列序号, 步号, 类型]）；J2 读数取两次平均并逐次列出。")
    A("- **J1 的 B 条件**是我在看数前写定的数值化（>=5% 且 >=3×s=0.25 的变化）；指引原文只说“s=0 时骤变”。")
    A("- **E3(a)** 平移 +4 px 不做图像边界裁剪（事件 x 可超过 239）；ray 直接由像素坐标算出，只有 token 的 x/W 超出 1。**(b)** 不在同 run 内 ±20 步的包不可用（见可用比例）。**(c)** 空包经 `forward_packet` 逐位返回 prev，|Δ|=0 是模型设计（ZERO_EVENT_GATE + 早退），不是学到的行为。")
    A("- **E3 期望位移** 用 prev 真值手腕关节深度（FK 关节 0）近似：4 px / fx_render × z。")
    A("- **关联区**：D 的 q 缩放不改变 vid / A / valid / hit；因此 near 事件（q<1）的 mass 与 hit 事件（q=1）的 mass 同时随 s 缩放（D1），D2/D3 分别只缩放其一。TF 包的 hit 占比、残差质量见 4.1 表的基线行。")
    A("- 本报告不对“关联是否瓶颈”下结论；各读数只按第 2 节预先写定的口径给出。")
    A("- **脚本演进（如实记录）**：先启动了 D1、D4、D23、E1、E2、E3、闭环与两个种子检查点的长跑；之后追加了 (i) `report` 部分（只读 JSON、生成本文件），(ii) `tfD1b` 路径隔离（在 `Rig` 里新增 `vin` 输入的 lm_fix 与 `slot_proj` 输入的 res_fix 两个 hook，spec=None 时恒等，对其余部分无影响），(iii) E3 的修复：首次 E3 运行里 (b) 变体在无可用包时只写入 `avail` 而未补 NaN，导致 local / overall 的 (b) 列因长度不齐被丢弃；已补齐 NaN 并**重跑了主检查点的 E3**（首次结果丢弃，本报告用的是重跑结果）。其余计算部分（hook、TF、闭环、E2）自启动后未改动，最终脚本与产生结果的版本在这些部分逐行相同。\n")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    log("wrote", out_dir / "REPORT.md")
    return L


# ============================================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, help="comma list of tfD1,tfD23,tfD4,tfE1,tfE2,tfE3,clD,clE1")
    ap.add_argument("--run-dir", default=P.RUN_DEFAULT)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--labels-dir", default=P.LABELS_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--max-steps", type=int, default=0, help="self-test: first N steps of each sequence")
    ap.add_argument("--stride", type=int, default=2, help="E2 rollouts start at every stride-th packet (0 = none)")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--reps", type=int, default=2, help="D4 random draws")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--no-assert-ref", action="store_true")
    ap.add_argument("--cl-extra", action="store_true", help="also closed-loop D2/D3 s=0")
    args = ap.parse_args()
    os.chdir(REPO)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    parts = [p for p in args.part.split(",") if p]
    if parts == ["report"]:
        make_report(args, out_dir)
        return
    with P.limited_runs(args.max_steps):
        env = make_env(args)
        log(f"device={env['device']} tag={env['tag']} ckpt={env['ckpt']} n={env['n']} parts={parts}")
        for part in parts:
            ts = time.time()
            if part.startswith("tf") and part != "tfE2" and part != "tfE3":
                part_tf(args, env, part, out_dir)
            elif part == "tfE2":
                part_e2(args, env, out_dir)
            elif part == "tfE3":
                part_e3(args, env, out_dir)
            elif part in ("clD", "clE1"):
                part_cl(args, env, part, out_dir)
            else:
                raise SystemExit(f"unknown part {part}")
            log(f"== {part} done in {time.time() - ts:.0f}s")


if __name__ == "__main__":
    main()
