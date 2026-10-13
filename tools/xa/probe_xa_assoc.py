#!/usr/bin/env python3
r"""[XYZ-ASSOC] P2: single-checkpoint probe of the ray/mesh association (xa_c1_s3407, last step).

Pre-registration: docs/EXPERIMENT_RECORDS.md, XYZ-ASSOC-20261009, section 3 (thresholds are quoted
there and implemented here verbatim; nothing in this file chooses or changes a threshold).

Read-only on the repository and on the checkpoint. The model's default behaviour is untouched: the
probe sets `model.xyz_keep_aux = True` (an existing probe switch) and, for P2-c only, replaces
`semkine.xyz_mesh.associate_rays` by a wrapper for the duration of a `with` block.

    P2-a  closed loop. `eval_track.track_sequence` under the select_checkpoint protocol (rng(0), 50 ms,
          val_core = zgz_global + zgz_local, noise scale 1) with a `state_hook` that reads, after every
          step's forward, `model.route_stats` and `model.xyz_last` (association of the 2048 sampled
          events), aligns them to the P1 source labels and counts mismatches. The hook returns `pred`
          unchanged.
    P2-b  teacher-forced stratified correlation. Per step, prev = GT at the end of the previous step
          (first step of a run: GT at the run start), no noise. m_i = mismatch mass share of the
          sampled live events, e_i = root-rotation update error (deg) and finger update error (rad).
          Spearman rho(m_i, e_i) per sequence and per fingertip_in_front stratum; binned table.
    P2-c  in-distribution intervention on the same TF packets with m_i > 0. Mismatch events get q := 0
          (their unit mass goes to the residual slot, r = 1); the control moves the same number of
          events with (about) the same q-sum, drawn at random from correctly associated events
          (rng(k), k = 0..4, averaged). Reports the relative drop of the mean root-rotation update
          error, intervention vs control, in percentage points.

Definitions (all on the <= 2048 events `stride_sample` hands to the model, in that order):
    joint mass of an event     = sum_r q * A_r [lbs_hard(vid_r) == j], j = 0..15 (as P1's `summarize`)
    event "has association"    = valid and total joint mass > 0
    pred joint                 = argmax_j joint mass (plurality), used for the exact part agreement
    mismatch event             = src_kind in {1, 2} and src_part >= 1 and has association and joint 0 holds
                                 MORE THAN HALF of the event's mass (P1's `maj0`; the pre-registered
                                 "关联多数质量落到关节 0"). The plurality variant (pred joint == 0) is counted too.
    m_i  (primary)             = #mismatch events / #live events   (each event carries unit mass)
    m_i^q (sensitivity)        = sum q over mismatch events / #live events
    M                          = #mismatch / #(src_kind in {1,2} and src_part >= 1)    ("M_all", the P1 form)
                                 and the same over those with an association ("M_has")
    exact part agreement       = pred joint == src_part, over events with src_part >= 0 and an association
                                 ("agree_exact"); over all src_part >= 0 events, unassociated counted as
                                 disagreement ("agree_exact_allhand", P1's denominator)

Usage (cwd-independent; run on the server):
    CUDA_VISIBLE_DEVICES=2 python tools/xa/probe_xa_assoc.py --part a,b,c,report
    CUDA_VISIBLE_DEVICES= python tools/xa/probe_xa_assoc.py --random-init --max-steps 10 --part b,c \
        --out-dir /tmp/xa_step2_selftest
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.dont_write_bytecode = True

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                   # noqa: E402
from model import MNISTModel                                       # noqa: E402
from semkine import eval_track as ET                               # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest   # noqa: E402
from semkine.xyz_mesh import stride_sample                         # noqa: E402
import semkine.xyz_mesh as XM                                      # noqa: E402

ST = 50                         # step = window, ms
MAX_NODES = 2048                # ENCODER_MAX_NODES
RAD2DEG = 180.0 / math.pi
RUN_DEFAULT = "outputs/semkine/xa_c1_s3407"
LABELS_DEFAULT = "outputs/xa/step1"
OUT_DEFAULT = "outputs/xa/step2"
# pre-registered thresholds (docs/EXPERIMENT_RECORDS.md, XYZ-ASSOC section 3), quoted not chosen
RHO_SUPPORT, RHO_REJECT = 0.3, 0.1
DPP_SUPPORT, DPP_REJECT = 25.0, 5.0
RA_REPRO_TOL_MM = 0.05          # the repo's reproduction gate (make_s36_row.py)
MIN_CF_FOR_DIST = 20            # steps with fewer finger-contour events are left out of per-step M quantiles
LOCAL = lambda s: "_local" in s  # noqa: E731
GROUP = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])   # joint -> palm/index/middle/pinky/ring/thumb
ROUTE_KEYS = ("xyz_valid", "xyz_hit", "xyz_resid_mass", "xyz_verts_direct", "xyz_verts_prop")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=1, ensure_ascii=False))
    log("wrote", path)


# --------------------------------------------------------------------------- sequence + labels
@dataclass
class Step:
    j: int
    run_id: int
    end: int
    prev_idx: int        # ms index of the GT state that precedes the window (run start for the first step)
    n_ev: int


class SeqCtx:
    """The evaluator's step enumeration of one sequence (same loop as `track_sequence`)."""

    def __init__(self, root: Path, legacy_dir: str, seq: str):
        self.seq = seq
        self.events, self.offsets, self.aux, self.pos51 = ET.load_sequence(root, legacy_dir, seq)
        p = root / legacy_dir / f"{seq}_tsub.npy"
        self.tsub = np.load(p, mmap_mode="r") if p.exists() else None
        runs = np.asarray(self.aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        self.steps = []
        for run_id, (a, b) in enumerate(runs):
            ends = np.arange(a + ST - 1, b, ST, dtype=np.int64)
            prev_end = int(a)
            for end in ends:
                n = int(self.offsets[int(end) + 1] - self.offsets[int(end) - ST + 1])
                self.steps.append(Step(len(self.steps), run_id, int(end), prev_end, n))
                prev_end = int(end)


class Labels:
    """P1 source labels of one sequence (`outputs/xa/step1/labels_<seq>.npz`, tools/xa/step1_source_labels.py).

    Keys used: src_part, src_kind (per event, `_window_events` order), step_ev_start (S+1 offsets),
    step_end_ms / step_prev_ms / run_id / step_a0 (per step), fingertip_in_front (per step),
    samp_local / samp_start (the sampled indices P1 used), ev_x / ev_y (event pixel, for alignment)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        z = np.load(self.path)
        self.files = list(z.files)
        self.part = z["src_part"]
        self.kind = z["src_kind"]
        self.start = z["step_ev_start"]
        self.end_ms = z["step_end_ms"].astype(np.int64)
        self.prev_ms = z["step_prev_ms"].astype(np.int64)
        self.run_id = z["run_id"].astype(np.int64)
        self.a0 = z["step_a0"].astype(np.int64)
        self.fit = z["fingertip_in_front"].astype(bool)
        self.samp = z["samp_local"]
        self.samp_start = z["samp_start"]
        self.ev_x = z["ev_x"]
        self.ev_y = z["ev_y"]

    def check_ctx(self, ctx: SeqCtx):
        n = len(ctx.steps)
        assert n <= len(self.end_ms), (ctx.seq, n, len(self.end_ms))
        e = np.array([s.end for s in ctx.steps])
        p = np.array([s.prev_idx for s in ctx.steps])
        r = np.array([s.run_id for s in ctx.steps])
        c = np.array([s.n_ev for s in ctx.steps])
        a0 = np.array([int(ctx.offsets[s.end - ST + 1]) for s in ctx.steps])
        assert (e == self.end_ms[:n]).all(), "step_end_ms differs from the evaluator enumeration"
        assert (p == self.prev_ms[:n]).all(), "step_prev_ms differs"
        assert (r == self.run_id[:n]).all(), "run_id differs"
        assert (c == np.diff(self.start)[:n]).all(), "per-step event counts differ"
        assert (a0 == self.a0[:n]).all(), "step_a0 differs"

    def step(self, st: Step, ev5=None):
        """(src_part, src_kind, fingertip_in_front) of the model's sampled events of step `st`, plus the
        sampled local indices. If `ev5` (the window's events) is given, the pixel coordinates are checked
        against the label file's, and the recomputed `stride_sample` indices against P1's."""
        a, b = int(self.start[st.j]), int(self.start[st.j + 1])
        assert b - a == st.n_ev
        idx = sampled_index(st.n_ev)
        s0, s1 = int(self.samp_start[st.j]), int(self.samp_start[st.j + 1])
        assert np.array_equal(idx, self.samp[s0:s1]), f"sampled index differs from P1 at step {st.j}"
        if ev5 is not None and st.n_ev:
            assert np.array_equal(ev5[:, 1].astype(np.int64), self.ev_x[a:b]), f"ev_x misaligned at step {st.j}"
            assert np.array_equal(ev5[:, 2].astype(np.int64), self.ev_y[a:b]), f"ev_y misaligned at step {st.j}"
        return self.part[a:b][idx], self.kind[a:b][idx], bool(self.fit[st.j]), idx


def sampled_index(n: int) -> np.ndarray:
    """The local event indices `_xyz_mesh_forward` keeps from a packet of `n` events (B = 1)."""
    src, mask = stride_sample(torch.tensor([0, n], dtype=torch.int64), MAX_NODES)
    return src[0][mask[0]].numpy().astype(np.int64)


@contextlib.contextmanager
def limited_runs(max_steps):
    """Truncate every sequence to its first `max_steps` evaluation steps (self-test only); both
    `track_sequence` and `SeqCtx` read sequences through `ET.load_sequence`, so they stay in step."""
    if not max_steps:
        yield
        return
    orig = ET.load_sequence

    def patched(root, legacy_dir, seq):
        ev, off, aux, pos51 = orig(root, legacy_dir, seq)
        aux = {k: aux[k] for k in aux.files}
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        keep, left = [], int(max_steps)
        for a, b in runs:
            n = len(np.arange(a + ST - 1, b, ST))
            if left <= 0:
                break                       # keep run indices (also of step-less runs) aligned with the full enumeration
            if n > left:
                b = a + ST * left          # exactly `left` window ends fit below b
                n = left
            keep.append((int(a), int(b)))
            left -= n
        aux["valid_runs_ms"] = np.asarray(keep, dtype=np.int64).reshape(-1, 2)
        return ev, off, aux, pos51

    ET.load_sequence = patched
    try:
        yield
    finally:
        ET.load_sequence = orig


# --------------------------------------------------------------------------- association analysis
@torch.no_grad()
def event_majority(assoc, lbs_hard):
    """Per sampled event (batch 0), numpy: plurality joint `maj`, joint-0 mass share `m0` (nan without
    association), has-association flag, q, valid, hit."""
    vid, A, q = assoc["vid"][0], assoc["A"][0].float(), assoc["q"][0].float()
    w = q.unsqueeze(-1) * A
    jm = torch.zeros(vid.shape[0], 16, dtype=torch.float32, device=vid.device)
    jm.scatter_add_(1, lbs_hard[vid], w)
    tot = jm.sum(1)
    valid = assoc["valid"][0]
    has = valid & (tot > 0)
    m0 = torch.where(has, jm[:, 0] / tot.clamp_min(1e-12), torch.full_like(tot, float("nan")))
    c = lambda t: t.cpu().numpy()            # noqa: E731
    return {"maj": c(jm.argmax(1)).astype(np.int64), "m0": c(m0), "has": c(has), "q": c(q), "valid": c(valid),
            "hit": c(assoc["hit"][0])}


def grp(j):
    return np.where(j >= 0, GROUP[np.clip(j, 0, 15)], -1)


def count_step(part, kind, ev):
    """Event counts of one packet and the masks P2-c needs. All arrays are over the N sampled events."""
    maj, m0, has, q, valid, hit = (ev[k] for k in ("maj", "m0", "has", "q", "valid", "hit"))
    hand = part >= 0
    cf = ((kind == 1) | (kind == 2)) & (part >= 1)               # finger contour foreground
    with np.errstate(invalid="ignore"):
        mis = cf & has & (m0 > 0.5)                              # joint 0 holds the majority of the event's mass
    mis_plur = cf & has & (maj == 0)
    den = hand & has
    agree = den & (maj == part)
    gagree = den & (grp(maj) == grp(part))
    c = {"n_live": int(len(part)), "n_valid": int(valid.sum()), "n_hit": int(hit.sum()), "n_has": int(has.sum()),
         "n_hand": int(hand.sum()), "n_nonhand": int((~hand).sum()),
         "valid_hand": int((valid & hand).sum()), "valid_nonhand": int((valid & ~hand).sum()),
         "hit_hand": int((hit & hand).sum()), "hit_nonhand": int((hit & ~hand).sum()),
         "n_cf": int(cf.sum()), "n_cf_has": int((cf & has).sum()), "n_mis": int(mis.sum()),
         "n_mis_plur": int(mis_plur.sum()), "n_mis_hit": int((mis & hit).sum()), "q_mis": float(q[mis].sum()),
         "n_den": int(den.sum()), "n_agree": int(agree.sum()), "n_gagree": int(gagree.sum())}
    for k in (0, 1, 2):
        c[f"den_k{k}"] = int((den & (kind == k)).sum())
        c[f"agree_k{k}"] = int((agree & (kind == k)).sum())
    return c, mis, agree


def update_errors(pred, prev, gt):
    """Update errors of one packet: pred / prev / gt are 51-vectors. (pred-prev) vs (gt-prev), as pre-registered."""
    pred, prev, gt = (np.asarray(x, np.float64).reshape(-1) for x in (pred, prev, gt))
    dp, dg = pred - prev, gt - prev
    geo = (Rotation.from_rotvec(pred[3:6]).inv() * Rotation.from_rotvec(gt[3:6])).magnitude() * RAD2DEG
    return {"e_rot_deg": float(np.linalg.norm(dp[3:6] - dg[3:6]) * RAD2DEG),
            "e_rot_geo_deg": float(geo),
            "e_fing_rad": float(np.linalg.norm(dp[6:51] - dg[6:51])),
            "e_t_mm": float(np.linalg.norm(dp[0:3] - dg[0:3]) * 1000.0),
            "gt_rot_deg": float(np.linalg.norm(dg[3:6]) * RAD2DEG),
            "gt_fing_rad": float(np.linalg.norm(dg[6:51]))}


# --------------------------------------------------------------------------- derived rates
def rates(s):
    """Derived rates from summed counts `s`."""
    d = lambda a, b: (a / b) if b else float("nan")      # noqa: E731
    return {"M_all": d(s["n_mis"], s["n_cf"]), "M_has": d(s["n_mis"], s["n_cf_has"]),
            "M_plurality_all": d(s["n_mis_plur"], s["n_cf"]),
            "m_count": d(s["n_mis"], s["n_live"]), "m_q": d(s["q_mis"], s["n_live"]),
            "agree_exact": d(s["n_agree"], s["n_den"]), "agree_group": d(s["n_gagree"], s["n_den"]),
            "agree_exact_allhand": d(s["n_agree"], s["n_hand"]), "agree_group_allhand": d(s["n_gagree"], s["n_hand"]),
            "agree_exact_k0_interior": d(s["agree_k0"], s["den_k0"]),
            "agree_exact_k1_silhouette": d(s["agree_k1"], s["den_k1"]),
            "agree_exact_k2_depthdisc": d(s["agree_k2"], s["den_k2"]),
            "valid_frac": d(s["n_valid"], s["n_live"]), "hit_frac": d(s["n_hit"], s["n_live"]),
            "valid_frac_hand": d(s["valid_hand"], s["n_hand"]), "valid_frac_nonhand": d(s["valid_nonhand"], s["n_nonhand"]),
            "hit_frac_hand": d(s["hit_hand"], s["n_hand"]), "hit_frac_nonhand": d(s["hit_nonhand"], s["n_nonhand"]),
            "cf_share": d(s["n_cf"], s["n_live"]), "mis_hit_share": d(s["n_mis_hit"], s["n_mis"])}


COUNT_KEYS = ("n_live", "n_valid", "n_hit", "n_has", "n_hand", "n_nonhand", "valid_hand", "valid_nonhand", "hit_hand",
              "hit_nonhand", "n_cf", "n_cf_has", "n_mis", "n_mis_plur", "n_mis_hit", "q_mis", "n_den", "n_agree", "n_gagree",
              "den_k0", "agree_k0", "den_k1", "agree_k1", "den_k2", "agree_k2")


def pooled(cols, sel=None):
    """Event-pooled counts and rates over the steps selected by boolean `sel` (None = all)."""
    s = {k: float(np.sum(np.asarray(cols[k])[sel] if sel is not None else np.asarray(cols[k]))) for k in COUNT_KEYS}
    n = int(len(cols["n_live"]) if sel is None else int(np.sum(sel)))
    return {"n_steps": n, "counts": s, **rates(s)}


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            **{f"p{int(q * 100)}": float(np.quantile(x, q)) for q in qs}}


# --------------------------------------------------------------------------- model
def load_everything(args):
    run = REPO / args.run_dir
    cfg_path = args.config or str(next(iter(sorted(run.glob("*.yaml"))), "")) or \
        json.loads((run / "training_metadata.json").read_text())["config_path"]
    cfg = load_config(cfg_path)
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.set_num_threads(args.threads)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    seqs = sequences_for_split(root, "val_core", mani)
    if args.random_init:
        # self-test model: seeded random weights. fc2 is zero-initialised (so a fresh model outputs prev
        # whatever the association did and every comparison would be vacuous), hence it is re-drawn small.
        torch.manual_seed(0)
        model = MNISTModel(cfg=cfg)
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            model.xyz.fc2.weight.copy_(torch.randn(model.xyz.fc2.weight.shape, generator=g) * 0.01)
            model.xyz.fc2.bias.copy_(torch.randn(model.xyz.fc2.bias.shape, generator=g) * 0.002)
        ckpt = None
        model = model.float().to(device).eval()
    else:
        if args.ckpt:
            ckpt = Path(args.ckpt)
        else:
            cands = sorted(run.glob("*step=6000.ckpt"))
            assert len(cands) == 1, f"expected exactly one step=6000 ckpt in {run}, found {cands}"
            ckpt = cands[0]
        model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device).to(device).eval()
    assert getattr(model, "xyz_mesh", False), "this probe is for ENCODER=xyz_mesh"
    return cfg, cfg_path, device, mano, root, seqs, model, ckpt


def build_ctx_labels(root, seqs, labels_dir):
    out = {}
    for seq, d in seqs:
        ctx = SeqCtx(root, d, seq)
        lab = Labels(Path(labels_dir) / f"labels_{seq}.npz")
        lab.check_ctx(ctx)
        out[seq] = (ctx, lab)
    return out


def meta(args, cfg_path, ckpt, device, extra=None):
    m = {"script": "tools/xa/probe_xa_assoc.py", "argv": sys.argv[1:], "date": time.strftime("%F %T"),
         "run_dir": args.run_dir, "config": str(cfg_path), "ckpt": None if ckpt is None else str(ckpt),
         "ckpt_mtime": None if ckpt is None else time.strftime("%F %T", time.localtime(Path(ckpt).stat().st_mtime)),
         "random_init": bool(args.random_init), "max_steps": args.max_steps,
         "device": str(device), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
         "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
         "torch": torch.__version__, "labels_dir": args.labels_dir,
         "label_files": {q.name: {"bytes": q.stat().st_size, "mtime": time.strftime("%F %T", time.localtime(q.stat().st_mtime))}
                         for q in sorted((REPO / args.labels_dir).glob("labels_zgz_*.npz")) + [REPO / args.labels_dir / "assoc_stats.json"]
                         if q.exists()}}
    m.update(extra or {})
    return m


# --------------------------------------------------------------------------- P2-a
class ClosedLoopHook:
    """`track_sequence(state_hook=...)`: reads the forward just finished, returns `pred` untouched."""

    def __init__(self, model, ctx: SeqCtx, lab: Labels, lbs_hard, check_every=25):
        self.model, self.ctx, self.lab, self.lbs_hard = model, ctx, lab, lbs_hard
        self.i = 0
        self.cols = defaultdict(list)
        self.check_every = check_every

    def __call__(self, kind, prev_t, pred):
        if kind != "step":
            return pred
        st = self.ctx.steps[self.i]
        self.i += 1
        c = self.cols
        ev5 = None
        if st.j < 20 or st.j % self.check_every == 0:
            ev5 = ET._window_events(self.ctx.events, self.ctx.offsets, self.ctx.tsub, st.end, ST)
        part_s, kind_s, fit, idx = self.lab.step(st, ev5)
        c["j"].append(st.j); c["run_id"].append(st.run_id); c["end_ms"].append(st.end); c["fit"].append(fit)
        c["n_ev"].append(st.n_ev)
        prev = prev_t.detach().double().cpu().numpy()[0]
        gprev = self.ctx.pos51[st.prev_idx].astype(np.float64)
        c["prev_err_t_mm"].append(float(np.linalg.norm(prev[0:3] - gprev[0:3]) * 1000))
        c["prev_err_rot_deg"].append(float(np.linalg.norm(prev[3:6] - gprev[3:6]) * RAD2DEG))
        c["prev_err_pose_rad"].append(float(np.linalg.norm(prev[6:51] - gprev[6:51])))
        rs = self.model.route_stats
        for k in ROUTE_KEYS:
            c[k].append(float(rs[k]) if st.n_ev and k in rs else float("nan"))
        if st.n_ev == 0:                      # empty packet: the model returns prev and xyz_last is stale
            for k in COUNT_KEYS:
                c[k].append(0)
            return pred
        xl = self.model.xyz_last
        assert int(xl["mask"].sum()) == len(idx), "sampled live count differs from stride_sample"
        ev = event_majority(xl["assoc"], self.lbs_hard)
        cnt, _, _ = count_step(part_s, kind_s, ev)
        # the aux dump and route_stats must describe the same packet
        assert abs(cnt["n_valid"] / cnt["n_live"] - float(rs["xyz_valid"])) < 1e-5, "xyz_last / route_stats valid mismatch"
        assert abs(cnt["n_hit"] / cnt["n_live"] - float(rs["xyz_hit"])) < 1e-5, "xyz_last / route_stats hit mismatch"
        for k in COUNT_KEYS:
            c[k].append(cnt[k])
        return pred


def summarize_p2a(cols):
    cols = {k: np.asarray(v) for k, v in cols.items()}
    fit = cols["fit"].astype(bool)
    per_step_M = np.where(cols["n_cf"] >= MIN_CF_FOR_DIST, cols["n_mis"] / np.maximum(cols["n_cf"], 1), np.nan)
    per_step_m = cols["n_mis"] / np.maximum(cols["n_live"], 1)
    per_step_agree = np.where(cols["n_den"] > 0, cols["n_agree"] / np.maximum(cols["n_den"], 1), np.nan)
    return {"n_steps": int(len(fit)), "n_steps_fit_true": int(fit.sum()),
            "pooled": pooled(cols),
            "pooled_fit_true": pooled(cols, fit) if fit.any() else None,
            "pooled_fit_false": pooled(cols, ~fit) if (~fit).any() else None,
            "per_step": {"M_all": quantiles(per_step_M), "min_cf_events_for_M": MIN_CF_FOR_DIST,
                         "n_steps_with_cf": int((cols["n_cf"] > 0).sum()),
                         "n_steps_M_defined": int(np.isfinite(per_step_M).sum()),
                         "m_count": quantiles(per_step_m),
                         "frac_steps_with_mismatch": float((cols["n_mis"] > 0).mean()),
                         "agree_exact": quantiles(per_step_agree)},
            "route_stats_mean": {k: float(np.nanmean(cols[k])) for k in ROUTE_KEYS},
            "closed_loop_state_error": {"prev_err_t_mm": quantiles(cols["prev_err_t_mm"]),
                                        "prev_err_rot_deg": quantiles(cols["prev_err_rot_deg"]),
                                        "prev_err_pose_rad": quantiles(cols["prev_err_pose_rad"])}}


def find_ra_reference(run: Path, step=6000):
    """Step-`step` RA of the selection grid: the selection JSON if it exists, else the select log."""
    for p in sorted(run.glob("selection_val_core_step50*.json")):
        d = json.loads(p.read_text())
        rows = [r for r in d["grid"] if r["step"] == step]
        if rows:
            return {"source": str(p.relative_to(REPO)), "overall": rows[0]["mpjpe_ra_mm"], "per_seq": rows[0]["per_seq"]}
    lp = REPO / "logs" / f"{run.name}_select.log"
    if lp.exists():
        m = re.search(rf"step={step}\s+RA=\s*([0-9.]+)", lp.read_text())
        if m:
            return {"source": str(lp.relative_to(REPO)), "overall": float(m.group(1)), "per_seq": None}
    return None


def compute_ra_check(ra, ref):
    diffs = {s: abs(ra["per_seq"][s] - ref["per_seq"][s]) for s in ra["per_seq"]} if ref.get("per_seq") else {}
    d_all = abs(ra["overall"] - ref["overall"])
    return {"reference": ref, "this_run_overall": ra["overall"], "abs_diff_overall": d_all, "abs_diff_per_seq": diffs,
            "tolerance_mm": RA_REPRO_TOL_MM,
            "match": bool(d_all < RA_REPRO_TOL_MM and all(v < RA_REPRO_TOL_MM for v in diffs.values()))}


def p1_reference(path: Path):
    """The slice of P1's assoc_stats.json this report compares against (current rule, sampled events)."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for seq, S in d["sets"]["sampled"].items():
        c = S["R0"]["cur"]
        out[seq] = {"n_steps": d["checks"][seq]["n_steps"], "M_by_regime": {rg: S[rg]["cur"]["M"]["maj0_rate"] for rg in S},
                    "R0": {k: c[k] for k in ("all_events", "hand", "M", "fip0_hand", "fip0_M", "fip1_hand", "fip1_M")}}
    return out


def run_p2a(args, env, out_dir):
    cfg, cfg_path, device, mano, root, seqs, model, ckpt = env["cfg"], env["cfg_path"], env["device"], env["mano"], \
        env["root"], env["seqs"], env["model"], env["ckpt"]
    lbs_hard = model.xyz.lbs_hard.to(device)
    model.xyz_keep_aux = True
    t0 = time.time()
    rng = np.random.default_rng(0)                         # one rng across the sequences, as select_checkpoint
    rng_ref = np.random.default_rng(0)
    res, hooks, nohook = {}, {}, {}
    for seq, d in seqs:
        ctx, lab = env["cl"][seq]
        if args.check_nohook:
            model.xyz_keep_aux = False
            nohook[seq] = ET.track_sequence(model, mano, cfg, root, d, seq, ST, device, rng_ref, 1.0, 1.0, None)
            model.xyz_keep_aux = True
        hook = ClosedLoopHook(model, ctx, lab, lbs_hard)
        ts = time.time()
        res[seq] = ET.track_sequence(model, mano, cfg, root, d, seq, ST, device, rng, 1.0, 1.0, None, state_hook=hook)
        assert hook.i == len(ctx.steps) == res[seq]["n_frames"], (hook.i, len(ctx.steps), res[seq]["n_frames"])
        hooks[seq] = hook
        log(f"P2-a {seq}: {hook.i} steps RA={res[seq]['mpjpe_ra_mm']:.4f} ({time.time() - ts:.0f}s)")
        if args.check_nohook:
            keys = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "jitter_all_mm_per_step")
            same = all(res[seq][k] == nohook[seq][k] for k in keys)
            log(f"   hook vs no-hook bit-identical metrics: {same}")
            assert same, "state_hook changed the closed loop"
    model.xyz_keep_aux = False
    n_tot = sum(r["n_frames"] for r in res.values())
    ra_all = sum(r["mpjpe_ra_mm"] * r["n_frames"] for r in res.values()) / max(n_tot, 1)
    out = {"meta": meta(args, cfg_path, ckpt, device, {"seconds": time.time() - t0, "protocol": "rng(0), 50 ms, val_core, noise 1.0, shared rng across sequences"}),
           "sequences": {}, "ra": {"overall": ra_all, "per_seq": {s: r["mpjpe_ra_mm"] for s, r in res.items()},
                                   "n_frames": {s: r["n_frames"] for s, r in res.items()}}}
    if args.check_nohook:
        out["hook_vs_nohook_identical"] = True
    ref = find_ra_reference(REPO / args.run_dir) if not (args.random_init or args.max_steps) else None
    out["ra_check"] = compute_ra_check(out["ra"], ref) if ref else {"reference": None, "note": "no selection result available (yet), or self-test; the report step fills it in"}
    if ref:
        log("RA check vs", ref["source"], "->", out["ra_check"]["match"], f"diff={out['ra_check']['abs_diff_overall']:.5f}")
    for seq, hook in hooks.items():
        s = summarize_p2a(hook.cols)
        s["columns"] = {k: np.asarray(v) for k, v in hook.cols.items()}
        out["sequences"][seq] = s
    out["p1_reference"] = p1_reference(REPO / args.labels_dir / "assoc_stats.json")
    dump(out_dir / "p2a_closedloop.json", out)
    return out


# --------------------------------------------------------------------------- TF packets
def tf_packets(ctx: SeqCtx, device):
    """Teacher-forced eval packets: prev = GT at the end of the previous step (run start for the first), no noise."""
    betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(ctx.aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    for st in ctx.steps:
        ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, st.end, ST)
        prev = ctx.pos51[st.prev_idx].astype(np.float32)
        gt = ctx.pos51[st.end].astype(np.float32)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        yield st, ev5, prev, gt, ET.make_eval_packet(ev5, prev_t, betas, K, ST, device)


def set_context(model, ctx, device):
    betas = torch.tensor(ctx.aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(ctx.aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)             # as track_sequence does


# --------------------------------------------------------------------------- statistics helpers
def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sstats.spearmanr(x, y).statistic)


BLOCK = 20          # moving-block bootstrap length in steps (1 s): consecutive packets are strongly autocorrelated


def block_idx(n, rng, L=BLOCK):
    L = max(1, min(L, n))
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, nb)
    return (starts[:, None] + np.arange(L)[None, :]).reshape(-1)[:n]


def spearman_ci(x, y, B=1000, seed=0):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if len(x) < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        i = block_idx(len(x), rng)
        v.append(spearman(x[i], y[i]))
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def partial_spearman(x, y, z):
    """Spearman partial correlation of x, y given z (rank-based)."""
    rx, ry, rz = (sstats.rankdata(a) for a in (x, y, z))
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    rxy, rxz, ryz = r(rx, ry), r(rx, rz), r(ry, rz)
    den = math.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    return (rxy - rxz * ryz) / den


BINS = (("m=0", 0.0, 0.0), ("(0,5%]", 0.0, 0.05), ("(5%,15%]", 0.05, 0.15), (">15%", 0.15, np.inf))


def bin_table(m, e_rot, e_fing):
    out = []
    for name, lo, hi in BINS:
        sel = (m == 0) if name == "m=0" else ((m > lo) & (m <= hi))
        out.append({"bin": name, "n": int(sel.sum()),
                    "e_rot_deg_mean": float(e_rot[sel].mean()) if sel.any() else None,
                    "e_rot_deg_median": float(np.median(e_rot[sel])) if sel.any() else None,
                    "e_fing_rad_mean": float(e_fing[sel].mean()) if sel.any() else None,
                    "m_mean": float(m[sel].mean()) if sel.any() else None})
    return out


def rho_block(c, sel):
    """Spearman block over the packets selected by `sel`."""
    m, mq, er, eg, ef, gt, gf = (np.asarray(c[k])[sel] for k in ("m_count", "m_q", "e_rot_deg", "e_rot_geo_deg", "e_fing_rad", "gt_rot_deg", "gt_fing_rad"))
    n = int(sel.sum())
    b = {"n_packets": n, "frac_m_gt0": float((m > 0).mean()) if n else None,
         "m_count_mean": float(m.mean()) if n else None,
         "rho_m_e_rot": spearman(m, er), "rho_m_e_rot_ci95": spearman_ci(m, er),
         "rho_mq_e_rot": spearman(mq, er), "rho_m_e_rot_geo": spearman(m, eg), "rho_m_e_fing": spearman(m, ef),
         "e_rot_deg_mean": float(er.mean()) if n else None, "e_fing_rad_mean": float(ef.mean()) if n else None}
    if n >= 10 and np.ptp(m) > 0:
        b["confound"] = {"rho_m_gtrot": spearman(m, gt), "rho_e_rot_gtrot": spearman(er, gt),
                         "partial_rho_m_e_rot_given_gtrot": partial_spearman(m, er, gt),
                         "rho_m_gtfing": spearman(m, gf), "rho_e_fing_gtfing": spearman(ef, gf),
                         "partial_rho_m_e_fing_given_gtfing": partial_spearman(m, ef, gf)}
        b["bins"] = bin_table(m, er, ef)
    return b


# --------------------------------------------------------------------------- P1 cross-check
class P1Events:
    """P1's per-event R0 / current-rule outputs on the sampled events (`assoc_events_<seq>.npz`, if present):
    the TF packets of P2-b are exactly P1's R0 inputs, so the two association implementations must agree
    event by event (modulo CPU/GPU float differences in the FK)."""

    KEYS = ("R0_cur_pred", "R0_cur_maj0", "R0_cur_atype", "samp_step")

    def __init__(self, path: Path, lab: Labels):
        z = np.load(path)
        self.d = {k: z[k] for k in self.KEYS}
        self.samp_start = lab.samp_start
        self.cnt = defaultdict(int)

    def compare(self, j, ev):
        a, b = int(self.samp_start[j]), int(self.samp_start[j + 1])
        assert (self.d["samp_step"][a:b] == j).all(), "P1 event file is not aligned with the label sampling"
        pred, maj0, at = self.d["R0_cur_pred"][a:b], self.d["R0_cur_maj0"][a:b], self.d["R0_cur_atype"][a:b]
        has1 = pred >= 0
        both = has1 & ev["has"]
        with np.errstate(invalid="ignore"):
            m0 = ev["has"] & (ev["m0"] > 0.5)
        c = self.cnt
        c["n_events"] += len(pred)
        c["has_differs"] += int((has1 != ev["has"]).sum())
        c["pred_differs"] += int((pred[both] != ev["maj"][both]).sum())
        c["maj0_differs"] += int((maj0 != m0).sum())
        c["hit_differs"] += int(((at == 2) != ev["hit"]).sum())


def p1_events_for(seq, lab, args):
    path = REPO / args.labels_dir / f"assoc_events_{seq}.npz"
    return P1Events(path, lab) if path.exists() else None


# --------------------------------------------------------------------------- P2-b
def run_p2b(args, env, out_dir):
    device, model, cfg_path, ckpt = env["device"], env["model"], env["cfg_path"], env["ckpt"]
    lbs_hard = model.xyz.lbs_hard.to(device)
    model.xyz_keep_aux = True
    t0 = time.time()
    out = {"sequences": {}}
    for seq, d in env["seqs"]:
        ctx, lab = env["cl"][seq]
        set_context(model, ctx, device)
        cols = defaultdict(list)
        p1ev = p1_events_for(seq, lab, args)
        ts = time.time()
        for st, ev5, prev, gt, batch in tf_packets(ctx, device):
            part_s, kind_s, fit, idx = lab.step(st, ev5)
            with torch.no_grad():
                pred = model.forward_packet(batch)
            err = update_errors(pred.double().cpu().numpy()[0], prev, gt)
            if st.n_ev:
                xl = model.xyz_last
                assert int(xl["mask"].sum()) == len(idx)
                ev = event_majority(xl["assoc"], lbs_hard)
                cnt, _, _ = count_step(part_s, kind_s, ev)
                if p1ev is not None:
                    p1ev.compare(st.j, ev)
            else:
                cnt = {k: 0 for k in COUNT_KEYS}
            rs = model.route_stats
            for k, v in {**err, **cnt, "fit": fit, "j": st.j, "end_ms": st.end, "run_id": st.run_id,
                         **{k_: (float(rs[k_]) if st.n_ev and k_ in rs else float("nan")) for k_ in ROUTE_KEYS}}.items():
                cols[k].append(v)
            cols["m_count"].append(cnt["n_mis"] / max(cnt["n_live"], 1))
            cols["m_q"].append(cnt["q_mis"] / max(cnt["n_live"], 1))
            if (st.j + 1) % 200 == 0:
                log(f"P2-b {seq}: {st.j + 1}/{len(ctx.steps)} steps ({time.time() - ts:.0f}s)")
        c = {k: np.asarray(v) for k, v in cols.items()}
        fit = c["fit"].astype(bool)
        allp = np.ones(len(fit), bool)
        out["sequences"][seq] = {"all": rho_block(c, allp),
                                 "fingertip_in_front_true": rho_block(c, fit),
                                 "fingertip_in_front_false": rho_block(c, ~fit),
                                 "pooled_counts": pooled(c),
                                 "pooled_fit_true": pooled(c, fit) if fit.any() else None,
                                 "pooled_fit_false": pooled(c, ~fit) if (~fit).any() else None,
                                 "route_stats_mean": {k_: float(np.nanmean(c[k_])) for k_ in ROUTE_KEYS},
                                 "p1_event_crosscheck": dict(p1ev.cnt) if p1ev is not None else None,
                                 "columns": c}
        log(f"P2-b {seq}: n={len(fit)} rho(m,e_rot)={out['sequences'][seq]['all']['rho_m_e_rot']:.4f}")
    model.xyz_keep_aux = False
    out["meta"] = meta(args, cfg_path, ckpt, device, {"seconds": time.time() - t0, "protocol": "TF, prev = GT at previous step end (run start for first step), no noise",
                                                      "m_definition": "primary m_count = #mismatch / #live sampled events (unit mass per event); sensitivity m_q = sum q over mismatch / #live"})
    local = [s for s in out["sequences"] if LOCAL(s)]
    if local:
        r = out["sequences"][local[0]]["all"]["rho_m_e_rot"]
        out["verdict_local"] = verdict_rho(r)
    dump(out_dir / "p2b_tf_corr.json", out)
    return out


def verdict_rho(r):
    if not math.isfinite(r):
        return {"rho_local": None, "verdict": "undefined"}
    v = "L2 支持" if r >= RHO_SUPPORT else ("L2 不支持" if r <= RHO_REJECT else "不下结论")
    return {"rho_local": r, "thresholds": {"support_ge": RHO_SUPPORT, "reject_le": RHO_REJECT}, "verdict": v}


# --------------------------------------------------------------------------- P2-c
class AssocPatch:
    """Context manager: `semkine.xyz_mesh.associate_rays` -> wrapper that zeroes `q` of the events in `self.sel`
    (indices into the packet's sampled events). `_xyz_mesh_forward` imports the function inside the call, so the
    module attribute is what it sees. Restored on exit."""

    def __init__(self):
        self.sel = None
        self.calls = 0
        self.zeroed = 0
        self.orig = None

    def wrapper(self, *a, **kw):
        out = self.orig(*a, **kw)
        self.calls += 1
        if self.sel is not None and len(self.sel):
            q = out["q"].clone()
            assert q.shape[0] == 1
            q[0, self.sel] = 0.0
            out["q"] = q
            self.zeroed += int(len(self.sel))
        return out

    def __enter__(self):
        self.orig = XM.associate_rays
        XM.associate_rays = self.wrapper
        return self

    def __exit__(self, *exc):
        XM.associate_rays = self.orig
        return False


Q_EDGES = np.array([0.25, 0.5, 0.75, 1.0])       # q bins: [0,.25) [.25,.5) [.5,.75) [.75,1) [1]  (1 = a surface hit)


def pick_control(rng, q, pool, target_q):
    """Random events from `pool`, as many as `target_q`, stratified so the q-sum is matched in distribution."""
    need = np.bincount(np.digitize(target_q, Q_EDGES), minlength=5)
    pb = np.digitize(q[pool], Q_EDGES)
    chosen = []
    for b in range(5):
        cand = pool[pb == b]
        k = min(int(need[b]), len(cand))
        if k:
            chosen.append(rng.choice(cand, size=k, replace=False))
    chosen = np.concatenate(chosen) if chosen else np.zeros(0, np.int64)
    short = len(target_q) - len(chosen)
    if short > 0:
        rest = np.setdiff1d(pool, chosen)
        if len(rest):
            chosen = np.concatenate([chosen, rng.choice(rest, size=min(short, len(rest)), replace=False)])
    return chosen.astype(np.int64)


def p2c_stats(P, sel, n_ctrl, B=2000, seed=0):
    """Relative drops of the mean update error, intervention vs control, over the packets in `sel`."""
    out = {"n_packets": int(sel.sum())}
    if not sel.any():
        return out
    for tag, key in (("rot", "rot_deg"), ("fing", "fing_rad")):
        e0 = P[f"e0_{key}"][sel]
        ei = P[f"ei_{key}"][sel]
        ek = P[f"ek_{key}"][sel]                      # (n, K)
        E0, Ei, Ek = e0.mean(), ei.mean(), ek.mean(0)
        rd_i = 1 - Ei / E0
        rd_k = 1 - Ek / E0
        rd_c = float(rd_k.mean())
        d = {"E0_mean": float(E0), "E_int_mean": float(Ei), "E_ctl_mean_over_k": float(Ek.mean()),
             "E_ctl_per_k": Ek.tolist(), "rel_drop_int": float(rd_i), "rel_drop_ctl": rd_c,
             "rel_drop_ctl_per_k": rd_k.tolist(), "delta_pp": float(100 * (rd_i - rd_c))}
        rng = np.random.default_rng(seed)
        n = len(e0)
        bs = []
        for _ in range(B):
            i = block_idx(n, rng)
            m0 = e0[i].mean()
            bs.append(100 * ((1 - ei[i].mean() / m0) - (1 - ek[i].mean(0) / m0).mean()))
        d["delta_pp_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[tag] = d
    out["n_mis_mean"] = float(P["n_mis"][sel].mean())
    out["q_mis_mean"] = float(P["q_mis"][sel].mean())
    out["n_ctrl_mean"] = float(P["n_ctrl"][sel].mean())
    out["q_ctrl_mean"] = float(P["q_ctrl"][sel].mean())
    out["q_sum_ratio_ctrl_over_int"] = float(P["q_ctrl"][sel].sum() / max(P["q_mis"][sel].sum(), 1e-12))
    out["frac_packets_control_short"] = float((P["n_ctrl_min"][sel] < P["n_mis"][sel]).mean())
    out["m_count_mean"] = float(P["m_count"][sel].mean())
    return out


def verdict_dpp(d):
    if d is None or not math.isfinite(d):
        return {"delta_pp_local": None, "verdict": "undefined"}
    v = "L2 支持" if d >= DPP_SUPPORT else ("L2 不支持" if d <= DPP_REJECT else "不下结论")
    return {"delta_pp_local": d, "thresholds": {"support_ge_pp": DPP_SUPPORT, "reject_le_pp": DPP_REJECT}, "verdict": v}


def run_p2c(args, env, out_dir):
    device, model, cfg_path, ckpt = env["device"], env["model"], env["cfg_path"], env["ckpt"]
    lbs_hard = model.xyz.lbs_hard.to(device)
    n_ctrl = args.n_ctrl
    model.xyz_keep_aux = True
    t0 = time.time()
    out = {"sequences": {}}
    patch = AssocPatch()
    with patch:
        for seq, d in env["seqs"]:
            ctx, lab = env["cl"][seq]
            set_context(model, ctx, device)
            rngs = [np.random.default_rng(k) for k in range(n_ctrl)]
            cols = defaultdict(list)
            n_pk = 0
            ts = time.time()
            for st, ev5, prev, gt, batch in tf_packets(ctx, device):
                n_pk += 1
                if st.n_ev == 0:
                    continue
                part_s, kind_s, fit, idx = lab.step(st, ev5)
                patch.sel = None
                with torch.no_grad():
                    pred0 = model.forward_packet(batch)
                xl = model.xyz_last
                ev = event_majority(xl["assoc"], lbs_hard)
                q = ev["q"]
                cnt, mis, agree = count_step(part_s, kind_s, ev)
                if cnt["n_mis"] == 0:
                    continue
                r0 = float(xl["r"].sum())
                e0 = update_errors(pred0.double().cpu().numpy()[0], prev, gt)

                def forward_with(sel):
                    patch.sel = torch.from_numpy(np.asarray(sel, np.int64)).to(device)
                    with torch.no_grad():
                        p = model.forward_packet(batch)
                    x2 = model.xyz_last
                    s_t = patch.sel
                    # the wrapper really took effect: no vertex mass, full residual mass on the chosen events
                    assert float(x2["w"][0, s_t].abs().sum()) == 0.0, "w of the selected events is not zero"
                    assert bool((x2["r"][0, s_t] == 1.0).all()), "r of the selected events is not one"
                    qs = float(q[np.asarray(sel)].sum())
                    assert abs(float(x2["r"].sum()) - r0 - qs) < 5e-3 + 1e-3 * qs, "residual mass did not rise by sum q"
                    return update_errors(p.double().cpu().numpy()[0], prev, gt)

                sel = np.flatnonzero(mis)
                ei = forward_with(sel)
                pool = np.flatnonzero(agree)
                ek = []
                n_ctrl_k, q_ctrl_k = [], []
                for k in range(n_ctrl):
                    ctrl = pick_control(rngs[k], q, pool, q[sel])
                    n_ctrl_k.append(len(ctrl))
                    q_ctrl_k.append(float(q[ctrl].sum()))
                    ek.append(forward_with(ctrl) if len(ctrl) else dict(e0))
                row = {"j": st.j, "end_ms": st.end, "fit": fit, "m_count": cnt["n_mis"] / cnt["n_live"],
                       "n_mis": cnt["n_mis"], "q_mis": cnt["q_mis"], "n_live": cnt["n_live"],
                       "n_ctrl": float(np.mean(n_ctrl_k)), "n_ctrl_min": int(np.min(n_ctrl_k)), "q_ctrl": float(np.mean(q_ctrl_k)),
                       "n_pool": int(len(pool))}
                for key in ("rot_deg", "fing_rad"):
                    row[f"e0_{key}"] = e0[f"e_{key}"]
                    row[f"ei_{key}"] = ei[f"e_{key}"]
                    row[f"ek_{key}"] = [e[f"e_{key}"] for e in ek]
                for k_, v_ in row.items():
                    cols[k_].append(v_)
                if len(cols["j"]) % 100 == 0:
                    log(f"P2-c {seq}: {len(cols['j'])} intervention packets, step {st.j + 1}/{len(ctx.steps)} ({time.time() - ts:.0f}s)")
            P = {k: np.asarray(v) for k, v in cols.items()}
            seqres = {"n_packets_total": n_pk, "n_packets_m_gt0": int(len(P.get("j", [])))}
            if len(P.get("j", [])):
                fit = P["fit"].astype(bool)
                seqres["all"] = p2c_stats(P, np.ones(len(fit), bool), n_ctrl)
                seqres["fingertip_in_front_true"] = p2c_stats(P, fit, n_ctrl)
                seqres["fingertip_in_front_false"] = p2c_stats(P, ~fit, n_ctrl)
                seqres["columns"] = P
            out["sequences"][seq] = seqres
            log(f"P2-c {seq}: {seqres['n_packets_m_gt0']}/{n_pk} packets with m>0")
    assert patch.calls > 0, "the associate_rays wrapper was never called"
    assert XM.associate_rays is patch.orig, "associate_rays was not restored"
    model.xyz_keep_aux = False
    out["meta"] = meta(args, cfg_path, ckpt, device, {
        "seconds": time.time() - t0, "wrapper_calls": patch.calls, "events_zeroed_total": patch.zeroed, "n_ctrl": n_ctrl,
        "protocol": "TF packets of P2-b with m>0; intervention q:=0 on mismatch events; control q:=0 on an equal number of "
                    "correctly associated events (valid, src_part>=0, majority joint == src_part), q-stratified, rng(k) k=0..4"})
    local = [s for s in out["sequences"] if LOCAL(s)]
    if local and "all" in out["sequences"][local[0]]:
        out["verdict_local"] = verdict_dpp(out["sequences"][local[0]]["all"]["rot"]["delta_pp"])
    dump(out_dir / "p2c_intervention.json", out)
    return out


# --------------------------------------------------------------------------- report
def f(x, nd=3, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:.{nd - 1}f}%" if pct else f"{x:.{nd}f}"


def make_report(args, out_dir):
    L = []
    A = json.loads((out_dir / "p2a_closedloop.json").read_text()) if (out_dir / "p2a_closedloop.json").exists() else None
    B = json.loads((out_dir / "p2b_tf_corr.json").read_text()) if (out_dir / "p2b_tf_corr.json").exists() else None
    C = json.loads((out_dir / "p2c_intervention.json").read_text()) if (out_dir / "p2c_intervention.json").exists() else None
    if A and not (A["meta"]["random_init"] or A["meta"]["max_steps"]):
        ref = find_ra_reference(REPO / A["meta"]["run_dir"])
        if ref:                                    # the selection run may have finished after P2-a
            A["ra_check"] = compute_ra_check(A["ra"], ref)
            dump(out_dir / "p2a_closedloop.json", A)
    L.append("# [XYZ-ASSOC] P2 单检查点探针报告（xa_c1_s3407，末步 step 6000）\n")
    mm = (A or B or C)["meta"]
    L.append(f"- 脚本：`tools/xa/probe_xa_assoc.py`；产物：`outputs/xa/step2/`。预注册：`docs/EXPERIMENT_RECORDS.md#xyz-assoc-20261009` 第 3 节（阈值原样引用，未改）。")
    L.append(f"- 检查点：`{mm['ckpt']}`（mtime {mm['ckpt_mtime']}）；配置：`{mm['config']}`；设备：{mm['device']}"
             f"（CUDA_VISIBLE_DEVICES={mm['cuda_visible_devices']}，{mm['gpu_name']}）；torch {mm['torch']}。")
    L.append(f"- 标签：`{mm['labels_dir']}/labels_zgz_{{global,local}}.npz`（P1 来源标签；本探针对事件像素坐标、抽样索引、步枚举逐项校验一致）。")
    L.append("- 抽样口径：模型实际看到的 `stride_sample(ptr, 2048)` 事件（≤2048/步），闭环与 TF 均如此；序列：local = zgz_local，global = zgz_global。")
    L.append("- 定义（对 ≤2048 抽样事件）：联合质量 = Σ_r q·A_r 按硬 LBS 关节累加；关联存在 = valid 且联合质量>0；**错配事件** = src_kind∈{1,2} 且 src_part≥1 且有关联且关节 0 的质量占该事件总质量 **> 1/2**（P1 的 `maj0` 口径）；"
             "m_i（主）= 错配事件数 / 抽样内 live 事件数（每事件单位质量），m_i^q（敏感性）= 错配事件 q 之和 / live 事件数；M_all = 错配数 / 全部手指轮廓前景事件数（P1 的 M），M_has = 错配数 / 其中有关联者；"
             "精确一致率 = argmax 关节 == src_part（分母：src_part≥0 且有关联；另列分母为全部手部事件者，与 P1 `exact` 同）。")
    seqs = lambda D: list(D["sequences"].keys())     # noqa: E731
    L.append("## 判据汇总（预注册第 3 节；本节由脚本从各 JSON 生成）\n")
    L.append("| 探针 | 判据 | 本次数值（序列 / 包数） | 判定 |")
    L.append("|---|---|---|---|")
    L.append("| P2-a | 无预注册阈值，仅记录闭环口径 | 见下 | 不适用 |")
    if B and B.get("verdict_local"):
        v = B["verdict_local"]
        ls = [s for s in seqs(B) if LOCAL(s)][0]
        L.append(f"| P2-b | ρ(local) ≥ {RHO_SUPPORT} 支持 / ≤ {RHO_REJECT} 不支持 / 其间不下结论 | zgz_local，{B['sequences'][ls]['all']['n_packets']} 包，ρ = {f(v['rho_local'], 3)}"
                 f" [{f(B['sequences'][ls]['all']['rho_m_e_rot_ci95'][0], 3)}, {f(B['sequences'][ls]['all']['rho_m_e_rot_ci95'][1], 3)}] | **{v['verdict']}** |")
    if C and C.get("verdict_local"):
        v = C["verdict_local"]
        ls = [s for s in seqs(C) if LOCAL(s)][0]
        ci = C["sequences"][ls]["all"]["rot"]["delta_pp_ci95"]
        L.append(f"| P2-c | Δ(local，干预相对下降 − 对照相对下降) ≥ {DPP_SUPPORT:.0f} pp 支持 / ≤ {DPP_REJECT:.0f} pp 不支持 / 其间不下结论 | zgz_local，{C['sequences'][ls]['all']['n_packets']} 包(m>0)，Δ = {f(v['delta_pp_local'], 2)} pp [{f(ci[0], 2)}, {f(ci[1], 2)}] | **{v['verdict']}** |")
    L.append("")
    if A:
        L.append("## P2-a 闭环关联统计（无预注册判据，仅记录）\n")
        ra = A["ra"]
        L.append("### 闭环 RA 核对\n")
        L.append("| 序列 | 步数 | 本次 RA (mm) | 选点结果 step 6000 RA | |差| |")
        L.append("|---|---|---|---|---|")
        ck = A.get("ra_check", {})
        ref = ck.get("reference") or {}
        for s in seqs(A):
            rr = (ref.get("per_seq") or {}).get(s)
            L.append(f"| {s} | {ra['n_frames'][s]} | {f(ra['per_seq'][s], 4)} | {f(rr, 4)} | {f((ck.get('abs_diff_per_seq') or {}).get(s), 5)} |")
        L.append(f"| 全体（帧加权） | {sum(ra['n_frames'].values())} | {f(ra['overall'], 4)} | {f(ref.get('overall'), 4)} | {f(ck.get('abs_diff_overall'), 5)} |")
        if ref:
            L.append(f"\n参考来源：`{ref['source']}`；容差 {ck.get('tolerance_mm')} mm（仓库复现门）；结论：**{'一致' if ck.get('match') else '不一致或未比较'}**。")
        else:
            L.append("\n选点结果（selection JSON / select 日志）在本报告生成时尚不存在，RA 一致性**未核对**。")
        L.append("\n### 闭环关联统计与 P1 R0 口径对照\n")
        L.append("三个来源：**闭环** = 本探针（prev = 模型自身上一步输出，`track_sequence` 协议）；**TF(R0)** = 本探针 P2-b 的同批包（prev = 窗首真值，未加噪）；"
                 "**P1 R0** = `outputs/xa/step1/assoc_stats.json`（`sets.sampled.<序列>.R0.cur`，P1 用库函数 `associate_rays` 在窗首真值网格上算，同一批 ≤2048 抽样事件）。"
                 "事件池化口径（各步事件数求和后相除）；P1 的“精确一致率(有关联)”取其 `exact_given_valid`。\n")
        L.append("| 序列 | 层 | 来源 | 步数 | M_all | M_has | M(plurality) | 精确一致率(有关联的手部事件) | 精确一致率(全部手部事件) | 组一致率(有关联) | valid | hit | resid 质量 | 手指轮廓前景占比 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        p1 = A.get("p1_reference") or {}
        strata = (("全部", "pooled", "pooled_counts", "pooled_counts", None), ("fingertip_in_front=真", "pooled_fit_true", "pooled_fit_true", "pooled_fit_true", "fip1"),
                  ("fingertip_in_front=假", "pooled_fit_false", "pooled_fit_false", "pooled_fit_false", "fip0"))

        def row(sq, tag, src, P, resid, n_steps):
            return (f"| {sq} | {tag} | {src} | {n_steps} | {f(P['M_all'], 3, True)} | {f(P['M_has'], 3, True)} | {f(P['M_plurality_all'], 3, True)} | {f(P['agree_exact'], 3, True)} | "
                    f"{f(P['agree_exact_allhand'], 3, True)} | {f(P['agree_group'], 3, True)} | {f(P['valid_frac'], 3, True)} | {f(P['hit_frac'], 3, True)} | {f(resid, 3, True)} | {f(P['cf_share'], 3, True)} |")

        for sq in seqs(A):
            for tag, ka, kb, _, fk in strata:
                PA = A["sequences"][sq].get(ka)
                if not PA:
                    continue
                L.append(row(sq, tag, "闭环", PA, A["sequences"][sq]["route_stats_mean"]["xyz_resid_mass"] if ka == "pooled" else None, PA["n_steps"]))
                PB = B["sequences"][sq].get(kb) if B else None
                if PB:
                    L.append(row(sq, tag, "TF(R0)", PB, B["sequences"][sq]["route_stats_mean"]["xyz_resid_mass"] if kb == "pooled_counts" else None, PB["n_steps"]))
                o = (p1.get(sq) or {}).get("R0")
                if o:
                    if fk is None:
                        e, h, m = o["all_events"], o["hand"], o["M"]
                    else:
                        e, h, m = None, o[f"{fk}_hand"], o[f"{fk}_M"]
                    L.append(f"| {sq} | {tag} | P1 R0 | {(p1[sq].get('n_steps') if fk is None else '')} | {f(m['maj0_rate'], 3, True)} | {f(m['maj0_given_valid'], 3, True)} | {f(m['plur0_rate'], 3, True)} | "
                             f"{f(h['exact_given_valid'], 3, True)} | {f(h['exact'], 3, True)} | {f(h['group_given_valid'], 3, True)} | {f(e['valid_rate'], 3, True) if e else ''} | "
                             f"{f(e['hit_rate'], 3, True) if e else ''} | {f(e['resid_mean'], 3, True) if e else ''} | {f(m['n'] / e['n'], 3, True) if e else ''} |")
        if p1:
            L.append("\nM 随 prev 质量的变化（P1 现行规则，抽样事件，池化）与本探针闭环 / TF 的 M_all：\n")
            L.append("| 序列 | P1 R3(逐毫秒真值) | P1 R0(窗首真值) | P1 R1s(+小噪声) | P1 R1l(+大噪声) | 本探针 TF(R0) | 本探针 闭环 |")
            L.append("|---|---|---|---|---|---|---|")
            for sq in seqs(A):
                mb = (p1.get(sq) or {}).get("M_by_regime", {})
                tf = B["sequences"][sq]["pooled_counts"]["M_all"] if B else None
                L.append(f"| {sq} | {f(mb.get('R3'), 3, True)} | {f(mb.get('R0'), 3, True)} | {f(mb.get('R1s'), 3, True)} | {f(mb.get('R1l'), 3, True)} | {f(tf, 3, True)} | {f(A['sequences'][sq]['pooled']['M_all'], 3, True)} |")
        else:
            L.append("\nP1 的 `assoc_stats.json` 在 P2-a 运行时不存在，未并列。")
        L.append("\n事件池化口径（各步事件数求和后相除）。逐步分布（仅 n_cf≥%d 的步）：\n" % MIN_CF_FOR_DIST)
        L.append("| 序列 | 量 | 步数 | 均值 | p10 | p50 | p90 |")
        L.append("|---|---|---|---|---|---|---|")
        for s in seqs(A):
            ps = A["sequences"][s]["per_step"]
            for name, key in (("逐步 M_all", "M_all"), ("逐步 m_i（错配数/live）", "m_count"), ("逐步精确一致率", "agree_exact")):
                q = ps[key]
                if q.get("n", 0):
                    L.append(f"| {s} | {name} | {q['n']} | {f(q['mean'], 3, True)} | {f(q['p10'], 3, True)} | {f(q['p50'], 3, True)} | {f(q['p90'], 3, True)} |")
        L.append("\n闭环 route_stats 均值（逐步平均）与 prev 偏离真值的程度（逐步均值）：\n")
        L.append("| 序列 | valid | hit | resid_mass | verts_direct | verts_prop | prev 根平移误差 mm | prev 根旋转误差 ° | prev 手指误差 rad |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in seqs(A):
            S = A["sequences"][s]
            r = S["route_stats_mean"]
            e = S["closed_loop_state_error"]
            L.append(f"| {s} | {f(r['xyz_valid'], 3, True)} | {f(r['xyz_hit'], 3, True)} | {f(r['xyz_resid_mass'], 3, True)} | {f(r['xyz_verts_direct'], 1)} | {f(r['xyz_verts_prop'], 1)} | "
                     f"{f(e['prev_err_t_mm']['mean'], 2)} | {f(e['prev_err_rot_deg']['mean'], 2)} | {f(e['prev_err_pose_rad']['mean'], 3)} |")
    if B:
        L.append("\n## P2-b 真值前态 TF 分层相关\n")
        L.append(f"prev = 上一步末真值（run 首步为 run 起点真值），不加噪；e_i = ‖(pred−prev)[3:6]−(gt−prev)[3:6]‖·180/π（度）；手指 e_i = [6:51] 同形范数（rad）。Spearman ρ（含并列平均秩），方括号为 95% 区间（滑动块 bootstrap，块长 {BLOCK} 步，1000 次；考虑相邻步自相关）。\n")
        L.append("| 序列 | 层 | 包数 | m_i>0 占比 | m_i 均值 | ρ(m_i, e_rot) | 95% CI | ρ(m_i^q, e_rot) | ρ(m_i, e_rot 测地线) | ρ(m_i, e_fing) |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in seqs(B):
            for tag, key in (("全部", "all"), ("fingertip_in_front=真", "fingertip_in_front_true"), ("fingertip_in_front=假", "fingertip_in_front_false")):
                b = B["sequences"][s][key]
                if not b["n_packets"]:
                    L.append(f"| {s} | {tag} | 0 | - | - | - | - | - | - | - |")
                    continue
                ci = b["rho_m_e_rot_ci95"]
                L.append(f"| {s} | {tag} | {b['n_packets']} | {f(b['frac_m_gt0'], 3, True)} | {f(b['m_count_mean'], 4)} | {f(b['rho_m_e_rot'], 3)} | "
                         f"[{f(ci[0], 3)}, {f(ci[1], 3)}] | {f(b['rho_mq_e_rot'], 3)} | {f(b['rho_m_e_rot_geo'], 3)} | {f(b['rho_m_e_fing'], 3)} |")
        L.append("\n补充（非预注册，仅作混杂检查）：m_i 与 e_i 是否都随真值根旋转更新幅度 ‖gt−prev‖[3:6] 变化。\n")
        L.append("| 序列 | ρ(m_i, 真值旋转更新幅度) | ρ(e_rot, 真值旋转更新幅度) | 偏 Spearman ρ(m_i, e_rot \\| 旋转幅度) | ρ(m_i, 真值手指更新幅度) | ρ(e_fing, 真值手指更新幅度) | 偏 Spearman ρ(m_i, e_fing \\| 手指幅度) |")
        L.append("|---|---|---|---|---|---|---|")
        for s in seqs(B):
            cf = B["sequences"][s]["all"].get("confound")
            if cf:
                L.append(f"| {s}（全部 {B['sequences'][s]['all']['n_packets']} 包） | {f(cf['rho_m_gtrot'], 3)} | {f(cf['rho_e_rot_gtrot'], 3)} | {f(cf['partial_rho_m_e_rot_given_gtrot'], 3)} | "
                         f"{f(cf['rho_m_gtfing'], 3)} | {f(cf['rho_e_fing_gtfing'], 3)} | {f(cf['partial_rho_m_e_fing_given_gtfing'], 3)} |")
        L.append("\nm_i 分箱的 e_i 均值（°；手指 rad）：\n")
        L.append("| 序列 | 层 | 箱 | 包数 | e_rot 均值 | e_rot 中位 | e_fing 均值 | m_i 均值 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for sq in seqs(B):
            for tag, key in (("全部", "all"), ("fingertip_in_front=真", "fingertip_in_front_true"), ("fingertip_in_front=假", "fingertip_in_front_false")):
                for r in B["sequences"][sq][key].get("bins", []):
                    L.append(f"| {sq} | {tag} | {r['bin']} | {r['n']} | {f(r['e_rot_deg_mean'], 3)} | {f(r['e_rot_deg_median'], 3)} | {f(r['e_fing_rad_mean'], 4)} | {f(r['m_mean'], 4)} |")
        v = B.get("verdict_local")
        if v:
            L.append(f"\n**P2-b 判据（ρ(local) ≥ {RHO_SUPPORT} → 支持；≤ {RHO_REJECT} → 不支持；其间不下结论）**：zgz_local 全部 {B['sequences'][[s for s in seqs(B) if LOCAL(s)][0]]['all']['n_packets']} 包，"
                     f"ρ = {f(v['rho_local'], 3)} → **{v['verdict']}**。")
    if C:
        L.append("\n## P2-c 分布内干预（TF 包，仅 m_i>0）\n")
        L.append("干预 = 把错配事件 q 置 0（质量进残差槽）；对照 = 把等数量、q 之和近似相等的随机“正确关联”事件（有关联、src_part≥0、多数关节 == src_part）的 q 置 0，rng(k)，k=0..%d 取均值。相对下降 = 1 − 误差均值(干预)/误差均值(基线)；Δ = 相对下降(干预) − 相对下降(对照)，单位百分点。\n" % (C["meta"]["n_ctrl"] - 1))
        L.append("| 序列 | 层 | 包数(m>0)/总包数 | 基线 e_rot ° | 干预 e_rot ° | 对照 e_rot ° | 相对下降(干预) | 相对下降(对照) | Δ (pp) | Δ 95% CI | 平均错配事件数 | 对照/干预 q 和 | 对照事件数不足的包占比 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for s in seqs(C):
            S = C["sequences"][s]
            for tag, key in (("全部", "all"), ("fingertip_in_front=真", "fingertip_in_front_true"), ("fingertip_in_front=假", "fingertip_in_front_false")):
                if key not in S or not S[key].get("n_packets"):
                    continue
                r = S[key]["rot"]
                L.append(f"| {s} | {tag} | {S[key]['n_packets']}/{S['n_packets_total']} | {f(r['E0_mean'], 3)} | {f(r['E_int_mean'], 3)} | {f(r['E_ctl_mean_over_k'], 3)} | "
                         f"{f(r['rel_drop_int'], 3, True)} | {f(r['rel_drop_ctl'], 3, True)} | {f(r['delta_pp'], 2)} | [{f(r['delta_pp_ci95'][0], 2)}, {f(r['delta_pp_ci95'][1], 2)}] | "
                         f"{f(S[key]['n_mis_mean'], 1)} | {f(S[key]['q_sum_ratio_ctrl_over_int'], 3)} | {f(S[key]['frac_packets_control_short'], 3, True)} |")
        L.append("\n手指更新误差（rad，补充）：\n")
        L.append("| 序列 | 包数 | 基线 | 干预 | 对照 | 相对下降(干预) | 相对下降(对照) | Δ (pp) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for s in seqs(C):
            S = C["sequences"][s]
            if "all" in S and S["all"].get("n_packets"):
                r = S["all"]["fing"]
                L.append(f"| {s} | {S['all']['n_packets']} | {f(r['E0_mean'], 4)} | {f(r['E_int_mean'], 4)} | {f(r['E_ctl_mean_over_k'], 4)} | {f(r['rel_drop_int'], 3, True)} | {f(r['rel_drop_ctl'], 3, True)} | {f(r['delta_pp'], 2)} |")
        v = C.get("verdict_local")
        if v:
            ls = [s for s in seqs(C) if LOCAL(s)][0]
            L.append(f"\n**P2-c 判据（local 根旋转误差相对下降，干预 − 对照：≥ {DPP_SUPPORT:.0f} pp → 支持；≤ {DPP_REJECT:.0f} pp → 不支持；其间不下结论）**："
                     f"zgz_local {C['sequences'][ls]['all']['n_packets']} 包，Δ = {f(v['delta_pp_local'], 2)} pp → **{v['verdict']}**。")
        L.append(f"\n包装器调用 {C['meta']['wrapper_calls']} 次，累计置零事件 {C['meta']['events_zeroed_total']} 个；每次前向都断言了所选事件 w=0、r=1 且残差质量增量 = 所选 q 之和。")
    L.append("\n## 实现说明与口径选择（均在看结果前由脚本固定）\n")
    L.append("- **错配的“多数质量”**：取联合质量 q·A 按硬 LBS 关节累加后关节 0 占该事件总质量的 **> 1/2**（与 P1 `summarize` 的 `maj0` 同口径）；“plurality”（argmax=0）版本另列为 M(plurality)。精确一致率用 argmax 关节（P1 的 `pred`）。")
    sens = ""
    if B:
        lsq = [x for x in seqs(B) if LOCAL(x)]
        if lsq:
            rq = B["sequences"][lsq[0]]["all"]["rho_mq_e_rot"]
            sens = f"敏感性：zgz_local 用 m_i^q 的 ρ = {f(rq, 3)}，判定 {verdict_rho(rq)['verdict']}（与主口径{'相同' if verdict_rho(rq)['verdict'] == B['verdict_local']['verdict'] else '不同'}）。"
    L.append("- **m_i 的“质量”**：主口径按事件计数（每个 live 事件单位质量，分母为抽样内 live 事件数）；q 加权版本 m_i^q 作敏感性列出。P2-b 判据用主口径。" + sens)
    L.append("- **e_i**：按任务书取参数空间 ‖(pred−prev)[3:6] − (gt−prev)[3:6]‖·180/π（轴角向量差范数，度）；预注册里的 ∠ 另以测地线角（R_pred⁻¹R_gt）作为补充列“测地线”，未用于判定。")
    L.append("- **P2-b 的 local 判定集**：zgz_local 的全部 TF 包（含 m_i=0 者），ρ 为含并列平均秩的 Spearman。fingertip_in_front 取 P1 标签的 primary 版本；zgz_global 该标志在全部步上为假，故真层为空。")
    L.append(f"- **P2-c 对照组**：从“有关联、src_part≥0、argmax 关节 == src_part”的事件池中，按 q 分箱（[0,.25) [.25,.5) [.5,.75) [.75,1) 与 q=1 的命中）分层随机抽取与错配事件同数量的事件（池不足时尽量补足，不足的包占比见表），rng=default_rng(k)，k=0..{(C or {'meta': {'n_ctrl': 5}})['meta']['n_ctrl'] - 1} 各一条在序列内跨包连续使用。相对下降用“各包误差的均值”之比（非逐包比值均值）；区间为滑动块 bootstrap（块长 {BLOCK}）。")
    L.append("- **配置**：使用运行目录中的 `config_resolved.yaml`（与 `tools/select_checkpoint.py` 一致）；与 `configs/xa/xa_c1_s3407.yaml` 逐键比较仅 `_config_path` 不同。")
    hk = ""
    if A and A.get("hook_vs_nohook_identical"):
        hk = "本次 P2-a 以 `--check-nohook` 运行：两条序列各先跑一遍无 hook、不开 `xyz_keep_aux` 的 `track_sequence`，其 mpjpe_ra / mpvpe_ra / mpjpe_abs / mpvpe_abs / jitter 与带 hook 的运行**逐位相同**。"
    L.append("- **探针不改变闭环**：`state_hook('step')` 原样返回 pred；`xyz_keep_aux` 只存引用。" + hk + "每步另断言 `xyz_last` 的 valid/hit 计数与 `route_stats` 一致、事件像素坐标与 P1 标签一致、抽样索引与 P1 的 `samp_local` 一致。")
    if B and C:
        for sq in seqs(C):
            cc = C["sequences"][sq].get("columns")
            if not cc:
                continue
            m = dict(zip(B["sequences"][sq]["columns"]["end_ms"], B["sequences"][sq]["columns"]["e_rot_deg"]))
            dd = [abs(m[int(e)] - v) for e, v in zip(cc["end_ms"], cc["e0_rot_deg"])]
            L.append(f"- P2-c 基线 e_rot 与 P2-b 同包 e_rot 的一致性（{sq}，{len(dd)} 包）：max|差| = {max(dd):.2e}°。")
    gp = out_dir / "gpu_choice.txt"
    if gp.exists():
        L.append("- 设备选择：" + " ".join(gp.read_text().split()))
    for nm, D in (("P2-a", A), ("P2-b", B), ("P2-c", C)):
        if D:
            L.append(f"- {nm} 运行：{D['meta']['seconds']:.0f} s（{D['meta']['device']}，{D['meta']['gpu_name']}）。")
    if B:
        for sq in seqs(B):
            xc = B["sequences"][sq].get("p1_event_crosscheck")
            if xc:
                L.append(f"- P2-b 与 P1 `assoc_events_{sq}.npz`（R0/当前规则）逐事件核对：{int(xc['n_events'])} 个事件，有无关联不同 {int(xc['has_differs'])}，argmax 关节不同 {int(xc['pred_differs'])}，"
                         f"joint-0 多数标志不同 {int(xc['maj0_differs'])}，命中标志不同 {int(xc['hit_differs'])}。")
    (out_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    log("wrote", out_dir / "REPORT.md")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="a,b,c,report")
    ap.add_argument("--run-dir", default=RUN_DEFAULT)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None, help="default: the config copied into the run dir (what select_checkpoint uses)")
    ap.add_argument("--labels-dir", default=LABELS_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--n-ctrl", type=int, default=5)
    ap.add_argument("--random-init", action="store_true", help="self-test: seeded random weights instead of a checkpoint")
    ap.add_argument("--max-steps", type=int, default=0, help="self-test: first N evaluation steps of each sequence")
    ap.add_argument("--check-nohook", action="store_true", help="P2-a: also run track_sequence without hook and require bit-identical metrics")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.chdir(REPO)                                  # DATA.ROOT / MANO.NPZ are repo-relative
    parts = set(args.part.split(","))
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else REPO / out_dir
    if parts & {"a", "b", "c"}:
        with limited_runs(args.max_steps):
            cfg, cfg_path, device, mano, root, seqs, model, ckpt = load_everything(args)
            log(f"device={device} ckpt={ckpt} cfg={cfg_path} seqs={[s for s, _ in seqs]}")
            lab_dir = REPO / args.labels_dir
            env = dict(cfg=cfg, cfg_path=cfg_path, device=device, mano=mano, root=root, seqs=seqs, model=model, ckpt=ckpt,
                       cl=build_ctx_labels(root, seqs, lab_dir))
            for p, fn in (("a", run_p2a), ("b", run_p2b), ("c", run_p2c)):
                if p in parts:
                    fn(args, env, out_dir)
    if "report" in parts:
        make_report(args, out_dir)


if __name__ == "__main__":
    main()
