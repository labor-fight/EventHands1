#!/usr/bin/env python3
"""CMP extension arm `fme` (docs/EXPERIMENT_RECORDS.md#cmp-amend8-20261110): FastMETRO with event input (Cho et al.,
ECCV 2022), the event-only hand baseline of EvRGBHand (Jiang et al., CVPR 2024; official `src/configs/config/
fastmetro_ev.yaml`), on the CMP protocol. Model semkine/cmp_fme.py, official network semkine/fastmetro_official/.

Registered by tools/cmp/arms.py (`EXTENSIONS`): `ARMS`, `EVAL`, `configure`, `CODE`, `e1`; the deployed step for the cost
tool is `Adapter` (tools/cmp/nets.py `EXT_ADAPTERS`).

Recipe (`configure`, written into the config's CMP.FME block): the official network, loss, optimiser, photometric event
augmentation and frame layout; see semkine/cmp_fme.py. The protocol's changes:
  input      the full 240 x 180 frame zero-padded to 256 x 192 (official: a 192 x 192 crop around the annotated hand box,
             rate 1.6), LNES of a fixed 100 ms window clipped at the run start (official: the window that holds 0.4
             events per box pixel)
  geometry   the official ranges as domain randomisation of the full frame: roll uniform in +-0.4 rad (official: N(0, 0.2)
             clipped at +-0.4 rad, about the principal point) and scale log-uniform in [0.9, 1.1] (official: the crop's
             rate x (1 + N(0, 0.05) clipped at +-0.1)); no shift (the official translation draw is not applied), no event
             thinning, hot pixels, speed augmentation, polarity flip or swap (not in the official recipe)
  labels     joints / vertices from MANO FK of the 51-D label with the sequence betas
  schedule   one GPU, batch 32 (the official per-GPU batch), CMP's warm-up then cosine per step to the official eta_min

Reading (`e1`): the network outputs a mesh, not a MANO state, so the engine's E1 cannot score it; `e1` walks the engine's
zgz endpoint grid itself (tools/pfo/engine.py: the same sequences, runs, endpoints and order; frame-wise, so no closed
loop and no initial noise), one forward per endpoint, and applies the engine's metric definitions (tools/xcl/evaluate.py
`score`) to predicted against ground-truth joints / vertices (`score_mesh`):
  mpjpe_ra_mm          the 21 joint tokens, wrist-aligned, against ManoLayer's joints of the label -- the joints the
                       official evaluation scores (train.py `run_eval_and_show`) and the ones supervised with FK joints
  mpvpe_ra_mm          the 778 fine vertices, aligned to vertex 0 (the engine's `root_align` on vertices)
  mp*_abs_mm           the raw outputs against the label; the official loss aligns every term to the wrist, so the
                       absolute position is not supervised and these fields carry no meaning (kept for the row schema)
  mpjpe_ra_mesh_mm     joints regressed from the fine mesh (MANO J_regressor + fingertip vertices) against FK joints;
                       the regressor applied to the ground-truth mesh is itself 1.0 / 2.4 mm (zgz_global / zgz_local)
                       from the FK joints, which is why this is not the main field
  mpjpe_ra_meshreg_mm  the same regressed joints against the regressor applied to the ground-truth mesh
  mpvpe_wrist_mm       vertices aligned to the regressed wrist on both sides (the official mesh alignment)
readings/cmp_fme_s<seed>/e1/{launch.json, states.npz, report.json} in the engine's schema (tools/cmp/rows.py and
diag_history.py read it unchanged); states.npz also keeps the predicted joints and vertices.

    python -m tools.cmp.run e1 --arm fme --seed S                      # the reading (accepted checkpoint, GPU 4-7)
    python -m tools.cmp.ext_fme e1 --seed S --smoke [--limit N]        # engineering smoke on the smoke checkpoint
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "model")) if p not in sys.path]
import numpy as np  # noqa: E402

from tools.cmp.common import OUT, now, read, require, save, sha  # noqa: E402

#: 192000 x 32 = the frame arms' 6000 x 1024 samples (as eho's 96000 x 64); the official 80 epochs of EvRealHands +
#: InterHand at batch 32 per GPU are of the same order
ARMS = {"fme": {"model": "fme", "stage": "frame", "steps": 192000, "lr": 1.0e-4, "warmup": 500, "batch": 32,
                "accumulate": 1, "workers": 16}}
EVAL = {"fme": {"WINDOW_MS": 100}}
CODE = ("tools/cmp/ext_fme.py", "semkine/cmp_fme.py", "semkine/fastmetro_official/fastmetro.py",
        "semkine/fastmetro_official/transformer.py", "semkine/fastmetro_official/data/mano_195_downsampling.npy",
        "semkine/fastmetro_official/data/mano_195_adjmat_indices.pt")
#: fastmetro_ev.yaml `model` (the blocks FastMETRO_Hand_Network reads)
NET = {"BACKBONE": "resnet34", "PRETRAINED": True, "HIDDEN_DIM_1": 512, "DIM_FEEDFORWARD_1": 2048, "HIDDEN_DIM_2": 128,
       "DIM_FEEDFORWARD_2": 512, "ENC_LAYERS": 3, "DEC_LAYERS": 3, "NHEAD": 8, "DROPOUT": 0.1}
#: fastmetro_ev.yaml `exper.loss` terms Loss.get_loss_mano uses for an output without camera / heat map (SmoothL1 beta 0.01)
LOSS = {"VERTICES": 100.0, "VERTICES_SUB": 100.0, "JOINTS_3D": 500.0, "JOINTS_3D_FROM_MESH": 500.0, "BETA": 0.01}
#: train.py `run`: AdamW(betas (0.9, 0.999), weight_decay 1e-4), lr / lr_backbone of fastmetro_ev.yaml, clip_grad_norm_ 1.0,
#: CosineAnnealingLR eta_min 1e-6
OPT = {"LR_BACKBONE": 1.0e-5, "BACKBONE_NAMES": ["backbone.layer"], "WEIGHT_DECAY": 1.0e-4, "BETAS": [0.9, 0.999],
       "CLIP_NORM": 1.0, "ETA_MIN": 1.0e-6}
#: fastmetro_ev.yaml `exper.augment.event_photometry` (colour jitter and blur have probability 0)
PHOTOMETRY = {"GAUSS_P": 0.1, "GAUSS_VAR": [0.05, 0.1], "SALT_PEPPER_P": 0.1, "SALT_PEPPER_RATE": 0.05}
#: fastmetro_ev.yaml `exper.augment.geometry` rot 0.2 / scale 0.05 (clipped at two sigma) as uniform ranges
DOMRAND = {"ENABLED": True, "ROLL_DEG": 22.92, "SCALE_MIN": 0.9, "SCALE_MAX": 1.1, "SHIFT_PX": 0.0, "KEEP_MIN": 1.0,
           "KEEP_MAX": 1.0, "HOT_PIXEL_RATE": 0.0, "SCALE_MODE": "focal"}
PAD = [8, 8, 6, 6]                       # left, right, top, bottom: 240 x 180 -> 256 x 192
SAVE_EVERY = 48000
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "mpjpe_ra_mesh_mm", "mpjpe_ra_meshreg_mm",
           "mpvpe_wrist_mm")


def configure(cfg, arm, seed, phase):
    """The official recipe's blocks on top of the registry's frame-stage config."""
    smoke = phase == "smoke"
    cfg["AUG"] = {"DOMRAND": dict(DOMRAND), "SPEED_AUG": False, "POLARITY_FLIP": False, "PIXEL_POLARITY_SWAP": False}
    cfg["DATA"]["WINDOW_MIN"] = cfg["DATA"]["WINDOW_MAX"] = int(EVAL["fme"]["WINDOW_MS"])
    cfg["MODEL"]["BACKBONE"] = NET["BACKBONE"]
    cfg["TRAIN"].update({"PRECISION": 32, "OPTIMIZER": "adamw", "SAVE_EVERY_N_STEPS": 10 if smoke else SAVE_EVERY})
    cfg["CMP"]["FME"] = {"NET": dict(NET), "PAD": list(PAD), "LOSS": dict(LOSS), "OPT": {**OPT}, "PHOTOMETRY": {**PHOTOMETRY}}
    return cfg


def attention_macs(d, n_enc, n_dec, src, tgt):
    """Multiply-accumulates of the attention modules of one encoder-decoder (nn.MultiheadAttention runs functional
    linears, which the cost tool's Linear hooks do not see): per attention the q / k / v and output projections
    (d^2 per token) and the two products with the attention matrix (2 Lq Lk d)."""
    att = lambda lq, lk: (lq + 2 * lk + lq) * d * d + 2 * lq * lk * d  # noqa: E731
    return n_enc * att(src, src) + n_dec * (att(tgt, tgt) + att(tgt, src))


class Adapter:
    """The deployed step for tools/cmp/nets.py and tools/cmp/cost.py: one forward on the LNES of the 100 ms window."""

    FORMS = ("main",)
    history = 1

    def __init__(self, net):
        self.net = net

    def bind(self, ctx):
        pass

    def predict(self, ctx, end):
        """(21, 3) joint tokens and (778, 3) fine vertices (float32, metres) of the window ending at `end`."""
        n = self.net
        out = n.model.predict(n.lnes(ctx, end, min(n.window, end - n.run_start(end) + 1)))
        return (out["pred_3d_joints"][0].float().cpu().numpy(), out["pred_vertices"][0].float().cpu().numpy())

    def step(self, ctx, end, prev):
        raise RuntimeError("FastMETRO regresses a mesh, not a 51-D MANO state: its reading is tools.cmp.ext_fme.e1 "
                           "(python -m tools.cmp.run e1 --arm fme), not the engine's E1")

    def cost_build(self, ctx, end, a, prev=None):
        return {"x": self.net.lnes(ctx, end, min(self.net.window, end - a + 1)), "a": a}

    def cost_step(self, item, prev, state):
        """The forward; the first 51 of the 63 joint coordinates stand in for the state the cost tool copies back."""
        return self.net.model.predict(item["x"])["pred_3d_joints"].reshape(1, -1)[:, :51]

    def extra_macs(self, state):
        m = self.net.model
        net, c = m.net, m.cfg
        l, r, t, b = m.pad
        src = ((int(c["DATA"]["HEIGHT"]) + t + b) // 32) * ((int(c["DATA"]["WIDTH"]) + l + r) // 32)
        tgt = net.num_joints + net.num_vertices_coarse
        n = c["CMP"]["FME"]["NET"]
        return sum(attention_macs(int(n[f"HIDDEN_DIM_{i}"]), int(n["ENC_LAYERS"]), int(n["DEC_LAYERS"]), src, tgt)
                   for i in (1, 2))


def score_mesh(pj, pv, gj, gv, jreg):
    """tools/xcl/evaluate.py `score`'s position metrics on joints (n, 21, 3) and vertices (n, 778, 3) instead of 51-D
    states (same expressions: `root_align` = minus element 0, mean Euclidean distance, mm), plus the mesh-joint fields."""
    from semkine import eval_track as ET
    ra = ET.root_align
    pjm, gjm = np.einsum("jv,nvk->njk", jreg, pv), np.einsum("jv,nvk->njk", jreg, gv)
    return {"mpjpe_ra_mm": np.linalg.norm(ra(pj) - ra(gj), axis=-1).mean(-1) * 1000,
            "mpvpe_ra_mm": np.linalg.norm(ra(pv) - ra(gv), axis=-1).mean(-1) * 1000,
            "mpjpe_abs_mm": np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000,
            "mpvpe_abs_mm": np.linalg.norm(pv - gv, axis=-1).mean(-1) * 1000,
            "mpjpe_ra_mesh_mm": np.linalg.norm(ra(pjm) - ra(gj), axis=-1).mean(-1) * 1000,
            "mpjpe_ra_meshreg_mm": np.linalg.norm(ra(pjm) - ra(gjm), axis=-1).mean(-1) * 1000,
            "mpvpe_wrist_mm": np.linalg.norm((pv - pjm[:, :1]) - (gv - gjm[:, :1]), axis=-1).mean(-1) * 1000}


def e1(arm, seed, form="main", smoke=False, limit=None):
    import torch
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from tools.cmp import nets as CN
    from tools.cmp.arms import ARMS as REG, SMOKE_STEPS, checkpoint, config_path
    from tools.pfo import engine as E
    from tools.pfo import nets as PN
    from tools.pfo.common import COUNTS, ZGZ
    from tools.xcl import evaluate as XE
    require(arm in ARMS and form == "main", "fme has one reading: e1, form main")
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(4, 8))), "one GPU among 4-7")
    require(smoke or limit is None, "a limited reading is an engineering smoke")
    XE.configure_environment()
    XE.set_policy()                                                    # the engine's deterministic evaluation flags
    if smoke:                                    # engineering smoke: smoke checkpoint, outputs/cmp/20261110/smoke_fme/
        ck = checkpoint(arm, seed, SMOKE_STEPS, "smoke")
        rec = {"status": "PASS", "checkpoint": str(ck), "checkpoint_sha256": sha(ck),
               "config": str(config_path(arm, seed, "smoke"))}
        base = OUT / "smoke_fme"
    else:
        rec = read(OUT / "receipts" / f"cmp_{arm}_s{seed}.json")
        base = OUT
    require(rec["status"] == "PASS" and sha(rec["checkpoint"]) == rec["checkpoint_sha256"], "unaccepted checkpoint")
    tag = f"cmp_{arm}_s{seed}"
    out = base / "readings" / tag / "e1"
    require(not out.exists(), "preserve existing reading: " + str(out))
    spec = {"kind": "cmp", "model": REG[arm]["model"], "checkpoint": rec["checkpoint"], "config": rec["config"],
            "form": form}
    save(base / "specs" / f"{tag}.json", spec)
    out.mkdir(parents=True)
    device = torch.device("cuda")
    net = CN.load(tag, device=device, spec=spec)
    launch = {"net": tag, "task": "e1", "route": None, "tau": None, "intervene": [], "identity": net.identity,
              "checkpoint_sha256": rec["checkpoint_sha256"], "config_sha256": sha(rec["config"]),
              "reader_sha256": sha(__file__), "model_sha256": sha(ROOT / "semkine/cmp_fme.py"),
              "engine_sha256": sha(Path(E.__file__)), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "limit": limit,
              "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    mano = ManoLayer(net.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    jreg = net.model.jreg.detach().cpu().numpy()
    root, seqs = E.sequences(net.cfg, ZGZ)
    arrays, blocks = {}, {}
    with torch.no_grad():
        for seq, directory in seqs:
            ctx = PN.sequence(root, directory, seq, device)
            net.bind(ctx)
            rows = E.endpoint_rows(ctx.aux)
            require(limit is not None or len(rows) == COUNTS[seq], "endpoint denominator changed")
            n = len(rows) if limit is None else min(len(rows), limit)
            rec_ = {k: [] for k in ("pred_joints", "pred_vertices", "target", "end_ms", "run_id", "elapsed_ms")}
            i = 0
            for rid, (a, b) in enumerate(np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)):
                prev_ms = int(a)
                for end in np.arange(a + 49, b, 50, dtype=np.int64):
                    if i >= n:
                        break
                    require(rows[i][:3] == (rid, int(end), prev_ms), "endpoint grid mismatch")
                    i += 1
                    joints, verts = net.ext.predict(ctx, int(end))
                    if not (np.isfinite(joints).all() and np.isfinite(verts).all()):
                        raise FloatingPointError(f"nonfinite endpoint retained as failure: {seq}:{int(end)}")
                    rec_["pred_joints"].append(joints); rec_["pred_vertices"].append(verts)
                    rec_["target"].append(np.asarray(ctx.gt[int(end)], np.float32).copy())
                    rec_["end_ms"].append(int(end)); rec_["run_id"].append(rid); rec_["elapsed_ms"].append(int(end - a))
                    prev_ms = int(end)
            for key, v in rec_.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            beta = torch.tensor(np.asarray(ctx.aux["betas"], np.float32), device=device).view(1, -1)
            gj, gv = ET.mano_fk(mano, arrays[f"{seq}/target"], beta, device)
            m = score_mesh(arrays[f"{seq}/pred_joints"], arrays[f"{seq}/pred_vertices"], gj, gv, jreg)
            for key, v in m.items():
                arrays[f"{seq}/{key}"] = v
            blocks[seq] = {k: float(v.mean()) for k, v in m.items()}
            blocks[seq]["n"] = int(len(rec_["end_ms"]))
    np.savez_compressed(out / "states.npz", **arrays)
    total = sum(blocks[s]["n"] for s in ZGZ)
    pooled = {k: sum(blocks[s][k] * blocks[s]["n"] for s in ZGZ) / total for k in METRICS}
    save(out / "report.json", {**launch, "status": "COMPLETE", "finished_utc": now(), "per_sequence": blocks,
                               "overall": pooled, "states_sha256": sha(out / "states.npz"),
                               "fields": {"mpjpe_ra_mm": "joint tokens", "mpvpe_ra_mm": "fine mesh, vertex-0 aligned",
                                          "mpjpe_abs_mm / mpvpe_abs_mm": "raw outputs, absolute position not supervised",
                                          "mpjpe_ra_mesh_mm": "joints regressed from the fine mesh vs FK joints",
                                          "mpjpe_ra_meshreg_mm": "regressed joints, both sides",
                                          "mpvpe_wrist_mm": "fine mesh, regressed-wrist aligned"}}, exclusive=True)
    print("COMPLETE", out, {s: round(blocks[s]["mpjpe_ra_mm"], 3) for s in blocks}, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("e1",))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    e1("fme", a.seed, "main", a.smoke, a.limit)


if __name__ == "__main__":
    main()
