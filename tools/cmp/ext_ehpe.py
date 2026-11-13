"""CMP extension arm `ehpe` (docs/EXPERIMENT_RECORDS.md#cmp-amend7-20261110): EventHPE (Zou et al., ICCV 2021), the
event-based human-mesh tracker, on the CMP protocol with MANO (semkine/cmp_ehpe.py; official modules vendored in
semkine/eventhpe_official/).

  ehpe  official EventTrackNet (ResNet-50 on [8-partition event frame | flow] -> GRU(2048) -> per-joint 6D delta
        rotations + delta translation accumulated from the initial state) and official losses / weights
        (event_pose_estimation/train.py defaults: tran 1, theta 10 geodesic, joints3d 1, joints2d 10, flow 0.1); snippets of
        the official num_steps = 8 packets of 50 ms from the tracker's initial state (GT at the snippet start + B2's
        prior noise); official Adam, lr 1e-3, batch 8 under CMP's warm-up + cosine; no augmentation (the official loader
        has none); flow = the frozen FlowNet of stage `epf` of the seed (the official unsupervised FlowNet needs gray
        frames); 45000 steps = the 2.88 M packet-samples of `ep` (7500 x 32 x 12)
Deployed form `main` (the official test procedure, online): snippets of 8 packets chained on the own last state, the
recurrent state reset to zero at every snippet start (and at every run start); `carry` (descriptive): never reset
within a run (tools/cmp/nets.py reserves the name `rolling` for the GRU baseline). `gtinit` (amendment 9,
#cmp-amend9-20261110, descriptive, not deployable): the official evaluation, which tracks num_steps = 8 steps from the
ground-truth pose of the clip's first frame (event_pose_estimation/test.py, test_steps 8) -- every snippet starts from
the GT state at its first packet's start instead of the tracker's own state.
"""
from __future__ import annotations

import copy

import numpy as np

from .common import require

PACKET_MS = 50
ARMS = {
    "ehpe": {"model": "ehpe", "stage": "seq", "init": ("epf", 3000), "steps": 45000, "lr": 1.0e-3, "warmup": 500,
             "batch": 8, "accumulate": 1, "workers": 16, "seq": {"K": 8, "VOXEL_BINS": 5, "EVENT_BINS": 8}},
}
EVAL = {"ehpe": {"WINDOW_MS": PACKET_MS, "HISTORY": 1}}
#: event_pose_estimation/train.py `get_args` defaults (delta_tran_loss 0, tran_loss 1, theta_loss 10, joints3d_loss 1,
#: joints2d_loss 10, flow_loss 0.1, use_geodesic_loss 1) and loss_funcs.py `compute_flow_loss` (|flow| > 4 px)
LOSS = {"DELTA_TRAN": 0.0, "TRAN": 1.0, "THETA": 10.0, "JOINTS3D": 1.0, "JOINTS2D": 10.0, "FLOW": 0.1, "GEODESIC": True,
        "FLOW_VALID_PX": 4.0}
#: the official dataloader augments nothing
AUG = {"DOMRAND": {"ENABLED": False}, "SPEED_AUG": False, "POLARITY_FLIP": False, "PIXEL_POLARITY_SWAP": False}
SAVE_EVERY = 5000
CODE = ("tools/cmp/ext_ehpe.py", "semkine/cmp_ehpe.py", "semkine/cmp_seq_ehpe.py", "semkine/eventhpe_official/model.py",
        "semkine/eventhpe_official/geometry.py", "semkine/eventhpe_official/loss_funcs.py",
        "semkine/evhandpose_official/backbone.py")


def configure(cfg, arm, seed, phase):
    """The official recipe's blocks on top of the registry's config: no augmentation, the loss weights, sparse saves."""
    require(arm in ARMS, "not an EventHPE arm")
    cfg["AUG"] = copy.deepcopy(AUG)
    cfg["CMP"]["LOSS"] = dict(LOSS)
    if phase == "formal":
        cfg["TRAIN"]["SAVE_EVERY_N_STEPS"] = SAVE_EVERY
    return cfg


def build_dataset(ds, cfg, split):
    """tools/cmp/train.py: the training dataset re-indexed into snippets with EventHPE's event frames."""
    if split != "train":
        return ds
    from semkine import cmp_seq_ehpe as S
    s = cfg["CMP"]["SEQ"]
    return S.upgrade(ds, int(s["K"]), int(s["PACKET_MS"]), int(s["VOXEL_BINS"]), int(s["EVENT_BINS"]))


class Adapter:
    """The deployed step of `ehpe` (tools/cmp/nets.py `CmpNet.ext`): one 50 ms packet per update, the state = the
    engine's previous state composed with the predicted delta; the recurrent state lives here and never reads GT."""

    FORMS = ("main", "carry", "gtinit")
    history = 1

    def __init__(self, net):
        self.net = net
        s = net.cfg["CMP"]["SEQ"]
        self.chunk, self.bins = int(s["K"]), int(s["EVENT_BINS"])
        self.hidden, self.last, self.n = None, None, 0

    def bind(self, ctx):
        self.hidden, self.last, self.n = None, None, 0

    def inputs(self, ctx, end):
        """Host part of an update: the packet's event frame and voxel grid on the device."""
        import torch
        from semkine.cmp_seq import voxel_grid
        from semkine.cmp_seq_ehpe import event_frames
        from tools.cmp.nets import window_raw
        net = self.net
        H, W = int(net.cfg["DATA"]["HEIGHT"]), int(net.cfg["DATA"]["WIDTH"])
        xs, ys, ps, us = window_raw(ctx, end, PACKET_MS)
        ev = torch.from_numpy(event_frames(xs, ys, us, PACKET_MS, self.bins, H, W)).unsqueeze(0).to(net.device)
        vox = torch.from_numpy(voxel_grid(xs, ys, ps, us, PACKET_MS, net.model.voxel_bins, H, W))
        return ev, vox.unsqueeze(0).to(net.device)

    def fresh(self, a, end, last, n):
        """A snippet starts here: another run, a gap in the packets, or (main) num_steps packets done."""
        return last != (a, end - PACKET_MS) or (self.net.form in ("main", "gtinit") and n >= self.chunk)

    def step(self, ctx, end, prev):
        import torch
        net = self.net
        end = int(end)
        a = net.run_start(end)
        if self.fresh(a, end, self.last, self.n):
            self.hidden, self.n = None, 0
            if net.form == "gtinit":                     # amendment 9: the official evaluation's initial pose (GT)
                prev = np.asarray(ctx.gt[max(end - PACKET_MS, a)], np.float32)
        ev, vox = self.inputs(ctx, end)
        prev_t = torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(net.device)
        y, self.hidden = net.model.track_step(ev, vox, prev_t, self.hidden)
        self.last, self.n = (a, end), self.n + 1
        return y.detach().float().cpu().numpy()[0].copy()

    # ------------------------------------------------------------------ tools/cmp/cost.py
    def cost_build(self, ctx, end, a, prev=None):
        ev, vox = self.inputs(ctx, end)
        return {"ev": ev, "vox": vox, "a": int(a), "end": int(end)}

    def cost_step(self, item, prev, state):
        """The deployed step on a staged item; `state` is the cost tool's per-pass dict ("a" None starts a pass)."""
        if state.get("a") != item["a"] or self.fresh(item["a"], item["end"], state.get("last"), state.get("n", 0)):
            state["hidden"], state["n"] = None, 0
        y, state["hidden"] = self.net.model.track_step(item["ev"], item["vox"], prev, state["hidden"])
        state["a"], state["last"], state["n"] = item["a"], (item["a"], item["end"]), state["n"] + 1
        return y

    def extra_macs(self, state):
        """The GRU's multiply-accumulates of one update (one recurrent step; analytic, as the cost tool counts eg's)."""
        g = self.net.model.tmp_encoder.gru
        return 3 * (g.input_size + g.hidden_size) * g.hidden_size


__all__ = ["ARMS", "AUG", "Adapter", "CODE", "EVAL", "LOSS", "build_dataset", "configure"]
