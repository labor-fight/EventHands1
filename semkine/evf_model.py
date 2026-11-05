"""EVF model (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): raw event points x the previous state's virtual frame ->
51-D state, as a Lightning module the frozen training script (semkine/train.py, via tools/evf/train.py) and the PFO
readout engine (via tools/evf/nets.py) can use in place of MNISTModel.

The loss, optimiser, schedule, logging and checkpoint layout are BaseModel's, unchanged (LOSS.TYPE so3_trans_fk ...).
MNISTModel's constructor is not run (it would build the dense ResNet18 the EVF arm does not use, and DDP refuses unused
parameters); its MANO / intrinsics helpers are reused. Config: the MODEL block keeps the keys the dataset and the
helpers read (POSE_REPR, OUTPUT_DIM, PREDICT_DELTA, RENDER_*); the network is described by the EVF block:

  NET          point                     A, the point version (semkine/evf_net.py)
  VF           true | false              virtual-frame node features (false: the "events only" control)
  LBS          false | true              add the LBS weights of the surface / contour points to the virtual-frame samples
  PREV48       true | false              the prev[3:51] prior path into the readout
  HIDDEN, LAYERS, SLOT_DIM, READOUT_HIDDEN
  VOXEL_PX, TIME_BINS, MAX_NODES         node selection (semkine/evf_nodes.py)
  ITERS        1 | 2                     update steps within one packet (2: re-render after the first, shared weights)
  ARM, SEED, PHASE, STUDY, SECTION       bookkeeping (tools/evf/arms.py)
"""
from __future__ import annotations

import torch

from model import BaseModel, MNISTModel
from semkine import evf_nodes as EN
from semkine.evf_net import EVFPointNet, compose
from semkine.evf_render import HAND_SCALE_M, Frame, project

EVF_KEYS = {"NET": "point", "VF": True, "LBS": False, "PREV48": True, "HIDDEN": 96, "LAYERS": 3, "SLOT_DIM": 32,
            "READOUT_HIDDEN": 256, "VOXEL_PX": EN.VOXEL_PX, "TIME_BINS": EN.TIME_BINS, "MAX_NODES": EN.MAX_NODES,
            "ITERS": 1, "ARM": None, "SEED": None, "PHASE": None, "STUDY": None, "SECTION": None}
NETS = ("point",)


def evf_options(cfg):
    e = dict(cfg.get("EVF") or {})
    unknown = sorted(set(e) - set(EVF_KEYS))
    if unknown:
        raise ValueError(f"unknown EVF keys: {unknown}")
    o = {**EVF_KEYS, **e}
    if o["NET"] not in NETS:
        raise ValueError(f"EVF.NET must be one of {NETS}, got {o['NET']!r}")
    if int(o["ITERS"]) not in (1, 2):
        raise ValueError("EVF.ITERS must be 1 or 2")
    if o["LBS"] and not o["VF"]:
        raise ValueError("EVF.LBS needs the virtual frame")
    return o


class RootFrame:
    """The projected root joint and the hand scale only (the control without the virtual frame)."""

    def __init__(self, joints, fx, fy, cx, cy):
        ru, rv, rz = project(joints[:, :1].float(), fx, fy, cx, cy)
        self.root_u, self.root_v = ru[:, 0], rv[:, 0]
        self.root_z = rz[:, 0].clamp_min(0.05)
        self.scale = (fx * fy).sqrt() * HAND_SCALE_M / self.root_z


class EVFModel(MNISTModel):
    def __init__(self, cfg=None, num_classes=None):
        BaseModel.__init__(self, cfg)
        model_cfg = self.cfg.get("MODEL", {})
        self.evf = evf_options(self.cfg)
        if self.output_dim != 51 or not bool(model_cfg.get("PREDICT_DELTA", False)):
            raise ValueError("the EVF model writes a 51-D state: OUTPUT_DIM 51, PREDICT_DELTA true")
        if self.loss_type != "so3_trans_fk":
            raise ValueError("the EVF study keeps LOSS.TYPE so3_trans_fk")
        self.predict_delta = True
        self.zero_event_gate = True
        self.encoder_name = "evf"
        self.xyz_mesh = False
        self.prev_render = False
        self.prevpos_embed = False
        self._ctx_betas = None
        self._ctx_K = None
        self._build_mano(model_cfg)
        with torch.no_grad():
            j_rest = self.mano.J_regressor @ self.mano.v_template
            tmpl = self.mano.v_template - j_rest[0:1]
        self.register_buffer("evf_template", tmpl.clone(), persistent=False)
        self.register_buffer("evf_lbs", self.mano.weights.detach().clone(), persistent=False)
        o = self.evf
        vf = bool(o["VF"])
        node_dim = (EN.NODE_DIM_VF + (EN.LBS_DIM if o["LBS"] else 0)) if vf else EN.NODE_DIM_NOVF
        packet_dim = EN.PACKET_DIM_VF if vf else EN.PACKET_DIM_NOVF
        self.net = EVFPointNet(node_dim, packet_dim, hidden=int(o["HIDDEN"]), layers=int(o["LAYERS"]),
                               slot_dim=int(o["SLOT_DIM"]), readout_hidden=int(o["READOUT_HIDDEN"]),
                               prev48=bool(o["PREV48"]))
        #: diagnostics of the last forward (node counts before / after the cap, clipped faces)
        self.evf_stats = {}

    # ------------------------------------------------------------------ one update from `prev`
    def _inputs(self, batch, prev, betas, K):
        """Everything the network reads, from the events and `prev` (no gradient)."""
        o = self.evf
        H, W = self.render_h, self.render_w
        fx, fy, cx, cy = self._intrinsics(K)
        with torch.no_grad(), torch.autocast(device_type=prev.device.type, enabled=False):
            verts, joints = self._fk(prev.float(), betas)
            vf = bool(o["VF"])
            frame = Frame(verts, joints, self.mano.f, fx, fy, cx, cy, H, W) if vf else RootFrame(joints, fx, fy, cx, cy)
            src, mask, n_nodes = EN.select_nodes(batch.events, batch.ptr, batch.delta_t_s, H, W, int(o["VOXEL_PX"]),
                                                 int(o["TIME_BINS"]), int(o["MAX_NODES"]))
            attrs = EN.event_attributes(batch.events, batch.ptr, batch.delta_t_s, H, W)
            feat, sums = EN.node_features(batch.events, attrs, src, mask, frame, self.evf_template, H, W, vf=vf,
                                          lbs=self.evf_lbs if o["LBS"] else None)
            packet = EN.packet_from_sums(batch.counts, n_nodes, batch.delta_t_s, mask.sum(1), sums, frame, vf=vf)
        self.evf_stats = {"nodes": n_nodes.detach(), "kept": mask.sum(1).detach(),
                          "clipped_faces": getattr(frame, "clipped", torch.zeros_like(n_nodes)).detach()}
        return feat, mask, packet, joints[:, 0], frame, (fx, fy, cx, cy)

    def _update(self, batch, prev, betas, K):
        feat, mask, packet, root, frame, (fx, fy, cx, cy) = self._inputs(batch, prev, betas, K)
        params = self.net(feat, mask, packet, prev[:, 3:51] if self.evf["PREV48"] else None)
        return compose(prev, params, root, frame.scale, fx, fy, cx, cy)

    def forward_packet(self, batch):
        """Raw-event forward for an `EventPacketBatch`; empty packets return `prev` bitwise."""
        prev = batch.prev_state.float()
        betas, K = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
        out = prev
        for _ in range(int(self.evf["ITERS"])):
            out = self._update(batch, out, betas, K)
        empty = (batch.counts <= 0).unsqueeze(-1)
        return torch.where(empty, prev, out)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        raise RuntimeError("the EVF model reads raw events; call forward_packet")


__all__ = ["EVFModel", "EVF_KEYS", "RootFrame", "evf_options"]
