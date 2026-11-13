r"""EventHPE (Zou et al., ICCV 2021) ported to the CMP protocol (docs/EXPERIMENT_RECORDS.md#cmp-amend7-20261110).

The network and the losses are the official code (semkine/eventhpe_official/, vendored from github.com/JimmyZou/EventHPE
at 2af451f): `EventTrackNet` = ResNet-50 (random init, conv1 on [event frame | flow]) -> 2048-d per step ->
uni-directional GRU(2048, one layer, residual) -> two linear heads, a 6D delta rotation per joint and a delta
translation; the rotations and the translation accumulate from the given initial state (R_k = dR_k ... dR_1 R_init per
joint, t_k = t_init + sum dt_i); loss = tran (MSE, 1) + theta (geodesic angle, 10) + joints3d (MSE, 1) + joints2d (MSE of
the clamped normalised projections, 10) + flow coherence (1 - cosine between the projected vertex motion and the
optical flow sampled at the vertices, where |flow| > 4 px; 0.1), the translation detached in the vertices and joints as
in the official forward. `EventTrackNet.forward` is restated here for MANO (16 joints, 778 vertices, 21 joints of the
MANO layer, sequence GT betas) in place of SMPL; what the protocol fixes:
  * input frame: the full 240 x 180 event frame zero-padded to 256 x 192 (as semkine/cmp_evhp.py), the official event
    frame of a step (semkine/cmp_seq_ehpe.py `event_frames`: 8 temporal partitions, binary, polarity ignored);
  * step: one 50 ms packet; snippet: the official num_steps = 8;
  * initial state: the tracker state (training: GT at the snippet start + B2's prior noise; test: the engine's `prev`)
    instead of the GT / HMR / VIBE pose of the first gray frame; GRU initial state zeros (the official option without
    the HMR feature of the first gray frame);
  * flow: the official FlowNet is trained unsupervised on gray frames this data does not have; the flow of a packet
    is the frozen CMP FlowNet of stage `epf` (EvHandPose's official FlowNet trained on mesh-flow pseudo ground truth,
    semkine/cmp_evhp.py), finest scale, in pixels;
  * optimiser: the official Adam, lr 1e-3, batch 8 snippets, under CMP's warm-up + cosine schedule; mixed precision
    (official use_amp) = CMP's bf16, the GRU and the losses in float32; no augmentation (the official loader has none).
Deployed form (tools/cmp/ext_ehpe.py `Adapter`): the official test procedure, snippets of num_steps chained on the own
last state with the recurrent state reset at every snippet start, one packet per update (`track_step`).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F

from model import BaseModel, MNISTModel

from .cmp_evhp import FLOW_CHUNK, FLOW_SCALE_PX, PAD, official_config
from .evhandpose_official.backbone import FlowNet
from .eventhpe_official import geometry as G
from .eventhpe_official import loss_funcs as L
from .eventhpe_official.model import Regressor, TemporalEncoder, resnet50_encoder

N_JOINTS = 16
FEATURE = 2048


def mano_rotmats(mano, betas, rot_mats, transl=None):
    """`ManoLayer.forward` from the joint rotation matrices (N, 16, 3, 3) instead of axis-angles: (verts, 21 joints)."""
    from mano_layer import _transform_mat
    batch = betas.shape[0]
    device, dtype = betas.device, betas.dtype
    v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)
    J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).reshape(batch, -1)
    v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, mano.posedirs)
    rel_joints = J.clone()
    rel_joints[:, 1:] -= J[:, mano.parents[1:]]
    transforms_mat = _transform_mat(rot_mats.reshape(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).view(batch, 16, 4, 4)
    parents = mano._parents_py
    chain = [transforms_mat[:, 0]]
    for i in range(1, mano.num_joints):
        chain.append(torch.matmul(chain[parents[i]], transforms_mat[:, i]))
    transforms = torch.stack(chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    joints_homogen = torch.cat([J, torch.zeros((batch, 16, 1), dtype=dtype, device=device)], dim=2)
    rest_mats = torch.matmul(transforms, joints_homogen.unsqueeze(-1))
    joint_rel = transforms.clone()
    joint_rel[:, :, :3, 3] = transforms[:, :, :3, 3] - rest_mats[:, :, :3, 0]
    T = torch.einsum("bnj,bjkl->bnkl", mano.weights.expand(batch, -1, -1), joint_rel)
    rest_shape_h = torch.cat([v_posed, torch.ones((batch, v_posed.shape[1], 1), dtype=dtype, device=device)], dim=2)
    verts = torch.matmul(T, rest_shape_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]
    tips = verts[:, mano.fingertip_vertex_ids]
    joints21 = torch.cat([posed_joints, tips], dim=1)[:, mano.mano16_tips_to_openpose21]
    if transl is not None:
        verts = verts + transl.unsqueeze(1)
        joints21 = joints21 + transl.unsqueeze(1)
    return verts, joints21


class EHPEModel(BaseModel):
    """CMP.MODEL ehpe on semkine/cmp_seq_ehpe.py batches (K = num_steps packets, VOXEL_BINS 5, EVENT_BINS 8)."""

    _build_mano = MNISTModel._build_mano
    _fk = MNISTModel._fk
    _resolve_betas_K = MNISTModel._resolve_betas_K
    _intrinsics = MNISTModel._intrinsics
    set_hand_context = MNISTModel.set_hand_context
    LR_FLOOR = 0.02

    def __init__(self, cfg=None, num_classes=None):
        super().__init__(cfg)
        c = self.cfg["CMP"]
        if c["MODEL"] != "ehpe" or self.output_dim != 51 or self.pose_repr != "mano_full_axis_angle":
            raise ValueError("EHPEModel: CMP.MODEL ehpe, 51-D axis-angle state")
        seq, w = c["SEQ"], c["LOSS"]
        self.num_steps = int(seq["K"])
        self.voxel_bins = int(seq["VOXEL_BINS"])
        self.event_bins = int(seq["EVENT_BINS"])
        #: the `args` of the official compute_losses (event_pose_estimation/train.py defaults in CMP.LOSS)
        self.loss_args = SimpleNamespace(
            delta_tran_loss=float(w["DELTA_TRAN"]), tran_loss=float(w["TRAN"]), theta_loss=float(w["THETA"]),
            joints3d_loss=float(w["JOINTS3D"]), joints2d_loss=float(w["JOINTS2D"]), flow_loss=float(w["FLOW"]),
            use_geodesic_loss=bool(w["GEODESIC"]), flow_valid_px=float(w["FLOW_VALID_PX"]))
        self.flow_encoder = FlowNet(official_config(3))                     # EvHandPose's FlowNet, frozen from stage epf
        for p in self.flow_encoder.parameters():
            p.requires_grad = False
        self.img_encoder = resnet50_encoder(self.event_bins + 2)
        self.tmp_encoder = TemporalEncoder(n_layers=1, hidden_size=FEATURE, bidirectional=False, add_linear=False,
                                           use_residual=True)
        self.regressor = Regressor(pose_dim=N_JOINTS * 6)
        self.mse = nn.MSELoss()
        self._build_mano(self.cfg["MODEL"])
        self._ctx_betas = None
        self._ctx_K = None

    # ------------------------------------------------------------------ network pieces
    def flow_px(self, voxel):
        """(N, bins, H, W) voxel grids -> (N, 2, H', W') flow of the packet in pixels on the padded frame (frozen)."""
        self.flow_encoder.eval()
        out = []
        with torch.no_grad():
            for v in voxel.split(FLOW_CHUNK):
                o = self.flow_encoder(F.pad(v, PAD))                        # (n, 4, 2, H', W')
                out.append(o.reshape(o.shape[0], -1, o.shape[-2], o.shape[-1])[:, -2:].float())
        return torch.cat(out) * FLOW_SCALE_PX

    def encode(self, ev, flow):
        """(N, EVENT_BINS, H, W) event frames and (N, 2, H', W') flow -> (N, 2048): the official encoder on [events | flow]."""
        return self.img_encoder(torch.cat([F.pad(ev, PAD), flow.to(ev.dtype)], dim=1))

    def temporal(self, feats, h0=None):
        """The official TemporalEncoder on (B, T, 2048) features from h0 (B, 2048; zeros when None), in float32."""
        feats = feats.float()
        if h0 is None:
            h0 = feats.new_zeros(feats.shape[0], feats.shape[-1])
        with torch.autocast(device_type=feats.device.type, enabled=False):
            return self.tmp_encoder(feats, h0.float()).contiguous()         # official: .contiguous() before the heads

    def temporal_step(self, feat, hidden=None):
        """One recurrent step of the TemporalEncoder: (B, 2048) feature and the GRU state (layers, B, 2048; zeros when
        None) -> (output (B, 2048) with the residual, new GRU state)."""
        x = feat.float().unsqueeze(1)
        if hidden is None:
            hidden = x.new_zeros(self.tmp_encoder.h0_size, x.shape[0], x.shape[-1])
        with torch.autocast(device_type=x.device.type, enabled=False):
            y, hidden = self.tmp_encoder.gru(x, hidden)
        return (y + x)[:, 0], hidden

    # ------------------------------------------------------------------ state <-> rotations
    def state_rotmats(self, state):
        """51-D states (N, 51) -> joint rotations (N, 16, 3, 3): root, then the 15 local joints (mean pose added)."""
        aa = torch.cat([state[:, 3:6], state[:, 6:51] + self.mano.hands_mean.view(1, -1)], dim=1)
        return G.batch_rodrigues(aa.reshape(-1, 3)).view(-1, N_JOINTS, 3, 3)

    def rotmats_state(self, trans, rot):
        """Translation (N, 3) and joint rotations (N, 16, 3, 3) -> 51-D states (the official test's conversion)."""
        aa = G.rotation_matrix_to_angle_axis(rot.reshape(-1, 3, 3).float()).view(-1, N_JOINTS, 3)
        return torch.cat([trans.float(), aa[:, 0], aa[:, 1:].reshape(-1, 45) - self.mano.hands_mean.view(1, -1)], dim=1)

    def compose(self, init_state, delta_pose, delta_tran):
        """The official accumulation from the initial state (EventTrackNet.forward): init_state (B, 51), delta_pose
        (B, T, 16 x 6), delta_tran (B, T, 3) -> rotations (B, T, 16, 3, 3), translations (B, T, 3)."""
        B, T = delta_pose.shape[:2]
        delta_rot = G.rot6d_to_rotmat(delta_pose.float().contiguous()).view(B, T, N_JOINTS, 3, 3)
        rot = G.delta_rotmat_to_rotmat(self.state_rotmats(init_state.float()), delta_rot, T)
        acc, trans = init_state[:, :3].float(), []
        for k in range(T):                             # init + cumsum (torch.cumsum has no deterministic CUDA kernel)
            acc = acc + delta_tran[:, k].float()
            trans.append(acc)
        return rot, torch.stack(trans, dim=1)

    def snippet(self, ev, voxel, init_state, hidden=None):
        """The official forward on a snippet: ev (B, T, bins, H, W), voxel (B, T, 5, H, W), init_state (B, 51)."""
        B, T = ev.shape[:2]
        flow = self.flow_px(voxel.reshape(B * T, *voxel.shape[2:]))
        feats = self.encode(ev.reshape(B * T, *ev.shape[2:]).float(), flow).view(B, T, -1)
        delta_pose, delta_tran = self.regressor(self.temporal(feats, hidden))
        rot, trans = self.compose(init_state, delta_pose, delta_tran)
        return {"rot": rot, "trans": trans, "delta_tran": delta_tran.float(), "flow": flow}

    def snippet_states(self, out):
        """(B, T, 51) states of a `snippet` output."""
        B, T = out["trans"].shape[:2]
        return self.rotmats_state(out["trans"].reshape(B * T, 3), out["rot"].reshape(B * T, N_JOINTS, 3, 3)).view(B, T, 51)

    @torch.no_grad()
    def track_step(self, ev, voxel, prev, hidden=None):
        """One update of the deployed tracker: ev (1, bins, H, W), voxel (1, 5, H, W), prev (1, 51) and the recurrent
        state of the snippet so far (None at a snippet start) -> (state (1, 51), recurrent state)."""
        y, hidden = self.temporal_step(self.encode(ev.float(), self.flow_px(voxel)), hidden)
        delta_pose, delta_tran = self.regressor(y.unsqueeze(1).contiguous())
        rot, trans = self.compose(prev, delta_pose, delta_tran)
        return self.rotmats_state(trans[:, 0], rot[:, 0]), hidden

    # ------------------------------------------------------------------ official losses with MANO
    def loss_inputs(self, out, init_state, target, betas, camera_K):
        """`out` / `target` dicts of the official compute_losses (tran, pred_rotmats / theta, joints3d, joints2d, verts,
        flows, cam_intr) for a `snippet` output and the GT states (B, T, 51); float32."""
        B, T = out["trans"].shape[:2]
        rot, trans = out["rot"].float(), out["trans"].float()
        bt = betas.float().unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
        verts, joints = mano_rotmats(self.mano, bt, rot.reshape(B * T, N_JOINTS, 3, 3))
        tdet = trans.detach().reshape(B * T, 1, 3)                          # official: verts / joints + trans.detach()
        verts = (verts + tdet).view(B, T, -1, 3)
        joints3d = (joints + tdet).view(B, T, -1, 3)
        init_verts, _ = self._fk(init_state.float(), betas.float())
        cam = tuple(v.view(B, 1, 1) for v in self._intrinsics(camera_K.float()))
        H, W = self.render_h, self.render_w
        left, _, top, _ = PAD
        flows = out["flow"].float()[..., top:top + H, left:left + W].reshape(B, T, 2, H, W).contiguous()
        gt = target.float().reshape(B * T, 51)
        _, gt_joints = self._fk(gt, bt)
        gt_joints = gt_joints.view(B, T, -1, 3)
        theta = torch.cat([gt[:, 3:6], gt[:, 6:51] + self.mano.hands_mean.view(1, -1)], dim=1).view(B, T, -1)
        pred = {"delta_tran": out["delta_tran"], "tran": trans.unsqueeze(2), "pred_rotmats": rot, "joints3d": joints3d,
                "joints2d": G.projection_torch(joints3d, cam, H, W),
                "verts": torch.cat([init_verts.unsqueeze(1), verts], dim=1), "cam_intr": cam}
        tgt = {"tran": gt[:, :3].view(B, T, 1, 3), "theta": theta, "joints3d": gt_joints,
               "joints2d": G.projection_torch(gt_joints, cam, H, W), "flows": flows}
        return pred, tgt

    @torch.no_grad()
    def flow_valid_fraction(self, pred, tgt):
        """Share of (step, vertex) pairs the flow-coherence loss uses (|sampled flow| > the validity threshold)."""
        flows = tgt["flows"]
        B, T, _, H, W = flows.shape
        uv = G.projection_torch(pred["verts"][:, :-1], pred["cam_intr"])
        grid = (2.0 * uv / torch.tensor([W, H], device=uv.device) - 1.0).reshape(B * T, 1, -1, 2)
        sampled = F.grid_sample(flows.reshape(B * T, 2, H, W), grid)
        return (sampled.norm(dim=1) > self.loss_args.flow_valid_px).float().mean()

    def losses(self, out, init_state, target, betas, camera_K):
        """The official training loss (sum of the weighted terms) and its parts."""
        with torch.autocast(device_type=out["trans"].device.type, enabled=False):
            pred, tgt = self.loss_inputs(out, init_state, target, betas, camera_K)
            parts = L.compute_losses(pred, tgt, self.mse, out["trans"].device, self.loss_args)
            loss = (parts["delta_tran"] + parts["tran"] + parts["theta"] + parts["joints3d"] + parts["joints2d"]
                    + parts["flow"]).sum()
            logs = {f"loss_{k}": v.detach().sum() for k, v in parts.items()}
            logs["joint_error_mm"] = (pred["joints3d"] - tgt["joints3d"]).detach().norm(dim=-1).mean() * 1000.0
            logs["rotation_error_deg"] = parts["theta"].detach().sum() / max(self.loss_args.theta_loss, 1e-12) * (180.0 / math.pi)
            logs["flow_valid"] = self.flow_valid_fraction(pred, tgt)
        return loss, logs

    # ------------------------------------------------------------------ training
    def training_step(self, batch, batch_nb):
        if not isinstance(batch, dict) or "ev8" not in batch or "voxel" not in batch:
            raise TypeError("EventHPE trains on semkine.cmp_seq_ehpe batches")
        ev, states = batch["ev8"], batch["states"]
        B, K = ev.shape[:2]
        if K != self.num_steps:
            raise ValueError("snippet length differs from CMP.SEQ.K")
        out = self.snippet(ev, batch["voxel"], batch["prev0"])
        loss, logs = self.losses(out, batch["prev0"], states[:, 1:], batch["betas"], batch["camera_K"])
        self.log("train_loss", loss, prog_bar=True, batch_size=B)
        for key, value in logs.items():
            self.log(f"train_{key}", value, batch_size=B)
        return loss.float()

    def init_from_abs_checkpoint(self, ckpt_path):
        """semkine/train.py's MODEL.INIT_FROM: FlowNet (every `flow_encoder.` tensor) from stage epf of the seed."""
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        own = self.state_dict()
        keys = [k for k in own if k.startswith("flow_encoder.")]
        missing = [k for k in keys if k not in sd or sd[k].shape != own[k].shape]
        if missing:
            raise RuntimeError(f"INIT_FROM lacks FlowNet tensors: {missing[:3]}")
        self.load_state_dict({**own, **{k: sd[k] for k in keys}})
        return len(keys)

    def train(self, mode=True):
        super().train(mode)
        self.flow_encoder.eval()                                            # frozen, kept in eval
        return self

    def configure_optimizers(self):
        t = self.cfg["TRAIN"]
        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=float(t["LR"]), weight_decay=float(t.get("WEIGHT_DECAY_L2", 0.0)))
        warmup, total = int(t["WARMUP_STEPS"]), int(t["MAX_STEPS"])

        def lr_lambda(step):
            if step < warmup:
                return float(step + 1) / float(warmup)
            p = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
            return self.LR_FLOOR + (1.0 - self.LR_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * p))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def forward(self, x, prevpos=None, betas=None, camera_K=None):
        raise RuntimeError("EventHPE reads packet snippets: use snippet / track_step")


def model_class(kind):
    if kind != "ehpe":
        raise ValueError(f"semkine.cmp_ehpe has no model kind {kind!r}")
    return EHPEModel


__all__ = ["EHPEModel", "FEATURE", "N_JOINTS", "mano_rotmats", "model_class"]
