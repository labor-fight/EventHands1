"""EVF model, second registry (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 and the rounds that register these arms).

`semkine/evf_model.py` (frozen with round 1) plus three switches; with the round-1 settings an EVFModel2 is the same
network as an EVFModel (same parameters in the same order, same forward):

  NET      point | graph        graph = starting point B (semkine/evf_graph.py): the point features at level 0, three
                                pooled space-time levels with virtual-frame injection and radius edges
  VFMODE   rel | concat         rel = A's virtual-frame features ("event minus prediction": signed distances, directions,
                                nearest-point attributes, position relative to the projected root); concat = the
                                rendered values at the node's own pixel only, appended to the event attributes and the
                                image position (inside, root-relative depth and template coordinates of the surface under
                                the pixel, zero outside), packet statistics in image units plus the projected root and
                                the hand scale: no explicit relative quantity ("只拼接" ablation)
  READOUT  query | surface      surface = slot k pools the node values with the LBS weight of joint k at the predicted
                                surface point under / nearest to the node (soft, from the virtual frame), slot 17 is the
                                plain mean; no learned attention
"""
from __future__ import annotations

import torch

from model import BaseModel, MNISTModel
from semkine import evf_nodes as EN
from semkine.evf_model import EVF_KEYS, RootFrame
from semkine.evf_net import EVFPointNet, compose
from semkine.evf_render import Frame, sample

EVF2_KEYS = {**EVF_KEYS, "VFMODE": "rel", "READOUT": "query"}
NETS2 = ("point", "graph")
NODE_DIM_CONCAT = 7 + 5
PACKET_DIM_CONCAT = EN.PACKET_DIM_NOVF + 3 + 5


def evf2_options(cfg):
    e = dict(cfg.get("EVF") or {})
    unknown = sorted(set(e) - set(EVF2_KEYS))
    if unknown:
        raise ValueError(f"unknown EVF keys: {unknown}")
    o = {**EVF2_KEYS, **e}
    if o["NET"] not in NETS2 or o["VFMODE"] not in ("rel", "concat") or o["READOUT"] not in ("query", "surface"):
        raise ValueError(f"bad EVF switches {o}")
    if int(o["ITERS"]) not in (1, 2):
        raise ValueError("EVF.ITERS must be 1 or 2")
    if (o["LBS"] or o["VFMODE"] == "concat" or o["READOUT"] == "surface") and not o["VF"]:
        raise ValueError("LBS / concat / surface need the virtual frame")
    if o["NET"] == "graph" and (o["VFMODE"] != "rel" or o["READOUT"] != "query"):
        raise ValueError("the graph version uses the rel features and the query readout")
    return o


@torch.no_grad()
def concat_rows(events, attrs, rows, bi, frame, template, H: int, W: int):
    """`(M, NODE_DIM_CONCAT)` features and `(M, 5)` image-moment terms of the nodes: the rendered values at the pixel."""
    px = events[rows, 1].long().clamp(0, W - 1)
    py = events[rows, 2].long().clamp(0, H - 1)
    face = frame.face[bi, py, px]
    from semkine.evf_render import surface_at
    bary, depth = surface_at(frame.u, frame.v, frame.z, frame.faces, bi, face, px.float(), py.float())
    fv = frame.faces.long()[face.clamp_min(0)]
    inside = (face >= 0).float()
    tmpl = (bary[..., None] * template[fv]).sum(1) * inside[:, None]
    dz = (depth - frame.root_z[bi]) * inside
    x, y = px.float() / W, py.float() / H
    feat = torch.cat([attrs[rows].float(), torch.stack([x, y], -1), inside[:, None], (dz / 0.1)[:, None], tmpl / 0.1], -1)
    return feat, torch.stack([x, y, x * x, y * y, x * y], -1)


class SurfacePointNet(EVFPointNet):
    """A's network with the surface readout: no attention logits (the LBS weights pool)."""

    def __init__(self, node_dim, packet_dim, **kw):
        super().__init__(node_dim, packet_dim, **kw)
        del self.att

    def node(self, feat):
        return None, self.val(self.phi(feat)).float()

    def pool_surface(self, value, mask, lbs):
        m = mask.unsqueeze(-1).float()
        w = torch.cat([lbs.float() * m, m], -1)                                    # (B, N, 17)
        tot = w.sum(1)
        slots = torch.einsum("bnk,bnd->bkd", w, value) / tot.clamp_min(1e-6).unsqueeze(-1)
        return slots, torch.log1p(tot)


class EVFModel2(MNISTModel):
    def __init__(self, cfg=None, num_classes=None):
        BaseModel.__init__(self, cfg)
        model_cfg = self.cfg.get("MODEL", {})
        self.evf = evf2_options(self.cfg)
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
        if not vf:
            node_dim, packet_dim = EN.NODE_DIM_NOVF, EN.PACKET_DIM_NOVF
        elif o["VFMODE"] == "concat":
            node_dim, packet_dim = NODE_DIM_CONCAT, PACKET_DIM_CONCAT
        else:
            node_dim, packet_dim = EN.NODE_DIM_VF + (EN.LBS_DIM if o["LBS"] else 0), EN.PACKET_DIM_VF
        kw = dict(slot_dim=int(o["SLOT_DIM"]), readout_hidden=int(o["READOUT_HIDDEN"]), prev48=bool(o["PREV48"]))
        if o["NET"] == "graph":
            from semkine.evf_graph import EVFGraphNet
            self.net = EVFGraphNet(node_dim, packet_dim, hidden=int(o["HIDDEN"]), **kw)
        elif o["READOUT"] == "surface":
            self.net = SurfacePointNet(node_dim, packet_dim, hidden=int(o["HIDDEN"]), layers=int(o["LAYERS"]), **kw)
        else:
            self.net = EVFPointNet(node_dim, packet_dim, hidden=int(o["HIDDEN"]), layers=int(o["LAYERS"]), **kw)
        self.evf_stats = {}

    def _inputs(self, batch, prev, betas, K):
        o = self.evf
        H, W = self.render_h, self.render_w
        fx, fy, cx, cy = self._intrinsics(K)
        extra = {}
        with torch.no_grad(), torch.autocast(device_type=prev.device.type, enabled=False):
            verts, joints = self._fk(prev.float(), betas)
            vf = bool(o["VF"])
            frame = Frame(verts, joints, self.mano.f, fx, fy, cx, cy, H, W) if vf else RootFrame(joints, fx, fy, cx, cy)
            src, mask, n_nodes = EN.select_nodes(batch.events, batch.ptr, batch.delta_t_s, H, W, int(o["VOXEL_PX"]),
                                                 int(o["TIME_BINS"]), int(o["MAX_NODES"]))
            attrs = EN.event_attributes(batch.events, batch.ptr, batch.delta_t_s, H, W)
            Bn, N = src.shape
            flat = mask.reshape(-1).nonzero().squeeze(1)
            rows = src.reshape(-1)[flat]
            bi = torch.div(flat, N, rounding_mode="floor")
            if vf and o["VFMODE"] == "concat":
                f, mom = concat_rows(batch.events, attrs, rows, bi, frame, self.evf_template, H, W)
            else:
                f, mom = EN.node_rows(batch.events, attrs, rows, bi, frame, self.evf_template, H, W, vf,
                                      self.evf_lbs if o["LBS"] else None)
            feat = EN._scatter_rows(flat, f, Bn, N)
            sums = EN._scatter_rows(flat, mom, Bn, N).sum(1)
            if vf and o["VFMODE"] == "concat":
                base = EN.packet_from_sums(batch.counts, n_nodes, batch.delta_t_s, mask.sum(1), sums, frame, vf=False)
                sm = frame.sil_moments
                root = torch.stack([frame.root_u / W, frame.root_v / H, frame.scale / 100.0], -1)
                packet = torch.cat([base, root, sm], -1)
            else:
                packet = EN.packet_from_sums(batch.counts, n_nodes, batch.delta_t_s, mask.sum(1), sums, frame, vf=vf)
            if o["READOUT"] == "surface":
                px = batch.events[rows, 1].long().clamp(0, W - 1)
                py = batch.events[rows, 2].long().clamp(0, H - 1)
                smp = sample(frame, bi, px, py, self.evf_template, frame.root_z, self.evf_lbs)
                extra["lbs"] = EN._scatter_rows(flat, smp["lbs_s"], Bn, N)
            if o["NET"] == "graph":
                from semkine.evf_graph import build_levels
                extra["levels"] = build_levels(batch.events, src, mask, batch.delta_t_s, frame, self.evf_template, H,
                                               W, vf=vf)
        self.evf_stats = {"nodes": n_nodes.detach(), "kept": mask.sum(1).detach(),
                          "clipped_faces": getattr(frame, "clipped", torch.zeros_like(n_nodes)).detach()}
        return feat, mask, packet, joints[:, 0], frame, (fx, fy, cx, cy), extra

    def _update(self, batch, prev, betas, K):
        o = self.evf
        feat, mask, packet, root, frame, (fx, fy, cx, cy), extra = self._inputs(batch, prev, betas, K)
        p48 = prev[:, 3:51] if o["PREV48"] else None
        if o["NET"] == "graph":
            params = self.net(feat, mask, extra["levels"], packet, p48)
        elif o["READOUT"] == "surface":
            _, value = self.net.node(feat)
            slots, mass = self.net.pool_surface(value, mask, extra["lbs"])
            params = self.net.readout(slots, mass, packet, p48).float()
        else:
            params = self.net(feat, mask, packet, p48)
        return compose(prev, params, root, frame.scale, fx, fy, cx, cy)

    def forward_packet(self, batch):
        prev = batch.prev_state.float()
        betas, K = self._resolve_betas_K(prev, batch.betas, batch.camera_K)
        out = prev
        for _ in range(int(self.evf["ITERS"])):
            out = self._update(batch, out, betas, K)
        empty = (batch.counts <= 0).unsqueeze(-1)
        return torch.where(empty, prev, out)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        raise RuntimeError("the EVF model reads raw events; call forward_packet")


__all__ = ["EVF2_KEYS", "EVFModel2", "NODE_DIM_CONCAT", "PACKET_DIM_CONCAT", "SurfacePointNet", "concat_rows",
           "evf2_options"]

