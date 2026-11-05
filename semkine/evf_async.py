"""EVF asynchronous runtime (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): the point model (A) driven event by
event between two state updates, equal to `EVFModel.forward_packet` on the whole packet.

    rt = AsyncPoint(model)
    rt.begin(prev, betas, K, delta_t_s)        state update: FK, the virtual frame, empty maps and aggregates
    rt.push(events)                            any number of times, events (n, 4) [x, y, t_rel, p] in arrival order
    state = rt.query()                         the readout on the aggregates (prev bitwise when no event arrived)

Per arriving event, O(1) work against the previous state's virtual frame: the per-pixel last-event times (same /
opposite polarity), the node-cell occupancy (a new node only if its cell is empty, so the node set is the synchronous
one), and for a new node only: its features, the shared MLP, an online-softmax update of the 17 pooled queries
(running max, normaliser, weighted value sum) and the moment sums. Nothing already computed is recomputed. A chunk of
events is processed together with the history it needs (each touched pixel's last times, as pseudo-events ahead of the
chunk), which gives the same numbers as one event at a time.

One update per packet only (EVF.ITERS 1): a second iteration re-renders with the first result and reads every node
again. A packet whose node count exceeds EVF.MAX_NODES is refused (the synchronous form strides it).
"""
from __future__ import annotations

import torch

from semkine import evf_nodes as EN
from semkine.evf_net import MASK_FILL, compose
from semkine.evf_render import Frame


class AsyncPoint:
    def __init__(self, model):
        if int(model.evf["ITERS"]) != 1 or model.evf["NET"] != "point":
            raise ValueError("the asynchronous runtime covers the point model with one update per packet")
        self.m = model

    @torch.no_grad()
    def begin(self, prev, betas, K, delta_t_s):
        m, o = self.m, self.m.evf
        self.prev = prev.float().view(1, -1)
        self.dev = self.prev.device
        H, W = m.render_h, m.render_w
        self.H, self.W = H, W
        betas, K = m._resolve_betas_K(self.prev, betas, K)
        self.intr = m._intrinsics(K)
        fx, fy, cx, cy = self.intr
        with torch.autocast(device_type=self.dev.type, enabled=False):
            verts, joints = m._fk(self.prev, betas)
        self.root = joints[:, 0].float()
        self.vf = bool(o["VF"])
        from semkine.evf_model import RootFrame
        self.frame = Frame(verts, joints, m.mano.f, fx, fy, cx, cy, H, W) if self.vf else \
            RootFrame(joints, fx, fy, cx, cy)
        self.dt = torch.tensor([float(delta_t_s)], device=self.dev)
        self.last_t = torch.full((2, H * W), -1.0, device=self.dev)
        gw, gh = (W + int(o["VOXEL_PX"]) - 1) // int(o["VOXEL_PX"]), (H + int(o["VOXEL_PX"]) - 1) // int(o["VOXEL_PX"])
        self.occupied = torch.zeros(int(o["TIME_BINS"]) * 2 * gh * gw, dtype=torch.bool, device=self.dev)
        K_ = self.m.net.att.out_features
        D = self.m.net.val.out_features
        self.run_max = torch.full((K_,), MASK_FILL, device=self.dev)
        self.run_sum = torch.zeros(K_, device=self.dev)
        self.run_val = torch.zeros(K_, D, device=self.dev)
        self.mass = torch.zeros(K_, device=self.dev)
        self.mom = torch.zeros(EN.MOMENT_DIM_VF if self.vf else EN.MOMENT_DIM_NOVF, device=self.dev)
        self.n_events = 0
        self.n_nodes = 0

    @torch.no_grad()
    def push(self, ev):
        """`ev (n, 4)`: x, y, t_rel (s), polarity, in arrival order (after every event already pushed)."""
        n = int(ev.shape[0])
        if n == 0:
            return
        m, o, H, W = self.m, self.m.evf, self.H, self.W
        ev5 = torch.cat([torch.zeros(n, 1, device=self.dev), ev.float().to(self.dev)], 1)
        pix = ev5[:, 2].long().clamp(0, H - 1) * W + ev5[:, 1].long().clamp(0, W - 1)
        pol = (ev5[:, 4] > 0.5).long()
        t = ev5[:, 3]
        # history of the touched pixels as pseudo-events ahead of the chunk (their order among themselves is free)
        upix = torch.unique(pix)
        hp, hq = upix.repeat(2), torch.arange(2, device=self.dev).repeat_interleave(upix.numel())
        ht = self.last_t[hq, hp]
        keep = ht >= 0
        hp, hq, ht = hp[keep], hq[keep], ht[keep]
        nh = int(hp.numel())
        same_t, opp_t = EN.previous_times(torch.cat([hp, pix]), torch.cat([hq, pol]), torch.cat([ht, t]))
        attrs = EN.attributes_from_times(pol, t, same_t[nh:], opp_t[nh:], self.dt.expand(n))
        # the last time of every touched (polarity, pixel): the last chunk row of each
        key = pol * (H * W) + pix
        rev = torch.flip(torch.arange(n, device=self.dev), [0])
        last_rows = rev[EN.first_of_keys(torch.flip(key, [0]))]
        self.last_t[pol[last_rows], pix[last_rows]] = t[last_rows]
        self.n_events += n
        # new nodes: the first chunk row of every cell that is still empty
        cell = EN.node_keys(ev5, self.dt, H, W, int(o["VOXEL_PX"]), int(o["TIME_BINS"]))
        first = EN.first_of_keys(cell)
        new = first[~self.occupied[cell[first]]]
        if new.numel() == 0:
            return
        self.occupied[cell[new]] = True
        self.n_nodes += int(new.numel())
        if self.n_nodes > int(o["MAX_NODES"]):
            raise ValueError("node cap exceeded: the asynchronous runtime does not stride")
        bi = torch.zeros_like(new)
        feat, mom = EN.node_rows(ev5, attrs, new, bi, self.frame, m.evf_template, H, W, self.vf,
                                 m.evf_lbs if o["LBS"] else None)
        logit, value = m.net.node(feat)
        mx = torch.maximum(self.run_max, logit.max(0).values)
        scale = torch.exp(self.run_max - mx)
        e = torch.exp(logit - mx[None])
        self.run_sum = self.run_sum * scale + e.sum(0)
        self.run_val = self.run_val * scale[:, None] + torch.einsum("nk,nd->kd", e, value)
        self.run_max = mx
        self.mass = self.mass + torch.sigmoid(logit).sum(0)
        self.mom = self.mom + mom.sum(0)

    @torch.no_grad()
    def query(self):
        m, o = self.m, self.m.evf
        if self.n_events == 0:
            return self.prev.clone()
        slots = (self.run_val / self.run_sum.clamp_min(1e-12)[:, None])[None]
        mass = torch.log1p(self.mass)[None]
        dev = self.dev
        packet = EN.packet_from_sums(torch.tensor([self.n_events], device=dev), torch.tensor([self.n_nodes], device=dev),
                                     self.dt, torch.tensor([self.n_nodes], device=dev), self.mom[None], self.frame,
                                     vf=self.vf)
        params = m.net.readout(slots, mass, packet, self.prev[:, 3:51] if o["PREV48"] else None).float()
        fx, fy, cx, cy = self.intr
        return compose(self.prev, params, self.root, self.frame.scale, fx, fy, cx, cy)


__all__ = ["AsyncPoint"]
