"""S43 deviation 9: the event-driven tracker's query as ONE CUDA graph per capacity bucket.

`semkine.async_tracker.EventDrivenTracker` issues ~400 kernels and synchronises with the host ~30 times per query (the
sparse executor's index building, the filter, the mesh cache): 9.7 ms on an L20 of which the GPU is busy about 1 ms.
`GraphTracker` keeps the same contract (set_hand / reset / ingest / query, the very same arithmetic) with the whole
query recorded as a CUDA graph on static device buffers:

    SAE (device, written per event by `ingest`)  -> LNES of the window [start, t]        (static: start, w are 0-d buffers)
    state (1, 51) -> MANO FK -> point splat (render_fast.RenderFast: no boolean indexing, Python-int kinematic chain)
    [LNES | render] -> StaticSparseTrunk (bucket b) -> prev_mlp -> zero-event hold -> AdaptiveFilter (device, float64)
    -> update dead band -> out (1, 51); the level counts of the input -> cnt (5,)

Host work per query: four scalar writes, one graph replay, one device-to-host copy (out + cnt, 56 numbers), the bucket
check. The bucket is the one that fitted last time; if the counts read back exceed it the query is replayed with the
next bucket (the state is only committed after the check), and after `down_after` consecutive queries that would have
fitted the smaller bucket the tracker steps down. The incremental mesh cache (`IncrementalMano`) is updated on the host
from the committed state, off the device's critical path (`mesh=True`; the render inside the graph uses the full FK).
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from semkine.anchored import AdaptiveFilter
from semkine.async_tracker import IncrementalMano, _Fixed, _NEVER
from semkine.kinematic_gate import deadband_state

__all__ = ["GraphTracker"]


class GraphTracker:
    def __init__(self, model, spec: Optional[dict], window_ms: int = 200, hop_ms: int = 50, device=None,
                 height: int = 180, width: int = 240, eps_f: float = 0.0, eps_r: float = 0.0, buckets=None,
                 mesh: bool = True, down_after: int = 20, graph: bool = True):
        from deploy_fast import enable_fast_render
        from sparse_graph import BUCKETS, StaticSparseTrunk
        if not getattr(model, "sparse_trunk", False):
            raise ValueError("GraphTracker runs an S43 checkpoint (MODEL.CNN_BACKBONE sparse_*)")
        if getattr(model, "transl_head", None) is not None:
            raise NotImplementedError("GraphTracker: MODEL.TRANSL_HEAD is not wired (the candidate has none)")
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("GraphTracker needs the model on a CUDA device")
        enable_fast_render(self.model)                                    # sync-free raster + kinematic chain
        self.st = StaticSparseTrunk(model.conv1, model.rn, height, width, buckets or BUCKETS)
        self.filter = None if spec is None else AdaptiveFilter.from_spec(_Fixed(), spec).to(self.device).eval()
        self.window_ms, self.hop_ms, self.h, self.w = int(window_ms), int(hop_ms), height, width
        self.eps_f, self.eps_r = float(eps_f), float(eps_r)
        self.use_mesh, self.down_after, self.use_graph = bool(mesh), int(down_after), bool(graph)
        dev = self.device
        self.sae = torch.full((height * width * 2,), _NEVER, dtype=torch.long, device=dev)
        self.start_buf = torch.zeros((), dtype=torch.long, device=dev)
        self.w_buf = torch.ones((), dtype=torch.float32, device=dev)
        self.n_buf = torch.zeros(1, dtype=torch.float64, device=dev)
        self.state_buf = torch.zeros(1, 51, device=dev)
        self.out_buf = torch.zeros(1, 51, device=dev)
        self.cnt_buf = torch.zeros(5, dtype=torch.long, device=dev)
        self.betas_buf = torch.zeros(1, 10, device=dev)
        self.K_buf = torch.eye(3, device=dev)[None].clone()
        self.pin = torch.zeros(56, dtype=torch.float64).pin_memory()      # out (51) + counts (5), one D2H per query
        self.ret_buf = torch.zeros(56, dtype=torch.float64, device=dev)
        self.ev_n: Dict[int, int] = {}
        self._graphs = {}
        self.bucket = 1
        self._fit_smaller = 0
        self.timing = {"ingest": 0.0, "query": 0.0, "n_query": 0, "n_events": 0, "replays": 0, "overflow": 0}
        self.macs = []

    # ---- context
    def set_hand(self, betas: torch.Tensor, camera_K: torch.Tensor):
        self.sae.fill_(_NEVER)
        self.ev_n = {}
        b = betas.reshape(1, -1).float().to(self.device)
        k = camera_K.reshape(1, 3, 3).float().to(self.device)
        self.model.set_hand_context(b, k)                 # the model's context (its render cache keys on these tensors)
        self.betas_buf.copy_(b)                           # the graph reads the static buffers: no re-capture per sequence
        self.K_buf.copy_(k)
        if self.use_mesh:
            self.mesh = IncrementalMano(self.model.mano, self.betas_buf, self.device)

    def reset(self, state51: torch.Tensor, floor_ms: int):
        self.state_buf.copy_(state51.reshape(1, 51).float())
        self.floor = int(floor_ms)
        if self.filter is not None:
            self.filter.reset_state(self.state_buf)
        if self.use_mesh:
            self.mesh.reset(self.state_buf[0])

    @property
    def state(self):
        return self.state_buf

    @state.setter
    def state(self, v):
        self.state_buf.copy_(v.reshape(1, 51).float())

    # ---- events
    @torch.no_grad()
    def ingest(self, xs, ys, ps, t_ms):
        t0 = time.perf_counter()
        xs = torch.as_tensor(xs, dtype=torch.long, device=self.device)
        ys = torch.as_tensor(ys, dtype=torch.long, device=self.device)
        ps = torch.as_tensor(ps, dtype=torch.long, device=self.device).clamp(0, 1)
        self.sae.scatter_reduce_(0, (ys * self.w + xs) * 2 + ps, torch.full_like(xs, int(t_ms)), reduce="amax")
        self.ev_n[int(t_ms)] = self.ev_n.get(int(t_ms), 0) + int(xs.numel())
        self.timing["ingest"] += time.perf_counter() - t0
        self.timing["n_events"] += int(xs.numel())

    def _recent_count(self, t_ms: int) -> int:
        lo = t_ms - self.hop_ms + 1
        for k in [k for k in self.ev_n if k < lo]:
            del self.ev_n[k]
        return int(sum(n for k, n in self.ev_n.items() if k <= t_ms))

    # ---- the stage (eager and under capture)
    def _stage(self, bucket: int) -> None:
        m = self.model
        s = self.sae.view(self.h, self.w, 2)
        val = (s - self.start_buf).to(torch.float32) / self.w_buf
        lnes = torch.where(s >= self.start_buf, val, torch.zeros_like(val))
        rend = m._render_prev(self.state_buf, self.betas_buf, self.K_buf)[0].to(lnes.dtype)
        img = torch.cat([lnes, rend], dim=-1)
        self.cnt_buf.copy_(self.st.counts(img))
        logits = self.st.run(img, bucket)[None]
        prev = self.state_buf
        pm = m.prev_mlp(prev)
        if not m.prev_mlp_transl:
            t = m.slices.transl
            pm = torch.cat([pm[:, : t.start], torch.zeros_like(pm[:, t]), pm[:, t.stop:]], dim=-1)
        empty = (lnes.abs().sum() <= 0).reshape(1, 1)
        z = torch.where(empty, prev, prev + (logits + pm))
        out = z
        if self.filter is not None:
            self.filter.trk_model.z = z
            out = self.filter((~empty).to(torch.float32), prev, n_events=self.n_buf)
        out = torch.where(empty, prev, deadband_state(prev, out, self.eps_f, self.eps_r))
        self.out_buf.copy_(out)
        self.ret_buf[:51].copy_(out[0].to(torch.float64))
        self.ret_buf[51:].copy_(self.cnt_buf.to(torch.float64))

    def _graph(self, bucket: int):
        if bucket not in self._graphs:
            from render_fast import capture_checked
            snap = (self.state_buf.clone(), self.out_buf.clone())
            g, _ = capture_checked(lambda: self._stage(bucket))
            self.state_buf.copy_(snap[0])
            self.out_buf.copy_(snap[1])
            self._graphs[bucket] = g
        return self._graphs[bucket]

    def _run(self, bucket: int):
        if self.use_graph:
            self._graph(bucket).replay()
        else:
            with torch.no_grad():
                self._stage(bucket)
        self.timing["replays"] += 1
        self.pin.copy_(self.ret_buf, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        r = self.pin.numpy()
        return r[:51].astype(np.float32).copy(), [int(v) for v in r[51:]]

    # ---- one query
    def query(self, t_ms: int) -> Dict:
        t0 = time.perf_counter()
        w = min(self.window_ms, t_ms - self.floor + 1)
        self.start_buf.fill_(t_ms - w + 1)
        self.w_buf.fill_(float(w))
        n_ev = self._recent_count(t_ms)
        self.n_buf.fill_(float(n_ev))
        b = self.bucket
        while True:
            out, cnt = self._run(b)
            fit = self.st.bucket_for(cnt)
            if fit is None:
                raise RuntimeError(f"active set {cnt} exceeds every bucket")
            if fit <= b:
                break
            self.timing["overflow"] += 1
            b = fit
        if fit < b:
            self._fit_smaller += 1
            if self._fit_smaller >= self.down_after:
                b, self._fit_smaller = b - 1, 0
        else:
            self._fit_smaller = 0
        self.bucket = b
        self.state_buf.copy_(self.out_buf)
        info = {"joints_changed": -2, "verts_reskinned": 0}
        if self.use_mesh:
            info = self.mesh.update(torch.from_numpy(out).to(self.device))
        self.timing["query"] += time.perf_counter() - t0
        self.timing["n_query"] += 1
        self.macs.append(0)
        return {"state": self.state_buf, "macs": 0, "n_events": n_ev, "window": w, "mesh": info, "gains": None,
                "bucket": b, "counts": cnt}
