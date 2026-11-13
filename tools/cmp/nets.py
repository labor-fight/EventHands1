"""CMP networks for PFO's frozen engine (tools/pfo/engine.py; docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110).

`load(name, spec=..., device=...)` returns an object with the engine's `Net` interface (`cfg`, `identity`, `bind(ctx)`,
`step(ctx, end, prev)` -> the 51-D state after the 50 ms update ending at `end`). Spec: {kind: cmp, model, checkpoint,
config, form}. Every window is clipped at the start of the valid run that contains `end` (the engine's segments), and
every event image is built by the frame stages' own builder (semkine/eval_track.py `build_lnes`: ms bins, as the
training splat), the voxel grid by semkine/cmp_seq.py `voxel_grid` on the microsecond stamps.

  eh   LNES over min(WINDOW_MS, end - run start + 1) ms (form `w<ms>` overrides WINDOW_MS); the state is not read
  em   LNES of the 50 ms packet; prev + gated delta
  eg   sliding form (main): the GRU over the last HISTORY packets of the run (ends end, end - 50, ...; per-packet
       features cached) from a zero hidden state, the state of the last one -- the training form; form `rolling`
       (descriptive): one hidden state carried from the run start (sequential calls only)
  ep   the same sliding form over EvHandPose's per-packet fusion features (FlowNet + fusion encoder cached), ConvGRU
       from a zero state, the predictor of the last packet
  epc  (amendment 1) ep on EvHandPose's 128 x 128 hand crop: the crop of the packet ending at `end` comes from `prev`,
       the state at the packet's start (its 21 MANO joints projected with the event-frame intrinsics, no shift), the
       packet's feature and crop are cached, and the last packet's crop decodes the translation
  eho  (amendment 4) EventHands as published: the 12-D output of the network on the LNES of the WINDOW_MS (100) ms ending
       at `end`, decoded to the 51-D state; forms `kf` / `kff` (descriptive): the official pipeline, the network at
       1 kHz over the run (window min(WINDOW_MS, elapsed) at every millisecond) followed by the official Kalman filter
       (filter/filter.py, filter_fast.py), the filtered output at `end`
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import require, sha

PACKET_MS = 50
FORMS = ("main", "rolling") + tuple(f"w{w}" for w in (50, 100, 200)) + ("warm", "gtcrop", "kf", "kff")
#: amendment 3 (#cmp-amend3-20261110, descriptive): `warm` = history windows not clipped at the segment start (eh: the
#: whole WINDOW_MS; eg / ep / epc: HISTORY packets, those before the start included; epc crops them around the state the
#: tracker is initialised with); `gtcrop` (epc only, the official evaluation's crop, not deployable) = warm history with
#: every packet cropped around the GT state at its start (GT(a) for packets before the segment start a)
#: amendment 4 (#cmp-amend4-20261110): the official EventHands Kalman filters, (R scale, Q variance) of filter/filter.py
#: (`kf`) and filter/filter_fast.py (`kff`); dt = 1 ms, the filters run on the 1 kHz outputs
KF = {"kf": (5.0, 0.1), "kff": (1.0, 3.0)}
#: amendment 6 (#cmp-amend6-20261110): the deployed step of an extension arm's model kind is an `Adapter(net)` of its own
#: module (bind(ctx), step(ctx, end, prev) -> 51-D state, FORMS, and for the cost tool history, cost_build(ctx, end, a,
#: prev) -> item, cost_step(item, prev, state) -> state tensor, extra_macs(state)); CMP.EVAL.AS evaluates a model kind
#: with the adapter code of another (epx: the cropped EvHandPose's)
EXT_ADAPTERS = {"ehpe": "tools.cmp.ext_ehpe", "fme": "tools.cmp.ext_fme"}
KF_DT = 1e-3
KF_BATCH = 50


def kf_update(state, z, r, q):
    """One output of the official Kalman filter (filter/filter.py with (r, q) = (5, 0.1), filter_fast.py with (1, 3)).

    The script's filter is one 24-D filterpy KalmanFilter whose F / H / Q are block-diagonal over the twelve outputs
    (position and velocity per step, F = [[1, 1], [0, 1]]), Q = q x Q_discrete_white_noise(2, dt = 1 ms), R = r I, P0 = I,
    filterpy's predict and Joseph-form update: twelve independent two-state filters, run here as such (state = (x (12, 2),
    P (12, 2, 2)), None before the first output). The first output sets the position with velocity 0; the script writes
    `np.r_[z, 0 z]` into the interleaved layout, which scrambles the first state, and the intended initialisation is used.
    """
    if state is None:
        return np.stack([z, np.zeros_like(z)], -1), np.repeat(np.eye(2)[None], len(z), 0)
    x, P = state
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[KF_DT ** 4 / 4, KF_DT ** 3 / 2], [KF_DT ** 3 / 2, KF_DT ** 2]])
    H = np.array([1.0, 0.0])
    x = x @ F.T
    P = F @ P @ F.T + Q
    K = P[:, :, 0] / (P[:, 0, 0] + r)[:, None]                                 # P H^T (H P H^T + R)^-1
    x = x + K * (z - x[:, 0])[:, None]
    I_KH = np.eye(2)[None] - K[:, :, None] * H[None, None, :]
    return x, I_KH @ P @ I_KH.transpose(0, 2, 1) + r * K[:, :, None] * K[:, None, :]


def window_raw(ctx, end, window):
    """Events of [end - window + 1, end]: xs, ys, ps (0 / 1), us (microseconds from the window start)."""
    start = end - window + 1
    a0, a1 = int(ctx.offsets[start]), int(ctx.offsets[end + 1])
    if a1 <= a0:
        z = np.zeros(0, np.int64)
        return z, z, z, z
    ev = np.asarray(ctx.events[a0:a1])
    counts = np.diff(np.asarray(ctx.offsets[start:end + 2])).astype(np.int64)
    us = np.repeat(np.arange(window, dtype=np.int64), counts) * 1000
    if ctx.tsub is not None:
        us = us + np.asarray(ctx.tsub[a0:a1]).astype(np.int64)
    return ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64), np.clip(ev[:, 2].astype(np.int64), 0, 1), us


class CmpNet:
    kind = "cmp"

    def __init__(self, name, model, cfg, device, identity, form):
        self.name, self.model, self.cfg, self.device, self.identity = name, model, cfg, device, identity
        ev = cfg["CMP"].get("EVAL", {})
        self.model_kind = ev.get("AS", cfg["CMP"]["MODEL"])
        self.form = form
        self.ext = None
        if self.model_kind in EXT_ADAPTERS:                                    # amendment 6
            import importlib
            self.ext = importlib.import_module(EXT_ADAPTERS[self.model_kind]).Adapter(self)
        self.window = int(form[1:]) if form[:1] == "w" and form[1:].isdigit() else int(ev.get("WINDOW_MS", PACKET_MS))
        self.history = int(ev.get("HISTORY", 1))
        require(form in (FORMS if self.ext is None else self.ext.FORMS), "unknown form " + form)
        require(not (form.startswith("w") and form != "warm") or self.model_kind == "eh", "window forms are EventHands readings")
        require(form != "rolling" or self.model_kind == "eg", "the rolling form is the GRU's")
        require(form != "warm" or self.model_kind in ("eh", "eg", "ep", "epc"), "warm history: eh / eg / ep / epc")
        require(form != "gtcrop" or self.model_kind == "epc", "the GT crop is the cropped EvHandPose's diagnostic")
        require(form not in KF or self.model_kind == "eho", "the Kalman forms are the published EventHands pipeline's")
        require(self.model_kind != "eho" or form in ("main", *KF), "EventHands as published: main / kf / kff")
        from semkine import events as EV
        self.channels = EV.event_channels(cfg)
        self.route = None
        self.tau = None

    # ------------------------------------------------------------------ engine interface
    def bind(self, ctx):
        self.model.set_hand_context(ctx.betas, ctx.K)
        self.runs = np.asarray(ctx.aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        self.cache, self.rolling, self.kf = {}, None, None
        if self.ext is not None:
            self.ext.bind(ctx)

    def run_start(self, end):
        i = int(np.searchsorted(self.runs[:, 0], end, side="right")) - 1
        require(i >= 0 and self.runs[i, 0] <= end < self.runs[i, 1], f"endpoint {end} outside every valid run")
        return int(self.runs[i, 0])

    def lnes(self, ctx, end, window):
        import torch
        from semkine import eval_track as ET
        x = ET.build_lnes(ctx.events, ctx.offsets, int(end), int(window), self.channels)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)

    def packet_feature(self, ctx, end):
        """Cached per-packet feature of the packet ending at `end` (a function of its events only)."""
        if end in self.cache:
            return self.cache[end]
        if self.model_kind == "eg":
            f = self.model.features(self.lnes(ctx, end, PACKET_MS))
        else:
            import torch
            from semkine.cmp_seq import voxel_grid
            xs, ys, ps, us = window_raw(ctx, end, PACKET_MS)
            H, W = int(self.cfg["DATA"]["HEIGHT"]), int(self.cfg["DATA"]["WIDTH"])
            vox = torch.from_numpy(voxel_grid(xs, ys, ps, us, PACKET_MS, self.model.voxel_bins, H, W))
            f = self.model.packet_features(self.lnes(ctx, end, PACKET_MS), vox.unsqueeze(0).to(self.device))
        self.cache[end] = f
        return f

    def crop_feature(self, ctx, end, prev_t):
        """(feature, crop) of the packet ending at `end`, cut to EvHandPose's crop around the state `prev_t`."""
        lnes, vox, crop = self.crop_inputs(ctx, end, prev_t)
        return self.model.packet_features(lnes, vox), crop

    def crop_inputs(self, ctx, end, prev_t):
        """(LNES, voxel grid, crop) on the device of the packet ending at `end`, cut around the state `prev_t`."""
        import torch
        from semkine.cmp_seq import voxel_grid
        from semkine.cmp_seq_crop import crop_events, crop_params
        from semkine import events as EV
        S, rate = int(self.cfg["CMP"]["CROP"]["SIZE"]), float(self.cfg["CMP"]["CROP"]["RATE"])
        rs = float(self.cfg["MODEL"]["RENDER_SCALE"])
        K = ctx.K.view(3, 3).float().cpu().numpy()
        fx, fy, cx, cy = K[0, 0] * rs, K[1, 1] * rs, K[0, 2] * rs, K[1, 2] * rs
        with torch.no_grad():
            _, joints = self.model._fk(prev_t.float(), ctx.betas.float())
        J = joints[0].double().cpu().numpy()
        z = np.clip(J[:, 2], 1e-6, None)
        j2 = np.stack([fx * J[:, 0] / z + cx, fy * J[:, 1] / z + cy], -1)
        cxy = np.array([cx, cy])
        scale, tl = crop_params(j2, cxy, S, rate)
        xs, ys, ps, us = window_raw(ctx, end, PACKET_MS)
        u, v, keep = crop_events(xs, ys, scale, tl, cxy, S)
        lnes = EV.splat_event_image(u[keep], v[keep], ps[keep], us[keep] // 1000, PACKET_MS, S, S, self.channels)
        vox = voxel_grid(u[keep], v[keep], ps[keep], us[keep], PACKET_MS, self.model.voxel_bins, S, S)
        crop = torch.tensor([[scale, tl[0], tl[1], fx, fy, cx, cy]], dtype=torch.float32, device=self.device)
        return (torch.from_numpy(lnes).unsqueeze(0).to(self.device), torch.from_numpy(vox).unsqueeze(0).to(self.device),
                crop)

    def kf_filter(self, ctx, end, a):
        """The official Kalman filter (form kf / kff, `kf_update`) on the network's 1 kHz outputs of the run [a, end],
        the window of every millisecond m clipped at the run start (min(WINDOW_MS, m - a + 1)); the filtered output at
        `end`. A pure function of the run's events up to `end`: the cache only spares recomputation for the engine's
        in-order calls."""
        import torch
        r, q = KF[self.form]
        c = self.kf
        if c is None or c["a"] != a or c["t"] > end:
            c = self.kf = {"a": a, "t": a - 1, "state": None}
        ms = np.arange(c["t"] + 1, end + 1)
        for lo in range(0, len(ms), KF_BATCH):
            chunk = [int(m) for m in ms[lo:lo + KF_BATCH]]
            x = torch.cat([self.lnes(ctx, m, min(self.window, m - a + 1)) for m in chunk])
            with torch.no_grad():
                zs = self.model(x, None).double().cpu().numpy()
            for z in zs:
                c["state"] = kf_update(c["state"], z, r, q)
        c["t"] = end
        return c["state"][0][:, 0]

    def step(self, ctx, end, prev):
        import torch
        end = int(end)
        if self.ext is not None:                                               # amendment 6
            value = np.asarray(self.ext.step(ctx, end, prev), np.float32)
            if value.shape != (51,) or not np.isfinite(value).all():
                raise FloatingPointError(f"nonfinite endpoint retained as failure: {ctx.seq}:{end}")
            return value
        a = self.run_start(end)
        prev_t = torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(self.device)
        if self.model_kind == "eho":                                           # amendment 4
            if self.form in KF:
                out = torch.from_numpy(self.kf_filter(ctx, end, a)).float().view(1, -1).to(self.device)
            else:
                out = self.model(self.lnes(ctx, end, min(self.window, end - a + 1)), prev_t)
            value = self.model.state(out).detach().float().cpu().numpy()[0].copy()
            if value.shape != (51,) or not np.isfinite(value).all():
                raise FloatingPointError(f"nonfinite endpoint retained as failure: {ctx.seq}:{end}")
            return value
        warm = self.form in ("warm", "gtcrop")
        lo = PACKET_MS - 1 if warm else a + PACKET_MS - 1                     # earliest packet end in the history
        if self.model_kind == "eh":
            pred = self.model(self.lnes(ctx, end, min(self.window, end + 1 if warm else end - a + 1)), prev_t)
        elif self.model_kind == "em":
            pred = self.model(self.lnes(ctx, end, PACKET_MS), prev_t)
        elif self.model_kind == "epc":
            ends = [e for e in range(end - PACKET_MS * (self.history - 1), end + 1, PACKET_MS) if e >= lo]
            for e in ends:
                if e not in self.cache:
                    if self.form == "gtcrop":
                        st = np.asarray(ctx.gt[max(e - PACKET_MS, a)], np.float32)
                        state = torch.from_numpy(np.ascontiguousarray(st)).view(1, -1).to(self.device)
                    else:
                        state = prev_t                      # this packet's start state; the initial state before a
                    self.cache[e] = self.crop_feature(ctx, e, state)
            feats = torch.stack([self.cache[e][0] for e in ends], 1)
            crops = torch.stack([self.cache[e][1] for e in ends], 1)
            pred = self.model.sequence_state(feats, crops)
        elif self.form == "rolling":
            f = self.packet_feature(ctx, end)
            h0 = self.rolling[1] if self.rolling is not None and self.rolling[0] == end - PACKET_MS else None
            y, hn = self.model.run(f.unsqueeze(1), h0)
            self.rolling = (end, hn)
            pred = y[:, -1]
        else:
            ends = [e for e in range(end - PACKET_MS * (self.history - 1), end + 1, PACKET_MS) if e >= lo]
            feats = torch.stack([self.packet_feature(ctx, e) for e in ends], 1)
            pred = self.model.run(feats)[0][:, -1] if self.model_kind == "eg" else self.model.sequence_state(feats)
        value = pred.detach().float().cpu().numpy()[0].copy()
        if value.shape != (51,) or not np.isfinite(value).all():
            raise FloatingPointError(f"nonfinite endpoint retained as failure: {ctx.seq}:{end}")
        return value


def load(name, route=None, tau=None, device=None, spec=None):
    import copy
    import torch
    from config import load_config
    from semkine.cmp_models import model_class
    require(spec is not None and spec.get("kind") == "cmp", "CMP networks are loaded from a CMP spec")
    require(route in (None, "router") and tau is None, "CMP networks have no routes")
    device = device or torch.device("cuda")
    cfg = copy.deepcopy(load_config(str(spec["config"])))
    kind = cfg["CMP"]["MODEL"]
    require(kind == spec["model"], "spec / config model kind differ")
    model = model_class(kind).load_from_checkpoint(str(spec["checkpoint"]), cfg=cfg, map_location=device, strict=True)
    identity = {"net": name, "kind": "cmp", "model": kind, "checkpoint": str(spec["checkpoint"]),
                "config": str(spec["config"]), "form": spec.get("form", "main"),
                "adapter_sha256": sha(Path(__file__)), "models_sha256": sha(Path(__file__).parents[2] / "semkine/cmp_models.py")}
    if kind == "eho":
        identity["eho_sha256"] = sha(Path(__file__).parents[2] / "semkine/cmp_eho.py")
    return CmpNet(name, model.to(device).eval(), cfg, device, identity, spec.get("form", "main"))


__all__ = ["CmpNet", "FORMS", "KF", "kf_update", "load", "window_raw"]
