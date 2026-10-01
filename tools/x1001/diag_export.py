#!/usr/bin/env python3
"""x1001 Phase 1 (E1a / H2 and E4 / H5): one feature export, reused by every probe and sweep.

Frozen historical checkpoints (trained on the 9-subject split; used here for diagnostics only):
  S37 routed : s37_routed_s3407 / s3408, at the zgz-selected step ("sel") and at step 6000 ("last")
  CNN abs    : s37diag_cnnabs_s3407 / s3408, at the zgz-selected step
Sets (protocol packets: every 50 ms end a+49, a+99, ... inside each valid run):
  train  the 72 training sequences of the 9 training subjects (splits_semkine.json; zgz excluded)
  dev    zgz_global / zgz_local; E4 thinning keeps, one fixed event mask per (sequence, keep),
         shared by every model and representation
  occl   every 4th dev packet at keep 1 with one 5 ms slice of the 50 ms window removed (10 slices)
Per packet: GT 51D, event count, S37 state-free pooled vector pf = [feat | node mean | node max]
(768) per S37 model, CNN penultimate (512) and output (51) per CNN model, 2x2x30x40 histogram
(4800), S37 graph geometry (weight-independent), LNES active pixels.
Routed evidence r = [feat | e (16 x 257)] (the S37 root head input) for the "sel" checkpoints under
three routing states, stored separately so GT never enters a normal-inference conclusion:
  oracle  prev = GT of the previous protocol step                    (GT routing; reference only)
  gtinit  prev = own closed loop, each segment started at GT + protocol noise   (protocol)
  cold    prev = own closed loop, each segment started at the training mean pose (no GT at all)
Closed-loop outputs (gtinit at every keep, cold at keep 1) give the E4 RA curves.

    python tools/x1001/diag_export.py --stage all [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.dataset import _read_meta51, sequences_for_split         # noqa: E402
from semkine.encoder import event_tokens                              # noqa: E402
from semkine.event_gnn import T_COL                                   # noqa: E402
from semkine.routed_readout import pool_joint_evidence                # noqa: E402

MAIN = Path("/data1/lyq/code/mesh/EventHands1")
PROG = Path("/data1/lyq/code/mesh/EventHands1_x1001")
OUT = PROG / "diag" / "e1a_e4"
DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
DEV_MANIFEST = DATA / "splits_semkine.json"
STEP = 50
H, W = 180, 240
KEEPS = {"zgz_global": (1.0, 0.5, 0.2, 0.1, 0.05, 0.03), "zgz_local": (1.0, 0.5, 0.25, 0.13)}
N_BINS = 10                      # 5 ms occlusion slices of the 50 ms window
THIN_SEED = 20261001


# ------------------------------------------------------------------------------------ models
def selected_ckpt(run: Path) -> tuple:
    sel = json.loads(sorted(run.glob("selection_val_core_step50*.json"))[0].read_text())["selected"]
    return sel["ckpt"] if os.path.isabs(sel["ckpt"]) else str(MAIN / sel["ckpt"]), int(sel["step"])


def model_specs():
    s37_cfg = MAIN / "configs/semkine/s37_routed_s3407.yaml"
    cnn_cfg = MAIN / "configs/semkine/s1_abs_domrand.yaml"
    specs = {}
    for tag, seed in (("a", 3407), ("b", 3408)):
        run = MAIN / f"outputs/semkine/s37_routed_s{seed}"
        ck, st = selected_ckpt(run)
        specs[f"s37{tag}_sel"] = ("s37", s37_cfg, ck, st)
        specs[f"s37{tag}_last"] = ("s37", s37_cfg, str(run / f"s37_routed_s{seed}-step=6000.ckpt"), 6000)
        run = MAIN / f"outputs/semkine/s37diag_cnnabs_s{seed}"
        ck, st = selected_ckpt(run)
        specs[f"cnn{tag}_sel"] = ("cnn", cnn_cfg, ck, st)
    return specs


def load_models(device, names=None):
    out = {}
    for name, (kind, cfgp, ck, st) in model_specs().items():
        if names and name not in names:
            continue
        cfg = load_config(cfgp)
        m = MNISTModel.load_from_checkpoint(ck, cfg=cfg, map_location=device).to(device).eval()
        if kind == "cnn":
            store = {}
            m.rn.fc.register_forward_hook(lambda mod, inp, o, store=store: store.__setitem__("pen", inp[0].detach()))
            m._x1001_store = store
        out[name] = (kind, m, cfg, ck, st)
    return out


def channels_of(models):
    cfg = next((c for k, m, c, _, _ in models.values() if k == "cnn"), None)
    return EV.event_channels(cfg) if cfg is not None else ("last",)


# ------------------------------------------------------------------------------------ sequences
class SeqData:
    """One sequence's protocol packets; `keep < 1` thins the events with a fixed mask."""

    def __init__(self, d: str, seq: str, keep: float = 1.0):
        base = str(DATA / d / seq)
        self.name, self.keep = seq, float(keep)
        self.events = np.load(base + "_events.npy", mmap_mode="r")
        self.offsets = np.load(base + "_offsets.npy")
        tp = base + "_tsub.npy"
        self.tsub = np.load(tp, mmap_mode="r") if os.path.exists(tp) else None
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        self.betas = np.asarray(aux["betas"], np.float32).reshape(-1)
        self.K = np.asarray(aux["camera_K"], np.float32).reshape(3, 3)
        self.pos51 = _read_meta51(base + ".meta")
        self.runs = []
        for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
            if len(ends):
                self.runs.append((int(a), ends))
        if self.keep < 1.0:
            rs = np.random.default_rng([THIN_SEED, int(self.keep * 1000), sum(map(ord, seq))])
            mask = rs.random(len(self.events)) < self.keep
            idx = np.flatnonzero(mask)
            self.events = np.asarray(self.events[idx])
            self.tsub = None if self.tsub is None else np.asarray(self.tsub[idx])
            self.offsets = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])[self.offsets]

    def rows(self):
        """[(run_idx, a, prev_end, end)] in protocol order."""
        out = []
        for r, (a, ends) in enumerate(self.runs):
            p = a
            for e in ends:
                out.append((r, a, p, int(e)))
                p = int(e)
        return out

    def ev5(self, end: int, drop_bin: int = -1) -> np.ndarray:
        ev = ET._window_events(self.events, self.offsets, self.tsub, int(end), STEP)
        if drop_bin >= 0 and len(ev):
            ms = np.floor(ev[:, 3] * 1000.0 + 5e-4)
            keep = (ms // (STEP // N_BINS)) != drop_bin
            ev = ev[keep]
        return ev


def lnes_of(ev5: np.ndarray, channels) -> np.ndarray:
    if not len(ev5):
        return np.zeros((H, W, 2 * len(channels)), np.float32)
    ms = np.floor(ev5[:, 3] * 1000.0 + 5e-4).astype(np.float32)
    return EV.splat_event_image(ev5[:, 1].astype(np.uint8), ev5[:, 2].astype(np.uint8),
                                ev5[:, 4].astype(np.uint8), ms, STEP, H, W, channels)


def hist_of(ev: np.ndarray) -> np.ndarray:
    """2 channels (log1p count, latest normalised time) x 2 polarities x 30 x 40 bins of 6 px."""
    f = np.zeros((2, 2, 30, 40), np.float32)
    if len(ev):
        xb = np.clip((ev[:, EV.EV_X] // 6).astype(np.int64), 0, 39)
        yb = np.clip((ev[:, EV.EV_Y] // 6).astype(np.int64), 0, 29)
        pb = (ev[:, EV.EV_P] > 0).astype(np.int64)
        np.add.at(f[0], (pb, yb, xb), 1.0)
        t = ev[:, EV.EV_T]
        tn = (t - t.min()) / max(float(t.max() - t.min()), 1e-9)
        np.maximum.at(f[1], (pb, yb, xb), tn.astype(np.float32))
    f[0] = np.log1p(f[0])
    return f.reshape(-1)


class PacketSet(torch.utils.data.Dataset):
    """CPU side of one packet list: raw events, LNES, histogram."""

    def __init__(self, seqs, items, channels):
        self.seqs, self.items, self.channels = seqs, items, channels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, end, drop = self.items[i]
        ev = self.seqs[si].ev5(end, drop)
        return {"i": i, "ev": ev, "lnes": lnes_of(ev, self.channels), "hist": hist_of(ev)}


def collate(items):
    return items


def ident(x):
    return x


def loader(ds, workers, bs=256):
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                                       collate_fn=collate, prefetch_factor=4 if workers else None,
                                       persistent_workers=False)


# ------------------------------------------------------------------------------------ GPU parts
def make_batch(ev_list, prev, betas, K, device):
    B = len(ev_list)
    evs, ptr = [], [0]
    for j, ev in enumerate(ev_list):
        ev = ev.copy()
        ev[:, 0] = j
        evs.append(ev)
        ptr.append(ptr[-1] + len(ev))
    allev = np.concatenate(evs) if evs else np.zeros((0, 5), np.float32)
    batch = ET.make_eval_packet(allev, prev, betas, K, STEP, device)
    batch.ptr = torch.tensor(ptr, dtype=torch.int64, device=device)
    for f in ("sequence_id", "t_start_us", "t_end_us", "delta_t_s", "is_sequence_start", "is_sequence_end"):
        setattr(batch, f, getattr(batch, f).expand(B).contiguous())
    return batch


@torch.no_grad()
def s37_parts(m, batch, route: bool = True):
    """(out51, pf, r) with out51 == forward_packet (checked by `selfcheck`); r is None if not route."""
    feat, h, _, px, py, mask = m.event_encoder(batch.events, batch.ptr, batch.delta_t_s, None, return_nodes=True)
    if h.shape[1] == 0:
        mean = peak = torch.zeros(feat.shape[0], h.shape[-1], device=feat.device, dtype=feat.dtype)
    else:
        live = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
        mean = (h * mask.unsqueeze(-1).to(h.dtype)).sum(1) / live
        peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values * mask.any(1, keepdim=True).to(h.dtype)
    pf = torch.cat([feat, mean, peak], 1).float()
    if not route:
        return None, pf, None
    prev = batch.prev_state
    if h.shape[1] == 0:
        ev = torch.zeros(feat.shape[0], 16, 2 * h.shape[-1] + 1, device=feat.device, dtype=feat.dtype)
    else:
        betas_f, k_f = m._resolve_betas_K(prev, batch.betas, batch.camera_K)
        a, _, _ = m._route_nodes(px, py, mask, prev, betas_f, k_f)
        ev, _ = pool_joint_evidence(h, a, mask)
    out = m._decode_active(feat, prev, ev)
    out = out + m.prev_mlp(prev.to(out.dtype))
    delta = torch.where((batch.counts <= 0).unsqueeze(-1), torch.zeros_like(out), out)
    out = (delta + prev.to(out.dtype)).float()
    return out, pf, torch.cat([feat.float(), ev.flatten(1).float()], 1)


@torch.no_grad()
def cnn_parts(m, lnes):
    out = m(lnes, torch.zeros(lnes.shape[0], 51, device=lnes.device)).float()
    return out, m._x1001_store["pen"].float()


@torch.no_grad()
def graph_stats(enc, batch):
    """S37 graph geometry (weight-independent), per packet: events, nodes, retention, median 1-hop
    and 3-hop temporal span of a node's ancestors (ms), median half-diagonal of the 3-hop ancestor
    box (px), median edge length (px), share of the events' 4 px cells that hold a node."""
    events, ptr = batch.events, batch.ptr
    B = int(ptr.numel() - 1)
    dev = events.device
    counts = (ptr[1:] - ptr[:-1]).float()
    out = torch.zeros(B, 8, device=dev)
    out[:, 0] = counts
    if events.shape[0] == 0:
        return out
    tok = event_tokens(events, ptr, batch.delta_t_s, enc.height, enc.width)
    src, mask = enc._sample(events, ptr)
    N = mask.shape[1]
    flat = src.reshape(-1)
    ev = events[flat].reshape(B, N, -1)
    tn = tok[flat, T_COL].reshape(B, N)
    p = torch.stack([ev[..., 1] / enc.width, ev[..., 2] / enc.height, tn * enc.t_scale], -1) * mask.unsqueeze(-1)
    idx, dp, emask = enc._edges(p.float(), mask)
    t = ev[..., 3].float() * 1000.0
    x, y = ev[..., 1].float(), ev[..., 2].float()
    ok = (emask > 0) & mask.unsqueeze(-1)

    def nb(v, fill):
        return v.gather(1, idx.reshape(B, -1)).reshape(idx.shape).masked_fill(~ok, fill)
    big = 1e9
    tmin, xmin, xmax, ymin, ymax = t, x, x, y, y
    span1 = None
    for hop in range(3):
        tmin, xmin, ymin, xmax, ymax = (torch.minimum(t, nb(tmin, big).min(-1).values),
                                        torch.minimum(x, nb(xmin, big).min(-1).values),
                                        torch.minimum(y, nb(ymin, big).min(-1).values),
                                        torch.maximum(x, nb(xmax, -big).max(-1).values),
                                        torch.maximum(y, nb(ymax, -big).max(-1).values))
        if hop == 0:
            span1 = t - tmin
    nan = float("nan")

    def med(v):
        return v.masked_fill(~mask, nan).nanmedian(1).values
    rad3 = 0.5 * torch.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2)
    elen = torch.sqrt((dp[..., 0].float() * enc.width) ** 2 + (dp[..., 1].float() * enc.height) ** 2)
    n = mask.sum(1).float()
    out[:, 1] = n
    out[:, 2] = n / counts.clamp_min(1)
    out[:, 3] = med(span1)
    out[:, 4] = med(t - tmin)
    out[:, 5] = med(rad3)
    out[:, 6] = elen.masked_fill(~ok, nan).reshape(B, -1).nanmedian(1).values
    bi = torch.repeat_interleave(torch.arange(B, device=dev), (ptr[1:] - ptr[:-1]))
    key = bi * 4096 + (events[:, 2].long() // 4) * 64 + events[:, 1].long() // 4
    cells_all = torch.bincount(torch.unique(key) // 4096, minlength=B).float()
    bn = torch.arange(B, device=dev).unsqueeze(1).expand(B, N)[mask]
    keyn = bn * 4096 + (ev[..., 2][mask].long() // 4) * 64 + ev[..., 1][mask].long() // 4
    cells_node = torch.bincount(torch.unique(keyn) // 4096, minlength=B).float()
    out[:, 7] = cells_node / cells_all.clamp_min(1)
    return torch.nan_to_num(out, nan=0.0)


GSTAT_NAMES = ("events", "nodes", "retention", "span1_ms", "span3_ms", "rad3_px", "edge_px", "cell_cov")


# ------------------------------------------------------------------------------------ passes
class Writer:
    def __init__(self, d: Path):
        self.d = d
        d.mkdir(parents=True, exist_ok=True)
        self.buf = {}

    def add(self, key, arr):
        self.buf.setdefault(key, []).append(np.asarray(arr))

    def close(self, meta: dict):
        for k, v in self.buf.items():
            a = np.concatenate(v)
            if a.dtype == np.float32 and (k.startswith("pf_") or k.startswith("pen_") or k.startswith("r_")
                                          or k == "hist"):
                a = a.astype(np.float16)
            np.save(self.d / f"{k}.npy", a)
        (self.d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))


@torch.no_grad()
def feature_pass(name, seqs, items, rows_meta, models, device, workers, oracle_route=True):
    """pf / pen / out / hist / graph stats / oracle routed evidence for a packet list."""
    d = OUT / name
    if (d / "meta.json").exists():
        print(f"[skip] {name}", flush=True)
        return
    t0 = time.time()
    chans = channels_of(models)
    wr = Writer(d)
    enc = next(m for k, m, *_ in models.values() if k == "s37").event_encoder
    ds = PacketSet(seqs, items, chans)
    for ci, chunk in enumerate(loader(ds, workers)):
        if ci % 50 == 0:
            print(f"   {name}: chunk {ci} / {len(ds) // 256 + 1}  {time.time() - t0:.0f} s", flush=True)
        idx = [c["i"] for c in chunk]
        evs = [c["ev"] for c in chunk]
        si = [items[i][0] for i in idx]
        ends = [items[i][1] for i in idx]
        betas = torch.from_numpy(np.stack([seqs[s].betas for s in si])).to(device)
        K = torch.from_numpy(np.stack([seqs[s].K for s in si])).to(device)
        prev_gt = torch.from_numpy(np.stack([seqs[s].pos51[rows_meta[i]["prev_end"]] for s, i in zip(si, idx)])).to(device)
        batch = make_batch(evs, prev_gt, betas, K, device)
        wr.add("gstat", graph_stats(enc, batch).cpu().numpy())
        lnes = torch.from_numpy(np.stack([c["lnes"] for c in chunk])).to(device)
        wr.add("lnes_active", (lnes.abs().sum(-1) > 0).flatten(1).sum(1).cpu().numpy())
        wr.add("hist", np.stack([c["hist"] for c in chunk]))
        for mn, (kind, m, cfg, _, _) in models.items():
            if kind == "s37":
                route = oracle_route and mn.endswith("_sel")
                out, pf, r = s37_parts(m, batch, route=route)
                wr.add(f"pf_{mn}", pf.cpu().numpy())
                if route:
                    wr.add(f"r_oracle_{mn}", r.cpu().numpy())
                    wr.add(f"tf_out_{mn}", out.cpu().numpy())
            else:
                out, pen = cnn_parts(m, lnes)
                wr.add(f"pen_{mn}", pen.cpu().numpy())
                wr.add(f"out_{mn}", out.cpu().numpy())
        wr.add("gt", np.stack([seqs[s].pos51[e] for s, e in zip(si, ends)]))
        wr.add("count", np.array([len(e) for e in evs]))
    wr.add("seq", np.array([items[i][0] for i in range(len(items))]))
    wr.add("end", np.array([items[i][1] for i in range(len(items))]))
    wr.add("drop", np.array([items[i][2] for i in range(len(items))]))
    wr.add("run", np.array([rows_meta[i]["run"] for i in range(len(items))]))
    wr.add("elapsed", np.array([items[i][1] - rows_meta[i]["a"] for i in range(len(items))]))
    wr.close({"set": name, "seqs": [s.name for s in seqs], "keeps": [s.keep for s in seqs], "n": len(items),
              "models": {k: {"kind": v[0], "ckpt": v[3], "step": v[4]} for k, v in models.items()},
              "gstat_names": GSTAT_NAMES, "seconds": round(time.time() - t0, 1)})
    print(f"[done] {name}: {len(items)} packets in {time.time() - t0:.0f} s", flush=True)


class StepSet(torch.utils.data.Dataset):
    """Item t = the raw packets of every segment still active at step t (closed-loop order kept)."""

    def __init__(self, seqs, segs):
        self.seqs, self.segs = seqs, segs
        self.T = max(len(s["ends"]) for s in segs)

    def __len__(self):
        return self.T

    def __getitem__(self, t):
        act = [j for j, s in enumerate(self.segs) if len(s["ends"]) > t]
        return t, act, [self.seqs[self.segs[j]["si"]].ev5(int(self.segs[j]["ends"][t])) for j in act]


@torch.no_grad()
def closed_loop(m, seqs, segs, inits, device, workers, capture=False):
    """Batched closed loop over independent segments; each segment advances strictly in time
    order from its own initial state. Returns per-segment outputs (and routed evidence)."""
    S = len(segs)
    prev = torch.from_numpy(np.stack(inits).astype(np.float32)).to(device)
    betas = torch.from_numpy(np.stack([seqs[s["si"]].betas for s in segs])).to(device)
    K = torch.from_numpy(np.stack([seqs[s["si"]].K for s in segs])).to(device)
    outs = [[] for _ in range(S)]
    caps = [[] for _ in range(S)]
    dl = torch.utils.data.DataLoader(StepSet(seqs, segs), batch_size=None, shuffle=False, collate_fn=ident,
                                     num_workers=workers, prefetch_factor=4 if workers else None)
    for t, act, evs in dl:
        act_t = torch.tensor(act, device=device)
        batch = make_batch(evs, prev[act_t], betas[act_t], K[act_t], device)
        out, _, r = s37_parts(m, batch, route=True)
        prev[act_t] = out
        o = out.cpu().numpy()
        rr = r.cpu().numpy().astype(np.float16) if capture else None
        for k, j in enumerate(act):
            outs[j].append(o[k])
            if capture:
                caps[j].append(rr[k])
    return outs, caps


def protocol_segments(seqs):
    segs = []
    for si, s in enumerate(seqs):
        for r, (a, ends) in enumerate(s.runs):
            segs.append({"si": si, "run": r, "a": a, "ends": ends})
    return segs


def gt_inits(cfg, seqs, segs):
    rng = np.random.default_rng(0)                       # the protocol's draw order: sequences, runs
    return [seqs[s["si"]].pos51[s["a"]] + ET.sample_init_noise(cfg, rng, 1.0) for s in segs]


@torch.no_grad()
def loop_pass(name, seqs, models, device, workers, mode, mean_pose=None, capture_sel=True, only_sel=False):
    d = OUT / name
    if (d / "meta.json").exists():
        print(f"[skip] {name}", flush=True)
        return
    t0 = time.time()
    wr = Writer(d)
    segs = protocol_segments(seqs)
    for mn, (kind, m, cfg, _, _) in models.items():
        if kind != "s37" or (only_sel and not mn.endswith("_sel")):
            continue
        inits = gt_inits(cfg, seqs, segs) if mode == "gtinit" else [mean_pose.copy() for _ in segs]
        cap = capture_sel and mn.endswith("_sel")
        outs, caps = closed_loop(m, seqs, segs, inits, device, workers, capture=cap)
        wr.add(f"loop_out_{mn}", np.concatenate([np.stack(o) for o in outs]))
        if cap:
            wr.add(f"r_{mode}_{mn}", np.concatenate([np.stack(c) for c in caps]).astype(np.float16))
        print(f"   {name} {mn}: {time.time() - t0:.0f} s", flush=True)
    wr.add("seq", np.concatenate([[s["si"]] * len(s["ends"]) for s in segs]))
    wr.add("end", np.concatenate([s["ends"] for s in segs]))
    wr.add("run", np.concatenate([[s["run"]] * len(s["ends"]) for s in segs]))
    wr.add("gt", np.concatenate([seqs[s["si"]].pos51[s["ends"]] for s in segs]))
    wr.close({"set": name, "mode": mode, "seqs": [s.name for s in seqs], "keeps": [s.keep for s in seqs],
              "n_segments": len(segs), "seconds": round(time.time() - t0, 1)})
    print(f"[done] {name} in {time.time() - t0:.0f} s", flush=True)


def items_of(seqs, drop_every=None):
    items, meta = [], []
    for si, s in enumerate(seqs):
        rows = s.rows()
        if drop_every:
            rows = rows[::drop_every]
        for (r, a, p, e) in rows:
            bins = range(N_BINS) if drop_every else (-1,)
            for b in bins:
                items.append((si, e, b))
                meta.append({"run": r, "a": a, "prev_end": p})
    return items, meta


@torch.no_grad()
def selfcheck(models, device):
    """forward_packet == s37_parts; lnes_of == ET.build_lnes; batched loop RA vs main rows."""
    s = SeqData("val", "zgz_local")
    rows = s.rows()[:48]
    evs = [s.ev5(e) for _, _, _, e in rows]
    prev = torch.from_numpy(np.stack([s.pos51[p] for _, _, p, _ in rows])).to(device)
    b = torch.from_numpy(np.stack([s.betas] * len(rows))).to(device)
    K = torch.from_numpy(np.stack([s.K] * len(rows))).to(device)
    rep = {}
    for mn, (kind, m, cfg, _, _) in models.items():
        if kind != "s37":
            continue
        batch = make_batch(evs, prev, b, K, device)
        ref = m.forward_packet(batch).float()
        out, _, _ = s37_parts(m, batch)
        rep[f"forward_packet_maxdiff_{mn}"] = float((ref - out).abs().max())
    chans = channels_of(models)
    md = 0.0
    for _, _, _, e in rows:
        a = ET.build_lnes(s.events, s.offsets, int(e), STEP, chans)
        md = max(md, float(np.abs(a - lnes_of(s.ev5(e), chans)).max()))
    rep["lnes_maxdiff"] = md
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=("selfcheck", "train", "trainloop", "dev", "devloop", "occl", "all"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--smoke", action="store_true", help="2 training sequences, 200 packets each")
    ap.add_argument("--models", default="", help="comma list (default all)")
    ap.add_argument("--out", default="", help="output sub-directory of diag/ (default e1a_e4)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = OUT.parent / a.out
    if a.smoke:
        OUT = OUT.parent / (OUT.name + "_smoke")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    names = [x for x in a.models.split(",") if x] or None
    models = load_models(device, names)
    print("models:", {k: (v[0], Path(v[3]).name) for k, v in models.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage in ("selfcheck", "all"):
        rep = selfcheck(models, device)
        print("selfcheck:", rep, flush=True)
        (OUT / "selfcheck.json").write_text(json.dumps(rep, indent=1))
        assert all(v < 1e-4 for v in rep.values()), rep
    train = [SeqData(d, s) for s, d in sequences_for_split(DATA, "train", DEV_MANIFEST)]
    assert len(train) == 64 and not any(s.name.startswith(("ly_", "zgz_")) for s in train)
    if a.smoke:
        train = train[:1] + train[-1:]
        for s in train:
            s.runs = [(s.runs[0][0], s.runs[0][1][:200])]
    if a.stage in ("train", "all"):
        items, meta = items_of(train)
        feature_pass("train", train, items, meta, models, device, a.workers)
    gt_train = np.load(OUT / "train" / "gt.npy")
    mean_pose = gt_train.mean(0).astype(np.float32)
    np.save(OUT / "train_mean_pose.npy", mean_pose)
    if a.stage in ("trainloop", "all"):
        if any(n.endswith("_sel") and v[0] == "s37" for n, v in models.items()):
            for mode in ("gtinit", "cold"):
                loop_pass(f"train_loop_{mode}", train, models, device, a.workers, mode, mean_pose, only_sel=True)
    dev_names = ["zgz_global", "zgz_local"]
    if a.stage in ("dev", "all"):
        for seq in dev_names:
            for keep in KEEPS[seq]:
                s = SeqData("val", seq, keep)
                if a.smoke:
                    s.runs = s.runs[:1]
                    s.runs = [(s.runs[0][0], s.runs[0][1][:120])]
                items, meta = items_of([s])
                feature_pass(f"dev_{seq}_k{keep:g}", [s], items, meta, models, device, a.workers,
                             oracle_route=(keep == 1.0))
    if a.stage in ("devloop", "all"):
        def dev_seqs(keep):
            out = []
            for seq in dev_names:
                if keep in KEEPS[seq]:
                    s = SeqData("val", seq, keep)
                    if a.smoke:
                        s.runs = [(s.runs[0][0], s.runs[0][1][:120])]
                    out.append(s)
            return out
        for keep in sorted({k for ks in KEEPS.values() for k in ks}, reverse=True):
            loop_pass(f"devloop_k{keep:g}_gtinit", dev_seqs(keep), models, device, a.workers, "gtinit",
                      capture_sel=(keep == 1.0))
        loop_pass("devloop_k1_cold", dev_seqs(1.0), models, device, a.workers, "cold", mean_pose)
    if a.stage in ("occl", "all"):
        for seq in dev_names:
            s = SeqData("val", seq, 1.0)
            if a.smoke:
                s.runs = [(s.runs[0][0], s.runs[0][1][:40])]
            items, meta = items_of([s], drop_every=4)
            feature_pass(f"occl_{seq}", [s], items, meta, models, device, a.workers, oracle_route=False)
    print("all stages done", flush=True)


if __name__ == "__main__":
    main()
