#!/usr/bin/env python3
"""Root-tracking round: the debug gate a candidate passes before its screening run.

    python tools/rt/debug_arm.py --config configs/rt/rt_g3_2k.yaml [--ckpt CKPT] [--batch 64] [--steps 40]

Each check prints PASS / FAIL with the number behind it; the exit status is 1 if any check fails.
  1 forward/backward  a train-mode bf16 step on real training packets: finite loss, and every trainable
                      parameter receives a finite gradient (none silently unused); grad norm per module
  2 learns            `--steps` Adam steps on one fixed batch bring its loss down (overfit check)
  3 eval              eval-mode forward twice on the same packets is bitwise identical; train/eval gap
  4 state contract    tracking arms: an event-free packet returns exactly `prev`; the output follows a
                      10 deg root rotation of `prev` (state actually read, in the camera frame)
  5 geometry          routed arms: share of graph nodes routed to the hand under the GT state against the
                      state shifted 15 cm sideways (the projection / camera frame the readout relies on)
  6 closed loop       the protocol loop on zgz (both sequences): finite; RA and root error printed. Without
                      `--ckpt` this runs the untrained weights (a plumbing check); with it, the real one
  7 runtime           batch-1 latency per packet on real packets (evalx.latency_model), raw and anchored
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "x1001")]
from config import load_config                                   # noqa: E402
from model import MNISTModel                                     # noqa: E402
from mano_layer import ManoLayer                                 # noqa: E402
from semkine import eval_track as ET                             # noqa: E402
from semkine.dataset import build_dataset, sequences_for_split, splits_manifest  # noqa: E402
from semkine.events import EventPacket, collate_packets          # noqa: E402
import evalx as EX                                               # noqa: E402

FAILS = []


def report(name, ok, msg):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}", flush=True)
    if not ok:
        FAILS.append(name)


def to_dev(batch, dev):
    if hasattr(batch, "events"):
        return batch.to(dev)
    return tuple(t.to(dev) if torch.is_tensor(t) else t for t in batch)


def collate(items):
    if isinstance(items[0], EventPacket):
        return collate_packets(items)
    return torch.utils.data.default_collate(items)


def step_loss(model, batch):
    """The training objective: `MNISTModel.training_step`'s loss before the log10."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred, y, betas, _ = model._predict_batch(batch)
        loss, parts = model._compute_loss(pred, y, betas)
    loss = loss.float()
    return (loss.log10() if model.log10_loss else loss), pred


def rot(aa_deg_axis):
    return aa_deg_axis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--no-loop", action="store_true", help="skip the closed loop and latency (6, 7)")
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device("cuda")
    cfg = load_config(a.config)
    model = (MNISTModel.load_from_checkpoint(a.ckpt, cfg=cfg, map_location="cpu") if a.ckpt
             else MNISTModel(cfg)).to(dev)
    # the weights under test, before any train-mode pass touches the BN running statistics
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{a.config}: {n_par / 1e6:.3f} M parameters; predict_delta={model.predict_delta} "
          f"routed={getattr(model, 'routed', False)} encoder={model.encoder_name or 'resnet18/LNES'}", flush=True)

    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    raw = ds.input_mode != "legacy_lnes"
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True,
                    collate_fn=collate if raw else None, generator=torch.Generator().manual_seed(0))
    it = iter(dl)
    batches = [to_dev(next(it), dev) for _ in range(2)]

    # 1 forward / backward
    model.train()
    model.zero_grad(set_to_none=True)
    loss, pred = step_loss(model, batches[0])
    loss.backward()
    unused, bad, norms = [], [], {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            unused.append(name)
            continue
        if not torch.isfinite(p.grad).all():
            bad.append(name)
        key = ".".join(name.split(".")[:2]) if name.startswith("event_encoder") else name.split(".")[0]
        norms[key] = norms.get(key, 0.0) + float(p.grad.float().pow(2).sum())
    report("forward/backward", math.isfinite(float(loss)) and not bad and not unused and torch.isfinite(pred).all(),
           f"loss={float(loss):.4f} unused={unused[:5]} nonfinite={bad[:5]}")
    print("    grad norm per module: " + ", ".join(f"{k}={math.sqrt(v):.3g}" for k, v in sorted(norms.items())))

    # 2 learns (overfit one batch). lr 1e-4: training reaches 4e-3 only after a 500-step warmup, and a
    # fresh S37 at 1e-3 from step 0 rises before it falls. The weights are restored afterwards.
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    hist = []
    for _ in range(a.steps):
        opt.zero_grad(set_to_none=True)
        loss, _ = step_loss(model, batches[0])
        loss.backward()
        opt.step()
        hist.append(float(loss))
    report("learns", all(map(math.isfinite, hist)) and hist[-1] < hist[0] - 0.1,
           f"log10 loss {hist[0]:.3f} -> {hist[-1]:.3f} over {a.steps} steps")
    model.load_state_dict(init_state)

    # 3 eval determinism and train / eval gap
    with torch.no_grad():
        model.eval()
        o1 = model._predict_batch(batches[1])[0].float()
        o2 = model._predict_batch(batches[1])[0].float()
        model.train()
        ot = model._predict_batch(batches[1])[0].float()
        model.eval()
    # a train-mode forward moves the BN running statistics even under no_grad: put them back, so the
    # closed loop below runs exactly the weights under test (and reproduces evalx for a --ckpt)
    model.load_state_dict(init_state)
    with torch.no_grad():
        o3 = model._predict_batch(batches[1])[0].float()
    report("eval determinism", torch.equal(o1, o2) and torch.equal(o1, o3),
           f"max |diff| {float((o1 - o2).abs().max()):.3g}; restored after a train-mode pass: "
           f"{float((o1 - o3).abs().max()):.3g}; train/eval gap max {float((o1 - ot).abs().max()):.3g} (BN statistics)")

    # 4 state contract, 5 geometry: on real zgz packets with the GT state
    root = Path(cfg["DATA"]["ROOT"])
    seqs = dict((s, d) for s, d in sequences_for_split(root, "val_core", splits_manifest(cfg)))
    events, offsets, aux, pos51 = ET.load_sequence(root, seqs["zgz_global"], "zgz_global")
    tsub_p = root / seqs["zgz_global"] / "zgz_global_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, K)
    a0, b0 = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)[0]
    ends = np.arange(a0 + 1049, min(b0, a0 + 1049 + 50 * 60), 50)

    def fwd(end, prev, empty=False):
        if raw:
            ev5 = ET._window_events(events, offsets, tsub, int(end), 50)
            return model.forward_packet(ET.make_eval_packet(ev5[:0] if empty else ev5, prev, betas, K, 50, dev))
        x = torch.from_numpy(ET.build_lnes(events, offsets, int(end), 50, ("last",))).unsqueeze(0).to(dev)
        return model(torch.zeros_like(x) if empty else x, prev)

    with torch.no_grad():
        if model.predict_delta:
            same, moved = [], []
            for end in ends[:20]:
                prev = torch.from_numpy(pos51[end - 50].copy()).view(1, -1).to(dev)
                same.append(bool(torch.equal(fwd(end, prev, empty=True), prev)))
                p2 = prev.clone().cpu().numpy()[0]
                p2[3:6] = EX.aa_compose(np.array([0.0, 0.0, np.deg2rad(10.0)]), p2[3:6])
                o_a = fwd(end, prev).cpu().numpy()
                o_b = fwd(end, torch.from_numpy(p2).view(1, -1).to(dev)).cpu().numpy()
                moved.append(float(EX.rot_err_deg(o_a, o_b)[0]))
            report("state contract", all(same) and min(moved) > 1.0,
                   f"empty packet == prev on {sum(same)}/{len(same)}; output root moves "
                   f"{np.mean(moved):.2f} deg (min {min(moved):.2f}) for a 10 deg rotation of prev")
        else:
            print("    (absolute arm: no state contract)")
        if getattr(model, "routed", False):
            fr = {}
            for tag, dx in (("gt", 0.0), ("shift15cm", 0.15)):
                vals = []
                for end in ends:
                    p = pos51[end - 50].copy()
                    p[0] += dx
                    fwd(end, torch.from_numpy(p).view(1, -1).to(dev))
                    vals.append(float(model.route_stats.get("route_frac_routed", float("nan"))))
                fr[tag] = float(np.nanmean(vals))
            report("geometry", fr["gt"] > fr["shift15cm"] + 0.2 and fr["gt"] > 0.5,
                   f"nodes routed to the hand: GT state {fr['gt']:.3f}, state shifted 15 cm {fr['shift15cm']:.3f}")

    if a.no_loop:
        sys.exit(1 if FAILS else 0)
    # 6 closed loop on the protocol (both zgz sequences)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
    rng = np.random.default_rng(0)
    res = {}
    t0 = time.time()
    for s, d in seqs.items():
        r = EX.run_sequence(model, cfg, root, d, s, dev, rng)
        m = EX.per_step_metrics(mano, r, dev)
        res[s] = {k: float(np.mean(m[k])) for k in ("mpjpe_ra_mm", "root_rot_deg")}
        res[s]["finite"] = bool(np.isfinite(r["pred"]).all())
    report("closed loop", all(v["finite"] for v in res.values()),
           json.dumps(res) + f" ({time.time() - t0:.0f} s)")

    # 7 runtime
    lat = EX.latency_model(model, cfg, dev)
    anchor = EX.latency_anchor(dev)
    print(f"[INFO] runtime: {lat:.2f} ms/packet raw, anchor {anchor:.2f} ms -> {lat * 1.75 / anchor:.2f} ms "
          f"on the 1.75 ms full-model scale", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
