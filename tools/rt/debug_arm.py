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
  7 runtime           batch-1 latency per packet on real packets (evalx.latency_model), raw and anchored;
                      with `--budget CONFIG` also the budget arm's latency and both arms' MACs in the same process
S38 additions (docs/S38_ROOT_TRACKING_VERDICT.md):
  8 rotation          `ROOT_MEAS: abs`: ROOT_REF is the training split's mean root; a zero head output measures
                      R_ref exactly; the training (GPU, float32) and inference (host, float64) compositions agree;
                      the root loss is zero at the target
  9 micro-overfit     `--overfit-steps` Adam steps (peak `--overfit-lr`, 10 % warmup, cosine to 0) on one fixed batch of
                      32 training packets:
                      log10 loss falls by >= 1 and the batch's root error ends < 3 deg (S37 itself: 0.18 deg)
 10 correction        (needs `--ckpt`) the closed loop on a *training* sequence (lyq_global, never zgz) and
                      10 / 20 deg root perturbations of the fed-back state, 20-step branches: the retained share
                      must decay; for a filtered absolute root it must follow (1 - gain)^k, the filter's law
 11 health           (needs `--ckpt`) on 256 training packets: share of live units after the encoder's readout
                      projections (reported; the sparse pyramid's node projection must keep >= 25 %), and for an
                      absolute root the measurement must beat the constant R_ref (error <= 0.8 x) -- the 500-step
                      gate caught a projection switched off by the translation gradient, leaving a constant
                      measurement
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
    ap.add_argument("--no-loop", action="store_true", help="skip the closed loop and latency (6, 7, 10)")
    ap.add_argument("--overfit-steps", type=int, default=1500)
    ap.add_argument("--overfit-lr", type=float, default=1e-3)
    ap.add_argument("--only-overfit", action="store_true", help="stop after the micro-overfit (8, 9)")
    ap.add_argument("--budget", default=None, help="config of the arm whose runtime is the budget (S38: rt_s37)")
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
            ok = all(same) and min(moved) > 1.0
            if getattr(model, "root_meas", "delta") == "abs":
                # a state-free measurement filtered with gain g: the output root keeps (1 - g) of prev's rotation
                want = 10.0 * (1.0 - model.root_filter_gain)
                ok = all(same) and abs(float(np.mean(moved)) - want) < 1.0
                if want == 0.0:
                    print("    (gain 1: the root reads no state)")
            report("state contract", ok,
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

    # 8 rotation convention of the absolute root measurement
    if getattr(model, "root_meas", "delta") == "abs":
        sys.path.insert(0, str(REPO / "tools" / "s38"))
        import make_configs as MC
        ref_train = np.asarray(MC.training_root_ref())
        ref_cfg = np.asarray(cfg["MODEL"]["ROOT_REF"], dtype=np.float64)
        with torch.no_grad():
            z = torch.zeros(4, 3, device=dev)
            prev4 = torch.zeros(4, 51, device=dev)
            cnt = torch.full((4,), 10, device=dev)
            g0 = model.root_filter_gain
            model.root_filter_gain = 1.0
            model.eval()
            at_ref_eval = model._abs_root(z, prev4, cnt, 1).cpu().numpy()
            model.train()
            at_ref_train = model._abs_root(z, prev4, cnt, 1).cpu().numpy()
            r = torch.randn(64, 3, device=dev) * 0.4                          # deviations up to ~70 deg
            pr = torch.zeros(64, 51, device=dev)
            c64 = torch.full((64,), 10, device=dev)
            tr = model._abs_root(r, pr, c64, 1).cpu().numpy()
            model.eval()
            ev = model._abs_root(r, pr, c64, 1).cpu().numpy()
            model.root_filter_gain = g0
            y = batches[1].target.float()
            loss_at_target, parts = model._compute_loss(y.clone(), y, batches[1].betas)
        model.load_state_dict(init_state)
        pad = lambda v: np.concatenate([np.zeros((len(v), 3), np.float32), v, np.zeros((len(v), 45), np.float32)], 1)  # noqa: E731
        d_ref = float(np.abs(ref_train - ref_cfg).max())
        d_zero = float(max(np.abs(at_ref_eval - ref_cfg).max(), np.abs(at_ref_train - ref_cfg).max()))
        d_path = float(EX.rot_err_deg(pad(tr), pad(ev)).max())
        report("rotation convention", d_ref < 1e-5 and d_zero < 1e-5 and d_path < 1e-3
               and float(parts["rot_loss"]) == 0.0,
               f"ROOT_REF vs training-split mean {d_ref:.2e}; zero head -> R_ref within {d_zero:.2e}; "
               f"train vs inference composition max {d_path:.2e} deg; root loss at target "
               f"{float(parts['rot_loss']):.1e} ({model.root_loss})")

    # 9 micro-overfit: one fixed batch, a few hundred steps
    small = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, drop_last=True,
                       collate_fn=collate if raw else None, generator=torch.Generator().manual_seed(1))
    fixed = to_dev(next(iter(small)), dev)
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.overfit_lr)
    warm = max(1, a.overfit_steps // 10)
    # the training schedule's shape (linear warmup, cosine to zero): at a constant rate the unnormalised
    # event graph oscillates on a fixed batch under the log10 objective and S37 itself would fail
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (
        1.0 + math.cos(math.pi * min(1.0, max(0.0, (i - warm) / max(1, a.overfit_steps - warm))))))
    hist = []
    for _ in range(a.overfit_steps):
        opt.zero_grad(set_to_none=True)
        loss, pred = step_loss(model, fixed)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
    with torch.no_grad():
        _, pred = step_loss(model, fixed)
    tgt = fixed.target if hasattr(fixed, "target") else fixed[2]
    rot_fit = float(np.mean(EX.rot_err_deg(pred.float().cpu().numpy(), tgt.float().cpu().numpy())))
    report("micro-overfit", all(map(math.isfinite, hist)) and hist[-1] < hist[0] - 1.0 and rot_fit < 3.0,
           f"log10 loss {hist[0]:.3f} -> {hist[-1]:.3f} in {a.overfit_steps} steps at lr {a.overfit_lr:g}; "
           f"root error on the batch {rot_fit:.2f} deg; loss at 25/50/75 %: "
           + ", ".join(f"{hist[int(len(hist) * q) - 1]:.2f}" for q in (0.25, 0.5, 0.75)))
    model.load_state_dict(init_state)
    model.eval()

    if a.no_loop or a.only_overfit:
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

    # 10 closed-loop correction on a training sequence (never the evaluation subject)
    if a.ckpt:
        tr_seq, tr_dir = "lyq_global", "train"
        rng = np.random.default_rng(0)
        r = EX.run_sequence(model, cfg, root, tr_dir, tr_seq, dev, rng)
        m = EX.per_step_metrics(mano, r, dev)
        pert = EX.perturb_trials(model, cfg, root, tr_dir, tr_seq, dev, r)
        ret = {th: [float(v.mean(0)[k] / th) for k in (0, 1, 4, 9)] for th, dd in pert.items() for v in [dd["div"]]}
        ok = np.isfinite(r["pred"]).all() and all(v[3] < 0.5 for v in ret.values())
        law = ""
        if getattr(model, "root_meas", "delta") == "abs":
            g = model.root_filter_gain
            want = [(1.0 - g) ** k for k in (1, 2, 5, 10)]
            ok = ok and all(abs(v[0] - want[0]) < 0.05 and v[2] < want[2] + 0.05 for v in ret.values())
            law = f"; filter law (1 - {g:g})^k = {[round(w, 3) for w in want]}"
        report("correction", ok,
               f"{tr_seq} closed loop RA {float(m['mpjpe_ra_mm'].mean()):.2f} mm, root {float(m['root_rot_deg'].mean()):.2f} deg; "
               f"retained share at k = 1, 2, 5, 10: "
               + ", ".join(f"{th:g} deg {[round(x, 3) for x in v]}" for th, v in ret.items()) + law)

    # 11 feature health
    if a.ckpt and raw:
        hb = to_dev(next(iter(DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, collate_fn=collate,
                                         generator=torch.Generator().manual_seed(2)))), dev)
        enc = model.event_encoder
        acts, hooks = {}, []
        for name in ("proj", "node_proj"):
            mod = getattr(enc, name, None)
            if mod is None:
                continue
            relu = next(x for x in mod if isinstance(x, (torch.nn.ReLU, torch.nn.GELU)))
            hooks.append(relu.register_forward_hook(lambda m_, i, o, n=name: acts.__setitem__(n, o.detach())))
        with torch.no_grad():
            model._predict_batch(hb)
        for h_ in hooks:
            h_.remove()
        # a unit is alive when some packet drives it into its linear regime (> 0; GELU's negative tail is ~0)
        alive = {k: float((v.reshape(-1, v.shape[-1]) > 0).any(0).float().mean()) for k, v in acts.items()}
        # the projected `feat` may be switched off where the routed evidence carries the update (S37's own is
        # ~0.006 in magnitude); what must stay alive is what the S38 arm reads: the node projection and the root
        ok = alive.get("node_proj", 1.0) >= 0.25 or model.encoder_name != "sparse_pyramid"
        msg = "live readout units " + ", ".join(f"{k} {v:.2f}" for k, v in alive.items())
        if getattr(model, "root_meas", "delta") == "abs":
            with torch.no_grad():
                enc(hb.events, hb.ptr, hb.delta_t_s)
                r = model.root_abs_head(enc.pooled.float())
                g0 = model.root_filter_gain
                model.root_filter_gain = 1.0
                meas = model._abs_root(r, hb.prev_state, hb.counts, int(hb.events.shape[0])).cpu().numpy()
                model.root_filter_gain = g0
            tgt = hb.target.float().cpu().numpy()
            pad = lambda v: np.concatenate([np.zeros((len(v), 3), np.float32), v, np.zeros((len(v), 45), np.float32)], 1)  # noqa: E731
            e_meas = float(EX.rot_err_deg(pad(meas), tgt).mean())
            const = np.broadcast_to(np.asarray(cfg["MODEL"]["ROOT_REF"], np.float32), meas.shape)
            e_const = float(EX.rot_err_deg(pad(np.ascontiguousarray(const)), tgt).mean())
            ok = ok and e_meas <= 0.8 * e_const
            msg += f"; absolute root on the batch {e_meas:.2f} deg vs constant R_ref {e_const:.2f} deg"
        if getattr(model, "finger_meas", "delta") == "abs":
            with torch.no_grad():
                enc(hb.events, hb.ptr, hb.delta_t_s)
                fm = model.finger_abs_head(enc.pooled.float())
            tf_ = hb.target.float()[:, 6:]
            e_f, e_0 = float((fm - tf_).square().mean()), float(tf_.square().mean())
            ok = ok and e_f <= 0.8 * e_0
            msg += f"; absolute fingers MSE {e_f:.4f} vs the mean hand {e_0:.4f}"
        report("health", ok, msg)

    # 7 runtime
    lat = EX.latency_model(model, cfg, dev)
    anchor = EX.latency_anchor(dev)
    print(f"[INFO] runtime: {lat:.2f} ms/packet raw, anchor {anchor:.2f} ms -> {lat * 1.75 / anchor:.2f} ms "
          f"on the 1.75 ms full-model scale", flush=True)
    if a.budget:
        sys.path.insert(0, str(REPO / "tools"))
        import make_s36_row as MR
        bcfg = load_config(a.budget)
        bmodel = MNISTModel(bcfg).to(dev).eval()
        lat_b = EX.latency_model(bmodel, bcfg, dev)
        lat = EX.latency_model(model, cfg, dev)                     # interleaved, same thermal state
        macs, macs_b = MR.macs_forward_packet(cfg)[0], MR.macs_forward_packet(bcfg)[0]
        report("budget", lat <= 1.05 * lat_b and macs <= macs_b,
               f"latency {lat:.2f} vs {lat_b:.2f} ms raw ({lat / lat_b:.3f}x, indicative; the formal number is "
               f"measured in isolation); MACs {macs / 1e9:.3f} vs {macs_b / 1e9:.3f} G")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
