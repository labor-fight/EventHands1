#!/usr/bin/env python3
"""Produce a main-table row for an `event_gnn`-family tracking run (S36 layout).

Accuracy re-runs the exact selection protocol (`select_checkpoint.py`: val_core = the held-out
subject zgz under `splits_semkine.json`, 50 ms recursive steps, rng seed 0, the checkpoint each
seed's selection JSON picked) and aggregates the two zgz sequences into the table's local / global columns,
frames-weighted, root-aligned. The reproduced overall RA is asserted against the
selection JSON so a protocol drift cannot go unnoticed.

Cost columns use batch-1 latency on this machine scaled
so that EventHands-Full = 1.75 ms, thop MACs for the frontend only (the CNN rows count
conv1+resnet only; heads add ~0.0005 G and are noted), params = every nn.Parameter.
The one necessary deviation: an event frontend's latency is data dependent, so instead
of a fixed random input it replays real 50 ms packets from a val sequence and reports
the mean per step (min over 3 passes); the anchor stays min-over-rounds like the table.

The head-MACs footnote assumes the S36 head layout (ACTIVE_HEAD + PREVPOS_EMBED); the
script refuses runs configured otherwise rather than printing a silently wrong number.

    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py                      # S36
    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py --run <arm>          # any later arm
    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py --run <arm> --split test --subject zgz
        # score the selected checkpoint on another set for the record (val_core is zgz already)
    CUDA_VISIBLE_DEVICES=0 python tools/make_s36_row.py --run xa_c1 --step last
        # XYZ-ASSOC: the last step (step=6000 checkpoint, else the largest step) of every seed, no selection
        # JSON, no drift assertion; written to <run>_main_row_last.json. `--step 3000` scores that step
        # (<run>_main_row_step3000.json). The default `--step selection` is the behaviour above.

The FLOPs column reports the whole `forward_packet` (thop, standard layers only) next to the
frontend-only figure the S36 row used, so an arm whose conditioning path holds parameters
outside `event_encoder` is not under-counted.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                   # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402

SEEDS = (3407, 3408)
STEP_MS = 50
FULL_PUBLISHED_MS = 1.75
LOCAL = lambda s: "_local" in s          # noqa: E731
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def wmean(rows, key):
    n = sum(r["n_frames"] for r in rows)
    return sum(r[key] * r["n_frames"] for r in rows) / max(n, 1)


def step_policy_arg(text: str):
    """`--step`: `selection` (the selection JSON's pick, the default), `last`, or an integer step."""
    t = str(text).strip().lower()
    if t in ("selection", "last"):
        return t
    try:
        n = int(t)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--step must be selection, last or an integer step, got {text!r}") from None
    if n <= 0:
        raise argparse.ArgumentTypeError(f"--step must be a positive step, got {n}")
    return n


def row_tag(split: str, subject: str | None, step_policy) -> str:
    """Suffix of `<run>_main_row<tag>.json`: the split / subject tag of a non-default evaluation set, then
    `_last` / `_step<n>` for a non-default checkpoint policy. The default (selection on val_core) is ''."""
    tag = "" if (split == "val_core" and not subject) else f"_{split}_{subject or 'all'}"
    if step_policy == "last":
        tag += "_last"
    elif step_policy != "selection":
        tag += f"_step{step_policy}"
    return tag


def resolve_step_ckpt(run: Path, policy) -> tuple[int, Path]:
    """`(step, checkpoint)` of the non-selection policies. `last`: the `*step=6000.ckpt` (the configured
    MAX_STEPS) if the run has it, else the largest `step=<n>` checkpoint; an integer: that step. Two
    checkpoints for one step (Lightning's `-v1` copies) are ambiguous and raise."""
    grid: dict[int, list[Path]] = {}
    for p in sorted(run.glob("*step=*.ckpt")):
        m = re.search(r"step=(\d+)", p.name)
        if m:
            grid.setdefault(int(m.group(1)), []).append(p)
    if not grid:
        raise FileNotFoundError(f"no *step=<n>.ckpt in {run}")
    step = (6000 if 6000 in grid else max(grid)) if policy == "last" else int(policy)
    if step not in grid:
        raise FileNotFoundError(f"no step={step} checkpoint in {run} (have {sorted(grid)})")
    if len(grid[step]) > 1:
        raise RuntimeError(f"several step={step} checkpoints in {run}: {[p.name for p in grid[step]]}")
    if policy == "last" and step != 6000:
        print(f"  WARNING {run.name}: no step=6000 checkpoint, 'last' = step {step}")
    return step, grid[step][0]


def eval_seed(run_name: str, seed: int, device, split: str = "val_core", subject: str | None = None,
              step_policy="selection"):
    """Under the current protocol `val_core` *is* the held-out subject zgz (`zgz_global`,
    `zgz_local`), which is also what the main table reports on; `split`/`subject` remain for
    scoring a selected checkpoint on some other set for the record. `step_policy` other than
    `selection` scores the run's last / a given step instead of the selection JSON's pick (no
    selection file needed, no drift assertion)."""
    run = REPO / f"outputs/semkine/{run_name}_s{seed}"
    if step_policy == "selection":
        sel_files = sorted(run.glob(f"selection_val_core_step{STEP_MS}*.json"))
        assert sel_files, f"no selection file in {run}"
        sel = json.loads(sel_files[0].read_text())["selected"]
    else:
        step, ckpt = resolve_step_ckpt(run, step_policy)
        sel = {"step": step, "ckpt": str(ckpt)}
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    seqs = sequences_for_split(root, split, mani)
    if subject:
        seqs = [(s, d) for s, d in seqs if s.split("_")[0] == subject]
    assert seqs, f"no sequences for split={split} subject={subject}"
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg,
                                            map_location=device).to(device).eval()
    rng = np.random.default_rng(0)
    res = [ET.track_sequence(model, mano, cfg, root, d, s, STEP_MS, device, rng, 1.0, 1.0, None)
           for s, d in seqs]
    res = [r for r in res if r]
    agg = {"step": sel["step"], "ckpt": sel["ckpt"], "sequences": [r["seq"] for r in res],
           "n_frames": int(sum(r["n_frames"] for r in res)),
           "overall": {k: wmean(res, k) for k in METRICS},
           "local": {k: wmean([r for r in res if LOCAL(r["seq"])], k) for k in METRICS},
           "global": {k: wmean([r for r in res if not LOCAL(r["seq"])], k) for k in METRICS},
           "per_seq": {r["seq"]: {k: r[k] for k in METRICS} for r in res}}
    if step_policy == "selection" and split == "val_core" and not subject:
        drift = abs(agg["overall"]["mpjpe_ra_mm"] - sel["mpjpe_ra_mm"])
        print(f"  s{seed}: step={sel['step']} RA={agg['overall']['mpjpe_ra_mm']:.4f} "
              f"(selection said {sel['mpjpe_ra_mm']:.4f}, drift {drift:.4f} mm)")
        assert drift < 0.05, "re-run does not reproduce the selection protocol"
    else:
        print(f"  s{seed}: step={sel['step']} {split}/{subject or 'all'} "
              f"{agg['sequences']} RA={agg['overall']['mpjpe_ra_mm']:.4f} "
              f"local={agg['local']['mpjpe_ra_mm']:.2f} global={agg['global']['mpjpe_ra_mm']:.2f}")
    del model
    torch.cuda.empty_cache()
    return agg, cfg, seqs, root


@torch.no_grad()
def latency_anchor(device, iters=300, rounds=8):
    m = MNISTModel(load_config(REPO / "configs/eventhands_abs_full51.yaml")).to(device).eval()
    x = torch.rand(1, 180, 240, 2, device=device)
    prev = torch.randn(1, 51, device=device) * 0.02
    prev[:, 2] += 0.4
    for _ in range(iters):
        m(x, prev)
    torch.cuda.synchronize()
    best = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x, prev)
        torch.cuda.synchronize()
        best.append((time.perf_counter() - t0) / iters * 1e3)
    del m
    torch.cuda.empty_cache()
    return min(best)


@torch.no_grad()
def latency_s36(cfg, ckpt, seqs, root, device, max_packets=600, passes=3):
    """Mean forward_packet time over real 50 ms packets of one sequence, staged on GPU.
    Always `lyq_local` (the sequence every recorded latency was measured on; a training sequence
    under the current protocol, which is irrelevant for a timing), so the column stays comparable
    across arms."""
    model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device).to(device).eval()
    seq, d = next((s, d) for split in ("val_core", "train")
                  for s, d in sequences_for_split(root, split, splits_manifest(cfg)) if s == "lyq_local")
    events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
    tsub_p = root / d / f"{seq}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    packets, counts, gts = [], [], []
    #: XYZ-ASSOC oracle arm: its association needs the window-end ground truth for every packet
    oracle = getattr(model, "xyz_assoc_prev", "self") == "target"
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        prev = torch.from_numpy(pos51[a].copy()).view(1, -1).to(device)
        for end in np.arange(a + STEP_MS - 1, b, STEP_MS, dtype=np.int64):
            ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
            packets.append(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, device))
            counts.append(len(ev5))
            if oracle:
                gts.append(torch.from_numpy(pos51[end].copy()).view(1, -1).to(device))
            if len(packets) >= max_packets:
                break
        if len(packets) >= max_packets:
            break
    for i, b_ in enumerate(packets[:100]):
        if oracle:
            model.xyz_target_override = gts[i]
        model.forward_packet(b_)
    torch.cuda.synchronize()
    means = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for i, b_ in enumerate(packets):
            if oracle:
                model.xyz_target_override = gts[i]
            model.forward_packet(b_)
        torch.cuda.synchronize()
        means.append((time.perf_counter() - t0) / len(packets) * 1e3)
    del model
    torch.cuda.empty_cache()
    return min(means), int(np.mean(counts)), len(packets)


def macs_s36(cfg):
    from thop import profile
    mc = cfg.get("MODEL", {})
    if str(mc.get("ENCODER", "")).lower() == "xyz_mesh":
        # S37-XYZ: no S36 heads. "heads" = the common readout (grouped slot projection + two layers);
        # "frontend" = everything else thop counts in the whole forward_packet (relation encoder on
        # the association records, vertex projection, mesh layers), which depends on the packet.
        m = MNISTModel(cfg).eval()
        x = m.xyz
        heads = (x.slot_proj.in_channels // x.slot_proj.groups * x.slot_proj.out_channels
                 + x.fc1.in_features * x.fc1.out_features + x.fc2.in_features * x.fc2.out_features)
        total, _ = macs_forward_packet(cfg)
        n_params = sum(p.numel() for p in m.parameters())
        n_head = sum(p.numel() for n, p in m.named_parameters()
                     if n.startswith(("xyz.slot_proj.", "xyz.fc1.", "xyz.fc2.")))
        return total - heads, heads, n_params, n_params - n_head
    assert bool(mc.get("ACTIVE_HEAD")) and bool(mc.get("PREVPOS_EMBED")), \
        "head-MACs formula below is for the S36 head layout only"
    m = MNISTModel(cfg).eval()
    enc = copy.deepcopy(m.event_encoder)
    hid = int(mc.get("ACTIVE_HIDDEN", 64))
    if str(mc.get("ENCODER", "")).lower() in ("fk_graph", "mesh_graph"):
        # S37 FK graph / mesh graph: the encoder consumes per-node observations, not events; its
        # cost does not depend on the event count at all (the assignment scatter, the visibility
        # test and the LBS pool are not standard layers).
        hidden = int(mc.get("ENCODER_HIDDEN", 128))
        obs = torch.zeros(1, enc.n_nodes, enc.obs_embed.in_features)
        macs, _ = profile(enc, inputs=(obs,), verbose=False)
        if str(mc.get("ENCODER", "")).lower() == "mesh_graph":
            # fingers read [mean, max, coverage] of their joint; root reads all sixteen + background
            ev_dim = 2 * hidden + 1
            heads = (16 * ev_dim + hidden) * 6 + 15 * ((ev_dim + 3) * hid + hid * 3) + 2 * 51 * 64
        else:
            heads = (17 * hidden) * 6 + 15 * ((hidden + 3) * hid + hid * 3) + 2 * 51 * 64
        n_params = sum(p.numel() for p in m.parameters())
        n_enc = sum(p.numel() for p in enc.parameters())
        return macs, heads, n_params, n_enc
    n_ev = 4096                               # > max_nodes, so the padded width is saturated
    ev = torch.zeros(n_ev, 5)
    ev[:, 1] = torch.rand(n_ev) * 240
    ev[:, 2] = torch.rand(n_ev) * 180
    ev[:, 3] = torch.linspace(0, 0.05, n_ev)
    ev[:, 4] = (torch.rand(n_ev) > 0.5).float()
    ptr = torch.tensor([0, n_ev])
    dt = torch.tensor([0.05])
    extra = torch.zeros(n_ev, enc.in_dim - 7)
    macs, _ = profile(enc, inputs=(ev, ptr, dt, extra), verbose=False)
    feat = int(mc.get("ENCODER_FEAT", 512))
    if bool(mc.get("ROUTED_READOUT", False)):
        # S37 routed readout: fingers read their joint's evidence, the root reads all sixteen
        ev_dim = 2 * int(mc.get("ENCODER_HIDDEN", 128)) + 1
        if str(mc.get("ROOT_FUSION", "concat")).lower() == "per_joint":
            root = 16 * ((feat + ev_dim) * hid + hid * 6)       # sixteen [f; e_j] MLPs, summed
        else:
            root = (feat + 16 * ev_dim) * 6
        heads = root + 15 * ((ev_dim + 3) * hid + hid * 3) + 2 * 51 * 64
    else:
        heads = feat * 6 + 15 * ((feat + 3) * hid + hid * 3) + 2 * 51 * 64
    n_params = sum(p.numel() for p in m.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    return macs, heads, n_params, n_enc


def macs_forward_packet(cfg):
    """thop MACs of the whole `forward_packet` (frontend + conditioning path + heads) on one
    saturated packet (4096 events, prev at 0.45 m so the projected hand lands in the frame).

    Complements `macs_s36`, which counts the frontend alone. thop only counts standard layers
    (Linear / Conv / BN): knn, scatter, cdist / topk, MANO FK and the S36 render are not counted
    in any arm, so the column is comparable across arms but is a lower bound for all of them.
    """
    from thop import profile
    from semkine.events import EventPacketBatch

    class _Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, batch):
            return self.m.forward_packet(batch)

    m = copy.deepcopy(MNISTModel(cfg).eval())
    n_ev = 4096
    g = torch.Generator().manual_seed(0)
    ev = torch.zeros(n_ev, 5)
    ev[:, 1] = (122 + 30 * torch.randn(n_ev, generator=g)).clamp(0, 239).round()
    ev[:, 2] = (91 + 30 * torch.randn(n_ev, generator=g)).clamp(0, 179).round()
    ev[:, 3] = torch.linspace(0, 0.05, n_ev)
    ev[:, 4] = (torch.rand(n_ev, generator=g) > 0.5).float()
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.45
    K = torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796], [0, 0, 1.0]]).view(1, 3, 3)
    batch = EventPacketBatch(events=ev, ptr=torch.tensor([0, n_ev]), sequence_id=torch.zeros(1, dtype=torch.long),
                             t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
                             delta_t_s=torch.tensor([0.05]), is_sequence_start=torch.zeros(1, dtype=torch.bool),
                             is_sequence_end=torch.zeros(1, dtype=torch.bool), target=prev.clone(),
                             prev_state=prev, betas=torch.zeros(1, 10), camera_K=K, lnes=None)
    from semkine.routed_readout import PerJointRootFusion

    def _count_root_fusion(mod, x, y):
        # sixteen [f; e_j] -> hidden -> 6 MLPs; thop does not see inside the batched matmuls
        mod.total_ops += torch.DoubleTensor([x[0].shape[0] * mod.n_joints
                                             * (mod.in_dim * mod.hidden + mod.hidden * mod.out_dim)])

    try:
        macs, _ = profile(_Wrap(m), inputs=(batch,), verbose=False,
                          custom_ops={PerJointRootFusion: _count_root_fusion})
        return float(macs), "forward_packet"
    except Exception as e:                   # e.g. a CUDA-only conditioning path on CPU
        return float("nan"), f"forward_packet not profilable on CPU: {type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s36_eventgnn",
                    help="run prefix under outputs/semkine/, without the _s<seed> suffix")
    ap.add_argument("--split", default="val_core",
                    help="sequences for the accuracy columns (val_core = the selection protocol)")
    ap.add_argument("--subject", default=None,
                    help="keep only this subject's sequences, e.g. --split test --subject zgz "
                         "for the main table's held-out pair")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                    help="comma-separated seeds; a single seed is a screen row, not an adoption row")
    ap.add_argument("--step", type=step_policy_arg, default="selection", metavar="{selection|last|<int>}",
                    help="which checkpoint of each seed to score: the selection JSON's pick (default), the last "
                         "step (step=6000 checkpoint, else the largest step; no selection file, no drift "
                         "assertion; row -> <run>_main_row_last.json) or a given step (-> _main_row_step<n>.json)")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    device = torch.device("cuda")
    tag = row_tag(args.split, args.subject, args.step)
    ckpt_what = {"selection": "selected checkpoints", "last": "last-step checkpoints (no selection)"}.get(
        args.step, f"step-{args.step} checkpoints (no selection)")
    print(f"== {args.run}: accuracy ({args.split}/{args.subject or 'all'}, "
          f"50 ms recursive, {ckpt_what}) ==")
    per_seed, cfg, seqs, root = {}, None, None, None
    for seed in seeds:
        agg, cfg, seqs, root = eval_seed(args.run, seed, device, args.split, args.subject, args.step)
        per_seed[seed] = agg

    mean2 = {blk: {k: float(np.mean([per_seed[s][blk][k] for s in seeds]))
                   for k in METRICS} for blk in ("overall", "local", "global")}

    print("== cost (same machine, main-table method) ==")
    anchor = latency_anchor(device)
    raw_lat, mean_events, n_pk = latency_s36(cfg, per_seed[seeds[0]]["ckpt"], seqs, root, device)
    lat = raw_lat / anchor * FULL_PUBLISHED_MS
    macs, head_macs, n_params, n_enc = macs_s36(cfg)
    macs_full, macs_full_note = macs_forward_packet(cfg)
    print(f"  Full anchor raw {anchor:.3f} ms | {args.run} raw {raw_lat:.3f} ms over {n_pk} packets "
          f"(mean {mean_events} events) -> scaled {lat:.2f} ms")
    print(f"  MACs frontend {macs/1e9:.3f} G (+ heads {head_macs/1e6:.2f} M) | "
          f"whole forward_packet {macs_full/1e9:.3f} G ({macs_full_note}) | "
          f"params {n_params/1e6:.2f} M (frontend {n_enc/1e6:.2f} M)")

    row = {"per_seed": per_seed, "two_seed_mean": mean2,
           "latency_ms_scaled_full1p75": lat, "latency_ms_raw": raw_lat,
           "anchor_full_raw_ms": anchor, "latency_packets": n_pk,
           "latency_mean_events_per_packet": mean_events,
           "macs_frontend": macs, "macs_heads": head_macs,
           "macs_forward_packet": macs_full, "macs_forward_packet_note": macs_full_note,
           "params_total": n_params, "params_frontend": n_enc,
           "protocol": {"split": args.split, "subject": args.subject,
                        "manifest": (splits_manifest(cfg).name if splits_manifest(cfg) else "splits_semkine.json"),
                        "step_ms": STEP_MS, "sequences": per_seed[seeds[0]]["sequences"],
                        "note": ("accuracy = two-seed mean of each seed's selected step" if len(seeds) > 1
                                 else f"accuracy = seed {seeds[0]} only (screen row, not an adoption row)")}}
    if args.step != "selection":
        # the default row keeps its keys; a last-step / fixed-step row says which policy produced it
        row["step_policy"] = "last" if args.step == "last" else f"step{args.step}"
        row["protocol"]["step_policy"] = row["step_policy"]
        row["protocol"]["note"] = (f"accuracy = {'two-seed mean' if len(seeds) > 1 else 'seed ' + str(seeds[0])}"
                                   f" of each seed's {row['step_policy']} checkpoint, steps "
                                   f"{[per_seed[s]['step'] for s in seeds]} (no checkpoint selection)")
    out = REPO / f"outputs/semkine/{args.run}_main_row{tag}.json"
    out.write_text(json.dumps(row, indent=2))

    m = mean2
    print("\n| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global |"
          " RA-MPJPE(递推) | Latency | FLOPs/step | Params |")
    print("|---|---|---|---|---|---|---|---|---|")
    print(f"| {args.run} (双种子均值, {args.split}/{args.subject or 'all'}) | {m['local']['mpjpe_ra_mm']:.2f} "
          f"| {m['global']['mpjpe_ra_mm']:.2f} | {m['local']['mpvpe_ra_mm']:.2f} "
          f"| {m['global']['mpvpe_ra_mm']:.2f} | {m['overall']['mpjpe_ra_mm']:.2f} "
          f"| {lat:.2f} ms | {macs_full/1e9:.3f} G (frontend {macs/1e9:.3f} G) | {n_params/1e6:.2f} M |")
    for s in seeds:
        p = per_seed[s]
        print(f"|   └ s{s} (step {p['step']}) | {p['local']['mpjpe_ra_mm']:.2f} "
              f"| {p['global']['mpjpe_ra_mm']:.2f} | {p['local']['mpvpe_ra_mm']:.2f} "
              f"| {p['global']['mpvpe_ra_mm']:.2f} | {p['overall']['mpjpe_ra_mm']:.2f} "
              f"| — | — | — |")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
