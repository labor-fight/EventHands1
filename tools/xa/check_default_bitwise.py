#!/usr/bin/env python3
r"""[XYZ-ASSOC] default-path bitwise reference (docs/EXPERIMENT_RECORDS.md#xyz-assoc-20261009, section 0).

The three new keys (`MODEL.XYZ_ASSOC_MODE`, `XYZ_CONTOUR_GAP`, `XYZ_ASSOC_PREV`) are opt-in; with them
absent the code must run the same ops as before. This script pins that: for the C1 and C0 configs,
a seeded random-initialised model (no checkpoint, fp32, CPU), a fixed batch of 8 training samples
(`semkine.dataset` train set, `torch.Generator().manual_seed(0)` picks the indices) and two real 50 ms
zgz_local evaluation packets (`make_eval_packet`; a busy step and the busiest step of the first 60) go through `forward_packet`; outputs, `route_stats`,
the association and pooled slots (`xyz_keep_aux`), and a train-mode loss + every parameter gradient
are stored. `--check` repeats the run and compares bit for bit.

    CUDA_VISIBLE_DEVICES= python tools/xa/check_default_bitwise.py --save     # before the edit
    CUDA_VISIBLE_DEVICES= python tools/xa/check_default_bitwise.py --check    # after: max|diff| must be 0

The XYZ readout's last layer (`fc2`) is zero-initialised, so a fresh model outputs `prev` whatever
the association did and the comparison would be vacuous. The script therefore overwrites `fc2` with
seeded random values (and nothing else), which makes every upstream module reach the output and
every parameter receive a nonzero gradient. CPU only: the GPUs belong to the training runs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
os.chdir(REPO)

from config import load_config                                     # noqa: E402
from model import MNISTModel                                       # noqa: E402
from semkine import eval_track as ET                               # noqa: E402
from semkine.dataset import build_dataset, sequences_for_split, splits_manifest   # noqa: E402
from semkine.events import collate_packets                         # noqa: E402

ARMS = {"c1": "configs/xa/xa_c1_s3407.yaml", "c0": "configs/xa/xa_c0_s3407.yaml"}
OUT = REPO / "outputs" / "xa"
N_TRAIN = 8
EVAL_SEQ = "zgz_local"
EVAL_SCAN_STEPS = 60          # evaluation packets are picked among the first 60 steps of the first valid run
EVAL_MIN_EVENTS = 1000        # "evalpkt": the first step with at least this many events
ASSOC_KEYS = ("vid", "A", "q", "z", "valid", "hit")
AUX_KEYS = ("slots", "O", "S", "w", "r", "delta")


def build_model(cfg) -> MNISTModel:
    torch.manual_seed(0)
    model = MNISTModel(cfg=cfg)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        model.xyz.fc2.weight.copy_(torch.randn(model.xyz.fc2.weight.shape, generator=g) * 0.05)
        model.xyz.fc2.bias.copy_(torch.randn(model.xyz.fc2.bias.shape, generator=g) * 0.01)
    return model.float().cpu().eval()


def fixed_batches(cfg):
    """(train batch of 8 samples, {name: real 50 ms zgz_local packet}, descriptors). `evalpkt` is the
    first 50 ms step of the first valid run with >= EVAL_MIN_EVENTS events; `evalpkt_max` is the busiest
    of the first EVAL_SCAN_STEPS steps (more than the 2048 sampled nodes, so `stride_sample` is exercised)."""
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(len(ds), (N_TRAIN,), generator=g).tolist()
    train_batch = collate_packets([ds[i] for i in idx])
    root = Path(cfg["DATA"]["ROOT"])
    seqs = dict(sequences_for_split(root, "val_core", splits_manifest(cfg)))
    events, offsets, aux, pos51 = ET.load_sequence(root, seqs[EVAL_SEQ], EVAL_SEQ)
    tsub_p = root / seqs[EVAL_SEQ] / f"{EVAL_SEQ}_tsub.npy"
    tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    a, b = next((int(a), int(b)) for a, b in runs if b - a > 50 * (EVAL_SCAN_STEPS + 2))
    ends = np.arange(a + 49, b, 50, dtype=np.int64)[:EVAL_SCAN_STEPS]
    counts = np.array([int(offsets[e + 1] - offsets[e - 49]) for e in ends])
    picks = {"evalpkt": int(ends[int(np.argmax(counts >= EVAL_MIN_EVENTS))]),
             "evalpkt_max": int(ends[int(np.argmax(counts))])}
    betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3)
    eval_batches, meta_e = {}, {}
    for name, end in picks.items():
        ev5 = ET._window_events(events, offsets, tsub, end, 50)
        prev = pos51[end - 49].copy() + ET.sample_init_noise(cfg, np.random.default_rng(0), 1.0)
        eval_batches[name] = ET.make_eval_packet(ev5, torch.from_numpy(prev).view(1, -1), betas, K, 50,
                                                 torch.device("cpu"))
        meta_e[name] = {"end_ms": end, "events": int(len(ev5))}
    meta = {"train_idx": idx, "train_counts": train_batch.counts.tolist(), "eval_seq": EVAL_SEQ,
            "eval_run": [a, b], "eval": meta_e}
    return train_batch, eval_batches, meta


def _flat(prefix, d, keys, out):
    for k in keys:
        out[f"{prefix}/{k}"] = d[k].detach().clone().cpu()


def run_arm(arm: str, train_batch, eval_batches, threads: int) -> dict:
    torch.set_num_threads(threads)
    cfg = load_config(ARMS[arm])
    model = build_model(cfg)
    model.xyz_keep_aux = True
    out: dict = {}
    # (a) eval-mode no-grad forward on the real evaluation packet and on the training batch
    for name, batch in (*eval_batches.items(), ("trainbatch", train_batch)):
        model.set_hand_context(None, None)
        model.eval()
        with torch.no_grad():
            pred = model.forward_packet(batch)
        out[f"{name}/eval/pred"] = pred.detach().clone()
        for k, v in model.route_stats.items():
            out[f"{name}/eval/stats/{k}"] = torch.as_tensor(v).detach().clone().cpu()
        _flat(f"{name}/eval/assoc", model.xyz_last["assoc"], ASSOC_KEYS, out)
        _flat(f"{name}/eval/aux", model.xyz_last, AUX_KEYS, out)
    # (b) train-mode forward + loss + backward on the training batch (the path training uses)
    model.train()
    model.zero_grad(set_to_none=True)
    pred = model.forward_packet(train_batch)
    loss, parts = model._compute_loss(pred, train_batch.target, train_batch.betas)
    loss.backward()
    out["trainbatch/train/pred"] = pred.detach().clone()
    out["trainbatch/train/loss"] = loss.detach().clone()
    for k, v in model.route_stats.items():
        out[f"trainbatch/train/stats/{k}"] = torch.as_tensor(v).detach().clone().cpu()
    for n, p in model.named_parameters():
        out[f"trainbatch/train/grad/{n}"] = (p.grad.detach().clone() if p.grad is not None
                                             else torch.full_like(p, float("nan")))
    out["__n_params"] = torch.tensor(sum(p.numel() for p in model.parameters()))
    return out


def diff(a: torch.Tensor, b: torch.Tensor):
    """(bitwise equal, max|a-b| as float64)."""
    if a.shape != b.shape:
        return False, float("inf")
    eq = torch.equal(a, b) or (a.is_floating_point() and bool(((a == b) | (a.isnan() & b.isnan())).all()))
    d = (a.double() - b.double()).abs()
    d = d.nan_to_num(nan=0.0, posinf=float("inf"))
    return bool(eq), float(d.max()) if d.numel() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", action="store_true", help="write the reference (run BEFORE the code change)")
    g.add_argument("--check", action="store_true", help="compare against the reference; exit 1 on any difference")
    ap.add_argument("--arms", default="c1,c0")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (kept fixed between save and check)")
    a = ap.parse_args()
    arms = a.arms.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg0 = load_config(ARMS[arms[0]])
    train_batch, eval_batches, meta = fixed_batches(cfg0)
    print(f"train batch: idx={meta['train_idx']} events/sample={meta['train_counts']}")
    print(f"eval packets ({meta['eval_seq']}, run {meta['eval_run']}): {meta['eval']}")
    bad = 0
    for arm in arms:
        path = OUT / f"bitwise_ref_{arm}.pt"
        if a.save:
            r1 = run_arm(arm, train_batch, eval_batches, a.threads)
            r2 = run_arm(arm, train_batch, eval_batches, a.threads)
            nd = [k for k in r1 if k not in r2 or not diff(r1[k], r2[k])[0]]
            assert not nd, f"{arm}: the reference run is not deterministic on this machine: {nd[:5]}"
            tmp = path.with_suffix(".pt.tmp")
            torch.save({"tensors": r1, "meta": {**meta, "arm": arm, "config": ARMS[arm],
                                                "threads": a.threads, "torch": torch.__version__}}, tmp)
            os.replace(tmp, path)
            print(f"[{arm}] saved {path}  ({len(r1)} tensors, params {int(r1['__n_params'])}, "
                  f"run-to-run identical)")
            continue
        ref = torch.load(path, map_location="cpu", weights_only=False)
        threads = int(ref["meta"].get("threads", a.threads))
        new = run_arm(arm, train_batch, eval_batches, threads)
        keys_ref, keys_new = set(ref["tensors"]), set(new)
        worst, n_bad = 0.0, 0
        groups: dict = {}
        for k in sorted(keys_ref & keys_new):
            ok, md = diff(ref["tensors"][k], new[k])
            grp = "/".join(k.split("/")[:3])
            gm = groups.setdefault(grp, [0.0, 0])
            gm[0] = max(gm[0], md)
            gm[1] += int(not ok)
            worst = max(worst, md)
            n_bad += int(not ok)
        for k in sorted(keys_ref - keys_new):
            print(f"[{arm}] MISSING in new run: {k}")
            n_bad += 1
        for k in sorted(keys_new - keys_ref):
            print(f"[{arm}] EXTRA in new run: {k}")
            n_bad += 1
        for grp, (md, nb) in groups.items():
            print(f"[{arm}]   {grp:<34s} max|diff| = {md:.3e}  mismatched tensors = {nb}")
        print(f"[{arm}] {len(keys_ref & keys_new)} tensors compared; max|diff| = {worst:.3e}; "
              f"mismatched/missing/extra = {n_bad}  -> {'IDENTICAL' if n_bad == 0 and worst == 0 else 'DIFFERENT'}")
        bad += int(n_bad > 0 or worst != 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
