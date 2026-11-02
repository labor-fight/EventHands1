"""Bit-identity fingerprints of the core training path (CPU): the runner behind `tests/test_core_fingerprint.py`.

For each config: seed exactly as `semkine/train.py main()` does, build the train dataset, build `MNISTModel`, hash the
initial state_dict; collate a fixed set of samples (fixed indices, fixed RNG) the way `main()` does; one `training_step`
on CPU (no trainer attached), hash the loss and every gradient; one optimiser step, hash every parameter.

It runs in its own process (the hashes of float32 results depend on the process's thread configuration, which other
tests change) and writes a JSON; `python tests/core_fingerprint_runner.py <out.json> [<name,name,...>]` from anywhere.
The reference is `tests/data/core_fingerprint_20261101.json`, computed on 2026-11-01 from the code BEFORE the core
cleanup of that day (dead opt-in features removed from `model/model.py`, `semkine/train.py`, `dataset.py`, ...).
"""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import pytorch_lightning as pl  # noqa: E402

torch.set_num_threads(4)

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine.dataset import build_dataset  # noqa: E402
from semkine.events import EventPacket, collate_packets  # noqa: E402

CONFIGS = {
    "S45": "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml",
    "C1": "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
    "C1SF_dz": "configs/c1sf_20261029/r2/c1sf_dz_s3407.yaml",
    "S39_dt_dz_l3": "configs/dt/dt_dz_l3.yaml",
    "S43_s43n_sp": "configs/s43/s43n_sp.yaml",
    "EH_abs_full51": "configs/eventhands_abs_full51.yaml",
    "EH_track_delta51": "configs/eventhands_track_delta51.yaml",
    "S45TR_t1d": "configs/s45tr_20261030/s45tr_t1d_s3407.yaml",
    "S45AT_t1ddz": "configs/s45at_20261031/s45at_t1ddz_s3407.yaml",
}
INDICES = [0, 37, 911, 2048, 4097, 8191, 12345, 20000]
INDICES_2 = [5, 777, 3001, 9999, 15000, 30001, 60000, 123457]


def h(t):
    a = t.detach().cpu().contiguous()
    return hashlib.sha256(a.reshape(-1).view(torch.uint8).numpy().tobytes() + str(a.dtype).encode()
                          + str(tuple(a.shape)).encode()).hexdigest()


def hash_state(sd):
    g = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        g.update(k.encode())
        g.update(h(v).encode() if torch.is_tensor(v) else repr(v).encode())
    return g.hexdigest()


def collate(items):
    if isinstance(items[0], EventPacket):
        return collate_packets(items)
    if isinstance(items[0], tuple) and isinstance(items[0][0], EventPacket):
        return tuple(collate_packets(list(col)) for col in zip(*items))
    return torch.utils.data.default_collate(items)


def run(name, path):
    t0 = time.time()
    cfg = load_config(path)
    cfg["TRAIN"]["NUM_WORKERS"] = 0
    pl.seed_everything(int(cfg.get("SEED", 0)), workers=True)
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    model = MNISTModel(cfg)
    rec = {"config": path, "n_train": len(ds), "init_state": hash_state(model.state_dict()),
           "n_params": int(sum(p.numel() for p in model.parameters()))}
    opt = model.configure_optimizers()
    if isinstance(opt, dict):
        opt = opt["optimizer"]
    elif isinstance(opt, (list, tuple)):
        opt = opt[0][0] if isinstance(opt[0], (list, tuple)) else opt[0]
    model.train()
    for step, (seed, idx) in enumerate(((1234, INDICES), (4321, INDICES_2))):
        pl.seed_everything(seed, workers=True)
        batch = collate([ds[i % len(ds)] for i in idx])
        torch.manual_seed(99 + step)
        opt.zero_grad(set_to_none=True)
        out = model.training_step(batch, step)
        loss = out["loss"] if isinstance(out, dict) else out
        rec[f"loss{step}"] = h(loss)
        rec[f"loss_value{step}"] = float(loss)
        loss.backward()
        g = hashlib.sha256()
        nonzero = 0
        for n, p in model.named_parameters():
            g.update(n.encode())
            g.update((h(p.grad) if p.grad is not None else "none").encode())
            nonzero += int(p.grad is not None and bool(p.grad.any()))
        rec[f"grads{step}"] = g.hexdigest()
        rec[f"params_with_grad{step}"] = nonzero
        opt.step()
        rec[f"after_step{step}"] = hash_state({n: p for n, p in model.named_parameters()})
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    out_path = Path(sys.argv[1])
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    res = {}
    for name, path in CONFIGS.items():
        if only and name not in only:
            continue
        if not Path(path).is_file():
            res[name] = {"config": path, "status": "missing"}
            continue
        try:
            res[name] = {"status": "ok", **run(name, path)}
        except Exception as e:  # record, keep going
            res[name] = {"config": path, "status": "error", "error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()[-1500:]}
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
