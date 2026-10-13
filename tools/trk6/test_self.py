#!/usr/bin/env python3
"""trk6 / S42 candidate B debug gate (CPU):
1. clip windows: with domain randomisation / polarity augmentation off, every decoded clip LNES equals the evaluation's
   `eval_track.build_lnes` of the same (end, window) bit for bit, on the legacy and on the fast path; targets = GT; the
   50 ms counts = the raw counts; the initial state = GT(e_0 - 50) + small noise;
2. CLIP_K = 0 leaves the plain samples unchanged (same items as the dataset built without the keys);
3. `MNISTModel._self_filter` equals `semkine.anchored.AdaptiveFilter` (ramp1000 spec) on random states.
"""
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "model")); sys.path.insert(0, str(REPO / "tools"))
from config import load_config                                        # noqa: E402
from model import MNISTModel                                          # noqa: E402
from semkine import eval_track as ET                                  # noqa: E402
from semkine import events as EV                                      # noqa: E402
from semkine.anchored import AdaptiveFilter                           # noqa: E402
from semkine.dataset import build_dataset                             # noqa: E402

SPEC = REPO / "outputs/dt3/gate_specs/ramp1000.json"
base = load_config(str(REPO / "configs/dt/dt_dz_l3_2k.yaml"))
comps = np.load(base["MANO"]["NPZ"])["hands_components"].astype(np.float32)


def ds_of(clip, fast, aug):
    c = copy.deepcopy(base)
    if clip:
        c["DATA"].update({"CLIP_K": 6, "CLIP_WINDOW": 200, "CLIP_FRAC": 1.0})
    if not aug:
        c["AUG"]["DOMRAND"]["ENABLED"] = False
        c["AUG"]["POLARITY_FLIP"] = False
        c["AUG"]["PIXEL_POLARITY_SWAP"] = False
    os.environ["SEMKINE_FAST_PIPE"] = "1" if fast else "0"
    return build_dataset(c, "train", comps, train=True)


ok = True
for fast in (False, True):
    ds = ds_of(True, fast, aug=False)
    rng = np.random.default_rng(0)
    n_valid = n_win = 0
    worst_init = 0.0
    for idx in rng.choice(len(ds), 400, replace=False):
        it = ds[int(idx)]
        if not it["valid"]:
            continue
        n_valid += 1
        si = int(ds.index[idx][0]); run_a = int(ds.index[idx][2])
        h = ds._handle(si)
        end = int(ds._window(int(idx), np.random.default_rng((ds.seed * 1_000_003 + int(idx)) & 0x7FFFFFFF))[1])
        ends = [end - 50 * (5 - k) for k in range(6)]
        for k, e in enumerate(ends):
            w = min(200, e - run_a + 1)
            ref = ET.build_lnes(h.events, h.offsets, e, w, ("last",))
            dec = (it["clip_codes"][k].float() / it["clip_w"][k]).numpy()
            if not np.array_equal(dec, ref):
                ok = False
                print("LNES mismatch", fast, idx, k, np.abs(dec - ref).max())
            if not np.allclose(it["clip_tgt"][k].numpy(), ds._to_target(h.pos51[e]), atol=0):
                ok = False
                print("target mismatch", fast, idx, k)
            if float(it["clip_n50"][k]) != float(h.offsets[e + 1] - h.offsets[e - 49]):
                ok = False
                print("count mismatch", fast, idx, k)
            n_win += 1
        worst_init = max(worst_init, float(np.abs(it["clip_init"].numpy() - ds._to_target(h.pos51[ends[0] - 50])).max()))
    print(f"fast={fast}: valid clips {n_valid}, windows checked {n_win}, max |init - GT| {worst_init:.3f}")
    ok &= n_valid > 50 and worst_init < 0.5

# 2. plain samples unchanged by the clip keys (augmentation on, default stream)
for fast in (False, True):
    a, b = ds_of(False, fast, aug=True), ds_of(True, fast, aug=True)
    for idx in (3, 1000, 123456, 2000000):
        x, y = a[idx], b[idx]["main"]
        same = all(torch.equal(u, v) for u, v in zip(x, y))
        ok &= same
        if not same:
            print("plain sample changed", fast, idx)
print("plain samples unchanged:", ok)

# 3. filter equivalence
cfg = copy.deepcopy(base)
cfg["DATA"]["CLIP_K"] = 6
cfg["TRACK"].update({"SELF_P": 0.5, "SELF_RAMP": [0, 10], "SELF_FILTER": str(SPEC)})
m = MNISTModel(cfg).eval()
torch.manual_seed(0)
B = 64
prev = torch.randn(B, 51) * 0.3
z = prev + torch.randn(B, 51) * 0.1
x = (torch.rand(B, 180, 240, 2) > 0.95).float() * torch.rand(B, 180, 240, 2)
x[:3] = 0
n = torch.tensor(np.exp(np.random.default_rng(0).uniform(0, 11, B)), dtype=torch.float64)
n[5] = 0


class Fixed(torch.nn.Module):
    def forward(self, x, p, betas=None, camera_K=None):
        return z


af = AdaptiveFilter.from_spec(Fixed(), AdaptiveFilter.canonical_spec(json.loads(SPEC.read_text())))
ref = af(x, prev, n_events=n)
got = m._self_filter(prev, z, n, x)
err = float((ref - got).abs().max())
print("filter max |diff|", err)
ok &= err < 1e-5
print("GATE", "PASS" if ok else "FAIL")
