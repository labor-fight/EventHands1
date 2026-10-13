"""S43 runtime: the SAE gives the evaluator's LNES bit for bit; the local mesh cache equals the full MANO forward."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from mano_layer import ManoLayer                         # noqa: E402
from semkine import eval_track as ET                     # noqa: E402
from semkine.async_tracker import IncrementalMano, _NEVER  # noqa: E402
from semkine.kinematic_gate import subtree_matrix        # noqa: E402

MANO_NPZ = REPO / "assets" / "mano_right.npz"


def test_sae_lnes_is_build_lnes():
    rng = np.random.default_rng(0)
    T, H, W = 400, 180, 240
    counts = rng.integers(0, 40, T)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = int(offsets[-1])
    events = np.stack([rng.integers(0, W, n), rng.integers(0, H, n), rng.integers(0, 2, n)], 1).astype(np.int16)
    sae = torch.full((H * W * 2,), _NEVER, dtype=torch.long)
    for ms in range(T):
        e = events[offsets[ms]:offsets[ms + 1]].astype(np.int64)
        if len(e):
            idx = torch.from_numpy((e[:, 1] * W + e[:, 0]) * 2 + np.clip(e[:, 2], 0, 1))
            sae.scatter_reduce_(0, idx, torch.full((len(e),), ms, dtype=torch.long), reduce="amax")
        for floor, win in ((0, 200), (150, 200), (ms - 30 if ms > 30 else 0, 50)):
            if ms < floor or ms % 37:
                continue
            w = min(win, ms - floor + 1)
            ref = ET.build_lnes(events, offsets, ms, w)
            s = sae.view(H, W, 2)
            start = ms - w + 1
            got = torch.where(s >= start, (s - start).to(torch.float32) / float(w), torch.zeros(()))
            assert np.array_equal(got.numpy(), ref), (ms, floor, win)


def test_subtree_matrix():
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    m = subtree_matrix(parents)
    assert m[3].tolist()[:3] == [1, 1, 1]          # a distal joint supports its whole finger chain
    assert m[1].tolist()[:3] == [1, 0, 0]          # the proximal joint only itself
    assert m[0].sum() == 0                          # the wrist supports no finger joint


def test_incremental_mesh_matches_full_forward():
    torch.manual_seed(0)
    mano = ManoLayer(str(MANO_NPZ), add_mean=False)
    betas = torch.randn(10) * 0.5
    im = IncrementalMano(mano, betas, torch.device("cpu"))
    p = torch.zeros(51)
    p[2] = 0.55
    p[3:6] = torch.tensor([1.5, 0.9, 1.3])
    p[6:] = torch.randn(45) * 0.2
    im.reset(p)
    assert im.check(betas) < 1e-6
    for k in range(20):
        q = im.params.clone()
        if k % 3 == 0:                              # root and translation only: rigid
            q[0:6] += torch.randn(6) * 0.02
        else:                                       # a few finger joints
            j = torch.randint(0, 15, (2,))
            for jj in j.tolist():
                q[6 + 3 * jj:9 + 3 * jj] += torch.randn(3) * 0.1
        info = im.update(q)
        assert im.check(betas) < 2e-6, k
        if k % 3 == 0:
            assert info["verts_reskinned"] == 0
        else:
            assert 0 < info["verts_reskinned"] < 778


def test_deadband_holds_small_updates_exactly():
    from semkine.kinematic_gate import deadband_state
    import math
    prev = torch.zeros(2, 51)
    prev[:, 3:6] = torch.tensor([1.5, 0.9, 1.3])
    out = prev.clone()
    out[:, 6:9] += 0.001          # joint 1: small change (held at 0.5 deg)
    out[:, 9:12] += 0.05          # joint 2: large change (written)
    out[:, 3] += 0.001            # tiny root change (held at 0.25 deg)
    out[:, 0] += 0.01             # translation is never held
    res = deadband_state(prev, out, math.radians(0.5), math.radians(0.25))
    assert torch.equal(res[:, 6:9], prev[:, 6:9])
    assert torch.equal(res[:, 9:12], out[:, 9:12])
    assert torch.equal(res[:, 3:6], prev[:, 3:6])
    assert torch.equal(res[:, 0:3], out[:, 0:3])
    assert torch.equal(deadband_state(prev, out, 0.0, 0.0), out)
