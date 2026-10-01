"""x1001 E8: whole-packet causal neighbour rule."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
from semkine.event_gnn import EventGNN  # noqa: E402


def _p(B=3, N=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(B, N, 3, generator=g)
    p[..., 2] = torch.sort(p[..., 2], dim=1).values
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 30:] = False
    return p, mask


def test_window_is_default_and_unchanged():
    enc = EventGNN(max_nodes=50, k=4, window=8)
    assert enc.nbr_mode == "window"
    p, mask = _p()
    a = enc._edges(p, mask)
    b = enc._edges_window(p, mask)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_causal_all_matches_brute_force_and_is_causal():
    enc = EventGNN(max_nodes=50, k=4, nbr_mode="causal_all", nbr_t_scale=0.1)
    p, mask = _p()
    idx, dp, emask = enc._edges(p, mask)
    w = torch.tensor([1.0, 1.0, 0.1])
    for b in range(p.shape[0]):
        for i in range(p.shape[1]):
            if not mask[b, i]:
                continue
            cand = [j for j in range(i) if mask[b, j]]
            if not cand:
                assert emask[b, i].sum() == 0
                continue
            d = ((p[b, cand] * w - p[b, i] * w) ** 2).sum(-1)
            want = sorted(cand[j] for j in d.topk(min(4, len(cand)), largest=False).indices.tolist())
            got = sorted(idx[b, i][emask[b, i] > 0].tolist())
            assert got == want, (b, i, got, want)
            assert all(j < i for j in got)
    # edge features keep the unscaled time axis
    b, i = 0, 40
    j = int(idx[b, i, 0])
    assert torch.allclose(dp[b, i, 0], p[b, j] - p[b, i])


def test_e8_config_single_variable():
    from config import load_config
    from model import MNISTModel
    cfg = load_config(ROOT / "configs/x1001/x1001_e8.yaml")
    ref = load_config(ROOT / "configs/x1001/x1001_s37.yaml")
    a = {k: v for k, v in cfg["MODEL"].items() if k not in ("ENCODER_NBR", "ENCODER_NBR_T_SCALE")}
    assert a == ref["MODEL"]
    m = MNISTModel(cfg).eval()
    assert m.event_encoder.nbr_mode == "causal_all" and m.event_encoder.nbr_t_scale == 0.1
