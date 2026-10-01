"""x1001 E9: the spatial sampler keeps the stride sampler's contract and adds cell round-robin."""
import torch

from semkine.event_gnn import EventGNN
from semkine.events import EV_X, EV_Y


def _packets(counts, seed=0, hot=None):
    g = torch.Generator().manual_seed(seed)
    evs, ptr = [], [0]
    for b, n in enumerate(counts):
        e = torch.zeros(n, 5)
        e[:, 0] = b
        if hot is not None and n > 0:
            # most events on a few hot pixels, the rest spread out (a dense, clumped packet)
            k = int(n * hot)
            e[:k, EV_X] = torch.randint(100, 104, (k,), generator=g).float()
            e[:k, EV_Y] = torch.randint(80, 84, (k,), generator=g).float()
            e[k:, EV_X] = torch.randint(0, 240, (n - k,), generator=g).float()
            e[k:, EV_Y] = torch.randint(0, 180, (n - k,), generator=g).float()
            perm = torch.randperm(n, generator=g)
            e[:, EV_X], e[:, EV_Y] = e[perm, EV_X], e[perm, EV_Y]
        else:
            e[:, EV_X] = torch.randint(0, 240, (n,), generator=g).float()
            e[:, EV_Y] = torch.randint(0, 180, (n,), generator=g).float()
        e[:, 3] = torch.linspace(0, 0.05, n) if n else e[:, 3]
        e[:, 4] = (torch.rand(n, generator=g) > 0.5).float()
        evs.append(e)
        ptr.append(ptr[-1] + n)
    return torch.cat(evs), torch.tensor(ptr)


def test_stride_is_the_default_and_unchanged():
    enc = EventGNN(max_nodes=64)
    assert enc.sample_mode == "stride"
    ev, ptr = _packets([10, 300, 0, 64])
    src, mask = enc._sample(ev, ptr)
    s2, m2 = enc._sample_stride(ev, ptr)
    assert torch.equal(src, s2) and torch.equal(mask, m2)


def test_sparse_packets_identical_to_stride():
    a = EventGNN(max_nodes=64, sample_mode="stride")
    b = EventGNN(max_nodes=64, sample_mode="spatial")
    ev, ptr = _packets([10, 64, 0, 33])
    sa, ma = a._sample(ev, ptr)
    sb, mb = b._sample(ev, ptr)
    assert torch.equal(ma, mb)
    assert torch.equal(sa[ma], sb[mb])


def test_dense_packets_full_sorted_unique_and_round_robin():
    enc = EventGNN(max_nodes=256, sample_mode="spatial", sample_cell=4)
    ev, ptr = _packets([5000, 3000, 100], hot=0.8)
    src, mask = enc._sample(ev, ptr)
    n = mask.sum(1)
    assert n.tolist() == [256, 256, 100]
    for bi in range(3):
        s = src[bi][mask[bi]]
        assert bool((s[1:] > s[:-1]).all()), "time order"
        assert bool(((s >= ptr[bi]) & (s < ptr[bi + 1])).all()), "own packet"
        if n[bi] == 256:
            e = ev[s]
            cell = (e[:, EV_Y].long() // 4) * 60 + e[:, EV_X].long() // 4
            per = torch.bincount(cell, minlength=60 * 45)
            allc = (ev[ptr[bi]:ptr[bi + 1], EV_Y].long() // 4) * 60 + ev[ptr[bi]:ptr[bi + 1], EV_X].long() // 4
            avail = torch.bincount(allc, minlength=60 * 45)
            occ = avail > 0
            # round robin: no cell gets r+1 nodes while another occupied cell with spare events has < r
            short = occ & (per < avail)
            if bool(short.any()):
                assert int(per[occ].max()) - int(per[short].min()) <= 1
            # covers more distinct cells than the stride sampler
            st = EventGNN(max_nodes=256)._sample(ev, ptr)[0][bi][mask[bi]]
            es = ev[st]
            cs = torch.unique((es[:, EV_Y].long() // 4) * 60 + es[:, EV_X].long() // 4).numel()
            assert int((per > 0).sum()) >= cs


def test_e9_config_builds_and_runs():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "model"))
    from config import load_config
    from model import MNISTModel
    cfg = load_config(root / "configs/x1001/x1001_e9.yaml")
    m = MNISTModel(cfg).eval()
    assert m.event_encoder.sample_mode == "spatial" and m.event_encoder.sample_cell == 4
    ref = load_config(root / "configs/x1001/x1001_s37.yaml")
    a = {k: v for k, v in cfg["MODEL"].items() if k not in ("ENCODER_SAMPLE", "ENCODER_SAMPLE_CELL")}
    assert a == ref["MODEL"], "E9 differs from x1001_s37 only in the sampling keys"
    assert {k: v for k, v in cfg.items() if k not in ("MODEL", "TRAIN", "EVAL", "_config_path")} == \
        {k: v for k, v in ref.items() if k not in ("MODEL", "TRAIN", "EVAL", "_config_path")}
