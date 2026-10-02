"""S38 contracts (docs/S38_ROOT_TRACKING_VERDICT.md): the sparse pyramid's operators against dense references,
its packet independence and empty-packet identity; the absolute root measurement's rotation convention,
loss, train / inference agreement and constant-gain filter; S37 left untouched.

    python -m pytest tests/test_s38.py -q
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]
from semkine.events import EventPacketBatch                                    # noqa: E402
from semkine.sparse_pyramid import SparsePyramid, SubmConv, SparseDown, _Level  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _byx(lv):
    """(packet, row, column) of every cell of a level, without the border offset."""
    b, yq, xq = lv.bxy()
    return b, yq - 1, xq - 1


def _packet_events(n, seed, b=0, cx=120, cy=90, spread=25):
    g = torch.Generator().manual_seed(seed)
    ev = torch.zeros(n, 5)
    ev[:, 0] = b
    ev[:, 1] = (cx + spread * torch.randn(n, generator=g)).clamp(0, 239).round()
    ev[:, 2] = (cy + spread * torch.randn(n, generator=g)).clamp(0, 179).round()
    ev[:, 3] = torch.sort(torch.rand(n, generator=g) * 0.05).values
    ev[:, 4] = (torch.rand(n, generator=g) > 0.5).float()
    return ev


def _batch(packets):
    ev = torch.cat([p for p in packets], 0) if packets else torch.zeros(0, 5)
    counts = torch.tensor([len(p) for p in packets])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    for i, p in enumerate(packets):
        ev[ptr[i]:ptr[i + 1], 0] = i
    return ev.to(DEV), ptr.to(DEV), torch.full((len(packets),), 0.05, device=DEV)


# ------------------------------------------------------------------ sparse operators vs dense references
def test_submanifold_conv_equals_dense_conv_on_active_sites():
    torch.manual_seed(0)
    H, W, C = 9, 11, 5
    occ = torch.rand(2, H, W) < 0.35
    b, y, x = occ.nonzero(as_tuple=True)
    key = _Level.make_key(b, y, x, H, W)
    key, order = torch.sort(key)
    lv = _Level(key, H, W)
    lb, ly, lx = _byx(lv)
    feat = torch.randn(len(key), C)
    conv = SubmConv(C, 7).eval()
    with torch.no_grad():
        conv.bn.running_mean.zero_()
        conv.bn.running_var.fill_(1.0 - conv.bn.eps)            # BN = identity
        out = conv.lin(SubmConvInput(conv, feat, lv.neighbours()))
        dense = torch.zeros(2, C, H, W)
        dense[lb, :, ly, lx] = feat
        w = conv.lin.weight.reshape(7, 3, 3, C).permute(0, 3, 1, 2)   # (out, in, ky, kx), row-major offsets
        ref = F.conv2d(dense, w, padding=1)[lb, :, ly, lx]
    assert torch.allclose(out, ref, atol=1e-5)


def SubmConvInput(conv, h, nbr):
    idx, valid = nbr
    M, C = h.shape
    return (torch.index_select(h, 0, idx.reshape(-1)).reshape(M, 9, C) * valid.to(h.dtype)).reshape(M, 9 * C)


def test_sparse_down_equals_dense_stride2_conv_on_parent_sites():
    torch.manual_seed(1)
    H, W, C = 7, 9, 4
    occ = torch.rand(1, H, W) < 0.4
    b, y, x = occ.nonzero(as_tuple=True)
    key, _ = torch.sort(_Level.make_key(b, y, x, H, W))
    lv = _Level(key, H, W)
    lb, ly, lx = _byx(lv)
    feat = torch.randn(len(key), C)
    nxt, inv, slot = lv.parent()
    nb, ny, nx = _byx(nxt)
    down = SparseDown(C, 6).eval()
    with torch.no_grad():
        down.bn.running_mean.zero_()
        down.bn.running_var.fill_(1.0 - down.bn.eps)
        buf = feat.new_zeros(nxt.key.numel() * 4, C)
        buf[inv * 4 + slot] = feat
        out = down.lin(buf.reshape(nxt.key.numel(), -1))
        dense = torch.zeros(1, C, H + 1, W + 1)
        dense[lb, :, ly, lx] = feat
        w = down.lin.weight.reshape(6, 2, 2, C).permute(0, 3, 1, 2)
        ref = F.conv2d(dense, w, stride=2)[nb, :, ny, nx]
    assert torch.allclose(out, ref, atol=1e-5)
    # every occupied child reaches exactly one parent slot
    assert len(set((inv * 4 + slot).tolist())) == len(key)


def test_neighbours_never_wrap_across_rows_or_packets():
    H, W = 4, 5
    # cells on the right edge of row 0 and the left edge of row 1, and the last cell of packet 0 /
    # first cell of packet 1: none of them may be each other's neighbour unless truly adjacent
    pts = [(0, 0, W - 1), (0, 1, 0), (0, H - 1, W - 1), (1, 0, 0)]
    b, y, x = (torch.tensor(v) for v in zip(*pts))
    key, _ = torch.sort(_Level.make_key(b, y, x, H, W))
    lv = _Level(key, H, W)
    lb, ly, lx = _byx(lv)
    idx, valid = lv.neighbours()
    for i in range(len(key)):
        for j in range(9):
            if valid[i, j, 0]:
                k = int(idx[i, j])
                assert int(lb[k]) == int(lb[i])
                assert (j // 3 - 1, j % 3 - 1) == (int(ly[k] - ly[i]), int(lx[k] - lx[i]))
    # the true neighbour pairs above: (0, 0, W-1) has no right / lower-right neighbour in another row
    assert int(valid.sum()) == len(key)                    # only the centre taps: nobody is adjacent


# ------------------------------------------------------------------------------ encoder contracts
def _encoder():
    torch.manual_seed(0)
    return SparsePyramid(hidden=32, feat_dim=64, channels=(8, 16, 24, 32), blocks=(1, 1, 2, 1)).to(DEV).eval()


def test_empty_packet_is_zero_and_has_no_nodes():
    enc = _encoder()
    ev, ptr, dt = _batch([torch.zeros(0, 5)])
    with torch.no_grad():
        out, h, g, px, py, mask = enc(ev, ptr, dt, return_nodes=True)
    assert torch.count_nonzero(out) == 0 and mask.numel() == 0
    # an empty packet next to a full one: still exactly zero, and no live node
    ev, ptr, dt = _batch([_packet_events(500, 3), torch.zeros(0, 5)])
    with torch.no_grad():
        out, h, g, px, py, mask = enc(ev, ptr, dt, return_nodes=True)
    assert torch.count_nonzero(out[1]) == 0 and not bool(mask[1].any()) and bool(mask[0].any())


def test_packets_do_not_leak_into_each_other():
    enc = _encoder()
    pa, pb = _packet_events(800, 4, cx=60), _packet_events(1500, 5, cx=170)
    with torch.no_grad():
        both = enc(*_batch([pa.clone(), pb.clone()]), return_nodes=True)
        a = enc(*_batch([pa.clone()]), return_nodes=True)
        b = enc(*_batch([pb.clone()]), return_nodes=True)
    assert torch.allclose(both[0][0], a[0][0], atol=1e-5) and torch.allclose(both[0][1], b[0][0], atol=1e-5)
    na, nb = int(a[5].sum()), int(b[5].sum())
    assert int(both[5][0].sum()) == na and int(both[5][1].sum()) == nb
    assert torch.allclose(both[1][0, :na], a[1][0], atol=1e-5)
    assert torch.allclose(both[3][1, :nb], b[3][0]) and torch.allclose(both[4][1, :nb], b[4][0])


def test_eval_is_bitwise_deterministic_and_nodes_sit_on_events():
    enc = _encoder()
    p = _packet_events(3000, 6)
    with torch.no_grad():
        o1 = enc(*_batch([p.clone()]), return_nodes=True)
        o2 = enc(*_batch([p.clone()]), return_nodes=True)
    assert all(torch.equal(u, v) for u, v in zip(o1, o2))
    # every node is the centre of a 4 px cell that holds at least one event
    px, py = o1[3][0], o1[4][0]
    cells = {(int(x) // 4, int(y) // 4) for x, y in zip(p[:, 1], p[:, 2])}
    assert {(int(x) // 4, int(y) // 4) for x, y in zip(px, py)} == cells


def test_all_events_count_no_node_cap():
    enc = _encoder()
    # a dense packet far above S37's 2048-node cap: every occupied 4 px cell is a site, every event counts
    p = _packet_events(40000, 7, spread=40)
    lv, f = enc._level0(*_batch([p])[0:1], torch.full((1,), 0.05, device=DEV))
    cells = {(int(x) // 4, int(y) // 4) for x, y in zip(p[:, 1], p[:, 2])}
    assert lv.key.numel() == len(cells)
    site = torch.searchsorted(lv.key, _Level.make_key(torch.zeros(len(p), dtype=torch.long, device=DEV),
                                                       p[:, 2].long().to(DEV) // 4, p[:, 1].long().to(DEV) // 4,
                                                       45, 60))
    n = torch.bincount(site, minlength=len(cells)).float()
    assert torch.allclose(torch.expm1(f[:, 0]) + torch.expm1(f[:, 1]), n, rtol=1e-4)
    # sub-cell offsets are means of values in [0, 1); newest times are maxima of times in [0, 1]
    assert bool((f[:, 5:7] >= 0).all() and (f[:, 5:7] < 1).all() and (f[:, 2:4] <= 1).all())


def test_level0_launch_lean_form_is_bitwise_the_reference_on_real_packets():
    """`_level0` (written for few launches) and `_level0_reference` (the literal form the S38 screening
    arms were trained with) agree bit for bit on real zgz and training packets, at batch 1 and batched."""
    from semkine import eval_track as ET
    root = Path(_cfg("configs/s38/s38_spabs_2k.yaml")["DATA"]["ROOT"])
    enc = _encoder()
    pk = []
    for d, s in (("val", "zgz_global"), ("val", "zgz_local"), ("train", "ch_global_v2")):
        events, offsets, aux, _ = ET.load_sequence(root, d, s)
        tsub = np.load(root / d / f"{s}_tsub.npy", mmap_mode="r")
        a = int(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)[0][0])
        for end in (a + 1049, a + 5049, a + 20049):
            pk.append(torch.from_numpy(ET._window_events(events, offsets, tsub, end, 50)))
    for group in ([p] for p in pk):
        ev, ptr, dt = _batch([g.clone() for g in group])
        l1, f1 = enc._level0(ev, dt)
        l2, f2 = enc._level0_reference(ev, dt)
        assert torch.equal(l1.key, l2.key) and torch.equal(f1, f2)
    ev, ptr, dt = _batch([p.clone() for p in pk])
    dt = dt * torch.linspace(0.6, 1.0, len(pk), device=DEV)                 # per-packet windows
    l1, f1 = enc._level0(ev, dt)
    l2, f2 = enc._level0_reference(ev, dt)
    assert torch.equal(l1.key, l2.key) and torch.equal(f1, f2)


# --------------------------------------------------------------------------- root measurement (M1)
def _cfg(path):
    return yaml.safe_load((REPO / path).read_text())


def _model(cfg):
    from model import MNISTModel
    return MNISTModel(copy.deepcopy(cfg))


def test_chordal_root_loss_matches_axis_angle_mse_for_small_errors_and_has_no_seam():
    from semkine.lie import so3_exp
    torch.manual_seed(0)
    aa = torch.randn(256, 3)
    aa = aa / aa.norm(dim=-1, keepdim=True) * 2.3
    err = aa / aa.norm(dim=-1, keepdim=True) * 0.01                   # 0.57 deg about the axis
    chord = (so3_exp(aa + err) - so3_exp(aa)).square().sum(dim=(-2, -1)).mean() / 6.0
    assert torch.allclose(chord, F.mse_loss(aa + err, aa), rtol=1e-3)
    # one rotation 1e-3 rad short of pi, written on both sides of the seam (k (pi - e) and the
    # 2e-3 rad rotation away from it): axis-angle MSE sees a 2 pi jump, chordal a 2e-3 rad step
    k = torch.tensor([[0.0, 0.0, 1.0]])
    a1, a2 = k * (np.pi - 1e-3), -k * (np.pi - 1e-3)
    assert F.mse_loss(a1, a2) > 3.0
    assert (so3_exp(a1) - so3_exp(a2)).square().sum() / 6.0 < 1e-5


def test_abs_root_train_and_inference_paths_agree_and_filter_is_a_geodesic_step():
    cfg = _cfg("configs/s38/s38_s37abs_2k.yaml")
    m = _model(cfg).to(DEV)
    torch.manual_seed(0)
    B = 16
    r = torch.randn(B, 3, device=DEV) * 0.3                               # measured deviations from R_ref
    prev = torch.randn(B, 51, device=DEV) * 0.1
    prev[:, 3:6] = torch.tensor(cfg["MODEL"]["ROOT_REF"], device=DEV) + 0.2 * torch.randn(B, 3, device=DEV)
    counts = torch.full((B,), 100, device=DEV)
    counts[3] = 0
    m.train()
    tr = m._abs_root(r, prev, counts, 1)
    m.eval()
    m.root_filter_gain = 1.0
    ev = m._abs_root(r, prev, counts, 1)
    sys.path.insert(0, str(REPO / "tools" / "x1001"))
    from evalx import rot_err_deg as _rot51

    def rot_err_deg(a, b):                    # evalx reads columns 3:6 of 51D states
        pa, pb = np.zeros((len(a), 51), np.float32), np.zeros((len(b), 51), np.float32)
        pa[:, 3:6], pb[:, 3:6] = a, b
        return _rot51(pa, pb)
    tr, ev = tr.detach().cpu().numpy(), ev.detach().cpu().numpy()
    assert rot_err_deg(tr, ev).max() < 1e-3                                   # train path == inference path
    assert np.array_equal(ev[3], prev[3, 3:6].cpu().numpy())                  # empty packet holds prev
    m.root_filter_gain = 0.5
    half = m._abs_root(r, prev, counts, 1).detach().cpu().numpy()
    p = prev[:, 3:6].cpu().numpy()
    full = rot_err_deg(p, ev)
    keep = np.arange(B) != 3
    assert np.allclose(rot_err_deg(half, p)[keep], full[keep] / 2, atol=1e-3)
    assert np.allclose(rot_err_deg(half, ev)[keep], full[keep] / 2, atol=1e-3)
    assert np.array_equal(half[3], p[3])


def test_abs_root_reads_no_state():
    """The measurement must not move when only prev moves (filter off)."""
    cfg = _cfg("configs/s38/s38_s37abs_2k.yaml")
    cfg["MODEL"]["ROOT_FILTER_GAIN"] = 1.0
    m = _model(cfg).to(DEV).eval()
    r = torch.randn(4, 3, device=DEV) * 0.2
    counts = torch.full((4,), 50, device=DEV)
    p1 = torch.zeros(4, 51, device=DEV)
    p2 = p1.clone()
    p2[:, 3:6] += 0.5
    assert torch.equal(m._abs_root(r, p1, counts, 1), m._abs_root(r, p2, counts, 1))


def test_root_ref_is_the_training_split_mean():
    cfg = _cfg("configs/s38/s38_spabs_2k.yaml")
    sys.path.insert(0, str(REPO / "tools" / "s38"))
    import make_configs as MC
    assert np.allclose(MC.training_root_ref(), cfg["MODEL"]["ROOT_REF"], atol=1e-6)


def test_s38_arms_differ_from_their_parents_only_where_registered():
    sys.path.insert(0, str(REPO / "tools" / "s38"))
    import make_configs as MC
    ref = _cfg("configs/s38/s38_spabs_2k.yaml")["MODEL"]["ROOT_REF"]
    for name, (parent, over, _) in MC.arms(ref).items():
        got = _cfg(f"configs/s38/{name}.yaml")
        want = MC.merge(copy.deepcopy(yaml.safe_load(parent.read_text())), copy.deepcopy(over))
        for k in ("MODEL", "LOSS", "TRACK", "AUG", "DATA"):
            assert got[k] == want[k], (name, k)


def test_s37_config_builds_the_s37_model_unchanged():
    """S37's config never reaches an S38 branch: delta root, six-wide routed root head, axis-angle loss."""
    m = _model(_cfg("configs/x1001/x1001_s37.yaml"))
    assert m.root_meas == "delta" and m.root_filter_gain == 1.0 and m.root_loss == "mse_aa"
    assert m.root_head.out_features == 6 and not hasattr(m, "root_abs_head")
    assert type(m.event_encoder).__name__ == "EventGNN"
