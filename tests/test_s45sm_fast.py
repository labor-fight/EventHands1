"""CPU tests: semkine/s45sm_fast.py gives the same routes as semkine/s45sm.py, bit for bit."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import encoder as ENC  # noqa: E402
from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"
T0P = ROOT / "outputs/s45rt/20261031/t0p/t0p_final.ckpt"


def packets(seed=0, n_per=(900, 0, 400, 1500)):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(80, 140, (n,), generator=g).float()       # crowded: many repeated pixels
        ev[:, EV_Y] = torch.randint(60, 110, (n,), generator=g).float()
        t = torch.sort(torch.rand(n, generator=g) * 0.05).values
        if n:
            hot = torch.randint(0, n, (max(1, n // 20),), generator=g)       # out-of-order rows, like injected events
            t[hot] = torch.rand(len(hot), generator=g) * 0.05
        ev[:, EV_T] = t
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((len(n_per),), 0.05)


class Batch:
    def __init__(self, events, ptr, dt):
        self.events, self.ptr, self.delta_t_s = events, ptr, dt


def test_event_image_equal():
    for seed in range(3):
        events, ptr, dt = packets(seed)
        assert torch.equal(RT.event_image(events, ptr, dt), FAST.event_image(events, ptr, dt))


def test_token_memo():
    events, ptr, dt = packets(1)
    a = FAST.event_tokens(events, ptr, dt, 180, 240)
    assert FAST.event_tokens(events, ptr, dt, 180, 240) is a
    assert torch.equal(a, ENC.event_tokens(events, ptr, dt, 180, 240))
    ev2 = events.clone()
    assert FAST.event_tokens(ev2, ptr, dt, 180, 240) is not a


def compare(router, ckpt, tau, tmp_path=None):
    events, ptr, dt = packets(2)
    batch = Batch(events, ptr, dt)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    try:
        SM.install(router, ckpt, "face_route", tau)
        slow = XAG.sample_sources(batch, src, mask)
    finally:
        SM.uninstall()
    try:
        FAST.install(router, ckpt, "face_route", tau)
        ENC.event_tokens(events, ptr, dt, 180, 240)                          # S45's own call fills the memo
        fast = XAG.sample_sources(batch, src, mask)
    finally:
        FAST.uninstall()
    assert set(slow) == set(fast)
    for k in slow:
        assert torch.equal(slow[k], fast[k]), k
    assert ENC.event_tokens is FAST._NATIVE_TOKENS and XAG.sample_sources is SM._NATIVE["sample_sources"]


def test_routes_equal_t0p():
    compare("t0p", T0P, 0.5)
    compare("t0p", T0P, 0.0)


def test_routes_equal_sr(tmp_path):
    torch.manual_seed(3)
    net = SM.build(NPZ)
    ckpt = tmp_path / "sr.ckpt"
    SM.save_checkpoint(net, ckpt, step=0)
    compare("sr", ckpt, 0.5)
    compare("sr", ckpt, 0.0)


def test_dense_pfa_equal():
    import numpy as np
    with np.load(NPZ, allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        lbs = torch.tensor(np.asarray(z["weights"], np.float32))
    g = torch.Generator().manual_seed(5)
    B, N = 3, 257
    V = verts.unsqueeze(0).repeat(B, 1, 1) + torch.rand(B, 1, 3, generator=g) * 0.01
    V[..., 2] += 0.45
    mask = torch.rand(B, N, generator=g) > 0.25
    face = torch.randint(-3, faces.shape[0] + 3, (B, N), generator=g)                 # some out-of-range faces
    bary = torch.rand(B, N, 3, generator=g)
    bary[0, :5] = 0.0                                                                  # degenerate rows
    valid = torch.rand(B, N, generator=g) > 0.3
    d = torch.cat([torch.rand(B, N, 2, generator=g) * 0.2 - 0.1, torch.ones(B, N, 1)], -1)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((B,), 0.02)
    e_slow = SM.empty_association(d, mask, V, faces, vf, band, band / 2, 0.01)
    e_fast = FAST.empty_association(d, mask, V, faces, vf, band, band / 2, 0.01)
    assert set(e_slow) == set(e_fast) and all(torch.equal(e_slow[k], e_fast[k]) and e_slow[k].dtype == e_fast[k].dtype
                                              for k in e_slow)
    kw = dict(source_part=torch.zeros_like(face), source_face=face, source_bary=bary, source_valid=valid)
    o1, a1 = SM.pfa_association("face_route", e_slow, d, mask, V, faces, vf, lbs, band, band / 2, 0.01, **kw)
    o2, a2 = FAST.pfa_association("face_route", e_fast, d, mask, V, faces, vf, lbs, band, band / 2, 0.01, **kw)
    for k in o1:
        assert torch.equal(o1[k], o2[k]) and o1[k].dtype == o2[k].dtype, k
    for k in a1:
        assert torch.equal(a1[k], a2[k]), k
