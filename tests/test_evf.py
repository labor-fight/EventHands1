"""EVF (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): CPU tests of the virtual frame, the event nodes, the point
network and the asynchronous runtime."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [p for p in (str(ROOT), str(ROOT / "model")) if p not in sys.path]

H, W = 180, 240


def base_cfg(**evf):
    return {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "MANO_NCOMPS": 45, "PREDICT_DELTA": True,
                  "PREDICT_BETA": False, "RENDER_H": H, "RENDER_W": W, "RENDER_SCALE": 0.375},
        "LOSS": {"TYPE": "so3_trans_fk", "ROT_WEIGHT": 1.0, "TRANS_WEIGHT": 1.0, "FK_WEIGHT": 2.0, "LAMBDA_POSE": 450.0,
                 "LAMBDA_R": 60.0, "LAMBDA_T": 30000.0, "LOG10": True, "NORMALIZER": 51},
        "TRAIN": {"LR": 0.004, "WARMUP_STEPS": 500, "MAX_STEPS": 6000, "LR_SCHEDULE": "cosine"},
        "MANO": {"NPZ": str(ROOT / "assets/mano_right.npz")},
        "EVF": {"NET": "point", **evf},
    }


def model(seed=0, **evf):
    from semkine.evf_model import EVFModel
    torch.manual_seed(seed)
    return EVFModel(base_cfg(**evf)).eval()


K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])


def states(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.zeros(n, 51)
    s[:, 0:3] = torch.tensor([0.0, 0.0, 0.5]) + 0.05 * torch.randn(n, 3, generator=g)
    s[:, 3:6] = torch.tensor([3.0, 0.0, 0.0]) + 0.3 * torch.randn(n, 3, generator=g)
    s[:, 6:51] = 0.3 * torch.randn(n, 45, generator=g)
    return s


def packets(m, prev, seed=0, n_per=(300, 2000, 0, 50), dts=(0.05, 0.12, 0.05, 0.03)):
    """Synthetic events around each state's projected vertices, in time order."""
    from semkine.events import EventPacketBatch
    g = torch.Generator().manual_seed(seed)
    B = prev.shape[0]
    betas = torch.zeros(B, 10)
    Kb = K.expand(B, 3, 3)
    fx, fy, cx, cy = m._intrinsics(Kb)
    with torch.no_grad():
        verts, _ = m._fk(prev, betas)
    rows = []
    for b in range(B):
        n = n_per[b % len(n_per)]
        v = verts[b, torch.randint(0, 778, (n,), generator=g)]
        u = fx[b] * v[:, 0] / v[:, 2] + cx[b] + 3 * torch.randn(n, generator=g)
        w = fy[b] * v[:, 1] / v[:, 2] + cy[b] + 3 * torch.randn(n, generator=g)
        x, y = u.round().clamp(0, W - 1), w.round().clamp(0, H - 1)
        t = torch.sort(torch.rand(n, generator=g) * dts[b % len(dts)]).values
        p = torch.randint(0, 2, (n,), generator=g).float()
        rows.append(torch.stack([torch.full((n,), float(b)), x, y, t, p], 1))
    ev = torch.cat(rows)
    counts = torch.tensor([r.shape[0] for r in rows])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    dt = torch.tensor([dts[b % len(dts)] for b in range(B)], dtype=torch.float32)
    z = torch.zeros(B, dtype=torch.long)
    return EventPacketBatch(events=ev, ptr=ptr, sequence_id=z, t_start_us=z, t_end_us=(dt * 1e6).long(), delta_t_s=dt,
                            is_sequence_start=z.bool(), is_sequence_end=z.bool(), target=prev.clone(),
                            prev_state=prev.clone(), betas=betas, camera_K=Kb.clone())


def randomize_readout(m, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.net.fc2.weight.copy_(0.01 * torch.randn(m.net.fc2.weight.shape, generator=g))
        m.net.fc2.bias.copy_(0.01 * torch.randn(m.net.fc2.bias.shape, generator=g))


def frame(m, prev):
    from semkine import evf_render as RD
    B = prev.shape[0]
    fx, fy, cx, cy = m._intrinsics(K.expand(B, 3, 3))
    with torch.no_grad():
        verts, joints = m._fk(prev, torch.zeros(B, 10))
    return RD.Frame(verts, joints, m.mano.f, fx, fy, cx, cy, H, W)


def test_rasterizer_matches_brute_force():
    m = model()
    prev = states(2, seed=3)
    fr = frame(m, prev)
    f = m.mano.f.long()
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    px, py = xx.reshape(1, -1), yy.reshape(1, -1)
    for b in range(2):
        u, v, z = fr.u[b][f], fr.v[b][f], fr.z[b][f]
        a = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (v[:, 1] - v[:, 0]) * (u[:, 2] - u[:, 0])
        b0 = ((u[:, 2:3] - u[:, 1:2]) * (py - v[:, 1:2]) - (v[:, 2:3] - v[:, 1:2]) * (px - u[:, 1:2])) / a[:, None]
        b1 = ((u[:, 0:1] - u[:, 2:3]) * (py - v[:, 2:3]) - (v[:, 0:1] - v[:, 2:3]) * (px - u[:, 2:3])) / a[:, None]
        b2 = 1 - b0 - b1
        inside = (b0 >= -1e-6) & (b1 >= -1e-6) & (b2 >= -1e-6) & (a.abs()[:, None] > 1e-12)
        zp = torch.where(inside, 1 / (b0 / z[:, 0:1] + b1 / z[:, 1:2] + b2 / z[:, 2:3]), torch.full_like(b0, np.inf))
        zmin, face = zp.min(0)
        ref = torch.where(torch.isfinite(zmin), face, torch.full_like(face, -1))
        got = fr.face[b].reshape(-1)
        assert int((ref >= 0).sum()) > 500
        assert torch.equal(got >= 0, ref >= 0)
        # a different face only where two faces tie in depth within the z-buffer quantum
        diff = (got != ref) & (ref >= 0)
        assert int(diff.sum()) <= 2
        assert int(fr.clipped[b]) == 0


def test_jfa_matches_brute_force():
    from semkine import evf_render as RD
    m = model()
    fr = frame(m, states(1, seed=4))
    seeds = fr.boundary[0].nonzero().float()
    yy, xx = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    P = torch.stack([yy.reshape(-1), xx.reshape(-1)], -1)
    d = torch.cdist(P, seeds).min(1).values
    got = fr.s1[2][0].reshape(-1)
    near = d < RD.FAR_PX - 2
    err = (got[near].float().sqrt() - d[near]).abs()
    assert float(err.max()) < 0.75 and float(err.mean()) < 1e-3
    assert bool((got[d > RD.FAR_PX + 1] == torch.iinfo(torch.int32).max).all())


def test_previous_times_matches_a_loop():
    from semkine import evf_nodes as EN
    g = torch.Generator().manual_seed(0)
    n = 400
    pix = torch.randint(0, 12, (n,), generator=g)
    pol = torch.randint(0, 2, (n,), generator=g)
    t = torch.sort(torch.rand(n, generator=g)).values
    same, opp = EN.previous_times(pix, pol, t)
    last = {}
    for i in range(n):
        k_s, k_o = (int(pix[i]), int(pol[i])), (int(pix[i]), 1 - int(pol[i]))
        assert float(same[i]) == last.get(k_s, -1.0)
        assert float(opp[i]) == last.get(k_o, -1.0)
        last[k_s] = float(t[i])


def test_nodes_are_prefix_stable():
    from semkine import evf_nodes as EN
    m = model()
    batch = packets(m, states(1, seed=5), n_per=(3000,), dts=(0.05,))
    src, mask, n = EN.select_nodes(batch.events, batch.ptr, batch.delta_t_s, H, W, cap=10 ** 6)
    full = src[0][mask[0]]
    for k in (1, 17, 500, 2999):
        part = batch.events[:k]
        s2, m2, _ = EN.select_nodes(part, torch.tensor([0, k]), batch.delta_t_s, H, W, cap=10 ** 6)
        assert torch.equal(s2[0][m2[0]], full[full < k])


def test_forward_identity_at_init_and_empty_packets():
    m = model()
    prev = states(4, seed=6)
    batch = packets(m, prev)
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert out.shape == (4, 51)
    assert torch.equal(out, prev)
    randomize_readout(m)
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert torch.isfinite(out).all()
    assert float((out - prev).abs().max()) > 1e-4
    assert torch.equal(out[2], prev[2])                       # the empty packet keeps prev bitwise


def test_initial_values_reproducible_and_seed_dependent():
    a, b, c = model(seed=7), model(seed=7), model(seed=8)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa if sa[k].is_floating_point() and "net." in k)


@pytest.mark.parametrize("evf", [{}, {"VF": False}, {"LBS": True}, {"PREV48": False}])
def test_async_matches_whole_packet(evf):
    from semkine.evf_async import AsyncPoint
    m = model(seed=9, **evf)
    randomize_readout(m)
    prev = states(4, seed=10)
    batch = packets(m, prev, seed=11)
    with torch.no_grad():
        ref = m.forward_packet(batch)
    rt = AsyncPoint(m)
    g = torch.Generator().manual_seed(12)
    for b in range(4):
        ev = batch.events[batch.ptr[b]:batch.ptr[b + 1], 1:]
        for chunking in ("one", "random", "all"):
            rt.begin(prev[b:b + 1], batch.betas[b:b + 1], batch.camera_K[b:b + 1], float(batch.delta_t_s[b]))
            i = 0
            while i < ev.shape[0]:
                step = 1 if chunking == "one" else (int(torch.randint(1, 64, (1,), generator=g)) if chunking == "random"
                                                    else ev.shape[0])
                if chunking == "one" and ev.shape[0] > 400 and i >= 200:
                    step = ev.shape[0] - i                      # keep the test short: one-by-one for the first 200
                rt.push(ev[i:i + step])
                i += step
            got = rt.query()
            assert float((got[0] - ref[b]).abs().max()) <= 1e-5, (evf, b, chunking)


def test_two_iterations_and_control_build():
    m2 = model(seed=13, ITERS=2)
    randomize_readout(m2)
    prev = states(2, seed=14)
    batch = packets(m2, prev, n_per=(400, 100), dts=(0.05, 0.05))
    with torch.no_grad():
        out = m2.forward_packet(batch)
    assert torch.isfinite(out).all()
    c = model(seed=13, VF=False)
    assert c.net.phi[0].in_features == 7
    with pytest.raises(ValueError):
        model(BOGUS=1)


ALLOWED_GPU = __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8)))


@pytest.mark.skipif(not (ALLOWED_GPU and torch.cuda.is_available()), reason="one GPU among 2-7 only")
def test_fused_jfa_equals_the_torch_pass():
    from semkine import evf_render as RD
    if RD.triton is None:
        pytest.skip("no Triton")
    g = torch.Generator().manual_seed(0)
    seed = (torch.rand(3, H, W, generator=g) < 0.002).cuda()
    seed[1] = False
    a, b = RD.jfa(seed, fused=True), RD.jfa(seed, fused=False)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
