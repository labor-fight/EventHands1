"""MRF receptive-field variants (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): `legacy` is C1 bit for bit, every
variant keeps C1's initial values, each variant's receptive field and support are what it claims, the chunked ball and
the sparse local attention equal their dense references, and the probe tail is AGG778's tail for `legacy`."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import mrf as M  # noqa: E402
from semkine.xag import GlobalResidual  # noqa: E402
from semkine.xyz_mesh import (XYZMeshLayer, XYZMeshNet, associate_rays, mesh_ring,  # noqa: E402
                              vertex_face_table)

V = 778


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def mano():
    with np.load(ROOT / "assets" / "mano_right.npz", allow_pickle=False) as z:
        return (torch.tensor(np.asarray(z["f"], np.int64)), torch.tensor(np.asarray(z["weights"], np.float32)),
                torch.tensor(np.asarray(z["v_template"], np.float32)))


def scene(B=2, N=192, seed=0):
    """Two posed copies of the MANO template at z ~ 0.5 m and events aimed at random vertices: hits, near events,
    padded rows, a partial direct support."""
    faces, lbs, v = mano()
    g = torch.Generator().manual_seed(seed)
    verts = torch.stack([v + torch.tensor([0.0, 0.0, 0.5]) + 0.01 * torch.randn(1, 3, generator=g)
                         for _ in range(B)])
    pick = torch.randint(0, V, (B, N), generator=g)
    target = verts[torch.arange(B)[:, None], pick] + 0.002 * torch.randn(B, N, 3, generator=g)
    d = target / target[..., 2:3]
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -16:] = False
    assoc = associate_rays(d, mask, verts, faces, vertex_face_table(faces, V), torch.full((B,), 16 / 300.0),
                           torch.full((B,), 3 / 300.0), 0.01)
    tok = torch.randn(B, N, 7, generator=g)
    prev48 = torch.randn(B, 48, generator=g) / 10
    return faces, lbs, (tok, d, mask, verts, assoc, prev48)


def activate(m):
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        m.fc2.weight.copy_(torch.randn(m.fc2.weight.shape, generator=g) * .02)
    return m


def build(rf=None, xag_gnn="legacy"):
    faces, lbs, _ = scene()
    torch.manual_seed(3407)
    if rf is None:
        return XYZMeshNet(lbs, faces, xag_gnn=xag_gnn)
    return M.MRFXYZMeshNet(lbs, faces, rf=rf)


def run(m, x=None):
    m.eval()
    with torch.no_grad():
        return m(*(x or scene()[2]), capture=True)


def bfs_hops(faces):
    idx, emask = mesh_ring(faces, V)
    nb = [idx[v][emask[v] > 0].tolist() for v in range(V)]
    D = np.full((V, V), -1, np.int64)
    for s in range(V):
        D[s, s], front, h = 0, [s], 0
        while front:
            h += 1
            nxt = []
            for u in front:
                for w in nb[u]:
                    if D[s, w] < 0:
                        D[s, w] = h
                        nxt.append(w)
            front = nxt
    return D


# --------------------------------------------------------------------------- neighbourhoods
def test_hop_matrix_and_balls_match_bfs():
    faces, _, _ = mano()
    D = bfs_hops(faces)
    Dm = M.hop_matrix(faces, V, 4).numpy()
    np.testing.assert_array_equal(Dm, np.where((D >= 0) & (D <= 4), D, 5))
    idx1, m1 = M.ball_table(faces, V, 1)
    ridx, rmask = mesh_ring(faces, V)
    for v in range(V):
        assert sorted(idx1[v][m1[v] > 0].tolist()) == sorted(ridx[v][rmask[v] > 0].tolist())
    idx4, m4 = M.ball_table(faces, V, 4)
    for v in (0, 100, 333, 777):
        assert sorted(idx4[v][m4[v] > 0].tolist()) == sorted(np.nonzero((D[v] >= 1) & (D[v] <= 4))[0].tolist())
    assert int(m4.sum(1).max()) == 111


def test_ball_index_chunks_cover_every_vertex_once():
    faces, _, _ = mano()
    b = M.BallIndex(faces, V)
    rows = torch.cat([r for r, _, _ in b.chunks()])
    assert sorted(rows.tolist()) == list(range(V))
    assert torch.equal(rows[b.inv], torch.arange(V))
    idx4, m4 = M.ball_table(faces, V, 4)
    for r, nbr, msk in b.chunks():
        assert torch.equal((nbr * msk.long()), (idx4[r, :nbr.shape[1]] * m4[r, :nbr.shape[1]].long()))
        assert int(m4[r].sum(1).max()) == nbr.shape[1]


def test_ball_round_on_the_1_ring_is_the_mesh_layer():
    faces, lbs, (tok, d, mask, verts, assoc, prev48) = scene()
    torch.manual_seed(1)
    layer, edge = XYZMeshLayer(128, 32), torch.nn.Linear(4, 32)
    ball1 = M.BallIndex(faces, V, hops=1)
    net = XYZMeshNet(lbs, faces)
    H = torch.randn(2, V, 128)
    O = (torch.rand(2, V) < 0.4).float()
    H = H * O.unsqueeze(-1)
    gemb = torch.relu(edge(net.edge_geometry(verts)))
    with torch.no_grad():
        ref = layer(H, O, gemb, net.ring_idx, net.ring_mask)
        got = M.ball_round(layer, edge, H, O, verts, ball1, expand=True)
        ref_np = M.fixed_round(layer, H, O, gemb, net.ring_idx, net.ring_mask)
        got_np = M.ball_round(layer, edge, H, O, verts, ball1, expand=False)
    torch.testing.assert_close(got[0], ref[0], rtol=0, atol=1e-6)
    assert torch.equal(got[1], ref[1])
    torch.testing.assert_close(got_np[0], ref_np[0], rtol=0, atol=1e-6)
    assert torch.equal(got_np[1], O)


def test_ball_round_gradient_through_checkpoint_equals_plain():
    faces, _, (_, _, _, verts, _, _) = scene()
    torch.manual_seed(2)
    layer, edge = XYZMeshLayer(128, 32), torch.nn.Linear(4, 32)
    ball = M.BallIndex(faces, V)
    O = (torch.rand(2, V) < 0.3).float()
    H = (torch.randn(2, V, 128) * O.unsqueeze(-1)).requires_grad_(True)
    out, _ = M.ball_round(layer, edge, H, O, verts, ball, expand=False)
    gH, gW = torch.autograd.grad(out.square().sum(), (H, layer.gate.weight))
    # plain (no checkpoint) dense reference on the padded ball
    idx4, m4 = M.ball_table(faces, V, 4)
    H2 = H.detach().clone().requires_grad_(True)
    vf = verts.float()
    K = idx4.shape[1]
    dv = vf[:, idx4.reshape(-1)].view(2, V, K, 3) - vf.unsqueeze(2)
    g = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], -1) / M.L_EDGE * m4.view(1, V, K, 1)
    out2, _ = M.fixed_round(layer, H2, O, torch.relu(edge(g)), idx4, m4)
    gH2, gW2 = torch.autograd.grad(out2.square().sum(), (H2, layer.gate.weight))
    torch.testing.assert_close(out, out2, rtol=0, atol=1e-5)
    torch.testing.assert_close(gH, gH2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gW, gW2, rtol=1e-5, atol=1e-4)


def test_local_attention_equals_dense_masked_attention():
    torch.manual_seed(4)
    faces, lbs, _ = mano()
    ridx, rmask = mesh_ring(faces, V)
    keys, kmask = M.local_keys(ridx, rmask)
    blk = M.LocalResidual().eval()
    H = torch.randn(2, V, 128)
    S = (torch.rand(2, V) < 0.6).float()
    H = H * S.unsqueeze(-1)
    with torch.no_grad():
        got = blk(H, S, lbs, keys, kmask)
        # dense reference: key j usable by query i iff j in ring(i) and supported, or j == i
        allowed = torch.zeros(V, V, dtype=torch.bool)
        allowed[torch.arange(V).unsqueeze(1).expand_as(ridx)[rmask > 0], ridx[rmask > 0]] = True
        sup = S > 0
        ok = allowed.unsqueeze(0) & sup.unsqueeze(1) | torch.eye(V, dtype=torch.bool).unsqueeze(0)
        value = blk.norm1(H)
        query = value + blk.identity(lbs).unsqueeze(0)
        upd, _ = blk.attention(query, query, value, attn_mask=(~ok).repeat_interleave(4, 0), need_weights=False)
        live = sup.unsqueeze(-1).float()
        ref = (H + upd * live) * live
        ref = (ref + blk.ff(blk.norm2(ref)) * live) * live
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------- the model variant
def test_legacy_variant_is_c1_bit_for_bit():
    base, var = activate(build()), activate(build("legacy"))
    sb, sv = base.state_dict(), var.state_dict()
    assert sb.keys() == sv.keys() and all(torch.equal(sb[k], sv[k]) for k in sb)
    x = scene()[2]
    db, _, ab = run(base, x)
    dv, _, av = run(var, x)
    assert torch.equal(db, dv)
    for k in ("H0", "H3", "O", "S", "slots", "bslot"):
        assert torch.equal(ab[k], av[k]), k
    base.train(), var.train()
    torch.use_deterministic_algorithms(True)                # CPU scatter backward is otherwise thread-order noisy
    try:
        gb = torch.autograd.grad(base(*x)[0].square().sum(), list(base.parameters()), allow_unused=True)
        gv = torch.autograd.grad(var(*x)[0].square().sum(), list(var.parameters()), allow_unused=True)
    finally:
        torch.use_deterministic_algorithms(False)
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(gb, gv))


@pytest.mark.parametrize("rf", ["self3", "noprop", "ring4np", "global_loc"])
def test_variants_keep_every_c1_initial_value(rf):
    base, var = build(), build(rf)
    sb, sv = base.state_dict(), var.state_dict()
    for k, v in sb.items():
        assert k in sv and torch.equal(v, sv[k]), k
    extra = set(sv) - set(sb)
    if rf == "global_loc":
        assert extra and all(k.startswith("mrf_global.") for k in extra)
        fplus = build(xag_gnn="global_residual").state_dict()
        for k in extra:
            assert torch.equal(sv[k], fplus["xag_global." + k[len("mrf_global."):]]), k
    else:
        assert not extra


@pytest.mark.parametrize("rf", ["self3", "noprop", "ring4np", "global_loc"])
def test_variants_run_and_hold_or_grow_support_as_claimed(rf):
    m = activate(build(rf))
    delta, stats, aux = run(m)
    assert torch.isfinite(delta).all() and delta.abs().sum() > 0
    if rf == "global_loc":
        c1 = activate(build())
        assert torch.equal(aux["S"], run(c1)[2]["S"]), "global_loc keeps C1's support expansion"
        assert (aux["S"] >= aux["O"]).all() and (aux["S"] > aux["O"]).any()
    else:
        assert torch.equal(aux["S"], aux["O"]), "no inpainting"
        assert (aux["H3"][aux["O"] == 0] == 0).all()


def test_self3_is_per_vertex_and_freezes_the_unused_maps():
    m = activate(build("self3"))
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {"edge.weight", "edge.bias"} | {f"layers.{i}.{a}.{b}" for i in range(3)
                                                     for a, b in (("msg", "weight"), ("gate", "weight"),
                                                                  ("gate", "bias"))}
    _, _, aux = run(m)
    with torch.no_grad():
        for layer in m.layers:
            layer.msg.weight.mul_(3.0)
            layer.gate.bias.add_(1.0)
        m.edge.weight.mul_(-2.0)
    _, _, aux2 = run(m)
    assert torch.equal(aux["H3"], aux2["H3"])
    # per-vertex: the Jacobian of H3 w.r.t. H0 is block diagonal
    H0 = aux["H0"][:1].clone().requires_grad_(True)
    O = aux["O"][:1]
    H3, _ = M.run_g("self3", H0, O, layers=m.layers, edge=m.edge)
    j = int(O[0].nonzero()[0])
    g, = torch.autograd.grad(H3[0, j].sum(), H0)
    others = torch.ones(V, dtype=torch.bool)
    others[j] = False
    assert (g[0, others] == 0).all() and g[0, j].abs().sum() > 0


@pytest.mark.parametrize("variant,hops", [("legacy", 3), ("noprop", 3), ("ring4np", 12), ("ring4", 12),
                                          ("diffuse", 3), ("none", 0), ("self3", 0)])
def test_receptive_field_in_hops(variant, hops):
    faces, lbs, (_, _, _, verts, _, _) = scene(B=1)
    D = torch.from_numpy(bfs_hops(faces))
    torch.manual_seed(5)
    layers = torch.nn.ModuleList([XYZMeshLayer(128, 32) for _ in range(3)])
    edge = torch.nn.Linear(4, 32)
    net = XYZMeshNet(lbs, faces)
    ball = M.BallIndex(faces, V)
    O = torch.ones(1, V)                                                       # full support: pure topology
    H0 = torch.randn(1, V, 128).requires_grad_(True)
    H3, _ = M.run_g(variant, H0, O, layers=layers, edge=edge, edge_g=net.edge_geometry(verts), verts=verts,
                    ring_idx=net.ring_idx, ring_mask=net.ring_mask, ball=ball)
    for j in (0, 400):
        g, = torch.autograd.grad(H3[0, j].sum(), H0, retain_graph=True)
        reach = g[0].abs().sum(-1) > 0
        assert not reach[D[j] > hops].any()
        if variant not in ("diffuse",):
            assert reach[D[j] <= hops].all(), (variant, j)


def test_noprop_messages_only_between_supported_vertices():
    faces, lbs, (_, _, _, verts, _, _) = scene(B=1)
    torch.manual_seed(7)
    layers = torch.nn.ModuleList([XYZMeshLayer(128, 32) for _ in range(3)])
    edge = torch.nn.Linear(4, 32)
    net = XYZMeshNet(lbs, faces)
    O = (torch.rand(1, V) < 0.4).float()
    H0 = (torch.randn(1, V, 128) * O.unsqueeze(-1)).requires_grad_(True)
    H3, S = M.run_g("noprop", H0, O, layers=layers, edge=edge, edge_g=net.edge_geometry(verts),
                    ring_idx=net.ring_idx, ring_mask=net.ring_mask)
    assert torch.equal(S, O) and (H3[0, O[0] == 0] == 0).all()
    j = int(O[0].nonzero()[3])
    g, = torch.autograd.grad(H3[0, j].sum(), H0)
    assert (g[0, O[0] == 0] == 0).all()


def test_unknown_or_incompatible_variant_fails_closed():
    faces, lbs, _ = scene()
    with pytest.raises(ValueError):
        M.MRFXYZMeshNet(lbs, faces, rf="nope")
    with pytest.raises(ValueError):
        M.MRFXYZMeshNet(lbs, faces, rf="noprop", geom=False)
    with pytest.raises(ValueError):
        M.MRFXYZMeshNet(lbs, faces, rf="noprop", xag_gnn="global_residual")
    with pytest.raises(ValueError):
        M.MRFXYZMeshNet(lbs, faces, rf="noprop", contour=True)
    assert M.RF_MODES[0] == "legacy"


WIRING_CHECK = r"""
import copy, sys, torch
sys.path[:0] = [sys.argv[1], sys.argv[2]]
from config import load_config
from model import MNISTModel
from semkine.mrf import MRFXYZMeshNet
from semkine.xyz_mesh import XYZMeshNet
base = load_config(sys.argv[2] + "/configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml")

def mk(extra):
    cfg = copy.deepcopy(base)
    cfg["MODEL"].update(extra)
    torch.manual_seed(3407)
    m = MNISTModel(cfg)
    return m, torch.rand(8)

a, ra = mk({})
b, rb = mk({"XYZ_RF": "legacy"})
assert type(a.xyz) is XYZMeshNet and type(b.xyz) is XYZMeshNet and torch.equal(ra, rb)
sa, sb = a.state_dict(), b.state_dict()
assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
for rf in ("self3", "noprop", "ring4np", "global_loc"):
    c, rc = mk({"XYZ_RF": rf})
    assert type(c.xyz) is MRFXYZMeshNet and c.xyz.rf == rf, rf
    assert torch.equal(rc, ra), "random stream after construction is C1's"
    sc = c.state_dict()
    assert all(torch.equal(sa[k], sc[k]) for k in sa), rf
for bad in ({"XYZ_RF": "bogus"}, {"XYZ_RF": "noprop", "XYZ_AGG": "mass"},
            {"XYZ_RF": "noprop", "XAG_GNN": "global_residual"}):
    try:
        mk(bad)
    except ValueError:
        continue
    raise SystemExit("accepted " + repr(bad))
print("WIRING OK")
"""


def test_model_key_wiring_default_bit_identical_and_dispatch(tmp_path):
    """On the wired model/model.py (a wired copy while AGG778 R2 still freezes the original)."""
    import shutil
    import subprocess
    from tools.mrf import wiring as W
    text = W.MODEL_PY.read_text()
    model_dir = tmp_path / "model"
    shutil.copytree(ROOT / "model", model_dir, ignore=shutil.ignore_patterns("__pycache__"))
    if '"XYZ_RF"' not in text:
        (model_dir / "model.py").write_text(W.wired(text))
    r = subprocess.run([sys.executable, "-c", WIRING_CHECK, str(model_dir), str(ROOT)], cwd=ROOT,
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                                             "OMP_NUM_THREADS": "4"})
    assert r.returncode == 0 and "WIRING OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]


# --------------------------------------------------------------------------- probe tail
def _tail(variant, f_in=3, seed=0):
    from tools.mrf.heads import MRFTail
    faces, lbs, _ = scene()
    xyz = XYZMeshNet(lbs, faces)
    torch.manual_seed(seed)
    return MRFTail(f_in, xyz, variant, faces=faces)


def test_probe_tail_legacy_is_agg778_tail_and_common_initial_values_are_shared():
    from tools.agg778.heads import Tail
    faces, lbs, (_, _, _, verts, _, _) = scene()
    xyz = XYZMeshNet(lbs, faces)
    torch.manual_seed(0)
    ref = Tail(3, xyz)
    leg = _tail("legacy")
    assert all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), leg.state_dict().values()))
    for v in ("self3", "noprop", "ring4np", "ring4", "global", "global_loc", "none", "diffuse", "nogeo"):
        sv = _tail(v).state_dict()
        for k, x in ref.state_dict().items():
            assert torch.equal(sv[k], x), (v, k)
    sg, sl = _tail("global").state_dict(), _tail("global_loc").state_dict()
    assert all(torch.equal(sg[k], sl[k]) for k in sg if k.startswith("blocks."))
    with torch.no_grad():
        ref.fc2.weight.normal_(0, .02)
        leg.fc2.weight.copy_(ref.fc2.weight)
    O = (torch.rand(2, V) < 0.4).float()
    f = torch.randn(2, V, 3) * O.unsqueeze(-1)
    e = xyz.edge_geometry(verts)
    bs, p48 = torch.zeros(2, 258), torch.zeros(2, 48)
    assert torch.equal(ref(f, O, e, bs, p48), leg(f, O, e, bs, p48, verts))


@pytest.mark.parametrize("variant", ["self3", "noprop", "ring4np", "ring4", "global", "global_loc", "none",
                                     "diffuse", "nogeo"])
def test_probe_tail_variants_train(variant):
    faces, lbs, (_, _, _, verts, _, _) = scene()
    t = _tail(variant)
    with torch.no_grad():
        t.fc2.weight.normal_(0, .02)
    O = (torch.rand(2, V) < 0.4).float()
    f = torch.randn(2, V, 3) * O.unsqueeze(-1)
    e = XYZMeshNet(lbs, faces).edge_geometry(verts)
    out = t(f, O, e, torch.zeros(2, 258), torch.zeros(2, 48), verts)
    assert out.shape == (2, 51) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert t.inp.weight.grad is not None and torch.isfinite(t.inp.weight.grad).all()
