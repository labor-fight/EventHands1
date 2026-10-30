"""AURA association variants (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): `c1` is the library association bit
for bit, every variant is what it claims (band, metric band, metric shell, 2D proximity, visibility, binding nulls,
occupancy, joint binning, shell nodes), the probe modules share AGG778's initial values, `MODEL.XYZ_AURA: shell` keeps
every C1 initial value, and the model wiring is C1 bit for bit when absent and dispatches every variant."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from semkine import aura as AU  # noqa: E402
from semkine.xyz_mesh import XYZMeshNet, associate_rays, mesh_ring, vertex_face_table  # noqa: E402

V = 778
F_PX = 226.2


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


def scene(B=3, N=400, seed=0):
    """Posed copies of the template at z ~ 0.5 m; events aimed at vertices (hits / near), displaced off the
    silhouette by up to ~40 px (near / none), and padded rows."""
    faces, lbs, v = mano()
    g = torch.Generator().manual_seed(seed)
    verts = []
    for _ in range(B):
        a = 0.6 * torch.randn(3, generator=g)
        th = a.norm()
        k = a / th
        Kx = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = torch.eye(3) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
        verts.append((v - v.mean(0)) @ R.T + torch.tensor([0.0, 0.0, 0.5]) + 0.01 * torch.randn(3, generator=g))
    verts = torch.stack(verts)
    pick = torch.randint(0, V, (B, N), generator=g)
    target = verts[torch.arange(B)[:, None], pick]
    off = torch.randn(B, N, 3, generator=g) * torch.linspace(0.0, 0.04, N).view(1, N, 1)
    target = target + off
    d = target / target[..., 2:3]
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -20:] = False
    f = torch.full((B,), F_PX)
    intr = (f, f, torch.full((B,), 120.0), torch.full((B,), 90.0))
    return faces, lbs, verts, d, mask, intr


def c1_lib(faces, verts, d, mask):
    """The library call exactly as model/model.py makes it (band = 16.0 / f_px in float32)."""
    B = d.shape[0]
    f = torch.full((B,), F_PX)
    f_px = (f * f).sqrt()
    return associate_rays(d, mask, verts, faces, vertex_face_table(faces, V), band=16.0 / f_px,
                          front_tol=3.0 / f_px, z_tol=0.01, k=16)


def assoc(rule, sc, **kw):
    faces, lbs, verts, d, mask, intr = sc
    return AU.associate(rule, d, mask, verts, faces, vertex_face_table(faces, V), intr, **kw)


# --------------------------------------------------------------------------- association rules
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_c1_is_the_library_association_bit_for_bit(seed):
    sc = scene(seed=seed)
    lib = c1_lib(sc[0], sc[2], sc[3], sc[4])
    mine = assoc("c1", sc)
    for k in lib:
        assert torch.equal(lib[k], mine[k]), k
    assert lib["hit"].any() and (lib["valid"] & ~lib["hit"]).any() and (~lib["valid"] & sc[4]).any()


@pytest.mark.parametrize("rule", AU.RULES)
def test_every_rule_conserves_mass_and_is_well_formed(rule):
    sc = scene()
    a = assoc(rule, sc)
    mask = sc[4]
    w = a["q"].unsqueeze(-1) * a["A"]
    tot = w.sum(-1) + (1 - a["q"])
    assert torch.allclose(tot[mask], torch.ones_like(tot[mask]), atol=1e-5)
    assert (a["q"][~a["valid"]] == 0).all() and (a["q"] >= 0).all() and (a["q"] <= 1 + 1e-6).all()
    assert (a["z"][a["valid"]] > 0).all() and not (a["valid"] & ~mask).any()
    s = a["A"].sum(-1)[a["valid"]]
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)
    if rule not in ("p2d", "p2dvis"):
        assert (a["q"][a["hit"]] == 1).all() or rule == "p2dh"


def test_surface_only_and_band_dose():
    sc = scene()
    c1, surf = assoc("c1", sc), assoc("surf", sc)
    assert torch.equal(surf["valid"], c1["hit"]) and torch.equal(surf["hit"], c1["hit"])
    for k in ("vid", "A", "q", "z"):
        assert torch.equal(surf[k][c1["hit"]], c1[k][c1["hit"]]), k
    b8, b32 = assoc("b8", sc), assoc("b32", sc)
    assert not (b8["valid"] & ~c1["valid"]).any() and not (c1["valid"] & ~b32["valid"]).any()
    for a in (b8, b32):
        assert torch.equal(a["hit"], c1["hit"])
        for k in ("vid", "A", "z"):
            assert torch.equal(a[k][c1["hit"]], c1[k][c1["hit"]]), k


def _perp(verts, d, vid):
    Vr = verts[torch.arange(verts.shape[0]).view(-1, 1, 1), vid]                     # (B, N, R, 3)
    dd = d.unsqueeze(2)
    zj = (Vr * dd).sum(-1) / (dd * dd).sum(-1)
    return (Vr - zj.unsqueeze(-1) * dd).norm(dim=-1)


@pytest.mark.parametrize("rule,r", [("m10", 0.01), ("m20", 0.02), ("mM", AU.R_MATCH)])
def test_metric_band_members_lie_within_the_radius(rule, r):
    sc = scene()
    faces, lbs, verts, d, mask, intr = sc
    a = assoc(rule, sc)
    near = a["valid"] & ~a["hit"]
    assert near.any()
    p = _perp(verts, d, a["vid"])
    w = a["A"] > 0
    assert (p[near.unsqueeze(-1) & w] <= r + 1e-6).all()
    c1 = assoc("c1", sc)
    assert torch.equal(a["hit"], c1["hit"])


@pytest.mark.parametrize("rule,r", [("aura10", 0.01), ("aura20", 0.02)])
def test_metric_shell_distance_is_the_ray_to_candidate_face_minimum(rule, r):
    sc = scene(B=2, N=240)
    faces, lbs, verts, d, mask, intr = sc
    a = assoc(rule, sc)
    shell = a["valid"] & ~a["hit"]
    assert shell.any()
    # the record's foot point (sum A V) and the closest ray point (z d): their distance is delta, q = 1 - (delta/r)^2
    Vr = verts[torch.arange(2).view(-1, 1, 1), a["vid"]]
    foot = (a["A"].unsqueeze(-1) * Vr).sum(2)
    X = a["z"].unsqueeze(-1) * d
    delta = (X - foot).norm(dim=-1)
    assert torch.allclose(a["q"][shell], (1 - (delta[shell] / r) ** 2).clamp(min=0), atol=2e-4)
    assert (delta[shell] <= r + 1e-6).all()
    # brute force over the same candidate faces (dense edge and interior sampling): nothing closer than delta
    vf = vertex_face_table(faces, V)
    ok = 0
    for b, n in shell.nonzero()[:40].tolist():
        dd = d[b, n]
        vu, vv = verts[b, :, 0] / verts[b, :, 2], verts[b, :, 1] / verts[b, :, 2]
        nid = ((vu - dd[0]) ** 2 + (vv - dd[1]) ** 2).topk(16, largest=False).indices
        cf = vf[nid].reshape(-1)
        cf = cf[cf >= 0].unique()
        P = verts[b][faces[cf]].double()                                            # (C, 3, 3)
        s = torch.linspace(0, 1, 401, dtype=torch.float64)
        g1, g2 = torch.meshgrid(s, s, indexing="ij")
        keep = (g1 + g2) <= 1
        w1, w2 = g1[keep], g2[keep]
        pts = (P[:, 0:1] * (1 - w1 - w2).view(1, -1, 1) + P[:, 1:2] * w1.view(1, -1, 1)
               + P[:, 2:3] * w2.view(1, -1, 1)).reshape(-1, 3)
        u = dd.double() / dd.double().norm()
        dist = (pts - (pts @ u).unsqueeze(-1) * u).norm(dim=-1).min()
        assert float(dist) >= float(delta[b, n]) - 2e-5 and float(dist) <= float(delta[b, n]) + 5e-5
        ok += 1
    assert ok > 5


def test_visibility_copy_is_evflow_visible_and_p2d_rules():
    from s44 import evflow as EF
    sc = scene()
    faces, lbs, verts, d, mask, intr = sc
    vis = AU.visible(verts, intr)
    uv = AU.project(verts, intr)
    for b in range(verts.shape[0]):
        assert torch.equal(vis[b], EF.visible(uv[b], verts[b, :, 2]))
    assert vis.any() and (~vis).any()
    p, pv, ph, c1 = assoc("p2d", sc), assoc("p2dvis", sc), assoc("p2dh", sc), assoc("c1", sc)
    assert not p["hit"].any() and not pv["hit"].any()
    for k in ("vid", "A", "q", "z", "valid"):
        assert torch.equal(pv[k], ph[k]), k
    assert torch.equal(ph["hit"], c1["hit"] & pv["valid"])
    used = pv["A"] > 0
    assert vis[torch.arange(3).view(-1, 1, 1).expand_as(pv["vid"])[used], pv["vid"][used]].all()
    assert not vis[torch.arange(3).view(-1, 1, 1).expand_as(p["vid"])[p["A"] > 0], p["vid"][p["A"] > 0]].all()
    v = assoc("vis", sc)
    assert torch.equal(v["hit"], c1["hit"])
    nr = v["valid"] & ~v["hit"]
    assert vis[torch.arange(3).view(-1, 1, 1).expand_as(v["vid"])[(v["A"] > 0) & nr.unsqueeze(-1)],
               v["vid"][(v["A"] > 0) & nr.unsqueeze(-1)]].all()


# --------------------------------------------------------------------------- binding nulls and helpers
def _multiset(a, mask, b):
    w = a["q"].unsqueeze(-1) * a["A"] * mask.float().unsqueeze(-1)
    keep = w[b] > 0
    pairs = torch.stack([a["vid"][b][keep].double(), w[b][keep].double()], -1)
    return pairs[torch.argsort(pairs[:, 0] * 10 + pairs[:, 1])]


@pytest.mark.parametrize("by_part", [False, True])
def test_shuffle_preserves_support_and_masses_and_uses_no_random_state(by_part):
    sc = scene()
    faces, lbs, verts, d, mask, intr = sc
    c1 = assoc("c1", sc)
    seeds = torch.tensor([11, 22, 33])
    part = lbs.argmax(1) if by_part else None
    state = torch.random.get_rng_state()
    s1, info = AU.shuffle_assoc(c1, d, mask, verts, seeds, part)
    assert torch.equal(state, torch.random.get_rng_state())
    s2, _ = AU.shuffle_assoc(c1, d, mask, verts, seeds, part)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    for b in range(3):
        assert torch.allclose(_multiset(c1, mask, b), _multiset(s1, mask, b), atol=1e-6)
    h0, n0 = AU.occupancy(c1, mask, V)
    h1, n1 = AU.occupancy(s1, mask, V)
    assert torch.allclose(h0, h1, atol=1e-5) and torch.allclose(n0, n1, atol=1e-5)
    el, p = info["eligible"], info["partner"]
    assert (p[~el] == -1).all() and (p[el] >= 0).all()
    moved = el & (p != torch.arange(p.shape[1]).view(1, -1))
    assert moved.float().mean() > 0.5
    if by_part:
        w = c1["q"].unsqueeze(-1) * c1["A"]
        main = c1["vid"].gather(2, w.argmax(-1, keepdim=True)).squeeze(-1)
        pm = main.gather(1, p.clamp_min(0))
        assert torch.equal(part[main][el], part[pm][el])
    for k in ("vid", "A", "q", "z", "valid", "hit"):
        assert torch.equal(s1[k][~el], c1[k][~el]), k


def test_none_occupancy_joints_and_shell_tables():
    sc = scene()
    faces, lbs, verts, d, mask, intr = sc
    c1 = assoc("c1", sc)
    nn_ = AU.none_assoc(c1)
    h, n = AU.occupancy(nn_, mask, V)
    assert (h == 0).all() and (n == 0).all() and (nn_["q"] == 0).all()
    h, n = AU.occupancy(c1, mask, V)
    w = c1["q"].unsqueeze(-1) * c1["A"] * mask.float().unsqueeze(-1)
    assert torch.allclose(h.sum() + n.sum(), w.sum(), atol=1e-3)
    J = torch.einsum("jv,bvc->bjc", AU.joint_weights(lbs), verts)
    ja = AU.joint_assoc(d, mask, J)
    assert torch.equal(ja["valid"], mask) and (ja["vid"][..., 0] < 16).all() and (ja["vid"][..., 1:] == 0).all()
    idx, em = mesh_ring(faces, V)
    i2, m2 = AU.shell_tables(idx, em)
    assert i2.shape == (2 * V, idx.shape[1] + 1)
    assert torch.equal(i2[:V, -1], torch.arange(V) + V) and torch.equal(i2[V:, -1], torch.arange(V))
    assert torch.equal(i2[V:, :-1], idx + V) and torch.equal(m2[:V, :-1], em) and (m2[:, -1] == 1).all()
    pos = AU.shell_positions(verts, faces)
    gap = (pos[:, V:] - pos[:, :V]).norm(dim=-1)
    assert torch.allclose(gap, torch.full_like(gap, AU.SHELL_R), atol=1e-6)


# --------------------------------------------------------------------------- probe modules
def test_probe_records_pool_and_ops_match_agg778():
    from tools.agg778.heads import RecordOp, Tail, c1_records, pool_records
    from tools.aura import heads as H
    sc = scene()
    faces, lbs, verts, d, mask, intr = sc
    c1 = assoc("c1", sc)
    tok = torch.randn(3, d.shape[1], 7)
    ref = c1_records({"assoc": c1, "d": d, "mask": mask, "verts": verts, "tok": tok})
    mine = H.records(c1, d, mask, verts, tok)
    for k in ("rb", "rn", "vrec", "wrec", "feat"):
        assert torch.equal(ref[k], mine[k]), k
    phi = torch.randn(mine["feat"].shape[0], 128)
    xa, Oa = pool_records(phi, ref["rb"], ref["vrec"], ref["wrec"], 3, "c1")
    xb, Ob, _ = H.pool(phi, mine["rb"], mine["node"], mine["wrec"], 3)
    assert torch.equal(xa, xb) and torch.equal(Oa, Ob)
    torch.manual_seed(5)
    r0 = RecordOp()
    torch.manual_seed(5)
    r1 = H.RecordOpX()
    torch.manual_seed(5)
    r2 = H.RecordOpX(side=1)
    for k, v in r0.state_dict().items():
        assert torch.equal(v, r1.state_dict()[k]) and torch.equal(v, r2.state_dict()[k]), k
    ctr = dict(mine, ctr=torch.ones(mine["feat"].shape[0], 1))
    assert torch.equal(r1(mine, 3)[0], r2(ctr, 3)[0])
    xyz = XYZMeshNet(lbs, faces)
    torch.manual_seed(7)
    t0 = Tail(257, xyz)
    for cls in (H.SplitTail, H.ShellTail, H.JointTail):
        torch.manual_seed(7)
        t1 = cls(257, xyz)
        for k, v in t0.state_dict().items():
            assert torch.equal(v, t1.state_dict()[k]), (cls.__name__, k)
    torch.manual_seed(7)
    ts = H.SplitTail(257, xyz)
    with torch.no_grad():
        ts.fc2.weight.normal_(0, .02)
        t0.fc2.weight.copy_(ts.fc2.weight)
    x, O = r1(mine, 3)
    e = xyz.edge_geometry(verts)
    bs, p48 = torch.zeros(3, 258), torch.zeros(3, 48)
    assert torch.allclose(t0(x, O, e, bs, p48), ts((x, torch.zeros_like(x)), O, e, bs, p48), atol=1e-6)
    (xh, xn), Os = r1(mine, 3, split=True)
    assert torch.equal(Os, O)
    pos = AU.shell_positions(verts, faces)
    rs = H.records(c1, d, mask, verts, tok, shell_pos=pos)
    x2, O2 = r1(rs, 3, Vn=2 * V)
    assert torch.equal(O2[:, :V] + O2[:, V:] > 0, O > 0)
    torch.manual_seed(7)
    tsh = H.ShellTail(257, xyz)
    out = tsh(x2, O2, pos, bs, p48)
    out.square().sum().backward()
    assert out.shape == (3, 51)


# --------------------------------------------------------------------------- D: AuraXYZMeshNet
def test_shell_net_keeps_every_c1_initial_value_and_the_random_stream():
    faces, lbs, verts, d, mask, intr = scene()
    torch.manual_seed(3407)
    base = XYZMeshNet(lbs, faces)
    ra = torch.rand(8)
    torch.manual_seed(3407)
    var = AU.AuraXYZMeshNet(lbs, faces)
    rb = torch.rand(8)
    assert torch.equal(ra, rb)
    sa, sb = base.state_dict(), var.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    with torch.no_grad():
        var.fc2.weight.normal_(0, .02)
    c1 = AU.associate("c1", d, mask, verts, faces, vertex_face_table(faces, V), intr)
    tok = torch.randn(3, d.shape[1], 7)
    var.train()
    delta, stats, aux = var(tok, d, mask, verts, c1, torch.zeros(3, 48))
    assert delta.shape == (3, 51) and torch.isfinite(delta).all()
    assert aux["O"].shape == (3, 2 * V) and (aux["O"][:, V:] > 0).any() and (aux["O"][:, :V] > 0).any()
    delta.square().sum().backward()
    assert all(p.grad is not None for p in var.parameters())
    with pytest.raises(ValueError):
        AU.AuraXYZMeshNet(lbs, faces, aura="none")
    with pytest.raises(ValueError):
        AU.AuraXYZMeshNet(lbs, faces, contour=True)


@pytest.mark.parametrize("mode", [m for m in AU.AURA_MODES if m not in ("legacy", "shell")])
def test_apply_aura_modes(mode):
    from types import SimpleNamespace
    faces, lbs, verts, d, mask, intr = sc = scene()
    c1 = AU.associate("c1", d, mask, verts, faces, vertex_face_table(faces, V), intr)
    batch = SimpleNamespace(sequence_id=torch.tensor([0, 1, 2]), t_end_us=torch.tensor([50000, 100000, 150000]))
    out = AU.apply_aura(mode, c1, d, mask, verts, faces, vertex_face_table(faces, V), intr, batch, lbs.argmax(1))
    assert set(out) == set(c1) and all(out[k].shape == c1[k].shape for k in c1)
    if mode == "none":
        assert (out["q"] == 0).all()
    elif mode != "shuffle":
        ref = assoc(AU.D_RULE[mode], sc, c1=c1)
        assert all(torch.equal(out[k], ref[k]) for k in c1)


WIRING_CHECK = r"""
import copy, sys, torch, numpy as np
sys.path[:0] = [sys.argv[1], sys.argv[2], sys.argv[2] + "/tools"]
from config import load_config
from model import MNISTModel
from semkine.aura import AuraXYZMeshNet
from semkine.xyz_mesh import XYZMeshNet
from tools.agg778 import common as C
torch.set_num_threads(4)
base = load_config(sys.argv[2] + "/configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml")

def mk(extra):
    cfg = copy.deepcopy(base)
    cfg["MODEL"].update(extra)
    torch.manual_seed(3407)
    m = MNISTModel(cfg)
    return m, torch.rand(8)

a, ra = mk({})
b, rb = mk({"XYZ_AURA": "legacy"})
assert type(a.xyz) is XYZMeshNet and type(b.xyz) is XYZMeshNet and torch.equal(ra, rb)
sa = {k: v.clone() for k, v in a.state_dict().items()}                         # initial values (a is re-drawn below)
sb = b.state_dict()
assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
ds = C.Endpoints(C.zgz_seqs(), stride=400)
items = [ds[0], ds[len(ds) - 1]]
batch = C.make_batch(items, np.stack([x["gt_prev"] for x in items]), torch.device("cpu"), with_sources=False)
torch.manual_seed(1)
for p in a.parameters():
    p.data.normal_(0, 0.02)
b.load_state_dict(a.state_dict())
a.eval(); b.eval()
with torch.no_grad():
    oa, ob = a.forward_packet(batch), b.forward_packet(batch)
assert torch.equal(oa, ob), "legacy wiring changed the C1 output"
for mode in ("none", "shuffle", "surf", "b8", "b32", "mmatch", "m10", "aura10", "p2dh", "vis", "shell"):
    c, rc = mk({"XYZ_AURA": mode})
    assert torch.equal(rc, ra), "random stream after construction is C1's: " + mode
    sc = c.state_dict()
    assert all(torch.equal(sa[k], sc[k]) for k in sa) and set(sc) == set(sa), mode
    assert (type(c.xyz) is AuraXYZMeshNet) == (mode == "shell"), mode
    c.load_state_dict(a.state_dict())
    c.eval()
    with torch.no_grad():
        oc = c.forward_packet(batch)
    assert torch.isfinite(oc).all() and not torch.equal(oc, oa), mode
for bad in ({"XYZ_AURA": "bogus"}, {"XYZ_AURA": "none", "XYZ_AGG": "mass"},
            {"XYZ_AURA": "none", "XAG_GNN": "global_residual"}, {"XYZ_AURA": "none", "XYZ_ASSOC_MODE": "contour"}):
    try:
        mk(bad)
    except ValueError:
        continue
    raise SystemExit("accepted " + repr(bad))
print("WIRING OK")
"""


@pytest.mark.parametrize("state", ["r2", "mrf"])
def test_model_key_wiring_default_bit_identical_and_dispatch(tmp_path, state):
    """On an AURA-wired copy of model/model.py from either accepted pre-state (R2's file, or MRF-wired)."""
    import shutil
    import subprocess
    from tools.aura import wiring as W
    from tools.mrf import wiring as MW
    text = W.MODEL_PY.read_text()
    if '"XYZ_AURA"' in text:
        pytest.skip("model/model.py is already AURA-wired")
    if state == "mrf" and '"XYZ_RF"' not in text:
        text = MW.wired(text)
    model_dir = tmp_path / "model"
    shutil.copytree(ROOT / "model", model_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (model_dir / "model.py").write_text(W.wired(text))
    r = subprocess.run([sys.executable, "-c", WIRING_CHECK, str(model_dir), str(ROOT)], cwd=ROOT,
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                                                             "OMP_NUM_THREADS": "4"})
    assert r.returncode == 0 and "WIRING OK" in r.stdout, r.stdout[-2000:] + r.stderr[-4000:]
