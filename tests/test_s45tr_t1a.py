"""S45-TR T1a (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): the node features equal EventGNN's own for the same
stride-sampled events; the state dict and random stream are S45's except the registered relation-encoder weight and
the new event-graph tensors; the record really reads the graph."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_t1a as T  # noqa: E402
from semkine import xyz_mesh as X  # noqa: E402
from semkine.encoder import event_tokens  # noqa: E402
from semkine.xyz_mesh import associate_rays, stride_sample, vertex_face_table  # noqa: E402
from tools.s45tr.common import T1A_GNN  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"


def mano():
    with np.load(NPZ, allow_pickle=False) as z:
        return (torch.tensor(np.asarray(z["f"], np.int64)), torch.tensor(np.asarray(z["weights"], np.float32)),
                torch.tensor(np.asarray(z["v_template"], np.float32)))


def events(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    counts = [2500, 700, 0][:B]
    rows, ptr = [], [0]
    for b, n in enumerate(counts):
        t = torch.sort(torch.rand(n, generator=g) * 0.05).values
        rows.append(torch.stack([torch.full((n,), float(b)),
                                 torch.randint(0, 240, (n,), generator=g).float(),
                                 torch.randint(0, 180, (n,), generator=g).float(), t,
                                 torch.randint(0, 2, (n,), generator=g).float()], -1))
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((B,), 0.05)


def test_node_features_equal_eventgnn_forward():
    ev, ptr, dt = events()
    T.T1aXYZMeshNet.GNN = dict(T1A_GNN)
    faces, lbs, _ = mano()
    torch.manual_seed(1)
    net = T.T1aXYZMeshNet(lbs, faces).eval()
    tok = event_tokens(ev, ptr, dt, 180, 240)
    src, mask = stride_sample(ptr, 2048)
    B = ptr.numel() - 1
    sampled = tok[src.reshape(-1)].view(B, -1, tok.shape[-1]) * mask.unsqueeze(-1)
    with torch.no_grad():
        mine = T.node_features(net.evgnn, sampled, mask)
        _, ref, *_ , ref_mask = net.evgnn(ev, ptr, dt, return_nodes=True)
    N = mine.shape[1]
    assert torch.equal(ref_mask[:, :N], mask) and not ref_mask[:, N:].any()
    assert torch.allclose(mine, ref[:, :N], atol=1e-6, rtol=0)


def test_state_dict_and_stream_are_s45s_except_registered():
    faces, lbs, _ = mano()
    T.T1aXYZMeshNet.GNN = dict(T1A_GNN)
    torch.manual_seed(3407)
    a = X.XYZMeshNet(lbs, faces)
    after_a = torch.randn(5)
    torch.manual_seed(3407)
    b = T.T1aXYZMeshNet(lbs, faces)
    after_b = torch.randn(5)
    assert torch.equal(after_a, after_b)
    sa, sb = a.state_dict(), b.state_dict()
    for k, v in sa.items():
        if k == "rel.0.weight":
            assert sb[k].shape == (v.shape[0], T1A_GNN["HIDDEN"] + 6)
        else:
            assert torch.equal(v, sb[k]), k
    assert all(k.startswith("evgnn.") for k in set(sb) - set(sa)) and set(sa) <= set(sb)


def test_record_reads_the_graph():
    faces, lbs, v = mano()
    T.T1aXYZMeshNet.GNN = dict(T1A_GNN)
    torch.manual_seed(3407)
    net = T.T1aXYZMeshNet(lbs, faces).eval()
    with torch.no_grad():
        net.fc2.weight.normal_(0, 0.05)
    B, N = 1, 256
    g = torch.Generator().manual_seed(0)
    verts = (v + torch.tensor([0.0, 0.0, 0.5])).unsqueeze(0)
    pick = torch.randint(0, 778, (B, N), generator=g)
    d = verts[0, pick] / verts[0, pick][..., 2:3]
    mask = torch.ones(B, N, dtype=torch.bool)
    assoc = associate_rays(d, mask, verts, faces, vertex_face_table(faces, 778), torch.full((B,), 16 / 300.0),
                           torch.full((B,), 3 / 300.0), 0.01)
    tok = torch.rand(B, N, 7, generator=g)
    tok[..., 3] = torch.linspace(0, 1, N)
    prev48 = torch.zeros(B, 48)
    with torch.no_grad():
        out = net(tok, d, mask, verts, assoc, prev48)[0]
        tok2 = tok.clone()
        tok2[0, 100, 4:] += 1.0           # change one event's attributes: its graph neighbours' features move too
        h1 = T.node_features(net.evgnn, tok, mask)
        h2 = T.node_features(net.evgnn, tok2, mask)
        out2 = net(tok2, d, mask, verts, assoc, prev48)[0]
    moved = (h1 - h2).abs().sum(-1)[0]
    assert moved[100] > 0 and int((moved[101:] > 0).sum()) > 0 and torch.count_nonzero(moved[:100]) == 0
    assert not torch.equal(out, out2)


def test_install_routes_constructor():
    saved = X.XYZMeshNet
    try:
        T.install(T1A_GNN)
        assert T.installed() and X.XYZMeshNet is T.T1aXYZMeshNet
    finally:
        X.XYZMeshNet = saved
