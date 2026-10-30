"""S45-TR T0 / T1b (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): T0's nodes are S45's sampled events; its labels
are the XAG sources at those events (template point of face + barycentrics); the projection recovers points on a face;
T1b's predicted sources have the shapes / dtypes the face route reads and only mark live, confident events."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_t0 as T0  # noqa: E402
from semkine import s45tr_t1b as T1B  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine.xyz_mesh import stride_sample  # noqa: E402
from tools.s45tr import common as C  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"


def net():
    verts, faces = T0.template(NPZ)
    torch.manual_seed(0)
    return T0.T0Net(faces, verts, C.T1A_GNN).eval()


def xag_batch():
    from config import load_config
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = load_config(str(C.S45_CONFIG))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(C.XAG_SOURCES)
    components = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", components, train=True)
    return collate_packets([ds[i] for i in (5, 400000, 2000000)])


def test_projection_recovers_points_on_the_face():
    n = net()
    g = torch.Generator().manual_seed(1)
    face = torch.randint(0, n.faces.shape[0], (500,), generator=g)
    bary = torch.rand(500, 3, generator=g)
    bary = bary / bary.sum(-1, keepdim=True)
    tri = n.tmpl[n.faces[face]]
    p = (bary.unsqueeze(-1) * tri).sum(-2)
    normal = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    off = p + 0.002 * normal / normal.norm(dim=-1, keepdim=True)        # 2 mm off the plane
    assert torch.allclose(T0.project_bary(off, tri), bary, atol=1e-4)
    assert torch.allclose(n.template_point(face, bary), p, atol=1e-7)


def test_nodes_labels_and_loss_on_real_packets():
    n = net()
    batch = xag_batch()
    out = n(batch.events, batch.ptr, batch.delta_t_s)
    src, mask = stride_sample(batch.ptr, 2048)
    assert torch.equal(out["src"], src) and torch.equal(out["mask"], mask)
    tgt = T0.targets(n, batch, src, mask)
    s = XAG.sample_sources(batch, src, mask)
    assert torch.equal(tgt["valid"] & ~s["xag_valid"], torch.zeros_like(mask))
    v = tgt["valid"]
    assert int(v.sum()) > 100
    expect = n.template_point(s["xag_face"][v], s["xag_bary"][v])
    assert torch.allclose(tgt["point"][v], expect)
    total, parts = T0.loss(n.train(), out, tgt, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(total) and set(parts) == {"onhand", "face", "part", "coord"}
    total.backward()
    assert all(p.grad is not None for p in n.parameters())


def test_t1b_sources_and_install(tmp_path):
    n = net()
    ckpt = tmp_path / "t0.ckpt"
    torch.save({"state_dict": n.state_dict(), "faces": n.faces, "tmpl": n.tmpl, "gnn": n.gnn_cfg}, ckpt)
    batch = xag_batch()
    src, mask = stride_sample(batch.ptr, 2048)
    try:
        T1B.install(ckpt, 0.5)
        assert XAG.sample_sources is T1B.predicted_sources and T1B.installed()["tau"] == 0.5
        s = XAG.sample_sources(batch, src, mask)
    finally:
        T1B.uninstall()
    assert XAG.sample_sources is T1B._ORIGINAL and T1B.installed() is None
    assert s["xag_face"].shape == mask.shape and s["xag_face"].dtype == torch.int64
    assert s["xag_bary"].shape == (*mask.shape, 3) and s["xag_bary"].dtype == torch.float32
    assert s["xag_valid"].dtype == torch.bool and not bool((s["xag_valid"] & ~mask).any())
    sel = s["xag_valid"]
    assert torch.allclose(s["xag_bary"][sel].sum(-1), torch.ones(int(sel.sum())), atol=1e-5)
