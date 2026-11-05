r"""PFO trunk variants (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103 and its phase-2 sections): opt-in subclasses of S45's
`XYZMeshNet`, selected by `install(variant)` before the model is built (`model/model.py` imports `XYZMeshNet` from
`semkine.xyz_mesh` at construction time). `uninstall()` restores the native class.

Every variant first runs the native constructor unchanged (the same random draws, so the parameters both networks
have start equal at the same seed); anything it adds is built inside an isolated RNG scope.

  base     the native network (a subclass with no change; for uniform bookkeeping)
  nomp     no message passing: the three MeshGNN rounds and the edge embedding are removed; H3 := H0, S := O
  self     the same parameters and MACs as base, no information between vertices: every vertex's 1-ring is replaced by
           K copies of itself (edge vectors 0), so the rounds are per-vertex residual MLPs and S stays O
  rewire   the same parameters and MACs as base on a random graph with every vertex's degree kept: the undirected
           1-ring edge set rewired by double-edge swaps (Maslov-Sneppen, 10 x |E| accepted swaps, fixed seed); edge
           vectors are those of the random vertex pairs
"""
from __future__ import annotations

import numpy as np
import torch

from . import xyz_mesh as XM

VARIANTS = ("base", "nomp", "self", "rewire")
REWIRE_SEED = 20261103
_NATIVE = {"cls": XM.XYZMeshNet}
_STATE = {"variant": None}


def rewired_ring(ring_idx: torch.Tensor, ring_mask: torch.Tensor, seed: int = REWIRE_SEED, swaps_per_edge: int = 10):
    """Degree-preserving random rewiring of the undirected graph given by a (V, K) ring table."""
    V, K = ring_idx.shape
    rows, cols = np.nonzero(ring_mask.numpy() > 0)
    nbr = ring_idx.numpy()[rows, cols]
    edges = {(int(min(a, b)), int(max(a, b))) for a, b in zip(rows, nbr) if a != b}
    E = sorted(edges)
    eset = set(E)
    rng = np.random.default_rng(seed)
    target, done, tries = swaps_per_edge * len(E), 0, 0
    while done < target and tries < 100 * target:
        tries += 1
        i, j = rng.integers(len(E), size=2)
        if i == j:
            continue
        (a, b), (c, d) = E[i], E[j]
        if rng.random() < 0.5:
            c, d = d, c
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if a == d or c == b or e1 in eset or e2 in eset or e1 == e2:
            continue
        eset.discard(E[i]); eset.discard(E[j])
        eset.add(e1); eset.add(e2)
        E[i], E[j] = e1, e2
        done += 1
    deg = np.zeros(V, np.int64)
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    old_deg = (ring_mask.numpy() > 0).sum(1)
    if not np.array_equal(deg, old_deg):
        raise RuntimeError("rewiring changed a vertex degree")
    idx = np.repeat(np.arange(V)[:, None], K, 1)
    mask = np.zeros((V, K), np.float32)
    fill = np.zeros(V, np.int64)
    for a, b in sorted(E):
        idx[a, fill[a]], mask[a, fill[a]] = b, 1.0
        fill[a] += 1
        idx[b, fill[b]], mask[b, fill[b]] = a, 1.0
        fill[b] += 1
    return torch.from_numpy(idx), torch.from_numpy(mask), {"swaps": done, "edges": len(E)}


class PFOMeshNet(XM.XYZMeshNet):
    variant = "base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        v = self.variant
        if v == "nomp":
            # the rounds and the edge embedding go; the native forward then embeds a dummy (B, 1, 1, 4) edge tensor
            # through a parameter-free identity and runs no round
            del self.layers
            del self.edge
            self.layers = torch.nn.ModuleList()
            self.edge = torch.nn.Identity()
        elif v == "self":
            V, K = self.ring_idx.shape
            self.ring_idx = torch.arange(V).unsqueeze(1).repeat(1, K)
        elif v == "rewire":
            idx, mask, info = rewired_ring(self.ring_idx.cpu(), self.ring_mask.cpu())
            self.ring_idx, self.ring_mask = idx, mask
            self.rewire_info = info
        elif v != "base":
            raise ValueError(v)

    @torch.no_grad()
    def edge_geometry(self, verts):
        if self.variant == "nomp":
            return verts.new_zeros((verts.shape[0], 1, 1, 4))
        return super().edge_geometry(verts)


def install(variant):
    if variant not in VARIANTS:
        raise ValueError(f"variant in {VARIANTS}")
    if XM.XYZMeshNet is not _NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PFOMeshNet_{variant}", (PFOMeshNet,), {"variant": variant})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = _NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PFOMeshNet", "VARIANTS", "install", "installed", "rewired_ring", "uninstall"]
