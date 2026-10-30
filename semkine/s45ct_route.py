r"""S45-CT (docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030): corrupted oracle face routes for the correspondence-
tolerance arms, on top of S45-TR T1d (S45 + `MODEL.XAG_ASSOC: face_route`).

Opt-in only: `install(mode, npz)` wraps `semkine.xag.sample_sources` (to keep the sampled events' source rows) and
`semkine.xag.route_association` for this process; with nothing installed T1d's code path is untouched.

Only the events the oracle face route replaces are changed (mask & q > 0 & valid & a known XAG source, i.e.
`route_association`'s own eligibility), and only their (face, barycentrics):
  sigma16  the source face becomes a face at MANO-template geodesic distance in [8, 24] mm (uniform over that ring,
           the face nearest to 16 mm when the ring is empty), barycentrics 1/3 each -- S45-RF P2's noise16
  shuffle  the (face, barycentrics) pairs are permuted among those events within each packet: a cyclic shift in the
           order of a hash of their source rows (every event reassigned when there are two or more, the packet's
           routed face multiset unchanged) -- S45-RF P2's shuffled
Each draw is a deterministic function of the event's original source row (`xag_source_index`), so the same raw event
gets the same corruption whatever the batch, augmentation or epoch; no global random stream is touched.
"""
from __future__ import annotations

import numpy as np
import torch

from . import xag as XAG

MODES = ("sigma16", "shuffle")
RING_M = 0.016
SALT = {"sigma16": 0x5163, "shuffle": 0x5348}
_STATE = {"mode": None, "npz": None, "native_route": None, "native_sample": None, "rows": None, "tables": {},
          "capture": None}


def face_geodesic(tmpl, faces):
    """(F, F) geodesic distance between face centroids on the template (metres): Dijkstra on edge adjacency."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    tmpl, faces = np.asarray(tmpl, np.float64), np.asarray(faces, np.int64)
    cent = tmpl[faces].mean(1)
    edge, rows, cols = {}, [], []
    for f, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            for g in edge.get(key, ()):
                rows.append(f)
                cols.append(g)
            edge.setdefault(key, []).append(f)
    rows, cols = np.asarray(rows), np.asarray(cols)
    w = np.linalg.norm(cent[rows] - cent[cols], axis=1)
    W = sp.coo_matrix((w, (rows, cols)), shape=(len(faces), len(faces))).tocsr()
    return shortest_path(W, directed=False).astype(np.float32)


def ring_table(dist, r_m):
    """Per face, the faces at geodesic distance in [r/2, 3r/2] (else the one nearest to r): padded table + counts."""
    lists = []
    for f in range(len(dist)):
        idx = np.flatnonzero((dist[f] >= 0.5 * r_m) & (dist[f] <= 1.5 * r_m))
        if not len(idx):
            d = np.abs(dist[f] - r_m)
            d[f] = np.inf
            idx = np.array([int(np.argmin(d))])
        lists.append(idx)
    table = np.zeros((len(lists), max(len(x) for x in lists)), np.int64)
    for f, idx in enumerate(lists):
        table[f, :len(idx)] = idx
    return table, np.array([len(x) for x in lists], np.int64)


def hash01(rows, salt):
    """Deterministic uniform [0, 1) per source row (32-bit integer mix, no overflow in int64)."""
    x = (rows.long() & 0xFFFFFFFF) ^ int(salt)
    for _ in range(2):
        x = ((x ^ (x >> 16)) * 0x45D9F3B) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return x.double() / 4294967296.0


def tables(npz, device):
    key = (str(npz), str(device))
    if key not in _STATE["tables"]:
        with np.load(str(npz), allow_pickle=False) as z:
            dist = face_geodesic(z["v_template"], z["f"])
        table, count = ring_table(dist, RING_M)
        _STATE["tables"][key] = (torch.from_numpy(table).to(device), torch.from_numpy(count).to(device))
    return _STATE["tables"][key]


def corrupt(mode, face, bary, eligible, rows, npz):
    """Corrupted copies of `face (B, N)` / `bary (B, N, 3)`; only `eligible` events change."""
    out_face, out_bary = face.clone(), bary.clone()
    if not bool(eligible.any()):
        return out_face, out_bary
    if mode == "sigma16":
        table, count = tables(npz, face.device)
        f = face[eligible].long()
        k = (hash01(rows[eligible], SALT[mode]) * count[f].double()).long().clamp_max(table.shape[1] - 1)
        out_face[eligible] = table[f, k].to(face.dtype)
        out_bary[eligible] = 1.0 / 3.0
    elif mode == "shuffle":
        b, n = eligible.nonzero(as_tuple=True)
        order = torch.argsort(b.double() * 2.0 + hash01(rows[b, n], SALT[mode]))     # by packet, then by hash
        bs = b[order]
        pos = torch.arange(len(order), device=b.device)
        start = torch.ones_like(bs, dtype=torch.bool)
        start[1:] = bs[1:] != bs[:-1]
        last = torch.ones_like(bs, dtype=torch.bool)
        last[:-1] = bs[1:] != bs[:-1]
        nxt = pos + 1
        nxt[last] = pos[start][torch.cumsum(start.long(), 0)[last] - 1]               # wrap to the packet's first
        dst, src = order, order[nxt]
        out_face[b[dst], n[dst]] = face[b[src], n[src]]
        out_bary[b[dst], n[dst]] = bary[b[src], n[src]]
    else:
        raise ValueError("unregistered S45-CT mode " + repr(mode))
    return out_face, out_bary


def install(mode, npz):
    if mode not in MODES:
        raise ValueError("unregistered S45-CT mode " + repr(mode))
    if _STATE["mode"] is not None:
        raise RuntimeError("S45-CT routes already installed in this process")
    _STATE.update(mode=mode, npz=str(npz), native_route=XAG.route_association, native_sample=XAG.sample_sources)

    def sample_sources(batch, src, mask, **kwargs):
        out = _STATE["native_sample"](batch, src, mask, **kwargs)
        _STATE["rows"] = out["xag_source_index"]
        return out

    def route_association(route_mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol, z_tol, *,
                          source_part, source_face, source_bary, source_valid, source_depth=None, k=16, chunk=4096):
        rows, _STATE["rows"] = _STATE["rows"], None          # one packet's rows serve one routing call only
        if route_mode == "face_route":
            if rows is None or rows.shape != mask.shape:
                raise RuntimeError("S45-CT: the packet's source rows are missing")
            known = source_valid & (source_face >= 0) & (source_face < len(faces))
            known = known & torch.isfinite(source_bary).all(-1) & (source_bary >= 0).all(-1) & (source_bary.sum(-1) > 0)
            eligible = mask & (assoc["q"] > 0) & assoc["valid"] & known
            face, bary = corrupt(_STATE["mode"], source_face, source_bary, eligible, rows, _STATE["npz"])
            if _STATE["capture"] is not None:
                _STATE["capture"].append({"eligible": eligible.cpu(), "rows": rows.cpu(), "face": source_face.cpu(),
                                          "bary": source_bary.cpu(), "new_face": face.cpu(), "new_bary": bary.cpu()})
            source_face, source_bary = face, bary
        return _STATE["native_route"](route_mode, assoc, d, mask, verts, faces, vert_faces, lbs_weights, band, front_tol,
                                      z_tol, source_part=source_part, source_face=source_face, source_bary=source_bary,
                                      source_valid=source_valid, source_depth=source_depth, k=k, chunk=chunk)

    XAG.sample_sources, XAG.route_association = sample_sources, route_association


def installed():
    return _STATE["mode"]


def capture(on=True):
    """Probe-only: keep (eligible, rows, face, new face, ...) of every routed packet until switched off."""
    _STATE["capture"] = [] if on else None


def captured():
    return _STATE["capture"]


__all__ = ["MODES", "capture", "captured", "corrupt", "face_geodesic", "hash01", "install", "installed", "ring_table"]
