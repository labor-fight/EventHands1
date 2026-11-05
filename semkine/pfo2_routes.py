r"""PFO2 P0-B (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103): label routes with controlled location noise and
part misassignment, for the routing-precision dose-response (diagnostic only, not deployable).

On top of the label routing (semkine/s45sm_oracle.py: the batch's XAG per-event labels, PFA), every labelled live node
gets, independently and from one seeded generator:
  * with probability `mis`: a random face of a uniformly chosen other part (area-weighted), uniform barycentrics;
  * otherwise, location noise of scale `sigma` (mm): a face f' drawn from the surface kernel
    p(f' | f) ~ A_f' exp(-D_ff'^2 / (2 s^2)) (D = geodesic distance between face centroids on the rest template, along
    the face-adjacency graph) with uniform barycentrics in f', kept unchanged with probability `keep`; (s, keep) are
    calibrated on the rest template so that the median displacement |p' - p| (template Euclidean, mm, the router
    reading's point-error metric, tools/pfo/router_read.py) equals sigma for area-uniform label points.
The random draws of a packet come from a generator keyed on its events (`packet_generator`): the same packet always
gets the same noisy routes, different packets independent ones. `install(native, npz, sigma, mis, seed=0)` returns the
wrapped sample-sources function; `uninstall()` clears the state.
"""
from __future__ import annotations

import numpy as np
import torch

_STATE = {"on": False, "native": None, "geo": None, "kernel": None, "gen": None, "sigma": None, "mis": None,
          "keep": 0.0, "stats": []}


def template(npz):
    with np.load(str(npz), allow_pickle=False) as z:
        v = np.asarray(z["v_template"], np.float64)
        f = np.asarray(z["f"], np.int64)
        w = np.asarray(z["weights"], np.float64)
    from .s45sm import face_part_table
    return v, f, face_part_table(w, f)


def face_geodesics(v, f):
    """(F, F) geodesic distances (m) between face centroids along the face-adjacency graph, face areas, centroids."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    c = v[f].mean(1)
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    edges = {}
    for i, tri in enumerate(f):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(i)
    rows, cols = [], []
    for faces in edges.values():
        for x in faces:
            for y in faces:
                if x != y:
                    rows.append(x)
                    cols.append(y)
    rows, cols = np.asarray(rows), np.asarray(cols)
    w = np.linalg.norm(c[rows] - c[cols], axis=1)
    G = coo_matrix((w, (rows, cols)), shape=(len(f), len(f))).tocsr()
    return dijkstra(G, directed=False), area, c


def kernel(D, area, s):
    K = area[None, :] * np.exp(-0.5 * (D / max(s, 1e-9)) ** 2)
    return K / K.sum(1, keepdims=True)


def _points(v, f, face, bary):
    return (bary[:, :, None] * v[f[face]]).sum(1)


def displacement_median(v, f, D, area, s, keep, n=20000, seed=0):
    """Median |p' - p| (mm) of the location noise for area-uniform label points (the calibration target)."""
    rng = np.random.default_rng(seed)
    face = rng.choice(len(f), size=n, p=area / area.sum())
    bary = rng.dirichlet(np.ones(3), size=n)
    K = kernel(D, area, s)
    cum = np.cumsum(K[face], 1)
    new = (cum < rng.random(n)[:, None]).sum(1).clip(0, len(f) - 1)
    nb = rng.dirichlet(np.ones(3), size=n)
    stay = rng.random(n) < keep
    p, q = _points(v, f, face, bary), _points(v, f, new, nb)
    d = np.where(stay, 0.0, np.linalg.norm(q - p, axis=1)) * 1000.0
    return float(np.median(d))


def calibrate(v, f, D, area, sigma_mm, seed=0):
    """(s, keep): s by bisection to median displacement sigma; below the s -> 0 floor, keep > 0 brings the median down."""
    floor = displacement_median(v, f, D, area, 1e-6, 0.0, seed=seed)
    if sigma_mm <= floor:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if displacement_median(v, f, D, area, 1e-6, mid, seed=seed) > sigma_mm:
                lo = mid
            else:
                hi = mid
        return 1e-6, hi, floor
    lo, hi = 1e-6, 0.2
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if displacement_median(v, f, D, area, mid, 0.0, seed=seed) < sigma_mm:
            lo = mid
        else:
            hi = mid
    return hi, 0.0, floor


def perturb(face, bary, valid, part_of_face, geo, K, mis, keep, gen):
    """New (face, bary) for the valid nodes; torch tensors on any device, sampling on the CPU generator."""
    dev = face.device
    f_cpu, b_cpu, v_cpu = face.cpu(), bary.float().cpu(), valid.cpu()
    idx = v_cpu.reshape(-1).nonzero().flatten()
    out_f, out_b = f_cpu.reshape(-1).clone(), b_cpu.reshape(-1, 3).clone()
    n = idx.numel()
    if n == 0:
        return face, bary, {"n": 0}
    src = out_f[idx]
    u_mis = torch.rand(n, generator=gen)
    u_keep = torch.rand(n, generator=gen)
    noise_face = torch.multinomial(K[src], 1, generator=gen).squeeze(1)
    nb = -torch.log(torch.rand(n, 3, generator=gen).clamp_min(1e-12))
    nb = nb / nb.sum(1, keepdim=True)
    other = torch.randint(0, geo["n_parts"] - 1, (n,), generator=gen)
    own = part_of_face[src]
    other = other + (other >= own).long()                                    # a uniformly chosen other part
    pick = torch.rand(n, generator=gen)
    cum = geo["part_cum"][other]                                             # (n, M) area cumsum of the part's faces
    j = (cum < pick.unsqueeze(1)).sum(1).clamp_max(cum.shape[1] - 1)
    mis_face = geo["part_faces"][other, j]
    is_mis = u_mis < mis
    stay = (~is_mis) & (u_keep < keep)
    orig_b = out_b[idx].clone()
    new_f = torch.where(is_mis, mis_face, noise_face)
    new_f = torch.where(stay, src, new_f)
    new_b = torch.where(stay.unsqueeze(1), orig_b, nb)
    out_f[idx], out_b[idx] = new_f, new_b
    tm = geo["tmpl_t"]
    fv = geo["faces_t"]
    p = (orig_b.unsqueeze(-1) * tm[fv[src]]).sum(1)
    q = (new_b.unsqueeze(-1) * tm[fv[new_f]]).sum(1)
    disp = (q - p).norm(dim=1) * 1000.0
    stats = {"n": int(n), "mis": int(is_mis.sum()), "disp_mm": disp.numpy()}
    return out_f.view_as(f_cpu).to(dev), out_b.view_as(b_cpu).to(dev).to(bary.dtype), stats


def packet_generator(batch, seed):
    """One generator per packet, keyed on its events: the same packet always gets the same noise (so the base and the
    injected-state steps of one endpoint see identical routes), different packets independent noise. Evaluation
    packets only (one packet per batch)."""
    if int(batch.ptr.numel()) != 2:
        raise ValueError("noisy label routes are an evaluation diagnostic (one packet per batch)")
    ev = batch.events
    n = int(ev.shape[0])
    key = n
    if n:
        e = ev[:, :3].double()
        key = hash((n, float((e[:, 0] * 1009.0 + e[:, 1] * 7.0 + e[:, 2]).sum()), float(e[0, 0]), float(e[-1, 1])))
    return torch.Generator().manual_seed((int(seed) * 1000003 + key) % (2 ** 62))


def build_geo(npz):
    v, f, part = template(npz)
    D, area, _ = face_geodesics(v, f)
    n_parts = int(part.max()) + 1
    members = [np.nonzero(part == k)[0] for k in range(n_parts)]
    M = max(len(m) for m in members)
    pf = np.zeros((n_parts, M), np.int64)
    pc = np.ones((n_parts, M), np.float64) * 2.0                              # padding never drawn (cumsum > 1)
    for k, m in enumerate(members):
        a = area[m] / area[m].sum()
        pf[k, :len(m)] = m
        pc[k, :len(m)] = np.cumsum(a)
        pc[k, len(m) - 1] = 1.0
    return {"v": v, "f": f, "part": part, "D": D, "area": area, "n_parts": n_parts,
            "part_faces": torch.from_numpy(pf), "part_cum": torch.from_numpy(pc).float(),
            "tmpl_t": torch.from_numpy(v).float(), "faces_t": torch.from_numpy(f)}


def install(native, npz, sigma, mis, seed=0):
    """Return the noisy sample-sources function wrapping `native` (install it where the caller needs it)."""
    if _STATE["on"]:
        raise RuntimeError("PFO2 noisy routes already installed")
    geo = build_geo(npz)
    s, keep, floor = calibrate(geo["v"], geo["f"], geo["D"], geo["area"], float(sigma)) if sigma > 0 else (None, 1.0, None)
    K = torch.from_numpy(kernel(geo["D"], geo["area"], s)).float() if s is not None else None
    part_t = torch.from_numpy(geo["part"]).long()
    _STATE.update(on=True, native=native, geo=geo, kernel=K, gen=None, sigma=float(sigma), mis=float(mis), keep=keep,
                  stats=[], scale=s, floor=floor, seed=int(seed))

    def noisy(batch, src, mask, *args, **kwargs):
        lab = native(batch, src, mask, *args, **kwargs)
        gen = packet_generator(batch, int(seed))
        F_ = len(geo["f"])
        lb = lab["xag_bary"].float()
        ok = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
        ok = ok & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
        face = lab["xag_face"].clamp(0, F_ - 1)
        bary = torch.where(ok.unsqueeze(-1), lb / lb.sum(-1, keepdim=True).clamp_min(1e-12), torch.zeros_like(lb))
        kern = K if K is not None else torch.eye(F_)
        nf, nb, st = perturb(face, bary, ok, part_t, geo, kern, float(mis), keep if K is not None else 1.0, gen)
        _STATE["stats"].append(st)
        out = dict(lab)
        out["xag_face"] = torch.where(ok, nf, lab["xag_face"])
        out["xag_bary"] = torch.where(ok.unsqueeze(-1), nb.to(lab["xag_bary"].dtype), lab["xag_bary"])
        out["xag_part"] = torch.where(ok, part_t.to(nf.device)[nf], lab["xag_part"])
        return out
    return noisy


def summary():
    st = [s for s in _STATE["stats"] if s["n"]]
    d = np.concatenate([s["disp_mm"] for s in st]) if st else np.zeros(0)
    n = sum(s["n"] for s in st)
    return {"sigma_mm": _STATE["sigma"], "mis": _STATE["mis"], "scale_m": _STATE.get("scale"), "keep": _STATE["keep"],
            "floor_mm": _STATE.get("floor"), "nodes": int(n), "mis_share": sum(s["mis"] for s in st) / max(n, 1),
            "disp_mm_median": float(np.median(d)) if len(d) else None,
            "disp_mm_median_located": float(np.median(d[d > 0])) if (d > 0).any() else None}


def uninstall():
    _STATE.update(on=False, native=None, geo=None, kernel=None, gen=None, stats=[])


__all__ = ["build_geo", "calibrate", "displacement_median", "face_geodesics", "install", "kernel", "perturb",
           "summary", "template", "uninstall"]
