#!/usr/bin/env python3
"""MRF Stage A (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): implementation correctness of C1's receptive-field
module G (`semkine/xyz_mesh.py` L572-581 with `XYZMeshLayer` L324-351 and the support it hands to `lbs_slot_pool`),
zero training, CPU only, on the frozen C1 (A0) and A3F0 (A3). Packets: zgz every 10th endpoint with E3 anchors (GT at
the previous endpoint) and E2 anchors (C1 strict-trace incoming), selection subjects every 40th endpoint (E3).

    A1  topology: the ring equals the independent face-edge set; the hop matrix and the 4-hop balls equal scipy BFS
    A2  support: S_k equals the exact k-hop dilation of O on every packet, k = 1..3
    A3  reference: independent float64 NumPy rounds (edge embedding, gate, mean over supported neighbours, residual,
        support) vs H1..H3 of the module, relative error <= 1e-5; the module's own capture H3 vs the re-run
    A4  structure: dH3_j / dH0_i exactly 0 beyond 3 hops (self entry nonzero); readout gradient exactly 0 off S; H3
        exactly 0 off S
    A5  invariances: vertex relabelling, batch composition, ring padding slots
    A6  numerics (descriptive): bf16 autocast (CPU) vs fp32 for G alone
    A7  receptive field (descriptive): the precheck measures (outputs/mrf/precheck_20261024) for both models
    A8  tooling: pytest tests/test_mrf.py (variants, probe tail, wiring on a copy)

Gate: A1-A5 pass (A8 is the engineering precondition of B / D).

    CUDA_VISIBLE_DEVICES="" python tools/mrf/a_audit.py --out outputs/mrf/a
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.mrf import common as MC  # noqa: E402
from semkine.xyz_mesh import L_EDGE, mesh_ring  # noqa: E402

TOL = 1e-5
PARTS = ["wrist/palm", "index1", "index2", "index3", "middle1", "middle2", "middle3", "pinky1", "pinky2", "pinky3",
         "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]
CPU = torch.device("cpu")


def q(x, ps=(10, 50, 90)):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return None
    return {"mean": float(x.mean()), **{f"p{p}": float(np.percentile(x, p)) for p in ps},
            "min": float(x.min()), "max": float(x.max())}


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)) if a.size else 0.0


# ------------------------------------------------------------------------------------------------ A1 / structural
def a1_structural():
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    from semkine import mrf as M
    with np.load(C.REPO / "assets" / "mano_right.npz", allow_pickle=False) as z:
        f = np.asarray(z["f"], np.int64)
        vt = np.asarray(z["v_template"], np.float64)
        W = np.asarray(z["weights"], np.float64)
        tips = np.asarray(z["fingertip_vertex_ids"], np.int64)
    V = vt.shape[0]
    idx, em = mesh_ring(torch.from_numpy(f), V)
    idx, em = idx.numpy(), em.numpy().astype(bool)
    rows = np.repeat(np.arange(V), idx.shape[1])[em.reshape(-1)]
    cols = idx.reshape(-1)[em.reshape(-1)]
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    ring_pairs = np.unique(np.sort(np.stack([rows, cols], 1), 1), axis=0)
    edges_match = bool(len(e) == len(ring_pairs) and (e == ring_pairs).all())
    symmetric = bool(len(rows) == 2 * len(e))
    A = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(V, V))
    hop = shortest_path(A, unweighted=True, directed=False)
    Dm = M.hop_matrix(torch.from_numpy(f), V, 4).numpy()
    hop_match = bool(np.array_equal(Dm, np.where(hop <= 4, hop, 5).astype(np.int64)))
    bidx, bm = M.ball_table(torch.from_numpy(f), V, 4)
    ball_match = all(sorted(bidx[v][bm[v] > 0].tolist()) == sorted(np.nonzero((hop[v] >= 1) & (hop[v] <= 4))[0].tolist())
                     for v in range(V))
    L = np.linalg.norm(vt[rows] - vt[cols], axis=1)
    geo = shortest_path(csr_matrix((L, (rows, cols)), shape=(V, V)), directed=False)
    hard = W.argmax(1)
    eu = np.linalg.norm(vt[:, None] - vt[None], axis=-1)
    res = {"V": V, "F": int(len(f)), "undirected_edges": int(len(e)), "A1_ring_equals_face_edges": edges_match,
           "A1_ring_symmetric": symmetric, "A1_hop_matrix_equals_bfs": hop_match, "A1_ball4_equals_bfs": ball_match,
           "degree": q(em.sum(1)), "edge_len_mm": q(np.linalg.norm(vt[e[:, 0]] - vt[e[:, 1]], axis=1) * 1e3),
           "graph_diameter_hops": float(hop[np.isfinite(hop)].max()),
           "tip_parts": [PARTS[int(hard[t])] for t in tips]}
    balls = {}
    for k in (1, 2, 3, 4, 6, 12):
        inb = hop <= k
        size = inb.sum(1)
        r_eu = np.where(inb, eu, 0).max(1) * 1e3
        r_geo = np.where(inb, geo, 0).max(1) * 1e3
        balls[k] = {"size": q(size), "r_eu_mm": q(r_eu), "r_geo_mm": q(r_geo),
                    "per_part_r_eu_mm_p50": {PARTS[j]: float(np.median(r_eu[hard == j])) for j in range(16)}}
    res["balls"] = balls
    res["tip_hops_to_part"] = {f"{t}:{PARTS[int(hard[t])]}": {PARTS[j]: float(hop[t, hard == j].min())
                                                               for j in range(16)} for t in tips}
    res["parts_in_3hop_ball"] = q([len(np.unique(hard[hop[v] <= 3])) for v in range(V)])
    res["hops_to_other_part"] = q([hop[v, hard != hard[v]].min() for v in range(V)])
    res["slot_support_w_gt0"] = {PARTS[j]: int((W[:, j] > 0).sum()) for j in range(16)}
    return res, hop.astype(np.int64)


# ------------------------------------------------------------------------------------------------ G re-runs
def run_gnn(xyz, H0, O, g, record=False):
    gemb = torch.relu(xyz.edge(g))
    H, S = H0, O
    hist = []
    for layer in xyz.layers:
        Hn, Sn = layer(H, S, gemb, xyz.ring_idx, xyz.ring_mask)
        if record:
            with torch.no_grad():
                gate = torch.sigmoid(layer.gate(gemb))
            hist.append({"H_prev": H.detach(), "H": Hn.detach(), "S": Sn.detach(), "gate": gate})
        H, S = Hn, Sn
    return H, S, hist


def dilate_exact(O, hop, k):
    """k-hop dilation of O by the hop matrix (independent of the ring tensor)."""
    reach = torch.from_numpy((hop <= k).astype(np.float32))                     # (V, V)
    return ((O @ reach) > 0).float()


def np_rounds(xyz, H0, O, verts):
    """Independent float64 NumPy rounds: (H1, S1), (H2, S2), (H3, S3)."""
    idx = xyz.ring_idx.numpy()
    msk = xyz.ring_mask.numpy().astype(np.float64)
    vf = verts.double().numpy()
    dv = vf[:, idx] - vf[:, :, None]                                             # (B, V, K, 3)
    g = np.concatenate([dv, np.linalg.norm(dv, axis=-1, keepdims=True)], -1) / L_EDGE * msk[None, :, :, None]
    We, be = xyz.edge.weight.double().numpy(), xyz.edge.bias.double().numpy()
    gemb = np.maximum(g @ We.T + be, 0.0)
    H, S = H0.double().numpy(), O.double().numpy()
    out = []
    for layer in xyz.layers:
        Wm, Ws = layer.msg.weight.double().numpy(), layer.slf.weight.double().numpy()
        Wg, bg = layer.gate.weight.double().numpy(), layer.gate.bias.double().numpy()
        Mn = (H @ Wm.T)[:, idx]                                                  # (B, V, K, C)
        Sn = S[:, idx] * msk[None]
        gate = 1.0 / (1.0 + np.exp(-(gemb @ Wg.T + bg)))
        agg = (gate * Mn * Sn[..., None]).sum(2) / np.maximum(Sn.sum(2), 1.0)[..., None]
        S_next = np.maximum(S, Sn.max(2))
        H = H + S_next[..., None] * np.maximum(H @ Ws.T + agg, 0.0)
        S = S_next
        out.append((H, S))
    return out


# ------------------------------------------------------------------------------------------------ A2-A6 per batch
def audit_batch(frozen, cap, hop, rng, n_jac):
    xyz = frozen.xyz
    last = frozen.xyz_last
    H0, O, S_cap, H3_cap = last["H0"].float(), cap["O"], cap["S"], last["H3"].float()
    verts = cap["verts"]
    g = xyz.edge_geometry(verts)
    r = {}
    with torch.no_grad():
        H3, S3, hist = run_gnn(xyz, H0, O, g, record=True)
    # A2
    r["A2_S_equals_dilation"] = all(bool(torch.equal(h["S"], dilate_exact(O, hop, k + 1))) for k, h in enumerate(hist))
    r["A2_capture_S_equals_S3"] = bool(torch.equal(S_cap, S3))
    # A3
    ref = np_rounds(xyz, H0, O, verts)
    r["A3_rel_err_H1"] = rel_err(hist[0]["H"].numpy(), ref[0][0])
    r["A3_rel_err_H2"] = rel_err(hist[1]["H"].numpy(), ref[1][0])
    r["A3_rel_err_H3"] = rel_err(H3.numpy(), ref[2][0])
    r["A3_S_equal"] = all(bool(np.array_equal(hist[k]["S"].numpy(), ref[k][1])) for k in range(3))
    r["A3_capture_rel_err"] = rel_err(H3_cap.numpy(), H3.numpy())
    # A4
    r["A4_H3_zero_off_S"] = bool((H3[S3 == 0] == 0).all()) and bool((H3_cap[S_cap == 0] == 0).all())
    r["A4_H0_zero_off_O"] = bool((H0[O == 0] == 0).all())
    jac_ok, self_nz, read_ok = True, True, True
    hop_t = torch.from_numpy(hop)
    for b in range(min(n_jac, H0.shape[0])):
        ob = (O[b] > 0).nonzero().squeeze(1).numpy()
        if len(ob) < 2:
            continue
        tg = torch.from_numpy(rng.choice(ob, size=min(8, len(ob)), replace=False))
        T = len(tg)
        Hr = H0[b:b + 1].expand(T, -1, -1).clone().requires_grad_(True)
        Hj, _, _ = run_gnn(xyz, Hr, O[b:b + 1].expand(T, -1), g[b:b + 1].expand(T, -1, -1, -1))
        u = torch.from_numpy(rng.standard_normal((T, Hj.shape[-1])).astype(np.float32))
        (gr,) = torch.autograd.grad((Hj[torch.arange(T), tg] * u).sum(), Hr)
        e = gr.abs().sum(-1)                                                    # (T, V)
        far = hop_t[tg] > 3
        jac_ok &= bool((e[far] == 0).all())
        self_nz &= bool((e[torch.arange(T), tg] > 0).all())
        h3 = H3[b:b + 1].clone().requires_grad_(True)
        capb = {"S": S3[b:b + 1], "O": O[b:b + 1], "bslot": last["bslot"][b:b + 1].float(),
                "prev48": last["prev48"][b:b + 1].float(), "H3": h3}
        out = xyz.xag_readout_from_h3(h3, capb)
        rv = torch.from_numpy(rng.standard_normal(51).astype(np.float32))
        (gh,) = torch.autograd.grad((out[0] * rv).sum(), h3)
        read_ok &= bool((gh[0][S3[b] == 0] == 0).all())
    r["A4_jacobian_zero_beyond_3_hops"] = jac_ok
    r["A4_jacobian_self_nonzero"] = self_nz
    r["A4_readout_grad_zero_off_S"] = read_ok
    # A5 relabelling, batch composition, padding slots
    with torch.no_grad():
        V = H0.shape[1]
        perm = torch.from_numpy(rng.permutation(V))
        inv = torch.argsort(perm)
        idx_n = inv[xyz.ring_idx[perm]]
        msk_n = xyz.ring_mask[perm]
        vn = verts[:, perm]
        K = idx_n.shape[1]
        dv = vn[:, idx_n.reshape(-1)].view(vn.shape[0], V, K, 3) - vn.unsqueeze(2)
        gn = torch.cat([dv, dv.norm(dim=-1, keepdim=True)], -1) / L_EDGE * msk_n.view(1, V, K, 1)
        gemb = torch.relu(xyz.edge(gn))
        Hn, Sn = H0[:, perm], O[:, perm]
        for layer in xyz.layers:
            Hn, Sn = layer(Hn, Sn, gemb, idx_n, msk_n)
        r["A5_relabel_rel_err"] = rel_err(Hn.numpy(), H3[:, perm].numpy())
        r["A5_relabel_S_equal"] = bool(torch.equal(Sn, S3[:, perm]))
        errs = []
        for b in range(min(4, H0.shape[0])):
            H1b, _, _ = run_gnn(xyz, H0[b:b + 1], O[b:b + 1], g[b:b + 1])
            errs.append(rel_err(H1b[0].numpy(), H3[b].numpy()))
        r["A5_batch_rel_err"] = max(errs)
        pad = xyz.ring_mask == 0
        idx_p = xyz.ring_idx.clone()
        idx_p[pad] = torch.from_numpy(rng.integers(0, V, int(pad.sum())))
        gemb = torch.relu(xyz.edge(g))
        Hp, Sp = H0, O
        for layer in xyz.layers:
            Hp, Sp = layer(Hp, Sp, gemb, idx_p, xyz.ring_mask)
        r["A5_padding_equal"] = bool(torch.equal(Hp, H3)) and bool(torch.equal(Sp, S3))
        # A6 (descriptive)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            Hb, _, _ = run_gnn(xyz, H0, O, g)
        live = S3 > 0
        r["A6_bf16_rel_err_H3_on_S"] = rel_err(Hb.float()[live].numpy(), H3[live].numpy())
    return r


# ------------------------------------------------------------------------------------------------ A7 (precheck measures)
def a7_measure(frozen, ds, hop, anchors_fn, n_jac=24, n_upd=32, n_route=24, seed=0):
    """The precheck quantities on one packet set (rf_measure / rf_measure2 / rf_measure3 of the precheck)."""
    from torch.func import jvp
    xyz = frozen.xyz
    hop_t = torch.from_numpy(hop)
    W = xyz.lbs_w.float()
    rng = np.random.default_rng(seed)
    st = {k: [] for k in ("O", "S3", "prop_share_slot", "z_mm", "r3_px_obs", "upd_rel_obs", "parts_O", "parts_S3")}
    per_layer = {l: {"upd_rel_obs": [], "upd_rel_prop": [], "gate_mean": []} for l in range(3)}
    jac, jac_n = np.zeros((2, 8)), np.zeros(2)
    upd, upd_n, ratio = np.zeros((2, 8)), np.zeros(2), []
    read = {"t": np.zeros(3), "r": np.zeros(3), "f": np.zeros(3)}
    route = {"t": np.zeros(3), "r": np.zeros(3), "f": np.zeros(3)}
    blocks = (("t", slice(0, 3)), ("r", slice(3, 6)), ("f", slice(6, 51)))
    done_j = done_u = done_r = 0
    for items in C.loader(ds, 32, False, 0, workers=4):
        anchors = np.stack([anchors_fn(x) for x in items])
        batch = C.make_batch(items, anchors, CPU)
        cap = C.forward_capture(frozen, batch)
        last = frozen.xyz_last
        H0, O, verts = last["H0"].float(), cap["O"], cap["verts"]
        g = xyz.edge_geometry(verts)
        with torch.no_grad():
            H3, S3, hist = run_gnn(xyz, H0, O, g, record=True)
        st["O"] += O.sum(1).tolist()
        st["S3"] += S3.sum(1).tolist()
        a = W.unsqueeze(0) * S3.unsqueeze(-1)
        live = a.sum(1) > 0
        prop = (a * (1 - O).unsqueeze(-1)).sum(1) / a.sum(1).clamp_min(1e-9)
        st["prop_share_slot"] += [float(prop[b][live[b]].mean()) for b in range(len(items)) if live[b].any()]
        hard = xyz.lbs_hard
        for b in range(len(items)):
            st["parts_O"].append(int(torch.unique(hard[O[b] > 0]).numel()))
            st["parts_S3"].append(int(torch.unique(hard[S3[b] > 0]).numel()))
        fx, fy, cx, cy = C.intrinsics(frozen, batch)
        uv = C.project(verts, fx, fy, cx, cy)
        st["z_mm"] += (verts[..., 2].mean(1) * 1e3).tolist()
        within = hop_t <= 3
        for b in range(len(items)):
            ob = O[b] > 0
            if ob.sum() == 0:
                continue
            ids = ob.nonzero().squeeze(1)[:64]
            du = (uv[b, ids].unsqueeze(1) - uv[b].unsqueeze(0)).norm(dim=-1)
            st["r3_px_obs"].append(float(torch.where(within[ids], du, torch.zeros_like(du)).amax(1).median()))
        for l, h in enumerate(hist):
            dH = (h["H"] - h["H_prev"]).norm(dim=-1)
            base = H0.norm(dim=-1).clamp_min(1e-6)
            obs, pr = O > 0, (h["S"] > 0) & (O == 0)
            per_layer[l]["upd_rel_obs"].append(float((dH[obs] / base[obs]).median()) if obs.any() else np.nan)
            ref = H0[obs].norm(dim=-1).median() if obs.any() else torch.tensor(1.0)
            per_layer[l]["upd_rel_prop"].append(float((dH[pr] / ref).median()) if pr.any() else np.nan)
            gm = h["gate"].mean(-1)
            per_layer[l]["gate_mean"].append(float(gm[xyz.ring_mask.bool().unsqueeze(0).expand_as(gm)].mean()))
        st["upd_rel_obs"] += ((H3 - H0).norm(dim=-1)[O > 0] / H0.norm(dim=-1)[O > 0].clamp_min(1e-6)).tolist()[:4000]
        for b in range(len(items)):
            ob = (O[b] > 0).nonzero().squeeze(1).numpy()
            pr = ((S3[b] > 0) & (O[b] == 0)).nonzero().squeeze(1).numpy()
            if len(ob) < 8 or len(pr) < 4:
                continue
            Ob, gb = O[b:b + 1], g[b:b + 1]
            # Jacobian energy by hop of H3 (with the skip) and of the learned update H3 - H0 (without)
            for kind, pool in ((0, ob), (1, pr)):
                if done_j >= n_jac and done_u >= n_upd:
                    break
                tg = rng.choice(pool, size=min(16, len(pool)), replace=False)
                T = len(tg)
                ti = torch.from_numpy(tg)
                Hr = H0[b:b + 1].expand(T, -1, -1).clone().requires_grad_(True)
                Hj, _, _ = run_gnn(xyz, Hr, Ob.expand(T, -1), gb.expand(T, -1, -1, -1))
                u = torch.from_numpy(rng.standard_normal((T, Hj.shape[-1])).astype(np.float32))
                u = u / u.norm(dim=-1, keepdim=True)
                hh = hop_t[ti].clamp(max=7)
                for which, acc, accn in (("full", jac, jac_n), ("update", upd, upd_n)):
                    if (which == "full" and done_j >= n_jac) or (which == "update" and done_u >= n_upd):
                        continue
                    s = ((Hj[torch.arange(T), ti] - (Hr[torch.arange(T), ti] if which == "update" else 0)) * u).sum()
                    (gr,) = torch.autograd.grad(s, Hr, retain_graph=True)
                    e = gr.pow(2).sum(-1) * (O[b] > 0).float().unsqueeze(0)
                    tot = e.sum(1, keepdim=True).clamp_min(1e-30)
                    for k in range(8):
                        acc[kind, k] += float(((e * (hh == k)) / tot).sum())
                    accn[kind] += T
                    if which == "update" and kind == 0:
                        self_e = e[torch.arange(T), ti]
                        ratio += ((e.sum(1) - self_e) / self_e.clamp_min(1e-30)).tolist()
            done_j += 1
            done_u += 1
            # readout reliance and output routing of observed evidence
            if done_r < n_route:
                h3 = H3[b:b + 1].clone().requires_grad_(True)
                capb = {"S": S3[b:b + 1], "O": Ob, "bslot": last["bslot"][b:b + 1].float(),
                        "prev48": last["prev48"][b:b + 1].float(), "H3": h3}
                out = xyz.xag_readout_from_h3(h3, capb)
                grads = {}
                o_, s_ = Ob[0] > 0, S3[b] > 0
                for key, sl in blocks:
                    gs = []
                    for _ in range(4):
                        rv = torch.from_numpy(rng.standard_normal(sl.stop - sl.start).astype(np.float32))
                        (gr,) = torch.autograd.grad((out[0, sl] * rv).sum(), h3, retain_graph=True)
                        e = gr[0].pow(2).sum(-1)
                        tot = float(e.sum()) + 1e-30
                        read[key] += np.array([float(e[o_].sum()) / tot, float(e[s_ & ~o_].sum()) / tot,
                                               float(e[~s_].sum()) / tot])
                        gs.append(gr[0])
                    grads[key] = torch.stack(gs)

                def gnn(h):
                    return run_gnn(xyz, h, Ob.expand(h.shape[0], -1), gb.expand(h.shape[0], -1, -1, -1))[0]
                tg = torch.from_numpy(rng.choice(ob, size=min(32, len(ob)), replace=False))
                T = len(tg)
                tang = torch.zeros(T, *H0.shape[1:])
                v = torch.from_numpy(rng.standard_normal((T, H0.shape[-1])).astype(np.float32))
                tang[torch.arange(T), tg] = v / v.norm(dim=-1, keepdim=True)
                _, dH3 = jvp(gnn, (H0[b:b + 1].expand(T, -1, -1).contiguous(),), (tang,))
                prop_b = s_ & ~o_
                onehot = torch.zeros(T, H0.shape[1], dtype=torch.bool)
                onehot[torch.arange(T), tg] = True
                for key, _ in blocks:
                    c = torch.einsum("rvc,tvc->rtv", grads[key], dH3)
                    own = c[:, torch.arange(T), tg]
                    other = (c * (o_.unsqueeze(0) & ~onehot).unsqueeze(0)).sum(-1)
                    via = (c * prop_b.view(1, 1, -1)).sum(-1)
                    route[key] += np.array([float(own.abs().sum()), float(other.abs().sum()), float(via.abs().sum())])
                done_r += 1
    out = {k: q(v) for k, v in st.items()}
    out["n_packets"] = len(st["O"])
    out["per_layer"] = {l: {k: float(np.nanmedian(v)) for k, v in d.items()} for l, d in per_layer.items()}
    names = ("target_observed", "target_propagated_only")
    hop_names = [f"hop{k}" if k < 7 else "hop>=7" for k in range(8)]
    out["erf_energy_by_hop"] = {n: dict(zip(hop_names, (jac[i] / max(jac_n[i], 1)).tolist())) for i, n in enumerate(names)}
    out["update_erf_by_hop"] = {n: dict(zip(hop_names, (upd[i] / max(upd_n[i], 1)).tolist())) for i, n in enumerate(names)}
    out["neighbour_to_self_energy_ratio_observed"] = q(ratio)
    out["readout_grad_share_O_prop_outside"] = {k: (v / max(4 * done_r, 1)).tolist() for k, v in read.items()}
    out["routing_own_otherobs_viaprop"] = {k: (v / max(v.sum(), 1e-30)).tolist() for k, v in route.items()}
    return out


def packet_sets():
    trace = C.strict_trace()
    sets = []
    for dom in C.ZGZ:
        ds = C.Endpoints([x for x in C.zgz_seqs() if x[0] == dom], stride=10)
        sets.append((f"{dom}/E3", ds, lambda it: it["gt_prev"]))
        sets.append((f"{dom}/E2", ds, lambda it, tr=trace[dom]: tr[it["k"]]))
    sets.append(("sel/E3", C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=40), lambda it: it["gt_prev"]))
    return sets


def pytest_tooling():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_mrf.py"], cwd=C.REPO, capture_output=True,
                       text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-500:]
    return {"returncode": r.returncode, "summary": tail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="first 32 packets of one set, not a result")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    if not args.smoke:
        MC.verify_protocol("A")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {"tooling_A8": pytest_tooling()}
    print("A8", res["tooling_A8"], flush=True)
    res["structural"], hop = a1_structural()
    print("A1", {k: v for k, v in res["structural"].items() if k.startswith("A1")}, flush=True)
    rng = np.random.default_rng(20261024)
    sets = packet_sets()
    if args.smoke:
        sets = sets[:1]
    for cond in ("A0", "A3"):
        frozen, cfg = C.load_frozen(cond, CPU)
        rc = res.setdefault(cond, {})
        for name, ds, anchor_fn in sets:
            if args.smoke:
                ds.index = ds.index[:32]
            rows = []
            for items in C.loader(ds, 32, False, 0, workers=4):
                anchors = np.stack([anchor_fn(x) for x in items])
                cap = C.forward_capture(frozen, C.make_batch(items, anchors, CPU))
                rows.append(audit_batch(frozen, cap, hop, rng, n_jac=2))
            summ = {}
            for k in rows[0]:
                vals = [r_[k] for r_ in rows]
                summ[k] = all(vals) if isinstance(vals[0], bool) else float(np.max(vals))
            rc[name] = {"checks": summ, "n_packets": len(ds)}
            if not args.smoke:
                rc[name]["A7"] = a7_measure(frozen, ds, hop, anchor_fn)
            print(cond, name, json.dumps(summ), f"{time.time() - t0:.0f} s", flush=True)
            C.save_json(out / "result.partial.json", res, exclusive=False)
        del frozen
    gates = {"A1": all(v for k, v in res["structural"].items() if k.startswith("A1"))}
    checks = [res[c][n]["checks"] for c in ("A0", "A3") for n in res[c]]
    gates["A2"] = all(c["A2_S_equals_dilation"] and c["A2_capture_S_equals_S3"] for c in checks)
    gates["A3"] = all(c["A3_rel_err_H1"] <= TOL and c["A3_rel_err_H2"] <= TOL and c["A3_rel_err_H3"] <= TOL
                      and c["A3_S_equal"] and c["A3_capture_rel_err"] <= TOL for c in checks)
    gates["A4"] = all(c["A4_H3_zero_off_S"] and c["A4_H0_zero_off_O"] and c["A4_jacobian_zero_beyond_3_hops"]
                      and c["A4_jacobian_self_nonzero"] and c["A4_readout_grad_zero_off_S"] for c in checks)
    gates["A5"] = all(c["A5_relabel_rel_err"] <= TOL and c["A5_relabel_S_equal"] and c["A5_batch_rel_err"] <= TOL
                      and c["A5_padding_equal"] for c in checks)
    res["gates"] = gates
    res["pass"] = all(gates.values())
    res["tooling_pass"] = res["tooling_A8"]["returncode"] == 0
    res["wall_s"] = time.time() - t0
    C.save_json(out / "result.json", res, exclusive=False)
    (out / "result.partial.json").unlink(missing_ok=True)
    MC.receipt(out, {"stage": "a_audit", "smoke": bool(args.smoke)}, scripts=(__file__,))
    print(json.dumps({"pass": res["pass"], "gates": gates, "tooling_pass": res["tooling_pass"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
