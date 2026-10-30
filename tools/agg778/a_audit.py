#!/usr/bin/env python3
"""AGG778 Stage A (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): implementation correctness of the 778-vertex
aggregation `semkine/xyz_mesh.py` L521-569, zero training, on the frozen C1 (A0) and A3F0 (A3).

    A1  reference equivalence: an independent NumPy float64 pooling (np.add.at / np.maximum.at) on the module's own
        phi vs the captured pre-vin x and H0; phi recomputed in float64 from the captured records; the probe record
        builder `heads.c1_records` vs the records the module actually encoded
    A2  properties: record order, batch composition, padding, zero-weight exclusion, empty vertices, duplicated records
    A3  index semantics: hit records' barycentric point = z d; vid / ring / LBS identity
    A4  mass-semantics statistics (descriptive)
    A5  numerics: training numerics (bf16 autocast, fp32 accumulation) vs evaluation numerics per packet, and C1's full
        zgz closed loop under both (the first must reproduce XCL E1 exactly)

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/agg778/a_audit.py --out outputs/agg778/a
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778.heads import c1_records  # noqa: E402

XCL_E1 = {"A0": "outputs/xcl/20261019/evaluations/xyz_c1_stable_s3407/e1_step6000_repeat1_none/states.npz",
          "A3": "outputs/xcl/20261019/evaluations/xcl_a3f0_s3407/e1_step6000_repeat1_none/states.npz"}
#: pooled x is fp64-accumulated then rounded to fp32 (1e-6); phi, vin and H0 are fp32 matmuls (1e-5)
TOL_POOL, TOL_PHI, TOL_GEOM = 1e-6, 1e-5, 1e-5


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)) if a.size else 0.0


def reference_pool(phi, rb, vrec, wrec, B):
    """Independent float64 NumPy pooling: weighted mean, unweighted max, log1p(mass), zero where empty."""
    phi = np.asarray(phi, np.float64)
    key = np.asarray(rb, np.int64) * C.V + np.asarray(vrec, np.int64)
    w = np.asarray(wrec, np.float64)
    Cc = phi.shape[1]
    mass = np.zeros(B * C.V)
    num = np.zeros((B * C.V, Cc))
    peak = np.full((B * C.V, Cc), -1e4)
    np.add.at(mass, key, w)
    np.add.at(num, key, w[:, None] * phi)
    np.maximum.at(peak, key, phi)
    has = mass > 0
    mean = num / np.maximum(mass, 1e-12)[:, None] * has[:, None]
    peak = peak * has[:, None]
    return np.concatenate([mean, peak, np.log1p(mass)[:, None]], 1).reshape(B, C.V, 2 * Cc + 1), has.reshape(B, C.V)


def hooked_forward(model, batch):
    """Forward with the relation encoder's first call (the vertex records) captured."""
    store = []
    h = model.xyz.rel.register_forward_hook(lambda m, i, o: store.append((i[0].detach(), o.detach())))
    try:
        cap = C.forward_capture(model, batch)
    finally:
        h.remove()
    cap["rel_in"], cap["rel_out"] = store[0]
    return cap


def a1(model, cap):
    B = cap["O"].shape[0]
    rec = c1_records(cap)
    feat, phi = cap["rel_in"].float(), cap["rel_out"].float()
    x_ref, has = reference_pool(phi.cpu().numpy(), rec["rb"].cpu().numpy(), rec["vrec"].cpu().numpy(),
                                rec["wrec"].cpu().numpy(), B)
    W1, b1 = model.xyz.rel[0].weight.double().cpu().numpy(), model.xyz.rel[0].bias.double().cpu().numpy()
    W2, b2 = model.xyz.rel[2].weight.double().cpu().numpy(), model.xyz.rel[2].bias.double().cpu().numpy()
    f64 = feat.double().cpu().numpy()
    phi64 = np.maximum(f64 @ W1.T + b1, 0.0) @ W2.T + b2
    Wv = model.xyz.vin.weight.double().cpu().numpy()
    H_ref = cap["x257"].double().cpu().numpy() @ Wv.T
    return {"pool_rel_err": rel_err(cap["x257"].cpu().numpy(), x_ref),
            "support_equal": bool(np.array_equal(has, (cap["O"] > 0).cpu().numpy())),
            "phi_rel_err": rel_err(phi.cpu().numpy(), phi64),
            "vin_rel_err": rel_err(cap["H0"].cpu().numpy(), H_ref),
            "records_equal_n": int(feat.shape[0]) == int(rec["feat"].shape[0]),
            "records_feat_rel_err": rel_err(rec["feat"].cpu().numpy(), feat.cpu().numpy())
            if int(feat.shape[0]) == int(rec["feat"].shape[0]) else float("inf"),
            "n_records": int(feat.shape[0])}


@torch.no_grad()
def xyz_call(model, tok, d, mask, verts, assoc, prev48):
    model.xyz(tok, d, mask, verts, assoc, prev48, capture=True)
    B = tok.shape[0]
    return model._agg778_store["x257"].float().reshape(B, C.V, -1).clone(), model.xyz.xag_last_capture["H0"].clone(), \
        model.xyz.xag_last_capture["O"].clone()


def take(assoc, idx):
    return {k: (v[:, idx] if v.dim() >= 2 else v) for k, v in assoc.items()}


def cat(assoc_a, assoc_b):
    return {k: torch.cat([assoc_a[k], assoc_b[k]], 1) for k in assoc_a}


@torch.no_grad()
def a2(model, cap, gen):
    tok, d, mask, verts, assoc, prev48 = cap["tok"], cap["d"], cap["mask"], cap["verts"], \
        {k: v for k, v in cap["assoc"].items() if k in ("vid", "A", "q", "z", "valid", "hit")}, cap["prev48"]
    x0, H0, O0 = xyz_call(model, tok, d, mask, verts, assoc, prev48)
    B, N = mask.shape
    out = {}
    perm = torch.randperm(N, generator=gen).to(mask.device)
    x, H, _ = xyz_call(model, tok[:, perm], d[:, perm], mask[:, perm], verts, take(assoc, perm), prev48)
    out["record_order_x"], out["record_order_H"] = rel_err(x.cpu(), x0.cpu()), rel_err(H.cpu(), H0.cpu())
    ex, eh = [], []
    for b in range(min(B, 8)):
        x1, H1, _ = xyz_call(model, tok[b:b + 1], d[b:b + 1], mask[b:b + 1], verts[b:b + 1],
                             {k: v[b:b + 1] for k, v in assoc.items()}, prev48[b:b + 1])
        ex.append(rel_err(x1[0].cpu(), x0[b].cpu()))
        eh.append(rel_err(H1[0].cpu(), H0[b].cpu()))
    out["batch_composition_x"], out["batch_composition_H"] = max(ex), max(eh)
    P = 64
    pad = {k: torch.zeros(B, P, *v.shape[2:], dtype=v.dtype, device=v.device) for k, v in assoc.items()}
    x, H, _ = xyz_call(model, torch.cat([tok, tok.new_zeros(B, P, tok.shape[2])], 1),
                       torch.cat([d, torch.tensor([0., 0., 1.], device=d.device).expand(B, P, 3)], 1),
                       torch.cat([mask, mask.new_zeros(B, P)], 1), verts, cat(assoc, pad), prev48)
    out["padding_x"], out["padding_H"] = rel_err(x.cpu(), x0.cpu()), rel_err(H.cpu(), H0.cpu())
    drop = (torch.rand(B, N, generator=gen) < 0.1).to(mask.device)
    a_q = dict(assoc)
    a_q["q"] = torch.where(drop, torch.zeros_like(assoc["q"]), assoc["q"])
    xq, Hq, _ = xyz_call(model, tok, d, mask, verts, a_q, prev48)
    xm, Hm, _ = xyz_call(model, tok, d, mask & ~drop, verts, assoc, prev48)
    out["zero_weight_exclusion_x"], out["zero_weight_exclusion_H"] = rel_err(xq.cpu(), xm.cpu()), rel_err(Hq.cpu(), Hm.cpu())
    empty = O0 == 0
    out["empty_vertices_max_abs"] = float(max(x0[empty].abs().max() if empty.any() else 0.0,
                                              H0[empty].abs().max() if empty.any() else 0.0))
    xd, Hd, Od = xyz_call(model, torch.cat([tok, tok], 1), torch.cat([d, d], 1), torch.cat([mask, mask], 1), verts,
                          cat(assoc, assoc), prev48)
    Cc = (x0.shape[-1] - 1) // 2
    m0 = torch.expm1(x0[..., -1].double())
    out["duplicate_mean_peak_x"] = rel_err(xd[..., :2 * Cc].cpu(), x0[..., :2 * Cc].cpu())
    out["duplicate_logmass_shift_x"] = rel_err(xd[..., -1].double().cpu(), torch.log1p(2 * m0).cpu())
    out["duplicate_support_equal"] = bool(torch.equal(Od, O0))
    return out


@torch.no_grad()
def a3(model, cap, cond):
    assoc, d, verts = cap["assoc"], cap["d"], cap["verts"]
    hit = assoc["hit"].bool() & cap["mask"]
    A = assoc["A"].float()
    Vr = torch.gather(verts.unsqueeze(1).expand(-1, d.shape[1], -1, -1), 2, assoc["vid"].unsqueeze(-1).expand(-1, -1, -1, 3))
    Xp = (A.unsqueeze(-1) * Vr).sum(2) / A.sum(-1, keepdim=True).clamp_min(1e-12)
    X = assoc["z"].float().unsqueeze(-1) * d
    out = {"hit_bary_point_max_m": float((Xp - X).norm(dim=-1)[hit].max()) if (hit.any() and cond == "A0") else None}
    f = model.mano.f.long()
    ring = model.xyz.ring_idx
    ok = True
    for a_, b_ in ((0, 1), (1, 2), (2, 0)):
        ok &= bool((ring[f[:, a_]] == f[:, b_].unsqueeze(-1)).any(-1).all())
    out["face_edges_in_ring"] = ok
    if cond == "A3" and "xag_part" in cap:
        w = assoc["q"].unsqueeze(-1) * A * cap["mask"].float().unsqueeze(-1)
        rb, rn, rr = (w > 0).nonzero(as_tuple=True)
        vid = assoc["vid"][rb, rn, rr]
        part = cap["xag_part"][rb, rn]
        hard = model.xyz.lbs_hard
        agree = float((hard[vid] == part).float().mean())
        g = torch.Generator(device=vid.device).manual_seed(1)
        shuf = torch.randint(0, C.V, vid.shape, generator=g, device=vid.device)
        out.update(lbs_part_agreement=agree, lbs_part_agreement_shuffled=float((hard[shuf] == part).float().mean()))
    return out


@torch.no_grad()
def a4(cap):
    rec = c1_records(cap)
    phi = cap["rel_out"].float()
    B = cap["O"].shape[0]
    key = rec["rb"] * C.V + rec["vrec"]
    Cc = phi.shape[1]
    peak = torch.full((B * C.V, Cc), -1e4, device=phi.device).scatter_reduce(0, key.unsqueeze(-1).expand(-1, Cc), phi,
                                                                            reduce="amax")
    is_arg = phi == peak[key]
    low = (rec["wrec"] < 0.05).unsqueeze(-1) & is_arg
    mass = torch.zeros(B * C.V, device=phi.device).index_add(0, key, rec["wrec"])
    cnt = torch.zeros(B * C.V, device=phi.device).index_add(0, key, torch.ones_like(rec["wrec"]))
    has = mass > 0
    live = cap["mask"]
    return {"peak_argmax_low_weight_frac": float(low.sum() / is_arg.sum().clamp(min=1)),
            "support_low_mass_frac": float(((mass < 0.05) & has).sum() / has.sum().clamp(min=1)),
            "records_per_supported_vertex_median": float(cnt[has].median()),
            "records_per_supported_vertex_p90": float(torch.quantile(cnt[has].float(), 0.9)),
            "hit_frac": float((cap["assoc"]["hit"].bool() & live).sum() / live.sum()),
            "valid_frac": float((cap["assoc"]["valid"].bool() & live).sum() / live.sum()),
            "residual_mass_frac": float((cap["r"] * live.float()).sum() / live.sum()),
            "direct_vertices_mean": float((cap["O"] > 0).float().sum(1).mean())}


def packet_sets(cfg):
    """256 zgz endpoints (E3 and C1-incoming anchors) and 256 fit-subject endpoints (R-noise anchors)."""
    zg = C.Endpoints(C.zgz_seqs())
    zg.index = zg.index[::max(1, len(zg.index) // 256)][:256]
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ))
    fit.index = fit.index[::max(1, len(fit.index) // 256)][:256]
    trace = C.strict_trace()
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(99)
    noise = [draw(rng)[0] for _ in range(len(fit))]
    return [("zgz_e3", zg, lambda it: it["gt_prev"]), ("zgz_loop", zg, lambda it: trace[it["seq"]][it["k"]]),
            ("fit_noise", fit, lambda it: it["gt_prev"] + noise[it["i"]])]


def own_loop(model, cfg, mano, dev, train_numerics=False):
    """C1's (or A3F0's) own zgz closed loop through `eval_track.track_sequence` (the XCL E1 contract: initial noise from
    rng 0, every endpoint kept); optionally the forward in training numerics (bf16 autocast, fp32 accumulation)."""
    from dataclasses import replace
    from semkine import eval_track as ET
    from tools.xcl.evaluate import score
    rng = np.random.default_rng(0)
    native = model.forward_packet
    res, preds_all = {}, {}
    for s, d in C.zgz_seqs():
        sq = C.Seq(s, d)
        rows = sq.grid
        preds, tgts = [], []

        def fp(packet):
            rid, end, previous, first = rows[len(preds)]
            packet = replace(packet, target=torch.from_numpy(sq.pos51[end].copy()).view(1, -1).to(dev))
            if train_numerics:
                model.xyz.train()
                try:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = native(packet)
                finally:
                    model.xyz.eval()
                out = out.float()
            else:
                out = native(packet)
            preds.append(out.detach().cpu().numpy()[0].copy())
            tgts.append(sq.pos51[end].copy())
            return out
        model.forward_packet = fp
        try:
            r = ET.track_sequence(model, mano, cfg, C.DATA_ROOT, d, s, 50, dev, rng, 1.0, 1.0, None)
        finally:
            model.forward_packet = native
        if r is None or r["n_frames"] != len(preds) or len(preds) != C.ZGZ_COUNTS[s]:
            raise RuntimeError("incomplete closed loop")
        beta = torch.tensor(sq.betas, device=dev).view(1, -1)
        m = score(mano, np.stack(preds).astype(np.float32), np.stack(tgts).astype(np.float32), beta, dev)
        res[s] = {k: float(v.mean()) for k, v in m.items()}
        preds_all[s] = np.stack(preds).astype(np.float32)
    return res, preds_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = C.require_gpu()
    if not args.smoke:
        C.verify_protocol()
    C.set_policy()
    out = Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit(f"{out} holds a result; preserve it")
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(20261024)
    res = {}
    for cond in ("A0", "A3"):
        model, cfg = C.load_frozen(cond, dev)
        mano = C.mano32(cfg, dev)
        rc = res.setdefault(cond, {})
        for name, ds, anchor_fn in packet_sets(cfg):
            if args.smoke:
                ds.index = ds.index[:32]
            agg = {"A1": [], "A2": [], "A3": [], "A4": [], "A5": []}
            for items in C.loader(ds, 32, False, 0, workers=args.workers):
                anchors = np.stack([anchor_fn(it) for it in items])
                batch = C.make_batch(items, anchors, dev)
                cap = hooked_forward(model, batch)
                agg["A1"].append(a1(model, cap))
                agg["A2"].append(a2(model, cap, gen))
                agg["A3"].append(a3(model, cap, cond))
                agg["A4"].append(a4(cap))
                # A5 (i): the same packets in training numerics
                model.xyz.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out_t = model.forward_packet(batch).float()
                model.xyz.eval()
                d_eval = (cap["out"] - batch.prev_state)
                d_tr = (out_t - batch.prev_state)
                from tools.xcl.evaluate import score
                diffs = []
                for b, it in enumerate(items):
                    beta = torch.tensor(it["betas"], device=dev).view(1, -1)
                    sc = score(mano, out_t[b:b + 1].cpu().numpy(), cap["out"][b:b + 1].cpu().numpy(), beta, dev)
                    diffs.append(float(sc["mpjpe_ra_mm"][0]))
                agg["A5"].append({"delta_abs_max": float((d_eval - d_tr).abs().max()),
                                  "delta_rel": float((d_eval - d_tr).norm() / d_eval.norm().clamp(min=1e-12)),
                                  "pred_ra_mm_median": float(np.median(diffs)), "pred_ra_mm_max": float(np.max(diffs))})
            summ = {}
            for k, rows in agg.items():
                keys = rows[0].keys()
                summ[k] = {}
                for kk in keys:
                    vals = [r[kk] for r in rows if r[kk] is not None]
                    if not vals:
                        summ[k][kk] = None
                    elif isinstance(vals[0], bool):
                        summ[k][kk] = all(vals)
                    elif kk.endswith(("frac", "median", "p90", "mean", "agreement", "shuffled", "delta_rel",
                                      "pred_ra_mm_median")):
                        summ[k][kk] = float(np.mean(vals))
                    else:
                        summ[k][kk] = float(np.max(vals))
            rc[name] = summ
        if not args.smoke:
            # A5 (ii): the closed loop in evaluation numerics (must reproduce XCL E1) and in training numerics
            e1_eval, p_eval = own_loop(model, cfg, mano, dev, False)
            e1_train, p_train = own_loop(model, cfg, mano, dev, True)
            rc["closed_loop"] = {"eval_numerics": e1_eval, "train_numerics": e1_train,
                                 "pred_max_abs_diff": {s: float(np.abs(p_eval[s] - p_train[s]).max()) for s in p_eval}}
            if cond in XCL_E1:
                with np.load(C.REPO / XCL_E1[cond]) as z:
                    rc["closed_loop"]["xcl_e1_bit_identical"] = {s: bool(np.array_equal(z[s + "/prediction"], p_eval[s]))
                                                                 for s in p_eval}
        del model
        torch.cuda.empty_cache()
    gates = {}
    for cond, rc in res.items():
        for name, s in rc.items():
            if name == "closed_loop":
                continue
            gates[f"{cond}/{name}"] = {
                "A1": s["A1"]["pool_rel_err"] <= TOL_POOL and s["A1"]["phi_rel_err"] <= TOL_PHI
                and s["A1"]["vin_rel_err"] <= TOL_PHI and s["A1"]["support_equal"] and s["A1"]["records_equal_n"]
                and s["A1"]["records_feat_rel_err"] <= TOL_POOL,
                "A2": all(v <= (TOL_POOL if k.endswith("_x") else TOL_PHI) for k, v in s["A2"].items()
                          if isinstance(v, float) and k != "empty_vertices_max_abs")
                and s["A2"]["empty_vertices_max_abs"] == 0.0 and s["A2"]["duplicate_support_equal"],
                "A3": (s["A3"]["hit_bary_point_max_m"] is None or s["A3"]["hit_bary_point_max_m"] <= TOL_GEOM)
                and s["A3"]["face_edges_in_ring"]
                and (cond != "A3" or (s["A3"]["lbs_part_agreement"] >= 0.9 and s["A3"]["lbs_part_agreement_shuffled"] <= 0.3))}
    res["gates"] = gates
    res["pass"] = all(all(v.values()) for v in gates.values())
    C.save_json(out / "result.json", res, exclusive=False)
    C.receipt(out, {"stage": "a_audit", "smoke": bool(args.smoke)}, scripts=(__file__, Path(__file__).with_name("heads.py")))
    print(json.dumps({"pass": res["pass"], "gates": gates}, indent=1), flush=True)


if __name__ == "__main__":
    main()
