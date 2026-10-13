#!/usr/bin/env python3
"""Architecture-decision probe, 2026-09-30 (docs/ARCHITECTURE_DECISION*.md).

--part main
  H0  the batch-1 closed loop reproduces the S37 main row (drift gate 0.05 mm).
  H2  absolute decodability of frozen features, fitted on training subjects only (ridge lambda and
      MLP early stop on a training-subject tail hold-out), scored on zgz:
        hist        2 pol x 30 x 40 event histogram (learning-free)
        <s37>:pf    S37 prev-free pooled features [feat 512 | node mean 128 | node max 128]
        <s37>:route [feat | 16 routed joint evidences] = the root head's input; train routed by GT
                    prev, test routed by GT prev (oracle) and by the model's own closed-loop prev
        <cnn>:pen   per-frame CNN penultimate features
      The MLP logs step-0 gradient norm, the share of parameters with a gradient and the relative
      parameter change, so a probe that never trained cannot pass silently.
  R1  candidate: prev-free absolute root-rotation head on <s37>:pf, fused into the S37 closed loop
      by SO(3) interpolation with a fixed alpha. All variants advance in one batch with the alpha=0
      baseline, so the matched comparison shares numerics; controls: training-mean rotation
      (shrinkage), histogram head (trivial features), CNN head (reference, not a candidate).
--part h5
  H5  one thinning mask per keep rate on zgz_global, shared by S37 and CNN, every derived
      representation recomputed; S37 closed loop, CNN per frame, and the S37 graph's actual
      1-hop / 3-hop coverage (ms, px, ancestors) and node retention.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "model", REPO / "tools", REPO / ".experiments" / "s37_debug_20260928",
           REPO / ".experiments" / "lit_hyp_20260929" / "debug"):
    sys.path.insert(0, str(_p))
import probe_structure_debug as PSD                                                    # noqa: E402
from probe_decomp import STEP, aa_to_R, decomp, summ                                   # noqa: E402
from probe_rot import R_to_aa                                                          # noqa: E402
from probe_s37_route import load_run                                                   # noqa: E402
from semkine import eval_track as ET                                                   # noqa: E402
from semkine import events as EV                                                       # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest                       # noqa: E402
from semkine.encoder import event_tokens                                               # noqa: E402
from semkine.event_gnn import T_COL                                                    # noqa: E402
from semkine.routed_readout import pool_joint_evidence                                 # noqa: E402

S37_RUNS = PSD.S37_RUNS
CNN_RUNS = PSD.CNN_RUNS
MAIN_ROW = PSD.MAIN_ROW
KEEPS = (1.0, 0.5, 0.2, 0.1, 0.05)
ALPHA = 0.5                     # fixed before the run (X1c precedent); 1.0 = replacement ablation
RUN = time.strftime("run_%H%M%S")


def r3(x):
    return PSD.r3(x)


def T1(x, device):
    return torch.from_numpy(np.asarray(x, np.float32).copy()).view(1, -1).to(device)


def truncate(s, n_rows):
    """Smoke mode: keep the first `n_rows` packets of each sequence."""
    s = copy.copy(s)
    runs, left = [], n_rows
    for a, ends in s.runs:
        if left <= 0:
            break
        runs.append((a, ends[:left]))
        left -= len(ends[:left])
    s.runs = runs
    return s


def load_models(device):
    s37s, cnns, hooks, cfg, cnn_cfg, ck = {}, {}, {}, None, None, {}
    for r in S37_RUNS:
        m, c, sel = load_run(r, device)
        m.eval()
        s37s[Path(r).name] = m
        cfg = cfg or c
        ck[Path(r).name] = sel["ckpt"]
    for r in CNN_RUNS:
        m, c, sel = load_run(r, device)
        m.eval()
        name = Path(r).name
        store = {}
        m.rn.fc.register_forward_hook(lambda mod, inp, out, store=store: store.__setitem__("pen", inp[0].detach()))
        cnns[name], cnn_cfg, hooks[name] = m, (cnn_cfg or c), store
        ck[name] = sel["ckpt"]
    return s37s, cnns, hooks, cfg, cnn_cfg, ck


# ------------------------------------------------------------------------------------ closed loops
@torch.no_grad()
def loop_variants(m, s, inits, device, heads):
    """All variants advance together as one batch (identical numerics for the matched baseline).
    heads: [(name, Rabs (rows, 3, 3) on device or None, alpha)]. Empty packets are not fused."""
    V = len(heads)
    preds, prevs, gts = [[] for _ in range(V)], [[] for _ in range(V)], []
    i = 0
    for (a, ends), init in zip(s.runs, inits):
        prev = T1(s.pos51[a] + init, device).expand(V, -1).contiguous()
        for end in ends:
            ev5 = s.packet_events(end)
            out = m.forward_packet(PSD.packet_batch([ev5] * V, prev, s, device)).float().clone()
            if len(ev5):
                for v, (_, Rabs, alpha) in enumerate(heads):
                    if Rabs is not None and alpha > 0:
                        Rm = aa_to_R(out[v:v + 1, 3:6])
                        rel = R_to_aa(Rm.transpose(-1, -2) @ Rabs[i:i + 1])
                        out[v:v + 1, 3:6] = R_to_aa(Rm @ aa_to_R(alpha * rel))
            for v in range(V):
                preds[v].append(out[v:v + 1])
                prevs[v].append(prev[v:v + 1])
            gts.append(T1(s.pos51[end], device))
            prev = out
            i += 1
    return [torch.cat(p) for p in preds], [torch.cat(p) for p in prevs], torch.cat(gts)


# ------------------------------------------------------------------------------------ probes
def ridge_fit(X, Y, device, lams=(1e-2, 1e-1, 1, 10, 100, 1000), hold=0.15):
    """Closed-form ridge on standardised features; lambda chosen on the training tail hold-out."""
    X, Y = X.double().to(device), Y.double().to(device)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    A = (X - mu) / sd
    n = len(A)
    k = int(n * (1 - hold))
    ym = Y[:k].mean(0)
    evals, V = torch.linalg.eigh(A[:k].T @ A[:k])
    B = V.T @ (A[:k].T @ (Y[:k] - ym))
    AkV = A[k:] @ V
    errs = {}
    for lam in lams:
        errs[lam] = float(((AkV @ (B / (evals + lam * k).unsqueeze(1)) + ym - Y[k:]) ** 2).sum(1).mean())
    lam = min(errs, key=errs.get)
    hold_pred = (AkV @ (B / (evals + lam * k).unsqueeze(1)) + ym).cpu()
    yma = Y.mean(0)
    w = torch.linalg.solve(A.T @ A + lam * n * torch.eye(A.shape[1], dtype=A.dtype, device=device),
                           A.T @ (Y - yma))

    def predict(Xte):
        return ((((Xte.double().to(device) - mu) / sd) @ w) + yma).cpu()
    state = {"mu": mu.cpu(), "sd": sd.cpu(), "w": w.cpu(), "ym": yma.cpu(), "lam": lam}
    return predict, {"lambda": lam, "holdout_sq_err_by_lambda": errs}, hold_pred, state


def mlp_fit(X, Y, device, steps=3000, seed=0, hold=0.15, tag=""):
    """2 x 256 MLP; early stop on the training tail hold-out; logs proof that it trained."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    ym, ys = Y.mean(0), Y.std(0).clamp_min(1e-6)
    A = ((X - mu) / sd).float().to(device)
    y = ((Y - ym) / ys).float().to(device)
    n = len(A)
    k = int(n * (1 - hold))
    net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 256),
                              torch.nn.ReLU(), torch.nn.Linear(256, y.shape[1])).to(device)
    p0 = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    curve, best, ver = [], (float("inf"), None, -1), {}
    with torch.enable_grad():
        for it in range(steps):
            idx = torch.randint(0, k, (512,), generator=g).to(device)
            loss = ((net(A[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            if it == 0:
                grads = [p.grad for p in net.parameters()]
                ver = {"requires_grad_all": all(p.requires_grad for p in net.parameters()),
                       "grad_enabled_in_loop": torch.is_grad_enabled(),
                       "loss_step0": float(loss),
                       "grad_norm_step0": float(torch.sqrt(sum((gg.float() ** 2).sum() for gg in grads if gg is not None))),
                       "params_with_nonzero_grad": sum(int(gg is not None and bool((gg != 0).any())) for gg in grads),
                       "n_param_tensors": len(grads)}
            opt.step()
            if it % 150 == 149:
                with torch.no_grad():
                    v = float(((net(A[k:]) - y[k:]) ** 2).mean())
                curve.append([it + 1, float(loss), v])
                if v < best[0]:
                    best = (v, {kk: vv.clone() for kk, vv in net.state_dict().items()}, it + 1)
    num = sum(float(((p.detach() - q) ** 2).sum()) for p, q in zip(net.parameters(), p0))
    den = sum(float((q ** 2).sum()) for q in p0)
    ver.update({"rel_param_change": math.sqrt(num / max(den, 1e-12)), "best_step": best[2],
                "final_train_loss": curve[-1][1] if curve else None, "best_holdout_loss": best[0], "curve": curve})
    net.load_state_dict(best[1])
    net.eval()

    @torch.no_grad()
    def predict(Xte):
        return net(((Xte - mu) / sd).float().to(device)).cpu().double() * ys + ym
    return predict, ver, predict(X[k:])


def rot_err_deg(pred9, gt51):
    R = PSD.project_so3(pred9[:, :9].reshape(-1, 3, 3).float())
    return PSD.geodesic_deg(R, aa_to_R(gt51[:, 3:6].float()))


# ------------------------------------------------------------------------------------ features
@torch.no_grad()
def s37_feats(m, batch, routes):
    """prev-free pooled vector, and [feat | routed evidence] for each routing prev in `routes`."""
    feat, h, _, px, py, mask = m.event_encoder(batch.events, batch.ptr, batch.delta_t_s, None, return_nodes=True)
    live = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
    mean = (h * mask.unsqueeze(-1).to(h.dtype)).sum(1) / live
    peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values * mask.any(1, keepdim=True).to(h.dtype)
    out = {"pf": torch.cat([feat, mean, peak], 1).float()}
    for name, prev in routes.items():
        if h.shape[1] == 0:
            ev = torch.zeros(feat.shape[0], 16, 2 * h.shape[-1] + 1, device=feat.device, dtype=feat.dtype)
        else:
            betas_f, k_f = m._resolve_betas_K(prev, batch.betas, batch.camera_K)
            a, _, _ = m._route_nodes(px, py, mask, prev, betas_f, k_f)
            ev, _ = pool_joint_evidence(h, a, mask)
        out[name] = torch.cat([feat.float(), ev.flatten(1).float()], 1)
        out[name + ":_raw"] = (feat, ev)
    return out


@torch.no_grad()
def check_extraction(m, s, rows, device):
    """The routed evidence extracted here is what the heads read: rebuild forward_packet."""
    chunk = rows[:32]
    evs = [s.packet_events(e) for _, e in chunk]
    G0 = PSD.T([s.pos51[p] for p, _ in chunk], device)
    batch = PSD.packet_batch(evs, G0, s, device)
    ref = m.forward_packet(batch).float()
    f = s37_feats(m, batch, {"gt": G0})
    feat, ev = f["gt:_raw"]
    rec = m._decode_active(feat, G0, ev)
    rec = rec + m.prev_mlp(G0.to(rec.dtype))
    rec = (rec + G0.to(rec.dtype)).float()
    d = float((rec - ref).abs().max())
    return d


@torch.no_grad()
def main_part(a):
    device = torch.device("cuda")
    torch.set_num_threads(16)
    t0 = time.time()
    res = {"protocol": vars(a), "alpha": ALPHA, "run": RUN}
    s37s, cnns, hooks, cfg, cnn_cfg, ck = load_models(device)
    res["ckpts"] = ck
    fkm = next(iter(s37s.values()))
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    zgz = [PSD.SeqNC(root, d, s, device) for s, d in sequences_for_split(root, "val_core", mani)]
    if a.smoke:
        zgz = [truncate(s, 40) for s in zgz]
    lnes_ch = EV.event_channels(cnn_cfg)
    rng0 = np.random.default_rng(0)
    inits = {s.name: [ET.sample_init_noise(cfg, rng0, 1.0) for _ in s.runs] for s in zgz}
    nfr = {s.name: sum(len(e) for _, e in s.runs) for s in zgz}

    # ------------------------------------------------------------ H0 batch-1 closed loop = main row
    closed = {}
    for name, m in s37s.items():
        closed[name] = {}
        for s in zgz:
            m.set_hand_context(s.betas, s.K)
            preds, prevs, gts = loop_variants(m, s, inits[s.name], device, [("base", None, 0.0)])
            d = summ(decomp(m, preds[0], gts, s.betas))
            closed[name][s.name] = {"prevs": prevs[0], "preds": preds[0], "gts": gts, "summary": d}
            ref = MAIN_ROW.get(name, {}).get(s.name)
            res.setdefault("H0", {})[f"{name}:{s.name}"] = {**d, "main_row": ref,
                                                            "drift": None if ref is None else abs(d["ra"] - ref)}
            print("H0", name, s.name, r3(d), "main row", ref, f"{time.time() - t0:.0f}s", flush=True)

    # ------------------------------------------------------------ H2 feature extraction
    rng2 = np.random.default_rng(2)
    train = [PSD.SeqNC(root, d, s, device) for s, d in sequences_for_split(root, "train", mani)]
    if a.smoke:
        train = train[:4]
    res["n_train_sequences"] = len(train)
    for name, m in s37s.items():
        res.setdefault("H2_extract_max_abs_diff", {})[name] = check_extraction(m, zgz[0], PSD.rows_of(zgz[0]), device)

    def extract(seq_rows, own):
        F_ = {}

        def put(k, v):
            F_.setdefault(k, []).append(v.cpu() if torch.is_tensor(v) else v)
        spans = []
        for s, rows in seq_rows:
            i0 = sum(len(v) for v in F_.get("gt51", []))
            for n_, m in s37s.items():
                m.set_hand_context(s.betas, s.K)
            for i in range(0, len(rows), 64):
                chunk = rows[i:i + 64]
                evs = [s.packet_events(e) for _, e in chunk]
                B = len(chunk)
                G0 = PSD.T([s.pos51[p] for p, _ in chunk], device)
                batch = PSD.packet_batch(evs, torch.zeros(B, 51, device=device), s, device)
                for n_, m in s37s.items():
                    routes = {"gt": G0}
                    if own:
                        routes["own"] = closed[n_][s.name]["prevs"][i:i + B]
                    f = s37_feats(m, batch, routes)
                    put(f"{n_}:pf", f["pf"])
                    put(f"{n_}:route_gt", f["gt"])
                    if own:
                        put(f"{n_}:route_own", f["own"])
                put("hist", torch.from_numpy(np.stack([PSD.hist_feat(ev) for ev in evs])))
                x = torch.from_numpy(np.stack([ET.build_lnes(s.events, s.offsets, int(e), STEP, lnes_ch)
                                               for _, e in chunk])).to(device)
                for n_, c in cnns.items():
                    o = c(x, torch.zeros(B, 51, device=device)).float()
                    put(f"{n_}:pen", hooks[n_]["pen"].float())
                    put(f"{n_}:out", o)
                put("gt51", torch.from_numpy(np.stack([s.pos51[e] for _, e in chunk]).astype(np.float32)))
                put("nev", torch.tensor([len(ev) for ev in evs]))
            spans.append((s.name, slice(i0, i0 + len(rows)), s.betas))
        return {k: torch.cat(v) for k, v in F_.items()}, spans

    Ftr, _ = extract([(s, PSD.rows_of(s, a.train_per_seq, rng2)) for s in train], own=False)
    print("extract train", len(Ftr["gt51"]), f"{time.time() - t0:.0f}s", flush=True)
    Fte, spans = extract([(s, PSD.rows_of(s)) for s in zgz], own=True)
    print("extract zgz", len(Fte["gt51"]), f"{time.time() - t0:.0f}s", flush=True)
    te = {"gt51": Fte["gt51"], "seqs": spans}

    def target(F_):
        g = F_["gt51"].double()
        return torch.cat([aa_to_R(g[:, 3:6]).reshape(-1, 9), g[:, 6:51]], 1)
    Ytr = target(Ftr)
    ktr = int(len(Ytr) * 0.85)
    gt_hold = Ftr["gt51"][ktr:]
    res["H2"] = {"n_train": int(len(Ytr)), "n_test": int(len(Fte["gt51"]))}

    refs = {"train_mean": Ytr.mean(0, keepdim=True).expand(len(Fte["gt51"]), -1).clone()}
    for n_ in cnns:
        o = Fte[f"{n_}:out"].double()
        refs[f"{n_}:own_head"] = torch.cat([aa_to_R(o[:, 3:6]).reshape(-1, 9), o[:, 6:51]], 1)
    for n_ in s37s:
        o = torch.cat([closed[n_][s.name]["preds"] for s in zgz]).double().cpu()
        refs[f"{n_}:closed_loop_output"] = torch.cat([aa_to_R(o[:, 3:6]).reshape(-1, 9), o[:, 6:51]], 1)
    for rname, p in refs.items():
        sc = PSD.score_abs(p, te, fkm, device)
        res["H2"][rname] = sc
        print("H2", rname, r3(sc), flush=True)

    heads = {}          # (feature set) -> test predictions for R1
    fsets = ["hist"] + [f"{n_}:pf" for n_ in s37s] + [f"{n_}:route_gt" for n_ in s37s] + [f"{n_}:pen" for n_ in cnns]
    for fs in fsets:
        Xtr = Ftr[fs]
        tests = {"test": Fte[fs]}
        if fs.endswith(":route_gt"):
            tests = {"test_route_gt(oracle)": Fte[fs], "test_route_own": Fte[fs.replace("route_gt", "route_own")]}
        rp, rinfo, rhold, rstate = ridge_fit(Xtr, Ytr, device)
        mp, mver, mhold = mlp_fit(Xtr, Ytr, device, tag=fs)
        rec = {"dim": int(Xtr.shape[1]), "ridge_info": rinfo, "mlp_verify": mver,
               "holdout_rot_deg": {"ridge": float(rot_err_deg(rhold, gt_hold).mean()),
                                   "mlp": float(rot_err_deg(mhold, gt_hold).mean())}}
        for tn, Xte in tests.items():
            pr, pm = rp(Xte), mp(Xte)
            rec[tn] = {"ridge": PSD.score_abs(pr, te, fkm, device), "mlp": PSD.score_abs(pm, te, fkm, device)}
            if tn == "test":
                heads[fs] = {"ridge": pr, "mlp": pm, "ridge_state": rstate}
        res["H2"][fs] = rec
        print("H2", fs, r3({k: v for k, v in rec.items() if k != "mlp_verify"}), f"{time.time() - t0:.0f}s", flush=True)
    (HERE / f"results_main_{RUN}.json").write_text(json.dumps(r3(res), indent=1, default=str))

    # ------------------------------------------------------------ R1 candidate in the real closed loop
    res["R1"] = {}
    mean_R = PSD.project_so3(Ytr[:, :9].mean(0).reshape(1, 3, 3).float())
    for (n_, m), cn in zip(s37s.items(), cnns):
        pf = res["H2"][f"{n_}:pf"]["holdout_rot_deg"]
        primary = "ridge" if pf["ridge"] <= pf["mlp"] else "mlp"
        torch.save(heads[f"{n_}:pf"]["ridge_state"], HERE / f"r1_ridge_head_{n_}.pt")

        def Rof(p):
            return PSD.project_so3(p[:, :9].reshape(-1, 3, 3).float()).to(device)
        variants = [("base_alpha0", None, 0.0),
                    (f"R1_{primary}_a{ALPHA}", Rof(heads[f"{n_}:pf"][primary]), ALPHA),
                    (f"R1_{'mlp' if primary == 'ridge' else 'ridge'}_a{ALPHA}",
                     Rof(heads[f"{n_}:pf"]["mlp" if primary == "ridge" else "ridge"]), ALPHA),
                    (f"R1_{primary}_a1.0", Rof(heads[f"{n_}:pf"][primary]), 1.0),
                    (f"ctrl_trainmean_a{ALPHA}", mean_R.to(device).expand(len(Fte["gt51"]), -1, -1).contiguous(), ALPHA),
                    (f"ctrl_hist_ridge_a{ALPHA}", Rof(heads["hist"]["ridge"]), ALPHA),
                    (f"ref_cnn_ridge_a{ALPHA}", Rof(heads[f"{cn}:pen"]["ridge"]), ALPHA)]
        out = {"primary_head": primary, "holdout_rot_deg_pf": pf, "variants": {}}
        for s, (sname, sl, _) in zip(zgz, spans):
            m.set_hand_context(s.betas, s.K)
            vs = [(vn, None if R is None else R[sl], al) for vn, R, al in variants]
            preds, _, gts = loop_variants(m, s, inits[s.name], device, vs)
            for (vn, _, _), p in zip(vs, preds):
                out["variants"].setdefault(vn, {})[s.name] = summ(decomp(m, p, gts, s.betas))
            # mechanism: error of the absolute head vs the S37 closed-loop output (same packets)
            Rg = aa_to_R(gts[:, 3:6])
            e_s37 = R_to_aa(aa_to_R(preds[0][:, 3:6]) @ Rg.transpose(-1, -2)).cpu().double()
            Rp = variants[1][1][sl]
            e_abs = R_to_aa(Rp @ Rg.transpose(-1, -2)).cpu().double()
            lens = PSD.run_lengths(s)
            ml = 20 if max(lens) > 40 else 5
            out.setdefault("mechanism", {})[s.name] = {
                "abs_head_rot_deg": float(torch.rad2deg(e_abs.norm(dim=1)).mean()),
                "s37_closed_rot_deg": float(torch.rad2deg(e_s37.norm(dim=1)).mean()),
                "err_vector_cos_mean": float(torch.nn.functional.cosine_similarity(e_abs, e_s37, dim=1).mean()),
                "err_norm_corr": float(np.corrcoef(e_abs.norm(dim=1).numpy(), e_s37.norm(dim=1).numpy())[0, 1]),
                "acf_abs_head": {k: v["rho"] for k, v in PSD.acf(e_abs, lens, ml).items() if k in (1, 5, 20)},
                "acf_s37_closed": {k: v["rho"] for k, v in PSD.acf(e_s37, lens, ml).items() if k in (1, 5, 20)}}
        for vn, per in out["variants"].items():
            per["frame_weighted_ra"] = sum(per[s.name]["ra"] * nfr[s.name] for s in zgz) / sum(nfr.values())
        res["R1"][n_] = out
        print("R1", n_, r3(out), f"{time.time() - t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / f"results_main_{RUN}.json").write_text(json.dumps(r3(res), indent=1, default=str))
    print("done main", res["seconds"], "s", flush=True)


# ------------------------------------------------------------------------------------ H5
@torch.no_grad()
def coverage(enc, batch):
    """Actual 1-hop / 3-hop ancestors of each sampled node of the S37 graph (weights irrelevant)."""
    events, ptr = batch.events, batch.ptr
    if events.shape[0] == 0:
        return None
    tok = event_tokens(events, ptr, batch.delta_t_s, enc.height, enc.width)
    src, mask = enc._sample(events, ptr)
    N = mask.shape[1]
    flat = src.reshape(-1)
    feat = tok[flat].reshape(1, N, -1) * mask.unsqueeze(-1)
    ev = events[flat].reshape(1, N, -1)
    p = torch.stack([ev[..., EV.EV_X] / enc.width, ev[..., EV.EV_Y] / enc.height,
                     feat[..., T_COL] * enc.t_scale], dim=-1) * mask.unsqueeze(-1)
    idx, _, emask = enc._edges(p.to(torch.float32), mask)
    n = int(mask.sum())
    idx, emask = idx[0, :n], emask[0, :n] > 0
    A = torch.eye(n, device=events.device)
    rows = torch.arange(n, device=events.device).unsqueeze(1).expand_as(idx)
    A[rows[emask], idx[emask]] = 1.0
    t_ms = ev[0, :n, EV.EV_T].float() * 1000.0
    xy = torch.stack([ev[0, :n, EV.EV_X], ev[0, :n, EV.EV_Y]], -1).float()
    out = {"n_events": int(events.shape[0]), "n_nodes": n, "retention": n / max(int(events.shape[0]), 1)}
    e_len = (xy[idx] - xy.unsqueeze(1)).norm(dim=-1)[emask]
    e_dt = (t_ms.unsqueeze(1) - t_ms[idx])[emask]
    out["edge_len_px_med"] = float(e_len.median()) if e_len.numel() else float("nan")
    out["edge_dt_ms_med"] = float(e_dt.median()) if e_dt.numel() else float("nan")
    R = A.clone()
    for hop in (1, 2, 3):
        if hop > 1:
            R = ((R @ A) > 0).float()
        if hop in (1, 3):
            anc = R > 0
            tmin = torch.where(anc, t_ms.unsqueeze(0).expand(n, n), torch.full_like(R, float("inf"))).min(1).values
            dist = torch.where(anc, torch.cdist(xy, xy), torch.zeros_like(R)).max(1).values
            out[f"hop{hop}_span_ms_med"] = float((t_ms - tmin).median())
            out[f"hop{hop}_radius_px_med"] = float(dist.median())
            out[f"hop{hop}_ancestors_med"] = float(anc.sum(1).float().median())
    return out


@torch.no_grad()
def h5_part(a):
    device = torch.device("cuda")
    torch.set_num_threads(8)
    t0 = time.time()
    s37s, cnns, _, cfg, cnn_cfg, ck = load_models(device)
    fkm = next(iter(s37s.values()))
    root = Path(cfg["DATA"]["ROOT"])
    mani = splits_manifest(cfg)
    zgz = [PSD.SeqNC(root, d, s, device) for s, d in sequences_for_split(root, "val_core", mani)]
    rng0 = np.random.default_rng(0)
    inits = {s.name: [ET.sample_init_noise(cfg, rng0, 1.0) for _ in s.runs] for s in zgz}
    base = next(s for s in zgz if "global" in s.name)
    if a.smoke:
        base = truncate(base, 40)
    lnes_ch = EV.event_channels(cnn_cfg)
    seqs = [PSD.ThinSeq(base, k) for k in KEEPS]
    for ts in seqs:
        ts.runs = base.runs
    rows = PSD.rows_of(base)
    res = {"keeps": KEEPS, "ckpts": ck, "run": RUN, "S37": {}, "CNN": {}, "coverage": {}}
    # S37: all keep rates advance together as one batch (packet j of the batch = keep rate j)
    for name, m in s37s.items():
        m.set_hand_context(base.betas, base.K)
        preds = [[] for _ in KEEPS]
        gts = []
        for (a0, ends), init in zip(base.runs, inits[base.name]):
            prev = T1(base.pos51[a0] + init, device).expand(len(KEEPS), -1).contiguous()
            for end in ends:
                evs = [ts.packet_events(end) for ts in seqs]
                out = m.forward_packet(PSD.packet_batch(evs, prev, base, device)).float()
                for j in range(len(KEEPS)):
                    preds[j].append(out[j:j + 1])
                gts.append(T1(base.pos51[end], device))
                prev = out
        gts = torch.cat(gts)
        for k, p in zip(KEEPS, preds):
            res["S37"].setdefault(str(k), {})[name] = summ(decomp(m, torch.cat(p), gts, base.betas))
        print("H5 S37", name, r3({k: v[name]["ra"] for k, v in res["S37"].items()}), f"{time.time() - t0:.0f}s", flush=True)
    G1 = PSD.T([base.pos51[e] for _, e in rows], device)
    for k, ts in zip(KEEPS, seqs):
        for name, c in cnns.items():
            outs = []
            for i in range(0, len(rows), 64):
                chunk = rows[i:i + 64]
                x = torch.from_numpy(np.stack([ET.build_lnes(ts.events, ts.offsets, int(e), STEP, lnes_ch)
                                               for _, e in chunk])).to(device)
                outs.append(c(x, torch.zeros(len(chunk), 51, device=device)).float())
            res["CNN"].setdefault(str(k), {})[name] = summ(decomp(fkm, torch.cat(outs), G1, base.betas))
        cov = []
        enc = fkm.event_encoder
        for _, e in rows[::10]:
            ev5 = ts.packet_events(e)
            b = ET.make_eval_packet(ev5, G1[:1], base.betas, base.K, STEP, device)
            c_ = coverage(enc, b)
            if c_ is not None:
                cov.append(c_)
        res["coverage"][str(k)] = {kk: float(np.nanmedian([c_[kk] for c_ in cov])) for kk in cov[0]} if cov else {}
        res["coverage"][str(k)]["n_packets"] = len(cov)
        print("H5", k, r3({"cnn": {n: v["ra"] for n, v in res["CNN"][str(k)].items()}, "cov": res["coverage"][str(k)]}),
              f"{time.time() - t0:.0f}s", flush=True)
    # natural sparse regime for reference: coverage on zgz_local (no thinning)
    loc = next(s for s in zgz if "local" in s.name)
    cov = []
    for _, e in PSD.rows_of(loc)[::10]:
        b = ET.make_eval_packet(loc.packet_events(e), G1[:1], loc.betas, loc.K, STEP, device)
        c_ = coverage(fkm.event_encoder, b)
        if c_ is not None:
            cov.append(c_)
    res["coverage"]["zgz_local_natural"] = {kk: float(np.nanmedian([c_[kk] for c_ in cov])) for kk in cov[0]}
    res["seconds"] = round(time.time() - t0, 1)
    (HERE / f"results_h5_{RUN}.json").write_text(json.dumps(r3(res), indent=1, default=str))
    print("done h5", res["seconds"], "s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("main", "h5"), required=True)
    ap.add_argument("--train-per-seq", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_per_seq = 30
    (main_part if args.part == "main" else h5_part)(args)
