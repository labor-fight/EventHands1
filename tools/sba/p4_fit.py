#!/usr/bin/env python3
"""SBA P4 fit (docs/EXPERIMENT_RECORDS.md#sba-p4-prereg-20261022): Poisson maximum likelihood of the emission weights
theta >= 0 (per channel of semkine/sba_emission.py) and the outlier fraction eta on the seven fit subjects, with a
smoothness penalty between neighbouring template patches; its strength is selected by the held-out likelihood per
event on the two selection subjects (ycy, ylf). zgz is not read.

Data parallel over GPUs 4-7 (torchrun, NCCL all-reduce of loss and gradient); full-batch L-BFGS on log-parameters.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 tools/sba/p4_fit.py --cache raw_b1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402

OUT = CM.OUT / "p4"
LAMBDAS = (0.0, 0.01, 0.1, 1.0)


def guard():
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not v or any(g not in ("4", "5", "6", "7") for g in v.split(",")):
        raise SystemExit(f"P4 runs on physical GPUs 4-7 only (CUDA_VISIBLE_DEVICES={v!r})")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID")


def load(cache_dir: Path, rank: int, world: int, dev):
    """All windows of the cache, partitioned across ranks; each rank keeps fit / selection tensors on its GPU."""
    recs = []
    meta = None
    for f in sorted(cache_dir.glob("shard*.pt")):
        d = torch.load(f)
        meta = {k: d[k] for k in ("n_ch", "n_patches", "smooth", "blur")}
        recs += d["recs"]
    recs.sort(key=lambda r: (r["seq"], r["end"]))
    mine = recs[rank::world]

    def pack(rs):
        if not rs:
            return None
        lens = torch.tensor([len(r["N"]) for r in rs])
        return {"N": torch.cat([r["N"] for r in rs]).float().to(dev),
                "F": torch.cat([r["F"] for r in rs]).float().to(dev),
                "BU": torch.cat([torch.stack([r["B"], r["U"]], 1) for r in rs]).to(dev),
                "F_tot": torch.stack([r["F_tot"] for r in rs]).sum(0).float().to(dev),
                "B_tot": float(sum(r["B_tot"] for r in rs)), "U_tot": float(sum(r["U_tot"] for r in rs)),
                "n_events": float(sum(r["n_crop"] for r in rs)), "n_windows": len(rs), "lens": lens}
    return meta, pack([r for r in mine if r["role"] == "fit"]), pack([r for r in mine if r["role"] == "sel"])


def nll(params, data, n_ch):
    th = params[:n_ch].exp()
    eta = params[n_ch].exp()
    lam = data["F"] @ th + data["BU"][:, 0] + eta * data["BU"][:, 1]
    return -(data["N"] * lam.clamp(min=1e-12).log()).sum() + data["F_tot"] @ th + data["B_tot"] + eta * data["U_tot"]


def patch_graph(dev, n_patches, k=6):
    """Symmetric k-NN adjacency of the template patches (rest-pose centroids)."""
    from mano_layer import ManoLayer
    from semkine import sba as SB
    mano = ManoLayer(str(CM.EF.MANO_NPZ), add_mean=False)
    topo = SB.Topology(mano.f)
    v = mano.v_template.float()
    mid = 0.5 * (v[topo.edges[:, 0]] + v[topo.edges[:, 1]])
    patch = torch.from_numpy(np.load(OUT / "patches.npy")).long()
    cen = torch.stack([mid[patch == i].mean(0) for i in range(n_patches)])
    d = torch.cdist(cen, cen)
    nn = d.topk(k + 1, largest=False).indices[:, 1:]
    pairs = {(min(i, j), max(i, j)) for i in range(n_patches) for j in nn[i].tolist()}
    return torch.tensor(sorted(pairs), dtype=torch.long, device=dev)


def allsum(x):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    guard()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    meta, fit, sel = load(OUT / "cache" / a.cache, rank, world, dev)
    n_ch, n_p = meta["n_ch"], meta["n_patches"]
    off = n_ch - n_p
    pairs = patch_graph(dev, n_p)
    tot_fit = allsum(torch.tensor([fit["n_events"] if fit else 0.0], device=dev)).item()
    tot_sel = allsum(torch.tensor([sel["n_events"] if sel else 0.0], device=dev)).item()
    t0 = time.time()
    results = {}
    best = None
    for lam_s in LAMBDAS:
        params = torch.full((n_ch + 1,), float(np.log(0.5)), device=dev)
        params[n_ch] = float(np.log(0.3))
        params.requires_grad_(True)
        opt = torch.optim.LBFGS([params], lr=1.0, max_iter=a.iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            local = (nll(params, fit, n_ch) if fit else params.sum() * 0) / tot_fit
            th = params[off:n_ch]
            local = local + lam_s * ((th[pairs[:, 0]] - th[pairs[:, 1]]) ** 2).mean() / world
            local.backward()
            allsum(params.grad)
            return allsum(local.detach().reshape(1).clone())[0]
        opt.step(closure)
        with torch.no_grad():
            lsel = nll(params, sel, n_ch) if sel else torch.zeros((), device=dev)
            lsel = allsum(lsel.reshape(1))[0].item() / tot_sel
            lfit = allsum((nll(params, fit, n_ch) if fit else torch.zeros((), device=dev)).reshape(1))[0].item() / tot_fit
        results[str(lam_s)] = {"fit_nll_per_event": lfit, "sel_nll_per_event": lsel,
                               "theta": params[:n_ch].exp().tolist(), "eta": float(params[n_ch].exp())}
        if best is None or lsel < results[best]["sel_nll_per_event"]:
            best = str(lam_s)
        if rank == 0:
            print(json.dumps({"lambda": lam_s, "fit": round(lfit, 4), "sel": round(lsel, 4),
                              "eta": round(float(params[n_ch].exp()), 3), "s": round(time.time() - t0)}), flush=True)
    # reference: the S0 model (one silhouette / one internal kappa, no material) refit on the same fit windows
    tie = torch.zeros(3, device=dev, requires_grad=True)

    def ref_params(t):
        p = torch.full((n_ch + 1,), -30.0, device=dev)
        return torch.cat([t[0].expand(6), t[1].expand(6), p[12:n_ch], t[2:3]])
    opt = torch.optim.LBFGS([tie], lr=1.0, max_iter=a.iters, line_search_fn="strong_wolfe")

    def closure_ref():
        opt.zero_grad()
        local = (nll(ref_params(tie), fit, n_ch) if fit else tie.sum() * 0) / tot_fit
        local.backward()
        allsum(tie.grad)
        return allsum(local.detach().reshape(1).clone())[0]
    opt.step(closure_ref)
    with torch.no_grad():
        rsel = allsum((nll(ref_params(tie), sel, n_ch) if sel else torch.zeros((), device=dev)).reshape(1))[0].item() / tot_sel
    if rank == 0:
        out = {"cache": a.cache, "meta": meta, "lambdas": results, "selected_lambda": best,
               "selected": results[best], "reference_contour_only_sel_nll_per_event": rsel,
               "gain_per_event_vs_reference_on_selection": rsel - results[best]["sel_nll_per_event"],
               "events_fit": tot_fit, "events_sel": tot_sel, "world": world, "wall_s": time.time() - t0}
        path = OUT / f"fit_{a.cache}.json"
        path.write_text(json.dumps(out, indent=1))
        CM.receipt(OUT / f"fit_{a.cache}", {"probe": "P4-fit", "cache": a.cache}, scripts=(__file__,))
        print(json.dumps({k: out[k] for k in ("selected_lambda", "reference_contour_only_sel_nll_per_event",
                                               "gain_per_event_vs_reference_on_selection")}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
