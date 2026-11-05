#!/usr/bin/env python3
"""PCR checks, acceptance, readings and router probes (docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105). Readings go
through PFO's frozen engine (tools/pfo/engine.py, tools/pfo/nets.py, unchanged) with the output root redirected to
outputs/pcr/20261105; the new router is reached through the trunk's evaluation config (S45SM router block replaced).

    python -m tools.pcr.run consistency
        CPU: on 48 training packets of the routers' data, the part map drawn from the *target* pose must cover the
        labelled events (any part >= 0.85, own part >= 0.70) and not the unlabelled ones (any part <= 0.30)
    python -m tools.pcr.run accept --router R --phase P
    python -m tools.pcr.run read --trunk l1|b2 --router r_s2d|pcr|pcrb --task e1|e3|pert|f1
        frozen trunk (IDP L1 / PFO2 B2, seed 3407, step 6000) with the router's routes; tag <trunk>__<router>
    python -m tools.pcr.run check
        K: l1__r_s2d e1 equals IDP's idp_l1_s3407 e1 bitwise (this entry adds nothing for r_s2d)
    python -m tools.pcr.run probe --router R --seq S --prior loop|gt
        router accuracy against the XAG labels on every 2nd zgz endpoint, the packet's prior = L1's own closed-loop
        state (loop: IDP's idp_l1_s3407 e1 incoming states) or the GT state at the window start (gt)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pcr.common import (CONFIGS, L1_DEPLOYED_E1, OUT, R_S2D, TRUNKS, ZGZ, now, read, require, save,  # noqa: E402
                              sha)

ROUTER_NAMES = ("r_s2d", "pcr", "pcrb")
TASKS = ("e1", "e3", "pert", "f1")


def router_ckpt(name):
    if name == "r_s2d":
        return R_S2D
    from tools.pcr.train import checkpoint
    return checkpoint(name)


def accepted_ckpt(name):
    if name == "r_s2d":
        return R_S2D
    rc = read(OUT / "receipts" / f"router_{name}.json")
    require(rc["status"] == "PASS", "router not accepted: " + name)
    ck = router_ckpt(name)
    require(sha(ck) == rc["checkpoint_sha256"], "router checkpoint changed since acceptance")
    return ck


# --------------------------------------------------------------------------- consistency (CPU)
def consistency():
    import torch
    from semkine import pcr_router as PCR
    from semkine import s45sm as SM
    from semkine.events import collate_packets
    from semkine.xyz_mesh import stride_sample
    from tools.s45sm import router_train as S
    torch.set_num_threads(16)
    cfg = S._data({"dz": True})
    ds = S.dataset(cfg, "ylf")
    idx = np.random.default_rng(0).choice(len(ds), 48, replace=False)
    batch = collate_packets([ds[int(i)] for i in idx])
    net = PCR.build(cfg["MANO"]["NPZ"], stem="s2d")
    src, mask = stride_sample(batch.ptr, SM.MAX_NODES)
    tgt = SM.targets(net, batch, src, mask)
    res = {}
    for name, state in (("target", batch.target), ("prev", batch.prev_state)):
        m = PCR.prior_maps(state, batch.betas, batch.camera_K, net.vert_part)
        ev = batch.events[src.reshape(-1)].view(src.shape[0], src.shape[1], -1)
        x2 = (ev[..., 1].long() // 2).clamp(0, PCR.W2 - 1)
        y2 = (ev[..., 2].long() // 2).clamp(0, PCR.H2 - 1)
        b = torch.arange(src.shape[0]).view(-1, 1).expand_as(x2)
        anyc, own = m.max(1).values[b, y2, x2], m[b, tgt["part"], y2, x2]
        v = tgt["valid"]
        res[name] = {"labelled_any": float((anyc[v] > 0.05).float().mean()),
                     "labelled_own": float((own[v] > 0.05).float().mean()),
                     "unlabelled_any": float((anyc[mask & ~v] > 0.05).float().mean()), "n_labelled": int(v.sum())}
    t = res["target"]
    ok = {"labelled_any": t["labelled_any"] >= 0.85, "labelled_own": t["labelled_own"] >= 0.70,
          "unlabelled_any": t["unlabelled_any"] <= 0.30}
    receipt = {"check": "consistency", "readings": res, "checks": ok, "status": "PASS" if all(ok.values()) else "FAIL",
               "checked_at_utc": now()}
    save(OUT / "receipts" / "consistency.json", receipt, exclusive=True)
    print(receipt["status"], json.dumps(res), flush=True)
    require(receipt["status"] == "PASS", "prior map consistency failed")


# --------------------------------------------------------------------------- acceptance
def accept(name, phase):
    from tools.pcr.train import PARAMS, ROUTERS, checkpoint, router_dir
    require(name in ROUTERS and phase in ("preflight", "formal"), "router / phase")
    tag = f"router_{name}" + ("" if phase == "formal" else f"_{phase}")
    out = OUT / "receipts" / f"{tag}.json"
    require(not out.exists(), "accepted once")
    d, ck = router_dir(name, phase), checkpoint(name, phase)
    launch = read(d / "launch.json")
    done = read(ck.with_suffix(".json")) if ck.with_suffix(".json").exists() else {}
    steps = ROUTERS[name]["steps"] if phase == "formal" else 40
    ok = {"complete": done.get("status") == "COMPLETE" and int(done.get("step", -1)) == steps,
          "checkpoint": ck.exists() and done.get("checkpoint_sha256") == sha(ck),
          "spec": launch["spec"] == json.loads(json.dumps(ROUTERS[name])) and launch["phase"] == phase,
          "params": int(done.get("params", -1)) == PARAMS,
          "prior_source": ("n4 bank" in launch["prior_source"]) == bool(ROUTERS[name]["bank"])}
    if ok["checkpoint"]:
        from semkine import pcr_router as PCR
        net, obj = PCR.load(ck)
        ok["loads"] = obj["kind"] == "pcr" and obj["router"] == name
        ok["prior_in_trained"] = float(net.prior_in.weight.abs().sum()) > 0
    receipt = {"router": name, "phase": phase, "checks": ok, "status": "PASS" if all(ok.values()) else "FAIL",
               "checkpoint_sha256": sha(ck) if ck.exists() else None, "accepted_at_utc": now()}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


# --------------------------------------------------------------------------- readings
def write_once(path, obj):
    from tools.pfo2.specs import write_once_text
    if str(path).endswith(".json"):
        return write_once_text(path, json.dumps(obj, indent=2, allow_nan=False) + "\n")
    import yaml
    return write_once_text(path, yaml.safe_dump(obj, sort_keys=False))


def reading_spec(trunk, router):
    import yaml
    sp_path, cfg_path = TRUNKS[trunk]
    sp = read(sp_path)
    require(sha(sp["checkpoint"]) == sp.get("checkpoint_sha256", sha(sp["checkpoint"])), "trunk checkpoint changed")
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if router == "r_s2d":
        require(cfg["S45SM"]["ROUTER_KIND"] == "pfr" and Path(cfg["S45SM"]["ROUTER_CHECKPOINT"]) == R_S2D, "trunk router")
        cfg_out = cfg_path
    else:
        cfg["S45SM"].update(ROUTER=router, ROUTER_KIND="pcr", ROUTER_CHECKPOINT=str(accepted_ckpt(router)))
        cfg_out = write_once(CONFIGS / "eval" / f"{trunk}__{router}.yaml", cfg)
    tag = f"{trunk}__{router}"
    spec = {"kind": "routed", "checkpoint": sp["checkpoint"], "checkpoint_sha256": sha(sp["checkpoint"]),
            "config": str(cfg_out), "trunk": sp.get("trunk", "gshare"), "module": sp.get("module", "trunk4"),
            "router": router, "router_checkpoint_sha256": sha(router_ckpt(router)), "trunk_spec": str(sp_path)}
    return tag, write_once(OUT / "specs" / f"{tag}.json", spec)


def prepare(sp):
    from semkine import pcr_router as PCR
    from semkine import pfo_router as PR
    from tools.pfo2.train import install_trunk
    s = read(sp)
    PR.register()
    PCR.register()
    install_trunk(s["module"], s["trunk"])
    return s


def engine(trunk, router, task):
    from tools.pfo import engine as E
    tag, sp = reading_spec(trunk, router)
    prepare(sp)
    E.OUT = OUT
    return E.run(tag, task, None, None, (), str(sp))


def check():
    a = np.load(OUT / "readings" / "l1__r_s2d" / "e1" / "states.npz", allow_pickle=False)
    b = np.load(L1_DEPLOYED_E1 / "states.npz", allow_pickle=False)
    rows = {f"{s}/prediction": bool(np.array_equal(a[f"{s}/prediction"], b[f"{s}/prediction"])) for s in ZGZ}
    receipt = {"check": "k", "rows": rows, "status": "PASS" if all(rows.values()) else "FAIL", "checked_at_utc": now()}
    save(OUT / "receipts" / "k.json", receipt, exclusive=True)
    print(receipt["status"], rows, flush=True)
    require(receipt["status"] == "PASS", "engine check failed")


# --------------------------------------------------------------------------- router probe
def probe(router, seq, prior):
    import torch
    from config import load_config
    from semkine import eval_track as ET
    from semkine import pcr_router as PCR
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine.xyz_mesh import stride_sample
    from tools.pfo.common import XAG_SOURCE_ROOT
    from tools.pfo.engine import endpoint_rows, sequences
    from tools.pfo.nets import sequence
    require(seq in ZGZ and prior in ("loop", "gt"), "seq / prior")
    out_path = OUT / "probes" / f"{router}__{seq}__{prior}.json"
    require(not out_path.exists(), "probe written once")
    dev = torch.device("cuda")
    if router == "r_s2d":
        net, _ = PR.load(R_S2D)
    else:
        net, _ = PCR.load(accepted_ckpt(router))
    net = net.to(dev).eval()
    cfg = load_config(str(TRUNKS["l1"][1]))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    root, seqs = sequences(cfg, (seq,))
    _, directory = seqs[0]
    ctx = sequence(root, directory, seq, dev)
    rows = endpoint_rows(ctx.aux)
    loop = np.load(L1_DEPLOYED_E1 / "states.npz", allow_pickle=False)
    inc, ends = loop[f"{seq}/incoming"], loop[f"{seq}/end_ms"]
    require(len(inc) == len(rows) and all(int(e) == r[1] for e, r in zip(ends, rows)), "loop states / grid mismatch")
    g = lambda q: torch.where(q == 0, torch.zeros_like(q), (q - 1) // 3 + 1)  # noqa: E731
    tot = {"n": 0, "part": 0, "finger": 0, "onhand_routed": 0}
    errs = []
    with torch.no_grad():
        for i in range(0, len(rows), 2):
            _, end, prev_ms, _ = rows[i]
            state = inc[i] if prior == "loop" else np.asarray(ctx.gt[int(prev_ms)], np.float32)
            ev5 = ET._window_events(ctx.events, ctx.offsets, ctx.tsub, int(end), 50)
            if len(ev5) == 0:
                continue
            prev = torch.from_numpy(np.ascontiguousarray(state, np.float32)).view(1, -1).to(dev)
            b = ET.make_eval_packet(ev5, prev, ctx.betas, ctx.K, 50, dev)
            b = ET.attach_xag_sources(b, cfg, seq, ctx.offsets, int(end), 50, dev)
            src, mask = stride_sample(b.ptr, SM.MAX_NODES)
            if router == "r_s2d":
                out = FAST.router_forward(net, b.events, b.ptr, b.delta_t_s, src, mask)
            else:
                out = PCR.pcr_forward(net, b, src, mask)
            routed = mask & (torch.sigmoid(out["onhand"].float()) >= 0.5)
            face, bary = FAST.route(net, "sr", out, routed)
            lab = SM._NATIVE["sample_sources"](b, src, mask)
            F_ = net.faces.shape[0]
            lv = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
            lb = lab["xag_bary"].float()
            lv = lv & torch.isfinite(lb).all(-1) & (lb >= 0).all(-1) & (lb.sum(-1) > 0)
            lf = lab["xag_face"].clamp(0, F_ - 1)
            both = routed & lv
            lp, rp = net.face_part[lf], net.face_part[face]
            tot["n"] += int(both.sum())
            tot["part"] += int((both & (lp == rp)).sum())
            tot["finger"] += int((both & (g(lp) == g(rp))).sum())
            tot["onhand_routed"] += int(routed.sum())
            lpt = ((lb / lb.sum(-1, keepdim=True).clamp_min(1e-12)).unsqueeze(-1) * net.tmpl[net.faces[lf]]).sum(-2)
            rpt = (bary.float().unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
            ok = both & (lp == rp)
            if int(ok.sum()):
                errs.append(((rpt - lpt).norm(dim=-1)[ok] * 1e3).cpu().numpy())
    res = {"router": router, "seq": seq, "prior": prior, "nodes": tot["n"], "part_acc": tot["part"] / max(tot["n"], 1),
           "finger_acc": tot["finger"] / max(tot["n"], 1),
           "point_err_part_correct_median_mm": float(np.median(np.concatenate(errs))) if errs else None,
           "written_utc": now()}
    save(out_path, res, exclusive=True)
    print(json.dumps(res), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("consistency")
    s = sub.add_parser("accept")
    s.add_argument("--router", required=True, choices=("pcr", "pcrb"))
    s.add_argument("--phase", required=True, choices=("preflight", "formal"))
    s = sub.add_parser("read")
    s.add_argument("--trunk", required=True, choices=tuple(TRUNKS))
    s.add_argument("--router", required=True, choices=ROUTER_NAMES)
    s.add_argument("--task", required=True, choices=TASKS)
    sub.add_parser("check")
    s = sub.add_parser("probe")
    s.add_argument("--router", required=True, choices=ROUTER_NAMES)
    s.add_argument("--seq", required=True, choices=ZGZ)
    s.add_argument("--prior", required=True, choices=("loop", "gt"))
    a = p.parse_args(argv)
    if a.verb == "consistency":
        return consistency()
    if a.verb == "accept":
        return accept(a.router, a.phase)
    if a.verb == "read":
        return engine(a.trunk, a.router, a.task)
    if a.verb == "check":
        return check()
    return probe(a.router, a.seq, a.prior)


if __name__ == "__main__":
    main()
