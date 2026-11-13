"""[#nopart-prereg-20261113] outputs/evhandtrack/analysis/nopart_20261113/capture_nopart.py with semkine/nopart.py enabled, so a
router trained without the part loss decodes with c_i alone; output under outputs/nopart/20261113/router_compare/capture.
Records per node: label / router selection, label / decoded part and template point, the raw c_i, the c_i-only decode.

Zero-training capture of a router's routes against the labels at the tracker's own nodes (GT prior, 50 ms grid).
Carrier: EvHandTrack seed 3407 in the packet form (only its packets and nodes are used). Descriptive, unregistered.

    CUDA_VISIBLE_DEVICES=<5|7> python router_capture.py --name r_s2d --rkind pfr --ckpt <router ckpt> [--seqs a,b]
        -> router_capture/<name>.npz : seq, pkt, l_ok, r_sel, lpart, rpart, lp (label template point, m), rp (router point)
"""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path("/data1/lyq/code/mesh/EventHands1")
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
OUT = ROOT / "outputs/nopart/20261113/router_compare/capture"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--rkind", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seqs", default="zgz_local,zgz_global,ylf_local,lyq_local")
    p.add_argument("--rdr", help="an RDR checkpoint (semkine/rdr_router.py): its routes are captured; the carrier keeps --rkind / --ckpt")
    p.add_argument("--zoom", type=float, default=1.0, help="scale the router's input events about their median (router only)")
    a = p.parse_args()
    import numpy as np
    import torch
    from semkine import nopart as NP
    NP.enable()
    from tools.pmr import arms_any4  # noqa: F401
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.pfo.common import read
    from tools.pmr import run as R
    from tools.pmr import run_rol2 as RR
    from tools.xcl import evaluate as XE
    XE.configure_environment()
    XE.set_policy()
    tag, sp = RR.arm_spec("d1r_rol5", 3407)
    R.prepare(sp)
    N._router_for = lambda route, cfg: (a.rkind, a.ckpt)
    spec = read(sp)
    spec = {**spec, "checkpoint": Path(spec["checkpoint"]), "config": Path(spec["config"])}
    dev = torch.device("cuda")
    net = N.load(tag, route="label_onhand", device=dev, spec=spec)
    from semkine import s45sm as SM
    from semkine import s45sm_fast as FAST
    from semkine import xag as XAG
    rec = {k: [] for k in ("seq", "pkt", "l_ok", "r_sel", "lpart", "rpart", "ppart", "upart", "lp", "rp", "up", "cp", "pconf")}
    rdr = None
    if a.rdr:
        from semkine import rdr_router as RD
        rdr = RD.load(a.rdr, map_location=dev)[0].to(dev).eval()
    cur = {"seq": 0, "pkt": 0}

    def capture(batch, src, mask, *, require_depth=False):
        st = SM._STATE
        rn = SM._net(src.device)
        ev = batch.events
        inb = torch.ones_like(mask)
        if a.zoom != 1.0 and ev.shape[0] > 0:
            from semkine.events import EV_X, EV_Y
            ev = ev.clone()
            cx, cy = ev[:, EV_X].median(), ev[:, EV_Y].median()
            x = (ev[:, EV_X] - cx) * a.zoom + cx
            y = (ev[:, EV_Y] - cy) * a.zoom + cy
            ok_xy = (x >= 0) & (x <= SM.WIDTH - 1) & (y >= 0) & (y <= SM.HEIGHT - 1)
            ev[:, EV_X] = x.round().clamp(0, SM.WIDTH - 1)
            ev[:, EV_Y] = y.round().clamp(0, SM.HEIGHT - 1)
            inb = ok_xy[src.reshape(-1)].view_as(mask)
        out = FAST.router_forward(rn, ev, batch.ptr, batch.delta_t_s, src, mask)
        r_sel = mask & inb & (torch.sigmoid(out["onhand"].float()) >= st["onhand_tau"])
        r_face, r_bary = FAST.route(rn, st["router"], out, mask)
        carrier = (r_sel, r_face, r_bary)
        if rdr is not None:
            o2 = rdr.forward_at(ev, batch.ptr, batch.delta_t_s, src, mask)
            r_sel = mask & inb & (torch.sigmoid(o2["onhand"].float()) >= st["onhand_tau"])
            r_face, r_bary = rdr.route(o2, mask)
        # unconstrained decode: c_i alone, nearest template face centroid over all faces
        cpt = out["coord"].float() * 0.01
        cflat = cpt.reshape(-1, 3)
        uface = torch.zeros(cflat.shape[0], dtype=torch.long, device=cflat.device)
        for ch in torch.arange(cflat.shape[0], device=cflat.device).split(8192):
            uface[ch] = (rn.centroid.unsqueeze(0) - cflat[ch].unsqueeze(1)).norm(dim=-1).argmin(1)
        uface = uface.view(cpt.shape[:-1])
        from semkine.s45tr_t0 import project_bary as _pb
        ubary = _pb(cpt, rn.tmpl[rn.faces[uface]])
        pr = torch.softmax(out["part"].float(), -1)
        ppart, pconf = pr.argmax(-1), pr.max(-1).values
        lab = SM._NATIVE["sample_sources"](batch, src, mask)
        F_ = rn.faces.shape[0]
        l_bary = lab["xag_bary"].float()
        l_ok = lab["xag_valid"] & mask & (lab["xag_face"] >= 0) & (lab["xag_face"] < F_)
        l_ok = l_ok & torch.isfinite(l_bary).all(-1) & (l_bary >= 0).all(-1) & (l_bary.sum(-1) > 0)
        l_face = lab["xag_face"].clamp(0, F_ - 1)
        l_bary = torch.where(l_ok.unsqueeze(-1), l_bary / l_bary.sum(-1, keepdim=True).clamp_min(1e-12), torch.zeros_like(l_bary))
        tri = rn.tmpl[rn.faces]
        lp = (l_bary.unsqueeze(-1) * tri[l_face]).sum(-2)
        rp = (r_bary.float().unsqueeze(-1) * tri[r_face]).sum(-2)
        up = (ubary.float().unsqueeze(-1) * tri[uface]).sum(-2)
        m = mask[0]
        n = int(m.sum())
        rec["seq"].append(np.full(n, cur["seq"], np.int16)); rec["pkt"].append(np.full(n, cur["pkt"], np.int32))
        rec["l_ok"].append(l_ok[0][m].cpu().numpy()); rec["r_sel"].append(r_sel[0][m].cpu().numpy())
        rec["lpart"].append(rn.face_part[l_face][0][m].cpu().numpy().astype(np.int8))
        rec["rpart"].append(rn.face_part[r_face][0][m].cpu().numpy().astype(np.int8))
        rec["lp"].append(lp[0][m].float().cpu().numpy()); rec["rp"].append(rp[0][m].float().cpu().numpy())
        rec["ppart"].append(ppart[0][m].cpu().numpy().astype(np.int8)); rec["upart"].append(rn.face_part[uface][0][m].cpu().numpy().astype(np.int8))
        rec["up"].append(up[0][m].float().cpu().numpy()); rec["cp"].append(cpt[0][m].float().cpu().numpy())
        rec["pconf"].append(pconf[0][m].float().cpu().numpy())
        c_sel, c_face, c_bary = carrier
        face = torch.where(c_sel, c_face, torch.zeros_like(c_face))
        zero = torch.zeros_like(face)
        return {"xag_source_index": zero, "xag_source_ms": zero, "xag_part": rn.face_part[face], "xag_face": face,
                "xag_bary": c_bary.to(torch.float32), "xag_valid": c_sel & mask}

    XAG.sample_sources = capture
    names = tuple(a.seqs.split(","))
    root, seqs = E.sequences(net.cfg, names)
    with torch.no_grad():
        for si, (seq, directory) in enumerate(seqs):
            ctx = N.sequence(root, directory, seq, dev)
            net.bind(ctx)
            rows = E.endpoint_rows(ctx.aux)
            for i, (_, end, prev_ms, _) in enumerate(rows):
                cur.update(seq=si, pkt=i)
                net.step(ctx, end, np.asarray(ctx.gt[int(prev_ms)], np.float32).copy())
            print(seq, len(rows), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / (f"{a.name}.npz" if a.zoom == 1.0 else f"{a.name}__zoom{a.zoom:.2f}.npz"), seqs=np.asarray(names), **{k: np.concatenate(v) for k, v in rec.items()})
    print("saved", a.name, flush=True)


if __name__ == "__main__":
    main()
