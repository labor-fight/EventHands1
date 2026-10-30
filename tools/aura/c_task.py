#!/usr/bin/env python3
"""AURA Stage C task-level probes (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): the association isolated under the
AGG778 R2 instrument (state-blind tail: prev48 = 0, the anchor's edge geometry kept, residual slot 0 except the "MR"
arms; bf16-mixed training, fp32 evaluation, deterministic algorithms; 4000 steps x 128, Adam 2e-3, warmup 300, cosine
to 2 %, clip 10; fit = the 7 fit subjects; evaluation on the selection subjects every 4th endpoint and all 2590 zgz
endpoints). The frozen C1 supplies FK, rays, tokens, its own association (c1) and the V1 single step; every arm gets a
fresh phi and tail, and arms differ only in the association that builds their records (`semkine/aura.py`):

    T0s                no events, no state (calibration)
    T6@X / T6s12@X     GT vertex displacement / L_REL on the support of association X (/ + N(0, 12 mm))
    M@X                fresh phi + C1 pooling on association X, residual slot 0
    MR@X               the same with the residual slot rebuilt by the arm's phi from X's residual mass (1 - q) mask
    occ / occ_tot / occ_surf / occD   occupancy inputs only (occD: donor-window events, c1 on the current anchor)
    M@shell            near records on 778 shell nodes (ShellTail);  MR@split: surface / near pooled apart (SplitTail)
    M@ctr              contour association, ctr flag as a zero-initialised side input
    j16                every event to its nearest projected joint node, joint slots straight into the readout

One process = one (regime, group) of `GROUPS`; all arms of a process train on the same batches and anchors. Arms of one
family and seed index start from the same values for their common parameters (`FAMILY_SEED`, extras built last).
Group `aa` repeats M@c1 with another batch order (loader seed 3408): the A/A calibration of training-order noise.

    CUDA_VISIBLE_DEVICES=4 CUDA_DEVICE_ORDER=PCI_BUS_ID python tools/aura/c_task.py --regime e3 --group occ \
        --out outputs/aura/c/probe/e3_occ
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.agg778 import b_task as BT  # noqa: E402
from tools.agg778.heads import Tail  # noqa: E402
from tools.aura import common as AC  # noqa: E402
from tools.aura import heads as H  # noqa: E402
from semkine import aura as AU  # noqa: E402

GROUPS = {
    ("e3", "cal"): {"T0s": 3, "T6@c1": 3, "T6s12@c1": 3, "T6@surf": 2, "T6@b32": 2, "T6@mM": 2, "T6@aura10": 2,
                    "T6@p2dvis": 2, "T6@A3": 2},
    ("e3", "occ"): {"M@c1": 3, "occ": 3, "occD": 3, "occ_tot": 2, "occ_surf": 2, "M@shufG": 3, "M@shufP": 3},
    ("e3", "geo"): {"M@p2dh": 3, "M@p2dvis": 2, "M@p2d": 1, "M@shell": 3, "M@vis": 2, "M@ctr": 1, "j16": 2},
    ("e3", "aura1"): {"MR@c1": 3, "MR@none": 2, "MR@surf": 3, "M@surf": 2, "MR@b8": 2, "MR@b32": 2},
    ("e3", "aura2"): {"MR@mM": 2, "MR@m10": 2, "MR@aura10": 3, "MR@m20": 1, "MR@aura20": 1, "MR@split": 1},
    ("e3", "oracle"): {"M@A1": 3, "M@A2": 2, "M@A3": 2},
    ("e3", "aa"): {"M@c1": 3},
    ("noise", "cal"): {"T0s": 3, "T6@c1": 3, "T6s12@c1": 2, "T6@surf": 2, "T6@b32": 2, "T6@aura10": 2, "T6@A3": 2},
    ("noise", "occ"): {"M@c1": 3, "occ": 3, "occD": 2, "M@shufG": 3, "M@shufP": 3},
    ("noise", "aura1"): {"MR@c1": 3, "MR@none": 2, "MR@surf": 3, "MR@b8": 2, "MR@b32": 2},
    ("noise", "aura2"): {"MR@mM": 2, "MR@m10": 2, "MR@aura10": 3, "MR@m20": 1, "MR@aura20": 1, "M@p2dh": 2,
                         "M@shell": 2},
    ("noise", "oracle"): {"M@A1": 3, "M@A2": 2, "M@A3": 2, "j16": 2},
    ("noise", "aa"): {"M@c1": 3},
}
#: E3 groups whose arms also get the sigma*-scaled-mesh zgz readout (descriptive)
E3S_ARMS = {"M@c1", "MR@c1", "MR@surf", "MR@b8", "MR@b32", "M@p2dh"}
LOADER_SEED = {"aa": 3408}
FAMILY_SEED = {"T0s": 50000, "T6": 50100, "T6s12": 50200, "M": 50300, "occ": 50400}
STEPS, BATCH, LR, WARMUP, LR_FLOOR, CLIP = 4000, 128, 2e-3, 300, 0.02, 10.0
NOISE_SEED_TRAIN, NOISE_SEED_EVAL, PERTURB_SEED = 3407, 777, 4242
NOISE_MM = 12.0
SIGMA_STAR = {"zgz_local": 0.94, "zgz_global": 0.96}
RULE_OF = {"surf": "surf", "b8": "b8", "b32": "b32", "mM": "mM", "m10": "m10", "m20": "m20", "aura10": "aura10",
           "aura20": "aura20", "p2d": "p2d", "p2dvis": "p2dvis", "p2dh": "p2dh", "vis": "vis"}


def lr_lambda(step, steps):
    if step < WARMUP:
        return float(step + 1) / WARMUP
    p = min(max((step - WARMUP) / float(steps - WARMUP), 0.0), 1.0)
    return LR_FLOOR + (1.0 - LR_FLOOR) * 0.5 * (1.0 + np.cos(np.pi * p))


def parse(arm):
    """(family, input kind, association) of an arm name."""
    if arm == "T0s":
        return "T0s", "T0s", None
    if arm in ("occ", "occ_tot", "occ_surf", "occD"):
        return "occ", arm, "c1"
    if arm == "j16":
        return "M", "j16", "j16"
    kind, assoc = arm.split("@")
    return {"T6": "T6", "T6s12": "T6s12", "M": "M", "MR": "M"}[kind], kind, assoc


def needs(spec):
    out = set()
    for a in spec:
        fam, kind, assoc = parse(a)
        if assoc in (None, "c1"):
            out.add("c1")
        elif assoc in ("split",):
            out.add("c1")
        else:
            out.add(assoc)
        if a == "occD":
            out.add("donor")
        if kind == "T6s12":
            out.add("R0s12")
    return out


# ------------------------------------------------------------------------------------------------ inputs
@torch.no_grad()
def seeds_of(items, dev):
    return torch.tensor([int(x["si"]) * 1000003 + int(x["k"]) for x in items], dtype=torch.long, device=dev)


@torch.no_grad()
def associations(frozen, batch, cap, items, need, dev, verts=None):
    """{name: association dict} of the needed associations on the anchor mesh (verts: an override, E3s)."""
    from semkine.xag import route_association, sample_sources
    from semkine.xyz_mesh import associate_rays, vertex_normals
    d, mask = cap["d"], cap["mask"]
    verts = cap["verts"] if verts is None else verts
    faces, vf = frozen.mano.f.long(), frozen.xyz.vert_faces
    intr = C.intrinsics(frozen, batch)
    f_px = (intr[0] * intr[1]).sqrt()
    out = {"c1": cap["assoc"] if verts is cap["verts"] else AU.associate("c1", d, mask, verts, faces, vf, intr)}
    c1 = out["c1"]
    vis = AU.visible(verts, intr) if need & {"p2dvis", "p2dh", "vis", "T6@p2dvis"} else None
    for name in need:
        if name in out or name in ("donor", "R0s12"):
            continue
        if name in RULE_OF:
            out[name] = AU.associate(RULE_OF[name], d, mask, verts, faces, vf, intr, c1=c1, vis=vis)
        elif name == "none":
            out[name] = AU.none_assoc(c1)
        elif name in ("shufG", "shufP"):
            out[name] = AU.shuffle_assoc(c1, d, mask, verts, seeds_of(items, dev),
                                         frozen.xyz.lbs_hard if name == "shufP" else None)[0]
        elif name == "ctr":
            out[name] = associate_rays(d, mask, verts, faces, vf, band=16.0 / f_px, front_tol=3.0 / f_px, z_tol=0.01,
                                       k=16, mode="contour", normals=vertex_normals(verts.float(), faces), gap=0.01)
        elif name in ("A1", "A2", "A3"):
            mode = {"A1": "part_route", "A2": "face_route", "A3": "full_oracle"}[name]
            src = sample_sources(batch, cap["src"], mask, require_depth=True)
            kw = {"source_depth": src["xag_depth"]} if name == "A3" else {}
            out[name], _ = route_association(mode, c1, d, mask, verts, faces, vf, frozen.xyz.lbs_w, 16.0 / f_px,
                                             3.0 / f_px, 0.01, source_part=src["xag_part"], source_face=src["xag_face"],
                                             source_bary=src["xag_bary"], source_valid=src["xag_valid"], **kw)
        elif name == "j16":
            J = torch.einsum("jv,bvc->bjc", AU.joint_weights(frozen.xyz.lbs_w), verts.float())
            out[name] = AU.joint_assoc(d, mask, J)
            out["joints"] = J
        elif name == "shell":
            out[name] = c1
            out["shell_pos"] = AU.shell_positions(verts, faces)
        else:
            raise ValueError(name)
    return out


@torch.no_grad()
def batch_inputs(frozen, items, anchors, dev, seed, need, sigma=False):
    batch = C.make_batch(items, anchors, dev)
    cap = C.forward_capture(frozen, batch)
    verts = cap["verts"]
    if sigma:
        betas_f, _ = frozen._resolve_betas_K(batch.prev_state, batch.betas, batch.camera_K)
        _, joints = frozen._fk(batch.prev_state.float(), betas_f)
        s = torch.tensor([SIGMA_STAR[x["seq"]] for x in items], device=dev).view(-1, 1, 1)
        z = verts[..., 2:3]
        xw = (joints[:, 0, :2] / joints[:, 0, 2:3]).unsqueeze(1)
        verts = torch.cat([(xw + s * (verts[..., :2] / z - xw)) * z, z], -1).float()
    B = cap["O"].shape[0]
    vgt = C.fk_verts(frozen, batch.target, batch)
    R0 = (vgt - verts) / 0.01
    inp = {"batch": batch, "cap": cap, "B": B, "R0": R0, "edge": frozen.xyz.edge_geometry(verts), "verts": verts,
           "own": cap["out"], "tok": cap["tok"], "d": cap["d"], "mask": cap["mask"]}
    inp["assoc"] = associations(frozen, batch, cap, items, need, dev, verts=None if not sigma else verts)
    if "R0s12" in need:
        g = torch.Generator(device=dev).manual_seed(int(seed))
        inp["R0s12"] = R0 + torch.randn(R0.shape, generator=g, device=dev) * (NOISE_MM * 1e-3 / 0.01)
    if "donor" in need:
        dbatch = C.make_batch(items, anchors, dev, events=[x["donor"]["ev5"] for x in items],
                              srcs=[x["donor"]["src"] for x in items])
        dcap = C.forward_capture(frozen, dbatch)
        inp["donor"] = dcap
    return inp


def support(assoc, mask):
    m_hit, m_near = AU.occupancy(assoc, mask, H.V)
    return ((m_hit + m_near) > 0).float()


# ------------------------------------------------------------------------------------------------ arms
class Arm(nn.Module):
    def __init__(self, name, frozen):
        super().__init__()
        self.name = name
        self.family, self.kind, self.assoc = parse(name)
        self.op = None
        xyz = frozen.xyz
        if self.family == "M":
            self.op = H.RecordOpX(side=1 if self.assoc == "ctr" else 0)
            tail_cls = {"shell": H.ShellTail, "split": H.SplitTail, "j16": H.JointTail}.get(self.assoc, Tail)
            self.tail = tail_cls(self.op.out_dim, xyz)
        elif self.family == "occ":
            self.tail = Tail({"occ": 3, "occD": 3, "occ_tot": 2, "occ_surf": 2}[self.kind], xyz)
        else:
            self.tail = Tail({"T0s": 1}.get(self.family, 3), xyz)

    def forward(self, inp):
        B, tok, d, mask = inp["B"], inp["tok"], inp["d"], inp["mask"]
        verts = inp["verts"]
        zeros_b = verts.new_zeros(B, 2 * 128 + 2)
        p48 = verts.new_zeros(B, 48)
        A = inp["assoc"]
        if self.family == "T0s":
            O = verts.new_zeros(B, H.V)
            delta = self.tail(verts.new_zeros(B, H.V, 1), O, inp["edge"], zeros_b, p48)
        elif self.family in ("T6", "T6s12"):
            O = support(A[self.assoc], mask)
            f = inp["R0" if self.family == "T6" else "R0s12"] * O.unsqueeze(-1)
            delta = self.tail(f, O, inp["edge"], zeros_b, p48)
        elif self.family == "occ":
            a = inp["donor"]["assoc"] if self.kind == "occD" else A["c1"]
            m = inp["donor"]["mask"] if self.kind == "occD" else mask
            f, O = H.occupancy_features(a, m, "occ" if self.kind == "occD" else self.kind)
            delta = self.tail(f * O.unsqueeze(-1), O, inp["edge"], zeros_b, p48)
        else:
            key = "c1" if self.assoc == "split" else self.assoc
            a = A[key]
            bslot = zeros_b
            if self.kind == "MR":
                r = (1.0 - a["q"]) * mask.float()
                bslot = H.residual_slot(self.op.rel, r, tok, mask)
            if self.assoc == "j16":
                rec = H.records(a, d, mask, A["joints"], tok)
                phi = self.op.phi(rec)
                js = H.joint_slots(phi, rec, B, mask.float().sum(1))
                delta = self.tail(js, bslot, p48)
            elif self.assoc == "shell":
                pos = A["shell_pos"]
                rec = H.records(a, d, mask, verts, tok, shell_pos=pos)
                x, O = self.op(rec, B, Vn=2 * H.V)
                delta = self.tail(x * (O > 0).unsqueeze(-1).to(x.dtype), O, pos, bslot, p48)
            elif self.assoc == "split":
                rec = H.records(a, d, mask, verts, tok)
                (xh, xn), O = self.op(rec, B, split=True)
                Om = (O > 0).unsqueeze(-1).to(xh.dtype)
                delta = self.tail((xh * Om, xn * Om), O, inp["edge"], bslot, p48)
            else:
                rec = H.records(a, d, mask, verts, tok)
                x, O = self.op(rec, B)
                delta = self.tail(x * (O > 0).unsqueeze(-1).to(x.dtype), O, inp["edge"], bslot, p48)
        return torch.where((inp["cap"]["counts"] <= 0).unsqueeze(-1), torch.zeros_like(delta), delta)


def build(spec, frozen, dev, steps):
    arms, names, opts, scheds = [], [], [], []
    for a, n in spec.items():
        for s in range(n):
            fam = parse(a)[0]
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.manual_seed(FAMILY_SEED[fam] + s)
                m = Arm(a, frozen).to(dev)
            o = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=LR)
            arms.append(m)
            names.append((a, s))
            opts.append(o)
            scheds.append(torch.optim.lr_scheduler.LambdaLR(o, lambda it, n_=steps: lr_lambda(it, n_)))
    return arms, names, opts, scheds


def train(args, frozen, cfg, dev, spec, need, log):
    fit = C.Endpoints(C.subject_seqs(C.FIT_SUBJ), with_donor="donor" in need)
    if args.smoke:
        fit.index = fit.index[:4 * args.batch]
    ld = C.loader(fit, args.batch, True, LOADER_SEED.get(args.group, 3407), workers=args.workers, drop_last=True)
    draw = C.noise_sampler(cfg)
    rng = np.random.default_rng(NOISE_SEED_TRAIN)
    arms, names, opts, scheds = build(spec, frozen, dev, args.steps)
    log.update(n_fit=len(fit), arms=[f"{a}/{s}" for a, s in names],
               params={f"{a}/{s}": sum(p.numel() for p in m.parameters() if p.requires_grad)
                       for m, (a, s) in zip(arms, names) if s == 0})
    run = np.zeros(len(arms))
    t0, it, curve, peak = time.time(), 0, [], 0.0
    while it < args.steps:
        for items in ld:
            if args.regime == "e3":
                anchors = np.stack([x["gt_prev"] for x in items])
            else:
                anchors = np.stack([x["gt_prev"] + draw(rng)[0] for x in items])
            inp = batch_inputs(frozen, items, anchors, dev, PERTURB_SEED + it, need)
            for k, (m, o, sc) in enumerate(zip(arms, opts, scheds)):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    delta = m(inp)
                loss, _ = frozen._compute_loss(inp["batch"].prev_state + delta.float(), inp["batch"].target)
                loss = loss.float().log10()
                o.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
                o.step()
                sc.step()
                run[k] += float(loss)
            if dev.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
            it += 1
            if it % 100 == 0 or it == args.steps:
                n = 100 if it % 100 == 0 else it % 100
                ent = {f"{a}/{s}": round(run[k] / n, 4) for k, (a, s) in enumerate(names)}
                curve.append({"it": it, **ent})
                print(it, json.dumps(ent), f"{time.time() - t0:.0f} s", f"peak {peak:.1f} GiB", flush=True)
                run[:] = 0
            if it >= args.steps:
                break
    log.update(loss_curve=curve, train_wall_s=time.time() - t0, peak_alloc_gib=peak)
    torch.save({"arms": [m.state_dict() for m in arms], "names": names}, args.out / "arms.pt")
    return arms, names


@torch.no_grad()
def evaluate(frozen, arms, names, ds, afn, need, dev, mano, workers, sigma=False):
    from tools.xcl.evaluate import score
    preds = [[] for _ in arms]
    base, tgt, meta, anc = [], [], [], []
    for items in C.loader(ds, 64, False, 0, workers=workers):
        anchors = np.stack([afn(x) for x in items])
        inp = batch_inputs(frozen, items, anchors, dev, PERTURB_SEED + 10 ** 6 + items[0]["i"], need, sigma=sigma)
        for k, m in enumerate(arms):
            preds[k].append((inp["batch"].prev_state + m(inp)).float().cpu().numpy())
        base.append(inp["own"].cpu().numpy())
        tgt.append(np.stack([x["gt_end"] for x in items]))
        anc.append(anchors.astype(np.float32))
        meta += [(x["seq"], x["si"], x["k"], x["rid"], x["blk"]) for x in items]
    out = BT.finish(preds, base, tgt, meta, ds, mano, dev, score)
    out["anchor"] = np.concatenate(anc)
    return out


def summary_of(res, names, key, summary):
    for dom in ("local", "global"):
        sel_ = np.array([("_" + dom) in s for s in res["seq"]])
        if sel_.any():
            summary[f"{key}/{dom}"] = {f"{a}/{s}": float(res["mpjpe_ra_mm"][j, sel_].mean()) for j, (a, s) in
                                       enumerate(names)}
            summary[f"{key}/{dom}"]["frozen_own"] = float(res["mpjpe_ra_mm"][-1, sel_].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("e3", "noise"), required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="tiny run (3 steps, 64 endpoints), not a result")
    ap.add_argument("--cpu", action="store_true", help="engineering smoke on the CPU (with --smoke only)")
    args = ap.parse_args()
    if args.cpu and not args.smoke:
        raise SystemExit("--cpu is for the engineering smoke only")
    dev = torch.device("cpu") if args.cpu else AC.require_gpu()
    spec = GROUPS[(args.regime, args.group)]
    args.steps, args.batch = (3, 8 if args.cpu else BATCH) if args.smoke else (STEPS, BATCH)
    if not args.smoke:
        AC.verify_protocol("C")
    C.set_policy()
    args.out = Path(args.out)
    if (args.out / "result.json").exists():
        raise SystemExit(f"{args.out} holds a result; preserve it")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(3407)
    frozen, cfg = C.load_frozen("A0", dev)
    mano = C.mano32(cfg, dev)
    need = needs(spec)
    log = {"regime": args.regime, "group": args.group, "spec": spec, "loader_seed": LOADER_SEED.get(args.group, 3407),
           "budget": {"steps": args.steps, "batch": args.batch, "lr": LR, "warmup": WARMUP, "clip": CLIP},
           "instrument": "AGG778 R2: state-blind tail (prev48 = 0), residual slot 0 except MR, bf16-mixed train, fp32 eval"}
    arms, names = train(args, frozen, cfg, dev, spec, need, log)
    for m in arms:
        m.eval()
    draw = C.noise_sampler(cfg)
    summary = {}
    don = "donor" in need
    evals = [("sel", C.Endpoints(C.subject_seqs(C.SELECT_SUBJ), stride=4, with_donor=don), False, None),
             ("zgz", C.Endpoints(C.zgz_seqs(), with_donor=don), False, None)]
    if args.regime == "noise":
        evals.append(("zgz_E2", C.Endpoints(C.zgz_seqs(), with_donor=don), False, "E2"))
    if args.regime == "e3" and any(a in E3S_ARMS for a in spec) and args.group != "aa":
        evals.append(("zgz_E3s", C.Endpoints(C.zgz_seqs(), with_donor=don), True, None))
    trace = C.strict_trace()
    for name, ds, sigma, special in evals:
        if args.smoke:
            ds.index = ds.index[:16 if args.cpu else 64]
        rng = np.random.default_rng(NOISE_SEED_EVAL)
        noise = {i: draw(rng) for i in range(len(ds))}
        if special == "E2":
            afn = lambda it: trace[it["seq"]][it["k"]]                                    # noqa: E731
        elif args.regime == "e3":
            afn = lambda it: it["gt_prev"]                                                # noqa: E731
        else:
            afn = lambda it, nz=noise: it["gt_prev"] + nz[it["i"]][0]                     # noqa: E731
        res = evaluate(frozen, arms, names, ds, afn, need, dev, mano, 0 if args.cpu else args.workers, sigma=sigma)
        if args.regime == "noise" and special is None:
            res["noise_kind"] = np.array([noise[i][1] for i in range(len(ds))])
        np.savez_compressed(args.out / f"eval_{name}.npz", **res)
        summary_of(res, names, name, summary)
    log["summary_mpjpe_ra"] = summary
    log["names"] = [f"{a}/{s}" for a, s in names]
    C.save_json(args.out / "result.json", log, exclusive=False)
    AC.receipt(args.out, {"stage": "C", "regime": args.regime, "group": args.group, "smoke": bool(args.smoke)},
               scripts=(__file__, Path(__file__).with_name("heads.py"), C.REPO / "semkine" / "aura.py"))
    print("COMPLETE", args.out, flush=True)


if __name__ == "__main__":
    main()
