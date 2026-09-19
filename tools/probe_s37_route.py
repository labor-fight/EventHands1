#!/usr/bin/env python3
"""S37 arms (routed readout / mesh query): the five pre-registered mechanism probes.

Everything is measured on the held-out subject's sequences (val_core = zgz) with the trained
checkpoints; nothing here trains. For each run the tool reports, in the order of the hypotheses
in docs/S37_ROUTED_READOUT_PREREG.md section 5:

  H1  routing quality (recall = share of nodes inside the 16 px band, purity = share of routed
      nodes whose argmax joint agrees with the routing under the ground-truth prev) with the
      previous state taken as ground truth / curriculum small noise / curriculum large noise /
      the model's own closed-loop prev; plus the same-checkpoint evidence ablation (zeroed e_j)
      and an event-swap test (events replaced by another packet's, 1 s away in the same run).
  H2  closed-loop RA with the model's own prev, and with *oracle routing* (routing computed from
      the ground-truth prev while everything else still sees the model's own prev).
  H3  absolute error decomposed into wrist translation / global rotation / root-aligned fingers,
      and the root head's own response to a prev perturbation with the events held fixed.
  H4  teacher-forced single-step RA on a few training sequences against the held-out ones.
  H5  teacher-forced update ratio (applied / needed) and direction cosine of the pose update.

Runs without an evidence path (S36) get every probe that does not need one, so the arms are
compared on identical packets. `has_evidence(model)` is true for both S37 arms; the node -> joint
responsibility comes from `model._node_responsibility` (hard LBS routing, or attention mass
carried by skinning weights for the mesh query).

    CUDA_VISIBLE_DEVICES=6 python tools/probe_s37_route.py \\
        --runs outputs/semkine/s36_eventgnn_s3407 outputs/semkine/s36_eventgnn_s3408 \\
               outputs/semkine/s37_routed_s3407 outputs/semkine/s37_routed_s3408 \\
        --out outputs/semkine/probe_s37_route.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                      # noqa: E402
from model import MNISTModel                                        # noqa: E402
from semkine import eval_track as ET                                # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest    # noqa: E402
from semkine.events import EV_X, EV_Y                               # noqa: E402

SMALL = (0.005, 0.05, 0.05)       # TRACK.PREV_NOISE_T / R / POSE of the S36 recipe
LARGE = (0.05, 0.3, 0.3)          # TRACK.PREV_NOISE_LARGE_*, 30% of training samples
TRAIN_SEQS = ("lyq_local", "lr_global", "ch_local", "ylf_global")


# --------------------------------------------------------------------------- helpers
def load_run(spec: str, device):
    """`RUN_DIR` (its selected checkpoint) or `RUN_DIR:CKPT` (an explicit checkpoint, smoke tests)."""
    run_dir, _, ckpt = spec.partition(":")
    run = REPO / run_dir
    if ckpt:
        sel = {"ckpt": ckpt, "step": None}
    else:
        sels = sorted(run.glob("selection_val_core_step50*.json"))
        if not sels:
            raise SystemExit(f"{run_dir}: no selection JSON; run tools/select_checkpoint.py first")
        sel = json.loads(sels[-1].read_text())["selected"]
    cfg = load_config(json.loads((run / "training_metadata.json").read_text())["config_path"])
    model = MNISTModel.load_from_checkpoint(sel["ckpt"], cfg=cfg, map_location=device)
    return model.to(device).eval(), cfg, sel


def has_evidence(model) -> bool:
    return bool(getattr(model, "routed", False) or getattr(model, "mesh_query", False))


def arm_of(run_name: str) -> str:
    return re.sub(r"_s\d{4}$", "", run_name)


def noise51(rng, scales):
    n = np.zeros(51, np.float32)
    n[0:3] = rng.standard_normal(3) * scales[0]
    n[3:6] = rng.standard_normal(3) * scales[1]
    n[6:51] = rng.standard_normal(45) * scales[2]
    return n


def rot_angle_deg(aa_a: torch.Tensor, aa_b: torch.Tensor) -> torch.Tensor:
    """Geodesic angle between two axis-angle rotations, in degrees (batched)."""
    def to_mat(aa):
        th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        k = aa / th
        K = torch.zeros(aa.shape[0], 3, 3, device=aa.device)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        th = th.unsqueeze(-1)
        return torch.eye(3, device=aa.device) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    R = to_mat(aa_a).transpose(1, 2) @ to_mat(aa_b)
    tr = (R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp(-1.0, 3.0)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1.0, 1.0)))


class Recorder:
    """State hook for `ET.track_sequence`: records the loop's own prev and pred per step and,
    in oracle mode, tells the routed model to route with the ground-truth prev of that step."""

    def __init__(self, model, gt_by_run, oracle=False):
        self.model, self.gt_by_run, self.oracle = model, gt_by_run, oracle
        self.run = -1
        self.step = 0
        self.prevs, self.preds = [], []

    def __call__(self, tag, prev_t, pred):
        if tag == "run_start":
            self.run += 1
            self.step = 0
            if self.oracle:
                self.model.route_prev_override = torch.from_numpy(
                    self.gt_by_run[self.run]["init"].copy()).view(1, -1)
            return pred
        if tag == "step" and pred is not None:
            self.prevs.append(prev_t.detach().cpu().numpy()[0].copy())
            self.preds.append(pred.detach().cpu().numpy()[0].copy())
            if self.oracle:
                gts = self.gt_by_run[self.run]["gts"]
                nxt = gts[min(self.step, len(gts) - 1)]
                self.model.route_prev_override = torch.from_numpy(nxt.copy()).view(1, -1)
            self.step += 1
        return pred


def runs_of(aux, pos51, step_ms):
    """The valid runs of a sequence exactly as `ET.track_sequence` enumerates them."""
    out = []
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if len(ends):
            out.append({"a": int(a), "ends": ends, "init": pos51[a], "gts": [pos51[e] for e in ends]})
    return out


# --------------------------------------------------------------------------- probes
@torch.no_grad()
def closed_loop(model, cfg, root, seqs, step_ms, device, oracle=False, ablate=False):
    """Recursive tracking on the sequences; returns pooled RA/abs and the decomposition."""
    if has_evidence(model):
        model.ablate_evidence = bool(ablate)
    ra, ab, n, wrist, rot, prevs_all, preds_all, gts_all = 0.0, 0.0, 0, [], [], [], [], []
    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        runs = runs_of(aux, pos51, step_ms)
        rec = Recorder(model, runs, oracle=oracle and has_evidence(model))
        r = ET.track_sequence(model, model.mano, cfg, root, d, seq, step_ms, device,
                              np.random.default_rng(0), 1.0, 1.0, None, state_hook=rec)
        if has_evidence(model):
            model.route_prev_override = None
        if r is None:
            continue
        ra += r["mpjpe_ra_mm"] * r["n_frames"]
        ab += r["mpjpe_abs_mm"] * r["n_frames"]
        n += r["n_frames"]
        gts = np.concatenate([np.stack(x["gts"]) for x in runs])
        preds = np.stack(rec.preds)
        assert len(preds) == len(gts), (len(preds), len(gts))
        p, g = torch.from_numpy(preds).to(device), torch.from_numpy(gts).to(device)
        wrist.append(((p[:, :3] - g[:, :3]).norm(dim=-1) * 1000).cpu().numpy())
        rot.append(rot_angle_deg(p[:, 3:6], g[:, 3:6]).cpu().numpy())
        prevs_all.append(np.stack(rec.prevs))
        preds_all.append(preds)
        gts_all.append(gts)
    if has_evidence(model):
        model.ablate_evidence = False
    wrist, rot = np.concatenate(wrist), np.concatenate(rot)
    return {"ra_mm": ra / max(n, 1), "abs_mm": ab / max(n, 1), "n_frames": n,
            "wrist_mm_p50": float(np.median(wrist)), "wrist_mm_p90": float(np.percentile(wrist, 90)),
            "rot_deg_p50": float(np.median(rot)), "rot_deg_p90": float(np.percentile(rot, 90))}, \
        (np.concatenate(prevs_all), np.concatenate(preds_all), np.concatenate(gts_all))


@torch.no_grad()
def teacher_forced(model, cfg, root, seqs, step_ms, device, max_packets=None, closed_prevs=None,
                   swap_gap_ms=1000, seed=0):
    """One pass over the packets with GT prev: TF accuracy, update ratio, event-swap dependence,
    evidence ablation and (routed arms) routing quality under the four prev conditions."""
    routed = has_evidence(model)
    rng = np.random.default_rng(seed)
    gap = max(1, swap_gap_ms // step_ms)
    tf_ra, tf_abs, up_ratio_num, up_ratio_den, cos = [], [], [], [], []
    swap_disp, true_step = [], []
    abl_ra = []
    route = {c: {"in_band": 0, "agree": 0, "routed_gt": 0, "live": 0}
             for c in ("gt", "small", "large", "closed")}
    k_closed = 0
    n_done = 0

    def ra_of(pred, gt, betas):
        jp = model._fk(pred.float(), betas)[1]
        jg = model._fk(gt.float(), betas)[1]
        jp = jp - jp[:, :1]
        jg = jg - jg[:, :1]
        return float((jp - jg).norm(dim=-1).mean()) * 1000

    def abs_of(pred, gt, betas):
        jp = model._fk(pred.float(), betas)[1]
        jg = model._fk(gt.float(), betas)[1]
        return float((jp - jg).norm(dim=-1).mean()) * 1000

    def routing(prev, ev5, betas, K):
        pk = ET.make_eval_packet(ev5, prev, betas, K, step_ms, device)
        src, mask = model.event_encoder._sample(pk.events, pk.ptr)
        flat = src.reshape(-1)
        px = pk.events[flat, EV_X].reshape(mask.shape).float() * mask
        py = pk.events[flat, EV_Y].reshape(mask.shape).float() * mask
        bf, kf = model._resolve_betas_K(prev, betas, K)
        a = model._node_responsibility(px, py, mask, prev, bf, kf)
        return a[0], mask[0]

    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        tsub_p = root / d / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        model.set_hand_context(betas, K)
        for run in runs_of(aux, pos51, step_ms):
            ends = run["ends"]
            prev_idx = run["a"]
            cache = {}

            def window(end):
                if end not in cache:
                    cache[end] = ET._window_events(events, offsets, tsub, int(end), step_ms)
                return cache[end]

            for i, end in enumerate(ends):
                if max_packets is not None and n_done >= max_packets:
                    break
                prev = torch.from_numpy(pos51[prev_idx].copy()).view(1, -1).to(device)
                gt = torch.from_numpy(pos51[int(end)].copy()).view(1, -1).to(device)
                ev_t = window(int(end))
                pk = ET.make_eval_packet(ev_t, prev, betas, K, step_ms, device)
                f0 = model.forward_packet(pk)
                tf_ra.append(ra_of(f0, gt, betas))
                tf_abs.append(abs_of(f0, gt, betas))
                dp, dn = (f0 - prev)[0, 6:], (gt - prev)[0, 6:]
                up_ratio_num.append(float(dp.norm()))
                up_ratio_den.append(float(dn.norm()))
                cos.append(float(torch.dot(dp, dn) / (dp.norm() * dn.norm() + 1e-9)))
                jg = model._fk(gt.float(), betas)[1]
                jp = model._fk(prev.float(), betas)[1]
                true_step.append(float((jg - jp).norm(dim=-1).mean()) * 1000)
                j = i + gap if i + gap < len(ends) else i - gap
                if 0 <= j < len(ends):
                    f2 = model.forward_packet(ET.make_eval_packet(window(int(ends[j])), prev, betas, K,
                                                                  step_ms, device))
                    j0, j2 = model._fk(f0.float(), betas)[1], model._fk(f2.float(), betas)[1]
                    swap_disp.append(float((j2 - j0).norm(dim=-1).mean()) * 1000)
                if routed:
                    model.ablate_evidence = True
                    fa = model.forward_packet(pk)
                    model.ablate_evidence = False
                    abl_ra.append(ra_of(fa, gt, betas))
                    a_gt, m = routing(prev, ev_t, betas, K)
                    hard_gt = a_gt.argmax(-1)
                    routed_gt = a_gt.sum(-1) > 0
                    conds = {"gt": prev,
                             "small": prev + torch.from_numpy(noise51(rng, SMALL)).to(device).view(1, -1),
                             "large": prev + torch.from_numpy(noise51(rng, LARGE)).to(device).view(1, -1)}
                    if closed_prevs is not None and k_closed < len(closed_prevs):
                        conds["closed"] = torch.from_numpy(closed_prevs[k_closed].copy()).view(1, -1).to(device)
                    for c, pv in conds.items():
                        a_c, _ = routing(pv, ev_t, betas, K)
                        in_band = a_c.sum(-1) > 0
                        agree = (a_c.argmax(-1) == hard_gt) & in_band & routed_gt
                        route[c]["in_band"] += int(in_band.sum())
                        route[c]["agree"] += int(agree.sum())
                        route[c]["routed_gt"] += int(routed_gt.sum())
                        route[c]["live"] += int(m.sum())
                k_closed += 1
                n_done += 1
                prev_idx = int(end)

    out = {"n_packets": n_done,
           "tf_ra_mm": float(np.mean(tf_ra)), "tf_abs_mm": float(np.mean(tf_abs)),
           "update_ratio_pose": float(np.sum(up_ratio_num) / max(np.sum(up_ratio_den), 1e-9)),
           "alignment_pose": float(np.mean(cos)),
           "true_step_mm": float(np.mean(true_step)),
           "event_swap_displacement_mm": float(np.mean(swap_disp)) if swap_disp else None,
           "event_dependence": float(np.mean(swap_disp) / max(np.mean(true_step), 1e-9)) if swap_disp else None}
    if routed:
        out["tf_ra_evidence_zeroed_mm"] = float(np.mean(abl_ra))
        out["routing"] = {c: {"recall_in_band": v["in_band"] / max(v["live"], 1),
                              "purity_vs_gt": v["agree"] / max(v["routed_gt"], 1),
                              "n_nodes": v["live"]} for c, v in route.items() if v["live"] > 0}
    return out


@torch.no_grad()
def root_gain(model, cfg, root, seqs, step_ms, device, max_packets=400):
    """Root head's own correction of a prev perturbation with the events fixed.

    For a perturbation `d` of the previous state's root block, the reported number is the signed
    ratio -<delta_root(prev+d) - delta_root(prev), d> / |d|^2 of the root head's output (the
    `prev_mlp` contribution is subtracted): 1 = fully corrected, 0 = blind, <0 = follows the error.
    """
    trans = torch.zeros(51, device=device)
    trans[0] = 0.036
    rots = []
    for ax in range(3):
        r = torch.zeros(51, device=device)
        r[3 + ax] = math.radians(10.0)
        rots.append(r)
    perts = {"trans_36mm": trans, "rot_x_10deg": rots[0], "rot_y_10deg": rots[1], "rot_z_10deg": rots[2]}
    acc = {k: [] for k in perts}
    n = 0

    def head_delta(pk):
        out = model.forward_packet(pk)
        delta = out - pk.prev_state
        if model.prevpos_embed:
            delta = delta - model.prev_mlp(pk.prev_state)
        return delta[0]

    for seq, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, seq)
        tsub_p = root / d / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        model.set_hand_context(betas, K)
        for run in runs_of(aux, pos51, step_ms):
            for end in run["ends"][::4]:            # every 4th packet is plenty for a response test
                if n >= max_packets:
                    break
                # GT state one step before the window: the conditioning a tracker would see
                prev = torch.from_numpy(pos51[max(int(end) - step_ms, run["a"])].copy()).view(1, -1).to(device)
                ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                if ev5.shape[0] == 0:
                    continue
                base = head_delta(ET.make_eval_packet(ev5, prev, betas, K, step_ms, device))
                for k, dvec in perts.items():
                    resp = head_delta(ET.make_eval_packet(ev5, prev + dvec.view(1, -1), betas, K, step_ms, device))
                    diff = (resp - base)[:6]
                    dd = dvec[:6]
                    acc[k].append(float(-(diff * dd).sum() / (dd * dd).sum()))
                n += 1
    return {k: float(np.mean(v)) for k, v in acc.items()} | {"n_packets": n}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--train-seqs", nargs="*", default=list(TRAIN_SEQS))
    ap.add_argument("--train-max-packets", type=int, default=1500)
    ap.add_argument("--control", default="s36_eventgnn")
    ap.add_argument("--smoke", action="store_true", help="first sequence, few packets, no train seqs")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = torch.device("cuda")

    rows = {}
    for run_dir in a.runs:
        name = Path(run_dir.partition(":")[0]).name
        t0 = time.time()
        model, cfg, sel = load_run(run_dir, device)
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, splits_manifest(cfg))
        if a.smoke:
            seqs = seqs[:1]
        routed = has_evidence(model)
        r = {"ckpt": sel["ckpt"], "step": sel.get("step"), "routed": bool(routed),
             "arm_kind": "mesh_query" if getattr(model, "mesh_query", False) else ("routed" if getattr(model, "routed", False) else "pooled")}

        closed, (prevs, _, _) = closed_loop(model, cfg, root, seqs, a.step_ms, device)
        r["closed"] = closed
        if routed:
            r["closed_evidence_zeroed"], _ = closed_loop(model, cfg, root, seqs, a.step_ms, device, ablate=True)
            r["closed_oracle_routing"], _ = closed_loop(model, cfg, root, seqs, a.step_ms, device, oracle=True)
        r["tf"] = teacher_forced(model, cfg, root, seqs, a.step_ms, device, closed_prevs=prevs,
                                 max_packets=200 if a.smoke else None)
        r["root_gain"] = root_gain(model, cfg, root, seqs, a.step_ms, device, max_packets=40 if a.smoke else 400)

        train_all = sequences_for_split(root, "train", splits_manifest(cfg))
        picked = [] if a.smoke else [(s, d) for s, d in train_all if s in set(a.train_seqs)]
        r["train_tf"] = {}
        for s, d in picked:
            tr = teacher_forced(model, cfg, root, [(s, d)], a.step_ms, device, max_packets=a.train_max_packets)
            r["train_tf"][s] = {"tf_ra_mm": tr["tf_ra_mm"], "n_packets": tr["n_packets"]}
        if picked:
            tr_mean = float(np.mean([v["tf_ra_mm"] for v in r["train_tf"].values()]))
            r["tf_gap_zgz_minus_train_mm"] = r["tf"]["tf_ra_mm"] - tr_mean
            r["train_tf_mean_mm"] = tr_mean
        r["seconds"] = round(time.time() - t0, 1)
        rows[name] = r
        print(f"{name}: closed RA {closed['ra_mm']:.2f} abs {closed['abs_mm']:.1f} | TF RA {r['tf']['tf_ra_mm']:.2f} "
              f"| amp x{closed['ra_mm'] / max(r['tf']['tf_ra_mm'], 1e-9):.2f} | wrist p50 {closed['wrist_mm_p50']:.1f} "
              f"rot p50 {closed['rot_deg_p50']:.1f} | event_dep {r['tf']['event_dependence']:.2f} "
              f"| gain t {r['root_gain']['trans_36mm']:.2f} rz {r['root_gain']['rot_z_10deg']:.2f} "
              f"| tf gap {r.get('tf_gap_zgz_minus_train_mm', float('nan')):.2f}", flush=True)
        if routed:
            print(f"    evidence zeroed: closed RA {r['closed_evidence_zeroed']['ra_mm']:.2f} "
                  f"(TF {r['tf']['tf_ra_evidence_zeroed_mm']:.2f}); oracle routing: closed RA "
                  f"{r['closed_oracle_routing']['ra_mm']:.2f}", flush=True)
            for c, v in r["tf"]["routing"].items():
                print(f"    routing[{c:>6}]: recall {v['recall_in_band']:.3f} purity {v['purity_vs_gt']:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # arm means and the H-verdict inputs
    arms = {}
    for name, r in rows.items():
        arms.setdefault(arm_of(name), []).append(r)

    def m(vals):
        return float(np.mean(vals))

    summary = {}
    for k, v in arms.items():
        s = {"n_seeds": len(v),
             "closed_ra": m([x["closed"]["ra_mm"] for x in v]),
             "closed_abs": m([x["closed"]["abs_mm"] for x in v]),
             "tf_ra": m([x["tf"]["tf_ra_mm"] for x in v]),
             "amplification": m([x["closed"]["ra_mm"] / x["tf"]["tf_ra_mm"] for x in v]),
             "wrist_mm_p50": m([x["closed"]["wrist_mm_p50"] for x in v]),
             "rot_deg_p50": m([x["closed"]["rot_deg_p50"] for x in v]),
             "event_dependence": m([x["tf"]["event_dependence"] for x in v]),
             "update_ratio_pose": m([x["tf"]["update_ratio_pose"] for x in v]),
             "alignment_pose": m([x["tf"]["alignment_pose"] for x in v]),
             "root_gain": {kk: m([x["root_gain"][kk] for x in v]) for kk in v[0]["root_gain"] if kk != "n_packets"},
             "tf_gap_zgz_minus_train_mm": m([x["tf_gap_zgz_minus_train_mm"] for x in v if "tf_gap_zgz_minus_train_mm" in x])}
        if v[0]["routed"]:
            s["closed_ra_evidence_zeroed"] = m([x["closed_evidence_zeroed"]["ra_mm"] for x in v])
            s["closed_ra_oracle_routing"] = m([x["closed_oracle_routing"]["ra_mm"] for x in v])
            s["routing"] = {c: {kk: m([x["tf"]["routing"][c][kk] for x in v if c in x["tf"]["routing"]])
                                for kk in ("recall_in_band", "purity_vs_gt")}
                            for c in v[0]["tf"]["routing"]}
        summary[k] = s

    out = {"protocol": {"split": a.split, "step_ms": a.step_ms, "train_seqs": a.train_seqs,
                        "train_max_packets": a.train_max_packets, "control": a.control,
                        "small_noise": SMALL, "large_noise": LARGE},
           "runs": rows, "arms": summary, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
