#!/usr/bin/env python3
"""S45-TR mechanism readouts (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030), one network at a time.

  readouts.py run --net <s45|arm> --task <task>    one GPU among 2-7
  readouts.py analyze --net <s45|arm>              CPU; writes readouts/<net>/summary.json

The four readouts are S45-LOC / S45-DEEP's, with the same tasks, draws and formulas, for any network:
  1  prior retention: GT previous + (translation 10 mm | root 5 deg | fingers N(0, 0.05^2)), one step, parameter-space
     survival rho_param (tools/s45loc/analyze.py retention); 1b on / off-manifold: fingers + 0.25 rad along the top-5 /
     bottom-15 training pose components (S45-DEEP f3)
  2  one-step shrinkage toward the training mean pose: k of the GT-previous step (S45-DEEP `shrink`), all endpoints and
     the lowest finger-motion quartile
  3  the same packet fed 20 times from the GT / mean-pose / own-loop state (S45-DEEP f1, every 8th endpoint)
  4  GT-pose clamp share (E1 - E1_gtpose) / (E1 - E3), GT-translation clamp as the control (S45-LOC amendment 1)
on zgz_local and the nine training local recordings. For `--net s45` every array must equal the published S45-LOC /
S45-DEEP array bit for bit (`analyze` checks it) before any arm's readout is read.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45loc import common as L  # noqa: E402
from tools.s45tr.common import (FORMAL_ARMS, OUT, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG, now, read,  # noqa: E402
                                require, save, sha, verify_protocol)

DEEP = ROOT / "outputs/s45loc/20261029/deep"
LOC = ROOT / "outputs/s45loc/20261029"
STRIDE, K_ITER, SAVE_ITERS = 8, 20, (1, 2, 3, 5, 10, 20)
PERT_NORM, TOP, BOTTOM = 0.25, 5, 15
DOMAINS = {"zl": ("zgz_local",), "tl": L.TRAIN_LOCAL, "zgz": ("zgz_global", "zgz_local")}
TASKS = {
    "zl_e3": ("zl", "e3", None), "tl_e3": ("tl", "e3", None),
    "zgz_e1": ("zgz", "e1", None), "tl_e1": ("tl", "e1", None),
    "zgz_e1_gtpose": ("zgz", "e1_gtpose", None), "zgz_e1_gttransl": ("zgz", "e1_gttransl", None),
    "tl_e1_gtpose": ("tl", "e1_gtpose", None), "tl_e1_gttransl": ("tl", "e1_gttransl", None),
    **{f"{d}_{k}": (d, "perturb", k) for d in ("zl", "tl") for k in ("T10", "R5", "Fs")},
    **{f"{d}_{w}": (d, "pert", w) for d in ("zl", "tl") for w in ("top", "bottom")},
    "zl_f1": ("zl", "f1", None), "tl_f1": ("tl", "f1", None),
}
#: f1 reads the same network's own loop
F1_LOOP = {"zl_f1": "zgz_e1", "tl_f1": "tl_e1"}
#: --net s45: the published array each task must reproduce, (directory, task) per domain
S45_REFERENCE = {
    "zl_e3": (LOC, "b0_zgz_e3"), "tl_e3": (LOC, "b0_train_e3"), "zgz_e1": (LOC, "b0_zgz_e1"),
    "tl_e1": (LOC, "c1_train_e1"), "zgz_e1_gtpose": (LOC, "d1_zgz_e1_gtpose"),
    "zgz_e1_gttransl": (LOC, "d1_zgz_e1_gttransl"), "tl_e1_gtpose": (LOC, "d1_train_local_e1_gtpose"),
    "tl_e1_gttransl": (LOC, "d1_train_local_e1_gttransl"),
    **{f"zl_{k}": (LOC, f"c4_zgz_local_{k}") for k in ("T10", "R5", "Fs")},
    **{f"tl_{k}": (LOC, f"c4_train_local_{k}") for k in ("T10", "R5", "Fs")},
    "zl_top": (DEEP, "f3_top_zl"), "zl_bottom": (DEEP, "f3_bottom_zl"), "tl_top": (DEEP, "f3_top_tl"),
    "tl_bottom": (DEEP, "f3_bottom_tl"), "zl_f1": (DEEP, "f1_fixedpoint_zl"), "tl_f1": (DEEP, "f1_fixedpoint_tl"),
}
COMPARED = ("prediction", "incoming_used", "end_ms", "delta", "gt_iter20", "mean_iter20", "loop_iter20", "mpjpe_ra_mm")


def out_dir(net, out_root=None):
    return Path(out_root or OUT / "readouts") / net


def pose_basis():
    b = read(DEEP / "protocol.json")["pose_basis"]
    return np.asarray(b["mean"]), np.asarray(b["components"])


def load(net, device):
    from config import load_config
    from model import MNISTModel
    if net == "s45":
        require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
        checkpoint, config = S45_CHECKPOINT, S45_CONFIG
        cfg = load_config(str(config))
    else:
        from tools.s45tr.evaluate import install_eval_hooks, resolve
        checkpoint, config = resolve(net, 6000)
        cfg = load_config(str(config))
        install_eval_hooks(cfg)
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True)
    return model.to(device).eval(), cfg, checkpoint


def run_task(net, task, limit=None, device_name="cuda", out_root=None):
    """`limit` / `device_name` / `out_root`: engineering smoke only (first endpoints, CPU, scratch directory)."""
    import torch
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.xcl import evaluate as XE
    domain, mode, param = TASKS[task]
    smoke = limit is not None or device_name != "cuda" or out_root is not None
    require(smoke or os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    if not smoke:
        XE.configure_environment()
        XE.set_policy()
    verify_protocol()
    mu, V = pose_basis()
    device = torch.device(device_name)
    model, cfg, checkpoint = load(net, device)
    from mano_layer import ManoLayer
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    out = out_dir(net, out_root) / task
    require(not out.exists(), "preserve existing readout: " + str(out))
    out.mkdir(parents=True)
    save(out / "launch.json", {"net": net, "task": task, "checkpoint_sha256": sha(checkpoint), "script_sha256": sha(__file__),
                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now()}, exclusive=True)
    loop = None
    if mode == "f1":
        loop = np.load(out_dir(net, out_root) / F1_LOOP[task] / "states.npz", allow_pickle=False)
    arrays = {}
    shared_rng = np.random.default_rng(0)
    with torch.no_grad():
        for seq in DOMAINS[domain]:
            events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
            tsub_path = root / found[seq] / f"{seq}_tsub.npy"
            tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
            betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
            K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
            model.set_hand_context(betas, K)
            rows = XE.endpoint_grid(aux)
            n = len(rows)
            g = lambda ms: np.asarray(gt[ms], np.float32).copy()  # noqa: E731

            def step(end, prev):
                packet = ET.make_eval_packet(
                    ET._window_events(events, offsets, tsub, int(end), 50),
                    torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(device), betas, K, 50, device)
                packet = ET.attach_xag_sources(packet, cfg, seq, offsets, int(end), 50, device)
                value = model.forward_packet(packet).detach().cpu().numpy()[0].copy()
                if value.shape != (51,) or not np.isfinite(value).all():
                    raise FloatingPointError(f"nonfinite endpoint retained as failure: {seq}:{end}")
                return value
            rec = {}

            def put(key, value):
                rec.setdefault(key, []).append(np.asarray(value))
            if mode in ("e1", "e1_gtpose", "e1_gttransl"):
                runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
                rng = shared_rng if domain == "zgz" else np.random.default_rng(0)
                i = 0
                for rid, (a, b) in enumerate(runs):
                    ends = np.arange(a + 49, b, 50, dtype=np.int64)
                    if not len(ends):
                        continue
                    prev = g(int(a)) + ET.sample_init_noise(cfg, rng, 1.0)
                    prev_ms = int(a)
                    for end in ends:
                        if limit is not None and i >= limit:
                            break
                        require(rows[i][:3] == (rid, int(end), prev_ms), "endpoint grid mismatch")
                        i += 1
                        truth = g(prev_ms)
                        used = np.asarray(prev, np.float32).copy()
                        if mode == "e1_gtpose":
                            used[3:] = truth[3:]
                        elif mode == "e1_gttransl":
                            used[:3] = truth[:3]
                        pred = step(end, used)
                        put("incoming_used", used); put("prediction", pred); put("gt_previous", truth)
                        put("target", g(int(end))); put("end_ms", int(end)); put("run_id", rid)
                        prev, prev_ms = pred, int(end)
                require(i == n or limit is not None, "no endpoints may be dropped: " + seq)
            elif mode in ("e3", "perturb", "pert"):
                for i in range(n if limit is None else min(n, limit)):
                    _, end, prev_ms, _ = rows[i]
                    truth, used = g(prev_ms), g(prev_ms)
                    if mode == "perturb":
                        used, dp = L.perturb_state(truth, param, L.rng_for(seq, int(end), 7))
                        put("delta_param", np.pad(np.asarray(dp, np.float64), (0, 45 - len(dp))))
                    elif mode == "pert":
                        rng = L.rng_for(seq, int(end), 11 if param == "top" else 12)
                        idx = np.arange(TOP) if param == "top" else np.arange(45 - BOTTOM, 45)
                        c = rng.standard_normal(len(idx))
                        d = (c[:, None] * V[idx]).sum(0)
                        d = PERT_NORM * d / np.linalg.norm(d)
                        used[6:51] = (used[6:51].astype(np.float64) + d).astype(np.float32)
                        put("delta", d)
                    put("prediction", step(end, used)); put("incoming_used", used); put("gt_previous", truth)
                    put("target", g(int(end))); put("end_ms", int(end))
            else:  # f1
                loop_in = loop[f"{seq}/incoming_used"]
                require(len(loop_in) == n or limit is not None, "own-loop endpoint count differs")
                for i in range(0, n if limit is None else min(n, limit * STRIDE, len(loop_in)), STRIDE):
                    _, end, prev_ms, _ = rows[i]
                    truth, cur = g(prev_ms), g(end)
                    meanpose = truth.copy()
                    meanpose[6:51] = mu.astype(np.float32)
                    for start_name, x0 in (("gt", truth), ("mean", meanpose), ("loop", loop_in[i])):
                        x = np.asarray(x0, np.float32).copy()
                        for k in range(1, K_ITER + 1):
                            x = step(end, x)
                            if k in SAVE_ITERS:
                                put(f"{start_name}_iter{k}", x)
                    put("index", i); put("end_ms", end); put("target", cur); put("gt_previous", truth)
            for key, v in rec.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            if mode != "f1":
                beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
                for key, v in XE.score(mano, arrays[f"{seq}/prediction"], arrays[f"{seq}/target"], beta, device).items():
                    arrays[f"{seq}/{key}"] = v
            save(out / "progress.json", {"done": seq})
    np.savez_compressed(out / "states.npz", **arrays)
    save(out / "report.json", {"net": net, "task": task, "status": "COMPLETE", "finished_utc": now(),
                               "states_sha256": sha(out / "states.npz")}, exclusive=True)


# --------------------------------------------------------------------------- analysis
def analyze(net):
    from scipy.spatial.transform import Rotation as R
    from semkine import eval_track as ET
    from tools.s45loc.analyze import FK
    mu, V = pose_basis()
    fk = FK()
    A = lambda t, k: np.load(out_dir(net) / t / "states.npz", allow_pickle=False)[k]  # noqa: E731
    s = {"net": net, "computed_utc": now(), "script_sha256": sha(__file__)}
    if net == "s45":
        same = {}
        for task, (base, ref_task) in S45_REFERENCE.items():
            mine = np.load(out_dir(net) / task / "states.npz", allow_pickle=False)
            ref = np.load(base / "tasks" / ref_task / "states.npz", allow_pickle=False)
            seqs = {k.split("/")[0] for k in mine.files}
            keys = [k for k in ref.files if k.split("/")[0] in seqs and k.split("/", 1)[1] in COMPARED]
            require(keys and all(k in mine.files for k in keys), "reference arrays missing for " + task)
            same[task] = all(np.array_equal(mine[k], ref[k]) for k in keys)
        s["s45_bitwise_reproduction"] = same
        require(all(same.values()), f"the readout tool does not reproduce S45-LOC / S45-DEEP: {same}")

    def ra(seq, a, b):
        return float(np.linalg.norm(ET.root_align(fk.joints(seq, a)) - ET.root_align(fk.joints(seq, b)), axis=-1).mean())

    def shrink(x, ref):
        d, m = x[:, 6:51].astype(np.float64) - ref[:, 6:51], mu - ref[:, 6:51].astype(np.float64)
        return float((d * m).sum() / (m * m).sum())

    def domain(d):
        return {"zgz_local": d["zgz_local"]} if "zgz_local" in d else d

    def med(per):
        return float(np.median(list(per.values())))
    # 1 retention, parameter space (S45-LOC analyze.retention, rho_param)
    r1 = {}
    for dom, pre in (("zgz_local", "zl"), ("train_local", "tl")):
        res = {}
        for kind in ("T10", "R5", "Fs"):
            per = {}
            for q in DOMAINS[pre]:
                pd_, p0 = A(f"{pre}_{kind}", f"{q}/prediction").astype(np.float64), A(f"{pre}_e3", f"{q}/prediction").astype(np.float64)
                used, truth = A(f"{pre}_{kind}", f"{q}/incoming_used").astype(np.float64), A(f"{pre}_{kind}", f"{q}/gt_previous").astype(np.float64)
                if kind == "T10":
                    din, dout = used[:, 0:3] - truth[:, 0:3], pd_[:, 0:3] - p0[:, 0:3]
                elif kind == "R5":
                    din = (R.from_rotvec(used[:, 3:6]) * R.from_rotvec(truth[:, 3:6]).inv()).as_rotvec()
                    dout = (R.from_rotvec(pd_[:, 3:6]) * R.from_rotvec(p0[:, 3:6]).inv()).as_rotvec()
                else:
                    din, dout = used[:, 6:51] - truth[:, 6:51], pd_[:, 6:51] - p0[:, 6:51]
                per[q] = float((din * dout).sum() / max((din * din).sum(), 1e-18))
            res[kind] = per[DOMAINS[pre][0]] if pre == "zl" else med(per)
        # 1b on / off manifold (S45-DEEP f3)
        for w in ("top", "bottom"):
            per = {}
            for q in DOMAINS[pre]:
                d = A(f"{pre}_{w}", f"{q}/delta")
                dout = (A(f"{pre}_{w}", f"{q}/prediction").astype(np.float64) - A(f"{pre}_e3", f"{q}/prediction"))[:, 6:51]
                per[q] = float((dout * d).sum() / (d * d).sum())
            res[w] = per[DOMAINS[pre][0]] if pre == "zl" else med(per)
        r1[dom] = res
    s["r1_retention"] = r1
    # 2 one-step shrinkage toward the mean pose: all endpoints and the lowest finger-motion quartile
    r2 = {}
    for dom, pre in (("zgz_local", "zl"), ("train_local", "tl")):
        alls, lows = {}, {}
        for q in DOMAINS[pre]:
            p, g0, tg = A(f"{pre}_e3", f"{q}/prediction"), A(f"{pre}_e3", f"{q}/gt_previous"), A(f"{pre}_e3", f"{q}/target")
            motion = np.linalg.norm(tg[:, 6:51].astype(np.float64) - g0[:, 6:51], axis=1)
            low = motion <= np.quantile(motion, 0.25)
            alls[q], lows[q] = shrink(p, g0), shrink(p[low], g0[low])
        r2[dom] = {"k_all": alls[DOMAINS[pre][0]] if pre == "zl" else med(alls),
                   "k_low_motion": lows[DOMAINS[pre][0]] if pre == "zl" else med(lows)}
    s["r2_shrinkage"] = r2
    # 3 the same packet fed 20 times (S45-DEEP f1): RA and k at iteration 20 per start
    r3 = {}
    for dom, task in (("zgz_local", "zl_f1"), ("train_local", "tl_f1")):
        seqs = DOMAINS["zl" if dom == "zgz_local" else "tl"]
        res = {}
        for start in ("gt", "mean", "loop"):
            per = {}
            for k in SAVE_ITERS:
                vals = [(ra(q, A(task, f"{q}/{start}_iter{k}"), A(task, f"{q}/target")),
                         shrink(A(task, f"{q}/{start}_iter{k}"), A(task, f"{q}/target"))) for q in seqs]
                per[k] = {"ra_mm_median": float(np.median([v[0] for v in vals])),
                          "shrink_k_median": float(np.median([v[1] for v in vals]))}
            res[start] = per
        r3[dom] = res
    s["r3_fixed_point"] = r3
    # 4 clamp shares (S45-LOC amendment 1): mean RA of each loop, (E1 - E1_clamp) / (E1 - E3)
    r4 = {}
    for dom, pre, e1 in (("zgz_local", "zl", "zgz_e1"), ("train_local", "tl", "tl_e1")):
        per = {}
        for q in DOMAINS[pre]:
            def mean_ra(task):
                return float(A(task, f"{q}/mpjpe_ra_mm").mean())
            v1, v3, vp, vt = mean_ra(e1), mean_ra(f"{pre}_e3"), mean_ra(e1 + "_gtpose"), mean_ra(e1 + "_gttransl")
            per[q] = {"e1": v1, "e3": v3, "e1_gtpose": vp, "e1_gttransl": vt,
                      "pose_clamp_share": (v1 - vp) / (v1 - v3), "transl_clamp_share": (v1 - vt) / (v1 - v3)}
        if pre == "zl":
            r4[dom] = per["zgz_local"]
        else:
            r4[dom] = {k: float(np.median([v[k] for v in per.values()])) for k in next(iter(per.values()))}
    s["r4_clamp"] = r4
    save(out_dir(net) / "summary.json", s)
    print(out_dir(net) / "summary.json")


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("action", choices=("run", "analyze"))
    p.add_argument("--net", required=True, choices=("s45",) + FORMAL_ARMS)
    p.add_argument("--task", choices=tuple(TASKS))
    a = p.parse_args(argv)
    if a.action == "run":
        require(a.task is not None, "--task required")
        run_task(a.net, a.task)
    else:
        verify_protocol()
        analyze(a.net)


if __name__ == "__main__":
    main()
