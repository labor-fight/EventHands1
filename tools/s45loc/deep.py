#!/usr/bin/env python3
"""S45-DEEP (docs/EXPERIMENT_RECORDS.md#s45deep-prereg-20261029): is S45's update an event-driven measurement update or a
pose-prior shrinkage with a weak event term? Zero training of the network; the only fit is a closed-form ridge probe.

    f1  fixed points: the same packet re-applied K times from GT previous / the mean pose / the own-loop state
    f2  event dependence: the E3 update with the packet of the endpoint 20 steps later (1 s) vs its own packet
    f3  retention of finger perturbations along the top-5 vs bottom-15 training pose components, own vs shifted events
    f4  own-loop replay capturing the 17-slot features zs, for a leave-one-recording-out ridge probe of the pose

Actions: freeze | run --task NAME | chain | analyze. Outputs in <S45LOC_OUT>/deep/. GPU 4-7, two processes per card.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45loc import common as C  # noqa: E402

DEEP = C.OUT / "deep"
PROTOCOL = DEEP / "protocol.json"
SOURCES = ("tools/s45loc/deep.py", "tools/s45loc/common.py", "tools/s45loc/run.py", "tools/xcl/evaluate.py",
           "semkine/xyz_mesh.py", "model/model.py", "semkine/eval_track.py")
#: every STRIDE-th endpoint for f1 (K iterations x 3 starts per endpoint)
STRIDE, K_ITER, SAVE_ITERS = 8, 20, (1, 2, 3, 5, 10, 20)
SHIFT = 20                      # f2 / f3: the packet of the endpoint SHIFT grid steps later (cyclic in the recording)
PERT_NORM = 0.25                # f3: finger axis-angle perturbation norm (rad)
TOP, BOTTOM = 5, 15
RIDGE = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
DOMAINS = {"zl": ("zgz_local",), "tl": C.TRAIN_LOCAL}
TASKS = {
    "f1_fixedpoint_zl": ("zl", "f1", None), "f1_fixedpoint_tl": ("tl", "f1", None),
    "f2_shifted_zl": ("zl", "e3_shift", None), "f2_shifted_tl": ("tl", "e3_shift", None),
    "f3_top_zl": ("zl", "pert", "top"), "f3_bottom_zl": ("zl", "pert", "bottom"),
    "f3_top_shift_zl": ("zl", "pert_shift", "top"),
    "f3_top_tl": ("tl", "pert", "top"), "f3_bottom_tl": ("tl", "pert", "bottom"),
    "f3_top_shift_tl": ("tl", "pert_shift", "top"),
    "f4_loop_zgz": ("zgz", "e1_capture", None), "f4_loop_tl": ("tl", "e1_capture", None),
}
SIZE = {"zl": 1, "tl": 9, "zgz": 2}
#: the S45-LOC arrays each task is anchored to (GT-previous E3 baseline / own-loop E1 states)
E3_BASE = {"zgz_local": "b0_zgz_e3", **{s: "b0_train_e3" for s in C.TRAIN_LOCAL}}
E1_BASE = {"zgz_local": "b0_zgz_e1", "zgz_global": "b0_zgz_e1", **{s: "c1_train_e1" for s in C.TRAIN_LOCAL}}


def base_arr(task, key):
    """The completed S45-LOC readings (always the reading directory, also for smoke runs of S45-DEEP)."""
    return np.load(C.DEFAULT_OUT / "tasks" / task / "states.npz", allow_pickle=False)[key]


# --------------------------------------------------------------------------- freeze
def freeze():
    rows = [base_arr("b0_train_e3", f"{s}/gt_previous")[:, 6:51] for s in C.TRAIN_BASE]
    P = np.concatenate(rows).astype(np.float64)
    mu = P.mean(0)
    _, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    section = DEEP / "prereg_section.md"
    C.require(C.SMOKE or section.exists(), "write the pre-registration section before freezing")
    C.save(PROTOCOL, {
        "study": "S45-DEEP", "status": "SMOKE" if C.SMOKE else "FROZEN", "frozen_utc": C.now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45deep-prereg-20261029",
        "prereg_section_sha256": C.sha(section) if section.exists() else None,
        "tasks": {k: list(v) for k, v in TASKS.items()},
        "pose_basis": {"source": "GT previous finger axis-angle of the 18 base training recordings' W50 endpoints "
                                 "(S45-LOC b0_train_e3)", "n": int(len(P)), "mean": mu.tolist(),
                       "components": Vt.tolist(), "variance_share": (S ** 2 / (S ** 2).sum()).tolist()},
        "constants": {"stride": STRIDE, "k_iter": K_ITER, "save_iters": SAVE_ITERS, "shift": SHIFT,
                      "pert_norm": PERT_NORM, "top": TOP, "bottom": BOTTOM, "ridge": RIDGE},
        "s45_checkpoint_sha256": C.CHECKPOINT_SHA256,
        "sources_sha256": {n: C.sha(C.ROOT / n) for n in SOURCES},
        "no_training_of_the_network": True,
    }, exclusive=True)
    print(PROTOCOL)


def protocol():
    p = C.read(PROTOCOL)
    C.require(p["status"] == ("SMOKE" if C.SMOKE else "FROZEN"), "S45-DEEP protocol status / directory mismatch")
    for name, digest in p["sources_sha256"].items():
        C.require(C.sha(C.ROOT / name) == digest, f"frozen source changed: {name}")
    return p


# --------------------------------------------------------------------------- run
def run_task(name, limit=None):
    import torch
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45loc.run import load_model
    from tools.xcl import evaluate as XE
    domain, mode, which = TASKS[name]
    proto = protocol()
    basis = proto["pose_basis"]
    mu, V = np.asarray(basis["mean"]), np.asarray(basis["components"])
    device = torch.device("cuda")
    model, cfg = load_model(device)
    model.xyz_keep_aux = mode == "e1_capture"
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    out = DEEP / "tasks" / name
    C.require(not out.exists(), "preserve existing task output: " + str(out))
    out.mkdir(parents=True)
    C.save(out / "launch.json", {"task": name, "protocol_sha256": C.sha(PROTOCOL), "started_utc": C.now(),
                                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}, exclusive=True)
    arrays = {}
    seqs = ("zgz_global", "zgz_local") if domain == "zgz" else DOMAINS[domain]
    shared_rng = np.random.default_rng(0)
    with torch.no_grad():
        for seq in seqs:
            events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
            tsub_path = root / found[seq] / f"{seq}_tsub.npy"
            tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
            betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
            K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
            model.set_hand_context(betas, K)
            rows = XE.endpoint_grid(aux)
            n = len(rows)
            g = lambda ms: np.asarray(gt[ms], np.float32).copy()  # noqa: E731
            pkt = lambda end, prev: ET.make_eval_packet(  # noqa: E731
                ET._window_events(events, offsets, tsub, int(end), 50),
                torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(device), betas, K, 50, device)
            rec = {}

            def put(key, value):
                rec.setdefault(key, []).append(np.asarray(value))
            if mode == "f1":
                loop_in = base_arr(E1_BASE[seq], f"{seq}/incoming_used")
                for i in range(0, n if limit is None else min(n, limit * STRIDE), STRIDE):
                    _, end, prev_ms, _ = rows[i]
                    truth, cur = g(prev_ms), g(end)
                    meanpose = truth.copy()
                    meanpose[6:51] = mu.astype(np.float32)
                    for start_name, x0 in (("gt", truth), ("mean", meanpose), ("loop", loop_in[i])):
                        x = np.asarray(x0, np.float32).copy()
                        packet = pkt(end, x)
                        for k in range(1, K_ITER + 1):
                            packet.prev_state = torch.from_numpy(x).view(1, -1).to(device)
                            x = model.forward_packet(packet).cpu().numpy()[0].copy()
                            if k in SAVE_ITERS:
                                put(f"{start_name}_iter{k}", x)
                    put("index", i); put("end_ms", end); put("target", cur); put("gt_previous", truth)
            elif mode in ("e3_shift", "pert", "pert_shift"):
                for i in range(n if limit is None else min(n, limit)):
                    _, end, prev_ms, _ = rows[i]
                    truth, cur = g(prev_ms), g(end)
                    used = truth.copy()
                    if mode in ("pert", "pert_shift"):
                        rng = C.rng_for(seq, int(end), 11 if which == "top" else 12)
                        idx = np.arange(TOP) if which == "top" else np.arange(45 - BOTTOM, 45)
                        c = rng.standard_normal(len(idx))
                        d = (c[:, None] * V[idx]).sum(0)
                        d = PERT_NORM * d / np.linalg.norm(d)
                        used[6:51] = (used[6:51].astype(np.float64) + d).astype(np.float32)
                        put("delta", d)
                    end_events = rows[(i + SHIFT) % n][1] if mode in ("e3_shift", "pert_shift") else end
                    packet = pkt(end_events, used)
                    put("prediction", model.forward_packet(packet).cpu().numpy()[0])
                    put("incoming_used", used); put("gt_previous", truth); put("target", cur); put("end_ms", end)
                    put("events_end_ms", end_events)
            else:   # e1_capture: replay the own loop exactly (shared init-noise generator for zgz) with zs captured
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
                        i += 1
                        model.xyz_last = None
                        packet = pkt(end, prev)
                        pred = model.forward_packet(packet).cpu().numpy()[0].copy()
                        last = getattr(model, "xyz_last", None)
                        zs = last["zs"][0].float().cpu().numpy() if last is not None else np.zeros(17 * 32, np.float32)
                        put("zs", zs); put("incoming_used", prev); put("prediction", pred)
                        put("gt_previous", g(prev_ms)); put("target", g(int(end))); put("end_ms", int(end))
                        put("run_id", rid); put("has_events", last is not None)
                        prev, prev_ms = pred, int(end)
            for key, v in rec.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            C.save(out / "progress.json", {"done": seq})
    np.savez_compressed(out / "states.npz", **arrays)
    C.save(out / "report.json", {"task": name, "status": "COMPLETE", "finished_utc": C.now(),
                                 "states_sha256": C.sha(out / "states.npz")}, exclusive=True)


def chain(limit=None):
    from tools.xcl import evaluate as XE  # noqa: F401  (environment policy is applied inside each worker)
    proto = protocol()
    slots = [(g, s) for g in C.PHYSICAL_GPUS for s in range(2)]
    pending = sorted(TASKS, key=lambda t: -SIZE[TASKS[t][0]])
    running, failed = {}, []
    log = DEEP / "logs"
    log.mkdir(parents=True, exist_ok=True)
    while pending or running:
        free = [s for s in slots if s not in running.values()]
        while pending and free and not failed:
            task, slot = pending.pop(0), free.pop(0)
            cmd = [C.PYTHON, str(ROOT / "tools/s45loc/deep.py"), "run", "--task", task]
            if limit is not None:
                cmd += ["--limit", str(limit)]
            running[subprocess.Popen(cmd, cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0])),
                                     stdout=(log / f"{task}.log").open("w"), stderr=subprocess.STDOUT)] = (slot, task)
        for proc in list(running):
            if proc.poll() is not None:
                slot, task = running.pop(proc)
                if proc.returncode != 0:
                    failed.append(task)
        if failed and not running:
            break
        time.sleep(5)
    C.save(DEEP / "chain_status.json", {"failed": failed, "pending": pending, "at_utc": C.now(),
                                        "protocol_sha256": C.sha(PROTOCOL)})
    C.require(not failed, f"S45-DEEP tasks failed: {failed}")
    code = subprocess.call([C.PYTHON, str(ROOT / "tools/s45loc/deep.py"), "analyze"], cwd=str(ROOT))
    C.require(code == 0, "S45-DEEP analysis failed")


# --------------------------------------------------------------------------- analysis
def analyze():
    from tools.s45loc.analyze import FK
    from semkine import eval_track as ET
    proto = protocol()
    mu, V = np.asarray(proto["pose_basis"]["mean"]), np.asarray(proto["pose_basis"]["components"])
    fk = FK()
    A = lambda t, k: np.load(DEEP / "tasks" / t / "states.npz", allow_pickle=False)[k]  # noqa: E731

    def ra(seq, a, b):
        Ja, Jb = fk.joints(seq, a), fk.joints(seq, b)
        return float(np.linalg.norm(ET.root_align(Ja) - ET.root_align(Jb), axis=-1).mean())

    def base(q, key, n):
        """The S45-LOC baseline array aligned to a task's n endpoints (all of them outside smoke runs)."""
        v = base_arr(E3_BASE[q], f"{q}/{key}")
        C.require(C.SMOKE or len(v) == n, "baseline / task endpoint counts differ")
        return v[:n]

    def shrink(x, ref):
        """k in x_f - ref_f = k (mu - ref_f): share of the way from the reference finger pose to the mean pose."""
        d, m = x[:, 6:51].astype(np.float64) - ref[:, 6:51], mu - ref[:, 6:51].astype(np.float64)
        return float((d * m).sum() / (m * m).sum())
    s = {"protocol_sha256": C.sha(PROTOCOL), "computed_utc": C.now()}
    # f1 fixed points
    f1 = {}
    for dom, task in (("zgz_local", "f1_fixedpoint_zl"), ("train_local", "f1_fixedpoint_tl")):
        seqs = DOMAINS["zl" if dom == "zgz_local" else "tl"]
        out = {}
        for start in ("gt", "mean", "loop"):
            per = {}
            for k in SAVE_ITERS:
                vals = [(ra(q, A(task, f"{q}/{start}_iter{k}"), A(task, f"{q}/target")),
                         shrink(A(task, f"{q}/{start}_iter{k}"), A(task, f"{q}/target"))) for q in seqs]
                per[k] = {"ra_mm_median": float(np.median([v[0] for v in vals])),
                          "shrink_k_median": float(np.median([v[1] for v in vals]))}
            out[start] = per
        # do the three starts reach the same state after K iterations?
        gaps = [float(np.linalg.norm(A(task, f"{q}/gt_iter{K_ITER}")[:, 6:51] - A(task, f"{q}/loop_iter{K_ITER}")[:, 6:51], axis=1).mean())
                for q in seqs]
        out["gt_vs_loop_finger_gap_rad_median_at_K"] = float(np.median(gaps))
        out["gt_vs_loop_finger_gap_rad_median_at_0"] = float(np.median([
            float(np.linalg.norm(A(task, f"{q}/gt_previous")[:, 6:51]
                                 - base_arr(E1_BASE[q], f"{q}/incoming_used")[A(task, f"{q}/index")][:, 6:51], axis=1).mean())
            for q in seqs]))
        f1[dom] = out
    s["f1_fixed_points"] = f1
    # f2 event dependence of the E3 update (finger part in the pose basis, root rotation vector)
    f2 = {}
    for dom, task in (("zgz_local", "f2_shifted_zl"), ("train_local", "f2_shifted_tl")):
        seqs = DOMAINS["zl" if dom == "zgz_local" else "tl"]
        per = {}
        for q in seqs:
            nq = len(A(task, f"{q}/end_ms"))
            own = base(q, "prediction", nq).astype(np.float64)
            g0 = base(q, "gt_previous", nq).astype(np.float64)
            sh = A(task, f"{q}/prediction").astype(np.float64)
            C.require(np.array_equal(A(task, f"{q}/end_ms"), base(q, "end_ms", nq)), "endpoint mismatch")
            u_own, u_sh = (own - g0)[:, 6:51] @ V.T, (sh - g0)[:, 6:51] @ V.T
            m = (A(task, f"{q}/target").astype(np.float64) - g0)[:, 6:51] @ V.T
            r2 = lambda a, b: float(1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum())  # noqa: E731
            ev = u_own - u_sh
            per[q] = {"finger_update_r2_explained_by_shifted": r2(u_own, u_sh),
                      "finger_update_norm_own": float(np.linalg.norm(u_own, axis=1).mean()),
                      "finger_update_norm_shifted": float(np.linalg.norm(u_sh, axis=1).mean()),
                      "event_part_gain_along_true_motion": float((ev * m).sum() / (m * m).sum()),
                      "own_gain_along_true_motion": float((u_own * m).sum() / (m * m).sum()),
                      "shrink_k_own": shrink(own, g0), "shrink_k_shifted": shrink(sh, g0),
                      "ra_own": ra(q, own.astype(np.float32), base(q, "target", nq)),
                      "ra_shifted": ra(q, sh.astype(np.float32), A(task, f"{q}/target"))}
        f2[dom] = {"per_recording": per, **{f"median_{k}": float(np.median([v[k] for v in per.values()]))
                                            for k in next(iter(per.values()))}}
    s["f2_event_dependence"] = f2
    # f3 retention of on- vs off-manifold finger perturbations, own vs shifted events
    f3 = {}
    for dom, suf in (("zgz_local", "zl"), ("train_local", "tl")):
        seqs = DOMAINS[suf]
        res = {}
        for cond, task, ref in (("top_own", f"f3_top_{suf}", "e3"), ("bottom_own", f"f3_bottom_{suf}", "e3"),
                                ("top_shifted", f"f3_top_shift_{suf}", "shift")):
            vals = []
            for q in seqs:
                nq = len(A(task, f"{q}/end_ms"))
                p0 = base(q, "prediction", nq) if ref == "e3" else A(f"f2_shifted_{suf}", f"{q}/prediction")[:nq]
                d = A(task, f"{q}/delta")
                dout = (A(task, f"{q}/prediction").astype(np.float64) - p0)[:, 6:51]
                vals.append(float((dout * d).sum() / (d * d).sum()))
            res[cond] = {"rho_median": float(np.median(vals)), "per_recording": dict(zip(seqs, vals))}
        f3[dom] = res
    s["f3_manifold_retention"] = f3
    # f4: the replay must be S45-LOC's own loop bit for bit, then the ridge probe of the true finger pose
    replay = {}
    for q, task in [("zgz_local", "f4_loop_zgz")] + [(r, "f4_loop_tl") for r in C.TRAIN_LOCAL]:
        mine = A(task, f"{q}/prediction")
        ref = base_arr(E1_BASE[q], f"{q}/prediction")
        C.require(C.SMOKE or len(mine) == len(ref), "replay endpoint count differs")
        replay[q] = bool(np.array_equal(mine, ref[:len(mine)]))
    s["f4_replay_equals_s45loc_e1"] = replay
    C.require(all(replay.values()), f"f4 replay differs from the S45-LOC own loop: {replay}")
    s["f4_probe"] = probe(A, mu, V)
    C.save(DEEP / "summary.json", s)
    print(DEEP / "summary.json")


def probe(A, mu, V, n_pc=10):
    def data(q, task):
        zs, inc = A(task, f"{q}/zs").astype(np.float64), A(task, f"{q}/incoming_used").astype(np.float64)
        tgt, pred = A(task, f"{q}/target").astype(np.float64), A(task, f"{q}/prediction").astype(np.float64)
        has = A(task, f"{q}/has_events").astype(bool)
        y = (tgt[:, 6:51] - mu) @ V[:n_pc].T
        prior = (inc[:, 6:51] - mu) @ V[:n_pc].T
        head = (pred[:, 6:51] - mu) @ V[:n_pc].T
        return zs[has], prior[has], y[has], head[has]

    def fit(X, Y, lam):
        m, sd = X.mean(0), X.std(0) + 1e-8
        Z = (X - m) / sd
        Wt = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
        return lambda Xn: ((Xn - m) / sd) @ Wt + Y.mean(0)
    tl = {q: data(q, "f4_loop_tl") for q in C.TRAIN_LOCAL}
    zl = data("zgz_local", "f4_loop_zgz")
    rms = lambda a, b: float(np.sqrt(((a - b) ** 2).sum(1).mean()))  # noqa: E731
    out = {"n_pc": n_pc, "folds": {}}
    for feats in ("zs+prior", "prior"):
        build = (lambda d: np.hstack([d[0], d[1]])) if feats == "zs+prior" else (lambda d: d[1])
        fold = {}
        for held in C.TRAIN_LOCAL:
            train = [tl[q] for q in C.TRAIN_LOCAL if q != held]
            X, Y = np.vstack([build(d) for d in train]), np.vstack([d[2] for d in train])
            # inner choice of lambda: the other recordings, leave-one-out, pre-registered grid
            best = min(RIDGE, key=lambda lam: np.mean([
                rms(fit(np.vstack([build(d) for j, d in enumerate(train) if j != k]),
                        np.vstack([d[2] for j, d in enumerate(train) if j != k]), lam)(build(train[k])), train[k][2])
                for k in range(len(train))]))
            fold[held] = {"lambda": best, "rms": rms(fit(X, Y, best)(build(tl[held])), tl[held][2])}
        X, Y = np.vstack([build(tl[q]) for q in C.TRAIN_LOCAL]), np.vstack([tl[q][2] for q in C.TRAIN_LOCAL])
        lam = float(np.median([v["lambda"] for v in fold.values()]))
        out["folds"][feats] = {"per_recording": fold, "median_rms": float(np.median([v["rms"] for v in fold.values()])),
                               "zgz_local_rms": rms(fit(X, Y, lam)(build(zl)), zl[2]), "zgz_lambda": lam}
    out["head_rms"] = {"train_local_median": float(np.median([rms(d[3], d[2]) for d in tl.values()])),
                       "zgz_local": rms(zl[3], zl[2])}
    out["prior_rms"] = {"train_local_median": float(np.median([rms(d[1], d[2]) for d in tl.values()])),
                        "zgz_local": rms(zl[1], zl[2])}
    out["mean_pose_rms"] = {"train_local_median": float(np.median([rms(0 * d[2], d[2]) for d in tl.values()])),
                            "zgz_local": rms(0 * zl[2], zl[2])}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("action", choices=("freeze", "run", "chain", "analyze"))
    p.add_argument("--task", choices=sorted(TASKS))
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)
    C.require(args.limit is None or C.SMOKE, "--limit is for smoke directories only")
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        from tools.xcl import evaluate as XE
        C.require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, C.PHYSICAL_GPUS)), "GPU 4-7 only")
        XE.configure_environment()
        XE.set_policy()
        run_task(args.task, args.limit)
    elif args.action == "chain":
        chain(args.limit)
    else:
        analyze()


if __name__ == "__main__":
    main()
