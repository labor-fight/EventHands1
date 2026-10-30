#!/usr/bin/env python3
"""S45-RF P1 (docs/EXPERIMENT_RECORDS.md#s45rf-prereg-20261030): is the one-step correction of S45's closed-loop
states event-driven? Zero training; frozen S45-TR networks (step 6000); zgz only; one physical GPU among 2-7 per process.

Tasks (E2 = the network fed S45's own zgz E1 state at each endpoint, as S45-TR's E2):
  e2_true        E2 with the endpoint's own packet; must equal the S45-TR E2 arrays bit for bit (S45: its own E1)
  e2_shift5      E2 with the packet of the endpoint 5 grid steps (250 ms) later, cyclic in the recording (primary)
  e2_shift20     the same, 20 steps (1 s, S45-DEEP f2's shift; fallback)
  e3 / e3_lag25  (s45, n4) the GT previous state / the GT state 25 ms earlier (N4's bank-lag check)
Decomposition (finger block, isotropic scalar gains, per-recording per-dimension fixed effects), in = incoming state:
  own packet      Δ = a_m (g_end - g_prev) + a_e (g_prev - in) + β (μ - in)
  shifted packet  Δ = c_M (s_end - s_prev) + K (s_prev - g_prev) + c_E (g_prev - in) + c_B (μ - in)
A pull toward the pose the events show contributes K = its gain (and the same to c_E); a correction from the incoming
state alone gives K = 0 and c_E = a_e. Event share s = K / a_e, paired block bootstrap (20-endpoint blocks).

  p1.py power                  cached S45-TR E2 + GT only: the design's power (before freezing)
  p1.py freeze                 after the ledger section is written; refuses to overwrite
  p1.py run --net N --task T   one GPU among 2-7
  p1.py chain                  every job on GPUs 2-7 (two processes per card), then check and analyze
  p1.py check | analyze        CPU
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

from tools.s45rf.common import (P1, PHYSICAL_GPUS, POSE_BASIS, S45_E1_TRACE, S45_E3_TRACE, S45TR, SEQS,  # noqa: E402
                                ledger_section, now, pose_mean, read, require, save, sha)

PROTOCOL = P1 / "protocol.json"
POWER = P1 / "power.json"
TAG = "S45RF-PREREG-20261030"
NETS = ("s45", "t1d", "n4", "n2", "t1a")
DECISION_NET, CONTROL_NET = "n4", "t1d"
#: task -> (prior kind, grid shift of the packet (e2) or prior lag in ms (e3))
TASKS = {"e2_true": ("e2", 0), "e2_shift5": ("e2", 5), "e2_shift20": ("e2", 20), "e3": ("e3", 0), "e3_lag25": ("e3", 25)}
SHIFT_ORDER = ("e2_shift5", "e2_shift20")       # primary, then fallback; chosen by the T1d control only
LAG_NETS = ("s45", "n4")
JOBS = tuple((n, t) for n in NETS for t in ("e2_true",) + SHIFT_ORDER) + tuple((n, t) for n in LAG_NETS for t in ("e3", "e3_lag25"))
BLOCK, NBOOT, BOOT_SEED = 20, 2000, 20261030
EVENT_SHARE, EVENT_LOW, STATE_SHARE, STATE_HIGH = 0.5, 0.3, 0.3, 0.5
TOL_ABS, TOL_REL = 0.04, 0.30
BLOCKS = {"finger": slice(6, 51), "root": slice(3, 6), "transl": slice(0, 3)}
SOURCES = ("tools/s45rf/__init__.py", "tools/s45rf/common.py", "tools/s45rf/p1.py", "tests/test_s45rf.py",
           "tools/s45tr/common.py", "tools/s45tr/readouts.py", "tools/s45tr/evaluate.py", "tools/xcl/evaluate.py",
           "semkine/eval_track.py", "model/model.py", "model/config.py", "semkine/xyz_mesh.py", "semkine/xag.py",
           "semkine/xag_sources.py", "semkine/s45tr_t1a.py", "semkine/event_gnn.py")


def e2_reference(net):
    return S45_E1_TRACE if net == "s45" else S45TR / "evaluations" / net / "e2_step6000" / "states.npz"


def e3_reference(net):
    return S45_E3_TRACE if net == "s45" else S45TR / "evaluations" / net / "e3_step6000" / "states.npz"


def shifted_index(i, n, k):
    """The endpoint whose packet is fed at endpoint i: k grid steps later, cyclic in the recording (S45-DEEP f2)."""
    return (i + k) % n


def lagged_ms(prev_ms, run_start, lag):
    """The prior's time for the lag check: `lag` ms before the GT previous state, never before its valid run."""
    return max(int(run_start), int(prev_ms) - int(lag))


def task_dir(net, task, out_root=None):
    return Path(out_root or P1 / "tasks") / net / task


# --------------------------------------------------------------------------- protocol
def protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45RF_P1_20261030" and p["status"] == "FROZEN", "S45-RF P1 protocol identity")
    for name, digest in {**p["sources_sha256"], **p["inputs_sha256"]}.items():
        require(sha(ROOT / name) == digest, "frozen S45-RF P1 file changed: " + name)
    require(sha(P1 / "prereg_section.md") == p["prereg_section_sha256"]
            and (P1 / "prereg_section.md").read_text() == ledger_section(TAG), "pre-registration changed")
    return p


def freeze():
    require(not PROTOCOL.exists(), "protocol already frozen")
    require(POWER.is_file(), "run the power analysis before freezing")
    section = P1 / "prereg_section.md"
    with section.open("x") as stream:
        stream.write(ledger_section(TAG))
    receipts = {n: read(S45TR / "receipts" / f"formal_{n}.json") for n in NETS if n != "s45"}
    from tools.s45tr.common import S45_CHECKPOINT_SHA256
    inputs = [S45_E1_TRACE, S45_E3_TRACE, POSE_BASIS, POWER] + [e2_reference(n) for n in NETS if n != "s45"] \
        + [e3_reference(n) for n in LAG_NETS if n != "s45"]
    save(PROTOCOL, {
        "study": "S45RF_P1_20261030", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45rf-prereg-20261030", "prereg_section_sha256": sha(section),
        "jobs": [list(j) for j in JOBS], "tasks": {k: list(v) for k, v in TASKS.items()}, "shift_order": SHIFT_ORDER,
        "decision_net": DECISION_NET, "control_net": CONTROL_NET,
        "constants": {"block": BLOCK, "nboot": NBOOT, "boot_seed": BOOT_SEED, "finger_block": [6, 51],
                      "pose_mean": "S45-DEEP pose_basis.mean"},
        "thresholds": {"event_share": EVENT_SHARE, "event_ci_low": EVENT_LOW, "state_share": STATE_SHARE,
                       "state_ci_high": STATE_HIGH, "consistency_tol_abs": TOL_ABS, "consistency_tol_rel": TOL_REL},
        "checkpoints_sha256": {"s45": S45_CHECKPOINT_SHA256,
                               **{n: r["checkpoint_sha256_by_step"]["6000"] for n, r in receipts.items()}},
        "inputs_sha256": {str(Path(p).relative_to(ROOT)): sha(p) for p in inputs},
        "sources_sha256": {name: sha(ROOT / name) for name in SOURCES},
        "physical_gpus": PHYSICAL_GPUS, "no_training": True, "single_seed_descriptive": True,
    }, exclusive=True)
    print(PROTOCOL)


# --------------------------------------------------------------------------- run
def run_task(net, task, limit=None, device_name="cuda", out_root=None):
    """`limit` / `device_name` / `out_root`: engineering smoke only (first endpoints, CPU, scratch directory)."""
    import torch
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45tr.common import verify_protocol as verify_s45tr
    from tools.s45tr.readouts import load
    from tools.xcl import evaluate as XE
    require((net, task) in JOBS, "unregistered P1 job")
    kind, k = TASKS[task]
    smoke = limit is not None or device_name != "cuda" or out_root is not None
    require(smoke or os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    if not smoke:
        XE.configure_environment()
        XE.set_policy()
        protocol()
    verify_s45tr()
    device = torch.device(device_name)
    model, cfg, checkpoint = load(net, device)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    found = dict(sequences_for_split(root, "val_core", splits_manifest(cfg)))
    require(tuple(found) == SEQS, "expected the original zgz_global / zgz_local")
    out = task_dir(net, task, out_root)
    require(not out.exists(), "preserve existing task output: " + str(out))
    out.mkdir(parents=True)
    save(out / "launch.json", {"net": net, "task": task, "checkpoint_sha256": sha(checkpoint), "script_sha256": sha(__file__),
                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "smoke": smoke, "started_utc": now()},
         exclusive=True)
    common = np.load(S45_E1_TRACE, allow_pickle=False) if kind == "e2" else None
    arrays = {}
    with torch.no_grad():
        for seq in SEQS:
            events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
            tsub_path = root / found[seq] / f"{seq}_tsub.npy"
            tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
            betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
            K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
            model.set_hand_context(betas, K)
            rows = XE.endpoint_grid(aux)
            n = len(rows)
            require(n == XE.COUNTS[seq], "endpoint denominator changed: " + seq)
            runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
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
            if common is not None:
                incoming = common[f"{seq}/incoming"]
                require(incoming.shape == (n, 51), "incomplete S45 E1 trace")
                require(np.array_equal(common[f"{seq}/end_ms"], [r[1] for r in rows]), "S45 trace endpoint identity differs")
            rec = {}

            def put(key, value):
                rec.setdefault(key, []).append(np.asarray(value))
            for i in range(n if limit is None else min(n, limit)):
                rid, end, prev_ms, _ = rows[i]
                truth, current = g(prev_ms), g(end)
                if kind == "e2":
                    used = np.asarray(incoming[i], np.float32).copy()
                    _, ev_end, ev_prev, _ = rows[shifted_index(i, n, k)]
                    put("events_end_ms", ev_end); put("events_gt_previous", g(ev_prev)); put("events_target", g(ev_end))
                else:
                    prior_ms = lagged_ms(prev_ms, runs[rid][0], k)
                    used, ev_end = g(prior_ms), end
                    put("prior_ms", prior_ms)
                put("prediction", step(ev_end, used)); put("incoming_used", used); put("gt_previous", truth)
                put("target", current); put("end_ms", int(end)); put("run_id", rid)
            for key, v in rec.items():
                arrays[f"{seq}/{key}"] = np.stack(v)
            beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
            for key, v in XE.score(mano, arrays[f"{seq}/prediction"], arrays[f"{seq}/target"], beta, device).items():
                arrays[f"{seq}/{key}"] = v
            save(out / "progress.json", {"done": seq})
    np.savez_compressed(out / "states.npz", **arrays)
    save(out / "report.json", {"net": net, "task": task, "status": "COMPLETE", "finished_utc": now(),
                               "states_sha256": sha(out / "states.npz")}, exclusive=True)


def gpu_busy(gpu):
    out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0]) > 1000


def chain():
    protocol()
    busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
    require(not busy, f"GPUs {busy} are in use; P1 waits for idle cards among 2-7")
    slots = [(g, s) for g in PHYSICAL_GPUS for s in range(2)]
    pending, running, failed = list(JOBS), {}, []
    log = P1 / "logs"
    log.mkdir(parents=True, exist_ok=True)
    while pending or running:
        free = [s for s in slots if s not in running.values()]
        while pending and free and not failed:
            job, slot = pending.pop(0), free.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--net", job[0], "--task", job[1]]
            running[subprocess.Popen(cmd, cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0])),
                                     stdout=(log / f"{job[0]}_{job[1]}.log").open("w"), stderr=subprocess.STDOUT)] = (slot, job)
        for proc in list(running):
            if proc.poll() is not None:
                slot, job = running.pop(proc)
                if proc.returncode != 0:
                    failed.append(list(job))
        if failed and not running:
            break
        time.sleep(5)
    save(P1 / "chain_status.json", {"failed": failed, "not_started": [list(j) for j in pending], "at_utc": now()})
    require(not failed, f"P1 jobs failed (preserved, not retried): {failed}")
    for action in ("check", "analyze"):
        require(subprocess.call([sys.executable, str(Path(__file__).resolve()), action], cwd=str(ROOT)) == 0,
                "P1 " + action + " failed")


# --------------------------------------------------------------------------- check
def load_task(net, task, out_root=None):
    with np.load(task_dir(net, task, out_root) / "states.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check():
    protocol()
    res = {}
    for net in NETS:
        mine = load_task(net, "e2_true")
        with np.load(e2_reference(net), allow_pickle=False) as ref:
            res[f"{net}/e2_true"] = all(np.array_equal(mine[f"{s}/{key}"], ref[f"{s}/{key}"])
                                        for s in SEQS for key in ("incoming_used", "prediction", "target", "gt_previous"))
        n_rows = {s: len(mine[f"{s}/end_ms"]) for s in SEQS}
        for task in SHIFT_ORDER:
            shifted = load_task(net, task)
            k = TASKS[task][1]
            same = all(np.array_equal(shifted[f"{s}/{key}"], mine[f"{s}/{key}"])
                       for s in SEQS for key in ("incoming_used", "target", "gt_previous", "end_ms"))
            j = {s: shifted_index(np.arange(n_rows[s]), n_rows[s], k) for s in SEQS}
            mapped = all(np.array_equal(shifted[f"{s}/events_target"], mine[f"{s}/target"][j[s]])
                         and np.array_equal(shifted[f"{s}/events_gt_previous"], mine[f"{s}/gt_previous"][j[s]])
                         and np.array_equal(shifted[f"{s}/events_end_ms"], mine[f"{s}/end_ms"][j[s]]) for s in SEQS)
            res[f"{net}/{task}"] = same and mapped
    for net in LAG_NETS:
        mine = load_task(net, "e3")
        with np.load(e3_reference(net), allow_pickle=False) as ref:
            res[f"{net}/e3"] = all(np.array_equal(mine[f"{s}/{key}"], ref[f"{s}/{key}"])
                                   for s in SEQS for key in ("incoming_used", "prediction", "target"))
        lagged = load_task(net, "e3_lag25")
        res[f"{net}/e3_lag25"] = all(np.array_equal(lagged[f"{s}/target"], mine[f"{s}/target"]) for s in SEQS)
    save(P1 / "check.json", {"computed_utc": now(), "checks": res, "status": "PASS" if all(res.values()) else "FAIL"})
    require(all(res.values()), f"P1 reproduction / identity checks failed: {res}")
    print(P1 / "check.json")


# --------------------------------------------------------------------------- decomposition
def design(true, shift, seq, blk, mu=None):
    """Per-endpoint regressors of one recording and parameter block, per-dimension fixed effects removed."""
    f = lambda a, key: a[f"{seq}/{key}"].astype(np.float64)[:, blk]  # noqa: E731
    xi, xp, xt = f(true, "incoming_used"), f(true, "gt_previous"), f(true, "target")
    d = {"D": f(true, "prediction") - xi, "Mc": xt - xp, "E": xp - xi}
    if mu is not None:
        d["Bm"] = mu[None] - xi
    if shift is not None:
        sp, se = f(shift, "events_gt_previous"), f(shift, "events_target")
        d.update(W=f(shift, "prediction") - xi, Mw=se - sp, Sw=sp - xp)
    return {key: v - v.mean(0, keepdims=True) for key, v in d.items()}


def block_sums(d, ykey, xkeys, block=BLOCK):
    """Per-block X'X and X'y, rows = (endpoint, dimension) pairs, scalar gain per regressor."""
    X = np.stack([d[k] for k in xkeys], -1)
    y = d[ykey]
    xtx, xty = np.einsum("nij,nik->njk", X, X), np.einsum("nij,ni->nj", X, y)
    b = np.arange(len(y)) // block
    A, v = np.zeros((b.max() + 1,) + xtx.shape[1:]), np.zeros((b.max() + 1, xty.shape[1]))
    np.add.at(A, b, xtx)
    np.add.at(v, b, xty)
    return A, v


def solve(parts, picks=None):
    if picks is None:
        return np.linalg.solve(sum(a.sum(0) for a, _ in parts), sum(w.sum(0) for _, w in parts))
    A = sum(a[p].sum(1) for (a, _), p in zip(parts, picks))
    v = sum(w[p].sum(1) for (_, w), p in zip(parts, picks))
    return np.linalg.solve(A, v[..., None])[..., 0]


def decompose(designs, with_mean, nboot=NBOOT, seed=BOOT_SEED):
    """Own-packet and shifted-packet fits pooled over recordings, with the same (paired) block resamples."""
    xt, xs = ("Mc", "E") + (("Bm",) if with_mean else ()), ("Mw", "Sw", "E") + (("Bm",) if with_mean else ())
    pt, ps = [block_sums(d, "D", xt) for d in designs], [block_sums(d, "W", xs) for d in designs]
    rng = np.random.default_rng(seed)
    picks = [rng.integers(0, len(a), (nboot, len(a))) for a, _ in pt]
    return solve(pt), solve(ps), solve(pt, picks), solve(ps, picks)


def tol(ref):
    return max(TOL_ABS, TOL_REL * abs(ref))


def classify(share, lo, hi):
    if share >= EVENT_SHARE and lo >= EVENT_LOW:
        return "event"
    if share <= STATE_SHARE and hi <= STATE_HIGH:
        return "state"
    return "uncertain"


def summarize(ft, fs, bt, bs):
    a_m, a_e, c_m, k, c_e = ft[0], ft[1], fs[0], fs[1], fs[2]
    out = {"a_m": float(a_m), "a_e": float(a_e), "c_M": float(c_m), "K": float(k), "c_E": float(c_e),
           "K_ci95": [float(x) for x in np.quantile(bs[:, 1], [0.025, 0.975])],
           "a_e_ci95": [float(x) for x in np.quantile(bt[:, 1], [0.025, 0.975])],
           "consistent": bool(abs(c_e - a_e) <= tol(a_e) and abs(c_m - a_m) <= tol(a_m))}
    if abs(a_e) > 1e-6:
        draws = bs[:, 1] / bt[:, 1]
        lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
        out.update(share=float(k / a_e), share_ci95=[lo, hi], cls=classify(k / a_e, lo, hi) if out["consistent"]
                   else "uncertain_inconsistent")
    return out


def analyze():
    protocol()
    mu = pose_mean()
    res = {}
    for net in NETS:
        true = load_task(net, "e2_true")
        res[net] = {}
        for task in SHIFT_ORDER:
            shift = load_task(net, task)
            block = {}
            for name, blk in BLOCKS.items():
                m = mu if name == "finger" else None
                block[name] = summarize(*decompose([design(true, shift, s, blk, m) for s in SEQS], m is not None))
                block[name]["per_recording"] = {s: summarize(*decompose([design(true, shift, s, blk, m)], m is not None))
                                                for s in SEQS}
            block["ra_mm"] = {s: {"own": float(true[f"{s}/mpjpe_ra_mm"].mean()), "shifted": float(shift[f"{s}/mpjpe_ra_mm"].mean())} for s in SEQS}
            res[net][task] = block
    # validity: the T1d control picks the shift; N4 is read only on that shift
    chosen = next((t for t in SHIFT_ORDER if res[CONTROL_NET][t]["finger"]["consistent"]
                   and res[CONTROL_NET][t]["finger"].get("cls") == "event"), None)
    if chosen is None:
        decision = {"status": "STOP_TOOL_INVALID", "shift": None, "n4": None}
    else:
        decision = {"status": "READ", "shift": chosen, "n4": res[DECISION_NET][chosen]["finger"]["cls"]}
    # N4 bank-lag check: per-endpoint (lag - no lag) of N4 minus S45, pooled over both recordings
    lag = {}
    diff = {}
    for s in SEQS:
        d = {net: load_task(net, "e3_lag25")[f"{s}/mpjpe_ra_mm"].astype(np.float64)
             - load_task(net, "e3")[f"{s}/mpjpe_ra_mm"].astype(np.float64) for net in LAG_NETS}
        lag[s] = {f"L_{net}": float(d[net].mean()) for net in LAG_NETS}
        diff[s] = d["n4"] - d["s45"]
    rng = np.random.default_rng(BOOT_SEED)
    pooled = np.concatenate([diff[s] for s in SEQS])
    blocks = [np.array_split(diff[s], int(np.ceil(len(diff[s]) / BLOCK))) for s in SEQS]
    draws = []
    for _ in range(NBOOT):
        parts = [np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))]) for bl in blocks]
        draws.append(np.concatenate(parts).mean())
    lo, hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))
    lag["pooled"] = {"dL_n4_minus_s45": float(pooled.mean()), "ci95": [lo, hi],
                     "reading": "supported" if hi < 0 else ("not_supported" if lo > 0 else "uncertain")}
    s = {"computed_utc": now(), "protocol_sha256": sha(PROTOCOL), "decomposition": res, "decision": decision,
         "lag_check": lag, "single_seed_descriptive": True}
    save(P1 / "summary.json", s)
    write_report(s)
    print(P1 / "summary.json")


def write_report(s):
    f = lambda v: "—" if v is None else f"{v:+.3f}"  # noqa: E731
    lines = ["# S45-RF P1 结果（自动生成；冻结网络、零训练、单 seed 3407 描述性）", "",
             f"判读：{s['decision']['status']}；所用平移 {s['decision']['shift']}；N4 = {s['decision']['n4']}", "",
             "| 网络 | 平移 | a_m | a_e | c_M | K | c_E | 事件份额 s [95 %] | 自检 | 判定 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for net, by in s["decomposition"].items():
        for task, block in by.items():
            r = block["finger"]
            ci = r.get("share_ci95")
            share = "—" if "share" not in r else f"{r['share']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
            lines.append(f"| {net} | {task} | {f(r['a_m'])} | {f(r['a_e'])} | {f(r['c_M'])} | {f(r['K'])} | {f(r['c_E'])} | "
                         f"{share} | {'过' if r['consistent'] else '不过'} | {r.get('cls', '—')} |")
    lag = s["lag_check"]
    lines += ["", f"N4 状态库滞后检查：ΔL（N4 − S45，lag25 − 无滞后，RA mm）{lag['pooled']['dL_n4_minus_s45']:+.2f} "
                  f"[{lag['pooled']['ci95'][0]:+.2f}, {lag['pooled']['ci95'][1]:+.2f}] → {lag['pooled']['reading']}"]
    (P1 / "report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- power (before freezing)
def resample_blocks(r, rng, block=BLOCK):
    """Rows of r in random contiguous blocks of `block`, concatenated and cut to len(r)."""
    n, starts, idx = len(r), np.arange(0, len(r), block), []
    while len(idx) < n:
        b = int(rng.choice(starts))
        idx.extend(range(b, min(b + block, n)))
    return r[np.asarray(idx[:n])]


def power(nrep=50, nboot=400):
    """Cached S45-TR own-packet E2 + GT only (no new reading): simulate the shifted-packet response for a known event
    share s, with block-resampled own-packet residuals x f, and count the registered classifications."""
    require(not POWER.exists(), "power analysis already recorded")
    mu, blk = pose_mean(), BLOCKS["finger"]
    rng = np.random.default_rng(BOOT_SEED)
    out = {"computed_utc": now(), "nrep": nrep, "nboot": nboot, "fits": {}, "cells": {}}
    for net in (DECISION_NET, CONTROL_NET):
        with np.load(e2_reference(net), allow_pickle=False) as z:
            true = {k: z[k] for k in z.files}
        base = {s: design(true, None, s, blk, mu) for s in SEQS}
        a_m, a_e, beta = solve([block_sums(d, "D", ("Mc", "E", "Bm")) for d in base.values()])
        resid = {s: d["D"] - (a_m * d["Mc"] + a_e * d["E"] + beta * d["Bm"]) for s, d in base.items()}
        out["fits"][net] = {"a_m": float(a_m), "a_e": float(a_e), "beta": float(beta)}
        for task in SHIFT_ORDER:
            k, shift = TASKS[task][1], {}
            for s in SEQS:
                n = len(true[f"{s}/end_ms"])
                j = shifted_index(np.arange(n), n, k)
                shift[f"{s}/events_gt_previous"] = true[f"{s}/gt_previous"][j]
                shift[f"{s}/events_target"] = true[f"{s}/target"][j]
                shift[f"{s}/prediction"] = true[f"{s}/prediction"]       # replaced by each simulated response
            geo = {s: design(true, shift, s, blk, mu) for s in SEQS}
            for fnoise in (1.0, 2.0, 3.0):
                for share in (0.0, 0.25, 0.5, 0.75, 1.0):
                    counts = {"event": 0, "state": 0, "uncertain": 0, "uncertain_inconsistent": 0}
                    for _ in range(nrep):
                        sim = dict(shift)
                        for s in SEQS:
                            d = geo[s]
                            w = (a_m * d["Mw"] + share * a_e * d["Sw"] + a_e * d["E"] + beta * d["Bm"]
                                 + fnoise * resample_blocks(resid[s], rng))
                            pred = true[f"{s}/incoming_used"].astype(np.float64).copy()
                            pred[:, blk] += w
                            sim[f"{s}/prediction"] = pred
                        r = summarize(*decompose([design(true, sim, s, blk, mu) for s in SEQS], True, nboot=nboot,
                                                 seed=int(rng.integers(1 << 31))))
                        counts[r["cls"]] += 1
                    out["cells"][f"{net}/{task}/noise{fnoise:.0f}/s{share:.2f}"] = counts
    save(POWER, out, exclusive=True)
    print(POWER)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("power", "freeze", "run", "chain", "check", "analyze"))
    p.add_argument("--net", choices=NETS)
    p.add_argument("--task", choices=tuple(TASKS))
    a = p.parse_args(argv)
    if a.action == "run":
        require(a.net is not None and a.task is not None, "--net and --task required")
        run_task(a.net, a.task)
    else:
        {"power": power, "freeze": freeze, "chain": chain, "check": check, "analyze": analyze}[a.action]()


if __name__ == "__main__":
    main()
