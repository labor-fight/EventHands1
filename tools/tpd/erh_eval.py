#!/usr/bin/env python3
"""TPD EvRealHands final test (docs/EXPERIMENT_RECORDS.md#tpd-erh-prereg-20261106; report only: no reading here enters any
rule, selection or main row). Recordings from tools/tpd/erh_convert.py (right hand; tiers 3d = manually checked MANO
labels, the main tier; 3d_unchecked = unchecked labels, descriptive).

    python -m tools.tpd.erh_eval check --net b2|c0|s43           K_E1: this runner's stepping reproduces PFO's engine e1
                                                                  predictions on zgz_local bit for bit (first 50 packets)
    python -m tools.tpd.erh_eval run --net b2|s43|rol5c|arm --seed S [--arm A] --tier 3d|3d_unchecked [--mask gt]
    python -m tools.tpd.erh_eval table                            analysis/erh.md, decisions/erh_table.json (once)

Readings (labels at 15 Hz; packets of 50 ms; 3 labels = 200 ms = 4 packets, so every third label falls on a packet end):
  loop     per contiguous labelled run: start at its first label with GT + the engine's initial noise (the network's own
           TRACK.PREV_NOISE_*, a per-run seeded draw), own closed loop on the 50 ms grid, scored at the labels on packet
           ends (5 Hz)
  h200     from GT at every scored label, 4 packets, scored at the next scored label (single-step quality, descriptive)
Baseline "hold": the start state held (no update), computed by `table` without a network.
Diagnostic `--mask gt` (uses GT, never deployable; tag suffix __gtmask): only the events inside the GT hand silhouette of
the nearest label (our FK of the label, the model's K, dilated 6 px; events farther than 33 ms from every label dropped),
so the reading separates the off-hand events of a full-scene recording from everything else.
Networks run as in their zgz rows: B2 (PFO2 fgs, B2's node sampler), S43 (its own checkpoint), rol5 in the causal form
(B2 + rol5 fine-tune, the ROL-c fallback), rolk (rol5 trained in the causal form, step 1500), a synthesis arm in its
deployment form; the causal form gets its stride k from
the 50 ms before every packet (semkine/tpd_causal.py), set just before the step. No label source is read: the deployed
route of `tools.pfo.nets.load` replaces `attach_xag_sources` by a no-op, as in every zgz reading (K_E1).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.tpd.common import OUT, PFO2_OUT, now, read, require, save, sha  # noqa: E402

ERH = OUT / "erh"
TIERS = ("3d", "3d_unchecked")
TABLE_TIERS = ("3d",)                    # the registered readings (3d_unchecked: labels not checked by the dataset)
SCENES = ("normal", "highlight", "flash")
NETS = ("b2", "s43", "rol5c", "rolk", "arm")
SEEDS = (3407, 3408, 3409)
PACKET_MS = 50
STRIDE = 3                       # labels between scored labels (3 x 66.67 ms = 200 ms)
HORIZON = 4                      # packets between scored labels
SCORED_MS = PACKET_MS * HORIZON
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm", "wrist_mm", "root_deg", "finger_deg")
CHECK_PACKETS = 50
MASK_DILATE = 6
MASK_HALF_MS = 33


# --------------------------------------------------------------------------- recordings
def trials(tier):
    t = read(ERH / "trials.json")["trials"]
    smoke = os.environ.get("TPD_ERH_SMOKE_TRIALS")
    return sorted((k for k, v in t.items() if v["tier"] == tier and (not smoke or k in smoke.split(","))), key=int)


def scored_runs(labels):
    """[(label_ms, params51)] per contiguous labelled run with at least one scored label after its start; label k of a
    run is scored when k % STRIDE == 0 (k > 0), and its ms is the run start + 200 ms x k / STRIDE."""
    from tools.tpd import erh_convert as EC
    ids, ms, P = labels["frame_id"], np.asarray(labels["label_ms"], np.int64), labels["params51"]
    out = []
    for a, b in EC.label_runs(ids):
        if b - a < STRIDE:
            continue
        m, p = ms[a:b + 1], np.asarray(P[a:b + 1], np.float32)
        k = np.arange(0, len(m), STRIDE)
        require(np.array_equal(m[k] - m[0], k // STRIDE * SCORED_MS), "label grid is not 200 ms per 3 labels")
        out.append((m, p))
    return out


def gt_mask_context(ctx):
    """Diagnostic only (GT): the recording with only the events inside the dilated GT silhouette of the nearest label."""
    import dataclasses
    from tools.tpd import erh_convert as EC
    ms = np.asarray(ctx.labels["label_ms"], np.int64)
    _, V = EC.fk51(ctx.labels["params51"], ctx.betas.view(-1).cpu().numpy())
    K = ctx.K.view(3, 3).cpu().numpy().astype(np.float64) * EC.RENDER_SCALE
    uv = np.stack([K[0, 0] * V[..., 0] / V[..., 2] + K[0, 2], K[1, 1] * V[..., 1] / V[..., 2] + K[1, 2]], -1)
    faces = EC.get_mano().f.numpy()
    n_ms = len(ctx.offsets) - 1
    mid = np.concatenate([[ms[0] - MASK_HALF_MS], (ms[:-1] + ms[1:] + 1) // 2, [ms[-1] + MASK_HALF_MS + 1]])
    keep = np.zeros(int(ctx.offsets[-1]), bool)
    ev = np.asarray(ctx.events)
    for i, m in enumerate(ms):
        lo, hi = max(int(mid[i]), int(m) - MASK_HALF_MS, 0), min(int(mid[i + 1]), int(m) + MASK_HALF_MS + 1, n_ms)
        if hi <= lo:
            continue
        a, b = int(ctx.offsets[lo]), int(ctx.offsets[hi])
        mask = EC.silhouette(uv[i], faces, MASK_DILATE)
        keep[a:b] = mask[ev[a:b, 1].astype(np.int64), ev[a:b, 0].astype(np.int64)]
    rows = np.concatenate([[0], np.cumsum(keep)])
    return dataclasses.replace(ctx, events=ev[keep], offsets=rows[np.asarray(ctx.offsets)].astype(np.int64),
                               tsub=np.asarray(ctx.tsub)[keep])


# --------------------------------------------------------------------------- networks
def _spec_obj(sp):
    s = read(sp)
    return {**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])}


def load_net(net, seed, arm=None, device=None):
    """(tag, tools.pfo.nets.Net, causal)."""
    from semkine import tpd_causal as TC
    from tools.pfo import nets as N
    if net == "rol5c":
        from tools.tpd import ns_seed
        ns_seed.enter()                                         # before tools.tpd.arms_rol is imported (seed 3409)
    if net == "rolk":
        from tools.tpd import ns_rolk
        ns_rolk.enter()
    from tools.tpd import run_rol as RR
    causal = False
    if net == "s43":
        tag, spec = f"s43_s{seed}", None
    elif net in ("b2", "b2c"):
        from tools.tpd.common import b2_checkpoint, b2_spec
        b2_checkpoint(seed)
        sp = b2_spec(seed)
        RR.prepare(sp)
        tag, spec, causal = f"pfo2_fgs_s{seed}" + ("__causal" if net == "b2c" else ""), _spec_obj(sp), net == "b2c"
    elif net == "rolk":
        from tools.tpd import run_rolk as RK
        tag, sp = RK.spec(seed)
        RR.prepare(sp)
        spec, causal = _spec_obj(sp), True
    elif net == "rol5c":
        tag, sp = RR.arm_spec("rol5", seed)
        RR.prepare(sp)
        tag, spec, causal = tag + "__causal", _spec_obj(sp), True
    else:
        from tools.tpd import arms_final as A
        from tools.tpd import run_final as RF
        tag, sp = RF.arm_spec(arm, seed)
        RR.prepare(sp)
        spec, causal = _spec_obj(sp), bool(A.ARMS[arm]["causal"])
    if causal:
        TC.install()
    model_net = N.load(tag, device=device, spec=spec)
    if causal:
        TC.retoken()
    return tag, model_net, causal


def stepper(net, causal):
    if not causal:
        return net.step
    from semkine import tpd_causal as TC

    def step(ctx, end, prev):
        TC.set_k([TC.k_from_prev(TC.n_prev_events(ctx.offsets, int(end)))])
        return net.step(ctx, end, prev)
    return step


def init_noise(cfg, trial, rid):
    from semkine import eval_track as ET
    return ET.sample_init_noise(cfg, np.random.default_rng([int(trial), int(rid)]), 1.0)


# --------------------------------------------------------------------------- readings
def policy():
    from tools.xcl import evaluate as XE
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    XE.configure_environment()
    XE.set_policy()


def run(net, seed, tier, arm=None, mask=None):
    import torch
    from mano_layer import ManoLayer
    from tools.tpd import erh_convert as EC
    from tools.xcl.evaluate import score
    require(tier in TIERS and mask in (None, "gt"), "tier / mask")
    policy()
    device = torch.device("cuda")
    tag, N_, causal = load_net(net, seed, arm, device)
    tag += "__gtmask" if mask else ""
    smoke = os.environ.get("TPD_ERH_SMOKE_DIR")                # engineering smoke: scratch root, TPD_ERH_SMOKE_TRIALS
    out = (Path(smoke) if smoke else ERH / "readings") / tag / tier
    require(not out.exists(), "preserve existing reading: " + str(out))
    out.mkdir(parents=True)
    launch = {"net": net, "seed": seed, "arm": arm, "tag": tag, "tier": tier, "causal": causal, "mask": mask, "identity": {
        k: str(v) for k, v in N_.identity.items()}, "checkpoint_sha256": sha(N_.identity["checkpoint"]),
              "runner_sha256": sha(__file__), "convert_sha256": sha(Path(EC.__file__)),
              "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": now()}
    save(out / "launch.json", launch, exclusive=True)
    mano = ManoLayer(N_.cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    step = stepper(N_, causal)
    info = read(ERH / "trials.json")["trials"]
    arrays, per_trial = {}, {}
    pooled = {"loop": {k: [] for k in KEYS}, "h200": {k: [] for k in KEYS}}
    scene = {}
    with torch.no_grad():
        for trial in trials(tier):
            ctx = EC.load(trial, device)
            if mask:
                ctx = gt_mask_context(ctx)
            N_.bind(ctx)
            rec = {"loop": {"prediction": [], "target": [], "elapsed_ms": [], "run": []},
                   "h200": {"prediction": [], "target": []}}
            for rid, (ms, P) in enumerate(scored_runs(ctx.labels)):
                a = int(ms[0])
                prev = P[0] + init_noise(N_.cfg, trial, rid)
                last = int(ms[(len(ms) - 1) // STRIDE * STRIDE])
                for end in range(a + PACKET_MS, last + 1, PACKET_MS):
                    prev = step(ctx, end, prev)
                    if (end - a) % SCORED_MS == 0:
                        k = (end - a) // SCORED_MS * STRIDE
                        require(int(ms[k]) == end, "scored label off the packet grid")
                        rec["loop"]["prediction"].append(prev); rec["loop"]["target"].append(P[k])
                        rec["loop"]["elapsed_ms"].append(end - a); rec["loop"]["run"].append(rid)
                for k in range(0, len(ms) - STRIDE, STRIDE):
                    s0, x = int(ms[k]), P[k]
                    for j in range(1, HORIZON + 1):
                        x = step(ctx, s0 + j * PACKET_MS, x)
                    rec["h200"]["prediction"].append(x); rec["h200"]["target"].append(P[k + STRIDE])
            per_trial[trial] = {"scene": info[trial]["scene"], "subject": info[trial]["subject_id"],
                                "gesture": info[trial]["gesture"]}
            for task in ("loop", "h200"):
                pred = np.asarray(rec[task]["prediction"], np.float32)
                tgt = np.asarray(rec[task]["target"], np.float32)
                arrays[f"{trial}/{task}/prediction"], arrays[f"{trial}/{task}/target"] = pred, tgt
                if task == "loop":
                    arrays[f"{trial}/loop/elapsed_ms"] = np.asarray(rec["loop"]["elapsed_ms"], np.int64)
                    arrays[f"{trial}/loop/run"] = np.asarray(rec["loop"]["run"], np.int64)
                if not len(pred):
                    continue
                m = score(mano, pred, tgt, ctx.betas, device)
                per_trial[trial][task] = {k: float(np.mean(m[k])) for k in KEYS} | {"n": int(len(pred))}
                for k in KEYS:
                    pooled[task][k].append(np.asarray(m[k], np.float64))
                if task == "loop":
                    sc = scene.setdefault(info[trial]["scene"], [])
                    sc.append(np.asarray(m["mpjpe_ra_mm"], np.float64))
            save(out / "progress.json", {"done": trial, "at": now()})
    overall = {t: {k: float(np.concatenate(v).mean()) for k, v in d.items()} | {"n": int(sum(map(len, d["mpjpe_ra_mm"])))}
               for t, d in pooled.items()}
    by_scene = {s: {"mpjpe_ra_mm": float(np.concatenate(v).mean()), "n": int(sum(map(len, v)))} for s, v in scene.items()}
    np.savez_compressed(out / "states.npz", **arrays)
    save(out / "report.json", {**launch, "status": "COMPLETE", "finished_utc": now(), "overall": overall,
                               "loop_by_scene": by_scene, "per_trial": per_trial, "states_sha256": sha(out / "states.npz")},
         exclusive=True)
    print("COMPLETE", out, {t: round(overall[t]["mpjpe_ra_mm"], 2) for t in overall}, flush=True)


# --------------------------------------------------------------------------- K_E1
CHECK_READINGS = {"b2": (PFO2_OUT / "readings/pfo2_fgs_s3407/e1/states.npz", ("b2", 3407, None)),
                  "c0": (OUT / "readings/tpd_c0_s3407/e1/states.npz", ("arm", 3407, "c0")),
                  "s43": (ROOT / "outputs/pfo/20261103/readings/s43_s3407/e1/states.npz", ("s43", 3407, None))}


def check(name):
    import torch
    from tools.pfo import nets as N
    from tools.pfo.engine import sequences
    policy()
    out = OUT / "receipts" / f"erh_k_e1_{name}.json"
    require(not out.exists(), "checked once")
    src, (net, seed, arm) = CHECK_READINGS[name]
    device = torch.device("cuda")
    tag, N_, causal = load_net(net, seed, arm, device)
    root, seqs = sequences(N_.cfg, ("zgz_local",))
    ctx = N.sequence(root, seqs[0][1], "zgz_local", device)
    N_.bind(ctx)
    step = stepper(N_, causal)
    with np.load(src, allow_pickle=False) as z:
        inc, pred, ends = (z[f"zgz_local/{k}"][:CHECK_PACKETS] for k in ("incoming", "prediction", "end_ms"))
        elapsed = z["zgz_local/elapsed_ms"][:CHECK_PACKETS]
    require(elapsed[0] == 49 and np.all(np.diff(elapsed) == 50), "the first packets are one run from its start")
    mine, prev = [], inc[0]
    with torch.no_grad():
        for end in ends:
            prev = step(ctx, int(end), prev)
            mine.append(prev)
    mine = np.asarray(mine, np.float32)
    ok = bool(np.array_equal(mine, pred))
    save(out, {"status": "PASS" if ok else "FAIL", "net": tag, "causal": causal, "reference": str(src),
               "reference_sha256": sha(src), "packets": int(len(ends)), "max_abs_diff": float(np.abs(mine - pred).max()),
               "written_utc": now()}, exclusive=True)
    print(name, "PASS" if ok else "FAIL", float(np.abs(mine - pred).max()))
    return 0 if ok else 1


# --------------------------------------------------------------------------- table
def _report(tag, tier):
    p = ERH / "readings" / tag / tier / "report.json"
    return read(p) if p.exists() else None


def hold(tier):
    """The start state held: pooled loop / h200 metrics over the tier (CPU, no network)."""
    import torch
    from mano_layer import ManoLayer
    from tools.tpd import erh_convert as EC
    from tools.xcl.evaluate import score
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    acc = {"loop": {k: [] for k in KEYS}, "h200": {k: [] for k in KEYS}}
    for trial in trials(tier):
        ctx = EC.load(trial, "cpu")
        pairs = {"loop": [], "h200": []}
        for ms, P in scored_runs(ctx.labels):
            k = np.arange(STRIDE, len(ms), STRIDE)
            pairs["loop"].append((np.repeat(P[:1], len(k), 0), P[k]))
            k0 = np.arange(0, len(ms) - STRIDE, STRIDE)
            pairs["h200"].append((P[k0], P[k0 + STRIDE]))
        for task, pp in pairs.items():
            m = score(mano, np.concatenate([a for a, _ in pp]), np.concatenate([b for _, b in pp]), ctx.betas,
                      torch.device("cpu"))
            for key in KEYS:
                acc[task][key].append(np.asarray(m[key], np.float64))
    return {t: {k: float(np.concatenate(v).mean()) for k, v in d.items()} | {"n": int(sum(map(len, d["mpjpe_ra_mm"])))}
            for t, d in acc.items()}


def table():
    out = OUT / "decisions" / "erh_table.json"
    require(not out.exists(), "written once")
    keep = OUT / "decisions" / "syn_keep.json"
    final = read(keep)["final_arm"] if keep.exists() else None
    groups = [("S43", [f"s43_s{s}" for s in SEEDS]), ("B2", [f"pfo2_fgs_s{s}" for s in SEEDS]),
              ("B2 + rol5（因果形式）", [f"tpd_rol5_s{s}__causal" for s in SEEDS]),
              ("B2 + rolk（因果形式训练的 rol5）", [f"tpd_rolk_s{s}" for s in SEEDS])]
    if final:
        groups.append((f"TPD 最终配置 {final}", [f"tpd_{final}_s{s}" for s in SEEDS]))
    groups += [("诊断：S43，GT 轮廓内事件", ["s43_s3407__gtmask"]), ("诊断：B2，GT 轮廓内事件", ["pfo2_fgs_s3407__gtmask"]),
               ("诊断：B2 + rol5（因果），GT 轮廓内事件", ["tpd_rol5_s3407__causal__gtmask"])]
    rows, L = {}, []
    for tier in TABLE_TIERS:
        h = hold(tier)
        rows[f"{tier}/hold"] = h
        scenes = []
        L += [f"### {tier}", "", "| 网络 | 种子 | RA-MPJPE | RA-MPVPE | MPJPE | MPVPE | 200 ms RA-MPJPE | 标签数 |",
              "|---|---|---|---|---|---|---|---|",
              f"| 保持起点状态（不更新） | – | {h['loop']['mpjpe_ra_mm']:.2f} | {h['loop']['mpvpe_ra_mm']:.2f} | "
              f"{h['loop']['mpjpe_abs_mm']:.2f} | {h['loop']['mpvpe_abs_mm']:.2f} | {h['h200']['mpjpe_ra_mm']:.2f} | "
              f"{h['loop']['n']} |"]
        for name, tags in groups:
            got = [(t, _report(t, tier)) for t in tags]
            got = [(t, r) for t, r in got if r is not None]
            for t, r in got:
                o = r["overall"]
                rows[f"{tier}/{t}"] = {"loop": o["loop"], "h200": o["h200"], "by_scene": r["loop_by_scene"]}
            if not got:
                continue
            mean = lambda task, k: sum(r["overall"][task][k] for _, r in got) / len(got)  # noqa: E731
            L.append(f"| {name} | {len(got)} 种子 | {mean('loop', 'mpjpe_ra_mm'):.2f} | {mean('loop', 'mpvpe_ra_mm'):.2f} | "
                     f"{mean('loop', 'mpjpe_abs_mm'):.2f} | {mean('loop', 'mpvpe_abs_mm'):.2f} | "
                     f"{mean('h200', 'mpjpe_ra_mm'):.2f} | {got[0][1]['overall']['loop']['n']} |")
            scenes.append((name, {sc: sum(r["loop_by_scene"][sc]["mpjpe_ra_mm"] for _, r in got) / len(got)
                                  for sc in SCENES if all(sc in r["loop_by_scene"] for _, r in got)}))
        L += ["", "闭环 RA-MPJPE 按场景（种子均值）：", "", "| 网络 | " + " | ".join(SCENES) + " |",
              "|---|" + "---|" * len(SCENES)]
        L += [f"| {n} | " + " | ".join(f"{d[sc]:.2f}" if sc in d else "–" for sc in SCENES) + " |" for n, d in scenes]
        L.append("")
    text = "## TPD EvRealHands 最终测试（只报；右手；DAVIS346 → 240 × 180；mm）\n\n" + "\n".join(L)
    (OUT / "analysis").mkdir(exist_ok=True)
    (OUT / "analysis" / "erh.md").write_text(text)
    save(out, {"rows": rows, "final_arm": final, "written_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    c = sub.add_parser("check")
    c.add_argument("--net", required=True, choices=tuple(CHECK_READINGS))
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=NETS)
    r.add_argument("--seed", type=int, required=True, choices=SEEDS)
    r.add_argument("--arm")
    r.add_argument("--tier", required=True, choices=TIERS)
    r.add_argument("--mask", choices=("gt",))
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "check":
        return check(a.net)
    if a.verb == "run":
        require((a.net == "arm") == (a.arm is not None), "--arm goes with --net arm")
        return run(a.net, a.seed, a.tier, a.arm, a.mask)
    return table()


if __name__ == "__main__":
    sys.exit(main())
