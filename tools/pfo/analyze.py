#!/usr/bin/env python3
"""PFO readings -> tables (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103); CPU only, read-only on the readings.

    python -m tools.pfo.analyze phase1 [--out FILE]       D1 / D2 / D3 / cost tables (markdown + json)
    python -m tools.pfo.analyze probes [--out FILE]       D4 ridge probes
    python -m tools.pfo.analyze runs [--out FILE]         phase-2 trained arms (per seed and seed means)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import OUT, ZGZ, read  # noqa: E402
from tools.pfo.engine import net_tag  # noqa: E402

R = OUT / "readings"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_deg", "finger_deg", "wrist_mm")


def rep(tag, task):
    p = R / tag / task / "report.json"
    return read(p) if p.exists() else None


def row5(r):
    """local / global MPJPE, local / global MPVPE, overall RA."""
    if r is None:
        return None
    s = r["per_sequence"]
    return (s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], r["overall"]["mpjpe_ra_mm"])


def fmt(v, nd=2):
    return "–" if v is None else f"{v:.{nd}f}"


def tl_mean(r):
    if r is None:
        return None
    vals = [v["mpjpe_ra_mm"] for v in r["per_sequence"].values()]
    return float(np.median(vals)), float(np.mean(vals))


def f1_c(tag):
    r = rep(tag, "f1")
    if r is None:
        return None
    out = {}
    for s in ZGZ:
        b = r["per_sequence"][s]
        out[s] = {"C": b["mean_iter20"] / b["mean_start"], "mean_start": b["mean_start"], "mean20": b["mean_iter20"],
                  "gt1": b["gt_iter1"], "gt20": b["gt_iter20"], "loop1": b["loop_iter1"], "loop20": b["loop_iter20"]}
    n = {"zgz_global": 1386, "zgz_local": 1204}
    out["pooled_C"] = sum(out[s]["mean20"] * n[s] for s in ZGZ) / sum(out[s]["mean_start"] * n[s] for s in ZGZ)
    return out


def pert_rho(tag):
    r = rep(tag, "pert")
    if r is None:
        return None
    return {s: {k: (v["rho_median"], v["mpjpe_ra_mm"]) for k, v in r["per_sequence"][s].items() if isinstance(v, dict)}
            for s in ZGZ}


DRIFT_BINS = (0, 1000, 2000, 4000, 8000, 10 ** 9)


def drift(tag, task="e1"):
    """E1 RA by elapsed time since the segment start (ms bins), zgz_local and zgz_global."""
    p = R / tag / task / "states.npz"
    if not p.exists():
        return None
    z = np.load(p)
    out = {}
    for s in ZGZ:
        if f"{s}/elapsed_ms" not in z.files:
            return None
        el, ra = z[f"{s}/elapsed_ms"], z[f"{s}/mpjpe_ra_mm"]
        out[s] = [(lo, hi, float(ra[(el >= lo) & (el < hi)].mean()) if ((el >= lo) & (el < hi)).any() else None,
                   int(((el >= lo) & (el < hi)).sum())) for lo, hi in zip(DRIFT_BINS[:-1], DRIFT_BINS[1:])]
    return out


def phase1():
    lines, js = [], {}
    # D1
    nets = ("pfasrdz", "xt0cos_srdwa", "pfat0pdz_cos", "pfagtdz", "s43_s3407", "s43_s3408", "s43_s3409")
    lines += ["## D1 闭环分解", "",
              "| 网络 | E1 local / global / RA | E3 local / global / RA | 放大 local / global / RA | tl_e1 中位 | tl_e3 中位 | C（均值起点 20 次） | GT 起点 1 → 20 次 RA |",
              "|---|---|---|---|---|---|---|---|"]
    for n in nets:
        e1, e3 = rep(n, "e1"), rep(n, "e3")
        a, b = row5(e1), row5(e3)
        amp = None if a is None or b is None else (a[0] / b[0], a[1] / b[1], a[4] / b[4])
        tl1, tl3 = tl_mean(rep(n, "tl_e1")), tl_mean(rep(n, "tl_e3"))
        f = f1_c(n)
        js[n] = {"e1": a, "e3": b, "amp": amp, "tl_e1": tl1, "tl_e3": tl3, "f1": f, "pert": pert_rho(n)}
        gt = None if f is None else (np.mean([f[s]["gt1"] for s in ZGZ]), np.mean([f[s]["gt20"] for s in ZGZ]))
        lines.append(f"| {n} | {fmt(a and a[0])} / {fmt(a and a[1])} / {fmt(a and a[4])} | "
                     f"{fmt(b and b[0])} / {fmt(b and b[1])} / {fmt(b and b[4])} | "
                     f"{fmt(amp and amp[0])} / {fmt(amp and amp[1])} / {fmt(amp and amp[2])} | "
                     f"{fmt(tl1 and tl1[0])} | {fmt(tl3 and tl3[0])} | {fmt(f and f['pooled_C'], 3)} | "
                     f"{fmt(gt and gt[0])} → {fmt(gt and gt[1])} |")
    lines += ["", "E1 漂移：段内经过时间分箱的 RA（mm；local | global；箱 0–1 / 1–2 / 2–4 / 4–8 / ≥8 s）：", "",
              "| 网络 | zgz_local | zgz_global |", "|---|---|---|"]
    for n in nets:
        d = drift(n)
        js[n]["drift"] = d
        if d is None:
            continue
        cell = lambda s: " / ".join("–" if v is None else f"{v:.1f}" for _, _, v, _ in d[s])  # noqa: E731
        lines.append(f"| {n} | {cell('zgz_local')} | {cell('zgz_global')} |")
    lines += ["", "注入误差的保留率 ρ 中位（zgz_local / zgz_global；0 = 一步完全纠正，1 = 原样保留）与该注入下的 RA：", "",
              "| 网络 | T10 | T30 | R5 | R15 | F05 | F15 | N（训练先验） |", "|---|---|---|---|---|---|---|---|"]
    for n in nets:
        p = js[n]["pert"]
        if p is None:
            continue
        cells = []
        for k in ("T10", "T30", "R5", "R15", "F05", "F15", "N"):
            cells.append(f"{p['zgz_local'][k][0]:.2f} / {p['zgz_global'][k][0]:.2f}（{p['zgz_local'][k][1]:.1f} / {p['zgz_global'][k][1]:.1f}）")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    # loop1
    lines += ["", "自身闭环入态走一步（loop1）：入态误差 → 一步后误差，同端点 E3（RA，zgz_local / zgz_global）：", "",
              "| 网络 | 入态 | 一步后 | E3 |", "|---|---|---|---|"]
    for n in nets:
        r = rep(n, "loop1")
        if r is None:
            continue
        s = r["per_sequence"]
        lines.append(f"| {n} | {s['zgz_local']['incoming']['mpjpe_ra_mm']:.2f} / {s['zgz_global']['incoming']['mpjpe_ra_mm']:.2f} | "
                     f"{s['zgz_local']['mpjpe_ra_mm']:.2f} / {s['zgz_global']['mpjpe_ra_mm']:.2f} | "
                     f"{s['zgz_local']['e3']['mpjpe_ra_mm']:.2f} / {s['zgz_global']['e3']['mpjpe_ra_mm']:.2f} |")
    # D2
    from tools.pfo.jobs import D2_TRUNKS, TAUS, D3_ALL, D3_LABEL
    base_route = {"pfasrdz": "sr1(部署)", "pfat0pdz_cos": "t0p(训练时)", "pfagtdz": "label(训练时)"}
    lines += ["", "## D2 路由阶梯（E1 local / global / RA；E3 RA）", "",
              "| 主干（训练路由） | 测试路由 | E1 local | E1 global | E1 RA | E3 RA |", "|---|---|---|---|---|---|"]
    js["d2"] = {}
    for trunk, routes in D2_TRUNKS.items():
        for route in (None,) + routes:
            tag = net_tag(trunk, route)
            a, b = row5(rep(tag, "e1")), row5(rep(tag, "e3"))
            js["d2"][tag] = {"e1": a, "e3": b}
            lines.append(f"| {trunk} | {route or base_route[trunk]} | {fmt(a and a[0])} | {fmt(a and a[1])} | "
                         f"{fmt(a and a[4])} | {fmt(b and b[4])} |")
    for tau in TAUS:
        tag = net_tag("pfasrdz", None, tau)
        a, b = row5(rep(tag, "e1")), row5(rep(tag, "e3"))
        js["d2"][tag] = {"e1": a, "e3": b}
        lines.append(f"| pfasrdz | sr1 τ={tau:g} | {fmt(a and a[0])} | {fmt(a and a[1])} | {fmt(a and a[4])} | {fmt(b and b[4])} |")
    # D3
    lines += ["", "## D3 测试时干预（对同一权重不干预的差；E1 local / global / RA，E3 RA）", "",
              "| 主干 | 干预 | ΔE1 local | ΔE1 global | ΔE1 RA | ΔE3 RA |", "|---|---|---|---|---|---|"]
    js["d3"] = {}
    for trunk, ivs in (("pfasrdz", D3_ALL), ("pfagtdz", D3_LABEL)):
        a0, b0 = row5(rep(trunk, "e1")), row5(rep(trunk, "e3"))
        for iv in ivs:
            tag = net_tag(trunk, None, None, (iv,))
            a, b = row5(rep(tag, "e1")), row5(rep(tag, "e3"))
            d = None if a is None or a0 is None else (a[0] - a0[0], a[1] - a0[1], a[4] - a0[4])
            d3 = None if b is None or b0 is None else b[4] - b0[4]
            js["d3"][tag] = {"e1": a, "e3": b, "d_e1": d, "d_e3": d3}
            lines.append(f"| {trunk} | {iv} | {fmt(d and d[0])} | {fmt(d and d[1])} | {fmt(d and d[2])} | {fmt(d3)} |")
    # cost
    lines += ["", "## C0 成本块（GPU 4 独占，lyq_local 600 包）", "",
              "| 网络 / 实现 / 标志 | 参数 | 执行 MACs 中位 | 主行口径时延（原始 / ×1.75 锚点） | 端到端 p50 / p99 / max | 一步峰值显存 |",
              "|---|---|---|---|---|---|"]
    js["cost"] = {}
    for d in sorted((OUT / "costs").glob("*")) if (OUT / "costs").exists() else []:
        r = read(d / "report.json") if (d / "report.json").exists() else None
        if r is None:
            continue
        js["cost"][d.name] = {k: r[k] for k in ("params_total", "macs_layers", "latency_row", "latency_e2e_ms",
                                                "memory_peak_step_mib", "breakdown_ms_mean", "macs_by_module_mean")}
        lines.append(f"| {d.name} | {r['params_total']:,} | {r['macs_layers']['median'] / 1e6:.1f} M | "
                     f"{r['latency_row']['raw_ms']:.2f} / {r['latency_row']['scaled_ms']:.2f} ms | "
                     f"{r['latency_e2e_ms']['p50']:.2f} / {r['latency_e2e_ms']['p99']:.2f} / {r['latency_e2e_ms']['max']:.2f} ms | "
                     f"{r['memory_peak_step_mib']:.1f} MiB |")
    return lines, js


def ridge_fit(X, Y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    ym = Y.mean(0)
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    W = np.linalg.solve(A, Xs.T @ (Y - ym))
    return lambda Z: ((Z - mu) / sd) @ W + ym


def probes():
    """Per stage: ridge from the captured features to the needed correction (GT end - incoming), fitted on the training
    local recordings (prior noise), scored on zgz as RA MPJPE of incoming + probe, next to the network's own output."""
    import torch
    from mano_layer import ManoLayer
    from tools.xcl.evaluate import score
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    lines = ["## D4 信息探针（岭回归，训练 local 拟合，zgz 测；RA mm，local / global）", "",
             "| 网络 | 测试态 | 网络自身 | 槽(H0) | 槽(H3) | zs | h |", "|---|---|---|---|---|---|---|"]
    js = {}
    stages = {"slots_h0": "cap/slots_h0", "slots": "cap/slots", "zs": "cap/zs", "h": "cap/h"}
    for tag in (net_tag("pfasrdz"), net_tag("pfasrdz", "label"), net_tag("pfagtdz")):
        fit = R / tag / "cap_tl" / "states.npz"
        if not fit.exists():
            continue
        zf = np.load(fit)
        seqs = sorted({k.split("/")[0] for k in zf.files if not k.startswith("cap/")})
        inc = np.concatenate([zf[f"{s}/incoming"] for s in seqs])
        tgt = np.concatenate([zf[f"{s}/target"] for s in seqs])
        Y = (tgt - inc).astype(np.float64)
        models = {}
        for st, key in stages.items():
            X = zf[key].reshape(len(Y), -1).astype(np.float64)
            models[st] = ridge_fit(X, Y, lam=float(X.shape[1]))
        for test in ("cap_zgz_e3", "cap_zgz_loop"):
            p = R / tag / test / "states.npz"
            if not p.exists():
                continue
            z = np.load(p)
            res = {}
            off = 0
            for s in ZGZ:
                n = len(z[f"{s}/incoming"])
                inc_s, tgt_s, own = z[f"{s}/incoming"], z[f"{s}/target"], z[f"{s}/prediction"]
                def three(pred):
                    a = pred.copy(); a[:, 3:6] = tgt_s[:, 3:6]
                    c = pred.copy(); c[:, 6:51] = tgt_s[:, 6:51]
                    return [float(score(mano, x, tgt_s, _betas(s), "cpu")["mpjpe_ra_mm"].mean()) for x in (pred, a, c)]
                row = {"own": three(own)}
                for st, key in stages.items():
                    X = z[key][off:off + n].reshape(n, -1).astype(np.float64)
                    pred = (inc_s + models[st](X)).astype(np.float32)
                    row[st] = three(pred)
                res[s] = row
                off += n
            js[f"{tag}/{test}"] = res
            for j, part in enumerate(("全部", "只手指(GT根)", "只根(GT手指)")):
                cells = [f"{res['zgz_local'][k][j]:.2f} / {res['zgz_global'][k][j]:.2f}" for k in ("own",) + tuple(stages)]
                lines.append(f"| {tag} | {test} {part} | " + " | ".join(cells) + " |")
    return lines, js


_BETAS = {}


def _betas(seq):
    import torch
    if seq not in _BETAS:
        from tools.pfo.engine import sequences
        from config import load_config
        from semkine import eval_track as ET
        cfg = load_config(str(ROOT / "configs/s45sm_20261031/s45sm_pfasrdz_s3407.yaml"))
        root, seqs = sequences(cfg, (seq,))
        _, _, aux, _ = ET.load_sequence(root, seqs[0][1], seq)
        _BETAS[seq] = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)
    return _BETAS[seq]


def runs():
    from tools.pfo.arms import ARMS, SEEDS, run_name
    lines = ["## 第 2 阶段训练臂（第 6000 步；local / global MPJPE、local / global MPVPE、RA；E3 RA；tl_e1 中位；C）", "",
             "| 臂 | 种子 | E1 五项 | E1@5000 / 5500 RA | E3 RA | tl_e1 | C |", "|---|---|---|---|---|---|---|"]
    js = {}
    for arm in ARMS:
        per = {}
        for seed in SEEDS:
            tag = run_name(arm, seed)
            a = row5(rep(tag, "e1"))
            if a is None:
                continue
            e3 = row5(rep(tag, "e3"))
            late = [row5(rep(f"{tag}@{s}", "e1")) for s in (5000, 5500)]
            tl = tl_mean(rep(tag, "tl_e1"))
            f = f1_c(tag)
            per[seed] = {"e1": a, "e3": e3, "late": late, "tl": tl, "C": f and f["pooled_C"]}
            lines.append(f"| {arm} | {seed} | {' / '.join(fmt(v) for v in a)} | "
                         f"{' / '.join(fmt(l and l[4]) for l in late)} | {fmt(e3 and e3[4])} | {fmt(tl and tl[0])} | "
                         f"{fmt(f and f['pooled_C'], 3)} |")
        if per:
            m = np.mean([v["e1"] for v in per.values()], 0)
            lines.append(f"| {arm} | 均值（{len(per)}） | {' / '.join(fmt(v) for v in m)} | | | | |")
            js[arm] = {"per_seed": per, "mean": m.tolist(), "n": len(per)}
    return lines, js


def decomp(tags=("pfasrdz", "xt0cos_srdwa", "pfat0pdz_cos", "pfagtdz", "s43_s3407", "s43_s3408", "s43_s3409"),
           task="e1"):
    """RA of the saved predictions with the GT root rotation swapped in (finger error only) and with the GT fingers
    swapped in (root-rotation error only); translation never matters for RA."""
    from mano_layer import ManoLayer
    from tools.xcl.evaluate import score
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
    lines = [f"## {task}：RA 分解（mm；local / global）", "", "| 网络 | 原始 | GT 根旋转（只剩手指） | GT 手指（只剩根） |", "|---|---|---|---|"]
    js = {}
    for tag in tags:
        p = R / tag / task / "states.npz"
        if not p.exists():
            continue
        z = np.load(p)
        row = {}
        for s in ZGZ:
            pred, tgt = z[f"{s}/prediction"].copy(), z[f"{s}/target"]
            b = _betas(s)
            a = pred.copy(); a[:, 3:6] = tgt[:, 3:6]
            c = pred.copy(); c[:, 6:51] = tgt[:, 6:51]
            row[s] = [float(score(mano, x, tgt, b, "cpu")["mpjpe_ra_mm"].mean()) for x in (pred, a, c)]
        js[tag] = row
        lines.append(f"| {tag} | " + " | ".join(f"{row['zgz_local'][i]:.2f} / {row['zgz_global'][i]:.2f}" for i in range(3)) + " |")
    return lines, js


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("phase1", "probes", "runs", "decomp", "decomp_e3"))
    p.add_argument("--out")
    a = p.parse_args(argv)
    lines, js = {"phase1": phase1, "probes": probes, "runs": runs, "decomp": decomp,
                 "decomp_e3": lambda: decomp(task="e3")}[a.what]()
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(js, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
