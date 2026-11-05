#!/usr/bin/env python3
"""EVF mechanism table (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 deliverable 2), CPU, read-only on PFO-engine
readings of any study.

    python -m tools.evf.mech --nets name=ROOT:TAG[,name=ROOT:TAG ...] [--out FILE.md]
        ROOT = evf | pfo2 | pfo (outputs/<study>/<date>/readings), TAG = the reading tag

Per network:
  E1 five (MPJPE-L / MPJPE-G / MPVPE-L / MPVPE-G / RA), E3 RA, amplification E1 RA / E3 RA (overall, local, global)
  identity = the RA of predicting the GT previous state itself on the E3 endpoints; E3 - identity separates tracking
             (close to 0) from re-estimation
  retention (zgz_local, median rho) R15 / F15 / T10 / T30
  f1: RA after 1 and 20 feeds of the same packet from the GT previous state; C = RA after 20 feeds from the mean hand
      pose / RA of the mean-pose start (pooled over zgz; < 1 = pulled towards the truth)
  drift: E1 RA by time since the segment start (0-1 / 1-2 / 2-4 / 4-8 / >= 8 s), zgz_local and zgz_global
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evf.common import OUT, PFO2_OUT, ZGZ, read  # noqa: E402

ROOTS = {"evf": OUT, "pfo2": PFO2_OUT, "pfo": ROOT / "outputs/pfo/20261103"}
DRIFT_BINS = (0, 1000, 2000, 4000, 8000, 10 ** 9)
_MANO = {}


def rep(root, tag, task):
    p = ROOTS[root] / "readings" / tag / task / "report.json"
    return read(p) if p.exists() else None


def states(root, tag, task):
    p = ROOTS[root] / "readings" / tag / task / "states.npz"
    return np.load(p) if p.exists() else None


def identity_ra(root, tag):
    """RA of the GT previous state against the target on the E3 endpoints (pooled over zgz)."""
    import torch
    z = states(root, tag, "e3")
    if z is None:
        return None
    from tools.pfo.engine import score, sequences
    from mano_layer import ManoLayer
    if "mano" not in _MANO:
        _MANO["mano"] = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()
        import yaml
        cfg = yaml.safe_load((ROOT / "configs/pfo2_20261103/eval/pfo2_fgs_s3407.yaml").read_text())
        root_dir, seqs = sequences(cfg, ZGZ)
        _MANO["betas"] = {}
        for s, d in seqs:
            aux = np.load(Path(root_dir) / d / f"{s}_aux.npz", allow_pickle=True)
            _MANO["betas"][s] = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)
    tot, n = 0.0, 0
    out = {}
    for s in ZGZ:
        m = score(_MANO["mano"], z[f"{s}/gt_previous"], z[f"{s}/target"], _MANO["betas"][s], torch.device("cpu"))
        out[s] = float(m["mpjpe_ra_mm"].mean())
        tot += float(m["mpjpe_ra_mm"].sum())
        n += len(m["mpjpe_ra_mm"])
    out["overall"] = tot / n
    return out


def row(root, tag):
    e1, e3, pt, f1 = (rep(root, tag, t) for t in ("e1", "e3", "pert", "f1"))
    r = {"tag": tag, "root": root}
    if e1:
        s = e1["per_sequence"]
        r["five"] = [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
                     s["zgz_global"]["mpvpe_ra_mm"], e1["overall"]["mpjpe_ra_mm"]]
        r["e1_lg"] = [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"]]
    if e3:
        s = e3["per_sequence"]
        r["e3"] = e3["overall"]["mpjpe_ra_mm"]
        r["e3_lg"] = [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"]]
        ident = identity_ra(root, tag)
        r["identity"] = ident
        r["e3_minus_identity"] = r["e3"] - ident["overall"]
    if e1 and e3:
        r["amp"] = [r["five"][4] / r["e3"], r["e1_lg"][0] / r["e3_lg"][0], r["e1_lg"][1] / r["e3_lg"][1]]
    if pt:
        loc = pt["per_sequence"]["zgz_local"]
        r["retention"] = {k: loc[k]["rho_median"] for k in ("R15", "F15", "T10", "T30")}
    if f1:
        b = f1["per_sequence"]
        n = {s: b[s]["n"] if "n" in b[s] else 1 for s in ZGZ}
        r["gt1"] = float(np.mean([b[s]["gt_iter1"] for s in ZGZ]))
        r["gt20"] = float(np.mean([b[s]["gt_iter20"] for s in ZGZ]))
        r["C"] = sum(b[s]["mean_iter20"] * n[s] for s in ZGZ) / sum(b[s]["mean_start"] * n[s] for s in ZGZ)
    z = states(root, tag, "e1")
    if z is not None and f"{ZGZ[0]}/elapsed_ms" in z.files:
        r["drift"] = {}
        for s in ZGZ:
            el, ra = z[f"{s}/elapsed_ms"], z[f"{s}/mpjpe_ra_mm"]
            r["drift"][s] = [float(ra[(el >= lo) & (el < hi)].mean()) if ((el >= lo) & (el < hi)).any() else None
                             for lo, hi in zip(DRIFT_BINS[:-1], DRIFT_BINS[1:])]
    return r


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def table(rows):
    L = ["| 网络 | E1 五项 | E3 RA | 恒等 RA | E3 − 恒等 | 放大 E1 / E3（全 / local / global） | R15 / F15 / T10 / T30 | GT 起点 1 → 20 次 | C |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, r in rows:
        five = " / ".join(fmt(x) for x in r["five"]) if "five" in r else "–"
        amp = " / ".join(fmt(x) for x in r["amp"]) if "amp" in r else "–"
        ret = " / ".join(fmt(r["retention"][k]) for k in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        gt = f"{fmt(r.get('gt1'))} → {fmt(r.get('gt20'))}" if "gt1" in r else "–"
        ident = r.get("identity") or {}
        L.append(f"| {name} | {five} | {fmt(r.get('e3'))} | {fmt(ident.get('overall'))} | {fmt(r.get('e3_minus_identity'))} | "
                 f"{amp} | {ret} | {gt} | {fmt(r.get('C'), 3)} |")
    L += ["", "段内漂移（E1 RA，0–1 / 1–2 / 2–4 / 4–8 / ≥ 8 s）：", "", "| 网络 | zgz_local | zgz_global |", "|---|---|---|"]
    for name, r in rows:
        if "drift" in r:
            L.append(f"| {name} | " + " | ".join(" / ".join(fmt(v, 1) for v in r["drift"][s])
                                             for s in ("zgz_local", "zgz_global")) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nets", required=True)
    p.add_argument("--out")
    a = p.parse_args(argv)
    rows = []
    for item in a.nets.split(","):
        name, rest = item.split("=", 1)
        root, tag = rest.split(":", 1)
        rows.append((name, row(root, tag)))
    text = table(rows)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
