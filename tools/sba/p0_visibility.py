#!/usr/bin/env python3
"""SBA post-hoc analysis of the P0 / Q0 stops (docs/EXPERIMENT_RECORDS.md#sba-why-20261022); descriptive, no new
likelihood evaluations, CPU only.

    run        for every P0 packet, render the visible part-group label map (painter's algorithm, 640 x 480 source)
               at the GT end knot and under each P0 hypothesis, plus two reference finger perturbations that P0 does
               not use (+-20 deg about local z = flexion, local y = abduction); record the changed-label area in
               binned (240 x 180) px^2 and the max vertex displacement in binned px
    summarize  join with the P0 per-packet likelihoods (profiled kappa, informative packets) and the Q0 sigma*:
               GT preference vs visible change per hypothesis family, finger preference by joint level, and the
               per-recording z* against depth x (1 / sigma* - 1)

    python tools/sba/p0_visibility.py run
    python tools/sba/p0_visibility.py summarize
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB  # noqa: E402
from sba.p0_landscape import packets, ZLINE  # noqa: E402

OUT = CM.OUT / "p0v"
BIN2 = (3 / 8) ** 2
FAM = {"finger": ["f+20", "f-20"], "root": CM.ROOT10, "trans": ["x+20", "x-20", "y+20", "y-20", "z+50", "z-50"]}
EXTRA = {"flex+20": (2, 20), "flex-20": (2, -20), "abd+20": (1, 20), "abd-20": (1, -20)}
AREA_BINS = (0, 5, 10, 20, 40, 80, 160, 320, 10 ** 9)


def perturb_axis(ctx, s, joint, axis, deg):
    """As CM.perturb's finger branch, about local axis `axis` instead of x."""
    s = s.clone().view(1, 51)
    full = s[0, 6 + 3 * joint:9 + 3 * joint] + ctx.mean[joint]
    w = torch.zeros(3, device=s.device)
    w[axis] = math.radians(deg)
    s[0, 6 + 3 * joint:9 + 3 * joint] = CM.so3_log(CM.so3_exp(full)[None] @ CM.so3_exp(w)[None])[0] - ctx.mean[joint]
    return s[0]


def render(uv, z, faces, fgroup, box):
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = box
    img = Image.new("L", (x1 - x0, y1 - y0), 0)
    dr = ImageDraw.Draw(img)
    for f in np.argsort(-z[faces].mean(1)):                                  # far -> near
        p = uv[faces[f]] - (x0, y0)
        dr.polygon([tuple(p[0]), tuple(p[1]), tuple(p[2])], fill=int(fgroup[f]) + 1)
    return np.asarray(img)


def work(job):
    torch.set_num_threads(1)
    seq, d, items = job
    ctx = CM.Ctx(torch.device("cpu"))
    sd = CM.SeqData(seq, d, torch.device("cpu"))
    faces = ctx.mano.f.long().numpy()
    vg = ctx.geo.groups.numpy()
    fgroup = np.array([np.bincount(vg[f], minlength=6).argmax() for f in faces])
    out = []
    for index, end in items:
        x1 = sd.gt(end)
        joint = index % 15
        xs = {"clean": x1}
        for h in sum(FAM.values(), []):
            kind, val = CM.PERTURB[h]
            xs[h] = CM.perturb(ctx, x1, kind, val, joint, sd.betas)
        for h, (ax, deg) in EXTRA.items():
            xs[h] = perturb_axis(ctx, x1, joint, ax, deg)
        names = list(xs)
        with torch.no_grad():
            v, j = ctx.geo.fk(torch.stack([xs[n] for n in names]), sd.betas)
            uv = SB.project_src(v, sd.K).numpy()
        z = v[..., 2].numpy()
        jr = (j - j[:, :1]).numpy() * 1000
        lo = np.floor(uv.reshape(-1, 2).min(0)).astype(int) - 2
        hi = np.ceil(uv.reshape(-1, 2).max(0)).astype(int) + 2
        box = (int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]))
        maps = [render(uv[i], z[i], faces, fgroup, box) for i in range(len(names))]
        rec = {"seq": seq, "index": index, "joint": joint, "depth": float(x1[2]),
               "hand_px": float(np.sqrt((maps[0] > 0).sum() * BIN2))}
        for i, n in enumerate(names[1:], 1):
            rec[n] = {"area": float((maps[i] != maps[0]).sum() * BIN2),
                      "dmax": float(np.linalg.norm(uv[i] - uv[0], axis=1).max() * 3 / 8),
                      "mpjpe_mm": float(np.linalg.norm(jr[i] - jr[0], axis=1).mean())}
        out.append(rec)
    return out


def run(procs):
    jobs = []
    for domain in ("zgz_local", "zgz_global", "check"):
        by = {}
        for seq, d, index, row, rows in packets(domain):
            by.setdefault((seq, d), []).append((index, int(row["end_ms"])))
        for (seq, d), items in by.items():
            jobs += [(seq, d, items[k:k + 40]) for k in range(0, len(items), 40)]
    with Pool(procs) as pool:
        res = [r for chunk in pool.imap_unordered(work, jobs) for r in chunk]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "visibility.json").write_text(json.dumps(res))
    CM.receipt(OUT / "run", {"probe": "P0 post-hoc visibility", "packets": len(res)}, scripts=(__file__,))
    print(len(res), "packets")


def summarize():
    vis = {(r["seq"], r["index"]): r for r in json.loads((OUT / "visibility.json").read_text())}
    p0 = [r for f in sorted(glob.glob(str(CM.OUT / "p0/raw/*of*.json"))) for r in json.loads(Path(f).read_text())["records"]
          if r["n_events"] >= CM.MIN_EVENTS]
    q0 = {(r["seq"], r["index"]): r["H"]["clean"]["sigma_star"]
          for f in sorted(glob.glob(str(CM.OUT / "q0/raw/*of*.json"))) for r in json.loads(Path(f).read_text())["records"]
          if "H" in r}
    dom_of = lambda s: s if s in CM.ZGZ else "check"                                           # noqa: E731
    rows = []
    for r in p0:
        v = vis[(r["seq"], r["index"])]
        for fam, hs in FAM.items():
            for h in hs:
                rows.append((dom_of(r["seq"]), fam, h, v["joint"] % 3, v[h]["area"], r["A"]["clean"][1] > r["A"][h][1]))
    res = {"families": {}, "finger_by_level": {}, "pref_vs_area": {}, "reference_area_median": {}, "mpjpe_mm_mean": {},
           "recordings": {}}
    for d in ("zgz_local", "zgz_global", "check"):
        rd = [x for x in rows if x[0] == d]
        res["families"][d] = {f: {"n": len([x for x in rd if x[1] == f]),
                                  "area_median": float(np.median([x[4] for x in rd if x[1] == f])),
                                  "pref": float(np.mean([x[5] for x in rd if x[1] == f]))} for f in FAM}
        res["finger_by_level"][d] = {L: {"n": len(s), "area_median": float(np.median([x[4] for x in s])),
                                         "pref": float(np.mean([x[5] for x in s]))}
                                     for L, k in (("MCP", 0), ("PIP", 1), ("DIP", 2))
                                     for s in [[x for x in rd if x[1] == "finger" and x[3] == k]]}
        res["pref_vs_area"][d] = {f"[{lo},{hi})": {f: (lambda s: {"n": len(s), "pref": float(np.mean(s)) if s else None})(
            [x[5] for x in rd if x[1] == f and lo <= x[4] < hi]) for f in FAM}
            for lo, hi in zip(AREA_BINS[:-1], AREA_BINS[1:])}
        vd = [v for v in vis.values() if dom_of(v["seq"]) == d]
        res["reference_area_median"][d] = {k: float(np.median([v[h]["area"] for v in vd for h in hs])) for k, hs in
                                           (("twist_P0", FAM["finger"]), ("flexion", ["flex+20", "flex-20"]),
                                            ("abduction", ["abd+20", "abd-20"]), ("root10", FAM["root"]))}
        res["mpjpe_mm_mean"][d] = {k: float(np.mean([v[h]["mpjpe_mm"] for v in vd for h in hs])) for k, hs in
                                   (("twist_P0", FAM["finger"]), ("flexion", ["flex+20", "flex-20"]),
                                    ("abduction", ["abd+20", "abd-20"]), ("root10", FAM["root"]))}
    by = collections.defaultdict(list)
    for r in p0:
        zs = ZLINE[int(np.argmax([r["A"][f"zl{q:+d}"][1] for q in ZLINE]))]
        base = r["seq"].replace("_v2", "").replace("_v3", "").replace("_v4", "")
        by[base].append((zs, q0.get((r["seq"], r["index"])), vis[(r["seq"], r["index"])]["depth"]))
    for s, xs in sorted(by.items()):
        sg = float(np.median([b for _, b, _ in xs if b is not None]))
        dep = float(np.median([c for _, _, c in xs]))
        res["recordings"][s] = {"n": len(xs), "zstar_median_mm": float(np.median([a for a, _, _ in xs])),
                                "sigma_star_q0_median": sg, "depth_m": dep, "pred_zstar_mm": dep * (1 / sg - 1) * 1000}
    rec = res["recordings"]
    res["pearson_pred_vs_zstar"] = float(np.corrcoef([v["pred_zstar_mm"] for v in rec.values()],
                                                     [v["zstar_median_mm"] for v in rec.values()])[0, 1])
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    CM.receipt(OUT / "summary", {"probe": "P0 post-hoc visibility summary"}, scripts=(__file__,))
    print(json.dumps({k: res[k] for k in ("families", "finger_by_level", "reference_area_median", "mpjpe_mm_mean",
                                          "pearson_pred_vs_zstar")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    run(a.procs) if a.cmd == "run" else summarize()


if __name__ == "__main__":
    main()
