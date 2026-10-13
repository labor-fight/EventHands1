#!/usr/bin/env python3
"""S43 (docs/EXPERIMENT_RECORDS.md#s43-prereg, G4): the kinematic-subtree evidence gate on top of the S39 stack.

    eval    one run's checkpoint under the S39 stack (W200 clip-run-start, ramp1000 spec, Count50: exactly
            tools/trk6/screen_eval.sh) with `semkine.kinematic_gate.KinematicGateFilter` for each n0 of --n0
            ("inf" = gate off = the S39 stack itself, which must reproduce the S39-stack json of the run bit for bit).
    select  the pre-registered two-fold cross-fit over runs: steps fold by the parity of their 10 s block
            (floor(end_ms / 10000) % 2, per sequence, as tools/dt/filter_sweep_2fold.py); per run and fold the n0 with the
            smallest fold RA is chosen and reported on the other fold against the gate-off RA there (6 selections for 3
            seeds). G4 passes iff all selections are negative out of fold and their mean is <= -0.05 mm.

    python tools/s43/gate_eval.py eval --run-dir outputs/semkine/dt_dz_l3_s3407 --out-dir outputs/s43/gate/dt_dz_l3_s3407
    python tools/s43/gate_eval.py select --in-dirs outputs/s43/gate/dt_dz_l3_s340{7,8,9} --out outputs/s43/reports/g4_s39.json
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

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]
from mano_layer import ManoLayer                                 # noqa: E402
from semkine.dataset import sequences_for_split                  # noqa: E402
from semkine.kinematic_gate import KinematicGateFilter           # noqa: E402
import evalx as EX                                               # noqa: E402
import filter_eval as FE                                         # noqa: E402

SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
N0_GRID = "inf,4,8,16,32,64"


def n0_tag(n0: float) -> str:
    return "off" if math.isinf(n0) else f"n0_{n0:g}"


def cmd_eval(a):
    if getattr(a, "strict_fp32", False):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    run = Path(a.run_dir)
    cfg, model, ckpt, step, _ = FE.build_run(run, a.ckpt, device)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    trained = json.loads((run / "training_metadata.json").read_text()).get("train_sequences", []) \
        if (run / "training_metadata.json").exists() else []
    assert not ({s.split("_")[0] for s, _ in seqs} & {s.split("_")[0] for s in trained}), "subject leak"
    ns = FE.eval_args(window_ms=200, clip_run_start=True, count_mode="last50")
    ns.init_seed = int(a.init_seed)
    spec = json.loads(SPEC.read_text())
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = [(float(f), float(r), float(ef), float(er)) for f in a.n0.split(",") for r in a.n0_root.split(",")
              for ef in a.eps_f.split(",") for er in a.eps_r.split(",")]
    for n0, n0r, ef, er in combos:
        tag = n0_tag(n0) + ("" if math.isinf(n0r) else f"_root{n0r:g}") + (f"_r{a.radius:g}" if a.radius != 8.0 else "")
        if tag.startswith("off_root"):
            tag = "fingeroff" + tag[3:]
        if ef > 0 or er > 0:
            tag += "_df%g_dr%g" % (ef, er)
        if a.init_seed:
            tag += "_init%d" % a.init_seed
        if (out_dir / f"{tag}.json").exists() and not a.force:
            print("exists", out_dir / f"{tag}.json")
            continue
        net = KinematicGateFilter(model, spec, n0, radius_px=a.radius, n0_root=n0r, eps_f=math.radians(ef),
                                  eps_r=math.radians(er)).to(device).eval()
        t0 = time.time()
        summary, arrays = EX.evaluate(net, cfg, mano, root, seqs, device, ns, label=f"{run.name} {tag}")
        out = {"run": run.name, "ckpt": str(ckpt), "step": step, "n0": n0, "n0_root": n0r, "radius_px": a.radius,
               "spec": net.spec(),
               "window": EX.window_record(ns), "wall_s": time.time() - t0}
        out.update(summary)
        keep = {k: v for k, v in arrays.items() if k.split("|")[-1] in
                ("mpjpe_ra_mm", "mpvpe_ra_mm", "root_rot_deg", "transl_mm", "end", "pred", "gt", "run", "count")}
        np.savez_compressed(out_dir / f"{tag}.npz", **keep)
        (out_dir / f"{tag}.json").write_text(json.dumps(out, indent=1))
        print(f"wrote {out_dir / tag}.json RA={summary['model']['overall']['mpjpe_ra_mm']:.4f}", flush=True)


def _fold_sums(z, key="mpjpe_ra_mm"):
    """{fold: (sum, n)} pooled over the sequences in npz `z`."""
    acc = {0: [0.0, 0], 1: [0.0, 0]}
    seqs = sorted({k.split("|")[1] for k in z.files if k.startswith("model|")})
    for s in seqs:
        v, e = z[f"model|{s}|{key}"], z[f"model|{s}|end"]
        f = (e // 10000) % 2
        for k in (0, 1):
            acc[k][0] += float(v[f == k].sum())
            acc[k][1] += int((f == k).sum())
    return {k: (s_, n) for k, (s_, n) in acc.items()}


def written_stats(z) -> dict:
    """Per sequence: finger joints written per step (of 15) and the fraction of steps whose root rotation was written,
    from consecutive outputs of the same segment (a held row is an exact copy of the previous output)."""
    out = {}
    for s in sorted({k.split("|")[1] for k in z.files if k.startswith("model|")}):
        p, run = z[f"model|{s}|pred"], z[f"model|{s}|run"]
        same = run[1:] == run[:-1]
        d = p[1:][same] - p[:-1][same]
        fing = (np.abs(d[:, 6:].reshape(-1, 15, 3)).max(axis=-1) > 0).sum(axis=1)
        out[s] = {"fingers_written_per_step": float(fing.mean()),
                  "root_written_frac": float((np.abs(d[:, 3:6]).max(1) > 0).mean())}
    return out


def cmd_select(a):
    rows, sel = [], []
    for d in [Path(p) for p in a.in_dirs]:
        tags = sorted(p.stem for p in d.glob("*.npz") if re.fullmatch(a.tags, p.stem))
        if "off" not in tags:
            raise SystemExit(f"{d}: the gate-off evaluation is missing")
        fold = {t: _fold_sums(np.load(d / f"{t}.npz")) for t in tags}
        ra = {t: {k: s / n for k, (s, n) in f.items()} for t, f in fold.items()}
        overall = {t: json.loads((d / f"{t}.json").read_text())["model"]["overall"]["mpjpe_ra_mm"] for t in tags}
        for k in (0, 1):
            pick = min(tags, key=lambda t: (ra[t][k], t))
            o = 1 - k
            sel.append({"run": d.name, "select_on": k, "pick": pick, "in_fold": ra[pick][k] - ra["off"][k],
                        "out_of_fold": ra[pick][o] - ra["off"][o]})
        rows.append({"run": d.name, "overall": overall, "fold_ra": ra,
                     "written": {t: written_stats(np.load(d / f"{t}.npz")) for t in tags}})
    oof = [s["out_of_fold"] for s in sel]
    res = {"selections": sel, "runs": rows, "oof_mean": float(np.mean(oof)), "oof_all_negative": all(v < 0 for v in oof),
           "G4_pass": bool(all(v < 0 for v in oof) and float(np.mean(oof)) <= -0.05)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for s in sel:
        print(f"{s['run']} fold{s['select_on']} pick={s['pick']:10s} in={s['in_fold']:+.3f} oof={s['out_of_fold']:+.3f}")
    print(f"oof mean {res['oof_mean']:+.4f}  all negative {res['oof_all_negative']}  G4 {'PASS' if res['G4_pass'] else 'fail'}")
    for r in rows:
        print(r["run"], {t: round(v, 4) for t, v in r["overall"].items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--ckpt", default="last")
    e.add_argument("--n0", default=N0_GRID)
    e.add_argument("--n0-root", default="inf", help="palm-evidence gate thresholds of root / translation (deviation 1)")
    e.add_argument("--radius", type=float, default=8.0)
    e.add_argument("--init-seed", type=int, default=0, help="seed of the segments' noisy initial states (0 = protocol)")
    e.add_argument("--eps-f", default="0", help="finger update dead bands, degrees (deviation 7)")
    e.add_argument("--eps-r", default="0", help="root-rotation update dead bands, degrees (deviation 7)")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--device", default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--strict-fp32", action="store_true", help="cuDNN without TF32 (reference for the G6 comparison)")
    s = sub.add_parser("select")
    s.add_argument("--in-dirs", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tags", default=".*", help="regex the candidate tags must match (the gate-off tag 'off' always must)")
    a = ap.parse_args()
    (cmd_eval if a.cmd == "eval" else cmd_select)(a)


if __name__ == "__main__":
    main()
