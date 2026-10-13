#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 side check (not part of the pre-registered metrics): is the GT pose of the event's own millisecond
the best-aligned one?  For a random subset of steps, every event is labelled with the GT frame of ms (t + delta),
delta in -4..+4, and the non-hand fraction / interior fraction are reported per delta."""
import json
import multiprocessing as mp
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_source_labels as L                                    # noqa: E402
import step1_common as C                                           # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402

DELTAS = list(range(-4, 5))


def task(step):
    s, end = step
    g = L._G
    start = end - 49
    off = g["offsets"]
    ev5 = C._window_events(g["events"], off, g["tsub"], end, 50)
    counts = np.diff(off[start:end + 2]).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(counts)])
    ex, ey = ev5[:, 1].astype(np.int64), ev5[:, 2].astype(np.int64)
    out = np.zeros((len(DELTAS), 5))                 # n, nonhand, interior, silhouette, depth-disc
    idx = np.arange(start - 4, end + 5)
    idx = np.clip(idx, 0, len(g["pos51"]) - 1)
    verts, _ = C.fk_batch(g["mano"], g["betas"], g["pos51"][idx])
    verts = verts.numpy().astype(np.float64)
    for i in range(50):
        if not counts[i]:
            continue
        sl = slice(cs[i], cs[i + 1])
        for j, dl in enumerate(DELTAS):
            Z, P, _ = L.rasterize(verts[i + 4 + dl], g["faces"], g["face_part"], g["fxyc"])
            lp, lk, ld, _ = L.label_maps(Z, P)
            k = lk[ey[sl], ex[sl]]
            out[j] += [counts[i], (k == -1).sum(), (k == 0).sum(), (k == 1).sum(), (k == 2).sum()]
    return out


if __name__ == "__main__":
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    res = {}
    for seq, legacy in C.sequences():
        ev, off, tsub, aux, pos51, betas, K = C.load_seq_bundle(seq, legacy)
        rid, ends, prev = C.enumerate_steps(aux)
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(ends), min(n_steps, len(ends)), replace=False))
        tasks = [(int(s), int(ends[s])) for s in pick]
        with mp.get_context("fork").Pool(min(64, len(tasks)), initializer=L._init_worker, initargs=(seq, legacy)) as pool:
            tot = sum(pool.map(task, tasks))
        res[seq] = {str(dl): {"n_events": int(tot[j, 0]), "nonhand": float(tot[j, 1] / tot[j, 0]), "interior": float(tot[j, 2] / tot[j, 0]),
                              "silhouette": float(tot[j, 3] / tot[j, 0]), "depth_disc": float(tot[j, 4] / tot[j, 0])} for j, dl in enumerate(DELTAS)}
        print(seq, "steps", len(tasks))
        for dl in DELTAS:
            r = res[seq][str(dl)]
            print(f"  delta={dl:+d} ms  n={r['n_events']:8d}  nonhand={r['nonhand']:.4f}  interior={r['interior']:.4f}  sil={r['silhouette']:.4f}  disc={r['depth_disc']:.4f}")
    (C.OUT_DIR / "check_timing.json").write_text(json.dumps(res, indent=1))
