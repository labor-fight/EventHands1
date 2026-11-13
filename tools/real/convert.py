#!/usr/bin/env python3
"""Real DAVIS346 hand data -> the hand_data51 format (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110).

Amendment 1 (#real-amend1-20261110): every method gets the same undistorted stream at the DAVIS346's native 346 x 260
and applies its own official preprocessing, so the virtual camera below is written on three canvases, all scalings of
the one 640 x 480-grid camera (the aux K stays on that grid; the canvas's render scale maps it):
  n346   346 x 260, scale 346 / 640 (f 259.5 px), uint16 coordinates -- the native stream (EventHands, EvHandPose)
  n256   256 x 192, scale 0.4 (f 192 px) -- EventHPE's official 256-pixel frame (`--img_size 256`), aspect kept
  c240   240 x 180, scale 0.375 -- the superseded first conversion (hand_data_real51)
Source: /data1/lyq/data/hand_data_real/data_collection (README there). Output: <out>/{train,val}/<seq>{_events.npy,
_offsets.npy, _tsub.npy, .meta, _aux.npz, .done.json}, <out>/splits_real.json, <out>/conversion.json -- the files
semkine/dataset.py and the PFO engine read for hand_data51 (no .evc: only the legacy model/fastevc.py reads it).

  camera   every session shares one DAVIS calibration (K + 5 distortion coefficients). Events of the recordings and of
           the _v2 / _v3 copies are undistorted through one per-pixel table into a virtual pinhole camera on the
           640 x 480 grid of hand_data51 (render scale 0.375 -> 240 x 180, unchanged): pixel centres whose distortion is
           not invertible (Newton on the branch before the model's fold, re-distortion error > 0.05 px) are dropped;
           the principal point centres the invertible extent and the isotropic focal is F_VIRTUAL = 480 px, the largest
           whole-pixel focal at which no labeled frame whose 21 joints all lie on the DAVIS sensor loses a joint from
           the frame (0 of 185,312 at 480 and 481, 1 at 482; keeps 97.7 % of the sensor's pixels; lossless would be
           414 px, half the DAVIS centre resolution). _v4 copies are already pinhole (K_v4 = jittered DAVIS pinhole): their pixels go through the similarity
           p' = a (p - c_davis) + c_virtual, a = f_virtual / f_davis, which maps the nominal DAVIS pinhole onto the
           virtual camera and keeps each copy's rotation / focal / principal-point jitter; K' = the same map of K_v4.
  time     labels (DAVIS clock, ns) of normal-light sessions and their copies are shifted by +6 ms (the README's measured
           event / label offset; dark sessions already carry +18 ms); t0 = the first label, events before t0 or after
           the last label's millisecond are dropped, as hand_data51's builder does.
  labels   tools/prepare_hand_data.py's own interpolation (accepted, non-interpolated labels; PCHIP translation,
           rotation splines, within continuous runs) to the 1 kHz 51-D meta; the 12-D EventHands labels and constant
           betas are checked as there.
  names    <subject>_<motion>[_dark][_v2|_v3|_v4]; aux category = global / local (as hand_data51).

    python tools/real/convert.py run --canvas n346|n256|c240 [--workers N]   # all sequences, skip finished ones
    python tools/real/convert.py camera --canvas ...                          # print the virtual camera only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

SRC = Path("/data1/lyq/data/hand_data_real/data_collection")
DATA = Path("/data1/lyq/code/mesh/EventHands/data")
CANVASES = {"c240": (240, 180, "hand_data_real51"), "n346": (346, 260, "hand_data_real_n346"),
            "n256": (256, 192, "hand_data_real_n256")}
OUT = DATA / "hand_data_real51"
MANO = ROOT / "assets" / "mano_right.npz"
W_DAVIS, H_DAVIS = 346, 260
GRID_W, GRID_H = 640, 480                      # hand_data51's K grid
OUT_W, OUT_H = 240, 180                        # stored event grid of the canvas (set_canvas)
SCALE = OUT_W / GRID_W
INVALID = 65535
SHIFT_NORMAL_NS = 6_000_000                    # README: events align at depth time + 6 ms (normal light)
REDISTORT_TOL_PX = 0.05
MARGIN = 0.002
F_VIRTUAL = 480.0

SUBJECTS = ("ch", "dzy", "kqj", "lbh", "lfz", "lqh", "lr", "lyh", "lyq", "sbl", "sm", "wh", "wjh", "wzh", "zy")
TEST = ("kqj", "lqh")                          # label-quality rule, docs/EXPERIMENT_RECORDS.md#real-prereg-20261110
DEV = ("wh",)
TRAIN = tuple(s for s in SUBJECTS if s not in TEST + DEV)
MOTIONS = ("global", "local")
LIGHTS = ("normal", "dark")
COPIES = ("", "v2", "v3", "v4")


def set_canvas(name):
    global OUT, OUT_W, OUT_H, SCALE
    OUT_W, OUT_H, d = CANVASES[name]
    OUT = DATA / d
    SCALE = OUT_W / GRID_W
    assert int(np.floor(GRID_H * SCALE - 1e-9)) < OUT_H + 1


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def seq_name(subject, motion, light, copy):
    return f"{subject}_{motion}" + ("_dark" if light == "dark" else "") + (f"_{copy}" if copy else "")


def source(subject, motion, light, copy):
    """(source directory, event subdirectory, label category) of one sequence."""
    d = SRC / (f"{subject}_{motion}" + (f"_{copy}" if copy else ""))
    cat = f"{subject}_{motion}" + (f"_{copy}" if copy else "") + ("_dark" if light == "dark" else "")
    sub = "dark" if light == "dark" else ""
    return d, ("event_dark" if sub else "event"), ("mesh_dark" if sub else "mesh"), cat


def plan():
    rows = []
    for subj in SUBJECTS:
        role = "test" if subj in TEST else ("dev" if subj in DEV else "train")
        for motion in MOTIONS:
            for light in LIGHTS:
                for copy in (COPIES if role == "train" else ("",)):
                    rows.append({"seq": seq_name(subj, motion, light, copy), "subject": subj, "motion": motion,
                                 "light": light, "copy": copy, "role": role,
                                 "legacy_dir": "train" if role == "train" else "val"})
    return rows


# ---------------------------------------------------------------------------------------------- camera
def davis_camera():
    m = json.loads((SRC / "ch_global" / "mesh" / "ch_global_manifest.json").read_text())["camera"]
    return np.asarray(m["K"], np.float64), np.asarray(m["distortion"], np.float64)


def check_single_calibration(K, dist):
    man = json.loads((SRC / "dataset_manifest.json").read_text())
    for s in man["sessions"]:
        d = s["participant_directory"]
        cat = d if s["environment"] == "normal" else d + "_dark"
        sub = "mesh" if s["environment"] == "normal" else "mesh_dark"
        c = json.loads((SRC / d / sub / f"{cat}_manifest.json").read_text())["camera"]
        assert np.allclose(c["K"], K) and np.allclose(c["distortion"], dist), f"{cat}: another calibration"


def distort(x, y, dist):
    """OpenCV's 5-coefficient model on normalized coordinates, with its Jacobian."""
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    dr = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    j11 = rad + 2 * x * x * dr + 2 * p1 * y + 6 * p2 * x
    j12 = 2 * x * y * dr + 2 * p1 * x + 2 * p2 * y
    j21 = 2 * x * y * dr + 2 * p1 * x + 2 * p2 * y
    j22 = rad + 2 * y * y * dr + 6 * p1 * y + 2 * p2 * x
    return xd, yd, j11, j12, j21, j22


def undistort_table(K, dist):
    """Newton from the distorted point (the physical branch before the model's fold); valid = re-distortion error
    <= REDISTORT_TOL_PX and a positive Jacobian determinant (not past the fold)."""
    xs, ys = np.meshgrid(np.arange(W_DAVIS, dtype=np.float64), np.arange(H_DAVIS, dtype=np.float64))
    u, v = xs.ravel(), ys.ravel()
    xd0 = (u - K[0, 2]) / K[0, 0]
    yd0 = (v - K[1, 2]) / K[1, 1]
    x, y = xd0.copy(), yd0.copy()
    for _ in range(100):
        xd, yd, a, b, c, d = distort(x, y, dist)
        rx, ry = xd - xd0, yd - yd0
        det = a * d - b * c
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        dx = (d * rx - b * ry) / det
        dy = (-c * rx + a * ry) / det
        step = np.hypot(dx, dy)
        lim = np.minimum(1.0, 0.05 / np.maximum(step, 1e-15))   # damped: at most 0.05 per iteration
        x -= dx * lim
        y -= dy * lim
    xd, yd, a, b, c, d = distort(x, y, dist)
    err = np.hypot((xd - xd0) * K[0, 0], (yd - yd0) * K[1, 1])
    ok = np.isfinite(err) & (err <= REDISTORT_TOL_PX) & (a * d - b * c > 0)
    return np.stack([x, y], 1), ok, err


def virtual_camera():
    K, dist = davis_camera()
    norm, ok, err = undistort_table(K, dist)
    lo, hi = norm[ok].min(0), norm[ok].max(0)
    f = F_VIRTUAL
    c = np.array([GRID_W / 2 - f * (hi[0] + lo[0]) / 2, GRID_H / 2 - f * (hi[1] + lo[1]) / 2])
    Kv = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    u = norm[:, 0] * f + c[0]
    v = norm[:, 1] * f + c[1]
    x = np.floor(u * SCALE)
    y = np.floor(v * SCALE)
    keep = ok & (x >= 0) & (x < OUT_W) & (y >= 0) & (y < OUT_H)
    lut = np.full((H_DAVIS * W_DAVIS, 2), INVALID, np.uint16)
    lut[keep, 0] = x[keep]
    lut[keep, 1] = y[keep]
    f_davis = float((K[0, 0] + K[1, 1]) / 2)
    info = {"davis_K": K.tolist(), "davis_distortion": dist.tolist(), "virtual_K_640x480": Kv.tolist(),
            "f_davis_mean": f_davis, "similarity_a": f / f_davis, "invertible_pixels": int(ok.sum()),
            "dropped_pixels": int((~keep).sum()), "dropped_fraction": float((~keep).mean()),
            "redistort_tol_px": REDISTORT_TOL_PX, "max_redistort_err_kept_px": float(err[ok].max()),
            "normalized_extent": [lo.tolist(), hi.tolist()], "canvas": [OUT_W, OUT_H], "render_scale": SCALE,
            "focal_on_canvas_px": f * SCALE,
            "focal_rule": "largest whole-pixel focal with 0 labeled on-sensor frames losing a joint (480: 0, 482: 1)"}
    return K, Kv, lut, keep, f_davis, info


def v4_table(K, Kv, f_davis):
    a = Kv[0, 0] / f_davis
    xs, ys = np.meshgrid(np.arange(W_DAVIS, dtype=np.float64), np.arange(H_DAVIS, dtype=np.float64))
    u = a * (xs.ravel() - K[0, 2]) + Kv[0, 2]
    v = a * (ys.ravel() - K[1, 2]) + Kv[1, 2]
    x, y = np.floor(u * SCALE), np.floor(v * SCALE)
    keep = (x >= 0) & (x < OUT_W) & (y >= 0) & (y < OUT_H)
    lut = np.full((H_DAVIS * W_DAVIS, 2), INVALID, np.uint16)
    lut[keep, 0] = x[keep]
    lut[keep, 1] = y[keep]
    return lut, keep, a


def v4_K(Kc, K, Kv, a):
    out = np.array(Kc, np.float64).copy()
    out[0, 0] *= a
    out[1, 1] *= a
    out[0, 2] = a * (Kc[0, 2] - K[0, 2]) + Kv[0, 2]
    out[1, 2] = a * (Kc[1, 2] - K[1, 2]) + Kv[1, 2]
    return out


# ---------------------------------------------------------------------------------------------- one sequence
def read_events(d, sub):
    recs = [json.loads(l) for l in (d / sub / "index.jsonl").read_text().splitlines() if l.strip()]
    t, x, y, p = [], [], [], []
    for r in recs:
        e = np.load(d / r["file"])
        assert len(e["t_us"]) == r["event_count"], f"{d / r['file']}: count"
        t.append(e["t_us"].astype(np.int64)); x.append(e["x"].astype(np.int32)); y.append(e["y"].astype(np.int32))
        p.append(e["p"].astype(np.uint8))
    t, x, y, p = (np.concatenate(a) for a in (t, x, y, p))
    if np.any(np.diff(t) < 0):
        o = np.argsort(t, kind="stable")
        t, x, y, p = t[o], x[o], y[o], p[o]
    assert x.min() >= 0 and x.max() < W_DAVIS and y.min() >= 0 and y.max() < H_DAVIS and p.max() <= 1
    return t, x, y, p, len(recs)


def convert_one(row, cam, skip=True):
    from tools.prepare_hand_data import continuous_valid_runs, interpolate_pose_ms, project_pca6, write_meta
    K, Kv, lut, lut4, a = cam["K"], cam["Kv"], cam["lut"], cam["lut4"], cam["a"]
    seq = row["seq"]
    out_dir = OUT / row["legacy_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / f"{seq}.done.json"
    if skip and done.exists():
        return json.loads(done.read_text())
    t_start = time.time()
    d, esub, msub, cat = source(row["subject"], row["motion"], row["light"], row["copy"])
    mesh = np.load(d / msub / f"{cat}_mesh.npz")
    eh12 = np.load(d / msub / f"{cat}_eventhands12.npz")
    shift = SHIFT_NORMAL_NS if row["light"] == "normal" else 0
    ts_ns = mesh["timestamp_ns"].astype(np.int64) + shift
    valid_ann = mesh["valid"].astype(bool) & ~mesh["interpolated"].astype(bool)
    assert valid_ann.any()
    first = int(np.flatnonzero(valid_ann)[0])
    t0_ns = int(ts_ns[first])
    t_ann_s = (ts_ns - t0_ns) / 1e9
    last = int(np.flatnonzero(valid_ann)[-1])
    n_ms = int(np.ceil(t_ann_s[last] * 1000)) + 1
    transl = mesh["camera_transl_right_canonical_m"].astype(np.float64)
    go = mesh["global_orient_rotvec"].astype(np.float64)
    hp = mesh["hand_pose_residual_aa"].astype(np.float64)
    betas = mesh["betas"].astype(np.float32)
    Kc = mesh["camera_K"].astype(np.float64)
    Kc = Kc[first] if Kc.ndim == 3 else Kc
    comps = np.load(MANO)["hands_components"].astype(np.float64)
    re12 = np.concatenate([project_pca6(hp, comps), transl, go], 1).astype(np.float32)
    err12 = float(np.abs(re12[valid_ann] - eh12["params_12"].astype(np.float32)[valid_ann]).max())
    assert err12 < 1e-4, f"{seq}: params_12 mismatch {err12}"
    beta_range = float((betas[valid_ann].max(0) - betas[valid_ann].min(0)).max())
    assert beta_range < 1e-4, f"{seq}: betas vary {beta_range}"
    beta0 = betas[first]
    keep_ann = valid_ann & (t_ann_s >= 0)
    params51, valid_ms = interpolate_pose_ms(t_ann_s, transl, go, hp, keep_ann, n_ms)
    gt_ms_idx = np.clip(np.rint(t_ann_s * 1000).astype(np.int64), 0, n_ms - 1)

    t_us, x, y, p, n_chunks = read_events(d, esub)
    n_in = len(t_us)
    rel = t_us * 1000 - t0_ns                                 # ns relative to t0 (exact integers)
    ms = np.floor_divide(rel, 1_000_000)
    inside = (ms >= 0) & (ms < n_ms)
    table = lut4 if row["copy"] == "v4" else lut
    xy = table[y.astype(np.int64) * W_DAVIS + x.astype(np.int64)]
    mapped = xy[:, 0] != INVALID
    keep = inside & mapped
    ms, xy, p, rel = ms[keep], xy[keep], p[keep], rel[keep]
    tsub = ((rel - ms * 1_000_000) // 1000).astype(np.uint16)
    order = np.argsort(ms, kind="stable")                     # already time-ordered; stable keeps the order
    ms, xy, p, tsub = ms[order], xy[order], p[order], tsub[order]
    counts = np.bincount(ms, minlength=n_ms).astype(np.int64)
    offsets = np.zeros(n_ms + 1, np.int64)
    np.cumsum(counts, out=offsets[1:])
    ev = np.empty((len(ms), 3), np.uint8 if OUT_W <= 256 else np.uint16)
    ev[:, 0], ev[:, 1], ev[:, 2] = xy[:, 0], xy[:, 1], p
    np.save(out_dir / f"{seq}_events.npy", ev)
    np.save(out_dir / f"{seq}_offsets.npy", offsets)
    np.save(out_dir / f"{seq}_tsub.npy", tsub)
    write_meta(out_dir / f"{seq}.meta", params51)
    runs = continuous_valid_runs(valid_ms)
    K_out = v4_K(Kc, K, Kv, a) if row["copy"] == "v4" else Kv
    aux = {"betas": beta0, "camera_K": K_out.astype(np.float32), "valid_ms": valid_ms, "gt_ms_idx": gt_ms_idx,
           "gt_valid": keep_ann, "gt_timestamp_ns": ts_ns, "t0_ns": np.int64(t0_ns), "n_ms": np.int64(n_ms),
           "valid_runs_ms": np.asarray(runs, np.int64).reshape(-1, 2),
           "category": np.array(row["motion"]), "split": np.array(row["legacy_dir"]),
           "canonical_hand": np.array("right"),
           "params_layout": np.array(["t[0:3]", "R_root[3:6]", "hand_pose_residual_aa[6:51]"]),
           "light": np.array(row["light"]), "subject": np.array(row["subject"]), "copy": np.array(row["copy"] or "rec"),
           "source_category": np.array(cat), "label_shift_ns": np.int64(shift)}
    np.savez_compressed(out_dir / f"{seq}_aux.npz", **aux)
    summary = {**row, "source_dir": str(d), "source_category": cat, "label_shift_ns": shift, "t0_ns": t0_ns,
               "n_ms": n_ms, "n_ann": int(len(ts_ns)), "n_valid_ann": int(keep_ann.sum()),
               "n_valid_ms": int(valid_ms.sum()), "n_runs": len(runs), "n_chunks": n_chunks, "events_in": int(n_in),
               "events_in_window": int(inside.sum()), "events_out": int(len(ms)),
               "events_dropped_unmapped": int((inside & ~mapped).sum()), "params12_max_abs_err": err12,
               "camera_K": K_out.tolist(), "betas": beta0.tolist(),
               "label_bundle_sha256": sha(d / msub / f"{cat}_mesh.npz"),
               "seconds": round(time.time() - t_start, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    done.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[{row['legacy_dir']}/{seq}] ms={n_ms} events {n_in} -> {len(ms)} valid_ms={int(valid_ms.sum())} "
          f"runs={len(runs)} {summary['seconds']}s", flush=True)
    return summary


_CAM = None


def _init(cam):
    global _CAM
    _CAM = cam


def _work(row):
    return convert_one(row, _CAM)


def camera_bundle():
    K, Kv, lut, keep, f_davis, info = virtual_camera()
    lut4, keep4, a = v4_table(K, Kv, f_davis)
    info["v4_pixels_kept_fraction"] = float(keep4.mean())
    return {"K": K, "Kv": Kv, "lut": lut, "lut4": lut4, "a": a}, info


def splits(rows):
    def block(sel):
        return {"trials": [r["seq"] for r in sel], "subjects": sorted({r["subject"] for r in sel}),
                "legacy_dir": {r["seq"]: r["legacy_dir"] for r in sel}}
    tr = [r for r in rows if r["role"] == "train"]
    dv = [r for r in rows if r["role"] == "dev"]
    te = [r for r in rows if r["role"] == "test"]
    return {"_protocol": "docs/EXPERIMENT_RECORDS.md#real-prereg-20261110", "train": block(tr), "val_core": block(dv),
            "val": block(dv), "test": block(te)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=("run", "camera"))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--canvas", required=True, choices=sorted(CANVASES))
    args = ap.parse_args()
    set_canvas(args.canvas)
    K, dist = davis_camera()
    check_single_calibration(K, dist)
    cam, info = camera_bundle()
    print(json.dumps(info, indent=1), flush=True)
    if args.verb == "camera":
        return
    rows = plan()
    if args.only:
        rows = [r for r in rows if r["seq"] in args.only]
    OUT.mkdir(parents=True, exist_ok=True)
    results, errors = [], []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(cam,)) as ex:
        futs = {ex.submit(_work, r): r for r in rows}
        for f in as_completed(futs):
            try:
                results.append(f.result())
            except Exception as e:                            # noqa: BLE001
                errors.append({"seq": futs[f]["seq"], "error": repr(e)})
                print("ERROR", futs[f]["seq"], repr(e), flush=True)
    if args.only:
        return
    full = plan()
    sp = splits(full)
    (OUT / "splits_real.json").write_text(json.dumps(sp, indent=1) + "\n")
    (OUT / "conversion.json").write_text(json.dumps({
        "tool": "tools/real/convert.py", "tool_sha256": sha(Path(__file__)), "source": str(SRC), "canvas": args.canvas,
        "source_manifest_sha256": sha(SRC / "dataset_manifest.json"), "camera": info,
        "label_shift_normal_ns": SHIFT_NORMAL_NS, "subjects": {"train": TRAIN, "dev": DEV, "test": TEST},
        "n_sequences": len(full), "errors": errors,
        "sequences": sorted(results, key=lambda r: r["seq"]),
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
    print("DONE", len(results), "sequences,", len(errors), "errors", flush=True)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
