#!/usr/bin/env python3
"""MV2 M10 / SDG (P6-E1-R4): sensor-domain-gap manifest and statistics gate. INPUT statistics only.

EvRealHands (`/data1/lyq/code/mesh/EventHands/data/evrealhands12/`, test-only by the user's 2026-10-25 decision; DAVIS346 346x260
already binned to the 240 x 180 grid and to 1 ms frames by the dataset converter) is compared with the 9 training subjects'
v2e event streams (`hand_data51` base recordings, 240 x 180, 1 ms) on per-50 ms-packet event statistics on the evaluator's grid.
Never read: the label arrays (`*_labels_15fps.npz`), the evaluation JSON / CSV files of that directory, any model output. The
trial table `summary.json` (subject / hand / scene / duration / annotation window) is read as metadata.

Per packet (the 50 ms window [end - 49, end], ends = a + 49 + 50 k inside each valid run; ERH: inside the annotated span):
  log_rate      log10(1 + n)                     n = events in the packet (all packets)
  zero_ms_frac  fraction of the 50 ms with no event          (active packets: n >= 50, the first ramp node of the S39 gate)
  fano          variance / mean of the 50 per-ms counts      (active packets)  -- ms-level timestamp clustering
  on_frac       fraction of polarity-1 events                (active packets)
  log_slots     log10(occupied (pixel, polarity) slots)      (active packets)  -- spatial footprint
  mult          events per occupied slot                     (active packets)  -- the LNES overwrite factor
  bbox_px       sqrt(width x height) of the central 98 % of the events  (active packets)  -- hand footprint in pixels
Distance between two subjects' packet distributions: the 1-D Wasserstein-1 distance in the statistic's own unit.
U_q = the maximum over the 36 pairs of training subjects; D_q = the maximum over training subjects of the distance to the pooled
right-hand ERH distribution (a domain 'inside the diameter' of the training cloud). Core statistics (rate, clustering) over the
bound => `real_sensor_claims_blocked`.

    python tools/mv2/m10_sdg.py --smoke
    python tools/mv2/m10_sdg.py --out outputs/mv2/m10_sdg [--workers 16]
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wasserstein_distance

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402

ERH_DIR = Path("/data1/lyq/code/mesh/EventHands/data/evrealhands12")
TRAIN_SUBJECTS = C.FIT7 + C.SELECT                      # the 9 training subjects
STEP = 50
ACTIVE_MIN = 50                                          # events: the first node of the ramp1000 gate (Count50)
STATS = ("log_rate", "zero_ms_frac", "fano", "on_frac", "log_slots", "mult", "bbox_px")
ACTIVE_STATS = STATS[1:]
CORE = ("log_rate", "zero_ms_frac", "fano")
CLAIM_OF = {"log_rate": "any zero-shot real-sensor claim (event rate)", "zero_ms_frac": "any zero-shot real-sensor claim (ms clustering)",
            "fano": "any zero-shot real-sensor claim (ms clustering)", "on_frac": "signed / polarity channels (M3, M4, M8)",
            "log_slots": "LNES occupancy-dependent claims (footprint)", "mult": "LNES overwrite-factor-dependent claims",
            "bbox_px": "scale / depth-dependent claims (hand footprint)"}
GRID_W, GRID_H = 240, 180
SPLIT_SEED = 20261026
POSCTRL_FACTOR = 10.0                                    # synthetic x10 event-rate shift: the gate must flag it

#: the K / pixel-binning / clock checklist (346x260 DAVIS346 vs the 640x480 -> 240x180 v2e training stream)
CHECKLIST = [
    {"item": "sensor array", "training": "v2e on 640x480 RGB-D video frames (AEDAT4 header: DVXplorer sample, 640x480)",
     "erh": "DAVIS346, 346x260", "status": "known", "consequence": "different source pixel pitch / field of view"},
    {"item": "binning to the 240x180 grid", "training": "x_out = floor(3 x_src / 8), y_out = floor(3 y_src / 8): 2 or 3 source px per output px "
     "(2.667 on average); 0.4375 px mean offset of the integer-pixel ray convention (#mv2-stage0-20261025 A1b)",
     "erh": "346 -> 240 and 260 -> 180: 1.442 and 1.444 source px per output px (1 or 2 merged); the converter script that wrote "
     "evrealhands12 is not in this repository, so the exact rule (floor / round, phase) is unverified; the data only confirm that "
     "coordinates are already within 240x180", "status": "unverified", "consequence": "the merged-pixel statistics (multiplicity, "
     "refractory-on-merged-pixel) differ by construction; a binning-phase effect on local accuracy cannot be excluded"},
    {"item": "K / render scale", "training": "RealSense K (603.45, 602.96, 325.09, 242.10) x 0.375 (RENDER_SCALE), isotropic",
     "erh": "DAVIS346 K from the dataset calibration (not read); scale 240/346 = 0.6936 (x) and 180/260 = 0.6923 (y): anisotropic by 0.2 %, "
     "while MODEL.RENDER_SCALE is one scalar", "status": "unverified", "consequence": "a run on ERH needs its own K and an "
     "anisotropic-aware render scale; the depth-scale sigma* gap (0.93-1.01) was measured with K x 0.375 only"},
    {"item": "time base", "training": "events binned to 1 ms; sub-ms microseconds kept in <seq>_tsub.npy; v2e common time layers "
     "(98-99.9 % of adjacent events share a timestamp, layer spacing ~0.93 ms)", "erh": "1 ms frames only (.evc frame = 1 ms); no sub-ms times",
     "status": "known", "consequence": "sub-ms clustering is not assessable on ERH; only ms-level clustering (zero-ms fraction, Fano factor) is compared"},
    {"item": "label clock", "training": "59.67 Hz source frames interpolated to 1 ms (PCHIP / RotationSpline)", "erh": "15 Hz annotations "
     "interpolated to 1 ms", "status": "known", "consequence": "label bandwidth 7.5 vs 30 Hz: lag / carry readings are not comparable"},
    {"item": "hand region", "training": "v2e events limited to the hand bounding box x 1.5 with a mask-border suppression (8 px source)",
     "erh": "full-scene events (background, clutter, flicker)", "status": "known", "consequence": "event-rate and footprint statistics are "
     "expected to differ for non-hand reasons; the data-driven-ROI variant is reported as descriptive"},
    {"item": "polarity convention", "training": "p in {0, 1}, 1 = ON", "erh": "p in {0, 1}; convention of the converter not documented",
     "status": "unverified", "consequence": "a flipped convention would show as an ON-fraction gap"},
    {"item": "handedness", "training": "right hand only (MANO right)", "erh": "40 right / 40 left trials", "status": "known",
     "consequence": "the primary comparison uses right-hand trials; left-hand trials need mirroring before any model run"},
]


# ----------------------------------------------------------------------------------------------- readers
def read_evc(path: Path):
    """`.evc` (EventHands format): records <HBB> = x (uint16), y (uint8), p (uint8); a record with p == 255 closes a 1 ms frame
    (the first record opens frame 0). Returns x, y, p of the events and per-ms offsets (n_frames + 1)."""
    rec = np.fromfile(path, dtype=np.dtype([("x", "<u2"), ("y", "u1"), ("p", "u1")]))
    mark = np.flatnonzero(rec["p"] == 255)
    counts = np.diff(mark) - 1
    keep = rec["p"] != 255
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return rec["x"][keep], rec["y"][keep], rec["p"][keep], offsets


def erh_trials() -> Dict[str, dict]:
    return json.loads((ERH_DIR / "summary.json").read_text())["trials"]


def train_stream(seq: str):
    events, offsets, aux, _ = C.load_seq(seq)
    runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
    ends = np.concatenate([np.arange(a + STEP - 1, b, STEP, dtype=np.int64) for a, b in runs if b - a >= STEP] or
                          [np.zeros(0, np.int64)])
    return np.asarray(events[:, 0]), np.asarray(events[:, 1]), np.asarray(events[:, 2]), np.asarray(offsets), ends


# ----------------------------------------------------------------------------------------------- statistics
def data_driven_roi(x, y, offsets, ends, expand: float = 1.5):
    """x / y bounds of the central 99 % of a recording's events (sampled), expanded by `expand` about the centre."""
    sel = np.linspace(0, len(x) - 1, min(len(x), 400000)).astype(np.int64)
    out = []
    for v, hi in ((x, GRID_W), (y, GRID_H)):
        lo_, hi_ = np.percentile(v[sel], [0.5, 99.5])
        c, h = (lo_ + hi_) / 2, (hi_ - lo_) / 2 * expand
        out.append((max(0.0, c - h), min(float(hi), c + h)))
    return out


def packet_stats(x, y, p, offsets, ends, roi=None) -> Dict[str, np.ndarray]:
    rows = {k: [] for k in STATS}
    for end in ends:
        s = int(end) - STEP + 1
        a0, a1 = int(offsets[s]), int(offsets[int(end) + 1])
        cnt = np.diff(offsets[s:int(end) + 2]).astype(np.float64)
        xs, ys, ps = x[a0:a1], y[a0:a1], p[a0:a1]
        if roi is not None and len(xs):
            m = (xs >= roi[0][0]) & (xs <= roi[0][1]) & (ys >= roi[1][0]) & (ys <= roi[1][1])
            if not m.all():
                cnt = np.bincount(np.repeat(np.arange(STEP), np.diff(offsets[s:int(end) + 2]).astype(np.int64))[m], minlength=STEP).astype(np.float64)
            xs, ys, ps = xs[m], ys[m], ps[m]
        n = len(xs)
        rows["log_rate"].append(np.log10(1.0 + n))
        if n < ACTIVE_MIN:
            for k in ACTIVE_STATS:
                rows[k].append(np.nan)
            continue
        key = (ys.astype(np.int64) * GRID_W + xs.astype(np.int64)) * 2 + np.clip(ps, 0, 1).astype(np.int64)
        slots = len(np.unique(key))
        w = np.subtract(*np.percentile(xs, [99, 1]))
        h = np.subtract(*np.percentile(ys, [99, 1]))
        rows["zero_ms_frac"].append(float((cnt == 0).mean()))
        rows["fano"].append(float(cnt.var() / cnt.mean()))
        rows["on_frac"].append(float((ps == 1).mean()))
        rows["log_slots"].append(float(np.log10(slots)))
        rows["mult"].append(float(n / slots))
        rows["bbox_px"].append(float(np.sqrt(max(w, 1.0) * max(h, 1.0))))
    return {k: np.asarray(v, np.float64) for k, v in rows.items()}


def _train_task(seq: str, max_packets: Optional[int], roi_mode: str):
    x, y, p, off, ends = train_stream(seq)
    if max_packets:
        ends = ends[:max_packets]
    roi = data_driven_roi(x, y, off, ends) if roi_mode == "dd" else None
    return {"seq": seq, "subject": C.subject_of(seq), "stats": packet_stats(x, y, p, off, ends, roi), "n_events": int(len(x)),
            "n_packets": int(len(ends))}


def _erh_task(trial: str, meta: dict, max_packets: Optional[int], roi_mode: str):
    x, y, p, off = read_evc(ERH_DIR / f"{trial}.evc")
    n_frames = len(off) - 1
    ok_len = n_frames == int(meta["duration_ms_frames"])
    lo = int(np.ceil(meta["annotation_time_start_s"] * 1000.0))
    hi = int(min(n_frames, np.floor(meta["annotation_time_end_s"] * 1000.0)))
    ends = np.arange(lo + STEP - 1, hi, STEP, dtype=np.int64)
    if max_packets:
        ends = ends[:max_packets]
    roi = data_driven_roi(x, y, off, ends) if roi_mode == "dd" else None
    return {"trial": trial, "subject": int(meta["subject_id"]), "hand": meta["hand_type"], "scene": meta["scene"],
            "frames_match_summary": bool(ok_len), "x_max": int(x.max()) if len(x) else -1, "y_max": int(y.max()) if len(y) else -1,
            "p_values": sorted(int(v) for v in np.unique(p)), "nonempty_ms_frac": float((np.diff(off) > 0).mean()),
            "stats": packet_stats(x, y, p, off, ends, roi), "n_events": int(len(x)), "n_packets": int(len(ends))}


def _pool(rows: List[dict], key=lambda r: True) -> Dict[str, np.ndarray]:
    sel = [r for r in rows if key(r)]
    return {k: np.concatenate([r["stats"][k] for r in sel]) if sel else np.zeros(0) for k in STATS}


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")


def distances(train_rows: List[dict], erh_rows: List[dict]) -> dict:
    subj = {s: _pool(train_rows, lambda r, s=s: r["subject"] == s) for s in TRAIN_SUBJECTS}
    erh_all = _pool(erh_rows, lambda r: r["hand"] == "right")
    out: Dict[str, dict] = {}
    for q in STATS:
        pair = {f"{a}|{b}": _w1(subj[a][q], subj[b][q]) for a, b in itertools.combinations(TRAIN_SUBJECTS, 2)}
        cross = {s: _w1(erh_all[q], subj[s][q]) for s in TRAIN_SUBJECTS}
        per_erh_subject = {}
        for e in sorted({r["subject"] for r in erh_rows if r["hand"] == "right"}):
            pe = _pool(erh_rows, lambda r, e=e: r["subject"] == e and r["hand"] == "right")
            per_erh_subject[str(e)] = max(_w1(pe[q], subj[s][q]) for s in TRAIN_SUBJECTS)
        # split-half control (same subject, alternating blocks of 20 packets): must be small
        halves = []
        for s in TRAIN_SUBJECTS:
            vals = np.concatenate([r["stats"][q] for r in train_rows if r["subject"] == s])
            blk = (np.arange(len(vals)) // 20) % 2
            halves.append(_w1(vals[blk == 0], vals[blk == 1]))
        out[q] = {"U_train_pairwise_max": float(np.nanmax(list(pair.values()))), "U_train_pairwise_median": float(np.nanmedian(list(pair.values()))),
                  "D_erh_max_over_train": float(np.nanmax(list(cross.values()))), "D_erh_min_over_train": float(np.nanmin(list(cross.values()))),
                  "cross": cross, "per_erh_subject_max": per_erh_subject, "split_half_max": float(np.nanmax(halves)),
                  "n_train_packets": int(np.isfinite(np.concatenate([subj[s][q] for s in TRAIN_SUBJECTS])).sum()),
                  "n_erh_packets": int(np.isfinite(erh_all[q]).sum())}
    # positive control: x10 on one training subject's event counts must exceed the training bound for the rate statistic
    ref = subj[TRAIN_SUBJECTS[0]]["log_rate"]
    shifted = np.log10(1.0 + POSCTRL_FACTOR * (10.0 ** ref - 1.0))
    out["_posctrl_rate_distance"] = _w1(ref, shifted)
    return out


def run(workers: int = 1, max_packets: Optional[int] = None, roi_mode: str = "full", seqs: Optional[Sequence[str]] = None,
        trials: Optional[Sequence[str]] = None) -> dict:
    seqs = list(seqs) if seqs is not None else C.base_seqs(TRAIN_SUBJECTS)
    meta = erh_trials()
    trials = list(trials) if trials is not None else sorted(meta, key=int)
    t0 = time.time()
    if workers <= 1:
        tr = [_train_task(s, max_packets, roi_mode) for s in seqs]
        er = [_erh_task(t, meta[t], max_packets, roi_mode) for t in trials]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            ft = [pool.apply_async(_train_task, (s, max_packets, roi_mode)) for s in seqs]
            fe = [pool.apply_async(_erh_task, (t, meta[t], max_packets, roi_mode)) for t in trials]
            tr, er = [f.get() for f in ft], [f.get() for f in fe]
    return {"train": tr, "erh": er, "roi": roi_mode, "secs": time.time() - t0}


def summarize(raw: dict) -> dict:
    erh = raw["erh"]
    validity = {"frames_match_summary_all": all(r["frames_match_summary"] for r in erh),
                "coords_within_grid_all": all(r["x_max"] < GRID_W and r["y_max"] < GRID_H for r in erh),
                "polarity_values_01_all": all(set(r["p_values"]) <= {0, 1} for r in erh),
                "n_train_subjects": len({r["subject"] for r in raw["train"]}), "n_erh_trials": len(erh),
                "n_erh_right_trials": sum(r["hand"] == "right" for r in erh)}
    dist = distances(raw["train"], erh)
    # descriptive: per-scene rate distances of right-hand ERH trials
    return {"validity": validity, "distances": dist, "roi": raw["roi"],
            "erh_trial_table": [{k: r[k] for k in ("trial", "subject", "hand", "scene", "n_events", "n_packets", "nonempty_ms_frac")}
                                for r in erh],
            "train_table": [{k: r[k] for k in ("seq", "subject", "n_events", "n_packets")} for r in raw["train"]]}


# ----------------------------------------------------------------------------------------------- decision
def decide(result: dict) -> dict:
    """Gate of SDG (tools/mv2/prereg_m10.md section 2). Per statistic q: exceed_q = D_q > U_q (D_q = max over training subjects
    of the Wasserstein-1 distance to the pooled right-hand ERH distribution, U_q = max pairwise training distance).
    tool validity (else `tool_defect`): every ERH trial's frame count equals the summary's duration and its coordinates lie in the
    240x180 grid; >= 9 training subjects and >= 20 right-hand ERH trials; the split-half control is below U_q for every statistic;
    the x10 rate shift exceeds U_q(log_rate).
    verdict: `comparable` (nothing exceeds), `partially_comparable` (only non-core exceed; the matching claims in CLAIM_OF are
    blocked), `real_sensor_claims_blocked` (a core statistic - rate or ms clustering - exceeds)."""
    v = result["validity"]
    if not (v["frames_match_summary_all"] and v["coords_within_grid_all"] and v["polarity_values_01_all"]):
        return {"verdict": "tool_defect", "notes": ["ERH file format check failed", {k: v[k] for k in v}]}
    if v["n_train_subjects"] < 9 or v["n_erh_right_trials"] < 20:
        return {"verdict": "tool_defect", "notes": ["too few subjects / right-hand trials"]}
    d = result["distances"]
    for q in STATS:
        if not (d[q]["split_half_max"] <= d[q]["U_train_pairwise_max"]):
            return {"verdict": "tool_defect", "notes": [f"split-half control above the training bound for {q}"]}
    if not (d["_posctrl_rate_distance"] > d["log_rate"]["U_train_pairwise_max"]):
        return {"verdict": "tool_defect", "notes": ["a x10 event-rate shift was not flagged by the rate bound"]}
    exceed = {q: bool(d[q]["D_erh_max_over_train"] > d[q]["U_train_pairwise_max"]) for q in STATS}
    ratio = {q: d[q]["D_erh_max_over_train"] / d[q]["U_train_pairwise_max"] for q in STATS}
    blocked = sorted({CLAIM_OF[q] for q in STATS if exceed[q]})
    if any(exceed[q] for q in CORE):
        verdict = "real_sensor_claims_blocked"
    elif any(exceed.values()):
        verdict = "partially_comparable"
    else:
        verdict = "comparable"
    return {"verdict": verdict, "exceed": exceed, "D_over_U": ratio, "blocked_claims": blocked,
            "not_assessable": ["sub-ms timestamp clustering (no tsub in .evc)", "K / binning rule of the converter (unverified)",
                               "polarity convention (undocumented)"], "roi": result.get("roi")}


# ----------------------------------------------------------------------------------------------- main
def smoke(args) -> int:
    t0 = time.time()
    ok = True
    # --- parser / statistics unit checks on a synthetic .evc (no real data)
    import tempfile
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as td:
        n_ms = 120
        counts = rng.poisson(30, n_ms)
        recs = [(0, 0, 255)]
        xs, ys, ps = [], [], []
        for c in counts:
            for _ in range(int(c)):
                xs.append(int(rng.integers(0, GRID_W))); ys.append(int(rng.integers(0, GRID_H))); ps.append(int(rng.integers(0, 2)))
                recs.append((xs[-1], ys[-1], ps[-1]))
            recs.append((0, 0, 255))
        arr = np.array(recs, dtype=np.dtype([("x", "<u2"), ("y", "u1"), ("p", "u1")]))
        f = Path(td) / "t.evc"
        arr.tofile(f)
        x, y, p, off = read_evc(f)
        parse_ok = (len(off) - 1 == n_ms and np.array_equal(np.diff(off), counts) and np.array_equal(x, np.array(xs))
                    and np.array_equal(p, np.array(ps)))
        print(f"[m10_sdg smoke] .evc parser round-trip on a synthetic file ({n_ms} ms frames): {parse_ok}")
        ok &= parse_ok
        st = packet_stats(x, y, p, off, np.array([49, 99]))
        print(f"  packet statistics on the synthetic stream: {len(STATS)} statistics x {len(st['log_rate'])} packets, "
              f"active-only fields finite = {bool(np.isfinite(st['fano']).all())}")
        ok &= bool(np.isfinite(st["fano"]).all())
    # --- the real readers on a tiny slice: shapes and counts only (no distance, no statistic is printed)
    meta = erh_trials()
    trial = min(meta, key=lambda t: (ERH_DIR / f"{t}.evc").stat().st_size)
    r = _erh_task(trial, meta[trial], args.max_packets, "full")
    print(f"  ERH trial {trial}: frames match summary={r['frames_match_summary']}, coordinates inside 240x180="
          f"{r['x_max'] < GRID_W and r['y_max'] < GRID_H}, polarity values {r['p_values']}, {r['n_packets']} packets, "
          f"stats arrays {[v.shape for v in list(r['stats'].values())[:2]]}")
    ok &= r["frames_match_summary"] and r["x_max"] < GRID_W and r["y_max"] < GRID_H and r["n_packets"] > 0
    q = _train_task("lyh_local", args.max_packets, "full")
    print(f"  training stream lyh_local: {q['n_packets']} packets, {len(q['stats'])} statistics")
    ok &= q["n_packets"] > 0
    q2 = _train_task("lyh_local", args.max_packets, "dd")
    r2 = _erh_task(trial, meta[trial], args.max_packets, "dd")
    print(f"  data-driven-ROI variant ran: {q2['n_packets']} / {r2['n_packets']} packets")
    # --- decision plumbing on a synthetic distance table
    def fake(exc_q=None):
        d = {q_: {"U_train_pairwise_max": 1.0, "U_train_pairwise_median": 0.5, "D_erh_max_over_train": 2.0 if q_ == exc_q else 0.8,
                  "D_erh_min_over_train": 0.3, "split_half_max": 0.1} for q_ in STATS}
        d["_posctrl_rate_distance"] = 5.0
        return {"validity": {"frames_match_summary_all": True, "coords_within_grid_all": True, "polarity_values_01_all": True,
                             "n_train_subjects": 9, "n_erh_trials": 80, "n_erh_right_trials": 40}, "distances": d, "roi": "full"}
    v = [decide(fake(None))["verdict"], decide(fake("on_frac"))["verdict"], decide(fake("fano"))["verdict"]]
    print(f"  decide() plumbing on synthetic tables: {v}")
    ok &= v == ["comparable", "partially_comparable", "real_sensor_claims_blocked"]
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m10_sdg"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"ok": bool(ok)}, indent=1))
    print(f"[m10_sdg smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-packets", type=int, default=24)
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m10_sdg (formal) or --smoke")
    C.verify_protocol()
    out_dir = Path(a.out)
    res = {"checklist": CHECKLIST}
    for roi_mode in ("full", "dd"):
        raw = run(workers=a.workers, roi_mode=roi_mode)
        res[roi_mode] = summarize(raw)
    res["primary"] = "full"
    res["decision"] = decide(res["full"])
    res["decision_descriptive_dd_roi"] = decide(res["dd"])
    C.save_json(out_dir / "result.json", res)
    C.receipt(out_dir, {"probe": "m10_sdg", "verdict": res["decision"]["verdict"], "reads": "evrealhands12 .evc + summary.json only"},
              scripts=["tools/mv2/m10_sdg.py", "tools/mv2/common.py"])
    print(json.dumps(res["decision"], indent=1, default=C._np))


if __name__ == "__main__":
    main()
