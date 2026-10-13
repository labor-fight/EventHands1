#!/usr/bin/env python3
"""DAVIS346 calibration helpers (no network needed).

hotpix   Record a STATIC scene (no hand, nothing moving) and mask pixels that fire far more than the rest.
             python calibrate_davis346.py hotpix --seconds 5 --out hotpix.npy
counts   Record while moving the right hand in front of the camera (at the tracking distance), measure the sensor's
         sensitivity as the median number of events per active pixel in 50 ms windows (nearly independent of how fast
         the hand moves: 1.2-1.5 for every training local sequence while their event counts differ 10x), and print the
         evidence gate's --count-scale = (640*480 / (W*H)) x training value / this camera's value; with --calib, write
         it (and the hot-pixel mask path) into calib.json.
             python calibrate_davis346.py counts --seconds 20 --calib calib.json --hotpix hotpix.npy
Both accept --source rec.aedat4 to use a recording instead of the live camera.

Why the count matters: S39's filter gain (how much of each network update is taken) is a function of the number of
events in the last 50 ms, calibrated on the training camera (v2e-simulated 640 x 480 events). The DAVIS346 has 3.4x
fewer pixels and its own contrast thresholds and noise, so its raw count must be rescaled to the training scale,
otherwise the tracker damps the updates too much (sluggish) or too little (jittery).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eh346.sources import DavisCamera, open_source  # noqa: E402


def record(src, seconds: float):
    """All events of `seconds` (sensor time) from the source: t_us, x, y, p."""
    live = isinstance(src, DavisCamera)
    ts, xs, ys, ps, t_first = [], [], [], [], None
    wall0 = time.perf_counter()
    while True:
        try:
            b = src.read()
        except StopIteration:
            break
        if b is None:
            if live:
                time.sleep(0.0005)
                if time.perf_counter() - wall0 > seconds + 5:
                    break
            continue
        t, x, y, p = b
        if t_first is None:
            t_first = int(t[0])
        ts.append(t); xs.append(x); ys.append(y); ps.append(p)
        if int(t[-1]) - t_first >= seconds * 1e6:
            break
    if not ts:
        sys.exit("no events received")
    return np.concatenate(ts), np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["hotpix", "counts"])
    ap.add_argument("--source", default="camera")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--ba-us", type=int, default=2000, help="on-camera background-activity filter (us), as for tracking")
    ap.add_argument("--hotpix", default="", help="counts: hot-pixel mask to apply (from the hotpix mode)")
    ap.add_argument("--rate-hz", type=float, default=30.0, help="hotpix: minimum event rate of a hot pixel")
    ap.add_argument("--out", default="hotpix.npy")
    ap.add_argument("--calib", default="", help="counts: calib.json to update with count_scale and hotpix")
    ap.add_argument("--reference", default="local", choices=["local", "global"],
                    help="counts: training reference: local = hand moving in place (default), global = whole arm moving")
    a = ap.parse_args()
    src = open_source(a.source, ba_filter_us=a.ba_us) if a.source.startswith("camera") else open_source(a.source)
    W, H = src.resolution
    secs = a.seconds or (5.0 if a.mode == "hotpix" else 20.0)
    print(f"{getattr(src, 'name', a.source)} {W}x{H}: recording {secs:.0f} s "
          f"({'keep the scene STATIC' if a.mode == 'hotpix' else 'move your RIGHT hand as in normal use'}) ...")
    t, x, y, p = record(src, secs)
    dur = (t[-1] - t[0]) / 1e6
    if a.mode == "hotpix":
        cnt = np.zeros((H, W), np.int64)
        np.add.at(cnt, (y.astype(np.intp), x.astype(np.intp)), 1)
        rate = cnt / max(dur, 1e-3)
        active = rate[rate > 0]
        thr = max(a.rate_hz, 20.0 * float(np.median(active)) if len(active) else a.rate_hz)
        mask = rate > thr
        np.save(a.out, mask)
        print(f"{len(t)} events in {dur:.1f} s; {int(mask.sum())} hot pixels (> {thr:.1f} Hz) -> {a.out}; "
              f"they carried {cnt[mask].sum() / max(len(t), 1) * 100:.1f}% of the events")
        return
    keep = np.ones(len(t), bool)
    if a.hotpix:
        hp = np.load(a.hotpix)
        keep = ~hp[y.astype(np.intp), x.astype(np.intp)]
    tk, xk, yk = t[keep], x[keep].astype(np.int64), y[keep].astype(np.int64)
    win = (tk - tk[0]) // 50000
    n = np.bincount(win)
    act = np.bincount(np.unique(win * (W * H) + yk * W + xk) // (W * H), minlength=len(n))
    pix_ratio = (640.0 * 480.0) / (W * H)
    busy = n >= 300.0 / pix_ratio                    # windows with a moving hand (300 events on the training camera)
    if busy.sum() < 20:
        sys.exit(f"only {int(busy.sum())} windows with enough events: move the hand more / record longer")
    epap = float(np.median(n[busy] / act[busy]))
    cfg = json.loads((ROOT / "configs" / "s39.json").read_text())
    ref = cfg["count_reference"]["events_per_active_pixel"][a.reference]["median"]
    scale = pix_ratio * ref / epap
    q = np.percentile(n[busy], [25, 50, 75])
    print(f"busy 50 ms windows: {int(busy.sum())} of {len(n)}; events per window p25/50/75 = {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f}")
    print(f"events per active pixel (sensor sensitivity): this camera {epap:.3f}, training ({a.reference}) {ref:.3f}")
    print(f"recommended --count-scale = pixel ratio {pix_ratio:.3f} x {ref:.3f} / {epap:.3f} = {scale:.3f}")
    print("(valid when the hand appears 60-90 px wide in the tracker's 240 x 180 image, as the viewer reports)")
    if a.calib:
        c = json.loads(Path(a.calib).read_text())
        c["count_scale"] = round(float(scale), 4)
        if a.hotpix:
            c["hotpix"] = str(Path(a.hotpix).resolve())
        Path(a.calib).write_text(json.dumps(c, indent=1))
        print(f"written to {a.calib}")


if __name__ == "__main__":
    main()
