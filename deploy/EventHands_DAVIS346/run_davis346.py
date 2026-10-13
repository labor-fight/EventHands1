#!/usr/bin/env python3
"""Real-time 3D hand mesh tracking with S39 (dt_dz_l3, W=200 + adaptive filter + evidence gate ramp1000).

    # live DAVIS346: put the RIGHT hand on the yellow template (palm to the camera), press SPACE
    python run_davis346.py --calib calib.json
    # a recording through the same loop, as fast as possible (or --source sim:rec.aedat4 = replayed in real time)
    python run_davis346.py --source rec.aedat4 --calib calib.json --init my_init.json --save out.npz --video out.mp4

Keys: SPACE start / restart from the template (or the --init file)   T stop tracking   +/- template size
      S save now   Q / ESC quit
See README.md: calibration (intrinsics, hot pixels, event-count scale), the initial state, the output format.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from eh346.events import EventBuffer, SensorMap, build_lnes  # noqa: E402
from eh346.sources import Clock, open_source  # noqa: E402


def load_init(path: str, mano):
    """--init JSON: {"state51": [51]} or MANO keys {"transl", "global_orient", "hand_pose", "flat_hand_mean"}, all in the
    camera frame (metres, axis-angle); optional "betas": [10]."""
    d = json.loads(Path(path).read_text())
    if "state51" in d:
        s = np.asarray(d["state51"], np.float32)
    else:
        from eh346.mano import state_from_mano
        s = state_from_mano(d["transl"], d["global_orient"], d["hand_pose"], bool(d.get("flat_hand_mean", False)),
                            mano.hands_mean.cpu()).numpy()[0]
    return s, (np.asarray(d["betas"], np.float32) if "betas" in d else None)


def camera_model(calib: dict, W: int, H: int, hfov: float):
    if calib:
        if calib.get("_template"):
            sys.exit("calib.json is still the template: put your own camera calibration in it (README, camera calibration)")
        if (int(calib["width"]), int(calib["height"])) != (W, H):
            sys.exit(f"calib.json is for {calib['width']}x{calib['height']}, the source is {W}x{H}")
        return np.asarray(calib["K"], np.float64), calib.get("dist")
    if hfov > 0:
        f = (W / 2) / math.tan(math.radians(hfov) / 2)
        print(f"WARNING: no calibration, assuming HFOV {hfov} deg (f = {f:.1f} px); calibrate for real use")
        return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]]), None
    sys.exit("give --calib calib.json (README: camera calibration) or, for a first look only, --hfov <deg>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="camera", help="camera | camera:<serial> | rec.aedat4 | rec.npz | sim:rec.aedat4")
    ap.add_argument("--calib", default="", help="calib.json: width, height, K [, dist, count_scale, betas, hotpix, rot180]")
    ap.add_argument("--hfov", type=float, default=0.0, help="without calibration: assumed horizontal field of view (deg)")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "s39_s3408.pt"),
                    help="a weight file, or several comma separated = seed ensemble (x3 compute)")
    ap.add_argument("--device", default="", help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--step-ms", type=int, default=50, help="output step; 50 = the evaluated S39 contract (README)")
    ap.add_argument("--count-scale", type=float, default=0.0,
                    help="evidence-gate event-count scale (default: calib.json, else 640*480/(W*H))")
    ap.add_argument("--init", default="template", help="template | init.json")
    ap.add_argument("--start-ms", type=int, default=-1, help="session ms where tracking starts with --init (default: files the first event, live: now)")
    ap.add_argument("--ba-us", type=int, default=2000, help="camera: on-camera background-activity filter (us, 0 = off)")
    ap.add_argument("--sw-ba-us", type=int, default=0, help="software background-activity filter (us, 0 = off)")
    ap.add_argument("--hotpix", default="", help="hot-pixel mask .npy (calibrate_davis346.py hotpix)")
    ap.add_argument("--rot180", action="store_true", help="camera mounted upside down")
    ap.add_argument("--save", default="", help="write states / joints (/ vertices with --save-verts) to this .npz")
    ap.add_argument("--save-verts", action="store_true")
    ap.add_argument("--video", default="", help="also write the visualisation to this .mp4")
    ap.add_argument("--no-viz", action="store_true", help="no window")
    ap.add_argument("--scale", type=int, default=3, help="display scale of the 240 x 180 image")
    ap.add_argument("--show-aps", action="store_true", help="draw on the DAVIS346 APS frame instead of the events")
    ap.add_argument("--duration-s", type=float, default=0.0, help="stop after this much sensor time")
    ap.add_argument("--cuda-graph", action="store_true", help="CNN replayed from a CUDA graph (about -1 ms per step)")
    ap.add_argument("--max-lag-ms", type=int, default=150, help="live: skip ahead when processing falls this far behind")
    ap.add_argument("--t0-us", type=int, default=-1, help="session origin (default: first event, floored to 1 ms)")
    a = ap.parse_args()

    import torch
    from eh346.tracker import S39Tracker
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    calib = json.loads(Path(a.calib).read_text()) if a.calib else {}
    src = (open_source(a.source, ba_filter_us=a.ba_us, sw_ba_filter_us=a.sw_ba_us) if a.source.startswith("camera")
           else open_source(a.source, sw_ba_filter_us=a.sw_ba_us))
    live = bool(getattr(src, "live", False))
    W, H = src.resolution
    K, dist = camera_model(calib, W, H, a.hfov)
    hot = a.hotpix or calib.get("hotpix", "")
    smap = SensorMap(W, H, K, dist, rot180=a.rot180 or bool(calib.get("rot180", False)), hotpix=np.load(hot) if hot else None)
    count_scale = a.count_scale or float(calib.get("count_scale", 0.0)) or (640.0 * 480.0) / (W * H)
    trk = S39Tracker([w for w in a.weights.split(",") if w], device=device, step_ms=a.step_ms, count_scale=count_scale,
                     cuda_graph=a.cuda_graph)
    trk.set_camera(smap.K_model())
    trk.warmup()                         # first CUDA / cuDNN calls are slow: do them before the first real step
    if "betas" in calib:
        trk.betas = torch.as_tensor(np.asarray(calib["betas"], np.float32).reshape(1, 10), device=trk.device)
    init_state, init_betas = (None, None) if a.init == "template" else load_init(a.init, trk.mano)
    print(f"source {getattr(src, 'name', a.source)} {W}x{H}{' (live)' if live else ''} | device {device} | "
          f"weights {a.weights} | step {a.step_ms} ms | count scale {count_scale:.3f}")

    show = not a.no_viz
    draw = show or bool(a.video)
    if draw:
        import cv2
        from eh346.viz import draw_hand, lnes_image, put_lines
    if show:
        cv2.namedWindow("S39 DAVIS346", cv2.WINDOW_AUTOSIZE)
    writer = None

    buf, clock = None, Clock()
    st = {"tracking": False, "next_end": None, "tpl_px": 72.0}
    last_out = None
    rec = {"end_ms": [], "state": [], "joints": [], "verts": [], "n50": [], "compute_ms": []}
    skips, last_viz, first_ms = 0, 0.0, None
    pending_start = (not live) or init_state is not None   # files start by themselves; live only with an --init file

    def start(state, betas, at_ms):
        trk.reset(state, at_ms, betas)
        st["next_end"], st["tracking"] = trk.first_end(), True
        print(f"tracking from session ms {at_ms}")

    def save():
        if not a.save or not rec["end_ms"]:
            return
        out = {"end_ms": np.asarray(rec["end_ms"]), "state51": np.asarray(rec["state"]), "joints21": np.asarray(rec["joints"]),
               "n50": np.asarray(rec["n50"]), "compute_ms": np.asarray(rec["compute_ms"]), "t0_us": buf.t0,
               "K_model": smap.K_model(), "betas": trk.betas.cpu().numpy()[0], "step_ms": a.step_ms}
        if a.save_verts:
            out["verts778"] = np.asarray(rec["verts"])
        np.savez_compressed(a.save, **out)
        print(f"saved {len(rec['end_ms'])} steps -> {a.save}")

    def frame(now):
        sc = a.scale
        if st["tracking"] and last_out is not None:
            lnes = last_out["lnes"]
        else:
            xyp, msr = buf.window(buf.latest_ms, 50)
            lnes = build_lnes(xyp, msr, 50)
        img = lnes_image(lnes, sc)
        if a.show_aps and getattr(src, "last_frame", None) is not None:
            fr = src.last_frame
            if dist is not None and np.any(np.abs(np.asarray(dist, np.float64)) > 0):
                if "aps_maps" not in st:          # the frame is distorted; the mesh is projected undistorted
                    Kd = np.asarray(K, np.float64)
                    st["aps_maps"] = cv2.initUndistortRectifyMap(Kd, np.asarray(dist, np.float64), None, Kd,
                                                                 (fr.shape[1], fr.shape[0]), cv2.CV_32FC1)
                fr = cv2.remap(fr, st["aps_maps"][0], st["aps_maps"][1], cv2.INTER_LINEAR)
            fr = fr[::-1, ::-1] if smap.rot180 else fr
            fr = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_GRAY2BGR) if fr.ndim == 2 else fr
            img = cv2.resize(fr, (240 * sc, 180 * sc), interpolation=cv2.INTER_LINEAR)
        if st["tracking"]:
            v, j = trk.mesh()
            uv = trk.project(v)
            draw_hand(img, uv, trk.project(j), sc, (0, 255, 0))
            side = float(max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1])))
            c = np.asarray(rec["compute_ms"][-20:]) if rec["compute_ms"] else np.zeros(1)
            lines = [f"TRACKING  t {st['next_end'] - trk.step_ms} ms  compute {np.median(c):.1f} ms  "
                     f"n50 {last_out['n50'] if last_out else 0}  skips {skips}",
                     f"hand {side:.0f} px (training 60-90)  depth {np.median(v[:, 2]):.2f} m"]
            if not 45 <= side <= 110:
                lines.append("hand size outside the training range: move closer / farther")
        else:
            s = init_state if init_state is not None else trk.template_state(st["tpl_px"])
            v, j = trk.mesh(s)
            draw_hand(img, trk.project(v), trk.project(j), sc, (0, 255, 255))
            lines = ["NOT TRACKING: RIGHT hand on the yellow template, palm to the camera",
                     "SPACE start   +/- template size   Q quit"]
        return put_lines(img, lines)

    try:
        while True:
            # ---- events: drain what has arrived (live), or one batch (file)
            eof, got = False, 0
            while True:
                try:
                    batch = src.read()
                except StopIteration:
                    eof = True
                    break
                if batch is None:
                    break
                t, x, y, p = batch
                if buf is None:
                    buf = EventBuffer(a.t0_us if a.t0_us >= 0 else int(t[0]) - int(t[0]) % 1000)
                mx, my, mp, ok = smap.map(x, y, p)
                buf.push(np.asarray(t)[ok], mx, my, mp)
                clock.update(buf.latest_ms)
                first_ms = buf.latest_ms if first_ms is None else first_ms
                got += 1
                if not live or got >= 200:
                    break
            if buf is None:
                if eof:
                    break
                time.sleep(0.0005)
                continue
            if not got and live and not eof:
                time.sleep(0.0003)
            now = buf.latest_ms + 1 if eof else clock.now_ms(live)
            if pending_start and now is not None:
                at = a.start_ms if a.start_ms >= 0 else (int(buf.ms[buf.lo]) if not live else now + 1)
                start(init_state if init_state is not None else trk.template_state(st["tpl_px"]), init_betas, int(at))
                pending_start = False
            # ---- every step whose window is complete (an event after it arrived / the clock passed it)
            stepped = False
            while st["tracking"] and now is not None and now > st["next_end"]:
                if live and now - st["next_end"] > a.max_lag_ms:
                    st["next_end"] += (now - st["next_end"]) // trk.step_ms * trk.step_ms
                    skips += 1
                    continue
                last_out = trk.step(buf, st["next_end"])
                v, j = trk.mesh()
                rec["end_ms"].append(st["next_end"]); rec["state"].append(last_out["state"]); rec["joints"].append(j)
                rec["n50"].append(last_out["n50"]); rec["compute_ms"].append(last_out["compute_ms"])
                if a.save_verts:
                    rec["verts"].append(v)
                st["next_end"] += trk.step_ms
                stepped = True
            buf.prune((st["next_end"] if st["tracking"] else buf.latest_ms) - 400)
            if a.duration_s and buf.latest_ms - first_ms > a.duration_s * 1000:
                eof = True
            if not draw:
                if stepped and len(rec["end_ms"]) % 200 == 0:
                    c = np.asarray(rec["compute_ms"][-200:])
                    print(f"step {len(rec['end_ms'])}: t {st['next_end']} ms, compute p50 {np.median(c):.2f} ms, "
                          f"n50 {last_out['n50']}, skips {skips}")
                if eof:
                    break
                continue
            # ---- display / video: every step, and ~30 Hz while not tracking
            wall = time.perf_counter()
            if not stepped and (wall - last_viz < 0.033 or st["tracking"]) and not eof:
                continue
            last_viz = wall
            img = frame(now)
            if a.video:
                if writer is None:
                    writer = cv2.VideoWriter(a.video, cv2.VideoWriter.fourcc(*"mp4v"), 1000.0 / trk.step_ms,
                                             (img.shape[1], img.shape[0]))
                writer.write(img)
            if show:
                cv2.imshow("S39 DAVIS346", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    s, b = (init_state, init_betas) if init_state is not None else (trk.template_state(st["tpl_px"]), None)
                    start(s, b, (now if now is not None else buf.latest_ms) + 1)
                elif key == ord("t"):
                    st["tracking"] = False
                elif key in (ord("+"), ord("=")):
                    st["tpl_px"] = min(st["tpl_px"] + 4, 110)
                elif key == ord("-"):
                    st["tpl_px"] = max(st["tpl_px"] - 4, 45)
                elif key == ord("s"):
                    save()
            if eof:
                break
    except KeyboardInterrupt:
        pass
    if writer is not None:
        writer.release()
    if rec["compute_ms"]:
        c = np.asarray(rec["compute_ms"])
        print(f"{len(c)} steps, compute per step p50 {np.median(c):.2f} ms, p99 {np.percentile(c, 99):.2f} ms, skips {skips}")
    save()


if __name__ == "__main__":
    main()
