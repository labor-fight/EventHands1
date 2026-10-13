#!/usr/bin/env python3
"""S43: per-stage cost of the event-driven runtime (`semkine.async_tracker`) on real zgz streams, batch 1, fp32.

Stages per query: lnes (SAE -> LNES of the window), render (splat of the cached mesh), trunk (active-site executor),
head (prev_mlp + zero-event hold + count-gain filter + gate), mesh (local update of the cache). Ingest is timed per event.
Each stage is bracketed by device synchronisation, so the stage times add up to the query time. Run on an idle device.

    python tools/s43/bench_runtime.py --run-dir outputs/semkine/s43n_sp_s3407 --device cuda --out outputs/s43/reports/bench_gpu.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]
from semkine import eval_track as ET                             # noqa: E402
from semkine.async_tracker import EventDrivenTracker             # noqa: E402
from semkine.dataset import sequences_for_split                  # noqa: E402
import filter_eval as FE                                         # noqa: E402

SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"


class Timed(EventDrivenTracker):
    """EventDrivenTracker.query with a synchronised timer around each stage (same arithmetic)."""

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def query(self, t_ms):
        from semkine.kinematic_gate import gate_state
        T = {}
        m = self.model
        self._sync(); t0 = time.perf_counter()
        lnes, w = self.lnes(t_ms)
        n_ev = self._recent_count(t_ms)
        prev = self.state
        empty = not bool((lnes != 0).any())
        self._sync(); T["lnes"] = time.perf_counter() - t0
        macs = 0
        if empty:
            z = prev.clone()
            T["render"] = T["trunk"] = 0.0
        else:
            t0 = time.perf_counter()
            rend = self.render().to(lnes.dtype)
            self._sync(); T["render"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            logits, macs, _ = self.exec.run(torch.cat([lnes, rend], dim=-1))
            self._sync(); T["trunk"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if not empty:
            z = prev + (logits[None].to(self.host) + self.prev_mlp(prev))
        out = z
        if self.filter is not None:
            self.filter.trk_model.z = z
            out = self.filter(torch.full((1, 1), 0.0 if empty else 1.0, device=self.host), prev, n_events=n_ev)
        if not empty and not (np.isinf(self.n0) and np.isinf(self.n0_root)):
            out = gate_state(prev.to(out.dtype), out, self._joint_counts(lnes, w)[None].to(self.host), self.n0, self.n0_root)
        self._sync(); T["head"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        info = self.mesh.update(out[0])
        self._sync(); T["mesh"] = time.perf_counter() - t0
        self.state = out
        self.macs.append(macs)
        return {"state": out, "T": T, "macs": macs, "mesh": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n0", type=float, default=float("inf"))
    ap.add_argument("--n0-root", type=float, default=float("inf"))
    ap.add_argument("--max-queries", type=int, default=800)
    ap.add_argument("--host-state", action="store_true", help="state / filter / mesh cache on the CPU, trunk on the device")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    cfg, model, ckpt, step, _ = FE.build_run(Path(a.run_dir), "last", dev)
    root = Path(cfg["DATA"]["ROOT"])
    trk = Timed(model, json.loads(SPEC.read_text()), n0=a.n0, n0_root=a.n0_root, device=dev, host_state=a.host_state)
    rec = {"T": [], "ingest_s": 0.0, "events": 0, "seq": []}
    rng = np.random.default_rng(0)
    for s, d in sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"])):
        events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
        trk.set_hand(torch.tensor(aux["betas"]), torch.tensor(aux["camera_K"]))
        nq = 0
        for ra, rb in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            ends = np.arange(ra + 49, rb, 50)
            if not len(ends) or nq >= a.max_queries:
                continue
            trk.reset(torch.from_numpy(pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)), floor_ms=int(ra))
            t_last = int(ra) - 1
            for t in ends:
                for ms in range(t_last + 1, int(t) + 1):
                    e0, e1 = int(offsets[ms]), int(offsets[ms + 1])
                    if e1 > e0:
                        ev = np.asarray(events[e0:e1])
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        trk.ingest(ev[:, 0], ev[:, 1], ev[:, 2], ms)
                        if dev.type == "cuda":
                            torch.cuda.synchronize()
                        rec["ingest_s"] += time.perf_counter() - t0
                        rec["events"] += e1 - e0
                t_last = int(t)
                r = trk.query(int(t))
                rec["T"].append(r["T"])
                rec["seq"].append(s)
                nq += 1
                if nq >= a.max_queries:
                    break
    keys = ("lnes", "render", "trunk", "head", "mesh")
    seqs = np.asarray(rec["seq"])
    out = {"run": Path(a.run_dir).name, "device": str(dev), "host_state": a.host_state, "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
           "threads": a.threads, "n0": a.n0, "n0_root": a.n0_root, "queries": len(rec["T"]),
           "ingest_us_per_event": 1e6 * rec["ingest_s"] / max(rec["events"], 1)}
    for sel_name in ("all", "zgz_global", "zgz_local"):
        sel = np.ones(len(seqs), bool) if sel_name == "all" else seqs == sel_name
        Ts = [t for t, k in zip(rec["T"], sel) if k]
        tot = np.array([sum(t[k] for k in keys) for t in Ts]) * 1e3
        out[sel_name] = {"query_ms_p50": float(np.median(tot)), "query_ms_mean": float(tot.mean()),
                         "query_ms_p90": float(np.percentile(tot, 90)),
                         "stage_ms_mean": {k: float(np.mean([t[k] for t in Ts]) * 1e3) for k in keys}}
    out["macs_median"] = float(np.median(trk.macs))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
