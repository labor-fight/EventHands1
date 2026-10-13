#!/usr/bin/env python3
"""S43 G6: run an S43 checkpoint through the event-driven asynchronous runtime (`semkine.async_tracker`) on zgz and
compare it with the packet evaluation of the same checkpoint (`tools/s43/gate_eval.py`, S39 stack).

Events are ingested millisecond by millisecond in time order (each event one O(1) write of the SAE); the tracker is
queried every `--hop` ms (50 = the protocol's cadence: the queries are exactly the evaluated steps). Every segment starts
from the protocol's noisy ground truth (rng 0, drawn in evalx's order) and the window is clipped to the segment start.
At a hop below 50 ms the count gains and the kinematic gate are converted to the hop, a_h = 1 - (1 - a_50)^(h / 50)
(the same per-50-ms contraction), and only the protocol's steps are scored.

    python tools/s43/async_eval.py --run-dir outputs/semkine/s43_sp_s3407 --hop 50 --n0 inf \
        --ref outputs/s43/gate/s43_sp_s3407/off.npz --out outputs/s43/async/s43_sp_s3407_off_h50.json
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
from mano_layer import ManoLayer                                 # noqa: E402
from semkine import eval_track as ET                             # noqa: E402
from semkine.async_tracker import EventDrivenTracker             # noqa: E402
from semkine.dataset import sequences_for_split                  # noqa: E402
import evalx as EX                                               # noqa: E402
import filter_eval as FE                                         # noqa: E402

SPEC = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"


def hop_spec(spec: dict, hop: int) -> dict:
    """The 50 ms gain spec converted to a hop of `hop` ms: a -> 1 - (1 - a)^(hop / 50) at every node."""
    if hop == 50:
        return spec
    f = hop / 50.0
    conv = lambda g: 1.0 - (1.0 - float(g)) ** f                                            # noqa: E731
    out = {}
    for k, v in spec.items():
        if k in ("a_root", "a_rest", "a_trans"):
            out[k] = conv(v) if isinstance(v, (int, float)) else [[n, conv(g)] for n, g in v]
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--hop", type=int, default=50)
    ap.add_argument("--n0", type=float, default=float("inf"))
    ap.add_argument("--n0-root", type=float, default=float("inf"))
    ap.add_argument("--eps-f", type=float, default=0.0, help="finger update dead band, degrees")
    ap.add_argument("--eps-r", type=float, default=0.0, help="root-rotation update dead band, degrees")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--executor", default="sparse", choices=("sparse", "masked"))
    ap.add_argument("--ref", default=None, help="gate_eval npz of the same checkpoint / n0 (hop 50 only)")
    ap.add_argument("--check-every", type=int, default=50, help="compare the mesh cache with full MANO every N queries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-fp32", action="store_true", help="cuDNN without TF32 (the G6 comparison is run so on both sides)")
    ap.add_argument("--render-full-fk", action="store_true", help="diagnostic: render with the full MANO FK, not the cache")
    ap.add_argument("--host-state", action="store_true", help="state / filter / mesh cache on the CPU, trunk on the device")
    ap.add_argument("--graph", action="store_true", help="semkine.async_graph.GraphTracker: the query as one CUDA graph (deviation 9)")
    ap.add_argument("--no-mesh", action="store_true", help="--graph: skip the host mesh cache (critical-path latency)")
    ap.add_argument("--replay-ref", action="store_true",
                    help="open loop: before every query set the state (and the mesh cache) to the reference evaluation's own "
                         "previous output, so each step compares the two implementations on identical inputs (hop 50, --ref)")
    a = ap.parse_args()
    if a.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if 50 % a.hop:
        raise SystemExit("--hop must divide 50")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Path(a.run_dir)
    cfg, model, ckpt, step, _ = FE.build_run(run, a.ckpt, device)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
    spec = hop_spec(json.loads(SPEC.read_text()), a.hop)
    n0 = a.n0 if (np.isinf(a.n0) or a.hop == 50) else a.n0       # gate threshold in pixels of the last 50 ms (unchanged)
    if a.graph:
        from semkine.async_graph import GraphTracker
        if not (np.isinf(a.n0) and np.isinf(a.n0_root)):
            raise SystemExit("--graph: the kinematic evidence gate is not part of the graphed runtime (use eps-f / eps-r)")
        trk = GraphTracker(model, spec, window_ms=200, hop_ms=50, device=device, eps_f=np.radians(a.eps_f),
                           eps_r=np.radians(a.eps_r), mesh=not a.no_mesh)
    else:
        trk = EventDrivenTracker(model, spec, n0=n0, radius_px=a.radius, window_ms=200, hop_ms=50, executor=a.executor,
                                 device=device, n0_root=a.n0_root, host_state=a.host_state,
                                 eps_f=np.radians(a.eps_f), eps_r=np.radians(a.eps_r))
    trk.render_full_fk = a.render_full_fk
    if a.graph and a.render_full_fk:
        pass                                              # the graphed runtime always renders from the full FK
    gain_pow = a.hop / 50.0
    if not (np.isinf(a.n0) and np.isinf(a.n0_root)) and a.hop != 50 and not a.graph:
        # the gate gain per query: 1 - (1 - g)^(hop/50); implemented by wrapping the count -> gain map
        base = trk._joint_counts

        def counts_hop(lnes, w):
            c = base(lnes, w)
            n0v = torch.tensor([trk.n0_root] + [trk.n0] * 15, device=c.device)
            g = (c / n0v).clamp(max=1.0)
            return torch.where(torch.isinf(n0v), c, (1.0 - (1.0 - g) ** gain_pow) * n0v)
        trk._joint_counts = counts_hop
    ref = np.load(a.ref) if a.ref else None
    rng = np.random.default_rng(0)
    res, diffs, lat, mesh_err, reskin, jchg, mesh_seq = {}, [], [], [], [], [], {}
    t_all = time.time()
    for s, d in seqs:
        events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device)
        trk.set_hand(betas, K)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        preds, gts, ends_all, runs_all, counts = [], [], [], [], []
        seq_reskin, seq_jchg, seq_held = [], [], []
        for run_id, (ra, rb) in enumerate(runs):
            ends = np.arange(ra + 49, rb, 50, dtype=np.int64)
            if not len(ends):
                continue
            prev = pos51[ra].copy() + ET.sample_init_noise(cfg, rng, 1.0)
            trk.reset(torch.from_numpy(prev), floor_ms=int(ra))
            ref_prev = prev
            t_last = int(ra) - 1
            end_set = set(int(e) for e in ends)
            for t in range(int(ra) + a.hop - 1, int(ends[-1]) + 1, a.hop):
                # ingest the events of the milliseconds (t_last, t], one millisecond at a time
                for ms in range(t_last + 1, t + 1):
                    e0, e1 = int(offsets[ms]), int(offsets[ms + 1])
                    if e1 > e0:
                        ev = np.asarray(events[e0:e1])
                        trk.ingest(ev[:, 0], ev[:, 1], ev[:, 2], ms)
                t_last = t
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if a.replay_ref and t in end_set:
                    st = torch.from_numpy(np.asarray(ref_prev, dtype=np.float32)).view(1, 51).to(getattr(trk, "host", device))
                    trk.state = st
                    if getattr(trk, "use_mesh", True):
                        trk.mesh.reset(st[0].to(trk.mesh.device))
                q0 = time.perf_counter()
                out = trk.query(t)
                if a.replay_ref and t in end_set:
                    ref_prev = ref[f"model|{s}|pred"][len(preds)]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                lat.append(time.perf_counter() - q0)
                if out["mesh"]["joints_changed"] >= 0:                  # -1 = a periodic full refresh
                    reskin.append(out["mesh"]["verts_reskinned"])
                    jchg.append(out["mesh"]["joints_changed"])
                    seq_reskin.append(out["mesh"]["verts_reskinned"])
                    seq_jchg.append(out["mesh"]["joints_changed"])
                if a.check_every and trk.timing["n_query"] % a.check_every == 0:
                    mesh_err.append(trk.mesh.check(betas))
                if t in end_set:
                    preds.append(out["state"][0].cpu().numpy())
                    gts.append(pos51[t])
                    ends_all.append(t)
                    runs_all.append(run_id)
                    counts.append(int(offsets[t + 1] - offsets[t - 49]))
        r = {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
             "end": np.asarray(ends_all), "run": np.asarray(runs_all), "count": np.asarray(counts), "betas": aux["betas"]}
        m = EX.per_step_metrics(mano, r, device)
        res[s] = (r, m)
        mesh_seq[s] = {"verts_reskinned_mean": float(np.mean(seq_reskin)) if seq_reskin else None,
                       "joints_changed_mean": float(np.mean(seq_jchg)) if seq_jchg else None,
                       "frac_queries_no_finger_change": float(np.mean(np.asarray(seq_jchg) == 0)) if seq_jchg else None}
        if ref is not None:
            rp = ref[f"model|{s}|pred"]
            assert np.array_equal(ref[f"model|{s}|end"], r["end"]), "step sets differ"
            diffs.append(float(np.abs(rp - r["pred"]).max()))
        print(f"{s}: RA={m['mpjpe_ra_mm'].mean():.4f} n={len(r['end'])}", flush=True)
    n = sum(len(m["mpjpe_ra_mm"]) for _, m in res.values())
    overall = {k: float(sum(m[k].sum() for _, m in res.values()) / n) for k in next(iter(res.values()))[1]}
    out = {"run": run.name, "ckpt": str(ckpt), "step": step, "hop_ms": a.hop, "n0": a.n0, "executor": a.executor,
           "graph": a.graph, "mesh_cache": not a.no_mesh, "graph_stats": trk.timing if a.graph else None,
           "eps_f_deg": a.eps_f, "eps_r_deg": a.eps_r,
           "overall": overall, "per_seq": {s: {k: float(v.mean()) for k, v in m.items()} for s, (r, m) in res.items()},
           "n_frames": n, "n_queries": trk.timing["n_query"], "n_events": trk.timing["n_events"],
           "latency_query_ms": {"p50": float(np.median(lat) * 1e3), "p90": float(np.percentile(lat, 90) * 1e3),
                                "mean": float(np.mean(lat) * 1e3)},
           "ingest_us_per_event": 1e6 * trk.timing["ingest"] / max(trk.timing["n_events"], 1),
           "macs": {"median": float(np.median(trk.macs)), "mean": float(np.mean(trk.macs)),
                    "p90": float(np.percentile(trk.macs, 90))},
           "mesh": {"max_err_m": float(max(mesh_err)) if mesh_err else None, "checks": len(mesh_err),
                    "verts_reskinned_mean": float(np.mean(reskin)), "joints_changed_mean": float(np.mean(jchg)),
                    "rigid_only_frac": float(np.mean(np.asarray(jchg) == 0)), "per_seq": mesh_seq},
           "ref": a.ref, "max_abs_diff_vs_ref": max(diffs) if diffs else None, "wall_s": time.time() - t_all}
    if ref is not None:
        ref_ra = sum(float(ref[f"model|{s}|mpjpe_ra_mm"].sum()) for s in res) / n
        out["ref_ra"] = ref_ra
        out["ra_diff_vs_ref"] = overall["mpjpe_ra_mm"] - ref_ra
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out.get(k) for k in ("overall", "latency_query_ms", "macs", "mesh", "max_abs_diff_vs_ref", "ra_diff_vs_ref")}, indent=1))


if __name__ == "__main__":
    main()
