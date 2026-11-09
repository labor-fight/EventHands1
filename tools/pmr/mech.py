#!/usr/bin/env python3
"""PMR mechanism reading (d) (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): does P carry cross-part information?
Zero training, delivered with the screen, not a gate.

    python -m tools.pmr.mech capture --arm A [--seed S]        (one GPU among 2-7)
        the accepted arm at step 6000 stepped on zgz with the GT previous state (E3, the deployed routes); per endpoint
        its 17 node inputs, prior joints, prior and target state; mech/<run>/capture.npz
    python -m tools.pmr.mech analyze --arm A [--seed S]        (CPU)
        per-joint counterfactual with the arm's own weights (fixed draws, rng 0): for every endpoint b and finger joint j,
        the inputs of every node except j (the 15 other parts and the hand node) are replaced by those of a donor
        endpoint of the same sequence >= 1 s away, node j keeps its own input and the edge gates keep b's geometry;
          e_intact(j) / e_others(j) / e_own(j) = |d theta_j - d theta_j^GT| (rad) with the intact inputs / the others
          swapped / node j's own input swapped (the others intact)
        cross-part reliance = mean e_others / mean e_intact (1.0 exactly when no message reaches j, i.e. the noprop arm;
        > 1 when the other parts' evidence of this packet makes joint j's increment more accurate); also the share of
        the output change caused by the others, |f(x_j, x'_-j) - f(x)| / (|f(x_j, x'_-j) - f(x)| + |f(x'_j, x_-j) - f(x)|);
        root / translation with the 16 part nodes swapped (the hand node kept). Gate statistics per edge type.
        mech/<run>/report.json and .md
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pmr import arms as A  # noqa: E402
from tools.pmr.common import OUT, SEED, ZGZ, now, read, require, save  # noqa: E402

DONOR_MIN_MS = 1000
GROUPS = {"MCP": (1, 4, 7, 10), "PIP": (2, 5, 8, 11), "DIP": (3, 6, 9, 12), "thumb": (13, 14, 15)}


def capture(arm, seed):
    import torch
    from tools.pfo import engine as E
    from tools.pfo import nets as N
    from tools.xcl import evaluate as XE
    from tools.pmr import run as R
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, range(2, 8))), "one GPU among 2-7")
    d = OUT / "mech" / A.run_name(arm, seed)
    require(not (d / "capture.npz").exists(), "written once")
    d.mkdir(parents=True, exist_ok=True)
    XE.configure_environment()
    XE.set_policy()
    tag, sp = R.arm_spec(arm, seed)
    s = R.prepare(sp)
    device = torch.device("cuda")
    net = N.load(tag, route=None, tau=None, device=device,
                 spec={**s, "checkpoint": Path(s["checkpoint"]), "config": Path(s["config"])})
    model = net.model
    model.xyz_keep_aux = True
    model.xyz.xag_capture_enabled = True
    rec = {k: [] for k in ("x_part", "x_hub", "J", "prior", "target", "seq", "end_ms", "delta")}
    root, seqs = E.sequences(net.cfg, ZGZ)
    t0 = time.time()
    with torch.no_grad():
        for si, (seq, directory) in enumerate(seqs):
            ctx = N.sequence(root, directory, seq, device)
            net.bind(ctx)
            rows = E.endpoint_rows(ctx.aux)
            g = lambda ms: np.asarray(ctx.gt[int(ms)], np.float32).copy()  # noqa: E731
            for i in range(len(rows)):
                _, end, prev_ms, _ = rows[i]
                prior = g(prev_ms)
                net.step(ctx, end, prior)
                L = model.xyz_last
                rec["x_part"].append(L["x_part"][0].float().cpu().numpy())
                rec["x_hub"].append(L["x_hub"][0].float().cpu().numpy())
                rec["J"].append(L["J"][0].float().cpu().numpy())
                rec["delta"].append(L["delta"][0].float().cpu().numpy())
                rec["prior"].append(prior)
                rec["target"].append(g(end))
                rec["seq"].append(si)
                rec["end_ms"].append(int(end))
            print(seq, len(rows), round(time.time() - t0, 1), flush=True)
    arrays = {k: np.stack(v) if k not in ("seq", "end_ms") else np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(d / "capture.npz", **arrays)
    save(d / "capture.json", {"arm": arm, "seed": seed, "spec": str(sp), "rows": len(rec["seq"]), "written_utc": now()},
         exclusive=True)


def load_trunk(arm, seed):
    import torch
    from tools.pmr import run as R
    import yaml
    tag, sp = R.arm_spec(arm, seed)
    s = read(sp)
    cfg = yaml.safe_load(Path(s["config"]).read_text())
    import importlib
    mod = importlib.import_module(cfg["PMR"]["MODULE"])
    from mano_layer import ManoLayer
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False)
    m = cfg["MODEL"]
    net = type("MechNet", (mod.PMRNet,), {"variant": cfg["PMR"]["VARIANT"]})(
        mano.weights, mano.f, hidden=int(m["ENCODER_HIDDEN"]), n_layers=int(m["ENCODER_LAYERS"]),
        rel_hidden=int(m["XYZ_REL_HIDDEN"]), readout_hidden=int(m["XYZ_READOUT_HIDDEN"]))
    sd = torch.load(s["checkpoint"], map_location="cpu")["state_dict"]
    own = {k[len("xyz."):]: v for k, v in sd.items() if k.startswith("xyz.")}
    missing = net.load_state_dict(own, strict=True)
    return net.eval(), cfg


def analyze(arm, seed):
    import torch
    d = OUT / "mech" / A.run_name(arm, seed)
    require(not (d / "report.json").exists(), "written once")
    z = np.load(d / "capture.npz", allow_pickle=False)
    net, cfg = load_trunk(arm, seed)
    xp, xh, J = (torch.from_numpy(z[k]).double() for k in ("x_part", "x_hub", "J"))
    net = net.double()
    n = len(z["seq"])
    rng = np.random.default_rng(0)
    donor = np.zeros(n, np.int64)
    for i in range(n):
        ok = np.nonzero((z["seq"] == z["seq"][i]) & (np.abs(z["end_ms"] - z["end_ms"][i]) >= DONOR_MIN_MS))[0]
        donor[i] = rng.choice(ok)
    dn = torch.from_numpy(donor)

    def f(x_part, x_hub, J_):
        with torch.no_grad():
            h0 = net.encode(x_part, x_hub)
            if net.variant == "mlp":
                return net.fc2(torch.relu(net.fc1(h0.reshape(h0.shape[0], -1))))
            use = net.variant == "r"
            return net.readout(net.propagate(h0, net.gates(J_) if use else None, messages=use))
    base = f(xp, xh, J)
    gt = torch.from_numpy(z["target"] - z["prior"]).double()
    rep = {"arm": arm, "seed": seed, "n": n, "variant": net.variant,
           "check_capture_delta_max_abs_diff": float((base.float() - torch.from_numpy(z["delta"])).abs().max())}
    per = {}
    for j in range(1, 16):
        sl = slice(6 + 3 * (j - 1), 9 + 3 * (j - 1))
        xo, ho = xp[dn].clone(), xh[dn].clone()
        xo[:, j] = xp[:, j]
        others = f(xo, ho, J)
        xs = xp.clone()
        xs[:, j] = xp[dn, j]
        own = f(xs, xh, J)
        e = {k: (v[:, sl] - gt[:, sl]).norm(dim=-1) for k, v in (("intact", base), ("others", others), ("own", own))}
        ch_o, ch_s = (others[:, sl] - base[:, sl]).norm(dim=-1), (own[:, sl] - base[:, sl]).norm(dim=-1)
        per[j] = {"e_intact": float(e["intact"].mean()), "e_others": float(e["others"].mean()),
                  "e_own": float(e["own"].mean()), "reliance": float(e["others"].mean() / e["intact"].mean()),
                  "share_change_others": float((ch_o / (ch_o + ch_s).clamp_min(1e-12)).mean())}
    rep["joints"] = per
    rep["groups"] = {g: {k: float(np.mean([per[j][k] for j in js])) for k in per[1]} for g, js in GROUPS.items()}
    xo = xp[dn].clone()
    root = f(xo, xh, J)
    er = {k: (v[:, 3:6] - gt[:, 3:6]).norm(dim=-1).mean() for k, v in (("intact", base), ("parts_swapped", root))}
    et = {k: (v[:, 0:3] - gt[:, 0:3]).norm(dim=-1).mean() * 1000 for k, v in (("intact", base), ("parts_swapped", root))}
    rep["root_rad"] = {k: float(v) for k, v in er.items()}
    rep["transl_mm"] = {k: float(v) for k, v in et.items()}
    if net.variant == "r":
        with torch.no_grad():
            g = net.gates(J)
        from semkine.pmr_trunk import EDGE_TYPES
        rep["gates"] = {t: {"mean": float(g[:, net.e_type == k].mean()), "std_over_edges_and_packets":
                            float(g[:, net.e_type == k].std())} for k, t in enumerate(EDGE_TYPES)}
    rep["written_utc"] = now()
    save(d / "report.json", rep, exclusive=True)
    L = [f"# PMR mechanism (d): {A.run_name(arm, seed)} (zgz E3, zero training; per-joint counterfactual, donor >= 1 s)", "",
         "| joints | e intact (rad) | e others swapped | e own swapped | reliance = e_others / e_intact | share of change from others |",
         "|---|---|---|---|---|---|"]
    for gname, v in rep["groups"].items():
        L.append(f"| {gname} | {v['e_intact']:.4f} | {v['e_others']:.4f} | {v['e_own']:.4f} | {v['reliance']:.3f} | "
                 f"{v['share_change_others']:.3f} |")
    L += ["", f"root (rad): intact {rep['root_rad']['intact']:.4f}, parts swapped {rep['root_rad']['parts_swapped']:.4f}; "
          f"translation (mm): intact {rep['transl_mm']['intact']:.2f}, parts swapped {rep['transl_mm']['parts_swapped']:.2f}",
          f"capture check (replayed increment vs engine, max abs diff): {rep['check_capture_delta_max_abs_diff']:.2e}"]
    if "gates" in rep:
        L.append("gates: " + "; ".join(f"{t} {v['mean']:.3f} ± {v['std_over_edges_and_packets']:.3f}"
                                       for t, v in rep["gates"].items()))
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("capture", "analyze"):
        s = sub.add_parser(verb)
        s.add_argument("--arm", required=True)
        s.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args(argv)
    A.parse(a.arm)
    return capture(a.arm, a.seed) if a.verb == "capture" else analyze(a.arm, a.seed)


if __name__ == "__main__":
    main()
