#!/usr/bin/env python3
"""S44 P0.3: how much of the frozen tracker's response to a state error goes through the render channel
(docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019 section 2).

At GT(end - 50) of every 2nd zgz step (W200 LNES clipped to the segment start, as the S39 stack), a state error delta
is injected either into the render input only (trunk(LNES, render(p + delta)) - trunk(LNES, render(p))) or into the
prev_mlp input only (prev_mlp(p + delta) - prev_mlp(p)); the skip connection is untouched in both. delta: translation
5 mm, root 3 deg (random axis, additive axis-angle), every finger joint 0.1 rad (random axes). Reported per perturbed
block: mean norm of the output change in that block, their ratio render / prev_mlp, and the correction fractions
c = -<change, delta> / |delta|^2.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "dt"))
from s44 import evflow as EF  # noqa: E402
import filter_eval as FE  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from sparse_trunk import input_mask  # noqa: E402

EX = FE.EX
BLOCKS = {"transl": slice(0, 3), "root": slice(3, 6), "fingers": slice(6, 51)}


def trunk_out(m, lnes, state):
    """The network output before prev_mlp / skip (`forward` up to `out`), render of `state`."""
    betas_f, k_f = m._resolve_betas_K(state, None, None)
    rend = m._render_prev(state.float(), betas_f, k_f).to(dtype=lnes.dtype)
    x = torch.cat([lnes, rend], dim=-1).permute(0, 3, 1, 2).contiguous()
    if m.sparse_trunk:
        m0 = input_mask(x)
        feats, _ = m.rn.pooled(m.conv1(x) * m0, m0)
        return m.rn.fc(feats)
    return m.rn(m.conv1(x))


def pm_out(m, state):
    pm = m.prev_mlp(state)
    if not m.prev_mlp_transl:
        t = m.slices.transl
        pm = torch.cat([pm[:, : t.start], torch.zeros_like(pm[:, t]), pm[:, t.stop:]], dim=-1)
    return pm


def deltas(rng, n):
    out = {}
    d = np.zeros((n, 51))
    u = rng.standard_normal((n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    d[:, 0:3] = u * 0.005
    out["transl"] = d
    d = np.zeros((n, 51))
    u = rng.standard_normal((n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    d[:, 3:6] = u * np.deg2rad(3.0)
    out["root"] = d
    d = np.zeros((n, 51))
    u = rng.standard_normal((n, 15, 3)); u /= np.linalg.norm(u, axis=2, keepdims=True)
    d[:, 6:] = (u * 0.1).reshape(n, 45)
    out["fingers"] = d
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    res = {}
    for run in a.runs:
        cfg, m, ckpt, step, _ = FE.build_run(Path(run), "last", dev)
        ev_ch = EV.event_channels(cfg)
        acc = {b: {"render": [], "mlp": [], "c_render": [], "c_mlp": []} for b in BLOCKS}
        rng = np.random.default_rng(0)
        for s, d in EF.split_seqs("val_core"):
            events, offsets, aux, pos51 = ET.load_sequence(EF.DATA_ROOT, d, s)
            betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
            K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
            m.set_hand_context(betas, K)
            runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            items = []
            for ra, rb in runs:
                for e in np.arange(ra + EF.STEP - 1, rb, EF.STEP)[1::2]:
                    if e - EF.STEP >= ra:
                        items.append((int(ra), int(e)))
            for i0 in range(0, len(items), 64):
                chunk = items[i0:i0 + 64]
                lnes = torch.from_numpy(np.stack([ET.build_lnes(events, offsets, e, EX.window_of(offsets, e, "fixed", 200,
                                                                                                0, 300, floor=ra), ev_ch)
                                                  for ra, e in chunk])).to(dev)
                p = torch.from_numpy(np.stack([pos51[e - EF.STEP] for _, e in chunk])).float().to(dev)
                base_t, base_m = trunk_out(m, lnes, p), pm_out(m, p)
                for b, dd in deltas(rng, len(chunk)).items():
                    dlt = torch.from_numpy(dd).float().to(dev)
                    ch_r = trunk_out(m, lnes, p + dlt) - base_t
                    ch_m = pm_out(m, p + dlt) - base_m
                    sl = BLOCKS[b]
                    dn = (dlt[:, sl] ** 2).sum(1)
                    acc[b]["render"].append(ch_r[:, sl].norm(dim=1).cpu().numpy())
                    acc[b]["mlp"].append(ch_m[:, sl].norm(dim=1).cpu().numpy())
                    acc[b]["c_render"].append((-(ch_r[:, sl] * dlt[:, sl]).sum(1) / dn).cpu().numpy())
                    acc[b]["c_mlp"].append((-(ch_m[:, sl] * dlt[:, sl]).sum(1) / dn).cpu().numpy())
        ent = {}
        for b, v in acc.items():
            r_, m_ = np.concatenate(v["render"]), np.concatenate(v["mlp"])
            ent[b] = {"n": int(len(r_)), "mean_change_render": float(r_.mean()), "mean_change_mlp": float(m_.mean()),
                      "ratio_render_over_mlp": float(r_.mean() / max(m_.mean(), 1e-12)),
                      "correction_frac_render": float(np.concatenate(v["c_render"]).mean()),
                      "correction_frac_mlp": float(np.concatenate(v["c_mlp"]).mean())}
        res[Path(run).name] = {"ckpt": str(ckpt), "step": step, "blocks": ent}
        print(Path(run).name, json.dumps({b: round(e["ratio_render_over_mlp"], 3) for b, e in ent.items()}), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
