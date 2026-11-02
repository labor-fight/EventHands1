"""T1A-UB round-2 CPU tests (docs/EXPERIMENT_RECORDS.md#t1aub-r2-prereg-20261102): state-arbitrated routing through the
round-2 install path. Each case runs in a fresh interpreter (the hooks patch module attributes)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

PRELUDE = r'''
import sys, json
sys.path[:0] = [%r, %r]
import numpy as np, torch
import pytorch_lightning as pl
from pathlib import Path
from semkine import xag as XAG
from tools.t1aub import r2, common as C

def real_packet(cfg, model):
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    seq, d = next((s, d) for s, d in sequences_for_split(root, "train", splits_manifest(cfg)) if s == "lyq_local")
    events, offsets, aux, gt = ET.load_sequence(root, d, seq)
    tsub = np.load(root / d / f"{seq}_tsub.npy", mmap_mode="r") if (root / d / f"{seq}_tsub.npy").exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3)
    model.set_hand_context(betas, K)
    a0, _ = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)[0]
    end = int(a0 + 49 + 50 * 40)
    prev = torch.from_numpy(np.asarray(gt[end - 50], np.float32).copy()).view(1, -1)
    return ET.make_eval_packet(ET._window_events(events, offsets, tsub, end, 50), prev, betas, K, 50, torch.device("cpu"))

def run_cell(cfg):
    from model import MNISTModel
    captured = {}
    pl.seed_everything(3407)
    m = MNISTModel(cfg).eval()
    native = XAG.route_association
    def cap(*a, **k):
        out = native(*a, **k)
        captured["assoc"] = {kk: v.clone() for kk, v in out[0].items()}
        return out
    XAG.route_association = cap
    with torch.no_grad():
        o = m.forward_packet(real_packet(cfg, m))
    XAG.route_association = native
    return o, captured["assoc"]
''' % (str(ROOT), str(ROOT / "model"))


def run(body, timeout=900):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", PRELUDE + body], capture_output=True, text=True, timeout=timeout, env=env,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_part_first_cell_equals_part_first_decoding():
    out = run(r'''
from tools.s45sm.routing import NEEDS, register_kinds
from semkine import s45sm as SM
cfg = r2.expected_cell("b_pf"); cfg["_config_path"] = "test"
st = r2.install_for_config(cfg, evaluation=True)
o1, a1 = run_cell(cfg)
from semkine import t1aub_arb as A, t1aub as M, xyz_mesh as XM, s45tr_t1a as T1A
A.uninstall(); SM.uninstall(); M.uninstall(); XM.XYZMeshNet = T1A._ORIGINAL
# reference: the same trunk config routed by the part-first decoding kind (semkine/s45sm_hier.py), no arbitration
ref = r2.expected_cell("b_pf"); ref["_config_path"] = "test"
ref["T1AUB"].pop("ARB"); ref["T1AUB"]["ROUTE"]["ROUTER_KIND"] = "dwh"
C.install_for_config(ref, evaluation=True)
o2, a2 = run_cell(ref)
print(json.dumps({"arb": st["arb"], "valid": int(a1["valid"].sum()), "assoc_equal": all(torch.equal(a1[k], a2[k]) for k in a1),
                  "out_equal": bool(torch.equal(o1, o2))}))
''')
    assert out["arb"] == {"mode": "point", "sigma_px": None, "band_px": 16.0}
    assert out["valid"] > 100 and out["assoc_equal"] and out["out_equal"]


def test_arbitrated_cells_run_and_switch():
    out = run(r'''
from semkine import t1aub_arb as A, s45sm as SM, t1aub as M, xyz_mesh as XM, s45tr_t1a as T1A
res = {}
for cell in ("b_pt2", "b_ray", "a_ray"):
    cfg = r2.expected_cell(cell); cfg["_config_path"] = "test"
    st = r2.install_for_config(cfg, evaluation=True)
    o, a = run_cell(cfg)
    res[cell] = {"finite": bool(torch.isfinite(o).all()), "valid": int(a["valid"].sum()), **A.last(), "arb": st["arb"]}
    A.uninstall(); SM.uninstall(); M.uninstall(); XM.XYZMeshNet = T1A._ORIGINAL
print(json.dumps(res))
''')
    for cell in ("b_pt2", "b_ray", "a_ray"):
        assert out[cell]["finite"] and out[cell]["valid"] > 100
    assert out["b_ray"]["switched"] > 0.0 and out["b_ray"]["arb"]["mode"] == "ray"
    assert out["b_pt2"]["arb"]["sigma_px"] == 2.0


def test_round2_configs_on_disk_equal_generator():
    out = run(r'''
print(json.dumps(r2.check_configs(2)))
''')
    assert out and all(out.values()), [k for k, v in out.items() if not v]
