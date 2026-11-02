"""T1A-UB CPU tests (docs/EXPERIMENT_RECORDS.md#t1aub-prereg-20261102). Each case runs in a fresh interpreter: the hooks
patch module attributes. Run with the training environment: `python -m pytest tests/test_t1aub.py -q`."""
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
import torch
import pytorch_lightning as pl
from config import load_config
from semkine import t1aub as M, s45tr_t1a as T1A, xyz_mesh as XM
from semkine.events import EventPacketBatch
from tools.t1aub import common as C

def packet(n_ev=4096, seed=0, xag=False, n_faces=1538):
    g = torch.Generator().manual_seed(seed)
    ev = torch.zeros(n_ev, 5)
    ev[:, 1] = (122 + 20 * torch.randn(n_ev, generator=g)).clamp(0, 239).round()
    ev[:, 2] = (91 + 20 * torch.randn(n_ev, generator=g)).clamp(0, 179).round()
    ev[:, 3] = torch.linspace(0, 0.05, n_ev)
    ev[:, 4] = (torch.rand(n_ev, generator=g) > 0.5).float()
    prev = torch.zeros(1, 51); prev[0, 2] = 0.45
    K = torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796], [0, 0, 1.0]]).view(1, 3, 3)
    b = EventPacketBatch(events=ev, ptr=torch.tensor([0, n_ev]), sequence_id=torch.zeros(1, dtype=torch.long),
                         t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
                         delta_t_s=torch.tensor([0.05]), is_sequence_start=torch.zeros(1, dtype=torch.bool),
                         is_sequence_end=torch.zeros(1, dtype=torch.bool), target=prev.clone(),
                         prev_state=prev, betas=torch.zeros(1, 10), camera_K=K, lnes=None)
    if xag:
        b.xag_source_index = torch.arange(n_ev)
        b.xag_source_ms = torch.zeros(n_ev, dtype=torch.long)
        b.xag_face = torch.randint(0, n_faces, (n_ev,), generator=g)
        b.xag_part = torch.zeros(n_ev, dtype=torch.long)
        w = torch.rand(n_ev, 3, generator=g); b.xag_bary = (w / w.sum(-1, keepdim=True)).float()
        b.xag_valid = torch.rand(n_ev, generator=g) > 0.3
    return b

def build(cfg):
    from model import MNISTModel
    pl.seed_everything(3407)
    return MNISTModel(cfg).eval()

def arm_cfg(name, phase="formal"):
    import yaml
    cfg = C.expected_raw(name, phase) if C.family(name) == "arm" else C.expected_cell(name)
    cfg["_config_path"] = "test"
    return cfg
''' % (str(ROOT), str(ROOT / "model"))


def run(body, timeout=900):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-c", PRELUDE + body], capture_output=True, text=True, timeout=timeout, env=env,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_noop_subclass_is_t1a_bitwise():
    out = run(r'''
t1a = load_config(str(C.T1A_CONFIG))
T1A.install(t1a["S45TR"]["T1A_GNN"]); ref = build(t1a)
M.install({"eg": True, "readout": None, "gtfeat": False, "mano_npz": t1a["MANO"]["NPZ"]}); new = build(t1a)
a, b = ref.state_dict(), new.state_dict()
for m in (ref, new):
    torch.manual_seed(1); m.xyz.fc2.weight.data.normal_(0, 0.01)
with torch.no_grad():
    o1, o2 = ref.forward_packet(packet()), new.forward_packet(packet())
    M.set_reparam(True); o3 = new.forward_packet(packet()); M.set_reparam(False)
print(json.dumps({"keys": list(a) == list(b), "equal": all(torch.equal(a[k], b[k]) for k in a),
                  "out": bool(torch.equal(o1, o2)), "reparam": float((o3 - o1).abs().max()),
                  "params": sum(p.numel() for p in ref.parameters()), "cls": type(new.xyz).__name__}))
''')
    assert out["keys"] and out["equal"] and out["out"] and out["cls"] == "T1aubXYZMeshNet"
    assert out["params"] == 494323
    assert out["reparam"] < 1e-6


def test_readout_widths_and_parameter_budget():
    out = run(r'''
res = {}
for arm in ("t0ps16", "t0pnoeg20", "t0p", "t0pnoeg"):
    cfg = arm_cfg(arm)
    if cfg["T1AUB"]["EG"]:
        T1A.install(cfg["T1AUB"]["T1A_GNN"])
    ro = cfg["T1AUB"].get("READOUT")
    M.install({"eg": cfg["T1AUB"]["EG"], "gtfeat": False, "mano_npz": cfg["MANO"]["NPZ"],
               "readout": {"slot_dim": ro["SLOT_DIM"], "hidden": ro["HIDDEN"]} if ro else None})
    m = build(cfg)
    res[arm] = {"params": sum(p.numel() for p in m.parameters()), "slot_out": m.xyz.slot_proj.out_channels,
                "fc2_zero": bool((m.xyz.fc2.weight == 0).all() and (m.xyz.fc2.bias == 0).all()),
                "state": {k: v.clone() for k, v in m.state_dict().items()}}
    M.uninstall(); XM.XYZMeshNet = T1A._ORIGINAL
same = lambda a, b, skip: all(torch.equal(res[a]["state"][k], res[b]["state"][k]) for k in res[a]["state"] if not k.startswith(skip))
print(json.dumps({**{k: {kk: v for kk, v in r.items() if kk != "state"} for k, r in res.items()},
                  "s16_rest_equal": same("t0ps16", "t0p", ("xyz.slot_proj", "xyz.fc1", "xyz.fc2")),
                  "s20_rest_equal": same("t0pnoeg20", "t0pnoeg", ("xyz.slot_proj", "xyz.fc1", "xyz.fc2"))}))
''')
    assert out["t0p"]["params"] == 494323 and out["t0pnoeg"]["params"] == 459059
    assert out["t0ps16"]["params"] == 354243 and out["t0pnoeg20"]["params"] == 353999
    assert out["t0ps16"]["slot_out"] == 17 * 16 and out["t0pnoeg20"]["slot_out"] == 17 * 20
    assert all(out[a]["fc2_zero"] for a in ("t0ps16", "t0pnoeg20", "t0p", "t0pnoeg"))
    assert out["s16_rest_equal"] and out["s20_rest_equal"]
    # in-budget candidates with the 137,291-parameter srdwkda router
    assert out["t0ps16"]["params"] + 137291 <= 494323 and out["t0pnoeg20"]["params"] + 137291 <= 494323


def test_gtfeat_feature_and_initial_function():
    out = run(r'''
t1a = load_config(str(C.T1A_CONFIG))
T1A.install(t1a["S45TR"]["T1A_GNN"]); ref = build(t1a); XM.XYZMeshNet = T1A._ORIGINAL
cfg = arm_cfg("gtf")
T1A.install(cfg["T1AUB"]["T1A_GNN"])
M.install({"eg": True, "gtfeat": True, "mano_npz": cfg["MANO"]["NPZ"], "readout": None})
new = build(cfg)
for m in (ref, new):
    torch.manual_seed(1); m.xyz.fc2.weight.data.normal_(0, 0.01)
b = packet(xag=True)
with torch.no_grad():
    o1, o2 = ref.forward_packet(packet()), new.forward_packet(b)
# the feature itself on a hand-made stash
mask = torch.ones(1, 3, dtype=torch.bool)
M._STATE["stash"] = {"face": torch.tensor([[0, 5, 7]]), "bary": torch.tensor([[[1., 0, 0], [0.2, 0.3, 0.5], [1, 0, 0]]]),
                     "valid": torch.tensor([[True, True, False]])}
f = M.gt_features(new.xyz, mask)
tm, fa = new.xyz.t1aub_tmpl, new.xyz.t1aub_faces
p0 = tm[fa[0, 0]] / 0.01
p1 = (0.2 * tm[fa[5, 0]] + 0.3 * tm[fa[5, 1]] + 0.5 * tm[fa[5, 2]]) / 0.01
w = new.xyz.rel[0].weight
print(json.dumps({"diff": float((o1 - o2).abs().max()), "rel_in": new.xyz.rel_in,
                  "zero_cols": bool((w[:, 96:100] == 0).all()), "f0": float((f[0, 0, :3] - p0).abs().max()),
                  "f1": float((f[0, 1, :3] - p1).abs().max()), "flags": f[0, :, 3].tolist(),
                  "invalid_zero": bool((f[0, 2] == 0).all()), "route": M.installed()["gtfeat_route"]}))
''')
    assert out["rel_in"] == 106 and out["zero_cols"] and out["route"]
    assert out["diff"] < 1e-5
    assert out["f0"] < 1e-4 and out["f1"] < 1e-4 and out["flags"] == [1.0, 1.0, 0.0] and out["invalid_zero"]


def test_routes_install_and_run_on_cpu():
    out = run(r'''
res = {}
for name in ("t0p", "t0ps16_srdwa"):
    cfg = arm_cfg(name)
    st = C.install_for_config(cfg, evaluation=True)
    m = build(cfg)
    with torch.no_grad():
        o = m.forward_packet(packet(2048))
    r = C.router_net()
    res[name] = {"finite": bool(torch.isfinite(o).all()), "router": st["route"]["router"],
                 "router_params": sum(p.numel() for p in r.parameters()), "model_params": sum(p.numel() for p in m.parameters()),
                 "cls": type(m.xyz).__name__, "eg": bool(m.xyz.t1aub_eg)}
    from semkine import s45sm as SM
    SM.uninstall(); M.uninstall(); XM.XYZMeshNet = T1A._ORIGINAL
print(json.dumps(res))
''')
    assert out["t0p"]["finite"] and out["t0p"]["router"] == "t0p" and out["t0p"]["router_params"] == 1606683
    c = out["t0ps16_srdwa"]
    assert c["finite"] and c["router"] == "dw" and c["router_params"] == 137291
    assert c["model_params"] + c["router_params"] == 491534 <= 494323


def test_cell_configs_drop_label_sources():
    out = run(r'''
cfg = C.expected_cell("gt_srdwa")
print(json.dumps({"src": cfg["DATA"].get("XAG_SOURCE_ROOT"), "route": cfg["T1AUB"]["ROUTE"]["ROUTER"],
                  "trunk": cfg["T1AUB"]["TRUNK"], "lr": cfg["TRAIN"].get("LR_SCHEDULE")}))
''')
    assert out["src"] is None and out["route"] == "srdwkda" and out["trunk"] == "gt" and out["lr"] is None


def test_configs_on_disk_equal_generator():
    out = run(r'''
print(json.dumps(C.check_configs(1)))
''')
    assert out and all(out.values()), [k for k, v in out.items() if not v]
