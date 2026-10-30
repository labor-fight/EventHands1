"""C1SF (docs/EXPERIMENT_RECORDS.md#c1sf-prereg-20261029): arm ZR changes only the reference of the relation
encoder's depth channel; with a constant 0.45 m reference it is C1 bit for bit; the constructor routing keeps C1's
state dict and random stream; every arm config is the canonical C1 recipe with exactly its one registered change."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import c1sf_zrel as Z  # noqa: E402
from semkine import xyz_mesh as X  # noqa: E402
from semkine.xyz_mesh import associate_rays, vertex_face_table  # noqa: E402
from tools.c1sf import common as C  # noqa: E402

NPZ = ROOT / "assets" / "mano_right.npz"


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(old)


def scene(B=2, N=192, seed=0):
    with np.load(NPZ, allow_pickle=False) as z:
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        lbs = torch.tensor(np.asarray(z["weights"], np.float32))
        v = torch.tensor(np.asarray(z["v_template"], np.float32))
    g = torch.Generator().manual_seed(seed)
    verts = torch.stack([v + torch.tensor([0.0, 0.0, 0.5 + 0.2 * b]) + 0.01 * torch.randn(1, 3, generator=g)
                         for b in range(B)])
    pick = torch.randint(0, 778, (B, N), generator=g)
    target = verts[torch.arange(B)[:, None], pick] + 0.002 * torch.randn(B, N, 3, generator=g)
    d = target / target[..., 2:3]
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -16:] = False
    assoc = associate_rays(d, mask, verts, faces, vertex_face_table(faces, 778), torch.full((B,), 16 / 300.0),
                           torch.full((B,), 3 / 300.0), 0.01)
    tok = torch.randn(B, N, 7, generator=g)
    prev48 = torch.randn(B, 48, generator=g) / 10
    return faces, lbs, (tok, d, mask, verts, assoc, prev48)


def pair():
    faces, lbs, x = scene()
    Z.ZRelXYZMeshNet.WRIST = Z.wrist_weights(NPZ)
    torch.manual_seed(3407)
    a = X.XYZMeshNet(lbs, faces)
    torch.manual_seed(3407)
    b = Z.ZRelXYZMeshNet(lbs, faces)
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        w = torch.randn(a.fc2.weight.shape, generator=g) * .02
        a.fc2.weight.copy_(w)
        b.fc2.weight.copy_(w)
    return a.eval(), b.eval(), x


def captured_features(net, x):
    seen = []
    handle = net.rel[0].register_forward_pre_hook(lambda m, inp: seen.append(inp[0].detach().clone()))
    with torch.no_grad():
        out = net(*x)[0]
    handle.remove()
    return out, seen[0]


def test_state_dict_and_parameters_are_c1s():
    a, b, _ = pair()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_constant_reference_is_c1_bit_for_bit(monkeypatch):
    a, b, x = pair()
    monkeypatch.setattr(Z.ZRelXYZMeshNet, "root_depth", lambda self, verts: torch.full((verts.shape[0],), 0.45))
    oa, fa = captured_features(a, x)
    ob, fb = captured_features(b, x)
    assert torch.equal(oa, ob) and torch.equal(fa, fb)
    assert X.Z_REF == 0.45


def test_zn_is_relative_to_previous_root_depth_and_nothing_else_moves():
    a, b, x = pair()
    _, fa = captured_features(a, x)
    _, fb = captured_features(b, x)
    tok, d, mask, verts, assoc, prev48 = x
    rb = Z.record_rows(assoc, mask)
    root = b.root_depth(verts)
    zrec = (fa[:, 10] * X.Z_SCALE + 0.45)                      # C1's zn decoded back to metres
    assert torch.allclose(fb[:, 10], (zrec - root[rb]) / X.Z_SCALE, atol=1e-5)
    keep = [i for i in range(fa.shape[1]) if i != 10]
    assert torch.equal(fa[:, keep], fb[:, keep])
    assert float(fa[:, 10].abs().mean()) > 0.5 > float(fb[:, 10].abs().mean())   # 0.5 / 0.7 m absolute vs in-hand relative
    assert X.Z_REF == 0.45


def test_reference_restored_after_failure():
    a, b, x = pair()
    tok, d, mask, verts, assoc, prev48 = x
    with pytest.raises(Exception):
        b(tok[:, :5], d, mask, verts, assoc, prev48)
    assert X.Z_REF == 0.45


def test_ablate_and_oracle_geometry():
    a, b, x = pair()
    with torch.no_grad():
        assert torch.equal(a(*x, ablate=True)[0], b(*x, ablate=True)[0])
    with pytest.raises(ValueError):
        b(*x, verts_geo=x[3])


def test_wrist_row_tracks_fk_wrist_depth():
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    mano = ManoLayer(str(NPZ), add_mean=False).eval()
    rng = np.random.default_rng(0)
    params = np.zeros((64, 51), np.float32)
    params[:, :3] = rng.normal(0, 0.05, (64, 3)) + [0, 0, 0.6]
    params[:, 3:6] = rng.normal(0, 1.0, (64, 3))
    params[:, 6:] = rng.normal(0, 0.3, (64, 45))
    beta = torch.zeros(1, 10)
    with torch.no_grad():
        joints, verts = ET.mano_fk(mano, params, beta, torch.device("cpu"))
    w = Z.wrist_weights(NPZ).numpy()
    err = np.abs(verts[..., 2] @ w - joints[:, 0, 2])
    assert err.max() < 0.005, err.max()


def test_arm_configs_are_c1_plus_one_registered_change():
    canonical = yaml.safe_load(C.CANONICAL.read_text())
    for arm in C.ARMS:
        for phase in ("preflight", "formal"):
            if phase == "formal" and arm not in C.FORMAL_ARMS:
                continue
            cfg = yaml.safe_load(C.config_path(arm, phase).read_text())
            assert cfg == C.expected_raw(arm, phase, canonical)
            def flat(node, prefix=""):
                if not isinstance(node, dict):
                    return {prefix[:-1]: node}
                return {k: v for key, value in node.items() for k, v in flat(value, prefix + key + ".").items()}
            fa, fb = flat(canonical), flat(cfg)
            diff = {k: fb.get(k) for k in set(fa) | set(fb) if fa.get(k) != fb.get(k)}
            bookkeeping = {"C1SF.ARM", "TRAIN.RUN_NAME", "TRAIN.OUTPUT_DIR", "EVAL.OUTPUT_DIR",
                           "TRAIN.NUM_WORKERS", "TRAIN.PREFETCH_FACTOR"}  # loader resources (amendment 1)
            assert {k: cfg["TRAIN"][k] for k in C.RESOURCES} == C.RESOURCES
            if phase == "preflight":
                bookkeeping |= {"TRAIN.MAX_STEPS", "TRAIN.VAL_CHECK_INTERVAL"}
            science = {k: v for k, v in diff.items() if k not in bookkeeping}
            assert science == C.ARMS[arm], (arm, phase, science)


ROUTING = r"""
import copy, sys, torch
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/model"]
from config import load_config
from model import MNISTModel
base = load_config(sys.argv[1] + "/configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml")
torch.manual_seed(3407); a = MNISTModel(copy.deepcopy(base)); ra = torch.rand(8)
from semkine import c1sf_zrel
c1sf_zrel.install(base["MANO"]["NPZ"])
torch.manual_seed(3407); b = MNISTModel(copy.deepcopy(base)); rb = torch.rand(8)
assert type(a.xyz).__name__ == "XYZMeshNet" and type(b.xyz) is c1sf_zrel.ZRelXYZMeshNet
assert torch.equal(ra, rb), "random stream after construction is C1's"
sa, sb = a.state_dict(), b.state_dict()
assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
cfg = copy.deepcopy(base); cfg["LOSS"].update(TYPE="so3_trans_fk", ROT_WEIGHT=1.0, TRANS_WEIGHT=1.0, FK_WEIGHT=2.0)
torch.manual_seed(3407); c = MNISTModel(cfg)
y = torch.zeros(4, 51); y[:, 2] = 0.6; y[:, 3:] = 0.1 * torch.randn(4, 48)
loss, parts = c._compute_loss(y + 0.01 * torch.randn(4, 51), y, torch.zeros(4, 10))
assert torch.isfinite(loss) and float(parts["loss_fk"]) > 0
print("ROUTING OK")
"""


def test_model_constructor_routing_and_so3_loss_on_xyz(tmp_path):
    script = tmp_path / "routing.py"
    script.write_text(ROUTING)
    r = subprocess.run([sys.executable, str(script), str(ROOT)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "ROUTING OK" in r.stdout, r.stdout + r.stderr
