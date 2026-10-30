"""S45-PX (docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030): both arms add only the associated previous-mesh vertex
position to every matched record, through a zero-initialised map; the initial function, every S45 tensor and the
random stream after construction are S45's; the features are the registered ones; the model constructor routes to
the arm's class; every arm config is the S45 recipe with exactly its one registered change."""
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

from semkine import s45px as P  # noqa: E402
from semkine import xyz_mesh as X  # noqa: E402
from semkine.xyz_mesh import associate_rays, vertex_face_table  # noqa: E402

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


def pair(frame, x=None):
    faces, lbs, scene_x = scene()
    P.PXXYZMeshNet.FRAME, P.PXXYZMeshNet.WRIST = frame, P.wrist_weights(NPZ)
    torch.manual_seed(3407)
    a = X.XYZMeshNet(lbs, faces)
    ra = torch.rand(8)
    torch.manual_seed(3407)
    b = P.PXXYZMeshNet(lbs, faces)
    rb = torch.rand(8)
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        w = torch.randn(a.fc2.weight.shape, generator=g) * .02
        a.fc2.weight.copy_(w)
        b.fc2.weight.copy_(w)
    return a.eval(), b.eval(), (scene_x if x is None else x), (ra, rb)


def px_inputs(net, x):
    seen = []
    handle = net.sign_in.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].detach().clone()))
    with torch.no_grad():
        out = net(*x)[0]
    handle.remove()
    return out, seen[0]


def records(x):
    tok, d, mask, verts, assoc, prev48 = x
    w = assoc["q"].unsqueeze(-1) * assoc["A"] * mask.to(torch.float32).unsqueeze(-1)
    rb, rn, rr = (w > 0).nonzero(as_tuple=True)
    return rb, assoc["vid"][rb, rn, rr]


@pytest.mark.parametrize("frame", P.FRAMES)
def test_initial_function_tensors_and_random_stream_are_s45s(frame):
    a, b, x, (ra, rb) = pair(frame)
    assert torch.equal(ra, rb), "random stream after construction is S45's"
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"sign_in.weight"} and set(sa) <= set(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert sb["sign_in.weight"].shape == (64, 3) and not sb["sign_in.weight"].any()
    with torch.no_grad():
        oa, ob = a(*x)[0], b(*x)[0]
    assert torch.equal(oa, ob)
    assert sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters()) == 192


def test_features_are_the_registered_vertex_positions():
    _, cam, x, _ = pair("camera")
    _, mano, _, _ = pair("mano")
    tok, d, mask, verts, assoc, prev48 = x
    rb, vrec = records(x)
    v = verts[rb, vrec]
    wrist = torch.einsum("v,bvc->bc", P.wrist_weights(NPZ), verts)
    _, fc = px_inputs(cam, x)
    _, fm = px_inputs(mano, x)
    assert fc.shape == fm.shape == (rb.numel(), 3) and rb.numel() > 100
    assert torch.allclose(fc, (v - torch.tensor([0.0, 0.0, 0.45])) / 0.1, atol=1e-5)
    assert torch.allclose(fm, (v - wrist[rb]) / 0.1, atol=1e-5)
    # the camera vertex is already a function of the record (X = z d, V = X - L_REL dXYZ): only the frame is new
    assert float(fm.abs().max()) < 3.0 and float(fc[:, 2].abs().mean()) > 0.5


def test_mano_frame_is_translation_invariant_and_camera_frame_is_not():
    shift = torch.tensor([0.03, -0.02, 0.05])
    _, cam, x, _ = pair("camera")
    _, mano, _, _ = pair("mano")
    verts = x[3]
    rb, vrec = records(x)
    m0, m1 = mano.sign_features(verts, rb, vrec, None, None), mano.sign_features(verts + shift, rb, vrec, None, None)
    c0, c1 = cam.sign_features(verts, rb, vrec, None, None), cam.sign_features(verts + shift, rb, vrec, None, None)
    assert torch.allclose(m0, m1, atol=1e-5)
    assert torch.allclose(c1 - c0, (shift / 0.1).expand_as(c0), atol=1e-5)


@pytest.mark.parametrize("frame", P.FRAMES)
def test_new_map_receives_gradient_and_changes_the_function(frame):
    _, b, x, _ = pair(frame)
    b.train()
    out = b(*x)[0]
    (out ** 2).sum().backward()
    assert b.sign_in.weight.grad is not None and float(b.sign_in.weight.grad.abs().sum()) > 0
    b.eval()
    with torch.no_grad():
        before = b(*x)[0].clone()
        b.sign_in.weight.normal_(0, 0.1)
        after = b(*x)[0]
    assert not torch.equal(before, after)


def test_ablate_and_oracle_geometry():
    a, b, x, _ = pair("mano")
    with torch.no_grad():
        assert torch.equal(a(*x, ablate=True)[0], b(*x, ablate=True)[0])
    with pytest.raises(ValueError):
        b(*x, verts_geo=x[3])


def test_install_requires_registered_frame():
    with pytest.raises(ValueError):
        P.install("world", NPZ)


def test_arm_configs_are_s45_plus_one_registered_change():
    from tools.s45px import common as C
    s45 = yaml.safe_load(C.S45_CONFIG.read_text())
    for arm in C.ARMS:
        for phase in ("preflight", "formal"):
            cfg = yaml.safe_load(C.config_path(arm, phase).read_text())
            assert cfg == C.expected_raw(arm, phase, s45)
            assert cfg["S45PX"] == {"ARM": arm, "FRAME": C.ARMS[arm]}
            base = {k: v for k, v in s45.items() if k != "C1SF"}
            assert {k: v for k, v in cfg.items() if k not in ("S45PX", "TRAIN", "EVAL")} == \
                {k: v for k, v in base.items() if k not in ("TRAIN", "EVAL")}
            changed = {k for k in cfg["TRAIN"] if cfg["TRAIN"][k] != base["TRAIN"].get(k)}
            allowed = {"RUN_NAME", "OUTPUT_DIR"} | ({"MAX_STEPS", "VAL_CHECK_INTERVAL"} if phase == "preflight" else set())
            assert changed <= allowed, (arm, phase, changed)


ROUTING = r"""
import copy, sys, torch
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/model"]
from config import load_config
from model import MNISTModel
base = load_config(sys.argv[1] + "/configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml")
torch.manual_seed(3407); a = MNISTModel(copy.deepcopy(base)); ra = torch.rand(8)
from semkine import s45px
s45px.install(sys.argv[2], base["MANO"]["NPZ"])
torch.manual_seed(3407); b = MNISTModel(copy.deepcopy(base)); rb = torch.rand(8)
assert type(a.xyz).__name__ == "XYZMeshNet" and type(b.xyz) is s45px.PXXYZMeshNet and b.xyz.px_frame == sys.argv[2]
assert s45px.installed() == sys.argv[2]
assert torch.equal(ra, rb), "random stream after construction is S45's"
sa, sb = a.state_dict(), b.state_dict()
assert set(sb) - set(sa) == {"xyz.sign_in.weight"} and all(torch.equal(sa[k], sb[k]) for k in sa)
assert not sb["xyz.sign_in.weight"].any()
print("ROUTING OK")
"""


@pytest.mark.parametrize("frame", P.FRAMES)
def test_model_constructor_routing(tmp_path, frame):
    script = tmp_path / "routing.py"
    script.write_text(ROUTING)
    r = subprocess.run([sys.executable, str(script), str(ROOT), frame], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "ROUTING OK" in r.stdout, r.stdout + r.stderr
