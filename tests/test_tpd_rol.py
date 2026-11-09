"""TPD stage ROL CPU tests (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105). Run with CUDA_VISIBLE_DEVICES="" in a
process of its own (the routing / trunk / geometry hooks are module patches).

- The arms' configs differ from B2's (PFO2 fgs, same seed and phase) only in the registered keys.
- `pfa_association_grad` equals semkine/s45sm.py `pfa_association` record for record and passes gradient to the vertices;
  `edge_geometry_grad` equals the native edge geometry and passes gradient.
- The rollout dataset: K consecutive 50 ms packets of one run, one augmentation realisation (same K, same domrand draw,
  same polarity flip), prior noise only on packet 0, targets = the transformed GT at the packet ends.
- On real packets (B2's checkpoint, r_s2d routes, gshare trunk): the ROL forward's delta equals B2's forward's delta bit
  for bit; with GRAD a two-packet rollout puts gradient on the previous state through the geometry, with the prior
  detached it does not; per-packet gradient checkpointing gives the same loss and gradients.
- The job list is well formed; the written rules on synthetic numbers.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import arms_rol as A  # noqa: E402

BASE_KEYS = {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "TRAIN.MAX_STEPS", "TRAIN.LR", "TRAIN.WARMUP_STEPS",
             "TRAIN.SAVE_EVERY_N_STEPS", "TRAIN.VAL_CHECK_INTERVAL", "MODEL.INIT_FROM", "ROL.K", "ROL.GRAD",
             "ROL.CHECKPOINT", "ROL.WEIGHTS", "ROL.PACKET_MS", "PFO.ARM", "PFO.BASE_ARM", "PFO.SECTION", "PFO.STUDY"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


@pytest.mark.parametrize("arm", tuple(A.ARMS))
@pytest.mark.parametrize("phase", A.PHASES)
@pytest.mark.parametrize("seed", (3407, 3408))
def test_configs_differ_from_b2_only_in_registered_keys(arm, phase, seed):
    from tools.pfo2 import arms as P2
    a, b = flat(A.build_config(arm, seed, phase)), flat(P2.build_config("fgs", seed, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff <= BASE_KEYS
    assert a["PFO.ROUTER"] == "r_s2d" and a["PFO.ROUTER_KIND"] == "pfr" and a["PFO.TRUNK"] == "gshare"
    assert a["ROL.K"] == A.ARMS[arm]["k"] and a["ROL.GRAD"] == A.ARMS[arm]["grad"]
    assert a["MODEL.INIT_FROM"].endswith(f"pfo2_fgs_s{seed}-step=6000.ckpt")
    assert a["TRAIN.LR_SCHEDULE"] == "cosine" and a["TRAIN.BATCH_SIZE_PER_GPU"] == 512 and a["TRAIN.DEVICES"] == 2


# ----------------------------------------------------------------------------------------------- geometry
def _mano():
    with np.load(ROOT / "assets/mano_right.npz") as z:
        return (torch.tensor(np.asarray(z["weights"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64)),
                torch.tensor(np.asarray(z["v_template"], np.float32)))


def test_pfa_association_grad_equals_native():
    from semkine import s45sm as SM
    from semkine import tpd_rol as R
    from semkine import xyz_mesh as XM
    w, f, v = _mano()
    g = torch.Generator().manual_seed(0)
    B, N = 3, 50
    verts = (v.unsqueeze(0) + 0.01 * torch.randn(B, *v.shape, generator=g)) + torch.tensor([0.0, 0.0, 0.45])
    mask = torch.rand(B, N, generator=g) > 0.2
    face = torch.randint(0, len(f), (B, N), generator=g)
    bary = torch.rand(B, N, 3, generator=g)
    valid = torch.rand(B, N, generator=g) > 0.3
    d = torch.cat([0.1 * torch.randn(B, N, 2, generator=g), torch.ones(B, N, 1)], -1)
    vf = XM.vertex_face_table(f, v.shape[0])
    empty = SM._NATIVE["associate_rays"](d, torch.zeros_like(mask), verts, f, vf, torch.full((B,), 0.03),
                                         torch.full((B,), 0.005), 0.01)
    kw = dict(source_part=torch.zeros_like(face), source_face=face, source_bary=bary, source_valid=valid)
    a, aux_a = SM.pfa_association("face_route", empty, d, mask, verts, f, vf, w, None, None, 0.01, **kw)
    vg = verts.clone().requires_grad_(True)
    b, aux_b = R.pfa_association_grad("face_route", empty, d, mask, vg, f, vf, w, None, None, 0.01, **kw)
    for key in ("vid", "A", "q", "z", "valid", "hit"):
        assert torch.equal(a[key], b[key].detach()), key
    for key in aux_a:
        assert torch.equal(aux_a[key], aux_b[key]), key
    b["z"].sum().backward()
    assert vg.grad is not None and vg.grad[..., 2].abs().sum() > 0 and vg.grad[..., :2].abs().sum() == 0


def test_edge_geometry_grad_equals_native():
    from semkine import pfo_trunk4 as T4
    from semkine import tpd_rol as R
    w, f, v = _mano()
    T4.install("gshare")
    try:
        from semkine import xyz_mesh as XM
        net = XM.XYZMeshNet(w, f)
    finally:
        T4.uninstall()
    verts = v.unsqueeze(0).repeat(2, 1, 1) + 0.001 * torch.randn(2, *v.shape, generator=torch.Generator().manual_seed(1))
    native = T4._NATIVE["cls"].edge_geometry(net, verts)
    vg = verts.clone().requires_grad_(True)
    mine = R.edge_geometry_grad(net, vg)
    assert torch.equal(native, mine.detach())
    mine.square().sum().backward()
    assert torch.isfinite(vg.grad).all() and vg.grad.abs().sum() > 0
    R.install_grad_geometry()
    try:
        assert T4._NATIVE["cls"].edge_geometry is R.edge_geometry_grad
        with pytest.raises(RuntimeError):
            R.install_grad_geometry()
    finally:
        R.uninstall_grad_geometry()
    assert T4._NATIVE["cls"].edge_geometry is not R.edge_geometry_grad


# ----------------------------------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def train_ds():
    from semkine.dataset import build_dataset
    cfg = A.build_config("rol3", 3407, "formal")
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return cfg, comps, build_dataset(cfg, "train", comps, train=True)


def test_rollout_dataset(train_ds):
    import copy
    from semkine import tpd_rol as R
    from semkine.dataset import build_dataset
    cfg, comps, _ = train_ds
    ds = R.upgrade(build_dataset(cfg, "train", comps, train=True), 3)
    assert len(ds) > 100000 and ds.k == 3 and ds.packet_ms == 50
    seen = 0
    for idx in np.random.default_rng(0).integers(0, len(ds), 6):
        item = ds[int(idx)]
        assert isinstance(item, R.RolloutItem) and len(item.packets) == 3
        si, ends, a, b = ds.ends(int(idx))
        assert ends[1] - ends[0] == 50 and ends[2] - ends[1] == 50 and ends[0] - 49 >= a and ends[2] < b
        dr = [p.meta["domrand"] for p in item.packets]
        assert dr[0] is dr[1] is dr[2]
        for i, p in enumerate(item.packets):
            assert p.t_end_us - p.t_start_us == 50000 and p.meta["end_idx"] == ends[i] and p.sequence_id == si
            if p.n_events:
                assert p.events[:, 2].min() >= 0 and p.events[:, 2].max() < 0.05
            assert np.array_equal(p.camera_K, item.packets[0].camera_K)
        # packet k >= 1 carries the (transformed) GT at the previous packet's end as its reference prior
        assert np.array_equal(item.packets[1].prev_state, item.packets[0].target)
        assert np.array_equal(item.packets[2].prev_state, item.packets[1].target)
        seen += 1
    assert seen == 6
    # without augmentation and noise, the states are the raw labels
    plain = copy.deepcopy(cfg)
    plain["AUG"]["DOMRAND"]["ENABLED"] = False
    plain["TRACK"].update(PREV_NOISE_T=0.0, PREV_NOISE_R=0.0, PREV_NOISE_POSE=0.0, PREV_NOISE_MODE="gaussian")
    dz = R.upgrade(build_dataset(plain, "train", comps, train=True), 2)
    item = dz[123]
    si, ends, a, b = dz.ends(123)
    h = dz._handle(si)
    assert np.array_equal(item.packets[0].prev_state, h.pos51[ends[0] - 49])
    assert np.array_equal(item.packets[0].target, h.pos51[ends[0]])
    assert np.array_equal(item.packets[1].target, h.pos51[ends[1]])
    batch = R.collate_any([dz[i] for i in (5, 6, 7)])
    assert isinstance(batch, R.RolloutBatch) and batch.batch_size == 3 and len(batch.packets) == 2
    plain_batch = R.collate_any([item.packets[0], item.packets[1]])
    assert plain_batch.batch_size == 2


# ----------------------------------------------------------------------------------------------- model
@pytest.fixture(scope="module")
def model_and_batch(train_ds):
    from semkine import pfo_router as PR
    from semkine import s45sm as SM
    from semkine import tpd_rol as R
    from semkine import pfo_trunk4 as T4
    cfg, comps, _ = train_ds
    from semkine.dataset import build_dataset
    p = cfg["PFO"]
    PR.register()
    SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    T4.install("gshare")
    R.install_grad_geometry()
    try:
        cls = R.rol_model_class()
        torch.manual_seed(0)
        model = cls(cfg)
        n = model.init_from_abs_checkpoint(cfg["MODEL"]["INIT_FROM"])
        ds = R.upgrade(build_dataset(cfg, "train", comps, train=True), 3)
        batch = R.collate_any([ds[i] for i in (11, 222222, 333333)])
        yield model.eval(), batch, n
    finally:
        R.uninstall_grad_geometry()
        T4.uninstall()
        SM.uninstall()
        PR.unregister()


def test_rol_forward_equals_b2(model_and_batch):
    from model import MNISTModel
    model, batch, n = model_and_batch
    assert n == len(model.state_dict())
    pk = batch.packets[0]
    with torch.no_grad():
        src = model._routes(pk, model.xyz_max_nodes)
        mine = model._xyz_mesh_forward(pk, pk.prev_state, src)
        base = MNISTModel._xyz_mesh_forward(model, pk, pk.prev_state)
        b2 = MNISTModel.forward_packet(model, pk)
        y = model._rol_step(pk, pk.prev_state, src)
    assert torch.equal(mine, base)
    assert torch.equal(y, b2)                       # float32 without autocast: the same sum


def _rollout_loss(model, batch, grad, ckpt):
    model.rol_grad, model.rol_ckpt = grad, ckpt
    x = batch.packets[0].prev_state
    total = 0.0
    prevs = []
    for pk in batch.packets[:2]:
        src = model._routes(pk, model.xyz_max_nodes)
        if ckpt:
            from torch.utils.checkpoint import checkpoint
            y = checkpoint(model._rol_step, pk, x, src, use_reentrant=False)
        else:
            y = model._rol_step(pk, x, src)
        loss, _ = model._compute_loss(y, pk.target, pk.betas)
        total = total + loss
        prevs.append(y)
        x = y if grad else y.detach()
    return total, prevs


def test_rollout_gradient_paths(model_and_batch):
    model, batch, _ = model_and_batch
    params = [p for p in model.parameters() if p.requires_grad]
    grads = {}
    for grad, ckpt in ((True, False), (True, True), (False, False)):
        model.zero_grad(set_to_none=True)
        loss, prevs = _rollout_loss(model, batch, grad, ckpt)
        prevs[0].retain_grad()
        loss.backward()
        grads[(grad, ckpt)] = (float(loss), [p.grad.clone() if p.grad is not None else None for p in params],
                               prevs[0].grad)
    (l1, g1, pg1), (l2, g2, _), (l3, g3, pg3) = grads[(True, False)], grads[(True, True)], grads[(False, False)]
    assert l1 == l2 == l3
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(g1, g2))
    # the second packet's loss reaches the first packet's output only with GRAD: detached, the first output's gradient
    # is its own loss's alone
    model.zero_grad(set_to_none=True)
    y0 = _rollout_loss(model, batch, False, False)[1][0]
    y0.retain_grad()
    model._compute_loss(y0, batch.packets[0].target, batch.packets[0].betas)[0].backward()
    assert torch.equal(pg3, y0.grad) and not torch.equal(pg1, y0.grad)
    assert any(a is not None and not torch.equal(a, b) for a, b in zip(g1, g3))


def test_training_step_keeps_gradient_after_start_check(model_and_batch, tmp_path):
    """The first-step start check runs the trunk under no_grad inside the training autocast region; the training forward
    after it must still reach the weights (autocast weight-cast cache)."""
    model, batch, _ = model_and_batch
    model.cfg["TRAIN"]["OUTPUT_DIR"] = str(tmp_path)
    model.rol_k, model.rol_w, model.rol_grad, model.rol_ckpt = 3, [1.0, 1.0, 1.0], True, True
    model._rol_checked = False
    model.zero_grad(set_to_none=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model.training_step(batch, 0)
    assert out.requires_grad
    rec = __import__("json").loads((tmp_path / "rol_start_check_rank0.json").read_text())
    assert rec["delta_bitwise_equal"] and rec["init_equal"] and rec["autocast"]
    out.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_geometry_carries_gradient(model_and_batch):
    """With the readout's prior input and the skip connection cut, the gradient on the prior is the geometry's."""
    model, batch, _ = model_and_batch
    pk = batch.packets[1]
    x = pk.prev_state.clone().requires_grad_(True)
    src = model._routes(pk, model.xyz_max_nodes)
    verts, _ = model._fk(x.float(), pk.betas)
    assert verts.requires_grad
    delta = model._xyz_mesh_forward(pk, x, src)
    g, = torch.autograd.grad(delta.square().sum(), x)
    assert torch.isfinite(g).all() and g[:, 0:6].abs().sum() > 0


# ----------------------------------------------------------------------------------------------- jobs and rules
def test_jobs_and_rules(tmp_path):
    from tools.tpd import analyze_rol as AN
    from tools.tpd import jobs_rol as J
    jobs = J.stage(J.SEED_SCREEN)
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys))
    assert all(set(j) <= {"key", "kind", "command", "deps", "note", "mem_gib", "pairs"} for j in jobs)
    b2 = {"ra": 11.47, "e3": 5.59, "R15": 0.45, "F15": 0.41}
    assert AN.screen({"ra": 11.10, "e3": 5.70, "R15": 0.45, "F15": 0.41}, b2)["verdict"] == "SCREEN_PASS"
    assert AN.screen({"ra": 11.30, "e3": 5.70, "R15": 0.39, "F15": 0.35}, b2)["verdict"] == "SCREEN_PASS"
    assert AN.screen({"ra": 11.30, "e3": 5.70, "R15": 0.39, "F15": 0.40}, b2)["verdict"] == "SCREEN_FAIL"
    assert AN.screen({"ra": 10.00, "e3": 5.95, "R15": 0.30, "F15": 0.30}, b2)["verdict"] == "SCREEN_FAIL"
    assert AN.base_for(11.47, 11.40) == "b2" and AN.base_for(11.47, 10.60) == "rol1"
    assert AN.gate_rol5({"step_s_median": 4.0}) and not AN.gate_rol5({"step_s_median": 9.0})
