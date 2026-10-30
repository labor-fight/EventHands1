"""XCL controls: source recall/depth, oracle support and paired initialization."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine.xag import route_association, sample_sources, validate_config
from semkine.xyz_mesh import L_REL, XYZMeshNet, associate_rays, vertex_face_table


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def scene():
    verts = torch.tensor([[[-.2, -.2, 1.], [.2, -.2, 1.], [0., .2, 1.],
                           [-.4, -.4, 2.], [.4, -.4, 2.], [0., .4, 2.]]])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    lbs = torch.zeros(6, 16)
    lbs[:3, 0], lbs[3:, 1] = 1, 1
    d = torch.tensor([[[0., 0., 1.], [5., 5., 1.], [0., 0., 1.], [0., 0., 1.]]])
    mask = torch.tensor([[True, True, True, False]])
    vf = vertex_face_table(faces, 6)
    band, front = torch.tensor([.5]), torch.tensor([.4])
    assoc = associate_rays(d, mask, verts, faces, vf, band, front, .01)
    source = dict(source_part=torch.tensor([[1, 1, -1, 1]]), source_face=torch.tensor([[1, 1, -1, 1]]),
                  source_bary=torch.tensor([[[.25, .25, .5]] * 4]),
                  source_valid=torch.tensor([[True, True, False, True]]),
                  source_depth=torch.tensor([[1.25, 1.5, 0., 2.]]))
    return (assoc, d, mask, verts, faces, vf, lbs, band, front, .01), source


def net(mode="legacy", cls=XYZMeshNet):
    args, _ = scene()
    torch.manual_seed(3407)
    return cls(args[6], args[4], xag_gnn=mode)


def inputs():
    args, _ = scene()
    g = torch.Generator().manual_seed(33)
    return (torch.randn(1, 4, 7, generator=g), args[1], args[2], args[3], args[0],
            torch.randn(1, 48, generator=g) / 10)


def activate_head(m):
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        m.fc2.weight.copy_(torch.randn(m.fc2.weight.shape, generator=g) * .02)
    return m


def test_full_oracle_replaces_all_six_fields_recalls_none_and_unknown_becomes_none():
    args, source = scene()
    old = {k: v.clone() for k, v in args[0].items()}
    out, audit = route_association("full_oracle", *args, **source)
    assert args[0]["hit"][0, 0] and not args[0]["valid"][0, 1]
    assert args[0]["valid"][0, 2], "unknown source deliberately starts as a valid baseline hit"
    assert audit["xag_recalled"].tolist() == [[False, True, False, False]]
    assert audit["xag_replaced"].tolist() == [[True, True, False, False]]
    assert out["vid"][0, :2].tolist() == [[3, 4, 5, 3]] * 2
    assert out["A"][0, :2].tolist() == [[.25, .25, .5, 0.]] * 2
    for name in ("q", "valid", "hit"):
        assert out[name].tolist() == [[1, 1, 0, 0]]
    assert torch.equal(out["z"], torch.tensor([[1.25, 1.5, 0., 0.]]))
    assert all(not out[k][:, 2:].any() for k in out)
    assert all(torch.equal(old[k], args[0][k]) for k in old)
    q, mask = out["q"], args[2].float()
    torch.testing.assert_close(q * out["A"].sum(-1) + (1 - q) * mask, mask, rtol=0, atol=0)


def test_full_oracle_depth_is_used_in_relation_including_normal_displacement():
    args, source = scene()
    assoc, _ = route_association("full_oracle", *args, **source)
    m = net().eval()
    captured = []
    hook = m.rel.register_forward_pre_hook(lambda _m, xs: captured.append(xs[0].detach().clone()))
    tok, d, mask, verts, _, prev = inputs()
    m(tok, d, mask, verts, assoc, prev)
    hook.remove()
    b, n, r = (assoc["q"].unsqueeze(-1) * assoc["A"] > 0).nonzero(as_tuple=True)
    expected = (source["source_depth"][b, n, None] * d[b, n] - verts[b, assoc["vid"][b, n, r]]) / L_REL
    assert torch.equal(captured[0][:, 7:10], expected)
    assert captured[0][:, 9].abs().min() > 0, "source depth exposes the missing normal component"


@pytest.mark.parametrize("bad", [None, float("nan"), 0., -1.])
def test_full_oracle_requires_real_positive_source_depth(bad):
    args, source = scene()
    source["source_depth"] = None if bad is None else torch.full((1, 4), bad)
    with pytest.raises((RuntimeError, ValueError), match="source_depth"):
        route_association("full_oracle", *args, **source)


def test_full_oracle_source_gather_keeps_exact_stride_and_requires_depth():
    batch = SimpleNamespace(events=torch.zeros(4, 5), xag_source_index=torch.arange(4),
                            xag_source_ms=torch.arange(4), xag_part=torch.zeros(4, dtype=torch.long),
                            xag_face=torch.zeros(4, dtype=torch.long), xag_bary=torch.ones(4, 3) / 3,
                            xag_valid=torch.ones(4, dtype=torch.bool), xag_depth=torch.tensor([.1, .2, .3, .4]))
    src, mask = torch.tensor([[0, 2, 3]]), torch.tensor([[True, True, False]])
    got = sample_sources(batch, src, mask, require_depth=True)
    assert torch.equal(got["xag_depth"], batch.xag_depth[src])
    assert got["xag_valid"].tolist() == [[True, True, False]]
    del batch.xag_depth
    assert "xag_depth" not in sample_sources(batch, src, mask)
    with pytest.raises(RuntimeError, match="xag_depth"):
        sample_sources(batch, src, mask, require_depth=True)


def test_default_net_matches_frozen_c1_bitwise():
    frozen = ROOT / "outputs/xag_paired/20261018/source_archive/files/semkine/xyz_mesh.py"
    if not frozen.exists():
        pytest.skip("external frozen C1 source evidence is absent")
    spec = importlib.util.spec_from_file_location("semkine._xcl_test_frozen_xyz", frozen)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    old = activate_head(net(cls=old_module.XYZMeshNet)).eval()
    old_rng = torch.rand(64)
    new = activate_head(net()).eval()
    assert torch.equal(old_rng, torch.rand(64))
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    with torch.no_grad():
        a, sa, aa = old(*inputs())
        b, sb, ab = new(*inputs())
    assert torch.equal(a, b)
    assert all(torch.equal(v, sb[k]) for k, v in sa.items())
    assert all(torch.equal(v, ab[k]) for k, v in aa.items())


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_oracle_modes_preserve_shared_initial_tensors_and_next_rng(mode):
    old = net()
    after_old = torch.rand(64)
    m = net(mode)
    assert torch.equal(after_old, torch.rand(64))
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in old.state_dict().items())
    assert m.state_dict().keys() - old.state_dict().keys() == {
        "xag_oracle_projection.weight", "xag_oracle_projection.bias"}
    other = net("oracle_observed" if mode == "oracle_dense" else "oracle_dense")
    assert all(torch.equal(v, other.xag_oracle_projection.state_dict()[k])
               for k, v in m.xag_oracle_projection.state_dict().items())


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_oracle_replaces_h3_uses_direct_quality_mask_and_preserves_residual(mode):
    old = net().eval()
    m = net(mode).eval()
    x = inputs()
    gt = x[3].clone() + torch.tensor([.01, -.02, .03])
    _, _, original = old(*x, capture=True)
    _, _, aux = m(*x, verts_gt=gt, capture=True)
    expected_s = torch.ones_like(aux["O"]) if mode == "oracle_dense" else aux["O"]
    expected_h = m.xag_oracle_projection((gt - x[3]) / L_REL) * expected_s[..., None]
    assert torch.equal(aux["S"], expected_s)
    assert torch.equal(aux["H3"], expected_h)
    assert torch.equal(aux["O"], original["O"])
    assert torch.equal(aux["bslot"], original["bslot"])
    assert not aux["O"][:, 3:].any()
    if mode == "oracle_observed":
        assert not aux["H3"][:, 3:].any(), "projection bias must also be masked"
    else:
        assert aux["H3"][:, 3:].abs().sum() > 0
    # A route with zero event quality never counts as a direct observation.
    zero_assoc = {k: v.clone() for k, v in x[4].items()}
    zero_assoc["q"].zero_()
    _, _, empty = m(*x[:4], zero_assoc, x[5], verts_gt=gt, capture=True)
    assert not empty["O"].any()
    if mode == "oracle_observed":
        assert not empty["H3"].any() and not empty["S"].any()


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_oracle_only_projection_slots_and_head_learn_every_trainable_parameter_used(mode):
    m = activate_head(net(mode)).train()
    x = inputs()
    gt = (x[3] + .01).requires_grad_()
    frozen = {k: v.detach().clone() for k, v in m.named_parameters() if not v.requires_grad}
    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = m(*x, verts_gt=gt)[0].square().sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in m.parameters() if p.requires_grad)
        assert m.xag_oracle_projection.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        optimizer.step()
    assert gt.grad is None
    assert all(torch.equal(v, dict(m.named_parameters())[k]) for k, v in frozen.items())
    assert all(k.startswith(("xag_oracle_projection.", "slot_proj.", "fc1.", "fc2."))
               for k, p in m.named_parameters() if p.requires_grad)


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_oracle_requires_finite_target_vertices(mode):
    m = net(mode)
    with pytest.raises(RuntimeError, match="verts_gt"):
        m(*inputs())
    bad = inputs()[3].clone()
    bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="verts_gt"):
        m(*inputs(), verts_gt=bad)


@pytest.mark.parametrize("cfg", [
    {"XAG_GNN": "oracle_dense", "XAG_ASSOC": "full_oracle"},
    {"XAG_GNN": "oracle_observed", "XAG_ASSOC": "face_route"},
    {"XAG_GNN": "oracle_dense", "XYZ_GEOM": False},
    {"XAG_ASSOC": "full_oracle", "XYZ_ASSOC_PREV": "target"},
])
def test_unregistered_or_incompatible_crossings_fail_closed(cfg):
    with pytest.raises(ValueError):
        validate_config(cfg, xyz_mesh=True)


def test_c0_remains_an_accepted_base_control():
    assert validate_config({"XAG_ASSOC": "base", "XAG_GNN": "legacy", "XYZ_GEOM": False},
                           xyz_mesh=True) == ("base", "legacy")


def _model(mode="legacy", assoc="base"):
    from config import load_config
    from model import MNISTModel
    cfg = load_config(str(ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"))
    cfg["MODEL"].update(XAG_GNN=mode, XAG_ASSOC=assoc)
    torch.manual_seed(3407)
    return MNISTModel(cfg=cfg)


def _batch():
    from semkine.events import EventPacketBatch
    prev = torch.zeros(1, 51)
    prev[:, 2] = .45
    return EventPacketBatch(
        events=torch.tensor([[0., 120., 90., .001, 1.], [0., 121., 90., .002, 0.]]),
        ptr=torch.tensor([0, 2]), sequence_id=torch.zeros(1, dtype=torch.long),
        t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
        delta_t_s=torch.tensor([.05]), is_sequence_start=torch.tensor([False]),
        is_sequence_end=torch.tensor([False]), target=torch.zeros(1, 51), prev_state=prev,
        betas=torch.zeros(1, 10),
        camera_K=torch.tensor([[[603.45, 0., 325.09], [0., 602.96, 242.10], [0., 0., 1.]]]))


def test_model_oracle_uses_end_target_override_without_changing_association():
    m = _model("oracle_dense").eval()
    m.xyz_keep_aux = True
    activate_head(m.xyz)
    batch = _batch()
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="ground truth"):
            m.forward_packet(batch)
        batch.target = batch.prev_state.clone()
        first = m.forward_packet(batch)
        old_assoc = {k: v.clone() for k, v in m.xyz_last["assoc"].items()}
        m.xyz_target_override = batch.target.clone()
        m.xyz_target_override[:, 2] += .03
        second = m.forward_packet(batch)
    assert not torch.equal(first, second)
    assert all(torch.equal(v, m.xyz_last["assoc"][k]) for k, v in old_assoc.items())
    expected, _ = m._fk(m.xyz_target_override, batch.betas)
    assert torch.equal(m.xyz_last["verts_gt"], expected)


def test_model_full_oracle_passes_sampled_sidecar_depth_and_excludes_synthetic_event():
    m, batch = _model(assoc="full_oracle").eval(), _batch()
    m.xyz_keep_aux = True
    batch.xag_source_index, batch.xag_source_ms = torch.tensor([5, -1]), torch.tensor([10, 10])
    batch.xag_part, batch.xag_face = torch.tensor([0, 0]), torch.tensor([0, 0])
    batch.xag_bary = torch.tensor([[.2, .3, .5], [.2, .3, .5]])
    batch.xag_valid, batch.xag_depth = torch.ones(2, dtype=torch.bool), torch.tensor([.4, 0.])
    with torch.no_grad():
        m.forward_packet(batch)
    assoc = m.xyz_last["assoc"]
    assert torch.equal(assoc["z"], torch.tensor([[.4, 0.]]))
    assert assoc["valid"].tolist() == [[True, False]]
    assert torch.equal(assoc["vid"][0, 0, :3], m.mano.f[0].long())
    assert m.xyz_last["r"].tolist() == [[0., 1.]]
    del batch.xag_depth
    with pytest.raises(RuntimeError, match="xag_depth"):
        m.forward_packet(batch)


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_oracle_gnn_never_reads_source_cache(mode):
    m, batch = _model(mode).eval(), _batch()
    activate_head(m.xyz)
    batch.target = batch.prev_state.clone()
    with torch.no_grad():
        original = m.forward_packet(batch)
        batch.xag_valid = torch.ones(2, dtype=torch.bool)
        batch.xag_face = torch.full((2,), 90000, dtype=torch.long)
        batch.xag_bary = torch.full((2, 3), float("nan"))
        batch.xag_depth = torch.full((2,), float("nan"))
        assert torch.equal(original, m.forward_packet(batch))


@pytest.mark.parametrize("mode", ["oracle_dense", "oracle_observed"])
def test_model_empty_rank_keeps_prev_and_connects_all_trainable_parameters(mode):
    m, batch = _model(mode).train(), _batch()
    batch.events, batch.ptr = torch.empty(0, 5), torch.tensor([0, 0])
    output = m.forward_packet(batch)
    assert torch.equal(output, batch.prev_state)
    output.sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in m.parameters() if p.requires_grad)
