"""XAG scientific controls: routing conservation, capture / replay and the source-label isolation of the base model."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine.xag import face_parts, route_association, sample_sources, validate_config
from semkine.xyz_mesh import XYZMeshNet, associate_rays, stride_sample, vertex_face_table


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def scene():
    verts = torch.tensor([[[-.2, -.2, 1.], [.2, -.2, 1.], [0., .2, 1.],
                           [-.4, -.4, 2.], [.4, -.4, 2.], [0., .4, 2.]]])
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    lbs = torch.zeros(6, 16)
    lbs[:3, 0] = 1
    lbs[3:, 1] = 1
    d = torch.tensor([[[0., 0., 1.], [.25, 0., 1.], [5., 5., 1.], [0., 0., 1.]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    vf = vertex_face_table(faces, 6)
    band, front = torch.tensor([.5]), torch.tensor([.4])
    assoc = associate_rays(d, mask, verts, faces, vf, band, front, .01)
    source = dict(source_part=torch.tensor([[1, 1, 1, -1]]), source_face=torch.tensor([[1, 1, 1, -1]]),
                  source_bary=torch.tensor([[[.25, .25, .5]] * 4]),
                  source_valid=torch.tensor([[True, True, True, False]]))
    args = (assoc, d, mask, verts, faces, vf, lbs, band, front, .01)
    return args, source


@pytest.mark.parametrize("mode", ["part_route", "face_route"])
def test_only_route_changes_mass_is_preserved_and_unknown_none_fall_back(mode):
    args, source = scene()
    old = args[0]
    saved = {k: v.clone() for k, v in old.items()}
    new, audit = route_association(mode, *args, **source)
    assert old["hit"][0, 0] and not old["hit"][0, 1] and old["q"][0, 1] > 0
    assert not old["valid"][0, 2]
    assert torch.equal(audit["xag_replaced"], torch.tensor([[True, True, False, False]]))
    assert (new["vid"][0, :2][new["A"][0, :2] > 0] >= 3).all()
    for key in ("q", "z", "hit", "valid"):
        assert new[key] is old[key]
    for key in old:
        assert torch.equal(old[key], saved[key]), "intervention mutated the control"
        assert torch.equal(new[key][:, 2:], old[key][:, 2:])
    torch.testing.assert_close(new["A"].sum(-1) * new["q"], old["A"].sum(-1) * old["q"], rtol=0, atol=1e-7)
    # Back-face source deliberately does not supply its 2m depth to the model.
    assert new["z"][0, 0] == 1


def test_part_absent_in_original_candidate_set_falls_back():
    args, source = scene()
    source["source_part"][:] = 3
    new, audit = route_association("part_route", *args, **source)
    assert not audit["xag_replaced"].any()
    assert all(torch.equal(new[k], args[0][k]) for k in new)


def test_face_unknown_and_bad_bary_fall_back():
    args, source = scene()
    source["source_face"][0, 0] = 999
    source["source_bary"][0, 1, 0] = float("nan")
    new, audit = route_association("face_route", *args, **source)
    assert not audit["xag_replaced"].any()
    assert all(torch.equal(new[k], args[0][k]) for k in new)


def test_base_is_identity_not_a_clone():
    args, source = scene()
    new, _ = route_association("base", *args, **source)
    assert new is args[0]


def test_part_route_with_original_part_is_identity_in_matched_scene():
    args, source = scene()
    source["source_part"][:, :2] = 0
    new, audit = route_association("part_route", *args, **source)
    assert audit["xag_replaced"][:, :2].all()
    assert all(torch.equal(new[k], args[0][k]) for k in new)


def test_face_route_keeps_old_depth_in_relation_features():
    args, source = scene()
    assoc, _ = route_association("face_route", *args, **source)
    m = net().eval()
    captured = []
    handle = m.rel.register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach().clone()))
    tok, d, mask, verts, _, prev = inputs()
    m(tok, d, mask, verts, assoc, prev)
    handle.remove()
    rb, rn, rr = (assoc["q"].unsqueeze(-1) * assoc["A"] > 0).nonzero(as_tuple=True)
    expected = (assoc["z"][rb, rn, None] * d[rb, rn] - verts[rb, assoc["vid"][rb, rn, rr]]) / .01
    assert torch.equal(captured[0][:, 7:10], expected)
    assert torch.equal(captured[0][:, 10], (args[0]["z"][rb, rn] - .45) / .1)


def test_face_parts_uses_xa_majority_and_tie_not_bary_argmax():
    w = torch.tensor([[.51, .49, 0.], [.51, .49, 0.], [0., .99, .01],
                      [.6, .2, .2], [.01, .6, .39], [.01, .4, .59]])
    f = torch.tensor([[0, 1, 2], [3, 4, 5]])
    # First has majority 0 despite largest summed weights at 1; tie picks 1.
    assert torch.equal(face_parts(w, f), torch.tensor([0, 1]))


def test_source_gather_uses_exact_stride_and_invalidates_padding_and_hot_events():
    batch = SimpleNamespace(events=torch.zeros(7, 5), xag_source_index=torch.arange(7),
                            xag_source_ms=torch.arange(7) + 100, xag_part=torch.arange(7),
                            xag_face=torch.arange(7), xag_bary=torch.ones(7, 3) / 3,
                            xag_valid=torch.ones(7, dtype=torch.bool))
    batch.xag_source_index[5] = -1
    src, mask = stride_sample(torch.tensor([0, 5, 7]), 3)
    out = sample_sources(batch, src, mask)
    assert torch.equal(out["xag_part"], batch.xag_part[src])
    assert torch.equal(out["xag_valid"], mask & (batch.xag_source_index[src] >= 0))
    del batch.xag_face
    with pytest.raises(RuntimeError, match="xag_face"):
        sample_sources(batch, src, mask)


def net():
    args, _ = scene()
    torch.manual_seed(3407)
    return XYZMeshNet(args[6], args[4])


def inputs():
    args, _ = scene()
    assoc, d, mask, verts = args[:4]
    g = torch.Generator().manual_seed(33)
    return (torch.randn(1, 4, 7, generator=g), d, mask, verts, assoc,
            torch.randn(1, 48, generator=g) / 10)


def activate_head(m):
    with torch.no_grad():
        g = torch.Generator().manual_seed(6)
        m.fc2.weight.copy_(torch.randn(m.fc2.weight.shape, generator=g) * .02)
        m.fc2.bias.copy_(torch.randn(m.fc2.bias.shape, generator=g) * .01)
    return m


def test_capture_and_identity_hook_preserve_output_bitwise_and_replay():
    m = activate_head(net()).eval()
    x = inputs()
    with torch.no_grad():
        base = m(*x)[0]
        observed = m(*x, capture=True)[0]
        hooked = m(*x, capture=True, h3_transform=lambda H, S, O: H)[0]
    assert torch.equal(base, observed) and torch.equal(base, hooked)
    cap = m.xag_last_capture
    assert cap["H0"].shape == cap["H3"].shape == (1, 6, 128)
    assert cap["slots"].shape == (1, 17, 258) and cap["zs"].shape == (1, 544)
    h = cap["H3"].clone().requires_grad_(True)
    replay = m.xag_readout_from_h3(h, cap)
    assert torch.equal(replay, base)
    replay.square().sum().backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="preserve shape"):
        m(*x, h3_transform=lambda H, S, O: H[:, :1])


@pytest.mark.parametrize("cfg,xyz", [({"XAG_ASSOC": "oracle"}, True), ({"XAG_GNN": "wide"}, True),
                                     ({"XAG_ASSOC": "base"}, False),
                                     # the removed XCL oracle / global-residual modes are not accepted any more
                                     ({"XAG_ASSOC": "full_oracle"}, True),
                                     ({"XAG_GNN": "oracle_dense"}, True),
                                     ({"XAG_GNN": "global_residual"}, True),
                                     ({"XAG_ASSOC": "face_route", "ENCODER_LAYERS": 2}, True),
                                     ({"XAG_ASSOC": "face_route", "XYZ_GEOM": False}, True)])
def test_incompatible_config_is_rejected(cfg, xyz):
    with pytest.raises(ValueError):
        validate_config(cfg, xyz_mesh=xyz)


def model(assoc="base"):
    from config import load_config
    from model import MNISTModel
    cfg = load_config(str(ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"))     # the S45 identity (an XYZ-C1 model)
    cfg["MANO"]["NPZ"] = str(ROOT / "assets/mano_right.npz")
    cfg["MODEL"].update(XAG_GNN="legacy", XAG_ASSOC=assoc)
    torch.manual_seed(3407)
    return MNISTModel(cfg=cfg)


def empty_batch():
    from semkine.events import EventPacketBatch
    return EventPacketBatch(events=torch.empty(0, 5), ptr=torch.tensor([0, 0]), sequence_id=torch.zeros(1, dtype=torch.long),
                            t_start_us=torch.zeros(1, dtype=torch.long), t_end_us=torch.tensor([50000]),
                            delta_t_s=torch.tensor([.05]), is_sequence_start=torch.tensor([False]),
                            is_sequence_end=torch.tensor([False]), target=torch.zeros(1, 51), prev_state=torch.zeros(1, 51),
                            betas=torch.zeros(1, 10), camera_K=torch.eye(3)[None])


@pytest.mark.parametrize("assoc", ["base", "part_route"])
def test_model_all_empty_training_connects_every_parameter(assoc):
    m = model(assoc).train()
    batch = empty_batch()
    output = m.forward_packet(batch)
    assert torch.equal(output, batch.prev_state)
    output.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any()
               for p in m.parameters() if p.requires_grad)


def test_base_model_cannot_read_any_oracle_source_field():
    m = model("base").eval()
    activate_head(m.xyz)
    batch = empty_batch()
    batch.events = torch.tensor([[0., 120., 90., .001, 1.], [0., 121., 90., .002, 0.],
                                 [0., 119., 92., .003, 1.]])
    batch.ptr = torch.tensor([0, 3])
    batch.prev_state[:, 2] = .45
    batch.camera_K = torch.tensor([[[603.45, 0., 325.09], [0., 602.96, 242.10], [0., 0., 1.]]])
    with torch.no_grad():
        without_sources = m.forward_packet(batch)
        batch.xag_source_index = torch.tensor([-1, 9000, 123])
        batch.xag_source_ms = torch.tensor([-99, 5, 99999999])
        batch.xag_part = torch.tensor([15, -1, 900])
        batch.xag_face = torch.tensor([-1, 1537, 90000])
        batch.xag_bary = torch.full((3, 3), float("nan"))
        batch.xag_valid = torch.ones(3, dtype=torch.bool)
        with_poisoned_sources = m.forward_packet(batch)
    assert torch.equal(without_sources, with_poisoned_sources)

