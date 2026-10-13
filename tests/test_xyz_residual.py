"""Scientific-control and compatibility contracts for XYZ residual A-D."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semkine.xyz_mesh import XYZMeshNet  # noqa: E402
from semkine.xyz_residual import XYZResidualBranch, residual_permutation  # noqa: E402


def _branch(mode):
    torch.manual_seed(11)
    return XYZResidualBranch(mode)


def _tokens():
    # Every event occupies a different cell, with the same coarse time/polarity
    # stratum but distinct fine temporal content.
    t = torch.zeros(2, 16, 7)
    t[..., 0] = (torch.arange(16) % 4 + 0.5) / 4
    t[..., 1] = (torch.arange(16) // 4 + 0.5) / 4
    t[..., 2] = 1
    t[..., 3] = torch.arange(16) / 40
    t[..., 4] = torch.arange(16) / 10
    t[..., 5] = torch.arange(16) / 30
    t[..., 6] = torch.arange(16) / 20
    return t


def _net(mode="legacy"):
    w = torch.zeros(4, 16)
    w[torch.arange(4), torch.arange(4)] = 1
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    torch.manual_seed(21)
    return XYZMeshNet(w, faces, residual_mode=mode)


def _inputs(tok):
    B, N, _ = tok.shape
    mask = torch.ones(B, N, dtype=torch.bool)
    verts = torch.tensor([[-.1, -.1, .5], [.1, -.1, .5], [.1, .1, .5], [-.1, .1, .5]])
    verts = verts.unsqueeze(0).expand(B, -1, -1)
    d = torch.zeros(B, N, 3)
    d[..., 2] = 1
    assoc = {"q": torch.zeros(B, N), "A": torch.zeros(B, N, 4),
             "vid": torch.zeros(B, N, 4, dtype=torch.long), "z": torch.zeros(B, N),
             "valid": torch.zeros(B, N, dtype=torch.bool), "hit": torch.zeros(B, N, dtype=torch.bool)}
    return d, mask, verts, assoc, torch.zeros(B, 48)


def test_all_arms_have_identical_modules_initialization_and_capacity():
    nets = [_net(m) for m in "ABCD"]
    baseline = nets[0].state_dict()
    for net in nets[1:]:
        assert net.state_dict().keys() == baseline.keys()
        assert all(torch.equal(value, net.state_dict()[key]) for key, value in baseline.items())
    assert len({sum(p.numel() for p in n.parameters()) for n in nets}) == 1
    assert all(n.fc1.in_features == 17 * 32 + 48 + 64 for n in nets)
    legacy = _net()
    assert sum(p.numel() for p in legacy.parameters()) == 459059
    assert not hasattr(legacy, "residual")
    assert legacy.fc1.in_features == 17 * 32 + 48
    assert _net("legacy").state_dict().keys() == legacy.state_dict().keys()


def test_a_contains_no_event_local_occupancy_or_position_and_keeps_slot_identity():
    branch = _branch("A")
    tok = _tokens()
    rw = torch.linspace(.1, 1., 16).expand(2, -1)
    mask = torch.ones_like(rw, dtype=torch.bool)
    out, aux = branch(tok, rw, mask)
    changed_xy = tok.clone()
    changed_xy[..., :2] = changed_xy[..., :2].flip(1)
    out2, _ = branch(changed_xy, rw, mask)
    assert torch.equal(out, out2)
    assert torch.equal(aux["residual_content"], aux["residual_content"][:, :1].expand(-1, 16, -1))
    global_meta = aux["residual_metadata"][..., :3]
    assert torch.equal(global_meta, global_meta[:, :1].expand(-1, 16, -1))
    assert torch.equal(aux["residual_metadata"][..., -2:], branch.centers.unsqueeze(0).expand(2, -1, -1))
    assert not torch.equal(aux["residual_slot_features"], aux["residual_slot_features"][:, :1].expand(-1, 16, -1))


def test_b_c_d_preserve_destination_counts_mass_and_fixed_position():
    tok = _tokens()
    rw = torch.linspace(.76, .99, 16).expand(2, -1)
    mask = torch.ones_like(rw, dtype=torch.bool)
    result = [_branch(mode)(tok, rw, mask)[1] for mode in "BCD"]
    for aux in result:
        assert torch.equal(aux["residual_count"], torch.ones(2, 16))
        assert torch.equal(aux["residual_mass"], rw)
        assert torch.allclose(aux["residual_metadata"][..., -2:], tok[..., :2])
    assert torch.equal(result[0]["residual_metadata"], result[1]["residual_metadata"])
    assert torch.equal(result[0]["residual_metadata"], result[2]["residual_metadata"])
    assert not result[2]["residual_content"].any()


def test_c_is_effective_exchange_with_quantified_quality_change():
    tok = _tokens()
    rw = torch.linspace(.76, .99, 16).expand(2, -1)
    mask = torch.ones_like(rw, dtype=torch.bool)
    bout, baux = _branch("B")(tok, rw, mask)
    cout, caux = _branch("C")(tok, rw, mask)
    assert not torch.equal(bout, cout)
    assert not torch.equal(baux["residual_content"], caux["residual_content"])
    assert float(caux["residual_swap_fraction"]) == 1
    assert float(caux["residual_eligible_swap_fraction"]) == 1
    assert float(caux["residual_crosscell_fraction"]) == 1
    assert float(caux["residual_weighted_crosscell_fraction"]) == 1
    assert float(caux["residual_swap_distance"]) > 0
    assert float(caux["residual_feature_diff"]) > 0
    assert float(caux["residual_quality_diff"]) > 0
    # Full residual packets isolate content-position binding from weight binding.
    _, pure = _branch("C")(tok, torch.ones_like(rw), mask)
    assert float(pure["residual_r1_crosscell_fraction"]) == 1
    assert float(pure["residual_r1_quality_diff"]) == 0


def test_c_exchange_stays_in_quality_time_polarity_strata():
    tok = _tokens().repeat(1, 4, 1)
    tok[:, 16:32, 2] = -1
    tok[:, 32:48, 3] += .5
    rw = torch.ones(2, 64)
    rw[:, 48:] = .4
    _, aux = _branch("C")(tok, rw, torch.ones_like(rw, dtype=torch.bool))
    assert torch.equal(aux["residual_group"], aux["residual_group"].gather(1, aux["residual_permutation"]))
    assert float(aux["residual_eligible_swap_fraction"]) == 1


def test_c_full_residual_stratum_preserves_quality_among_weak_neighbors():
    tok = _tokens()
    rw = torch.ones(2, 16)
    rw[:, :8] = .8
    _, aux = _branch("C")(tok, rw, torch.ones_like(rw, dtype=torch.bool))
    source_rw = rw.gather(1, aux["residual_permutation"])
    assert torch.equal(rw, source_rw)
    assert float(aux["residual_r1_quality_diff"]) == 0


def test_c_does_not_consume_global_rng_and_is_stable_across_modes_padding_and_batch():
    branch = _branch("C")
    tok = _tokens()
    rw = torch.ones(2, 16)
    mask = torch.ones_like(rw, dtype=torch.bool)
    state = torch.random.get_rng_state().clone()
    branch.train()
    out, aux = branch(tok, rw, mask)
    assert torch.equal(state, torch.random.get_rng_state())
    branch.eval()
    out2, aux2 = branch(tok, rw, mask)
    assert torch.equal(out, out2)
    assert torch.equal(aux["residual_permutation"], aux2["residual_permutation"])
    _, single = branch(tok[:1], rw[:1], mask[:1])
    assert torch.equal(aux["residual_permutation"][:1], single["residual_permutation"])
    padded_tok = torch.cat([tok, torch.randn(2, 9, 7)], 1)
    padded_rw = torch.cat([rw, torch.zeros(2, 9)], 1)
    padded_mask = padded_rw > 0
    _, padded = branch(padded_tok, padded_rw, padded_mask)
    assert torch.equal(aux["residual_permutation"], padded["residual_permutation"][:, :16])
    assert torch.equal(out, padded["residual_extra"])


@pytest.mark.parametrize("mode", list("BCD"))
def test_local_branch_has_no_raw_xy_leak_within_cells(mode):
    branch = _branch(mode)
    tok = _tokens()
    rw = torch.ones(2, 16)
    mask = torch.ones_like(rw, dtype=torch.bool)
    captured = []
    hook = branch.content[0].register_forward_pre_hook(lambda module, inp: captured.append(inp[0].detach().clone()))
    out, _ = branch(tok, rw, mask)
    moved = tok.clone()
    moved[..., :2] += .04
    out2, _ = branch(moved, rw, mask)
    hook.remove()
    assert torch.equal(out, out2)
    assert captured[0].shape[-1] == 5
    assert torch.equal(captured[0], tok[..., 2:])
    assert torch.equal(captured[0], captured[1])


def test_d_cannot_use_temporal_content_and_still_has_all_trainable_parameters():
    branch = _branch("D")
    tok = _tokens()
    rw = torch.ones(2, 16)
    mask = torch.ones_like(rw, dtype=torch.bool)
    out, _ = branch(tok, rw, mask)
    altered = tok.clone()
    altered[..., 2:] = torch.randn_like(altered[..., 2:]) * 100
    out2, _ = branch(altered, rw, mask)
    assert torch.equal(out, out2)
    out.square().sum().backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    assert all(not p.grad.any() for p in branch.content.parameters())
    assert any(p.grad.any() for p in branch.slot.parameters())


@pytest.mark.parametrize("mode", list("ABCD"))
@pytest.mark.parametrize("n", [0, 16])
def test_empty_and_zero_evidence_are_exactly_zero(mode, n):
    branch = _branch(mode)
    tok = _tokens()[:, :n]
    rw = torch.zeros(2, n)
    mask = torch.ones_like(rw, dtype=torch.bool)
    out, aux = branch(tok, rw, mask)
    assert out.shape == (2, 64)
    assert not out.any()
    assert all(torch.isfinite(value).all() for value in aux.values())


def test_singleton_strata_do_not_claim_exchange_and_padding_is_not_a_source():
    live = torch.tensor([[True, False, True, False]])
    group = torch.tensor([[1, 0, 2, 0]])
    p = residual_permutation(live, groups=group)
    assert torch.equal(p, torch.arange(4).unsqueeze(0))
    group[0, 2] = 1
    p = residual_permutation(live, groups=group)
    assert p.tolist() == [[2, 1, 0, 3]]


@pytest.mark.parametrize("mode", list("ABCD"))
def test_integration_keeps_old_slots_and_ablation_silences_extra(mode):
    tok = _tokens()
    inp = _inputs(tok)
    net = _net(mode)
    delta, stats, aux = net(tok, *inp)
    legacy = _net()
    _, _, old = legacy(tok, *inp)
    assert torch.equal(aux["slots"], old["slots"])
    assert delta.shape == (2, 51) and aux["residual_extra"].shape == (2, 64)
    assert "xyz_residual_weighted_crosscell_fraction" in stats
    _, _, quiet = net(tok, *inp, ablate=True)
    assert not quiet["residual_extra"].any()
    # With a nonzero head, every active branch reaches the output.
    with torch.no_grad():
        net.fc2.weight.normal_(std=.05)
    net(tok, *inp)[0].square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    if mode != "D":
        assert any(p.grad.any() for p in net.residual.content.parameters())


def test_invalid_modes_are_rejected():
    with pytest.raises(ValueError):
        _branch("legacy")
    with pytest.raises(ValueError):
        _net("unknown")
