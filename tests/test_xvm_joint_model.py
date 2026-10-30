"""CPU contracts for the new XVM joint record and fixed-reference update.

These tests check algebra, units, support reuse and trainable gradients. They
do not load events, perform a model-accuracy experiment, or assert a method gain.
"""
import torch

from tools.xvm_joint.model import (
    JointLocalConfig, JointLocalOperator, fixed_reference_joint_solve,
    local_record_loss, source_normalized_assignment,
)


def _scalar_problem(a, d, nominal=2.0, damping=1.0):
    means = torch.tensor([[[a, d]]], dtype=torch.float64)
    cov = torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2)
    jac = torch.tensor([[[[1.0], [0.0]]]], dtype=torch.float64)
    normal = torch.tensor([[[1.0, 0.0]]], dtype=torch.float64)
    nominal_xy = torch.tensor([[[nominal, 0.0]]], dtype=torch.float64)
    return fixed_reference_joint_solve(
        means, cov, jac, jac, normal, nominal_xy,
        torch.ones(1, 1, dtype=torch.float64), torch.ones(1, dtype=torch.float64),
        start_damping=damping, end_damping=damping)


def _inputs(B=1, V=4):
    generator = torch.Generator().manual_seed(8142)
    patches = torch.randn(B, V, 37, 16, 16, generator=generator)
    rho = torch.full((B, V), 7.0)
    observed = torch.ones(B, V, dtype=torch.bool)
    b0 = torch.randn(B, V, 2, 51, generator=generator)
    b1 = torch.randn(B, V, 2, 51, generator=generator)
    normal = torch.randn(B, V, 2, generator=generator)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    nominal = torch.randn(B, V, 2, generator=generator)
    source = torch.tensor([[0, 0, 1, 1]]).expand(B, V) if V == 4 else torch.arange(V).expand(B, V)
    return patches, rho, observed, b0, b1, normal, nominal, source


def test_joint_block_matches_known_two_endpoint_solution():
    # Objective: (x0-3)^2 + (x1-x0-2)^2 + x0^2 + x1^2.
    # The exact minimizer is x0=0.8, x1=1.4. This catches both the
    # negative c0 block in the motion row and the endpoint anchor damping.
    d0, d1, aux = _scalar_problem(3.0, 4.0)
    torch.testing.assert_close(d0, torch.tensor([[0.8]], dtype=torch.float64))
    torch.testing.assert_close(d1, torch.tensor([[1.4]], dtype=torch.float64))
    torch.testing.assert_close(aux["C"], torch.tensor([[[[1.0, 0.0], [-1.0, 1.0]]]], dtype=torch.float64))


def test_position_offset_and_true_motion_are_separate_variables():
    # In the weak-prior limit, start correction is a and endpoint correction
    # is a+(d-d_nominal). A velocity-only update could not recover both.
    d0, d1, _ = _scalar_problem(3.0, 4.0, damping=1e-8)
    torch.testing.assert_close(d0, torch.tensor([[3.0]], dtype=torch.float64), atol=1e-6, rtol=0)
    torch.testing.assert_close(d1, torch.tensor([[5.0]], dtype=torch.float64), atol=1e-6, rtol=0)
    z0, z1, _ = _scalar_problem(0.0, 4.0, damping=1e-8)
    torch.testing.assert_close(z0, torch.zeros_like(z0), atol=1e-6, rtol=0)
    torch.testing.assert_close(z1, torch.tensor([[2.0]], dtype=torch.float64), atol=1e-6, rtol=0)


def test_residual_initialization_preserves_global_proposal_and_units():
    model = JointLocalOperator()
    p, rho, obs, b0, b1, n, nominal, src = _inputs()
    d0, d1, aux = model(p, rho, obs, b0, b1, n, nominal, source_ids=src)
    assert torch.count_nonzero(d0) == 0
    assert torch.count_nonzero(d1) == 0
    torch.testing.assert_close(aux["a_normal_px"], torch.zeros_like(rho))
    torch.testing.assert_close(aux["d_normal_px"], (n * nominal).sum(-1))
    torch.testing.assert_close(aux["covariance_px"],
                               aux["covariance_normalized"] * rho[..., None, None].square())


def test_no_observed_support_gives_exact_zero_local_update():
    model = JointLocalOperator()
    with torch.no_grad():
        model.mean_head.bias.copy_(torch.tensor([0.7, -0.3]))
    p, rho, obs, b0, b1, n, nominal, src = _inputs()
    obs.zero_()
    d0, d1, aux = model(p, rho, obs, b0, b1, n, nominal, source_ids=src)
    assert torch.count_nonzero(d0) == torch.count_nonzero(d1) == 0
    assert torch.count_nonzero(aux["weights"]) == 0
    torch.testing.assert_close(aux["unknown_score"], torch.ones_like(rho))


def test_duplicating_candidates_does_not_create_independent_evidence():
    score = torch.tensor([[0.8, -0.4]], dtype=torch.float64)
    ids = torch.tensor([[3, 3]])
    live = torch.ones_like(ids, dtype=torch.bool)
    null = torch.tensor(0.2, dtype=torch.float64)
    weight, unknown, count = source_normalized_assignment(score, live, ids, null)
    cloned_weight, cloned_unknown, cloned_count = source_normalized_assignment(
        score.repeat_interleave(2, dim=1), live.repeat_interleave(2, dim=1),
        ids.repeat_interleave(2, dim=1), null)
    torch.testing.assert_close(cloned_weight.reshape(1, 2, 2).sum(-1), weight)
    torch.testing.assert_close(cloned_unknown[:, ::2], unknown)
    torch.testing.assert_close(cloned_count, count)
    assert float(weight.sum()) <= 1.0


def test_static_assignment_does_not_read_temporal_features_or_d():
    model = JointLocalOperator(JointLocalConfig(motion_conditioned_assignment=False))
    p, rho, obs, b0, b1, n, nominal, src = _inputs()
    spatial = torch.tensor([[0.3, -0.2, 0.4, 0.8]])
    _, _, first = model(p, rho, obs, b0, b1, n, nominal,
                        source_ids=src, spatial_logits=spatial)
    with torch.no_grad():
        model.mean_head.bias.copy_(torch.tensor([8.0, -5.0]))
    _, _, second = model(-p, rho, obs, b0, b1, n, -nominal,
                         source_ids=src, spatial_logits=spatial)
    torch.testing.assert_close(first["association_logits"], second["association_logits"])
    torch.testing.assert_close(first["association_mass"], second["association_mass"])


def test_outer_gradients_reach_means_uncertainty_and_motion_assignment():
    model = JointLocalOperator()
    # At the public zero-residual initial point, some endpoint gradients
    # intentionally vanish. This checks a state after mean-head warmup.
    with torch.no_grad():
        model.mean_head.weight.normal_(0.0, 0.01)
        model.mean_head.bias.copy_(torch.tensor([0.2, -0.1]))
    p, rho, obs, b0, b1, n, nominal, src = _inputs()
    d0, d1, aux = model(p, rho, obs, b0, b1, n, nominal, source_ids=src)
    extra = local_record_loss(aux, torch.full_like(rho, 1.7), torch.full_like(rho, -0.4),
                              rho, obs, validity_target=obs)
    loss = d0.square().sum() + d1.square().sum() + extra["local_nll"] + extra["validity_bce"]
    loss.backward()
    for parameter in (model.mean_head.weight, model.covariance_head.weight,
                      model.association_head[-1].weight, model.cnn[0].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert float(parameter.grad.abs().sum()) > 0


def test_cpu_autocast_keeps_covariance_and_linear_solve_in_float32():
    model = JointLocalOperator()
    p, rho, obs, b0, b1, n, nominal, src = _inputs()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        d0, d1, aux = model(p, rho, obs, b0, b1, n, nominal, source_ids=src)
        loss = local_record_loss(aux, torch.zeros_like(rho), torch.ones_like(rho), rho, obs)["local_nll"]
    assert aux["covariance_normalized"].dtype == torch.float32
    assert aux["normal_matrix"].dtype == torch.float32
    assert torch.isfinite(d0).all() and torch.isfinite(d1).all() and torch.isfinite(loss)
    assert torch.count_nonzero(d0) == torch.count_nonzero(d1) == 0


def test_source_balanced_auxiliary_loss_is_invariant_to_candidate_cloning():
    # Two sources have different errors. Cloning every candidate of only the
    # first source must not give that source three times the supervision mass.
    aux = {
        "mean_normalized": torch.tensor([[[0.5, -0.3], [1.2, 0.7]]], dtype=torch.float64),
        "covariance_normalized": torch.eye(2, dtype=torch.float64).reshape(1, 1, 2, 2).repeat(1, 2, 1, 1),
        "validity_logits": torch.tensor([[0.2, -0.8]], dtype=torch.float64),
    }
    rho = torch.ones(1, 2, dtype=torch.float64)
    target_a = torch.tensor([[1.0, 3.0]], dtype=torch.float64)
    target_d = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    supervised = torch.ones(1, 2, dtype=torch.bool)
    valid = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    original = local_record_loss(aux, target_a, target_d, rho, supervised,
                                 validity_target=valid, record_weights=torch.ones_like(rho))
    indices = torch.tensor([0, 0, 0, 1])
    cloned_aux = {key: value[:, indices] for key, value in aux.items()}
    fixed = torch.tensor([[1/3, 1/3, 1/3, 1.0]], dtype=torch.float64)
    cloned = local_record_loss(cloned_aux, target_a[:, indices], target_d[:, indices],
                              rho[:, indices], supervised[:, indices],
                              validity_target=valid[:, indices], record_weights=fixed)
    torch.testing.assert_close(original["local_nll"], cloned["local_nll"])
    torch.testing.assert_close(original["validity_bce"], cloned["validity_bce"])
    torch.testing.assert_close(original["supervised_source_mass"], cloned["supervised_source_mass"])
