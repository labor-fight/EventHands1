"""Opt-in local position/motion records and one fixed-reference MANO update.

This module does not read a dataset, construct GT-conditioned patches, or modify
the legacy S43/XYZ encoders. The caller supplies causal dual-time patches and
pixel Jacobians in the *same* 51D parameter convention as its state update.

The learned records are candidate-conditioned, same-source predictions. The
linear layer below is a discriminative update, not an independent-sensor
posterior or exact Gauss--Newton of a state-dependent measurement network.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class JointLocalConfig:
    patch_channels: int = 37
    patch_size: int = 16
    dof: int = 51
    hidden: int = 128
    translation_scale_m: float = 0.02
    rotation_scale_rad: float = 0.1
    start_damping: float = 1.0
    end_damping: float = 1.0
    covariance_floor: float = 0.03
    initial_std_normalized: float = 1.0
    motion_conditioned_assignment: bool = True
    check_finite: bool = True

    def __post_init__(self):
        if self.patch_channels != 37 or self.patch_size != 16:
            raise ValueError("The registered local contract is 37 channels and 16x16 patches")
        if self.dof != 51:
            raise ValueError("JointLocalOperator uses the original 51D MANO state")
        if min(self.translation_scale_m, self.rotation_scale_rad,
               self.start_damping, self.end_damping, self.covariance_floor) <= 0:
            raise ValueError("State scales, damping, and covariance floor must be positive")
        if self.initial_std_normalized <= self.covariance_floor:
            raise ValueError("Initial standard deviation must exceed covariance floor")


def _require_shape(value: Tensor, shape: Tuple[int, ...], name: str):
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


def source_normalized_assignment(
    logits: Tensor,
    observed: Tensor,
    source_ids: Tensor,
    unknown_logit: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Candidate softmax with an unknown branch and bounded source mass.

    ``source_ids[B,V]`` identifies repeated candidates for the SAME observed
    support. It is not a vertex identity. Unrelated supports must have different
    IDs; overlapping supports need to be grouped upstream. IDs may be arbitrary
    nonnegative integers and are local to each batch lane.

    Candidate logits include a uniform 1/n-candidate prior. Thus duplicating
    the complete set of equal candidate records does not manufacture trust.
    Sources are then averaged, so adding cloned vertices cannot increase the
    total mass above one. Unknown and weights are scores, not calibrated
    probabilities or information precision.
    """
    if logits.ndim != 2 or observed.shape != logits.shape or source_ids.shape != logits.shape:
        raise ValueError("logits, observed and source_ids must all be [B,V]")
    if source_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("source_ids must be integer")
    if unknown_logit.numel() != 1:
        raise ValueError("unknown_logit must be scalar")
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    unknown_logit = unknown_logit.to(logits)
    weights, unknown, count_out = [], [], []
    for lane in range(logits.shape[0]):
        ids = source_ids[lane]
        valid = observed[lane].bool() & (ids >= 0)
        # Invalid IDs do not create another source. unique is used only for a
        # bounded local record table; no event identities are inferred here.
        _, inverse = torch.unique(ids.clamp_min(0), sorted=True, return_inverse=True)
        groups = int(inverse.max().detach()) + 1 if inverse.numel() else 0
        if not groups:
            weights.append(logits[lane] * 0)
            unknown.append(torch.ones_like(logits[lane]))
            count_out.append(logits.new_zeros(()))
            continue
        sizes = logits.new_zeros(groups).scatter_add(0, inverse, valid.to(logits.dtype))
        candidate = logits[lane] - sizes[inverse].clamp_min(1).log()
        candidate = candidate.masked_fill(~valid, -torch.inf)
        maxima = logits.new_full((groups,), -torch.inf).scatter_reduce(
            0, inverse, candidate, reduce="amax", include_self=True)
        maxima = torch.maximum(maxima, unknown_logit.reshape(()).expand_as(maxima))
        numer = torch.exp(candidate - maxima[inverse])
        null = torch.exp(unknown_logit.reshape(()) - maxima)
        denominator = logits.new_zeros(groups).scatter_add(0, inverse, numer) + null
        alpha = numer / denominator[inverse]
        null_per_record = null[inverse] / denominator[inverse]
        active_count = (sizes > 0).to(logits.dtype).sum()
        weights.append(alpha / active_count.clamp_min(1))
        unknown.append(null_per_record)
        count_out.append(active_count)
    return torch.stack(weights), torch.stack(unknown), torch.stack(count_out)


def _fixed_reference_joint_solve_impl(
    means_px: Tensor,
    covariance_px: Tensor,
    B0: Tensor,
    B1: Tensor,
    normal: Tensor,
    nominal_delta: Tensor,
    weights: Tensor,
    state_scale: Tensor,
    *,
    start_damping: float = 1.0,
    end_damping: float = 1.0,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Solve the two-endpoint block problem at one fixed reference.

    B0/B1 are d(pixel position)/d(original physical MANO parameter), [B,V,2,D].
    They MUST NOT already include D. ``state_scale`` supplies D and is applied
    once here. normal is a caller-defined UNIT image normal, [B,V,2].

    means_px is [a_normal, d_normal], and nominal_delta is p(qA1)-p(q0ref).
    The block Jacobian is C=[c0,0;-c0,c1], and endpoint deltas are expressed
    around q0ref and qA1, respectively. No information gives exact zero deltas.
    The caller composes these deltas in its registered additive/retraction
    convention and enforces its separately declared camera/trust-region rules.
    """
    if means_px.ndim != 3 or means_px.shape[-1] != 2:
        raise ValueError("means_px must be [B,V,2]")
    B, V, _ = means_px.shape
    if B0.ndim != 4 or B0.shape[:3] != (B, V, 2) or B1.shape != B0.shape:
        raise ValueError("B0 and B1 must be [B,V,2,D]")
    dof = B0.shape[-1]
    _require_shape(covariance_px, (B, V, 2, 2), "covariance_px")
    _require_shape(normal, (B, V, 2), "normal")
    _require_shape(nominal_delta, (B, V, 2), "nominal_delta")
    _require_shape(weights, (B, V), "weights")
    _require_shape(state_scale, (dof,), "state_scale")
    if min(float(start_damping), float(end_damping)) <= 0:
        raise ValueError("Both damping blocks must be positive")
    # Small linear solves are deliberately outside mixed precision. Casting
    # preserves the outer supervised gradient to local predictions and weights.
    dtype = torch.float64 if means_px.dtype == torch.float64 else torch.float32
    means, covariance = means_px.to(dtype), covariance_px.to(dtype)
    b0, b1, n = B0.to(dtype), B1.to(dtype), normal.to(dtype)
    scale = state_scale.to(device=means.device, dtype=dtype)
    c0 = torch.einsum("bvi,bvid->bvd", n, b0) * scale
    c1 = torch.einsum("bvi,bvid->bvd", n, b1) * scale
    zero = torch.zeros_like(c0)
    C = torch.stack((torch.cat((c0, zero), -1), torch.cat((-c0, c1), -1)), -2)
    nominal_normal = (n * nominal_delta.to(dtype)).sum(-1)
    residual = torch.stack((means[..., 0], means[..., 1] - nominal_normal), -1)
    cov_factor, cov_info = torch.linalg.cholesky_ex(covariance)
    if bool((cov_info != 0).any()):
        raise ArithmeticError("Local covariance is not positive definite")
    precision = torch.cholesky_inverse(cov_factor)
    W = precision * weights.to(dtype)[..., None, None]
    H = torch.einsum("bvri,bvrs,bvsj->bij", C, W, C)
    gradient = torch.einsum("bvri,bvrs,bvs->bi", C, W, residual)
    damping = torch.cat((means.new_full((dof,), float(start_damping)),
                         means.new_full((dof,), float(end_damping))))
    H = (H + H.transpose(-1, -2)) * 0.5 + torch.diag(damping).unsqueeze(0)
    factor, info = torch.linalg.cholesky_ex(H)
    if bool((info != 0).any()):
        raise ArithmeticError("Regularized joint system is not positive definite")
    xi = torch.cholesky_solve(gradient.unsqueeze(-1), factor).squeeze(-1)
    delta0, delta1 = xi[:, :dof] * scale, xi[:, dof:] * scale
    return delta0, delta1, {
        "xi0": xi[:, :dof], "xi1": xi[:, dof:], "C": C,
        "residual_px": residual, "precision_px": precision, "weights": weights,
        "normal_matrix": H, "normal_rhs": gradient,
    }


def fixed_reference_joint_solve(
    means_px: Tensor,
    covariance_px: Tensor,
    B0: Tensor,
    B1: Tensor,
    normal: Tensor,
    nominal_delta: Tensor,
    weights: Tensor,
    state_scale: Tensor,
    *,
    start_damping: float = 1.0,
    end_damping: float = 1.0,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Public FP32/FP64 solve; see the implementation for the exact contract.

    Disabling outer autocast matters: merely calling ``.float()`` before an
    einsum does not prevent a surrounding BF16 context from downcasting it.
    """
    with torch.autocast(device_type=means_px.device.type, enabled=False):
        return _fixed_reference_joint_solve_impl(
            means_px, covariance_px, B0, B1, normal, nominal_delta,
            weights, state_scale, start_damping=start_damping,
            end_damping=end_damping)


class JointLocalOperator(nn.Module):
    """Shared local CNN, source-wise motion assignment, and joint update.

    The mean head is residual-initialized: a starts at zero, while total d
    starts at the nominal global endpoint displacement. At initialization the
    local state update is therefore exactly zero, including on nonzero-motion
    references. This is public initialization, not a measurement claim.

    Geometry arrays must be legal reference/candidate quantities. GT belongs
    solely in the separate loss function, never in this forward signature.
    """
    def __init__(self, config: Optional[JointLocalConfig] = None):
        super().__init__()
        self.config = config or JointLocalConfig()
        c = self.config
        self.cnn = nn.Sequential(
            nn.Conv2d(37, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
        )
        self.encode = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, c.hidden), nn.SiLU())
        self.mean_head = nn.Linear(c.hidden, 2)
        self.covariance_head = nn.Linear(c.hidden, 3)
        # Features include the observed d prediction and geometric sensitivity,
        # not correction residual magnitude. Large a is never a hard reject.
        assoc_in = c.hidden + 1 + 2 + 2 * c.dof + 1
        self.association_head = nn.Sequential(nn.Linear(assoc_in, 64), nn.SiLU(), nn.Linear(64, 1))
        self.validity_head = nn.Linear(c.hidden, 1)
        self.unknown_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("state_scale", torch.tensor(
            [c.translation_scale_m] * 3 + [c.rotation_scale_rad] * 48, dtype=torch.float32))
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.covariance_head.weight)
        std_raw = torch.log(torch.expm1(torch.tensor(c.initial_std_normalized - c.covariance_floor)))
        with torch.no_grad():
            self.covariance_head.bias.copy_(torch.tensor([std_raw, 0.0, std_raw]))
        nn.init.zeros_(self.association_head[-1].weight)
        nn.init.zeros_(self.association_head[-1].bias)
        nn.init.zeros_(self.validity_head.weight)
        nn.init.constant_(self.validity_head.bias, 2.0)

    def forward(self, patches: Tensor, rho: Tensor, observed: Tensor,
                B0: Tensor, B1: Tensor, normal: Tensor, nominal_delta: Tensor,
                *, source_ids: Optional[Tensor] = None,
                candidate_mask: Optional[Tensor] = None,
                spatial_logits: Optional[Tensor] = None):
        if patches.ndim != 5 or patches.shape[2:] != (37, 16, 16):
            raise ValueError("patches must be [B,V,37,16,16]")
        B, V = patches.shape[:2]
        for value, name in ((rho, "rho"), (observed, "observed")):
            _require_shape(value, (B, V), name)
        for value, name in ((B0, "B0"), (B1, "B1")):
            _require_shape(value, (B, V, 2, self.config.dof), name)
        _require_shape(normal, (B, V, 2), "normal")
        _require_shape(nominal_delta, (B, V, 2), "nominal_delta")
        if self.config.check_finite:
            for value, name in ((patches, "patches"), (rho, "rho"), (B0, "B0"),
                                (B1, "B1"), (normal, "normal"), (nominal_delta, "nominal_delta")):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Nonfinite {name}; camera-invalid inputs must be handled upstream")
        if bool((rho <= 0).any()):
            raise ValueError("rho must be strictly positive, including masked records")
        norm = normal.norm(dim=-1)
        candidate_mask = torch.ones_like(observed, dtype=torch.bool) if candidate_mask is None else candidate_mask.bool()
        _require_shape(candidate_mask, (B, V), "candidate_mask")
        active = observed.bool() & candidate_mask & (norm > 1e-6)
        unit_normal = normal / norm.clamp_min(1e-6)[..., None]
        source_ids = torch.arange(V, device=patches.device).expand(B, V) if source_ids is None else source_ids
        _require_shape(source_ids, (B, V), "source_ids")
        features = self.encode(self.cnn(patches.reshape(B * V, 37, 16, 16))).reshape(B, V, -1)
        raw_mean = self.mean_head(features).to(rho.dtype)
        nominal_normal = (unit_normal * nominal_delta).sum(-1)
        mean_normalized = torch.stack((raw_mean[..., 0],
                                       raw_mean[..., 1] + nominal_normal / rho), -1)
        cov_raw = self.covariance_head(features)
        with torch.autocast(device_type=patches.device.type, enabled=False):
            cov_raw = cov_raw.double() if rho.dtype == torch.float64 else cov_raw.float()
            s0 = F.softplus(cov_raw[..., 0]) + self.config.covariance_floor
            s1 = F.softplus(cov_raw[..., 2]) + self.config.covariance_floor
            off = torch.tanh(cov_raw[..., 1])
            zeros = torch.zeros_like(s0)
            chol = torch.stack((torch.stack((s0, zeros), -1), torch.stack((off, s1), -1)), -2)
            covariance_normalized = chol @ chol.transpose(-1, -2)
        # Preserve an exactly zero initial update: avoid a divide/multiply
        # round trip of the nominal displacement when raw_mean is zero.
        means_px = torch.stack((raw_mean[..., 0] * rho,
                                raw_mean[..., 1] * rho + nominal_normal), -1)
        covariance_px = covariance_normalized * rho[..., None, None].square()
        c0 = torch.einsum("bvi,bvid->bvd", unit_normal, B0) * self.state_scale.to(B0)
        c1 = torch.einsum("bvi,bvid->bvd", unit_normal, B1) * self.state_scale.to(B1)
        assoc_features = torch.cat((features, mean_normalized[..., 1:2], unit_normal,
                                    c0 / rho[..., None], c1 / rho[..., None], active[..., None].to(features)), -1)
        logits = self.association_head(assoc_features).squeeze(-1)
        if not self.config.motion_conditioned_assignment:
            # This control bypasses ALL temporal H/d features in assignment.
            # Merely deleting the explicit d input would not remove motion.
            if spatial_logits is None:
                spatial_logits = torch.zeros_like(logits)
            _require_shape(spatial_logits, (B, V), "spatial_logits")
            logits = spatial_logits.to(logits)
        weights, unknown, source_count = source_normalized_assignment(
            logits, active, source_ids, self.unknown_logit)
        association_mass = weights
        validity_logits = self.validity_head(features).squeeze(-1)
        validity = torch.sigmoid(validity_logits)
        weights = weights * validity
        delta0, delta1, solve_aux = fixed_reference_joint_solve(
            means_px, covariance_px, B0, B1, unit_normal, nominal_delta, weights,
            self.state_scale, start_damping=self.config.start_damping,
            end_damping=self.config.end_damping)
        return delta0, delta1, {
            "a_normal_px": means_px[..., 0], "d_normal_px": means_px[..., 1],
            "mean_normalized": mean_normalized, "raw_mean_residual": raw_mean,
            "covariance_normalized": covariance_normalized, "covariance_px": covariance_px,
            "association_logits": logits, "unknown_score": unknown,
            "association_mass": association_mass,
            "validity_logits": validity_logits, "validity_score": validity,
            "active": active, "source_count": source_count, "normal": unit_normal,
            **solve_aux,
        }

    def metadata(self):
        return {
            "schema": "xvm_joint_local_operator_v1", "config": asdict(self.config),
            "patch_layout": "F0[16],F1[16],boundary0,boundary1,x,y,both_exposed",
            "geometry_input": "pixel Jacobian versus original physical 51D parameters; D applied once internally",
            "mean": "normalized a residual, normalized d residual around nominal global displacement",
            "covariance": "joint a/d normalized covariance; covariance_px=rho^2*covariance_normalized",
            "association": "motion-conditioned source-candidate scores with unknown; not posterior",
            "update": "fixed-reference one-step discriminative linear solve; no independent-sensor claim",
            "initialization": "zero mean residual; local update exactly zero before training",
            "source_grouping": "caller must identify shared observed supports; default unique record IDs is explicit",
        }


def local_record_loss(aux: Dict[str, Tensor], target_a_px: Tensor, target_d_px: Tensor,
                      rho: Tensor, supervised: Tensor, *, validity_target: Optional[Tensor] = None,
                      record_weights: Optional[Tensor] = None):
    """Auxiliary joint normal NLL on supported directions, separate from input construction.

    Targets are caller-created supervision-only geometry proxies. This function
    does not infer an event source identity or mark an unsupported vertex as an
    observed motion sensor. The full model's endpoint/mesh loss remains primary.

    ``record_weights`` are fixed supervision weights, normally the inverse
    number of supervised candidates of each source. They must not be learned
    association scores: a model may not avoid its targets by lowering alpha.
    The caller also uses zero for non-supervised/padded lanes. Source weights
    apply to both the normal NLL and optional validity BCE aggregation.
    """
    target = torch.stack((target_a_px, target_d_px), -1) / rho[..., None]
    mask = supervised.to(target.dtype)
    if record_weights is None:
        fixed_weight = torch.ones_like(mask)
    else:
        _require_shape(record_weights, tuple(mask.shape), "record_weights")
        if record_weights.requires_grad:
            raise ValueError("record_weights must be fixed supervision weights, not learned association scores")
        if not bool(torch.isfinite(record_weights).all()) or bool((record_weights < 0).any()):
            raise ValueError("record_weights must be finite and nonnegative")
        fixed_weight = record_weights.to(mask)
    nll_weight = mask * fixed_weight
    residual = aux["mean_normalized"] - target
    covariance = aux["covariance_normalized"]
    factor = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(factor, residual.unsqueeze(-1), upper=False).squeeze(-1)
    nll = 0.5 * whitened.square().sum(-1) + factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    nll_mean = (nll * nll_weight).sum() / nll_weight.sum().clamp_min(1)
    if validity_target is None:
        valid_loss = nll_mean * 0
    else:
        valid_per_record = F.binary_cross_entropy_with_logits(
            aux["validity_logits"], validity_target.to(target.dtype), reduction="none")
        valid_loss = (valid_per_record * fixed_weight).sum() / fixed_weight.sum().clamp_min(1)
    return {"local_nll": nll_mean, "validity_bce": valid_loss,
            "supervised_records": mask.sum(), "supervised_source_mass": nll_weight.sum()}


__all__ = ["JointLocalConfig", "JointLocalOperator", "fixed_reference_joint_solve",
           "source_normalized_assignment", "local_record_loss"]
