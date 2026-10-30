"""Explicit paired-residual gain fitting and checkpoint-compatible inference.

This module does not collect data, select checkpoints, or activate a policy.
Callers must supply matched residuals from the same actual inference states.
The fitted coefficients replace the old effective coefficients; multiplying
them by a previous policy's conservative scales again would be incorrect.
"""
from __future__ import annotations

import copy
import math

import torch

from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel


GROUPS = ((0, 1), (1, 2), (2, 3), (3, 6)) + tuple(
    (start, start + 3) for start in range(6, 51, 3))


def coefficient_vector(values):
    if not isinstance(values, (list, tuple)) or len(values) != 19:
        raise ValueError('exactly 19 fixed effective coefficients are required')
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) or not 0. <= x <= 1. for x in values):
        raise ValueError('effective coefficients must be finite numbers in [0, 1]')
    return tuple(float(x) for x in values)


def fit_paired_gains(i_residual, d_residual, gates, fallback, *,
                     denominator_floor=1e-18, psd_tolerance=1e-12):
    """Fit fixed coefficients for g*k, retaining bias and paired covariance.

    Nx51 columns are physical joint-centroid errors (metres), root errors in
    the GT SO(3) tangent (radians), then 15 local finger AA errors (radians).
    Root tangent fitting is an approximation: actual clipped SO(3) trajectories
    must still be validated. No row, condition, or identity is removed here.
    """
    fallback = coefficient_vector(fallback)
    for value in (denominator_floor, psd_tolerance):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('strictly positive finite numeric tolerances required')
    ei = torch.as_tensor(i_residual, dtype=torch.float64, device='cpu')
    ed = torch.as_tensor(d_residual, dtype=torch.float64, device='cpu')
    gate = torch.as_tensor(gates, dtype=torch.float64, device='cpu')
    if ei.ndim != 2 or ei.shape[1] != 51 or ed.shape != ei.shape or len(ei) == 0:
        raise ValueError('nonempty paired Nx51 residuals required')
    if gate.shape != (len(ei),):
        raise ValueError('one observed evidence gate per paired row required')
    if not all(bool(torch.isfinite(x).all()) for x in (ei, ed, gate)):
        raise ValueError('nonfinite paired observation; no finite-subset fitting')
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError('evidence gates must lie in [0, 1]')
    records, coefficients = [], []
    for index, (start, stop) in enumerate(GROUPS):
        a, b = ei[:, start:stop], ed[:, start:stop]
        mean_i, mean_d = a.mean(0), b.mean(0)
        vi, vd, cov = (a.square().sum(1).mean(), b.square().sum(1).mean(),
                       (a * b).sum(1).mean())
        limit = vi * vd
        if bool(cov.square() > limit + psd_tolerance * max(float(limit), 1e-30)):
            raise ValueError('paired residual moments violate Cauchy-Schwarz')
        difference = a - b
        numerator = (gate * (a * difference).sum(1)).mean()
        denominator = (gate.square() * difference.square().sum(1)).mean()
        degenerate = float(denominator) <= denominator_floor
        unconstrained = None if degenerate else float(numerator / denominator)
        coefficient = fallback[index] if degenerate else min(1., max(0., unconstrained))
        coefficients.append(coefficient)
        records.append(dict(
            group=index, columns=[start, stop], dimensions=stop-start,
            mean_i=mean_i.tolist(), mean_d=mean_d.tolist(),
            vi_uncentered=float(vi), vd_uncentered=float(vd), cid_uncentered=float(cov),
            vi_centered=float((a-mean_i).square().sum(1).mean()),
            vd_centered=float((b-mean_d).square().sum(1).mean()),
            cid_centered=float(((a-mean_i)*(b-mean_d)).sum(1).mean()),
            numerator=float(numerator), denominator=float(denominator),
            unconstrained=unconstrained, coefficient=coefficient,
            degenerate_fallback=degenerate,
            objective_at_baseline=float((a-gate[:, None]*fallback[index]*difference).square().sum(1).mean()),
            objective_at_fit=float((a-gate[:, None]*coefficient*difference).square().sum(1).mean())))
    return dict(rows=len(ei), coefficients=coefficients, groups=records,
                denominator_floor=denominator_floor, psd_tolerance=psd_tolerance,
                moment_convention='uncentered paired group energy, bias retained',
                root_optimality='GT tangent approximation; actual clipped SO3 validation required')


class PairedStatisticsPolicy(MotionCountFusionPolicy):
    def __init__(self, spec):
        value = copy.deepcopy(spec)
        self.fixed_effective_gains = coefficient_vector(value.pop('fixed_effective_gains'))
        if (value.get('gain_scale') != [1., 1., 1.]
                or value.get('count_gate') != 'ramp1000'
                or value.get('motion_count_gate') != 'ramp1000'
                or value.get('translation_reference') != 'joint_centroid'
                or set(value) - {'id', 'gain_scale', 'count_gate', 'motion_count_gate', 'translation_reference'}):
            raise ValueError('paired coefficients require the declared centroid/I1000/D1000 policy without extra scaling')
        super().__init__(value)
        self.spec['fixed_effective_gains'] = list(self.fixed_effective_gains)

    def gains(self, base, counts):
        # base supplies only device/dtype; stale EMA moments are not reused.
        coefficients = base.new_tensor(self.fixed_effective_gains).float()
        return coefficients[None] * (counts.float()/1000.).clamp(0., 1.)[:, None]

    def is_identity(self):
        return False


class XKFPairedStatisticsModel(XKFStableMotionGateModel):
    def configure_fusion(self, spec=None):
        self.fusion_policy = None if spec is None else PairedStatisticsPolicy(spec)
        return self
