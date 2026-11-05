r"""ACU engineering amendment 2 (docs/EXPERIMENT_RECORDS.md#acu-amend2-20261104): PR's causal window sums with exact
integer prefix sums, so the router runs on CUDA under torch.use_deterministic_algorithms(True).

semkine/acu_pr.py (frozen) builds the causal cell / image sums of `causal_cell_sums` from a float64 `cumsum`; CUDA has no
deterministic floating-point cumsum, so the PR preflight stopped at its first step. Here the features are rounded to the
fixed-point grid 2^-20 and summed as int64 (integer cumsum is exact, hence deterministic on every device); a window sum is
the difference of two exact prefix sums, divided back. The function is otherwise `acu_pr.causal_cell_sums` line by line
(same segments, keys, searchsorted bounds and output layout); the only change is the rounding of each summand to 2^-20
(<= 4.8e-7 absolute). The window sums are exact (no float64 cancellation), so they depend only on the events inside the
window; causality is unchanged (later events never change an earlier node's outputs, bitwise).

    install()     rebinds semkine.acu_pr.causal_cell_sums (PRNet.forward looks it up at call time); idempotent
"""
from __future__ import annotations

import torch

from . import acu_pr as PR

FIX_BITS = 20
SCALE = float(2 ** FIX_BITS)
_native = PR.causal_cell_sums


def _prefix(feat_sorted: torch.Tensor) -> torch.Tensor:
    """(n + 1, C) int64 prefix sums of the fixed-point features (row 0 is zero)."""
    q = torch.round(feat_sorted.double() * SCALE).long()
    if q.numel():
        # every partial sum stays far inside int64: |sum| <= n * max|q|
        lim = int(q.abs().max().item()) * q.shape[0]
        if lim >= 2 ** 62:
            raise OverflowError(f"fixed-point window sums would overflow int64 (bound {lim})")
    z = torch.zeros(1, q.shape[1], device=q.device, dtype=torch.long)
    return torch.cat([z, q.cumsum(0)])


@torch.no_grad()
def causal_cell_sums(b, pos, t_us, x, y, feat, cell, tau_us):
    """`acu_pr.causal_cell_sums` with exact fixed-point prefix sums (see the module docstring)."""
    n, C = feat.shape
    dev = feat.device
    ncx, ncy = PR.WIDTH // cell + 2, PR.HEIGHT // cell + 2
    gx, gy = (x // cell).long() + 1, (y // cell).long() + 1
    P = int(pos.max().item()) + 2 if n else 2
    t0 = t_us.min() if n else torch.zeros((), device=dev, dtype=torch.long)
    T = int(t_us.max().item() - t_us.min().item()) + int(tau_us) + 2 if n else 2

    def window_sums(seg):
        key_p = seg * P + pos
        order = torch.argsort(key_p)
        kp, ks, kt = key_p[order], seg[order], t_us[order]
        key_t = ks * T + (kt - t0)
        cs = _prefix(feat[order])
        cn = torch.arange(n + 1, device=dev, dtype=torch.long)
        return kp, key_t, cs, cn

    def diff(cs, up, lo):
        return ((cs[up] - cs[lo]).double() / SCALE).float()

    out_c, out_n = [], []
    seg_all = PR._segments(b, gx, gy, ncx, ncy)
    kp, key_t, cs, cn = window_sums(seg_all)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            s2 = PR._segments(b, gx + dx, gy + dy, ncx, ncy)
            up = torch.searchsorted(kp, s2 * P + pos)
            lo = torch.searchsorted(key_t, s2 * T + (t_us - int(tau_us) - t0))
            lo = torch.minimum(lo, up)
            out_c.append(diff(cs, up, lo))
            out_n.append((cn[up] - cn[lo]).float())
    seg_b = b.long()
    kp, key_t, cs, cn = window_sums(seg_b)
    up = torch.searchsorted(kp, seg_b * P + pos)
    lo = torch.minimum(torch.searchsorted(key_t, seg_b * T + (t_us - int(tau_us) - t0)), up)
    return torch.stack(out_c, 1), torch.stack(out_n, 1), diff(cs, up, lo), (cn[up] - cn[lo]).float()


def install() -> str:
    PR.causal_cell_sums = causal_cell_sums
    return "acu_pr_det"


def uninstall() -> None:
    PR.causal_cell_sums = _native
