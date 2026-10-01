#!/usr/bin/env python3
r"""S16 frontend battle: one arm at a time, parameter budget within 10%.

The plan's novelty defence says the frontend is not a contribution. These modules exist so a
reviewer who asks "why not AEGNN / SAST / a PointNet++" gets a measured answer rather than a
citation. Each arm is a *minimum mechanism*, not a framework port:

* `lnes`         the existing dense surface (not reimplemented)
* `raw_scan`     S2 gated scan
* `sparse_cell`  S2 fallback
* `event_gnn`    S36 AEGNN proper: the events themselves are the nodes, edges are a causal k-NN in
                 a temporal window, and the message carries the relative `(dx, dy, dt)`. See
                 `event_gnn.py`.

Three graph-shaped arms were removed on 2026-08-28 after being measured to a conclusion. Their
numbers, and the reasoning that retires them, are in `docs/GNN_ARMS_ARCHIVE_20260828.md`; do not
reintroduce any of them without reading it first.

* `aegnn_lite`   a full-packet `torch.cdist` k-NN. Structural NO-GO: `O(N^2)`, and packets reach
                 788 817 events, so cost grew with the square of the event rate. `event_gnn`
                 replaces it with a windowed causal search of the same neighbourhood semantics at
                 `O(N * window)`.
* `cell_gnn`     an active-cell lattice that called itself a graph. S35 showed it is a 3x3
                 convolution with its nine taps tied to two matrices. `event_gnn` beat it by
                 5.23 mm recursive RA on half the parameters.
* `keg`          message passing on 16 MANO nodes with state-conditioned routing. Its single-step
                 error was *better* than the dense arm's (12.43 vs 15.42 mm) and its recursive
                 error much worse (26.59 vs 17.17 mm), because routing on the previous pose put
                 the closed-loop gain at 0.910 against the dense arm's 0.449.

SAST / FARSE / SNN thought-arms are represented by the same graph module with a different
neighbour rule rather than by importing those repositories. rpg_asynet is not used.

A battle run trains one frontend against the frozen S1 Track+domrand protocol, same loss, same
seeds, parameter count within 10% of `raw_scan`. Winner is the one whose paired CI on recursive
RA excludes zero in its favour; a tie keeps LNES.
"""
from __future__ import annotations

from torch import nn

from .encoder import RawEventEncoder, SparseCellEncoder


def _pick(kw, *names):
    return {k: kw[k] for k in names if k in kw}


def build_frontend(kind: str, **kw) -> nn.Module:
    kind = (kind or "raw_scan").lower()
    shared = _pick(kw, "height", "width", "hidden", "feat_dim", "extra_channels")
    if kind in ("raw_scan", "scan", "raw"):
        return RawEventEncoder(**shared)
    if kind in ("sparse_cell", "fallback", "cell"):
        return SparseCellEncoder(**shared, **_pick(kw, "cell"))
    if kind in ("event_gnn", "eventgnn", "aegnn"):
        from .event_gnn import EventGNN
        return EventGNN(**shared,
                        **_pick(kw, "k", "n_layers", "max_nodes", "window", "t_scale",
                                "node_attrs", "readout", "sample_mode", "sample_cell",
                                "nbr_mode", "nbr_t_scale"))
    raise ValueError(f"unknown frontend {kind!r}")


def frontend_param_count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())
