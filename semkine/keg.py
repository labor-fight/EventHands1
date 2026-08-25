#!/usr/bin/env python3
r"""S18 KEG: the Kinematic Event Graph frontend.

An event says "brightness changed at pixel `u` at time `t`". LNES answers that by writing `t` into
a dense `(180, 240, 2)` buffer, which overwrites 70-88% of the events (`ARCHITECTURE_AUDIT.md` §7),
quantises `t` to the millisecond, and fixes the window at 50 ms. This module answers it differently:
*which piece of the hand was at `u`, according to the previous state* -- and then aggregates each
piece's events over continuous time.

Three design decisions, each forced by a measurement or a derivation rather than chosen.

**The graph is the kinematic tree, not a pixel k-NN graph.** AEGNN (CVPR'22) treats events as an
evolving graph and updates only the k-hop subgraph a new event touches. Ported literally, the graph
is built by k-NN in `(x, y, t)`, which for our packets means `cdist` over up to 29 255 events -- 8.6e8
pairwise distances per packet, and the resulting neighbourhoods carry no task meaning (two events on
different fingers can be pixel neighbours). Here the graph has 16 nodes, one per MANO joint, and an
event is assigned to the node its pixel's dominant skinning weight names. AEGNN's locality argument
survives -- a new event touches one node, and its influence reaches the rest of the hand along the
tree -- while the node count is fixed and the assignment is what the kinematics actually cares about.

**Temporal aggregation is a diagonal LTI recurrence, not an input-gated one.** Per hidden channel `c`
with rate `lambda_c > 0`, the state obeys `dh_c/dt = -lambda_c h_c` between events and jumps by
`v_{i,c}` at event `i`. Zero-order hold gives the recursion and its closed form:

    h_c(t_i) = e^{-lambda_c dt_i} h_c(t_{i-1}) + v_{i,c},
    h_c(T)   = sum_i e^{-lambda_c (T - t_i)} v_{i,c}.

The recursion is `O(1)` per event, which is the asynchronous deployment mode; the closed form is one
segmented `index_add_`, which is the trainable mode. They are the same operator, so training and
asynchronous inference agree exactly rather than approximately -- the equivalence contract
Messikommer et al. (ECCV'20) make for sparse convolutions.

Keeping the recurrence *linear time-invariant* is not a simplification, it is the claim. Zubic et al.
(CVPR'24) show an LTI continuous-time system can be re-discretised at any step size, which is why an
SSM backbone loses 3.31 mAP across train/test frequency mismatch where an RNN loses 21. An
input-dependent forget gate (`a_i = sigma(W_f x_i) e^{-lambda dt_i}`, as in `encoder.py`'s
`gated_scan`) destroys time-invariance and with it that guarantee. So the gate is dropped, and every
input channel is a function of absolute time differences in seconds -- never of the window length,
which is why `keg_tokens` does not divide `t` by `Delta t` the way `event_tokens` does. The learned
part is the per-event embedding `v_i = W x_i` and the rate bank `lambda`, which is the diagonal state
of an S4D layer with one timescale per channel.

**The per-event lift is geometric, not learned.** The residual an event constrains is the
contour-normal one, `r_i = n_i^T (u_i - pi(X_i(x)))` (S6/S8): motion along an edge produces no
brightness change, so only the normal component is observable. That is the residual every 2024-2026
model-based event tracker minimises (EDFT, LOPET, EDOPT, Event6D) -- all of them rigid. `n_i`,
the surface point, the skinning weights and the part code all come from KSSF evaluated at the
*previous* state, so the lift needs no answer to produce, and the articulated case is the gap.

Everything is off by default. With `kssf=None` the grouping falls back to a coarse spatial grid and
the 12 geometric channels are zero, which is the single-variable ablation G3 needs.
"""
from __future__ import annotations

from typing import Optional, Tuple

import os

import torch
from torch import nn

from .encoder import inter_event_dt, sae_times
from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y
from .gnn import TreeConv, mano_edges

#: MANO joints, wrist included. One graph node each.
N_NODES = 16
#: Events whose pixel carries no surface at the previous state. Not a tree node: nothing in the
#: kinematic model claims to move them, so they are pooled separately and never message-passed.
BG_NODE = N_NODES
N_GROUPS = N_NODES + 1

#: Per-event token width, `keg_tokens`.
TOKEN_DIM = 7
TOKEN_NAMES = ("x_norm", "y_norm", "polarity", "age", "dt_prev", "sae_same", "sae_opp")
#: Per-event geometric lift width, `kssf_event_channels`.
KSSF_CHANNELS = 12
KSSF_NAMES = ("visibility", "sdf_tanh", "normal_x", "normal_y",
              "lbs_w0", "lbs_w1", "lbs_w2", "lbs_w3",
              "sem_radial", "sem_cos", "sem_sin", "sem_depth")

#: Timescale of the bounded token channels, seconds. 10 ms sits inside every window size the S20
#: sweep uses, so the same channel is informative at 5 ms and at 50 ms.
TAU_TOKEN = 1.0e-2
#: "This pixel never fired this polarity in the packet", seconds. A constant, not the packet
#: duration, so the channel does not encode the window length.
SAE_NEVER = 1.0
#: Softening scale of the contour distance channel, pixels. KSSF's band is 12 px wide.
SDF_TANH_PX = 4.0
#: Rate bank limits, 1/seconds. 5 covers a 200 ms context, 2000 covers 500 us.
RATE_MIN, RATE_MAX = 5.0, 2000.0
#: How many nodes one event is shared between under the KSSF routing. MANO's skinning returns four
#: weights per surface point, and the routing is their convex combination.
ROUTE_SLOTS = 4
#: e-folding scale of the continuous visibility kernel `w_geo = exp(-[sdf]_+ / kappa)`, pixels.
#: Half of KSSF's 12 px SDF band: at the band edge the weight has decayed to exp(-2) ~ 0.14, and
#: outside the band the normal field is zero so the halo projection cannot land on the surface at
#: all -- the kernel needs no explicit cutoff. The debug-3eaac2 R3 measurement put the median |sdf|
#: of gate-deleted events at 4-9 px, i.e. inside one to one-and-a-half kappa.
HALO_KAPPA_PX = 6.0
#: Extra step past the contour when projecting an outside event onto the surface, pixels. The
#: closest point `x - d(x) n(x)` lies *on* the contour where the rasterised visibility is
#: fractional; one more pixel along `-n` lands strictly inside so the LBS gather is defined.
HALO_INSET_PX = 1.0
#: Additive guard on the normaliser of the exponentially weighted mean. Small against the mass of a
#: single event, which is `exp(-rate * age) <= 1`, so it perturbs an occupied node negligibly while
#: sending an empty one to exactly zero.
EMA_EPS = 1.0e-6


# ---------------------------------------------------------------------- tokens
def event_age(events: torch.Tensor, delta_t_s: torch.Tensor) -> torch.Tensor:
    """Seconds between each event and the end of its packet, `(N,)`, clamped at zero.

    This is the only place time enters the aggregation, and it is a physical interval: the same
    event 3 ms before the prediction instant produces the same weight whether the window is 5 ms
    or 50 ms long.
    """
    b = events[:, EV_BATCH].long()
    return (delta_t_s[b] - events[:, EV_T]).clamp_min(0.0)


def keg_tokens(events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
               height: int, width: int) -> torch.Tensor:
    """`(N, 7)` rate-invariant event tokens.

    Every temporal channel is `exp(-x / TAU_TOKEN)` of an interval in seconds: bounded in `[0, 1]`,
    monotone, and independent of the window length. `event_tokens` in `encoder.py` normalises time
    by the packet duration instead, which is fine for a fixed 50 ms pipeline and wrong here.
    """
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((0, TOKEN_DIM))
    age = event_age(events, delta_t_s)
    dt_prev = inter_event_dt(events, ptr)
    sae_s, sae_o = sae_times(events, ptr, height, width, fallback=SAE_NEVER)
    return torch.stack([
        events[:, EV_X] / float(width),
        events[:, EV_Y] / float(height),
        2.0 * events[:, EV_P] - 1.0,
        torch.exp(-age / TAU_TOKEN),
        torch.exp(-dt_prev / TAU_TOKEN),
        torch.exp(-sae_s / TAU_TOKEN),
        torch.exp(-sae_o / TAU_TOKEN),
    ], dim=-1)


def kssf_event_channels(fields, events: torch.Tensor, halo: bool = False
                        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Lift events onto the previous state's surface.

    Returns `(N, 12)` channels and a routing `(idx (N, 4), weight (N, 4))` that spreads each event
    over the four joints whose skinning weights MANO assigns to the surface behind it.

    **`halo=True` replaces the visibility hard gate with deletion-free routing (E5.5a).** The
    debug-3eaac2 probes measured what `keep = vis` costs: events fire on moving intensity edges
    (Gallego et al., TPAMI 2022), edges straddle the predicted silhouette, and the gate zeroes all
    twelve channels of the outside half -- vis_frac 0.62 at the ground-truth state on val_core,
    monotone in the state error (0.54 at 6.6 mm, 0.43 at 26 mm, 0.32 at 50 mm), with the deleted
    events at a median |sdf| of 4-9 px, i.e. exactly the contour evidence a tracker needs to
    correct itself. The halo path keeps every event with total routing mass 1:

        pi(x)     = x - (d(x) + 1) n(x)          closest-point projection onto the surface,
                                                  exact to first order in the 12 px band since
                                                  grad d = n with |n| = 1 (signed distance)
        w_geo(x)  = exp(-[d(x)]_+ / kappa)        continuous visibility, kappa = 6 px; the same
                                                  replacement of a binary coverage test by a
                                                  monotone function of the signed distance that
                                                  soft rasterisation uses (SoftRas, Liu et al.,
                                                  ICCV 2019)
        route(x)  = w_geo(x) * sum_k w_k(pi(x)) delta_{j_k(pi(x))} + (1 - w_geo(x)) delta_BG

    Inside events have d <= 0, so w_geo = 1, pi(x) = x, and the routing equals the gated one --
    the change is confined to the events the gate used to delete. The geometry channels (sdf,
    normal) always carry their true values: they *are* the mismatch measurement between the
    events and the predicted contour, which is the correction signal, and were being zeroed
    precisely when they were largest.

    **The routing is soft because the loop is closed.** The obvious assignment is the argmax joint,
    MANO's own answer to "which joint moves this piece of surface", and it is what this frontend did
    first. But the field is read at the *previous predicted* state, so the assignment is a function
    of the estimate, and `argmax` is discontinuous in it: a small state error near a skinning
    boundary moves a whole band of events from one node to another in one jump, the per-joint
    decoders read evidence belonging to a neighbour, and the resulting error feeds back into the
    next field query. Measured at 50 ms on `s20_keg_mixed` s3407, that showed up as an
    error amplification of x2.62 from the single-step regime to the recursive one against x1.91 for
    the dense control -- while the two arms' *single-step* errors were 11.59 mm and 10.95 mm, i.e.
    the representation was already at parity and the entire deficit was closed-loop.

    Convex weights make every node feature continuous in the state, which is what bounds the
    feedback gain: `sum_k w_k(x) f_k` moves by `O(||dx||)` where the argmax moves by `O(1)`. The
    weights are MANO's own `lbs_weights`, so this borrows no new quantity -- the hard version was
    discarding three of the four numbers KSSF already returns.
    """
    n = events.shape[0]
    dev = events.device
    if n == 0:
        z = torch.zeros(0, dtype=torch.long, device=dev)
        return (events.new_zeros((0, KSSF_CHANNELS)),
                (z.view(0, 1), events.new_zeros((0, 1))))
    q = fields.query(events[:, [EV_X, EV_Y]], events[:, EV_BATCH])
    vis = (q["visibility"] > 0.5)
    if halo:
        return _halo_channels(fields, events, q, vis)
    keep = vis.to(events.dtype).unsqueeze(-1)
    ch = torch.cat([
        vis.to(events.dtype).unsqueeze(-1),
        torch.tanh(q["sdf"] / SDF_TANH_PX).unsqueeze(-1),
        q["sdf_normal"],
        q["lbs_weights"],
        q["semantic"],
    ], dim=-1).to(events.dtype) * keep
    idx = q["lbs_indices"].long().clamp(0, N_NODES - 1)
    w = q["lbs_weights"].to(events.dtype).clamp_min(0.0)
    # Renormalise: KSSF returns the top four of MANO's sixteen weights, which need not sum to one.
    # A convex combination is what makes the readout a weighted *mean* rather than a rescaling.
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
    # Inference-time ablation switch for the closed-loop probe: collapse the convex routing back to
    # the argmax joint so hard-vs-soft can be compared on the *same* checkpoint, which is the
    # highest-priority evidence class in FAILURE_AND_CLEANUP_LEDGER §0. KSSF returns the top-4
    # weights sorted, so slot 0 is the argmax. Training never sets this variable.
    if os.environ.get("EVENTHANDS_KEG_ROUTE", "soft") == "hard":
        w = torch.zeros_like(w)
        w[:, 0] = 1.0
    # Events with no surface behind them carry no kinematic claim, so they go entirely to the
    # background node, whose geometric channels `keep` has already zeroed.
    bg = torch.full_like(idx[:, :1], BG_NODE)
    idx = torch.where(vis.unsqueeze(-1), idx, torch.cat([bg] * idx.shape[1], dim=-1))
    onehot = torch.zeros_like(w)
    onehot[:, 0] = 1.0
    w = torch.where(vis.unsqueeze(-1), w, onehot)
    return ch, (idx, w)


def _halo_channels(fields, events: torch.Tensor, q: dict, vis: torch.Tensor
                   ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Deletion-free lift: see `kssf_event_channels` (halo=True). Routing is `(N, 5)`:
    four surface joints scaled by `w_geo` plus the background node holding `1 - w_geo`."""
    dt = events.dtype
    sdf_pos = q["sdf"].clamp_min(0.0)
    xy_h = events[:, [EV_X, EV_Y]] - (sdf_pos + HALO_INSET_PX).unsqueeze(-1) * q["sdf_normal"]
    qh = fields.query(xy_h, events[:, EV_BATCH])
    # Outside events whose projection lands on the surface. Beyond the SDF band the normal is
    # zero, the projection stays put, visibility stays 0, and the event falls through to BG.
    hit = ~vis & (qh["visibility"] > 0.5)
    w_geo = torch.where(vis, torch.ones_like(sdf_pos),
                        torch.exp(-sdf_pos / HALO_KAPPA_PX) * hit.to(sdf_pos.dtype)).to(dt)
    pick = hit.unsqueeze(-1)
    keep = vis.to(dt).unsqueeze(-1)
    # Surface identity comes from the halo point for outside events; the mismatch geometry
    # (sdf, normal) keeps its true value everywhere instead of being zeroed with the gate.
    lbs_ch = torch.where(pick, qh["lbs_weights"], q["lbs_weights"] * keep)
    sem_ch = torch.where(pick, qh["semantic"], q["semantic"] * keep)
    ch = torch.cat([
        w_geo.unsqueeze(-1),
        torch.tanh(q["sdf"] / SDF_TANH_PX).unsqueeze(-1),
        q["sdf_normal"],
        lbs_ch,
        sem_ch,
    ], dim=-1).to(dt)
    src_idx = torch.where(pick, qh["lbs_indices"].long(),
                          q["lbs_indices"].long()).clamp(0, N_NODES - 1)
    src_w = torch.where(pick, qh["lbs_weights"], q["lbs_weights"]).to(dt).clamp_min(0.0)
    src_w = src_w / src_w.sum(-1, keepdim=True).clamp_min(1e-8)
    # Same inference-time ablation contract as the gated path: collapse the surface slots to the
    # argmax joint; w_geo and the BG slot are unaffected because they are not a routing choice.
    if os.environ.get("EVENTHANDS_KEG_ROUTE", "soft") == "hard":
        src_w = torch.zeros_like(src_w)
        src_w[:, 0] = 1.0
    idx = torch.cat([src_idx, torch.full_like(src_idx[:, :1], BG_NODE)], dim=-1)
    w = torch.cat([w_geo.unsqueeze(-1) * src_w, (1.0 - w_geo).unsqueeze(-1)], dim=-1)
    return ch, (idx, w)


def spatial_groups(events: torch.Tensor, height: int, width: int, grid: int = 4
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback routing: a `grid x grid` tiling of the frame, as a one-hot `(idx, weight)`.

    Used when KSSF is off. It has the same node count as the kinematic routing and the same
    aggregation downstream, so the G3 ablation isolates *what the groups mean* rather than
    conflating that with how many there are. It is state-independent by construction, so a single
    hard assignment carries no closed-loop risk and there is nothing to soften.
    """
    if events.shape[0] == 0:
        z = torch.zeros(0, dtype=torch.long, device=events.device)
        return z.view(0, 1), events.new_zeros((0, 1))
    gx = (events[:, EV_X] / float(width) * grid).long().clamp(0, grid - 1)
    gy = (events[:, EV_Y] / float(height) * grid).long().clamp(0, grid - 1)
    idx = (gy * grid + gx).clamp(0, N_NODES - 1)
    return idx.view(-1, 1), torch.ones_like(idx, dtype=events.dtype).view(-1, 1)


# ------------------------------------------------------------------ aggregation
def lti_aggregate(values: torch.Tensor, age: torch.Tensor, seg: torch.Tensor,
                  n_seg: int, rate: torch.Tensor,
                  weight: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    r"""Segmented diagonal-LTI readout. `values (N, H)`, `age (N,)`, `rate (H,)`.

    The recurrence is the zero-order-hold one, whose closed form is

        raw[s, c] = sum_{i in s} exp(-rate_c * age_i) * values[i, c],
        mass[s, c] = sum_{i in s} exp(-rate_c * age_i),

    and the returned `ema` is `raw / mass`, the same kernel normalised by itself. The unnormalised
    `raw` is *proportional to how many events the segment holds*, which on this data is a nuisance
    variable spanning 20x between validation sequences and another 10x across the 5-50 ms mixed
    window schedule; measured on a trained checkpoint it put four orders of magnitude between the
    median and the maximum of one feature block while the other blocks stayed inside [0, 9]. The
    ratio is bounded by `max_i |values[i, c]|` and carries the *shape* of the temporal profile with
    the rate divided out. Event mass is not discarded, it is returned separately as `mass` for the
    caller to compress.

    Normalising also strengthens the time-invariance this frontend exists for. `raw` is invariant
    only because ages are measured from the window end; shifting every age by a common `d` scales
    `raw` by `exp(-rate_c d)`. It scales `mass` by the same factor, so `ema` is invariant under any
    common shift, exactly.

    `weight (N,)` is an optional per-event routing weight, folded into the same kernel so that both
    the numerator and the normaliser see it and the result stays a weighted mean. It is what lets an
    event be shared between joints instead of assigned to one, and it enters every statistic so that
    a weakly claimed event is weakly present everywhere rather than fully present in one place.

    Also returns a scale-free peak and a count. One `index_add_` per statistic regardless of how
    many events a segment holds, so the cost is `O(N)` with no sequential steps -- the reason this
    replaces the `gated_scan` of `encoder.py`, whose padded loop runs for `max_i N_i` steps and is
    not trainable at 29 000 events per packet.

    `index_add_` accumulates with CUDA atomics, so the sum is not bitwise reproducible: measured at
    2.0e-7 relative for the batch sizes here (`test_aggregation_jitter_is_below_float32_epsilon`).
    `ARCHITECTURE_AUDIT.md` §5 registers bitwise-deterministic *inference* as the reason paired
    comparisons can use thresholds far below the 0.4 mm replicate floor, so this frontend weakens
    that guarantee and the weakening is measured rather than assumed: S19 re-measures it end to end
    on a trained checkpoint before any paired threshold is applied.
    """
    H = values.shape[-1]
    dev = values.device
    # Never below float32, whatever autocast is doing. A packet can hold 29 000 events, and bfloat16's
    # eight mantissa bits would lose every contribution more than 2^-8 of the running sum -- the
    # long tail of an exponentially weighted sum is exactly what this operator is for. `exp` of a
    # rate-time product in bfloat16 is likewise good to about two decimal digits.
    work = torch.float64 if torch.float64 in (values.dtype, age.dtype, rate.dtype) \
        else torch.float32
    values, age, rate = values.to(work), age.to(work), rate.to(work)
    raw = torch.zeros(n_seg, H, device=dev, dtype=work)
    mass = torch.zeros(n_seg, H, device=dev, dtype=work)
    peak = torch.zeros(n_seg, H, device=dev, dtype=work)
    count = torch.zeros(n_seg, device=dev, dtype=work)
    if values.shape[0]:
        w = torch.exp(-rate.unsqueeze(0) * age.unsqueeze(-1))
        rw = torch.ones_like(age) if weight is None else weight.to(work)
        w = w * rw.unsqueeze(-1)
        idx = seg.unsqueeze(-1).expand(-1, H)
        raw = raw.index_add(0, seg, values * w)
        mass = mass.index_add(0, seg, w)
        # `amax` with `include_self` leaves empty segments at the zero they were initialised to,
        # which is the right neutral value: an unobserved joint must look like no evidence, not
        # like -inf leaking into the projection.
        peak = peak.scatter_reduce(0, idx, values * rw.unsqueeze(-1),
                                   reduce="amax", include_self=True)
        count = count.index_add(0, seg, rw)
    # An empty segment has zero mass and must read as no evidence rather than as a division by
    # something tiny, so the guard is additive and the result is exactly zero there.
    ema = raw / (mass + EMA_EPS)
    return {"ema": ema, "raw": raw, "mass": mass, "peak": peak, "count": count}


def route_aggregate(values: torch.Tensor, age: torch.Tensor, events: torch.Tensor,
                    idx: torch.Tensor, weight: torch.Tensor, batch: int,
                    rate: torch.Tensor) -> Dict[str, torch.Tensor]:
    """`lti_aggregate` under a `(N, K)` soft routing, summed over the `K` destinations.

    Runs one pass per routing slot rather than materialising `K` copies of the token bank: with 29k
    events, 64 channels and `K = 4` the copy would be a third of a gigabyte per packet and buys
    nothing, since the four passes touch disjoint gradients of the same buffers.

    Summing `raw` and `mass` across slots before dividing is what keeps the result a weighted mean:
    a node's feature is `sum_i w_i k_i v_i / sum_i w_i k_i` over exactly the events routed to it,
    whatever fraction of each event that is.
    """
    H = values.shape[-1]
    dev = values.device
    work = torch.float64 if torch.float64 in (values.dtype, age.dtype, rate.dtype) \
        else torch.float32
    n_seg = batch * N_GROUPS
    raw = torch.zeros(n_seg, H, device=dev, dtype=work)
    mass = torch.zeros(n_seg, H, device=dev, dtype=work)
    peak = torch.zeros(n_seg, H, device=dev, dtype=work)
    count = torch.zeros(n_seg, device=dev, dtype=work)
    if values.shape[0]:
        base = events[:, EV_BATCH].long() * N_GROUPS
        for k in range(idx.shape[1]):
            part = lti_aggregate(values, age, base + idx[:, k], n_seg, rate, weight[:, k])
            raw = raw + part["raw"]
            mass = mass + part["mass"]
            count = count + part["count"]
            peak = torch.maximum(peak, part["peak"])
    return {"ema": raw / (mass + EMA_EPS), "raw": raw, "mass": mass,
            "peak": peak, "count": count}


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """`x` such that `softplus(x) == y`, stable for large `y`.

    The direct form `log(expm1(y))` overflows in float32 above about 88, and the fast rates in the
    bank reach 2000, so the whole top of the bank would initialise to infinity.
    """
    return y + torch.log(-torch.expm1(-y))


class KinematicEventGraph(nn.Module):
    """KEG frontend. `(events, ptr, delta_t_s)` in, one feature vector per packet out.

    The pose head stays in `MNISTModel` / the S10 per-joint decoders, same rule as
    `RawEventEncoder`: this module may not contain a second path to a finger angle.
    """

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 64,
                 feat_dim: int = 256, node_dim: int = 64, n_layers: int = 2,
                 extra_channels: int = 0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.hidden = int(hidden)
        self.feat_dim = int(feat_dim)
        self.node_dim = int(node_dim)
        #: set by `MNISTModel` when `MODEL.ENCODER_KSSF` is on; widens the token by 12
        self.kssf_channels = int(extra_channels)
        in_dim = TOKEN_DIM + self.kssf_channels
        self.embed = nn.Linear(in_dim, self.hidden)
        # One timescale per hidden channel, log-spaced over the bank, i.e. the diagonal state of an
        # S4D layer. Parameterised through softplus so a rate can never cross zero and turn the
        # decay into growth. Stored as the pre-activation so `softplus(log_rate) == rate` at init.
        rates = torch.logspace(torch.log10(torch.tensor(RATE_MIN)),
                               torch.log10(torch.tensor(RATE_MAX)), self.hidden)
        self.log_rate = nn.Parameter(_inv_softplus(rates))
        self.node_proj = nn.Linear(3 * self.hidden + 1, self.node_dim)
        self.layers = nn.ModuleList([TreeConv(self.node_dim) for _ in range(int(n_layers))])
        parents = torch.tensor([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
        self.register_buffer("edges", mano_edges(parents.long()), persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(3 * self.node_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    @property
    def rate(self) -> torch.Tensor:
        """Positive decay rates, 1/seconds, `(hidden,)`."""
        return torch.nn.functional.softplus(self.log_rate)

    def node_features(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                      extra: Optional[torch.Tensor] = None,
                      groups: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """`(B, 17, node_dim)` per-group features before message passing.

        `groups` is a routing `(idx (N, K), weight (N, K))`; `None` falls back to the spatial grid.
        """
        B = int(ptr.numel() - 1)
        tok = keg_tokens(events, ptr, delta_t_s, self.height, self.width)
        if self.kssf_channels:
            if extra is None or extra.shape[0] != tok.shape[0]:
                extra = tok.new_zeros((tok.shape[0], self.kssf_channels))
            tok = torch.cat([tok, extra], dim=-1)
        if groups is None:
            groups = spatial_groups(events, self.height, self.width)
        idx, wr = groups
        v = self.embed(tok)
        age = event_age(events, delta_t_s) if events.shape[0] else events.new_zeros(0)
        agg = route_aggregate(v, age, events, idx, wr, B, self.rate)
        # Shape, extremum, and mass, each on its own scale: the exponentially weighted mean and the
        # peak are bounded by the token embedding, and both event-mass channels are compressed by
        # `log1p` so that a node holding a hundred times more events differs by a constant rather
        # than by a factor of a hundred.
        feat = torch.cat([agg["ema"], agg["peak"], torch.log1p(agg["mass"]),
                          torch.log1p(agg["count"]).unsqueeze(-1)], dim=-1)
        return self.node_proj(feat.to(v.dtype)).view(B, N_GROUPS, self.node_dim)

    def forward_nodes(self, events, ptr, delta_t_s, extra=None, groups=None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """`(feature (B, feat_dim), joint nodes (B, 16, node_dim))`."""
        nodes = self.node_features(events, ptr, delta_t_s, extra, groups)
        h = nodes[:, :N_NODES]
        bg = nodes[:, BG_NODE]
        for layer in self.layers:
            h = h + layer(h, self.edges)
        pooled = torch.cat([h.mean(1), h.max(1).values, bg], dim=-1)
        return self.proj(pooled), h

    def forward(self, events, ptr, delta_t_s, extra: Optional[torch.Tensor] = None,
                groups: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        return self.forward_nodes(events, ptr, delta_t_s, extra, groups)[0]
