"""Finite phase-preserving event evidence for a fixed trajectory bank.

This is a NumPy implementation of the finite model in S43表征调研.md, part 8.
Integer *absolute* predicted log-brightness paths are conditions supplied by an
adapter, never measurements.  A likelihood here is only a likelihood under that
declared finite model; the module makes no claim about a real camera model.

Shapes: paths [K,M,P,B*S+1], counts [B,P,2], phi [K,M,P,Q].  Channel 0 is model
ON and channel 1 model OFF.  An adapter with unverified stored polarity must say
so in ``polarity_convention`` and must not claim physical ON/OFF calibration.

The state is bounded to one epoch.  Ingest only stages observations; advance
consumes certified complete bins; query is read-only.  A clock-only advance is
not free: an empty live bin still receives its normalized observation score.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import sys
from typing import Iterable

import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def logsumexp(x, axis=None):
    """A NumPy-only logsumexp which also supports all-negative-infinity rows."""
    x = np.asarray(x, dtype=np.float64)
    mx = np.max(x, axis=axis, keepdims=True)
    finite = np.isfinite(mx)
    shifted = np.full_like(x, -np.inf)
    np.subtract(x, mx, out=shifted, where=np.broadcast_to(finite, x.shape))
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = np.full_like(mx, -np.inf)
    np.log(total, out=out, where=total > 0)
    np.add(out, mx, out=out, where=finite)
    if axis is None:
        return float(out.reshape(-1)[0])
    return np.squeeze(out, axis=axis)


def _normalized_weights(values, shape, name):
    out = np.array(values, dtype=np.float64, copy=True)
    if out.shape != shape or not np.isfinite(out).all() or np.any(out < 0):
        raise ValueError(f"{name} must be finite nonnegative with shape {shape}")
    norm = out.sum(axis=-1, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError(f"{name} has a zero-mass row")
    return out / norm


def _binomial_logpmf(n, d, alpha):
    if d < 0 or d > n:
        return -math.inf
    if alpha == 0:
        return 0.0 if d == 0 else -math.inf
    if alpha == 1:
        return 0.0 if d == n else -math.inf
    # lgamma(n+1)-lgamma(n-d+1) loses digits for n>>d.  Small d is
    # exactly the low-count regime used in the capped sensor observation.
    short = min(d, n - d)
    if short < 128:
        choose = math.fsum(math.log(n - j) for j in range(short)) - math.lgamma(short + 1)
    else:
        choose = math.lgamma(n + 1) - math.lgamma(d + 1) - math.lgamma(n - d + 1)
    return choose + d * math.log(alpha) + (n - d) * math.log1p(-alpha)


def _poisson_survival(threshold, nu):
    """P(Poisson(nu)>=threshold), without subtracting a tiny tail from one."""
    if threshold <= 0:
        return 1.0
    if nu == 0:
        return 0.0
    cdf = math.fsum(math.exp(-nu + j * math.log(nu) - math.lgamma(j + 1))
                    for j in range(threshold))
    if cdf < .5:
        return max(0.0, 1.0 - cdf)
    term = math.exp(-nu + threshold * math.log(nu) - math.lgamma(threshold + 1))
    total, j = term, threshold
    # In this branch threshold is at/above the central mass, so the
    # positive-tail recurrence decreases rapidly.  Truncate only after
    # its additional relative contribution is below double precision.
    while term > 0:
        j += 1
        term *= nu / j
        total += term
        if term <= total * 2e-17:
            break
    return min(1.0, total)


def _binomial_survival(n, alpha, threshold):
    """P(Binomial(n,alpha)>=threshold), stable even for n large, alpha tiny."""
    if threshold <= 0:
        return 1.0
    if threshold > n or alpha == 0:
        return 0.0
    if alpha == 1:
        return 1.0
    cdf = math.fsum(math.exp(_binomial_logpmf(n, j, alpha)) for j in range(threshold))
    if cdf < .5:
        return max(0.0, 1.0 - cdf)
    term = math.exp(_binomial_logpmf(n, threshold, alpha))
    total, j = term, threshold
    while term > 0 and j < n:
        term *= ((n - j) / (j + 1)) * alpha / (1.0 - alpha)
        j += 1
        total += term
        if term <= total * 2e-17:
            break
    return min(1.0, total)


@lru_cache(maxsize=512)
def _cached_count_pmf(n: int, alpha: float, nu: float, cap: int):
    """min(Binomial(n,alpha)+Poisson(nu),cap), including overflow mass.

    The latent trigger count is never clipped before readout thinning.  Log
    combinatorics avoid overflow for larger latent counts.  This cache has a
    fixed entry limit and is included as fixed shared overhead in the budget.
    """
    values = []
    for y in range(cap):
        terms = []
        for d in range(min(n, y) + 1):
            lb = _binomial_logpmf(n, d, alpha)
            rest = y - d
            lp = (0.0 if rest == 0 else -math.inf) if nu == 0 else (
                -nu + rest * math.log(nu) - math.lgamma(rest + 1))
            terms.append(lb + lp)
        finite_terms = [v for v in terms if math.isfinite(v)]
        if finite_terms:
            m = max(finite_terms)
            values.append(math.exp(m) * sum(math.exp(v - m) for v in finite_terms))
        else:
            values.append(0.0)
    # Split on the finite binomial count.  For d>=cap the overflow is
    # certain; for d<cap retain the strictly positive Poisson survival.
    # Using 1-sum(values) here would turn nu=.02,cap=8 into a false zero.
    tail = _binomial_survival(n, alpha, cap) + math.fsum(
        math.exp(_binomial_logpmf(n, d, alpha)) * _poisson_survival(cap - d, nu)
        for d in range(min(n, cap - 1) + 1))
    if not 0 <= tail <= 1.0 + 1e-10:
        raise ArithmeticError("invalid readout survival probability")
    values.append(min(1.0, tail))
    return tuple(values)


def count_pmf(n, alpha=.9, nu=.02, cap=3):
    n, cap = _integer(n, "latent count"), _integer(cap, "cap")
    if n < 0 or cap < 1 or not 0 <= alpha <= 1 or not np.isfinite(nu) or nu < 0:
        raise ValueError("invalid count readout parameters")
    return np.asarray(_cached_count_pmf(n, float(alpha), float(nu), cap))


@dataclass(frozen=True)
class Event:
    uid: int
    t_us: int
    pixel: int
    polarity: int


class GridKernel:
    """Immutable trajectory conditions; paths are owned and read-only.

    The last S samples are internal discrete brightness jumps belonging to
    their bin.  This is a defined finite sensor law, not an exact discretization
    of arbitrary continuous threshold-crossing boundary times.
    """

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("kernel is immutable; construct a new bank and replay")
        object.__setattr__(self, name, value)

    def __init__(self, paths, bins=None, substeps=1, ap=2, am=2,
                 alpha=.9, nu=.02, cap=3, context_id="c0",
                 polarity_convention="physical_on_off"):
        self.S = _integer(substeps, "substeps")
        self.ap, self.am = _integer(ap, "ap"), _integer(am, "am")
        self.cap = _integer(cap, "cap")
        a = np.asarray(paths)
        if a.ndim != 4 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError("paths must be integer [K,M,P,B*S+1]")
        self.K, self.M, self.P, steps = a.shape
        if not 2 <= self.K <= 8 or min(self.M, self.P) < 1:
            raise ValueError("K=2..8 and positive M/P required")
        if min(self.S, self.ap, self.am, self.cap) < 1:
            raise ValueError("positive finite model dimensions required")
        if bins is None:
            if (steps - 1) % self.S:
                raise ValueError("path length not divisible by substeps")
            bins = (steps - 1) // self.S
        self.B = _integer(bins, "bins")
        if self.B < 1 or steps != self.B * self.S + 1:
            raise ValueError("path shape does not match bins/substeps")
        self.paths = np.array(a, dtype=np.int64, copy=True)
        self.paths.flags.writeable = False
        self.phases = np.arange(-self.am + 1, self.ap, dtype=np.int64)
        self.Q = len(self.phases)
        self.alpha = np.broadcast_to(np.asarray(alpha, dtype=np.float64), (self.M,)).copy()
        self.nu = np.broadcast_to(np.asarray(nu, dtype=np.float64), (self.M,)).copy()
        if (not np.isfinite(self.alpha).all() or not np.isfinite(self.nu).all()
                or np.any((self.alpha < 0) | (self.alpha > 1)) or np.any(self.nu < 0)):
            raise ValueError("invalid per-nuisance readout law")
        self.alpha.flags.writeable = self.nu.flags.writeable = False
        self.phases.flags.writeable = False
        self.context_id = str(context_id)
        self.polarity_convention = str(polarity_convention)
        self.static_id = (self.K, self.M, self.P, self.B, self.S, self.ap, self.am,
                          self.cap, tuple(self.alpha), tuple(self.nu), self.context_id,
                          self.polarity_convention)
        h = hashlib.sha256(repr(self.static_id).encode())
        h.update(self.paths.tobytes())
        self.fingerprint = h.hexdigest()
        self._sealed = True

    def predict(self, b):
        """Return destination phase index and ON/OFF latent counts [K,M,P,Q]."""
        b = _integer(b, "bin")
        if not 0 <= b < self.B:
            raise ValueError("bin outside epoch")
        r = np.broadcast_to(self.phases, (self.K, self.M, self.P, self.Q)).copy()
        plus, minus = np.zeros_like(r), np.zeros_like(r)
        for j in range(self.S):
            i = b * self.S + j
            delta = self.paths[..., i + 1] - self.paths[..., i]
            h = r + delta[..., None]
            pn = np.where(h >= self.ap, h // self.ap, 0)
            mn = np.where(h <= -self.am, (-h) // self.am, 0)
            r = h - pn * self.ap + mn * self.am
            plus += pn
            minus += mn
        if np.any((r < -self.am + 1) | (r >= self.ap)):
            raise ArithmeticError("phase transition escaped finite support")
        return r + self.am - 1, plus, minus

    def observation_log_prob(self, predicted_counts, observed):
        """Readout mass [K,M,P,Q], integrating thinning and additive noise."""
        if observed.shape != (self.P,):
            raise ValueError("observed pixel shape")
        out = np.full(predicted_counts.shape, -np.inf, dtype=np.float64)
        for m in range(self.M):
            n = predicted_counts[:, m]
            unique, indices = np.unique(n, return_inverse=True)
            table = np.asarray([_cached_count_pmf(int(v), float(self.alpha[m]),
                                                  float(self.nu[m]), self.cap) for v in unique])
            ix = indices.reshape(n.shape)
            pm = table[ix, observed[None, :, None]]
            np.log(pm, out=out[:, m], where=pm > 0)
        return out


class EvidenceMemory:
    """One bounded epoch; repeated reads never consume the same evidence again."""

    def __init__(self, kernel, omega=None, phase=None, pi=None, *, start_us=0,
                 bin_us=1000, max_sources=100000, max_refresh=1):
        if not isinstance(kernel, GridKernel):
            raise TypeError("GridKernel required")
        self.kernel = kernel
        k = kernel
        self.start_us, self.bin_us = _integer(start_us, "start_us"), _integer(bin_us, "bin_us")
        self.max_sources, self.max_refresh = _integer(max_sources, "max_sources"), _integer(max_refresh, "max_refresh")
        if self.bin_us < 1 or self.max_sources < 1 or self.max_refresh < 0:
            raise ValueError("positive time/source bounds and nonnegative refresh bound required")
        weights = np.ones((k.K, k.M)) if omega is None else omega
        weights = _normalized_weights(weights, (k.K, k.M), "omega")
        priors = np.ones(k.K) if pi is None else pi
        self.pi = _normalized_weights(priors, (k.K,), "pi")
        self.logomega = np.full(weights.shape, -np.inf)
        np.log(weights, out=self.logomega, where=weights > 0)
        initial = np.ones((k.K, k.M, k.P, k.Q)) if phase is None else phase
        self.initial_phase = _normalized_weights(initial, (k.K, k.M, k.P, k.Q), "phase")
        self.logphi = np.full(self.initial_phase.shape, -np.inf)
        np.log(self.initial_phase, out=self.logphi, where=self.initial_phase > 0)
        self.ell = np.zeros((k.K, k.M), dtype=np.float64)
        self.counts = np.zeros((k.B, k.P, 2), dtype=np.int64)
        self.live = np.ones((k.B, k.P), dtype=bool)
        self.seen = {}
        self.end = self.version = self.refresh_count = 0

    @property
    def phi(self):
        return np.exp(self.logphi)

    @property
    def cutoff_us(self):
        return self.start_us + self.end * self.bin_us

    def ingest(self, events: Iterable[Event]):
        """Stage unique source observations; validate entire packet before mutation."""
        incoming = {}
        k = self.kernel
        for e in events:
            if not isinstance(e, Event):
                raise TypeError("Event required")
            uid = _integer(e.uid, "source uid")
            stamp = _integer(e.t_us, "event t_us")
            pixel, polarity = _integer(e.pixel, "pixel"), _integer(e.polarity, "polarity")
            if uid < 0 or not 0 <= pixel < k.P or polarity not in (0, 1):
                raise ValueError("invalid source, pixel, or polarity")
            payload = (stamp, pixel, polarity)
            old = self.seen.get(uid, incoming.get(uid))
            if old is not None:
                if old != payload:
                    raise ValueError("source ID payload conflict")
                continue
            b = (stamp - self.start_us) // self.bin_us
            if not self.end <= b < k.B:
                raise ValueError("new event is late or outside epoch")
            incoming[uid] = payload
        if len(self.seen) + len(incoming) > self.max_sources:
            raise OverflowError("source cap exceeded; no observations silently dropped")
        for uid, payload in incoming.items():
            stamp, pixel, polarity = payload
            b = (stamp - self.start_us) // self.bin_us
            self.counts[b, pixel, polarity] = min(k.cap, self.counts[b, pixel, polarity] + 1)
            self.seen[uid] = payload
        return len(incoming)

    def _consume_bin(self, b):
        k = self.kernel
        qnext, plus, minus = k.predict(b)
        lp = (k.observation_log_prob(plus, self.counts[b, :, 0])
              + k.observation_log_prob(minus, self.counts[b, :, 1]))
        lp = np.where(self.live[b][None, None, :, None], lp, 0.0)
        contribution = (self.logphi + lp).reshape(-1, k.Q)
        destinations = qnext.reshape(-1, k.Q)
        dest = np.full_like(contribution, -np.inf)
        rows = np.arange(len(dest))
        for q in range(k.Q):
            np.logaddexp.at(dest, (rows, destinations[:, q]), contribution[:, q])
        dest = dest.reshape(self.logphi.shape)
        logc = logsumexp(dest, axis=-1)
        self.logphi.fill(-np.inf)
        np.subtract(dest, logc[..., None], out=self.logphi,
                    where=np.broadcast_to(np.isfinite(logc)[..., None], dest.shape))
        self.ell += logc.sum(axis=-1)
        self.end = b + 1
        self.version += 1

    def advance(self, cutoff_us, *, watermark_us, live=None):
        """Consume complete bins only, with an explicit certified input watermark.

        ``live`` is [B,P] (acquisition available), independent of event occupancy.
        All shape/dropout errors are detected before advancing any bin.
        """
        cutoff = _integer(cutoff_us, "cutoff_us")
        watermark = _integer(watermark_us, "watermark_us")
        distance = cutoff - self.start_us
        if distance % self.bin_us or not self.cutoff_us <= cutoff <= self.start_us + self.kernel.B * self.bin_us:
            raise ValueError("nonmonotone, unaligned, or outside-epoch cutoff")
        if watermark < cutoff:
            raise ValueError("prefix not certified by watermark")
        end = distance // self.bin_us
        mask = self.live.copy() if live is None else np.asarray(live)
        if mask.shape != self.live.shape or mask.dtype != np.bool_:
            raise ValueError("live must be boolean [B,P]")
        if not np.array_equal(mask[:self.end], self.live[:self.end]):
            raise ValueError("cannot rewrite consumed acquisition validity")
        if np.any((~mask[self.end:end]) & np.any(self.counts[self.end:end] != 0, axis=-1)):
            raise ValueError("a missing bin also contains observations")
        self.live[self.end:end] = mask[self.end:end]
        for b in range(self.end, end):
            self._consume_bin(b)
        return self

    def likelihoods(self):
        return logsumexp(self.logomega + self.ell, axis=-1)

    def query(self, coordinates=None, *, cutoff_us=None, target_us=None):
        """Read evidence at the current consumed prefix; never advance or commit.

        ``log_weights`` includes nuisance prior logomega once.  ``ell`` is the
        conditional evidence alone.  A changed numerical coordinate reference
        changes only the coordinate columns, not likelihoods or phase.
        """
        if cutoff_us is not None and _integer(cutoff_us, "query cutoff") != self.cutoff_us:
            raise ValueError("historical/future query requires its own prefix snapshot")
        target = self.cutoff_us if target_us is None else _integer(target_us, "target_us")
        if not self.start_us <= target <= self.cutoff_us:
            raise ValueError("target outside available causal prefix")
        k = self.kernel
        coords = np.zeros((k.K, 51)) if coordinates is None else np.asarray(coordinates, dtype=np.float64)
        if coords.shape != (k.K, 51) or not np.isfinite(coords).all():
            raise ValueError("coordinates must be finite [K,51]")
        ll = self.likelihoods()
        valid = np.isfinite(ll)
        relative = np.zeros(k.K)
        if valid.any():
            relative[valid] = ll[valid] - np.max(ll[valid])
        logpi = np.full(k.K, -np.inf)
        np.log(self.pi, out=logpi, where=self.pi > 0)
        score = logpi + ll
        norm = logsumexp(score)
        posterior = np.exp(score - norm) if np.isfinite(norm) else np.zeros(k.K)
        return {"status": "ok" if np.isfinite(norm) else "unsupported",
                "R": np.column_stack([coords, relative, valid.astype(float)]),
                "log_likelihood": ll.copy(), "log_weights": (self.logomega + self.ell).copy(),
                "ell": self.ell.copy(), "phi": self.phi, "posterior": posterior,
                "valid": valid.copy(), "posterior_defined": bool(np.isfinite(norm)),
                "cutoff_us": self.cutoff_us, "target_us": target, "version": self.version,
                "model_id": k.fingerprint, "polarity_convention": k.polarity_convention}

    def replay(self, kernel):
        """Replace the same retained evidence under changed window trajectories.

        The physical boundary, sensor law and nuisance identities must remain
        fixed.  A changed boundary requires an explicit restart, not this method.
        No old likelihood increment is multiplied into the replacement.
        """
        if self.refresh_count >= self.max_refresh:
            raise OverflowError("one-epoch refresh budget exceeded")
        if kernel.static_id != self.kernel.static_id or not np.array_equal(kernel.paths[..., 0], self.kernel.paths[..., 0]):
            raise ValueError("replay cannot silently change physical boundary/model identity")
        new = EvidenceMemory(kernel, omega=np.exp(self.logomega), phase=self.initial_phase,
                             pi=self.pi, start_us=self.start_us, bin_us=self.bin_us,
                             max_sources=self.max_sources, max_refresh=self.max_refresh)
        new.counts[:] = self.counts
        new.seen = self.seen.copy()
        new.advance(self.cutoff_us, watermark_us=self.cutoff_us, live=self.live)
        new.refresh_count = self.refresh_count + 1
        return new

    def _sensor_id(self):
        k = self.kernel
        return (k.M, k.P, k.S, k.ap, k.am, k.cap, tuple(k.alpha), tuple(k.nu),
                k.polarity_convention, self.bin_us)

    def branch_handoff(self, selected):
        """Conditional boundary after a hard branch choice; globally approximate.

        The nuisance posterior and pixel phase are preserved, but unselected
        path histories are discarded.  This is not a full-stream Bayes filter.
        """
        selected = _integer(selected, "selected candidate")
        if self.end != self.kernel.B or not 0 <= selected < self.kernel.K:
            raise ValueError("handoff requires an existing branch at epoch end")
        ll = self.likelihoods()
        if self.pi[selected] == 0 or not np.isfinite(ll[selected]):
            raise ValueError("cannot hand off an unsupported/zero-prior branch")
        lw = self.logomega[selected] + self.ell[selected] - ll[selected]
        return {"omega": np.exp(lw), "phase": self.phi[selected].copy(),
                "cutoff_us": self.cutoff_us, "bin_us": self.bin_us,
                "last_levels": self.kernel.paths[selected, :, :, -1].copy(),
                "sensor_id": self._sensor_id(), "selected": selected,
                "from_model_id": self.kernel.fingerprint,
                "approximation": "hard selected trajectory; discarded other histories"}

    @classmethod
    def from_handoff(cls, kernel, handoff, *, pi=None, max_sources=100000, max_refresh=1):
        """Start a fresh bounded epoch with a physically matching sensor boundary."""
        memory = cls(kernel, omega=np.broadcast_to(handoff["omega"], (kernel.K, kernel.M)),
                     phase=np.broadcast_to(handoff["phase"], (kernel.K, kernel.M, kernel.P, kernel.Q)),
                     pi=pi, start_us=handoff["cutoff_us"], bin_us=handoff["bin_us"],
                     max_sources=max_sources, max_refresh=max_refresh)
        if memory._sensor_id() != handoff["sensor_id"]:
            raise ValueError("handoff changed sensor/nuisance identity")
        if not np.array_equal(kernel.paths[..., 0], np.broadcast_to(
                handoff["last_levels"], (kernel.K, kernel.M, kernel.P))):
            raise ValueError("next paths do not match the physical brightness boundary")
        return memory

    def memory_bytes(self):
        """Resident array/source budget, not Python process RSS or peak workspace."""
        arrays = (self.kernel.paths, self.kernel.phases, self.kernel.alpha, self.kernel.nu,
                  self.pi, self.logomega, self.initial_phase, self.logphi, self.ell,
                  self.counts, self.live)
        source_bytes = sys.getsizeof(self.seen) + sum(
            sys.getsizeof(uid) + sys.getsizeof(payload) + sum(sys.getsizeof(v) for v in payload)
            for uid, payload in self.seen.items())
        return {"array_bytes": int(sum(a.nbytes for a in arrays)),
                "source_bytes": int(source_bytes), "source_count": len(self.seen),
                "source_cap": self.max_sources, "pmf_cache_max_entries": 512,
                "total_accounted_bytes": int(sum(a.nbytes for a in arrays) + source_bytes)}

    def digest(self):
        h = hashlib.sha256(self.kernel.fingerprint.encode())
        for a in (self.logphi, self.ell, self.counts, self.live, self.pi, self.logomega):
            h.update(a.tobytes())
        h.update(json.dumps([self.end, self.version, self.refresh_count,
                             sorted(self.seen.items())], separators=(",", ":")).encode())
        return h.hexdigest()


def counts_to_events(counts, *, start_us=0, bin_us=1000, first_uid=0):
    """Canonical bin-group event adapter, used for offline/stream equivalence.

    Overflow counts represent only their saturated observation; converting them
    to events does not pretend to reconstruct unobserved latent multiplicity.
    """
    a = np.asarray(counts)
    if a.ndim != 3 or a.shape[-1] != 2 or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise ValueError("counts must be nonnegative integer [B,P,2]")
    start_us, bin_us, uid = _integer(start_us, "start_us"), _integer(bin_us, "bin_us"), _integer(first_uid, "first_uid")
    if bin_us < 1 or uid < 0:
        raise ValueError("invalid event adapter bounds")
    result = []
    for b, p, pol in np.argwhere(a > 0):
        for _ in range(int(a[b, p, pol])):
            result.append(Event(uid, start_us + int(b) * bin_us, int(p), int(pol)))
            uid += 1
    return result


def encode_batch(paths, counts, *, live=None, coordinates=None, omega=None,
                 phase=None, pi=None, start_us=0, bin_us=1000, **kernel_options):
    """Offline constructor sharing exactly the streaming update implementation.

    Reads all counts only into a staging buffer, then consumes bins in order.
    A caller requiring a shorter prefix must pass only its matching paths/counts
    or use EvidenceMemory with the desired certified cutoff.
    """
    a = np.asarray(counts)
    if a.ndim != 3 or a.shape[-1] != 2 or not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise ValueError("counts must be nonnegative integer [B,P,2]")
    kernel = GridKernel(paths, bins=len(a), **kernel_options)
    if a.shape != (kernel.B, kernel.P, 2):
        raise ValueError("count/path shape mismatch")
    memory = EvidenceMemory(kernel, omega=omega, phase=phase, pi=pi,
                            start_us=start_us, bin_us=bin_us)
    memory.counts[:] = np.minimum(a, kernel.cap)
    memory.advance(start_us + kernel.B * bin_us,
                   watermark_us=start_us + kernel.B * bin_us, live=live)
    return memory.query(coordinates)
