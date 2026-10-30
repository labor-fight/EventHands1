"""Three representations of one common, saturated millisecond count stream.

No labels, true candidate, true nuisance, or future prefixes are accepted here.
The cheap enhancement is deliberately strong: first/last/count and five bins.
"""
from __future__ import annotations

import numpy as np

from .carrier import EvidenceMemory, GridKernel, logsumexp

ARMS = ("lnes", "enhanced", "br")
SENSOR = dict(ap=2, am=2, alpha=.9, nu=.02, cap=8)


def pixel_features(counts):
    a = np.asarray(counts)
    if a.ndim != 3 or a.shape[-1] != 2 or len(a) != 50 or np.any(a < 0):
        raise ValueError("expected nonnegative [50,P,2] counts")
    a = np.minimum(a, SENSOR['cap'])
    present = a > 0
    ticks = np.arange(len(a))[:, None, None]
    last = np.where(present, ticks, 0).max(0) / float(len(a))
    first = np.where(present, ticks, len(a)).min(0)
    first = np.where(first < len(a), first / float(len(a)), 0.)
    count = np.log1p(a.sum(0))
    valid = present.any(0).astype(float)
    coarse = np.log1p(a.reshape(5, 10, *a.shape[1:]).sum(1))
    enhanced = np.concatenate([last, first, count, valid, *coarse], axis=-1)
    return last.astype(np.float32), enhanced.astype(np.float32)


def temporal_intervention(counts, seed):
    """Destroy fine timing while preserving BOTH cheap representations exactly.

    Permute within each 10ms block, strictly inside each slot's first/last bin.
    The intervention changes observations, not physical trajectory labels.
    """
    a = np.minimum(np.asarray(counts), SENSOR['cap']).copy()
    rng = np.random.default_rng(seed)
    for p in range(a.shape[1]):
        for pol in range(2):
            occupied = np.flatnonzero(a[:, p, pol])
            if len(occupied) < 2:
                continue
            first, last = occupied[[0, -1]]
            for start in range(0, 50, 10):
                ix = np.arange(max(start, first + 1), min(start + 10, last))
                if len(ix) > 1:
                    a[ix, p, pol] = a[rng.permutation(ix), p, pol]
    before, after = pixel_features(counts), pixel_features(a)
    if not all(np.array_equal(x, y) for x, y in zip(before, after)):
        raise AssertionError("time intervention changed a protected baseline")
    return a


def evidence(paths, counts, *, need_remixed=False, context_id="experiment"):
    """Shared-eta evidence; optional wrong per-bin eta remix with phase intact."""
    kernel = GridKernel(paths, context_id=context_id, **SENSOR)
    memory = EvidenceMemory(kernel)
    memory.counts[:] = np.minimum(counts, kernel.cap)
    remixed = np.zeros(kernel.K)
    t0 = __import__('time').perf_counter()
    for b in range(kernel.B):
        old = memory.ell.copy()
        memory.advance((b + 1) * 1000, watermark_us=(b + 1) * 1000)
        if need_remixed:
            # Keep each branch's posterior phase; reset only eta mixing weights.
            delta = memory.ell - old
            remixed += logsumexp(memory.logomega + delta, axis=-1)
    advance_seconds = __import__('time').perf_counter() - t0
    t0 = __import__('time').perf_counter()
    result = memory.query()
    query_seconds = __import__('time').perf_counter() - t0
    if result['status'] != 'ok':
        raise ArithmeticError("all candidates unsupported; retain model mismatch")
    return {"relative": result['R'][:, 51].astype(np.float32),
            "likelihood": result['log_likelihood'],
            "posterior": result['posterior'],
            "remixed": remixed,
            "advance_seconds": advance_seconds, "query_seconds": query_seconds,
            "memory": memory.memory_bytes()}


def geometry_conditions(betas, K_cam):
    """Explicit beta/camera access for every arm, not only the renderer."""
    camera=np.asarray(K_cam,float).copy()
    camera[0]/=240.; camera[1]/=180.
    return np.concatenate([np.asarray(betas,float).reshape(10),camera.ravel()]).astype(np.float32)


def shared_context(history, candidates, paths, betas, K_cam):
    """Only legal window-start context, shared byte-for-byte by all arms.

    Candidate coordinates are scaled m/rad.  The static first radiance frame is
    a public geometry assumption, not an image, label or the generating eta.
    """
    scale = np.array([.05] * 3 + [.3] * 48)
    coords = (np.asarray(candidates)[:, -1] - np.asarray(history)) / scale
    return np.concatenate([np.asarray(history) / scale, coords.ravel(),
                           np.asarray(paths)[0, 1, :, 0] / 20.,
                           geometry_conditions(betas,K_cam)]).astype(np.float32)


def probabilities(scores):
    x = np.asarray(scores, float)
    return np.exp(x - logsumexp(x, axis=-1)[..., None])
