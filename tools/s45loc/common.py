"""S45-LOC identity, task registry and pure helpers (docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029).

S45-LOC = zero-training readings of the frozen S45 (`c1sf_so3_s3407`, step 6000) that localise where, along
simulated events -> ray x MANO association -> 13-dim relation records (token7 | dXYZ | zn | valid | hit) ->
learned 51D update, the zgz_local single-step (E3) excess and the closed-loop (E1) error come from.

Nothing here trains, selects or rewrites a weight. Interventions act only on the evaluation inputs (event subset,
virtual camera, state) or on one relation-input column through a forward pre-hook, and every one is restored.
Imports of torch / the model happen inside the functions that need them, so this module is CPU-only and cheap.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/s45loc/20261029"
#: engineering smoke runs point S45LOC_OUT at a scratch directory; only the default directory holds readings
OUT = Path(os.environ.get("S45LOC_OUT", str(DEFAULT_OUT)))
SMOKE = OUT != DEFAULT_OUT
PROTOCOL = OUT / "protocol.json"
BASELINE = ROOT / "outputs/s45/baseline.json"
CHECKPOINT = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407/c1sf_so3_s3407-step=6000.ckpt"
CHECKPOINT_SHA256 = "d0164d55660c31184e8fd0b70f22f56cf9a7a064a216073862c9c56653d7dce0"
CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
CONFIG_SHA256 = "521d2d4aa2558d1eac33a77e23d972a1258fdc7733c4b35406e62215eaccd29f"
#: the closed C1SF readings of S45 (E1 self loop, E2 on C1's incoming, E3 GT previous) and the C1 reference
S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
C1_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/c1"
XAG_SOURCES = ROOT / "outputs/xag/sources"
SIGMA_STAR = ROOT / "outputs/sba/p0v/result.json"
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
#: the XYZ topic's approved cards (C1SF amendment 1 allows 2-7; this round uses 4-7, two processes per card)
PHYSICAL_GPUS = (4, 5, 6, 7)
SEED = 20261029

TRAIN_SUBJECTS = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")
TRAIN_LOCAL = tuple(f"{s}_local" for s in TRAIN_SUBJECTS)
TRAIN_GLOBAL = tuple(f"{s}_global" for s in TRAIN_SUBJECTS)
TRAIN_BASE = tuple(q for s in TRAIN_SUBJECTS for q in (f"{s}_global", f"{s}_local"))
ZGZ = ("zgz_global", "zgz_local")

#: perturbation sizes (E3 + delta): translation 10 mm, root rotation 5 degrees (camera frame, about the root joint),
#: fingers = the training contract's "small" draw, N(0, 0.05^2) rad on each of the 45 local axis-angle entries
PERTURB = {"T10": 0.010, "R5": float(np.deg2rad(5.0)), "Fs": 0.05}

#: task -> (domain, mode, intervention). Domains: "zgz" = zgz_global then zgz_local in the official order with one
#: init-noise generator (so zgz E1 replays the cached C1SF E1 draws); the intervention touches zgz_local only.
#: "zgz_local" = zgz_local alone (E3 only: no closed-loop state crosses sequences). "train_local" / "train_base" =
#: the nine / eighteen base training recordings (in-sample), each with its own init-noise generator.
STAGE1 = {
    # --- baselines (zgz ones must equal the cached C1SF arrays bit for bit)
    "b0_zgz_e3": ("zgz", "e3", "none"),
    "b0_zgz_e1": ("zgz", "e1", "none"),
    "b0_train_e3": ("train_base", "e3", "none"),
    # --- S45 sign check of the two frozen C1SF interventions on in-sample local recordings
    "b1_train_local_geozero": ("train_local", "e3", "geometry_zero"),
    "b1_train_local_h3zero": ("train_local", "e3", "h3_zero"),
    # --- absolute depth (zn) channel: the zn-only column shift (primary) both ways, and on zgz_local the
    #     geometry-consistent virtual-zoom transplant (residual = the dolly-zoom perspective term, <= ~1.3 px)
    "b2_zgz_local_transplant_ref": ("zgz_local", "e3", "transplant_ref"),
    "b2_train_local_znshift_zgz": ("train_local", "e3", "znshift_zgz"),
    "b3_zgz_local_znshift_ref": ("zgz_local", "e3", "znshift_ref"),
    # --- event-side scale gap (SBA / Stage 0 sigma*): event pixels rescaled about the GT-previous wrist pixel
    "b7_zgz_local_sigma_ref": ("zgz_local", "e3", "sigma_ref"),
    "b7_train_local_sigma_zgz": ("train_local", "e3", "sigma_zgz"),
    # --- event density, alone and with the zgz depth (zn shift)
    "b4_train_local_subsample": ("train_local", "e3", "subsample_zgz"),
    "b4_train_local_zgzlike": ("train_local", "e3", "zgzlike"),
    # --- oracle event cleaning (GT pseudo-sources), each against a random drop of the same count
    "b5_zgz_local_offhand": ("zgz_local", "e3", "drop_offhand"),
    "b5_zgz_local_offhand_rand": ("zgz_local", "e3", "drop_offhand_rand"),
    "b5_zgz_local_clean": ("zgz_local", "e3", "drop_misassoc"),
    "b5_zgz_local_clean_rand": ("zgz_local", "e3", "drop_misassoc_rand"),
    "b5_train_local_offhand": ("train_local", "e3", "drop_offhand"),
    "b5_train_local_offhand_rand": ("train_local", "e3", "drop_offhand_rand"),
    "b5_train_local_clean": ("train_local", "e3", "drop_misassoc"),
    "b5_train_local_clean_rand": ("train_local", "e3", "drop_misassoc_rand"),
    # --- closed loop
    "c1_train_e1": ("train_base", "e1", "none"),
    "c2_train_local_zgzlike_e1": ("train_local", "e1", "zgzlike"),
    "c3_zgz_e1_transplant_ref": ("zgz", "e1", "transplant_ref"),
    "c3_zgz_e1_znshift_ref": ("zgz", "e1", "znshift_ref"),
    "c4_zgz_local_T10": ("zgz_local", "e3", "perturb_T10"),
    "c4_zgz_local_R5": ("zgz_local", "e3", "perturb_R5"),
    "c4_zgz_local_Fs": ("zgz_local", "e3", "perturb_Fs"),
    "c4_train_local_T10": ("train_local", "e3", "perturb_T10"),
    "c4_train_local_R5": ("train_local", "e3", "perturb_R5"),
    "c4_train_local_Fs": ("train_local", "e3", "perturb_Fs"),
    "c5_zgz_e1_clean": ("zgz", "e1", "drop_misassoc"),
    "c5_zgz_e1_clean_rand": ("zgz", "e1", "drop_misassoc_rand"),
}
#: stage 2 reads the stage-1 token captures (training-local quantiles) before it starts
STAGE2 = {"b6_zgz_local_tokenmap": ("zgz_local", "e3", "tokenmap")}
#: amendment 1 (docs/EXPERIMENT_RECORDS.md#s45loc-amend1-20261029, written after the main readings): XCL E4-style GT
#: clamps of the closed loop -- every step's incoming state keeps the loop's own translation with the GT previous
#: pose (e1_gtpose), or the loop's own pose with the GT previous translation (e1_gttransl)
AMEND1 = {
    "d1_zgz_e1_gtpose": ("zgz", "e1_gtpose", "none"),
    "d1_zgz_e1_gttransl": ("zgz", "e1_gttransl", "none"),
    "d1_train_local_e1_gtpose": ("train_local", "e1_gtpose", "none"),
    "d1_train_local_e1_gttransl": ("train_local", "e1_gttransl", "none"),
}
TASKS = {**STAGE1, **STAGE2, **AMEND1}
E1_MODES = ("e1", "e1_gtpose", "e1_gttransl")
INTERVENTIONS = ("none", "geometry_zero", "h3_zero", "transplant_ref", "znshift_ref", "znshift_zgz",
                 "sigma_ref", "sigma_zgz", "subsample_zgz", "zgzlike", "drop_offhand", "drop_offhand_rand", "drop_misassoc",
                 "drop_misassoc_rand", "perturb_T10", "perturb_R5", "perturb_Fs", "tokenmap")
#: relation-input columns: token7 = [x/W, y/H, polarity, t/dt, log1p(dt_ie/eps), SAE same / dt, SAE opposite / dt]
ZN_COL, VALID_COL = 10, 11
TOKEN_TIMING_COLS = (4, 5, 6)
TOKEN_SAMPLE = 200_000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(value):
    h = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(value):
    return json.loads(Path(value).read_text())


def save(destination, value, *, exclusive=False):
    p = Path(destination)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n"
    if exclusive:
        with p.open("x") as stream:
            stream.write(data)
        return
    temp = p.with_name(p.name + f".{os.getpid()}.tmp")
    temp.write_text(data)
    temp.replace(p)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def domain_sequences(domain):
    return {"zgz": ZGZ, "zgz_local": ("zgz_local",), "train_local": TRAIN_LOCAL, "train_base": TRAIN_BASE}[domain]


def rng_for(seq, end_ms, salt):
    """Per-(sequence, endpoint, use) generator, independent of process / order."""
    key = int.from_bytes(hashlib.sha256(seq.encode()).digest()[:4], "little")
    return np.random.default_rng([SEED, key, int(end_ms), int(salt)])


# --------------------------------------------------------------------------- state / camera transforms
def rodrigues(rv):
    rv = np.asarray(rv, np.float64)
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def log_so3(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, np.float64)).as_rotvec()


def transplant_state(state, k, j0_z):
    """Virtual zoom: the root joint's camera depth p0_z = j0_z + t_z becomes k p0_z; x, y, rotations and fingers keep
    their bits. With fx, fy scaled by the same k (principal point kept) every hand point projects to the same pixel to
    first order (the residual is the second-order perspective term, ~(1/k - 1) * depth extent * |x| / z^2)."""
    out = np.array(state, np.float32, copy=True)
    out[..., 2] = (k * (j0_z + out[..., 2].astype(np.float64)) - j0_z).astype(np.float32)
    return out


def transplant_K(K, k):
    out = np.array(K, np.float32, copy=True)
    out[0, 0] *= k
    out[1, 1] *= k
    return out


def perturb_state(state, kind, rng):
    """E3 + delta; `kind` in PERTURB. Returns (perturbed state, the delta's parameter-space description)."""
    out = np.array(state, np.float32, copy=True)
    if kind == "T10":
        u = rng.standard_normal(3)
        delta = PERTURB[kind] * u / np.linalg.norm(u)
        out[0:3] = (out[0:3].astype(np.float64) + delta).astype(np.float32)
    elif kind == "R5":
        u = rng.standard_normal(3)
        delta = PERTURB[kind] * u / np.linalg.norm(u)
        out[3:6] = log_so3(rodrigues(delta) @ rodrigues(out[3:6])).astype(np.float32)
    elif kind == "Fs":
        delta = PERTURB[kind] * rng.standard_normal(45)
        out[6:51] = (out[6:51].astype(np.float64) + delta).astype(np.float32)
    else:
        raise ValueError(kind)
    return out, np.asarray(delta, np.float64)


def rescale_events(ev5, center, factor, width=240, height=180):
    """Event pixels moved to rint(c + factor (q - c)) about `center` (render-frame pixels); returns (events, in-bounds
    mask over the input rows). Only x / y change; time and polarity keep their bits."""
    out = np.array(ev5, np.float32, copy=True)
    x = np.rint(center[0] + factor * (ev5[:, 1].astype(np.float64) - center[0]))
    y = np.rint(center[1] + factor * (ev5[:, 2].astype(np.float64) - center[1]))
    keep = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    out[:, 1], out[:, 2] = x, y
    return out[keep], keep


def keep_fraction_mask(n, p, rng):
    return rng.random(n) < float(p)


def random_drop_mask(n, n_drop, rng):
    keep = np.ones(n, bool)
    if n_drop > 0:
        keep[rng.choice(n, size=int(n_drop), replace=False)] = False
    return keep


# --------------------------------------------------------------------------- part assignment
def weighted_part(vid, weight, vertex_part, n_parts=16):
    """Majority body part of weighted vertex sets: vid / weight `(..., R)`, vertex_part `(V,)` -> `(...,)`, -1 where
    the weights are all zero."""
    vid = np.asarray(vid)
    weight = np.asarray(weight, np.float64)
    parts = np.asarray(vertex_part)[vid]
    votes = np.zeros(vid.shape[:-1] + (n_parts,), np.float64)
    for r in range(vid.shape[-1]):
        np.add.at(votes, (*np.indices(vid.shape[:-1]), parts[..., r]), weight[..., r])
    best = votes.argmax(-1)
    return np.where(votes.max(-1) > 0, best, -1)


__all__ = [name for name in dir() if not name.startswith("_")]
