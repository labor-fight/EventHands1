"""XVM shared helpers (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 section 0).

Everything here is a thin layer over existing, already-validated pieces: the S44 float64 MANO / SAE / source owners
(`tools/s44/evflow.py`), the evaluator's FK and root alignment (`semkine/eval_track.py`) and the XCL endpoint grid.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from s44 import evflow as EF  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.xyz_mesh import vertex_normals  # noqa: E402

OUT = REPO / "outputs" / "xvm"
STEP = 50
FINGER = slice(6, 51)
SIGMA_3D = 1e-3          # m
SIGMA_2D = 0.5           # px
LAM_P = 1.0
GN_ITERS = 3
CONTOUR_COS = 0.4
ZGZ = ("zgz_global", "zgz_local")
TRAIN_SUBJ = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SELECT_SUBJ = ("ycy", "ylf")


def require_gpu(allowed=("4", "5", "6", "7")) -> torch.device:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if v not in allowed:
        raise SystemExit(f"XVM jobs run on exactly one physical GPU 4-7 (CUDA_VISIBLE_DEVICES={v!r})")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID so that indices are physical")
    return torch.device("cuda")


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def receipt(out_dir: Path, extra: dict, scripts=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"scripts": {str(Path(s).relative_to(REPO)): sha(s) for s in (__file__, *scripts)},
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    (out_dir / "receipt.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


# ------------------------------------------------------------------------------------------------ sequences
def endpoint_grid(runs) -> List[Tuple[int, int, int, bool]]:
    """XCL `endpoint_grid`: (run_id, end, previous_ms, is_first) for ends a+49, a+99, ... < b."""
    rows = []
    for rid, (a, b) in enumerate(np.asarray(runs, np.int64).reshape(-1, 2)):
        previous = int(a)
        for end in np.arange(int(a) + 49, int(b), 50, dtype=np.int64):
            rows.append((rid, int(end), previous, previous == int(a)))
            previous = int(end)
    return rows


def zgz_seqs():
    s = EF.split_seqs("val_core")
    assert [x for x, _ in s] == list(ZGZ)
    return s


def subject_seqs(subjects, base_only=False):
    out = []
    for s, d in EF.split_seqs("train"):
        subj = s.split("_")[0]
        if subj in subjects and (not base_only or not s.split("_")[-1].startswith("v")):
            out.append((s, d))
    return out


def observed(sq: "EF.Seq", end: int, faces: torch.Tensor, window: int = STEP):
    """Vertices owning GT-source events of [end-window+1, end]: (vid (n,) long, mass (n,) float64)."""
    a0, a1 = int(sq.offsets[end - window + 1]), int(sq.offsets[end + 1])
    dev = faces.device
    if a1 <= a0:
        return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.float64, device=dev)
    ids, w, ok = EF.owners_oracle(sq, np.arange(a0, a1), faces)
    ids, w = ids[ok].reshape(-1), w[ok].reshape(-1)
    keep = w > 0
    mass = torch.zeros(778, dtype=torch.float64, device=dev).index_add_(0, ids[keep], w[keep])
    vid = (mass > 0).nonzero().flatten()
    return vid, mass[vid]


# ------------------------------------------------------------------------------------------------ geometry
def fd_jacobian(mano: "EF.Mano64", params: torch.Tensor, betas: torch.Tensor, intr):
    """params (B, 51) float64 -> verts (B,778,3), dX (B,778,3,51), uv (B,778,2), z (B,778), J2 (B,778,2,51).
    Central differences (eps 1e-4) in the additive 51-D parameters, the S44 convention."""
    B = params.shape[0]
    eps = EF.FD_EPS
    eye = torch.eye(51, dtype=torch.float64, device=params.device) * eps
    P = torch.cat([params[:, None], params[:, None] + eye[None], params[:, None] - eye[None]], 1)
    v, _ = mano.fk_chunked(P.reshape(-1, 51), betas)
    v = v.view(B, 103, 778, 3)
    v0 = v[:, 0]
    dX = ((v[:, 1:52] - v[:, 52:103]) / (2 * eps)).permute(0, 2, 3, 1).contiguous()   # (B,778,3,51)
    fx, fy, _, _ = intr
    X, Y, Z = v0[..., 0], v0[..., 1], v0[..., 2].clamp(min=1e-6)
    Jp = torch.zeros(B, 778, 2, 3, dtype=torch.float64, device=params.device)
    Jp[..., 0, 0] = fx / Z
    Jp[..., 0, 2] = -fx * X / Z ** 2
    Jp[..., 1, 1] = fy / Z
    Jp[..., 1, 2] = -fy * Y / Z ** 2
    J2 = Jp @ dX
    uv, z = EF.Mano64.project(v0, intr)
    return v0, dX, uv, z, J2, Jp


def projected_normals(verts: torch.Tensor, faces: torch.Tensor, Jp: torch.Tensor):
    """Unit normals n (B,V,3), |n . d_hat| (B,V) and the unit image direction e1 of the normal (B,V,2)."""
    n = vertex_normals(verts.float(), faces).to(verts.dtype)
    dhat = verts / verts.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cos = (n * dhat).sum(-1).abs()
    e = (Jp @ n[..., None]).squeeze(-1)
    e1 = e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return n, cos, e1


def increment_precision(dev) -> torch.Tensor:
    """Precision of the 50 ms 51-D increment: the S44 rate precision (1/s units) / 0.05^2."""
    prec, _ = EF.rate_prior(dev)
    return prec / (STEP * 1e-3) ** 2


def gn_step(A_y: torch.Tensor, dy: torch.Tensor, Q: torch.Tensor, m: torch.Tensor, W: torch.Tensor,
            P: torch.Tensor, lam: float, dx0: torch.Tensor) -> torch.Tensor:
    """One Gauss-Newton step for one state. Per measured vertex: residual r = Q (y(x) - y(x0)) - m with
    information W; A_y (n, dim, 51) d y / d x at x; dy (n, dim) = y(x) - y(x0); Q (n, k, dim); m (n, k); W (n, k, k);
    prior lam * P on dx0 = x - x0. Returns delta (51,)."""
    A = Q @ A_y                                                            # (n, k, 51)
    r = (Q @ dy[..., None]).squeeze(-1) - m                                # (n, k)
    H = torch.einsum("nka,nkl,nlb->ab", A, W, A) + lam * P
    g = torch.einsum("nka,nkl,nl->a", A, W, r) + lam * (P @ dx0)
    return -torch.linalg.solve(H, g)


def fk_h(mano: "EF.Mano64", params: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    return mano.fk_hand(params, betas)


def r_art_terms(mano, x0, x, gt_end, betas):
    """Per step e_art and e_zero (mm) of the finger part of x relative to x0 (S44 definition)."""
    n = x.shape[0]
    J = fk_h(mano, torch.cat([x, x0, gt_end]), betas)
    jx, j0, jg = J[:n], J[n:2 * n], J[2 * n:]
    e_art = (jx - jg).norm(dim=-1).mean(-1) * 1000
    e_zero = (j0 - jg).norm(dim=-1).mean(-1) * 1000
    return e_art, e_zero, (jx - jg), (j0 - jg)


def score_ra(mano32, preds: np.ndarray, targets: np.ndarray, betas_np, device) -> Dict[str, np.ndarray]:
    """The evaluator's root-aligned MPJPE / MPVPE per step (float32 ManoLayer, `ET.mano_fk`)."""
    beta = torch.tensor(np.asarray(betas_np), device=device, dtype=torch.float32).view(1, -1)
    pj, pv = ET.mano_fk(mano32, np.ascontiguousarray(preds, np.float32), beta, device)
    gj, gv = ET.mano_fk(mano32, np.ascontiguousarray(targets, np.float32), beta, device)
    return {"mpjpe_ra_mm": np.linalg.norm(ET.root_align(pj) - ET.root_align(gj), axis=-1).mean(-1) * 1000,
            "mpvpe_ra_mm": np.linalg.norm(ET.root_align(pv) - ET.root_align(gv), axis=-1).mean(-1) * 1000,
            "wrist_mm": np.linalg.norm(pj[:, 0] - gj[:, 0], axis=-1) * 1000}


def mano32(device):
    from mano_layer import ManoLayer
    return ManoLayer(str(EF.MANO_NPZ), add_mean=False).to(device).eval()


def smoothed_gt(pos51: np.ndarray, t: int, lo: int, hi: int, half: int = 25) -> np.ndarray:
    """+-`half` ms box average of the per-ms GT around t, clipped to [lo, hi] (the valid run)."""
    a, b = max(lo, t - half), min(hi, t + half)
    return np.asarray(pos51[a:b + 1], np.float64).mean(0)
