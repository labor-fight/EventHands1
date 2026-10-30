"""MV2 shared helpers (docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026): the ten-scheme v2.1 study.

A thin layer over already-validated pieces:
  * the frozen S39 trackers `outputs/semkine/dt_dz_l3_s{3407,3408,3409}` (step 6000; earlier steps for the S3 debug
    reference), loaded with `MNISTModel.load_from_checkpoint` exactly like `tools/trk6/carry_probe.py`;
  * the evaluator's W200 event image (`semkine.eval_track.build_lnes`, window clipped to the run start) with
    prev = GT(end - 50 ms), i.e. the E3 single step of the S39 stack (carry_probe's construction);
  * `resnet_forward_taps` for the layer1-3 maps (bit for bit `rn(x)`), `semkine.aura.visible` (3 x 3 vertex-splat
    z-buffer) and `semkine.xyz_mesh.vertex_normals` / `mesh_ring` for the MANO surface.
Every probe verifies the frozen protocol (`outputs/mv2/protocol.json`) before it reads anything (`--smoke` runs print
shapes and timings only) and refuses to overwrite a result.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine.dataset import sequences_for_split  # noqa: E402

OUT = REPO / "outputs" / "mv2"
PROTOCOL = OUT / "protocol.json"
DATA_ROOT = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
LEGACY_DIR = "train"
ZGZ = ("zgz_global", "zgz_local")
FIT7 = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SELECT = ("ycy", "ylf")
SEEDS = (3407, 3408, 3409)
STEP = 50
WINDOW = 200
S39_RUN = REPO / "outputs" / "semkine" / "dt_dz_l3_s{seed}"
TAPS = ("layer1", "layer2", "layer3")
TAP_STRIDE = {"layer1": 4, "layer2": 8, "layer3": 16}
H, W = 180, 240
PREREG_BEGIN = '<a id="mv2-prereg-20261026"></a>'
PREREG_END = "<!-- MV2-PREREG:END -->"
GPUS_ALLOWED = ("2", "3", "4", "5", "6", "7")


# ----------------------------------------------------------------------------------------------- protocol / io
def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_text() -> Optional[str]:
    doc = (REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text(encoding="utf-8")
    m = re.search(re.escape(PREREG_BEGIN) + r"(.*?)" + re.escape(PREREG_END), doc, re.S)
    return m.group(1) if m else None


def verify_protocol(require: bool = True) -> dict:
    """Every frozen file must match its registered SHA-256 and the pre-registration text its digest."""
    if not PROTOCOL.exists():
        if require:
            raise SystemExit("outputs/mv2/protocol.json missing: freeze before reading (tools/mv2/freeze.py)")
        return {}
    proto = json.loads(PROTOCOL.read_text())
    bad = [f for f, h in proto["files"].items() if not (REPO / f).exists() or sha(REPO / f) != h]
    bad += [c for c, h in proto.get("checkpoints", {}).items() if not Path(c).exists() or sha(c) != h]
    if bad:
        raise SystemExit(f"frozen files changed since the MV2 freeze: {bad}")
    txt = prereg_text()
    if txt is None or hashlib.sha256(txt.encode()).hexdigest() != proto["prereg_sha256"]:
        raise SystemExit("the MV2 pre-registration text changed after the freeze")
    return proto


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(path, obj, exclusive: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise SystemExit(f"{path} exists; preserve it")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_np, ensure_ascii=False))
    os.replace(tmp, path)


def receipt(out_dir, extra: dict, scripts: Iterable = ()):
    out_dir = Path(out_dir)
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "scripts": {str(Path(s).relative_to(REPO)) if Path(s).is_absolute() else str(s):
                       sha(REPO / s if not Path(s).is_absolute() else s) for s in scripts}}
    rec.update(extra)
    save_json(out_dir / "receipt.json", rec, exclusive=False)


def device_from_env(cpu: bool = False) -> torch.device:
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not vis or any(v.strip() not in GPUS_ALLOWED for v in vis.split(",")):
        raise SystemExit(f"MV2 GPU work only on physical GPUs {GPUS_ALLOWED}; CUDA_VISIBLE_DEVICES={vis!r}")
    return torch.device("cuda")


# ----------------------------------------------------------------------------------------------- data
def base_seqs(subjects: Sequence[str], domains=("local", "global")) -> List[str]:
    return [f"{s}_{d}" for s in subjects for d in domains]


def subject_of(seq: str) -> str:
    return seq.split("_")[0]


_SEQ_DIR: Dict[str, str] = {}


def seq_dir(seq: str) -> str:
    if not _SEQ_DIR:
        for split in ("train", "val_core"):
            for name, d in sequences_for_split(DATA_ROOT, split):
                _SEQ_DIR[name] = d
    return _SEQ_DIR[seq]


def load_seq(seq: str):
    """(events, offsets, aux, pos51) of a sequence (`semkine.eval_track.load_sequence`)."""
    return ET.load_sequence(DATA_ROOT, seq_dir(seq), seq)


def endpoints(aux, every: int = 8, phase: int = 0) -> np.ndarray:
    """Endpoint ms of the evaluator's 50 ms grid inside every valid run, skipping each run's first step (no prev inside
    the run), keeping every `every`-th one (offset `phase`). carry_probe's grid with a configurable stride."""
    out = []
    for ra, rb in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        ends = np.arange(ra + STEP - 1, rb, STEP, dtype=np.int64)
        for k, end in enumerate(ends):
            if k == 0 or (k - phase) % every:
                continue
            out.append((int(end), int(ra)))
    return np.asarray(out, dtype=np.int64).reshape(-1, 2)


class SeqSamples:
    """W200 single-step samples of one sequence: X (N, H, W, C) event images, prev = GT(end - 50), gt = GT(end)."""

    def __init__(self, seq: str, every: int = 8, phase: int = 0, window: int = WINDOW, channels=("last",),
                 max_n: Optional[int] = None):
        events, offsets, aux, pos51 = load_seq(seq)
        self.seq, self.window = seq, window
        self.betas = np.asarray(aux["betas"], np.float32).reshape(-1)[:10]
        self.K = np.asarray(aux["camera_K"], np.float32).reshape(3, 3)
        ep = endpoints(aux, every, phase)
        if max_n is not None:
            ep = ep[:max_n]
        self.ends, self.run_a = ep[:, 0], ep[:, 1]
        self.windows = np.minimum(window, self.ends + 1 - self.run_a)
        self._events, self._offsets, self.channels = events, offsets, channels
        self.pos51 = pos51
        self.prev = np.stack([np.asarray(pos51[int(e) - STEP], np.float32) for e in self.ends]) if len(ep) else \
            np.zeros((0, 51), np.float32)
        self.gt = np.stack([np.asarray(pos51[int(e)], np.float32) for e in self.ends]) if len(ep) else \
            np.zeros((0, 51), np.float32)

    def __len__(self):
        return len(self.ends)

    def lnes(self, idx: Sequence[int], window: Optional[int] = None, transform=None) -> np.ndarray:
        """Event images of samples `idx`; `transform(ev (n, 3) int x/y/p, ms_rel) -> (x, y, p, ms_rel)` lets a probe
        perturb the raw events before the splat (zoom, artefacts) with the evaluator's own splat."""
        out = []
        for i in idx:
            end = int(self.ends[i])
            w = int(min(window or self.window, end + 1 - int(self.run_a[i])))
            if transform is None:
                out.append(ET.build_lnes(self._events, self._offsets, end, w, self.channels))
                continue
            start = end - w + 1
            a0, a1 = int(self._offsets[start]), int(self._offsets[end + 1])
            ev = np.asarray(self._events[a0:a1])
            counts = np.diff(self._offsets[start: end + 2]).astype(np.int64)
            ms_rel = np.repeat(np.arange(w, dtype=np.float32), counts)
            x, y, p, ms_rel = transform(ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64),
                                        np.clip(ev[:, 2], 0, 1).astype(np.int64), ms_rel)
            if len(x) == 0:
                out.append(np.zeros((H, W, 2 * len(self.channels)), np.float32))
            else:
                out.append(EV.splat_event_image(x, y, p, ms_rel, w, H, W, self.channels))
        return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------------------------------------- frozen S39
def s39_ckpt(seed: int, step: int = 6000) -> Path:
    run = Path(str(S39_RUN).format(seed=seed))
    ck = run / f"dt_dz_l3_s{seed}-step={step}.ckpt"
    if not ck.exists():
        raise FileNotFoundError(ck)
    return ck


def load_s39(seed: int, device, step: int = 6000, cls=MNISTModel):
    run = Path(str(S39_RUN).format(seed=seed))
    cfg = load_config(run / "config_resolved.yaml")
    model = cls.load_from_checkpoint(str(s39_ckpt(seed, step)), cfg=cfg, map_location=device).to(device).eval()
    return model, cfg


@torch.no_grad()
def s39_forward_taps(model, X: torch.Tensor, P: torch.Tensor, taps=TAPS, check: bool = False):
    """The S39 configuration's `MNISTModel.forward` (PREV_RENDER sil/inv, resnet18_l3, PREVPOS_EMBED, PREDICT_DELTA,
    ZERO_EVENT_GATE; no CAM_PLANES / ROOT_HEAD / ROOT_COMPOSE / TRANSL_HEAD) with the layer maps `taps` returned.
    Returns (out51, {tap: map}, logits_before_prev_mlp). `check=True` asserts out51 == model(X, P) bit for bit."""
    from backbones import resnet_forward_taps
    assert model.prev_render and model.root_head_kind == "none" and not model.cam_planes
    assert model.predict_delta and model.prevpos_embed and not model.active_head and not model.glimpse
    lnes = X
    B = lnes.shape[0]
    betas_f, k_f = model._resolve_betas_K(P, None, None)
    rend = model._render_prev(P.float(), betas_f, k_f).to(dtype=lnes.dtype, device=lnes.device)
    x = torch.cat([lnes, rend], dim=-1).permute(0, 3, 1, 2).contiguous()
    logits, got = resnet_forward_taps(model.rn, model.conv1(x), taps)
    out = logits + model.prev_mlp(P.to(logits.dtype))
    empty = lnes.reshape(B, -1).abs().sum(dim=1, keepdim=True) <= 0
    delta = torch.where(empty, torch.zeros_like(out), out)
    out51 = delta + P.to(out.dtype)
    if check:
        ref = model(X, P)
        if not torch.equal(ref, out51):
            raise AssertionError(f"s39_forward_taps differs from model(): max {float((ref - out51).abs().max())}")
    return out51, got, logits


# ----------------------------------------------------------------------------------------------- MANO surface
_MESH_CACHE: Dict[str, torch.Tensor] = {}


def mesh_tables(model) -> Dict[str, torch.Tensor]:
    """Faces, LBS weights (778, 16), face 1-ring (idx, mask) of the model's MANO layer (cached, CPU)."""
    if not _MESH_CACHE:
        from semkine.xyz_mesh import mesh_ring
        faces = model.mano.f.detach().cpu().long()
        idx, emask = mesh_ring(faces, 778)
        _MESH_CACHE.update(faces=faces, lbs=model.mano.weights.detach().cpu().float(), ring_idx=idx, ring_mask=emask)
    return _MESH_CACHE


@torch.no_grad()
def fk_surface(model, P: torch.Tensor, betas: torch.Tensor, K: torch.Tensor) -> Dict[str, torch.Tensor]:
    """FK of `P` (B, 51) in float32: verts (B, 778, 3), joints (B, 21, 3), render-pixel uv (B, 778, 2), z (B, 778),
    vis (B, 778) bool (`semkine.aura.visible`), normals (B, 778, 3), intr (fx, fy, cx, cy)."""
    from semkine.aura import visible
    from semkine.xyz_mesh import vertex_normals
    B = P.shape[0]
    b = betas.float().view(1, -1).expand(B, -1).to(P.device)
    k = K.float().view(1, 3, 3).expand(B, -1, -1).to(P.device)
    verts, joints = model._fk(P.float(), b)
    intr = model._intrinsics(k)
    fx, fy, cx, cy = intr
    z = verts[..., 2].clamp(min=1e-6)
    uv = torch.stack([fx[:, None] * verts[..., 0] / z + cx[:, None], fy[:, None] * verts[..., 1] / z + cy[:, None]], -1)
    vis = visible(verts, intr)
    normals = vertex_normals(verts, model.mano.f)
    return {"verts": verts, "joints": joints, "uv": uv, "z": verts[..., 2], "vis": vis, "normals": normals,
            "intr": intr}


def sample_map(fmap: torch.Tensor, uv: torch.Tensor, stride: int) -> torch.Tensor:
    """Bilinear samples of `fmap` (B, C, h, w) at render pixels `uv` (B, N, 2) -> (B, N, C), using the stem geometry of
    `MNISTModel._anchor_features` (cell m of a stride-s map sits at input pixel s m; align_corners=True; border clamp)."""
    B, C, h, w = fmap.shape
    gx = 2.0 * (uv[..., 0] / stride) / (w - 1) - 1.0
    gy = 2.0 * (uv[..., 1] / stride) / (h - 1) - 1.0
    grid = torch.stack([gx, gy], -1).unsqueeze(2).to(fmap.dtype)
    return F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(-1).transpose(1, 2)


# ----------------------------------------------------------------------------------------------- statistics
def ridge_dual(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, lam: float) -> np.ndarray:
    """Kernel (dual) ridge with a linear kernel and an unpenalised mean: prediction for Xte. Works for d >> n."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    A, Bt = Xtr - mx, Xte - mx
    Kt = A @ A.T
    n = Kt.shape[0]
    alpha = np.linalg.solve(Kt + lam * np.trace(Kt) / n * np.eye(n), Ytr - my)
    return Bt @ (A.T @ alpha) + my


def block_bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 20261026,
                       stat=np.mean, q=(2.5, 97.5)) -> Tuple[float, float]:
    """Percentile CI of `stat` resampling whole groups (subjects or (sequence, block) units)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        bs.append(stat(np.concatenate([values[idx[g]] for g in pick])))
    return float(np.percentile(bs, q[0])), float(np.percentile(bs, q[1]))


def root_aligned_joints(model, P: np.ndarray, betas: np.ndarray, device, chunk: int = 512) -> np.ndarray:
    """(N, 21, 3) root-aligned joints (metres) of 51-d states, the evaluator's RA convention."""
    out = []
    b = torch.tensor(betas, dtype=torch.float32, device=device).view(1, -1)
    with torch.no_grad():
        for i in range(0, len(P), chunk):
            p = torch.from_numpy(np.asarray(P[i:i + chunk], np.float32)).to(device)
            _, j = model._fk(p, b.expand(len(p), -1))
            j = j - j[:, :1]
            out.append(j.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 21, 3), np.float32)
