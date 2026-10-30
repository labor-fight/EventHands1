"""AGG778 shared helpers (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024).

The frozen modules (C1 for the A0 condition, XCL A3F0 for the A3 condition) run through the unchanged
`MNISTModel.forward_packet` on evaluator packets: the 50 ms windows of the XCL endpoint grid built by
`eval_track._window_events`, with the probe-only capture (`model.xyz_keep_aux`) switched on. Nothing here changes a
model path; the only addition is a forward pre-hook on `xyz.vin` that keeps the pooled 257-d vertex input.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = REPO / "outputs" / "agg778"
PROTOCOL = OUT / "protocol.json"
DATA_ROOT = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
MANIFEST = DATA_ROOT / "splits_semkine.json"
SOURCE_ROOT = REPO / "outputs" / "xag" / "sources"
STEP = 50
V = 778
FIT_SUBJ = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SELECT_SUBJ = ("ycy", "ylf")
ZGZ = ("zgz_global", "zgz_local")
ZGZ_COUNTS = {"zgz_global": 1386, "zgz_local": 1204}
GPUS = ("4", "5", "6", "7")
BLOCK = 20                       # endpoints per bootstrap block (within one run)
CONTOUR_COS = 0.4                # |n . d_hat| below this is a contour vertex (XVM convention)
RHO_M = 0.02                     # vertex-frame unit (P4 PhiHead): rho = f * 0.02 / z
MODELS = {
    "A0": {"config": "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
           "ckpt": "outputs/semkine/xyz_c1_stable_s3407/xyz_c1_stable_s3407-step=6000.ckpt",
           "sha256": "41a8dc3feac3e7cffc5b19ec93491ec2c7a841c902ee583c39d8435eb5c08091"},
    "A3": {"config": "configs/xcl_20261019/xcl_a3f0_s3407.yaml",
           "ckpt": "outputs/xcl/20261019/runs/xcl_a3f0_s3407/repro_final.ckpt",
           "sha256": "4d2e6bf2b89b897759a00a497bcaa09d62e462ec0bd66da83a9fab34040ffd0c"},
}
STRICT_TRACE = {"path": "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407/strict_trace/states.npz",
                "sha256": "dc961819cd76e381dca5037595e2dc2280569af02e2b290258d1263fee726c9b"}
SOURCE_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face", "xag_bary", "xag_valid", "xag_depth")
POLICY_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}


# ------------------------------------------------------------------------------------------------ identity
def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def require_gpu() -> torch.device:
    """One process = one physical GPU of 4-7 (user authorisation 2026-10-24)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if v not in GPUS:
        raise SystemExit(f"AGG778 jobs run on exactly one physical GPU 4-7 (CUDA_VISIBLE_DEVICES={v!r})")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise SystemExit("CUDA_DEVICE_ORDER must be PCI_BUS_ID so that indices are physical")
    return torch.device("cuda")


def verify_protocol() -> dict:
    """Every GPU entry: the frozen file SHA-256 of outputs/agg778/protocol.json must match the tree."""
    if not PROTOCOL.is_file():
        raise SystemExit("AGG778 protocol.json missing: pre-register and freeze first")
    p = json.loads(PROTOCOL.read_text())
    bad = [k for k, v in p["files"].items() if sha(REPO / k) != v]
    if bad:
        raise SystemExit(f"AGG778 frozen sources changed: {bad}")
    return p


def set_policy():
    """The XCL evaluation flags (deterministic algorithms, no TF32)."""
    for k, v in POLICY_ENV.items():
        os.environ.setdefault(k, v)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")


def save_json(path: Path, value, exclusive: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as f:
            f.write(text)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        tmp.replace(path)


def receipt(out_dir: Path, extra: dict, scripts=()):
    rec = {"scripts": {str(Path(s).resolve().relative_to(REPO)): sha(s) for s in (__file__, *scripts)},
           "protocol_sha256": sha(PROTOCOL) if PROTOCOL.is_file() else None,
           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), **extra}
    save_json(Path(out_dir) / "receipt.json", rec, exclusive=False)


# ------------------------------------------------------------------------------------------------ sequences
def endpoint_grid(runs) -> List[tuple]:
    """XCL `endpoint_grid`: (run_id, end, previous_ms, is_first) for ends a+49, a+99, ... < b."""
    rows = []
    for rid, (a, b) in enumerate(np.asarray(runs, np.int64).reshape(-1, 2)):
        previous = int(a)
        for end in np.arange(int(a) + 49, int(b), STEP, dtype=np.int64):
            rows.append((rid, int(end), previous, previous == int(a)))
            previous = int(end)
    return rows


def split_seqs(split: str):
    from semkine.dataset import sequences_for_split
    return sequences_for_split(DATA_ROOT, split, MANIFEST)


def subject_seqs(subjects) -> list:
    return [(s, d) for s, d in split_seqs("train") if s.split("_")[0] in subjects]


def zgz_seqs() -> list:
    s = split_seqs("val_core")
    if [x for x, _ in s] != list(ZGZ):
        raise RuntimeError("expected the complete original zgz_global / zgz_local")
    return s


class Seq:
    """One recording: events (mmap), offsets, sub-ms times, GT 51-D per ms, valid runs, betas, K, endpoint grid."""

    def __init__(self, name: str, d: str):
        from semkine import eval_track as ET
        ev, off, aux, pos = ET.load_sequence(DATA_ROOT, d, name)
        tsub = DATA_ROOT / d / f"{name}_tsub.npy"
        self.name, self.d = name, d
        self.events, self.offsets = ev, np.asarray(off, np.int64)
        self.tsub = np.load(tsub, mmap_mode="r") if tsub.exists() else None
        self.pos51 = np.asarray(pos, np.float32)
        self.runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        self.betas = np.asarray(aux["betas"], np.float32).reshape(-1)
        self.K = np.asarray(aux["camera_K"], np.float32).reshape(3, 3)
        self.grid = endpoint_grid(self.runs)

    def window(self, end: int):
        """Evaluator ev5 rows [0, x, y, t_rel_s, p] of [end-49, end], their original rows and milliseconds."""
        from semkine import eval_track as ET
        start = int(end) - STEP + 1
        ev5 = ET._window_events(self.events, self.offsets, self.tsub, int(end), STEP)
        rows = np.arange(int(self.offsets[start]), int(self.offsets[int(end) + 1]), dtype=np.int64)
        ms = np.repeat(np.arange(start, int(end) + 1, dtype=np.int64),
                       np.diff(self.offsets[start:int(end) + 2]).astype(np.int64))
        if len(rows) != len(ev5):
            raise RuntimeError(f"window rows disagree with evaluator events: {self.name}:{end}")
        return ev5, rows, ms


def sources(seq_name: str, rows: np.ndarray, ms: np.ndarray) -> Dict[str, np.ndarray]:
    """GT source labels (XAG cache, with the XCL depth sidecar) of event rows, in caller order."""
    from semkine.xag_sources import load_sources
    return load_sources(str(SOURCE_ROOT), seq_name, rows, ms, require_depth=True)


class Endpoints(torch.utils.data.Dataset):
    """One item = one endpoint of the XCL grid: evaluator events of [end-49, end], GT at the previous endpoint and at
    end, and (always) the GT source labels of the same rows."""

    def __init__(self, seqs, stride: int = 1, offset: int = 0, with_donor: bool = False):
        self.with_donor = with_donor
        self.seqs = [Seq(s, d) for s, d in seqs]
        self.index = []
        for si, sq in enumerate(self.seqs):
            k_in_run = 0
            last_rid = -1
            for k, (rid, end, prev, first) in enumerate(sq.grid):
                k_in_run = 0 if rid != last_rid else k_in_run + 1
                last_rid = rid
                if k % stride == offset % stride:
                    self.index.append((si, k, rid, end, prev, bool(first), k_in_run // BLOCK))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, k, rid, end, prev, first, blk = self.index[i]
        sq = self.seqs[si]
        ev5, rows, ms = sq.window(end)
        out = {"i": i, "si": si, "k": k, "seq": sq.name, "rid": rid, "end": end, "prev": prev, "first": first,
               "blk": blk,
               "ev5": ev5, "rows": rows, "ms": ms, "src": sources(sq.name, rows, ms),
               "gt_prev": sq.pos51[prev].copy(), "gt_end": sq.pos51[end].copy(),
               "betas": sq.betas, "K": sq.K}
        if self.with_donor:
            out["donor"] = self.donor(si, k)
        return out

    def donor(self, si: int, k: int) -> dict:
        """Null events: the window half a recording away in the same sequence (its own rows, sources and count),
        retimed into the current window by construction (ev5 times are window-relative)."""
        sq = self.seqs[si]
        kd = (k + len(sq.grid) // 2) % len(sq.grid)
        end = sq.grid[kd][1]
        ev5, rows, ms = sq.window(end)
        return {"ev5": ev5, "src": sources(sq.name, rows, ms), "end": end}


def permute_positions(items, seed: int):
    """Position-only decomposition input: inside every packet the (x, y) of the rows are permuted while each row keeps
    its time and polarity (rows stay time ordered); the source labels follow the positions."""
    rng = np.random.default_rng(seed)
    evs, srcs = [], []
    for it in items:
        e = np.array(it["ev5"], np.float32, copy=True)
        pi = rng.permutation(len(e))
        e[:, 1:3] = e[pi, 1:3]
        evs.append(e)
        s = it["src"]
        srcs.append({k: np.asarray(s[k])[pi] for k in s})
    return evs, srcs


def collate(items):
    return items


def loader(ds, batch: int, shuffle: bool, seed: int, workers: int = 8, drop_last: bool = False):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, generator=g, num_workers=workers,
                                       collate_fn=collate, drop_last=drop_last, persistent_workers=workers > 0,
                                       prefetch_factor=2 if workers > 0 else None)


def block_ids(items) -> np.ndarray:
    """Bootstrap block per endpoint: (sequence index, run, 20-endpoint block within the run)."""
    return np.array([it["si"] * 10 ** 6 + it["rid"] * 10 ** 3 + it["blk"] for it in items], np.int64)


# ------------------------------------------------------------------------------------------------ anchors
def noise_sampler(cfg):
    """C1's `TRACK.PREV_NOISE_MODE: mixed` draw, by the dataset's own `_sample_prev_noise` (same semantics, bit for
    bit for a given generator state)."""
    from semkine.dataset import SemKineDataset
    t = cfg.get("TRACK", {}) or {}
    mix = (float(t.get("PREV_NOISE_P_SMALL", 0.5)), float(t.get("PREV_NOISE_P_LARGE", 0.3)),
           float(t.get("PREV_NOISE_P_CORR", 0.2)))
    s = sum(mix) or 1.0
    ns = SimpleNamespace(prev_noise=(float(t.get("PREV_NOISE_T", 0.0)), float(t.get("PREV_NOISE_R", 0.0)),
                                     float(t.get("PREV_NOISE_POSE", 0.0))),
                         prev_noise_mode=str(t.get("PREV_NOISE_MODE", "gaussian")),
                         prev_noise_large=(float(t.get("PREV_NOISE_LARGE_T", 0.05)),
                                           float(t.get("PREV_NOISE_LARGE_R", 0.3)),
                                           float(t.get("PREV_NOISE_LARGE_POSE", 0.3))),
                         prev_noise_mix=tuple(v / s for v in mix))

    def draw(rng: np.random.Generator):
        u_state = rng.bit_generator.state
        u = float(rng.random())
        rng.bit_generator.state = u_state
        n = SemKineDataset._sample_prev_noise(ns, rng)
        kind = 0 if u < ns.prev_noise_mix[0] else (1 if u < ns.prev_noise_mix[0] + ns.prev_noise_mix[1] else 2)
        return (np.zeros(51, np.float32) if n is None else n.astype(np.float32)), kind
    return draw


def strict_trace():
    """C1's own closed-loop incoming states on zgz (the E2 / R-loop anchors): {seq: (n, 51)} in endpoint-grid order
    (the strict trace stores `incoming` / `prediction` only, one row per grid endpoint)."""
    path = REPO / STRICT_TRACE["path"]
    if sha(path) != STRICT_TRACE["sha256"]:
        raise RuntimeError("canonical C1 strict trace changed")
    with np.load(path, allow_pickle=False) as z:
        out = {s: np.asarray(z[s + "/incoming"], np.float32) for s in ZGZ}
    for s in ZGZ:
        if out[s].shape != (ZGZ_COUNTS[s], 51) or not np.isfinite(out[s]).all():
            raise RuntimeError("incomplete canonical trace")
    return out


def mano32(cfg, device):
    """The evaluator's float32 ManoLayer (`tools/xcl/evaluate.py` scoring)."""
    from mano_layer import ManoLayer
    return ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()


# ------------------------------------------------------------------------------------------------ frozen modules
def load_frozen(cond: str, device):
    """The frozen C1 (A0) or A3F0 (A3) at step 6000, SHA-checked, eval mode, no gradients, capture on."""
    from config import load_config
    from model import MNISTModel
    m = MODELS[cond]
    ckpt = REPO / m["ckpt"]
    if sha(ckpt) != m["sha256"]:
        raise RuntimeError(f"frozen {cond} checkpoint SHA changed")
    cfg = load_config(str(REPO / m["config"]))
    model = MNISTModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location=device, strict=True).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.xyz_keep_aux = True
    store = {}

    def keep(module, inputs):
        store["x257"] = inputs[0].detach()
    model.xyz.vin.register_forward_pre_hook(keep)
    model._agg778_store = store
    return model, cfg


def make_batch(items, anchors: np.ndarray, device, events: Optional[Sequence[np.ndarray]] = None,
               srcs: Optional[Sequence[dict]] = None, with_sources: bool = True):
    """`EventPacketBatch` of B evaluator packets (as `eval_track.make_eval_packet`, batched). `events` / `srcs`
    optionally replace an item's rows (nulls, thinning) together with their source labels."""
    from semkine.events import EventPacketBatch
    B = len(items)
    evs, ptr, src = [], [0], {k: [] for k in SOURCE_FIELDS}
    for b, it in enumerate(items):
        e = np.array(events[b] if events is not None else it["ev5"], np.float32, copy=True)
        e[:, 0] = b
        evs.append(e)
        ptr.append(ptr[-1] + len(e))
        if with_sources:
            s = srcs[b] if srcs is not None else it["src"]
            if len(s["xag_face"]) != len(e):
                raise RuntimeError("source rows disagree with packet rows")
            for k in SOURCE_FIELDS:
                src[k].append(np.asarray(s[k]))
    ev = np.concatenate(evs) if ptr[-1] else np.zeros((0, 5), np.float32)
    f32 = dict(device=device, dtype=torch.float32)
    batch = EventPacketBatch(
        events=torch.from_numpy(ev).to(device), ptr=torch.tensor(ptr, dtype=torch.int64, device=device),
        sequence_id=torch.zeros(B, dtype=torch.int64, device=device),
        t_start_us=torch.zeros(B, dtype=torch.int64, device=device),
        t_end_us=torch.full((B,), STEP * 1000, dtype=torch.int64, device=device),
        delta_t_s=torch.full((B,), STEP * 1e-3, **f32),
        is_sequence_start=torch.zeros(B, dtype=torch.bool, device=device),
        is_sequence_end=torch.zeros(B, dtype=torch.bool, device=device),
        target=torch.as_tensor(np.stack([it["gt_end"] for it in items]), **f32),
        prev_state=torch.as_tensor(np.asarray(anchors, np.float32), **f32),
        betas=torch.as_tensor(np.stack([it["betas"] for it in items]), **f32),
        camera_K=torch.as_tensor(np.stack([it["K"] for it in items]), **f32), lnes=None)
    if with_sources:
        for k in SOURCE_FIELDS:
            a = np.concatenate(src[k]) if ptr[-1] else np.zeros((0, 3) if k == "xag_bary" else (0,))
            dt = {"xag_bary": torch.float32, "xag_depth": torch.float32, "xag_valid": torch.bool}.get(k, torch.int64)
            setattr(batch, k, torch.as_tensor(a).to(device=device, dtype=dt))
    return batch


@torch.no_grad()
def forward_capture(model, batch) -> dict:
    """The frozen module's own forward on `batch` and everything the probes read from it."""
    model.xyz_last = None
    out = model.forward_packet(batch)
    last = model.xyz_last
    if last is None or last.get("xag_empty"):
        raise RuntimeError("all-empty packet batch: no capture")
    B = out.shape[0]
    cap = {"out": out.float(), "H0": last["H0"].float(), "O": last["O"].float(), "S": last["S"].float(),
           "bslot": last["bslot"].float(), "prev48": last["prev48"].float(), "verts": last["verts"].float(),
           "d": last["d"].float(), "mask": last["mask"], "tok": last["tok"].float(), "w": last["w"].float(),
           "r": last["r"].float(), "assoc": last["assoc"], "src": last["src"],
           "x257": model._agg778_store["x257"].float().reshape(B, V, -1),
           "counts": batch.counts.clone()}
    for k in ("xag_depth", "xag_valid", "xag_face", "xag_bary", "xag_part"):
        if k in last:
            cap[k] = last[k]
    return cap


@torch.no_grad()
def fk_verts(model, params: torch.Tensor, batch) -> torch.Tensor:
    """(B, 778, 3) float32 camera-frame vertices of 51-D states, by the model's own FK (as its `verts`)."""
    betas_f, _ = model._resolve_betas_K(params, batch.betas, batch.camera_K)
    v, _ = model._fk(params.float(), betas_f)
    return v.float()


def intrinsics(model, batch):
    _, k_f = model._resolve_betas_K(batch.prev_state, batch.betas, batch.camera_K)
    return model._intrinsics(k_f)


def project(verts: torch.Tensor, fx, fy, cx, cy):
    z = verts[..., 2].clamp(min=1e-6)
    return torch.stack([fx[:, None] * verts[..., 0] / z + cx[:, None], fy[:, None] * verts[..., 1] / z + cy[:, None]], -1)


@torch.no_grad()
def vertex_geometry(model, batch, verts: torch.Tensor) -> dict:
    """Anchor-mesh vertex frame (P4 convention): uv, z, projected-normal frame R (columns e1 = image direction of the
    normal, e2), |n . d_hat|, rho = f 0.02 / z, and z-buffer visibility (S44 `visible`)."""
    from semkine import xvm as XV
    from semkine.xyz_mesh import vertex_normals
    from s44 import evflow as EF
    fx, fy, cx, cy = intrinsics(model, batch)
    uv = project(verts, fx, fy, cx, cy)
    z = verts[..., 2]
    normals = vertex_normals(verts.float(), model.mano.f)
    Jp = XV.projection_jacobian(verts.double(), fx.double(), fy.double())
    R, cos = XV.vertex_frames(verts.double(), normals.double(), Jp)
    f = torch.sqrt(fx * fy)
    rho = (f[:, None] * RHO_M / z.clamp(min=1e-3)).float()
    vis = torch.stack([EF.visible(uv[b].double(), z[b].double()) for b in range(verts.shape[0])])
    return {"uv": uv, "z": z, "R": R.float(), "cos": cos.float(), "rho": rho, "vis": vis, "f": f,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "normals": normals}


def gt_owned(items, device) -> torch.Tensor:
    """(B, 778) bool: vertices owning a GT-source event of the window (operator-neutral support; positive
    barycentric weight of the event's GT source face)."""
    out = torch.zeros(len(items), V, dtype=torch.bool, device=device)
    for b, it in enumerate(items):
        s = it["src"]
        ok = np.asarray(s["xag_valid"], bool) & (np.asarray(s["xag_face"]) >= 0)
        if ok.any():
            out[b] = owned_from_faces(np.asarray(s["xag_face"])[ok], np.asarray(s["xag_bary"])[ok], device)
    return out


_FACES = {}


def mano_faces() -> np.ndarray:
    if "f" not in _FACES:
        with np.load(REPO / "assets" / "mano_right.npz", allow_pickle=False) as z:
            _FACES["f"] = np.asarray(z["f"], np.int64)
    return _FACES["f"]


def owned_from_faces(face: np.ndarray, bary: np.ndarray, device) -> torch.Tensor:
    ids = mano_faces()[face]                                                   # (P, 3)
    m = np.zeros(V, bool)
    m[ids[np.asarray(bary) > 0]] = True
    return torch.from_numpy(m).to(device)


# ------------------------------------------------------------------------------------------------ statistics
def block_bootstrap(values: Dict[str, np.ndarray], blocks: np.ndarray, n: int = 2000, seed: int = 20261024):
    """Mean and 95 % percentile CI of per-endpoint values (dict of equal-length arrays), resampling blocks."""
    ub, inv = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ub), size=(n, len(ub)))
    out = {}
    for k, v in values.items():
        v = np.asarray(v, np.float64)
        s = np.bincount(inv, weights=v, minlength=len(ub))
        c = np.bincount(inv, minlength=len(ub)).astype(np.float64)
        bs = s[draws].sum(1) / c[draws].sum(1)
        out[k] = {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                  "n": int(len(v)), "blocks": int(len(ub))}
    return out


def corr_from_stats(st: np.ndarray) -> np.ndarray:
    """Pearson r from summed sufficient statistics (..., 6) = [n, sx, sy, sxy, sxx, syy]."""
    n, sx, sy, sxy, sxx, syy = (st[..., i] for i in range(6))
    cov = sxy / np.maximum(n, 1) - sx * sy / np.maximum(n, 1) ** 2
    vx = sxx / np.maximum(n, 1) - (sx / np.maximum(n, 1)) ** 2
    vy = syy / np.maximum(n, 1) - (sy / np.maximum(n, 1)) ** 2
    return cov / np.sqrt(np.maximum(vx * vy, 1e-30))


def r2_from_stats(st: np.ndarray) -> np.ndarray:
    """R^2 of prediction x for target y from [n, sx, sy, sxy, sxx, syy]."""
    n, sx, sy, sxy, sxx, syy = (st[..., i] for i in range(6))
    sse = syy - 2 * sxy + sxx
    sst = syy - sy ** 2 / np.maximum(n, 1)
    return 1.0 - sse / np.maximum(sst, 1e-30)


def bootstrap_stats(block_stats: np.ndarray, fn, n: int = 2000, seed: int = 20261024):
    """block_stats (n_blocks, ..., 6) summed per block; returns point, lo, hi of fn over block resamples."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, block_stats.shape[0], size=(n, block_stats.shape[0]))
    point = fn(block_stats.sum(0))
    bs = np.stack([fn(block_stats[d].sum(0)) for d in draws])
    return point, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
