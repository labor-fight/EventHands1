"""DT render-fast contracts: the host-synchronisation-free render path is bit-for-bit the existing one
(findings: outputs/dt/reports/render_breakdown.md).

The reference is the REAL existing implementation -- `MNISTModel._render_prev` / `_render_chunk` / `forward` and
`ManoLayer.forward` of the live repo (MAIN, read only) -- run in a separate process whose import path is MAIN only,
built from this tree's configs/rt/rt_cnntrack.yaml, and compared by hash of the raw bytes (NaN payloads and the sign of
zero included). The subject is model/render_fast.py of THIS tree. What is pinned:

  * 2400+ random states (typical, far = colliding vertices, near, behind the camera, off frame, wild), B in
    {1, 7, 256, 300 (two chunks)}, per-state betas / K, for every variant (original, rasteriser only, FK only, both,
    cached per-sequence terms); the reference's own determinism is checked first
  * hand-made vertex clouds against the real `_render_chunk` (its `_fk` replaced): behind the camera, z at the 1e-6 edge,
    pixel edges (+-0.5 rounding, 0 and W / H), many vertices on one pixel (with and without ties), an all-invalid sample
    inside a batch, NaN / +-Inf, huge and denormal values, -0.0
  * `FastMano` (with and without the per-sequence terms) against `ManoLayer.forward`
  * cache semantics: only the context tensors hit it, an in-place edit or another tensor falls back, a stale cache is
    never served
  * GPU: zero host synchronisations (sync debug mode "error" and the profiler), the existing path does trip them, and a
    CUDA-graph replay of the whole forward is bit-identical, also for a new sequence's betas / K, also in a closed loop
  * `ManoLayer.fast_chain` (default off): the Python-int kinematic chain is bitwise the original layer, has no host
    syncs, and changes no state_dict key
  * `dense_forward` replicates every forward option of the DT rounds (CAM_PLANES, ROOT_HEAD spatial / anchor,
    PREV_MLP_TRANSL false, ROOT_COMPOSE so3) and raises NotImplementedError for an arm it does not know
  * `GraphedForward` records the model's own `forward` by default (all arms), after `deploy_fast.enable_fast_render`
  * `deploy_fast.enable_fast_render` (opt-in, on the model instance; model/model.py untouched): on / off / new sequence,
    bitwise the original model, eager and graphed

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_dt_render.py -q -p no:cacheprovider
    CUDA_VISIBLE_DEVICES=7  OMP_NUM_THREADS=2 python -m pytest tests/test_dt_render.py -q -p no:cacheprovider -k gpu
"""
from __future__ import annotations

import collections
import copy
import faulthandler
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
MAIN = Path("/data1/lyq/code/mesh/EventHands1")       # the live repo: read only, imported by the reference process only
CONFIG = REPO / "configs" / "rt" / "rt_cnntrack.yaml"
BASE_K = torch.tensor([[603.4507, 0.0, 325.09183], [0.0, 602.95654, 242.09796], [0.0, 0.0, 1.0]])
H, W, N_VERT = 180, 240, 778


def _digest(t):
    return hashlib.blake2b(t.detach().cpu().contiguous().numpy().tobytes(), digest_size=16).hexdigest()


def _digests(t):
    return [_digest(t[i]) for i in range(t.shape[0])]


# ------------------------------------------------------------------------------------ reference process (MAIN only)
def _ref_worker_main(req_path, out_path):
    """Fresh process, import path = the live repo only: the outputs of the code as it exists."""
    sys.path[:0] = [str(MAIN), str(MAIN / "model")]
    req = torch.load(req_path)
    import mano_layer
    from config import load_config
    from model import MNISTModel

    cfg = load_config(req["config"])
    for sec, kv in (req.get("overrides") or {}).items():
        cfg[sec].update(kv)
    dev = torch.device(req["device"])
    torch.manual_seed(0)
    m = MNISTModel(cfg).to(dev).eval()
    if req.get("state_dict") is not None:
        m.load_state_dict(req["state_dict"])
    out = {"files": {"model": sys.modules["model"].__file__, "mano_layer": mano_layer.__file__}}
    kind = req["kind"]
    digs = []
    with torch.no_grad():
        if kind == "render":
            prev, betas, K = (req[k].to(dev) for k in ("prev", "betas", "K"))
            i, same = 0, True
            for B in req["batches"]:
                p, b, k = prev[i:i + B], betas[i:i + B], K[i:i + B]
                if req.get("ctx") and B == 1:
                    m.set_hand_context(b, k)

                    def call():
                        return m._render_prev(p, *m._resolve_betas_K(p, None, None))
                else:
                    def call():
                        return m._render_prev(p, b, k)
                r = call()
                same = same and bool(torch.equal(r, call()))          # is the reference itself deterministic?
                digs += _digests(r)
                i += B
            out["self_consistent"] = same
        elif kind == "verts":
            V, K = req["verts"].to(dev), req["K"].to(dev)
            cur = {}
            m._fk = lambda params, betas: (cur["v"], None)            # the real _render_chunk, any vertex cloud
            i = 0
            for B in req["batches"]:
                cur["v"] = V[i:i + B]
                r = m._render_chunk(torch.zeros(B, 51, device=dev), torch.zeros(B, 10, device=dev), K[i:i + B])
                digs += _digests(r)
                i += B
        elif kind == "mano":
            a = {k: req[k].to(dev) for k in ("betas", "orient", "pose", "transl")}
            vd, jd, i = [], [], 0
            for B in req["batches"]:
                v, j = m.mano(a["betas"][i:i + B], a["orient"][i:i + B], a["pose"][i:i + B], a["transl"][i:i + B])
                vd += _digests(v)
                jd += _digests(j)
                i += B
            out["verts"], out["joints"] = vd, jd
        elif kind == "forward":
            m.set_hand_context(req["betas"].to(dev), req["K"].to(dev))
            xs, prevs = req["x"].to(dev), req["prev"].to(dev)
            preds, prev = [], prevs[:1]
            for t in range(xs.shape[0]):
                pred = m(xs[t], prev if req["closed_loop"] else prevs[t:t + 1])
                preds.append(pred.cpu())
                prev = pred
            out["preds"] = torch.cat(preds)
        else:
            raise ValueError(kind)
    out["digests"] = digs
    torch.save(out, out_path)


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--ref-worker":
    _ref_worker_main(sys.argv[2], sys.argv[3])
    sys.exit(0)

sys.path[:0] = [str(REPO), str(REPO / "model")]
import render_fast as RF                                  # noqa: E402
import deploy_fast as DF                                  # noqa: E402
from config import load_config                            # noqa: E402
from model import MNISTModel                              # noqa: E402

pytestmark = pytest.mark.skipif(not (MAIN / "model" / "model.py").exists(), reason="live repo (reference) not found")
if torch.cuda.is_available():
    # a GPU test process must never hang on the shared GPU (faulthandler's timer is a C thread: it fires even if the main
    # thread is stuck inside a library call, e.g. a profiler); DT_TEST_WATCHDOG_S (default 1700 s) is the limit of the whole run
    faulthandler.dump_traceback_later(int(os.environ.get("DT_TEST_WATCHDOG_S", "1700")), exit=True)


def run_ref(req):
    with tempfile.TemporaryDirectory() as td:
        rp, op = Path(td) / "req.pt", Path(td) / "out.pt"
        torch.save(req, rp)
        r = subprocess.run([sys.executable, str(HERE), "--ref-worker", str(rp), str(op)], capture_output=True,
                           text=True, env=os.environ.copy(), timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        out = torch.load(op)
    assert out["files"]["model"].startswith(str(MAIN)), out["files"]          # the reference really is the live repo
    assert out["files"]["mano_layer"].startswith(str(MAIN)), out["files"]
    return out


def assert_same(got, ref, what):
    assert len(got) == len(ref), f"{what}: {len(got)} results vs {len(ref)}"
    bad = [i for i, (a, b) in enumerate(zip(got, ref)) if a != b]
    assert not bad, f"{what}: {len(bad)} of {len(ref)} results differ bitwise, first indices {bad[:10]}"


def bits_equal(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    return a.shape == b.shape and a.dtype == b.dtype == torch.float32 and torch.equal(a.view(torch.int32), b.view(torch.int32))


# ------------------------------------------------------------------------------------ inputs
def random_states(n, seed):
    """(prev (n, 51), betas (n, 10), K (n, 3, 3)); state i is of kind i % 10: 0-4 typical, 5 far (collisions),
    6 very near, 7 behind the camera, 8 off to one side, 9 wild."""
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g)

    def u(*s):
        return torch.rand(*s, generator=g)

    kind = torch.arange(n) % 10
    prev = torch.zeros(n, 51)
    t = torch.stack([0.06 * r(n), 0.05 * r(n), 0.35 + 0.45 * u(n)], 1)
    k = kind == 5
    t[k, 2] = 2.0 + 4.0 * u(int(k.sum()))
    k = kind == 6
    t[k, 2] = 0.04 + 0.10 * u(int(k.sum()))
    k = kind == 7
    t[k, 2] = -(0.2 + u(int(k.sum())))
    k = kind == 8
    sign = (u(int(k.sum())) < 0.5).float() * 2 - 1
    t[k, 0] = sign * (0.6 + u(int(k.sum())))
    k = kind == 9
    t[k] = 0.3 * r(int(k.sum()), 3) + torch.tensor([0.0, 0.0, 0.6])
    axis = r(n, 3)
    axis = axis / axis.norm(dim=1, keepdim=True)
    prev[:, 0:3] = t
    prev[:, 3:6] = axis * (math.pi * u(n, 1))
    prev[:, 6:] = ((0.35 + 0.85 * (kind == 9).float())[:, None] * r(n, 45)).clamp(-2.0, 2.0)
    betas = 0.8 * r(n, 10)
    K = BASE_K.repeat(n, 1, 1)
    K[:, 0, 0] *= 1 + 0.03 * r(n)
    K[:, 1, 1] *= 1 + 0.03 * r(n)
    K[:, 0, 2] += 10 * r(n)
    K[:, 1, 2] += 10 * r(n)
    return prev, betas, K


def random_packets(T, seed):
    """(T, 1, H, W, 2) event images; packet 0 is empty (the zero-event gate)."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(T, 1, H, W, 2, generator=g) < 0.03).float() * torch.rand(T, 1, H, W, 2, generator=g)
    x[0] = 0.0
    return x


def crafted_cases():
    """[(name, vertices (n, 778, 3) float32)] -- hand-made clouds; K is BASE_K for all."""
    g = torch.Generator().manual_seed(11)
    d, n = torch.float64, N_VERT

    def rnd(*s):
        return torch.rand(*s, generator=g, dtype=d)

    Kd = BASE_K.double()
    fx, fy, cx, cy = Kd[0, 0] * 0.375, Kd[1, 1] * 0.375, Kd[0, 2] * 0.375, Kd[1, 2] * 0.375

    def xyz(u, v, z):
        return torch.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], -1).float()

    def inframe(k):
        return 10 + 220 * rnd(k, n), 10 + 160 * rnd(k, n), 0.3 + 0.6 * rnd(k, n)

    cases = [("typical", xyz(*inframe(3)))]
    u, v, _ = inframe(2)
    z = -rnd(2, n)
    z[1] = 0.0
    z[0, 0] = -0.0
    cases.append(("behind_camera", xyz(u, v, z)))
    u, v, z = inframe(3)
    cases.append(("behind_mixed", xyz(u, v, torch.where(rnd(3, n) < 0.5, -rnd(3, n), z))))
    t1 = torch.tensor(1e-6, dtype=torch.float32)
    zs = torch.stack([t1, torch.nextafter(t1, torch.tensor(1.0)), torch.nextafter(t1, torch.tensor(0.0)),
                      torch.tensor(0.0), torch.tensor(-0.0), torch.tensor(1e-7), torch.tensor(2e-6), torch.tensor(1e-5)])
    V = torch.zeros(1, n, 3)
    V[0, :, 2] = zs[torch.arange(n) % len(zs)]
    cases.append(("z_threshold", V))                                       # x = y = 0: every vertex at (cx, cy)
    eu = torch.tensor([-1.0, -0.6, -0.5001, -0.5, -0.4999, -1e-3, 0.0, 0.4999, 0.5, 0.5001, 1.0, W - 1.5, W - 1.0,
                       W - 0.5001, W - 0.5, W - 0.4999, W - 1e-3, float(W), W + 0.5, W + 1.0], dtype=d)
    ev = torch.tensor([-1.0, -0.6, -0.5001, -0.5, -0.4999, -1e-3, 0.0, 0.4999, 0.5, 0.5001, 1.0, H - 1.5, H - 1.0,
                       H - 0.5001, H - 0.5, H - 0.4999, H - 1e-3, float(H), H + 0.5, H + 1.0], dtype=d)
    iu = torch.randint(len(eu), (4, n), generator=g)
    iv = torch.randint(len(ev), (4, n), generator=g)

    def jit():
        return (rnd(4, n) < 0.5).to(d) * (rnd(4, n) - 0.5) * 2e-4

    cases.append(("pixel_edges", xyz(eu[iu] + jit(), ev[iv] + jit(), 0.3 + 0.5 * rnd(4, n))))
    ones = torch.ones(2, n, dtype=d)
    u, v = 120.3 * ones, 90.2 * ones
    u[1], v[1] = 0.0, 0.0
    cases.append(("collide_same_z", xyz(u, v, 0.5 * ones)))                # all vertices on one pixel, exact ties
    ties = torch.tensor([0.31, 0.5, 0.5, 0.75, 0.31], dtype=d)
    z = torch.stack([ties[torch.randint(5, (n,), generator=g)], 0.3 + 0.6 * rnd(n)])
    cases.append(("collide_diff_z", xyz(119.9 * ones, 90.1 * ones, z)))
    u, v, z = inframe(3)
    u[:, :400], v[:, :400] = 77.2, 33.7
    cases.append(("collide_partial", xyz(u, v, z)))
    uA, vA, zA = inframe(1)
    uB, vB, zB = inframe(1)
    uC, vC, zC = 120.3 * ones[:1], 90.2 * ones[:1], 0.3 + 0.6 * rnd(1, n)
    cases.append(("all_invalid_inside_batch", torch.cat([xyz(uA, vA, zA), xyz(uB, vB, -zB), xyz(uC, vC, zC)])))
    base = xyz(*inframe(6))
    V = base.clone()
    V[rnd(6, n, 3) < 0.05] = float("nan")
    m = rnd(6, n, 3) < 0.03
    V[m] = ((rnd(int(m.sum())) < 0.5).float() * 2 - 1) * float("inf")
    V[5] = float("nan")
    V[4, :, 2] = float("inf")
    cases.append(("nan_inf", V))
    V = torch.zeros(3, n, 3)
    V[0, :, 2] = torch.tensor([1e10, 1e30, 3.0e38, float("inf"), 1e-30, 5.0])[torch.arange(n) % 6]
    V[1, :, 0] = V[1, :, 1] = 1e30
    V[1, :, 2] = 0.5
    V[2, :, 0] = V[2, :, 1] = float("inf")
    V[2, :, 2] = 0.5
    cases.append(("huge", V))
    u, v, z = 10 + 220 * rnd(4, n) - 40 + 80 * rnd(4, n), 10 + 160 * rnd(4, n) - 40 + 80 * rnd(4, n), 0.05 + 1.95 * rnd(4, n)
    cases.append(("random_wide", xyz(u, v, torch.where(rnd(4, n) < 0.05, -z, z))))
    V = torch.full((1, n, 3), -0.0)
    V[..., 2] = 0.5
    cases.append(("negative_zero_xy", V))
    V = torch.zeros(1, n, 3)
    V[0, :, 2] = torch.tensor([1e-38, 1.4e-45, 1e-20, 1e-7])[torch.arange(n) % 4]
    cases.append(("denormal_z", V))
    ar = torch.arange(n, dtype=d)
    cases.append(("dense_grid", xyz((20 + (ar % 40) * 5)[None], (20 + (ar // 40) * 7)[None], 0.3 + 0.6 * rnd(1, n))))
    z0 = torch.tensor(0.2, dtype=torch.float32)
    zc = torch.stack([z0 + k * (torch.nextafter(z0, torch.tensor(1.0)) - z0) for k in range(-3, 4)])
    zz = zc[torch.arange(n) % 7].double()[None]
    cases.append(("inv_clamp_boundary", xyz((5 + (ar % 230))[None], (20 + (ar // 230) * 5)[None], zz)))
    return cases


# ------------------------------------------------------------------------------------ subject side
def subject_variants(model):
    return {
        "stage_existing": None,
        "raster": RF.RenderFast(model, raster=True, fk=False, cache=False),
        "fk": RF.RenderFast(model, raster=False, fk=True, cache=False),
        "raster+fk": RF.RenderFast(model, raster=True, fk=True, cache=False),
        "cached": RF.RenderFast(model, raster=True, fk=True, cache=True),
    }


def subject_render(model, var, prev, betas, K, batches, use_ctx):
    digs, i = [], 0
    for B in batches:
        p, b, k = prev[i:i + B], betas[i:i + B], K[i:i + B]
        if use_ctx and B == 1:
            model.set_hand_context(b, k)
            if var is not None:
                var.set_context(model._ctx_betas, model._ctx_K)
            args = model._resolve_betas_K(p, None, None)
        else:
            args = (b, k)
        r = model._render_prev(p, *args) if var is None else var(p, *args)
        digs += _digests(r)
        i += B
    return digs


def coverage(model, prev, betas, K):
    """How many of the states exercise each corner (computed from the subject's own FK and projection)."""
    c = collections.Counter()
    for i0 in range(0, len(prev), 256):
        p, b, k = prev[i0:i0 + 256], betas[i0:i0 + 256], K[i0:i0 + 256]
        with torch.no_grad():
            verts, _ = model._fk(p, b)
        fx, fy, cx, cy = (t[:, None] for t in model._intrinsics(k))
        _, ui, vi, valid = RF.project_pixels(verts, fx, fy, cx, cy, H, W)
        z = verts[..., 2]
        for j in range(len(p)):
            v = valid[j]
            n_valid = int(v.sum())
            c["all_invalid"] += int(n_valid == 0)
            c["some_valid"] += int(n_valid > 0)
            c["collisions"] += int(n_valid > int((vi[j] * W + ui[j])[v].unique().numel()))
            c["some_behind"] += int(bool((z[j] <= 1e-6).any()))
            c["some_outside"] += int(bool(((z[j] > 1e-6) & ~v).any()))
    return c


def _reset(m):
    m.set_hand_context(None, None)


PLANS = [("B1", [1] * 400, True, 101), ("B7", [7] * 60, False, 102), ("B256", [256] * 4, False, 103),
         ("B300", [300] * 2, False, 104)]
PLANS_GPU = [("B1", [1] * 100, True, 201), ("B7", [7] * 8, False, 202), ("B256", [256], False, 203),
             ("B300", [300], False, 204)]


def _check_plans(model, dev, plans, min_total, min_cover):
    model = model.to(dev)
    total, cover = 0, collections.Counter()
    for name, batches, use_ctx, seed in plans:
        _reset(model)
        n = sum(batches)
        total += n
        prev, betas, K = random_states(n, seed)
        ref = run_ref(dict(kind="render", config=str(CONFIG), device=dev, prev=prev, betas=betas, K=K,
                           batches=batches, ctx=use_ctx))
        assert ref["self_consistent"], "the reference itself is not deterministic: bit-exactness is not decidable"
        p, b, k = prev.to(dev), betas.to(dev), K.to(dev)
        for vname, var in subject_variants(model).items():
            _reset(model)
            got = subject_render(model, var, p, b, k, batches, use_ctx)
            assert_same(got, ref["digests"], f"{name} {vname}")
            if vname == "cached" and use_ctx:
                assert var.cache_hits == len(batches) and var.cache_misses == 0, (var.cache_hits, var.cache_misses)
        cover.update(coverage(model, p, b, k))
    assert total >= min_total, total
    for key, need in min_cover.items():
        assert cover[key] >= need, f"corner '{key}' covered by only {cover[key]} states ({dict(cover)})"
    _reset(model)
    return total, cover


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cpu_model():
    torch.manual_seed(0)
    return MNISTModel(load_config(CONFIG)).eval()


@pytest.fixture(scope="module")
def gpu_model():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU (CUDA_VISIBLE_DEVICES=7)")
    torch.manual_seed(0)
    return MNISTModel(load_config(CONFIG)).eval().to("cuda")


# ------------------------------------------------------------------------------------ random states vs the live repo
def test_random_states_bitwise_cpu(cpu_model):
    total, cover = _check_plans(cpu_model, "cpu", PLANS, 2400, {"all_invalid": 100, "collisions": 100, "some_behind": 50,
                                                          "some_outside": 100, "some_valid": 1000})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_random_states_bitwise_gpu(gpu_model):
    _check_plans(gpu_model, "cuda", PLANS_GPU, 700, {"all_invalid": 30, "collisions": 30, "some_behind": 15,
                                                "some_outside": 30, "some_valid": 300})


@pytest.mark.parametrize("channels", [["sil"], ["inv"], ["inv", "sil"], ["sil", "inv", "semsil"]])
def test_channel_sets_cpu(channels):
    cfg = load_config(CONFIG)
    cfg["MODEL"]["RENDER_CHANNELS"] = channels
    torch.manual_seed(0)
    model = MNISTModel(cfg).eval()
    prev, betas, K = random_states(40, 7)
    batches = [1] * 12 + [7] * 4
    ref = run_ref(dict(kind="render", config=str(CONFIG), device="cpu", prev=prev, betas=betas, K=K, batches=batches,
                       ctx=True, overrides={"MODEL": {"RENDER_CHANNELS": channels}}))
    for vname, var in subject_variants(model).items():
        _reset(model)
        assert_same(subject_render(model, var, prev, betas, K, batches, True), ref["digests"], f"{channels} {vname}")


# ------------------------------------------------------------------------------------ hand-made vertex clouds
def _check_crafted(model, dev):
    model = model.to(dev)
    cases = crafted_cases()
    V = torch.cat([v for _, v in cases])
    batches = [len(v) for _, v in cases]
    K = BASE_K.repeat(len(V), 1, 1)
    ref = run_ref(dict(kind="verts", config=str(CONFIG), device=dev, verts=V, K=K, batches=batches))["digests"]
    V, K = V.to(dev), K.to(dev)
    cur = {}
    wrapper = RF.RenderFast(model, raster=True, fk=False, cache=False)
    got = {"direct": [], "masked_copy": [], "wrapper": []}
    model._fk = lambda params, betas: (cur["v"], None)                       # the vertex cloud replaces the FK
    try:
        i = 0
        for B in batches:
            v, k = V[i:i + B], K[i:i + B]
            cur["v"] = v
            fx, fy, cx, cy = (t[:, None] for t in model._intrinsics(k))
            args = (v, fx, fy, cx, cy, model.render_h, model.render_w, model.render_channels)
            got["direct"] += _digests(RF.rasterize_nosync(*args))
            got["masked_copy"] += _digests(RF.rasterize_masked(*args))
            got["wrapper"] += _digests(wrapper.render_chunk(torch.zeros(B, 51, device=dev),
                                                            torch.zeros(B, 10, device=dev), k))
            i += B
    finally:
        del model._fk
    for key, digs in got.items():
        assert_same(digs, ref, f"crafted vertex clouds, {key}")
    # the cases really are the corners they claim to be
    starts = dict(zip([n for n, _ in cases], torch.tensor([0] + batches).cumsum(0).tolist()))
    z_valid = lambda name, j: RF.project_pixels(  # noqa: E731
        V[starts[name] + j:starts[name] + j + 1], *(t[:, None] for t in model._intrinsics(K[:1])), H, W)[3][0]
    assert not z_valid("behind_camera", 0).any() and not z_valid("behind_camera", 1).any()
    assert not z_valid("all_invalid_inside_batch", 1).any() and z_valid("all_invalid_inside_batch", 0).any()
    pe = z_valid("pixel_edges", 0)
    assert pe.any() and not pe.all()
    cs = z_valid("collide_diff_z", 1)
    _, ui, vi, _ = RF.project_pixels(V[starts["collide_diff_z"] + 1:starts["collide_diff_z"] + 2],
                                     *(t[:, None] for t in model._intrinsics(K[:1])), H, W)
    assert int(cs.sum()) == N_VERT and int((vi * W + ui)[0].unique().numel()) == 1


def test_crafted_vertex_clouds_cpu(cpu_model):
    _check_crafted(cpu_model, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_crafted_vertex_clouds_gpu(gpu_model):
    _check_crafted(gpu_model, "cuda")


def test_all_invalid_state_renders_nothing_cpu(cpu_model):
    prev, betas, K = random_states(10, 5)
    for kind_idx in (7, 8):                                                   # behind the camera / off to one side
        p, b, k = prev[kind_idx:kind_idx + 1], betas[kind_idx:kind_idx + 1], K[kind_idx:kind_idx + 1]
        _reset(cpu_model)
        ref = run_ref(dict(kind="render", config=str(CONFIG), device="cpu", prev=p, betas=b, K=k, batches=[1], ctx=True))
        cpu_model.set_hand_context(b, k)
        rf = RF.RenderFast(cpu_model, cache=True)
        rf.set_context(cpu_model._ctx_betas, cpu_model._ctx_K)
        r = rf(p, *cpu_model._resolve_betas_K(p, None, None))
        assert float(r.abs().sum()) == 0.0
        assert _digests(r) == ref["digests"]
    _reset(cpu_model)


# ------------------------------------------------------------------------------------ MANO
def _mano_inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(n, 3, generator=g)
    axis = axis / axis.norm(dim=1, keepdim=True)
    return {"betas": 0.9 * torch.randn(n, 10, generator=g), "orient": axis * (3.1 * torch.rand(n, 1, generator=g)),
            "pose": 0.6 * torch.randn(n, 45, generator=g),
            "transl": 0.1 * torch.randn(n, 3, generator=g) + torch.tensor([0.0, 0.0, 0.5])}


def _check_mano(model, dev):
    model = model.to(dev)
    plan = [1] * 40 + [5] * 6 + [64] * 2
    a = _mano_inputs(sum(plan), 5)
    ref = run_ref(dict(kind="mano", config=str(CONFIG), device=dev, batches=plan, **a))
    a = {k: v.to(dev) for k, v in a.items()}
    fm = RF.FastMano(model.mano)
    got = collections.defaultdict(list)
    i = 0
    with torch.no_grad():
        for B in plan:
            b, o, p, t = (a[k][i:i + B] for k in ("betas", "orient", "pose", "transl"))
            v, j = model.mano(b, o, p, t)                                      # this tree's layer
            got["layer_v"] += _digests(v)
            got["layer_j"] += _digests(j)
            v, j = fm.forward(b, o, p, t)
            got["fast_v"] += _digests(v)
            got["fast_j"] += _digests(j)
            v, j = fm.forward(b, o, p, t, need_joints=False)
            assert j is None
            got["fast_nojoints_v"] += _digests(v)
            if B == 1:
                sh = fm.shape_terms(b)
                v, j = fm.forward(b, o, p, t, shape=sh)
                got["cached_v"] += _digests(v)
                got["cached_j"] += _digests(j)
                v, _ = fm.forward(b, o, p, t, shape=sh, need_joints=False)
                got["cached_nojoints_v"] += _digests(v)
            i += B
    n1 = sum(1 for B in plan if B == 1)                                      # the B == 1 calls come first
    for key, digs in got.items():
        full = ref["joints"] if key.endswith("_j") else ref["verts"]
        assert_same(digs, full[:n1] if key.startswith("cached") else full, f"mano {key}")


def test_fastmano_equals_layer_cpu(cpu_model):
    _check_mano(cpu_model, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fastmano_equals_layer_gpu(gpu_model):
    _check_mano(gpu_model, "cuda")


def test_shape_terms_are_batch_one_only_cpu(cpu_model):
    fm = RF.FastMano(cpu_model.mano)
    a = _mano_inputs(3, 1)
    with pytest.raises(AssertionError):
        fm.shape_terms(a["betas"])
    sh = fm.shape_terms(a["betas"][:1])
    with pytest.raises(AssertionError):
        fm.forward(a["betas"], a["orient"], a["pose"], a["transl"], shape=sh)


# ------------------------------------------------------------------------------------ cache semantics
def test_cache_only_hits_the_context_cpu(cpu_model):
    m = cpu_model
    prev, betas, K = random_states(10, 9)
    p, b, k = prev[:1], betas[:1], K[:1]
    m.set_hand_context(b, k)
    rf = RF.RenderFast(m, cache=True)
    assert RF.RenderFast(m).cache is False                                   # off by default
    rf.set_context(m._ctx_betas, m._ctx_K)
    args = m._resolve_betas_K(p, None, None)
    r0 = rf(p, *args)
    assert rf.cache_hits == 1 and rf.cache_misses == 0
    assert bits_equal(r0, m._render_prev(p, *args))
    # another tensor with other values: the cache is not served
    b2 = betas[1:2].clone()
    r2 = rf(p, b2, k)
    assert rf.cache_misses == 1
    assert bits_equal(r2, m._render_prev(p, b2, k))
    assert not bits_equal(r2, r0)                                            # the test is sensitive to the betas
    # same values in another tensor: still the uncached path (identity, not equality, decides)
    r3 = rf(p, b.clone(), k)
    assert rf.cache_misses == 2 and bits_equal(r3, r0)
    # an in-place edit of the context bumps its version: the cache must not be served stale
    m._ctx_betas.add_(0.3)
    args = m._resolve_betas_K(p, None, None)
    r4 = rf(p, *args)
    assert rf.cache_misses == 3 and rf.cache_hits == 1
    assert bits_equal(r4, m._render_prev(p, *args)) and not bits_equal(r4, r0)
    m._ctx_K.mul_(1.01)
    args = m._resolve_betas_K(p, None, None)
    assert bits_equal(rf(p, *args), m._render_prev(p, *args))
    # B > 1 never uses the per-sequence terms
    hits = rf.cache_hits
    assert bits_equal(rf(prev[:3], betas[:3], K[:3]), m._render_prev(prev[:3], betas[:3], K[:3]))
    assert rf.cache_hits == hits
    # a new context, then none
    m.set_hand_context(betas[2:3], K[2:3])
    rf.set_context(m._ctx_betas, m._ctx_K)
    args = m._resolve_betas_K(p, None, None)
    assert bits_equal(rf(p, *args), m._render_prev(p, *args)) and rf.cache_hits == hits + 1
    rf.set_context(None, None)
    assert bits_equal(rf(p, *args), m._render_prev(p, *args)) and rf.cache_hits == hits + 1
    _reset(m)


def test_graph_construction_guards_cpu(cpu_model):
    with pytest.raises(ValueError):
        RF.GraphedForward(cpu_model, render=RF.RenderFast(cpu_model, cache=True))
    with pytest.raises(ValueError):
        RF.GraphedForward(cpu_model, render=RF.RenderFast(cpu_model, raster=False))
    with pytest.raises(ValueError):
        RF.RenderFast(cpu_model, fk=False, cache=True)


# ------------------------------------------------------------------------------------ whole forward / closed loop
def test_dense_forward_replica_equals_model_forward_cpu(cpu_model):
    m = cpu_model
    x = random_packets(3, 4)
    prev, betas, K = random_states(10, 6)
    for B, i in ((1, 0), (3, 3)):
        xb = x[:3, 0] if B == 3 else x[1]
        pb = prev[i:i + B]
        _reset(m)
        assert bits_equal(RF.dense_forward(m, xb, pb), m(xb, pb))
        m.set_hand_context(betas[:1], K[:1])
        rf = RF.RenderFast(m, cache=True)
        rf.set_context(m._ctx_betas, m._ctx_K)
        assert bits_equal(RF.dense_forward(m, xb, pb, render=rf), m(xb, pb))
        assert bits_equal(RF.dense_forward(m, xb, pb, render=RF.RenderFast(m)), m(xb, pb))
    _reset(m)


def _quiet_model(model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.rn.fc.weight.mul_(0.02)
        m.rn.fc.bias.mul_(0.02)
    return m


def _closed_loop(model, dev, steps, with_graph):
    """Chaotic feedback: the output is the next step's input. Any 1-ulp difference of any variant would grow."""
    m = _quiet_model(model).to(dev)
    prev, betas, K = random_states(10, 8)
    xs = random_packets(steps, 12)
    ref = run_ref(dict(kind="forward", config=str(CONFIG), device=dev, state_dict={k: v.cpu() for k, v in m.state_dict().items()},
                       betas=betas[:1], K=K[:1], x=xs, prev=prev[:1], closed_loop=True))["preds"]
    m.set_hand_context(betas[:1].to(dev), K[:1].to(dev))
    xs, p0 = xs.to(dev), prev[:1].to(dev)
    rf_nc = RF.RenderFast(m)
    rf_c = RF.RenderFast(m, cache=True)
    rf_c.set_context(m._ctx_betas, m._ctx_K)
    runs = {"existing": lambda x, p: m(x, p),
            "raster+fk": lambda x, p: RF.dense_forward(m, x, p, render=rf_nc),
            "cached": lambda x, p: RF.dense_forward(m, x, p, render=rf_c)}
    if with_graph:
        gf = RF.GraphedForward(m, render=RF.RenderFast(m))                        # the replica
        gf.capture(xs[1], p0, betas[:1].to(dev), K[:1].to(dev))
        runs["graph"] = lambda x, p: gf(x, p)
        m2 = _quiet_model(model).to(dev)                                          # the model's own forward
        DF.enable_fast_render(m2)
        m2.set_hand_context(betas[:1].to(dev), K[:1].to(dev))
        gf2 = RF.GraphedForward(m2)
        gf2.capture(xs[1], p0, betas[:1].to(dev), K[:1].to(dev))
        runs["graph (model.forward)"] = lambda x, p: gf2(x, p)
    with torch.no_grad():
        for name, fn in runs.items():
            prev_t, traj = p0, []
            for t in range(steps):
                prev_t = fn(xs[t], prev_t)
                traj.append(prev_t.cpu())
            assert bits_equal(torch.cat(traj), ref), f"closed loop, {name}"
    assert not torch.equal(ref[0], ref[-1])                                  # the loop really moves
    assert rf_c.cache_hits == steps


def test_closed_loop_bitwise_cpu(cpu_model):
    _closed_loop(cpu_model, "cpu", 10, with_graph=False)


# ------------------------------------------------------------------------------------ GPU: host synchronisation
def _runtime_calls(fn, reps=3):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "t.json")
        prof.export_chrome_trace(path)
        events = json.load(open(path))["traceEvents"]
    c = collections.Counter(e["name"] for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"))
    m = collections.Counter("DtoH" if "DtoH" in e["name"] else "HtoD" if "HtoD" in e["name"] else "DtoD"
                            for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy")
    out = {k: v / reps for k, v in c.items()}
    out.update({"memcpy_" + k: v / reps for k, v in m.items()})
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fast_path_has_no_host_sync_gpu(gpu_model):
    m = gpu_model
    prev, betas, K = (t.to("cuda") for t in random_states(4, 123))
    m.set_hand_context(betas[:1], K[:1])
    p = prev[:1]
    rf = RF.RenderFast(m, cache=True)
    rf.set_context(m._ctx_betas, m._ctx_K)
    rf_nc = RF.RenderFast(m)
    only_raster = RF.RenderFast(m, raster=True, fk=False)
    only_fk = RF.RenderFast(m, raster=False, fk=True)
    x = random_packets(2, 3)[1].to("cuda")
    args = m._resolve_betas_K(p, None, None)
    for _ in range(2):
        for f in (rf, rf_nc, only_raster, only_fk, m._render_prev):
            f(p, *args)
        RF.dense_forward(m, x, p, render=rf)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        rf(p, *args)
        rf_nc(p, *args)
        rf_nc(prev, betas, K)                                                # B = 4
        RF.dense_forward(m, x, p, render=rf)                                 # the whole forward, CNN included
    finally:
        torch.cuda.set_sync_debug_mode(0)
    torch.cuda.synchronize()
    # The detector is live: each old part trips it. The 15 hidden syncs of the existing FK are the python-list-indexing
    # `.item()` calls of ManoLayer.forward; with `ManoLayer.fast_chain` on (default off) there are none, so then the
    # existing FK and "only the rasteriser fixed" must be sync-free too.
    layer_patched = bool(getattr(m.mano, "fast_chain", False))
    assert layer_patched is False, "ManoLayer.fast_chain must be off by default"
    cases = (("existing render", lambda: m._render_prev(p, *args), True),               # boolean indexing: always syncs
             ("only the FK fixed", lambda: only_fk(p, *args), True),                    # ... the masked rasteriser
             ("existing FK", lambda: m._fk(p, args[0]), not layer_patched),
             ("only the rasteriser fixed", lambda: only_raster(p, *args), not layer_patched))
    for name, f, must_sync in cases:
        torch.cuda.set_sync_debug_mode("error")
        try:
            if must_sync:
                with pytest.raises(RuntimeError, match="synchroniz"):
                    f()
            else:
                f()
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    _reset(m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_profiler_counts_gpu(gpu_model):
    # eager runs only: torch 2.1's profiler (CUPTI) never returns after a CUDA-graph replay, so this test must stay
    # before every graph test of this file
    m = gpu_model
    prev, betas, K = (t.to("cuda") for t in random_states(2, 5))
    m.set_hand_context(betas[:1], K[:1])
    p, args = prev[:1], m._resolve_betas_K(prev[:1], None, None)
    rf = RF.RenderFast(m, cache=True)
    rf.set_context(m._ctx_betas, m._ctx_K)
    for f in (rf, m._render_prev):
        f(p, *args)
    fast = _runtime_calls(lambda: rf(p, *args))
    old = _runtime_calls(lambda: m._render_prev(p, *args))
    assert fast.get("cudaStreamSynchronize", 0) == 0, fast
    assert fast.get("memcpy_DtoH", 0) == 0 and fast.get("memcpy_HtoD", 0) == 0, fast      # device-to-device copies are fine
    assert old.get("cudaStreamSynchronize", 0) > 0 and old.get("memcpy_DtoH", 0) > 0, old
    _reset(m)


# ------------------------------------------------------------------------------------ GPU: CUDA graph (keep last)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graph_forward_bitwise_gpu(gpu_model):
    m = gpu_model
    T = 10
    x = random_packets(T, 31)
    prev, betas, K = random_states(T, 32)
    sd = {k: v.cpu() for k, v in m.state_dict().items()}
    ref = run_ref(dict(kind="forward", config=str(CONFIG), device="cuda", state_dict=sd, betas=betas[:1], K=K[:1], x=x,
                       prev=prev, closed_loop=False))["preds"]
    xd, pd, bd, kd = x.to("cuda"), prev.to("cuda"), betas[:1].to("cuda"), K[:1].to("cuda")
    m.set_hand_context(bd, kd)
    gf = RF.GraphedForward(m, render=RF.RenderFast(m))
    assert not gf.ready and gf.matches(xd[0], pd[:1])
    gf.capture(xd[0], pd[:1], bd, kd)
    assert gf.ready and gf.matches(xd[1], pd[1:2]) and not gf.matches(xd[0].repeat(2, 1, 1, 1), pd[:2])
    with torch.no_grad():
        for t in range(T):
            out = gf(xd[t], pd[t:t + 1])
            assert bits_equal(out, ref[t:t + 1]), f"graph replay of packet {t}"
            assert bits_equal(out, m(xd[t], pd[t:t + 1])), f"graph vs this tree's eager forward, packet {t}"
    # a new sequence: betas / K are graph inputs, no re-capture
    ref2 = run_ref(dict(kind="forward", config=str(CONFIG), device="cuda", state_dict=sd, betas=betas[3:4], K=K[3:4], x=x,
                        prev=prev, closed_loop=False))["preds"]
    gf.set_context(betas[3:4].to("cuda"), K[3:4].to("cuda"))
    with torch.no_grad():
        for t in range(T):
            assert bits_equal(gf(xd[t], pd[t:t + 1]), ref2[t:t + 1]), f"new sequence, packet {t}"
    assert not bits_equal(ref, ref2)                                         # the context matters
    _reset(m)


#: every forward option of the DT rounds the graph / replica must cover (ROOT_HEAD adds the second root readout)
ARM_KEYS = [{}, {"CAM_PLANES": True}, {"ROOT_COMPOSE": "so3"}, {"PREV_MLP_TRANSL": False}, {"RENDER_FP32": True},
            {"CAM_PLANES": True, "ROOT_COMPOSE": "so3", "PREV_MLP_TRANSL": False},
            {"ROOT_HEAD": "spatial"}, {"ROOT_HEAD": "anchor"},
            {"ROOT_HEAD": "anchor", "CAM_PLANES": True, "ROOT_COMPOSE": "so3", "PREV_MLP_TRANSL": False},
            {"CNN_BACKBONE": "resnet18_l3"},
            {"CNN_BACKBONE": "resnet18_l3", "ROOT_HEAD": "spatial", "CAM_PLANES": True}]


def _arm_model(dt_keys, dev):
    """A model of the arm `dt_keys` (or skip if this tree has no such key) whose zero-initialised tensors -- the planes'
    conv slice, the root readout, biases -- are filled with small noise, so that every option changes the output and a
    replica that dropped one could not match bitwise."""
    cfg = load_config(CONFIG)
    for k in dt_keys:
        if k not in MNISTModel.MODEL_KEYS:
            pytest.skip(f"this tree's MNISTModel has no MODEL.{k}")
    cfg["MODEL"].update(dt_keys)
    torch.manual_seed(0)
    m = MNISTModel(cfg).eval()
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in m.parameters():
            z = p == 0
            if z.any():
                p[z] = 0.02 * torch.randn(int(z.sum()), generator=g)
    return _quiet_model(m).to(dev)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dt_keys", ARM_KEYS, ids=lambda k: "+".join(f"{a}={b}" for a, b in k.items()) or "default")
def test_graph_wraps_the_models_own_forward_gpu(dt_keys):
    """What `GraphedForward` does by default: the graph captures `model.forward` itself, its `_render_prev` dispatching to
    the fast render (`deploy_fast.enable_fast_render`). Replayed bits == this tree's eager forward (the original render,
    the original syncs), for every arm."""
    m = _arm_model(dt_keys, "cuda")
    T = 6
    x = random_packets(T, 51).to("cuda")
    prev, betas, K = random_states(T, 52)
    pd = prev.to("cuda")
    m.set_hand_context(betas[:1].to("cuda"), K[:1].to("cuda"))
    with torch.no_grad():
        eager = torch.cat([m(x[t], pd[t:t + 1]) for t in range(T)])              # the existing render, the existing syncs
        bad = RF.GraphedForward(m)                                                # not yet sync-free: refused, cleanly
        with pytest.raises(RuntimeError, match="synchroniz"):
            bad.capture(x[0], pd[:1], betas[:1].to("cuda"), K[:1].to("cuda"))
        assert not bad.ready and not bad.capturing
        DF.enable_fast_render(m)
        fast_eager = torch.cat([m(x[t], pd[t:t + 1]) for t in range(T)])
        gf = RF.GraphedForward(m)
        gf.capture(x[0], pd[:1], betas[:1].to("cuda"), K[:1].to("cuda"))
        graph = torch.cat([gf(x[t], pd[t:t + 1]) for t in range(T)])
        rep = torch.cat([RF.dense_forward(m, x[t], pd[t:t + 1], render=RF.RenderFast(m)) for t in range(T)])
        assert bits_equal(rep, eager), "dense_forward replica"
    assert bits_equal(fast_eager, eager), "eager forward with the fast render"
    assert bits_equal(graph, eager), "graph replay of the model's own forward"
    _reset(m)


@pytest.mark.parametrize("dt_keys", ARM_KEYS, ids=lambda k: "+".join(f"{a}={b}" for a, b in k.items()) or "default")
def test_dense_forward_covers_every_arm_cpu(dt_keys):
    """The replica is the live forward bit for bit for every option (it used to drop the root head and the SO(3) compose
    without a word)."""
    m = _arm_model(dt_keys, "cpu")
    x = random_packets(3, 4)
    prev, betas, K = random_states(10, 6)
    m.set_hand_context(betas[:1], K[:1])
    for xb, pb in ((x[1], prev[3:4]), (x[:3, 0], prev[3:6]), (x[0], prev[0:1])):         # x[0] is the empty packet
        if xb.shape[0] != pb.shape[0]:
            continue
        got = RF.dense_forward(m, xb, pb)
        assert bits_equal(got, m(xb, pb)), dt_keys
    rf = RF.RenderFast(m, cache=True)
    rf.set_context(m._ctx_betas, m._ctx_K)
    assert bits_equal(RF.dense_forward(m, x[1], prev[3:4], render=rf), m(x[1], prev[3:4]))
    _reset(m)


def test_dense_forward_refuses_other_arms_cpu(cpu_model):
    import types
    for attrs in ({"encoder_name": "gnn"}, {"active_head": True}, {"prev_render": False}):
        fake = types.SimpleNamespace(**{"encoder_name": "", "active_head": False, "prev_render": True, **attrs})
        with pytest.raises(NotImplementedError):
            RF.dense_forward(fake, torch.zeros(1, H, W, 2), torch.zeros(1, 51))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graph_capture_failure_is_clean_gpu(gpu_model):
    """A forward with a host sync cannot be captured. The error is raised, no half-built graph is left behind and CUDA
    keeps working (the model patch relies on this to fall back to eager)."""
    m = gpu_model
    prev, betas, K = (t.to("cuda") for t in random_states(2, 61))
    x = random_packets(2, 62)[1].to("cuda")
    bad = RF.GraphedForward(m, fn=lambda x_, p_, b_, k_: p_ + float(x_.sum().item()))      # .item(): a host sync
    with pytest.raises(RuntimeError, match="synchroniz"):                  # caught by the eager dry run, before any capture
        bad.capture(x, prev[:1], betas[:1], K[:1])
    assert not bad.ready and not bad.capturing
    m.set_hand_context(betas[:1], K[:1])
    good = RF.GraphedForward(m, render=RF.RenderFast(m))                          # the replica with the sync-free render
    good.capture(x, prev[:1], betas[:1], K[:1])
    with torch.no_grad():
        assert bits_equal(good(x, prev[:1]), m(x, prev[:1]))
    # the model's own forward is refused as long as it syncs (its `_render_prev` and FK are the originals), cleanly
    own = RF.GraphedForward(m)
    with pytest.raises(RuntimeError, match="synchroniz"):
        own.capture(x, prev[:1], betas[:1], K[:1])
    assert not own.ready and not own.capturing
    with torch.no_grad():
        assert bits_equal(good(x, prev[:1]), m(x, prev[:1]))                     # CUDA and the first graph still work
    _reset(m)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_closed_loop_bitwise_gpu(gpu_model):
    _closed_loop(gpu_model, "cuda", 20, with_graph=True)


# ------------------------------------------------------------------------------------ ManoLayer.fast_chain
def test_fast_chain_is_off_by_default_and_bitwise_cpu(cpu_model):
    m = cpu_model
    assert m.mano.fast_chain is False and m.mano._parents_py == m.mano.parents.tolist()
    keys = set(m.state_dict())
    assert not any("fast_chain" in k or "parents_py" in k for k in keys)             # checkpoints are unchanged
    a = _mano_inputs(40, 3)
    plan = [1] * 10 + [5] * 2 + [30]
    i, vs, js = 0, [], []
    for flag in (False, True):
        m.mano.fast_chain = flag
        i, d = 0, []
        for B in plan:
            v, j = m.mano(a["betas"][i:i + B], a["orient"][i:i + B], a["pose"][i:i + B], a["transl"][i:i + B])
            d += _digests(v) + _digests(j)
            i += B
        vs.append(d)
    m.mano.fast_chain = False
    assert_same(vs[1], vs[0], "fast_chain on vs off")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fast_chain_removes_the_syncs_and_is_bitwise_gpu(gpu_model):
    m = gpu_model
    a = {k: v.to("cuda") for k, v in _mano_inputs(1, 8).items()}
    args = (a["betas"], a["orient"], a["pose"], a["transl"])
    assert m.mano.fast_chain is False
    with torch.no_grad():
        ref_v, ref_j = m.mano(*args)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with pytest.raises(RuntimeError, match="synchroniz"):
                m.mano(*args)                                                     # the original chain: `.item()` per joint
        finally:
            torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()
        m.mano.fast_chain = True
        try:
            m.mano(*args)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                v, j = m.mano(*args)                                              # no sync at all
            finally:
                torch.cuda.set_sync_debug_mode(0)
            torch.cuda.synchronize()
        finally:
            m.mano.fast_chain = False
    assert bits_equal(v, ref_v) and bits_equal(j, ref_j)


# ------------------------------------------------------------------------------------ deploy_fast.enable_fast_render (opt-in)
def _enabled_forward_check(dev, graph):
    torch.manual_seed(0)
    m = _quiet_model(MNISTModel(load_config(CONFIG)).eval()).to(dev)
    assert not DF.is_fast_render(m) and m.mano.fast_chain is False                              # off by default
    T = 6
    x = random_packets(T, 41)
    prev, betas, K = random_states(T, 42)
    ref = run_ref(dict(kind="forward", config=str(CONFIG), device=dev, state_dict={k: v.cpu() for k, v in m.state_dict().items()},
                       betas=betas[:1], K=K[:1], x=x, prev=prev, closed_loop=False))["preds"]
    m.set_hand_context(betas[:1].to(dev), K[:1].to(dev))
    run = lambda: torch.cat([m(x[t].to(dev), prev[t:t + 1].to(dev)).cpu() for t in range(T)])      # noqa: E731
    with torch.no_grad():
        assert bits_equal(run(), ref)
        DF.enable_fast_render(m, cache=True)
        assert DF.is_fast_render(m) and m.mano.fast_chain is True
        gf = None
        if graph:
            gf = RF.GraphedForward(m)
            gf.capture(x[0].to(dev), prev[:1].to(dev), betas[:1].to(dev), K[:1].to(dev))
            run = lambda: torch.cat([gf(x[t].to(dev), prev[t:t + 1].to(dev)).cpu() for t in range(T)])  # noqa: E731
        assert bits_equal(run(), ref), "enabled"
        if not graph:
            assert m._render_prev.cache_hits >= T                                  # the per-sequence cache is really used
        m.set_hand_context(betas[3:4].to(dev), K[3:4].to(dev))                    # a new sequence (the hook refreshes the cache)
        if graph:
            gf.set_context(betas[3:4].to(dev), K[3:4].to(dev))
        ref2 = run_ref(dict(kind="forward", config=str(CONFIG), device=dev, state_dict={k: v.cpu() for k, v in m.state_dict().items()},
                            betas=betas[3:4], K=K[3:4], x=x, prev=prev, closed_loop=False))["preds"]
        assert bits_equal(run(), ref2), "enabled, new sequence"
        DF.disable_fast_render(m)
        assert not DF.is_fast_render(m) and m.mano.fast_chain is False and "set_hand_context" not in m.__dict__
        assert bits_equal(torch.cat([m(x[t].to(dev), prev[t:t + 1].to(dev)).cpu() for t in range(T)]), ref2), "disabled"


def test_enable_fast_render_eager_cpu():
    _enabled_forward_check("cpu", graph=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_enable_fast_render_eager_gpu():
    _enabled_forward_check("cuda", graph=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_enable_fast_render_graph_gpu():
    _enabled_forward_check("cuda", graph=True)


def test_enable_fast_render_wrappers_and_refusals_cpu(cpu_model):
    from semkine.anchored import AdaptiveFilter
    m = copy.deepcopy(cpu_model)
    flt = AdaptiveFilter(m, a_root=0.5, a_rest=0.8)
    DF.enable_fast_render(flt)                                                       # the model inside the wrapper
    assert DF.is_fast_render(m)
    DF.disable_fast_render(flt)
    assert not DF.is_fast_render(m)
    with pytest.raises(NotImplementedError):
        DF.enable_fast_render(torch.nn.Linear(2, 2))
    # a model that is deep-copied while enabled keeps a working, independent render (the hook is an object, not a closure)
    DF.enable_fast_render(m)
    prev, betas, K = random_states(4, 77)
    m.set_hand_context(betas[:1], K[:1])
    m2 = copy.deepcopy(m)
    x = random_packets(2, 5)[1]
    m2.set_hand_context(betas[1:2], K[1:2])
    ref2 = _fresh_model_output(betas[1:2], K[1:2], x, prev[:1])
    assert bits_equal(m2(x, prev[:1]), ref2)
    assert bits_equal(m(x, prev[:1]), _fresh_model_output(betas[:1], K[:1], x, prev[:1]))


def _fresh_model_output(betas, K, x, prev):
    torch.manual_seed(0)
    fresh = MNISTModel(load_config(CONFIG)).eval()
    fresh.set_hand_context(betas, K)
    return fresh(x, prev)
