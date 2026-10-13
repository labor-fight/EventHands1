#!/usr/bin/env python3
"""S43 deviation 9: the static-shape sparse trunk (`sparse_graph`) against the masked dense trunk on every zgz W200 step
(equivalence), and the batch-1 latency of: eager SparseExecutor, eager StaticSparseTrunk per bucket, its CUDA graph per
bucket, the masked dense trunk eager and as a CUDA graph. Run on an idle GPU (4-7).

    python tools/s43/bench_static.py --run-dir outputs/semkine/s43n_sp_s3407 --out outputs/s43/reports/static_trunk.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]
from semkine import eval_track as ET                             # noqa: E402
from semkine.dataset import sequences_for_split                  # noqa: E402
from sparse_graph import GraphedSparseTrunk                      # noqa: E402
from sparse_trunk import SparseExecutor, input_mask              # noqa: E402
from render_fast import capture_checked                          # noqa: E402
import filter_eval as FE                                         # noqa: E402


def timeit(fn, items, passes=3, warm=10):
    for it in items[:warm]:
        fn(it)
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(passes):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / len(items) * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-fp32", action="store_true")
    a = ap.parse_args()
    if a.strict_fp32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    cfg, model, ckpt, step, _ = FE.build_run(Path(a.run_dir), "last", dev)
    root = Path(cfg["DATA"]["ROOT"])
    ex = SparseExecutor(model.conv1, model.rn)
    g = GraphedSparseTrunk(model.conv1, model.rn)
    st = g.st
    items = []
    for s, d in sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"])):
        events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        for ra, rb in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
            for e in np.arange(ra + 49, rb, 50):
                w = min(200, int(e - ra + 1))
                l = torch.from_numpy(ET.build_lnes(events, offsets, int(e), w)).to(dev)
                prev = torch.from_numpy(pos51[max(int(e) - 50, int(ra))].copy()).view(1, -1).to(dev)
                with torch.no_grad():
                    r = model._render_prev(prev.float(), betas, K)[0]
                items.append((s, torch.cat([l, r], -1)))

    def dense(img):
        x = img.permute(2, 0, 1)[None]
        m0 = input_mask(x)
        return model.rn(model.conv1(x) * m0, m0)[0]

    # ---- equivalence on every step, bucket chosen from the counts
    diffs, diffs_ex, buckets, overflow = [], [], [], 0
    with torch.no_grad():
        for s, img in items:
            ref = dense(img)
            cnt = st.counts(img).tolist()
            b = st.bucket_for(cnt)
            if b is None:
                overflow += 1
                continue
            out = st.run(img, b)
            diffs.append(float((out - ref).abs().max()))
            buckets.append(b)
            g.img_buf.copy_(img)
            og, _, _ = g(cnt)
            diffs_ex.append(float((og - ref).abs().max()))
    print(f"static eager vs masked dense: max |d| {max(diffs):.2e}; graph replay vs masked dense: max |d| {max(diffs_ex):.2e}; "
          f"overflow {overflow}/{len(items)}; bucket histogram {np.bincount(buckets, minlength=len(st.buckets)).tolist()}")

    # ---- latency
    gl = [im for s, im in items if s == "zgz_global"][::12][:100]
    lo = [im for s, im in items if s == "zgz_local"][::12][:100]
    res = {"run": Path(a.run_dir).name, "gpu": torch.cuda.get_device_name(dev), "strict_fp32": a.strict_fp32,
           "max_abs_diff_static_vs_dense": max(diffs), "max_abs_diff_graph_vs_dense": max(diffs_ex),
           "bucket_histogram": np.bincount(buckets, minlength=len(st.buckets)).tolist(), "overflow": overflow, "ms": {}}
    with torch.no_grad():
        res["ms"]["sparse_executor_eager"] = {"global": timeit(lambda im: ex.run(im, count_macs=False), gl),
                                              "local": timeit(lambda im: ex.run(im, count_macs=False), lo)}
        res["ms"]["masked_dense_eager"] = {"global": timeit(dense, gl), "local": timeit(dense, lo)}
        for b in range(len(st.buckets)):
            res["ms"][f"static_eager_bucket{b}"] = {"global": timeit(lambda im, b=b: st.run(im, b), gl),
                                                    "local": timeit(lambda im, b=b: st.run(im, b), lo)}
            gr, out, _ = g.graph(b)

            def rep(im, gr=gr):
                g.img_buf.copy_(im)
                gr.replay()
                return out
            res["ms"][f"static_graph_bucket{b}"] = {"global": timeit(rep, gl), "local": timeit(rep, lo)}
        # bucket chosen per step from the counts (one host sync) + replay
        def auto(im):
            g.img_buf.copy_(im)
            cnt = st.counts(g.img_buf).tolist()
            return g(cnt)[0]
        res["ms"]["static_graph_auto_bucket"] = {"global": timeit(auto, gl), "local": timeit(auto, lo)}
        # the masked dense trunk as a CUDA graph (same static buffer)
        xb = torch.zeros_like(g.img_buf)
        gd, outd = capture_checked(lambda: dense(xb))

        def repd(im):
            xb.copy_(im)
            gd.replay()
            return outd
        res["ms"]["masked_dense_graph"] = {"global": timeit(repd, gl), "local": timeit(repd, lo)}
    for k, v in res["ms"].items():
        print(f"{k:28s} global {v['global']:.3f} ms   local {v['local']:.3f} ms")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
