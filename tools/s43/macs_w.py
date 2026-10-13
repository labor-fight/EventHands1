"""Median active-site MACs of the S43 executor on zgz steps for an evidence window W (open loop, prev = GT of the step
before; window clipped to the segment start as in the S39 stack). Writes {"macs": {...}} for tools/s43/make_row.py."""
import json, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/data1/lyq/code/mesh/EventHands1")
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "tracking"), str(REPO / "tools" / "dt")]
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split
from sparse_trunk import SparseExecutor
import filter_eval as FE
run, W, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = torch.device("cuda")
cfg, model, ckpt, step, _ = FE.build_run(REPO / "outputs/semkine" / run, "last", dev)
ex = SparseExecutor(model.conv1, model.rn)
root = Path(cfg["DATA"]["ROOT"])
macs, sizes = [], []
for s, d in sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"])):
    events, offsets, aux, pos51 = ET.load_sequence(root, d, s)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
    for a, b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
        for e in np.arange(a + 49, b, 50):
            w = min(W, int(e - a + 1))
            l = torch.from_numpy(ET.build_lnes(events, offsets, int(e), w)).to(dev)
            prev = torch.from_numpy(pos51[max(int(e) - 50, int(a))].copy()).view(1, -1).to(dev)
            with torch.no_grad():
                r = model._render_prev(prev.float(), betas, K)[0]
                _, m, sz = ex.run(torch.cat([l, r], -1))
            macs.append(m); sizes.append(sz["A0"])
res = {"run": run, "window_ms": W, "n": len(macs), "macs": {"median": float(np.median(macs)), "mean": float(np.mean(macs)),
       "p90": float(np.percentile(macs, 90))}, "A0_median": float(np.median(sizes))}
Path(out).write_text(json.dumps(res, indent=1))
print(res)
