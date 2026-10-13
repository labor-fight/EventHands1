import numpy as np, json
from pathlib import Path
root = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
out = {}
for d, s in [("val","zgz_global"),("val","zgz_local"),("train","lyq_global"),("train","lyq_local"),("train","ch_global_v2"),("train","ylf_local_v3")]:
    ev = np.load(root/d/f"{s}_events.npy", mmap_mode="r"); off = np.load(root/d/f"{s}_offsets.npy")
    aux = np.load(root/d/f"{s}_aux.npz", allow_pickle=True)
    runs = np.asarray(aux["valid_runs_ms"]).reshape(-1,2)
    n_ev, n_pix, n_c2, n_c4, n_c8 = [], [], [], [], []
    for a,b in runs:
        for end in range(a+49, b, 50)[::4]:
            a0, a1 = off[end-49], off[end+1]
            e = np.asarray(ev[a0:a1]); n_ev.append(len(e))
            if len(e)==0: n_pix.append(0); n_c2.append(0); n_c4.append(0); n_c8.append(0); continue
            x = e[:,0].astype(np.int64); y = e[:,1].astype(np.int64)
            n_pix.append(len(np.unique(y*240+x)))
            n_c2.append(len(np.unique((y//2)*120+x//2)))
            n_c4.append(len(np.unique((y//4)*60+x//4)))
            n_c8.append(len(np.unique((y//8)*30+x//8)))
    f = lambda v: {p: int(np.percentile(v,p)) for p in (5,50,95,99)}
    out[s] = {"events": f(n_ev), "pixels": f(n_pix), "cells2": f(n_c2), "cells4": f(n_c4), "cells8": f(n_c8)}
print(json.dumps(out, indent=0))
