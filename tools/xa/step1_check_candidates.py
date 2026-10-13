#!/usr/bin/env python3
"""[XYZ-ASSOC] P1 side check: in R3 (the event's own ms mesh), for events labelled interior (src_kind 0) whose
`associate_rays` hit depth differs from the rendered visible depth by >1 mm, is the rendered front-most face among
the candidate faces (faces incident to the k=16 angularly nearest vertices)?  Random windows, all events."""
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_source_labels as L                                    # noqa: E402
import step1_assoc_regimes as A                                    # noqa: E402
import step1_common as C                                           # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
from semkine.xyz_mesh import associate_rays                        # noqa: E402

torch.set_num_threads(1)
n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
res = {}
for seq, legacy in C.sequences():
    L._init_worker(seq, legacy)
    g = L._G
    ev, off, tsub, aux, pos51, betas, K = C.load_seq_bundle(seq, legacy)
    rid, ends, prev = C.enumerate_steps(aux)
    fx, fy, cx, cy = C.render_intrinsics_f32(K)
    f_px = (fx * fy).sqrt()
    band, front = (C.BAND_PX / f_px).contiguous(), (C.FRONT_PX / f_px).contiguous()
    vf = A.vertex_face_table(g["mano"].f, 778)
    rng = np.random.default_rng(1)
    pick = rng.choice(len(ends), min(n_steps, len(ends)), replace=False)
    tot = dict(n_interior=0, n_hit=0, n_zdiff_gt1mm=0, front_face_in_cand=0, hit_behind=0, hit_in_front=0)
    for s in pick:
        end = int(ends[s]); start = end - 49
        ev5 = C._window_events(g["events"], g["offsets"], g["tsub"], end, 50)
        counts = np.diff(g["offsets"][start:end + 2]).astype(np.int64)
        cs = np.concatenate([[0], np.cumsum(counts)])
        x, y = ev5[:, 1].astype(np.int64), ev5[:, 2].astype(np.int64)
        verts, _ = C.fk_batch(g["mano"], g["betas"], g["pos51"][start:end + 1])
        for i in range(0, 50, 5):                     # every 5th ms keeps it cheap
            if not counts[i]:
                continue
            Z, P, FID = L.rasterize(verts[i].numpy().astype(np.float64), g["faces"], g["face_part"], g["fxyc"])
            lp, lk, ld, _ = L.label_maps(Z, P)
            sl = slice(cs[i], cs[i + 1])
            kd = lk[y[sl], x[sl]]
            inter = np.where(kd == 0)[0]
            if not len(inter):
                continue
            ex, ey = x[sl][inter], y[sl][inter]
            d = torch.stack([(torch.from_numpy(ex).float() - cx) / fx, (torch.from_numpy(ey).float() - cy) / fy, torch.ones(len(ex))], -1).unsqueeze(0)
            out = associate_rays(d, torch.ones(1, len(ex), dtype=torch.bool), verts[i:i + 1], g["mano"].f, vf, band=band, front_tol=front, z_tol=C.Z_TOL, k=C.KNN)
            hit, z = out["hit"][0].numpy(), out["z"][0].numpy()
            zt = Z[ey, ex]
            tot["n_interior"] += len(ex); tot["n_hit"] += int(hit.sum())
            bad = hit & (np.abs(z - zt) > 1e-3)
            tot["n_zdiff_gt1mm"] += int(bad.sum())
            tot["hit_behind"] += int((hit & (z - zt > 1e-3)).sum()); tot["hit_in_front"] += int((hit & (zt - z > 1e-3)).sum())
            if bad.any():
                # candidate faces of the bad events
                v = verts[i].float()
                zv = v[:, 2]
                vu, vv = v[:, 0] / zv, v[:, 1] / zv
                bi = np.where(bad)[0]
                d2 = (vu[None] - d[0, bi, 0:1]).square() + (vv[None] - d[0, bi, 1:2]).square()
                nid = d2.topk(C.KNN, dim=-1, largest=False).indices
                cand = vf[nid].reshape(len(bi), -1).numpy()
                ff = FID[ey[bi], ex[bi]]
                tot["front_face_in_cand"] += int(np.any(cand == ff[:, None], axis=1).sum())
    res[seq] = tot
    print(seq, tot, flush=True)
(C.OUT_DIR / "check_candidates.json").write_text(json.dumps(res, indent=1))
