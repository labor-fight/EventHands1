#!/usr/bin/env python3
"""S45-RF P2 (docs/EXPERIMENT_RECORDS.md#s45rf-p2-prereg-20261030): does the frozen correspondence predictor T0, pooled
per hand part, carry an informative and unbiased innovation about the pose error of realistic priors? Zero training;
frozen S45 (its ray association) and frozen T0; T0's held-out subject ylf decides, zgz is descriptive.

For every W50 endpoint (the first endpoint of each valid run skipped), with the prior = S45's own closed-loop state at
the previous endpoint ("loop": the S45-TR N4 bank on training subjects, S45's E1 trace on zgz) or the GT previous
state ("gt"):
  common events E* = sampled on-hand events (XAG source valid) that S45's ray association matched (q > 0, valid)
  every source gives each event in E* a point on the prior mesh and a hand part:
    ray          S45's association: sum_r A_r V[vid_r] / sum_r A_r, part = argmax sum_r A_r w_lbs[vid_r]
    part_oracle  XAG part route (XCL A1: the GT part among the association's own candidates; trained: no gain on C1)
    face_oracle  XAG source face + barycentrics (XCL A2 / S45-TR T1d; trained: gain on C1 and S45)
    t0           T0's argmax face + projected barycentrics
    noise<r>     the source face replaced by a face drawn at template geodesic distance in [r/2, 3r/2] mm (centroid)
    partswap40   with probability 0.4 the source face replaced by a uniform face of another hand part (centroid)
    shuffled     the source (face, bary) permuted among the window's E* events
  innovation nu = event ray (x, y) - projection of that point, pooled per part (>= M_MIN events)
  truth D_p = mean over part-p vertices of projection(V_window_end) - projection(V_prior)
Informativeness I: the linear share of the (recording, part)-centred D energy on the oracle-visible pairs that the
source's pooled innovation explains (pairs it does not cover explain nothing). F: its false innovation at GT priors.

  p2.py run --domain D --seq S --prior P   one GPU among 2-7
  p2.py chain --domain D                   every (seq, prior) of the domain on GPUs 2-7, then analyze
  p2.py analyze --domain D                 CPU
  p2.py freeze                             after the ledger section is written (ylf / zgz need it; dry_ch runs before)
Domains: dry_ch (training subject ch, anchors only, no T0, before freezing), ylf (decision), zgz (descriptive).
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45rf.common import (OUT, PHYSICAL_GPUS, S45_E1_TRACE, S45TR, ledger_section, now, read,  # noqa: E402
                                require, save, sha)

P2 = OUT / "p2"
PROTOCOL = P2 / "protocol.json"
TAG = "S45RF-P2-PREREG-20261030"
#: domain -> (split, sequence prefix, run T0, priors)
DOMAINS = {"dry_ch": ("train", "ch_", False, ("loop",)), "ylf": ("train", "ylf_", True, ("loop", "gt")),
           "zgz": ("val_core", "zgz_", True, ("loop", "gt"))}
NOISE_MM = (4, 8, 16, 32, 64)
PARTSWAP = 0.4
SOURCES = ("ray", "part_oracle", "face_oracle", "t0") + tuple(f"noise{r}" for r in NOISE_MM) + ("partswap40", "shuffled")
N_PARTS, M_MIN, BLOCK, NBOOT, BOOT_SEED = 16, 3, 20, 2000, 20261030
VALID_GAP, SHUFFLE_FRAC = 0.15, 0.25
HIST_MM = 200
SOURCES_FROZEN = ("tools/s45rf/__init__.py", "tools/s45rf/common.py", "tools/s45rf/p2.py", "tests/test_s45rf_p2.py",
                  "tools/s45tr/common.py", "tools/xcl/evaluate.py", "semkine/eval_track.py", "semkine/xag.py",
                  "semkine/xag_sources.py", "semkine/s45tr_t0.py", "semkine/event_gnn.py", "semkine/xyz_mesh.py",
                  "model/model.py", "model/config.py")


# --------------------------------------------------------------------------- geometry (pure, tested on CPU)
def project(P):
    return P[..., :2] / P[..., 2:3]


def assoc_point(V, vid, A):
    w = A.clamp_min(0)
    return (w.unsqueeze(-1) * V[vid]).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1e-12)


def assoc_part(lbs, vid, A):
    return (A.clamp_min(0).unsqueeze(-1) * lbs[vid]).sum(-2).argmax(-1)


def face_point(V, faces, face, bary):
    w = bary / bary.sum(-1, keepdim=True).clamp_min(1e-12)
    return (w.unsqueeze(-1) * V[faces[face]]).sum(-2)


def pool(nu, part, n_parts=N_PARTS):
    import torch
    cnt = torch.bincount(part, minlength=n_parts)
    total = torch.zeros(n_parts, 2, dtype=torch.float64, device=nu.device).index_add_(0, part, nu.double())
    return cnt, total


def part_truth(Vp, Ve, vparts, n_parts=N_PARTS):
    import torch
    d = (project(Ve.double()) - project(Vp.double()))
    cnt = torch.bincount(vparts, minlength=n_parts).clamp_min(1).double()
    return torch.zeros(n_parts, 2, dtype=torch.float64, device=d.device).index_add_(0, vparts, d) / cnt.unsqueeze(-1)


def face_geodesic(tmpl, faces):
    """(F, F) geodesic distance between face centroids on the template (metres): Dijkstra on the edge-adjacency graph."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path
    tmpl, faces = np.asarray(tmpl, np.float64), np.asarray(faces, np.int64)
    cent = tmpl[faces].mean(1)
    edge = {}
    rows, cols = [], []
    for f, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            for g in edge.get(key, ()):
                rows.append(f)
                cols.append(g)
            edge.setdefault(key, []).append(f)
    rows, cols = np.asarray(rows), np.asarray(cols)
    w = np.linalg.norm(cent[rows] - cent[cols], axis=1)
    W = sp.coo_matrix((w, (rows, cols)), shape=(len(faces), len(faces))).tocsr()
    return shortest_path(W, directed=False).astype(np.float32)


def ring_table(dist, r_m):
    """Per face, the faces at geodesic distance in [r/2, 3r/2] (else the one nearest to r): padded table + counts."""
    lists = []
    for f in range(len(dist)):
        idx = np.flatnonzero((dist[f] >= 0.5 * r_m) & (dist[f] <= 1.5 * r_m))
        if not len(idx):
            d = np.abs(dist[f] - r_m)
            d[f] = np.inf
            idx = np.array([int(np.argmin(d))])
        lists.append(idx)
    table = np.zeros((len(lists), max(len(x) for x in lists)), np.int64)
    for f, idx in enumerate(lists):
        table[f, :len(idx)] = idx
    return table, np.array([len(x) for x in lists], np.int64)


def other_part_table(fparts):
    """Per part, the faces of every other part: padded table + counts (the partswap anchor's draw)."""
    lists = [np.flatnonzero(fparts != p) for p in range(N_PARTS)]
    table = np.zeros((N_PARTS, max(len(x) for x in lists)), np.int64)
    for p, idx in enumerate(lists):
        table[p, :len(idx)] = idx
    return table, np.array([len(x) for x in lists], np.int64)


# --------------------------------------------------------------------------- informativeness (pure, tested on CPU)
def centred(values, mask, group):
    """values (W, P, 2) minus their mean over `mask` within each (group, part); zero outside the mask."""
    out = np.zeros_like(values)
    for g in np.unique(group):
        rows = group == g
        m = mask[rows]
        v = values[rows]
        mean = (v * m[..., None]).sum(0) / np.maximum(m.sum(0), 1)[:, None]
        out[rows] = (v - mean[None]) * m[..., None]
    return out


def window_terms(D, cnt, total, vis, group, m_min=M_MIN):
    """Per-window sufficient statistics of one source: <D~, nu~>, |nu~|^2 on its covered pairs, |D~|^2 on V*."""
    cover = vis & (cnt >= m_min)
    nu = np.where(cnt[..., None] > 0, total / np.maximum(cnt, 1)[..., None], 0.0)
    dc, nc, dv = centred(D, cover, group), centred(nu, cover, group), centred(D, vis, group)
    return (dc * nc).sum((1, 2)), (nc * nc).sum((1, 2)), (dv * dv).sum((1, 2)), cover


def informativeness(a, b, d):
    num, den = a.sum(), b.sum()
    return float(num * num / den / d.sum()) if den > 0 and d.sum() > 0 else 0.0, float(num / den) if den > 0 else 0.0


# --------------------------------------------------------------------------- run
def job_dir(domain, seq, prior, out_root=None):
    return Path(out_root or P2) / domain / f"{seq}__{prior}"


def protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45RF_P2_20261030" and p["status"] == "FROZEN", "S45-RF P2 protocol identity")
    for name, digest in {**p["sources_sha256"], **p["inputs_sha256"]}.items():
        require(sha(ROOT / name) == digest, "frozen S45-RF P2 file changed: " + name)
    require((P2 / "prereg_section.md").read_text() == ledger_section(TAG), "P2 pre-registration changed")
    return p


def run_job(domain, seq, prior, limit=None, device_name="cuda", out_root=None):
    """`limit` / `device_name` / `out_root`: engineering smoke only."""
    import torch
    from config import load_config
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine import s45tr_t0 as T0
    from semkine import xag as XAG
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45tr.common import (N4_BANK, S45_CHECKPOINT, S45_CHECKPOINT_SHA256, S45_CONFIG, T0_CHECKPOINT,
                                    XAG_SOURCES, verify_protocol as verify_s45tr)
    from tools.xcl import evaluate as XE
    split, prefix, use_t0, priors = DOMAINS[domain]
    require(prior in priors and seq.startswith(prefix), "unregistered P2 job")
    smoke = limit is not None or device_name != "cuda" or out_root is not None
    require(smoke or os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
    if not smoke:
        XE.configure_environment()
        XE.set_policy()
        if domain != "dry_ch":
            protocol()
    verify_s45tr()
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    device = torch.device(device_name)
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=device, strict=True).to(device).eval()
    model.xyz_keep_aux = True
    cfg_x = copy.deepcopy(cfg)
    cfg_x["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    t0 = None
    if use_t0:
        done = read(T0_CHECKPOINT.with_suffix(".json"))
        require(done["status"] == "COMPLETE" and done["checkpoint_sha256"] == sha(T0_CHECKPOINT), "T0 checkpoint changed")
        t0 = T0.load(T0_CHECKPOINT)[0].to(device)
    tmpl, faces_np = T0.template(cfg["MANO"]["NPZ"])
    faces = model.mano.f.long().to(device)
    require(torch.equal(faces.cpu(), faces_np.long()), "MANO face tables differ")
    tmpl = tmpl.to(device)
    lbs = model.xyz.lbs_w.to(device)
    vparts, fparts = lbs.argmax(-1), XAG.face_parts(lbs, faces)
    dist = face_geodesic(tmpl.cpu().numpy(), faces_np.numpy())
    rings = {r: tuple(torch.from_numpy(x).to(device) for x in ring_table(dist, r / 1000.0)) for r in NOISE_MM}
    other_table, other_count = (torch.from_numpy(x).to(device) for x in other_part_table(fparts.cpu().numpy()))
    root = Path(cfg["DATA"]["ROOT"])
    found = dict(sequences_for_split(root, split, splits_manifest(cfg)))
    require(seq in found, "unknown sequence " + seq)
    events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
    tsub_path = root / found[seq] / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
    model.set_hand_context(betas, K)
    rows = XE.endpoint_grid(aux)
    g = lambda ms: np.asarray(gt[ms], np.float32).copy()  # noqa: E731
    if prior == "loop":
        if domain == "zgz":
            with np.load(S45_E1_TRACE, allow_pickle=False) as z:
                loop = z[f"{seq}/incoming"].copy()
                require(np.array_equal(z[f"{seq}/end_ms"], [r[1] for r in rows]), "S45 trace grid differs")
        else:
            with np.load(N4_BANK / f"{seq}.npz", allow_pickle=False) as z:
                require(np.array_equal(z["end_ms"], [r[1] for r in rows]), "bank grid differs")
                loop = np.concatenate([np.zeros((1, 51), np.float32), z["state"][:-1]])   # row i <- state after row i-1
    out = job_dir(domain, seq, prior, out_root)
    require(not out.exists(), "preserve existing P2 job: " + str(out))
    out.mkdir(parents=True)
    save(out / "launch.json", {"domain": domain, "seq": seq, "prior": prior, "script_sha256": sha(__file__),
                               "t0_checkpoint_sha256": sha(T0_CHECKPOINT) if use_t0 else None,
                               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "smoke": smoke, "started_utc": now()},
         exclusive=True)
    srcs = [s for s in SOURCES if use_t0 or s != "t0"]
    rec = {k: [] for k in ("end_ms", "run_id", "D", "n_live", "n_estar")}
    rec.update({f"cnt_{s}": [] for s in srcs})
    rec.update({f"sum_{s}": [] for s in srcs})
    hist = {k: np.zeros(HIST_MM + 1, np.int64) for k in ("t0_err_mm", "ray_err_mm")}
    tally = {"t0_face_ok": 0, "t0_part_ok": 0, "estar": 0}
    band_px, front_px = model.xyz_band_px, model.xyz_front_px
    n_done = 0
    with torch.no_grad():
        for i, (rid, end, prev_ms, first) in enumerate(rows):
            if first:
                continue
            if limit is not None and n_done >= limit:
                break
            n_done += 1
            state = loop[i] if prior == "loop" else g(prev_ms)
            prev = torch.from_numpy(np.ascontiguousarray(state, np.float32)).view(1, -1).to(device)
            packet = ET.make_eval_packet(ET._window_events(events, offsets, tsub, int(end), 50), prev, betas, K, 50, device)
            packet = ET.attach_xag_sources(packet, cfg_x, seq, offsets, int(end), 50, device)
            model.xyz_last = None
            model.forward_packet(packet)
            last = model.xyz_last
            target = torch.from_numpy(g(int(end))).view(1, -1).to(device)
            betas_f, k_f = model._resolve_betas_K(prev, packet.betas, packet.camera_K)
            Vp = model._fk(prev.float(), betas_f)[0][0]
            Ve = model._fk(target.float(), betas_f)[0][0]
            rec["end_ms"].append(int(end)); rec["run_id"].append(int(rid))
            rec["D"].append(part_truth(Vp, Ve, vparts).cpu().numpy())
            empty = last is None or "assoc" not in last
            if not empty:
                require(torch.allclose(last["verts"][0], Vp, atol=1e-6), "prior mesh differs from the association's")
                src, mask, assoc, d = last["src"], last["mask"], last["assoc"], last["d"]
                s = XAG.sample_sources(packet, src, mask)
                known = s["xag_valid"] & (s["xag_face"] >= 0) & (s["xag_face"] < len(faces))
                known = known & torch.isfinite(s["xag_bary"]).all(-1) & (s["xag_bary"] >= 0).all(-1) & (s["xag_bary"].sum(-1) > 0)
                matched = mask & (assoc["q"] > 0) & assoc["valid"] & (assoc["A"].clamp_min(0).sum(-1) > 0)
                estar = (known & matched)[0]
                empty = not bool(estar.any())
            if empty:
                for k in srcs:
                    rec[f"cnt_{k}"].append(np.zeros(N_PARTS, np.int64)); rec[f"sum_{k}"].append(np.zeros((N_PARTS, 2)))
                rec["n_live"].append(0 if last is None or "mask" not in last else int(last["mask"].sum()))
                rec["n_estar"].append(0)
                continue
            V = Vp
            ray_xy = d[0][estar][:, :2].double()
            vid, A = assoc["vid"][0][estar], assoc["A"][0][estar].float()
            sface = s["xag_face"][0][estar].long()
            sbary = s["xag_bary"][0][estar].float()
            pts, parts = {}, {}
            pts["ray"], parts["ray"] = assoc_point(V, vid, A), assoc_part(lbs, vid, A)
            fx, fy, _, _ = model._intrinsics(k_f)
            f_px = (fx * fy).sqrt()
            routed, _ = XAG.route_association(
                "part_route", assoc, d, mask, last["verts"], model.mano.f, model.xyz.vert_faces, model.xyz.lbs_w,
                band_px / f_px, front_px / f_px, model.xyz_z_tol, source_part=s["xag_part"], source_face=s["xag_face"],
                source_bary=s["xag_bary"], source_valid=s["xag_valid"], k=model.xyz_k)
            pvid, pA = routed["vid"][0][estar], routed["A"][0][estar].float()
            pts["part_oracle"], parts["part_oracle"] = assoc_point(V, pvid, pA), assoc_part(lbs, pvid, pA)
            pts["face_oracle"], parts["face_oracle"] = face_point(V, faces, sface, sbary), fparts[sface]
            seed = zlib.crc32(f"{domain}|{seq}|{prior}|{int(end)}".encode())
            rng = np.random.default_rng(seed)
            centroid = torch.full((len(sface), 3), 1.0 / 3.0, device=device)
            for r in NOISE_MM:
                table, count = rings[r]
                pick = torch.from_numpy(rng.random(len(sface))).to(device)
                k = (pick * count[sface]).long().clamp_max(table.shape[1] - 1)
                nf = table[sface, k]
                pts[f"noise{r}"], parts[f"noise{r}"] = face_point(V, faces, nf, centroid), fparts[nf]
            swap = torch.from_numpy(rng.random(len(sface)) < PARTSWAP).to(device)
            other = other_table[fparts[sface], torch.from_numpy(rng.random(len(sface))).to(device).mul(
                other_count[fparts[sface]]).long().clamp_max(other_table.shape[1] - 1)]
            nf = torch.where(swap, other, sface)
            pts["partswap40"] = face_point(V, faces, nf, torch.where(swap.unsqueeze(-1), centroid, sbary))
            parts["partswap40"] = fparts[nf]
            perm = torch.from_numpy(rng.permutation(len(sface))).to(device)
            pts["shuffled"], parts["shuffled"] = face_point(V, faces, sface[perm], sbary[perm]), fparts[sface[perm]]
            spoint = face_point(tmpl, faces, sface, sbary)
            if t0 is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    o = t0(packet.events, packet.ptr, packet.delta_t_s)
                require(torch.equal(o["mask"], mask) and torch.equal(o["src"][mask], src[mask]), "T0 / S45 nodes differ")
                tface, tbary, tpoint = t0.correspondence(o)
                tf, tb = tface[0][estar].long(), tbary[0][estar].float()
                pts["t0"], parts["t0"] = face_point(V, faces, tf, tb), fparts[tf]
                err = ((tpoint[0][estar].float() - spoint).norm(dim=-1) * 1000).cpu().numpy()
                hist["t0_err_mm"] += np.bincount(np.minimum(err.astype(np.int64), HIST_MM), minlength=HIST_MM + 1)
                tally["t0_face_ok"] += int((tf == sface).sum()); tally["t0_part_ok"] += int((fparts[tf] == fparts[sface]).sum())
            rerr = ((assoc_point(tmpl, vid, A) - spoint).norm(dim=-1) * 1000).cpu().numpy()
            hist["ray_err_mm"] += np.bincount(np.minimum(rerr.astype(np.int64), HIST_MM), minlength=HIST_MM + 1)
            tally["estar"] += int(estar.sum())
            for k in srcs:
                cnt, total = pool(ray_xy - project(pts[k].double()), parts[k])
                rec[f"cnt_{k}"].append(cnt.cpu().numpy()); rec[f"sum_{k}"].append(total.cpu().numpy())
            rec["n_live"].append(int(mask.sum())); rec["n_estar"].append(int(estar.sum()))
            if n_done % 250 == 0:
                save(out / "progress.json", {"windows": n_done})
    arrays = {k: np.asarray(v) for k, v in rec.items()}
    arrays.update({f"hist_{k}": v for k, v in hist.items()})
    np.savez_compressed(out / "states.npz", **arrays)
    save(out / "report.json", {"domain": domain, "seq": seq, "prior": prior, "status": "COMPLETE", "windows": n_done,
                               "tally": tally, "finished_utc": now(), "states_sha256": sha(out / "states.npz")},
         exclusive=True)


def gpu_busy(gpu):
    used = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return int(used.splitlines()[0]) > 1000


def domain_jobs(domain):
    from config import load_config
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45tr.common import S45_CONFIG
    split, prefix, _, priors = DOMAINS[domain]
    cfg = load_config(str(S45_CONFIG))
    seqs = [s for s, _ in sequences_for_split(Path(cfg["DATA"]["ROOT"]), split, splits_manifest(cfg)) if s.startswith(prefix)]
    require(len(seqs) == (2 if domain == "zgz" else 8), "domain sequence count changed")
    return [(s, p) for p in priors for s in seqs]


def chain(domain):
    if domain != "dry_ch":
        protocol()
    busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
    require(not busy, f"GPUs {busy} are in use; P2 waits for idle cards among 2-7")
    slots = [(g, s) for g in PHYSICAL_GPUS for s in range(2)]
    pending, running, failed = domain_jobs(domain), {}, []
    log = P2 / "logs"
    log.mkdir(parents=True, exist_ok=True)
    while pending or running:
        free = [s for s in slots if s not in running.values()]
        while pending and free and not failed:
            (seq, prior), slot = pending.pop(0), free.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", "--domain", domain, "--seq", seq, "--prior", prior]
            running[subprocess.Popen(cmd, cwd=str(ROOT), env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0])),
                                     stdout=(log / f"{domain}_{seq}_{prior}.log").open("w"),
                                     stderr=subprocess.STDOUT)] = (slot, (seq, prior))
        for proc in list(running):
            if proc.poll() is not None:
                slot, job = running.pop(proc)
                if proc.returncode != 0:
                    failed.append(list(job))
        if failed and not running:
            break
        time.sleep(5)
    save(P2 / f"chain_{domain}.json", {"failed": failed, "not_started": [list(j) for j in pending], "at_utc": now()})
    require(not failed, f"P2 jobs failed (preserved, not retried): {failed}")
    require(subprocess.call([sys.executable, str(Path(__file__).resolve()), "analyze", "--domain", domain],
                            cwd=str(ROOT)) == 0, "P2 analysis failed")


# --------------------------------------------------------------------------- analysis
def load_prior(domain, prior):
    """Windows of every recording of the domain under one prior, with a (base recording, time block) id for resampling."""
    jobs = [(s, p) for s, p in domain_jobs(domain) if p == prior]
    parts, group, blocks = [], [], []
    for gi, (seq, _) in enumerate(jobs):
        with np.load(job_dir(domain, seq, prior) / "states.npz", allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
        n = len(parts[-1]["end_ms"])
        group.append(np.full(n, gi))
        base = seq.split("_v")[0]
        blocks.append(np.array([f"{base}|{e // (BLOCK * 50)}" for e in parts[-1]["end_ms"]]))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if not k.startswith("hist_")}
    hist = {k: sum(p[k] for p in parts) for k in parts[0] if k.startswith("hist_")}
    return cat, np.concatenate(group), np.concatenate(blocks), hist, [s for s, _ in jobs]


def hist_median(h):
    c = np.cumsum(h)
    return float(np.searchsorted(c, c[-1] / 2)) + 0.5 if c[-1] else None


def analyze(domain):
    if domain != "dry_ch":
        protocol()
    _, _, use_t0, priors = DOMAINS[domain]
    srcs = [s for s in SOURCES if use_t0 or s != "t0"]
    res = {"domain": domain, "computed_utc": now(), "m_min": M_MIN, "priors": {}}
    fits = {}
    for prior in priors:
        cat, group, block, hist, seqs = load_prior(domain, prior)
        D = cat["D"]
        vis = cat["cnt_face_oracle"] >= M_MIN
        terms = {s: window_terms(D, cat[f"cnt_{s}"], cat[f"sum_{s}"], vis, group) for s in srcs}
        uniq, inv = np.unique(block, return_inverse=True)
        rng = np.random.default_rng(BOOT_SEED)
        draws = rng.integers(0, len(uniq), (NBOOT, len(uniq)))
        weight = np.stack([np.bincount(dr, minlength=len(uniq)) for dr in draws]).astype(np.float64)  # multiplicities
        per_block = lambda x: np.bincount(inv, weights=x, minlength=len(uniq))  # noqa: E731
        out = {"windows": int(len(D)), "sequences": seqs, "visible_pairs": int(vis.sum()), "sources": {}}
        boot = {}
        for s in srcs:
            a, b, dd, cover = terms[s]
            I, c = informativeness(a, b, dd)
            A_, B_, D_ = per_block(a), per_block(b), per_block(dd)
            num, den, tot = weight @ A_, weight @ B_, weight @ D_
            boot[s] = np.where(den > 0, num * num / np.maximum(den, 1e-300) / tot, 0.0)
            fingers = np.zeros_like(vis)
            fingers[:, 1:] = True
            sub = {}
            for name, keep in (("palm", ~fingers), ("fingers", fingers)):
                a2, b2, d2, _ = window_terms(D, cat[f"cnt_{s}"] * keep, cat[f"sum_{s}"] * keep[..., None], vis & keep, group)
                sub[name] = informativeness(a2, b2, d2)[0]
            out["sources"][s] = {"I": I, "I_ci95": [float(x) for x in np.quantile(boot[s], [0.025, 0.975])], "gain_c": c,
                                 "coverage": float((cover & vis).sum() / max(vis.sum(), 1)), "I_palm": sub["palm"],
                                 "I_fingers": sub["fingers"]}
            fits[(prior, s)] = c
        out["t0_template_error_median_mm"] = hist_median(hist.get("hist_t0_err_mm", np.zeros(1))) if use_t0 else None
        out["ray_template_error_median_mm"] = hist_median(hist["hist_ray_err_mm"])
        if prior == "loop":
            out["_boot"] = boot
        res["priors"][prior] = out
    # F: false innovation at GT priors with the loop-fitted gain
    if "gt" in priors:
        cat_l, group_l, _, _, _ = load_prior(domain, "loop")
        vis_l = cat_l["cnt_face_oracle"] >= M_MIN
        dv = centred(cat_l["D"], vis_l, group_l)
        scale = np.sqrt((dv * dv).sum() / max(vis_l.sum(), 1))
        cat, group, _, _, _ = load_prior(domain, "gt")
        vis = cat["cnt_face_oracle"] >= M_MIN
        for s in srcs:
            cover = vis & (cat[f"cnt_{s}"] >= M_MIN)
            nu = np.where(cat[f"cnt_{s}"][..., None] > 0, cat[f"sum_{s}"] / np.maximum(cat[f"cnt_{s}"], 1)[..., None], 0.0)
            r = fits[("loop", s)] * centred(nu, cover, group) - centred(cat["D"], cover, group)
            res["priors"]["gt"]["sources"][s]["F"] = float(np.sqrt((r * r).sum() / max(cover.sum(), 1)) / scale)
    loop = res["priors"]["loop"]["sources"]
    boot = res["priors"]["loop"].pop("_boot")
    gap = loop["face_oracle"]["I"] - loop["part_oracle"]["I"]
    valid = gap >= VALID_GAP and loop["shuffled"]["I"] <= SHUFFLE_FRAC * loop["face_oracle"]["I"]
    mid = 0.5 * (loop["part_oracle"]["I"] + loop["face_oracle"]["I"])
    dose = [r for r in NOISE_MM if loop[f"noise{r}"]["I"] >= mid]
    decision = {"anchor_gap": gap, "shuffled_over_face": loop["shuffled"]["I"] / max(loop["face_oracle"]["I"], 1e-12),
                "valid": bool(valid), "midpoint": mid, "required_precision_mm": max(dose) if dose else "<4"}
    if use_t0:
        It0 = loop["t0"]["I"]
        d_part = boot["t0"] - boot["part_oracle"]
        d_mid = boot["t0"] - 0.5 * (boot["part_oracle"] + boot["face_oracle"])
        cls = "uninformative" if It0 <= loop["part_oracle"]["I"] else ("promising" if It0 >= mid else "partial")
        ci_part, ci_mid = np.quantile(d_part, [0.025, 0.975]), np.quantile(d_mid, [0.025, 0.975])
        crosses = lambda ci: bool(ci[0] <= 0 <= ci[1])  # noqa: E731
        edge = {"uninformative": crosses(ci_part), "promising": crosses(ci_mid),
                "partial": crosses(ci_part) or crosses(ci_mid)}[cls]
        decision.update(t0=cls if valid else None, boundary=edge if valid else None,
                        t0_minus_part_ci95=[float(x) for x in ci_part], t0_minus_mid_ci95=[float(x) for x in ci_mid],
                        status="READ" if valid else "STOP_PROBE_INVALID")
    else:
        decision["status"] = "DRY_RUN_ANCHORS_ONLY"
    res["decision"] = decision
    save(P2 / f"summary_{domain}.json", res)
    write_report(res)
    print(P2 / f"summary_{domain}.json")


def write_report(res):
    lines = [f"# S45-RF P2 · {res['domain']}（自动生成；冻结 S45 关联与 T0、零训练、单 seed 描述性）", "",
             f"判读：{res['decision']}", ""]
    for prior, out in res["priors"].items():
        lines += [f"## 前态 {prior}（窗口 {out['windows']}，可见部位对 {out['visible_pairs']}；T0 模板误差中位 "
                  f"{out['t0_template_error_median_mm']} mm，射线 {out['ray_template_error_median_mm']} mm）", "",
                  "| 来源 | I [95 %] | 覆盖 | I 掌 | I 指 | 增益 c | F |", "|---|---|---|---|---|---|---|"]
        for s, r in out["sources"].items():
            lines.append(f"| {s} | {r['I']:.3f} [{r['I_ci95'][0]:.3f}, {r['I_ci95'][1]:.3f}] | {r['coverage']:.2f} | "
                         f"{r['I_palm']:.3f} | {r['I_fingers']:.3f} | {r['gain_c']:+.3f} | {r.get('F', float('nan')):.3f} |")
        lines.append("")
    (P2 / f"report_{res['domain']}.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- freeze
def freeze():
    require(not PROTOCOL.exists(), "P2 protocol already frozen")
    dry = P2 / "summary_dry_ch.json"
    require(dry.is_file(), "run the dry_ch anchors before freezing")
    section = P2 / "prereg_section.md"
    with section.open("x") as stream:
        stream.write(ledger_section(TAG))
    from tools.s45tr.common import N4_BANK, S45_CHECKPOINT_SHA256, T0_CHECKPOINT
    from tools.s45tr.common import XAG_SOURCES
    inputs = [S45_E1_TRACE, dry, T0_CHECKPOINT, XAG_SOURCES / "build_summary.json"] + sorted(N4_BANK.glob("ylf_*.npz"))
    save(PROTOCOL, {
        "study": "S45RF_P2_20261030", "status": "FROZEN", "frozen_utc": now(),
        "prereg": "docs/EXPERIMENT_RECORDS.md#s45rf-p2-prereg-20261030", "prereg_section_sha256": sha(section),
        "domains": {k: list(v) for k, v in DOMAINS.items()}, "sources": SOURCES, "noise_mm": NOISE_MM, "partswap": PARTSWAP,
        "constants": {"m_min": M_MIN, "block_endpoints": BLOCK, "nboot": NBOOT, "boot_seed": BOOT_SEED, "n_parts": N_PARTS},
        "thresholds": {"valid_gap": VALID_GAP, "shuffle_frac": SHUFFLE_FRAC},
        "s45_checkpoint_sha256": S45_CHECKPOINT_SHA256,
        "inputs_sha256": {str(Path(p).relative_to(ROOT)): sha(p) for p in inputs},
        "sources_sha256": {name: sha(ROOT / name) for name in SOURCES_FROZEN},
        "physical_gpus": PHYSICAL_GPUS, "no_training": True, "single_seed_descriptive": True,
    }, exclusive=True)
    print(PROTOCOL)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=("run", "chain", "analyze", "freeze"))
    p.add_argument("--domain", choices=tuple(DOMAINS))
    p.add_argument("--seq")
    p.add_argument("--prior", choices=("loop", "gt"))
    a = p.parse_args(argv)
    if a.action == "run":
        require(a.domain and a.seq and a.prior, "--domain, --seq and --prior required")
        run_job(a.domain, a.seq, a.prior)
    elif a.action == "freeze":
        freeze()
    else:
        require(a.domain is not None, "--domain required")
        (chain if a.action == "chain" else analyze)(a.domain)


if __name__ == "__main__":
    main()
