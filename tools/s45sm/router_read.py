"""Router readings on the hand surface (GPU, one process per router on one physical GPU among 2-7; read once per
router, descriptive, never gating or tuning anything).

  gate  (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031, the round-1 reading; kind sr only) on the held-out subject ylf:
        packets and event set as S45-RT's T0P gate (8192 samples drawn once, rng 3407, gate set = on-hand events that the
        frozen S45's own association matched); for the SR router being read and for T0P, with S45-RT's definitions:
        coverage (on-hand >= 0.5), part-cross share, face geodesic error median, template-point error median, each for the
        routed events and for every gate event; the measure-level occupancy overlap over 16 parts / T0P's 279 patches; the
        anchors sigma16 and pi (S45-CT's corruptions of the XAG routes). -> routers/<name>/eval.json
  d1    (#s45sm-d1-20261031) how well an events-only router labels the hand surface on evaluation packets of six recordings
        (lyq / ylf / zgz, local and global): the evaluator's 50 ms endpoint grid over every valid run, the router sees the
        events only (B = 1), labels are the XAG per-event sources of the same raw rows (reading only). Per sequence, over the
        router's stride-sampled nodes whose XAG source is valid: coverage, on-hand AUC, part-cross share, face geodesic
        median, template-point median / p75, the per-packet occupancy overlap over 16 parts / 279 patches.
        -> readouts_d1/<router>.json
  ctx   (#s45sm-r6c-prereg-20261031) D1 with the router looking at the last ctx ms of events (semkine/s45sm_ctx.py;
        ctx 50 is D1 itself; 100 / 150 / 200 / 300 add the events before the packet). Checks written into the result: `d1`
        (the ctx-50 metrics equal the router's D1 file) and `ctx_equal` (on every packet the context forward with the
        context equal to the packet gives D1's routes). -> readouts_ctx/<router>.json
"""
from __future__ import annotations

import numpy as np

from .paths import HELDOUT, OUT, PACKAGE, ROOT, S45_CONFIG, now, read, require, require_one_gpu, save, sha
from .protocol import verify_protocol
from .registry import router_checkpoint, router_dir, router_ready, router_spec
from .routing import NEEDS, register_kinds

N_SAMPLES, BATCH, TAU = 8192, 128, 0.5
SEQUENCES = ("lyq_local", "lyq_global", "ylf_local", "ylf_global", "zgz_local", "zgz_global")
STEP_MS = 50
CONTEXTS = (50, 100, 150, 200, 300)
COUNTS = ("packets", "nodes", "valid_nodes")
DISTANCES = ("face_geodesic_median_mm", "template_median_mm", "template_p75_mm")
TOL_SHARE, TOL_MM = 1e-4, 0.01


def auc(score, label):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos = label.astype(bool)
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def gate_metrics(sel, face, point, gt_face, gt_point, face_part, geo):
    if not sel.any():
        return {"events": 0}
    return {"events": int(sel.sum()),
            "part_cross_share": float((face_part[face[sel]] != face_part[gt_face[sel]]).mean()),
            "face_geodesic_median_mm": float(np.median(geo[face[sel], gt_face[sel]]) * 1000),
            "template_median_mm": float(np.median(np.linalg.norm(point[sel] - gt_point[sel], axis=-1)) * 1000),
            "template_p75_mm": float(np.percentile(np.linalg.norm(point[sel] - gt_point[sel], axis=-1), 75) * 1000)}


def overlap(packet, gate, face, gt_face, table, n_bins):
    """Measure-level fidelity: sum over packets of the histogram intersection / gate events."""
    sel = gate
    key_r = packet[sel] * n_bins + table[face[sel]]
    key_g = packet[sel] * n_bins + table[gt_face[sel]]
    size = int(packet.max() + 1) * n_bins
    hr = np.bincount(key_r, minlength=size)
    hg = np.bincount(key_g, minlength=size)
    return float(np.minimum(hr, hg).sum() / max(hg.sum(), 1))


def d1_metrics(c, n_packets, face_part, patch, geo):
    """The per-sequence D1 metrics from the concatenated arrays of one sequence (keys: valid, on, face, point, gt_face,
    gt_point, packet)."""
    g = c["valid"].astype(bool)
    err = np.linalg.norm(c["point"][g] - c["gt_point"][g], axis=-1) * 1000
    return {"packets": int(n_packets), "nodes": int(len(g)), "valid_nodes": int(g.sum()),
            "coverage_onhand_0p5": float((c["on"][g] >= 0.5).mean()), "onhand_auc": auc(c["on"], g),
            "part_cross_share": float((face_part[c["face"][g]] != face_part[c["gt_face"][g]]).mean()),
            "face_geodesic_median_mm": float(np.median(geo[c["face"][g], c["gt_face"][g]]) * 1000),
            "template_median_mm": float(np.median(err)), "template_p75_mm": float(np.percentile(err, 75)),
            "occupancy_overlap_part": overlap(c["packet"], g, c["face"], c["gt_face"], face_part, 16),
            "occupancy_overlap_patch": overlap(c["packet"], g, c["face"], c["gt_face"], patch, int(patch.max() + 1))}


def d1_check(got, d1):
    bad = []
    for seq, m in got.items():
        ref = d1["sequences"][seq]
        for k, v in m.items():
            tol = 0 if k in COUNTS else (TOL_MM if k in DISTANCES else TOL_SHARE)
            if abs(float(v) - float(ref[k])) > tol:
                bad.append({"sequence": seq, "metric": k, "probe": v, "d1": ref[k]})
    return bad


def _register(router):
    register_kinds(NEEDS[router_spec(router)["kind"]])


# --------------------------------------------------------------------------- gate (round 1 / 2 / 4 reading, kind sr)
def gate(router):
    import torch
    from tools.s45tr.common import S45_CHECKPOINT, S45_CHECKPOINT_SHA256, XAG_SOURCES
    verify_protocol()
    require_one_gpu()
    spec = router_spec(router)
    require(spec["kind"] == "sr" and spec["heldout"] == HELDOUT, "SR routers held out on ylf only")
    ckpt = router_ready(router)
    target = router_dir(router) / "eval.json"
    require(not target.exists(), "the router reading is read once: " + str(target))
    require(sha(S45_CHECKPOINT) == S45_CHECKPOINT_SHA256, "S45 checkpoint changed")
    from config import load_config
    from model import MNISTModel
    from semkine import s45ct_route as CT
    from semkine import s45rt_router as RT
    from semkine import s45sm as SM
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    from semkine import xag as XAG
    dev = torch.device("cuda")
    cfg = load_config(str(S45_CONFIG))
    model = MNISTModel.load_from_checkpoint(str(S45_CHECKPOINT), cfg=cfg, map_location=dev, strict=True).to(dev).eval()
    model.xyz_keep_aux = True
    sr, meta = SM.load(ckpt)
    sr = sr.to(dev)
    require(meta["heldout"] == HELDOUT, "SR held-out subject differs")
    t0p_path = router_checkpoint("t0p")
    router_ready("t0p")
    t0p, _ = RT.load(t0p_path)
    t0p = t0p.to(dev)
    require(torch.equal(t0p.face_part.cpu(), sr.face_part.cpu()), "T0P and SR part tables differ")
    face_part = sr.face_part.cpu().numpy()
    patch = t0p.patch_of_face.cpu().numpy()
    geo = CT.face_geodesic(sr.tmpl.cpu().numpy(), sr.faces.cpu().numpy())
    cent = sr.centroid.cpu().numpy()
    tmpl, faces = sr.tmpl.cpu().numpy(), sr.faces.cpu().numpy()
    npz = cfg["MANO"]["NPZ"]
    dcfg = load_config(str(S45_CONFIG))
    dcfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    components = np.load(dcfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(dcfg, "train", components, train=True)
    held = np.array([s.startswith(HELDOUT + "_") for s, _ in ds.sequences])
    rows = np.flatnonzero(held[ds.index[:, 0]])
    require(held.sum() == 8 and rows.size > N_SAMPLES, "held-out selection changed")
    pick = np.sort(np.random.default_rng(3407).choice(rows, N_SAMPLES, replace=False))
    keys = ("packet", "gate", "valid", "gt_face", "gt_point", "s_on", "s_face", "s_point", "t_on", "t_face", "t_point",
            "sig_face", "pi_face", "pi_point")
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for start in range(0, N_SAMPLES, BATCH):
            batch = collate_packets([ds[int(i)] for i in pick[start:start + BATCH]]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.forward_packet(batch)
                os_ = sr(batch.events, batch.ptr, batch.delta_t_s)
                ot = t0p(batch.events, batch.ptr, batch.delta_t_s)
            last = model.xyz_last
            src, mask, assoc = last["src"], last["mask"], last["assoc"]
            for o in (os_, ot):
                require(torch.equal(o["mask"], mask) and torch.equal(o["src"][mask], src[mask]), "node mismatch")
            s = XAG.sample_sources(batch, src, mask)
            tgt = SM.targets(sr, batch, src, mask)
            w = (assoc["A"].float() * (assoc["q"].float().unsqueeze(-1) > 0)).clamp_min(0)
            matched = mask & (assoc["q"] > 0) & assoc["valid"] & (w.sum(-1) > 0)
            gate_ = tgt["valid"] & matched
            fs, bs = sr.route(os_, mask)
            ft, bt, _ = t0p.route(ot, mask)
            sig_f, _ = CT.corrupt("sigma16", tgt["face"], s["xag_bary"], gate_, s["xag_source_index"], npz)
            pi_f, pi_b = CT.corrupt("shuffle", tgt["face"], s["xag_bary"], gate_, s["xag_source_index"], npz)
            pt = lambda f, b: (b.float().unsqueeze(-1) * sr.tmpl[sr.faces[f]]).sum(-2)  # noqa: E731
            pk = torch.arange(mask.shape[0], device=dev).unsqueeze(1).expand_as(mask) + start
            for k, v in (("packet", pk), ("gate", gate_), ("valid", tgt["valid"]), ("gt_face", tgt["face"]),
                         ("gt_point", tgt["point"]), ("s_on", torch.sigmoid(os_["onhand"].float())), ("s_face", fs),
                         ("s_point", pt(fs, bs)), ("t_on", torch.sigmoid(ot["onhand"].float())), ("t_face", ft),
                         ("t_point", pt(ft, bt)), ("sig_face", sig_f), ("pi_face", pi_f), ("pi_point", pt(pi_f, pi_b))):
                acc[k].append(v[mask].float().cpu() if v.dtype.is_floating_point else v[mask].cpu())
    c = {k: torch.cat(v).numpy() for k, v in acc.items()}
    g = c["gate"].astype(bool)
    packet = c["packet"].astype(np.int64)
    sig_point = cent[c["sig_face"]]

    def reading(face, point, on=None):
        r = {"every_gate_event": gate_metrics(g, face, point, c["gt_face"], c["gt_point"], face_part, geo),
             "occupancy_overlap_part": overlap(packet, g, face, c["gt_face"], face_part, 16),
             "occupancy_overlap_patch": overlap(packet, g, face, c["gt_face"], patch, int(patch.max() + 1))}
        if on is not None:
            routed = g & (on >= TAU)
            r["coverage_onhand_0p5"] = float(routed.sum() / max(g.sum(), 1))
            r["routed_onhand_0p5"] = gate_metrics(routed, face, point, c["gt_face"], c["gt_point"], face_part, geo)
            r["onhand_auc"] = auc(on, c["valid"].astype(bool))
        return r
    result = {
        "status": "DESCRIPTIVE", "router": router, "read_at_utc": now(), "heldout": HELDOUT, "samples": N_SAMPLES,
        "router_checkpoint_sha256": sha(ckpt), "router_params": SM.n_params(sr),
        "t0p_checkpoint_sha256": sha(t0p_path), "t0p_params": SM.n_params(t0p),
        "s45_checkpoint_sha256": S45_CHECKPOINT_SHA256, "live_events": int(len(g)), "gate_events": int(g.sum()),
        "sr": reading(c["s_face"], c["s_point"], c["s_on"]),
        "t0p": reading(c["t_face"], c["t_point"], c["t_on"]),
        "anchors": {"sigma16": reading(c["sig_face"], sig_point), "pi": reading(c["pi_face"], c["pi_point"]),
                    "trained_retention_R": {"sigma16": 0.99, "pi": 0.88, "t0_t1b": "0.18-0.23"}},
    }
    save(target, result, exclusive=True)
    print({k: result[k] for k in ("sr", "t0p")}, flush=True)
    return target


# --------------------------------------------------------------------------- D1 / ctx
def _sequence_setup(torch, dev, router):
    """Everything D1 and the context probe share: the router net, T0P's patch table, geodesics and the sequence index."""
    from config import load_config
    from semkine import s45ct_route as CT
    from semkine import s45rt_router as RT
    from semkine import s45sm as SM
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.s45tr.common import XAG_SOURCES
    kind = router_spec(router)["kind"]
    ckpt = router_ready(router)
    net = SM._load_router(kind, ckpt).to(dev)
    t0p, _ = RT.load(router_ready("t0p"))
    patch = t0p.patch_of_face.numpy()
    face_part = net.face_part.cpu().numpy()
    require(np.array_equal(face_part, t0p.face_part.numpy()), "part tables differ")
    geo = CT.face_geodesic(net.tmpl.cpu().numpy(), net.faces.cpu().numpy())
    cfg = load_config(str(S45_CONFIG))
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCES)
    from pathlib import Path
    data_root = Path(cfg["DATA"]["ROOT"])
    where = {s: d for split in ("train", "val_core") for s, d in sequences_for_split(data_root, split, splits_manifest(cfg))}
    return kind, ckpt, net, patch, face_part, geo, cfg, data_root, where


def d1(router):
    import torch
    verify_protocol()
    _register(router)
    require_one_gpu()
    target = OUT / "readouts_d1" / f"{router}.json"
    require(not target.exists(), "D1 is read once per router: " + str(target))
    from semkine import eval_track as ET
    from semkine import s45sm as SM
    from semkine import xag as XAG
    from semkine.xyz_mesh import stride_sample
    dev = torch.device("cuda")
    kind, ckpt, net, patch, face_part, geo, cfg, data_root, where = _sequence_setup(torch, dev, router)
    result = {"status": "DESCRIPTIVE", "router": router, "kind": kind, "router_checkpoint_sha256": sha(ckpt),
              "router_params": SM.n_params(net), "read_at_utc": now(), "sequences": {}}
    for seq in SEQUENCES:
        require(seq in where, "sequence not in the split manifest: " + seq)
        events, offsets, aux, pos51 = ET.load_sequence(data_root, where[seq], seq)
        tsub_p = data_root / where[seq] / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        acc = {k: [] for k in ("packet", "valid", "on", "face", "point", "gt_face", "gt_point")}
        n_packets = 0
        with torch.no_grad():
            for run_a, run_b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
                prev = torch.from_numpy(pos51[run_a].copy()).view(1, -1).to(dev)
                for end in np.arange(run_a + STEP_MS - 1, run_b, STEP_MS, dtype=np.int64):
                    ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
                    if len(ev5) == 0:
                        continue
                    batch = ET.attach_xag_sources(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, dev), cfg, seq,
                                                  offsets, int(end), STEP_MS, dev)
                    src, mask = stride_sample(batch.ptr, SM.MAX_NODES)
                    s = XAG.sample_sources(batch, src, mask)
                    out, routed, face, bary, on = SM.route_nodes(net, kind, batch.events, batch.ptr, batch.delta_t_s, 0.5)
                    every = mask.clone()
                    face, bary = (net.route(out, every)[:2])
                    valid = s["xag_valid"] & mask & (s["xag_face"] >= 0)
                    gf = s["xag_face"].clamp_min(0)
                    w = s["xag_bary"].clamp_min(0.0)
                    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
                    gp = (w.unsqueeze(-1) * net.tmpl[net.faces[gf]]).sum(-2)
                    pt = (bary.float().unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2)
                    for k, v in (("valid", valid), ("on", on), ("face", face), ("point", pt), ("gt_face", gf),
                                 ("gt_point", gp)):
                        acc[k].append(v[mask].float().cpu() if v.dtype.is_floating_point else v[mask].cpu())
                    acc["packet"].append(torch.full((int(mask.sum()),), n_packets, dtype=torch.long))
                    n_packets += 1
        c = {k: torch.cat(v).numpy() for k, v in acc.items()}
        result["sequences"][seq] = d1_metrics(c, n_packets, face_part, patch, geo)
        print(seq, result["sequences"][seq], flush=True)
    save(target, result, exclusive=True)
    return target


def ctx_probe(router):
    import torch
    verify_protocol()
    _register(router)
    require_one_gpu()
    target = OUT / "readouts_ctx" / f"{router}.json"
    require(not target.exists(), "the context probe is read once per router: " + str(target))
    d1_file = read(OUT / "readouts_d1" / f"{router}.json")
    from semkine import eval_track as ET
    from semkine import s45sm as SM
    from semkine import s45sm_ctx as CTX
    from semkine import xag as XAG
    from semkine.xyz_mesh import stride_sample
    dev = torch.device("cuda")
    kind = router_spec(router)["kind"]
    ckpt = router_ready(router)
    require(d1_file["router_checkpoint_sha256"] == sha(ckpt), "D1 was read on another checkpoint")
    kind, ckpt, net, patch, face_part, geo, cfg, data_root, where = _sequence_setup(torch, dev, router)
    result = {"status": "DESCRIPTIVE", "router": router, "kind": kind, "router_checkpoint_sha256": sha(ckpt),
              "router_params": SM.n_params(net), "contexts_ms": list(CONTEXTS), "script_sha256": sha(__file__),
              "ctx_module_sha256": sha(ROOT / "semkine/s45sm_ctx.py"), "read_at_utc": now(),
              "by_context": {str(c): {} for c in CONTEXTS}, "context_ms_median": {}, "ctx_equal_mismatch_packets": 0}
    mismatch = 0
    for seq in SEQUENCES:
        require(seq in where, "sequence not in the split manifest: " + seq)
        events, offsets, aux, pos51 = ET.load_sequence(data_root, where[seq], seq)
        tsub_p = data_root / where[seq] / f"{seq}_tsub.npy"
        tsub = np.load(tsub_p, mmap_mode="r") if tsub_p.exists() else None
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=dev).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=dev).view(1, 3, 3)
        acc = {c: {k: [] for k in ("packet", "valid", "on", "face", "point", "gt_face", "gt_point")} for c in CONTEXTS}
        widths = {c: [] for c in CONTEXTS}
        n_packets = 0
        with torch.no_grad():
            for run_a, run_b in np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2):
                prev = torch.from_numpy(pos51[run_a].copy()).view(1, -1).to(dev)
                for end in np.arange(run_a + STEP_MS - 1, run_b, STEP_MS, dtype=np.int64):
                    ev5 = ET._window_events(events, offsets, tsub, int(end), STEP_MS)
                    if len(ev5) == 0:
                        continue
                    batch = ET.attach_xag_sources(ET.make_eval_packet(ev5, prev, betas, K, STEP_MS, dev), cfg, seq,
                                                  offsets, int(end), STEP_MS, dev)
                    src, mask = stride_sample(batch.ptr, SM.MAX_NODES)
                    s = XAG.sample_sources(batch, src, mask)
                    valid = s["xag_valid"] & mask & (s["xag_face"] >= 0)
                    gf = s["xag_face"].clamp_min(0)
                    w = s["xag_bary"].clamp_min(0.0)
                    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
                    gp = (w.unsqueeze(-1) * net.tmpl[net.faces[gf]]).sum(-2)
                    every = mask.clone()
                    out, _, _, _, on = SM.route_nodes(net, kind, batch.events, batch.ptr, batch.delta_t_s, 0.5)
                    face, bary = net.route(out, every)[:2]
                    same = CTX.router_forward_ctx(net, batch.ptr, {"events": batch.events, "ptr": batch.ptr,
                                                                    "delta_t_s": batch.delta_t_s}, src, mask)
                    f2, b2 = net.route(same, every)[:2]
                    on2 = torch.sigmoid(same["onhand"].float())
                    if not (torch.equal(face[mask], f2[mask]) and torch.equal(bary[mask], b2[mask])
                            and torch.equal(on[mask], on2[mask])):
                        mismatch += 1
                    routes = {}
                    for c in CONTEXTS:
                        wc = min(c, int(end) - int(run_a) + 1)
                        widths[c].append(wc)
                        if wc == STEP_MS:
                            routes[c] = (face, bary, on)
                            continue
                        if wc in routes:
                            routes[c] = routes[wc]
                            continue
                        ctx5 = ET._window_events(events, offsets, tsub, int(end), wc)
                        CTX.check_tail(ev5, ctx5, wc)
                        ctx = {"events": torch.from_numpy(np.ascontiguousarray(ctx5)).to(dev),
                               "ptr": torch.tensor([0, len(ctx5)], dtype=torch.int64, device=dev),
                               "delta_t_s": torch.tensor([wc * 1e-3], dtype=torch.float32, device=dev)}
                        o = CTX.router_forward_ctx(net, batch.ptr, ctx, src, mask)
                        fc, bc = net.route(o, every)[:2]
                        routes[c] = routes[wc] = (fc, bc, torch.sigmoid(o["onhand"].float()))
                    for c in CONTEXTS:
                        fc, bc, oc = routes[c]
                        pt = (bc.float().unsqueeze(-1) * net.tmpl[net.faces[fc]]).sum(-2)
                        for k, v in (("valid", valid), ("on", oc), ("face", fc), ("point", pt), ("gt_face", gf),
                                     ("gt_point", gp)):
                            acc[c][k].append(v[mask].float().cpu() if v.dtype.is_floating_point else v[mask].cpu())
                        acc[c]["packet"].append(torch.full((int(mask.sum()),), n_packets, dtype=torch.long))
                    n_packets += 1
        for c in CONTEXTS:
            arrays = {k: torch.cat(v).numpy() for k, v in acc[c].items()}
            result["by_context"][str(c)][seq] = d1_metrics(arrays, n_packets, face_part, patch, geo)
            result["context_ms_median"].setdefault(str(c), {})[seq] = float(np.median(widths[c]))
        print(seq, {c: (round(result["by_context"][str(c)][seq]["occupancy_overlap_patch"], 4),
                        round(result["by_context"][str(c)][seq]["part_cross_share"], 4)) for c in CONTEXTS}, flush=True)
    bad = d1_check(result["by_context"]["50"], d1_file)
    result["ctx_equal_mismatch_packets"] = mismatch
    result["checks"] = {"d1": not bad, "d1_mismatches": bad, "ctx_equal": mismatch == 0}
    save(target, result, exclusive=True)
    print("CHECKS", result["checks"], flush=True)
    return target
