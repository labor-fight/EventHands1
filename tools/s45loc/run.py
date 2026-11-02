#!/usr/bin/env python3
"""S45-LOC: one process = one registered task (domain x E1/E3 x intervention) on one physical GPU among 4-7.

The packet / state / FK-scoring contract is the XCL evaluator's (tools/xcl/evaluate.py: endpoint grid, W50 packets,
`make_eval_packet`, `score`, deterministic policy), so the two zgz baselines must reproduce the closed C1SF S45 arrays
bit for bit before any other task is read (`b0_*` check in tools/s45loc/chain.py). Every task also records, per
endpoint, how the frozen model's association treated the events against the XAG GT pseudo-sources.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.frozen import accepts  # noqa: E402
from tools.s45loc import common as C  # noqa: E402
from tools.xcl import evaluate as XE  # noqa: E402

STAT_KEYS = ("n_raw", "n_nodes", "n_sv", "n_sv_hit", "n_sv_near", "n_sv_none", "n_sv_agree", "n_sv_hit_agree",
             "n_sv_near_agree", "n_sv_hit_face", "sv_point_err_mm_sum", "n_off", "n_off_assoc", "q_off", "q_all",
             "zn_qmean", "zn_max", "n_zn_over", "tok_dtie_mean", "tok_sae_same_mean", "tok_sae_opp_mean",
             "n_sae_same_never", "resid_mass", "n_direct", "d_trans_mm", "d_root_deg", "d_finger_rad",
             "n_drop") + tuple(f"n_sv_assoc_q{i}" for i in range(4)) + tuple(f"n_sv_agree_q{i}" for i in range(4))
#: zn above the deepest training-local recording's p90 root depth (lyh_local, 0.665 m): (0.665 - 0.45) / 0.1
ZN_OVER = 2.15


def protocol():
    p = C.read(C.PROTOCOL)
    C.require(p.get("status") == ("SMOKE" if C.SMOKE else "FROZEN"), "S45-LOC protocol status does not match its directory")
    # a registered amendment supersedes the hashes of the files it changed (the frozen protocol itself is never edited)
    for amendment in sorted(C.OUT.glob("amendment*.json")):
        a = C.read(amendment)
        C.require(a.get("status") == "FROZEN", "unfrozen amendment: " + amendment.name)
        p["sources_sha256"].update(a["sources_sha256"])
    for name, digest in p["sources_sha256"].items():
        C.require(accepts(C.ROOT / name, digest), f"frozen source changed: {name}")
    return p


def load_model(device):
    import torch
    from config import load_config
    from model import MNISTModel
    C.require(C.sha(C.CHECKPOINT) == C.CHECKPOINT_SHA256, "S45 checkpoint changed")
    C.require(accepts(C.CONFIG, C.CONFIG_SHA256), "S45 config changed")
    cfg = load_config(str(C.CONFIG))
    model = MNISTModel.load_from_checkpoint(str(C.CHECKPOINT), cfg=cfg, map_location=device, strict=True)
    return model.to(device).eval(), cfg


class Hooks:
    """Relation-input pre-hooks (zn shift / token quantile map) and the two C1SF frozen interventions."""

    def __init__(self, model):
        self.model, self.handles, self.zn_shift, self.tokenmap = model, [], None, None
        self.saved_transform = getattr(model.xyz, "xag_h3_transform", None)

    def install(self, intervention, tokenmap=None):
        import torch
        xyz = self.model.xyz
        if intervention == "geometry_zero":
            def zero(module, inputs):
                value = inputs[0].clone()
                value[..., 7:11] = 0
                return (value, *inputs[1:])
            self.handles.append(xyz.rel[0].register_forward_pre_hook(zero))
        elif intervention == "h3_zero":
            xyz.xag_h3_transform = lambda h, s, o: h * 0
        elif intervention in ("znshift_ref", "znshift_zgz", "zgzlike", "tokenmap"):
            def edit(module, inputs):
                value = inputs[0].clone()
                if self.zn_shift is not None:
                    live = value[:, C.VALID_COL] > 0.5
                    value[live, C.ZN_COL] = value[live, C.ZN_COL] + float(self.zn_shift)
                if self.tokenmap is not None:
                    for col, (src_q, dst_q) in self.tokenmap.items():
                        x = value[:, col].double()
                        s = torch.as_tensor(src_q, dtype=torch.float64, device=x.device)
                        d = torch.as_tensor(dst_q, dtype=torch.float64, device=x.device)
                        value[:, col] = interp(x, s, d).to(value.dtype)
                return (value, *inputs[1:])
            self.handles.append(xyz.rel[0].register_forward_pre_hook(edit))
            self.tokenmap = tokenmap
        return self

    def remove(self):
        for h in self.handles:
            h.remove()
        self.model.xyz.xag_h3_transform = self.saved_transform


def interp(x, xp, fp):
    """Monotone piecewise-linear map (torch, 1-D), constant beyond the ends. Ties in xp keep their mid value."""
    import torch
    idx = torch.searchsorted(xp, x).clamp(1, len(xp) - 1)
    x0, x1, y0, y1 = xp[idx - 1], xp[idx], fp[idx - 1], fp[idx]
    w = torch.where(x1 > x0, (x - x0) / (x1 - x0).clamp_min(1e-300), torch.full_like(x, 0.5))
    out = y0 + w.clamp(0, 1) * (y1 - y0)
    return torch.where(x <= xp[0], fp[0], torch.where(x >= xp[-1], fp[-1], out))


def prepass_parts(model, prev_t, packet, vertex_part):
    """The model's own association rule on every raw event of the packet (not just the stride sample), cast on the
    incoming state's mesh; returns (assoc valid, assigned part) per raw event."""
    import torch
    from semkine.events import EV_X, EV_Y
    from semkine.xyz_mesh import associate_rays
    ev = packet.events
    n = int(ev.shape[0])
    if n == 0:
        return np.zeros(0, bool), np.zeros(0, np.int64)
    with torch.no_grad():
        betas_f, k_f = model._resolve_betas_K(prev_t, packet.betas, packet.camera_K)
        verts, _ = model._fk(prev_t.float(), betas_f)
        fx, fy, cx, cy = model._intrinsics(k_f)
        px, py = ev[:, EV_X].float().view(1, -1), ev[:, EV_Y].float().view(1, -1)
        d = torch.stack([(px - cx[:, None]) / fx[:, None], (py - cy[:, None]) / fy[:, None],
                         torch.ones_like(px)], dim=-1)
        f_px = (fx * fy).sqrt()
        mask = torch.ones(1, n, dtype=torch.bool, device=ev.device)
        assoc = associate_rays(d, mask, verts, model.mano.f, model.xyz.vert_faces,
                               band=model.xyz_band_px / f_px, front_tol=model.xyz_front_px / f_px,
                               z_tol=model.xyz_z_tol, k=model.xyz_k)
    valid = assoc["valid"][0].cpu().numpy()
    part = C.weighted_part(assoc["vid"][0].cpu().numpy(), (assoc["A"][0] * assoc["q"][0, :, None]).cpu().numpy(),
                           vertex_part)
    return valid, np.where(valid, part, -1)


def source_parts(source, faces, vertex_part):
    face = source["xag_face"]
    ok = source["xag_valid"] & (face >= 0)
    part = np.full(len(face), -1, np.int64)
    if ok.any():
        part[ok] = C.weighted_part(faces[face[ok]], source["xag_bary"][ok], vertex_part)
    return ok, part


def packet_stats(model, packet, source, faces, vertex_part, collect_tokens):
    """Per-endpoint association / record statistics of the packet the model actually read (its stride sample)."""
    out = dict.fromkeys(STAT_KEYS, 0.0)
    out["n_raw"] = float(packet.events.shape[0])
    last = getattr(model, "xyz_last", None)
    if last is None or out["n_raw"] == 0:
        return out, None
    mask = last["mask"][0].cpu().numpy()
    src = last["src"][0].cpu().numpy()[mask]
    assoc = {k: v[0].cpu().numpy() for k, v in last["assoc"].items()}
    tok = last["tok"][0].cpu().numpy()[mask]
    verts = last["verts"][0].double().cpu().numpy()
    valid, hit, q = assoc["valid"][mask], assoc["hit"][mask], assoc["q"][mask]
    vid, A, z = assoc["vid"][mask], assoc["A"][mask], assoc["z"][mask]
    sv, spart = source_parts({k: v[src] for k, v in source.items()}, faces, vertex_part)
    apart = np.where(valid, C.weighted_part(vid, A * q[:, None], vertex_part), -1)
    agree = valid & sv & (apart == spart)
    out["n_nodes"] = float(mask.sum())
    out["n_sv"] = float(sv.sum())
    out["n_sv_hit"] = float((sv & hit).sum())
    out["n_sv_near"] = float((sv & valid & ~hit).sum())
    out["n_sv_none"] = float((sv & ~valid).sum())
    out["n_sv_agree"] = float(agree.sum())
    out["n_sv_hit_agree"] = float((agree & hit).sum())
    out["n_sv_near_agree"] = float((agree & ~hit).sum())
    sface = source["xag_face"][src]
    fv = faces[np.clip(sface, 0, None)]
    same_face = np.sort(vid[:, :3], axis=1) == np.sort(fv, axis=1)
    out["n_sv_hit_face"] = float((sv & hit & same_face.all(1)).sum())
    both = sv & valid
    if both.any():
        p_assoc = (A[both, :, None] * verts[vid[both]]).sum(1) / A[both].sum(1, keepdims=True).clip(1e-12)
        bary = source["xag_bary"][src][both].astype(np.float64)
        p_src = (bary[:, :, None] * verts[fv[both]]).sum(1) / bary.sum(1, keepdims=True).clip(1e-12)
        out["sv_point_err_mm_sum"] = float(np.linalg.norm(p_assoc - p_src, axis=1).sum() * 1000)
    tq = np.clip((tok[:, 3] * 4).astype(np.int64), 0, 3)
    for i in range(4):
        out[f"n_sv_assoc_q{i}"] = float((both & (tq == i)).sum())
        out[f"n_sv_agree_q{i}"] = float((agree & (tq == i)).sum())
    off = ~sv
    out["n_off"] = float(off.sum())
    out["n_off_assoc"] = float((off & valid).sum())
    out["q_off"] = float(q[off].sum())
    out["q_all"] = float(q.sum())
    if valid.any():
        zn = (z[valid] - 0.45) / 0.1
        out["zn_qmean"] = float((zn * q[valid]).sum() / max(q[valid].sum(), 1e-12))
        out["zn_max"] = float(zn.max())
        out["n_zn_over"] = float((zn > ZN_OVER).sum())
    out["tok_dtie_mean"] = float(tok[:, 4].mean())
    out["tok_sae_same_mean"] = float(tok[:, 5].mean())
    out["tok_sae_opp_mean"] = float(tok[:, 6].mean())
    out["n_sae_same_never"] = float((tok[:, 5] >= 1.0).sum())
    out["resid_mass"] = float(last["r"][0].cpu().numpy()[mask].sum())
    out["n_direct"] = float(last["O"][0].sum().item())
    delta = last["delta"][0].double().cpu().numpy()
    out["d_trans_mm"] = float(np.linalg.norm(delta[0:3]) * 1000)
    out["d_root_deg"] = float(np.rad2deg(np.linalg.norm(delta[3:6])))
    out["d_finger_rad"] = float(np.linalg.norm(delta[6:51]))
    return out, (tok[:, list(C.TOKEN_TIMING_COLS)].astype(np.float32) if collect_tokens else None)


def run_task(name, gpu_note, limit=None):
    import torch
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import mano_root_joint, sequences_for_split, splits_manifest
    from semkine.xag_sources import load_sources
    domain, mode, intervention = C.TASKS[name]
    proto = protocol()
    consts = proto["constants"]
    device = torch.device("cuda")
    model, cfg = load_model(device)
    model.xyz_keep_aux = True
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    faces = model.mano.f.long().cpu().numpy()
    vertex_part = model.xyz.lbs_hard.cpu().numpy()
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    out_dir = C.OUT / "tasks" / name
    C.require(not out_dir.exists(), "preserve existing task output: " + str(out_dir))
    out_dir.mkdir(parents=True)
    identity = {"task": name, "domain": domain, "mode": mode, "intervention": intervention,
                "protocol_sha256": C.sha(C.PROTOCOL), "checkpoint_sha256": C.sha(C.CHECKPOINT),
                "script_sha256": C.sha(__file__), "physical_gpu": gpu_note, "started_utc": C.now()}
    C.save(out_dir / "launch.json", identity, exclusive=True)
    hooks = Hooks(model)
    tokenmap = None
    if intervention == "tokenmap":
        tm = C.read(C.OUT / "tokenmap.json")
        tokenmap = {int(col): (np.asarray(v["src"]), np.asarray(v["dst"])) for col, v in tm["columns"].items()}
    hooks.install(intervention, tokenmap)
    arrays, per_seq = {}, []
    shared_rng = np.random.default_rng(0) if domain == "zgz" else None
    try:
        with torch.no_grad():
            for seq in C.domain_sequences(domain):
                active = intervention != "none" and (domain != "zgz" or seq == "zgz_local")
                events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
                tsub_path = root / found[seq] / f"{seq}_tsub.npy"
                tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
                betas_np = np.asarray(aux["betas"], np.float32)
                j0_z = float(mano_root_joint(betas_np, cfg["MANO"]["NPZ"])[2])
                k = float(consts["k_ref"][seq]) if active and intervention == "transplant_ref" else 1.0
                hooks.zn_shift = None
                if active and intervention == "znshift_ref":
                    hooks.zn_shift = float(consts["zn_shift_ref"][seq])
                elif active and intervention in ("znshift_zgz", "zgzlike"):
                    hooks.zn_shift = float(consts["zn_shift_zgz"][seq])
                if intervention == "tokenmap":
                    hooks.tokenmap = tokenmap if active else None
                p_keep = float(consts["p_sub"][seq]) if active and intervention in ("subsample_zgz", "zgzlike") else 1.0
                sigma = None
                if active and intervention == "sigma_ref":
                    sigma = float(consts["sigma_factor_ref"][seq])
                elif active and intervention == "sigma_zgz":
                    sigma = float(consts["sigma_factor_zgz"][seq])
                j0 = mano_root_joint(betas_np, cfg["MANO"]["NPZ"])
                rs = float(cfg["MODEL"]["RENDER_SCALE"])

                def state(ms):
                    s = np.asarray(gt[ms], np.float32).copy()
                    return C.transplant_state(s, k, j0_z) if k != 1.0 else s
                betas = torch.tensor(betas_np, dtype=torch.float32, device=device).view(1, -1)
                K_np = np.asarray(aux["camera_K"], np.float32)
                K_used = C.transplant_K(K_np, k) if k != 1.0 else K_np
                camera_K = torch.tensor(K_used, dtype=torch.float32, device=device).view(1, 3, 3)
                model.set_hand_context(betas, camera_K)
                rng = shared_rng if shared_rng is not None else np.random.default_rng(0)
                rows = XE.endpoint_grid(aux)
                runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
                rec = {k2: [] for k2 in ("incoming_used", "prediction", "target", "gt_previous", "end_ms", "run_id",
                                         "delta_param")}
                stats = {k2: [] for k2 in STAT_KEYS}
                tokens = []
                i = 0
                for rid, (a, b) in enumerate(runs):
                    ends = np.arange(a + 49, b, 50, dtype=np.int64)
                    if not len(ends):
                        continue
                    prev = state(int(a)) + ET.sample_init_noise(cfg, rng, 1.0)
                    prev_t = torch.from_numpy(prev).view(1, -1).to(device)
                    prev_ms = int(a)
                    for end in ends:
                        if limit is not None and i >= limit:
                            break
                        C.require(rows[i][:3] == (rid, int(end), prev_ms), "endpoint grid mismatch")
                        i += 1
                        truth, current = state(prev_ms), state(int(end))
                        delta_param = np.zeros(45, np.float64)
                        if mode == "e3":
                            used = truth.copy()
                            if active and intervention.startswith("perturb_"):
                                kind = intervention.split("_", 1)[1]
                                used, dp = C.perturb_state(truth, kind, C.rng_for(seq, int(end), 7))
                                delta_param[:len(dp)] = dp
                            used_t = torch.from_numpy(used).view(1, -1).to(device)
                        elif mode == "e1":
                            used_t = prev_t
                            used = prev_t.detach().cpu().numpy()[0].copy()
                        else:
                            # XCL E4 clamps (tools/xcl/evaluate.py select_previous): GT pose or GT translation
                            used = prev_t.detach().cpu().numpy()[0].copy()
                            if mode == "e1_gtpose":
                                used[3:] = truth[3:]
                            else:
                                used[:3] = truth[:3]
                            used_t = torch.from_numpy(used).view(1, -1).to(device)
                        ev5 = ET._window_events(events, offsets, tsub, int(end), 50)
                        start = int(end) - 49
                        row_ids = np.arange(int(offsets[start]), int(offsets[int(end) + 1]), dtype=np.int64)
                        event_ms = np.repeat(np.arange(start, int(end) + 1, dtype=np.int64),
                                             np.diff(offsets[start:int(end) + 2]).astype(np.int64))
                        C.require(len(row_ids) == len(ev5), "source rows disagree with packet events")
                        source = load_sources(C.XAG_SOURCES, seq, row_ids, event_ms)
                        if sigma is not None:
                            # the wrist (root joint) pixel of the GT previous state, render frame
                            p0 = j0 + truth[0:3].astype(np.float64)
                            center = (rs * (K_used[0, 0] * p0[0] / p0[2] + K_used[0, 2]),
                                      rs * (K_used[1, 1] * p0[1] / p0[2] + K_used[1, 2]))
                            ev5, inside = C.rescale_events(ev5, center, sigma)
                            source = {k2: v[inside] for k2, v in source.items()}
                        keep = np.ones(len(ev5), bool)
                        if p_keep < 1.0:
                            keep = C.keep_fraction_mask(len(ev5), p_keep, C.rng_for(seq, int(end), 1))
                        elif active and intervention.startswith("drop_"):
                            probe = ET.make_eval_packet(ev5, used_t, betas, camera_K, 50, device)
                            sv, spart = source_parts(source, faces, vertex_part)
                            drop = ~sv
                            if intervention.startswith("drop_misassoc"):
                                avalid, apart = prepass_parts(model, used_t, probe, vertex_part)
                                drop = drop | (sv & avalid & (apart != spart))
                            if intervention.endswith("_rand"):
                                keep = C.random_drop_mask(len(ev5), int(drop.sum()), C.rng_for(seq, int(end), 2))
                            else:
                                keep = ~drop
                        if not keep.all():
                            ev5 = ev5[keep]
                            source = {k2: v[keep] for k2, v in source.items()}
                        packet = ET.make_eval_packet(ev5, used_t, betas, camera_K, 50, device)
                        packet.target = torch.from_numpy(current).view(1, -1).to(device)
                        model.xyz_last = None
                        pred = model.forward_packet(packet)
                        value = pred.detach().cpu().numpy()[0].copy()
                        if value.shape != (51,) or value.dtype != np.float32 or not np.isfinite(value).all():
                            raise FloatingPointError(f"nonfinite endpoint retained as failure: {seq}:{end}")
                        st, tk = packet_stats(model, packet, source, faces, vertex_part,
                                              collect_tokens=(intervention == "none" and mode == "e3"))
                        st["n_drop"] = float((~keep).sum())
                        for k2 in STAT_KEYS:
                            stats[k2].append(st[k2])
                        if tk is not None:
                            tokens.append(tk)
                        for k2, v in (("incoming_used", used), ("prediction", value), ("target", current),
                                      ("gt_previous", truth), ("end_ms", int(end)), ("run_id", rid),
                                      ("delta_param", delta_param)):
                            rec[k2].append(v)
                        prev_t = pred
                        prev_ms = int(end)
                C.require(i == len(rows) or limit is not None, "no endpoints may be dropped: " + seq)
                beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
                pred_a, tgt_a = np.stack(rec["prediction"]), np.stack(rec["target"])
                metrics = XE.score(mano, pred_a, tgt_a, beta, device)
                ident = XE.score(mano, np.stack(rec["gt_previous"]), tgt_a, beta, device)
                incoming = XE.score(mano, np.stack(rec["incoming_used"]), np.stack(rec["gt_previous"]), beta, device)
                for k2, v in rec.items():
                    arrays[f"{seq}/{k2}"] = np.asarray(np.stack(v) if np.ndim(v[0]) else v)
                for k2, v in metrics.items():
                    arrays[f"{seq}/{k2}"] = v
                for k2, v in ident.items():
                    arrays[f"{seq}/identity_{k2}"] = v
                for k2, v in incoming.items():
                    arrays[f"{seq}/incoming_{k2}"] = v
                for k2, v in stats.items():
                    arrays[f"{seq}/stat_{k2}"] = np.asarray(v, np.float64)
                if tokens:
                    t = np.concatenate(tokens)
                    pick = np.random.default_rng([C.SEED, 99]).choice(len(t), size=min(len(t), C.TOKEN_SAMPLE),
                                                                      replace=False)
                    arrays[f"{seq}/token_sample"] = t[np.sort(pick)]
                for k2, v in arrays.items():
                    if k2.startswith(seq + "/") and v.dtype.kind == "f" and not np.isfinite(v).all():
                        raise FloatingPointError("nonfinite array cannot be omitted: " + k2)
                per_seq.append({"seq": seq, "n": len(pred_a), "k": k, "p_keep": p_keep,
                                **{k2: float(v.mean()) for k2, v in metrics.items()},
                                **{"identity_" + k2: float(v.mean()) for k2, v in ident.items()}})
                C.save(out_dir / "progress.json", {"done": [r["seq"] for r in per_seq]})
    finally:
        hooks.remove()
    np.savez_compressed(out_dir / "states.npz", **arrays)
    report = {**identity, "status": "COMPLETE", "finished_utc": C.now(), "per_sequence": per_seq,
              "states_sha256": C.sha(out_dir / "states.npz"), "no_training_or_selection": True}
    C.save(out_dir / "report.json", report, exclusive=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--task", required=True, choices=sorted(C.TASKS))
    p.add_argument("--limit", type=int, help="engineering smoke only (S45LOC_OUT): first N endpoints per sequence")
    args = p.parse_args(argv)
    C.require(args.limit is None or C.SMOKE, "--limit is for smoke directories only")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    C.require(gpu in tuple(map(str, C.PHYSICAL_GPUS)), "S45-LOC runs on exactly one physical GPU among 4-7")
    XE.configure_environment()
    XE.set_policy()
    try:
        run_task(args.task, gpu, args.limit)
    except BaseException as exc:
        out_dir = C.OUT / "tasks" / args.task
        if out_dir.exists() and not (out_dir / "report.json").exists():
            C.save(out_dir / "failure.json", {"task": args.task, "status": "FAILED_PRESERVED",
                                              "error": f"{type(exc).__name__}: {exc}", "at_utc": C.now()})
        raise
    print("COMPLETE", args.task, flush=True)


if __name__ == "__main__":
    main()
