#!/usr/bin/env python3
"""T1A-UB zero-training probes: information retention along the network (round-1 supplement R1P,
docs/EXPERIMENT_RECORDS.md#t1aub-r1p-prereg-20261102). Never trains or changes a network; the only fits are probes.

  probe.py slots  --net N   one GPU among 2-7: the network's own closed loop on the nine training local recordings and on
                            zgz_local, replayed exactly as its tl_e1 / E1 readings (checked bit for bit by `fit`), with the
                            readout input zs captured at every endpoint.
  probe.py events --net N   one GPU among 2-7: the event encoder's per-node features (token7 for networks without the event
                            graph, the event-graph node features otherwise) and, for the same nodes, the XAG label surface
                            location (16-way part, rest-template point in cm, valid flag); every 10th endpoint of the 18 base
                            training recordings, ylf and zgz, at most 256 sampled nodes per packet (seeded by endpoint).
                            `--net router:<name>` stores an S45-SM router's own predictions (part, template point) for the
                            same nodes as a reference.
  probe.py fit    --net N   CPU: (a) S45-DEEP's f4 ridge probe (tools/s45loc/deep.py probe, unchanged) of the true finger
                            pose (top-10 training-pose components) from [zs + previous state] vs [previous state] vs the
                            network's own output; (b) event probes: linear and two-layer MLP (256) classifiers of the part
                            and regressors of the template point, fit on the eight subjects other than ylf, tested on ylf /
                            zgz_local / zgz_global; writes probes/<N>/summary.json.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.t1aub import common as C  # noqa: E402

OUT = C.OUT / "probes"
SUBJECTS = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")
TL = tuple(f"{s}_local" for s in SUBJECTS)
EVENT_SEQS = tuple(f"{s}_{k}" for s in SUBJECTS for k in ("local", "global")) + ("zgz_local", "zgz_global")
EVENT_STRIDE, NODES_PER_PACKET, MLP_HIDDEN, MLP_STEPS = 10, 256, 256, 3000


def found_sequences(cfg):
    from semkine.dataset import sequences_for_split, splits_manifest
    root = Path(cfg["DATA"]["ROOT"])
    found = {}
    for split in ("val_core", "train"):
        found.update(dict(sequences_for_split(root, split, splits_manifest(cfg))))
    return root, found


def slots(name):
    import torch
    from tools.xcl import evaluate as XE
    from semkine import eval_track as ET
    from tools.t1aub.evaluate import load
    C.require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()
    out = OUT / name / "slots"
    C.require(not out.exists(), "preserve existing probe output: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    model, cfg, checkpoint, state = load(name, device)
    model.xyz_keep_aux = True
    C.save(out / "launch.json", {"net": name, "checkpoint_sha256": C.sha(checkpoint), "script_sha256": C.sha(__file__),
                                 "installed": state, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": C.now()},
           exclusive=True)
    root, found = found_sequences(cfg)
    arrays = {}
    with torch.no_grad():
        for seq in TL + ("zgz_local",):
            events, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
            tsub_path = root / found[seq] / f"{seq}_tsub.npy"
            tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
            betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
            K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
            model.set_hand_context(betas, K)
            g = lambda ms: np.asarray(gt[ms], np.float32).copy()  # noqa: E731
            # the readings' own loop: a fresh rng(0) per training recording; zgz_local after zgz_global in E1's shared rng
            rng = np.random.default_rng(0)
            if seq == "zgz_local":
                _, _, aux_g, _ = ET.load_sequence(root, found["zgz_global"], "zgz_global")
                for a, b in np.asarray(aux_g["valid_runs_ms"], np.int64).reshape(-1, 2):
                    if len(np.arange(a + 49, b, 50)):
                        ET.sample_init_noise(cfg, rng, 1.0)
            rec = {}
            for rid, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)):
                ends = np.arange(a + 49, b, 50, dtype=np.int64)
                if not len(ends):
                    continue
                prev = g(int(a)) + ET.sample_init_noise(cfg, rng, 1.0)
                for end in ends:
                    model.xyz_last = None
                    packet = ET.make_eval_packet(ET._window_events(events, offsets, tsub, int(end), 50),
                                                 torch.from_numpy(np.ascontiguousarray(prev, np.float32)).view(1, -1).to(device),
                                                 betas, K, 50, device)
                    packet = ET.attach_xag_sources(packet, cfg, seq, offsets, int(end), 50, device)
                    pred = model.forward_packet(packet).cpu().numpy()[0].copy()
                    last = getattr(model, "xyz_last", None)
                    has = last is not None and "zs" in last
                    zdim = model.xyz.slot_proj.out_channels
                    zs = last["zs"][0].float().cpu().numpy() if has else np.zeros(zdim, np.float32)
                    for k, v in (("zs", zs), ("incoming_used", prev), ("prediction", pred), ("target", g(int(end))),
                                 ("end_ms", int(end)), ("run_id", rid), ("has_events", has)):
                        rec.setdefault(k, []).append(np.asarray(v))
                    prev = pred
            for k, v in rec.items():
                arrays[f"{seq}/{k}"] = np.stack(v)
            C.save(out / "progress.json", {"done": seq})
    np.savez_compressed(out / "states.npz", **arrays)
    C.save(out / "report.json", {"net": name, "status": "COMPLETE", "finished_utc": C.now(),
                                 "states_sha256": C.sha(out / "states.npz")}, exclusive=True)
    print("COMPLETE", out, flush=True)


def events(name):
    import torch
    from tools.xcl import evaluate as XE
    from semkine import eval_track as ET
    from semkine import s45tr_t1a as T1A
    from semkine.encoder import event_tokens
    from semkine.events import EV_X, EV_Y  # noqa: F401
    from semkine.xag_sources import load_sources
    from semkine.xyz_mesh import stride_sample
    C.require_one_gpu()
    XE.configure_environment()
    XE.set_policy()
    C.verify_protocol()
    out = OUT / name.replace(":", "_") / "events"
    C.require(not out.exists(), "preserve existing probe output: " + str(out))
    out.mkdir(parents=True)
    device = torch.device("cuda")
    router = name.split(":", 1)[1] if name.startswith("router:") else None
    if router:
        info = C.router_info(router)
        from tools.s45sm.routing import NEEDS, register_kinds
        from semkine import s45sm as SM
        register_kinds(NEEDS[info["kind"]])
        net = SM._load_router(info["kind"], info["checkpoint"]).to(device)
        cfg = C.load_config_for("t1a")
        model, checkpoint = None, info["checkpoint"]
    else:
        from tools.t1aub.evaluate import load
        model, cfg, checkpoint, _ = load(name, device)
    with np.load(cfg["MANO"]["NPZ"], allow_pickle=False) as z:
        tmpl, faces = np.asarray(z["v_template"], np.float64), np.asarray(z["f"], np.int64)
    from semkine.s45sm import face_part_table
    with np.load(cfg["MANO"]["NPZ"], allow_pickle=False) as z:
        fpart = face_part_table(np.asarray(z["weights"]), faces)
    C.save(out / "launch.json", {"net": name, "checkpoint_sha256": C.sha(checkpoint), "script_sha256": C.sha(__file__),
                                 "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "started_utc": C.now()}, exclusive=True)
    root, found = found_sequences(cfg)
    src_root = str(C.XAG_SOURCES)
    arrays = {}
    with torch.no_grad():
        for seq in EVENT_SEQS:
            evs, offsets, aux, gt = ET.load_sequence(root, found[seq], seq)
            tsub_path = root / found[seq] / f"{seq}_tsub.npy"
            tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
            rows = XE.endpoint_grid(aux)
            feats, labels = [], []
            for i in range(0, len(rows), EVENT_STRIDE):
                end = int(rows[i][1])
                ev5 = ET._window_events(evs, offsets, tsub, end, 50)
                if len(ev5) == 0:
                    continue
                start = end - 50 + 1
                row_ids = np.arange(int(offsets[start]), int(offsets[end + 1]), dtype=np.int64)
                ms = np.repeat(np.arange(start, end + 1, dtype=np.int64), np.diff(offsets[start:end + 2]).astype(np.int64))
                lab = load_sources(src_root, seq, row_ids, ms)
                ev = torch.from_numpy(ev5).to(device)
                ptr = torch.tensor([0, len(ev5)], device=device)
                dts = torch.tensor([0.05], device=device)
                src, mask = stride_sample(ptr, 2048)
                n = int(mask.sum())
                pick = np.random.default_rng(end).permutation(n)[:NODES_PER_PACKET]
                node_src = src[0, :n].cpu().numpy()[pick]
                if router:
                    # the router's own routing decision (every node routed: tau 0): face -> part and rest-template point
                    _, _, face_r, bary_r, _ = SM.route_nodes(net, info["kind"], ev, ptr, dts, 0.0)
                    fr = face_r[0, :n].cpu().numpy()[pick]
                    br = bary_r[0, :n].float().cpu().numpy()[pick]
                    f = np.column_stack([fpart[fr], (br[:, :, None] * tmpl[faces[fr]]).sum(1) / 0.01])
                else:
                    tok = event_tokens(ev, ptr, dts, 180, 240)[src.reshape(-1)].view(1, -1, 7) * mask.unsqueeze(-1)
                    if model.xyz.t1aub_eg:
                        h = T1A.node_features(model.xyz.evgnn, tok, mask)
                    else:
                        h = tok
                    f = h[0, :n].float().cpu().numpy()[pick]
                face, bary, valid = lab["xag_face"][node_src], lab["xag_bary"][node_src], lab["xag_valid"][node_src]
                ok = valid & (face >= 0) & (face < len(faces)) & np.isfinite(bary).all(-1) & (bary >= 0).all(-1)
                w = np.clip(bary, 0, None)
                w = w / np.clip(w.sum(-1, keepdims=True), 1e-12, None)
                point = (w[:, :, None] * tmpl[faces[np.clip(face, 0, len(faces) - 1)]]).sum(1) / 0.01
                part = fpart[np.clip(face, 0, len(faces) - 1)]
                feats.append(f.astype(np.float32))
                labels.append(np.column_stack([ok, part, point]).astype(np.float32))
            arrays[f"{seq}/features"] = np.concatenate(feats)
            arrays[f"{seq}/labels"] = np.concatenate(labels)
            C.save(out / "progress.json", {"done": seq})
    np.savez_compressed(out / "states.npz", **arrays)
    C.save(out / "report.json", {"net": name, "status": "COMPLETE", "finished_utc": C.now(),
                                 "states_sha256": C.sha(out / "states.npz")}, exclusive=True)
    print("COMPLETE", out, flush=True)


# --------------------------------------------------------------------------- fits (CPU)
def fit_slots(name):
    from tools.s45loc import deep as D
    z = np.load(OUT / name / "slots/states.npz", allow_pickle=False)
    # bit-for-bit check against the network's own loop readings (tl_e1 readout and zgz E1)
    if name == "t1a":
        tl_ref = C.S45TR / "readouts/t1a/tl_e1/states.npz"
        zg_ref = C.S45TR / "evaluations/t1a/e1_step6000/states.npz"
    else:
        tl_ref = C.OUT / "readouts" / name / "tl_e1/states.npz"
        zg_ref = C.OUT / "evaluations" / name / "e1_step6000/states.npz"
    same = {}
    with np.load(tl_ref, allow_pickle=False) as a:
        for q in TL:
            same[q] = bool(np.array_equal(a[f"{q}/prediction"], z[f"{q}/prediction"]))
    with np.load(zg_ref, allow_pickle=False) as a:
        same["zgz_local"] = bool(np.array_equal(a["zgz_local/prediction"], z["zgz_local/prediction"]))
    basis = C.read(D.PROTOCOL)["pose_basis"]
    mu, V = np.asarray(basis["mean"]), np.asarray(basis["components"])
    task = {"f4_loop_tl": "", "f4_loop_zgz": ""}

    def A(t, key):
        C.require(t in task, "unexpected probe task")
        return z[key]
    return {"replay_equals_own_loop": same, "f4": D.probe(A, mu, V)}


def _mlp_fit(Xtr, Ytr, kind, steps=MLP_STEPS, seed=0):
    import torch
    torch.manual_seed(seed)
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    X = torch.from_numpy(((Xtr - m) / s).astype(np.float32))
    out_dim = 16 if kind == "part" else 3
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], MLP_HIDDEN), torch.nn.ReLU(), torch.nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
                              torch.nn.ReLU(), torch.nn.Linear(MLP_HIDDEN, out_dim))
    Y = torch.from_numpy(Ytr.astype(np.int64 if kind == "part" else np.float32))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (4096,), generator=g)
        o = net(X[idx])
        loss = torch.nn.functional.cross_entropy(o, Y[idx]) if kind == "part" else torch.nn.functional.smooth_l1_loss(o, Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return lambda Xn: net(torch.from_numpy(((Xn - m) / s).astype(np.float32))).detach().numpy()


def _linear_fit(Xtr, Ytr, kind, steps=MLP_STEPS, seed=0):
    import torch
    torch.manual_seed(seed)
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = (Xtr - m) / s
    if kind == "point":
        A_ = np.hstack([Xs, np.ones((len(Xs), 1))])
        W = np.linalg.solve(A_.T @ A_ + 1.0 * np.eye(A_.shape[1]), A_.T @ Ytr)
        return lambda Xn: np.hstack([(Xn - m) / s, np.ones((len(Xn), 1))]) @ W
    X = torch.from_numpy(Xs.astype(np.float32))
    Y = torch.from_numpy(Ytr.astype(np.int64))
    lin = torch.nn.Linear(X.shape[1], 16)
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, X.shape[0], (8192,), generator=g)
        loss = torch.nn.functional.cross_entropy(lin(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda Xn: lin(torch.from_numpy(((Xn - m) / s).astype(np.float32))).detach().numpy()


def fit_events(name):
    z = np.load(OUT / name.replace(":", "_") / "events/states.npz", allow_pickle=False)

    def data(seqs):
        F = np.concatenate([z[f"{q}/features"] for q in seqs])
        L = np.concatenate([z[f"{q}/labels"] for q in seqs])
        ok = L[:, 0] > 0.5
        return F[ok], L[ok, 1].astype(np.int64), L[ok, 2:5]
    train = [q for q in (f"{s}_{k}" for s in SUBJECTS if s != "ylf" for k in ("local", "global"))]
    tests = {"ylf_local": ["ylf_local"], "ylf_global": ["ylf_global"], "zgz_local": ["zgz_local"], "zgz_global": ["zgz_global"]}
    out = {"n_train_nodes": int(data(train)[0].shape[0])}
    if name.startswith("router:"):
        for t, seqs in tests.items():
            F, P, Y = data(seqs)
            out[t] = {"part_accuracy": float((F[:, 0].astype(np.int64) == P).mean()),
                      "point_err_median_mm": float(np.median(np.linalg.norm(F[:, 1:4] - Y, axis=1)) * 10), "n": int(len(F))}
        return out
    Ftr, Ptr, Ytr = data(train)
    rng = np.random.default_rng(0)
    keep = rng.permutation(len(Ftr))[:300000]
    Ftr, Ptr, Ytr = Ftr[keep], Ptr[keep], Ytr[keep]
    probes = {"linear": _linear_fit, "mlp": _mlp_fit}
    for kind_name, maker in probes.items():
        part_f, point_f = maker(Ftr, Ptr, "part"), maker(Ftr, Ytr, "point")
        res = {}
        for t, seqs in tests.items():
            F, P, Y = data(seqs)
            acc = float((part_f(F).argmax(-1) == P).mean())
            err = float(np.median(np.linalg.norm(point_f(F) - Y, axis=1)) * 10)      # cm -> mm
            res[t] = {"part_accuracy": acc, "point_err_median_mm": err, "n": int(len(F))}
        out[kind_name] = res
    return out


def fit(name):
    C.verify_protocol()
    s = {"net": name, "computed_utc": C.now(), "script_sha256": C.sha(__file__)}
    if (OUT / name / "slots/states.npz").is_file():
        s["slots"] = fit_slots(name)
    if (OUT / name.replace(":", "_") / "events/states.npz").is_file():
        s["events"] = fit_events(name)
    dst = OUT / name.replace(":", "_") / "summary.json"
    C.require(not dst.exists(), "summary exists: " + str(dst))
    C.save(dst, s, exclusive=True)
    print(dst, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("slots", "events", "fit"))
    p.add_argument("--net", required=True)
    a = p.parse_args(argv)
    {"slots": slots, "events": events, "fit": fit}[a.verb](a.net)


if __name__ == "__main__":
    main()
