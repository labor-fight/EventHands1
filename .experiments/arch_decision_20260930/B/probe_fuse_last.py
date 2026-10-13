"""Selection-free re-check of the CNN-root fusion signal: CNN last.ckpt heads (ridge, lambda on train tail),
fused at the a-priori alpha into the S37 closed loop, for the zgz-selected and the last S37 checkpoints."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_arch as PA  # noqa: E402

PSD, REPO = PA.PSD, PA.REPO


@torch.no_grad()
def main():
    device = torch.device("cuda")
    torch.set_num_threads(16)
    t0 = time.time()
    cnns, hooks = {}, {}
    for r in PSD.CNN_RUNS:
        m, cnn_cfg, _ = PA.load_run(f"{r}:{REPO / r / 'last.ckpt'}", device)
        m.eval()
        store = {}
        m.rn.fc.register_forward_hook(lambda mod, inp, out, store=store: store.__setitem__("pen", inp[0].detach()))
        cnns[Path(r).name], hooks[Path(r).name] = m, store
    _, cfg, _ = PA.load_run(PSD.S37_RUNS[0], device)
    root = Path(cfg["DATA"]["ROOT"])
    mani = PA.splits_manifest(cfg)
    zgz = [PSD.SeqNC(root, d, s, device) for s, d in PA.sequences_for_split(root, "val_core", mani)]
    train = [PSD.SeqNC(root, d, s, device) for s, d in PA.sequences_for_split(root, "train", mani)]
    lnes_ch = PA.EV.event_channels(cnn_cfg)

    def pen(seq_rows):
        F_ = {k: [] for k in list(cnns) + ["gt51"]}
        spans = []
        for s, rows in seq_rows:
            i0 = sum(len(v) for v in F_["gt51"])
            for i in range(0, len(rows), 64):
                chunk = rows[i:i + 64]
                x = torch.from_numpy(np.stack([PA.ET.build_lnes(s.events, s.offsets, int(e), PA.STEP, lnes_ch)
                                               for _, e in chunk])).to(device)
                for n_, c in cnns.items():
                    c(x, torch.zeros(len(chunk), 51, device=device))
                    F_[n_].append(hooks[n_]["pen"].float().cpu())
                F_["gt51"].append(torch.from_numpy(np.stack([s.pos51[e] for _, e in chunk]).astype(np.float32)))
            spans.append((s.name, slice(i0, i0 + len(rows))))
        return {k: torch.cat(v) for k, v in F_.items()}, spans

    rng2 = np.random.default_rng(2)
    Ftr, _ = pen([(s, PSD.rows_of(s, 300, rng2)) for s in train])
    Fte, spans = pen([(s, PSD.rows_of(s)) for s in zgz])
    print("extracted", len(Ftr["gt51"]), len(Fte["gt51"]), f"{time.time() - t0:.0f}s", flush=True)
    g = Ftr["gt51"].double()
    Ytr = torch.cat([PA.aa_to_R(g[:, 3:6]).reshape(-1, 9), g[:, 6:51]], 1)
    Rabs, res = {}, {"alpha": PA.ALPHA, "cnn_ckpt": "last.ckpt", "heads": {}, "loops": {}}
    for n_ in cnns:
        predict, info, hold_pred, _ = PA.ridge_fit(Ftr[n_], Ytr, device)
        p = predict(Fte[n_])
        Rabs[n_] = PSD.project_so3(p[:, :9].reshape(-1, 3, 3).float()).to(device)
        ktr = int(len(Ytr) * 0.85)
        res["heads"][n_] = {"lambda": info["lambda"],
                            "holdout_rot_deg": float(PA.rot_err_deg(hold_pred, Ftr["gt51"][ktr:]).mean()),
                            "zgz_rot_deg": {s: float(PA.rot_err_deg(p[sl], Fte["gt51"][sl]).mean()) for s, sl in spans}}
    rng0 = np.random.default_rng(0)
    inits = {s.name: [PA.ET.sample_init_noise(cfg, rng0, 1.0) for _ in s.runs] for s in zgz}
    nfr = {s.name: sum(len(e) for _, e in s.runs) for s in zgz}
    tags = (("selected", None),) if "--only-selected" in sys.argv else (("selected", None), ("last", "last.ckpt"))
    for r, cn in zip(PSD.S37_RUNS, cnns):
        for tag, ck in tags:
            m, _, sel = PA.load_run(r if ck is None else f"{r}:{REPO / r / ck}", device)
            m.eval()
            per = {}
            for s, (_, sl) in zip(zgz, spans):
                m.set_hand_context(s.betas, s.K)
                preds, _, gts = PA.loop_variants(m, s, inits[s.name], device,
                                                 [("base", None, 0.0), ("cnn_last_fuse", Rabs[cn][sl], PA.ALPHA)])
                for vn, pr in zip(("base", "cnn_last_fuse"), preds):
                    per.setdefault(vn, {})[s.name] = PA.summ(PA.decomp(m, pr, gts, s.betas))
                Rg = PA.aa_to_R(gts[:, 3:6])
                e_s = PA.R_to_aa(PA.aa_to_R(preds[0][:, 3:6]) @ Rg.transpose(-1, -2)).cpu().double()
                e_a = PA.R_to_aa(Rabs[cn][sl] @ Rg.transpose(-1, -2)).cpu().double()
                per.setdefault("mechanism", {})[s.name] = {
                    "cnn_head_rot_deg": float(torch.rad2deg(e_a.norm(dim=1)).mean()),
                    "s37_closed_rot_deg": float(torch.rad2deg(e_s.norm(dim=1)).mean()),
                    "err_vector_cos_mean": float(torch.nn.functional.cosine_similarity(e_a, e_s, dim=1).mean()),
                    "err_norm_corr": float(np.corrcoef(e_a.norm(dim=1).numpy(), e_s.norm(dim=1).numpy())[0, 1])}
            mech = per.pop("mechanism")
            for vn, v in per.items():
                v["frame_weighted_ra"] = sum(v[s.name]["ra"] * nfr[s.name] for s in zgz) / sum(nfr.values())
            key = f"{Path(r).name}:{tag}"
            res["loops"][key] = {"ckpt": str(sel["ckpt"]), **per, "mechanism": mech}
            print(key, PA.r3({vn: {k: (x["ra"] if isinstance(x, dict) else x) for k, x in v.items()}
                              for vn, v in per.items()}), f"{time.time() - t0:.0f}s", flush=True)
            del m
            torch.cuda.empty_cache()
    out = HERE / f"results_fuse_last_{time.strftime('%H%M%S')}.json"
    out.write_text(json.dumps(PA.r3(res), indent=1, default=str))
    print("done", out, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
