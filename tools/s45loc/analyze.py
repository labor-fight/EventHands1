#!/usr/bin/env python3
"""S45-LOC analysis (docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029); CPU only, reads task arrays, no model.

Writes <OUT>/summary.json and <OUT>/report.md. Every readout and every attribution label below is the pre-registered
definition; nothing is selected on the data. All RA numbers are mean root-aligned MPJPE in mm on the frozen
XCL scorer arrays; "identity" = holding the GT previous state.

Attribution labels for a factor's share of the zgz_local excess: >= 1/3 "major", [0.10, 1/3) "minor", otherwise
"negligible" (a wrong-signed share is "negligible"). Shares are reported separately and are not additive.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.s45loc import common as C  # noqa: E402

N_BOOT = 2000
_cache = {}


def arr(task, key):
    if task not in _cache:
        _cache[task] = dict(np.load(C.OUT / "tasks" / task / "states.npz", allow_pickle=False))
    return _cache[task][key]


def ra(task, seq):
    return float(arr(task, f"{seq}/mpjpe_ra_mm").mean())


def label(share):
    if share is None or not np.isfinite(share):
        return "n/a"
    return "major" if share >= 1 / 3 else "minor" if share >= 0.10 else "negligible"


def boot_diff(task_a, task_b, seq, key="mpjpe_ra_mm"):
    """Mean paired difference a - b over the same endpoints, with a run-block bootstrap 95 % interval."""
    a, b = arr(task_a, f"{seq}/{key}"), arr(task_b, f"{seq}/{key}")
    C.require(np.array_equal(arr(task_a, f"{seq}/end_ms"), arr(task_b, f"{seq}/end_ms")), "endpoint mismatch")
    d = a - b
    runs = arr(task_a, f"{seq}/run_id")
    ids = np.unique(runs)
    groups = [d[runs == r] for r in ids]
    rng = np.random.default_rng([C.SEED, 5])
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        stats.append(np.concatenate([groups[i] for i in pick]).mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)], "frac_negative": float((d < 0).mean()),
            "n_endpoints": int(len(d)), "n_runs": int(len(ids))}


def per_recording(task_a, task_b, seqs):
    """Recording-level RA differences a - b and their median / sign count (each recording paired with itself)."""
    vals = {s: ra(task_a, s) - ra(task_b, s) for s in seqs}
    v = np.array(list(vals.values()))
    return {"per_recording": vals, "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "n_positive": int((v > 0).sum()), "n": len(v)}


def assoc_profile(task, seq):
    """Recording-level association / event / record descriptors from the per-endpoint counts."""
    g = lambda k: arr(task, f"{seq}/stat_{k}")  # noqa: E731
    n_sv, n_nodes = g("n_sv").sum(), g("n_nodes").sum()
    assoc = g("n_sv_hit").sum() + g("n_sv_near").sum()
    out = {
        "events_per_packet_median": float(np.median(g("n_raw"))),
        "nodes_per_packet_median": float(np.median(g("n_nodes"))),
        "event_label_consistency": float(n_sv / max(n_nodes, 1)),
        "hand_events_hit": float(g("n_sv_hit").sum() / max(n_sv, 1)),
        "hand_events_near": float(g("n_sv_near").sum() / max(n_sv, 1)),
        "hand_events_none": float(g("n_sv_none").sum() / max(n_sv, 1)),
        "part_agreement": float(g("n_sv_agree").sum() / max(assoc, 1)),
        "part_agreement_hit": float(g("n_sv_hit_agree").sum() / max(g("n_sv_hit").sum(), 1)),
        "part_agreement_near": float(g("n_sv_near_agree").sum() / max(g("n_sv_near").sum(), 1)),
        "face_agreement_hit": float(g("n_sv_hit_face").sum() / max(g("n_sv_hit").sum(), 1)),
        "point_error_mm": float(g("sv_point_err_mm_sum").sum() / max(assoc, 1)),
        "part_agreement_by_time_quartile": [float(g(f"n_sv_agree_q{i}").sum() / max(g(f"n_sv_assoc_q{i}").sum(), 1))
                                            for i in range(4)],
        "offhand_share_of_nodes": float(g("n_off").sum() / max(n_nodes, 1)),
        "offhand_associated": float(g("n_off_assoc").sum() / max(g("n_off").sum(), 1)),
        "offhand_share_of_matched_mass": float(g("q_off").sum() / max(g("q_all").sum(), 1e-12)),
        "zn_mean": float(np.average(g("zn_qmean"), weights=np.maximum(g("q_all"), 1e-12))),
        "zn_over_share": float(g("n_zn_over").sum() / max(assoc + g("n_off_assoc").sum(), 1)),
        "token_dtie_mean": float(np.average(g("tok_dtie_mean"), weights=np.maximum(g("n_nodes"), 1e-12))),
        "token_sae_same_mean": float(np.average(g("tok_sae_same_mean"), weights=np.maximum(g("n_nodes"), 1e-12))),
        "token_sae_never_share": float(g("n_sae_same_never").sum() / max(n_nodes, 1)),
        "residual_mass_share": float(g("resid_mass").sum() / max(n_nodes, 1)),
        "direct_vertices_median": float(np.median(g("n_direct"))),
        "update_trans_mm_median": float(np.median(g("d_trans_mm"))),
        "update_root_deg_median": float(np.median(g("d_root_deg"))),
        "update_finger_rad_median": float(np.median(g("d_finger_rad"))),
    }
    return out


def range_of(profiles, key):
    v = np.array([p[key] for p in profiles.values()], np.float64)
    return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


# --------------------------------------------------------------------------- FK-based readouts
class FK:
    def __init__(self):
        import torch
        from config import load_config
        from mano_layer import ManoLayer
        cfg = load_config(str(C.CONFIG))
        self.cfg, self.device = cfg, torch.device("cpu")
        self.mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).eval()
        self.betas = {}

    def beta(self, seq):
        import torch
        from semkine import eval_track as ET
        from semkine.dataset import sequences_for_split, splits_manifest
        if seq not in self.betas:
            root = Path(self.cfg["DATA"]["ROOT"])
            found = {}
            for split in ("val_core", "train"):
                found.update(dict(sequences_for_split(root, split, splits_manifest(self.cfg))))
            aux = ET.load_sequence(root, found[seq], seq)[2]
            self.betas[seq] = torch.tensor(np.array(aux["betas"]), dtype=torch.float32).view(1, -1)
        return self.betas[seq]

    def joints(self, seq, states):
        import torch
        from semkine import eval_track as ET
        with torch.no_grad():
            j, _ = ET.mano_fk(self.mano, np.ascontiguousarray(states, np.float32), self.beta(seq), self.device)
        return j.astype(np.float64) * 1000.0   # mm


def update_decomposition(fk, task, seq):
    """E3 update in root-aligned joint space: U = RA(pred) - RA(prev), M = RA(cur) - RA(prev). Gain along the true
    motion beta = sum U.M / sum M.M; the off-motion part's RMS; both for all endpoints and the lowest-motion quartile."""
    from semkine import eval_track as ET
    prev, pred, cur = (fk.joints(seq, arr(task, f"{seq}/{k}")) for k in ("gt_previous", "prediction", "target"))
    U = ET.root_align(pred) - ET.root_align(prev)
    M = ET.root_align(cur) - ET.root_align(prev)
    m_norm = np.sqrt((M ** 2).sum((1, 2)) / M.shape[1])
    out = {}
    for name, sel in (("all", np.ones(len(M), bool)), ("low_motion_q1", m_norm <= np.percentile(m_norm, 25))):
        u, m = U[sel], M[sel]
        beta = float((u * m).sum() / max((m * m).sum(), 1e-12))
        off = u - beta * m
        out[name] = {"gain_along_motion": beta,
                     "off_motion_rms_mm": float(np.sqrt((off ** 2).sum((1, 2)).mean() / u.shape[1])),
                     "update_rms_mm": float(np.sqrt((u ** 2).sum((1, 2)).mean() / u.shape[1])),
                     "motion_rms_mm": float(np.sqrt((m ** 2).sum((1, 2)).mean() / m.shape[1])),
                     "ra_e3_mm": float(arr(task, f"{seq}/mpjpe_ra_mm")[sel].mean()),
                     "ra_identity_mm": float(arr(task, f"{seq}/identity_mpjpe_ra_mm")[sel].mean()),
                     "n": int(sel.sum())}
    return out


def retention(fk, task, base, seq, kind):
    """One-step survival of an injected prior error: parameter space and joint space (absolute for T, RA otherwise)."""
    from semkine import eval_track as ET
    C.require(np.array_equal(arr(task, f"{seq}/end_ms"), arr(base, f"{seq}/end_ms")), "endpoint mismatch")
    pd, p0 = arr(task, f"{seq}/prediction").astype(np.float64), arr(base, f"{seq}/prediction").astype(np.float64)
    used, truth = arr(task, f"{seq}/incoming_used").astype(np.float64), arr(task, f"{seq}/gt_previous").astype(np.float64)
    if kind == "T10":
        din, dout = used[:, 0:3] - truth[:, 0:3], pd[:, 0:3] - p0[:, 0:3]
    elif kind == "R5":
        from scipy.spatial.transform import Rotation as R
        din = (R.from_rotvec(used[:, 3:6]) * R.from_rotvec(truth[:, 3:6]).inv()).as_rotvec()
        dout = (R.from_rotvec(pd[:, 3:6]) * R.from_rotvec(p0[:, 3:6]).inv()).as_rotvec()
    else:
        din, dout = used[:, 6:51] - truth[:, 6:51], pd[:, 6:51] - p0[:, 6:51]
    rho_param = float((din * dout).sum() / max((din * din).sum(), 1e-18))
    J = {k: fk.joints(seq, v) for k, v in (("pd", pd), ("p0", p0), ("used", used), ("truth", truth))}
    if kind != "T10":
        J = {k: ET.root_align(v) for k, v in J.items()}
    jin, jout = J["used"] - J["truth"], J["pd"] - J["p0"]
    rho_joint = float((jin * jout).sum() / max((jin * jin).sum(), 1e-18))
    inj = float(arr(task, f"{seq}/incoming_mpjpe_ra_mm").mean()) if kind != "T10" else \
        float(arr(task, f"{seq}/incoming_mpjpe_abs_mm").mean())
    key = "mpjpe_abs_mm" if kind == "T10" else "mpjpe_ra_mm"
    carried = float(arr(task, f"{seq}/{key}").mean() - arr(base, f"{seq}/{key}").mean())
    return {"rho_param": rho_param, "rho_joint": rho_joint, "injected_error_mm": inj,
            "output_error_change_mm": carried, "metric": key}


def main():
    proto = C.read(C.PROTOCOL)
    s = {"protocol_sha256": C.sha(C.PROTOCOL), "analysis_sha256": C.sha(__file__), "computed_utc": C.now(),
         "smoke": C.SMOKE}
    zl, tl = "zgz_local", C.TRAIN_LOCAL
    # ---------------------------------------------------------------- R0: identity / cached S45 numbers
    cached = {m: dict(np.load(C.S45_EVAL / f"{m}_step6000_repeat1_none/states.npz")) for m in ("e1", "e2", "e3")}
    c1_e3 = dict(np.load(C.C1_EVAL / "e3_step6000_repeat1_none/states.npz"))
    r0 = {f"{m}_{q}": float(cached[m][f"{q}/mpjpe_ra_mm"].mean()) for m in cached for q in C.ZGZ}
    r0["identity_zgz_local"] = float(arr("b0_zgz_e3", f"{zl}/identity_mpjpe_ra_mm").mean())
    r0["c1_e3_zgz_local"] = float(c1_e3[f"{zl}/mpjpe_ra_mm"].mean())
    r0["excess_X"] = r0["e3_zgz_local"] - r0["identity_zgz_local"]
    r0["c1_gap_closed_share"] = (r0["c1_e3_zgz_local"] - r0["e3_zgz_local"]) / (r0["c1_e3_zgz_local"] - r0["identity_zgz_local"])
    r0["b0_check"] = C.read(C.OUT / "b0_check.json") if (C.OUT / "b0_check.json").exists() else None
    s["R0"] = r0
    X = r0["excess_X"]
    # ---------------------------------------------------------------- A: single step
    A = {}
    A["A1_in_sample_excess"] = {
        q: {"per_recording": {r: ra("b0_train_e3", r) - float(arr("b0_train_e3", f"{r}/identity_mpjpe_ra_mm").mean())
                              for r in rs}} for q, rs in (("local", C.TRAIN_LOCAL), ("global", C.TRAIN_GLOBAL))}
    for v in A["A1_in_sample_excess"].values():
        x = np.array(list(v["per_recording"].values()))
        v.update(median=float(np.median(x)), min=float(x.min()), max=float(x.max()), n_positive=int((x > 0).sum()))
    S_loc = A["A1_in_sample_excess"]["local"]["median"]
    Z = X - S_loc
    A["A2"] = {"X_zgz_local_excess": X, "S_in_sample_local_median": S_loc, "Z_zgz_specific": Z}
    A["A3_depth"] = {"D_transplant_ref_zgz_local": boot_diff("b2_zgz_local_transplant_ref", "b0_zgz_e3", zl),
                     "D_znshift_ref_zgz_local": boot_diff("b3_zgz_local_znshift_ref", "b0_zgz_e3", zl),
                     "M_znshift_zgz_train_local": per_recording("b2_train_local_znshift_zgz", "b0_train_e3", tl)}
    A["A3b_event_scale_gap"] = {"D_sigma_ref_zgz_local": boot_diff("b7_zgz_local_sigma_ref", "b0_zgz_e3", zl),
                                "M_sigma_zgz_train_local": per_recording("b7_train_local_sigma_zgz", "b0_train_e3", tl)}
    A["A4_density"] = {"M_subsample_train_local": per_recording("b4_train_local_subsample", "b0_train_e3", tl),
                       "M_zgzlike_train_local": per_recording("b4_train_local_zgzlike", "b0_train_e3", tl)}
    A["A5_token_timing"] = {"D_tokenmap_zgz_local": boot_diff("b6_zgz_local_tokenmap", "b0_zgz_e3", zl)}
    A["A6_cleaning"] = {
        "zgz_local": {"offhand_vs_rand": boot_diff("b5_zgz_local_offhand", "b5_zgz_local_offhand_rand", zl),
                      "clean_vs_rand": boot_diff("b5_zgz_local_clean", "b5_zgz_local_clean_rand", zl),
                      "offhand_vs_none": boot_diff("b5_zgz_local_offhand", "b0_zgz_e3", zl),
                      "clean_vs_none": boot_diff("b5_zgz_local_clean", "b0_zgz_e3", zl),
                      "drop_share_offhand": float(arr("b5_zgz_local_offhand", f"{zl}/stat_n_drop").sum()
                                                  / max(arr("b0_zgz_e3", f"{zl}/stat_n_raw").sum(), 1)),
                      "drop_share_clean": float(arr("b5_zgz_local_clean", f"{zl}/stat_n_drop").sum()
                                                / max(arr("b0_zgz_e3", f"{zl}/stat_n_raw").sum(), 1))},
        "train_local": {"offhand_vs_rand": per_recording("b5_train_local_offhand", "b5_train_local_offhand_rand", tl),
                        "clean_vs_rand": per_recording("b5_train_local_clean", "b5_train_local_clean_rand", tl)}}
    A["A7_sign_check_train_local"] = {"geometry_zero": per_recording("b1_train_local_geozero", "b0_train_e3", tl),
                                      "h3_zero": per_recording("b1_train_local_h3zero", "b0_train_e3", tl)}
    prof = {r: assoc_profile("b0_train_e3", r) for r in C.TRAIN_BASE}
    prof["zgz_local"] = assoc_profile("b0_zgz_e3", zl)
    prof["zgz_global"] = assoc_profile("b0_zgz_e3", "zgz_global")
    keys = [k for k in prof[zl] if not isinstance(prof[zl][k], list)]
    A["A8_profiles"] = {"zgz_local": prof[zl], "zgz_global": prof["zgz_global"],
                        "train_local_range": {k: range_of({r: prof[r] for r in tl}, k) for k in keys},
                        "train_global_range": {k: range_of({r: prof[r] for r in C.TRAIN_GLOBAL}, k) for k in keys},
                        "per_recording": prof}
    fk = FK()
    A["A9_update_decomposition"] = {"zgz_local": update_decomposition(fk, "b0_zgz_e3", zl),
                                    "train_local": {r: update_decomposition(fk, "b0_train_e3", r) for r in tl}}
    # zgz-specific shares are defined only when the zgz-specific part is at least 1 mm (else the excess is structural)
    zs = (lambda v: v / Z) if Z >= 1.0 else (lambda v: None)
    shares = {
        "in_sample_structural_of_X": S_loc / X,
        "depth_zn_shift_on_zgz": zs(-A["A3_depth"]["D_znshift_ref_zgz_local"]["mean"]),
        "depth_transplant_on_zgz": zs(-A["A3_depth"]["D_transplant_ref_zgz_local"]["mean"]),
        "depth_mirror_on_train": zs(A["A3_depth"]["M_znshift_zgz_train_local"]["median"]),
        "event_scale_gap_on_zgz": zs(-A["A3b_event_scale_gap"]["D_sigma_ref_zgz_local"]["mean"]),
        "event_scale_gap_mirror_on_train": zs(A["A3b_event_scale_gap"]["M_sigma_zgz_train_local"]["median"]),
        "density_emulation_on_train": zs(A["A4_density"]["M_subsample_train_local"]["median"]),
        "zgzlike_emulation_on_train": zs(A["A4_density"]["M_zgzlike_train_local"]["median"]),
        "token_timing_map_on_zgz": zs(-A["A5_token_timing"]["D_tokenmap_zgz_local"]["mean"]),
        # net effect of removing the events (vs keeping them); the count-matched contrast is reported, not labelled
        "oracle_association_cleaning_on_zgz": -A["A6_cleaning"]["zgz_local"]["clean_vs_none"]["mean"] / X,
        "oracle_offhand_cleaning_on_zgz": -A["A6_cleaning"]["zgz_local"]["offhand_vs_none"]["mean"] / X,
    }
    A["shares"] = {k: {"share": v, "label": label(v)} for k, v in shares.items()}
    s["A"] = A
    # ---------------------------------------------------------------- B: closed loop
    B = {}
    e1_zl, e3_zl = ra("b0_zgz_e1", zl), ra("b0_zgz_e3", zl)
    loop = {r: {"e1": ra("c1_train_e1", r), "e3": ra("b0_train_e3", r),
                "identity": float(arr("b0_train_e3", f"{r}/identity_mpjpe_ra_mm").mean())} for r in C.TRAIN_BASE}
    gaps = np.array([loop[r]["e1"] - loop[r]["e3"] for r in tl])
    B["B1_in_sample_loop"] = {"per_recording": loop, "train_local_e1_median": float(np.median([loop[r]["e1"] for r in tl])),
                              "train_local_gap_median": float(np.median(gaps)), "zgz_local_e1": e1_zl,
                              "zgz_local_e3": e3_zl, "zgz_local_gap": e1_zl - e3_zl}
    B["B2_zgzlike_loop_train_local"] = per_recording("c2_train_local_zgzlike_e1", "c1_train_e1", tl)
    B["B3_depth_transplant_loop_zgz_local"] = boot_diff("c3_zgz_e1_transplant_ref", "b0_zgz_e1", zl)
    B["B3_depth_znshift_loop_zgz_local"] = boot_diff("c3_zgz_e1_znshift_ref", "b0_zgz_e1", zl)
    B["B3_sanity_zgz_global_unchanged"] = all(
        bool(np.array_equal(arr(t, "zgz_global/prediction"), arr("b0_zgz_e1", "zgz_global/prediction")))
        for t in ("c3_zgz_e1_transplant_ref", "c3_zgz_e1_znshift_ref", "c5_zgz_e1_clean", "c5_zgz_e1_clean_rand"))
    B["B4_retention"] = {kind: {"zgz_local": retention(fk, f"c4_zgz_local_{kind}", "b0_zgz_e3", zl, kind),
                                "train_local": {r: retention(fk, f"c4_train_local_{kind}", "b0_train_e3", r, kind)
                                                for r in tl}} for kind in C.PERTURB}
    for kind, v in B["B4_retention"].items():
        for key in ("rho_param", "rho_joint"):
            v[f"train_local_{key}_median"] = float(np.median([x[key] for x in v["train_local"].values()]))
    B["B5_cleaning_loop_zgz_local"] = {"clean_vs_rand": boot_diff("c5_zgz_e1_clean", "c5_zgz_e1_clean_rand", zl),
                                       "clean_vs_none": boot_diff("c5_zgz_e1_clean", "b0_zgz_e1", zl)}
    B["B6_loop_association"] = {"zgz_local_e3": prof[zl], "zgz_local_e1": assoc_profile("b0_zgz_e1", zl),
                                "train_local_e1": {r: assoc_profile("c1_train_e1", r) for r in tl}}
    G = e1_zl - r0["identity_zgz_local"]
    loop_shares = {
        "in_sample_gap_vs_zgz_gap": float(np.median(gaps)) / (e1_zl - e3_zl),
        "depth_transplant_of_loop_excess": -B["B3_depth_transplant_loop_zgz_local"]["mean"] / G,
        "depth_znshift_of_loop_excess": -B["B3_depth_znshift_loop_zgz_local"]["mean"] / G,
        "zgzlike_emulation_of_loop_gap": B["B2_zgzlike_loop_train_local"]["median"]
        / (e1_zl - B["B1_in_sample_loop"]["train_local_e1_median"]),
        "oracle_cleaning_of_loop_excess": -B["B5_cleaning_loop_zgz_local"]["clean_vs_none"]["mean"] / G,
    }
    B["shares"] = {k: {"share": v, "label": label(v)} for k, v in loop_shares.items()}
    s["B"] = B
    C.save(C.OUT / "summary.json", s)
    write_report(s)
    print(C.OUT / "summary.json")


def write_report(s):
    table = subprocess.run(
        [C.PYTHON, str(ROOT / "tools/report_table.py"), "dt3_dz_l3_w200cfg", "s45",
         "--label", "dt3_dz_l3_w200cfg=S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *",
         "--label", "s45=S45：XYZ-C1 + SO(3)+平移+FK 损失（seed3407）"],
        cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    A, B, r0 = s["A"], s["B"], s["R0"]
    f = lambda x: f"{x:+.2f}"  # noqa: E731
    lines = ["# S45-LOC 结果（自动生成；冻结 S45、零训练、单 seed3407 描述性）", "", table, "",
             f"- E3 local {r0['e3_zgz_local']:.2f} / 恒等 {r0['identity_zgz_local']:.2f} → 超出 X {A['A2']['X_zgz_local_excess']:.2f}；"
             f"训练 local 中位超出 S {A['A2']['S_in_sample_local_median']:.2f}；zgz 特有 Z {A['A2']['Z_zgz_specific']:.2f}",
             f"- 深度移植（zgz_local→训练参考深度）E3 {f(A['A3_depth']['D_transplant_ref_zgz_local']['mean'])}；"
             f"zn 单列平移 {f(A['A3_depth']['D_znshift_ref_zgz_local']['mean'])}；训练 local 的 zn 平移到 zgz 深度中位 "
             f"{f(A['A3_depth']['M_znshift_zgz_train_local']['median'])}",
             f"- 事件侧尺度缺口（σ*）：zgz_local 事件绕腕缩放到训练中位 {f(A['A3b_event_scale_gap']['D_sigma_ref_zgz_local']['mean'])}；"
             f"训练 local 缩到 zgz 的 σ* 中位 {f(A['A3b_event_scale_gap']['M_sigma_zgz_train_local']['median'])}",
             f"- 训练 local 稀疏到 zgz 事件率中位 {f(A['A4_density']['M_subsample_train_local']['median'])}；稀疏+zgz 深度中位 "
             f"{f(A['A4_density']['M_zgzlike_train_local']['median'])}；zgz_local token 时间列分位映射 "
             f"{f(A['A5_token_timing']['D_tokenmap_zgz_local']['mean'])}",
             f"- oracle 清洗（对不丢 / 对随机同量丢弃）：去手外事件 {f(A['A6_cleaning']['zgz_local']['offhand_vs_none']['mean'])} / "
             f"{f(A['A6_cleaning']['zgz_local']['offhand_vs_rand']['mean'])}；去手外+错部位关联 "
             f"{f(A['A6_cleaning']['zgz_local']['clean_vs_none']['mean'])} / {f(A['A6_cleaning']['zgz_local']['clean_vs_rand']['mean'])}",
             f"- 闭环：zgz_local E1−E3 {B['B1_in_sample_loop']['zgz_local_gap']:.2f}，训练 local 中位 "
             f"{B['B1_in_sample_loop']['train_local_gap_median']:.2f}；深度移植 E1 {f(B['B3_depth_transplant_loop_zgz_local']['mean'])}，"
             f"zn 平移 E1 {f(B['B3_depth_znshift_loop_zgz_local']['mean'])}；"
             f"清洗 E1（对随机）{f(B['B5_cleaning_loop_zgz_local']['clean_vs_rand']['mean'])}",
             "- 保留率 ρ（参数 / 关节；1 = 不纠正）：" + "；".join(
                 f"{k} zgz_local {v['zgz_local']['rho_param']:.2f}/{v['zgz_local']['rho_joint']:.2f}，训练 local 中位 "
                 f"{v['train_local_rho_param_median']:.2f}/{v['train_local_rho_joint_median']:.2f}"
                 for k, v in B["B4_retention"].items()),
             "", "## 份额标签（预注册：≥1/3 major，0.10–1/3 minor，其余 negligible；不可相加）", ""]
    for block in ("A", "B"):
        for k, v in s[block]["shares"].items():
            sh = v["share"]
            lines.append(f"- {block}.{k}: {'n/a' if sh is None else f'{sh:.2f}'} → {v['label']}")
    (C.OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
