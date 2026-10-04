#!/usr/bin/env python3
"""Generate the U1a verdict from completed explicit-last artifacts only.

Default writes docs/U1A_SHARED_MEASUREMENT_VERDICT.md only after every 6k
training, verification, evaluation, raw, gate and completion check finishes.
--phase 2k --dry-run validates the existing screening schema and prints a
SCREENING preview. It cannot write a verdict. No selection file, training,
evaluation, model, checkpoint tensor loading, or CUDA operation is used.

  python tools/u1a/make_verdict.py --phase 2k --dry-run
  python tools/u1a/make_verdict.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "u1a"
ROWS = REPO / "outputs" / "semkine"
TARGET = REPO / "docs" / "U1A_SHARED_MEASUREMENT_VERDICT.md"
SEEDS = ("3407", "3408")
MODES = ("shared", "untied")
LIMITS = {"ra_mm": 1.1, "local_mm": 1.1, "translation_ratio": 1.1, "root_deg": 1.0}
sys.path.insert(0, str(REPO))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved(value):
    path = Path(value)
    return (path if path.is_absolute() else REPO / path).resolve()


class Evidence:
    def __init__(self):
        self.files = {}

    def add(self, value, role, expected=None):
        path = resolved(value)
        require(not path.name.lower().startswith("selection_"), "selection artifacts cannot enter an explicit-last verdict")
        require(path.is_file(), f"missing {role}: {path}")
        if path not in self.files:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            self.files[path] = {"sha256": h.hexdigest(), "roles": set()}
        item = self.files[path]
        require(expected is None or expected == item["sha256"], f"{role}: source hash changed: {path}")
        item["roles"].add(role)
        return path

    def json(self, value, role, expected=None):
        return json.loads(self.add(value, role, expected).read_text(encoding="utf-8"))

    def text(self, value, role, expected=None):
        return self.add(value, role, expected).read_text(encoding="utf-8")

    def sources(self, value, role):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                self.add(value["path"], role, value["sha256"])
            for key, item in value.items():
                self.sources(item, role + "/" + str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                self.sources(item, role + "/" + str(index))

    def recheck(self):
        for path, item in self.files.items():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            require(h.hexdigest() == item["sha256"], f"input changed during verdict generation: {path}")


def complete_jobs(state, names, label, exact=True):
    jobs = state["jobs"]
    actual = {j["run"] for j in jobs}
    require(len(jobs) == len(actual), f"{label}: duplicate run identity")
    require(actual == set(names) if exact else set(names) <= actual, f"{label}: run identities differ")
    require(all(j.get("status") == "complete" and j.get("exit_code") == 0 for j in jobs),
            f"{label}: a process is not complete with exit 0")


def metric(value):
    number = float(value)
    require(math.isfinite(number), "non-finite diagnostic")
    return number


def raw_root(js, part):
    return metric(js["raw"][part]["measurement_metrics"]["root_geodesic_deg"]["mean"])


def check_gate(gate, rows, raws):
    from tools.u1a import gates
    require(gate["thresholds"] == LIMITS, "gate thresholds differ from the preregistration")
    comparisons = (("sharing", "shared", "untied", True),
                   ("shared_vs_S38", "shared", "S38", False),
                   ("untied_vs_S38", "untied", "S38", False))
    for name, candidate, reference, paired in comparisons:
        a = {s: {p: raw_root(raws[candidate][s], p) for p in ("overall", "global", "local")} for s in SEEDS}
        b = {s: {p: raw_root(raws[reference][s], p) for p in ("overall", "global", "local")} for s in SEEDS}
        recomputed = gates.comparison(rows[candidate], rows[reference], a, b, paired=paired, label=name)
        stored = gate["gates"][name]
        require(stored["pass"] is recomputed["pass"], f"{name}: pass flag disagrees with its criteria")
        require(len(stored["criteria"]) == len(recomputed["criteria"]), f"{name}: missing criteria")
        for got, expected in zip(stored["criteria"], recomputed["criteria"]):
            require(got["name"] == expected["name"] and got["pass"] is expected["pass"],
                    f"{name}: criterion identity/pass changed")
            require(abs(metric(got["observed"]) - expected["observed"]) < 1e-8
                    and abs(metric(got["maximum"]) - expected["maximum"]) < 1e-8,
                    f"{name}/{got['name']}: metric or guard changed")
    eligible = gate["budget"] == 6000 and gate["gates"]["sharing"]["pass"] and gate["gates"]["shared_vs_S38"]["pass"]
    require(gate["final_adoption_eligible"] is eligible, "adoption flag can neither omit nor override a root guard")


def load_phase(phase, evidence):
    budget = {"2k": 2000, "6k": 6000}[phase]
    expected = [f"u1a_{mode}_{phase}_s{seed}" for seed in SEEDS for mode in MODES]
    training = evidence.json(OUT / f"{phase}_state.json", phase + "/training")
    require(training["phase"] == phase, "training phase mismatch")
    complete_jobs(training, expected, phase + "/training")
    verification = evidence.json(OUT / f"{phase}_verification.json", phase + "/verification")
    require(set(verification) == set(expected), "four-run verification identities differ")
    for name, item in verification.items():
        require(item["last_step"] == budget, f"{name}: wrong last step")
        require(all(item.get(k) is True for k in ("frozen_encoder_bitwise", "all_state_finite", "paired_rank_streams_equal")),
                f"{name}: last/frozen/paired audit failed")
    evaluation = evidence.json(OUT / f"{phase}_evaluation_state.json", phase + "/evaluation")
    require(evaluation["phase"] == phase and evaluation["budget"] == budget and evaluation["stage"] == "complete",
            "recursive evaluation/cost/table pipeline is incomplete")
    complete_jobs(evaluation, expected, phase + "/evaluation")
    require(len(evaluation["reports"]) == 2 and {j["arm"] for j in evaluation["reports"]} == {f"u1a_{m}_{phase}" for m in MODES}
            and all(j.get("status") == "complete" and j.get("exit_code") == 0 for j in evaluation["reports"]), "cost reports are incomplete")
    evidence.sources(evaluation, phase + "/evaluation_sources")
    raw_state = evidence.json(OUT / f"{phase}_raw_state.json", phase + "/raw_state")
    require(raw_state["phase"] == phase, "raw phase mismatch")
    complete_jobs(raw_state, expected, phase + "/raw_state", exact=False)
    require({j["run"] for j in raw_state["jobs"]} <= set(expected) | {"s38_spmeas_s3407", "s38_spmeas_s3408"},
            "raw state contains an unexpected run")
    gate = evidence.json(OUT / f"{phase}_gate.json", phase + "/gate")
    require(gate["kind"] == "U1a_preregistered_engineering_gates" and gate["budget"] == budget
            and gate["stage"] == ("final_6k" if phase == "6k" else "screening_2k")
            and gate["checkpoint_rule"] == "explicit last", "gate is not the requested explicit-last phase")
    evidence.add(REPO / "tools/u1a/gates.py", "gate executable", gate["gate_script_sha256"])
    evidence.sources(gate["inputs"], phase + "/gate_inputs")
    evidence.sources(gate["diagnostic_sources"], phase + "/raw_sources")
    rows, raw, extended = {}, {}, {}
    for mode in MODES:
        row = evidence.json(ROWS / f"u1a_{mode}_{phase}_main_row.json", phase + "/" + mode + "/row")
        ext = evidence.json(ROWS / f"u1a_{mode}_{phase}_extended.json", phase + "/" + mode + "/extended")
        require(row["mode"] == mode and row["budget"] == budget and row["ckpt_rule"] == "last"
                and set(row["per_seed"]) == set(SEEDS) and row["n_seeds"] == 2, "row mode/budget/seeds changed")
        require(ext["provenance"] == row["provenance"] and ext["checkpoint_rule"] == "last", "extended provenance drift")
        evidence.sources(row, phase + "/" + mode + "/row_sources")
        rows[mode], extended[mode] = row, ext
        raw[mode] = {}
        for seed in SEEDS:
            source = gate["diagnostic_sources"][mode][seed]
            js = evidence.json(source["path"], phase + "/" + mode + "/" + seed + "/probe", source["sha256"])
            require(js["gate_eligible"] is True and js["raw"]["gate_eligible_full_window_coverage"] is True
                    and js["step"] == budget and js["seed"] == int(seed), "raw probe is partial or mismatched")
            require(js["raw"]["protocol_windows_total"] == js["raw"]["sampled_windows_total"] == 2590, "raw coverage incomplete")
            require(js["gradient"]["status"] == "ok" and js["gradient"]["encoder_state_unchanged"] is True,
                    "gradient diagnostic missing or changed frozen encoder")
            require(js["ckpt_sha256"] == row["provenance"][seed]["checkpoint"]["sha256"], "raw/recursive checkpoints differ")
            raw[mode][seed] = js
            provenance = row["provenance"][seed]
            metadata = evidence.json(provenance["training_metadata"]["path"], phase + "/" + mode + "/" + seed + "/metadata")
            require(metadata["resumed_from"] is None and metadata["max_steps"] == budget,
                    "6k/2k must restart prepared initialization, not resume compressed 2k cosine")
            cfg_path = evidence.add(provenance["config"]["path"], phase + "/" + mode + "/" + seed + "/config")
            cfg = yaml.safe_load(cfg_path.read_text())
            require(resolved(cfg["MODEL"]["INIT_FROM"]) == resolved(provenance["initialization"]["path"])
                    and cfg["MODEL"]["U1A_FREEZE_ENCODER"] is True, "prepared initialization/freeze changed")
            run = resolved(row["per_seed"][seed]["run"])
            for rank in (0, 1):
                evidence.json(run / f"pair_stream_rank{rank}.json", phase + "/" + mode + "/" + seed + f"/rank{rank}_stream")
    baseline = extended["shared"]["baseline"]["row"]
    require(baseline == extended["untied"]["baseline"]["row"], "two modes have different S38 reference rows")
    rows["S38"], raw["S38"] = baseline, {}
    require(baseline["budget"] == 6000 and baseline["ckpt_rule"] == "last", "S38 reference is not fixed 6000 last")
    evidence.sources(baseline, phase + "/S38_sources")
    for seed in SEEDS:
        source = gate["diagnostic_sources"]["S38"][seed]
        raw["S38"][seed] = evidence.json(source["path"], phase + "/S38/" + seed + "/probe", source["sha256"])
        for rank in (0, 1):
            a = resolved(rows["shared"]["per_seed"][seed]["run"]) / f"pair_stream_rank{rank}.json"
            b = resolved(rows["untied"]["per_seed"][seed]["run"]) / f"pair_stream_rank{rank}.json"
            require(json.loads(a.read_text()) == json.loads(b.read_text()), "paired DDP stream fingerprints differ")
    check_gate(gate, rows, raw)
    table_path = OUT / f"{phase}_table.md"
    table = evidence.text(table_path, phase + "/standard_table", evaluation["table"]["sha256"])
    from tools import report_table as rt
    aliases = ["rt_s37_3seed_tf_pert", "s38_u1a_fixed_reference", f"u1a_shared_{phase}", f"u1a_untied_{phase}"]
    labels = ["S37（统一3种子 last 参照）", "S38 spmeas（匹配2种子 last）",
              f"U1a shared {phase}（冻结 S38）", f"U1a untied {phase}（冻结 S38）"]
    expected_table = "\n".join([rt.HEADER, rt.RULE, rt.BASELINE, *[rt.row(a, label) for a, label in zip(aliases, labels)]])
    actual_table = "\n".join(line for line in table.splitlines() if line.startswith("|"))
    require(actual_table == expected_table, "stored standard table drifted from its main rows")
    for alias in aliases:
        evidence.json(ROWS / f"{alias}_main_row.json", phase + "/table_row/" + alias)
    return {"phase": phase, "budget": budget, "table": actual_table, "gate": gate, "rows": rows,
            "raw": raw, "training": training, "evaluation": evaluation, "extended": extended}


def load_debug_and_u0(evidence, baseline):
    debug = evidence.json(OUT / "debug_contract.json", "debug/forward_gradient_FK_hold")
    for mode in MODES:
        require(all(debug[mode].get(k) is True for k in ("finite_forward", "raw_prev_independent", "frozen_encoder",
                                                       "all_head_gradients_finite", "mean_added_once", "empty_hold", "mixed_empty_hold")),
                "debug wiring contract failed")
    state = evidence.json(OUT / "debug_state.json", "debug/DDP_state")
    names = [f"u1a_{mode}_debug_s3407" for mode in MODES]
    complete_jobs(state, names, "debug/DDP")
    verification = evidence.json(OUT / "debug_verification.json", "debug/last_frozen_paired")
    require(set(verification) == set(names) and all(v["last_step"] == 20 and v["frozen_encoder_bitwise"]
                                                  and v["all_state_finite"] and v["paired_rank_streams_equal"]
                                                  for v in verification.values()), "debug DDP audit failed")
    for mode in MODES:
        run = ROWS / f"u1a_{mode}_debug_s3407"
        metadata = evidence.json(run / "training_metadata.json", "debug/" + mode + "/actual_metadata")
        require(metadata["max_steps"] == 20 and metadata["devices"] == 2, "debug recipe is not the completed 20-step DDP run")
        config = run / "config_resolved.yaml"
        evidence.add(config if config.is_file() else metadata["config_path"], "debug/" + mode + "/config")
        checkpoint = evidence.add(run / "last.ckpt", "debug/" + mode + "/last_checkpoint")
        require(checkpoint.stat().st_mtime_ns <= (OUT / "debug_verification.json").stat().st_mtime_ns,
                "debug checkpoint is newer than its frozen-state audit")
        for rank in (0, 1):
            a = ROWS / "u1a_shared_debug_s3407" / f"pair_stream_rank{rank}.json"
            b = ROWS / "u1a_untied_debug_s3407" / f"pair_stream_rank{rank}.json"
            require(evidence.json(a, "debug/shared/stream") == evidence.json(b, "debug/untied/stream"),
                    "debug DDP source stream differs")
    for name in ("u1a_shared_debug_s3407.log.attempt1", "u1a_untied_debug_s3407.log.attempt1"):
        if (OUT / name).is_file():
            evidence.add(OUT / name, "debug/preserved_first_DDP_attempt")
    u0 = evidence.json(OUT / "u0_equivalence_full.json", "U0/full_equivalence")
    require(u0["kind"] == "U0_folded_S38_interface_equivalence_no_training" and u0["not_an_accuracy_arm"] is True
            and u0["all_interface_checks_pass"] is True and {str(r["seed"]) for r in u0["runs"]} == set(SEEDS),
            "full U0 equivalence failed or missing seeds")
    evidence.add(REPO / "tools/u1a/u0_equivalence.py", "U0/executable", u0["script_sha256"])
    require(u0["device"] == "cpu" and u0["precision"] == "fp32 full model, independent fp64 random algebra; TF32 disabled",
            "U0 precision/device differs from the stated equivalence scope")
    require(u0["source_files_before_sha256"] == u0["source_files_after_sha256"], "U0 altered its source files")
    for path, digest in u0["source_files_after_sha256"].items():
        evidence.add(path, "U0/source_unchanged", digest)
    for run in u0["runs"]:
        seed = str(run["seed"])
        require(run["interface_equivalence_pass"] is True and run["step"] == 6000
                and run["fixed_windows"]["full_2590_coverage"] is True
                and run["fixed_windows"]["sampled_windows_total"] == 2590, "U0 full coverage or pass missing")
        require(run["random_pool"]["samples"] == 384 and run["fixed_windows"]["window_ms"] == run["fixed_windows"]["step_ms"] == 50,
                "U0 random/fixed-window scope changed")
        require(run["fold"]["root_hidden"] == 64 and run["fold"]["finger_hidden"] == 256 and run["fold"]["output"] == 48,
                "U0 is not the stated original root64/finger256 folded interface")
        require(run["checkpoint_sha256"] == baseline["provenance"][seed]["checkpoint"]["sha256"], "U0 uses a different S38 last source")
        for key in ("checkpoint", "config", "manifest"):
            evidence.add(run[key], "U0/" + seed + "/" + key, run[key + "_sha256"])
        contracts = run["contracts"]
        require(contracts["encoder_parameters_frozen"] and contracts["encoder_BN_eval_and_statistics_unchanged"]
                and contracts["T_original_routed_root_head_and_prev_mlp_retained"]
                and contracts["optimizer_or_training_steps"] == 0
                and all(contracts["unchanged_components_equal_before_after_and_between_models"].values()),
                "U0 fixed-component/training contract failed")
    return debug, u0


def pass_word(value):
    return "通过" if value else "失败"


def conclusion(gate, screening=False):
    checks = gate["gates"]
    sharing, shared, untied = [checks[k]["pass"] for k in ("sharing", "shared_vs_S38", "untied_vs_S38")]
    prefix = "2k仅为筛查：" if screening else "6k判定："
    if sharing and not shared and not untied:
        first = prefix + "共享门通过；shared 与 untied 均未通过 S38 可用守护，共享未额外损伤并不代表当前接口可用。"
    else:
        first = prefix + f"共享门{pass_word(sharing)}，shared/S38 守护{pass_word(shared)}，untied/S38 守护{pass_word(untied)}。"
    adoption = "符合本轮工程候选门" if gate["final_adoption_eligible"] else "不符合本轮共享候选采用门"
    return first + "\n\n" + ("2k不能作最终采用决定。" if screening else adoption + "；任一 raw 或递推根守护失败都不能由 RA 均值掩盖。")


def diagnostic_tables(bundle):
    phase, gate, rows, raw = bundle["phase"], bundle["gate"], bundle["rows"], bundle["raw"]
    lines = ["### 根守护诊断（角度单位 °，正差表示候选更差）", "",
             "global 指 zgz_global 序列；root 为相机坐标下的 MANO global orientation，不代表世界坐标系追踪。",
             "raw 来自全部固定50ms窗口的 nonempty 测量 Exp(raw) R_ref；teacher-forced gain0.5输出不能替代 raw。", "",
             "| 比较 | seed | 范围 | raw 候选 | raw 参照 | raw 差 | 递推候选 | 递推参照 | 递推差 | raw门 | 递推门 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    comparisons = (("sharing", "shared", "untied"), ("shared_vs_S38", "shared", "S38"), ("untied_vs_S38", "untied", "S38"))
    for name, candidate, reference in comparisons:
        criteria = {c["name"]: c for c in gate["gates"][name]["criteria"]}
        for seed in SEEDS:
            for part in ("overall", "global"):
                ra, rb = raw_root(raw[candidate][seed], part), raw_root(raw[reference][seed], part)
                ca, cb = [metric(rows[mode]["per_seed"][seed][part]["root_rot_deg"]) for mode in (candidate, reference)]
                raw_pass = criteria[f"s{seed}/{part}/raw_nonempty_root_delta_deg"]["pass"]
                recursive_pass = criteria[f"s{seed}/{part}/recursive_root_delta_deg"]["pass"]
                lines.append(f"| {candidate}/{reference} | {seed} | {part} | {ra:.4f} | {rb:.4f} | {ra-rb:+.4f} | {ca:.4f} | {cb:.4f} | {ca-cb:+.4f} | {pass_word(raw_pass)} | {pass_word(recursive_pass)} |")
    lines += ["", "每个种子、overall/global 的 raw 与递推根差均须不超过 +1.0°；不允许跨种子平均掉根失败。", "",
              "### TF、hold、noevents与扰动定位（overall辅助，不进入主表）", "",
              "| arm | seed | raw root° | TF root° | 递推root° | raw finger RA-mm | TF RA-mm | hold RA-mm | noevents RA-mm | 10° retention k1/k2 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in (*MODES, "S38"):
        diagnostics = bundle["extended"]["shared"]["baseline"]["per_seed"] if mode == "S38" else bundle["extended"][mode]["per_seed"]
        for seed in SEEDS:
            js, probe = diagnostics[seed], raw[mode][seed]
            finger = metric(probe["raw"]["overall"]["measurement_metrics"]["finger_articulation_root_relative_mpjpe_mm"]["mean"])
            retention = js["perturb"]["10"]["retention_k1_k2_k5_k10_k20"]
            lines.append(f"| {mode} | {seed} | {raw_root(probe, 'overall'):.4f} | {js['tf']['overall']['root_rot_deg']:.4f} | {js['model']['overall']['root_rot_deg']:.4f} | {finger:.4f} | {js['tf']['overall']['mpjpe_ra_mm']:.4f} | {js['hold']['overall']['mpjpe_ra_mm']:.4f} | {js['noevents']['overall']['mpjpe_ra_mm']:.4f} | {retention[0]:.6f}/{retention[1]:.6f} |")
    lines += ["", "TF仍经过gain0.5固定滤波；raw finger articulation按probe独立的root-relative FK合同计量，TF/递推RA并非纯finger误差。",
              "hold与noevents对照检验空包保持。retention接近0.5、0.25主要反映固定滤波与prev-independent测量合同，不能当作根测量合格或新的动态恢复机制证据。",
              "", "### 平移与非根守护", "", "| 比较 | 判据 | 观察值 | 上限 | 结果 |", "|---|---|---|---|---|"]
    for name, _, _ in comparisons:
        for check in gate["gates"][name]["criteria"]:
            if "root_delta" not in check["name"]:
                lines.append(f"| {name} | {check['name']} | {check['observed']:.6f} | {check['maximum']:.3f} | {pass_word(check['pass'])} |")
    lines += ["", "T 比为候选/参照的两种子平均绝对平移误差比；不是旋转或 mesh 测量精度的替代指标。",
              "", "### 梯度方向与幅值（独立训练单 batch）", "",
              "| phase | arm | seed | 参数组 | 坐标数 | norm T | norm root | norm finger | cos(T,R) | cos(T,F) | cos(R,F) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in MODES:
        for seed in SEEDS:
            gradient = raw[mode][seed]["gradient"]
            for group, values in gradient["groups"].items():
                norm, cos = values["norm"], values["pairwise_cosine"]
                def c(key):
                    return "-" if cos[key] is None else f"{cos[key]:.6f}"
                lines.append(f"| {phase} | {mode} | {seed} | {group} | {values['n_parameter_coordinates']} | {norm['translation']:.6f} | {norm['root']:.6f} | {norm['finger']:.6f} | {c('translation__root')} | {c('translation__finger')} | {c('root__finger')} |")
    lines += ["", "每个向量是其加权任务项对 grad(log10(total)) 的贡献，使用同一 detached 1/(ln10·total)，不是分别对 log10(各项) 求导。",
              "这是每个 checkpoint 的一个固定、增广训练 batch（16样本），不是总体梯度统计或因果证据；不同参数坐标数与输出尺度使 norm 不能直接代表训练质量。",
              "untied bank 的任务坐标互不重叠，其零 cosine 属于结构定义；identity embedding 的耦合单独列出。", "",
              "### 输出尺度（rad；T delta 为 m）", "",
              "| phase | arm | seed | 上下文 | root分量RMS | root范数RMS | GT root范数RMS | finger分量RMS | GT finger分量RMS | finger范数RMS | T delta RMS |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode in (*MODES, "S38"):
        for seed in SEEDS:
            js = raw[mode][seed]
            contexts = [("zgz全nonempty窗口", js["raw"]["overall"]["raw_scale"], None)]
            if mode in MODES:
                contexts.append(("独立训练单batch", js["gradient"]["raw_scale"], js["gradient"]["translation_delta_component_m"]["rms"]))
            for label, scale, trans in contexts:
                keys = ("root_reference_residual_component_rad", "root_reference_residual_norm_rad",
                        "root_GT_reference_residual_norm_rad", "finger_residual_component_rad",
                        "finger_GT_residual_component_rad", "finger_residual_joint_norm_rad")
                cells = " | ".join(f"{metric(scale[k]['rms']):.6f}" for k in keys)
                t = "-" if trans is None else f"{metric(trans):.6f}"
                lines.append(f"| {phase} | {mode} | {seed} | {label} | {cells} | {t} |")
    lines += ["", "root 是相对 R_ref 的旋转残差；finger 是 residual45，hands_mean 只由原 FK/decode 加一次。尺度差、norm 支持域、容量和 T 重参数化均是待隔离因素，不能据本表指定单一原因。"]
    return lines


def render(current, screen, debug, u0, evidence, completion):
    phase = current["phase"]
    lines = [current["table"], "", conclusion(current["gate"], phase == "2k"), "",
             "# U1a 共享测量接口 verdict" + ("（仅schema筛查预览，非6k最终文档）" if phase == "2k" else ""), "",
             "## 协议与预算", "",
             "训练固定9人72序列，zgz_global/local同时为开发与测试，仅使用种子3407/3408和 explicit last；S37表行为统一3种子last参照，不是本轮配对对照。",
             "2k与6k各自从同一 prepared init 重新训练，metadata.resumed_from=None；6k不是从2k压缩cosine checkpoint续训。",
             "encoder来源是各seed已经训练6000步的S38，权重及BN统计冻结；2k/6k为额外 head-only 步数，不是与S38从头训练等总预算。",
             "shared与untied保留同输入字段、mask、type/joint embedding、逐槽LN、hidden64和3D输出；同seed的17个untied heads从shared逐tensor复制，配对rank数据流一致。",
             "与原S38相比，两臂同时改了translation槽（原routed T+prev_mlp组合重参数化）、norm支持范围与finger表达；shared相对untied才是本轮主要配对比较，宽度相同也不代表参数量匹配。", "",
             "## 保留的2k筛查结果", "", screen["table"], "", conclusion(screen["gate"], True), "",
             "以上2k失败结果原样保留；6k新结果不能覆盖筛查证据或把筛查结果改称最终通过。", "",
             "## Debug、DDP与源合同", "",
             "主代理已执行：python -m pytest tests/test_u1a_readout.py tests/test_s38.py tests/test_s37_routed_readout.py -q --disable-warnings；stdout记录45 passed、18 warnings、5.15s、exit0。",
             "当时未单独重定向测试日志；这是主代理执行记录，末尾测试源码SHA256是当前源码证据，不冒充独立测试执行artifact，也未为生成文档重复运行测试。",
             "debug_contract验证完整forward、非零head下root/finger对prev独立、有限梯度、原mean只加一次、混合空包保持和真实S38损失下降。",
             "两臂各20步双卡DDP退出0；debug及正式phase的last/frozen/finite/paired审计均通过，rank0/1的前两批完整packet字段指纹匹配。",
             "首次DDP通信停滞attempt1已保留，统一NCCL_P2P_DISABLE=1后完成；通信排障不作为网络结构失败证据。", "",
             "## U0 full接口等价检查及范围", "",
             "U0在原S38两块输入支持与hidden子空间内折叠root64与finger256为48D块对角读出；保留原T、prev_mlp、MANO、encoder及BN，未新增优化步骤。",
             "两个S38 last seed均覆盖全部2590个固定50ms窗口，原模型与U0每次收到同一GT/noised prev；另含384个非零随机pool的FP32/FP64代数检查及empty/mixed/train-mode forward检查。",
             "通过仅说明当前CPU FP32、禁用TF32、同输入forward的数值等价；独立FP64随机代数亦通过。没有AMP等价、梯度或optimizer等价、闭环递推轨迹/精度等价的证明；train-mode forward不等于训练等价。",
             "U0不是学到的17槽共享head或新准确率臂；它不能证明padding/LN、容量、translation重参数化或训练预算中的任一因素是失败原因。", "",
             "| seed | full窗口 | 随机FP32 raw48最大差(rad) | 随机FP64 raw48最大差(rad) | 固定窗口raw48最大差(rad) | full51最大差 | 根旋转最大差(°) | T最大差(m) |",
             "|---|---|---|---|---|---|---|---|"]
    for run in u0["runs"]:
        fixed, random = run["fixed_windows"], run["random_pool"]
        maxima = fixed["maxima"]
        lines.append(f"| {run['seed']} | {fixed['sampled_windows_total']} | {random['fp32_raw48_maxabs_rad']:.3e} | {random['fp64_raw48_maxabs_rad']:.3e} | {maxima['raw48_maxabs_rad']:.3e} | {maxima['full51_maxabs']:.3e} | {maxima['raw_root_rotation_maxdiff_deg']:.3e} | {maxima['translation_maxabs_m']:.3e} |")
    lines += ["", "## " + phase + "守护与辅助诊断", "", *diagnostic_tables(current),
              "", "## 2k辅助诊断（保留原筛查失败）", "", *diagnostic_tables(screen), "", "## 下一步范围", ""]
    checks = current["gate"]["gates"]
    if not checks["shared_vs_S38"]["pass"] and not checks["untied_vs_S38"]["pass"]:
        lines += ["两臂均未过S38守护：先恢复S38读出输入与norm支持范围、finger hidden256表达，并保留原S38 routed T+prev_mlp；以此隔离共享因素，另行预注册后再测。",
                  "这是下一组受控对照的建议，不是已确认的padding-LN因果解释；不据失败直接否定共享，也不继续叠加图消息掩盖读出接口问题。"]
    elif current["gate"]["final_adoption_eligible"]:
        lines += ["共享臂符合当前6k工程候选门，下一阶段仍需独立注册支持域、状态/dirty执行与mesh合同实验；本生成器不自动运行新阶段或修改当前臂。"]
    else:
        lines += ["共享臂未满足本轮全部采用门；先逐项定位失败的根、RA/local或T守护，并保持输入支持、norm、finger表达与T接口的受控比较，另行注册后再测。"]
    lines += ["", "两个种子、一个开发/测试受试者、冻结表征head probe仅支持本设置的工程判定；不能宣称最优架构、总体显著、真实异步执行、局部mesh更新或NeurIPS/TPAMI级创新。",
              "本工具不修改AGENTS.md的当前臂；所有推荐均为后续待检验机制。", "", "## 输入与SHA256", "",
              "下表包含生成时实际读取的主表、gate全inputs及递归来源、状态、诊断、checkpoint/config/manifest/init、配对DDP指纹、debug、U0与代码来源。未读取selection文件。", "",
              "| 路径 | 证据用途 | SHA256 |", "|---|---|---|"]
    for path, item in sorted(evidence.files.items(), key=lambda x: str(x[0])):
        try:
            display = str(path.relative_to(REPO))
        except ValueError:
            display = str(path)
        roles = "; ".join(sorted(item["roles"]))
        lines.append(f"| [{display}]({path.as_posix()}) | {roles} | {item['sha256']} |")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("2k", "6k"), default="6k", help="2k is allowed only with --dry-run")
    parser.add_argument("--dry-run", action="store_true", help="validate/hash/render stdout only; never write a document")
    args = parser.parse_args(argv)
    require(args.phase == "6k" or args.dry_run, "2k cannot write a final verdict; use --dry-run")
    evidence = Evidence()
    completion = None
    if args.phase == "6k":
        completion = evidence.json(OUT / "6k_completion_state.json", "6k/completion")
        require(completion.get("stage") == "complete", "6k completion pipeline is not complete; no verdict written")
    screen = load_phase("2k", evidence)
    current = screen if args.phase == "2k" else load_phase("6k", evidence)
    if completion is not None:
        for key, gate_key in (("sharing_gate_pass", "sharing"), ("shared_S38_guard_pass", "shared_vs_S38"), ("untied_S38_guard_pass", "untied_vs_S38")):
            require(completion[key] is current["gate"]["gates"][gate_key]["pass"], "completion/gate status mismatch")
        require(completion["final_adoption_eligible"] is current["gate"]["final_adoption_eligible"], "completion adoption mismatch")
    debug, u0 = load_debug_and_u0(evidence, current["rows"]["S38"])
    evidence.json(OUT / "initialization.json", "prepared_initialization_registry")
    for relative in ("docs/U1A_SHARED_MEASUREMENT_PREREG.md", "tests/test_u1a_readout.py", "tests/test_s38.py",
                     "tests/test_s37_routed_readout.py", "model/model.py", "semkine/u1a_readout.py", "semkine/train.py",
                     "tools/u1a/prepare.py", "tools/u1a/verify.py", "tools/u1a/debug.py", "tools/u1a/report.py",
                     "tools/u1a/probe.py", "tools/u1a/gates.py", "tools/u1a/u0_equivalence.py",
                     "tools/u1a/evaluate_phase.py", "tools/u1a/run_raw.py", "tools/u1a/finish_6k.py",
                     "tools/report_table.py", "tools/tracking/evalx.py", "tools/u1a/make_verdict.py"):
        evidence.add(REPO / relative, "code_or_prereg_source")
    document = render(current, screen, debug, u0, evidence, completion)
    evidence.recheck()
    if args.dry_run:
        print(document, end="")
        print("\ndry-run: schema/source hashes passed; no document written.", file=sys.stderr)
        return
    if TARGET.exists():
        require(TARGET.read_text(encoding="utf-8") == document, "existing verdict differs; refusing implicit overwrite")
        print(f"unchanged: {TARGET}")
    else:
        with TARGET.open("x", encoding="utf-8") as stream:
            stream.write(document)
        print(f"wrote {TARGET}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(f"U1a verdict refused: {error}")

