"""Assemble sealed Stage5 artifacts; never train, predict, or open GT arrays.

The optional document write is idempotent and touches only current-round blocks.
No method is adopted and no repository default/configuration is changed here.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/s43_revisit_20261028"
ARMS = ("canonical", "s43", "field_box", "field_box_collapse",
        "field_static", "field_static_collapse")
LABELS = dict(canonical="冻结 S43", s43="同预算 S43",
    field_box="Field 手框／保留时间", field_box_collapse="Field 手框／时间坍缩",
    field_static="Field 骨段／保留时间", field_static_collapse="Field 骨段／时间坍缩")
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm",
           "mpvpe_abs_mm", "mpvpe_joint_wrist_aligned_mm", "root_error_mm")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def last_row(path):
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return None
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 131072))
        rows = stream.read().splitlines()
    return json.loads(rows[-1])


def status(train_root, eval_root):
    train_root, eval_root = Path(train_root), Path(eval_root)
    stages = {}
    for stage in ("small", "full"):
        rows = {}
        for arm in ARMS[1:]:
            folder = train_root / "train" / stage / arm
            row = last_row(folder / "curve.jsonl")
            rows[arm] = dict(started=(folder / "manifest.json").exists(),
                complete=(folder / "complete.json").exists(),
                failed=(folder / "failure.json").exists(),
                supervised_outputs=0 if row is None else row["supervised_outputs"],
                optimizer_updates=0 if row is None else row["update"],
                elapsed_training_curve_seconds=None if row is None else row["elapsed_seconds"])
        stages[stage] = rows
    complete = (train_root / "training_collection_reporting_complete.json").exists()
    failed = any((train_root / name).exists() for name in
        ("postmaster_failure.json", "full_execution_orchestrator_failure.json")) or any(
        r["failed"] for phase in stages.values() for r in phase.values())
    state = "COMPLETE" if complete else "FAILED_PRESERVED" if failed else "EVALUATED_AWAITING_REPORT" if (eval_root / "evaluation_complete.json").exists() else "IN_PROGRESS"
    return dict(status=state,
        phases=stages, all_full_sealed=(train_root / "all_final_weights_complete.json").exists(),
        all_predictions_complete=(eval_root / "all_predictions_complete.json").exists(),
        resources_complete=(eval_root / "resources_complete.json").exists(),
        note="A curve is progress, not a sealed final result. No task metrics are read by status.")


def sealed_inputs(train_root, eval_root):
    train_root, eval_root = Path(train_root), Path(eval_root)
    final = read(train_root / "all_final_weights_complete.json")
    audit = read(train_root / "audits/full.json")
    evaluation = read(eval_root / "evaluation_complete.json")
    scores = read(eval_root / "report/results.json")
    predictions = read(eval_root / "all_predictions_complete.json")
    resources = read(eval_root / "resources_complete.json")
    if (final.get("status") != "COMPLETE" or set(final["arms"]) != set(ARMS[1:])
            or final["supervised_outputs_per_arm"] != 6144000
            or audit.get("status") != "PASS"
            or not audit.get("all_per_update_source_target_traces_paired")
            or not audit.get("all_sources_unchanged")):
        raise RuntimeError("All five full budgets/source/pairing gates must pass.")
    if audit.get("sourcefreeze_sha256") != final.get("sourcefreeze_sha256"):
        raise RuntimeError("Full audit and weight cohort source identities differ.")
    if (evaluation.get("status") != "COMPLETE" or not evaluation.get("source_files_unchanged")
            or scores.get("status") != "COMPLETE" or predictions.get("status") != "COMPLETE"
            or predictions.get("total_traces") != 120 or resources.get("status") != "COMPLETE"
            or not scores.get("all_predictions_saved_before_GT_pose")):
        raise RuntimeError("Complete fixed evaluation artifacts required; no partial-score substitution.")
    if evaluation["report_sha256"] != sha(eval_root / "report/results.json"):
        raise RuntimeError("Saved score identity changed.")
    if evaluation["resources_sha256"] != sha(eval_root / "resources_complete.json"):
        raise RuntimeError("Saved resource identity changed.")
    gate = evaluation["allfive_full_weights_gate"]
    if (gate["marker_sha256"] != sha(train_root / "all_final_weights_complete.json")
            or gate["audit_sha256"] != sha(train_root / "audits/full.json")
            or scores.get("allfive_weights_gate") != gate):
        raise RuntimeError("Score/evaluation belongs to a different final weight cohort.")
    evaluation_freeze = eval_root / "actual_evaluation_sourcefreeze.json"
    freeze_sha = sha(evaluation_freeze)
    if (evaluation.get("sourcefreeze_sha256") != freeze_sha
            or predictions.get("sourcefreeze_sha256") != freeze_sha
            or resources.get("sourcefreeze_sha256") != freeze_sha):
        raise RuntimeError("Evaluation/prediction/resource source identities differ.")
    evaluation_dependencies = read(evaluation_freeze)["dependencies"]
    for path, expected in evaluation_dependencies.items():
        if sha(path) != expected:
            raise RuntimeError("Frozen evaluation dependency changed: " + path)
    full_freeze = Path(final["sourcefreeze_path"])
    if sha(full_freeze) != final["sourcefreeze_sha256"]:
        raise RuntimeError("Full executable source freeze identity changed.")
    full_source = read(full_freeze)
    for group in ("code", "controls", "configs", "engineering"):
        for path, expected in full_source[group].items():
            if sha(path) != expected:
                raise RuntimeError("Frozen full dependency changed: " + path)
    if len(predictions.get("predictions", {})) != 120:
        raise RuntimeError("Exactly120 sealed prediction archives required.")
    for prefix, expected in predictions["predictions"].items():
        if sha(Path(prefix).with_suffix(".npz")) != expected:
            raise RuntimeError("Sealed prediction archive changed: " + prefix)
    resource_rows = resources["results"]
    expected_resources = {}
    for arm in ARMS:
        for source in ("native", "noise30"):
            kind = "native_s43" if arm in ("canonical", "s43") else "field"
            profiles = [(25, 25, kind), (50, 50, kind), (100, 100, kind), (25, 50, kind)]
            if kind == "native_s43":
                profiles.append((50, 50, "shared_audit"))
            for hop, commit, runtime in profiles:
                name = f"resource_{arm}_{source}_q{hop}_c{commit}_{runtime}"
                expected_resources[name] = (arm, source, hop, commit, runtime == "native_s43")
    if (resources.get("fixed_workloads") != 52 or len(resource_rows) != 52
            or {r["name"] for r in resource_rows} != set(expected_resources)):
        raise RuntimeError("Exact52 preregistered resource workloads required.")
    files = [train_root / "all_final_weights_complete.json", train_root / "audits/full.json",
        eval_root / "evaluation_complete.json", eval_root / "report/results.json",
        eval_root / "resources_complete.json", eval_root / "all_predictions_complete.json",
        evaluation_freeze, full_freeze]
    for arm in ARMS[1:]:
        weight = Path(final["final_weights"][arm])
        receipt = read(weight.with_name("complete.json"))
        if (receipt["actual_supervised_outputs"] != 6144000
                or receipt["final_weights_sha256"] != sha(weight)
                or receipt["sourcefreeze_sha256"] != final["sourcefreeze_sha256"]):
            raise RuntimeError("Final weight/budget identity changed: " + arm)
        files.extend((weight, weight.with_name("complete.json")))
    for row in resources["results"]:
        if row.get("execution", {}).get("resource_exclusivity_valid") is False:
            raise RuntimeError("Resource GPU exclusivity was violated during execution.")
        path = Path(row["resource"])
        if sha(path) != row["sha256"]:
            raise RuntimeError("Resource identity changed: " + str(path))
        data = read(path)
        arm, source, hop, commit, native = expected_resources[row["name"]]
        if (data.get("arm") != arm or data.get("query_ms") != hop or data.get("commit_ms") != commit
                or data.get("native_s43_without_unused_field") != native
                or data.get("status") != row["status"]
                or Path(data["source"]).resolve() != (eval_root / "resource_inputs" / source).resolve()):
            raise RuntimeError("Resource row and actual workload identity differ.")
        if data.get("status") != "COMPLETE" and data.get("throughput") is not None:
            raise RuntimeError("Failed resource case contains misleading full-flow throughput.")
        source_path = path.with_suffix(".sourcefreeze.json")
        if sha(source_path) != data["sourcefreeze_sha256"]:
            raise RuntimeError("Resource execution source identity changed.")
        execution = read(source_path)
        if execution.get("engineering") is not False or execution.get("allfive_weights_gate") != gate:
            raise RuntimeError("Resource execution was not from the same complete full cohort.")
        for dependency, expected in execution["sources"].items():
            if sha(dependency) != expected:
                raise RuntimeError("Resource execution dependency changed: " + dependency)
        files.extend((path, source_path))
    if resources["failed_workloads"] != sum(row["status"] != "COMPLETE" for row in resource_rows):
        raise RuntimeError("Resource failure count differs from preserved workloads.")
    return final, audit, scores, resources, files


def comparison(primary, candidate, control):
    # Positive means candidate has smaller error; never label this significance.
    return {part: {metric: primary[control][part][metric] - primary[candidate][part][metric]
        for metric in METRICS} for part in ("local", "global", "overall")}


def resource_case(resources, arm, final):
    kind = "native_s43" if arm in ("canonical", "s43") else "field"
    name = f"resource_{arm}_native_q50_c50_{kind}"
    rows = [r for r in resources["results"] if r["name"] == name]
    if len(rows) != 1:
        raise RuntimeError("Exactly one fair native q50 resource case required: " + name)
    data = read(rows[0]["resource"])
    if (data.get("arm") != arm or data.get("engineering") is not False
            or data.get("query_ms") != 50 or data.get("commit_ms") != 50
            or data.get("threads") != 2 or not data.get("source_files_unchanged")
            or data.get("precision") != "FP32eval/TF32off/renderFP32/stateFP32"
            or not str(data.get("device", "")).startswith("cuda")
            or data.get("native_s43_without_unused_field") != (arm in ("canonical", "s43"))):
        raise RuntimeError("Primary resource case serving contract differs: " + name)
    if arm == "canonical":
        if data.get("checkpoint") not in (None, "None"):
            raise RuntimeError("Canonical resource case contains a replacement checkpoint.")
    elif Path(data["checkpoint"]).resolve() != Path(final["final_weights"][arm]).resolve():
        raise RuntimeError("Primary resource case uses another trained checkpoint.")
    if data.get("status") == "COMPLETE" and data.get("native_input_count") != 247738:
        raise RuntimeError("Primary resource case is not the complete native stream.")
    return data


def number(value, digits=3):
    if value is None:
        return "未得到"
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite value cannot be formatted as a result.")
    return f"{value:.{digits}f}"


def replace_block(path, marker, body):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    begin, end = f"<!-- {marker}_BEGIN -->", f"<!-- {marker}_END -->"
    block = begin + "\n" + body.strip() + "\n" + end
    if begin in text or end in text:
        if text.count(begin) != 1 or text.count(end) != 1:
            raise RuntimeError("Ambiguous generated document block: " + str(path))
        a, b = text.index(begin), text.index(end) + len(end)
        if text.index(end) < a:
            raise RuntimeError("Reversed generated document markers: " + str(path))
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    temporary = path.with_name(path.name + ".s43_report_tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def assemble(train_root, eval_root, *, write_docs=False):
    train_root, eval_root = Path(train_root), Path(eval_root)
    final, audit, scores, resources, files = sealed_inputs(train_root, eval_root)
    primary = scores["primary_heldperson"]
    required = set(ARMS)
    if set(primary) != required:
        raise RuntimeError("All six models must be reported.")
    effects = {f"{candidate}_vs_{control}": comparison(primary, candidate, control)
        for candidate, control in (("s43", "canonical"), ("field_box", "canonical"),
            ("field_box", "s43"), ("field_static", "s43"), ("field_static", "field_box"),
            ("field_box", "field_box_collapse"), ("field_static", "field_static_collapse"))}
    table = ["| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE（递推） | Latency P50 | FLOPs/step | Params |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    resource_primary = {}
    for arm in ARMS:
        p = primary[arm]
        r = resource_case(resources, arm, final)
        resource_primary[arm] = r
        throughput = r.get("throughput") if r.get("status") == "COMPLETE" else None
        latency = None if throughput is None else throughput["steady_cycle_p50_ms"]
        operations = r.get("FLOPs", {}).get("phases", {}).get("steady", {})
        flops = operations.get("actual_dense_operator_equivalent_NN_FLOPs_per_output") if r.get("status") == "COMPLETE" else None
        params = r["memory"]["model_parameter_count"]
        table.append("| " + " | ".join((LABELS[arm],
            number(p["local"]["mpjpe_ra_mm"]), number(p["global"]["mpjpe_ra_mm"]),
            number(p["local"]["mpvpe_ra_mm"]), number(p["global"]["mpvpe_ra_mm"]),
            number(p["overall"]["mpjpe_ra_mm"]),
            number(latency, 2) + (" ms" if latency is not None else ""),
            number(None if flops is None else flops / 1e9) + (" G_NN" if flops is not None else ""),
            number(params / 1e6) + " M")) + " |")
    consistency = scores["fixed_commit_extra_query_parity"]
    if set(consistency) != {a + "/" + r for a in ARMS for r in ("zgz_local", "zgz_global")}:
        raise RuntimeError("All12 model/recording consistency results required.")
    consistent = all(row["within_declared_numeric_tolerance"] for row in consistency.values())
    inherited = {str(path.resolve()): sha(path) for path in files}
    outcome = dict(schema="s43_full_field_postfinal_summary_v1", status="COMPLETE",
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        source_artifacts=inherited, supervised_outputs_per_arm=6144000,
        no_training_prediction_or_GT_arrays_read=True, primary=primary, effects_mm=effects,
        primary_resources=resource_primary, extra_query_consistency_pass=consistent,
        failed_resource_workloads=resources["failed_workloads"],
        interpretation="Effect sizes only; positive candidate-control differences do not establish significance, equivalence, causality or Nature-level originality. No posthoc numeric adoption gate and no automatic default switch.",
        default_configuration_changed=False)
    report = eval_root / "report"
    lines = [
        '<a id="s43-field-full-results-20261028"></a>',
        "## S43 第三轮完整前端：固定预算训练与完整回放结果",
        "",
        "本节由封存产物自动汇总；五臂各6144000实际监督输出、固定末权重，120条语义回放全部保存后评分。"
        "这是单seed3407的探索性证据；zgz为本轮未训练受试者，历史上已反复作开发／测试集，不称盲测。",
        "现有训练／评测录制沿历史v2e来源及整数ms事件合同，不能由软件回放宣称真实传感器硬件低延迟；合成物理实验仍与现有数据分报。",
        "",
        "β0部署、W50原始无滤波、1ms回溯；local/global指录制类型。J0关节与V0顶点分别对齐，"
        "RA为两类逐帧加权的递推J0误差，不是Procrustes。单位mm。",
        "", *table, "",
        "Latency为真实ch3300ms、CPU2、FP32评测的无内部profile整mesh服务P50；"
        "原S43使用无额外field的原生服务。FLOPs为实际Conv/Linear执行形状的直接卷积等价NN计数，"
        "2FLOPs/MAC；不含BN／激活／采样／MANO／CPU记忆维护，不是硬件指令或旧occupancy估计。",
        "",
        "| 误差下降：对照－候选（正值更好） | J0 local | J0 global | 递推RA |",
        "|---|---:|---:|---:|"]
    for key, e in effects.items():
        lines.append(f'| {key} | {number(e["local"]["mpjpe_ra_mm"])} | '
            f'{number(e["global"]["mpjpe_ra_mm"])} | {number(e["overall"]["mpjpe_ra_mm"])} |')
    lines += ["", "| 原生q50整mesh服务 | 状态 | P95 / P99 ms | 输出/s | 事件/s | CUDA allocated MiB | RSS观测峰 MiB |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        r = resource_primary[arm]
        t = (r.get("throughput") or {}) if r.get("status") == "COMPLETE" else {}
        m = r["memory"]
        lines.append("| " + " | ".join((LABELS[arm], r["status"],
            number(t.get("steady_cycle_p95_ms"), 2) + " / " + number(t.get("steady_cycle_p99_ms"), 2),
            number(t.get("outputs_per_second"), 2), number(t.get("events_per_second"), 0),
            number(m["torch_peak_allocated_bytes"] / 2**20, 2),
            number(m["peak_RSS_bytes"] / 2**20, 2))) + " |")
    lines += ["", "RSS为进程高水位，含模型／source mmap／临时输入及先前加载；不等于纯表示存储。"
        "CUDA与RSS是各自观测，不能相加。失败workload的峰只属其实际执行前缀，不能据此声称完整运行更省内存。",
        "", "| 分项profile均值 ms（嵌套项不相加） | 事件维护 | 整数bin CPU | 几何＋条件D2H | 全局trunk | 局部encoder | 输出MANO FK |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        measured = resource_primary[arm]
        phases = measured.get("component_profile", {}).get("phases", {}) if measured.get("status") == "COMPLETE" else {}
        names = ("sensor_maintenance_ms", "integer_timebin_readout_CPU_ms",
            "query_geometry_and_conditions_D2H_ms", "global_sparse_trunk_ms",
            "local_encoder_ms", "surface_fk_ms")
        values = [phases.get("steady/" + name, {}).get("mean_ms") for name in names]
        lines.append("| " + " | ".join([LABELS[arm]] + [number(v) for v in values]) + " |")
    lines += ["", "‘未得到’可指无此分支或该分项未完成，不能当作零成本；"
        "分项来自独立reset且输出一致的profile pass，不能替代第一遍整链计时。",
        "", "| β0绝对误差补充 mm | local J / V / root | global J / V / root |",
        "|---|---:|---:|"]
    for arm in ARMS:
        p = primary[arm]
        parts = [" / ".join(number(p[part][metric]) for metric in
            ("mpjpe_abs_mm", "mpvpe_abs_mm", "root_error_mm")) for part in ("local", "global")]
        lines.append("| " + " | ".join([LABELS[arm]] + parts) + " |")
    lines += ["", "| 显式给定β条件诊断（GT辅助，非部署／自动上界） | local J0 | global J0 | 逐帧加权RA |",
        "|---|---:|---:|---:|"]
    for arm in ARMS:
        known = [scores["all_recordings_profiles"][f"known_beta/{arm}/{r}"] for r in ("zgz_local", "zgz_global")]
        weighted = sum(v["n"] * v["mpjpe_ra_mm"] for v in known) / sum(v["n"] for v in known)
        lines.append("| " + " | ".join([LABELS[arm]] + [number(v["mpjpe_ra_mm"]) for v in known] + [number(weighted)]) + " |")
    lines += ["",
        f"固定commit50插入query25的数值一致性：{'全部通过' if consistent else '存在未通过，必须查看原报告'}；"
        f"资源压力case失败数：{resources['failed_workloads']}，失败case不以成功前缀填完整吞吐。",
        "",
        "归因按固定控制读取：新前端对同预算S43检验输入转换与完整系统；static对box检验骨段组织；"
        "各ordered对collapse检验时间信息。仅胜冻结起点不足以归因表示，"
        "仅MANO输出约束有用不足以证明骨语义输入必要。报告效应量，不以正负号自创显著性或非劣门。",
        "",
        "给定β只在明确标记的条件诊断中使用；全部absolute J/V、root、真关节腕对齐V、"
        "不同query间隔／局部手指分层／前态扰动结果保留于原固定报告，"
        "不能用J0对齐掩盖绝对发散。没有新S39/W200匹配行，不改S39／S43默认。",
        "",
        "可复用成果包括独立LNES-free前端、共同因果事件核心、显式query/commit与可恢复序列状态、"
        "精确MANO接口、原生S43公平成本对照及来源／预测隔离。"
        "本轮不自动认证Nature级原创性；若简单控制解释收益，应删去相应专用表示主张。",
        "",
        "论文贡献应围绕‘核心原理—作用机制—实际效应’组织：将可重解释的事件记忆与手态查询分离，"
        "在同训练输出预算下区分计数场、骨段组织和时间顺序的作用，并明确其可观测性与流式成本边界。"
        "基础计数场、ROI及MANO FK均有直接先例；未获独立证据的机制不能写成已成立的新方法贡献。",
        "",
        "完整机器报告、执行命令、权重及封存身份以本节产物链接为准。"]
    body = "\n".join(lines)
    (report / "main_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    (report / "postfinal_summary.md").write_text(body + "\n", encoding="utf-8")
    if write_docs:
        relative = "../" + str((report / "postfinal_summary.json").relative_to(ROOT))
        fixed = "../" + str((report / "results.json").relative_to(ROOT))
        links = f"\n\n[完整封存汇总]({relative}) · [120条固定条件与诊断]({fixed})。"
        replace_block(ROOT / "docs/EXPERIMENT_RECORDS.md", "S43_FULL_FIELD_FINAL", body + links)
        # 2026-10-28: docs/S43表征调研.md was merged into the research document as 附录 N
        # (headings one level deeper, same markers); the block is written there.
        research = ('<a id="s43-part-13"></a>\n### 第十三编｜完整前端结果归档\n\n'
            "完整训练、120条回放与资源测量已结束，实际数值见"
            "[实验账本](EXPERIMENT_RECORDS.md#s43-field-full-results-20261028)。"
            "第十二编的‘正在运行’是历史时点；原子阶段负门不因后续结果撤销。\n\n"
            + "\n".join(lines[4:]) + links)
        replace_block(ROOT / "docs/网络结构与可观测性诊断研究.md", "S43_FULL_FIELD_FINAL", research)
        # Current navigation is updated while the historical sections remain.
        documents = (ROOT / "docs/EXPERIMENT_RECORDS.md", ROOT / "docs/网络结构与可观测性诊断研究.md")
        prior = report / "document_navigation_before_first_update.json"
        if not prior.exists():
            dump(prior, {str(p): p.read_text(encoding="utf-8").splitlines()[:28] for p in documents})
        for path in documents:
            text = path.read_text(encoding="utf-8")
            rows = text.splitlines()
            for i, row in enumerate(rows):
                if row.startswith("**本轮进展（2026-10-28，S43第三轮）**"):
                    rows[i] = ("**本轮进展（2026-10-28，S43第三轮）**：归因、三案与交叉审查、原子实验及五臂完整预算训练／120条回放／资源测量均已有封存产物，见"
                        "[完整前端结果](#s43-field-full-results-20261028)。原子负门保留；单seed描述性结果不认证Nature级原创，不自动替换S43／S39。")
                elif row.startswith("- **本轮后续（2026-10-28）**"):
                    rows[i] = ("- **本轮后续（2026-10-28）**：[完整前端结果归档](#s43-part-13)已由封存产物更新。前十二编保留归因、设计、原子负结果及运行时点，不能用历史进行中状态覆盖最终数值。")
            text = "\n".join(rows) + "\n"
            if path.name == "网络结构与可观测性诊断研究.md":
                text = text.replace("| 第十二编（最新） |", "| 第十二编 |")
                if "| 第十三编（最新） |" not in text:
                    text = text.replace("| 第十二编 |", "| 第十三编（最新） | 完整前端固定预算结果与归因 | [阅读](#s43-part-13) |\n| 第十二编 |", 1)
            temporary = path.with_name(path.name + ".s43_report_tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
    # This COMPLETE artifact exists only after formatting and optional writes succeed.
    dump(report / "postfinal_summary.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, default=BASE / "s43_full_field_R1")
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--write-docs", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.status_only:
        value = status(args.train_root, args.eval_root)
    else:
        value = assemble(args.train_root, args.eval_root, write_docs=args.write_docs)
        value = dict(status=value["status"], extra_query_consistency_pass=value["extra_query_consistency_pass"],
            source_sha256=value["source_sha256"], default_configuration_changed=False)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
