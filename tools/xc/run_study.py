#!/usr/bin/env python3
"""Prepare, freeze, and execute the isolated XYZ geometry x causal-time study.

`prepare` only writes configurations and a draft.  A separate explicit `freeze`
requires the final preregistration and a source-matched passing preflight.  Every
GPU launch rechecks frozen identities.  Only exact step6000 checkpoints are
scored; there is no checkpoint selection or accuracy-dependent queue skipping.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/xc"
CONFIG_DIR = REPO / "configs/xc"
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
GPUS = (4, 5, 6, 7)
SEEDS = (3407, 3408, 3409)
ARMS = {"g0t0": (False, "static"), "g1t0": (True, "static"),
        "g0t1": (False, "extrapolate4"), "g1t1": (True, "extrapolate4")}
SECTION = "XYZ-GEOMETRY-CAUSAL-TIME-20261012"
MARKER = "xyz-geometry-causal-time-20261012"
STEPS = 6000
EFFECTIVE_BATCH = 1024
EVENT_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    def invalid_constant(value):
        raise ValueError(f"non-standard/nonfinite JSON constant {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=invalid_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def event(**value):
    with EVENT_LOCK:
        OUT.mkdir(parents=True, exist_ok=True)
        row = {"time": stamp(), **value}
        with (OUT / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)


def prereg_text():
    source = (REPO / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = (f"<!-- EXPERIMENT_RECORD:{SECTION}:{kind} -->" for kind in ("BEGIN", "END"))
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("final XC preregistration is missing or ambiguous; a draft cannot freeze")
    return source.split(begin, 1)[1].split(end, 1)[0]


def config_for(arm, seed, repeat=False):
    geometry, time_mode = ARMS[arm]
    cfg = yaml.safe_load((REPO / "configs/xa/xa_c0_s3407.yaml").read_text())
    run_name = f"xc_{arm}{'r' if repeat else ''}_s{seed}"
    cfg["SEED"] = seed
    # DATA records the requested contract; TRACK is the current dataset's actual
    # pair switch.  The isolated subclass uses the pair to estimate history and
    # does not replace main.prev (UNROLL_P=0).
    cfg["DATA"].update(UNROLL_PAIR=True,
                       SPLITS_MANIFEST=str(REPO / "data/hand_data51/splits_semkine.json"))
    cfg["TRACK"].update(UNROLL_PAIR=True, UNROLL_P=0.0)
    cfg["MODEL"].update(XYZ_GEOM=geometry, XYZ_RESIDUAL_MODE="legacy",
                         XYZ_CAUSAL_STUDY=True, XYZ_CAUSAL_TIME_MODE=time_mode)
    cfg["MODEL"].pop("INIT_FROM", None)
    cfg["TRAIN"].update(DEVICES=1, STRATEGY="auto", PRECISION="bf16",
                         BATCH_SIZE_PER_GPU=512, ACCUMULATE_GRAD_BATCHES=2,
                         MAX_STEPS=STEPS, LR=4.0e-3, WARMUP_STEPS=500,
                         LR_SCHEDULE="constant", OPTIMIZER="adam",
                         RUN_NAME=run_name, OUTPUT_DIR=f"outputs/semkine/{run_name}")
    cfg["TRAIN"].pop("TRAINABLE_PREFIXES", None)
    cfg["TRAIN"].pop("RESUME_FROM", None)
    cfg["EVAL"].update(WINDOW_MS=50, OUTPUT_DIR=f"outputs/semkine/{run_name}/eval")
    return run_name, cfg


def jobs():
    result = []
    for seed in SEEDS:
        for arm, (geometry, time_mode) in ARMS.items():
            name, _ = config_for(arm, seed)
            result.append({"run": name, "arm": arm, "geometry": geometry,
                           "time_mode": time_mode, "seed": seed, "repeat": False,
                           "phase": "full", "max_steps": STEPS,
                           "effective_batch": EFFECTIVE_BATCH,
                           "config": f"configs/xc/{name}.yaml"})
    for arm, (geometry, time_mode) in ARMS.items():
        name, _ = config_for(arm, 3407, repeat=True)
        result.append({"run": name, "arm": arm, "geometry": geometry,
                       "time_mode": time_mode, "seed": 3407, "repeat": True,
                       "phase": "full", "max_steps": STEPS,
                       "effective_batch": EFFECTIVE_BATCH,
                       "config": f"configs/xc/{name}.yaml"})
    return result


def draft_text():
    return f"""<!-- EXPERIMENT_RECORD:{SECTION}:BEGIN -->
<a id="{MARKER}"></a>

### [XC] 显式3D通道 × 可部署事件时刻关联：预注册草稿（2026-10-12）

本文件是尚待root审定的草稿，不是冻结许可；prepare不启动GPU。

0. 范围：用户授权物理GPU4–7四卡、全部CPU执行四组。严格隔离显式3D整组G与事件时刻关联T，不以ABCD剩余摘要结果替代两机制验证。旧S39默认/部署权重、旧XA/XB产物与历史预注册不修改。现行唯一只读划分为data/hand_data51/splits_semkine.json：九人72训练序列，zgz_global/local仍同时开发与测试。

1. 四组：xc_g0t0（G关闭、静止旧网格）、xc_g1t0（G开启、静止旧网格）、xc_g0t1（G关闭、四片预测网格）、xc_g1t1（G开启、四片预测网格）。G只控制原dXYZ/z_prior/edge整组，不能据此确定单个通道的因果贡献。原17槽及legacy残差不变；不引入新局部残差支路。

2. 公共历史：四组都取得相同raw leading/main配对窗口，DATA.UNROLL_PAIR=true并由TRACK.UNROLL_PAIR=true启用，UNROLL_P=0不替换main.prev。训练leading.prev仍由原课程GT+噪声生成，但其运动终点只由本模型对过去leading事件的预测得到，不读取leading.target、main.target或未来GT；部署历史完全由过去自身预测产生。明确保留这种训练初值与部署历史的分布差，不能称训练全程自生成前态。leading用no_grad且固定BN为eval，始终使用静态零历史fallback。由leading.prev→leading.pred的51维物理速度（平移米、16组SO3增量弧度）除以leading实际(t_end_us−t_start_us)×1e-6跨度，按当前查询终点相对incoming状态的实际跨度缩放，得到共同motion_delta51；四组以相同维数拼接读出。因此不能把多读一段过去历史的收益归于T。

3. T：T0即使有历史仍静止关联；T1从main.prev锚定，以当前真实跨度的.125/.375/.625/.875中心外推四个状态，事件仅按真实相对时刻分配到四片，各片FK/关联以及匹配记录的dXYZ/z采用该片预测状态；mesh边几何统一用incoming anchor，不构造四张持续动态mesh。训练main.prev在窗首，时间offset=0；部署首包incoming状态在段首，offset=0；续包incoming状态是前包末的自身预测，与新包首差1ms，offset=.001s。片中心的物理外推时刻是offset+center*packet.delta_t_s；共同motion_delta读出horizon为packet.delta_t_s−.001s+offset。部署初包历史端点间隔49ms、后续50ms，速度按显式标记的状态时刻计算。eval外层state_hook负责reset/set_history，不从packet内的局部0/W或GT猜历史。时间token、W50评估窗、采样最多2048、极性、数量、查询频率保持相同。无未来GT拟合、无未来网络输出回填。无历史或零速度必须回退到静态几何，预先验收其等价误差。该试验验证按事件时刻预测几何关联，不称持久流式顶点图已实现；异步执行与模型精度是不同命题。

4. 训练：四组seed3407/3408/3409；每个条件另有同配置同seed3407从零独立复跑（xc_g0t0r/g1t0r/g0t1r/g1t1r_s3407），不添加第五种机制条件。共16次完整训练。所有参数从零训练，不载入C0/S39/其他检查点，不冻结参数，不从部分run隐式续训。每次精确6000optimizer steps，单GPU512×累加2=EBS1024、bf16、Adam4e-3、warmup500、恒定LR，损失/增广/前态噪声继承XA C0。四组共同在FK/SO3/几何部分禁用autocast并用FP32，不称与旧XA bf16几何逐位等价。公共leading推理成本四组一致。总96000optimizer steps、98304000配对样本；若每样本leading/main各一次前向，合计196608000窗口前向样本，只有main监督反传。sample预算与事件数不是同一单位。训练随机窗口W30–300ms，四片中点离事件最远3.75–37.5ms；正式W50最远6.25ms，量化误差不能称连续事件时刻精确FK。当前stride采样身份依赖整包最终事件数，最多2048，全包批式前向；该实现不是前缀不变增量图，也不证明每个新事件只更新局部MANO状态。

5. 冻结与合同：先通过默认S39逐位保持、四组等参数/初始化、前三raw批次逐seed配对、过去历史因果性、T片映射与真实时间单位、零速度fallback、端点及有限性preflight；再将源码/config/split/预注册摘要冻结。任何冻结输入SHA漂移拒绝launch/score。初始三批审计不宣称整个训练流逐位相同。物理GPU4–7一任务一卡，动态空闲队列。精确末点step6000必需，保存grid仅恢复/审计，不选择中间点；缺末点/重复末点/非有限保持失败并阻止正常均值，不以较早点或有限子集替换。

6. 主评价：各模型自身预测推进下一步，不借其它模型或GT轨迹构成历史；rawW50、相同有效段、原段首初始化噪声、rng0、zgz两序列。每个run训练子进程退出后，在该同一卡独占完成exactstep6000单seed主行（outputs/xc/individual_rows），然后才释放该卡给下一训练。原非有限值写作null/FAIL或原失败输出保留；不删失败seed，不对有限子集求均值。三seed保留每seed与overall/local/global，正式组主行由已验证的单seed行算术聚合accuracy，不重复完整闭环；现有兼容two_seed_mean明确存三seed均值，cost按canonical固定取seed3407单行。各源行/检查点/配置/冻结源码SHA保留，任何hash或协议不匹配拒绝合并。用户主表只用规定九列，通过report_table.py读取新的main_row_last.json生成；S39/C0只作跨配方参考。

7. 不随结果改变的门（草稿，须root在结果前审定）：每个注册对比c、每列q用相应条件各自seed3407独立repeat构成完整对比，δcq=|contrast_repeat3407−contrast_primary3407|，τcq=max(1.1mm,2δcq)；不拿g0t0的一个差值代替其余条件或交互噪声。每组还单列自身repeat差，只有一个复跑差值，均是工程噪声门，不是置信区间/显著性。G在每个T层分别比较E0t−E1t；T在每个G层分别比较Eg0−Eg1；所有四个条件差均完整报告，不依据成绩选择解释层。某条件对照的效用门要求三个seedoverall差全>0、overall均值>该contrast的τoverall、local/global平均退化不超过该contrast的各τ。组合作用比较E00−E11，用自身contrast repeat门；交互增益I=E10+E01−E00−E11逐seed报告，交互噪声用完整四repeat的I_r与primary seed3407的I之差，不能从两个主效应均值自动推断协同。G仅某T层成功或T仅某G层成功，只允许条件性结论；未超过门称未检出，不能称证明无因果贡献。若T只有GT轨迹诊断有利或只有更长窗/更多事件有利，不称可部署异步收益。若G有利不能称独立观测深度。

8. 机制/恢复护栏（root补齐独立诊断具体实现与固定范围后冻结）：自然错误前态和大共同初值的真时间推进、正常动作响应、真实时间破坏控制以及零速度/无历史fallback需保留所有帧/种子。未执行只能标未验证，不由主表推断通过。调度收益需同一函数批式参考/增量执行等价验收，计FK、四片关联、等待/写入/查询/缓存失效与leading额外前向全部成本；本轮未实现增量执行时明确不作该主张。

9. 原始产物：configs/xc、tools/xc/train.py、tools/xc/make_row.py、semkine/xyz_time_study.py与测试；outputs/xc/study_manifest.json/frozen_manifest.json/preflight.json/events.jsonl/status.json；outputs/semkine/xc_*_s*/精确检查点、training_metadata.json、pair_stream_rank0.json。此草稿不回写旧docs，不冻结，不启动。
<!-- EXPERIMENT_RECORD:{SECTION}:END -->
"""


def prepare():
    if (OUT / "frozen_manifest.json").exists():
        raise RuntimeError("XC is frozen; refuse config/draft regeneration")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    job_list = jobs()
    for job in job_list:
        name, cfg = config_for(job["arm"], job["seed"], job["repeat"])
        path = REPO / job["config"]
        path.write_text(f"# [{SECTION}] {name}; draft, no launch\n"
                        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        job["config_sha256"] = sha(path)
    manifest = {"study": SECTION, "created_at": stamp(), "status": "draft_no_gpu_launch",
                "allowed_physical_gpus": list(GPUS), "python": PYTHON,
                "source_config": "configs/xa/xa_c0_s3407.yaml", "jobs": job_list,
                "primary_checkpoint": "exact step6000, no selection",
                "train_entry": "tools/xc/train.py", "eval_entry": "tools/xc/make_row.py",
                "common_history": {"source": "own leading packet prediction",
                                   "motion_delta_dim": 51, "leading_grad": False,
                                   "leading_bn": "eval", "unroll_replace_probability": 0.0},
                "time_slice_centers_fraction": [0.125, 0.375, 0.625, 0.875],
                "evaluation": {"window_ms": 50, "split": "val_core", "subject": "zgz",
                               "manifest": "data/hand_data51/splits_semkine.json", "rng_seed": 0},
                "total_optimizer_steps": STEPS * len(job_list),
                "total_pair_samples": STEPS * EFFECTIVE_BATCH * len(job_list),
                "total_leading_plus_main_window_samples": 2 * STEPS * EFFECTIVE_BATCH * len(job_list),
                "scope_limit": "G tests the original three-channel bundle; T tests four-slice causal association, not incremental execution"}
    write_json(OUT / "study_manifest.json", manifest)
    (OUT / "prereg_draft.txt").write_text(draft_text())
    lines = ["# XC queue (draft; no GPU task launched)", "",
             "16 complete from-scratch runs in four conditions; physical GPUs 4–7, one task/card; EBS1024.", "",
             "| Run | G | T | Seed | Steps | Config SHA256 |", "|---|---|---|---|---|---|"]
    lines.extend(f"| {j['run']} | {int(j['geometry'])} | {j['time_mode']} | {j['seed']} | {j['max_steps']} | {j['config_sha256']} |" for j in job_list)
    (OUT / "QUEUE.md").write_text("\n".join(lines) + "\n")
    print(f"Prepared {len(job_list)} runs; {manifest['total_optimizer_steps']} steps; no GPU launch.")


def source_files():
    names = ["AGENTS.md", "semkine/xyz_mesh.py", "semkine/xyz_residual.py",
             "semkine/xyz_time_study.py", "model/model.py", "model/mano_layer.py", "model/pose_repr.py",
             "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py",
             "semkine/encoder.py", "semkine/events.py", "semkine/lie.py", "model/config.py",
             "tools/make_s36_row.py", "tools/report_table.py", "tools/xb/diagnostics.py",
             "assets/mano_right.npz", "data/hand_data51/splits_semkine.json",
             "tests/test_xyz_time_study.py", "outputs/xb/c0_val_reference.pt",
             "outputs/xc/clip_selection.json"]
    paths = [REPO / name for name in names]
    paths.extend(sorted((REPO / "tools/xc").glob("*.py")))
    paths.extend(sorted((REPO / "tests").glob("test_xc*.py")))
    # Train/eval entries must exist even if the glob found another helper.
    paths.extend([REPO / "tools/xc/train.py", REPO / "tools/xc/make_row.py"])
    return sorted(set(paths))


def freeze():
    target = OUT / "frozen_manifest.json"
    if target.exists():
        raise RuntimeError("XC frozen manifest exists; historical freeze is immutable")
    manifest = read_json(OUT / "study_manifest.json")
    preflight = read_json(OUT / "preflight.json")
    if preflight.get("status") != "PASS" or preflight.get("failures") != []:
        raise RuntimeError("preflight must have status=PASS and failures=[]")
    paths = source_files() + [REPO / j["config"] for j in manifest["jobs"]]
    absent = [str(path.relative_to(REPO)) for path in paths if not path.is_file()]
    if absent:
        raise RuntimeError(f"missing source/config inputs: {absent}")
    identities = {str(path.relative_to(REPO)): sha(path) for path in paths}
    preflight_sources = preflight.get("source_sha256", {})
    essential = ["semkine/xyz_time_study.py", "tools/xc/train.py", "tools/xc/make_row.py",
                 "tests/test_xyz_time_study.py"]
    for name in essential:
        if preflight_sources.get(name) != identities[name]:
            raise RuntimeError(f"preflight source identity missing/drifted: {name}")
    for name, digest in preflight_sources.items():
        if not (REPO / name).is_file() or sha(REPO / name) != digest:
            raise RuntimeError(f"preflight source changed after PASS: {name}")
    for job in manifest["jobs"]:
        if identities[job["config"]] != job["config_sha256"]:
            raise RuntimeError(f"prepared config changed: {job['config']}")
    rules = prereg_text()
    manifest.update(status="frozen_before_gpu_training", frozen_at=stamp(),
                    files_sha256=identities, prereg_text=rules,
                    prereg_sha256=hashlib.sha256(rules.encode()).hexdigest(),
                    preflight_sha256=sha(OUT / "preflight.json"))
    write_json(target, manifest)
    event(action="freeze", files=len(identities), frozen_manifest_sha256=sha(target))


def frozen():
    manifest = read_json(OUT / "frozen_manifest.json")
    if manifest.get("study") != SECTION or manifest.get("status") != "frozen_before_gpu_training":
        raise RuntimeError("invalid XC frozen manifest")
    bad = [name for name, digest in manifest["files_sha256"].items()
           if not (REPO / name).is_file() or sha(REPO / name) != digest]
    if sha(OUT / "preflight.json") != manifest["preflight_sha256"]:
        bad.append("outputs/xc/preflight.json")
    if hashlib.sha256(prereg_text().encode()).hexdigest() != manifest["prereg_sha256"]:
        bad.append("XC preregistration section")
    if bad:
        raise RuntimeError(f"frozen inputs changed: {bad}; no launch/score permitted")
    return manifest


def final_checkpoint(job):
    folder = REPO / f"outputs/semkine/{job['run']}"
    matches = list(folder.glob(f"{job['run']}-step={STEPS}.ckpt"))
    other_matches = list(folder.glob(f"*-step={STEPS}.ckpt"))
    if len(matches) != len(other_matches) or len(matches) > 1:
        raise RuntimeError(f"ambiguous/wrong exact endpoint for {job['run']}: {other_matches}")
    return matches[0] if matches else None


def audit_checkpoint(job):
    import torch
    path = final_checkpoint(job)
    if path is None:
        raise RuntimeError(f"missing exact step6000: {job['run']}")
    checkpoint = torch.load(path, map_location="cpu")
    if int(checkpoint.get("global_step", -1)) != STEPS:
        raise RuntimeError(f"{job['run']} actual checkpoint step is not {STEPS}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{job['run']} checkpoint lacks state_dict")
    nonfinite = [name for name, tensor in state.items() if torch.is_tensor(tensor)
                 and (tensor.is_floating_point() or tensor.is_complex())
                 and not bool(torch.isfinite(tensor).all())]
    if nonfinite:
        raise RuntimeError(f"{job['run']} nonfinite checkpoint tensors: {nonfinite}")
    return {"checkpoint": str(path.relative_to(REPO)), "checkpoint_sha256": sha(path),
            "global_step": STEPS, "state_tensors": len(state), "finite": True}


def check_run_metadata(job):
    manifest = frozen()
    meta = read_json(REPO / f"outputs/semkine/{job['run']}/training_metadata.json")
    expected = {"seed": job["seed"], "max_steps": STEPS, "devices": 1,
                "batch_size_per_gpu": 512, "accumulate_grad_batches": 2,
                "effective_batch": EFFECTIVE_BATCH, "lr": 0.004,
                "lr_schedule": "constant", "precision": "bf16",
                "data_generator_seed": job["seed"], "resumed_from": None,
                "study": SECTION, "xyz_geometry": job["geometry"],
                "xyz_time_mode": job["time_mode"], "motion_delta_dim": 51,
                "frozen_manifest_sha256": sha(OUT / "frozen_manifest.json")}
    for key, value in expected.items():
        if meta.get(key) != value or (value is None and key not in meta):
            raise RuntimeError(f"{job['run']} metadata {key}: {meta.get(key)!r} != {value!r}")
    if meta.get("init_from") is not None:
        raise RuntimeError(f"{job['run']} must train from scratch, INIT_FROM found")
    visible = str(meta.get("cuda_visible_devices", ""))
    if visible not in tuple(map(str, GPUS)):
        raise RuntimeError(f"{job['run']} unauthorized/multiple CUDA_VISIBLE_DEVICES={visible}")
    subjects = ["ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"]
    if sorted(meta.get("train_subjects", [])) != subjects:
        raise RuntimeError(f"{job['run']} wrong training subjects")
    sequences = meta.get("train_sequences", [])
    if len(sequences) != 72 or len(set(sequences)) != 72:
        raise RuntimeError(f"{job['run']} wrong/duplicate training sequences")
    if sorted(meta.get("val_core_sequences", [])) != ["zgz_global", "zgz_local"]:
        raise RuntimeError(f"{job['run']} wrong validation sequences")
    if meta.get("provenance", {}).get("config_sha256") != sha(REPO / job["config"]):
        raise RuntimeError(f"{job['run']} config hash mismatch")
    code = meta.get("source_sha256", {})
    for name in ("semkine/xyz_time_study.py", "tools/xc/train.py", "semkine/dataset.py", "model/model.py"):
        if code.get(name) != manifest["files_sha256"][name]:
            raise RuntimeError(f"{job['run']} metadata source SHA mismatch/missing: {name}")
    return meta


def job_env(gpu):
    if gpu not in GPUS:
        raise RuntimeError(f"GPU {gpu} is outside authorized 4–7")
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NCCL_P2P_DISABLE": "1"}


def train_one(job, gpu):
    frozen()
    name = job["run"]
    if final_checkpoint(job):
        check_run_metadata(job)
        audit = audit_checkpoint(job)
        event(action="reuse_completed", run=name, gpu=gpu, **audit)
        score_one(job, gpu, audit)
        return True
    run_dir = REPO / f"outputs/semkine/{name}"
    if run_dir.exists() and (any(run_dir.iterdir())):
        raise RuntimeError(f"partial/nonempty existing {name}; no implicit rerun/resume")
    log = OUT / f"logs/{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "tools/xc/train.py", "--config", job["config"], "--seed", str(job["seed"]),
               "--run-name", name, "--output-dir", f"outputs/semkine/{name}"]
    event(action="train_start", run=name, gpu=gpu, command=command, log=str(log.relative_to(REPO)))
    status()
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, env=job_env(gpu), stdout=handle, stderr=subprocess.STDOUT)
    okay = result.returncode == 0 and final_checkpoint(job) is not None
    audit = None
    if okay:
        check_run_metadata(job)
        audit = audit_checkpoint(job)
    event(action="train_finish", run=name, gpu=gpu, returncode=result.returncode,
          exact_last=okay, audit=audit)
    status()
    if okay:
        score_one(job, gpu, audit)
    return okay


def run():
    manifest = frozen()
    job_list = manifest["jobs"]
    failures = []
    status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        remaining = iter(job_list)
        pending = {}
        for gpu in GPUS:
            job = next(remaining, None)
            if job:
                pending[pool.submit(train_one, job, gpu)] = (job, gpu)
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job, gpu = pending.pop(future)
                try:
                    if not future.result():
                        failures.append(job["run"])
                except Exception as error:
                    failures.append(job["run"])
                    event(action="job_exception", run=job["run"], error=str(error))
                # Launch-time freeze checks still apply; an accuracy/numeric
                # failure never chooses a different seed or skips another arm.
                next_job = next(remaining, None)
                if next_job:
                    pending[pool.submit(train_one, next_job, gpu)] = (next_job, gpu)
                status()
    if failures:
        raise RuntimeError(f"incomplete/invalid XC runs: {failures}; no scientific means")
    event(action="phase_train_done", phase="full")
    check_streams()


def check_streams():
    manifest = frozen()
    summary = []
    for seed in SEEDS:
        subset = [job for job in manifest["jobs"] if job["seed"] == seed]
        streams = {}
        for job in subset:
            check_run_metadata(job)
            records = read_json(REPO / f"outputs/semkine/{job['run']}/pair_stream_rank0.json")
            if not isinstance(records, list) or len(records) != 3:
                raise RuntimeError(f"{job['run']} requires exactly initial three raw batch records")
            values = [row.get("sha256") for row in records]
            if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
                raise RuntimeError(f"{job['run']} invalid raw batch SHA")
            streams[job["run"]] = values
        paired = len(streams) == (8 if seed == 3407 else 4) and len({tuple(v) for v in streams.values()}) == 1
        summary.append({"seed": seed, "streams": streams, "initial_three_batches_identical": paired})
    write_json(OUT / "paired_streams_full.json", {"scope": "initial three raw leading/main batches; not a full training-stream equality proof", "groups": summary})
    if not all(group["initial_three_batches_identical"] for group in summary):
        raise RuntimeError("XC initial raw batch pairing failed")
    print("XC three-seed initial raw batch pairing PASS.")


def row_tasks():
    return ([(f"xc_{arm}", SEEDS) for arm in ARMS]
            + [(f"xc_{arm}r", (3407,)) for arm in ARMS])


def numeric_failures(value, path="root"):
    result = []
    if isinstance(value, float) and not math.isfinite(value):
        result.append(path)
    elif isinstance(value, dict):
        if value.get("numeric_valid") is False:
            result.append(path + ".numeric_valid")
        for key, item in value.items():
            result.extend(numeric_failures(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(numeric_failures(item, f"{path}[{index}]"))
    return result


def validate_row(row, name, seeds):
    if numeric_failures(row):
        raise RuntimeError(f"{name} nonfinite/invalid row: {numeric_failures(row)[:20]}")
    actual = row.get("per_seed", {})
    if set(actual) != {str(seed) for seed in seeds}:
        raise RuntimeError(f"{name} omitted/unregistered seeds in main row")
    for seed, item in actual.items():
        if item.get("step") != STEPS:
            raise RuntimeError(f"{name}/{seed} row is not exact step{STEPS}")
        for scope in ("overall", "local", "global"):
            for metric in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm", "mpvpe_abs_mm"):
                score = item.get(scope, {}).get(metric)
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise RuntimeError(f"{name}/{seed}/{scope}/{metric}: missing/null/nonfinite")


def individual_row_path(job):
    return OUT / f"individual_rows/{job['run']}_main_row_last.json"


def score_one(job, gpu, checkpoint_audit=None):
    """Score the endpoint before releasing its GPU; cache is journal/hash audited."""
    manifest = frozen()
    check_run_metadata(job)
    audit = checkpoint_audit or audit_checkpoint(job)
    path = individual_row_path(job)
    if path.is_file():
        validate_cached_row(job, manifest, audit)
        event(action="reuse_individual_row", run=job["run"], gpu=gpu, path=str(path.relative_to(REPO)), sha256=sha(path))
        return
    name = f"xc_{job['arm']}{'r' if job['repeat'] else ''}"
    seeds = (job["seed"],)
    log = OUT / f"logs/{job['run']}_row.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "tools/xc/make_row.py", "--run", name,
               "--seeds", str(job["seed"]), "--step", "last", "--output", str(path)]
    event(action="individual_row_start", run=job["run"], gpu=gpu, command=command)
    status()
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, env=job_env(gpu), stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not path.is_file():
        event(action="individual_row_failed", run=job["run"], gpu=gpu,
              returncode=result.returncode, raw_output_exists=path.is_file())
        status()
        raise RuntimeError(f"XC individual row failed {job['run']}; raw result/log preserved at {log}")
    row = read_json(path)
    validate_row(row, job["run"], seeds)
    validate_cost(row, job["run"])
    row.setdefault("protocol", {}).update(n_seeds=len(seeds), effective_batch=EFFECTIVE_BATCH,
                                           raw_window_ms=50, study=SECTION,
                                           exact_step=STEPS,
                                           note=f"seed {job['seed']}, exact last6000, own causal history; no checkpoint selection")
    row["xc_provenance"] = {"run": job["run"], "seed": job["seed"],
                            "geometry": job["geometry"], "time_mode": job["time_mode"],
                            "checkpoint": audit["checkpoint"], "checkpoint_sha256": audit["checkpoint_sha256"],
                            "config_sha256": sha(REPO / job["config"]),
                            "frozen_manifest_sha256": sha(OUT / "frozen_manifest.json"),
                            "source_sha256": manifest["files_sha256"]}
    write_json(path, row)
    event(action="individual_row_finish", run=job["run"], gpu=gpu,
          path=str(path.relative_to(REPO)), sha256=sha(path), checkpoint_sha256=audit["checkpoint_sha256"])
    status()


def validate_cost(row, name):
    for key in ("latency_ms_scaled_full1p75", "latency_ms_raw", "anchor_full_raw_ms",
                "macs_forward_packet", "params_total"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"{name}/{key}: missing/null/nonpositive/nonfinite cost")


def validate_cached_row(job, manifest, checkpoint_audit=None):
    path = individual_row_path(job)
    row = read_json(path)
    validate_row(row, job["run"], (job["seed"],))
    validate_cost(row, job["run"])
    audit = checkpoint_audit or audit_checkpoint(job)
    expected = {"run": job["run"], "seed": job["seed"], "geometry": job["geometry"],
                "time_mode": job["time_mode"], "checkpoint": audit["checkpoint"],
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "config_sha256": sha(REPO / job["config"]),
                "frozen_manifest_sha256": sha(OUT / "frozen_manifest.json"),
                "source_sha256": manifest["files_sha256"]}
    if row.get("xc_provenance") != expected:
        raise RuntimeError(f"{job['run']} cached endpoint/source provenance changed")
    protocol = row.get("protocol", {})
    if protocol.get("study") != SECTION or protocol.get("raw_window_ms") != 50 or protocol.get("n_seeds") != 1:
        raise RuntimeError(f"{job['run']} invalid individual score protocol")
    expected_digest = None
    for line in (OUT / "events.jsonl").read_text().splitlines():
        item = json.loads(line)
        if item.get("action") == "individual_row_finish" and item.get("run") == job["run"]:
            expected_digest = item["sha256"]
    if expected_digest != sha(path):
        raise RuntimeError(f"{job['run']} cached score digest missing/changed; no silent reuse")
    return row


def rows():
    """Aggregate audited endpoints; never rescore an already evaluated trajectory."""
    manifest = frozen()
    check_streams()
    metrics = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
    failures = {}
    for name, seeds in row_tasks():
        cached, inputs = {}, {}
        group_errors = {}
        for seed in seeds:
            job = next(item for item in manifest["jobs"] if item["run"] == f"{name}_s{seed}")
            try:
                check_run_metadata(job)
                cached[str(seed)] = validate_cached_row(job, manifest)
                inputs[str(seed)] = {"path": str(individual_row_path(job).relative_to(REPO)),
                                     "sha256": sha(individual_row_path(job)),
                                     "provenance": cached[str(seed)]["xc_provenance"]}
            except Exception as error:
                group_errors[str(seed)] = str(error)
        if group_errors:
            failures[name] = group_errors
            event(action="row_aggregation_failed", run=name, required_seed_errors=group_errors,
                  no_finite_subset_mean=True)
            continue
        # Canonical cost convention: the first seed supplies cost; accuracy is
        # arithmetic over all required registered seeds, no finite-only means.
        row = copy.deepcopy(cached[str(seeds[0])])
        row["per_seed"] = {str(seed): cached[str(seed)]["per_seed"][str(seed)] for seed in seeds}
        row["two_seed_mean"] = {scope: {metric: sum(row["per_seed"][str(seed)][scope][metric]
                                                  for seed in seeds) / len(seeds)
                                            for metric in metrics}
                                for scope in ("overall", "local", "global")}
        row["xc_provenance"] = {"study": SECTION, "individual_rows": inputs,
                                "cost_seed": seeds[0], "cost_source_sha256": inputs[str(seeds[0])]["sha256"],
                                "frozen_manifest_sha256": sha(OUT / "frozen_manifest.json")}
        row["protocol"].update(n_seeds=len(seeds), effective_batch=EFFECTIVE_BATCH,
                               raw_window_ms=50, study=SECTION, exact_step=STEPS,
                               aggregation="arithmetic accuracy of audited individual rows; cost from seed3407",
                               note=f"{len(seeds)} registered seed(s), exact last6000, own causal history; no selection/rescoring")
        validate_row(row, name, seeds)
        validate_cost(row, name)
        path = REPO / f"outputs/semkine/{name}_main_row_last.json"
        write_json(path, row)
        event(action="row_finish", run=name, path=str(path.relative_to(REPO)), sha256=sha(path),
              individual_sources={seed: item["sha256"] for seed, item in inputs.items()})
    if failures:
        write_json(OUT / "row_failures.json", {"time": stamp(), "required_group_failures": failures,
                                               "no_finite_subset_mean": True})
        raise RuntimeError(f"required individual score failures prevent complete primary aggregation: {list(failures)}")


def validate_diagnostic(seed, path):
    """Require every original trajectory; measured nonfinite values remain FAIL."""
    payload = read_json(path)
    if payload.get("study") != SECTION or payload.get("seed") != seed:
        raise RuntimeError(f"diagnostic study/seed mismatch: {path}")
    if set(payload.get("arms", {})) != set(ARMS):
        raise RuntimeError(f"diagnostic has incomplete arm set: {path}")
    for arm in ARMS:
        rows_for_arm = payload["arms"][arm].get("rows", [])
        if len(rows_for_arm) != 192:
            raise RuntimeError(f"diagnostic {seed}/{arm} lacks all 192 required clips")
        for item in rows_for_arm:
            if len(item.get("mpjpe_ra_mm", [])) != 11 or len(item.get("mpjpe_abs_mm", [])) != 11:
                raise RuntimeError(f"diagnostic {seed}/{arm} omitted frame states")
    # numeric_valid=False is a measured scientific failure, never an excuse to
    # discard this seed or rerun until it happens to be finite.
    return payload


def diagnostic_one(seed, gpu):
    manifest = frozen()
    endpoint_errors = {}
    for arm in ARMS:
        job = next(item for item in manifest["jobs"] if item["run"] == f"xc_{arm}_s{seed}")
        try:
            check_run_metadata(job)
            audit_checkpoint(job)
        except Exception as error:
            endpoint_errors[arm] = str(error)
    if endpoint_errors:
        path = OUT / f"diagnostic_endpoint_failure_s{seed}.json"
        write_json(path, {"time": stamp(), "seed": seed, "status": "FAIL",
                          "required_endpoint_failures": endpoint_errors,
                          "reason": "four finite valid exact-step6000 endpoints required; no earlier substitution"})
        event(action="diagnostics_endpoint_failure", run=f"diagnostics_s{seed}", gpu=gpu,
              seed=seed, failures=endpoint_errors)
        raise RuntimeError(f"seed{seed} cannot replay all four valid exact endpoints: {endpoint_errors}")
    path = OUT / f"recovery_s{seed}.json"
    if path.is_file():
        validate_diagnostic(seed, path)
        event(action="reuse_diagnostics", run=f"diagnostics_s{seed}", gpu=gpu,
              seed=seed, path=str(path.relative_to(REPO)), sha256=sha(path))
        return
    log = OUT / f"logs/recovery_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "tools/xc/diagnostics.py", "run", "--seed", str(seed)]
    event(action="diagnostics_start", run=f"diagnostics_s{seed}", gpu=gpu, seed=seed,
          command=command, log=str(log.relative_to(REPO)))
    env = job_env(gpu)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not path.is_file():
        event(action="diagnostics_failed", run=f"diagnostics_s{seed}", gpu=gpu,
              seed=seed, returncode=result.returncode, raw_output_exists=path.is_file())
        raise RuntimeError(f"seed{seed} diagnostics failed; partial output and log preserved: {log}")
    payload = validate_diagnostic(seed, path)
    event(action="diagnostics_finish", run=f"diagnostics_s{seed}", gpu=gpu, seed=seed,
          path=str(path.relative_to(REPO)), sha256=sha(path), numeric_valid=payload.get("numeric_valid"))


def diagnostics():
    frozen()
    failures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SEEDS)) as pool:
        work = {pool.submit(diagnostic_one, seed, GPUS[index]): seed for index, seed in enumerate(SEEDS)}
        for future in concurrent.futures.as_completed(work):
            seed = work[future]
            try:
                future.result()
            except Exception as error:
                failures[str(seed)] = str(error)
    if failures:
        write_json(OUT / "diagnostic_execution_failures.json", {"time": stamp(), "failures": failures,
                                                                "no_omitted_seed_claim": True})
        raise RuntimeError(f"required diagnostic execution failures: {failures}")


def finalize():
    frozen()
    log = OUT / "logs/finalize.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "tools/xc/finalize.py"]
    event(action="finalize_start", command=command)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode or not (OUT / "verdict.json").is_file():
        raise RuntimeError(f"XC finalization failed; see {log}")
    verdict = read_json(OUT / "verdict.json")
    event(action="finalize_finish", verdict_sha256=sha(OUT / "verdict.json"),
          status=verdict.get("status"), decision=verdict.get("decision"))


def primary_failure_record(errors):
    manifest = frozen()
    endpoints = []
    for job in manifest["jobs"]:
        path = individual_row_path(job)
        record = {"run": job["run"], "raw_individual_score": str(path.relative_to(REPO)),
                  "raw_individual_score_exists": path.is_file(),
                  "raw_individual_score_sha256": sha(path) if path.is_file() else None,
                  "training_log": f"outputs/xc/logs/{job['run']}.log",
                  "scoring_log": f"outputs/xc/logs/{job['run']}_row.log"}
        raw_accuracy = Path(str(path) + ".raw_accuracy.json")
        record.update(raw_accuracy=str(raw_accuracy.relative_to(REPO)), raw_accuracy_exists=raw_accuracy.is_file(),
                      raw_accuracy_sha256=sha(raw_accuracy) if raw_accuracy.is_file() else None)
        relevant_events = []
        journal = OUT / "events.jsonl"
        if journal.is_file():
            for line in journal.read_text().splitlines():
                item = json.loads(line)
                if item.get("run") == job["run"]:
                    relevant_events.append(item)
        record["events"] = relevant_events
        try:
            check_run_metadata(job)
            record.update(exact_checkpoint_audit=audit_checkpoint(job))
            validate_cached_row(job, manifest, record["exact_checkpoint_audit"])
            record["individual_score_status"] = "PASS"
        except Exception as error:
            record.update(individual_score_status="FAIL", reason=str(error))
        endpoints.append(record)
    write_json(OUT / "primary_failures.json", {"time": stamp(), "status": "FAIL",
                                               "stage_errors": errors, "required_runs": endpoints,
                                               "no_finite_subset_mean": True,
                                               "original_failure_outputs_preserved": True})


def report():
    frozen()
    labels = {"g0t0": "G0T0：零显式3D，静止关联，共同过去历史",
              "g1t0": "G1T0：原3D，静止关联，共同过去历史",
              "g0t1": "G0T1：零显式3D，四片因果关联",
              "g1t1": "G1T1：原3D，四片因果关联"}
    names = [task[0] for task in row_tasks()]
    for name, seeds in row_tasks():
        validate_row(read_json(REPO / f"outputs/semkine/{name}_main_row_last.json"), name, seeds)
    command = [PYTHON, "tools/report_table.py", "--tag", "last", "xa_c0", *names,
               "--label", "xa_c0=旧C0（两seed、无新历史读出，仅参考）"]
    for arm, label in labels.items():
        command.extend(["--label", f"xc_{arm}={label}"])
    for arm in ARMS:
        command.extend(["--label", f"xc_{arm}r={arm.upper()}：同种子独立复跑"])
    result = subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)
    reference = subprocess.run([PYTHON, "tools/report_table.py", "--no-baseline", "dt3_dz_l3_w200cfg",
                                "--label", "dt3_dz_l3_w200cfg=S39：W200完整栈（参考）"],
                               cwd=REPO, check=True, capture_output=True, text=True)
    table = result.stdout.rstrip() + "\n" + "\n".join(reference.stdout.splitlines()[2:]) + "\n"
    (OUT / "TABLE_full.md").write_text(table)
    print(table, end="")


def gate_summary():
    manifest = frozen()
    data = {}
    for name, seeds in row_tasks():
        row = read_json(REPO / f"outputs/semkine/{name}_main_row_last.json")
        validate_row(row, name, seeds)
        data[name.removeprefix("xc_")] = row
    scopes = ("overall", "local", "global")
    def error(arm, seed, scope):
        return data[arm]["per_seed"][str(seed)][scope]["mpjpe_ra_mm"]
    arm_noise = {arm: {scope: abs(error(arm + "r", 3407, scope) - error(arm, 3407, scope))
                      for scope in scopes} for arm in ARMS}
    pairs = {"G_at_T0": ("g0t0", "g1t0"), "G_at_T1": ("g0t1", "g1t1"),
             "T_at_G0": ("g0t0", "g0t1"), "T_at_G1": ("g1t0", "g1t1"),
             "joint_vs_G0T0": ("g0t0", "g1t1")}
    contrasts = {}
    for name, (baseline, candidate) in pairs.items():
        values = {scope: [error(baseline, seed, scope) - error(candidate, seed, scope) for seed in SEEDS]
                  for scope in scopes}
        means = {scope: sum(values[scope]) / len(SEEDS) for scope in scopes}
        repeat = {scope: error(baseline + "r", 3407, scope) - error(candidate + "r", 3407, scope)
                  for scope in scopes}
        noise = {scope: abs(repeat[scope] - values[scope][0]) for scope in scopes}
        tau = {scope: max(1.1, 2 * noise[scope]) for scope in scopes}
        checks = {"all_seeds_overall_positive": all(value > 0 for value in values["overall"]),
                  "overall_mean_above_noise": means["overall"] > tau["overall"],
                  "local_mean_degradation_within_noise": means["local"] >= -tau["local"],
                  "global_mean_degradation_within_noise": means["global"] >= -tau["global"]}
        contrasts[name] = {"baseline": baseline, "candidate": candidate, "gain_mm": values,
                           "mean_gain_mm": means, "repeat_gain_mm": repeat,
                           "contrast_repeat_abs_difference_mm": noise, "threshold_mm": tau,
                           "checks": checks, "efficacy_pass": all(checks.values())}
    interaction = {scope: [error("g1t0", seed, scope) + error("g0t1", seed, scope)
                           - error("g0t0", seed, scope) - error("g1t1", seed, scope)
                           for seed in SEEDS] for scope in scopes}
    interaction_repeat = {scope: error("g1t0r", 3407, scope) + error("g0t1r", 3407, scope)
                          - error("g0t0r", 3407, scope) - error("g1t1r", 3407, scope) for scope in scopes}
    interaction_noise = {scope: abs(interaction_repeat[scope] - interaction[scope][0]) for scope in scopes}
    interaction_tau = {scope: max(1.1, 2 * interaction_noise[scope]) for scope in scopes}
    write_json(OUT / "gates.json", {"study": SECTION, "prereg_sha256": manifest["prereg_sha256"],
                                    "source": "strict finite required exact-last rows",
                                    "seeds": list(SEEDS), "arm_repeat_abs_difference_mm": arm_noise,
                                    "noise_scope": "each contrast uses repeats of its own endpoint conditions",
                                    "contrasts": contrasts,
                                    "interaction_gain_mm": interaction,
                                    "interaction_mean_gain_mm": {q: sum(v) / len(SEEDS) for q, v in interaction.items()},
                                    "interaction_repeat_gain_mm": interaction_repeat,
                                    "interaction_repeat_abs_difference_mm": interaction_noise,
                                    "interaction_threshold_mm": interaction_tau,
                                    "scope_limit": "G=bundled original geometry; T=four-slice causal association, not asynchronous runtime",
                                    "research_adoption": "pending independent mechanism/recovery/motion and cost verification"})
    print("Written primary factorial gates; primary efficacy does not imply mechanism adoption.")


def complete():
    frozen()
    path = OUT / "complete_status.json"
    record = {"study": SECTION, "started_at": stamp(), "status": "running", "finished": [],
              "attempted": [], "stage_errors": {},
              "independent_diagnostics": "mandatory even after measured primary failure if valid endpoints exist"}
    write_json(path, record)
    stages = (("train_full", run), ("rows_full", rows), ("report_full", report),
              ("primary_gates", gate_summary), ("independent_diagnostics", diagnostics), ("finalize", finalize))
    primary_labels = {"train_full", "rows_full", "report_full", "primary_gates"}
    primary_errors = {}
    for label, action in stages:
        record.update(current=label, updated_at=stamp())
        write_json(path, record)
        event(action="pipeline_stage_start", stage=label)
        try:
            action()
        except Exception as error:
            record["stage_errors"][label] = str(error)
            event(action="pipeline_error", stage=label, error=str(error))
            # A changed source is an execution-contract failure.  It must never
            # be called a measured model failure or allowed to launch diagnostics.
            try:
                frozen()
            except Exception as frozen_error:
                record.update(status="failed_frozen_contract", error=str(frozen_error), failed_at=stamp())
                write_json(path, record)
                raise
            if label in primary_labels:
                primary_errors[label] = str(error)
                primary_failure_record(primary_errors)
        else:
            record["finished"].append(label)
            event(action="pipeline_stage_finish", stage=label)
        record["attempted"].append(label)
        record.update(updated_at=stamp())
        write_json(path, record)
    verdict_path = OUT / "verdict.json"
    verdict = read_json(verdict_path) if verdict_path.is_file() else {}
    if "finalize" in record["stage_errors"]:
        final_status = "failed_finalization"
    elif primary_errors:
        final_status = "measured_primary_failure"
    elif "independent_diagnostics" in record["stage_errors"]:
        final_status = "diagnostic_execution_failure"
    else:
        final_status = "complete"
    record.update(status=final_status, completed_at=stamp(),
                  scientific_status=verdict.get("status"), decision=verdict.get("decision"),
                  diagnostics_attempted="independent_diagnostics" in record["attempted"],
                  all_stages_attempted=len(record["attempted"]) == len(stages))
    write_json(path, record)
    status()
    if final_status in ("failed_finalization", "diagnostic_execution_failure"):
        raise RuntimeError(f"XC required execution failed: {record['stage_errors']}")


def status():
    manifest = read_json(OUT / "study_manifest.json")
    observations = {}
    path = OUT / "events.jsonl"
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "run" in item:
                observations[item["run"]] = item
    result = []
    for job in manifest["jobs"]:
        folder = REPO / f"outputs/semkine/{job['run']}"
        steps = [int(match.group(1)) for candidate in folder.glob("*step=*.ckpt")
                 if (match := re.search(r"step=(\d+)", candidate.name))]
        item = observations.get(job["run"], {})
        log = OUT / f"logs/{job['run']}.log"
        progress_path = folder / "progress.json"
        progress = {}
        if progress_path.is_file():
            try:
                progress = read_json(progress_path)
            except (ValueError, OSError):
                progress = {"read_status": "temporarily_unreadable"}
        result.append({**job, "latest_saved_step": max(steps, default=0),
                       "latest_optimizer_step": progress.get("global_step", max(steps, default=0)),
                       "training_progress": progress,
                       "exact_last_exists": final_checkpoint(job) is not None,
                       "individual_row_exists": individual_row_path(job).is_file(),
                       "last_action": item.get("action", "not_launched"), "gpu": item.get("gpu"),
                       "last_event_time": item.get("time"), "log_bytes": log.stat().st_size if log.is_file() else 0})
    with STATE_LOCK:
        write_json(OUT / "status.json", {"time": stamp(), "frozen": (OUT / "frozen_manifest.json").is_file(), "jobs": result})
    for item in result:
        print(f"{item['last_action']:18s} {item['run']} step={item['latest_optimizer_step']}/{STEPS} saved={item['latest_saved_step']} gpu={item['gpu']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "run", "streams", "rows", "report", "gates", "diagnostics", "finalize", "complete", "status"))
    args = parser.parse_args()
    os.chdir(REPO)
    actions = {"prepare": prepare, "freeze": freeze, "run": run, "streams": check_streams,
               "rows": rows, "report": report, "gates": gate_summary, "diagnostics": diagnostics,
               "finalize": finalize, "complete": complete, "status": status}
    actions[args.command]()


if __name__ == "__main__":
    main()
