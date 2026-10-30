"""Freeze the user-authorized experiment before new scientific readings."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/s43_rel_20261027"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+"\n")
    tmp.replace(path)

def freeze():
    old=["model/model.py","model/mano_layer.py","semkine/train.py","semkine/dataset.py",
         "semkine/events.py","semkine/eval_track.py","semkine/xyz_mesh.py","assets/mano_right.npz",
         "outputs/semkine/s43n_sp_s3407/s43n_sp_s3407-step=6000.ckpt",
         "outputs/semkine/s43n_sp_s3407/config_resolved.yaml",
         "tools/s43_integration/model.py","tools/s43_integration/fast_geometry.py",
         "tools/s43_integration/raster.cpp","tools/s43_br/data.py"]
    plan=ROOT/"docs/S43_三种表征方案与主方案定义_第二轮重设计_20261027.md"
    split=Path("/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json")
    p={
      "schema":"s43_relation_execution_v1","date":"2026-10-27","seed":3407,
      "authorization":"Current user explicitly requests mechanisms, one correction, full training and streaming evaluation.",
      "plan":str(plan.relative_to(ROOT)),"plan_sha256":sha(plan),
      "frozen_existing_sources":{x:sha(ROOT/x) for x in old},
      "split_manifest":str(split),"split_sha256":sha(split),
      "allowed_gpus":[4,5,6,7],"forbidden_gpus":[0,1,2,3],
      "arms":{
        "s43":"Original S43, same full-network fine-tuning; all new contributions gated zero.",
        "enhanced":"S43 plus zero-initialized true last/first/count/4-bin/EMA 20ch increment.",
        "no_order":"S43 plus A6 and typed geometry; spatial reverse Z symmetrized at fixed p,q.",
        "full":"S43 plus A6 and typed ordered-Z f,a,lambda geometry."},
      "architecture_deviation":"Shared native-LNES/render bypass and true observation increments. Not an exclusive replacement or pseudo-LNES bridge. Added active parameters reported.",
      "core":{"width":240,"height":180,"cell_size":2,"offsets":"8-neighbor plus doubled",
              "tau_ms":[2,8,32],"gamma_ms":[16,64],"dtype":"float32",
              "prefix":"t_i < cutoff, complete integer-ms multiset",
              "state":"Continuous from valid-run boundary; no query/commit reset.",
              "output_lag_ms":1,"MANO_independent_write":True},
      "geometry":{"samples":128,"radius_px":4,"k":4,"direction_translation_m":0.005,
                  "direction_rotation_rad":0.1,"K_source_grid":[480,640],"K_event_scale":0.375,
                  "semantics":"polygonal self-occlusion silhouette normal action; switches flagged, not texture normal, velocity likelihood or Fisher"},
      "tool_gates":["direct-pair reference","half-open/future guard","same-ms multiplicity",
                    "chunks/empty/memory","mixed-seq/padding/release/checkpoint",
                    "SO3 action finite differences/near-pi","zero-increment legacy parity",
                    "actual backward","training-online prefix equality"],
      "mechanisms":{
        "E1":{"question":"Does time order improve the required MANO target beyond simple summaries?",
              "existing":"16 label-independent windows per base local/global FIT7 and SELECT2 recording; no zgz selection.",
              "synthetic":"Full-resolution MANO silhouette threshold events, root and real finger flexion;30contexts per axis,6observations each;contextsplit16/6/8.",
              "time_intervention":"Protect per-pixel polarity count,first,last,4 coarse-bin totals; change only interior integer-ms timestamps. Destructive diagnostic, not physical regenerated events.",
              "baselines":["LNES","20ch simple","unordered-Z","ordered-Z","TORE4 reconstructed-kernel"],
              "readouts":"Training-normalized RBF regression, median training bandwidth, ridge .01; typed MLP hidden128 Adam .001 batch32 300steps seed3407.",
              "support_rule":"SELECT task risk >=2% better full than no_order and enhanced in both readouts; otherwise unsupported/low-power, not a raw-information ceiling."},
        "E2":{"question":"Can the same evidence be requeried under changed deployable history and assist correction?",
              "perturbations":"root-x +5/+15mm; one actual finger flexion axis +.1/+.3rad, axis selected by FK geometry only before event scores.",
              "controls":["fresh binding","stale binding with same perturbed prior","raw same-core rebuild","sensor-only"],
              "parity":"Float32 allclose rtol1e-4 atol1e-5",
              "support_rule":"fresh task risk >=2% lower than stale on SELECT in named conditions; absolute and RA reported; oracle history diagnostic only.",
              "attribution_limit":"Oracle also changes prior/initialization; no unique routing attribution without component controls."},
        "E3":{"question":"Absent directional support versus cancellation, and separate actual resource assessment.",
              "synthetic":"Known normal/tangent/balanced-opposite supports at fixed MANO query; formal nullspace distinct from physical observability.",
              "existing":"Same-prefix real training/SELECT streaming/chunks and f-only/sensor-only diagnostics; no Gram-inverse posterior.",
              "resources":"10k/50k/150k events per50ms; query25/50/100; synthetic and actual costs separate.",
              "support_rule":"Declared factor nullspace and nonnegative support pass; task/ambiguity and cost conclusions separate. Passing algebra alone is not task innovation."}},
      "training":{
        "initialization":"Same canonical original seed3407 step6000 checkpoint; paired full-network fine-tuning, not fresh.",
        "small":{"steps":120,"batch":4,"clip_steps":2,"subjects":"FIT7","validation":"SELECT2"},
        "full":{"steps":6000,"batch":8,"clip_steps":4,"subjects":"all9 original train subjects/all72 manifest trials",
                "outputs_per_arm":192000,"fraction_of_historical_6144000_samples":0.03125},
        "optimizer":"Adam","lr":0.0001,"warmup":500,"schedule":"cosine","TBPTT":1,
        "BN_statistics_frozen_initially":True,"base_weights_trainable":True,
        "loss":"unchanged so3_trans_fk and original log10 objective",
        "state":"per-lane persistent sensor memory, public initialization, own detached feedback, no GT history",
        "paired_sampling":True,"checkpoint":"last only; no seed/best-step selection",
        "correction":"One targeted scientific correction from FIT/SELECT only, recorded before rerun; engineering bugs distinct.",
        "failure_policy":"Tool/nonfinite failure blocks until fixed. Negative science stays negative. Current user authorizes correction then complete paired training as falsification; success not forced."},
      "evaluation":{
        "primary":"W50/raw, complete valid runs, public init and common two-window canonical warmup",
        "recordings":["ycy_local","ycy_global","zgz_local","zgz_global"],
        "query_intervals_ms":[25,50,100],
        "asynchrony":"1ms versus variable complete-group chunks; pure query at fixed commit versus changed commit strategies separate.",
        "prior_errors":"root5/15mm, actual finger.1/.3rad; fixed sources",
        "metrics":["MPJPE local/global","MPVPE local/global","RA-MPJPE","actual synchronized service latency p50/p95/p99",
                   "outputs/sec","events/sec","CUDA allocated/reserved peak","resident/workspace bytes",
                   "ingest/seal/geometry/readout/network/commit maintenance","watermark/output lag"],
        "accuracy_goal":"Noninferiority within max(.5mm,5%) of paired S43 for local/global; one seed descriptive.",
        "latency":"Include real geometry/memory/transfer/network/commit; compilation/warmup/tail separately; no scaled proxy.",
        "references":"Frozen canonical S43 separate; legacy GT-assisted protocol separate diagnostic; S39 stack secondary if executed.",
        "zgz":"Final design locked first; report once; project dev=test identity unchanged.",
        "EvRealHands":"not touched; requires final-only source adaptation protocol."},
      "claims":"No promised improvement; classify representation implementation/readout/optimization/evidence/core failure."
    }
    write_json(OUT/"protocol.json",p)
    with (ROOT/"docs/EXPERIMENT_RECORDS.md").open("a") as f:
      f.write("\n\n<a id=\"s43-rel-prereg-20261027\"></a>\n\n"
        "### S43 有向关系记忆：机制实验、完整配对微调与在线评估预注册（2026-10-27）\n\n"
        "本轮用户明确授权从A原型推进到完整训练与在线推理。新科学读数前冻结 outputs/s43_rel_20261027/protocol.json。"
        "新增代码只在tools/s43_rel、tests/test_s43_rel_*、configs/s43_rel；生产/旧B-R/S43/S39指针不改。"
        "仅物理GPU4–7，0–3由他人使用；CPU并行。前置工程单测/真实checkpoint smoke已做且披露，不作任务读数。\n\n"
        "四臂S43、简单增强、空间反向Z平均抹顺序、完整Z。保原LN/render共同通路，加真实A输入通道和typed几何增量；"
        "不称完全替代LNES。全base可训练，新增支路zero-init，原MANO/51D/监督保留，新增结构单列。\n\n"
        "先工具验收再E1时间必要性、E2几何重读、E3支持与分列的成本。合成和现有v2e录制分开。"
        "基础录制16固定窗，FIT7/SELECT2选择，TORE4同核强控，common RBF与typed MLP300步。"
        "时间破坏保指定简单统计，只称诊断，不冒充物理重生成；公式成立不升级成任务有效。\n\n"
        "small120步×B4×clip2；根据FIT/SELECT做一轮记录修正。full6000实际optimizer步×B8×clip4，每臂192000输出，"
        "历史6.144M的1/32，全9人原72trials，四臂同canonical6000步初值/抽样/优化机会。"
        "这是完整预注册配对微调，不冒充历史同样本量fresh训练。Adam1e-4/warm500/cosine/TBPTT1，自反馈detach；"
        "sensor连续tail只在valid-run边界reset，末短尾消费再release，不以随机窗重置截尾。last固定，不择seed/最好点。\n\n"
        "完整W50/raw公共初始化与GT初始化上界分列，25/50/100ms查询、1ms异步/分块、局部屈曲/前态扰动。"
        "MPJPE/MPVPE local/global及实际端到端延迟/吞吐/显存/维护全账，非CPU核或缩放时延代理。"
        "非劣参考max(.5mm,5%)，单seed描述。科学负结果如实保留；当前用户授权一次修正后完整训练作否证，"
        "实现/非有限错误先修，不为交付强称成功。效果/资源/阈值/全部输入身份见冻结JSON。\n")
    print(OUT/"protocol.json")

if __name__=="__main__":freeze()
