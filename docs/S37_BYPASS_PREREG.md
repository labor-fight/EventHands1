# R1：冻结自身历史上的手指旁路移除诊断

预注册：2026-09-25，父代理 /root。本文创建在任何 R1 数据评分之前。原始提案见 `research_state/lit/G10_tracking_resume_20260925.md §5`；R0 每臂都保留旁路，所以本项不是重跑 R0。仅归入 C0r 最小修复的必要性排查，非第四架构、非正式主行。

## 1. 假说和边界

固定 S37 s3407 step2500、R0 抽取时的自身递推历史、同包事件/几何路由和已存预测。检查去掉 `prev_mlp` 的45D手指加性输出，是否优于原预测及保持原方向的等幅对照。结果只解释该 checkpoint 两条共同训练路径的条件组合；失败可能是共适应，不能证明旁路必要或一切历史先验有害。固定历史改进不等于新闭环改进。

不重跑网络、不开优化器、不换标签/帧、不读 zgz、不使用 GT 选择增量。原权重/模型源码/训练配置均不改。R0 中18条训练序列，每条固定128步；8人16序列2048步为fit，ylf两序列256步仅作为一次dev确认，backbone已见所有9人，不作泛化声明。

## 2. 干预与对照（固定后不调系数）

记 `d = pred[6:] - prev[6:]`，`g = prev_bypass[6:]`。原最终空包门使用 packet counts，由 `make_eval_packet` 的 ptr 差得到，等于缓存 `n_events`；空包令有效g=0并检查pred=prev。

- original：原51D预测。
- off：root6D保留原pred，手指=`prev+d-g`。
- amp：root同上，手指=`prev+αd`，`α=||d-g||₂/||d||₂`，不截断α。
- keep_fingers：root仍用原pred，手指保持prev。
- keep_all：完整prev，仅参考。

所有增量算术使用缓存 FP32 值提升到 NumPy FP64，构造完输出转 FP32 解码。`||d||=0`与`||d-g||=0`时α=0；前者为0而后者非0时amp保持，记录不可定义，完整分母保留，方向准入INCONCLUSIVE并停止，不删行。极小非零范数仍按公式计算，非有限直接失败。

## 3. 执行前合同与单位

运行只在CPU单进程，库线程4，无进程分片、CUDA查询或GPU预留。内部80s期限，外层`timeout --signal=TERM --kill-after=5s 90s`；超时保留INCOMPLETE、不自动续跑。所有命令、时间、源码/资产/缓存hash写入 RUNS 和 R1回执。

先做合成合同：普通同向/反向/正交、零对零、抵消导致不可定义及空包，检查输出root逐位不变，空包保持，FP64等幅最大误差≤1e-10 rad。实际FP32转回后等幅误差≤`1e-6+1e-5*max_norm` rad；任何非有限或形状/身份错误终止。仅这些检查通过才读数据评分。

只加载R0所需prev/pred/target/bypass/n_events/betas；复核R0 contract及其绑定源码/权重/资产hash、18个npz hash、序列顺序/身份/步数。源码是抽取时版本，读取缓存不重新生成样本。MANO FP32、`add_mean=False`，用 `decode_to_mano_inputs(mano_full_axis_angle)` 补 `hands_mean`，joint0/vertex0对齐，米转mm；与R0原评分相同。若进入dev，original/keep_all应与R0已知dev参考评分差≤1e-3mm，否则评分合同失败；不放宽容差。

## 4. 预定门与停止条件

按动作类对完整帧平均；fit各类1024帧，dev各类128帧。只看MPJPE门，MPVPE及每序列/每帧数值留回执说明，不用于择优。

先fit；仅以下全部通过才读dev并一次确认同门：

1. off相对original的local改善≥0.5mm，且相对amp的local改善≥0.5mm。
2. off global相对original回退≤0.3mm。
3. 每类无amp不可定义行，全部输出有限，等幅与root/空包合同通过。
4. 为限制45D角度等幅不等于实际关节等幅的混杂，计算同root+prev手指参考下每帧joint0对齐的平均关节位移 `s_off,s_amp`；每类 `mean(|s_off-s_amp|)/max(mean(s_off),1e-8 mm) ≤0.10`。这只是预定容差控制，不是关节方向完全匹配证明。

fit任一项失败：不读dev干预分数，结束此固定移除提案，不增删关节、不搜索旁路比例、不换checkpoint/子集。若仅第4项失败，方向解释INCONCLUSIVE，亦不进入dev；不事后补另一幅度控制。dev任何门失败则不采纳；两阶段都过也仅允许另立真实闭环Debug合同。不得把临时去除旁路直接部署或称为创新。

## 5. 产物

脚本 `scratch/goal_20260925/r1/bypass_diagnostic.py`；独立合成合同回执 `synthetic_receipt.json`；执行前身份 `identity.json`；唯一真实运行输出 `run_v1/receipt.json` 及完整已评帧 `*_rows.npz`。RUNS追加开始与结束；本文件与FAILURE账本追加最终判读。此处尚无R1结果。

## 6. 唯一执行终态（2026-09-25T17:50:09.228638+08:00）

`r1_bypass_frozen_v1` COMPLETED / exit0，CPU单进程4库线程；外层wall 2.786745285964571秒，GPU预算0。合成6例及原绑定身份检查通过；父代理从16个已评缓存行文件独立重算20个聚合分数，逐值一致，2048帧全分母保留。`research_state/audit/R1_terminal_audit_20260925.json`。

fit收益门、global门及FK幅度混杂门均未通过。移除在两动作类别均更差；方向归因仍INCONCLUSIVE，因为角度等幅不等于实际关节等幅。按原门停止，未产生任何ylf dev干预评分文件；没有比例、关节子集、checkpoint搜索或真实闭环修改。仅否定这一个固定权重的直接移除提案；共同训练补偿仍可解释其恶化，不能反推所有历史先验有益或旁路必需。

内部未舍入回执摘录（非正式主表）：

```json
{
  "scores": {
    "local": {
      "original": {
        "mpjpe_ra_mm": 25.084148606197232,
        "mpvpe_ra_mm": 19.93118241216507,
        "n_frames": 1024
      },
      "off": {
        "mpjpe_ra_mm": 27.77571577416893,
        "mpvpe_ra_mm": 22.36255688194433,
        "n_frames": 1024
      },
      "amp": {
        "mpjpe_ra_mm": 26.209080004491625,
        "mpvpe_ra_mm": 21.039791735802282,
        "n_frames": 1024
      },
      "keep_fingers": {
        "mpjpe_ra_mm": 25.0922441041439,
        "mpvpe_ra_mm": 19.946253803254876,
        "n_frames": 1024
      },
      "keep_all": {
        "mpjpe_ra_mm": 25.012714568788397,
        "mpvpe_ra_mm": 19.88211436275833,
        "n_frames": 1024
      }
    },
    "global": {
      "original": {
        "mpjpe_ra_mm": 19.115111156452258,
        "mpvpe_ra_mm": 14.421468486034428,
        "n_frames": 1024
      },
      "off": {
        "mpjpe_ra_mm": 23.72115833668431,
        "mpvpe_ra_mm": 18.549995053945167,
        "n_frames": 1024
      },
      "amp": {
        "mpjpe_ra_mm": 20.33042640414351,
        "mpvpe_ra_mm": 15.802817689746007,
        "n_frames": 1024
      },
      "keep_fingers": {
        "mpjpe_ra_mm": 19.09254982456332,
        "mpvpe_ra_mm": 14.403618366031878,
        "n_frames": 1024
      },
      "keep_all": {
        "mpjpe_ra_mm": 19.357040500381117,
        "mpvpe_ra_mm": 14.574330066125185,
        "n_frames": 1024
      }
    }
  },
  "gains": {
    "local_vs_original_mm": -2.691567167971698,
    "local_vs_amp_mm": -1.5666357696773048,
    "global_regression_mm": 4.6060471802320535
  },
  "categories": {
    "local": {
      "fk_step_relative_mismatch": 0.2995102990883496,
      "amp_undefined_count": 0,
      "empty_count": 0,
      "bypass_changed_count": 1024
    },
    "global": {
      "fk_step_relative_mismatch": 0.3687588662345553,
      "amp_undefined_count": 0,
      "empty_count": 0,
      "bypass_changed_count": 1024
    }
  },
  "gates": {
    "gain_vs_original": false,
    "gain_vs_amp": false,
    "global_regression": false,
    "amplitude_defined": true,
    "fk_step_mismatch": false
  },
  "pass": false,
  "direction_attribution": "INCONCLUSIVE"
}
```

执行前预注册已独立冻结为 `scratch/goal_20260925/r1/prereg_frozen.md`；identity绑定该副本，原文只追加终态。SIGTERM处理与外层终态记账在真实执行前独立审查后补全，不改变科学门。只有进入dev才触发的CPU/CUDA评分parity未执行，不能写为已通过；本轮只是fit条件诊断。
