# EventHands hand_data51 实验报告合集

> **当前裁决与 zgz 协议数字见** [FAILURE_AND_CLEANUP_LEDGER.md](FAILURE_AND_CLEANUP_LEDGER.md)。
> 本文保留旧协议 §1–14 的逐实验收据，不替代总览。


第 1-8 节由 `outputs/hand_data51/report_*.md` 的 8 份报告合并而成，正文逐字保留，仅调整了标题层级
与图片路径。按实验推进顺序排列。

第 7、8 节所述的 SPA / SPA round 2 代码曾随回退到 **EventHands-AbsRender** 一并删除；
那条路线的诊断过程已并入本文 §9–11（核心结论在 §10 被推翻）。
第 9 节是按 §7 的文字重建该头之后的结果：**17.91 mm 未能复现，且推理期消融显示这个头是惰性的**，
因此 §7 / §8 中依赖"两次独立训练之差"的消融结论都需要重新审视。
第 10 节用零训练的判别探针**修正了根因**：瓶颈不在输入表示也不在暴露偏差，而在
**事件→绝对姿态映射对 val 主体不泛化**（教师强制单步 train 5.4mm vs val 26.8mm）；
同轮落地了 delta 信任域，闭环 RA 19.26 → **18.73**（当前最好读数）。
第 11 节用一次 LOSO 训练**再次修正根因**：跨主体泛化其实很好（未见主体 local 6.0mm），
26.8mm 全部来自 `zgz_local` 这一条低事件率序列（692 事件/50ms，同主体 global 的 1/43）。
**19.26 / 18.73 这些历史读数是被单条异常序列支配的**，这解释了前四轮改动为何全部无效；
§10 的"主体泛化鸿沟"表述应按 §11 理解。同轮据此在数据层做域随机化（网络逐参数不变），
双副本 worse-of-two 闭环 RA **19.26 → 12.77mm**（当前最好读数），TF 单步 26.84 → 12.60mm。
第 12 节拆开 4 通道输入里的两个渲染通道：**剪影掩码与归一化逆深度互相可替代**，
单用任一个与两者都用在统计上不可区分；同时给出一条方法学警告——
本轮臂间效应（≤0.5mm）远小于同配置运行间散布（2.43mm），
§7 / §8 那类"两次独立训练之差"的结论都需要按 §12.3 的池化口径重算才可信。

## 目录

1. [绝对位姿基线：PCA-6 vs Full-51](#1-绝对位姿基线pca-6-vs-full-51)
2. [纯增量 tracking 的开环递推](#2-纯增量-tracking-的开环递推)
3. [Battle 分析：增量 tracking 方案裁决](#3-battle-分析增量-tracking-方案裁决)
4. [track_render51：render-and-compare + delta](#4-track_render51render-and-compare-+-delta)
5. [4 通道绝对模型消融（abs_render51 / EventHands-AbsRender）](#5-4-通道绝对模型消融abs_render51--eventhands-absrender)
6. [全实验报告（阶段 1-5 汇总）](#6-全实验报告阶段-1-5-汇总)
7. [SPA：语义部件锚定 delta 头](#7-spa语义部件锚定-delta-头)
8. [SPA round 2：EDD + 任务空间关节损失](#8-spa-round-2edd-+-任务空间关节损失)
9. [KSGN：LBS 语义关节 GN 头重建（未复现 17.91，头被判定惰性）](#9-ksgnlbs-语义关节-gn-头重建未复现-1791头被判定惰性)
10. [无训练判别探针：根因修正 + delta 信任域（19.26 → 18.73）](#10-无训练判别探针根因修正--delta-信任域1926--1873)
11. [LOSO 诊断 + 域随机化：根因再修正与 19.26 → 12.77mm](#11-loso-诊断--域随机化根因再修正与-1926--1277mm)
12. [渲染通道消融：silhouette vs 归一化逆深度](#12-渲染通道消融silhouette-vs-归一化逆深度)

---

## 1. 绝对位姿基线：PCA-6 vs Full-51

> 原文件：`outputs/hand_data51/report_abs_baseline.md`（原标题：EventHands Absolute Pose Baseline Report）

PCA-6 (12D) vs full axis-angle (51D) on hand_data51 val (`zgz_global`, `zgz_local`).

### Checkpoints

- **PCA6**: `outputs/hand_data51/abs_pca12/abs_pca12-step=11000-val_loss=val_loss=4.3319.ckpt` (best val_loss)
- **Full51**: `outputs/hand_data51/abs_full51/abs_full51-step=6000-val_loss=val_loss=1.1009.ckpt` (best val_loss, `LAMBDA_POSE=450 = 45/0.1`)

Note: A/B `val_loss` are not comparable (different NORMALIZER / layout). Compare MPJPE/MPVPE.

### Main metrics (mm)

| Model | dim | MPJPE (RA) | MPVPE (RA) | MPJPE (abs) | MPVPE (abs) | global MPJPE | local MPJPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| EventHands-PCA6 | 12 | 19.61 | 15.15 | 59.59 | 59.25 | 10.99 | 30.00 |
| EventHands-Full | 51 | 19.39 | 15.90 | 61.53 | 61.14 | 11.78 | 28.56 |

### Per-finger MPJPE root-aligned (mm)

| Model | thumb | index | middle | ring | pinky |
|---|---:|---:|---:|---:|---:|
| EventHands-PCA6 | 13.98 | 22.93 | 24.98 | 22.15 | 18.93 |
| EventHands-Full | 15.91 | 22.72 | 23.79 | 21.53 | 17.87 |

### Efficiency

| Model | Params | FLOPs | latency (ms, batch=1) |
|---|---:|---:|---:|
| EventHands-PCA6 | 11182725 | 1652710016 | 1.718 |
| EventHands-Full | 11202732 | 1652729984 | 1.755 |

### GT-side PCA-6 truncation ceiling (val)

Oracle: same `t`/`R`/betas as GT, local pose = `alpha6 @ C6` (no network).

- energy retained by PCA-6: **79.7%**
- residual RMS truncation error: **0.161 rad/dim**
- MPJPE (RA): **6.85 mm**, MPVPE (RA): **6.87 mm**
- per-finger MPJPE (RA): thumb 10.96, index 8.17, middle 5.02, ring 5.30, pinky 6.53

Per sequence:
- `zgz_global` (global): MPJPE=5.00, MPVPE=5.10, energy=85.3%
- `zgz_local` (local): MPJPE=8.90, MPVPE=8.82, energy=73.6%

### Conclusion: does 6D MANO PCA limit single-finger local pose?

1. **Network A vs B**: Full51 improves overall RA-MPJPE by only **0.22 mm** (19.61 → 19.39), and local-sequence RA-MPJPE by **1.44 mm**. MPVPE slightly favors PCA6 (15.15 vs 15.90). Per-finger: index/middle/ring/pinky slightly better with Full; thumb slightly worse.
2. **GT truncation ceiling** already sits at **6.9 mm** RA-MPJPE (thumb alone **11.0 mm**), and local sequence truncation is **8.9 mm**. Both trained models (~19.4–19.6 mm) are far above this ceiling, so fitting error is dominated by the LNES→pose regressor, not by the 6D PCA bottleneck.
3. **Answer**: On this absolute-pose baseline, **6D PCA does not appear to be the primary limiter of single-finger local pose fitting**. The representation truncates ~20% of local pose energy and imposes a ~7 mm oracle floor (higher on local/thumb), but unlocking full 45D axis-angle yields only marginal MPJPE gains under identical training. Further gains should come from temporal/tracking models or better event features, not from removing PCA alone.

---

## 2. 纯增量 tracking 的开环递推

> 原文件：`outputs/hand_data51/report_track_delta.md`（原标题：Tracking vs absolute baselines (recursive protocol)）

| model | mode | step(ms) | MPJPE RA | MPVPE RA | MPJPE abs | jitter static (mm/step) | jitter all |
|---|---|---:|---:|---:|---:|---:|---:|
| track_delta51 | track | 50 | 79.97 | 60.74 | 1010.14 | 16.877 | 23.018 |
| track_delta51 | track | 50 | 93.38 | 68.78 | 1670.84 | 2.779 | 8.551 |
| abs_full51 | absolute | 50 | 19.97 | 16.43 | 63.43 | 13.608 | 13.158 |

![drift](assets/report_track_delta.png)

---

## 3. Battle 分析：增量 tracking 方案裁决

> 原文件：`outputs/hand_data51/report_battle_analysis.md`（原标题：Battle 分析：事件增量 tracking 方案裁决报告）

日期：2026-08-12。评测协议：递推 tracking（step=50ms，GT 首帧+噪声初始化，init_noise_scale=1.0，seed=0），val = `zgz_global`（1386 帧）+ `zgz_local`（1204 帧，41 段）。

### 1. 证据账本（全部 observed，receipt 可审计）

| ID | 证据 | 来源 |
|---|---|---|
| E1 | abs_full51 递推协议 RA-MPJPE **19.97mm**，漂移曲线平坦（10-16mm @65s） | `abs_full51/eval/track_metrics_step50.json` |
| E2 | track_delta51 纯开环 RA-MPJPE **93.4mm**；绝对误差线性漂移：5s 231mm → 65s **5554mm**（有偏积分，非随机游走） | `track_delta51/eval/track_metrics_step50.json` |
| E3 | 增量模型 jitter_all：local **6.4 vs abs 14.5** mm/步（GT 真实运动 3.8）；global 10.7 vs 11.8 | 同上 |
| E4 | track val_loss 在 step2000 后单调恶化（0.083→0.145）；abs/pca 无此模式 | `logs/train_track_delta51.log` |
| E5 | 漂移主轴是平移（RA 后 93mm vs 绝对 1670mm）：t_z 从 2D 事件差分弱可观测 | E2 分解 |
| E6 | 第一部分：PCA6 vs Full51 仅差 0.2mm；GT-PCA6 截断上限 6.9mm << abs 的 19.4mm | `report_abs_baseline.md` |
| E7 | **零训练融合（Phase 1 裁决实验）**：`x_k=(1-α)(x_{k-1}+Δ̂)+α·ŷ_abs`，α=0.2 → RA-MPJPE **19.72**、jitter **6.79**；α=0.5 → **19.32** / 8.26；漂移曲线与纯 abs 重合 | `fuse_sweep.json` / `fuse_sweep.png` |

结构性根因（inferred）：(a) 训练 iid 高斯噪声 vs 推理相关有偏误差的 exposure bias；(b) prevpos 仅经零初始化 MLP 进输出端，输入中不存在"当前估计 vs 观测"的空间对齐反馈（对照 se(3)-TrackNet/ReFit 的 render-and-compare 设计）。

### 2. 2023-2026 顶会范式地图（第一性原理分类）

按"状态如何传播"分五族：

1. **逐帧绝对 + 全局后处理**（SOTA 主流）：HaMeR/WiLoR → HaWoR、Dyn-HaMR（CVPR'25：逐帧绝对回归 + SLAM 相机解耦 + 运动先验 infill）；WHAM/TRAM/SLAHMR 同构。原理：绝对观测无漂，时序仅平滑/补洞/世界坐标。
2. **帧内迭代增量**：ReFit（ICCV'23，逐参数 GRU ΔΘ）、PyMAF、FoundationPose——Δ 相对当前估计，输入必须含当前估计的投影反馈。
3. **生成先验后验推断**：ScoreHMR（CVPR'24）、HuMoR、HMP——先验补观测缺失。
4. **Memory/query 传播**：SAM2、MOTR——latent 状态传播，回避显式积分。
5. **显式增量积分**：EventHPE（ICCV'21）、WHAM root velocity、IMU 预积分——仅当观测本身是差分信号时使用，且**必配**累积损失/锚定/接触约束，无一裸用开环。

事件手部方向（EventHands/EvHandPose/EvRGBHand/EventEgo3D++/EventEgoHands, 2021-2025）全部为绝对回归 + 时序模块；"事件增量 + 绝对锚"的闭环手部 tracking 未见先例（not found in searched scope，截至 2026-08）。

### 3. Battle

**反方（放弃增量）**：SOTA 手部跟踪均为逐帧绝对+后处理；E2 开环爆炸；E6 显示 abs 回归器仍有 ~13mm 理论空间，加大 backbone 收益可能更直接；60s 级纯增量物理上不成立（惯导亦然）。

**正方（继续增量）**：E3 平滑收益真实且达 4 倍；E5 漂移集中在 t，可锚定；被否定的是无锚开环实现而非增量假设——文献中带锚增量（IMU+视觉、WHAM）全部成立；毫秒级事件增量的低延迟独特性是逐帧大模型不具备的。

**裁决（由 E7 决定性实验判定）：GO——闭环融合成立。**

- 预注册判据：存在 α 使 RA-MPJPE ≤ 19.97 且 jitter < 0.8×13.07=10.45 → `joint_0.2`、`joint_0.5` 双双达标。
- α 扫描呈完美 U 型：纯增量（α=0）93.8mm 爆炸、纯绝对（α=1）抖动 13.1；甜点 α∈[0.2,0.5] 同时取得**精度最优（19.32）与抖动近半（6.8-8.3）**。
- 切片细化（α_t 大/α_pose 小）未超过联合 α：局部姿态与平移同样需要锚定。

**结论**：原"纯增量"方案不必继续；升级为**绝对锚定 + 事件增量的闭环双头**（与 IMU+视觉互补滤波同构）。增量头价值在平滑性与高频更新，绝对头价值在无漂锚定。

### 4. 与第一部分结论的衔接

- 表示（PCA6 vs 51D）不是瓶颈（E6），回归器是；
- 闭环融合在不增加任何训练的情况下把递推协议成绩推到 19.32mm（略优于纯 abs 19.97），并将抖动从 13.1 降到 6.8-8.3 mm/步；
- 下一步（Phase 2A）：联合训练闭环双头（共享 backbone、abs+delta 双头、rollout 递推训练修复 exposure bias），目标在同一协议下进一步压 MPJPE 并保持低抖动。

### 5. Receipt 索引

- `outputs/hand_data51/fuse_sweep.json` / `fuse_sweep.png`（Phase 1 裁决）
- `outputs/hand_data51/abs_full51/eval/track_metrics_step50.json`（E1）
- `outputs/hand_data51/track_delta51/eval/track_metrics_step50.json`（E2/E3）
- `outputs/hand_data51/report_abs_baseline.md`（E6，第一部分）
- 临时脚本 `/tmp/eval_fuse_tmp.py` 已按约定删除；协议参数记录于 fuse_sweep.json 的 `protocol` 字段

---

## 4. track_render51：render-and-compare + delta

> 原文件：`outputs/hand_data51/report_track_render51.md`（原标题：track_render51 recursive evaluation）

Protocol: step=50ms, GT init + TRACK.PREV_NOISE (scale=1.0), seed=0, val=`zgz_global`+`zgz_local`.
Ckpt selected by **recursive RA-MPJPE**, not teacher-forced val_loss.

### 1. Ckpt selection (top-3 val_loss + last)

| ckpt | val_loss | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| render step1000 (val_loss best) **← selected** | 0.8015 | 19.26 | 15.68 | 62.21 | 27.97 / 11.69 | 11.384 | 4.545 | 28.9 |
| render step2000 | 1.0657 | 20.04 | 16.57 | 67.43 | 28.35 / 12.82 | 11.731 | 5.425 | 26.2 |
| render step5000 | 0.9326 | 21.76 | 17.31 | 59.67 | 33.11 / 11.90 | 11.723 | 5.268 | 21.9 |
| render last (15k) | — | 21.44 | 17.49 | 59.47 | 30.13 / 13.88 | 11.031 | 4.980 | 19.9 |

Selected: `outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt` (RA-MPJPE **19.26 mm**). Lightning val_loss-best coincides with recursive-best (step 1000).

### 2. Four-way recursive comparison

| model | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|---:|
| abs_full51 | 19.97 | 16.43 | 63.43 | 29.12 / 12.03 | 13.158 | 13.608 | 24.4 |
| track_delta51 | 93.38 | 68.78 | 1670.84 | 66.75 / 116.52 | 8.551 | 2.779 | 5554.3 |
| fuse α=0.5 (zero-train) | 19.32 | 16.16 | 62.55 | 27.89 / 11.87 | 8.261 | — | — |
| track_render51 (step1000) | 19.26 | 15.68 | 62.21 | 27.97 / 11.69 | 11.384 | 4.545 | 28.9 |

### 3. Pre-registered criteria

- Mechanism GO if RA-MPJPE < 40 **and** 65s abs error < 200mm: RA=19.26, glob@65s abs=28.9 → **PASS**
- Beat zero-train fuse if RA ≤ 20 **and** jitter < 10: RA=19.26, jitter=11.38 → **FAIL**
- vs abs_full51 recursive (19.97 / jitter 13.16): RA -0.71 mm, jitter -1.78

### 4. Notes

- Teacher-forced val_loss rose after step 1000 (0.80 → 1.09); recursive RA also worsened (19.26 → 21.44). Selecting by closed-loop RA agrees with val_loss-best this time.
- Render-and-compare converts open-loop 93mm Track into an abs-level tracker; jitter is lower than abs (and close to fuse).

![drift](assets/report_track_render51.png)

---

## 5. 4 通道绝对模型消融（abs_render51 / EventHands-AbsRender）

> 原文件：`outputs/hand_data51/report_ablation_4ch.md`（原标题：4-channel absolute ablation (abs_render51)）

Protocol: recursive step=50ms, GT init + TRACK.PREV_NOISE (scale=1.0), seed=0.
Factor: input channels (2/4) × output form (absolute / delta).

### 1. Ckpt selection (top-3 val_loss + last)

| ckpt | val_loss | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| step3000 **← selected** | 1.2341 | 20.08 | 16.41 | 63.27 | 30.28 / 11.23 | 11.849 | 15.872 | 28.1 |
| step6000 | 1.2366 | 20.16 | 16.26 | 63.79 | 30.45 / 11.22 | 11.551 | 14.242 | 27.9 |
| step13000 (val_loss best) | 1.2239 | 21.07 | 16.90 | 63.94 | 32.18 / 11.42 | 12.049 | 8.556 | 30.8 |
| last (15k) | — | 20.31 | 16.37 | 62.85 | 30.90 / 11.12 | 11.971 | 12.030 | 29.0 |

Selected by recursive RA-MPJPE: `outputs/hand_data51/abs_render51/abs_render51-step=3000-val_loss=val_loss=1.2341.ckpt` (**20.08 mm**).

### 2. 2×2 factor table (recursive RA-MPJPE)

| | output = absolute | output = prev + Δ |
|---|---:|---:|
| input 2ch (LNES) | **19.97** (Full) | **93.38** (Track) |
| input 4ch (+render prev) | **20.08** (AbsRender) | **19.26** (Render) |

### 3. Full recursive comparison

| model | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | local / global MPVPE | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 2ch-abs | 19.97 | 16.43 | 63.43 | 29.12 / 12.03 | 24.60 / 9.34 | 13.158 | 13.608 | 24.4 |
| Track 2ch-delta | 93.38 | 68.78 | 1670.84 | 66.75 / 116.52 | 49.05 / 85.92 | 8.551 | 2.779 | 5554.3 |
| Render 4ch-delta | 19.26 | 15.68 | 62.21 | 27.97 / 11.69 | 23.24 / 9.11 | 11.384 | 4.545 | 28.9 |
| AbsRender 4ch-abs (this) | 20.08 | 16.41 | 63.27 | 30.28 / 11.23 | 25.36 / 8.63 | 11.849 | 15.872 | 28.1 |

### 4. Interpretation

**结论：4ch 渲染进输入 alone 几乎不提升精度；`prev+Δ` 输出结构才是 track_render51 相对 Full 的主要收益来源。**

证据：
- AbsRender RA **20.08 ≈ Full 19.97**（+0.11 mm，统计上等价），并未逼近 Render 的 **19.26**。
- AbsRender jitter all 11.85，介于 Full 13.16 与 Render 11.38 之间，略好于 Full。
- 关键键差异在 **jitter static**：AbsRender **15.87** vs Render **4.55** vs Full 13.61——零事件门控 + delta 输出才把静止漂移压住；绝对输出下渲染通道无法提供同等平滑。
- 反过来，2ch-delta（开环 Track）爆炸到 93 mm，说明 **delta  alone 也不够**，必须与 render-and-compare 输入联用。

因此 track_render51 的成功是 **「渲染反馈输入 + delta 输出」缺一不可**，不是"只是多了 2 个通道信息"。

- vs Full: RA +0.11 mm, jitter all -1.31, jitter static +2.26
- vs track_render51: RA +0.82 mm, jitter all +0.47, jitter static +11.33
- vs Track open-loop: RA -73.30 mm

### 5. Efficiency (user-table row)

| Model | Acc (MPJPE) local / global | Acc (MPVPE) local / global | Latency | FLOPs/step | Params |
|---|---:|---:|---:|---:|---:|
| EventHands-AbsRender (4ch-abs) | 30.28 / 11.23 | 25.36 / 8.63 | 3.96 ms | 1.655 G | 11.20 M |

Latency aligned to published Full=1.75 ms via min-latency ratio (4.155/1.835). FLOPs from thop (CNN-dominated; MANO render not fully counted).

![drift](assets/report_ablation_4ch.png)

---

## 6. 全实验报告（阶段 1-5 汇总）

> 原文件：`outputs/hand_data51/report_full_experiment.md`（原标题：EventHands hand_data51 全实验报告）

日期汇总：2026-08-12 — 2026-08-16  
数据：`data/hand_data51/`（74 序列，51D meta `[t3, R3, residual45]`）  
环境：`EventHandsTrain`；训练常用 `CUDA_VISIBLE_DEVICES` + `NCCL_P2P_DISABLE=1` `NCCL_IB_DISABLE=1`

---

### 0. 总览与主结论

本系列实验按问题推进：

1. **表示是否瓶颈？** PCA-6 vs Full-51 绝对回归 → PCA 不是主瓶颈。
2. **纯增量 tracking 能否工作？** Track(Δ) 开环递推 → 失败（线性漂移）。
3. **绝对锚定 + 增量能否互补？** 零训练 α 融合 → GO。
4. **render-and-compare 能否修好 tracking？** Track(render) → 机制验证 PASS，精度略优于 Full。
5. **收益来自输入信息还是 delta 设计？** AbsRender 4ch-abs 消融 → **两者缺一不可，但相对 Full 的精度收益主要来自 `prev+Δ` 输出结构**。

#### 总表（用户表格口径）

由 `tools/make_main_table.py` 生成（图片 `outputs/hand_data51/main_table.png`，
数字 `main_table.md` / `main_table.json`）。精度尽可能直接读评测 JSON，成本三列在当前机器上
一次性重测，因此各行之间可比——历史文档里的延迟横跨多个测量会话，渲染各行相差可达 10%。

![主表](../outputs/hand_data51/main_table.png)

| Model | 权重来源 (run · ckpt) | MPJPE local | MPJPE global | MPVPE local | MPVPE global | RA-MPJPE (递推) | Latency | FLOPs/step | Params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EventHands-PCA6 [a] | `abs_pca12` · eval_abs | 30.00 | 10.99 | 23.58 | 8.15 | — | 1.76 ms | 1.653 G | 11.18 M |
| EventHands-Full [a,b] | `abs_full51` · eval_abs / eval | 28.56 | 11.78 | 24.01 | 9.16 | 19.97 | 1.75 ms | 1.653 G | 11.20 M |
| EventHands-Track (Δ) [c] | `track_delta51` · last.ckpt | 62.78 | 94.89 | 48.47 | 71.40 | 79.97 | 1.85 ms | 1.653 G | 11.21 M |
| EventHands-AbsRender (4ch abs) | `abs_render51` · step3000 | 30.28 | 11.23 | 25.36 | 8.63 | 20.08 | 4.37 ms | 1.655 G | 11.20 M |
| EventHands-Track (render) [d] | `track_render51` · step1000 | 27.97 | 11.69 | 23.24 | 9.11 | 19.26 | 4.53 ms | 1.655 G | 11.21 M |
| EventHands-Track (render, sil only) [e] | `…ch_sil` · step1500 | 28.16 | 12.78 | 22.45 | 9.97 | 19.93 | 4.41 ms | 1.654 G | 11.21 M |
| EventHands-Track (render, inv only) [e] | `…ch_inv_rep2` · step1000 | 28.32 | 12.73 | 22.73 | 10.92 | 19.98 | 4.50 ms | 1.654 G | 11.21 M |
| EventHands-Track (render, sil+inv) [d,e] | `…ch_both_rep2` · step1500 | 26.92 | 13.11 | 22.14 | 9.86 | 19.53 | 4.54 ms | 1.655 G | 11.21 M |

口径说明：

- **a** PCA6 / Full 的四列精度为逐帧绝对评测（`eval_abs.py`，更密采样）；其余各行为递推协议
  （step=50 ms，每段 valid run 开头 GT+噪声初始化，之后 `prev ← pred`，不再喂姿态 GT）。
- **b** Full 的 RA 取自同一权重的递推评测（`abs_full51/eval`），故 **RA 列全表同协议可比**；
  该协议下 Full 的 local/global 为 29.12 / 12.03。PCA6 无递推评测，RA 留空。
- **c** Track(Δ) 取 `last.ckpt`（与历史表格一致）；其 `eval_best` 更差（RA 93.38）。
- **d** 这两行是**同一个网络但不是同一份权重**，所以数字不该相同。网络侧完全一致（同为
  11,209,429 参数、`conv1` 同 4 通道、`RENDER_CHANNELS` 同为 `[sil, inv]`），因此
  **成本三列一致**（4.53 vs 4.54 ms 之差 0.01 ms 在测量散布内）；权重侧是两次独立训练：
  `sil+inv` 行 = seed 1、单卡 2048、3000 步、取 step1500；Track (render) 行 = seed 0、
  2 卡×1024、15000 步预算、取 step1000。精度差异全部来自**训练随机性 + 选点**，与结构无关——
  两者 RA 分布均值 20.70 vs 20.63（差 0.07 mm），19.26 只是原始 run 4 个测点中的最小值。详见 §12.4。
- **e** 三臂按每副本取最优步、每臂取更差副本；与 Track (render) 行的 best-of-4 **不同口径**，
  跨组比大小会偏向后者。臂间差异 ≤0.5 mm 低于本轮 1.1 mm 的分辨率，不可区分。见 §12。
- **\*** Latency：batch=1，同机逐个模型单独驻留测量（8 轮×300 次取最小），再按与 Full 的实测比值
  对齐到 Full=1.75 ms；**跨进程重测散布约 ±0.05 ms，故末位不具判别力**——PCA6 与 Full、
  以及三个渲染臂之间的差都在此散布内。FLOPs/step 为 thop 的 MAC 计数（未乘 2，以 CNN 主干为主，
  MANO 点撒渲染未计入）。
- **本表只列当前代码库里仍存在的方法。** KSGN（§9）、域随机化（§10–11）与 δ-trust 推理开关
  均已随回退到 EventHands-AbsRender 删除（`eval_track.py` 已无 `--delta-trust`），故不再入表；
  它们的历史读数保留在对应章节内，但**不应再与本表并列比较**。
- **本表所有 local 列都被 `zgz_local` 单条序列支配**：它是 val 里唯一的 local 序列，事件率仅
  692/50ms，是同主体 `zgz_global` 的 1/43、其他主体 local 的 1/4–1/7。§11 的留出主体诊断里，
  同协议下换成 ycy 做 val 时 local 只有 6.02 mm（该诊断所依赖的 LOSO / δ-trust 代码现已回退删除，
  数字仅作历史参考）。因此表内 27–30 mm 的 local 差异应读作"各方法在一条低事件率序列上的
  噪声输出之差"，**跨行 ±1 mm 的比较不具备判别力**。

#### 递推协议同协议对比（RA-MPJPE，含零训练融合）

| model | MPJPE RA | MPVPE RA | MPJPE abs | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|
| abs_full51 | 19.97 | 16.43 | 63.43 | 13.16 | 13.61 | 24.4 |
| track_delta51 | 93.38 | 68.78 | 1670.84 | 8.55 | 2.78 | **5554** |
| fuse α=0.5（零训练） | 19.32 | 16.16 | 62.55 | 8.26 | — | — |
| abs_render51（4ch-abs） | 20.08 | 16.41 | 63.27 | 11.85 | 15.87 | 28.1 |
| track_render51（4ch-delta） | 19.26 | 15.68 | 62.21 | 11.38 | **4.55** | 28.9 |

---

### 1. 实验设定

#### 1.1 数据与表示

- 布局：51D meta = `[t(3), R(3), residual45]`；12D 网络布局 = `[alpha6, t3, R3]`（与论文顺序不同，实现以仓库为准）。
- Val：`zgz_global` + `zgz_local`。
- MANO：`assets/mano_right.npz`；`betas` 用序列 GT；相机内参 `camera_K` 按序列（约 19 种），渲染时 ×0.375 缩放到 240×180。

#### 1.2 两种评测协议

**A. 逐帧绝对（abs）**  
每帧独立前向，`prev` 置零或忽略。用于 PCA6/Full 的主表精度。

**B. 递推 tracking（recursive）**  
`model/eval_track.py`：

- step = LNES 窗口 = 50ms；
- 每个 valid run：`prev = GT[start] + noise`（模拟 RGB 初始化）；
- 之后每步：`pred = model(LNES, prev)`，`prev ← pred`；
- GT 仅用于算误差，**不进网络**；
- `zgz_global` 约 1 段长序列；`zgz_local` 41 段，每段开头各 init 一次。

#### 1.3 Loss（51D）

`LAMBDA_POSE=450`（=45/0.1）、`LAMBDA_T=30000`、`LAMBDA_R=60`、`NORMALIZER=51`、`LOG10=true`。

---

### 2. 实验一：绝对位姿基线（PCA6 vs Full51）

#### 2.1 目的

检验 6D MANO PCA 是否限制单指局部姿态精度。

#### 2.2 配置与 ckpt

| | config | best ckpt |
|---|---|---|
| PCA6 | `configs/eventhands_abs_pca12.yaml` | `abs_pca12-step=11000-val_loss=4.3319.ckpt` |
| Full | `configs/eventhands_abs_full51.yaml` | `abs_full51-step=6000-val_loss=1.1009.ckpt` |

#### 2.3 结果（逐帧 abs）

| Model | dim | MPJPE RA | MPVPE RA | MPJPE abs | global / local MPJPE |
|---|---:|---:|---:|---:|---:|
| PCA6 | 12 | 19.61 | 15.15 | 59.59 | 10.99 / 30.00 |
| Full | 51 | 19.39 | 15.90 | 61.53 | 11.78 / 28.56 |

GT-PCA6 截断上限（oracle）：RA-MPJPE **6.85 mm**（保留能量 79.7%）。

#### 2.4 结论

Full 相对 PCA 仅改善 **0.22 mm** overall RA；两模型都远高于 6.9 mm 截断上限。  
**6D PCA 不是主瓶颈**，误差主要在 LNES→pose 回归器。

分报告：`outputs/hand_data51/report_abs_baseline.md`

---

### 3. 实验二：纯增量 Track(Δ)

#### 3.1 设计

- `PREDICT_DELTA=true`：`out = prev + Δ`
- `PREVPOS_EMBED=true`：51→64→51 MLP（零初始化末层）
- 训练噪声：iid 高斯（t=5mm, R/pose=0.05rad）
- config：`configs/eventhands_track_delta51.yaml`
- best（val_loss）：`track_delta51-step=2000-val_loss=0.0829.ckpt`
- 表格用 last 递推：local/global MPJPE **62.78 / 94.89**

#### 3.2 递推结果（best ckpt）

| | MPJPE RA | MPJPE abs | jitter all | global abs @65s |
|---|---:|---:|---:|---:|
| track_delta51 | 93.38 | 1670.84 | 8.55 | 5554 mm |

漂移：`zgz_global` 绝对误差约 231mm@5s → **5554mm@65s**（近似线性有偏积分）。

#### 3.3 失败根因

1. **开环积分不收缩**：`e_k = e_{k-1} + δ_k`，偏差线性累积。
2. **误差在输入空间不可见**：CNN 只看事件；prev 仅从输出端进入，没有“估计手 vs 事件位置”的空间错位信号。
3. **Exposure bias**：训练 iid 小噪声 vs 推理相关大误差。
4. **静止段零事件仍输出非零 Δ**（约 2.78 mm/步静态抖动 → ~56 mm/s 漂移）。

正面信号：抖动低于纯 abs（平滑收益真实）。

分报告：`report_track_delta.md`、`report_battle_analysis.md`

---

### 4. 实验三：零训练融合（Battle / Phase 1）

#### 4.1 公式

\[
x_k = (1-\alpha)\,(x_{k-1}+\hat\Delta_k) + \alpha\,\hat y^{\mathrm{abs}}_k
\]

冻住 abs_full51 + track_delta51，扫 α。

#### 4.2 结果（节选）

| combo | RA-MPJPE | jitter all |
|---|---:|---:|
| α=0（纯增量） | 93.81 | ~8.7 |
| α=0.2 | 19.72 | 6.79 |
| **α=0.5** | **19.32** | **8.26** |
| α=1（纯 abs） | 19.97 | ~13.1 |

裁决：**GO**。甜点 α∈[0.2, 0.5] 同时优于纯 abs 的精度与抖动。

结论：放弃无锚开环；升级为 **绝对锚 + 事件增量闭环**。

产物：`fuse_sweep.json` / `fuse_sweep.png`，`report_battle_analysis.md`

---

### 5. 实验四：Track(render) — render-and-compare

#### 5.1 设计

相对 Track(Δ) 的关键改动：

- `PREV_RENDER=true`：把 prev 姿态 MANO 前向 → z-buffer 点撒 → silhouette + inv-depth，与 LNES 拼成 **4 通道**；
- 保留 `prev + Δ` 输出；
- `ZERO_EVENT_GATE`：LNES 全零时强制 Δ=0；
- 混合噪声课程（小/大/相关）；
- warm-start 自 abs_full51（conv1 新通道零填充）。

config：`configs/eventhands_track_render51.yaml`  
训练：GPU 0–1，bsz 1024×2，LR 5.656e-3，15000 steps。

#### 5.2 Ckpt 选择（按递推 RA，非 val_loss）

| ckpt | val_loss | RA-MPJPE | local / global | jitter all | static |
|---|---:|---:|---:|---:|---:|
| **step=1000（选中）** | 0.8015 | **19.26** | 27.97 / 11.69 | 11.38 | **4.55** |
| step=2000 | 1.0657 | 20.04 | 28.35 / 12.82 | 11.73 | 5.43 |
| step=5000 | 0.9326 | 21.76 | 33.11 / 11.90 | 11.72 | 5.27 |
| last (15k) | — | 21.44 | 30.13 / 13.88 | 11.03 | 4.98 |

#### 5.3 判据

- 机制 GO：RA&lt;40 且 65s abs&lt;200 → **PASS**（19.26，28.9mm）
- 平滑性赢零训练 fuse（jitter&lt;10）→ **FAIL**（11.38）
- vs Full 递推：RA **−0.71 mm**，jitter **−1.78**

#### 5.4 Tracking 性质确认

是 tracking：姿态 GT 只用于每段开头初始化；之后只靠自身预测递推。`betas` / `camera_K` 为序列级上下文。

分报告：`report_track_render51.md` / `report_track_render51.png`

---

### 6. 实验五：AbsRender 消融（4ch-abs）

#### 6.1 目的

回答：Track(render) 变好，是因为 **多了渲染输入**，还是因为 **`prev+Δ` 设计**？

唯一变量：输出改绝对回归（与 Full 相同），输入仍 4ch（LNES+渲染 prev）。

| | 输出 absolute | 输出 prev+Δ |
|---|---|---|
| 输入 2ch | Full | Track(Δ) |
| 输入 4ch | **AbsRender（本实验）** | Track(render) |

config：`configs/eventhands_abs_render51.yaml`  
`PREDICT_DELTA=false`，`PREVPOS_EMBED=false`，`ZERO_EVENT_GATE=false`；其余（渲染、混合噪声、warm-start、双卡）同 track_render51。

#### 6.2 Ckpt 选择

| ckpt | val_loss | RA-MPJPE |
|---|---:|---:|
| **step=3000（选中）** | 1.2341 | **20.08** |
| step=6000 | 1.2366 | 20.16 |
| step=13000（val_loss best） | 1.2239 | 21.07 |
| last | — | 20.31 |

#### 6.3 2×2 结果（递推 RA-MPJPE）

| | output = absolute | output = prev + Δ |
|---|---:|---:|
| input 2ch | **19.97**（Full） | **93.38**（Track） |
| input 4ch | **20.08**（AbsRender） | **19.26**（Render） |

#### 6.4 结论

- AbsRender **≈ Full**（+0.11 mm）：只加渲染通道、绝对输出，**几乎不提升精度**。
- Track(render) 相对 Full 的增益（→19.26）主要来自 **`prev+Δ`（及零事件门控）**。
- 静止抖动：AbsRender **15.87** vs Render **4.55** vs Full 13.61 — 门控+delta 才压住静止漂移。
- 2ch-delta 爆炸 → **仅有 delta 也不够**，必须与 render-and-compare 联用。

**总判：渲染反馈输入 + delta 输出缺一不可；相对 Full 的精度收益主要来自输出侧 tracking 设计，不是“单纯多两通道”。**

分报告：`report_ablation_4ch.md` / `report_ablation_4ch.png`

---

### 7. 已删除 / 未保留实验

**DualClosed（联合训练双头）**：曾实现并训练（RA≈21.37，未超过零训练 fuse / Track(render)），后按计划清除代码、配置与产物。本报告不将其列入主对比表。

---

### 8. 配置与产物索引

| 实验 | config | 输出目录 | 关键 ckpt |
|---|---|---|---|
| PCA6 | `configs/eventhands_abs_pca12.yaml` | `outputs/hand_data51/abs_pca12/` | step=11000 |
| Full | `configs/eventhands_abs_full51.yaml` | `outputs/hand_data51/abs_full51/` | step=6000 |
| Track(Δ) | `configs/eventhands_track_delta51.yaml` | `outputs/hand_data51/track_delta51/` | step=2000 / last |
| Track(render) | `configs/eventhands_track_render51.yaml` | `outputs/hand_data51/track_render51/` | **step=1000** |
| AbsRender | `configs/eventhands_abs_render51.yaml` | `outputs/hand_data51/abs_render51/` | **step=3000** |

| 报告 | 路径 |
|---|---|
| 绝对基线 | `outputs/hand_data51/report_abs_baseline.md` |
| Battle / 融合 | `outputs/hand_data51/report_battle_analysis.md` |
| Track(Δ) | `outputs/hand_data51/report_track_delta.md` |
| Track(render) | `outputs/hand_data51/report_track_render51.md` |
| 4ch 消融 | `outputs/hand_data51/report_ablation_4ch.md` |
| **本总报告** | `outputs/hand_data51/report_full_experiment.md` |

关键代码：

- 模型：`model/model.py`（`PREV_RENDER` / `PREDICT_DELTA` / 零事件门控）
- 数据噪声：`model/fastevc.py`（`PREV_NOISE_MODE=mixed`）
- 训练：`model/train_abs.py`
- 递推评测：`model/eval_track.py`
- 绝对评测：`model/eval_abs.py`

---

### 9. 一句话总结

1. PCA 表示不是瓶颈。  
2. 纯开环增量 tracking 不可用（线性漂移）。  
3. 零训练 abs+Δ 融合证明闭环方向正确。  
4. Render-and-compare + delta 把 tracking 修到略优于 Full。  
5. 消融表明：**不能只靠多输入通道**；`prev+Δ`（及静止门控）才是相对 Full 的关键增益，且必须与渲染反馈联用。

---

## 7. SPA：语义部件锚定 delta 头

> 原文件：`outputs/hand_data51/report_spa.md`（原标题：SPA: semantic part-anchored delta head）

Two changes on top of `track_render51` (render-and-compare + delta, RA-MPJPE 19.26 mm):

1. **SPA head** — the rendered previous state additionally yields a 16-channel part
   histogram and the 21 projected joints. Those turn the ResNet feature pyramid into
   21 semantically identified joint nodes (mask pooling + anchor sampling + joint-ID
   embedding + current local rotation), which are propagated over the MANO kinematic
   tree and decoded into a per-joint pose delta. All final layers are zero-initialized,
   so a warm-started model starts out bit-identical to the baseline.
2. **Per-node 2D displacement loss** — node `j` predicts
   `pi(FK(y))_j - pi(FK(prev))_j`, supervised with a Huber loss. Targets are produced
   inside the model from the same projection used by the renderer, so the dataset is
   untouched.

Protocol: identical to [report_track_render51.md](#report_track_render51) —
`model/eval_track.py`, step = 50 ms, GT init + `TRACK.PREV_NOISE` (scale 1.0), seed 0,
val = `zgz_global` + `zgz_local`. Checkpoints are selected by **recursive RA-MPJPE**,
not by teacher-forced `val_loss`.

### 1. Headline

| model | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all | jitter static | global abs @~65s |
|---|---:|---:|---:|---:|---:|---:|---:|
| abs_full51 | 19.97 | 16.43 | 63.43 | 29.12 / 12.03 | 13.158 | 13.608 | 24.4 |
| fuse α=0.5 (zero-train) | 19.32 | 16.16 | 62.55 | 27.89 / 11.87 | 8.261 | — | — |
| track_render51 (baseline) | 19.26 | 15.68 | 62.21 | 27.97 / 11.69 | 11.384 | 4.545 | 28.9 |
| **SPA (8k run, last) ← selected** | **17.91** | **14.61** | 63.04 | 25.82 / 11.04 | 10.161 | 9.003 | **19.4** |
| SPA (8k run, step1000) | 18.52 | 15.07 | 60.01 | 26.70 / 11.42 | 10.746 | 4.349 | 28.8 |
| SPA (15k replicate, step2000) | 18.40 | 15.11 | 58.22 | 27.02 / 10.91 | 10.579 | 6.046 | 25.3 |

Selected: `outputs/hand_data51/track_render51_spa/last.ckpt` — **RA-MPJPE 17.91 mm,
−1.35 mm (−7.0%) vs the render+delta baseline**, and the best number recorded on this
dataset so far. MPVPE RA improves by 1.07 mm and both categories improve
(local 27.97 → 25.82, global 11.69 → 11.04).

### 2. Pre-registered criteria

| criterion | target | result | verdict |
|---|---|---|---|
| recursive RA-MPJPE | ≤ 18.7 (≥0.5 mm and ≥3%) | 17.91 (−1.35 mm, −7.0%) | **PASS** |
| jitter (all) | ≤ 10 | 10.161 (from 11.384) | **FAIL** (narrow) |
| 65 s absolute drift | not worse than 28.9 | 19.4 | **PASS** |
| zero-event gate | empty LNES ⇒ `pred == prev` | unit test `test_spa_zero_event_gate_unchanged` | **PASS** |

The jitter gate — the one the baseline also missed — is still missed, but by 0.16 mm/step
instead of 1.38. Every SPA checkpoint sits in 10.03–10.75 against the baseline's 11.38, so
the direction is consistent; nothing here crosses the threshold. Note the trade-off in the
selected checkpoint: **static** jitter rises (4.545 → 9.003) even as all-frame jitter falls.
The selected model is more willing to move, which is what buys the drift and RA gains; the
earlier checkpoints (step1000: RA 18.52, static jitter 4.349) keep the baseline's static
stability while still improving RA, and are the better pick if a static hand must not
shimmer.

Absolute MPJPE is 63.04 vs 62.21, i.e. 0.8 mm worse. The drift curve explains it: the
selected model is better at the sequence end (65 s: 19.4 vs 28.9) but worse mid-sequence
(35–55 s), so the average is a wash while the endpoint — the thing the drift gate measures
— clearly improves.

![drift](assets/report_spa.png)

### 3. Ablations

All ablations were trained for 2000 steps, because both the baseline and the SPA runs
reach their best recursive RA within the first two validation points. Compare them against
the full SPA restricted to the same budget (**18.52**), not against the 8000-step 17.91.

| variant | changed key | best RA | ΔRA vs full SPA | jitter all | jitter static |
|---|---|---:|---:|---:|---:|
| full SPA (≤2000 steps) | — | 18.52 | — | 10.746 | 4.349 |
| mask only | `PART_HEAD_ANCHOR: false` | 18.38 | −0.14 | 10.586 | 3.553 |
| λ_2D = 60 | `LAMBDA_J2D: 60` | 18.67 | +0.15 | 11.476 | 3.305 |
| λ_2D = 600 | `LAMBDA_J2D: 600` | 18.80 | +0.28 | 11.096 | 5.270 |
| anchor only | `PART_HEAD_MASK: false` | 19.00 | +0.48 | 10.277 | 9.253 |
| no graph | `GRAPH_LAYERS: 0` | 19.86 | +1.34 | 10.407 | 8.515 |
| no aux loss | `LAMBDA_J2D: 0` | 20.36 | +1.84 | 11.625 | 8.025 |

Three conclusions, in order of how far they clear the noise floor (§4):

- **The auxiliary 2D loss is not optional — it is what makes the head trainable.** Without
  it the SPA variant lands at 20.36, i.e. *worse than the 19.26 baseline it started from*.
  The mechanism is visible in the initialization: the pose heads are zero-initialized, so
  at step 0 the gradient reaching the node trunk through the pose loss is exactly zero, and
  the trunk only starts learning after the head's last layer drifts off zero. The 2D loss
  attaches directly to the node features and supplies gradient from step 1
  (`test_spa_aux_loss_trains_the_node_trunk` asserts exactly this). What is resolved is
  only *zero vs nonzero*: the three nonzero settings (60 → 18.67, 200 → 18.52,
  600 → 18.80) all sit inside the ±0.4 mm noise floor of §4 and are indistinguishable, so
  λ needs no tuning beyond "not zero" and 200 is not a demonstrated optimum.
- **Tree propagation is required.** Dropping it costs 1.34 mm and also lands below the
  baseline. Per-joint features that cannot exchange information along the kinematic chain
  are worse than no per-joint features at all — consistent with the derivation, where the
  propagation stands in for the tree-sparse `(J^T W J)^-1` that couples joints sharing an
  ancestor chain.
- **Part-mask pooling carries the spatial gain; the joint anchor does not.** Mask-only
  (18.38) matches full SPA within noise, while anchor-only (19.00) gives up most of the
  improvement. The useful signal is a joint's *pixel support region*, not the feature at
  its projected center — which is the PARE result rather than a keypoint-sampling result.
  Anchor sampling can be dropped for a slightly cheaper head; it is kept on in the default
  config because it costs 16k parameters and the two are statistically tied.

Two node-feature ingredients were never isolated and are therefore unproven, not proven:
the **21-way joint-ID embedding** and the **per-node current local rotation** concatenated
into `node_in`. Both are carried in every run above, so nothing here says whether the node
identity has to be an explicit learned code or is already implied by the mask/anchor
geometry. Ablating them needs two more 2000-step runs.

### 4. Run-to-run variance

`spa` (8000 steps) and `spa_long` (15000 steps) share seed, data order and LR schedule and
differ only in `MAX_STEPS`, so their first 8000 steps should coincide. They do not:
`spa` step2000 = 18.83 vs `spa_long` step2000 = 18.40. The scatter-based rasterizer and
part histogram are non-deterministic under floating-point accumulation order, so the two
runs diverge and act as independent replicates.

Practical consequence: **treat ±0.4 mm as the noise floor for RA-MPJPE.** The two runs give
best-checkpoint RA of 17.91 and 18.40, so the honest claim is a **0.9–1.35 mm (4.5–7.0%)
improvement** over the baseline, not a point estimate of 1.35. The ablation deltas for
`no aux` (+1.84) and `no graph` (+1.34) are well outside this floor; `mask only` vs full
SPA (−0.14) is inside it and is reported as a tie.

Training longer is not the answer: the 15000-step replicate peaked at step 2000 (18.40) and
regressed to 19.59 by the end. Both runs and the baseline agree that this setup saturates
early.

### 5. Implementation notes

- **Identity at initialization.** `test_spa_identity_at_init` and
  `test_spa_warm_start_from_render_checkpoint` assert that a warm-started SPA model
  reproduces the baseline output exactly (`atol=1e-5`, measured difference 0.0). This
  requires the explicit ResNet decomposition in `_backbone_features` to be bit-equivalent
  to `self.rn(x)`, which `test_spa_backbone_decomposition_equals_resnet` checks. Warm start
  loaded all 140 baseline tensors.
- **Cost.** 11.209 M → 11.479 M parameters (+269 k, +2.40%). Peak training memory at
  batch 1024 on one L20: 11.40 GB → 12.00 GB (+5.3%, within the +10% budget).
  Throughput was ~3.6–4.0 k samples/s on 2 GPUs, unchanged from the baseline.
- **The part histogram is never materialized densely.** Visible vertices are scattered
  straight into a `(B, 16, 23, 30)` accumulator at stride 8, so the semantic channels cost
  ~45 MB at batch 1024 instead of a full-resolution one-hot volume.
- **`grid_sample` has no bf16 CUDA kernel** in torch 2.1, so anchor sampling runs in fp32
  and casts back. This was the only bf16 incompatibility encountered.
- **`val_loss` stays pose-only** (the auxiliary term is added to the training objective
  before the `log10`, but is logged separately as `train_j2d_loss` / `val_j2d_loss`), so
  `val_loss` and the checkpoint monitor remain comparable with all earlier runs. Note that
  teacher-forced `val_loss` is again a poor guide: the selected model has the *worst*
  `val_loss` of the run while having the best recursive RA.
- **`_aux_out` is consumed on read**, so cached activations can never be counted into two
  optimizer steps (`test_spa_aux_out_is_consumed_once`).

### 6. Reproduce

```bash
# main run (2 GPUs)
CUDA_VISIBLE_DEVICES=0,2 NCCL_P2P_DISABLE=1 python model/train_abs.py \
    --config configs/eventhands_track_render51_spa.yaml

# closed-loop scoring of every checkpoint
python model/eval_track.py --config configs/eventhands_track_render51_spa.yaml \
    --ckpt outputs/hand_data51/track_render51_spa/last.ckpt --step-ms 50 \
    --out-dir outputs/hand_data51/track_render51_spa/eval_last

# ablation sweep (trains + scores all variants)
CUDA_VISIBLE_DEVICES=0,2 tools/run_spa_ablations.sh
```

### 7. Open items

- The jitter-all gate (≤ 10) is still unmet at 10.16, and the selected checkpoint trades
  static stability (4.545 → 9.003) for accuracy. A temporal smoothness penalty on the
  predicted delta, or the zero-event gate widened to a low-event gate, is the obvious next
  lever and is orthogonal to this change.
- Anchor sampling is statistically inert; if head cost ever matters it is the first thing
  to remove.
- Multi-step rollout training (the exposure-bias fix deferred from
  [report_battle_analysis.md](#report_battle_analysis)) remains untouched and orthogonal.
- Checkpoint selection still relies on `val_loss` top-3 + last for *which* checkpoints get
  saved, then recursive RA to choose among them. Since `val_loss` and recursive RA are
  anti-correlated here, saving on a fixed step grid would sample the trajectory better.

---

## 8. SPA round 2：EDD + 任务空间关节损失

> 原文件：`outputs/hand_data51/report_spa_round2.md`（原标题：SPA round 2: evidence-damped delta + task-space joint loss）

Two changes on top of SPA ([report_spa.md](#report_spa), selected RA-MPJPE **17.91 mm**), plus one deletion that round 1 already measured as inert:

1. **Delete anchor sampling.** Round 1: mask-only 18.38 vs full SPA 18.52 (Δ −0.14, inside ±0.4 mm noise). The `grid_sample` branch, `PART_HEAD_ANCHOR` / `PART_HEAD_MASK` switches, and the `spa_anchor_only` / `spa_mask_only` configs are gone. `joints2d` stays because the auxiliary 2D loss still needs `j2d_prev`.
2. **EDD — per-joint event-evidence damping.** Occupancy of the LNES, pooled onto the same stride-8 grid as the part masks, gives a scalar evidence \(E_j\) per articulated joint. The local-45 update is multiplied by \(c_j = \mathrm{clamp}(1 - a_j\exp(-E_j/\tau), 0, 1)\) with \(a_j\) zero-initialized, so a warm start is a no-op. Root / translation stay under the existing global empty-LNES gate.
3. **Task-space FK joint loss.** Root-aligned smooth-L1 on the 21 OpenPose joints produced by the same differentiable MANO call the renderer uses. First-order kinematics: \(\|\mathrm{FK}_{RA}(\hat\theta)-\mathrm{FK}_{RA}(\theta^*)\|^2 \approx \Delta\theta^\top J^\top J\,\Delta\theta\). Zero new parameters; `LAMBDA_J3D=0` skips the FK.

Protocol unchanged: `model/eval_track.py`, step = 50 ms, GT init + `TRACK.PREV_NOISE` (scale 1.0), seed 0, val = `zgz_global` + `zgz_local`. Checkpoints are selected by **recursive RA-MPJPE**. This round saved a **fixed 1000-step grid** (`TRAIN.SAVE_EVERY_N_STEPS`) because round 1 showed `val_loss` anti-correlated with closed-loop RA.

Hardware: 4× L20 (`CUDA_VISIBLE_DEVICES=4,5,6,7`), global batch 2048 (4×512), same LR as SPA. Two 8000-step copies of the identical config (CUDA `scatter_add` non-determinism = independent replicates). Claims use the **worse** of the two selected checkpoints.

`LAMBDA_J3D` calibration (pre-registered, one shot, no grid): smoke at 30 gave the J3D term 0.4% of the total loss, outside [10%, 60%]. One bump to 3000 landed at **54%**, inside the band; no second adjustment.

### 1. Headline

| model | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all | jitter static | global abs @~65s | global drift mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| track_render51 (baseline) | 19.26 | 15.68 | 62.21 | 27.97 / 11.69 | 11.384 | 4.545 | 28.9 | 34.1 |
| **SPA (8k, last) ← still selected** | **17.91** | **14.61** | 63.04 | **25.82 / 11.04** | 10.161 | 9.003 | **19.4** | 34.9 |
| spa2 copy A (step 6000) | 18.40 | 15.17 | 71.31 | 27.76 / 10.27 | 10.856 | 4.698 | 44.0 | 35.8 |
| spa2 copy B (step 3000) | 18.27 | 15.14 | 63.66 | 26.80 / 10.86 | 10.365 | 10.546 | 27.7 | 27.7 |
| spa2 **worse of two** (protocol) | 18.40 | 15.17 | 71.31 | 27.76 / 10.27 | 10.856 | 10.546* | 44.0 | 35.8 |

\*jitter-static protocol takes the worse replica's own selected ckpt (copy B, 10.546), not copy A's 4.698.

**Round 2 does not replace SPA.** The worse replica is 0.49 mm worse than 17.91, which is outside the ±0.4 mm noise floor. Local RA, the quantity J3D was supposed to move, is also worse (25.82 → 27.76 / 26.80). Copy A's 65 s absolute error **regresses** to 44.0 mm (SPA: 19.4).

![drift](assets/report_spa_round2.png)

### 2. Pre-registered criteria

Worse-of-two against SPA 17.91 / jitter 10.161 / static 9.003 / 65 s 19.4.

| criterion | target | worse replica | verdict |
|---|---|---|---|
| recursive RA-MPJPE | ≤ 17.5 (beyond noise vs 17.91) | 18.40 | **FAIL** |
| jitter (all) | ≤ 10 | 10.856 | **FAIL** |
| jitter (static) | ≤ 5.0 | 10.546 (copy B) / 4.698 (copy A) | **FAIL** (inconsistent across copies) |
| zgz_global 65 s abs | ≤ 19.4 | 44.0 | **FAIL** |
| drift-bucket mean | not worse than SPA (34.9) | 35.8 | **FAIL** (tie within noise, not an improvement) |
| empty LNES ⇒ `pred == prev` | unit test | `test_edd_zero_event_gate_still_holds`, `test_spa_zero_event_gate_unchanged` | **PASS** |
| val_j3d vs closed-loop RA correlates better than val_loss | side prediction | Spearman **+0.84** vs **−0.07** (n=16, 1k-step grid, both copies) | **PASS** (the one clear positive) |

Decision tree from the plan: EDD ablation did not hurt RA beyond noise; J3D ablation did not hurt RA beyond noise; **neither component is delivered as a method claim.** SPA `last.ckpt` remains the selected model.

### 3. Why EDD did not fire

`gate_logit` on every scored spa2 / spa2_rep2 / spa2_noj3d checkpoint is \(\approx -3\times 10^{-4}\). After `clamp(0, 1)` that is **identically zero**, so \(c_j \equiv 1\) at every joint. EDD is a no-op at the selected points, not a weak gate.

Mechanism: zero-init + `clamp` makes "turn the gate on" a one-sided request. Teacher-forced training always has a noisy `prevpos` that the pose / J3D terms want to correct, so the gradient on \(a_j\) is negative (keep full update). The parameter walks slightly below zero and stays there. The identity warm-start convention that made SPA loadable is exactly what prevented this gate from ever leaving identity.

Consequence: `spa2_nogate` (J3D, no EDD) at 2000 steps is 19.61 vs spa2@2000 = 19.30 (Δ +0.31, inside noise), which is what a no-op gate predicts. Jitter-all never crossed 10 because the damper that was supposed to suppress unsupported updates never engaged.

### 4. Ablations (2000-step budget)

Compare against spa2 restricted to the same budget (**19.30** at step 2000). Copy B at step 1000 is 18.70 — that 0.76 mm copy-to-copy gap at step 1000 is larger than the round-1 ±0.4 mm floor, so 2000-step deltas inside ~0.4 mm are reported as ties.

| variant | changed key | best RA ≤2000 | Δ vs spa2@2000 | jitter all | jitter static |
|---|---|---:|---:|---:|---:|
| spa2 (copy A, ≤2000) | — | 19.30 | — | 11.142 | 7.985 |
| spa2 (copy B, ≤2000) | replicate | 18.70 | −0.60 | 10.733 | 8.220 |
| no J3D (EDD only) | `LAMBDA_J3D: 0` | 19.02 | −0.28 | 10.361 | 5.763 |
| no EDD (J3D only) | `EVENT_GATE_PER_JOINT: false` | 19.61 | +0.31 | 10.894 | 5.106 |
| no joint-ID embedding | `PART_HEAD_ID_DIM: 0` | 19.25 | −0.05 | 10.527 | 5.122 |
| no prev local rotation | `PART_HEAD_PREV_AA: false` | 19.31 | +0.01 | 10.535 | 9.006 |

- **EDD and J3D are both inside noise at this budget.** Neither is isolated as the cause of the 8000-step regression vs SPA.
- **Joint-ID embedding and per-node prev local rotation are inert** (Δ −0.05 / +0.01). Round 1 left them untested; this round tests them. They can be dropped for a slightly smaller `node_proj` without changing the method claim. An 8000-step pruned confirmation was **not** run: the parent spa2 config already misses every closed-loop gate against SPA, and removing two noise-floor features cannot recover 0.49 mm. A pruned yaml is `configs/eventhands_track_render51_spa2_pruned.yaml` (ID_DIM 0, PREV_AA false) for a later run if needed.

### 5. Side prediction: val_j3d vs val_loss vs closed-loop RA

n = 16 checkpoints on the 1000-step grid of the two 8000-step copies (last.ckpt omitted as a duplicate of step 8000).

| selector | Spearman vs closed-loop RA | Pearson | checkpoint it would pick | that ckpt's RA |
|---|---:|---:|---|---:|
| `val_loss` (min) | **−0.07** | −0.04 | copy B step 1000 | 18.70 |
| `val_j3d` (min) | **+0.84** | +0.87 | copy B step 1000 | 18.70 |
| closed-loop RA (oracle) | 1 | 1 | copy B step 3000 | 18.27 |

`val_j3d` is the metric the derivation said it would be: same space as the eval, monotonically related to recursive RA. `val_loss` remains useless for selection (copy A step 6000 has the **worst** `val_loss` of that run, 1.31, and the **best** RA of that run, 18.40 — the same anti-pattern as round 1).

This does **not** rescue the RA gate. The best `val_j3d` still lands at 18.70, 0.79 mm behind SPA. Metric alignment improved the *monitor*, not the *tracker*. Teacher-forced FK of a one-step prediction is not the compounded closed-loop trajectory; J3D cannot substitute for on-policy prev (the OPR item removed from this round).

### 6. Implementation notes

- **42 unit tests pass**, including EDD identity / monotonicity / empty-LNES, J3D zero-loss / translation invariance / gradient / render consistency, and single-key diffs of the four ablation yamls.
- **Identity at `a_j = 0`.** `test_edd_identity_at_zero_init` loads SPA weights into spa2 and matches outputs at `atol=1e-6`. After training, `a_j` never left 0, so the trained gate is still that identity map.
- **Single FK source.** `_fk` / `_fk_joints21` is shared by the renderer (still `no_grad` + fp32 + chunk) and the J3D term (`test_j3d_fk_matches_direct_mano_call`, `test_j3d_render_projection_still_consistent`).
- **J3D is skipped when `LAMBDA_J3D=0`**, so `spa2_noj3d` does not pay for a differentiable LBS (`test_j3d_disabled_returns_none`). No new parameters, no DDP unused-param issue.
- **Cost.** 11.5 M parameters (gate_logit is 15 scalars). Peak memory 6.12 GB at batch 512 / GPU (SPA was 12.0 GB at 1024; global batch held at 2048). Smoke throughput 5.5–7.3 k samples/s on 4 GPUs.
- **Fixed grid.** `train_abs.py` writes every 1000 steps when `SAVE_EVERY_N_STEPS` is set; the old top-3-by-`val_loss` path is unchanged for configs that omit the key. This is the one infrastructure change that earned its keep: without it, copy A's RA-best (step 6000, `val_loss` 1.31, worst of the run) would not have been saved.

### 7. Full closed-loop grid

| run | ckpt | RA | MPVPE RA | abs | local RA | global RA | jitter all | jitter static | 65s abs | drift mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spa2 | 1000 | 19.46 | 15.87 | 61.80 | 29.15 | 11.03 | 11.118 | 8.044 | 27.3 | 32.4 |
| spa2 | 2000 | 19.30 | 15.95 | 61.63 | 28.96 | 10.91 | 11.142 | 7.985 | 24.4 | 30.6 |
| spa2 | 3000 | 18.53 | 15.48 | 65.18 | 27.38 | 10.85 | 10.628 | 4.975 | 29.2 | 42.4 |
| spa2 | 4000 | 19.28 | 15.92 | 61.56 | 28.82 | 10.99 | 10.449 | 3.431 | 23.6 | 33.1 |
| spa2 | 5000 | 18.98 | 15.67 | 62.25 | 28.29 | 10.90 | 10.883 | 6.818 | 21.4 | 34.4 |
| spa2 | **6000** | **18.40** | 15.17 | 71.31 | 27.76 | 10.27 | 10.856 | 4.698 | 44.0 | 35.8 |
| spa2 | 7000 | 18.98 | 15.68 | 62.21 | 28.88 | 10.38 | 10.949 | 7.232 | 23.5 | 30.9 |
| spa2 | 8000 / last | 19.48 | 16.02 | 62.42 | 29.66 | 10.64 | 10.737 | 5.028 | 24.2 | 34.3 |
| spa2_rep2 | 1000 | 18.70 | 15.49 | 62.48 | 27.76 | 10.82 | 10.733 | 8.220 | 24.9 | 31.1 |
| spa2_rep2 | 2000 | 19.54 | 16.06 | 60.58 | 28.75 | 11.53 | 10.923 | 10.229 | 22.9 | 31.2 |
| spa2_rep2 | **3000** | **18.27** | 15.14 | 63.66 | 26.80 | 10.86 | 10.365 | 10.546 | 27.7 | 27.7 |
| spa2_rep2 | 4000 | 19.04 | 15.49 | 60.79 | 28.16 | 11.12 | 10.239 | 7.399 | 21.9 | 33.9 |
| spa2_rep2 | 5000 | 19.44 | 16.00 | 63.87 | 29.03 | 11.11 | 10.593 | 5.697 | 26.1 | 28.9 |
| spa2_rep2 | 6000 | 19.32 | 16.21 | 63.08 | 26.68 | 12.93 | 10.293 | 4.626 | 26.4 | 36.0 |
| spa2_rep2 | 7000 | 20.29 | 16.57 | 63.73 | 29.95 | 11.90 | 10.772 | 5.518 | 26.2 | 32.3 |
| spa2_rep2 | 8000 / last | 19.54 | 15.97 | 61.08 | 30.01 | 10.45 | 10.906 | 6.978 | 25.8 | 32.8 |

Early saturation repeats: both copies peak at or before step 6000 and then wander. Training longer is still not the answer.

### 8. Reproduce

```bash
# tests
/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python -m pytest tests/test_abs_pose.py -q

# two 8000-step copies + 2000-step ablations on GPU 4-7
GPUS=4,5,6,7 tools/run_spa2_round2.sh

# closed-loop scoring of every checkpoint on the grid
GPUS=4,5,6,7 tools/eval_spa2_round2.sh

# table + Spearman
python tools/collect_round2.py
```

Selected model is unchanged: `outputs/hand_data51/track_render51_spa/last.ckpt`.

### 9. What this round actually bought

- **Anchor sampling is gone.** That was already justified in round 1; the code now matches the evidence.
- **A fixed checkpoint grid**, which is how copy A's RA-best (worst `val_loss`) got scored at all.
- **`val_j3d` is a usable monitor** (Spearman +0.84 vs closed-loop RA). Use it for future selection. Do not confuse a better monitor with a better tracker.
- **EDD as parameterized here is unlearnable under teacher forcing.** A later attempt needs a parameterization that can leave identity (e.g. a small positive init, or a softplus) *or* on-policy prev — the OPR item this round was told not to implement.
- **J3D did not move closed-loop local RA.** Teacher-forced one-step FK is the wrong trajectory. The remaining gap to the 6.85 mm oracle on local joints is still an exposure-bias / compounding problem, not a missing loss term in parameter vs task space.

Open: on-policy prev (OPR / DAgger, deferred), a gate that is allowed to turn on, and the still-unmet jitter-all ≤ 10 gate on the SPA checkpoint itself (10.16).

---

## 9. KSGN：LBS 语义关节 GN 头重建（未复现 17.91，头被判定惰性）

回退到 `track_render51` 基线后，按 §7 的文字描述重建 LBS 权重驱动的语义关节头（代码已随回退删除，
只能依据报告重写）。**结论：门槛未过，头是惰性的，17.91 mm 没有复现。** 本节记录推导、实现、
两副本结果与三项判别性诊断，以便后续轮次不再重复这条路径。

### 1. 数学骨架

**(a) delta 是摊销的阻尼 Gauss–Newton 步。** 状态 \(x=(t,\theta)\in\mathbb R^{51}\)，事件帧 \(E_t\)
与上一状态渲染 \(R(x_{t-1})\) 的对齐残差 \(r\)，一步阻尼高斯牛顿为

\[ \delta^\* = \arg\min_\delta \|r(x_{t-1}\oplus\delta;E_t)\|_W^2 + \lambda\|\delta\|^2
            = -(J^\top W J+\lambda I)^{-1}J^\top W\, r . \]

网络学习 \((E_t,R(x_{t-1}))\mapsto\delta\) 这个映射，即把迭代优化摊销进一次前向
（DeepIM ECCV'18、se(3)-TrackNet IROS'20、BA-Net ICLR'19、DROID-SLAM NeurIPS'21）。

**(b) LBS 权重矩阵给出 \(J\) 的支撑结构——这就是"语义"的本体。** MANO 蒙皮
\(v_i=\sum_j W_{ij}G_j(\theta)\bar v_i\)（`ManoLayer.weights`，778×16，[model/mano_layer.py](../model/mano_layer.py) 第 64/151 行），
链式法则给出 \(\partial v_i/\partial\theta_j\neq 0\) 当且仅当顶点 \(i\) 蒙皮到关节 \(j\) 的子树。
于是 \((J^\top W r)_j\) 的像素支撑恰是部件掩码
\(M_j=\{u:\arg\max_{j'}W_{i(u)j'}=j\}\)——**对渲染出的部件掩码做特征池化，等价于按 Jacobian 的行结构
收集每个关节的梯度统计量**（与 PARE ICCV'21 的部件注意同一结论，无需任何额外标注）。

**(c) 运动链传播 ≈ 法方程逆的树截断。** \(H=J^\top WJ+\lambda I\) 的填充沿运动树（共享祖先链的关节耦合），
Neumann 级数 \(H^{-1}=\lambda^{-1}\sum_k(-\lambda^{-1}\tilde H)^k\) 的 K 阶截断由 K 层归一化邻接
\(\hat A=D^{-1/2}(A+I)D^{-1/2}\) 消息传递实现（Kipf & Welling ICLR'17；树上高斯 BP 的精确性见 Weiss & Freeman）。取 K=2。

**(d) 辅助逐节点 2D 位移损失。** 姿态头零初始化保证暖启动逐位等价，但也使节点主干在第 0 步收不到姿态梯度，
辅助项 \(\mathcal L_{2d}=\sum_j\rho\big(\hat d_j-[\pi(FK(y))_j-\pi(FK(x_{t-1}))_j]/(W,H)\big)\)
（Huber）从第 1 步起向节点特征供梯度。

### 2. 实现与自洽性验证

- [model/model.py](../model/model.py)：`_register_hand_topology`（`weights` argmax → `vert_part`、
  指尖继承末端关节、21 节点 OpenPose 序、\(\hat A\)、node→part 行归一化散射、每节点 prev 局部旋转列索引）、
  `_render_chunk` 的部件直方图分支（stride-8 `scatter_add`，深度缓冲做可见性 z 测试，不物化稠密 one-hot）、
  `_part_nodes`（掩码加权池化 + `grid_sample` fp32 锚点采样 + 21 维 ID 嵌入 + prev 局部旋转）、
  2 层 `_GraphLayer`、零初始化 `local_head`/`root_head`、`aux_j2d_head` 与 `_aux_loss`。
- `_backbone_features` 显式展开 resnet18 以取出 stride-8 特征，`test_spa_backbone_decomposition_equals_resnet`
  断言与 `self.rn(x)` **逐位相等**；`test_spa_identity_at_init_matches_render_baseline` 断言暖启动后与基线
  输出逐位相同（实测差 **0.0**，暖启动载入全部 140 个张量）。
- 成本：11.209 M → 11.299 M 参数（+89 k，+0.80%），batch 1024/GPU 峰值 12.00 GB，4.24 k samples/s（2 卡），
  与 §7 记录的 12.00 GB / 3.6–4.0 k 一致。
- 测试：28 项全过（其中 KSGN 相关 11 项，含拓扑正确性、直方图稀疏性、零事件门、aux 目标一致性、
  `_aux_out` 只消费一次、辅助损失确实给主干供梯度而姿态损失在第 0 步不供）。

### 3. 结果：门槛未过

协议同 §7：`model/eval_track.py`，step = 50 ms，GT init + `TRACK.PREV_NOISE`（scale 1.0），seed 0，
val = `zgz_global` + `zgz_local`；按**递推 RA-MPJPE** 而非 `val_loss` 选点；每 1000 步固定网格存盘。
两副本（GPU 4,5 与 6,7，互异 `MASTER_PORT`，`NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`）各 8000 步。

| model | MPJPE RA | MPVPE RA | MPJPE abs | local / global RA | jitter all |
|---|---:|---:|---:|---:|---:|
| track_render51 基线（= 暖启动起点） | **19.26** | 15.68 | 62.21 | 27.97 / 11.69 | 11.384 |
| KSGN 副本 1 最优（step8000） | 19.26 | 15.79 | 62.90 | 28.78 / 10.99 | 11.293 |
| KSGN 副本 2 最优（step4000） | 19.50 | 15.74 | 64.92 | 29.02 / 11.24 | 11.363 |
| §7 历史记录（不可复现） | 17.91 | 14.61 | 63.04 | 25.82 / 11.04 | 10.161 |

预注册门槛：worse-of-two ∈ [17.5, 18.4] 复现历史；> 18.8 停下排查、不交付。
**实测 worse-of-two = 19.50，与基线 19.26 在 ±0.4 mm 噪声地板内打平 → FAIL，触发 STOP。**

效率（同 §6 口径，thop + batch-1 延迟按与 Full 实测比值对齐到 Full = 1.75 ms，产物
`outputs/hand_data51/track_render51_spa/efficiency.json`）：**params 11.30 M，FLOPs 1.657 G/step，latency 6.37 ms**，
对应 local/global MPVPE RA = 23.87 / 8.77。同一次测量里 Track(render) 复测 4.53 ms（发表值 4.96 ms），
所以 KSGN 的净开销是 **≈1.40× Track(render)**：+0.09 M 参数、+2 MFLOPs 是小项，真正的成本是部件直方图渲染分支
与两层树传播这两段延迟。这一行已填入 §6 总表末行。

两副本的整条轨迹都先变差再回升（副本 1：20.85 → 20.39 → 20.66 → 20.66 → 20.27 → 19.99 → 19.54 → 19.26），
终点只是回到起点。注意基线自身在同一配方下也是随训练退化的（step1000 19.26 → step2000 20.04 →
step5000 21.76 → last 21.44），所以"训练越久闭环越差"是这个设置的既有性质，不是本改动引入的。

![drift](assets/report_ksgn.png)

### 4. 判别性诊断（为什么没涨）

**(i) 头是惰性的 —— 用同一 checkpoint 做推理期消融，而不是比较两次独立训练。** 姿态解码器是零初始化的，
把 `local_head`/`root_head` 清零就恰好还原暖启动时的恒等映射，于是同一权重可以带/不带关节 delta 各评一次
（[tools/ablate_part_head.py](../tools/ablate_part_head.py)）。在最优 checkpoint 上：

| 量 | 数值 |
|---|---:|
| 关节 delta RMS / fc delta RMS | **0.079** |
| RA（头激活） | 19.26 |
| RA（头清零） | 19.42 |
| 头的贡献 | **+0.16 mm（噪声地板内）** |

即 §7 声称由这个头带来的 −1.35 mm，在本次重建里根本不存在：头只贡献了 0.16 mm，且方向上把 global
略微改好、local 略微改差。**任何"图传播 +1.34 mm / 辅助损失 +1.84 mm"的消融结论都不能用两次独立训练
的差值来支撑**——那个差值里混着 ±0.4 mm 噪声与训练轨迹差异；推理期消融才是干净的工具。

**(ii) 不是特征退化。** 怀疑过 stride-8 下手指部件没有像素支撑，实测否定：手掌轮廓 606 px（约 9.5 个
stride-8 cell 的面积），每个部件平均点亮 3–5 个 cell（腕部 25 个），没有任何 (样本, 部件) 对为空；
池化后 21 个节点特征的两两余弦相似度均值 0.546、最小 0.066，近重复对平均 0.5/20。
**节点特征是有区分度的，是加性头没有动机去接管 fc 通路已经拟合的那部分**——fc（512→51）已经直接回归
整个 delta，零初始化的旁路头在梯度上没有理由长大。

**(iii) 没有触及根因。** 运动增益分解（[tools/measure_motion_gain.py](../tools/measure_motion_gain.py)，
对闭环 \(\Delta p = g\,\Delta g + n\) 做最小二乘）：

| checkpoint | local g | local 正交能量 | global g | global 正交能量 |
|---|---:|---:|---:|---:|
| track_render51 基线 | 0.145 | 99.6% | 1.005 | 39.4% |
| KSGN step8000 | 0.160 | 99.4% | 0.966 | 40.4% |

与早期闭环诊断 §4 的历史读数（g=0.19、99.4% 正交）
一致，这同时验证了工具本身。**手指增量仍有 99.4% 的能量与真值正交**：解码端换成语义关节结构没有、
也不可能改变这一点，因为窗口内的运动方向信息在 LNES 输入处就已经被丢掉了（TBIN 待办）。

### 5. 过程中修掉的一个真 bug（值得留档）

第一轮训练 step1000 直接掉到 20.85，排查发现进度条上 `train_loss`（姿态项）= 0.05 而返回的
`log10(total)` = 2.14，即 total ≈ 138：辅助项是姿态项的 **2044 倍**。原因是我把 aux 目标按"一个
stride-8 cell = 8 px"归一化，目标量级 O(1)、Huber 落在线性区，λ=200 于是永久压制姿态目标——
和 round 2 里 J3D 梯度失衡 180–450 倍是同一个失效模式。改为按 (W, H) 逐轴归一化（§7 的原始约定）后，
失衡在 60 步内从 627× 衰减到 3–5× 并稳定，姿态项量级恢复正常。这轮的所有数字都来自修正后的配置；
失效那轮的产物归档在 `outputs/hand_data51/_discarded_j2d_scale8/`。

教训：**λ 和目标归一化只能一起报告**。§7 只记了 λ=200 而没记归一化，正是这次返工的直接原因。

### 6. 复现

```bash
# 双副本训练（GPU 4,5 与 6,7；NCCL 变通 + 互异 MASTER_PORT）
bash tools/run_spa.sh

# 全 checkpoint 闭环打分（可与训练并行，按副本分卡）
REPS=a GPU=0 bash tools/eval_spa.sh
REPS=b GPU=1 bash tools/eval_spa.sh

# 汇总 + 门槛判定
python tools/collect_spa.py

# 判别性诊断
python tools/ablate_part_head.py  --config configs/eventhands_track_render51_spa.yaml \
    --ckpt outputs/hand_data51/track_render51_spa/track_render51_spa-step=8000.ckpt
python tools/measure_motion_gain.py --config configs/eventhands_track_render51_spa.yaml \
    --ckpt outputs/hand_data51/track_render51_spa/track_render51_spa-step=8000.ckpt
```

选中模型不变，仍是 `outputs/hand_data51/track_render51/track_render51-step=1000-...ckpt`（19.26）。

### 7. 这一轮真正的产出

- **一个能判定"头是否在起作用"的工具**：零初始化解码器 + 推理期消融，同一 checkpoint 自比。
  §7 / §8 的全部消融结论都缺这一步，因此都需要重新审视。
- **闭环运动增益分解**成为常规读数，并给出了基线 g（local 0.145 / global 1.005），
  后续 TBIN 轮次可直接对照。
- **一个否定结论**：在这个数据集与训练配方下，LBS 语义关节头作为**加性旁路**是惰性的。
  若要继续这条路，唯一有机制依据的改法是让关节头成为局部关节参数的**唯一**通路
  （fc 只出 root + transl），代价是失去暖启动逐位等价这一干净的比较基准——属于新方法，需另立门槛。
- **根因未动**：local 手指增量 99.4% 正交这一读数没变，输入表示（TBIN）仍是下一个必须做的改动。
  ——**此条已被 §10 推翻**：99.4% 正交是闭环差分伪影，真正的根因是 train→val 泛化鸿沟，TBIN 已降级。

---

## 10. 无训练判别探针：根因修正 + delta 信任域（19.26 → 18.73）

日期 2026-08-19。本节所有结论**零训练**，只用冻结的 `track_render51 step=1000` 检查点做推理。
工具：`tools/debug_probe_local.py`（探针 P1–P8），产物 `outputs/hand_data51/probe_local_report.jsonl`。

### 1. 动机与方法

问题：之前所有轮次（EDD / J3D / CMN / KSGN）全部无效，最大的问题到底在哪？只改一处应该改哪？

此前的运动增益读数（local g≈0.15、正交 99.4%）全部来自**闭环** rollout，它区分不了两件事：
输入里根本没有方向信息，还是闭环误差滚雪球。本轮的关键设计是**教师强制单步探针**（TF）：
给模型干净的 GT 上一帧，误差就只剩"模型从这帧输入里读出了什么"。

协议自校验：探针闭环 s=1.0 与官方 `eval_track.py` **逐位一致**（19.258 / 11.690 / 27.969），
且两次独立运行逐位复现——本机评测是确定性的，以下所有差异都是真实排序。

### 2. 假设判定

| 假设 | 判定 | 决定性证据 |
|---|---|---|
| H1 LNES 在输入端毁掉方向信息 | **否定（作为主瓶颈）** | TF 下同一 LNES 输入：val 增益 0.54、train 增益 0.78（正交仅 0.67）。方向信息在输入里，网络在 train 上读得出来 |
| H2 暴露偏差 / 误差滚雪球 | **否定** | TF 单步 26.84mm ≈ 闭环 27.97mm，无累积；闭环 = 单步噪声地板 |
| H3 损失导致 delta 收缩 | **否定** | val 上输出幅度是 GT 增量的 2.3–17 倍（过冲，非收缩；train 上 ≈0.9 校准） |
| H4 模型不利用 prev | **证实（更强形式）** | 注入噪声后模型把误差放大 2.4–8 倍（负降噪）；copy-prev 单步 4.4mm，模型 26.8mm |
| H5 train/val 域差 | **证实为主因，但与事件密度无关** | 见下表：密度匹配后 train 仍 5.9mm，val 26.8mm |

### 3. 关键读数（`zgz_local` 为主，50ms 窗）

| 探针 | 读数 |
|---|---|
| TF 单步 RA（val zgz_local） | **26.84mm**（gain 0.54，正交 0.99；预测增量 rms 19.6mm，GT 仅 3.7mm） |
| TF 单步 RA（train ch_local） | **5.42mm**（gain 0.78，正交 0.67）——**4.5× 鸿沟** |
| train 事件稀释至 val 密度（keep=0.25≈1428 事件/窗） | 5.89mm（keep=0.1 → 7.66mm）：**密度不是原因** |
| val 误差按事件数分位（120→3610 事件/窗） | 28.6→24.6mm：val 内部对密度也不敏感 |
| val 窗口 50→150→300ms | 26.8→26.0→25.7mm：更多事件救不了 |
| copy-prev 单步基线 | val 4.39mm / train 3.77mm（模型比"什么都不做"差 6 倍） |
| 分组降噪率（中位） | t 噪声 0.008→0.067（放大 8×）、pose 噪声 0.59→1.40（放大 2.4×） |

### 4. 根因修正（推翻本文 §9 末条与早期闭环诊断的核心结论）

1. **"LNES 丢 72–89% 事件 → 方向信息进网络前就不存在"不成立。** 闭环增量的 99.4% 正交是
   **相关噪声差分的伪影**：闭环里 `pred(t) = GT(t) + e(t)`，逐步增量含 `e(t) − e(t−1)`，
   把逐步相关的重估计噪声差分成了"纯噪声"。TF 一测，同样的输入增益 0.54–0.78。
2. **闭环误差不是累积出来的**，就是单步重估计噪声地板（26.8 ≈ 27.97）。
3. **真正的根因：事件→绝对姿态映射对 val 主体（zgz）不泛化。** 训练集只有 ch 一个来源
   （v2–v4 是变体），模型基本背下了 train（正交 0.67），对 val 输出 19.6mm rms 的
   "自信垃圾"增量（GT 每步只动 3.7mm）。密度、窗长、事件数全部被排除。
4. **这解释了四轮改动为何全部无效**：EDD / J3D / CMN / KSGN 都作用在同一训练分布内的
   网络/损失/噪声模型上，碰不到这个地板；17.91 与 19.26 的历史差异也都活在地板之内。
5. **TBIN（K=4 计数分箱）前提失效，正式降级**——其立论"方向信息在输入端被丢弃"已被 TF 探针否定。

### 5. 已落地修复：delta 信任域（零训练，19.26 → 18.73）

既然单步重估计噪声大、且大部分与真实运动正交，对更新量做信任域收缩即可把独立噪声成分平均掉：
`model/eval_track.py --delta-trust s`，闭环链每步 `pred ← prev + s·(pred − prev)`，默认 1.0。

选型：2D 分组扫描（global/pose 各一个 s）最优 18.722，与单标量 s=0.5 的 18.730 只差 0.008mm，
盆地平坦（0.5–0.7 均在 18.7–18.9）→ 取**单标量 0.5**，避免在 2 条 val 序列上过拟合超参。

官方评测前后对照（同 ckpt、同协议；默认参数与旧 json **逐位相等**，代码改动无害）：

| 指标 | 基线 | δ-trust 0.5 |
|---|---:|---:|
| RA-MPJPE 总体 | 19.258 | **18.730** |
| zgz_local / zgz_global RA | 27.97 / 11.69 | **27.04** / 11.51 |
| RA-MPVPE 总体 | 15.680 | **15.354** |
| jitter local / global (mm/step) | 11.68 / 11.08 | **6.11** / 8.52 |
| MPJPE abs 总体 | 62.21 | 61.57 |
| zgz_local 漂移桶 0s→15s | 27.1→29.7 | 25.9→29.7（无滞后性漂移） |

三次独立运行（含 `conda activate EventHandsTrain` 路径）逐位一致。产物：
`outputs/hand_data51/track_render51/eval_step1000_trust05/`。

定位：这是**补丁不是解法**——它只消噪声的独立成分，消不掉主体差带来的相关偏差
（local 只从 27.97 → 27.04）。但它免费、无害、把抖动近乎减半，作为部署侧默认值成立。

### 6. 下一轮唯一改动（预注册）

**训练数据生成域随机化**（网络、损失、训练配方一行不动）：逐序列随机化
(i) MANO 形状 betas；(ii) 动作来源与速度/幅度重定时（覆盖 GT 每步 p50≈5.3°/50ms 的关节速度分布，
而非重复 ch 的动作风格）；(iii) 事件生成参数（对比度阈值、噪声率、密度——train/val 本身就差 4×）。

**验收门槛（预注册，比 RA 灵敏一个量级）**：TF 探针 val 单步 RA 从 26.8mm 显著向 train 的
5–6mm 收敛（`tools/debug_probe_local.py`，训练早期即可读出方向）；闭环 RA 预期跟随，
因为闭环 = 单步噪声地板。

### 7. 复现

```bash
conda activate EventHandsTrain   # base 环境没有 torch

# 全套探针（约 2.5 分钟，产物 outputs/hand_data51/probe_local_report.jsonl）
CUDA_VISIBLE_DEVICES=2 python tools/debug_probe_local.py

# 信任域官方评测（RA 18.73）
CUDA_VISIBLE_DEVICES=2 python model/eval_track.py \
    --config configs/eventhands_track_render51.yaml \
    --ckpt "outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt" \
    --delta-trust 0.5 --out-dir outputs/hand_data51/track_render51/eval_step1000_trust05
```

---

## 11. LOSO 诊断 + 域随机化：根因再修正与 19.26 → 12.77mm

第 10 节把根因定为"事件→绝对姿态映射对 val 主体不泛化"（TF 单步 train 5.4mm vs val 26.8mm）。
本轮用一次留一主体（LOSO）训练把这个结论**推翻**了：跨主体泛化其实很好（未见主体 local 6.0mm），
26.8mm 完全来自 `zgz_local` 这一条**低事件率**序列。前四轮（EDD / J3D / CMN / KSGN）之所以全部无效，
是因为它们优化的 19.26mm 指标被这一条异常序列支配。

据此把修复放到数据层（唯一改动 = dataloader 的域随机化，网络逐参数不变），
双副本 worse-of-two 闭环 RA **19.26 → 12.77mm（−34%）**，TF 单步 **26.84 → 12.60mm（−53%）**，
两条预注册门槛全部通过。

### 1. Step 0：过拟合动力学（零训练）

`track_render51` 四个存盘 ckpt 的 TF 单步 RA（`tools/debug_probe_local.py --tf-only`）：

| ckpt | train/ch_local | train/ycy_local | val/zgz_local | val/zgz_global |
|---|---:|---:|---:|---:|
| step1000 | 5.42 | 5.09 | 28.62 | 10.43 |
| step2000 | 4.95 | 4.03 | 26.74 | 11.36 |
| step5000 | 4.14 | 3.58 | 32.46 | 10.25 |
| last | 4.13 | 4.69 | 27.84 | 11.71 |

train 单调下降、val 在 26–32mm 无规律抖动，局部增量的正交能量比 train 0.67→0.47 而 val 恒为 0.99。
记忆化确认，且**训练越久对 val 没有任何帮助**，为早停提供了依据。

### 2. Step 1：LOSO 诊断训练（ycy 留出）

`data/hand_data51_loso`（符号链接 + 改 `splits.json`，ycy 的 8 条移出 train，共 64 条）
+ `configs/eventhands_track_render51_loso.yaml`（只改 `DATA.ROOT`/`OUTPUT_DIR`；因另一项目占满 8 卡，
改成单卡 256×累积 8 = 等效 2048，与参考配方的 2×1024 等效，LR 不变）。

held-out ycy 的 TF 单步 RA：

| ckpt | train/ch_local | **held-out ycy_local** | held-out ycy_global | val/zgz_local |
|---|---:|---:|---:|---:|
| step1000 | 4.12 | **5.11** | 5.97 | 27.58 |
| step2000 | 3.78 | 6.18 | 7.63 | 32.28 |
| step3000 | 3.40 | 7.68 | 8.33 | 32.49 |
| step5000 | 5.81 | 10.51 | 11.14 | 31.06 |

**ycy 完全没参与训练，误差 5.11mm，和它在训练集里时的 5.09mm 几乎相同。** 预注册的分叉判据
（≥15mm 判通用鸿沟 / ≤10mm 判 zgz 特有）明确落在后者。ycy 随训练步数从 5.11 升到 10.51，
是记忆化的独立证据。

### 3. 同协议闭环对照：held-out ycy vs zgz

同一个 LOSO step1000 模型，`data/hand_data51_valycy`（评测专用符号链接根，val 换成 ycy 的
local/global 两条，与 zgz 对称）跑与官方完全相同的递推协议：

| val 主体 | overall RA | local | global |
|---|---:|---:|---:|
| **held-out ycy**（δ-trust 1.0） | 12.61 | **6.06** | 18.21 |
| **held-out ycy**（δ-trust 0.5） | 11.97 | **6.02** | 17.07 |
| zgz（δ-trust 1.0） | 20.16 | 29.70 | 11.88 |
| zgz（δ-trust 0.5） | 19.62 | 28.72 | 11.72 |

**未见主体的 local 追踪是 6.0mm，`zgz_local` 是 28.7mm，差 4.8 倍。** 而 global 方向相反
（ycy 17.07 差于 zgz 11.72），说明这不是"某个主体整体难"，而是 `zgz_local` 这条序列本身。

### 4. Step 2B：`zgz_local` 到底异常在哪

逐项排除（全部零训练，工具随本轮入库）：

| 假设 | 工具 | 读数 | 判定 |
|---|---|---|---|
| GT 时间错位 | `tools/check_gt_event_alignment.py` | 覆盖度峰值在 **+0ms**（±200ms 扫描），所有序列一致 | 排除 |
| GT 空间错位 | 同上 | zgz_local 事件落在 GT 手掩码内 **0.958**，高于 ch_local 0.666 / ycy_local 0.706 | 排除 |
| 手指姿态 OOD | `tools/check_pose_coverage.py` | 45D 残差姿态最近邻距离 zgz_local **0.719** < held-out ycy_local 0.797 | 排除 |
| 手的表观尺寸/深度/腕部朝向 | 同上 | 557px / 0.689m / 2.44rad，均在训练分布内 | 排除 |
| 事件数量 | §10 thin 探针 | ch_local 稀释到 508 事件仅退化到 7.66mm | 不足以解释 |

唯一显著异常：**`zgz_local` 的事件率 692 事件/50ms**，是所有序列最低——ycy_local 2530、
ch_local 5082、lyq_local 4158；同一主体的 `zgz_global` 反而是最高的 29676（**同主体同相机相差 43 倍**，
其他主体的 global/local 只差 4–8 倍）。

结合"GT 逐步位移 3.8mm 属正常水平"，`zgz_local` 是**手在正常运动却几乎不产生事件**的采集：
低对比度/低光照或阈值异常。模型面对"运动大、证据少"就输出噪声增量（pred rms 19.6mm vs GT 3.8mm，
正交能量 0.99），这也解释了为什么静态的 δ-trust 0.5 对它特别有效——把噪声增量直接砍半。

### 5. Step 2A：域随机化（唯一改动 = dataloader）——19.26 → 12.77mm

Step 2B 的排查把 `zgz_local` 的异常锁定在"采集条件域外（事件率极低）"而不是标签错误，
对症的修复因此仍是数据层：让训练分布覆盖"证据稀疏"这个域。改动只在
`model/fastevc.py` 的 `HandData51Dataset`（网络/损失/优化器逐参数不变，11,209,429 参数）：

- **相机一致的几何增强**：绕主点的滚转 ±15°、缩放 [0.8, 1.25]、平移 ±14px。
  变换施加在**事件坐标**上（栅格化之前，无重采样），并精确镜像到标签侧：
  - 缩放/平移是纯内参变化，`K' = A·K`，3D 标签不变（逐样本 K 早已接到渲染分支的
    `_intrinsics`，无需改网络）；
  - 滚转把 3D 点映射为 `Q = R_z(θ)`，由于 MANO 组合方式是 `R_g(V − j0) + j0 + t`，
    精确的标签变换是 `R_g ← Q R_g`、`t ← Q(j0 + t) − j0`（`j0 = J(betas)[0]` 是 LBS 支点）。
    这一步是像素旋转的充要条件当 `fx = fy`；本相机 `fx/fy = 1.0008`，残差 <0.5px（有单元测试守住）。
- **事件统计增强**：随机丢弃 keep∈[0.25, 1]（直击 `zgz_local` 的 692 事件/50ms）+ 热像素 2e-4。

单元测试（`tests/test_abs_pose.py::test_domrand_*`，6 项）分别守住：全关时与基线 dataloader
逐位相同、滚转标签变换与像素旋转一致（<0.5px）、`K'` 与事件位移一致（<1e-3px）、
丢弃是同一批槽位的子集且标签不变、缩放/平移只改 K 不动 3D 标签、配置 diff 只含 AUG 与训练规模。

双副本（`configs/eventhands_track_render51_domrand{,_rep2}.yaml`，固定步网格存盘）结果：

| 副本 / ckpt | δ-trust | overall RA | local | global | jitter all |
|---|---:|---:|---:|---:|---:|
| rep1 step3000 | 0.5 | **12.77** | 14.67 | 11.12 | 6.52 |
| rep2 step3000 | 0.5 | **12.56** | 13.97 | 11.33 | 6.53 |
| rep1 step3000 | 1.0 | 13.23 | 15.52 | 11.23 | 9.15 |
| rep2 step3000 | 1.0 | 12.93 | 14.74 | 11.36 | 9.24 |
| 基线 track_render51 step1000 | 0.5 | 18.73 | 27.04 | 11.51 | 6.11 |

TF 单步 RA（灵敏读数）：`val/zgz_local` **26.84 → 12.06 / 12.60**（两副本），
`train/ch_local` 保持 4.6–5.0（未牺牲训练域）。

**预注册门槛判定：两条全部通过。**
TF worse-of-two = 12.60mm（需改善 ≥30%，实得 **−53%**）；
闭环 worse-of-two = **12.77mm**（需 <18.3，当前最好 18.73，实得 **−34%**）。

两点值得注意：

1. δ-trust 的收益从 −0.53mm 缩到 −0.46/−0.37mm，且 trust=1.0 时也已经到 12.9–13.2mm。
   说明模型**真正学会了在稀疏证据下少动**，不再依赖推理期的静态压制——这与 §4 的机理判断一致。
2. 最佳点在 step3000 而非 step1000，且 TF 读数不再随训练单调恶化（§1 的记忆化曲线被打断），
   说明增强确实扩大了有效训练分布而不是单纯正则化。

### 6. 结论与对历史读数的影响

1. **跨主体泛化不是瓶颈**：未见主体 local 6.0mm、overall 11.97mm。
2. **19.26 / 18.73mm 这些历史读数是被单条异常序列支配的**。`zgz_local` 占 val 约一半帧、
   贡献 28.7mm，`zgz_global` 只有 11.7mm。任何在这个指标上做的 ±0.5mm 比较，本质上是在
   比较各方法对一条低事件率序列的噪声输出，这正是前四轮全部"无效"的原因。
3. **δ-trust 是对症的**：它压制的正是证据不足时的噪声增量。
4. **域随机化把这条低事件率序列拉回分布内**：19.26 → **12.77mm**（worse-of-two，−34%），
   且网络零改动。这是四轮网络侧改动全部失败之后，第一个通过预注册门槛的修复，
   也印证了"瓶颈在数据分布覆盖，不在网络容量或损失设计"。
5. 后续候选（本轮冻结未做）：
   (a) 评测口径——val 应包含多个主体并区分正常/低事件率采集；建议把留出主体口径
       （§3 的表）作为并列主指标，避免再次被单条序列支配；
   (b) 机制——把 δ-trust 做成随证据量自适应的更新步长（事件少→信息矩阵弱→步长小，
       Gauss-Newton 的自然推论，静态 0.5 是它的特例）。本轮数据侧修复后 δ-trust 的
       边际收益已从 0.53 降到 0.37mm，这条路的空间需要重新估。

### 7. 复现

```bash
conda activate EventHandsTrain   # base 环境没有 torch

# Step 0 过拟合动力学
CUDA_VISIBLE_DEVICES=2 python tools/debug_probe_local.py --tf-only \
    --ckpt "outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt" \
    --seqs "val:zgz_local,val:zgz_global,train:ch_local,train:ycy_local" \
    --out outputs/hand_data51/probe_step0_dynamics.jsonl

# Step 1 LOSO 训练（~2h，单卡）
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 MASTER_PORT=29711 CUDA_VISIBLE_DEVICES=2 \
    python model/train_abs.py --config configs/eventhands_track_render51_loso.yaml

# Step 1 判读 + held-out ycy 的同协议闭环
CUDA_VISIBLE_DEVICES=2 python model/eval_track.py \
    --config configs/eventhands_track_render51_loso_valycy.yaml \
    --ckpt "outputs/hand_data51/track_render51_loso/track_render51_loso-step=1000.ckpt" \
    --delta-trust 0.5 --out-dir outputs/hand_data51/track_render51_loso/eval_valycy_step1000_trust0.5

# Step 2B 标签与分布排查（各 <10 秒）
CUDA_VISIBLE_DEVICES=2 python tools/check_gt_event_alignment.py
CUDA_VISIBLE_DEVICES=2 python tools/check_pose_coverage.py

# Step 2A 域随机化：单元测试 -> 双副本训练（~2.5h/副本）-> 评估 -> 门槛
python -m pytest tests/test_abs_pose.py -k domrand -q
GPU_A=3 GPU_B=6 bash tools/run_domrand.sh
GPU=2 bash tools/eval_domrand.sh
python tools/collect_domrand.py
```

---

## 12. 渲染通道消融：silhouette vs 归一化逆深度

> **⚠ 本节的三臂对比不成立。** `MODEL.RENDER_CHANNELS` 当时**从未被 `model/model.py`
> 读取**（`_render_chunk` 硬编码 `torch.stack([sil, inv])`、`in_ch` 硬编码为 4），所以
> `ch_sil` / `ch_inv` / `ch_both` 训的是**同一个 4 通道模型**，§12.3 的"三臂统计不可区分"
> 实际测到的是同配置的副本散布，§12.5 结论 1 与 3 不成立。§12.1（`inv` 的动态范围）、
> §12.2（置零依赖度）、§12.4（19.26 是 best-of-4）不受影响，它们测的都是既有的 4 通道模型。
>
> 这个静默失效是在后来一轮已撤回的输入通道实验中发现的：该键现已真正实现解析
> （`MNISTModel._parse_render_channels`，可选 `sil` / `inv`，默认 `[sil, inv]`），
> 并对 `MODEL` 下的**未知键直接报错**（`MNISTModel.MODEL_KEYS`），
> 由 `test_unknown_model_key_is_rejected` 守住，不会再出现"config 层消融、代码层静默失效"。
> 重构后 `track_render51-step=1000` 复评得 **19.25760436702419**，与记录值逐位相同，
> 故 `ch_both` 的 12 个测点仍是合法对照组。

问题：4 通道输入里的两个渲染通道（剪影掩码 `sil` + 归一化逆深度 `inv`），到底哪一个带来了
Track(render) 相对 Full 的收益？基线取 `track_render51`（闭环 RA 19.26，`step=1000`），
因为 §5 已证明渲染通道在**绝对输出**路径上无增益（AbsRender 20.08 ≈ Full 19.97），
在那条路上拆通道是在拆一个零效应。

### 12.1 两个通道不对称（先验修正）

渲染器最后一步是 `inv = clamp(1/z, 0, 5)/5 * sil`，所以 **`inv > 0` 恰好等价于 `sil == 1`**：
掩码可以从 `inv` 阈值化完全恢复。三臂的真实语义因此是

| 臂 | 输入 | 语义 |
|---|---|---|
| `ch_sil` | 3ch：LNES + sil | 只有掩码 |
| `ch_inv` | 3ch：LNES + inv | 掩码 **+ 深度数值** |
| `ch_both` | 4ch：LNES + sil + inv | 已发表配置（掩码被显式重复一份） |

实测（`tools/probe_render_channels.py --mode stats`，val 两条序列各 400 GT 帧）确认了支撑集恒等，
并否掉了"手在 0.4 m 处 `inv ≈ 常数、无信息"的先验：

| 序列 | 掩码像素占比 | `inv` 掩码内均值 | 帧内 std | 帧内极差 | 极差 / bf16 分辨率 |
|---|---:|---:|---:|---:|---:|
| zgz_global | 1.52% | 0.381 | 0.0148 | 0.0731 | **37.4×** |
| zgz_local | 1.23% | 0.291 | 0.0105 | 0.0527 | **27.0×** |

深度数值的展布是 bf16 量化步长的 27–37 倍，**是可分辨的真实信号**，不是 `sil` 的缩放副本。

### 12.2 零训练探针：训练好的网络重度依赖 inv

`track_render51-step=1000` 的 `conv1` 每输入通道 L2 范数：

| 通道 | LNES(off) | LNES(on) | sil | inv |
|---|---:|---:|---:|---:|
| 范数 | 2.656 | 2.672 | 1.306 | **2.236** |

`inv` 的范数是 `sil` 的 **1.71×**，且已达 LNES 通道的 84%（`sil` 只有 49%）。

推理期把某个渲染通道强制置零后跑同协议闭环（`--mode zero`，不改 `model/eval_track.py`）：

| 置零 | RA-MPJPE | ΔRA | local | global | jitter all (local) |
|---|---:|---:|---:|---:|---:|
| 无（控制组） | 19.254 | +0.00 | 27.96 | 11.69 | 11.79 |
| `sil` | 21.181 | **+1.93** | 31.00 | 12.65 | 15.34 |
| `inv` | 57.111 | **+37.86** | 34.44 | 76.80 | 221.4 |
| 两者 | 84.150 | +64.90 | 82.81 | 85.31 | 25.07 |

控制组复现官方 `eval_step1000` 到 3.4e-3 mm（19.2542 vs 19.2576，cuDNN 算法选择差异，
远小于 ±0.4 mm 噪声带），说明包装器本身是惰性的。

判读：**训练好的 4ch 模型把信息通路几乎全部压在 `inv` 上**——抹掉它，global 轨迹直接崩到 76.8 mm。
但这是"依赖度上界"，因为置零后 `sil=1 且 inv=0` 这种组合在训练里从未出现，属于 off-distribution 输入。
是否**必须**有 `inv`，只能靠重训练回答。

### 12.3 重训练三臂：两个通道互相可替代

`MODEL.RENDER_CHANNELS` 是本轮唯一的代码改动（`model/model.py`），只要不含 `inv` 就跳过
z-buffer 那一趟 `scatter_reduce_(amin)`。三臂 config 逐键校验为单变量差异
（`tools/diff_configs.py`：仅 `MODEL.RENDER_CHANNELS` + 输出路径），rep2 仅差 `SEED`。

每臂 2 副本 × 每 500 步存盘 × 3000 步 = 每臂 12 个已评测检查点，按**递推 RA-MPJPE** 排序。

先按仓库既有约定（每副本取最优步、每臂取更差副本）：

| 臂 | RA（更差副本） | rep1 / rep2 | 选中步 | local | global | jitter all | 延迟 b1（对齐 Full=1.75） |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ch_sil` | 19.93 | 19.93 / 19.68 | 1500 / 3000 | 28.16 | 12.78 | 11.64 | 4.41 ms |
| `ch_inv` | 19.98 | 17.55 / 19.98 | 500 / 1000 | 28.32 | 12.73 | 12.43 | 4.50 ms |
| `ch_both` | **19.53** | 18.73 / 19.53 | 500 / 1500 | 26.92 | 13.11 | 11.68 | 4.54 ms |

（延迟列的跨进程重测散布约 ±0.05 ms，三臂之差不具判别力；数字统一来自
`outputs/hand_data51/main_table.json`。）

臂间差异只有 +0.40 / +0.44 mm，而**副本散布最大 2.43 mm**（`ch_inv` 的 17.55 vs 19.98）。
"min over 6 步再 max over 2 副本"是个 2 样本次序统计量，方差正是这 2.43 mm，
所以这张表**分辨不出任何臂间差异**。

改用池化估计（每臂 12 个测点，零额外算力）：

| 臂 | n | 平均 RA | sd | sem | min | max |
|---|---:|---:|---:|---:|---:|---:|
| `ch_sil` | 12 | 20.79 | 0.81 | 0.23 | 19.68 | 22.79 |
| `ch_inv` | 12 | 20.23 | 1.05 | 0.30 | 17.55 | 21.66 |
| `ch_both` | 12 | 20.70 | 1.33 | 0.38 | 18.73 | 23.45 |

| 臂 | 平均 ΔRA vs `ch_both` | 差值的合并 sem |
|---|---:|---:|
| `ch_sil` | **+0.09** | 0.45 |
| `ch_inv` | **−0.47** | 0.49 |

两个差值都在 1 个 sem 以内。本轮预算的分辨率约 **1.1 mm**，低于此的差异不能声称。

### 12.4 `ch_both` 与主表 `EventHands-Track (render)` 的对齐

`ch_both` **就是** Track(render)：`RENDER_CHANNELS: [sil, inv]` 等于代码默认值，两者参数量同为
**11,209,429**、`conv1` 同为 4 输入通道、FLOPs 同为 1.655 G，同机实测 batch-1 延迟
3.803 vs 3.789 ms（差 0.37%，小于 ±0.05 ms 的重测散布）。`tests/test_abs_pose.py::test_render_channels_default_is_the_published_pair`
钉住"不写 `RENDER_CHANNELS` 时解析结果就是 `(sil, inv)` + `conv1` 4 通道"，另有一条测试把单通道臂的
渲染面与 4ch 渲染的对应面做 `torch.equal` 逐位比较。

逐键 diff 后，两个 config 的实质差异只在 TRAIN 块，全部与网络结构无关：

| 键 | Track(render) | `ch_both` | 是否影响数值 |
|---|---|---|---|
| `DEVICES` × `BATCH_SIZE_PER_GPU` | 2 × 1024 | 1 × 2048 | 等效 batch 同为 2048；但 BatchNorm 的每设备统计量由 1024 样本变为 2048 样本 |
| `MODEL.RENDER_CHUNK` | 256 | 1024 | 近似不影响：分块只沿 batch 维、样本间独立，但 MANO 前向的 GEMM 会随 batch 尺寸换 kernel，实测 700 样本下 0.01% 的像素因 1 ulp（1.2e-7）舍入差翻转取整下标，非逐位一致 |
| `MAX_STEPS` | 15000 | 3000 | 是（本轮只跑到 3000） |
| 存盘网格 | 1000 / 2000 / 5000 / last | 每 500 步 | 否，但改变"最优步"的候选集 |

所以 `ch_both` 是 Track(render) 的一次**独立复现**，两者的检查点集合可以直接并到一起看：

| | n | 平均 RA | sd | min（各自表内声称值） |
|---|---:|---:|---:|---:|
| Track(render) 基线（`eval_step{1000,2000,5000}` + `eval_last`） | 4 | 20.63 | 1.18 | **19.26** |
| `ch_both`（2 副本 × 6 步） | 12 | 20.70 | 1.33 | **18.73** |

**均值只差 0.07 mm**，`ch_both` 复现了基线的分布中心。由此可以反过来校准主表：

- 主表 Track(render) 行的 **19.26 是 4 个测点里的最小值**，而不是该配置的期望性能；同一配置
  换个种子、把存盘网格加密到 12 个点，最小值就落到 18.73。两个数都是 best-of-N 抽样，
  **分布中心在 20.6~20.7 mm**。
- 因此 §12.3 "臂间差异 ≤0.5 mm 不可分辨"的结论，同样适用于**任何以 19.26 为基线、只改一处再重训练
  的历史结论**（这些方法本身已随回退删除，不再入主表，但方法学教训仍然有效）：只有幅度显著超过
  1.1 mm 的差值才可信。§11 的域随机化 19.26 → 13.23（−6.0 mm）远超此阈值；§10 的 δ-trust
  19.26 → 18.73（−0.53 mm）落在分辨率之下，但它是**同一检查点的推理期改动**（配对比较，
  无训练随机性），所以仍然成立——这正是配对设计比跨训练比较更省算力的地方。
- 三臂的选点约定（每副本取最优、每臂取更差副本）与 Track(render) 行的 best-of-4 不同口径，
  跨行直接比大小会偏向后者，故主表用脚注标注了这一点。

### 12.5 结论

1. **两个渲染通道互相可替代。** 单独用 `sil`（20.79 ± 0.23）、单独用 `inv`（20.23 ± 0.30）
   与两者都用（20.70 ± 0.38）在统计上不可区分。渲染反馈提供的是"上一帧手在图像哪里、
   大致什么形状"这一件事，掩码和逆深度各自都足够编码它。
2. **"网络依赖 inv"与"inv 不可或缺"是两件事。** 推理期置零显示 4ch 模型几乎完全走 `inv` 通路
   （+37.86 mm），`conv1` 权重范数也是 1.71× 偏向 `inv`；但那只是训练收敛到的一条**路由选择**，
   不是信息需求——重训练成 3ch-sil 后精度一分不少。这也再次说明**推理期消融只能读"依赖"，
   不能读"必要性"**（与 §9 判定 KSGN 头惰性时的方法学教训同向）。
3. **省掉 z-buffer 不是有意义的效率收益。** `ch_sil` 3.692 ms vs `ch_both` 3.803 ms（同机原始值），
   只快 2.9%，且与 ±0.05 ms 的重测散布同量级：渲染路径的开销主要在可微 MANO 前向与点撒投影，
   深度 scatter 只占极小一块。
4. **方法学：本轮的臂间效应（≤0.5 mm）远小于同配置的运行间散布（2.43 mm）。**
   §7 / §8 那类"两次独立训练之差"的结论需要同样的池化处理才可信；要把分辨率压到 0.4 mm，
   按 (2.43/0.4)² 估算需要每臂约 37 个副本，不具性价比。

### 12.6 复现

```bash
conda activate EventHandsTrain

# Step 0 零训练探针（stats 约 5 秒；4 个置零变体并发约 2 分钟）
CUDA_VISIBLE_DEVICES=7 python tools/probe_render_channels.py --mode stats \
    --ckpt "outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt" \
    --out outputs/hand_data51/probe_render_channels/stats.json
for z in none sil inv both; do
  CUDA_VISIBLE_DEVICES=2 python tools/probe_render_channels.py --mode zero --zero $z \
      --ckpt "outputs/hand_data51/track_render51/track_render51-step=1000-val_loss=val_loss=0.8015.ckpt" \
      --out outputs/hand_data51/probe_render_channels/zero_$z.json &
done; wait

# 单元测试（8 项与本轮相关）
python -m pytest tests/test_abs_pose.py -q

# 三臂 × 2 副本，一卡一 run 全并行（各约 55 分钟）
gpus=(0 2 3 4 5 6); runs=(ch_sil ch_sil_rep2 ch_inv ch_inv_rep2 ch_both ch_both_rep2)
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=${gpus[$i]} nohup python model/train_abs.py \
      --config configs/eventhands_track_render51_${runs[$i]}.yaml \
      > outputs/hand_data51/logs_ch/${runs[$i]}.log 2>&1 &
done; wait

# 36 个 checkpoint 的闭环评测（每卡 6 并发，约 100 秒）+ 汇总
GPUS="0 2 3 4 5 6 7" PER_GPU=6 bash tools/eval_ch_ablation.sh
CUDA_VISIBLE_DEVICES=7 python tools/report_ch_ablation.py
```

吞吐配方（本轮实测）：单卡 bsz 2048 + `ACCUMULATE_GRAD_BATCHES: 1` + `RENDER_CHUNK: 1024`
得到 **2050 samples/s、峰值 23–24 GB**，即单卡就达到历史双卡 2440 samples/s 的 84%；
6 个 run 并发时每 run 仍是 2020–2050 samples/s（16 GB 数据集全驻页缓存，144 核喂得动 96 worker），
无争用。有效批保持 2048 以沿用 `LR: 5.656e-3`。

分报告：`outputs/hand_data51/report_ch_ablation.md` / `.json`

---
## 13. SO(3)+FK 训练目标：把 51D MSE 换成 L_rot + L_trans + 2·L_FK

> **本节替代了原 §13（LBS 关节驱动输入通道）。** 那轮实验因 LBS 编码的设置存在问题被撤回，
> 其代码（`vert_lbs` / `vert_wmax` 缓冲、`lbs`/`wmax` 渲染面、`RENDER_DEPTH_TOL`）、
> 4 个 config、4 个 run 目录与全部产物均已删除；它唯一保留下来的结论——
> `MODEL.RENDER_CHANNELS` 此前从未被读取——已并入 §12 顶部的警告框。
> 同期撤回的还有"实验三"（LBS 通道 + 本节 loss 的组合臂）。

问题：当前 51D 的训练目标是**逐元素加权 MSE**，把 45D 局部 axis-angle 残差、3D 平移（米）
和 3D 全局旋转放进同一个和里，靠 `LAMBDA_POSE/T/R = 450/60/30000` 拉平量纲。
这三个 λ 做的是**单位换算**而不是权衡，且 MSE 在 axis-angle 上并不度量真实的旋转差
（同一旋转的 `r` 与 `r(1+2π/|r|)` 表示会被判成天差地别）。本轮只换目标函数：

```
L = L_rot + L_trans + 2·L_FK
```

**网络结构、输出维度、dataset、label、MANO 参数定义、推理输出格式全部不动**，
ResNet 仍输出 51D，与对照臂 `ch_both` 的 config 差异经 `tools/diff_configs.py` 校验为
**仅 `LOSS` 块 + 输出路径**（由 `test_so3fk_config_diff_is_loss_only` 守住）。

### 13.1 先审计，不假设

| 项 | 审计结果 | 依据 |
|---|---|---|
| 51D 切片 | `transl 0:3`、`global_orient 3:6`、`local 6:51` | `pose_repr.Slices`（`mano_full_axis_angle`） |
| 45D 语义 | **不是**完整 axis-angle，是相对 `hands_mean` 的**残差** | `decode_to_mano_inputs`：`local_full = residual + hands_mean` |
| 旋转个数 | `1 (global) + 15 (local) = 16` | 45/3 + 1 |
| MANO joints | **21 个** OpenPose 序（16 关节 + 5 指尖） | `ManoLayer.mano16_tips_to_openpose21` |
| joints 单位 | **米** | GT `transl` 实测 `mean|t| = 0.2481`、`z = 0.6458` |
| root 索引 | **0**（腕） | OpenPose 21 序；记为 `BaseModel.FK_ROOT_JOINT`，未硬编码在 loss 里 |
| 旧 loss | `F.mse_loss` 三段加权和 / `NORMALIZER=51` | `BaseModel._mse_51d_loss` |

关键的一条是 **45D 是残差**：所有旋转量都必须先过 `decode_to_mano_inputs` 拿到
`local_full_aa` 才能转四元数，直接拿 51D 的 45 维去转是错的。

### 13.2 三项 loss 的实现

全部复用既有的 `decode_to_mano_inputs` → `ManoLayer` → `_fk` 通路，**没有第二套 MANO decode**。

**L_rot**：把 `[global_orient(3), local_full_aa(45)]` 拼成 `(B,16,3)`，转单位四元数后

```python
dot = (q_pred * q_gt).sum(-1)
loss_rot = (1.0 - dot.square().clamp(max=1.0)).mean()
```

平方使其天然满足 `q ≡ -q`（由 `test_rotation_term_is_antipodally_invariant` 守住：
给每个**解码后**的旋转加整圈 2π，loss 不变）。不用 `acos`（梯度在 ±1 处发散），
`clamp(max=1.0)` 只是防止完美预测因浮点误差得到负 loss——总 loss 之后要取 `log10`。

四元数转换 `pose_repr.axis_angle_to_quaternion` 用半角公式，并在 `|r|→0` 处切到泰勒展开、
模长取自 `clamp` 后的平方和，否则 `norm` 在原点给出 NaN 梯度——而**未训练的头恰好就在原点附近**。

**L_trans**：`F.smooth_l1_loss` 直接作用在米制 `transl` 上。工程里没有对平移做归一化，
故不额外引入缩放。

**L_FK**：pred 与 GT 各自过同一套 MANO 得到 21 关节，减去关节 0 后取**3D 欧氏距离的均值**
（不是 xyz 逐元素 MSE）：

```python
joint_dist = torch.norm(rel_pred - rel_gt, dim=-1)
loss_fk = joint_dist.mean()
```

GT 分支走 `no_grad`，pred 分支全程保留计算图。整个 `_so3_fk_loss` 在
`torch.autocast(enabled=False)` 里以 fp32 运行——变换链对精度敏感，而主干跑 bf16。
旧的 51D MSE 以 `legacy_loss_51d` 保留，`no_grad` 计算，**只进日志不进反传**。

### 13.3 零训练前置检查：量级与梯度

`tools/check_so3fk_loss.py`，未训练模型 + 64 个真实 val 样本：

| 项 | 值 | 判读 |
|---|---:|---|
| `loss_rot` | 0.1095 | 无量纲，对应平均旋转误差 37.00° |
| `loss_trans` | 0.0359 | 米制 SmoothL1 |
| `loss_fk` | 0.0248 | 米，对应 24.79 mm |
| `2·loss_fk` | 0.0496 | — |
| `loss_total` | 0.1949 | 三项同量级，**无单位错配** |
| `legacy_loss_51d` | 43.68 | 仅日志 |

`L_FK / L_trans = 0.7`：若 FK 误用毫米，这个比值会是 ~700，是最容易犯的错，这里没犯。

梯度（`torch.autograd.grad`，检查 51D 头的活跃行数）：

| 项 | fc 活跃行 | 期望 | 含义 |
|---|---:|---:|---|
| `loss_trans` | 3/51 | 3 | 只推平移 |
| `loss_rot` | 48/51 | 48 | 16 个旋转 = 3 + 45 |
| `loss_fk` | 51/51 | 51 | 经 MANO 反传到全部输出 |

`conv1`、`layer4`、`fc`、`prev_mlp` 均有非零且有限梯度；`legacy_loss_51d.requires_grad = False`。
**VERDICT: PASS**。

### 13.4 重训练：ΔRA −0.78 mm，未过门槛，判定打平

2 副本 × 3000 步 × 每 500 步存盘 = 12 个闭环评测检查点，配方与 `ch_both` 完全一致
（单卡 bsz 2048、`RENDER_CHUNK` 1024、warmup 500、`LOG10` 保留），rep2 仅差 `SEED`。

| 臂 | loss | n | 池化均值 RA | sd | sem | min | max |
|---|---|---:|---:|---:|---:|---:|---:|
| `ch_both`（对照） | 51D MSE | 12 | 20.70 | 1.33 | 0.38 | 18.73 | 23.45 |
| `so3fk` | `1·L_rot + 1·L_trans + 2·L_FK` | 12 | **19.92** | 1.66 | 0.48 | 17.73 | 24.04 |

**ΔRA = −0.78 mm，差值合并 sem = 0.61**。预注册门槛 ΔRA ≤ −1.1 mm，故判定 **打平（fail）**，
尽管它是三轮消融里唯一方向为负、且幅度最大的一个。按仓库既有口径
（每副本取最优步、每臂取更差副本）为 **18.27 vs 19.53（−1.26）**，是全表最好的递推 RA；
但按 §12.4 的教训，**不用这个更宽松的口径下结论**。

**代价：绝对定位显著退化。** 腕对齐指标看不到这一点：

| 臂 | 池化 RA-MPJPE | 池化**非对齐** MPJPE |
|---|---:|---:|
| `ch_both` | 20.70 | 63.71 |
| `so3fk` | 19.92 | **83.63** |

原因是结构性的：`L_trans` 是米制 SmoothL1，厘米级误差落在**二次区**（1e-2 量级的误差
贡献 ~5e-5），而 `L_FK` 又是 root-relative，**总目标里几乎没有约束绝对平移的项**。
新目标把容量花在了手指姿态上，绝对位置反而松了。这是本轮最重要的副作用。

### 13.5 结论

1. **换目标函数是三轮消融里唯一"方向正确"的改动**：−0.78 ± 0.61 mm，虽未过 −1.1 mm 的
   预注册门槛，但符号稳定、且在更宽松口径下达到 18.27（全表最优递推 RA）。
   与 §12/原 §13 的"加输入信息"路线形成对照：**瓶颈更可能在目标函数而不是输入信息量**。
2. **λ 的单位换算职能被消掉了**。新目标的权重是 1/1/2，每项都在自己的流形上，
   不再需要 450/60/30000 那种量纲补偿；实测三项量级也确实同阶（0.109 / 0.036 / 0.050）。
3. **暴露了一个此前被 MSE 掩盖的问题**：总目标对绝对平移几乎无约束，非对齐 MPJPE
   从 63.7 退到 83.6 mm。下一步若继续这条路线，应先补回绝对平移约束
   （例如把 `L_trans` 换出二次区，或加一项非 root-relative 的关节项），而不是继续加权重。
4. **数值稳定性的两个坑都在实现里堵住了**：`axis_angle_to_quaternion` 在原点的 NaN 梯度、
   以及 `1 - dot²` 在完美预测处的负值（会让 `log10` 变 NaN）。均有单元测试覆盖。

### 13.6 复现

```bash
conda activate EventHandsTrain

# 单元测试（32 项，含 10 项 so3fk / 四元数测试）
python -m pytest tests/test_abs_pose.py -q

# 零训练前置检查：量级 + 梯度（约 15 秒）
CUDA_VISIBLE_DEVICES=0 python tools/check_so3fk_loss.py \
    --config configs/eventhands_track_render51_so3fk.yaml --batch-size 64

# 2 副本训练 -> 检查点完整性 -> 12 个 checkpoint 闭环评测 -> 带门槛的汇总
GPUS="6 1" bash tools/run_loss_experiments.sh

# 仅重出报告（评测产物已在时）
python tools/report_ch_ablation.py --arms ch_both so3fk --control ch_both \
    --gate-mm -1.1 --out outputs/hand_data51/report_loss_so3fk.md
```

分报告：`outputs/hand_data51/report_loss_so3fk.md` / `.json`

---

## 14. 语义值化 mask：把 LBS 语义写进 silhouette 的取值（两种 loss 基线各测一次）

### 14.1 动机与设计

输入侧的 LBS 语义在此前已以**独立 7 通道**的形式被否证（ΔRA −0.14 ± 0.49，原 §13，已撤回）。
本轮重新设计的出发点是：语义不再单独占通道，而是**值化进已有的 mask 通道**——掩码形状、
通道数、参数量、FLOPs 全部不变，只把「填 1」换成「填一个表达该像素属于手的哪个部位的数」。

- **semsil 的定义**：`code_i = normalize(‖(W @ J_rest)_i − J_wrist‖)`。MANO 蒙皮权重矩阵
  `W`（778×16）的每一行完整回答了「该顶点由哪些内部关节驱动、各占多少」；把它与静止姿态的
  16 个关节位置相乘，得到每个顶点的规范关节位置，再取其到腕关节的距离，压成一个连续的
  **径向骨架坐标**（腕部 ≈ 0，指尖 ≈ 1）。用权重而非 `argmax` 部件 ID，是因为权重本身是混合的，
  所以该码在部件交界处连续。
- 掩码内取值 `0.35 + 0.65·code`，掩码外 0。0.35 的下界保证 mask 仍能被阈值化读回二值占据图，
  且远大于 bf16 步长。
- 光栅化时对每个像素取 z 缓冲容差内可见顶点的码求平均（与被撤回的 7ch 实现同一条光栅化路径，
  仅换码）。
- **深度通道保持米制不动**：§12.1 实测 `inv` 只有 27–37× bf16 步的动态范围，经不起语义调制。

**关键的实验设计决定**：同一改动在**两种 loss 基线上各测一次**，每臂与自己的对照只差
`MODEL.RENDER_CHANNELS` 一个键。

| 臂 | 基线 / 对照 | loss |
|---|---|---|
| `ch_semsil` | `ch_both`（原始发表配方） | 51D MSE |
| `sem` | `so3fk`（§13） | 1·L_rot + 1·L_trans + 2·L_FK |

先只做了 `sem`（叠在 so3fk 上），结果 +1.22 mm。但 so3fk 本身把非对齐 MPJPE 从 63.7 退到
83.6 mm，即对照已被新 loss 扭曲，因此在它上面打平**无法区分**「语义没用」和「语义与新 loss 打架」。
补做 `ch_semsil` 是对这个问题唯一干净的单变量检验——事实证明这一步是必要的（见 14.3）。

### 14.2 前置检查（零训练）

| 检查 | 结果 |
|---|---|
| 4ch 基线逐位复现 | 19.25760436702419 ✓ |
| so3fk step2500 逐位复现 | 18.2711665746328 ✓ |
| sem step2500 逐位复现 | 19.482833233395137 ✓ |
| semsil 支撑集 vs sil | 逐位相同 ✓ |
| semsil 动态范围 | 333× bf16 eps（`inv` 只有 27–37×），掩码内 std 0.228 |
| 参数量 / FLOPs | 与对照逐位相同（conv1 仍 4 输入）✓ |
| 单元测试 | 38 项全过（含 5 项 semsil + 1 项双臂单变量 config 校验） |

两臂的 batch 均为 2048×1，与各自对照完全一致，**没有任何 batch 或 LR 偏离**。

### 14.3 结果（各臂 2 副本 × 6 测点，池化）

| 臂 | loss | 池化 RA | sd | ΔRA vs 自己的对照 | 池化 sem | 门槛 ≤ −1.10 |
|---|---|---:|---:|---:|---:|---|
| `ch_both`（对照） | 51D MSE | 20.70 | 1.28 | — | — | — |
| **`ch_semsil`** | 51D MSE | **20.26** | **0.96** | **−0.44** | 0.48 | fail（打平） |
| `so3fk`（对照） | so3fk | 19.92 | 1.59 | — | — | — |
| `sem` | so3fk | 21.13 | 2.97 | **+1.21** | 0.97 | fail（打平） |

**两臂都没过门槛，但符号相反，这是本轮唯一重要的读数。**

- 在**原始 51D MSE** 基线上，语义 mask 微弱**有利**（−0.44），且把副本方差压到全表最低
  （sd 0.96 vs 对照 1.28，最差 checkpoint 21.54 vs 对照 23.45）。最优单点 18.26，
  已经和 so3fk 的 17.73 同量级。
- 在 **SO3+FK** 基线上，语义 mask 明显**不利**（+1.21），且方差最大（sd 2.97，最差 29.93）。
- 交互约 **1.65 mm**、约 1.5σ。按本预算的分辨率（约 2.5 mm）不可声称，但足以说明
  **「输入侧语义无用」这个结论只在 so3fk 基线上成立**，不能外推。若当初只做了 `sem` 就关闭这条路线，
  会是一个由对照选择导致的错误结论。

非对齐 MPJPE 与 jitter：

| 臂 | 池化非对齐 MPJPE | 最优 ckpt 非对齐 | 池化 jitter |
|---|---:|---:|---:|
| `ch_both` | 63.7 | 72.3 | 11.95 |
| `ch_semsil` | 69.1 | 74.7 | 12.17 |
| `so3fk` | 83.6 | 77.3 | 11.32 |
| `sem` | 83.3 | 86.3 | 11.24 |

语义 mask 在两种 loss 下都基本不影响绝对定位（+5.4 / −0.3 mm，远小于 so3fk 自身造成的
+19.9 mm 退化），也不影响 jitter。即它是一个**低风险、低收益**的改动。

### 14.4 已撤回：迭代 render-and-compare

同轮还测过迭代 render-and-compare（`it2` / `sem_it2`：以第一遍预测重渲染作为第二遍的 prev，
权重共享、零新参数，两遍都监督）。**已撤回，代码、config 与产物均已删除。**

撤回理由不是机制没实现对——训练侧确实有效，第二遍 loss 反超第一遍
（0.0111 < 0.0130）——而是**闭环稳定性被显著恶化**：非对齐 MPJPE 池化从 83.6 爆到 198.7 mm、
jitter 从 11.3 涨到 16.4，即把预测位姿反馈进渲染会在递推 rollout 中放大漂移（少数 checkpoint
落在稳定盆里，多数不在），同时延迟与 FLOPs 翻倍。RA 也未过槛（+0.97 / +0.83）。
若日后重启这条路线，应先解决反馈放大漂移（例如给第二遍的 prev 加信任域，或只在低事件率帧
触发第二遍），而不是加迭代次数。

### 14.5 结论

1. **语义值化 mask 在原始 loss 下方向正确但幅度不足**（−0.44 ± 0.48，门槛 −1.1）。它同时
   把训练方差压到全表最低，这一点比均值移动更稳健，值得在后续任何加大预算的实验里保留。
2. **不能宣布「输入侧语义无用」**。该结论只在 so3fk 基线上被观察到；换回原始 loss 后符号翻转。
   两个基线的交互（1.65 mm、1.5σ）是本轮最值得追的线索。
3. **主表最优行仍是 so3fk 的 18.27**；`ch_semsil` 的更差副本 18.61 紧随其后，且它与 so3fk 是
   两条独立的改进方向（一个改输入、一个改目标函数），尚未组合过——`ch_semsil` 与 so3fk 的组合
   就是 `sem`，而它是四臂里最差的，所以这两条路线目前是**互斥**的，不是可叠加的。

### 14.6 复现

```bash
conda activate EventHandsTrain

# 单元测试（38 项，含 5 项 semsil）
python -m pytest tests/test_abs_pose.py -q

# semsil 的 bf16 量级表
python tools/probe_render_channels.py --mode stats \
    --config configs/eventhands_track_render51_sem.yaml \
    --out outputs/hand_data51/probe_render_channels/stats_sem.json

# ch_semsil：2 副本训练 -> 12 checkpoint 闭环评测 -> 带门槛的分报告（约 1 小时，双卡）
GPUS="0 1" EXPERIMENTS="ch_semsil" CONTROL=ch_both bash tools/run_loss_experiments.sh

# loss x 语义 2x2 汇总
python tools/report_ch_ablation.py --arms ch_both ch_semsil so3fk sem \
    --control ch_both --gate-mm -1.1 --out outputs/hand_data51/report_semsil_2x2.md
```

汇总报告：`outputs/hand_data51/report_semsil_2x2.md` / `.json`
分报告：`outputs/hand_data51/report_loss_ch_semsil.md`、`report_loss_sem.md`

---
