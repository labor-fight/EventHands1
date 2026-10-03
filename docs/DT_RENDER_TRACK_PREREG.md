# DT 预注册：EventHands-Track（render + SO3 + FK + DomRand）的抖动、误差与参数量

> 2026-10-03。基线固定为 `dt_base`（该架构在项目统一配方下的重训），参照为 `rt_cnntrack`（同网络、MSE 目标）和历史 `track_render51_dr_so3fk`。结论写在 `docs/DT_RENDER_TRACK_VERDICT.md`。
> 本文登记的臂、键、判定门槛在任何 DT 臂被评测之前写下并提交，之后不再改动；之后的追加只写在文末"追加登记"，并标明日期和当时已有的结果。
> **时间线（如实记录）**：`dt_base`（种子 3407 / 3408 / 3409）、`rt_cnntrack_s3409`、`dt_tr`（3407 / 3408）、`dt_nos_s3407` 的训练在 06:53 启动，早于本文的提交。它们是基线臂和不需要新代码的臂，配置与键由已批准的计划固定；启动时没有任何 DT 评测结果，也没有任何 DT 训练跑完。其余臂在各自的代码通过测试之后启动。

## 0 问题与约束

**问题**：EventHands-Track（render + SO3 + FK + DomRand）是项目里精度最好的稠密跟踪器：ResNet18 读 4 通道 [LNES 两通道 + 上一状态渲染的剪影 + 逆深度]，输出 51 维 Δ 加到 prev 上，损失为 SO(3) 旋转 + 平移 + 根相对 FK，域随机化训练。在保持因果、在线、递推的前提下，能否 (a) 降低逐步抖动，(b) 降低误差（根对齐 RA-MPJPE 和绝对 MPJPE），(c) 降低参数量？

**约束**：
- 因果：只用当前 50 ms 包内的事件和上一步输出的状态。
- 协议不变（AGENTS.md）：9 人 72 序列训练；zgz 既是开发集也是测试集；只报最后一步，不选点；主表只经 `tools/report_table.py` 生成。
- S37 / S38 逐位可复现：所有新行为默认关闭，旧臂的代码路径逐位不变。
- 一个候选臂 = `dt_base` + 一个因素，差异只在表中列出的键。
- 不在 zgz 上调超参：增益、权重、范围都在看到结果之前固定。
- 本轮是稠密 CNN 臂，不属于 S37 / S38 的稀疏异步族。稀疏、异步的约束不适用，两族的结论互不替代。

## 1 诊断（立项依据）

| # | 事实 | 数字 | 出处 |
|---|---|---|---|
| F1 | 历史目标架构（第 3000 步，种子 0 / 1，zgz） | 递推 RA 12.254 / 12.192（均值 12.223），local 14.245，global 10.466；11.21 M 参数，1.655 GMACs，4.51 ms，其中渲染 + FK 约 2.7 ms；绝对 MPJPE 70–81 mm，每步抖动 12–13 mm（GT 运动约 2.9 mm / 步） | `docs/FAILURE_AND_CLEANUP_LEDGER.md` 2026-10-02 条；`EventHands/outputs/hand_data51/report_domrand_2x2.md` §2 |
| F2 | 统一配方、MSE 目标的同一网络 `rt_cnntrack`（6000 步） | RA 13.695 / 13.800；绝对 MPJPE 56.80（global 28.05，local 89.89）；教师强制 RA 10.03；闭环 / TF 放大 1.365；根速度比 1.52（偏抖），手指速度比 0.32（过阻尼） | `outputs/semkine/rt_cnntrack_s3407/evalx_val_core_last_tf_pert.json` |
| F3 | 抖动拆分（`rt_cnntrack_s3407`，本轮新增 `evalx.jitter_decomp`，反事实混合：把某一块换成 GT 或预测后再做 FK） | zgz_global：jit GT 7.56、预测 9.20 mm / 步；误差加速度 acc_err 6.50 mm，只保留某一块时 平移 / 根 / 手指 = 4.85 / 5.29 / 2.51。zgz_local：jit GT 3.79、预测 6.07；acc_err 10.28，平移 / 根 / 手指 = 7.61 / 8.88 / 5.62。根角加速度 预测 3.07° 对 GT 1.82°（global） | `tools/dt/test_jitter_decomp_scratch.py` 的真实数据段 |
| F4 | **绝对平移的误差是深度的系统偏差，且 zgz_local 的深度在训练范围之外** | 训练集 50 ms 步上的深度（9 人 72 序列，91832 帧）均值 564、标准差 54、99 分位 666、最大 710 mm，超过 650 mm 的占 7.1%，超过 700 mm 的不到 0.1%。zgz_local 深度 699–742（均值 720）mm，整条在训练范围之外；zgz_global 506–648（均值 562）在范围之内。`rt_cnntrack` 的深度偏差：local −89.3 / −89.3 mm（−12.4%），global +20.0 / +17.8 mm（+3.6%）；预测深度对 GT 深度的回归斜率 global 0.62 / 0.67，local −0.08 / −0.29；深度误差与 GT 深度的相关 −0.57 至 −0.65。local 的"保持初始位姿"平移误差只有 11.5 mm，闭环 90.6 mm，教师强制 66.0 mm | `tools/dt/transl_structure.py` → `outputs/dt/reports/transl_structure.md`；深度分布由训练集 `.meta` 的 50 ms 步位姿统计 |
| F5 | F4 的偏差不是某个臂的特例 | 同一统计下：`rt_cnn`（绝对 CNN）local 深度偏差 −110 至 −112 mm，`rt_s37` −83 至−100 mm，`rt_cnnf` 与 `rt_cnn` 相同 | 同上 |
| F6 | 平移几乎没有被监督 | `so3_trans_fk` 的平移项是 SmoothL1，转折点 beta = 1.0 m：厘米级误差全落在二次区，梯度接近 0；FK 项是根相对的。SO3FK 使绝对 MPJPE 从 63.7 退到 83.6，根相对 MPJPE 变好（历史 §13）。补救键 `LOSS.TRANS_BETA`、`LOSS.ABS_FK_WEIGHT` 已在代码里，从未在这个臂上训练过 | `model/model.py` `BaseModel.__init__` 的 S4 注释 |
| F7 | 尺度增强打断"像素尺寸 ↔ 深度"线索 | DomRand 的尺度以 K' = A·K 实现、3D 标签不变，等价于随机改焦距而手的深度不变；网络若不读 K'，同样大小的手在 ±25% 范围内可能对应不同深度。本数据集只有一台相机，焦距恒定，这一增强没有对应的测试时变化 | `semkine/domrand.py` `transform_labels`；主表脚注 |
| F8 | 根旋转用轴角相加更新 | 训练集根轴角模长中位数 132°，7.4% 超过 150°，0.7% 超过 170°；加法步长是真实转角的 1.19 倍；S38 已证明绝对根必须用参考旋转 + chordal 损失 | `docs/S38_ROOT_TRACKING_PREREG.md` §1 |
| F9 | 没有任何输出滤波 | 逐包 CNN 加常数增益因果滤波 −0.82 mm，根速度比 2.0 → 1.2；历史 δ-trust 0.5 在旧渲染跟踪器上把抖动从 11.7 压到 6.1、RA −0.5，代码已删、从未在 dr_so3fk 上测；渲染跟踪器本身从未被滤波 | `docs/S37_ROOT_TRACKING_VERDICT.md` §5、§6、§8.5；历史 §10 |
| F10 | 渲染 + FK 占时最多，且有两处主机同步 | `_render_chunk` 的 `[valid]` 布尔索引和 `if pix.numel() > 0` 各触发一次同步；从未分解测量 | `model/model.py` |
| F11 | 骨干对 180×240 输入偏大 | 11.209 M 参数（conv1 111，prev_mlp 6.5 k，ResNet18 11.18 M），layer4 约 8.4 M；EvHand-FPV 把 11.2 M 降到 1.2 M 精度反升，EventEgo3D++ 1.25 M | 文献（EvHand-FPV 2025，EventEgo3D++ IJCV 2025，Lite-HRNet CVPR 2021） |
| F12 | 训练只有教师强制 + 高斯 prev 噪声 | 闭环 / TF 放大 1.37；10° 扰动有 8% 永不恢复；unroll、保持率惩罚、迭代精修都已失败；评测从未包含加速度指标 | `rt_cnntrack` evalx；`docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md`；历史 §14.4 |

已被实测否定、本轮不重复：迭代 render-and-compare（发散）、语义 mask、KSGN、恒定 UNROLL_P、保持率惩罚、两网络锚定（全量后收益消失）、输入相关的学习增益。

**读 F4 时的限定**：F4 是看过 zgz 深度之后才发现的。AGENTS.md 把 zgz 同时定为开发集和测试集，这一点在结论文档里明说。F4 催生的候选（`dt_dz`）不调任何参数：尺度范围沿用 DomRand 已有的 0.8–1.25，只改尺度的标签语义（见 §2）。

## 2 臂与因素

训练配方与所有 `rt_*` 臂相同：单卡 × 512 × 梯度累积 2，Adam 4e-3，warmup 500，cosine 衰减到 2%，bf16，**6000 步**，500 步检查点网格。种子 3407 / 3408；基线 `dt_base`、`rt_cnntrack` 加种子 3409。所有臂都以 6000 步报告，不做 2000 步筛选判定（小网络在 2000 步收敛慢，会系统性偏低；2000 步配置由 `make_configs.py` 一并生成，只作信息）。

每个臂 = `configs/rt/rt_cnntrack.yaml` + `LOSS` 块（历史 so3fk 的原样：`TYPE: so3_trans_fk, ROT_WEIGHT: 1, TRANS_WEIGHT: 1, FK_WEIGHT: 2`，`LAMBDA_*`、`LOG10` 保留）+ 表中的键，由 `tools/dt/make_configs.py` 生成。

| 臂 | 因素 | 键（相对 `dt_base`） | 需要的代码 | 针对 |
|---|---|---|---|---|
| `dt_base` | 基线 | 无 | 无 | 统一配方下的目标架构 |
| `dt_tr`（C1） | 平移按其误差尺度监督 | `LOSS.TRANS_BETA: 0.01`，`LOSS.ABS_FK_WEIGHT: 1.0` | 无 | F6 |
| `dt_nos`（C2a） | 去掉尺度增强 | `AUG.DOMRAND.SCALE_MIN: 1.0`，`SCALE_MAX: 1.0`（随机数流不变：尺度从退化区间抽取） | 无 | F7；C2b、C2c 的对照 |
| `dt_cam`（C2b） | 射线平面输入 | `MODEL.CAM_PLANES: true`：两个额外输入通道 (u − cx) / fx、(v − cy) / fy，用渲染所用的（增强后的）内参，conv1 由 4 通道变 6 通道 | `model.py` | F7 |
| `dt_dz`（C2c） | 深度一致的尺度增强 | `AUG.DOMRAND.SCALE_MODE: depth`（默认 `focal` = 现有语义）：图像按 s 缩放，K 不变，标签的深度除以 s（X、Y 不变）。同一幅图像，原语义解释为"焦距乘 s"，这里解释为"手的深度除以 s"。尺度范围仍是 0.8–1.25 | `domrand.py`、`dataset.py` | F4、F7 |
| `dt_trnos` | C1 与 C2a 的析因臂 | `dt_tr` 的键 + `dt_nos` 的键 | 无 | 两个零代码因素的交互；无独立判定门 |
| `dt_so3c`（C3） | 根旋转在 SO(3) 上合成 | `MODEL.ROOT_COMPOSE: so3`（默认 `add`）：R = Exp(δ_root) · R_prev（左乘，与 evalx 的扰动、DomRand 的 roll、S38 的 R_ref 同一约定）；prev_mlp 的根项先并入 δ；空包逐位返回 prev；手指和平移不变 | `model.py` | F8 |
| `dt_acc`（C4） | 加速度损失 | `LOSS.ACCEL_WEIGHT: 1.0`，`DATA.TRIPLET: true`：三个等长的连续窗（目标间隔恰为一个窗长），同一个 DomRand 实现，各自 GT + 独立噪声的 prev（教师强制）；对绝对 FK 关节的二阶差分取 L1（预测减 GT），加到主损失里再取 log10；只对 1 / 3 的样本取三连窗 | `dataset.py`、`model.py` | F3、F12；证据弱，最后一个名额 |
| `dt_w05`（C5） | 半宽 ResNet18 | `MODEL.CNN_BACKBONE: resnet18_w0.5`（宽度 32-64-128-256，层结构、BN、7×7 步长 2 的茎、最大池化不变） | `model.py`、`model/backbones.py` | F11 |
| `dt_l3`（C6） | 去掉 layer4 | `MODEL.CNN_BACKBONE: resnet18_l3`（layer4 → Identity，fc 从 256 起） | 同上 | F11 |
| `dt_w05_kd`（C7，条件） | C5 + 蒸馏 | C5 + `MODEL.DISTILL_WEIGHT: 1.0`，`DISTILL_CKPT:` 同种子 `dt_base` 的最后检查点 | 同上，且需修复稠密批的蒸馏（现为静默空操作） | F11；只在 `dt_base` 跑完且 C5 / C6 有精度缺口时启动 |

`MODEL.CNN_BACKBONE` 不叫 `BACKBONE`：`MODEL.BACKBONE` 是许多旧配置里的死键。默认 `resnet18` 返回与现在完全相同的 `torchvision.models.resnet18(num_classes=51)`，旧臂的 state_dict 键和初值不变。

**输出滤波**（零训练，不占名额）：`semkine/anchored.py` 的 `FilteredTracker(trk_model, a_root, a_rest, a_trans)`，回灌状态 = `anchor_blend(prev, 跟踪器输出, …)`：根旋转测地插值，手指和平移线性，空包保持 prev，增益 1 = 裸跟踪器。**增益事先固定：根 0.5、手指 1.0、平移 0.5**（手指速度比已只有 0.32，不再压）。另报根 / 手指 / 平移 ∈ {0.25, 0.5, 0.75, 1.0} 的若干组合作敏感性参考，**只作描述，不在 zgz 上选择**。

## 3 调试门（训练之前，每个臂都要过）

`tools/rt/debug_arm.py`（前向 / 反向、能学、评测确定性、状态契约、微型过拟合、闭环通路、运行时间），再加每个新键各自的门：

| 门 | 要求 |
|---|---|
| SO(3) 合成（`dt_so3c`） | 空包逐位返回 prev；δ → 0 时输出等于 prev；prev 的根转 10° 时输出转 10°；训练与评测两条路径一致；根损失在 GT 处为 0 |
| 滤波律（`FilteredTracker`） | 根扰动的保留率符合 (1 − g)^k；增益 1 逐位等于裸跟踪器；空包返回 prev |
| 射线平面（`dt_cam`） | K 变则输出变，K 不变则输出逐位不变；平面数值与手算一致 |
| 深度一致增强（`dt_dz`） | `focal` 模式（默认）下增强结果逐位等于现有实现；`depth` 模式下 256 个训练包的顶点经图像变换后与"标签变换后在 K 下重新投影"的像素之差，平均 ≤ 2 px（180×240 像素） |
| 加速度损失（`dt_acc`） | 恒速轨迹上为 0；`TRIPLET: false` 时批逐位等于现有数据集；`TRIPLET: true` 时 w2 逐位等于非三连窗的同一样本 |
| 骨干（`dt_w05`、`dt_l3`） | 输出形状 (B, 51)；默认名的 state_dict 键、形状和（同一随机种子下的）初值与现有逐位相同；每个参数都有有限梯度 |
| 渲染无同步（并入主干的前提） | 对 2000 个随机状态（含相机后方、画面外、同像素碰撞、全无效、B = 1 / 7 / 256 / 300）与现有 `_render_chunk` 的输出 `torch.equal` |
| 可复现性 | 任何对 `model.py` / `dataset.py` / `domrand.py` 的改动之后：`rt_cnntrack_s3407` 最后一步重评的 RA 逐位等于 13.69480423503861；`tools/s38/s37_repro.py` 得 23.557357022200772 |

**已通过（训练之前，GPU 0 / 1 / 2，未训练权重）**：`dt_base`、`dt_tr`、`dt_nos` 全部通过 `debug_arm.py` 的 7 项：
- 参数量 11.209 M；前向 / 反向有限，无未用参数；
- 40 步内 log10 损失 −0.614 → −2.085（`dt_base`）、−0.054 → −1.085（`dt_tr`，损失含绝对 FK 项，量级不同）、−0.609 → −2.106（`dt_nos`）；
- 评测确定性（两次前向差 0）；状态契约（空包等于 prev，10° 扰动的输出转 10.00°）；
- 微型过拟合 1500 步：根误差 0.21° / 0.33° / 0.26°；
- 闭环通路有限；批 1 延迟 3.60 ms 原始，折合 4.48 ms（`dt_base`）。
日志在 `outputs/dt/debug/`。`dt_nos` 的尺度区间退化为 [1, 1]：`exp(uniform(log 1, log 1)) = 1.0`，且抽取的随机数个数不变，所以其余增强的随机数流与 `dt_base` 相同。

## 4 评测协议（基线与候选统一）

- `tools/tracking/evalx.py eval --ckpt last --controls --tf --perturb`，GPU、fp32、batch 1、50 ms 递推，每段从 GT 加协议噪声起步（rng 0），split `val_core`。**只报最后一步，不选点。** 12 个检查点的网格中位数（`tools/select_checkpoint.py`）只作稳健性参考，从不用来选点。
- zgz 是唯一的评测受试者，同时用作筛选判定。本轮的结论都带这个限定。
- 报告的量：
  - 主表各列（AGENTS.md 的格式，只由 `tools/report_table.py` 生成，每个入表的臂先过 `tools/make_s36_row.py`）；
  - 扩展指标在本文档和结论文档里：RA（overall / global / local）、根旋转误差、绝对平移与绝对 MPJPE（overall / global / local）、MPVPE、教师强制 RA 与闭环放大、10° / 20° 扰动在 k = 1、2、5、10、20 步的保留率、根速度比和手指速度比、失败段、按事件率分档的误差；
  - **新增抖动与加速度**（`evalx.jitter_decomp`，绝对关节，同一段内）：jit_pred / jit_gt（一阶差分，mm / 步）、jit_ratio、acc_pred / acc_gt（二阶差分）、acc_err（预测减 GT 的二阶差分范数）、acc_ratio、静止步抖动（0.5 mm 阈值）、根角加速度；
  - **抖动拆分**：把 acc_err 按平移 / 根 / 手指做 Shapley 归因（8 个联盟取值，三项之和 = acc_err，容差 1e-6）；
  - **平移结构**（`tools/dt/transl_structure.py`）：各轴偏差、深度偏差占比、深度回归斜率、按段偏移与段内波动；
  - 参数量、MACs（`forward_packet` 整体）、延迟（正式测速：空闲 GPU、固定 CPU 核、与 `dt_base` 交替，`tools/s38/bench.py`）。
- 滤波行：最终模型各报"裸跟踪器"和"`FilteredTracker`（0.5 / 1.0 / 0.5）"两行。

## 5 判定规则

记号：对臂 A 和种子 s，Δm(A, s) = m(A, s) − m(`dt_base`, s)（同种子配对，最后一步）；Δm̄ 为 3407 / 3408 两个种子的均值。RA、ABS、ROT、ACC 依次是 overall 的 `mpjpe_ra_mm`、`mpjpe_abs_mm`、`root_rot_deg` 和 `jitter.acc_err_mm`。**1.1 mm 为平局带；两个种子只能可靠分辨 ≥ 2.5–3 mm 的差；介于 1.1 与 2.5 mm 之间的结果一律标"待 3409 确认"，并用种子 3409 复核。**

**各候选的门**（6000 步，种子 3407 / 3408，须两个种子与均值同号）：

| 臂 | 通过条件 |
|---|---|
| `dt_tr` | ΔABS̄ ≤ −5 mm 且两个种子都为负；ΔRA̅ ≤ +0.5 mm |
| `dt_nos`、`dt_cam`、`dt_dz` | ΔABS̄ ≤ −3 mm 或 ΔRA̅ ≤ −1.1 mm |
| `dt_so3c` | ΔROT̄ ≤ −0.5° 且 ΔRA̅ ≤ −1.1 mm；护栏：global 根速度比 ≥ 0.9 |
| `dt_acc` | ACC̄ 相对 `dt_base` ≤ −15%；ΔRA̅ ≤ +0.3 mm；ΔABS̄ ≤ +1.1 mm；护栏：手指速度比下降 ≤ 0.05 |
| `dt_w05`、`dt_l3`、`dt_w05_kd` | RA̅ ≤ 1.02 × `dt_base` 的 RA̅，且（参数量 ≤ 0.5 × 或正式测得延迟 ≤ 0.8 ×）；或 ΔRA̅ ≤ −1.1 mm |

`dt_cam` 另与 `dt_nos` 比较（描述性）：射线平面的贡献是否超出"去掉尺度增强"本身。`dt_trnos` 与 `dt_tr`、`dt_nos`、`dt_base` 构成 2 × 2 析因，只报交互项，不设门。

**通用护栏**（任一触发，即使过门也不采纳）：ΔABS̄ > +2 mm；local RA 的 Δ̄ > +1 mm；jit_pred 或 acc_err 的种子均值上升超过 5%；global 根速度比 < 0.9 或手指速度比比 `dt_base` 下降超过 0.05；失败段数增加；闭环 / TF 放大（RA）比 `dt_base` 大超过 0.05。

**结论口径**：
- "优于 `dt_base`"要求过门、无护栏触发，且 1.1–2.5 mm 的结果经 3409 确认。
- 只在两个种子里有一个成立、或在 ±1.1 mm 内的，写"平局"，不写"有效"。
- 与历史 `track_render51_dr_so3fk` 的比较只作参照：配方不同（batch 2048、3000 步、warm-start），不做配对判定。
- 未取得提升或遇到硬阻塞时如实记录。

## 6 组合与最终确认

按下列事先固定的规则，在 Stage 2 的结果出来之后合成臂：
1. **精度包**：通过门的 C1、C2（C2a / C2b / C2c 三选一：取通过门且 ABS̄ 最低者；相差不到 1.1 mm 时取最简单的，顺序 `dt_nos` < `dt_dz` < `dt_cam`）、C3、C4。
2. **参数包**：C5 或 C6（取通过效率门且参数量更少者）。
3. 组合臂 = 精度包；组合臂 = 精度包 + 参数包。每个组合 6000 步，种子 3407 / 3408，再加 3409；`dt_base`、`rt_cnntrack` 也有 3409，三种子配对。
4. 最终模型套 `FilteredTracker`（0.5 / 1.0 / 0.5），报滤波前后两行。滤波"有效"的判据：ΔRA̅ ≤ 0 且 acc_err 下降 ≥ 15%。
5. 主表（`tools/report_table.py`）必须含 EventHands-PCA6 基线、S37 当前臂、历史 EventHands-Track 缓存行、`dt_base`、`rt_cnntrack` 和入选臂。

## 7 限制与声明

- zgz 既是开发集也是测试集（AGENTS.md）。F4 在看过 zgz 深度之后才被发现，`dt_dz` 因此带有"由开发集结果催生"的限定。它没有可调参数，门槛与其他候选相同。
- zgz_local 的深度在训练范围之外，其绝对误差反映外推能力，不只是跟踪精度；RA 对平移不敏感。主表分列 local / global，扩展指标里单独给出 global 的绝对误差。
- 两个种子的分辨力有限（见 §5）；`dt_base` 的 3409 种子用来估计种子间波动。
- 历史 `track_render51_dr_so3fk` 的 RA 12.223 与 `dt_base` 的配方不同；`dt_base` 的数字才是配对基线。
- 渲染无同步改写、`FilteredTracker`、`jitter_decomp` 都是只加不改：旧路径在可复现性门下逐位不变。

## 8 复现命令

```
python tools/dt/make_configs.py                                    # configs/dt/*.yaml
python tools/dt/jobs.py --arms dt_base dt_tr --seeds 3407 3408 --prio 100 > /tmp/jobs.json
SCHED_PROG=outputs/dt python tools/tracking/sched.py daemon        # 调度器，GPU 列表在 outputs/dt/sched/allowed_gpus
SCHED_PROG=outputs/dt python tools/tracking/sched.py submit /tmp/jobs.json
python tools/tracking/evalx.py eval --run-dir outputs/semkine/<run> --ckpt last --controls --tf --perturb
python tools/dt/transl_structure.py --runs "rt_cnntrack_s34*"     # F4、F5
python tools/rt/debug_arm.py --config configs/dt/<arm>.yaml        # 调试门
```


## 9 追加登记（2026-10-03 07:20；第一波训练进行中，没有任何 DT 训练跑完，没有任何 DT 评测结果）

### 9.1 新增诊断 F13：推理期 `prev_mlp` 消融

`tools/dt/ablate_eval.py`：对已记录的 `rt_cnntrack_s3407` / `s3408` 最后检查点，把 `prev_mlp` 最后一层的若干输出行（权重和偏置）置零，其余不动。GPU、fp32、批 1，协议同 §4。不重训，不选点。"无"行逐位复现记录值（RA 13.695 / 13.800）。平移误差、绝对 MPJPE 的单位是 mm。结果文件在 `outputs/dt/ablate/`。

| 种子 | `prev_mlp` 置零的输出 | RA | 绝对 MPJPE | 平移误差 global / local | 绝对 MPJPE global / local | 教师强制平移误差 global / local |
|---|---|---|---|---|---|---|
| 3407 | 无（原检查点） | 13.69 | 56.8 | 22.7 / 90.6 | 28.1 / 89.9 | 11.7 / 66.0 |
| 3407 | z 行（第 2 行） | 14.21 | 49.4 | 54.8 / 36.1 | 59.0 / 38.4 | 30.2 / 32.2 |
| 3407 | 平移三行（0:3） | 13.59 | 62.6 | 62.1 / 61.1 | 64.7 / 60.3 | 41.9 / 41.3 |
| 3407 | 根旋转三行（3:6） | 93.05 | 99373.9 | 175432.8 / 11862.2 | 175402.4 / 11852.7 | 11.7 / 66.0 |
| 3407 | 全部 51 行 | 109.53 | 115.4 | 45.4 / 81.4 | 118.8 / 111.5 | 41.9 / 41.3 |
| 3408 | 无（原检查点） | 13.80 | 57.2 | 21.1 / 91.7 | 25.6 / 93.6 | 10.9 / 68.0 |
| 3408 | z 行（第 2 行） | 15.41 | 44.8 | 42.0 / 39.2 | 46.7 / 42.5 | 20.2 / 36.8 |
| 3408 | 平移三行（0:3） | 14.84 | 57.6 | 50.2 / 64.0 | 52.8 / 63.1 | 33.8 / 47.9 |
| 3408 | 根旋转三行（3:6） | 90.90 | 124916.9 | 222349.8 / 12800.4 | 222318.9 / 12791.2 | 10.9 / 68.0 |
| 3408 | 全部 51 行 | 97.54 | 95.5 | 51.8 / 92.1 | 88.3 / 103.7 | 33.8 / 47.9 |


读数（两个种子方向一致）：
- **根旋转输出是必需的状态反馈。** 置零根旋转三行，闭环崩溃（RA 91–93，平移跑飞）；全部置零同样崩溃（RA 98–110）。网络离不开 `prev_mlp` 的旋转输出。
- **深度输出是向训练均值的拉力：训练范围内有益，训练范围外有害。** 置零平移三行后，zgz_local 的平移误差变小，zgz_global 的变大（上表）。只置零 z 行：zgz_local 平移误差 90.6 → 36.1 mm（种子 3408：91.7 → 39.2），教师强制 66.0 → 32.2 mm，绝对 MPJPE 56.8 → 49.4 mm（44.8）；zgz_global 平移误差 22.7 → 54.8 mm（21.1 → 42.0），RA 13.69 → 14.21（13.80 → 15.41）。
- 去掉 `prev_mlp` 的平移项后，仅由 CNN 给出的教师强制平移误差在两个序列上都在 34–48 mm：CNN 自己的平移输出有很大的系统偏差，在训练范围内由 `prev_mlp` 的平移项抵消，训练范围之外抵消失效。
- 这是推理期的切除，网络没有机会适应。它说明机制，不预测重训后的结果。

### 9.2 新增候选

| 臂 | 因素 | 键（相对 `dt_base`） | 需要的代码 | 针对 |
|---|---|---|---|---|
| `dt_nopm`（C9a） | 去掉事件无关的 `prev_mlp`，上一状态只经渲染进入 | `MODEL.PREVPOS_EMBED: false` | 无 | F4、F13 |
| `dt_pmt`（C9b） | 保留 `prev_mlp` 的根旋转与手指输出，屏蔽它的三个平移输出（层的形状不变，state_dict 可互相加载） | `MODEL.PREV_MLP_TRANSL: false`（默认 `true`） | `model.py` | F4、F13 |

判定门与 C2 族相同：ΔABS̄ ≤ −3 mm 或 ΔRA̅ ≤ −1.1 mm，两个种子须同号；通用护栏不变。
**事先写下的预期和风险**：F13 显示旋转输出对稳定是必需的，所以 `dt_nopm` 是强干预，很可能整体变差（根旋转失去状态反馈）。`dt_pmt` 保留旋转，只改平移；但 F13 同时显示平移项在训练范围内有益，重训后 zgz_global 的绝对误差可能变差，而 zgz_local 变好。两者都按同一门判定，不事先假定方向。

### 9.3 组合规则的补充

- 精度包的候选扩为：C1、C2（`dt_nos` / `dt_dz` / `dt_cam` 三选一，规则同 §6）、C9（`dt_nopm` / `dt_pmt` 二选一：取通过门且 ABS̄ 最低者，相差不到 1.1 mm 时取 `dt_pmt`）、C3、C4。每个因素只有单独通过门且无护栏触发时才进入精度包。
- 组合臂之外，时间允许时另加留一法臂（去掉对 ABS̄ 贡献最大的因素），只作描述。

### 9.4 主表行

入主表的臂先生成主行，再由 `tools/report_table.py` 出表（AGENTS.md）：
`python tools/tracking/evalx.py row --arm <arm> --runs outputs/semkine/<arm>_s3407 outputs/semkine/<arm>_s3408 --ckpt last --variant tf_pert`，在空闲 GPU 上执行，延迟按该工具的口径测量。

### 9.5 新增诊断 F14：训练序列上的深度精度

`tools/dt/depth_insample.py`：对 `rt_cnntrack_s3407` / `s3408` 的最后检查点，在**训练序列**上（唯一的 18 条，不含 `_v2`–`_v4` 重复）跑教师强制和闭环，按序列深度统计平移误差和深度（z）偏差。这是已记录检查点的样本内诊断，不是评测。结果文件 `outputs/dt/ablate/*_depth_insample.{json,md}`。

| 种子 | 序列数 | 序列深度 mm | 教师强制平移误差 mm（均值，范围） | 闭环平移误差 mm（均值，范围） | z 偏差绝对值最大值 mm（教师强制 / 闭环） | 深度 ≥ 600 mm 的序列：闭环平移误差均值 | z 偏差对深度的斜率（教师强制 / 闭环） |
|---|---|---|---|---|---|---|---|
| 3407 | 18 | 489–660 | 3.5（2.7–4.4） | 5.1（3.5–7.3） | 2.4 / 4.7 | 5.2（5 条） | -0.002 / -0.006 |
| 3408 | 18 | 489–660 | 4.1（3.2–4.9） | 6.0（4.1–8.6） | 3.1 / 7.1 | 5.9（5 条） | +0.009 / +0.014 |

对照：同一检查点在没见过的受试者 zgz 上，范围内的 zgz_global（562 mm）平移误差 22.7 / 21.1 mm、深度偏差 +20.0 / +17.8 mm，范围外的 zgz_local（720 mm）平移误差 90.6 / 91.7 mm、深度偏差 −89.3 mm（F4）。

读数：
- **网络在训练序列上把深度测得很准，与深度无关**：整个训练深度范围（489–660 mm）内平移误差只有几毫米，z 偏差对深度几乎没有斜率。所以 zgz 上的深度失败不是"深度读不准"的一般能力问题，而是泛化问题。
- **解释（待训练验证，不是结论）**：9 个受试者每人只有两条序列，local 序列的深度几乎不变（标准差 3–11 mm），global 序列在受试者自己的深度附近变化 ±30–50 mm。网络可以把"受试者外观 → 深度"当作捷径记下来，而不必从手的像素大小读深度。现有的尺度增强（`SCALE_MODE: focal`）改变手的像素大小而保持深度标签不变，等于告诉网络"像素大小与深度无关"，强化了这个捷径。
- **`dt_dz` 的作用机制**：深度一致的尺度增强让同一受试者以 0.8–1.25 倍深度出现，把深度与受试者身份解耦，并把训练深度覆盖扩展到约 890 mm（含 zgz_local 的 699–742 mm）。预期写在训练之前：`dt_dz` 在 zgz_local 上的深度偏差绝对值明显变小；如果没有变小，这个解释不成立。

### 9.6 新增析因臂 `dt_trdz`

`dt_trdz` = `dt_tr` 的键 + `dt_dz` 的键（`LOSS.TRANS_BETA: 0.01`，`LOSS.ABS_FK_WEIGHT: 1.0`，`AUG.DOMRAND.SCALE_MODE: depth`），与 `dt_base`、`dt_tr`、`dt_dz` 构成 2 × 2 析因，种子 3407 / 3408，6000 步；只报交互项，不设判定门（报告工具计算 `(trdz − dz) − (tr − base)`）。F14 指向的两个因素（平移监督、深度解耦）可能互补，这个臂让精度包里的组合提前有数据；是否进入精度包仍按 §6 的规则由单因素的门决定。

## 10 结果已知之后的追加（2026-10-03 11:55）

**这一节写在第一波结果出来之后。** 已知的结果：`dt_base` ×3、`rt_cnntrack_s3409`、`dt_tr`、`dt_nos`、`dt_nopm`、`dt_dz`、`dt_trdz` 都已训练并评测（`dt_trnos`、`dt_pmt`、`dt_so3c` 及其后各臂尚在训练）。结果表在 `docs/DT_RENDER_TRACK_VERDICT.md`，配对门判定在 `outputs/dt/reports/screen_6k_wave1.md`。本节所有内容都属于"看过结果之后的决定"，与 §1–§9 不是同一性质，读的时候要分开。

### 10.1 已知的门判定（按 §5 原样，两个种子 3407 / 3408）

| 臂 | 门 | 说明 |
|---|---|---|
| `dt_tr` | 不通过 | ΔABS̄ −1.6 mm（要 ≤ −5），ΔRA̅ +2.5 mm（要 ≤ +0.5） |
| `dt_nos` | 不通过 | ΔABS̄ −0.6 mm，ΔRA̅ +1.0 mm（要 ΔABS̄ ≤ −3 或 ΔRA̅ ≤ −1.1） |
| `dt_nopm` | 不通过 | ΔRA̅ +2.5 mm，ΔABS̄ −0.5 mm |
| `dt_dz` | **门通过，护栏 G6 触发** | ΔABS̄ −21.6 mm（两个种子 −24.4 / −18.8），ΔRA̅ −0.09 mm（平局）；G6 放大比 +0.051，阈值 0.05 |

### 10.2 对 `dt_dz` 的 G6 触发：登记一次偏离

按 §5 的原文，G6 触发意味着 `dt_dz` 不得进入精度包。这里明确**偏离这条规则**，理由和撤回条件写在这里，之后不再改：
- 超出阈值 0.001（+0.051 对 0.05），两个种子分别是 +0.095 和 +0.008，种子间波动（`dt_base` 三个种子的放大比 1.248 / 1.293 / 1.276，标准差 0.023）远大于超出量。
- 放大比的分母是教师强制 RA。`dt_dz` 的闭环 RA 与 `dt_base` 持平（−0.09 mm），教师强制 RA 变好（−0.41 mm），放大比上升来自分母变小，不是闭环变差。扰动保留率、失败段、根速度比、手指速度比都在护栏之内。
- **撤回条件**：种子 3409 补齐后，三个种子的放大比差（`dt_dz` 减 `dt_base`）的均值若仍大于 0.05 加上 `dt_base` 的种子标准差（0.023），即大于 0.073，则撤回这次偏离，`dt_dz` 退出精度包，结论文档照实写。
- 无论如何，结论文档里 `dt_dz` 的行都会标注"护栏 G6 触发，按 §10.2 偏离采纳"。

### 10.3 看过结果之后新增的臂

| 臂 | 构成 | 种子 | 说明 |
|---|---|---|---|
| `dt_dz` 种子 3409 | 已有配置 | 3409 | `dt_dz` 效应最大，补第三个种子作样本外确认；同时用于 §10.2 的撤回条件 |
| `dt_dz_w05` | `dt_dz` + `dt_w05` | 3407 / 3408 | §6 的"参数包"提前在 `dt_dz` 之上试：`dt_dz` 通过了门，参数缩减不必再等 `dt_w05` 在 `dt_base` 上的结果；`make_configs.py --pack` 生成 |
| `dt_dz_l3` | `dt_dz` + `dt_l3` | 3407 / 3408 | 同上 |

这些臂没有独立的新门：参数缩减按 §5 的效率门（相对同种子的 `dt_dz`，而不是 `dt_base`）、其余指标按通用护栏描述。精度包的最终成员由 §6 的规则加上 §10.2 的偏离决定。

### 10.4 零训练的输出滤波

按 §2 事先固定的增益（根 0.5、手指 1.0、平移 0.5），用 `tools/dt/filter_eval.py` 套在已训练的 `dt_base`、`dt_dz`、`dt_trdz` 的最后检查点上，报滤波前后两行。**增益不在这些臂上调**，敏感性组合只在 `rt_cnntrack` 上做过（`docs/DT_FILTER_CNNTRACK_RESULT_20261003.md`），仅作描述。

### 10.5 缩回原计划规模（2026-10-03 12:55，用户决定，经另一个会话转达）

用户要求把 DT 轮缩回已批准计划的规模（训练臂不超过 8 个，约 40–45 GPU·h），并在 12:35 先要求把 GPU 0、1 空出来（调度器的可用 GPU 改为 2–7）。据此：
- **在启动之前取消**：§10.3 的 `dt_dz_w05`、`dt_dz_l3`（各两个种子），以及 `dt_pmt` 种子 3407 的重排队任务和 `dt_trdz` 种子 3409。§10.3 的这两个组合臂因此**不会运行**，该表只作"曾登记、已撤销"的记录。
- **保留**：计划内的 `dt_w05`、`dt_l3`（父配置 `dt_base`，提高了优先级，排在 `dt_cam` 之前）、`dt_cam`、`dt_dz` 的种子 3409（样本外确认）；`dt_acc` 在它的调试门通过之后运行，`dt_w05_kd` 只在 `dt_w05` / `dt_l3` 有精度缺口时作后备。
- **已在运行的不动**：`dt_pmt` 种子 3408、`dt_so3c` 两个种子、`dt_trnos` 种子 3408、`dt_tr` 种子 3409。
- **GPU 空出来的代价**：12:35 停掉了 GPU 0、1 上各约 50 分钟的 `dt_cam_s3407` 和 `dt_pmt_s3407`（未跑完的输出保留在 `outputs/semkine/_killed_*`）。`dt_cam_s3407` 重新排队，`dt_pmt_s3407` 不再排队，所以 `dt_pmt` 只有种子 3408，不做配对判定，只描述。
- **Stage 3**：组合臂只在单因素通过各自的门之后做，至多一两个。

### 10.6 指标优先级与采纳规则（2026-10-03 13:35，用户决定，经另一个会话转达）

主表（根对齐 MPJPE / MPVPE、递推 RA、延迟 / FLOPs / 参数量）是主要目标，论文结论建立在主表上；绝对指标仍要报告，但是次要。因此：
1. **采纳规则**：一个臂或因素只有在主表变好、或者主表落在 1.1 mm 平局带内而绝对指标变好时才采纳。§5 的门照原样判定并记录；凡是只靠绝对分支通过的门（`dt_tr` 的 ABS 条件、`dt_nos` / `dt_cam` / `dt_dz` 的 ABS 条件、`dt_trdz` 的结果），结论文档明确写成"只有绝对指标的收益，对主表没有影响"。`dt_dz` 的 RA 在平局带内（−0.09 mm），绝对 MPJPE −21.6 mm，符合采纳规则；§10.2 的 G6 偏离仍然单独记录。
2. **主表行**：用 `tools/tracking/evalx.py row`（最后一步、`--variant tf_pert`，S38 / rt 轮次对稠密臂用的同一条路径，`tools/make_s36_row.py` 的头部 MACs 脚注只适用于 S36 头部布局，对稠密 CNN 臂会拒绝运行）在空闲 GPU 上为 `dt_base`、`dt_dz`（裸跟踪器与带预注册滤波的两行）以及每个通过的臂生成主行，再由 `tools/report_table.py` 出表，表里含基线行和当前臂行。
3. **抖动口径**：除绝对关节外，`evalx.jitter_decomp` 现在同时给出根对齐口径（`acc_err_ra_mm`、`jit_ra_*`、`acc_ratio_ra`，减去 0 号关节，对 21 个关节取均值，与 `per_step_metrics` 的 RA 口径一致），两种口径并列报告。
4. **D0a 的事后说明**：D0a 的决策规则是按绝对关节的抖动定的（平移占比 ≥ 50% → 优先 C1 和平移滤波）。在根对齐口径下平移消失，剩下的是根旋转和手指，所以对主表有意义的抖动手段是 C3 `dt_so3c` 和 C4 `dt_acc`。根对齐口径下滤波的 `acc_err` 只降 18–24%，并且预测的根对齐加速度降到 GT 的 0.6–0.8 倍（过平滑），而不是绝对口径下的 44%。
