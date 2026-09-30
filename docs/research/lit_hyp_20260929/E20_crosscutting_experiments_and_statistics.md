# E20：跨问题实验设计——表述、训练分布、统计与信息价值排序

> 子代理 E20 · 2026-09-29。独立于文献组。不改仓库代码/配置/测试/`outputs/`；不训练；不启 GPU。  
> 事实摘要：`docs/research/lit_hyp_20260929/00_CONTEXT.md`。主文档对照：`docs/网络结构分析.md` §5–§7。  
> 脚本落点（若日后跑 CPU 分析）：`.experiments/lit_hyp_20260929/E20/`。

---

## 0. 已核对的钩子与成本锚点

### 0.1 表述开关【代码事实】

| 开关 | 位置 | 作用 |
|---|---|---|
| `PREDICT_DELTA` | `model/model.py:659`；应用 `1525-1530`（packet）/`1550-1555`（CNN） | true → `out = prev + Δ`；false → 绝对 51 维 |
| `PREVPOS_EMBED` | `model/model.py:660`；应用 `1518-1524` | `prev_mlp(prev)` 加到输出；看不到事件 |
| `ACTIVE_HEAD` | `model/model.py:702`；解码 `_decode_active:1155-1192` | 根 `Linear` + 15 个关节 MLP；手指头拼接 `prev` 的 3 维角 |
| `ROUTED_READOUT` | `model/model.py:668`；路由 `_route_nodes:1388-1396` | 用 **prev FK** 做几何路由；绝对臂必须关 |
| `UNROLL_PAIR` / `UNROLL_P` / `UNROLL_RAMP` / `UNROLL_RESIDUAL` | `model/model.py:231-275`、`636-650` | 残差式：`cond = main.prev_state + (est − lead.target)`；replacement 会收窄课程 |
| `TRAIN.TRAINABLE_PREFIXES` | `model/model.py:952-958` | 冻结主干只训前缀（例：`s37_rootinnov_*.yaml`） |

### 0.2 配置对照【代码事实】

| 配置 | 编码器 | 表述 | 关键键 |
|---|---|---|---|
| `configs/semkine/s1_abs_domrand.yaml` | ResNet18 + LNES | **绝对** | `PREDICT_DELTA: false`，`PREVPOS_EMBED: false`，`PREV_RENDER: false` |
| `configs/semkine/s1_track_domrand.yaml` | ResNet18 + LNES | **跟踪** | `PREDICT_DELTA/PREVPOS_EMBED/PREV_RENDER: true`，渲染比较 |
| `configs/semkine/s37_routed_s3407.yaml` | EventGNN | **跟踪** | delta + prev_mlp + 课程；`ROUTED_READOUT`；`PREV_RENDER: false` |

### 0.3 协议脚本【代码事实】

- `tools/run_zgz_protocol.sh`：双卡并行两种子 → `select_checkpoint.py`（zgz 递推 RA 选点，非 val_loss）。
- `tools/make_s36_row.py:63-95`：`rng = np.random.default_rng(0)`；复现门 `drift < 0.05` mm。
- `tools/select_checkpoint.py`：固定网格；打印网格中位数与 spread（选点增益上界）。
- `tools/probe_s37_cnnroot.py`：零训练融合 CNN 根进 S37 闭环（已有先例）。

### 0.4 E5.5b 坍缩【实验事实】

- `docs/FAILURE_AND_CLEANUP_LEDGER.md` E5.5b 行：恒定 `UNROLL_P=0.5`（无退火）→ TF ~42.5 / 递推 ~144.5；最优解是无视 prev，坍缩回绝对回归（~40 mm 档）。
- `debug_e55b_unroll_20260825.md` 已删（账本 2026-09-29 条）；结论以账本 + `model/model.py:245-254` 注释为准，标「已删记录的旁证未复核原文」。
- 修复形态：`UNROLL_RAMP` 退火 + **`UNROLL_RESIDUAL: true`**（E5.5c），避免 replacement 把课程收窄成「prev 可信」。

### 0.5 成本锚点【实验事实】

| 作业 | 成本 |
|---|---|
| S37 两种子全量（2×512，6000 步） | ~1.0 s/it → **约 1 h 45 min / 种子**（`00_CONTEXT`；账本 09-07） |
| CNN 绝对单卡 batch 1024，~4000 步网格 | 10:14 启动 → 11:32 含选点与融合探针（`S37_ROTW_CNNROOT_PREREG.md` §B）→ **~1.3 h / 种子量级** |
| 零训练 zgz 闭环 / 探针 | **数分钟 / GPU** |
| S37 主行 | 7.72 ms，0.827 G，0.73 M；CNN 全模型锚 1.76 ms / 1.653 G / 11.18 M |

---

## 1. 缺失的一格：2×2 {编码器 × 表述}

### 1.1 现状矩阵【实验事实】

|  | 绝对（不读 prev） | 跟踪（delta + prev 通路） |
|---|---|---|
| **CNN-LNES** | **13.56**（`s37diag_cnnabs`；13.31/13.80）【实验事实：`S37_ROTW_CNNROOT_PREREG.md` §5B】 | **11.9–13.3**（`track_render51_dr_*`；账本 09-06 条称「同协议、主表已有」）【实验事实出处待核：数字在 `FAILURE_AND_CLEANUP_LEDGER.md`，配置/主行路径未在本任务内打开核对】 |
| **EventGNN** | **缺** | S36 **23.17**；S37 **20.74**【实验事实：主行】 |

【推断】矩阵对角线对比才能把「编码器天花板」与「跟踪表述损害」拆开；目前所有 GNN 数字都叠在跟踪表述上，所有强 absolute 数字都叠在 CNN 上。

### 1.2 实验 X1：GNN 绝对（补缺格）— **P0**

**Hypothesis**【待实验验证的假设｜对应 M6】  
在相同域随机化与 zgz 协议下，若 EventGNN 的逐包信念本身不弱，则「GNN + 绝对 51 维、不读 prev」应接近 CNN 绝对（~13–16 mm 量级）；若远差于 CNN 绝对而接近/差于 S37 跟踪，则瓶颈在编码器/感受野（M5），不在 delta+课程表述（M6）。

**Modification**【推断｜设计；实现时需改配置，本轮不写代码】  
新建臂（建议名 `s37diag_gnnabs`），相对 `s37_routed_s3407.yaml` 的单变量意图：

| 键 | S37 | GNN 绝对 | 理由 |
|---|---|---|---|
| `ENCODER` / 图超参 | event_gnn 128/3/8/2048/32 | **相同** | 固定编码器 |
| `INPUT_MODE` | `raw_packed` | **相同** | |
| `PREDICT_DELTA` | true | **false** | 绝对输出 |
| `PREVPOS_EMBED` | true | **false** | 不读 prev_mlp |
| `ROUTED_READOUT` | true | **false** | 路由依赖 prev FK【代码事实：`_route_nodes`】 |
| `PREV_RENDER` | false | false | 已状态无关 |
| `ACTIVE_HEAD` | true | **false**；`pose_head: Linear(512→51)` 或 `512→256→51`（非线性头） | 避免手指头仍拼接 `prev` 角【代码事实：`_decode_active:1190-1191`】 |
| `ZERO_EVENT_GATE` | true | false（无 delta 可关） | |
| `AUG.DOMRAND` + 极性 | 开 | **同开** | 与 CNN 绝对公平 |
| 训练日程 | 2×512，6000 | **同 S37**（或先单种子 6000 诊断） | |

评测：与 CNN 绝对相同——递推协议下每步不读自预测（`PREDICT_DELTA: false` 时 `forward` 不把 prev 加回去）【代码事实：`1525-1530`】；仍走 `select_checkpoint` / `make_s36_row` 以便主行可比。

**与对照差几个变量**【推断】

| 对比 | 差的变量（应尽量只留这些） |
|---|---|
| vs CNN 绝对（`s1_abs_domrand` / `s37diag_cnnabs`） | **编码器与输入表示**：EventGNN+raw tokens vs ResNet18+LNES；头：`pose_head` vs ResNet `fc`；参数量/MACs/延迟不同。表述（绝对、不读 prev）、domrand、损失、zgz 协议应对齐。 |
| vs S37 | **表述三件套**：绝对 vs delta；无 prev_mlp；无路由读出 / ACTIVE 手指头。编码器相同。 |

**Control**  
(1) 已有 `s37diag_cnnabs`；(2) 当前臂 `s37_routed` 主行；(3) 可选：同配置但 `PREDICT_DELTA: true` + 干净 GT prev 教师强制（诊断用，非主行）——若绝对差而 TF-delta 好，则是「无 prev 时不可观」而非编码器死。

**Metric**  
主表八列（`report_table.py`）；辅：根旋转 °、仅手指 RA、TF 单步（诊断 JSON，不进主表）。

**Expected observation**【推断】  
三种区间：  
- A：GNN 绝对 ≲ 16 mm（接近 CNN）→ 编码器够用，S37 的 20.74 主要被跟踪表述/课程拉坏。  
- B：GNN 绝对 ≳ 25 mm（接近/差于 S37）→ 编码器是天花板；改表述救不了根旋转盲。  
- C：介于 16–22 → 两者都贡献；下一步做「同编码器上的绝对→跟踪」课程消融。

**Positive result means**  
区间 A：采纳「先绝对后跟踪」或「绝对根支路」路线；H1/H4 曝光偏差优先级下降，M6 升为真。  
**Negative result means**  
区间 B：证伪「表述是主因」；转向 M5（感受野/层级）或 CNN 蒸馏/替换根，而不是再堆 unroll/门控。

**Compute cost**  
两种子 × ~1 h 45 min ≈ **3.5 GPU·h 墙钟**（可两种子并行占 4 卡）；单种子诊断可先 **~1.75 h**。  
**Priority：P0**

**钩子**  
`configs/semkine/s37_routed_s3407.yaml` 派生；`model/model.py` 的 `PREDICT_DELTA`/`PREVPOS_EMBED`/`ROUTED_READOUT`/`pose_head`；`tools/run_zgz_protocol.sh`；`tools/make_s36_row.py:eval_seed`；对照 `tools/probe_s37_cnnroot.py`。

---

## 2. 训练与测试 prev 分布（M8 / H4-exposure）

### 2.1 实验 X2：零训练——课程 prev vs 闭环 prev 单步误差 — **P0**

**Hypothesis**【待实验验证的假设｜M8】  
同一冻结 S37 checkpoint 上，在「课程噪声 prev」（训练分布）与「闭环自身 prev」（测试分布）下，单步 RA / 根旋误差的条件期望不同；若差距大且方向与「往均值收缩」一致，则曝光偏差成立；若两者接近且都远差于 GT-prev TF，则主因是逐包信念偏差（M2），不是分布偏移。

**Modification**  
零训练探针（脚本放 `.experiments/lit_hyp_20260929/E20/`，日后 GPU 跑）：对 zgz 每步固定事件包，分别条件于  
1. `prev = GT`（TF）；  
2. `prev = GT + 课程噪声`（复现 `semkine/dataset.py:221-251` 的混合：50/30/20）；  
3. `prev = 闭环上一步输出`（已有轨迹可从 `eval_track.track_sequence` 回放）；  
4. （可选）`prev = GT + 闭环残差`（模仿残差 unroll 的条件分布，不训练）。

**Control**  
同 checkpoint、同包、同 `rng` 协议（`make_s36_row` 的 seed 0）；变的只有 prev 来源。

**Metric**  
单步 RA、根旋 °、更新方向余弦、与「所需 Δ」的投影；分 global/local；分课程工况（小/大/相关）。

**Expected observation**【推断】  
【实验事实线索】TF 9.78 vs 闭环 20.71；同包迭代收敛≈闭环；prev 干净时单步比原地不动差 2.7–3.4×（`00_CONTEXT` §4）。预期：课程大噪声工况的单步接近闭环量级，小噪声/干净工况注入额外误差（与 §9.1 Wiener 收缩一致）。

**Positive result means**  
课程分布未能覆盖闭环相关误差，或干净 prev 上过度纠正 → 支持残差 unroll / 历史 dropout（TIP 类），进入 X3。  
**Negative result means**  
三种 prev 下单步信念误差平台相同 → 证伪 M8 为第一瓶颈；回到 X1/测量端。

**Compute cost**  
**数分钟–0.5 GPU·h**（零训练）。  
**Priority：P0**

**钩子**  
`semkine/eval_track.py:track_sequence`；`semkine/dataset.py` 噪声采样；已有探针风格 `tools/probe_s37_cnnroot.py` / `probe_s37_route.py`。

### 2.2 实验 X3：残差式 UNROLL（只改训练）— **P1**（X2 支持后）

**Hypothesis**【待实验验证的假设】  
在保留课程覆盖的前提下注入自误差的时间相关结构（`UNROLL_RESIDUAL: true` + `UNROLL_RAMP`），可缩小 TF→闭环差距，而不重演 E5.5b 坍缩。

**Modification**  
相对 `s37_routed` 只开：  
`TRACK.UNROLL_PAIR: true`，`UNROLL_P: 0.5`，`UNROLL_RAMP: [500, 2000]`，`UNROLL_RESIDUAL: true`【代码事实：`model/model.py:252-274`】。  
**禁止**：`UNROLL_RESIDUAL: false` 的 replacement；禁止从 step 0 恒定高 p（E5.5b）。

**Control**  
同种子 S37；可选 ablation：同 ramp 的 replacement（预期递推变差，复现注释中的 27.8→31–34）。

**Metric**  
主表 RA；TF 单步；「放大率」仅作描述；网格中位数（防选点噪声）。

**Expected observation**【推断】  
若 M8 真：闭环 −1–3 mm 且两种子同向；若 M2 真：TF 可能略好、闭环打平/变差（重演「过度信任历史」）。

**Positive / Negative**  
过门 ≥1.1 mm 且逐种子不劣 → 采纳为训练配方；否则记入账本，不再堆 scheduled sampling。

**Compute cost**  
全量两种子 ≈ **3.5 GPU·h**（额外 forward 使 step 变慢，【推断】墙钟可能 +20–40%）。  
**Priority：P1**

**钩子**  
`MNISTModel._maybe_unroll`；账本 E5.5b；`run_zgz_protocol.sh`。

---

## 3. 统计与协议：检出 1.1 mm 要几个种子；如何减选点=上报偏差

### 3.1 种子数（功效粗算）【推断｜基于实验事实散布】

记种子间差 `g = |s3407 − s3408|`，粗估 σ ≈ `g/√2`（两 i.i.d. 正态）。双侧 α=0.05、功效 0.8、正态近似、**相对已知基线**的均值检验：

| 散布来源 | g (mm) | σ̂ | 检出 **1.1 mm** 所需种子 n | 检出 2.0 mm | 检出 2.43 mm（S37−S36） |
|---|---|---|---|---|---|
| S36 overall | 3.5 | ~2.48 | **~40** | ~12 | ~8 |
| S37 overall | 3.03 | ~2.14 | **~30** | ~9 | ~6 |
| S37 local | ~6 | ~4.24 | **~117** | ~35 | ~24 |

【实验事实】采纳门：均值改善 ≥1.1 mm 且逐种子不劣；|Δ|<1.1 打平（`FAILURE_AND_CLEANUP_LEDGER.md` §1.2）。  
【推断】**在观测散布下，两种子对 1.1 mm 远不够功效**；两种子只对 ≳2.5–3 mm 且同向的效应可靠（S37 vs S36 属此类）。主表口径不改为「要 30 种子」，而是把 1.1 mm 门理解为**筛选阈值 + 同向约束**，小效应必须靠下方诊断协议，不能当已证伪。

两臂各 n 种子比较时，每臂样本量约为上表「vs 已知」的约 2 倍（方差 ×2）。

### 3.2 选点 = 上报的偏差怎么减（不改主表口径）【推断】

主表仍：`select_checkpoint` 选中点 → `make_s36_row` 复现。诊断层并行报告：

| 手段 | 做法 | 减小什么 |
|---|---|---|
| **网格中位数** | `select_checkpoint.py` 已打印；S37 中位 23.52/24.33 vs 选中 ~19–22【实验事实：`S37_ROUTED_READOUT_PREREG.md`】；CONTEXT：中位比选中高 **2–4 mm** | 选点乐观偏置 |
| **两条序列交叉选点** | 用 `zgz_global` 选点、在 `zgz_local` 上报（及对换）；或 leave-one-sequence | 选点集=上报集重合【实验事实：`00_CONTEXT` §2】 |
| **固定步报告** | 额外报 step=2500/5000/6000 的 RA，不只报 argmin | 网格捞针 |
| **逐段配对 bootstrap** | 对 90 个 local 段 + global 帧块做配对 Δ 的 bootstrap 95% CI（账本 §1.2 已列） | 种子少时的不确定性 |
| **预注册主指标用中位或配对** | 仅写入 prereg「机制门」，**不改** `report_table` 列 | 决策与论文主表分离 |

**最小改动诊断协议（建议预注册模板）**

1. 照常跑两种子 + 主行（口径不变）。  
2. 附加 JSON：网格中位数、交叉选点 RA、固定三步 RA、local 段配对 bootstrap CI。  
3. **采纳**：主表过 1.1 mm 且同向 **且**（中位数同向改善或交叉选点同向）。  
4. **打平**：主表 |Δ|<1.1 或种子反向；即使选中点好看也不叙事。

---

## 4. 延迟感知：7.72 ms 预算内的绝对根支路

目标：在 **≈7.72 ms** 总预算内加一条**绝对根（≥旋转）**通路，服务「CNN 根替换 −3.8 mm」类收益【实验事实：`probe_s37_cnnroot` fuse_R → 16.93】而不把系统做成双倍 ResNet。

| 选项 | MACs 量级【推断｜相对锚点】 | 延迟增量【推断】 | 可行性 |
|---|---|---|---|
| A. 完整 ResNet18-LNES 绝对根（现成 CNN） | +~1.65 G | +~1.75 ms → **合计 ~9.5 ms** | **超预算**；仅作 oracle 上界（已有探针） |
| B. 半分辨率 LNES（120×90）轻量 CNN → 6D 根 | ~1/4 → **~0.4 G** | ~0.4–0.6 ms → **合计 ~8.2–8.3** | 临界；需实测 `profile` |
| C. 更浅骨干（如 3–4 层 conv / MobileNet-ish）→ 6D | **0.05–0.2 G** | ~0.2–0.5 ms → **≤8.2 ms** | 优先实测 |
| D. 不新建编码器：冻结 S37，`TRAINABLE_PREFIXES` 只训「feat→绝对根」旁路，评测时与 delta 根融合 | ~0（复用 0.827 G） | ~0 | 零额外视觉；信息受 M5 限制【实验事实：§9.2 完美路由 R²≈0】 |
| E. 事件图节点投影到粗网格再 2D conv（共享同一包事件） | 视网格；目标 **<0.15 G** | 需 profile；避免二次建图 | 工程量大 → **P2** |
| F. 低占空比：仅每 K 步跑绝对根、中间用 S37 | ÷K 摊销 MACs | 摊销后可进预算 | 引入异步融合（H1 相关） |

【实验事实】S37 延迟 7.72 ms 对 0.827 G，**延迟/FLOPs 比远高于 CNN**（图算不规则）；加 CNN 时延迟近似可加，加在图上则不可按 FLOPs 线性外推。

**推荐测量序（零/少训练）**  
1. 已有 fuse_R 上界（16.93）确认收益；  
2. 用 `thop` + `profile_s36_latency` 风格脚本测 B/C 的 batch-1 ms；  
3. 仅当增量 ≤0.5 ms 且 fuse 仍 ≥半上界时，再训轻量根。

**Priority**：测延迟 **P0**；训轻量根 **P1**（X1 若显示 GNN 绝对根旋已够则可能取消）。

---

## 5. 信息价值排序

评分口径（本代理自用）：  
`IV = (#可证伪的核心机制) × (对「下一臂做什么」的决策影响 1–3) / 预估 GPU·小时`。

| 秩 | 实验 | 证伪对象 | 决策影响 | GPU·h | IV 粗分 | 优先级 |
|---|---|---|---|---|---|---|
| 1 | **X2 课程 vs 闭环 prev 单步** | M8 vs M2 | 高（是否改训练分布） | ~0.1–0.5 | **最高** | P0 |
| 2 | **X1 GNN 绝对** | M6 vs M5 | 极高（编码器 vs 表述） | ~2–3.5 | 极高 | P0 |
| 3 | 轻量绝对根延迟探针（§4 B/C） | 「根必须上大 CNN」 | 中高 | ~0.1 | 高 | P0 |
| 4 | H1 oracle 门控上界（主文档 §7，非本代理原创） | M1 | 高 | ~0.2 | 高 | P0（主文档） |
| 5 | X3 残差 unroll | M8 可修复性 | 中 | ~4 | 中 | P1 |
| 6 | 手指父链 / 偏移探针（H2/H3） | M4/M3 | 中高 | ~0.2–1 | 中高 | P0–P1 |
| 7 | 绝对根轻量训练 + 融合 | 工程化 M2 补丁 | 中 | ~2–4 | 中 | P1 |
| 8 | 层级/长时序编码器大改 | M5 | 高但贵 | ≫10 | 低 | P2 |
| 9 | 跨包特征记忆 GRU | P4/H4-memory | 低（主文档已降级） | ≫5 | 低 | P2 |

### 5.1 以最低成本最大程度减少不确定性的**单个**实验

**选 X2（零训练：课程噪声 prev vs 闭环 prev 的单步误差分解）。**

**理由**【推断】  
- 成本最低（分钟级），却直接切割 **M8（分布不一致）vs M2（时间相关系统偏差 / 逐包信念）**——二者对后续是「改 unroll」还是「改测量/编码器/表述」完全分流。  
- 不依赖新权重；可与主文档 §7 的门控/手指探针同脚本打包。  
- 若 X2 否定 M8，可**跳过**昂贵的 X3；若支持 M8，再花 3.5 h 跑 X3 才有信息量。  
- X1 信息量更大但贵一个数量级；应在 X2 之后或并行单种子启动。

（若限制「必须改模型结构的训练臂」，则次优是 **X1 GNN 绝对**——唯一补全 2×2、直接打 M5/M6。）

---

## 6. 实验卡片速查（交付摘要用）

### E-X1 GNN 绝对（P0，~3.5 GPU·h）
- 假设：缺格若接近 CNN 绝对 → 表述有害；若远差 → 编码器天花板。  
- 改动：S37 图 + `PREDICT_DELTA/PREVPOS_EMBED/ROUTED_READOUT=false` + 非线性/`pose_head` 绝对 51。  
- 判据：对 13.56 与 20.74 的相对位置（区间 A/B/C）。  

### E-X2 prev 分布探针（P0，分钟级）
- 假设：训练课程 prev 与闭环 prev 下单步误差系统不同（M8）。  
- 改动：无；条件 prev 来源四档。  
- 判据：工况分层后平台是否重合；是否支持上 X3。  

### E-X3 残差 unroll（P1，~4 GPU·h）
- 假设：残差自滚动缩小闭环差距且不坍缩。  
- 改动：仅 `UNROLL_*` + residual + ramp。  
- 判据：≥1.1 mm 同向；对照 E5.5b 失败模式。  

### E-X4 延迟内绝对根（P0 测 / P1 训）
- 假设：≤0.5 ms 增量的轻量根可兑现 fuse_R 部分收益。  
- 改动：半分辨率/浅 CNN→6D 或前缀微调。  
- 判据：profile ≤ 预算；融合 RA 改善。  

### E-X5 诊断统计协议（P0，零额外训练）
- 假设：选点乐观 2–4 mm；1.1 mm 门在 2 种子下功效不足。  
- 改动：只加报告（中位、交叉选点、bootstrap），不改主表。  
- 判据：过程规范；防止假阳性采纳。  

---

## 7. 最可能被推翻的结论

【推断】**「P4 / 曝光偏差（训练–闭环 prev 分布不一致）是当前第一瓶颈」**（主文档 §5 已倾向降级，但仍常被当作改 unroll 的动机）——更可能被 X2 + 同包迭代证据推翻为：**稳态由逐包系统偏差决定（M2），分布偏移是次要的**；因而残差 unroll（X3）更可能打平而非过门。

次热候选：【推断】**「GNN 跟踪差主要因为表述（M6）」**——更可能被 X1 推翻为编码器/感受野（M5）天花板（若 GNN 绝对 ≫ CNN 绝对）。

---

## 8. 未核实 / 边界

- `track_render51_dr_*` 的 11.9–13.3：**账本有数，主行路径与 zgz 严格对齐未在本任务打开核对** → 2×2 右上角标「待核」。  
- `debug_e55b_unroll_20260825.md` 已删；E5.5b 以账本 + 代码注释为准。  
- 功效计算为正态近似，非仓库内正式统计模块；σ 来自仅 2 点差，本身噪声大。  
- 轻量 CNN 的 ms/MACs 为按 ResNet18 锚点外推，**必须实测**。  
- 本代理未跑 GPU、未改配置；X1 的「非线性头」维数需在实现前用现有 `pose_head` 路径确认无需新算子（`model/model.py:873-874` 已有 `Linear(feat→51)`）。
)
