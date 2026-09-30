# E18：根融合（M1 / M2 / M3）实验设计与反证

> 2026-09-29。研究子代理 E18。仅基于代码与仓库实验；不改仓库代码/配置/测试/`outputs/`；不训练、不启 GPU。
> 脚本落点（若日后实现）：`.experiments/lit_hyp_20260929/E18/`。
> 对照臂：`s37_routed`（20.74；19.23 / 22.26）。机制候选见 `00_CONTEXT.md` §6 的 M1–M3；第一轮 H1 / Step 0 见 `docs/网络结构分析.md` §2、§6–§7。

---

## 0. 竞争机制（中性）

| 标签 | 内容 | 关键已有线索 |
|---|---|---|
| **M1** | `Δroot = F + G` 只能表示**常数**融合增益；改常数或加状态门即可 | 【实验事实】分工况最优增益 0.02 / 0.41 / 0（干净），模型约用 0.3（`S37_ROUTED_READOUT_PREREG.md` §9.1）；K0/K1/K2 落在同一常数增益权衡线 |
| **M2** | 逐包信念误差是**时间相关系统偏差**；增益/滤波上界 ≈ 信念本身 | 【实验事实】同包迭代收敛点 ≈ 闭环（§8）；TF 误差与运动量/事件数无关（§9）；CNN 根替换 −3.8 mm |
| **M3** | 缺观测减预测 / 几何条件（d、v*、r_prev、K） | 【代码事实】根头 `Linear(4624→6)` 不读 prev（`model/model.py:825,1133-1141`）；`dist/vid` 在调用处丢弃（`model/model.py:1503` ↔ `routed_readout.py:40-89`） |

本文件的目标：**用最低成本实验把 M1 与 M2 判开**，再决定门控（H1）还是更好的绝对根测量；M3 作为单变量冻结读出臂单独证伪。

---

## 1. 已核对钩子与脚本清单

### 1.1 `model/model.py`（自己打开核对）

| 钩子 | 位置 | 作用 | 本文件用法 |
|---|---|---|---|
| `ablate_evidence` | `:925, 1510-1511` | 证据置零 | 相关/增益对照；确认 F 依赖事件 |
| `route_prev_override` | `:930, 1478-1479` | 路由几何用另一份 51 维，解码仍用包内 prev | CNN 旋只改路由、不改输出；oracle 路由对照 |
| `ablate_joint_heads` | `:704, 1185-1188` | 手指头输出零（闭环则关节不动） | 根-only RA 诊断（可选） |
| `ablate_innovation` | `:948, 1499-1500` | 新息特征置零（仅 rootinnov 臂） | 对照已失败臂，本设计默认不用 |
| `prev_mlp_root` | `:939, 1520-1523` | `PREV_MLP_ROOT:false` → G 的根 6 维置零 | 单变量「删 G」对照；**不是**门控 |
| `TRAIN.TRAINABLE_PREFIXES` | `:952-958` | 按前缀 `requires_grad` | 冻结主干只训根读出（与 rootinnov 同机制） |
| F / G 分离 | `:1515-1530` | `prevpos_embed` 关 → 仅头；`prev_mlp(prev)` → G；`out=prev+Δ` | αβ 扫描与门控探针在脚本里拆 F/G，**无需改模型** |

【代码事实】闭环融合是 `out = prev + F_heads + G_prev_mlp`（有事件时），根 6 维与手指共用同一加法（`model/model.py:1518-1530`）。

### 1.2 仓库工具

| 脚本 | 关键 API | 本文件用法 |
|---|---|---|
| `tools/run_closed_loop_probe.py` | `_run`, `_DeltaHook`, `--prev-noise` | 噪声工况下的 TF / 闭环对照 |
| `tools/probe_s37_route.py` | `closed_loop`, `teacher_forced`, `root_gain`, `load_run` | 复用加载与闭环骨架；`root_gain` 已测响应增益 |
| `tools/probe_s37_rootinnov.py` | `g1_tf_rotation`, `g2_gain`, `g3_closed` | 任何新读出臂的机制门 G1/G2 |
| `tools/probe_s37_cnnroot.py` | `run_seq` variants `s37/cnn/fuse_R/fuse_RT` | 扩展 route-only / output-only |
| `tools/make_s36_row.py` | `eval_seed`：`default_rng(0)`、种子 3407/3408、zgz `val_core` | 凡报主行必须走此 rng 协议 |
| `tools/select_checkpoint.py` | 递推 RA 选点 | 冻结训读出后的选点 |

### 1.3 `.experiments` 探针（只读模板，脚本副本放 E18/）

| 脚本 | 关键函数 | 本文件用法 |
|---|---|---|
| `s37_debug_20260928/probe_decomp.py` | `teacher_forced`（full/heads/pmlp）、`loop`（oracle_rot / trust0.5） | **F/G 拆分模板**；trust0.5 ≈ 单点 (α=β=0.5) |
| `s37_debug_20260928/probe_rot.py` | `response`, `err_axes` | 10° 纠正与方向余弦 |
| `s37_debug_20260928/probe_info.py` | `features`, `ridge`/`mlp` | 新根输入的信息上界 |
| `s37_debug_20260928/probe_window.py` | `run_seq` | 窗口非本任务焦点 |
| `s37_debug_20260928/probe_train.py` | 冻结特征上的线性/MLP 拟合 | 无偏门控在训练受试者上拟合 |
| `s37_reanalysis_20260928/probe_regime.py` | `probe_seq` fixpt 迭代 | **逐包信念**（收敛点）误差序列 |

---

## 2. 实验设计

约定：主指标为 zgz 两种子递推 RA（`make_s36_row` 协议）；辅助 G1/G2（`probe_s37_rootinnov`）；采纳门 ≥1.1 mm 且逐种子不劣（`00_CONTEXT.md` §2）。成本按仓库实测：零训练 zgz 闭环**数分钟/次（GPU）**；rootinnov 式 1500 步 ≈ **25 min**；全量 6000 步 2×512 ≈ **1 h 45 min**。本轮设计**禁止**全量大模型。

---

### E18-1　M1 vs M2：信念误差自相关 + 滤波可达上界（零训练）

**Hypothesis**【待实验验证的假设】  
若 M2：逐包信念误差 ε_t（根旋转角或 RA）在滞后 1–10 步（50–500 ms）上 ACF 显著（如 lag-1 > 0.5），且在**冻结的信念序列**上最优常数增益 / 一阶低通 / oracle 调参标量 Kalman 相对「直接用信念」的改善 **< 1.1 mm**。  
若 M1：ACF 接近白噪声，或滤波上界 ≥ 1.1 mm。

**Modification**  
不改权重。对选中 S37 ckpt（3407/3408）：

1. 用 `probe_regime.probe_seq` 取每包 **fixpt 收敛态**（建议 8 次迭代）及 **TF 一步输出** 的根旋转 / RA 误差序列 ε_t（相对包终点 GT）。
2. 报 ACF(lag=1…20)、半相关时间、global vs local 分列。
3. **离线**（不动网络）：把「信念」当作测量 z_t，GT 为真值；在整段序列上网格求  
   - 最优常数增益：`x̂_t = (1−κ) x̂_{t−1} + κ z_t`（SO(3) 上用小角或轴角，与仓库轴角加法一致以便对照）；  
   - 最优一阶低通（等价离散 α）；  
   - 标量 Kalman：网格 (q, r) 或在训练受试者上调、zgz 上评（避免双重乐观）。  
   对照线：直接用 z_t；以及 `hold` / `copy-prev`（`probe_regime` 已有）。

**Control**  
同一帧集合；两种子；信念定义同时报 fixpt 与 TF（【推断】二者应接近，§8）。

**Metric**  
ACF；滤波后递推等价误差（开环平滑上的 RA / 根旋 °）；相对「用信念」的 Δmm。

**Expected observation**【推断】  
ACF lag-1 高（>0.4–0.6），滤波 Δ < 1.1 mm → 支持 M2。

**Positive result means**  
M2 占优：后续应做更好的**逐包绝对根**（E18-6），门控/αβ 期望打平。

**Negative result means**  
滤波上界 ≥ 1.1 mm 且 ACF 低 → M1 仍存活，进入 E18-2/E18-3。

**Compute cost**  
约 2 种子 × 数分钟闭环/fixpt + CPU 网格；**< 0.2 GPU·h**。脚本：`.experiments/.../E18/probe_belief_acf_filter.py`（模板：`probe_regime.py:probe_seq`，`probe_decomp.py:teacher_forced`）。

**Priority** **P0**

**钩子**  
无模型钩子；复用 `probe_regime` / `load_run`（`probe_s37_route.py:load_run`）。

---

### E18-2　常数 (α, β) 扫描：只缩放根上的 F 与 G（零训练）

**Hypothesis**【待实验验证的假设】  
问题主要是**常数增益设错**：存在 (α*, β*) 远离 (1,1)，闭环 RA 改善 ≥ 1.1 mm（两种子均值，且逐种子不劣）。

**Modification**  
闭环每步（仿 `probe_decomp.teacher_forced` 拆分）：

- `F = forward_packet` with `prevpos_embed=False`，取根 6 维相对 prev 的增量（或 `full - prev - G`）；  
- `G = prev_mlp(prev)[:6]`（空包则 0，同 `model/model.py:1527-1529`）；  
- 手指保持原 `full` 的 45 维；  
- 根：`prev[:6] + α·F[:6] + β·G`。  

网格例：α, β ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}（约 49 点；可先粗后细）。**只作用于根**。

**Control**  
(1,1) = 原 S37；(0,1) ≈ 仅 G；(1,0) ≈ 仅 F；(0.5,0.5) ≈ 已有 `trust0.5`（`probe_decomp.loop`）但是整 51 维——本实验必须**只改根**以免与手指混淆。

**Metric**  
zgz 两种子 overall RA；G1（TF 根旋 °）；可选 global/local。

**Expected observation**【推断】  
若 §9.1「测试 prev 近真值却用 0.3」为主因，则 α*<1 或 β* 偏离 1，且 Δ≥1.1 mm。

**Positive result means**  
**M1（常数增益错）**：状态相关门控的额外收益应对照「已调好的常数 (α*,β*)」；若门控相对 (α*,β*) 再无增益，则 H1 出局。

**Negative result means**  
最优点贴近 (1,1) 或改善 <1.1 mm → 常数重标度不够；转向 M2 或 M3。

**Compute cost**  
粗网格 ~50 次闭环 ≈ **数小时 GPU**（可单卡串行；可先单种子筛再两种子确认）。**< 2 GPU·h**。

**Priority** **P0**

**钩子**  
`prevpos_embed` 运行时开关（同 `probe_decomp.py:166-168`）；`prev_mlp`；空包门逻辑与 `model/model.py:1527-1529` 一致。不改 `PREV_MLP_ROOT`（那是硬删 G）。

---

### E18-3　无偏门控上界（修正第一轮 Step 0 的乐观偏差）

**Hypothesis**【待实验验证的假设】  
第一轮「每步用 GT 在 {prev, prev+F, prev+G, prev+F+G} 中挑」【主文档 §2 Step 0】高估可达收益；**在训练受试者上拟合、在 zgz 上评估**的门控，改善会明显更小，可能 < 1.1 mm。

**Modification**

1. **乐观对照（复现 Step 0）**：闭环每步用 GT 根旋挑四候选（或 αF+βG 网格中的 argmin），报上界。  
2. **无偏版**：  
   - 特征 ρ（测试时可见）：`|F|_rot`、`|G|_rot`、`coverage` / `route_frac_routed`（`route_stats`）、事件数、可选 **d 的池化统计**（需在探针里从 `_route_nodes` 收回 `dist`，**不改仓库**：E18 脚本内复制调用 `route_front_vertex_lbs`）。  
   - 在**训练受试者**序列上（TF 或短闭环），拟合：多项式 / 逻辑门 / 浅 MLP，`g(ρ)` → 混合权重或离散四选一；损失 = 根旋角或 RA。  
   - **zgz 上固定 g**，不得看 GT 选动作。  
3. 可选：**留一受试者**交叉，防训练受试者过拟合。

**Control**  
常数 (α*,β*) from E18-2；随机门；全选 prev+F+G。

**Metric**  
乐观上界 Δ vs 无偏 Δ（两种子 RA）；G1/G2。

**Expected observation**【推断】  
乐观 ≥1.1 mm 而无偏 <1.1 mm → 不能用 Step 0 给 H1 开绿灯。

**Positive result means**（无偏仍 ≥1.1 mm）  
状态依赖融合有真实空间 → 做 E18-5c（训门）。

**Negative result means**（无偏 <1.1 mm）  
**H1 出局**（与主文档「Step 0 上界 <1.1 → H1 出局」同逻辑，但以无偏为准）。

**Compute cost**  
零训练闭环 + CPU/小 GPU 拟合；**< 0.5 GPU·h**。模板：`probe_train.py`、`probe_decomp.py`。

**Priority** **P0**（与 E18-1/2 并行）

**钩子**  
`route_stats`；可选脚本内路由以取 d；不改模型。

---

### E18-4　CNN 根 vs S37 根：误差相关与融合收益拆解（零训练）

**Hypothesis**【待实验验证的假设】  
(a) S37 与 CNN 根旋误差**弱相关**（互补）或**强相关**（同偏差）；各自时间 ACF 可与 E18-1 对照。  
(b) `fuse_R` 的 −3.8 mm【实验事实】`S37_ROTW_CNNROOT_PREREG.md` §5B】中，一部分来自**路由变好**，一部分来自**输出根变好**。

**Modification**（扩展 `probe_s37_cnnroot.py`，变体写入 E18 副本）

| 变体 | 行为 |
|---|---|
| `s37` / `cnn` / `fuse_R` / `fuse_RT` | 已有【代码事实】`probe_s37_cnnroot.py:41,69-77` |
| `route_R_only` | `route_prev_override` 的根旋 = CNN，**输出仍为 S37 全量**；下一 prev 仍是 S37 输出 |
| `out_R_only` | 路由用 S37 prev；输出根旋换成 CNN（即现 `fuse_R`，但显式命名） |
| `route_R + out_R` | 两者都换（≈ fuse_R 且路由一致） |

另：对齐同一包，报 `corr(ε_S37, ε_CNN)`、各自 ACF；分 global/local。

**Control**  
`route_prev_override=None`；oracle 路由（GT prev）已有更差 +2.5 mm【实验事实】§6——作负对照。

**Metric**  
两种子 RA；根旋 °；误差相关；Δ(route_only) vs Δ(out_only) vs Δ(both)。

**Expected observation**【推断】  
若 Δ(out_only)≫Δ(route_only) → 收益在测量而非路由；与「完美路由无信息」§9.2 一致。

**Positive result means**  
可量化「更好绝对根」的杠杆拆分，指导 E18-6 是否值得带 CNN 路由。

**Negative result means**  
route_only 占大头 → 优先修路由几何（与 M3/路由纯度相关），而非根头融合。

**Compute cost**  
~6 变体 × 2 种子 × 数分钟；**< 1 GPU·h**。

**Priority** **P0**

**钩子**  
`route_prev_override`（`model/model.py:930,1478-1479`）；`probe_s37_cnnroot.run_seq`。

---

### E18-5　冻结主干、只训根读出的最小单变量臂

每次**只改一处**；`TRAIN.TRAINABLE_PREFIXES`；~1500 步热启动协议对齐 rootinnov；过 G1/G2 再考虑加步数。脚本/配置仅写在 E18 实验区或文档中——**本轮不训练**。

#### 与已失败 `s37_rootinnov` 的变量对照【实验事实】`S37_ROOT_INNOVATION_PREREG.md` §1、§5

| 变量 | rootinnov（失败） | E18-5a | E18-5b | E18-5c |
|---|---|---|---|---|
| 新息头 `root_innov_head` | **新增**加到 root | 无 | 无 | 无 |
| `PREV_MLP_ROOT` | **false（删 G）** | **保留 G** | 保留 G | 保留 G |
| 根头输入 | 原 z + 新息加项 | **z ‖ r_prev**（或替换加性 G） | **z ‖ 偏移统计(d,o,…)** | 原 z |
| 融合 | F'+Innov + 无 G | 非线性读 prev（**非** F+G） | F 输入变，融合仍可 F+G | **门控** σ(g)⊙F + σ(h)⊙G |
| 可训前缀 | `root_head., root_innov_head.` | `root_head.`（加宽） | `root_head.` | `gate.*`（+可选 root_head） |
| 结果 | RA 22.94；G1/G2 不过 | 待测 | 待测 | 待测 |

#### E18-5a　非线性根头读 r_prev（非加性）

**Hypothesis**  
加性 G 表达力不足；`root_head([z; r_prev])` 或小 MLP 可学状态相关增益。  

**Modification**  
增广根头输入 6（或 51）维 prev；**关掉**根上加性 `prev_mlp`（`prev_mlp_root=false`）以免双重先验——与 rootinnov「删 G」相同，但**无新息头、无创新特征**。  

**Control**  
同步数只重训原 `root_head`（无 prev 输入）；rootinnov 数字。  

**Metric** G1/G2/G4。  
**Positive** ≥1.1 mm 且 G1↓ → 非加性条件有用。  
**Negative** 重演「删 G 后发散/变差」→ 加性 G 不可丢，转 5c。  
**Cost** ~25 min × 2 种子 ≈ **0.8 GPU·h**。 **Priority P1**（仅当 E18-1 不支持纯 M2，或 E18-2 显示需状态依赖）。  
**钩子** `TRAINABLE_PREFIXES`、`prev_mlp_root`。

#### E18-5b　根头读 d / v* 偏移统计（M3）

**Hypothesis**  
丢弃的 d、顶点偏移携带根旋信息；拼进根头可降 G1。  

**Modification**  
探针已证 2D 残差信息量 ≥ z【实验事实】§9.2；本臂把**池化后的低维统计**（非完整新息头）拼进 `root_head`，**保留 G**，不删 prev_mlp 根。与 rootinnov 差在：无 lever×ψ 新息头、保留 G、特征更简。  

**Control**  
`ablate` 偏移通道；只重训原头。  

**Metric** G1/G2；信息探针复测。  
**Positive** G1≤3° 或 RA≥1.1 → M3 存活。  
**Negative** 打平 → 与 §9.2「干净样本仍注入 3.9–5°」一致，瓶颈不在拼几个标量。  
**Cost** ~0.8 GPU·h。 **Priority P1**（E18-1 偏 M2 时升为优先「测量」臂）。  
**钩子** 脚本内 `route_front_vertex_lbs`（`semkine/routed_readout.py:40`）；`TRAINABLE_PREFIXES`。

#### E18-5c　门控 F 与 G（H1 正式臂）

**Hypothesis**  
`Δroot = σ(g(ρ))⊙F + σ(h(ρ))⊙G`，ρ 仅用已有标量，可同时降 G1、保 G2。  

**Modification**  
极小门网络；**保留**完整 prev_mlp；不增加新息特征。入口条件：**E18-3 无偏上界 ≥1.1 mm**。  

**Control**  
常数 (α*,β*)；门输入置零。  

**Metric** G1/G2/G4；相对 (α*,β*) 的增量。  
**Positive** 相对最佳常数再 ≥1.1 mm → 状态依赖必要。  
**Negative** 只等价于缩小 α → H1 出局（主文档 §2 证伪条款）。  
**Cost** ~0.8 GPU·h。 **Priority P1**（门控于 E18-3）。  

---

### E18-6　若 M2 成立：延迟预算内最便宜的更好逐包绝对根

**前提** E18-1（及 E18-2 打平）支持 M2。

**Hypothesis**【待实验验证的假设】  
在延迟可接受范围内，**复用已有逐帧 CNN 根**（或其蒸馏）比改 S37 融合更便宜地换取 ≥1.1 mm。

**候选（由廉到贵）**

| 方案 | 改动 | 延迟影响【推断/已测】 | 成本 |
|---|---|---|---|
| **6a 已测上界** | `fuse_R` / `fuse_RT`（`probe_s37_cnnroot`） | 每步额外一次 CNN forward；S37 已 7.72 ms，CNN 诊断臂 ~EventHands 量级 1.7 ms 量级【实验事实】主行 Params/Latency 列 | **0**（已测：16.93 / 16.79） |
| **6b 条件 CNN** | 仅当 `|F|` 大或 coverage 低时跑 CNN 换根 | 平均延迟低于 6a | 零训练策略扫描，**< 0.5 GPU·h** |
| **6c 蒸馏根头** | 冻结 GNN+手指；根头回归 CNN 根旋（或 GT）；绝对或残差 | 推理**无**CNN，延迟 ≈ S37 | 1500–3000 步，**~1–2 GPU·h** |
| **6d 轻量绝对根支路** | 小 MLP/浅 CNN 看 LNES 或剪影残差，只出 3 维旋 | 需 profile；目标总延迟不显著超 S37+裕量 | P2，仅 6c 不够时 |

**Control**  
S37；fuse_R；rootinnov。  

**Metric** RA；Latency（`make_s36_row` 同锚）；G1。  

**Positive** 6c 接近 fuse_R 且延迟不增 → 采纳蒸馏。  
**Negative** 蒸馏远差于 fuse_R → 绝对根信息在 CNN 表征里，事件读出不够，需改编码器（交出 M5/M6，非本文件展开）。  

**Priority** **P1**（M2 确认后）；6d 为 **P2**。  

**钩子**  
`probe_s37_cnnroot`；`route_prev_override`（若 6b 与路由联用）；`TRAINABLE_PREFIXES: [root_head.]`。

---

## 3. 按「减少不确定性 / GPU 小时」排序

| 序 | 实验 | 主要消解的不确定性 | 估 GPU·h | 优先级 |
|---|---|---|---|---|
| 1 | **E18-1** 自相关+滤波上界 | M1 vs M2 一次判开 | <0.2 | P0 |
| 2 | **E18-2** (α,β) 扫描 | 「常数增益是否设错」 | <2 | P0 |
| 3 | **E18-3** 无偏门控上界 | H1 / Step 0 是否假阳性 | <0.5 | P0 |
| 4 | **E18-4** CNN 相关与 route/out 拆 | 融合收益机制；指导绝对根 | <1 | P0 |
| 5 | **E18-5c** 训门 | 无偏上界通过后：状态门是否真值 | ~0.8 | P1 |
| 6 | **E18-5b** 偏移进根头 | M3 | ~0.8 | P1 |
| 7 | **E18-5a** 根头读 r_prev | 非加性先验 | ~0.8 | P1 |
| 8 | **E18-6b/c** 条件 CNN / 蒸馏 | M2 下的可部署绝对根 | 0.5–2 | P1 |
| 9 | **E18-6d** 新轻量绝对支路 | 大改动 | 多 | P2 |

并行建议：P0 四条零训练可同周跑完；**禁止**在 E18-1～3 出结果前启动 5a–c 双种子训练。

---

## 4. 与第一轮 H1 / Step 0 的关系

- 第一轮 Step 0 = 本文件 **E18-3 乐观支** + 未拆根的 ρ 扫描；本文件强制 **无偏评估** 与 **根限定 (α,β)**（E18-2）。  
- 主文档证伪「上界 <1.1 → H1 出局」保留，但上界必须是 **E18-3 无偏**；仅乐观上界通过**不能**开训。  
- rootinnov 失败**不能**否定 H1：它删了 G 并加了新息头（§E18-5 表）；H1 是保留 G、门控 F 与 G。

---

## 5. 未核实 / 边界

- CNN 诊断臂精确 Latency 未在本任务重测；延迟论述依赖主行锚与 Params 列【推断】。  
- 轴角空间做 Kalman/低通与 SO(3) 几何滤波的差距未量化；E18-1 应同时报一种简单 SO(3) 插值敏感性。  
- 09-28 已删预注册中是否已扫过 αβ / 门控：**未读**（纪律：仅分配代理可读）；开跑前主代理应查 `85d76a1`。  
- 本代理未运行任何 GPU 作业；数字均引自 `00_CONTEXT.md` / 已有 prereg。
