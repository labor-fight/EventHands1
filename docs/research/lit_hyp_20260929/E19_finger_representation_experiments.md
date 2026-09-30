# E19：手指（M4）与证据表示（M5）——实验设计与反证

> 子代理 E19，2026-09-29。独立于文献组；不改仓库代码/配置/测试/`outputs/`；不训练、不用 GPU。
> 对照臂：`s37_routed`。机制标签：M4（手指孤立 / 缺父链 / 无逐关节门）、M5（时间近视、无层级、一阶池化 → 形状/绝对朝向不可恢复）。
> 脚本落点（若日后实现探针）：`.experiments/lit_hyp_20260929/E19/`。

---

## 0. 可用钩子核对（均已打开）

| 钩子 / 脚本 | 位置 | 用途 |
|---|---|---|
| `route_prev_override` | `model/model.py:926-930, 1478-1479` | 路由几何用 override，其余仍见包的 prev（oracle 路由） |
| `ablate_evidence` | `model/model.py:923-925, 1510-1511` | 证据置零（同 ckpt 机制消融） |
| `ablate_joint_heads` | `model/model.py:704, 1185-1188` | 手指头静默（Δ=0 → 关节不动） |
| `TRAIN.TRAINABLE_PREFIXES` | `model/model.py:951-958`；先例 `configs/semkine/s37_rootinnov_s340{7,8}.yaml` | 冻结主干只训读出 |
| `ENCODER_WINDOW/K/MAX_NODES` | `model/model.py:838-840` → `semkine/event_gnn.py:127-176` | 因果窗、近邻数、节点帽 |
| `event_encoder(..., return_nodes=True)` | `semkine/event_gnn.py:179-239` | 取出 `feat, h, g, px, py, mask` |
| `_route_nodes` → `(a, d, v)` | `model/model.py:1388-1396`；`routed_readout.py:40-90` | **d、v 在正向里丢弃**（`model/model.py:1503-1504` 只保留 `a`） |
| `pool_joint_evidence` | `semkine/routed_readout.py:93-118` | e = mean‖max‖coverage |
| mesh_query 逐关节门 | `model/model.py:1406-1419`；`semkine/mesh_query.py:19,112-114` | 对照：`cov>0` 门控；S37 routed **无**逐关节门 |
| `tools/probe_s37_route.py:load_run` | 加载选中 ckpt | |
| `tools/run_closed_loop_probe.py:150-172` | 更新比、方向投影 `alignment_pose` | |
| `tools/probe_s37_cnnroot.py` | CNN 根替换 / 对照特征源 | |
| `tools/make_s36_row.py` / `select_checkpoint.py` | 主行与选点 | |
| `.experiments/s37_debug_20260928/probe_info.py` | **冻结特征 + ridge/MLP 模板**（§9.2） | |
| `probe_decomp.py` | 根/手指 GT 替换、冻结段首 | |
| `probe_window.py` | 评测窗 50→W ms（**≠** `ENCODER_WINDOW`） | |

**【代码事实】** S37 手指头：`[e_{k+1} 257 ‖ θ_k^prev 3] → 64 → 3`（`model/model.py:868-870, 1190-1191`）；根头单层 `Linear(4624→6)`（`model/model.py:863, 1172-1179`）。
**【实验事实】** 仅手指 RA：CNN ≪ S37；方向投影 ~0.25；窗口加长（评测 W）根旋不变（`00_CONTEXT.md` §4、主文档 §3–§4）。

---

## 1. 实验总览与优先级

| ID | 名称 | 针对 | 训练？ | Priority | 粗算 GPU·h（两种子合计量级） |
|---|---|---|---|---|---|
| **E19-A** | 绝对姿态可解码性探针 | M5 | 零训练（拟合读出） | **P0** | ~0.5–1（特征提取 GPU 分钟级 × 多表示） |
| **E19-B** | 手指分层探针 (i)–(v) | M4 / H2 / H3 | 零训练 | **P0** | ~0.3–0.6 |
| **E19-C** | 感受野单变量重训 | M5 / H3-hier | 全量 2×6000 | **P1** | ~7（2 卡×1.75 h×2 种子；若再加层级臂再 ×2） |
| **E19-D** | 7% 抽样：等步长 vs 体素 | M5 | 全量 2×6000 | **P1** | ~7 |
| **E19-E** | 冻结主干：门 / 偏移 / 父链 | M4 / H3-offset | 冻结 1500 步 | **P0–P1** | ~0.8–1.5（3 臂×2 种子×~25 min） |
| **E19-F** | 2×2 感受野×手指条件 | M4×M5 / PARE | 全量 | **P2** | ~14+（仅当 E19-B/C/E 支持） |

**放弃事件图编码器路线的否决实验**见 §8：主要是 **E19-A**（平凡直方图 ≥ S37 特征，且 CNN 倒数层远优）配合 **E19-C 负结果**。

---

## 2. E19-A：绝对姿态可解码性探针（零训练）— P0

### Hypothesis【待实验验证的假设】
若 M5 成立，则冻结 S37 的 `feat‖e_*` **不能**比平凡事件直方图更好地解码绝对全局朝向与绝对手指角；CNN 倒数层应显著更好。若直方图 ≥ S37，说明瓶颈在编码器丢形状，而非读出头容量。

### Modification
不改模型权重。对每个 50 ms 包提取三组固定特征，再在其上拟合解码器（不给 prev，绝对回归）：

1. **S37 冻结特征**【代码事实】
   - `feat`：全局 mean‖max 经 `proj` 的 512 维（`event_gnn.py:226-233`）
   - `h` 上再可选：mean / max 池化的 128 维原始节点摘要（未过 proj）作对照
   - `e_0..e_15`：257×16（`pool_joint_evidence`）
   - 拼接 `z = [feat 512 ‖ e.flatten 4112]`（与根头输入同形，4624）
   - 指纹：仅用 `e_1..e_15`（不含腕 e_0）或逐关节子集（与 B 衔接）
2. **平凡事件直方图**
   - 极性 2 × 空间 30×40（覆盖 240×180 的 8×4.5 下采样格子）→ 2400 维计数，log1p
   - **时间面下采样**：包内按 t 分 4 片 × 2 极性 × 15×20 → 2400 维（与 (1) 同量级，隔离“有无时间切片”）
3. **逐帧 CNN 倒数第二层**【实验事实】ckpt：`outputs/semkine/s37diag_cnnabs_s340{7,8}/`
   - LNES 输入（`probe_s37_cnnroot.py` / `INPUT_MODE: legacy_lnes`）
   - 取 ResNet18 `avgpool` 后、最终 51 维头之前的 512 维向量（与 EventHands 全模型一致量级）

解码目标（**绝对**，相对相机系）：
- 全局朝向：`pos51[end, 3:6]` 轴角，或旋转矩阵 6D；报告 ° 误差（同 `probe_info.score`）
- 手指：`pos51[end, 6:51]` 全体 45 维；另报仅手指 RA（FK 后腕对齐、根旋用 GT）

解码器：
- **Ridge**：复用 `probe_info.py:ridge`（λ ∈ {1e-2…1e3}，训练内 85% 子切选 λ）
- **小 MLP**：2×128，Adam，wd=1e-4，早停在训练内 10% holdout（`probe_info.py:mlp`）

### Control
- Hold（预测 = 训练集目标均值）
- 仅用 `coverage` 标量（16 维）— 下界
- S37 自身闭环/ TF 更新作参照（非本探针主指标）

### Metric
- 全局朝向：mean °、干净子集 °、R² per-axis
- 手指：MSE(45)、°/关节、仅手指 RA mm（zgz）
- **关键判据**：直方图 ridge/MLP 的朝向 ° **≤** S37-z 的 °（差距 < 0.3° 或更差）→ M5 支持

### Expected observation【推断】
S37-z 朝向 ~8°（§9.2 已有相对/残差设定下的 8.02°；本探针为**绝对**，基线会更差，但三表示相对排序仍可比）；直方图接近或打平 S37；CNN 倒数层显著更好（与 CNN 闭环 8–9° 根旋一致）。

### Positive result means
编码器（抽样+短因果 EdgeConv+一阶池化）未保留可线性/浅 MLP 恢复的形状/朝向信息；M5 成立，应优先改感受野/层级/稠密表示，而非只改手指头。

### Negative result means
S37 特征显著优于直方图且接近 CNN 层 → M5（“特征不可解码”）弱化；瓶颈更可能在 delta/融合/手指头（M1/M4/M6）。

### 特征提取、样本数、正则与泄漏防护
| 项 | 规定 |
|---|---|
| 包定义 | 与评测一致：50 ms，end 落在 `valid_runs_ms` 网格上（`probe_info` / `eval_track`） |
| prev 条件 | **绝对解码不注入 prev**；路由用 **GT(end) 的几何** 还是 **GT(start)** 须各报一组——主报告用 GT(start)（与训练路由分布一致），附录用 GT(end)（上界） |
| 训练拟合集 | `splits` 的 train 9 受试者；每序列均匀抽样 **≤160 包**（同 `probe_info --train-per-seq 160`）→ 约 9×8×160 量级，实际按序列数截断后 ~1e4 |
| 测试 | **仅 zgz** `val_core` 全包（~2590 帧，勿再抽样） |
| 标准化 | 仅用训练集 μ/σ；禁止用测试统计 |
| λ / 早停 | 仅在训练受试者内 holdout；禁止看 zgz 调参 |
| 种子 | 特征扰动/MLP 初始化固定；两种子 ckpt（3407/3408）各提特征、各拟合、再平均 |
| 泄漏 | 禁止用 zgz 事件建直方图词典；禁止把闭环 pred 当特征；CNN 与 S37 必须同包同 end |

### Compute cost / Priority
CPU 可拟合 ridge；特征提取需 GPU 推理数分钟/表示。合计 **≪ 1 GPU·h**。**P0**。

### 现有钩子
`probe_s37_route.load_run`；`event_encoder(..., return_nodes=True)`；`_route_nodes` + `pool_joint_evidence`；模板 `.experiments/s37_debug_20260928/probe_info.py:features/ridge/mlp`；CNN：`probe_s37_cnnroot.py` 加载路径。

---

## 3. E19-B：手指探针 (i)–(v) — P0

### Hypothesis【待实验验证的假设】
手指更新方向错（投影 ~0.25【实验事实】）主要因相机系证据 ↔ 父系 Δθ 缺父链朝向（H2）；若再加全 e_j / 偏移仍无增益，则属主干上下文不足（与 PARE 前提对照，主文档 §3）。

### Modification（零训练拟合）
对关节头索引 `k=0..14`（证据行 `j=k+1`），目标：

\[
\Delta\theta_k^\* = \theta_k^{\mathrm{GT(end)}} - \theta_k^{\mathrm{prev}}
\]

（轴角向量差；附录可报 SO(3) 对数映射差。）prev = GT(start) + 课程小噪声（与 `probe_info.perturb` 同类，或 50% 干净），以匹配训练分布。

输入组（在冻结 `h,a,d,v,e` 上）：

| 组 | 内容 | 维数约 |
|---|---|---|
| (i) | `[e_{k+1}, θ_k]` | 257+3 |
| (ii) | (i) + **父链朝向**（prev FK：`G_parent` 的 6D 旋转）+ **根 Δ**（detach，`ΔR,Δt` 或 6 维） | +6+6 |
| (iii) | (i) + 全部 `e_0..e_15` | +4112 |
| (iv) | (i) + 部件内偏移统计：对路由到 j 的节点，`mean(du,dv), mean(d), mean\|d\|, coverage`（`d` 来自 `_route_nodes`） | +~5 |
| (v) | (ii) + **父骨骼相机系朝向**（父→子骨骼方向的 3D 单位向量或 6D；与 (ii) 的 `G_parent` 互补：显式骨骼轴） | +3~6 |

父链：用 `mano.parents`【代码事实】`model/mano_layer.py:81-83,135-138`。分层：

| 层 | MANO 关节 j（证据行） | 典型头 k |
|---|---|---|
| MCP | 1,4,7,10（四指近端） | 0,3,6,9 |
| PIP | 2,5,8,11 | 1,4,7,10 |
| DIP | 3,6,9,12 | 2,5,8,11 |
| 拇指 | 13,14,15 | 12,13,14 |

每层单独拟合 ridge / 64-MLP（与头同宽），禁止跨层共享权重（避免容量混淆坐标系效应）。

### Control
(i) 为基线；随机朝向噪声替换 (ii) 的父链（应无增益）作安慰剂。

### Metric
- **解释方差** R²（每轴 / 合成 ‖Δθ‖²）
- **方向投影**：`dot(Δθ̂, Δθ*) / ‖Δθ*‖²`（对齐 `run_closed_loop_probe.py:157-161` 的 pose 定义，但按关节）
- 分层：MCP / PIP / DIP / 拇指各自均值
- 附录：用拟合读出替换闭环手指头一步的仅手指 RA（可选，仍零训练）

### Expected observation
H2 真：(ii)(v) 在 **DIP/PIP** 上 R² 与投影升幅 > MCP；拇指视父链定义可能居中。
H3-offset 真：(iv) 有增益且与 (ii) 可加。
PARE 式：(iii) 在短感受野下几乎无用。

### Positive / Negative
- **正**：(ii) 或 (v) 远端显著升、近端弱 → 采纳冻结主干 + 父链条件臂（E19-E）。
- **负**：(ii)(iii)(v) 均不升 → **M4/H2 出局**，转向 M5（E19-A/C）；近端=远端同等小升 → 容量而非坐标系。

### Compute cost / Priority
与 A 同量级特征提取 + CPU/GPU 浅拟合。**P0**（可与 A 同一次特征缓存完成）。

### 钩子
`_route_nodes` 保留 `d,v`；`_fk` / `mano.parents`；`probe_info` 拟合；方向投影参照 `tools/run_closed_loop_probe.py:150-172`。

---

## 4. E19-C：感受野单变量重训 — P1

### Hypothesis
【待实验验证的假设】M5：节点只在前 32 因果窗内取邻（`event_gnn.py:149-176`），dense 包感受野 ~2–3 ms【实验事实】；把 `ENCODER_WINDOW` 32→256（或整包空间 kNN）应同时改善根旋与手指，并降低 TF 单步误差。若否，瓶颈不在感受野。

### Modification（单变量）
- **臂 C1**：仅改 `MODEL.ENCODER_WINDOW: 32 → 256`；`ENCODER_K=8`、`MAX_NODES=2048`、读出/损失/课程均不变。两种子全量训练（6000 步级，与 S37 同配方）。
- **臂 C2**（第二臂，仍单变量精神）：一级**因果集合抽象**（如每 4 个时间序节点平均 → N/4 超点后再 EdgeConv×2），窗保持 32 或略增；**不加**手指头改动。实现成本高于 C1；若 C1 已显著改善可推迟 C2。

备选单变量（与 C1 互斥报告）：整包**空间** kNN（忽略因果窗，仅空间 (x,y)），用于分离“时间近视”vs“缺空间上下文”。

### Control
`s37_routed` 同种子主行；禁止同时改抽样或头结构。

### Metric
主行 8 列（`make_s36_row.py`）；附加诊断（不进用户表）：TF 单步 RA、根旋 °、仅手指 RA、方向投影（`run_closed_loop_probe` / `probe_decomp`）。

### Expected observation
M5 真：C1 根旋与手指双降，TF 单步降，闭环 ≥1.1 mm。M5 假：打平（|Δ|<1.1），与 §9.3「评测窗加长无效」一致——因 §9.3 未改图内窗。

### Positive / Negative
- **正**：感受野是杠杆 → 再开 E19-F / 层级 novelty 清理。
- **负**：**瓶颈不在 ENCODER_WINDOW**；若同时 E19-A 负向（特征仍不可解码朝向）则需质疑整条事件图路线或转稠密/CNN 蒸馏。

### Compute cost / Priority
**【推断】成本**：边候选 `O(N·window)`，256/32=8×；EdgeConv 仍只聚 k=8，MACs 增幅小于 8×但显存（cand/dp）近似线性。训练仍约 **1.0 s/it 量级或略慢** → 2 卡×6000×2 种子 ≈ **7 GPU·h**。**P1**。
显存：若 OOM，先降 `MAX_NODES` 1024 **会破坏单变量**——应记失败并改用梯度检查点，而非默默改节点数。

### 钩子
配置键 `ENCODER_WINDOW`（`model/model.py:840`）；训练后 `select_checkpoint.py` → `make_s36_row.py`；机制 `probe_decomp.py`、`run_closed_loop_probe.py`。

---

## 5. E19-D：dense 包 7% 抽样——等步长 vs 空间均匀 — P1

### Hypothesis
【待实验验证的假设】等步长抽样（`event_gnn._sample`）在空间上偏时间均匀，可能系统丢掉轮廓/指尖；**同预算 2048** 下体素网格均匀抽样应改善形状可解码性与手指。

### Modification
- 保持 `MAX_NODES=2048`、窗 32、头结构不变。
- **D0**：现状等步长（对照）。
- **D1**：空间体素（如 16×12×4 时空格）内均匀/随机取点至 2048；不足则回退全保留。
- 仅改 `_sample` 策略；token 仍先全事件计算再抽【代码事实】`00_CONTEXT.md` §3。

### Control
D0 = S37；禁止同时改 WINDOW。

### Metric
主行；E19-A 子集（抽样后直方图 vs 图特征）可复测朝向可解码性。

### Expected / Positive / Negative
正：global 手指与根旋改善 → 抽样是 M5 的子因。负：打平 → 7% 不是主因（与“时间近视”分离）。

### Compute cost / Priority
全量两种子 ≈ **7 GPU·h**。**P1**（建议在 E19-A 显示“直方图≈S37”之后做，否则优先级降）。

### 钩子
`semkine/event_gnn.py:_sample`；评测链同 C。

---

## 6. E19-E：冻结主干单变量臂 — P0/P1

均用 `TRAIN.TRAINABLE_PREFIXES`【代码事实】`model/model.py:951-958`，初始化 S37 ckpt，~1500 步，两种子；机制门：仅手指 RA、方向投影、闭环 RA≥1.1 mm。

### E19-E1 逐关节门
- **Hypothesis**：无覆盖关节仍更新导致 local 过动（闭环位移 3×【实验事实】）。
- **Modification**：仿 mesh_query，`Δθ_k *= 1[coverage_{k+1}>0]`（或软门 `σ(α·log coverage)`）；只训门参数 + 可选 `joint_heads`。
- **Control**：S37；`ablate_joint_heads` 全关作下界。
- **正/负**：正 → 采纳硬门（并与 `s37_meshq` 对照，建议先 `make_s36_row --run s37_meshq`【主文档 §1】）。负 → 过动主因不是无覆盖更新。

### E19-E2 证据加偏移 (d, du, dv)
- **Hypothesis**：H3-offset——读出端补回相对 prev 表面的比较值改善手指。
- **Modification**：池化前或池化后把 `(du, dv, d)` 拼进节点/部件证据（`d` 已由 `_route_nodes` 算出）；结构仍 state-free 图。只训读出/`joint_heads`/`root_head`。
- **正/负**：正 → H3；负 → 与 §9.2 根上结论一致，偏移对手指也无新信息 → 转 P3b。

### E19-E3 父链朝向条件
- **Hypothesis**：H2；输入数值父链 6D（+ 可选根 Δ detach），证据图不变。
- **Modification**：扩展 `joint_heads` 输入维；`TRAINABLE_PREFIXES: [joint_heads.]`（或含小适配层）。
- **正/负**：见 E19-B 判决；远端>近端为坐标系机制。

### Compute cost
冻结 1500 步 ~25 min/跑 → 3 臂×2 种子 ≈ **2.5 GPU·h** 量级。E1/E2/E3 **先做完 E19-B 再开**（B 负则跳过 E3）。

### 钩子
`TRAINABLE_PREFIXES`；mesh_query 门作参考 `model/model.py:1413-1419`；`d`：`_route_nodes`；评测 `make_s36_row` / `probe_decomp`（手指换 GT 分解）。

---

## 7. E19-F：2×2 交互 — P2

### Hypothesis【待实验验证的假设】
PARE：主干有上下文时隔离读出可行。检验：

| | 手指头 = 现状隔离 | 手指头 = 父链条件或跨关节混合 |
|---|---|---|
| 窗 = 32 | S37（已有） | E19-E3 |
| 窗 = 256（或 C2 层级） | E19-C1 | **F11** 交互格 |

### Modification
在 C1 收敛 ckpt 上热启动，只改手指头输入（父链或轻量 e 上 GCN），冻结或微训主干；或两端都全量（更贵）。

### Metric
交互效应：`ΔRA(F11) ? ΔRA(C1)+ΔRA(E3)`；分层投影。

### Positive
超加性 → PARE 式交互成立，应用「加宽感受野 + 再隔离」或「窄感受野必须条件读出」的设计规则。

### Negative
无交互 → 两问题独立或都不是主因。

### Compute cost / Priority
至少再 2 种子全量或长冻结 ≈ **7–14 GPU·h**。**P2**，仅当 E19-B 正 **且** E19-C 或 E 正。

---

## 8. 按「减少不确定性 / GPU 小时」排序

| 序 | 实验 | 不确定性消除 | GPU·h | 比值 |
|---|---|---|---|---|
| 1 | **E19-A** 绝对可解码性 | M5 核心：特征有无形状/朝向 | ~0.5–1 | 最高 |
| 2 | **E19-B** 手指 (i)–(v) | M4/H2/H3 同时分层证伪 | ~0.3–0.6（可与 A 共享特征） | 极高 |
| 3 | **E19-E3←B** 父链冻结臂 | 把探针变成闭环 mm | ~0.8 | 高 |
| 4 | **E19-E2** 偏移冻结臂 | H3-offset | ~0.8 | 高 |
| 5 | **E19-E1** 逐关节门 | 过动机制 | ~0.8 | 中高 |
| 6 | **E19-C1** WINDOW 256 | M5 感受野 | ~7 | 中（诊断价值高但贵） |
| 7 | **E19-D** 体素抽样 | M5 抽样子因 | ~7 | 中低 |
| 8 | **E19-C2 / F** | 层级与交互 | 14+ | 低（P2） |

### 否决事件图编码器路线的触发器

**【推断】单一最低成本最大杀伤**：**E19-A**。

若同时满足：
1. 平凡直方图（及时间面）在绝对朝向与手指角上 **≥** S37 的 `feat‖e_*`（ridge 与 MLP 一致）；且
2. CNN 倒数层 **显著优于** 二者（与已有闭环 CNN 优势同向）；

则说明当前 EdgeConv 事件图在既定抽样与池化下**没有**提供超出计数图的姿态信息 → 继续堆手指头/门控/偏移的期望收益有限。

**升级为“放弃本路线”**（而非仅改读出）：再加 **E19-C1 负结果**（加宽窗全量训练仍 |ΔRA|<1.1 且 TF 根旋不降）。此时应转向稠密事件表示 / CNN 式主干 / 蒸馏，而不是更深的同构 GNN 读出修补。

若 A 显示 S37 ≫ 直方图，则**不应**因文献 novelty 放弃事件图，而应做 B→E→C。

---

## 9. 与已有实验的边界（避免重复）

| 已有 | 本设计差异 |
|---|---|
| §9.2 `probe_info` | 相对根旋残差 + 扰动 prev；**A 改为绝对、加直方图与 CNN 层、加手指目标** |
| §9.3 `probe_window` | 只加长**评测事件窗**，不改 `ENCODER_WINDOW`；**C 改图内因果窗并重训** |
| rotw / rootinnov | 根融合；本文件不重复，仅手指/表示 |
| meshq | 已有逐关节门但无主行；E1 前先补主行作对照 |

---

## 10. 交付自检

- [x] 每实验含 Hypothesis / Modification / Control / Metric / Expected / Positive / Negative / Cost / Priority / 钩子
- [x] 覆盖任务 1–6
- [x] 成本按仓库实测标度
- [x] 未改仓库代码；未训练；脚本目录预留 `.experiments/lit_hyp_20260929/E19/`
