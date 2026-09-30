# C9：事件相机手部 / 人体姿态与网格（结构对照 S37）

> 子代理 C9 · 2026-09-29 · 仓库根 `/data1/lyq/code/mesh/EventHands1`  
> 遵守 `00_CONTEXT.md` §1 证据标签、§7 纪律（不改代码/配置/测试/outputs、不训练、不用 GPU）、§8 交付。  
> 范围：与 M1–M8 **结构上相似**的事件手/人体姿态与网格；不做通用事件表示（C10）、通用时序 HMR（C11）。  
> 主文档附录与 `docs/research/dir12_20260928/` 已读篇目标「第一轮已读」，本轮只核关键数字或补本任务视角。

---

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29 |
| 检索式（WebSearch） | `EvRGBHand temporal attention ablation 23.5 47.3`；`EventEgoHands 2048 PointNet++`；`Ev2Hands window length milliseconds`；`Chen Efficient Human Pose Estimation 3D Event Point Cloud`；`event camera hand pose mesh CVPR/ICCV/ECCV 2023–2026`；`DHP19 event human pose` |
| 核实来源 | arXiv abs / ar5iv HTML、CVF OA PDF、官方项目页（EventHands、EvHand-FPV、E-3DPSM）、EventEgoHands 补充材料 PDF |
| 筛选规则 | (1) 事件→手或人体 3D 姿态/网格；(2) 结构上触及表示（类帧 vs 点/图）、prev/递推、根/全局、编码器消融或窗口长度——对应 M1–M8；(3) 必须能打开 abs/HTML/DOI/项目页；(4) 不按关键词凑数；通用表示学习、纯 RGB 时序 HMR 排除 |
| 覆盖缺口 | EvHandPose 期刊 PDF 版式与 ar5iv 一致处已用；EventEgoHands++ 主表数字未全部抄录（只读方法立场）；EBHNet（ECCV 2024 模糊图+事件）结构与 S37 递推图不相似，仅入候选目录；EventPointMesh / 部分 2025–2026 新预印本未全文打开 |

---

## 1. 候选目录表

| 短名 | 标题 | 第一 / 末位作者 | venue | 年 | 链接 | 读到 | 入深读？ |
|---|---|---|---|---|---|---|---|
| EventHands | Real-Time Neural 3D Hand Pose Estimation from an Event Stream | Rudnev / Theobalt（MPI 系；末位待以 CVF 为准） | ICCV | 2021 | https://arxiv.org/abs/2012.06475 | 全文 HTML | 是（第一轮已读，本轮重核窗口/KF） |
| EvHandPose | Event-based 3D Hand Pose Estimation with Sparse Supervision | Jiang / Shi 等 | TPAMI | 2024 | https://arxiv.org/abs/2303.02862 · DOI 10.1109/tpami.2024.3380648 | 全文 HTML | 是（第一轮已读） |
| Ev2Hands | 3D Pose Estimation of Two Interacting Hands from a Monocular Event Camera | Millerdurai / Golyanik | 3DV | 2024 | https://arxiv.org/abs/2312.14157 | 全文 HTML | 是（第一轮已读；核窗口） |
| EventEgoHands | Event-based Egocentric 3D Hand Mesh Reconstruction | Hara 等 | ICIP | 2025 | https://arxiv.org/abs/2505.19169 | HTML + 补充 PDF | 是 |
| EventEgoHands++ | … with Real Dataset | Hara 等 | arXiv（IEEE Access DOI 见第一轮） | 2026 | https://arxiv.org/abs/2609.17189 | HTML | 轻读（时序立场） |
| EvRGBHand | Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction | Jiang / Shi | CVPR | 2024 | https://arxiv.org/abs/2403.07346 · CVF | 全文 HTML | 是（核 TA 消融误读） |
| EventHPE | Event-based 3D Human Pose and Shape Estimation | Zou 等 | ICCV | 2021 | https://arxiv.org/abs/2108.06819 | 全文 HTML | 是（第一轮已读） |
| EventCap | Monocular 3D Capture of High-Speed Human Motions using an Event Camera | Xu 等 | CVPR | 2020 | https://arxiv.org/abs/1908.11505 | HTML | 是（优化+灰度锚） |
| EventEgo3D / ++ | EventEgo3D++: 3D Human Motion Capture from a Head-Mounted Event Camera | Millerdurai 等 | CVPR 2024 / IJCV | 2024/2025 | https://arxiv.org/abs/2502.07869 | HTML | 是（LNES+REPM） |
| E-3DPSM | Event-based 3D Pose State Machine（预印本） | Deshmukh / Golyanik | arXiv | 2026 | https://arxiv.org/abs/2604.08543 | 全文 HTML | 是（第一轮已读；M1 反例） |
| Chen-EPP | Efficient Human Pose Estimation via 3D Event Point Cloud | Chen / Wang | 3DV | 2022 | https://arxiv.org/abs/2206.04511 | 全文 HTML | 是（M5 点 vs 帧） |
| Yin-3DRep | Exploring Event-based Human Pose Estimation with 3D Event Representations | Yin / Wang | CVIU（arXiv） | 2023/刊 | https://arxiv.org/abs/2311.04591 | HTML | 是（Chen 扩展；DEV） |
| DHP19 | DHP19: Dynamic Vision Sensor 3D Human Pose Dataset | Calabrese / Delbruck | CVPRW | 2019 | CVF OA | PDF | 是（基准+帧 CNN） |
| EvHand-FPV | Efficient Event-Based 3D Hand Tracking from First-Person View | Xu / Chen | arXiv | 2025 | https://arxiv.org/abs/2509.13883 | HTML | 是（无状态 LNES） |
| EgoEV-HandPose | Egocentric 3D Hand Pose … Stereo Event Cameras | Wang / Wang | arXiv | 2026 | https://arxiv.org/abs/2605.12297 | abs+HTML 前段 | 轻读（立体，非单目） |
| EBHNet | 3D Hand Sequence Recovery from Real Blurry Images and Event Stream | — | ECCV | 2024 | ECVA PDF | 仅扫结构 | 否（模糊图+事件序列恢复，≠S37 递推图） |

---

## 2. 按 M1–M8 组织

### M1 常数增益融合（`Δ = F(事件;a(prev)) + G(prev)`）

- **支持（同形存在）**【论文已有结论】E-3DPSM（arXiv:2604.08543，全文）：直接姿态 \(\mathbf{P}^D\) 与增量 \(\mathbf{P}^\Delta\) 经可微 Kalman 式融合；**推理期 \(Q,R\) 为训练后固定常数**（§4.2）。去掉融合、只做增量：EE3D-R MPJPE **141.22**（表 3 最差）；仅直接姿态平滑误差 **17.22**；完整模型 **84.45** / 平滑 **8.40**（第一轮 G16 已记，本轮 HTML 复核）。【推断】与 S37 的 \(F+G\) 同属「稳态常数权衡两路」，不能当本项目创新。
- **反对（常数增益非充分）**【论文已有结论】同文：用 MLP 让 \(Q,R\) 依赖输入 → MPJPE **91.15**，差于常数 **84.45**（附录 E.5）。【实验事实】仓库：`prev_mlp` 斜率≈课程 Wiener 收缩（`00_CONTEXT` §9.1）；rootinnov / rotw 不救根旋。
- **空白**：事件**手部**递推里几乎无人做「显式新息 \(\times\) 状态依赖 \(K_t\)」；EventHands 的 KF 在**开环网络输出之外**事后平滑，不是端到端 \(F+G\)。

### M2 逐包误差高度时间相关（滤波救不了，需更好逐包）

- **支持**【论文已有结论】EvHandPose（TPAMI 2024，全文）：§III-B **motion ambiguity**——静止部位无事件，同一段事件可对应多个绝对姿态；相对 \(\Delta\theta\)（点名 EventHPE）也解不开。【实验事实】S37 同包迭代收敛≈闭环（§8）；TF 误差与运动量/事件数无关（§9）。
- **反例（时序仍有用，但多为开环记忆）**【论文已有结论】EvHandPose 用 Conv-GRU 跨子段；EventHPE 用 GRU 积 \(\Delta\)；EvRGBHand 用时间注意力——都改善**开环**序列稳定性，不是证明闭环滤波必要。
- **空白**：无论文在「闭环稳态 = 单包吸引子」设定下做受控对比。

### M3 缺少观测减预测 / 几何条件

- **支持（别人显式做了比较）**【论文已有结论】EventCap（CVPR 2020）：灰度锚 + 事件轨迹优化，渲染/轮廓约束。【论文已有结论】EvHandPose Pose-to-IWE：对比度最大化 + hand-edge（训练期几何对齐）。【论文已有结论】EgoEV-HandPose：立体重投影迭代（立体设定，非单目）。
- **反例**：多数学习法（EventHands、Ev2Hands、EventEgoHands、EvHand-FPV）是绝对回归，无「事件相对 prev 投影」新息。
- **空白**：与 S37「丢弃 d / v*、根头不读 \(r_{\mathrm{prev}},K\)」完全同形的消融外部证据少；完美路由使 z 对 prev 误差无信息是**【实验事实】**（`00_CONTEXT` §9.2），文献未复现。

### M4 手指头缺父链 / 证据混合 / 无逐关节门

- **弱支持**【论文已有结论】Ev2Hands：按分割标签的特征注意力分左右手；EventEgoHands：交叉注意力建模双手关系——部件级条件，但不是 MANO 父链局部位姿头。
- **空白**：事件手文献几乎不消融「逐关节门」或「父坐标系条件」；该 M 主要靠 RGB 手网格侧（C 其他子代理）与仓库 MESHQ 设计。

### M5 事件编码器时间感受野与层级（本任务核心）

**(a) 表示与感受野**

| 方法 | 表示 | 窗口 / 有效视野 | 层级 | 读到 |
|---|---|---|---|---|
| EventHands | LNES（整窗覆写时间） | **100 ms** 窗、99 ms 重叠 → 有效步进 **~11 ms**；消融到 33/300 ms | ResNet-18，深 CNN 层级 | 全文 |
| EvHandPose | LNES 边 + 体积累流；Conv-GRU | 子段监督约 **66.7 ms**；弱监督 10–100 ms；快动评测 **5 ms** | CNN + 时序记忆 | 全文 |
| Ev2Hands | 事件点云（同像素合并，时间=均值） | 实现写明 **2 ms** 窗、1 ms 重叠 → 1000 FPS | **PointNet++ 分层** + 分割注意力 | 全文 |
| EventEgoHands | LNES→分割掩膜 + 过滤点云 | 与 30 fps 视频窗对齐（合成 N-HOT3D） | PointNet++ | HTML+补 |
| S37 | 7 维 token + 因果 EdgeConv×3 | 评测 50 ms；邻域只在时间序前 **32** 节点 → global ~**0.78–2.3 ms** 三跳 | **无下采样、无层级** | 【代码事实】`event_gnn.py` |

**(b) prev / 递推**：见下节与 M6。

**(c) 根与全局**：EventHands / Ev2Hands / EvHandPose / EventEgoHands 均**绝对**回归 \(R,t\)（或 PCA 姿态）；EventHPE / E-3DPSM 有相对增量支路。无一篇把根旋隔离成「单层线性、不读 prev 几何」的同形设计。

**(d) 点/图 vs 类帧、窗口长度——M5 外部证据**

- **支持「整窗/深层级有助于姿态」**【论文已有结论】EventHands Table 1：LNES **100 ms** 优于同窗 EOI/ECI；LNES **300 ms** 仍稳健（合成 3D-AUC 约 0.87），短窗 33 ms 时差距缩小——说明**类帧+时间戳归一化**能吃长窗。【实验事实】仓库 CNN-LNES（`s37diag_cnnabs`）递推 RA **13.56** vs S37 **20.74**；S37 窗 50→300 ms **根旋不变**（§9.3）——与「感受野未变」一致。
- **支持「点云+层级可行，但不自动优于深 CNN」**【论文已有结论】Chen et al. 3DV 2022 Table 4：PointNet-2048 MPJPE3D **82.46 mm**、延迟 **12.29 ms**；Point Transformer **73.37 mm**；重实现 Pose-ResNet18/50 的类帧 CNN **更准但更重**。Table 2：点数 1024→2048→… 精度/速度权衡。**【推断】**这是人体 2D→三角化，不是 MANO 根旋；方向上支持「层级点编码器有用」，**不能**直接说 PointNet++ 会修好 S37 根旋。
- **非因果分层点云手**【论文已有结论】Ev2Hands：PointNet++ + 特征注意力，**无 prev**，窗 **2 ms**（全文 Implementation）。与 S37「2048 + 无层级因果图」同预算但结构不同。
- **反例 / 冲突澄清**：见 §重点核实。**空白**：无论文在同一 MANO 递推协议上做「GNN 无层级 vs CNN LNES」的 \(2\times2\)；仓库缺「GNN 绝对」格（M6）。

### M6 表述（delta + 课程噪声 + prev_mlp → 收缩）

- **支持同形**【论文已有结论】EventHPE：\(\hat d_t=\hat d_{t-1}+\Delta\hat d_t\)，\(\Delta\) 来自有记忆 GRU；需已知起始姿态。【论文已有结论】E-3DPSM：直接+增量+常数 \(Q,R\)。
- **反对「必须 delta」**【论文已有结论】EventHands、Ev2Hands、EventEgoHands、EvHand-FPV：**开环绝对**逐窗；EventHands 外挂常速度 KF。WHAM/GVHMR 类视频工作（主文档）也常去掉自回归——反例在时序人体侧。
- **空白**：事件手几乎无「课程噪声注入 prev」与测试闭环误差分布对齐的研究 → M8。

### M7 泛化

- **支持「合成→真实 / 跨相机难」**【论文已有结论】EvHandPose Table II：仅合成训练 → 真实 MPJPE 升 **20–50 mm**。【论文已有结论】EventHands 强调合成训练可泛化但仍用 KF 抑真实抖动。
- **与 S37**【实验事实】TF 训练受试者→zgz 差 3.7× S36——文献无同协议数字。

### M8 训练/测试 prev 误差分布不一致

- **空白为主**：事件手绝对法无 prev；递推法（EventHPE、E-3DPSM）通常用真值滚动或隐藏状态，**无**独立课程噪声课表。【推断】M8 几乎是仓库特有；文献只能提供「绝对开环避开曝光偏差」的旁证（EvHand-FPV、EventEgoHands++ 在抖动后仍倾向不强加时序——第一轮主文档已引）。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A — 整窗类帧感受野（对齐 M5 / M2）

- **数学**：LNES \(I(x,y,p)=(t-t_0)/T\) 覆写；CNN 层级使每像素有效感受野 ≈ 全 \(T\)（EventHands §5.1）。
- **接入**：对照已有 `INPUT_MODE: legacy_lnes` + ResNet（`s37diag_cnnabs` / `s1_abs_domrand`）；或把 GNN 改为「先栅格化再 CNN」只改编码器、保持路由读出。
- **成本**：EventHands 量级 ~ResNet18；仓库 CNN 臂已测 **11.18 M / ~1.65 G**。【实验事实】CNN 绝对已 **13.56**。
- **最小证伪**：冻结 S37 读出，只换「等步长 2048 EdgeConv」→「同包 LNES+CNN 特征再路由」；若根旋 TF 仍 ~6–8° 且与事件数无关 → **反对「只换表示就够」**（支持 M2：信念底噪）；若根旋显著降且随窗长降 → **支持 M5**。

### 机制 B — 分层点集抽象（对齐 M5，对照 Ev2Hands / Chen）

- **数学**：PointNet++ SA：半径/FPS 下采样 + 局部 MLP 聚合；或 Chen RasEPC 切片聚合 \((x,y,t_{\mathrm{avg}},p_{\mathrm{acc}},e_{\mathrm{cnt}})\)。
- **接入**：`semkine/event_gnn.py` 的 embed+EdgeConv×3 换为 2–3 级集合抽象；保持 2048 入、因果约束可先关（与 Ev2Hands 对齐）再开。
- **成本**：Ev2Hands/EventEgoHands 同量级；Chen PointNet-2048 Jetson **12.29 ms**（人体任务，仅数量级参考）。
- **最小证伪**：同一训练课表、只改编码器层级；主行 RA 与根旋。若分层无改善根旋（类似窗长 50→300）→ **反对「缺层级是主因」**；若改善且探针「完美路由 z」R² 上升 → 支持 M5。

### 机制 C — 显式运动场 / 短时记忆进 \(\Delta\)（对齐 M2 竞争假说与 M6）

- **数学**：EvHandPose：FlowNet 形状流 + Conv-GRU；EventHPE：光流条件的 \(\Delta\theta\) GRU。形式 \(\varphi_t = \mathrm{Decode}(h_t),\; h_t=\mathrm{GRU}(h_{t-1},f(E_t))\)，**开环**。
- **接入**：在 `forward_packet` 为根/手指头增加只读事件的短状态（或包内流特征）；**不要**先做成 E-3DPSM 式常数 \(Q,R\)（已占位）。
- **成本**：+一个轻量 GRU/流头（EvHandPose ResNet34 量级，远大于 S37 0.73 M——需砍）。
- **最小证伪**：TF 单步根旋是否随「包内运动量」变化；若仍与运动量无关（仓库 §9）→ 记忆未变成可用新息（M3 仍在）；若相关且闭环改善 ≥1.1 mm → 支持「逐包动态线索」而非常数 \(G\)。

---

## 4. 重点核实（任务 e）

### EvRGBHand「去掉时间注意力 23.5→47.3 mm」——**误读，应降级**

【论文已有结论】Table 4（ar5iv / CVF HTML，全文）：

| 消融 | Strong light MPJPE (mm) | 含义 |
|---|---|---|
| 完整模型 | **22.34** | SA+CF+TA + 全部 Degrader |
| 去掉 **TA** | **23.87** | 仅 −TA，约 **+1.5 mm** |
| 去掉 **MB**（Degrader 一项） | **23.50** | 不是 TA |
| 去掉 OE+MB+BO（全部 Degrader） | **47.34** | 与 TA 无关 |

【推断】第一轮主文档 §P4「去掉后强光 23.5→47.3」把 **Degrader 全关** 误记成 **去 TA**；**不能**再作为「时间注意力必要」的幅度证据。TA 的真实贡献是挑战场景约 **2.5–3 mm** 集体模块收益中的一部分（正文 §5.3）。

### EventEgoHands「2048 点」——**核实成立**

【论文已有结论】补充材料 B.2：Event Cloud \(E\in\mathbb{R}^{N\times 5}\)，**\(N=2048\)**（sigport 补充 PDF + arXiv PDF）。与 S37 `ENCODER_MAX_NODES: 2048` 同预算；差异在**手部分割过滤** vs **等步长抽样**，且 EvEgo 用 PointNet++ 分层、绝对 MANO。

### Ev2Hands 窗口「~11 ms / ~2 ms」——**冲突可消解**

【论文已有结论】Ev2Hands Implementation（ar5iv）：*「temporal resolution … 1000 FPS, … **2 ms** window … **1 ms** overlap」*。  
【论文已有结论】EventHands §5.1：*100 ms 窗、99 ms 重叠 → **effective temporal resolution of 11 ms***。  
【推断】「11 ms」来自 **EventHands** 有效步进，不是 Ev2Hands；主文档 §1「不引 Ev2Hands 窗口长度」的处理正确；本轮可恢复引用 **2 ms**，并注明勿与 EventHands 11 ms 混淆。

---

## 5. 限制与诚实声明

1. 未改仓库任何代码/配置/outputs；未训练；未用 GPU。  
2. 部分 venue 以 arXiv HTML 为准；E-3DPSM、EgoEV、EvHand-FPV、EventEgoHands++ 按**预印本/早期正式版**处理，不写成已录用 CVPR 除非 DOI 页确认（E-3DPSM：第一轮已注明 CVPR 2026 OA 无标题）。  
3. Chen / Yin 是**人体关键点三角化**，指标 mm 不可与手 MANO RA 混比，只借「点编码器 vs 类帧 CNN」结构。  
4. EvRGBHand 为 **RGB+事件**；其 TA 数字不可直接外推纯事件递推。  
5. EventEgoHands \(\theta\in\mathbb{R}^{15}\) 照录（第一轮已疑记法不全）。  
6. 深读 **12** 篇全文/HTML 级 + **2** 篇轻读；宁少勿滥。  
7. **最可能被推翻的结论**：见文末回复 ④——「无层级/短感受野是 S37 相对 CNN 的主因」；Chen 显示点云可接近但未必超过深 CNN，且仓库窗长实验已暗示瓶颈可能在读出/表述（M2/M6）而非窗长本身。

---

## 附录：深读条目卡片（支撑论断用）

### A1 EventHands · ICCV 2021 · https://arxiv.org/abs/2012.06475
Rudnev et al.（第一 Viktor Rudnev；项目页 MPI）。问题：单目事件→MANO。**M5/M6**。LNES+ResNet18 绝对回归；外挂常速度 KF（过程/观测噪声分低高速两档）。Table 1 表示消融。读：**全文**。第一轮已读。

### A2 EvHandPose · TPAMI 2024 · https://arxiv.org/abs/2303.02862
Jiang et al.。**M2/M5/M6**。运动歧义；Conv-GRU；弱监督 IWE。子段 5–100 ms。读：**全文**。第一轮已读。

### A3 Ev2Hands · 3DV 2024 · https://arxiv.org/abs/2312.14157
Millerdurai / Golyanik。**M5**。点云+PointNet++；**2 ms**；无 prev。读：**全文**。第一轮已读。

### A4 EventEgoHands · ICIP 2025 · https://arxiv.org/abs/2505.19169
**M5**。掩膜后 **N=2048** + PointNet++。读：HTML+补充。  

### A5 EvRGBHand · CVPR 2024 · https://arxiv.org/abs/2403.07346
**M6 旁证（误读已纠正）**。Table 4。读：**全文**。

### A6 EventHPE · ICCV 2021 · https://arxiv.org/abs/2108.06819
**M1/M6**。\(\Delta\)+GRU；起始姿态给定；事件帧约 15 ms×4 通道。读：**全文**。第一轮已读。

### A7 Chen-EPP · 3DV 2022 · https://arxiv.org/abs/2206.04511
Chen / Wang。**M5**。RasEPC；PointNet **82.46 mm** vs Point Transformer **73.37 mm** vs Pose-ResNet；2048 点。读：**全文**。

### A8 Yin-3DRep · CVIU / arXiv:2311.04591
Yin / Wang。**M5**。RasEPC+DEV；Chen 扩展。读：**HTML**。

### A9 E-3DPSM · arXiv:2604.08543
Deshmukh / Golyanik。**M1**。常数 \(Q,R\) 融合；表 3/附 E。读：**全文**。第一轮已读。

### A10 EventEgo3D++ · arXiv:2502.07869
**M5/M6**。LNES+REPM 残差传播；绝对热图路径。读：**HTML**。

### A11 EvHand-FPV · arXiv:2509.13883
Xu / Chen。**M6 反例**。LNES-Fast；轻量绝对；强调效率。读：**HTML**。

### A12 DHP19 · CVPRW 2019 · CVF OA
Calabrese / Delbruck。恒定计数事件帧 + CNN + 三角化；约 **8 cm** 级 3D。读：**PDF**。

### 轻读
- EventCap（优化+灰度）；EgoEV-HandPose（立体 BEV）；EventEgoHands++（实例检测+自适应注意力，时序非重点）。
