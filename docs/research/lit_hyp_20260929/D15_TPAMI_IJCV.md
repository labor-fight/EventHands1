# D15：IEEE TPAMI / IJCV 期刊文献（相对 M1–M8）

> 子代理 D15 · 检索日 2026-09-29 · 范围 2019–2026 · 纪律见 `00_CONTEXT.md` §1/§7/§8。  
> 期刊分区（二手汇总，**待官方 fenqubiao.com 复核**）：中科院升级版 2023 常见口径——TPAMI 计算机 1 区 TOP、IJCV 计算机 2 区；二者 JCR 通常均为 Q1。【论文已有结论】分区依据为腾讯云/CSDN 对 2023 中科院表的转载，非本代理登录官方库核验。

---

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 检索式 | `EvHandPose TPAMI`；`HybrIK-X TPAMI`；`PyMAF-X TPAMI`；`EventEgo3D++ IJCV`；`Event-based Vision Survey Gallego TPAMI`；`TORE volumes TPAMI Baldwin`；`Oberweger feedback loop TPAMI`；`EKLT Gehrig IJCV`；`Recovering 3D Human Mesh Survey Tian TPAMI`；`TPAMI event hand OR human pose 2019..2026`；`IJCV event camera tracking`；`TPAMI exposure bias tracking` |
| 来源 | arXiv abs/html、ar5iv HTML、IEEE Xplore/CSDL/EuropePMC 元数据、作者项目页（liuyebin.com/pymaf-x、jeffli.site/HybrIK、eventego3d.mpi-inf.mpg.de、rpg.ifi.uzh.ch）、GitHub README |
| 日期 | 2026-09-29 |
| 筛选规则 | **结构相似优先**：与 M1–M8 中“常数增益融合 / 逐包系统偏差 / 观测−预测几何 / 父链运动学 / 事件感受野与层级 / delta+课程表述 / 泛化 / 训测 prev 分布不一致”之一同构；不按“event/hand/mesh”关键词凑数。会议扩展版须写明期刊相对会议多了什么。主文档附录已列且本任务无新数字则标「第一轮已读」并只补本视角。 |
| 覆盖缺口 | (1) TPAMI/IJCV 上几乎无与 **KalmanNet 式学习增益** 或 **M8 时间相关闭环误差课程** 同构的手部跟踪论文（exposure bias 文献主要在 MOT/NLP 会议）；(2) 事件图网络（AEGNN 类）无 TPAMI/IJCV 扩展版；(3) VIBE/TCMR 等视频网格的期刊扩版未在本检索中核实到；(4) 部分 IEEE PDF 无法全文抓取，数字以 arXiv HTML / 实验室 PDF 为准并标注阅读形态。 |

主文档附录已覆盖、本任务仍深读以补期刊视角：EvHandPose、Oberweger、HybrIK-X、EventEgo3D++、PyMAF（会议版）。本任务新增或升级为期刊深读：Gallego Survey、PyMAF-X、Tian Survey、TORE、EKLT。

---

## 1. 候选目录表

| # | 标题 | 一作 / 末作 | venue·年 | 链接 | 对应 M | 深读？ | 阅读形态 |
|---|---|---|---|---|---|---|---|
| 1 | Event-based Vision: A Survey | Gallego / Scaramuzza | TPAMI 2022 | https://arxiv.org/abs/1904.08405 · DOI 待核（文内致谢 IEEE TPAMI） | M5 | 是 | arXiv HTML 全文 |
| 2 | EvHandPose: Event-based 3D Hand Pose Estimation with Sparse Supervision | Jiang / Shi | TPAMI 2024 | https://arxiv.org/abs/2303.02862 · TPAMI 46(9):6416–6430 | M2,M5,M6,M7 | 是 | arXiv HTML；第一轮已读 |
| 3 | HybrIK-X: Hybrid Analytical-Neural Inverse Kinematics for Whole-Body Mesh Recovery | Li / Lu | TPAMI 2025 | https://arxiv.org/abs/2304.05690 · DOI 10.1109/TPAMI.2025.3528979 | M4 | 是 | arXiv PDF 文本；第一轮已读 |
| 4 | EventEgo3D++: 3D Human Motion Capture from a Head-Mounted Event Camera | Millerdurai / Golyanik | IJCV 2025 | https://arxiv.org/abs/2502.07869 · 项目 https://eventego3d.mpi-inf.mpg.de | M2,M5 | 是 | arXiv HTML；第一轮部分 |
| 5 | PyMAF-X: Towards Well-Aligned Full-Body Model Regression From Monocular Images | Zhang / Liu | TPAMI 2023 | https://arxiv.org/abs/2207.06400 · DOI 10.1109/TPAMI.2023.3271691 | M1,M3,M4 | 是 | arXiv HTML |
| 6 | Recovering 3D Human Mesh from Monocular Images: A Survey | Tian / Wang | TPAMI 2023 | https://arxiv.org/abs/2203.01923 · DOI 10.1109/TPAMI.2023.3298850 | M3,M4,M6（综述） | 是 | arXiv HTML |
| 7 | Time-Ordered Recent Event (TORE) Volumes for Event Cameras | Baldwin / Hirakawa | TPAMI 2023 | https://arxiv.org/abs/2103.06108 · TPAMI 45(2):2519–2532 | M5 | 是 | ar5iv HTML |
| 8 | Generalized Feedback Loop for Joint Hand-Object Pose Estimation | Oberweger / Lepetit | TPAMI 2020 | https://arxiv.org/abs/1903.10883 · TPAMI 42(8):1898–1912 | M1,M3 | 是 | arXiv HTML；第一轮已读 |
| 9 | EKLT: Asynchronous Photometric Feature Tracking using Events and Frames | Gehrig / Scaramuzza | IJCV 2019 | https://arxiv.org/abs/1807.09713 · 实验室 PDF https://rpg.ifi.uzh.ch/docs/IJCV19_Gehrig.pdf | M3 | 是 | 实验室 PDF 全文 |
| 10 | PyMAF（会议版，对照期刊） | Zhang et al. | ICCV 2021 | https://arxiv.org/abs/2103.16507 | M1,M3 | 对照 | 第一轮已读；不重复深读 |

剔除（关键词相关但结构不入 M1–M8）：纯 federated / 无关跟踪的误抓 arXiv；会议-only 的 EventHPE、AEGNN、RVT（主文档已有）；MOT exposure-bias 论文（非 TPAMI/IJCV）。

---

## 2. 按 M1–M8 组织

### M1 常数增益融合（Δ = F(obs;a(prev))+G(prev)）

**支持什么**【论文已有结论】  
- **PyMAF / PyMAF-X**：显式用「当前网格与图像对齐状态」驱动参数修正，而非 F+G 无门加法。PyMAF-X Table X（Human3.6M）：反馈环迭代下 MPJPE 从初值 274.0 → 第 1/2/3 次迭代约 78.2 / 73.2 / **72.1** mm（带 SAA）；说明**条件于当前预测位置的残差更新**逐步收敛。【阅读：arXiv HTML】  
- **Oberweger TPAMI**：updater 吃「输入深度 ‖ 合成深度」，预测 Δpose；NYU 上 DeepPrior++ 初值 12.3 mm → 反馈后 **10.8 mm**（Table I）。显式比较而非常数加性融合。【阅读：arXiv HTML；第一轮已读】

**反对 / 反例**【论文已有结论】  
- Oberweger 比较了「图像差最小化」优化：同一合成器下平均 3D 误差 **32.3 mm**，劣于初值 12.2 mm（§V，Fig.11）——**有残差形式不等于可学常数增益有用**；目标函数与位姿误差可错配。【推断】对 S37：盲目加「渲染−事件」L2 也可能重蹈。

**未覆盖**  
- 期刊中未见与 S37「事件头 F + 无事件 prev_mlp G 加性、无新息门」同构的学习卡尔曼增益（KalmanNet 在 TSP 等，非本刊任务范围）。

### M2 逐包误差是时间相关系统偏差

**支持什么**【论文已有结论】  
- **EvHandPose**：静止/弱运动时事件缺绝对外观 →「motion ambiguity」；用 **Conv-GRU 跨子段携带特征** + 手部光流表示。Table II：去掉 flow 后 normal 场景 3D-MPJPE 19.82→**25.19** mm。说明单窗事件观测不足以消歧，需要跨窗记忆。【阅读：HTML；第一轮已读】  
- **EventEgo3D++**：静止几乎无事件；用 **frame buffer / REPM** 把过去 LNES×置信度加到当前。Table 5 消融：相对仅 EPM 基线，完整 REPM+损失约 **−8% MPJPE / −11% PA-MPJPE**（文内表述）。【阅读：HTML】

**反对什么**【论文已有结论】+【推断】  
- EvHandPose / EventEgo3D++ 仍是**开环或缓冲特征的逐窗回归**，不是闭环递推跟踪；它们证明「单包不够」但**不直接证明**「闭环稳态 ≈ 系统偏差、滤波无用」。与 S37 的 M2 竞争关系：文献更支持「加强逐包/跨窗表示」，而非否定滤波。

**未覆盖**  
- 期刊无对「同包迭代收敛点 ≈ 闭环误差」的受控复现；无 K0/K1/K2 式课程增益线讨论。

### M3 缺少观测−预测 / 几何条件

**支持什么**【论文已有结论】  
- **PyMAF-X**：在预测网格投影处采样特征再回归 Δθ（mesh-aligned evidence）。Table VII：mesh-aligned ≫ global / grid features。【阅读：HTML】  
- **Oberweger**：合成图 vs 输入的显式比较进 updater。【第一轮已读】  
- **EKLT（IJCV）**：亮度增量观测 L 与生成模型预测 L̂（帧梯度·光流）的**光度残差**做最大似然对齐；仿真平均跟踪误差约 **0.4 px**（Table 1）；相对 ICP/EM-ICP 更准且更长轨迹（Table 2–3）。结构同构于「观测减预测」。【阅读：实验室 PDF】  
- **Tian Survey**：系统归纳优化范式用 2D 证据作 data term、回归范式训练期有投影监督但推理常开环——与 PyMAF「推理期仍读对齐」对照。【阅读：HTML】

**反对什么**  
- 无期刊论文证明「丢弃 d/v*/g 仍然最优」；相反反馈类方法把对齐当作必要输入。

**未覆盖**  
- 事件相机 + MANO 路由下「完美路由 z 对 prev 误差无信息」的期刊先例（仓库探针级）。

### M4 手指头缺父链 / 证据混合 / 无逐关节门

**支持什么**【论文已有结论】  
- **HybrIK-X**：twist–swing + 沿运动学树求解；**Naive HybrIK 误差沿树累积（文中 Eq.15 / Fig.4）**，Adaptive / divide-and-conquer（HybrIK-X）缓解远端误差。AGORA 全身：HybrIK-X FB NMJE **115.7** vs PyMAF-X **140.0**（Table 3）。手部 FreiHAND：HybrIK PA-MPJPE **优于多数全身专家**（Table 4，文内相对全身法约 −1.5 mm MPJPE）。【阅读：PDF 文本；第一轮已读】  
- **PyMAF-X**：腕部用肘 twist 补偿的自适应整合；Table XI：learned / copy-paste / adaptive 在对齐与腕合理性上不同——**孤立手专家 + 朴素粘贴会伤腕**。【阅读：HTML】

**反对 / 反例**  
- 主文档已列 HaMeR/PARE 等会议反例；本期期刊未新挖到「无树结构仍 SOTA」的 TPAMI 手部文。

**未覆盖**  
- 事件路由证据上的逐关节硬门（仓库 meshq 设计）无期刊对照。

### M5 事件编码器时间感受野与层级

**支持什么**【论文已有结论】  
- **Gallego Survey**：事件是异步亮度变化，任务依赖表示选择（帧/体素/图/尖峰）；强调时间分辨率与数据驱动采样。【阅读：HTML】  
- **TORE**：按像素极性 FIFO 保留最近 K 个时间戳，`TORE=log(t−FIFO+1)`，**无固定时间窗**、带局部记忆。DHP19 上仅换表示：2D 姿态相对原表示约 **+23%**，整体管线约 **+29%**（文 §VII）；Table VI 报 3D MPJPE（mm）。【阅读：ar5iv】  
- **EvHandPose Table II**：同一网络换 ECI / Voxel / TORE / Time surface / EST / Matrix-LSTM——TORE normal 3D-MPJPE **23.78** vs 全文方法 **19.82**；**Matrix-LSTM 58.36** 远差。表示选择可差数十 mm。【阅读：HTML】  
- **EventEgo3D++**：LNES 窗 **T=33 ms**、序列 **N=20**；相对 S37「包内因果 32 邻域 ≈ 数 ms」是**整窗稠密 + 跨帧缓冲**。【阅读：HTML】

**反对什么**【推断】  
- TORE/LNES/CNN 成功说明「更好同步表示」有效，但**不证伪**「当前 EdgeConv 感受野过短」——它们换了表示族。

**未覆盖**  
- TPAMI/IJCV 无 AEGNN 式异步图层级扩版与 S37 单级池化的直接对照。

### M6 表述（delta + 课程噪声 + prev_mlp）

**支持什么**【论文已有结论】+【推断】  
- EvHandPose / EventEgo3D++ / EventHands 线多为**绝对/开环回归** + 时序特征，而非 delta-tracking + 课程噪声。Tian Survey 区分优化 vs 回归、单帧 vs 时序——**缺少「GNN×绝对」格与期刊证据一致（期刊也几乎不做事件 GNN 跟踪）**。  
- Oberweger 训练 updater 时**显式采样当前误差分布**（非仅高斯），聚焦常见偏差——结构上接近「训测误差分布对齐」，但是**单帧迭代**而非时间相关闭环。【阅读：HTML】

**反对什么**  
- 无期刊实验断言「delta+prev_mlp 必然学成均值收缩」。

**未覆盖**  
- 2×2 {GNN/CNN}×{绝对/跟踪} 的期刊空白；M6 仍主要靠仓库诊断。

### M7 泛化

**支持什么**【论文已有结论】  
- **EvHandPose**：合成/他域训练 → EvRealHands 上 MPJPE 可到 **70+ mm**（Table II domain-gap 块）；实采 EvRealHands 后 normal **19.82** mm。域差可达数十 mm。【阅读：HTML】  
- **EventEgo3D++**：合成 EE3D-S → 野外 EE3D-W 需微调；野外 MPJPE 均值 **166.19** mm vs 工作室 **102.15** mm（Table 2–3）。【阅读：HTML】

**未覆盖**  
- 与 S37「TF 训练受试者 vs zgz 差 3.7× S36」同构的期刊定量；无期刊讨论「路由读出伤害迁移」。

### M8 训练/测试 prev 误差分布不一致

**支持什么**【推断，弱】  
- Oberweger 对 updater 注入「经验误差分布」；反馈环迭代把模型暴露给自身中间状态——**单帧版 exposure mitigation**。  
- 主文档已有 Scheduled Sampling / DAgger（会议/经典）；**本刊检索未找到** TPAMI/IJCV 上手部/人体网格跟踪的 exposure-bias 专文。

**反对 / 空白**【论文已有结论】  
- EvHandPose、EventEgo3D++ **不做闭环 prev←pred 训练**，故**不覆盖 M8**。  
- MOT 的 exposure-bias 工作（如 arXiv 相关）不在本刊任务验收范围。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A — 预测位置条件的对齐残差（→ M1/M3）

- **数学形式**：\(\Delta\theta = R\big(\phi_{\text{mesh}}(\Pi(\hat M(\theta^-))), \phi_{\text{img}}\big)\)，而非 \(\Delta = F(z)+G(\theta^-)\)。  
- **接入点**【代码事实】：`semkine/routed_readout.py` 保留/传入距离 d、顶点 id；`model/model.py` 根头 `Linear(4624→6)` 改为读 `[z ‖ e ‖ r_prev ‖ geom_resid]`（对齐 PyMAF 采样 / Oberweger 比较）。  
- **成本量级**【推断】：额外投影+采样 MLP，相对现 0.827 G 主路径，估计数个百分点–低十几个百分点 MACs（冻结主干只训读出时可忽略主干）。  
- **最小证伪**：冻结 S37 主干，只训读出；若相对常数增益对照 |ΔRA| < 1.1 mm 且 G1/G2 不改善 → 机制在事件路由设定下失败（仓库 rootinnov 已有失败先例，需与「真几何残差」区分）。

### 机制 B — 运动学条件解码 / 防树累积（→ M4）

- **数学形式**：HybrIK Adaptive：目标方向用已重建父关节 \(q_{\mathrm{pa}}\) 而非噪声父关键点，避免式 (15) 累积。手指头：\(\Delta\theta_k = f(e_{k+1}, \theta_k^{\mathrm{prev}}, R_{\mathrm{pa}})\)。  
- **接入点**：`model/model.py` 手指头（约 830–834, 1146–1153）；FK 已在 `pose_repr` / `mano_layer`。  
- **成本**：15× 小 MLP 多拼 3–9 维，可忽略。  
- **最小证伪**：分层探针（主文档 H2）：仅远端关节加父朝向；若仅近端改善、远端不变 → 父链不是瓶颈。

### 机制 C — 跨包事件记忆 / 非固定短感受野（→ M2/M5）

- **数学形式**：EvHandPose Conv-GRU；或 EventEgo3D++ \(\hat L_q = L_q \oplus (C_{q-1}\odot \hat L_{q-1})\)；或 TORE 像素 FIFO。  
- **接入点**：`semkine/event_gnn.py` 包间状态；或评测环缓存上一包节点特征（对照 EvHandPose「特征记忆」）。  
- **成本**：GRU/缓冲相对 EdgeConv×3 的 99.7% MACs 为小头；TORE+CNN 则整换编码器，成本回到 CNN 档。  
- **最小证伪**：仅加包间特征缓冲、不改损失；若闭环 RA 不变且同包迭代收敛点不变 → 支持 M2（系统偏差在逐包估计），反对「缺记忆是主因」。

---

## 4. 限制与诚实声明

- 未改仓库代码/配置/outputs；未训练；未用 GPU。  
- 若干 DOI 以 EuropePMC / researchr / arXiv related-DOI 为准；Gallego 的精确 IEEE DOI 未在本次 abs 页解析到数字串，标**待核**。  
- EvHandPose / Oberweger / HybrIK-X 第一轮已进主文档；本文件侧重**期刊扩版增量**与 M1–M8 映射，非重复刷指标。  
- 分区依据为 2023 中科院升级版二手表，**非官方库截图**。  
- EKLT 日期：ECCV 2018 扩展，IJCV 正式发表常见记为 **2019**（GitHub `Gehrig19ijcv`）；任务书写「2020」与官方 2019 差一年，以 IJCV 卷期为准。  
- 无法从 IEEE Xplore 拉取付费 PDF 时，不以二手博客补数字。

---

## 5. 期刊证据对 M1–M8 的净增量（相对会议文献）

| M | 会议文献已有 | 本刊净增量 |
|---|---|---|
| M1 | PyMAF ICCV、DeepIM、se(3)-TrackNet | **PyMAF-X** 给出迭代表（Table X）与全身腕整合；**Oberweger 期刊版**把手–物反馈与「图像差优化失败 32.3 mm」写全——**反例更硬** |
| M2 | EvHandPose 已在附录 | 期刊版 Table II/III 完整消融与延迟；**EventEgo3D++** 相对 CVPR EventEgo3D：+2D/bone 损失、野外 EE3D-W、SMPL、REPM 消融 Table 5 |
| M3 | 同上 + CLIFF | **EKLT IJCV** 把「生成模型残差」做成经典光度跟踪；**Tian Survey** 把开环回归 vs 优化 data term 系统化 |
| M4 | HybrIK CVPR | **HybrIK-X TPAMI**：全身 divide-and-conquer、Eq.15 累积与 Adaptive 对照、AGORA/FreiHAND 全表 |
| M5 | AEGNN/RVT 会议 | **TORE TPAMI** 表示换骨实验；**Gallego Survey** 领域地图；EvHandPose 表示消融表（期刊完整） |
| M6 | 会议跟踪表述杂 | 期刊事件姿态几乎全是**开环回归**→ 强化「GNN×跟踪」格在期刊也空白 |
| M7 | 会议域适应 | EvHandPose / EventEgo3D++ 期刊给出**大域差毫米数**（70 mm / 野外 166 mm 量级） |
| M8 | DAgger/SS 会议 | **本刊几乎空白**；仅 Oberweger 误差分布采样可弱迁移 |

---

## 附：核心论文卡片（深读 9 篇）

### A. Gallego et al., Event-based Vision: A Survey（TPAMI 2022）
- 作者：Guillermo Gallego … Davide Scaramuzza  
- 链接：https://arxiv.org/abs/1904.08405  
- 问题：事件相机原理、表示、任务全景  
- 对应：M5（表示与时间采样）  
- 关键差异：综述，无手部跟踪系统  
- 为何参考：界定「事件≠帧」与表示税  
- 阅读：arXiv HTML 全文；DOI 待核  

### B. Jiang et al., EvHandPose（TPAMI 2024）【第一轮已读，本任务核 Table II】
- 一作 Jianping Jiang，末作 Boxin Shi  
- https://arxiv.org/abs/2303.02862 · TPAMI 46(9):6416–6430  
- 问题：稀疏标注下事件 3D 手部姿态  
- 对应：M2,M5,M6,M7  
- 关键：开环 + Conv-GRU + 流；非闭环路由  
- 数字：Table II EvHandPose normal 3D-MPJPE **19.82** / PA **10.19**；EventHands **38.51** / **14.18**；w/o flow **25.19**  
- 阅读：arXiv HTML  

### C. Li et al., HybrIK-X（TPAMI 2025）【第一轮已读，补期刊全身表】
- 一作 Jiefeng Li，末作 Cewu Lu  
- https://arxiv.org/abs/2304.05690 · DOI 10.1109/TPAMI.2025.3528979  
- 问题：全身网格混合 IK  
- 对应：M4  
- 期刊相对 HybrIK CVPR：手+脸、backward-updated 合并子树、截断稳健、ICE 相机  
- 数字：Table 3 AGORA FB NMJE HybrIK-X **115.7** vs PyMAF-X **140.0**；Naive 累积见 Fig.4 / Eq.15  
- 阅读：arXiv PDF 文本  

### D. Millerdurai et al., EventEgo3D++（IJCV 2025）
- 一作 Christen Millerdurai，末作 Vladislav Golyanik  
- https://arxiv.org/abs/2502.07869  
- 问题：头戴事件相机自我中心 3D 人体  
- 对应：M2,M5  
- 期刊相对 CVPR EventEgo3D：2D/bone 损失、EE3D-W、SMPL、allocentric RGB  
- 数字：Table 1 EE3D-S MPJPE **98.67**；Table 2 EE3D-R **102.15**；Table 3 EE3D-W **166.19**；140 Hz；T=33 ms  
- 阅读：arXiv HTML  

### E. Zhang et al., PyMAF-X（TPAMI 2023）
- 一作 Hongwen Zhang，末作 Yebin Liu  
- https://arxiv.org/abs/2207.06400 · DOI 10.1109/TPAMI.2023.3271691  
- 问题：全身网格对齐反馈  
- 对应：M1,M3,M4  
- 期刊相对 PyMAF ICCV：手脸专家、自适应腕整合、更全基准  
- 数字：Table X 迭代 MPJPE→**72.1**；ResNet-50 相对 baseline MPJPE −4.7/−5.5 mm（3DPW/H36M，文 §）  
- 阅读：arXiv HTML  

### F. Tian et al., Recovering 3D Human Mesh… Survey（TPAMI 2023）
- 一作 Yating Tian，末作 Limin Wang  
- https://arxiv.org/abs/2203.01923 · DOI 10.1109/TPAMI.2023.3298850  
- 问题：单目人体网格综述  
- 对应：M3,M4,M6（分类框架）  
- 关键：无新算法；用于避免重复发明  
- 阅读：arXiv HTML  

### G. Baldwin et al., TORE Volumes（TPAMI 2023）
- 一作 R. Wes Baldwin，末作 Keigo Hirakawa  
- https://arxiv.org/abs/2103.06108 · TPAMI 45(2):2519–2532  
- 问题：事件表示  
- 对应：M5  
- 数字：DHP19 相对原表示 2D **+23%**、整体 **+29%**（文 §VII）；式 (3)–(5)  
- 阅读：ar5iv HTML  

### H. Oberweger et al., Generalized Feedback Loop…（TPAMI 2020）【第一轮已读】
- 一作 Markus Oberweger，末作 Vincent Lepetit  
- https://arxiv.org/abs/1903.10883 · TPAMI 42(8):1898–1912  
- 问题：深度图手（–物）反馈精修  
- 对应：M1,M3  
- 期刊相对会议版：手–物联合反馈  
- 数字：NYU **10.8 mm**（Table I）；图像差优化 **32.3 mm**  
- 阅读：arXiv HTML  

### I. Gehrig et al., EKLT（IJCV 2019）
- 一作 Daniel Gehrig，末作 Davide Scaramuzza  
- https://arxiv.org/abs/1807.09713 · https://rpg.ifi.uzh.ch/docs/IJCV19_Gehrig.pdf  
- 问题：事件+帧光度特征跟踪  
- 对应：M3  
- 期刊相对 ECCV 2018：更长分析与基线  
- 数字：仿真 ~**0.4 px**（Table 1）；实数八序列优于 ICP/EM-ICP（Table 2–3）  
- 阅读：实验室 PDF 全文  
