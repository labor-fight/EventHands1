# D16：期刊文献之二——IEEE TIP / TMM / TCSVT / TNNLS 与 Pattern Recognition

> 子代理 D16 · 2026-09-29。事实摘要见 `00_CONTEXT.md`。主文档附录已列 Shao（PR）等；本文件按 **M1–M8 结构相似**筛选，宁少勿滥。全部陈述带证据标签。

---

## 0 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29（UTC+8） |
| 检索式（代表） | `event-based human pose` + `TCSVT\|TMM\|TIP\|Pattern Recognition`；`Graph-Based Spatio-Temporal Feature Learning Neuromorphic TIP Bi`；`EV-VGCNN`；`Spiking Spatiotemporal Transformer Zou TCSVT`；`Event Voxel Set Transformer TCSVT`；`SMR Spatial-Guided Model-Based Regression TCSVT`；`Dual-Branch Self-Boosting TIP hand`；`event graph TNNLS`；`hand mesh temporal TIP\|TMM\|Pattern Recognition 2019..2026` |
| 来源 | arXiv abs/html、ar5iv、IEEE Xplore DOI 页、ScienceDirect / CityUHK Scholars / dblp / 作者主页（Deng、Zou/Zuo、Ren） |
| 筛选规则 | (1) venue ∈ {TIP, TMM, TCSVT, TNNLS, Pattern Recognition} 且年 ∈ 2019–2026；(2) 与 M1–M8 **结构同构**（常数增益/系统偏差、观测−预测、运动学解码、事件图感受野/层级、表述、泛化、train–test prev 分布），非关键词凑数；(3) 会议扩展版须写明期刊多给了什么；(4) 读不到全文标「仅摘要」；(5) 主文档已列论文仅在关键数字或本任务视角时重读，标「第一轮已读」 |
| 覆盖缺口 | **IEEE TMM** 上与 M1–M8 同构的「事件姿态 / 事件图」几乎空白（检索到的时序双手多为 **ACM TOMM**，venue 不符）；**TNNLS** 多为事件去噪图网络或通用 GNN 综述，与姿态状态条件弱相关；**M1/M8** 在目标期刊中几乎无「观测−预测新息 + 课程噪声 vs 闭环误差」的直接同构实验；TIP/TMM 上手部网格多是单帧/自监督，少递推状态条件 |

**期刊分区（依据说明，不确定写待核）**

| 期刊 | 常用标签（非正式） | 本文件依据 |
|---|---|---|
| IEEE TIP | 中科院 1 区 / CCF-A | 社区常用口径，**分区待核最新分区表** |
| IEEE TMM | 中科院 1 区 / CCF-B | 同上，待核 |
| IEEE TCSVT | 中科院 1 区 / CCF-B | 同上，待核 |
| IEEE TNNLS | 中科院 1 区 / CCF-B | 同上，待核 |
| Pattern Recognition (Elsevier) | 中科院 1 区 / CCF-B | 同上，待核 |

---

## 1 候选目录表

深读 **11** 篇（结构相关）；另 **4** 篇核实后剔除或降级（venue 不符 / 结构弱）。

| # | 标题 | 第一 / 末位作者 | venue 年 | 链接 | 对应 M | 读深 | 去留 |
|---|---|---|---|---|---|---|---|
| 1 | A Temporal Densely Connected Recurrent Network for Event-based Human Pose Estimation | Zhanpeng Shao / Youfu Li | Pattern Recognition 2024（Vol.147, 110048；在线 2023） | https://arxiv.org/abs/2209.07034 · DOI [10.1016/j.patcog.2023.110048](https://doi.org/10.1016/j.patcog.2023.110048) | M2, M5,（弱）M8 | 全文 HTML | 入；**第一轮已读**，本轮补 M 视角与表号 |
| 2 | Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing | Yin Bi / Yiannis Andreopoulos | TIP 2020（Vol.29, 9084–9098） | https://arxiv.org/abs/1910.03579 · DOI [10.1109/TIP.2020.3023597](https://doi.org/10.1109/TIP.2020.3023597) | M5 | 全文 arXiv/PDF | 入 |
| 3 | Highly Efficient 3D Human Pose Tracking from Events with Spiking Spatiotemporal Transformer | Shihao Zou / Li Cheng | TCSVT 2025（35(10):9708–9722） | https://arxiv.org/abs/2303.09681 · 期刊页待核 DOI；researchr 确认 venue | M2, M5, M6,（弱）M7 | 全文 HTML（v5） | 入；会议/预印扩展 |
| 4 | Event Voxel Set Transformer for Spatiotemporal Representation Learning on Event Streams | Bochen Xie / Youfu Li | TCSVT 2024（34(12):13427–13440） | https://arxiv.org/abs/2303.03856 · DOI [10.1109/TCSVT.2024.3448615](https://doi.org/10.1109/TCSVT.2024.3448615)（CityUHK Scholars） | M5 | 全文 HTML/PDF | 入 |
| 5 | MVF-Net: A Multi-View Fusion Network for Event-Based Object Classification | Yongjian Deng / Youfu Li | TCSVT 2021（32(12):8275–8284） | DOI [10.1109/TCSVT.2021.3073673](https://doi.org/10.1109/TCSVT.2021.3073673) | M5（弱）, M7（弱） | 仅摘要 + 他文引用表 | 入（轻） |
| 6 | Learning From Images: A Distillation Learning Framework for Event Cameras | Yongjian Deng / Youfu Li | TIP 2021（Vol.30, 4919–4931） | DOI [10.1109/TIP.2021.3077136](https://doi.org/10.1109/TIP.2021.3077136) | M7 | 仅摘要 / PubMed | 入（轻） |
| 7 | SMR: Spatial-Guided Model-Based Regression for 3D Hand Pose and Mesh Reconstruction | Haifeng Sun / Jianxin Liao | TCSVT 2024（34(1):299–314；录用 2023） | DOI [10.1109/TCSVT.2023.3285153](https://doi.org/10.1109/TCSVT.2023.3285153) | M3, M4 | 仅摘要（dblp/作者页） | 入 |
| 8 | A Dual-Branch Self-Boosting Framework for Self-Supervised 3D Hand Pose Estimation | Pengfei Ren / Jianxin Liao | TIP 2022（Vol.31, 5052–5066） | DOI [10.1109/TIP.2022.3192708](https://doi.org/10.1109/TIP.2022.3192708) · https://github.com/PengfeiRen96/DSF | M4 | 仅摘要 | 入（轻） |
| 9 | A Voxel Graph CNN for Object Classification with Event Cameras（EV-VGCNN） | Yongjian Deng / Youfu Li | **CVPR 2022**（非目标期刊） | https://arxiv.org/abs/2106.00216 · DOI 10.1109/CVPR52688.2022.00124 | M5 | 全文 PDF | **剔除主表**；候选核实：venue=CVPR，非 TIP/TCSVT |
| 10 | EvGNN: An Event-driven GNN Accelerator… | Yufeng Yang / Charlotte Frenkel | **IEEE TCASAI 2025**（非目标五刊） | https://arxiv.org/abs/2404.19489 · DOI 10.1109/TCASAI.2024.3520905 | M5（硬件） | 全文 HTML | **剔除**；算法层对 S37 弱 |
| 11 | Exploiting Spatial-Temporal Context for Interacting Hand Reconstruction… | （ACM 页） | **ACM TOMM**（非 IEEE TMM） | https://doi.org/10.1145/3639707 | M4,（弱）M8 | 全文 ACM | **剔除 venue**；可作旁证 |
| 12 | Hierarchical neural network for hand pose estimation | Zheng Chen / Xiaohong Ma | **Signal Processing: Image Communication** 2020 | DOI 10.1016/j.image.2020.115909 | M4 | 仅摘要 | **剔除 venue**；旁证父链分层 |
| 13 | TNNLS 事件去噪 GNN-Transformer（IEEE Xplore 9893571） | （页内） | TNNLS 2024（35(3)） | https://ieeexplore.ieee.org/document/9893571 | （弱）M5 | 仅摘要片段 | 入旁证：图邻域相关，非姿态 |
| 14 | DiffHand… | — | Pattern Recognition（他文引用 2025） | ScienceDirect 摘要链 | M4（弱） | 未核全文 | **不入深读**（未核 DOI/作者） |

---

## 2 按结构问题 M1–M8 组织

### M1 常数增益融合

- **文献支持什么**：目标五刊中**几乎没有**「`Δ = F(obs) + G(state)` 且增益不依赖新息」的直接反驳或支持。【论文已有结论】KalmanNet 等在主文档，但属 TSP 等，非本任务期刊。
- **反对 / 反例**：无；Bi/EVSTr/Shao/Zou 均不做显式增益调度。【推断】期刊文献**不能**支撑「必须学自适应增益」。
- **没覆盖什么**：观测−预测残差、协方差门控、课程噪声下 Wiener 收缩（S37 §9.1）在 TIP/TCSVT/PR **无同构实验**。

**净结论：空白（对本刊范围）。**

### M2 逐包误差是时间相关系统偏差

- **支持**：【论文已有结论】Shao et al., Pattern Recognition 2024（全文 HTML）：单窗事件只含运动部件 → 身体部件「incomplete / disappeared」；用 **dense recurrent + 非顺序几何一致性** 跨步堆积信息。Table III：tDenseRNN 在 DHP19 上 MPJPE **5.08** vs RNN **5.36** vs DKD **5.40**；CDEHP AP **80.18** / PCK **79.70**（tDenseRNN）。Fig.5 / 文中：`T=2,4` 显著差于 `T=8,16`，说明**跨窗时间聚合**改善「单窗信息不全」——与「单包信念误差靠滤波抹平」竞争，更支持**拉长有效观测**。【第一轮已读】
- **支持（弱）**：【论文已有结论】Zou et al., TCSVT 2025 Table III：仅事件输入时，PEL-MPJPE（T=8）Ours **58.7** vs ResNet-GRU **60.0**；强调时空融合补早期时刻遮挡部件（Fig.10 注意力跨时刻），属**更好的逐窗/序列估计**，非闭环常数增益滤波。
- **反对**：无期刊直接证明「滤波无用、只需更好单包」。
- **没覆盖**：S37 同包迭代收敛≈闭环（§8）在期刊中**无对应实验设计**。

**净结论：弱支持「逐包/短窗系统偏差需靠更长时空聚合」；不支持「滤波必有用」。**

### M3 缺少观测−预测 / 几何条件

- **支持**：【论文已有结论】SMR, TCSVT 2024（仅摘要）：**spatial-aware / pose-guided** 特征增强后再回归 MANO 参数；强调用空间姿态信息帮助模型参数回归——与 S37「根头不读 `r_prev`、K、深度；路由丢 d/v\*」对照。【代码事实】`routed_readout.py` 丢弃距离与顶点 id；根头 `model/model.py` 单层线性。
- **反对**：无期刊证明「几何条件不必要」。
- **没覆盖**：完美路由下 z 对 prev 误差 R²≈0（§9.2）无期刊复现；render-and-compare / innovation 形式在目标刊**稀缺**。

**净结论：弱支持「空间/姿态条件化有助于 MANO 参数回归」；机制门级证据空白。**

### M4 手指头缺父链 / 证据混合 / 无逐关节门

- **支持**：【论文已有结论】Ren et al., TIP 2022（仅摘要）：双分支解耦 **3D 手模型拟合** 与像素级姿态，**part-aware model-fitting loss** 使分支互促——结构上贴近「按部位约束 / 不全局混证据」。SMR 用姿态引导的模型回归（同上）。
- **旁证（venue 外）**：Hierarchical NN（Image Communication 2020）分六部递进回归 + interference cancellation——**不入主表**。
- **反对**：无。
- **没覆盖**：S37 契约「他关节证据梯度为零」+ 无逐关节门的期刊对照实验缺失。

**净结论：弱支持分层/部位化解码；对「逐关节硬门」空白。**

### M5 事件编码器时间感受野与层级

- **强支持**：【论文已有结论】
  - **Bi et al., TIP 2020**（全文）：事件→图；**残差 GCN + 图池化**变粗；**Graph2Grid + 3D CNN** 跨多图长时依赖。Table 1：RG-CNNs N-Caltech101 Top-1 **0.657** vs G-CNNs **0.630** vs ResNet50 **0.637**；ASL-DVS **0.901**。Table 2：RG-CNNs **0.79 GFLOPs / 19.46 MB** vs ResNet50 **3.87 / 25.61**（N-Caltech101）。建图半径与 `Dmax=32`、非均匀采样——与 S37「等步抽样 + 前 32 因果邻 + 无池化」对照。
  - **EVSTr, TCSVT 2024**（全文）：体素集 + **MNEL 多尺度邻域** + VSAL 全局交互 + **S²TM 段级时序**。Table I：N-Caltech101 **0.797** vs VMV-GCN **0.778** vs EV-VGCNN **0.748**；Table II：Params **0.93 M**、MACs **0.34 G** vs EV-VGCNN **0.84 M / 0.70 G**。
  - **Shao PR**：`T` 过短性能崩（见 M2）——有效感受野要跨多个事件帧。
  - **Zou TCSVT**：Spiking Spatiotemporal Transformer **双向时空**融合 spike 姿态特征（相对单向短跳 EdgeConv）。
- **反例 / 边界**：【论文已有结论】EV-VGCNN（**CVPR**，非本刊）用 MFRL 分距离尺度，但**无层级池化到部件形状**也够分类；说明「层级」对分类非唯一充要——对 **绝对朝向回归** 是否充分仍【待实验验证的假设】。
- **没覆盖**：S37「窗口 50→300 ms 根旋不变」（§9.3）因**图内 RF 未变**——期刊无人手网格+固定跳数因果图的对照。

**净结论：支持「无层级/短时邻域限制时空表征」；对姿态朝向的充分性需自证。**

### M6 表述（delta + 课程噪声 + prev_mlp → 收缩）

- **支持（间接）**：【论文已有结论】Zou TCSVT：端到端从事件回归姿态序列（SNN tracking），**不依赖**「prev + 常数增益 δ」表述；与 EventHPE 等「灰度初值 + 事件」对照（Table III：EventHPE(MPS) G+E PEL **65.1** vs Ours 仅 E **58.7**）。【推断】期刊侧更常见绝对/序列回归，少见 S37 式 delta+课程噪声。
- **反对**：无期刊证明 delta 表述有害。
- **没覆盖**：2×2 {GNN/CNN}×{绝对/跟踪} 缺格；prev_mlp Wiener 收缩。

**净结论：空白偏弱支持「可换表述」；无证伪 delta 的期刊实验。**

### M7 泛化

- **支持**：【论文已有结论】Deng et al., TIP 2021（仅摘要）：图像域蒸馏提升事件模型特征——暗示事件表征难自学纹理/几何。【论文已有结论】Zou Table IV：Real&Syn + DA 相对 Real-only PEL-MPJPE 改善括号内 **2.9**（Ours 58.7→55.8 量级叙述），合成扩域有用。
- **没覆盖**：S37「TF 训练−zgz 差 3.7× S36」无期刊同构。

**净结论：弱支持域/模态迁移手段；受试者闭环泛化空白。**

### M8 训练/测试 prev 误差分布不一致

- **支持（弱、间接）**：【论文已有结论】Shao：训练用固定长度 clip 的稠密时序连接，评测同设定——**不是**「训练独立噪声 prev、测试自回归误差」。ACM TOMM 双手时序约束（venue 外）强调单帧信息不足需时序——仍非 exposure bias。
- **反对 / 空白**：目标五刊**未见** DAgger / scheduled sampling / replacement unrolling 对手部事件跟踪的直接期刊版（主文档 TIP=Transformer Inertial Poser 为 SIGGRAPH Asia，易混名）。

**净结论：空白。**

---

## 3 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A：事件图层级池化 + 多尺度邻域（对齐 M5）

- **数学形式（文献）**：Bi：图卷积后 **pooling 降节点**，再 Graph2Grid；EVSTr：MNEL 对邻域 \(k\) 用位置+语义注意力聚合，再 VSAL。对比 S37：`h ← h + mean_j ReLU(W[h_j−h_i; dp])` ×3，**无下采样**。【代码事实】`semkine/event_gnn.py:EdgeConv` / `EventGNN`。
- **接入点**：`semkine/event_gnn.py` 在 EdgeConv 栈间插入固定比池化（或体素聚类顶点），或 MFRL 式双半径邻域；**不改**读出契约时可先只换编码器。
- **成本量级**：【论文已有结论】EVSTr Table II MACs **0.34 G**（分类）；S37 主干已占 **~0.827 G** 整包——层级若减节点可降 MAC，但池化实现与 B=1 核启动需实测。【推断】参数 +0.1–0.5 M 量级可试。
- **最小证伪**：冻结读出与头，只换「3×EdgeConv 无池化」→「EdgeConv–Pool–EdgeConv–Pool」；zgz 协议两种子。**若**根旋 TF 与闭环均 |Δ|<1.1 mm → 否定「层级 RF 是当前瓶颈」；**若**仅 TF 降、闭环不降 → 支持 M2 胜于 M5。

### 机制 B：跨包稠密时序状态（对齐 M2/M5，非 M1 滤波）

- **数学形式**：Shao：对帧特征稠密跳连 \(H_t = f(H_{t-1},\ldots,H_{t-T}, x_t)\) + 时空注意力；Zou：spike 特征在 \(T\) 步上双向注意力。
- **接入点**：在 `forward_packet` 外维护 **短隐状态**（或上一包全局 `feat`），喂根头：`model/model.py:forward_packet` / `root_head`；或评测时允许 `T>1` 包堆叠仅编码器（训练对齐）。
- **成本**：Shao 取 `T=16`；Zou T=8 时 FLOPs **9.4 G**（全身 SNN，不可直接比）。S37 若只缓存 512-d feat：MAC 增量很小，延迟主要在多次 GNN。【推断】
- **最小证伪**：oracle 多包事件拼成「真长窗」但 **图仍 3 跳** vs 真加隐状态。若长窗拼事件仍不降根旋（呼应 §9.3）而隐状态降 → 支持「需跨包记忆」；两者都不降 → 支持 M2/M3（表征不可辨）而非 RF。

### 机制 C：空间/姿态条件化的模型参数回归（对齐 M3/M4）

- **数学形式**：SMR：先 SAR 得空间姿态信念，再引导 MANO 参数；非 `Linear([mean‖max; e])`。
- **接入点**：根头输入增加 `r_prev`、投影残差或路由距离 d：`routed_readout.py` + `model/model.py` root_head；手指头增加父关节轴角（已有 `θ_k^prev`，可扩父链）。
- **成本**：+数十维线性，可忽略 vs 0.827 G。
- **最小证伪**：仅拼接 `r_prev` 到根头（§9.2 已有 ridge 信号）。**若** TF 根旋不降 → 与「缺几何条件」叙事冲突（或条件形式不对）；**若**降而闭环不降 → 回到 M2/M8。

---

## 4 限制与诚实声明

1. **未改仓库、未训练、未用 GPU**；未生成主行数字。
2. SMR、Ren TIP、Deng TIP/MVF-Net、TNNLS 去噪文：**仅摘要或二次页面**，数字尽量不外推；SMR/Ren **无表内 mm 数字写入本文**。
3. Zou TCSVT 的 **DOI 字符串**未在 IEEE 页成功打开（researchr / 作者 CV / arXiv comment「Accepted by IEEE TCSVT」交叉核实 venue/页码）；Table 数字来自 arXiv html v5，**假定与期刊版一致，待 PDF 终稿核对**。
4. EV-VGCNN **不是** TIP/TCSVT，候选清单已剔除主结论。
5. IEEE TMM 本任务结构缺口大；勿把 ACM TOMM 或 SIGGRAPH「TIP」惯性位姿与 IEEE TMM 混淆。
6. 期刊分区写「待核」；未查当年中科院/CCF 官方表。
7. 分类/动作识别数字（Bi、EVSTr）到 **手部朝向回归** 的外推是【推断】。

---

## 5 期刊证据对 M1–M8 的净增量（相对会议文献）

| 相对会议文献多给了什么 | 说明 |
|---|---|
| **事件图 + 层级/长时模块的完整期刊叙事** | Bi TIP 2020（池化+Graph2Grid）、EVSTr TCSVT 2024（MNEL+S²TM）把「感受野/层级」写成可引用期刊证据，补 AEGNN/CVPR 点云图线之外的 **TIP/TCSVT 锚点** |
| **事件人体姿态的期刊跟踪结果表** | Zou TCSVT 2025 Table III 给出仅事件 SNN 与 EventHPE/ANN 的 **PEL/PA-MPJPE + FLOPs/能量**；主文档会议线外的效率对照 |
| **不完整事件观测的时序稠密连接** | Shao PR 用 Table III + T 消融把「单窗残缺 → 必须跨步堆积」钉死，服务 M2/M5 |
| **手部 MANO 空间条件回归的期刊表述** | SMR TCSVT / Ren TIP 提供 M3/M4 的期刊措辞，但**缺**与 S37 闭环同协议数字 |
| **几乎没多给的** | M1 自适应增益、M8 exposure bias、M6 delta 表述证伪——仍主要依赖会议/主文档（KalmanNet、DAgger、TIP-SA 等） |

**一句话**：本刊子集对 **M5（及弱 M2）增量最大**；对 **M1/M8 近乎零增量**；M3/M4 有抽象支持、无协议级数字。

---

## 附：核心论文卡片（深读）

### A. Shao et al. — Pattern Recognition 2024【第一轮已读】

- **作者**：Zhanpeng Shao … **Youfu Li**（末）；HTML 与 CityUHK 著录另含 Xueping Wang 等。
- **问题**：事件窗内静止部件消失 → 姿态不全。
- **与 S37**：M2/M5；跨步堆积 vs S37 单包 3 跳。
- **关键差异**：帧式 heatmap + CNN-RNN，非事件图；人体 2D/3D 关键，非 MANO 递推手。
- **值得参考**：dense recurrent 消融（短 T 崩）。
- **读到**：全文 HTML（arXiv 2209.07034v2）。
- **数字**：Table III tDenseRNN DHP19 MPJPE **5.08**；CDEHP AP **80.18**；+PoseAug **4.55** / AP **82.22**。

### B. Bi et al. — TIP 2020

- **作者**：Yin Bi … **Yiannis Andreopoulos**。
- **问题**：NVS 稀疏异步 → 需图表示 + 时空学习。
- **与 S37**：M5。
- **关键差异**：分类/动作，非回归姿态；有池化与 Graph2Grid。
- **读到**：全文 arXiv。
- **数字**：Table 1 RG-CNNs N-Caltech101 **0.657**，ASL-DVS **0.901**；Table 2 **0.79 GFLOPs**。

### C. Zou et al. — TCSVT 2025（预印扩展）

- **作者**：Shihao Zou … **Li Cheng**。
- **期刊版相对早期 arXiv**：接受 TCSVT；强化 SynEventHPD、效率（19.1% FLOPs / 3.6% energy）与 spiking attention 消融（Tab. 文中）。
- **与 S37**：M2/M5/M6；序列姿态跟踪、时空融合。
- **读到**：全文 HTML v5。
- **数字**：Table III Ours（E, T=8）PEL-MPJPE **58.7**、PA-MPJPE **44.1**、FLOPs **9.4 G**、Engy **0.0083**；ResNet-GRU（E）PEL **60.0**。

### D. Xie et al. — EVSTr, TCSVT 2024

- **作者**：Bochen Xie … **Youfu Li**。
- **与 S37**：M5 多尺度+段时序。
- **读到**：全文。
- **数字**：Table I N-Caltech101 **0.797**；Table II **0.93 M / 0.34 G MACs**。

### E. SMR — TCSVT 2024（仅摘要）

- **作者**：Haifeng Sun … **Jianxin Liao**。
- **与 S37**：M3/M4。
- **数字**：摘要未在本环境解析出表内 mm；不编造。

### F. Ren Dual-Branch — TIP 2022（仅摘要）

- **作者**：Pengfei Ren … **Jianxin Liao**。
- **与 S37**：M4 part-aware。
- **读到**：仅摘要 + 官方 GitHub 声明 TIP2022。
