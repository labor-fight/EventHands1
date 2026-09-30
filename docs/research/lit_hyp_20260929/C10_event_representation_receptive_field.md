# C10　事件表示学习：感受野、层级、抽样与点/图相对稠密的精度差距

> 研究子代理 C10。日期 **2026-09-29**。只写本文件；未改仓库代码/配置/测试/`outputs/`；未训练；未用 GPU。  
> 事实摘要：`docs/research/lit_hyp_20260929/00_CONTEXT.md`。避免重复 G17（增量图调度）与 G18（集合递推代数）；本文件只答 **(a)–(e)** 的精度/消融。  
> S37 结构锚点【代码事实】：等步长 ≤2048、`window=32` 因果 kNN、EdgeConv×3 无下采样、mean‖max 一阶池化（`semkine/event_gnn.py:_sample/_edges`）。

**提要**

- **(a)** 分类上稠密 EST/直方图 CNN 在多类（N-Caltech101）常高于浅层点/图；二分类（N-Cars）差距可消失。**回归**上 PEPNet（点云+层级）在 6-DoF 位姿上反超帧式 CNN-LSTM。几乎没有与 S37 同协议的「局部因果图 vs LNES+ResNet」对照——仓库 CNN 绝对臂 13.56 vs S37 20.74 是目前最直接的证据【实验事实】，不能从分类文献直接外推毫米数。
- **(b)** 体素化顶点、半径/深度扩大、时间窗加长、层级集合抽象、时间分箱，均有可核消融数字；全局注意力（GET）有小幅增益。单纯「多跳 EdgeConv 无池化」被 AEGNN 批评为感受野不足。
- **(c)** 等步长/均匀下采样是主流；非均匀网格对精度「仅边际」；体素按事件数选顶点远优于同预算随机事件点；压缩比增大精度缓降。
- **(d)** 异步相对同步多宣称输出等价；实测同构稀疏异步相对稠密同步掉约 1–2 个点（分类）。有向因果图相对无向主要赢在 FLOPs；纯事件+有向池化可大幅掉 mAP。
- **(e)** EST 明确：分开极性通道显著优于沿极性求和；光流任务极性敏感性弱于时间。多数图网络默认用极性作节点特征，但少有「去掉极性」的干净消融。

---

## 0. 检索记录

| # | 检索式 / 入口 | 来源 | 日期 | 结果与用法 |
|---|---|---|---|---|
| 1 | `event camera graph neural network vs dense CNN accuracy comparison voxel sampling ablation` | WebSearch | 2026-09-29 | 命中 AEGNN、EV-VGCNN、DAGr、eGSMV(WACV 2026) |
| 2 | `EV-VGCNN event voxel graph CNN sampling polarity ablation` | WebSearch | 同日 | arXiv:2106.00216；CVF PDF；**无极性消融表** |
| 3 | `PEPNet event point cloud sampling FPS voxel grid ablation CVPR 2024` | WebSearch | 同日 | CVF HTML/PDF；层级/Bi-LSTM 消融，非抽样策略表 |
| 4 | `AEGNN voxel pooling ablation N-Caltech` | WebSearch | 同日 | 全文承认聚类消融「仍开放」；Table 1 跨方法对比可用 |
| 5 | `event camera polarity ablation EST HATS Matrix-LSTM GET` | WebSearch | 同日 | EST 极性/投影消融最硬；GET Group Token |
| 6 | 已知 abs/HTML：`2203.17149` `1904.08245` `1908.06648` `1803.07913` `2001.03455` `2003.09148` `2106.00216` `2402.15584`；CVF PEPNet/GET；Nature DAGr；NeurIPS EGSST PDF | ar5iv / CVF / Nature / NeurIPS | 同日 | 深读入口；PDF 二进制用 HTML/CVF 抽取文本 |
| 7 | 目录避免重复 | `docs/网络结构分析.md` 附录；`G17_*`、`G18_*` 目录 | 同日 | 已读 AEGNN/EventNet/EST/Matrix-LSTM/RVT/SSM/GET/EGSST/PEPNet ——本轮只抽 (a)–(e) 数字，标「第一轮已读」 |

**筛选规则**

1. 结构上触碰 M5（感受野/层级/时间上下文）或与 S37 编码器同族（点/图/体素图 vs 稠密张量），或触及抽样、异步等价、极性通道。  
2. **不按关键词凑数**：纯手部姿态（C9）、纯增量调度（G17）、纯集合代数递推（G18）不展开机制。  
3. 数字必须出自原文表格或明确句子；OCR 冲突以 arXiv HTML/PDF 为准并注明。  
4. 读不到全文写「仅摘要」；不把预印本当已发表。

**覆盖缺口**

- 没有一篇在 **连续回归 + 稠密形状/位姿** 上做与 S37 同协议的「局部因果 kNN-GNN vs 稠密帧 CNN」对照。  
- AEGNN **未**发表体素池化尺寸消融（正文写开放问题）。  
- EventMamba、TPAMI 2025「Rethinking Point-Based…」全文本轮未取到，不引用数字。  
- eGSMV（WACV 2026）仅作目录，不深读（非分配核心）。  
- 极性「置零通道」消融在图网络中几乎空白。

---

## 1. 候选目录表

### 1.1 本轮深读 / 关键数字重读（12 篇）

| 标题 | 第一 / 末作者 | venue | 年 | 链接 | 读到 | 对应 M | 与 (a)–(e) |
|---|---|---|---|---|---|---|---|
| AEGNN: Asynchronous Event-based Graph Neural Networks | Schaefer / Scaramuzza | CVPR | 2022 | https://arxiv.org/abs/2203.17149 ；CVF | 全文（第一轮已读；本轮抽表） | M5 | a,b,c,d |
| End-to-End Learning of Representations…（EST） | Gehrig / Scaramuzza | ICCV | 2019 | https://arxiv.org/abs/1904.08245 | 全文 HTML（第一轮已读；本轮抽表） | M5 | a,b,e |
| Graph-Based Object Classification for Neuromorphic Vision Sensing | Bi / Andreopoulos | ICCV | 2019 | https://arxiv.org/abs/1908.06648 | 全文 HTML | M5 | a,b,c |
| A Voxel Graph CNN…（EV-VGCNN） | Deng / Li | CVPR | 2022 | https://arxiv.org/abs/2106.00216 | 全文 HTML | M5 | a,b,c |
| Event-based Asynchronous Sparse Convolutional Networks（AsyNet） | Messikommer / Scaramuzza | ECCV | 2020 | https://arxiv.org/abs/2003.09148 | 全文 HTML（第一轮目录；本轮抽表） | M5 | a,d |
| A Simple… Point-based Network…（PEPNet） | Ren / Cheng | CVPR | 2024 | https://arxiv.org/abs/2403.19412 ；CVF（主代理更正：原写 2312.04080，该编号是一篇物理论文） | 全文 | M5,M6 | a,b,c |
| HATS: Histograms of Averaged Time Surfaces… | Sironi / Benosman | CVPR | 2018 | https://arxiv.org/abs/1803.07913 | 全文 HTML | M5 | a（稠密手工基线） |
| A Differentiable Recurrent Surface…（Matrix-LSTM） | Cannici / Matteucci | ECCV | 2020 | https://arxiv.org/abs/2001.03455 | 全文 HTML（第一轮已读；本轮抽表） | M5 | b,e |
| GET: Group Event Transformer… | Peng / Wu | ICCV | 2023 | CVF + abs 检索 | 全文 CVF（第一轮已读；本轮抽表） | M5 | b,e |
| Low-latency automotive vision…（DAGr） | D. Gehrig / Scaramuzza | Nature | 2024 | https://doi.org/10.1038/s41586-024-07409-w | Nature HTML（第一轮已读；本轮抽有向） | M5 | b,d |
| State Space Models for Event Cameras | Zubić / Scaramuzza | CVPR | 2024 | https://arxiv.org/abs/2402.15584 | 全文 HTML（第一轮已读；本轮抽率/GNN 评语） | M5,M7 | b（时间上下文） |
| EGSST: Event-based Graph Spatiotemporal Sensitive Transformer | Wu / Hu | NeurIPS | 2024 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/da733d44e4be3902d952d6c1ffcb7db6-Abstract.html | 全文 PDF 抽取（第一轮已读；本轮抽消融） | M5 | b |

### 1.2 入目录、不深读或仅转引

| 标题 | 说明 |
|---|---|
| EventNet（CVPR 2019） | G18 已深读；无 (a) 型精度对照表 |
| RVT（CVPR 2023） | 稠密递推检测；与 SSM 同表；非点/图感受野 |
| EventMamba（AAAI 2025） | 本轮未取全文 |
| Bi TIP 2020 扩展 | 与 ICCV 题名不同；未读 |
| Ev2Hands / EventEgoHands | 手部任务 → C9；主文档已引层级 PointNet++ |

---

## 2. 按 M1–M8 组织（本任务主答 M5；其余标空白或弱相关）

### M1 常数增益融合

- **支持 / 反对 / 空白**：**空白**（所选表示论文不讨论 \(F(\mathrm{obs})+G(\mathrm{prev})\)）。  
- 【推断】表示学习文献不检验融合增益。

### M2 逐包系统偏差

- **净结论**：**弱空白**（侧证）。  
- 【论文已有结论】SSM（CVPR 2024）Table 2：RVT 在 Gen1 上 20→200 Hz 平均掉约 26 mAP；S5 掉约 3.3 mAP 量级（摘要/正文「3.31」相对 RVT/GET 的 21+）。说明**时间建模形式**影响跨率稳定性，但不等于「逐包偏差不可滤波」。  
- 未覆盖：手部闭环稳态与表示感受野的因果。

### M3 观测减预测 / 几何条件

- **净结论**：**空白**。  
- 无论文在事件图上保留「相对 prev 剪影」再与稠密 CNN 比。

### M4 手指头父链 / 门

- **净结论**：**空白**（排除手部专文）。

### M5 事件编码器时间感受野与层级　← 本任务主轴

#### 文献支持什么

1. **局部感受野的图相对稠密直方图，在多类识别上可以明显更差**  
   - 【论文已有结论】AEGNN Table 1（arXiv PDF）：N-Caltech101 上 EST（稠密直方图+CNN，同步）**0.817** vs AEGNN 图 **0.668**；N-Cars 上 EST 0.925 vs AEGNN **0.945**（二分类图可打平甚至略好）。读到：全文。  
   - 【论文已有结论】同表 AsyNet 0.745 / HATS 0.642 / NVS-S 0.670。AEGNN 正文写 NVS-S「感受野限于直接邻域」导致检测弱于带池化的 AEGNN（Table 3/检测段：NVS-S 0.346\* vs Ours 0.595 mAP，+7.7% 叙述）。

2. **同架构下稀疏异步 ≈ 稠密同步（差距小）**  
   - 【论文已有结论】AsyNet Table 1：Event Histogram + Standard Conv vs Ours：N-Caltech **0.761 → 0.745**，N-Cars **0.945 → 0.944**；FLOPs 降一个数量级。读到：全文。归因：计算路径异步化，不是换更大感受野。

3. **体素化/层级显著缩小「随机事件点图」的差距**  
   - 【论文已有结论】EV-VGCNN Table 5：同网络，Original events 2048 顶点 **0.565** vs Event voxels 2048 **0.748**（N-Cal）；即使点式加到 8192 仍 **0.619** < 体素 2048。读到：全文。  
   - 【论文已有结论】Bi 补充 Table 5：半径 \(R=1.5→3\)：准确率 **0.551→0.630**；再增大到 4.5/6 不升且 GFLOPs 升。Table 6：窗长 10→30 ms：**0.528→0.630**；50/70 ms 不更好。Table 7：深度 2→4：**0.514→0.630**。读到：全文。

4. **回归任务上，层级点云可反超帧 CNN**  
   - 【论文已有结论】PEPNet 摘要：相对传统帧方法约 **38%** 性能提升、约 **6%** 参数；Table 4（shape translation，random split）：仅层级+Max（Cond1）T+R=**3.04**；层级+Bi-LSTM+Temporal agg（Cond4）**2.25**。读到：全文。任务是相机 6-DoF 重定位，不是 MANO。

5. **时间分箱 / 极性–时间分组扩大有效上下文**  
   - 【论文已有结论】EST 补充 Table 7：时间 bins \(B=2,4,9,16\)，性能随更细时间离散化上升（N-Cars 约在 \(B=9\) 平台）。读到：全文。  
   - 【论文已有结论】GET：Group Token（按时间与极性分组）相对无该模块约 **+2.7%** top-1 / **+1.6** mAP；文中写 Group Token 表示 84.8% vs 其他 token 化 80.4%。读到：全文 CVF。

6. **仓库内现象与 M5 同向**  
   - 【实验事实】`00_CONTEXT` §9.3：窗 50→300 ms 闭环均值几乎不变、根旋转不降——与「邻域仍锁在前 32 节点 / 数 ms」一致【推断：机制尚未被文献在手上证伪】。

#### 文献反对什么（反例 / 限缩）

1. **「图一定逊于稠密 CNN」不成立**  
   - Bi Table 2：RG-CNN N-Caltech **0.657** vs ResNet-50 事件图 **0.637**，且 GFLOPs 更低（Table 3：RG 0.79 vs ResNet-50 3.87 @224）。读到：全文。  
   - PEPNet：点式回归优于帧式 CNN-LSTM（摘要 38%）。  

2. **异步本身不是精度来源**  
   - AEGNN/AsyNet/DAGr：异步规则对齐同步前向；精度差应≈0，收益在 FLOPs/早出。G17 已详述，此处不重复。  
   - 【论文已有结论】DAGr：有向池化可把计算降约 **91%**、mAP 约降 **2** 个百分点（影像+事件设定）；纯事件 directed pooling mAP **18.35**（正文，配置非其 SOTA 对比设定）。读到：Nature HTML。

3. **加长输入窗 ≠ 自动涨点**  
   - Bi：>30 ms 不升；S37 窗实验不升——二者同向【推断】。

#### 没有覆盖什么

- 因果「前 32」邻域 vs 对称半径图的精度差（主文档 P3 已承认空白）。  
- 手部绝对朝向 / RA-mm 上的层级消融公开表。  
- AEGNN 体素尺寸网格消融。

### M6 表述（绝对 / delta / 课程）

- **净结论**：**弱支持「点编码器可做绝对回归」**，不支持/反对 S37 的 delta+prev_mlp。  
- 【论文已有结论】PEPNet 明确放弃置换不变、用层级+Bi-LSTM 回归绝对 6-DoF。与 M6「缺 GNN×绝对格」叙事相容，但是相机位姿不是手。  
- 未覆盖：课程噪声与表述坍缩。

### M7 泛化

- **净结论**：**弱相关**。SSM 率泛化优于 RVT/GET（Table 2）；非受试者迁移。

### M8 训练/测试 prev 分布

- **净结论**：**空白**。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 1　一级因果集合抽象 / 体素池化（对标 AEGNN / Bi / EV-VGCNN）【待实验验证的假设】

- **数学形式（示意）**：在节点特征 \(h^{(L)}\) 上按时空体素 \(\mathrm{bin}(x,y,t)\) 做 max/mean 得粗图 \(\mathcal{G}'\)，再一层 EdgeConv；或 Bi 式坐标网格池化。感受野从「3 跳×32 因果窗」扩到整包空间尺度。  
- **接入点**：`semkine/event_gnn.py`：`EventGNN.forward` 在三层 EdgeConv 之间或之后插入池化；保持因果边则用 DAGr 式 directed voxel pooling。  
- **成本量级**：节点数 \(N\to N/4\)–\(N/8\) 时 EdgeConv MAC 近似平方下降；参数 + 一个小卷积块（≪ 现 394K 读出）。  
- **最小证伪**：冻结读出与头，只改主干；两种子短训或探针——若 zgz TF 根旋转与闭环 RA 相对 S37 改善 \(<1.1\,\mathrm{mm}\) 且根角不降，则「缺层级」不是天花板（对照主文档 D2 证伪句）。

### 机制 2　抽样改为「空间均匀 / 体素代表点」而非纯时间等步长【待实验验证的假设】

- **依据**：【论文已有结论】EV-VGCNN：同预算体素顶点 ≫ 随机事件点；AEGNN：等步长必要，非均匀网格仅边际。S37 `_sample` 是时间等步长【代码事实】`event_gnn.py:127-146`，高事件率时空间上仍可能扎堆。  
- **接入点**：`EventGNN._sample`：先按像素/粗网格分层再等步长，或按 voxel 事件计数取代表（EV-VGCNN 选顶点规则）。  
- **成本**：建网格 O(ΣN)，相对 0.827 G 主干通常可忽略。  
- **最小证伪**：同 checkpoint 仅改抽样（或同配方重训）；若闭环打平且探针「部件形状线性可读性」不变，则抽样不是主因。

### 机制 3　极性–时间分通道进读出前特征（EST/GET 型）【待实验验证的假设】

- **依据**：【论文已有结论】EST：沿极性求和损害分类（正文：「discarding the polarity… decrease… up to 77%」——幅度夸张但方向清楚）；GET Group Token 含极性轴。S37 token 已有 `2p−1`【代码事实】`encoder.py`，但三层因果邻域未必形成整掌极性–时间结构。  
- **接入点**：不改图拓扑；在 `proj` 前对正负极性分别 mean‖max，或仿 EST 把极性维拼进全局特征。  
- **成本**：特征维 ×2 量级，MAC 小。  
- **最小证伪**：训练时置零极性通道 vs 基线；若 RA/根角不变，则「极性未进有效感受野」假设弱。注意训练已有极性增强，与 CNN 配置相同【代码事实】`00_CONTEXT` §3。

**不推荐当作精度机制**：异步增量调度（G17 已否决对本协议延迟/信息的贡献）。

---

## 4. 限制与诚实声明

1. 深读 12 篇；关键数字均核对 arXiv abs/HTML、CVF 或 Nature/NeurIPS PDF 抽取。AEGNN CVF OCR 的 MFLOP/ev 与 arXiv PDF（7.31 / 0.47）不一致时，**以 arXiv PDF Table 1 为准**。  
2. 分类/检测 mAP 与 S37 RA-mm **不可换算**；PEPNet 是刚体相机位姿。  
3. 「第一轮已读」篇未整文重读，只抽 (a)–(e) 相关表；若表与主文档旧笔记冲突，以本轮打开的原文为准。  
4. 未读 EventMamba 全文、未核 EV-VGCNN 补充材料里全部密度曲线、未做 CPU 复现实测。  
5. 最可能被推翻的结论见文末对主代理摘要 ④：把 AEGNN 相对 EST 的 N-Caltech 差距直接归因于「缺层级」——也可能是 SplineConv 容量、训练配方或同步 CNN 骨干，而非池化 alone（AEGNN 自身未做池化消融）。

---

## 附：重点问题 (a)–(e) 一页对照

| 问 | 核实结论 | 关键数字（原文表） |
|---|---|---|
| (a) 点/图 vs 稠密 | 多类分类：稠密 EST 常大幅领先浅层图；二分类/部分图可打平；**回归** PEPNet 点式可赢帧 CNN | AEGNN T1：0.817 vs 0.668（N-Cal）；PEPNet 摘要 38% |
| (b) 何物缩小差距 | 体素顶点、半径/深度、层级+时序、时间 bins、Group Token；有向池化损精度换算力 | EV-VGCNN T5：0.565→0.748；Bi T5–T7；PEPNet T4：3.04→2.25；GET +2.7% |
| (c) 抽样 | 均匀/等步长实用；非均匀边际；体素选点≫随机事件点；压缩 k↑ 精度缓降 | AEGNN 正文；Bi T4：k=1→12：0.636→0.612 |
| (d) 异步 vs 同步 | 宣称等价；AsyNet 同构约 −1.6 pp（N-Cal）；有向为效率 | AsyNet T1；DAGr 有向 −2 pp / 纯事件 18.35 mAP |
| (e) 极性 | EST：分极性 > 求和；光流不敏感；图网默认用 p，少见去除消融 | EST T2/正文；GET Group Token |
