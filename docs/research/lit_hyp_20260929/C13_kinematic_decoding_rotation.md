# C13：运动学 / 层级解码、父链坐标系条件、旋转表示与增量复合

> 子代理 C13。日期 2026-09-29。只读文献与仓库代码；未改代码/配置/测试/`outputs/`，未训练，未用 GPU。  
> 任务边界：不覆盖根与相机条件（归 C14）。对照机制以 `00_CONTEXT.md` §6 的 **M4**（手指头缺父链/证据混合/门）与 **M6**（delta + 轴角加性表述）为主；其余 M 只给净结论。  
> 权威事实摘要：`docs/research/lit_hyp_20260929/00_CONTEXT.md`。主文档对照：`docs/网络结构分析.md` §3（H2）。G14 目录已浏览（MANO/可见性/6D·SVD 种子），本文件不重复其几何探针。

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29 |
| 检索式（英文） | `Hierarchical Kinematic Human Mesh Recovery HKMR ablation 71.08 77.10`；`Hamba GCN ablation PA-MPJPE 6.6 7.3`；`HOPE-Net adjacency 12.91 6.81`；`HybrIK Eq.15 error accumulation Adaptive Naive`；`CAPTRA pose canonicalization RotationNet residual`；`rotation residual SO(3) composition 6D axis-angle ablation`；`SimpleHand MobRecon HandOccNet kinematic decoder` |
| 来源 | arXiv abs / ar5iv HTML / CVF Open Access PDF 文本 / NeurIPS proceedings PDF / ECVA ECCV 页；WebSearch 仅作定位 |
| 筛选规则 | 结构上须与 S37 的某一环相似：（a）逐关节/层级读祖先姿态；（b）父系或 prev 规范化坐标系；（c）旋转连续表示或增量复合；（d）固定树/链式有害或单 token 足够的反例。关键词命中但结构不像（纯遮挡注意力、纯双目、纯效率工程）不进深读 |
| 核实手段 | 每篇至少打开 arXiv abs 或 HTML / CVF PDF；数字只引用原文表号 |
| 覆盖缺口 | 未找到「手部递推跟踪 + 轴角加性 vs SO(3) 右侧复合」的同协议消融；Aksan SPL 等序列先验未深读；MobRecon/HandOccNet/IntagHand 仅作结构对照（非父链条件消融）；NIKI 以摘要+开篇为主 |

## 1. 候选目录表

| # | 标题 | 第一作者 / 末位作者 | venue / 年 | 链接 | 读到 | 对应 M | 是否深读 |
|---|---|---|---|---|---|---|---|
| 1 | Hierarchical Kinematic Human Mesh Recovery (HKMR) | Georgakis / Wu | ECCV 2020 | https://arxiv.org/abs/2003.04232 | 全文（arXiv PDF 文本） | M4 | 是；第一轮已引，本轮核实数字 |
| 2 | Hamba: … Graph-guided Bi-Scanning Mamba | Dong / …（NeurIPS 论文页） | NeurIPS 2024 | https://arxiv.org/abs/2407.09646 | HTML 全文 | M4 | 是；核实去 GCN |
| 3 | HOPE-Net: A Graph-based Model for Hand-Object Pose Estimation | Doosti / … | CVPR 2020 | https://arxiv.org/abs/2004.00060 | CVF PDF 全文 | M4 反例（邻接初始化） | 是；核实 12.91/6.81 |
| 4 | HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution | Li / Lu | CVPR 2021 | https://arxiv.org/abs/2011.14672 | ar5iv 全文 | M4 反例语境（IK 误差累积） | 是；核实 Eq.15 |
| 5 | HybrIK-X: Hybrid Analytical-Neural IK for Whole-Body Mesh Recovery | Li / Lu | arXiv→TPAMI 线；文中称扩展 HybrIK | https://arxiv.org/abs/2304.05690 | ar5iv 全文 | M4 | 是；手部 Naive vs Adaptive 表 |
| 6 | Accurate 3D Hand Pose Estimation for Whole-Body … (Hand4Whole) | Moon / Lee | CVPRW 2022 | https://arxiv.org/abs/2011.11534 | ar5iv 全文 | M4（腕←子 MCP） | 是 |
| 7 | Encoder-decoder with Multi-level Attention … (MAED / KTD) | Wan / … | ICCV 2021 | https://arxiv.org/abs/2109.02303 | ar5iv 全文 | M4 | 是；补第一轮缺失的 KTD 消融 |
| 8 | PARE: Part Attention Regressor for 3D Human Body Estimation | Kocabas / … | ICCV 2021 | https://arxiv.org/abs/2104.08527 | ar5iv 全文 | M4 条件反例 | 是（第一轮已读视角） |
| 9 | On the Continuity of Rotation Representations | Zhou / Li | CVPR 2019 | https://arxiv.org/abs/1812.07035 | ar5iv 全文 | M6 | 是；G14 已列，本轮补经验段 |
| 10 | An Analysis of SVD for Deep Rotation Estimation | Levinson / … | NeurIPS 2020 | https://arxiv.org/abs/2006.14616 | ar5iv 全文 | M6 | 是 |
| 11 | CAPTRA: CAtegory-level Pose Tracking … | Weng / Guibas | ICCV 2021 | https://arxiv.org/abs/2104.03437 | ar5iv 全文 | M4+M6（prev 规范化 + SO(3) 增量） | 是 |
| 12 | HaMeR: Reconstructing Hands … | Pavlakos / … | CVPR 2024 | https://arxiv.org/abs/2312.05251 | HTML | M4 反例（单路径回归） | 中等深读 |
| 13 | A Simple Baseline for Efficient Hand Mesh Reconstruction (SimpleHand) | Zhou / … | CVPR 2024 | https://arxiv.org/abs/2403.01813 | HTML/abs | M4 反例（简化 token 解码） | 中等深读 |
| 14 | NIKI: Neural Inverse Kinematics with Invertible NNs | Li / Lu | CVPR 2023 | https://arxiv.org/abs/2305.08590 | 摘要+开篇 | M4（IK 线） | 浅 |
| 15 | IntagHand | Li / Liu | CVPR 2022 | https://arxiv.org/abs/2203.09364 | 摘要+开篇 | M4（跨手注意力，非父链） | 浅；结构不像则不夸大 |
| — | MobRecon / HandOccNet | Chen / Park 等 | CVPR 2022 | CVF 页 | 仅摘要/元数据 | — | 筛掉：遮挡/效率为主，无父链条件消融 |
| — | DeepIM | Li / Fox | ECCV 2018 | https://arxiv.org/abs/1804.00175 | abs | M6 先例 | 浅；增量位姿复合先例 |

**筛掉理由（宁少勿滥）。** MobRecon（螺旋图卷积+效率）、HandOccNet（遮挡特征注入）与 IntagHand（双手交叉注意力）不提供「逐关节读祖先姿态」或「父骨骼坐标系」的消融，故不计入核心深读。DeepIM 仅作增量复合先例索引。

## 2. 按结构问题 M1–M8 组织

### 核实第一轮关键数字（先行）

| 引用 | 核实结果 | 原文出处 | 读到 | 对主文档的修正 |
|---|---|---|---|---|
| HKMR 71.08→77.10 | **数字正确**【论文已有结论】 | Table 2，H36M Protocol #1 MPJPE：Full model **71.08**，No joint hierarchy **77.10** | 全文 | 消融语义须写清：No joint hierarchy =「每个关节只依赖**紧邻父关节**，不是全部祖先」；不是「去掉整棵树」。Forward only 75.99、Discriminator 74.21 |
| Hamba 去 GCN 6.6→7.3 | **数字正确**【论文已有结论】 | FreiHAND；文中写 PA-MPJPE 6.6→7.3、PA-MPVPE 6.3→7.2（w/o GCN；NeurIPS PDF / HTML Table 4 或 5，版本间表号 4/5 互换） | HTML 全文 | 这是 **GSS 块内去掉 MANO 邻接 GCN**，不是「去掉层级父链回归器」。与 S37「头 k 不读祖先证据」只是弱同构 |
| HOPE-Net 12.91 vs 6.81 | **数字正确，语义被夸大**【论文已有结论】 | Table 3：自适应图卷积的**邻接矩阵初始化**：Skeleton **12.91** mm，Identity **6.81** mm（2D→3D 平均误差） | CVF PDF | 两边最终都学邻接；不是「固定运动学树编码 vs 学习邻接」的最终结构对照。正确表述：从骨骼先验初始化劣于无信息 Identity 初始化 |
| HybrIK Eq.15 误差累积 | **公式与叙述属实，适用对象需收窄**【论文已有结论】 | §3.2 Eq.15：`p_k−q_k = Σ_{i∈A(k)} ε_i`（Naive IK）；Adaptive 使误差只依赖当前关节 | ar5iv 全文 | 说的是 **3D 关节→相对旋转的解析 IK 重建**，不是神经网络逐关节轴角头的链式解码。不宜直接写成「朴素链式解码有害」 |

### M1 常数增益融合

- **文献支持 / 反对 / 空白：** **空白**（本任务未检索常数增益融合文献）。
- 旋转侧仅有弱相关：S37 用轴角欧氏相加 `out = Δ + prev`【代码事实】`model/model.py:1525-1530`，与 Lie 右侧复合不同【代码事实】`semkine/lie.py:1-9,34-37`；这属 **M6**，不是 M1 的增益调度。

### M2 逐包系统偏差

- **空白**（运动学解码文献不覆盖时间相关信念误差）。

### M3 缺少观测−预测 / 几何条件

- **弱相关空白。** CAPTRA 用 prev 位姿把点云变到规范系再回归小增量【论文已有结论】（arXiv:2104.03437 §3），是「观测相对 prev」的先例，但是物体跟踪、深度点云，不是事件证据相对父骨骼。根头与 K/深度条件归 C14。

### M4 手指头缺父链坐标系 / 证据混合 / 无逐关节门

**【代码事实】** S37 手指头 k 输入仅为 `[e_{k+1} ‖ θ_k^prev]`，不读祖先证据或父骨骼朝向：`model/model.py:860-871`（构造）、`_decode_active` 中 `evidence[:, k+1]` 与 `prev[:, 6+3k:9+3k]`（约 1184–1191）；契约钉死对其它证据梯度为零（`tests/test_s37_routed_readout.py`，见 `00_CONTEXT` §3）。

#### 文献支持什么

1. **祖先姿态条件解码（人体）— HKMR**【论文已有结论】  
   - 问题：单帧 SMPL 回归未显式用运动学层级。  
   - 与 S37：每条链的回归器读共享特征 + **已更新的父链姿态**；S37 手指头不读父链更新。对应 **M4**。  
   - 关键差异：人体单帧、外循环迭代；非事件、非递推。  
   - 数字：H36M P1 MPJPE Full **71.08** vs No joint hierarchy **77.10**（Table 2）；文称层级建模约 **6 mm** 增益。  
   - 读参考：直接消融「只看紧邻父」vs「看全部祖先」。

2. **祖先姿态条件解码（人体视频）— MAED/KTD**【论文已有结论】  
   - 问题：迭代反馈回归同时出所有关节，忽略关节依赖。  
   - 机制：`ω_k = W_k · Concat(x, ω_ancestors)`，用 **6D** 表示。  
   - 数字：3DPW 上 CNN+STE+**KTD** PA-MPJPE **45.7** / MPJPE **79.1** vs CNN+STE+**Iterative** **47.5 / 80.2**；打乱树序 `KTD_random` **47.7 / 82.5**，反序 `KTD_reverse` **47.6 / 79.7**（Table 2）。  
   - 关键：第一轮写「消融数值未找到」→ **本轮已找到 Table 2**。  
   - 关键差异：视频人体、共享图像特征 x 很强；S37 事件证据短感受野。

3. **手部拓扑混合 — Hamba**【论文已有结论】  
   - 问题：单目手网格，需关节拓扑。  
   - 机制：GSS 内 GCN（MANO 邻接）+ Mamba；去掉 GCN → PA-MPJPE **6.6→7.3**（FreiHAND）。  
   - 对应 M4 的「跨关节混合」支路，**不是**父链坐标系条件。  
   - 关键：单帧图像 token，非递推 delta。

4. **腕旋转以子关节（MCP）为条件 — Hand4Whole**【论文已有结论】  
   - 问题：全身网格里腕/手指差。  
   - 机制：腕用 body + **8 个 MCP** 关节特征；手指头**去掉** body 粗特征。  
   - 数字：EHF 上手 MPVPE（无旋转对齐，故敏感于腕）：Body **50.4** → Body+MCP **39.8**；Body+All hand joints **43.4**（Table 1）。手指：去掉 body 粗特征改善 PA MPVPE（Table 2，方向性结论）。  
   - 与 S37：方向相反于「远端读近端」——这里是**近端（腕）读远端子节点**；对 S37 更像「父/子证据应共享」而非严格祖先条件。对应 **M4** 的证据混合。

5. **prev 局部系规范化 + 小增量旋转 — CAPTRA**【论文已有结论】  
   - 问题：类别级刚体/铰接物点云跟踪。  
   - 机制：`Z = R_t^{-1}(X−T_t)/s_t`；RotationNet 回归 **6D** 小增量，再与 prev 复合 `R_{t+1}=R_t R̂`。  
   - 消融：canonicalized CoordinateNet 显著优于未规范化；再加 RotationNet 更好（Table 4，刚性 NOCS-REAL275；文中定性，表内具体 mm 在 PDF 排版中未完整抽出——**完整单元格未在本轮 HTML 抽出，标待核**）。  
   - 对应：主文档 H2 的 D3「规范化」与 **M4/M6**；不是父骨骼，而是 **prev 部件坐标系**。

6. **父→子解析 IK（手部）— HybrIK / HybrIK-X**【论文已有结论】  
   - 用 3D 关节支架逐级求相对旋转；Adaptive 防误差沿树累积。  
   - HybrIK-X Table VII（噪声关节输入）：Naive 手 MPJPE 在中等噪声下 **57.6** vs Adaptive **16.5**（列对应噪声档；Body/Hand 分行）。  
   - 关键：需要可靠 3D 关节；S37 是参数空间 delta，无 IK 支架。

#### 文献反对什么（反例）

1. **字面固定骨骼邻接作初始化有害 — HOPE-Net**【论文已有结论】  
   - Skeleton 初始化 **12.91** vs Identity **6.81**（Table 3）。学到的邻接可连非骨骼边（如 index PIP–thumb TIP）。  
   - 含义：【推断】对 S37，「硬接线 MANO 树做证据混合」可能不如可学习混合；但本实验是 2D→3D 提升网络的邻接初始化，不是手指头条件消融。

2. **主干上下文足够时，逐部件隔离可成立 — PARE**【论文已有结论】  
   - 逐部件注意力池化 + 各自 MLP；依赖深层共享 CNN 上下文。  
   - 【推断】与主文档一致：隔离本身不是罪，前提是池化前特征有上下文。S37 主干感受野仅 2–7 ms（`00_CONTEXT` §3）→ PARE 式前提不成立。

3. **强视觉主干下单路径/简化解码足够 — HaMeR / SimpleHand**【论文已有结论】  
   - HaMeR：Transformer 直接回归 MANO；SimpleHand：token generator + 简化上采样网格回归，不强调运动学父链解码。  
   - 【推断】「单 token 解码器足够」的条件 ≈ **丰富外观特征 + 单帧绝对回归**；与 S37 稀疏事件 + 隔离证据头相反，不能用来否定父链条件，只能限制外推。

4. **硬关节约束在类别级跟踪上可能有害 — CAPTRA**【论文已有结论】  
   - §5.8：强制不准确关节约束使平移误差约 **+80%**；GT 轴硬约束增益很小（Table 2 Ours+Rot.Proj）。  
   - 【推断】「运动学树硬约束」≠「父朝向作条件特征」；后者更贴近 S37 的 D1。

#### 没有覆盖什么

- 无文献在**事件相机递推手跟踪**上做「是否读祖先证据 / 父骨骼 6D」的对照。  
- 无「逐关节门」与父链条件的交叉消融。  
- Hand4Whole 的方向是腕←MCP，不是指尖←父 MCP；远端父系条件的**手部数字**仍薄（主要靠人体 HKMR/KTD + 物体 CAPTRA 类推）。

### M5 事件编码器时间感受野与层级

- **空白**（本任务文献不覆盖事件图感受野）。PARE/HaMeR 仅说明「主干上下文丰富时可隔离解码」。

### M6 表述（delta + 课程噪声 + 轴角加性）

#### 支持

1. **连续旋转表示优于轴角/四元数（绝对回归）— Zhou et al.**【论文已有结论】  
   - 理论：SO(3) 在 ≤4 维欧氏空间无连续表示；6D/5D 连续。  
   - 经验：自编码器中 6D/5D 收敛更快；不连续表示仍可出现 **>170°** 误差，连续表示测例 **≤2°**（§5.1 / Fig.5）。  
   - 关键：主要是**绝对**旋转回归，不是残差加性 vs 群复合的消融。

2. **SVD 正交化 — Levinson et al.**【论文已有结论】  
   - 9D→SVD→SO(3)；相对 Gram-Schmidt（Zhou 6D 所用）在噪声下期望误差约一半（§3.3 推论）。多任务上 SVD 常优于经典表示。  
   - 关键：同样侧重表示/投影，非递推残差更新律。

3. **小增量在规范系用 6D 回归再复合 — CAPTRA**【论文已有结论】  
   - `R_{t+1} = R_t R̂`（SO(3) 复合），输出空间靠近恒等；附录 PCA：小角度 6D 残差前 3 主成分解释大部分方差（Table 6）。  
   - 与 S37：S37 是 **轴角向量相加**【代码事实】`model/model.py:1525-1530`，而仓库已有右侧复合约定【代码事实】`semkine/lie.py:34-37`，当前主路径未用。

#### 反对 / 空白

- **未找到**「同一手部递推设定下：轴角加性 Δ vs SO(3) 复合」的已发表消融表。  
- DeepIM 等 6D 物体迭代匹配用增量位姿（abs 页），结构先例，非手部数字。  
- 【推断】Zhou/Levinson **不能**直接证伪 S37 的轴角加性：50 ms 局部增量通常很小，加性近似误差可能低于 1.1 mm 门；需最小实验，不能靠文献定论。

### M7 泛化 / M8 训练–测试 prev 分布

- **空白**（本任务文献不覆盖）。

## 3. 对 S37 最有价值的可迁移机制（1–3 条）

### 机制 A — 父链 / 祖先姿态条件（优先）

- **数学形式（增量版）。** 对关节 k（MANO 局部序），令祖先集合 `A(k)`（或仅父 `pa(k)`）：  
  \[
  \Delta\theta_k = f_k\big(e_{k+1},\; \theta_k^{\mathrm{prev}},\; \{\theta_j^{\mathrm{prev}}\}_{j\in A(k)},\; [R_{\mathrm{pa}(k)}^{\mathrm{cam}}]_{6\mathrm{D}}\big)
  \]
  其中 \(R_{\mathrm{pa}}^{\mathrm{cam}}\) 由 prev 的 FK 得到、**detach**（与主文档 D1 一致）。  
- **接入点：** `model/model.py:_decode_active`（拼接输入）；FK 朝向可从现有 `_fk` / `model/mano_layer.py` 变换链取出。  
- **成本量级：** 每头多 3–18 维输入（仅父 vs 全祖先）；参数 +O(15×(Δin)×64)，相对 0.73 M 可忽略；延迟 ≪ 事件 GNN。  
- **文献锚：** HKMR Table 2；MAED Table 2；Hand4Whole Table 1（证据混合方向）。  
- **最小证伪实验：** 冻结主干；只训手指头。臂：(i) 现状；(ii) +父 θ_prev；(iii) +父相机系 6D；(iv) +全祖先。指标：仅手指 RA（根旋 GT 替换，仓库已有分解）；近端 vs 远端分层。若 (ii)(iii) 两种子均值改善 **<1.1 mm** 或远端不优于近端 → **推翻「父链坐标系是瓶颈」**【待实验验证的假设】。

### 机制 B — prev / 父系规范化后再回归增量（与 A 正交）

- **数学形式。** 将关节相关事件偏移或证据，用 \(R_{\mathrm{pa}}^{\mathrm{cam}\top}\) 旋到父系，再回归体坐标增量，更新用右侧复合：  
  \[
  R_k \leftarrow R_k\,\mathrm{Exp}(\delta\phi_k)
  \]
  （已存在于 `semkine/lie.py`，主路径未接）。  
- **接入点：** `semkine/routed_readout.py` 池化前；或头输入处；更新改 `forward_packet` 的 `out = delta + prev` 为 `retract_51d`。  
- **成本：** 15 次 3×3 旋转，可忽略。  
- **文献锚：** CAPTRA 规范化 + SO(3) 复合；HybrIK Adaptive（重建时用已更新父位置）。  
- **最小证伪：** 同 A 的冻结主干协议；2×2：{相机系证据, 父系证据} × {轴角加性, SO(3) 复合}。若规范化臂不优于相机系臂 → 推翻 H2-frame。

### 机制 C — 可学习跨关节混合（慎用固定树）

- **形式：** 在 16 个证据行上做轻量可学习邻接混合（初始化 Identity 或 MANO），再进各头；对照硬接线骨骼邻接。  
- **文献锚：** Hamba 去 GCN +0.7 mm；HOPE-Net Skeleton init 劣于 Identity。  
- **证伪：** 混合带来的手指 RA 增益 <1.1 mm，或 Skeleton 硬接线优于 Identity → 修正「可学习邻接必胜」叙事。

## 4. 限制与诚实声明

1. 深读核心 **12** 篇（表中 1–12 为主；13–15 浅），宁少勿滥；未做 2023–2026 全目录扫描。  
2. 所有文献数字来自 **RGB/深度单帧或短视频**，协议与 zgz 递推 RA **不可比**；只迁移机制，不迁移毫米值。  
3. CAPTRA Table 4 的逐格数值本轮 HTML 未完整抽出，只采信文中「canonicalization / RotationNet 优于消融」的方向句。  
4. HybrIK Eq.15 **不应**再被主文档写成「朴素链式神经解码有害」；应改为「Naive IK 重建误差沿祖先累加」。  
5. HOPE-Net 12.91/6.81 是 **初始化** 消融，不是固定树 vs 学习图的最终结构。  
6. **未核实：** 针对「手部残差回归」的轴角加性 vs SO(3) 复合的已发表对照表；Hamba 作者全序（仅核第一作者 Dong 与 venue）；HybrIK-X 正式 TPAMI 卷期页码（以 arXiv:2304.05690 为准）。  
7. 未改仓库；未跑 GPU；未写实验脚本。

---

**一句话总判。**【推断】文献对 **M4（祖先/父系条件与跨关节混合）** 支持较强且有消融数字；对 **M6（轴角加性有害）** 只有连续表示与物体跟踪先例，**没有**可直接搬用的手部递推反证；M1–M3、M5、M7–M8 在本专题基本空白。最可能被推翻的结论：在 S37 短感受野下「加上父链 6D 即可显著改善远端手指」——若瓶颈是证据本身不可辨识，父系条件会失败。
