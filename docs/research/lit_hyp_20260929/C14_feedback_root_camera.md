# C14：与预测比较的反馈特征、根 / 全局朝向估计与相机条件回归、事件域模型式跟踪

> 子代理 C14。2026-09-29。仅写本文件；未改代码 / 配置 / 测试 / `outputs/`；未训练、未用 GPU。
> 共用事实：`docs/research/lit_hyp_20260929/00_CONTEXT.md`。主文档已有条目见 `docs/网络结构分析.md` §2/§4 与附录；G01 目录见 `docs/research/dir12_20260928/G01_monocular_hand_mesh.md`（避免重复扩量）。
> 深读 **12** 篇（方法节 + 关键消融表已对 arXiv HTML / CVF PDF / 项目页核对）。PyMAF-X 等期刊版只列、归 D15。不覆盖时序状态（C11）与运动学父链（C13）。

证据标签按 `00_CONTEXT` §1：【代码事实】【实验事实】【论文已有结论】【推断】【待实验验证的假设】。

---

## 0. 检索记录

| 检索式 / 操作 | 来源 | 日期 | 筛选 |
|---|---|---|---|
| 种子标题：PyMAF、ReFit、DeepIM、se(3)-TrackNet、CAPTRA、CLIFF、SPEC、Zolly、HandDGP、WiLoR、Dyn-HaMR、Nehvi、Xue、EventCap | arXiv abs/html、CVF OA、项目页 | 2026-09-29 | 只保留与 M1–M8 **结构相似**（预测处取证据 / 显式比较 / 相机条件全局朝向 / 事件残差跟踪）；关键词命中但结构不像的剔除 |
| `mesh-aligned OR reprojection feedback` + human mesh | WebSearch → arXiv/CVF | 同上 | 要求有「按当前估计位置采样或渲染再回归残差」的消融 |
| `CLIFF ablation CI CS Human3.6M` | arXiv HTML `2208.00571` | 同上 | 核验主文档「约 2.6 mm」 |
| `HandDGP Table 2 CS-MVE 2.7` | arXiv HTML `2407.15844` | 同上 | 核验与 CLIFF / 他代理幅度冲突 |
| `Event-based Non-Rigid Reconstruction from Contours Xue BMVC` | arXiv `2210.06270`、MPI 项目页 | 同上 | 事件轮廓残差与失败条件 |
| `RRTrack 2607.23669` / 主文档 STORM `2511.09771` | arXiv | 同上 | 闭环精修漂出收敛域（反例臂） |

**筛选规则（相对 M1–M8）**

- **纳入**：在预测位姿处采样 / 渲染 / 重投影，并与观测比较或作为反馈喂回归器（M3）；相机 / 裁剪 / 焦距条件进回归器且有全局旋转相关消融（M3 几何条件）；事件域显式「模型→残差→更新」（M3，辅 M1）。
- **排除**：仅绝对回归无反馈；仅时序 RNN/SSM（C11）；仅运动学树解码（C13）；PyMAF-X / 期刊扩写（D15）。
- **已读重读**：WiLoR、Dyn-HaMR 在 G01 已深读，本轮只补「反馈 vs 比较」视角，标「第一轮已读」。

**覆盖缺口**：Oberweger TPAMI 2020 反馈环本轮未重开全文（主文档已 ✔）；STORM `2511.09771` 仅确认存在与主文档引用方向，未深读方法节；EventCap 灰度+事件批优化，与 S37 纯事件递推结构部分类似。未系统扫 TPAMI/IJCV 期刊反馈变体（归 D15）。

---

## 1. 候选目录表

| 标题 | 第一作者 / 末位 | venue | 年 | 链接 | 本轮 | 对应 M | 备注 |
|---|---|---|---|---|---|---|---|
| PyMAF: … Pyramidal Mesh Alignment Feedback Loop | Zhang / Sun | ICCV | 2021 | https://arxiv.org/abs/2103.16507 · CVF | **深读 HTML** | M3 | 网格对齐 vs 网格/全局特征消融 |
| ReFit: Recurrent Fitting Network for 3D Human Recovery | Wang / Daniilidis | ICCV | 2023 | https://arxiv.org/abs/2308.11184 · 项目 https://yufu-wang.github.io/refit_humans/ · DOI `10.1109/iccv51070.2023.01346` | **深读 PDF** | M3 | 重投影窗反馈；反馈 dropout |
| DeepIM: Deep Iterative Matching for 6D Pose Estimation | Li / Fox | ECCV | 2018 | https://arxiv.org/abs/1804.00175 | **深读 HTML** | M3 | 渲染–观测匹配，相对 SE(3) |
| se(3)-TrackNet | Wen / Bekris | IROS | 2020 | https://arxiv.org/abs/2007.13866 | **深读 HTML** | M3、M1 | 跨帧渲染比较；漂移与重初始化 |
| CAPTRA | Weng / Guibas | ICCV | 2021 | https://arxiv.org/abs/2104.03437 | **深读 HTML** | M3（规范化） | 按 prev 把观测变到规范系；**非**显式像素残差比较 |
| CLIFF | Li / Yan | ECCV | 2022 | https://arxiv.org/abs/2208.00571 | **深读 HTML** | M3 | CI/CS 消融 → 核实 2.6 mm |
| SPEC | Kocabas / Black | ICCV | 2021 | https://arxiv.org/abs/2110.00620 | **深读 HTML** | M3 | 相机参数拼接进回归 |
| Zolly | Wang / Komura | ICCV | 2023 | https://arxiv.org/abs/2303.13796 | **深读 PDF** | M3 | 焦距/透视畸变条件 |
| HandDGP | Valassakis / Garcia-Hernando | **venue 待核**（arXiv） | 2024 | https://arxiv.org/abs/2407.15844 | **深读 HTML** | M3 | 可微全局定位；幅度冲突已拆 |
| WiLoR | Potamias / Zafeiriou | CVPR | 2025 | https://arxiv.org/abs/2409.12259 | 第一轮已读（G01） | M3 | 手上网格对齐 Δθ |
| Dyn-HaMR | Yu / Birdal | CVPR | 2025 | https://arxiv.org/abs/2412.12861 | 第一轮已读（G01） | M3 | 测试期重投影优化根 |
| Differentiable Event Stream Simulator … | Nehvi / Theobalt | CVPRW | 2021 | https://arxiv.org/abs/2104.15139 | **深读 HTML** | M3 | 渲染差分事件 + 相关能量 |
| Event-based Non-Rigid Reconstruction from Contours | Xue / Stückler | BMVC | 2022 | https://arxiv.org/abs/2210.06270 | **深读 HTML** | M3 | 轮廓关联 EM；失败条件 |
| EventCap | Xu / Theobalt | CVPR | 2020 | https://arxiv.org/abs/1908.11505 | **深读 HTML** | M3 | 事件轨迹 + 批优化 + 强度图 |
| PyMAF-X | Zhang / … | TPAMI | 2023 | https://arxiv.org/abs/2207.06400 | **只列** | — | 归 D15 |
| RRTrack | Li / Li | arXiv | 2026 | https://arxiv.org/abs/2607.23669 | 摘要+方法开头 | 反例 M3 闭环 | 渲染掩膜一致性抑漂 |

---

## 2. 按 M1–M8：文献支持 / 反对 / 空白

### M1 常数增益融合（Δ = F + G）

- **支持（间接）**【论文已有结论】：滤波式跟踪与优化残差都用「当前状态 + 新息」而非常数先验 alone（KalmanNet 主文档已列；se(3)-TrackNet 每步回归相对 `Δξ`；Nehvi/Xue 每缓冲重算残差）。没有一篇把「单层线性 F + 看不见事件的 G」当作推荐结构。
- **反对 / 空白**：本批几乎无人做「常数增益 vs 状态依赖增益」的 head-to-head（增益门控属主文档 KalmanNet / Selective Fusion，不在本任务深读核心）。**【推断】** M1 不能单靠本批反馈论文证成或证伪。

### M2 逐包系统偏差（增益救不了）

- **空白（本批）**：反馈/相机论文多为单帧或单序列优化，不报告「同包迭代收敛点 ≈ 闭环」这类分解。【实验事实】该现象来自仓库 `S37_ROUTED_READOUT_PREREG.md` §8，非文献。
- **弱支持**【论文已有结论】：EventHands 开环+外置 KF、主文档 WHAM/GVHMR 对自回归漂移的批评——更好的逐包绝对测量比改增益更关键；与 CNN 根替换杠杆一致，但是旁证。

### M3 缺少观测减预测 / 几何条件（本批主战场）

#### (a) 预测处取特征 vs 显式比较残差

| 工作 | 做法 | 与 S37 | 消融：只对齐 vs 显式比较 |
|---|---|---|---|
| **PyMAF** | \(X_t=\Pi(\tilde M_t)\)，在投影点双线性取特征再回归 \(\Delta\Theta\) | S37：按 prev 分组 `a(prev)` 后**丢弃** \(d,v^*\)，证据是绝对池化特征【代码事实】`routed_readout.py:40-118`；`model.py` 路由后根头不读偏移 | **有**：Table 4 Human3.6M，无辅助监督。金字塔下 Mesh-aligned **76.8 / 50.9** vs Grid **80.5 / 54.7** vs Global baseline **84.1 / 55.6**（MPJPE / PA-MPJPE）。结论：仅均匀网格弱于**按当前网格投影对齐**；对齐特征 ≠ 有符号残差【论文已有结论】全文 HTML |
| **ReFit** | 关键点重投影到特征图开窗 \(r=3\)，拼 \(\Theta_t\) 与 bbox，26 路 GRU 更新 | 比 PyMAF 更接近「拟合」；仍是学习特征→\(\Delta\)，窗内不是 \(\Pi(M)-I\) | Table 3：反馈 dropout、多步、full-frame adjusted reprojection、26 GRU vs 1 GRU 均有收益；**没有**「关掉重投影、只喂全局特征」与「显式 2D 残差向量」的头对头【论文已有结论】PDF |
| **DeepIM** | 渲染当前估计 vs 观测，FlowNet 骨干回归相对 SE(3) | 显式 render-and-compare | 训练需多轮迭代匹配测试分布；初值差时单步不够【论文已有结论】 |
| **se(3)-TrackNet** | \(R_{t-1}\) 渲染 vs \(O_t\)，回归 \(\Delta\xi\in se(3)\) | 跨帧闭环；刚体 RGB-D | 相对 DeepIM tracking 更少重初始化；仍依赖 CAD【论文已有结论】 |
| **CAPTRA** | 用 \(T_t^{-1}\) **规范化点云**再回归小 \(\hat R\) | 「按 prev 对齐观测」；**没有**像素残差比较 | Table 4：canonicalized CoordinateNet ≫ 无规范化；属 M3 的「坐标系条件」而非「观测−预测」【论文已有结论】 |
| **WiLoR**（第一轮已读） | 粗网格多尺度顶点采样 → \(\Delta\theta\) | 手部 PyMAF 对应；聚合掉部件杠杆 | 主表 PA，不暴露相机系根旋 |
| **Dyn-HaMR**（第一轮已读） | 测试期 \(\mathcal L_{2d}\) 重投影优化，先动根 | 显式几何残差，但是优化器不是 50 ms 因果头 | 先验项使「残差=0 仍可动」 |

**【推断】** 文献强支持「在预测位置取条件证据」优于全局特征（PyMAF Table 4）；**弱支持**「必须构造显式 \(\mathrm{obs}-\mathrm{pred}\) 残差」——多数深度方法停在对齐特征。S37 现状是「按 prev 分组但不比较」，结构上更接近 CAPTRA 的规范化一半 + PyMAF 的采样一半，却丢掉了两者都保留的「当前位置上的比较信号」。

#### (b) 反馈回路在闭环中不稳定

- 【实验事实】仓库两遍 `it2`/`sem_it2` render-and-compare：**明确撤回**；abs MPJPE 与 jitter 爆炸，延迟近翻倍（`docs/FAILURE_AND_CLEANUP_LEDGER.md` 表行「两遍 iterative render-and-compare」）。
- 【论文已有结论】DeepIM / se(3)-TrackNet：初值远离时需迭代拉近渲染与观测；DeepIM 跟踪变体平均约每 340 帧需 PoseCNN 重初始化（se(3)-TrackNet 文中转述）。Xue：增量 EM **可漂移**，但有足够轮廓事件时可「snap」回轮廓；初值噪声大则失败。Nehvi：假设**第一帧形状准确投影**；黑底合成。RRTrack：用渲染掩膜一致性检测漂移并触发校正（承认闭环精修会漂）。
- **空白**：几乎没有与 S37 同设定（事件、MANO 51、50 ms 因果、无第二遍渲染）的「两遍反馈爆炸」复现；仓库失败是最直接证据。

#### (c) 全局朝向与相机 / 裁剪条件

| 工作 | 条件量 | 消融数字（核实） |
|---|---|---|
| **CLIFF** | \(I_{bbox}=[c_x/f,\,c_y/f,\,b/f]\)（CI）+ 全图重投影监督（CS） | **Table 2 Human3.6M**：w/o CI&CS **85.2** → w/o CI **84.0** → full **81.4**（MPJPE）。**去掉 CI 相对 full：+2.6 mm**（84.0−81.4）。去掉 CI&CS：+3.8 mm。PA-MPJPE 仅 54.5→52.1。【论文已有结论】**主文档「约 2.6 mm」成立，且是 MPJPE（含全局旋转），不是 PA** |
| **SPEC** | CamCalib：pitch/roll/vfov；拼进 HMR 式回归 | 在 SPEC-SYN/MTP 上 W-MPJPE 相对 IWP 假设方法改善显著；3DPW 上 PA 改善较小——收益主要在世界/相机系朝向【论文已有结论】Table 2–3 |
| **Zolly** | 畸变图 + 距离/焦距；透视+弱透视损失 | 针对近距透视畸变；固定内参 DAVIS 场景优先级低于 CLIFF 式位置编码【论文已有结论】 |
| **HandDGP** | 可微 DLT 式全局定位 + 图像整流 | 见下节 (e) 幅度拆解；改善的是 **相机系平移/尺度对齐**，不是根旋转角单独报告 |
| **ReFit** | full-frame adjusted reprojection + bbox | 与 CLIFF 同族；消融称忠实全图相机模型在「所有阶段」最好【论文已有结论】 |

**【代码事实】** S37 根头 `Linear(4624→6)` 不读 \(K\)、深度、\(r_{prev}\)（`model/model.py` 根头路径；`00_CONTEXT` §3）。事件 token 含归一化像素坐标，但全局读出与路由丢弃 \(d\) 后进不了根比较。

**【推断】** 对 S37：DAVIS 全图、固定 \(K\)，CLIFF 的「裁剪丢位置」不如「根头看不见几何比较」致命；更贴的迁移是 **把丢弃的 \(d\) / 投影偏移 / 深度杠杆送进根头**（类 PyMAF/ReFit 位置条件 + 仓库 §9.2 手工剪影残差），而不是再估 vfov（Zolly/SPEC 的主靶）。

#### (d) 事件域模型式跟踪：残差怎么造、何时失败

| 工作 | 残差构造 | 失败 / 限制 |
|---|---|---|
| **Nehvi CVPRW 2021** | 可微渲染前后两帧灰度差分 → 平滑阈值 → 与观测事件帧相关；另有 no-event、剪影项 | 需好初值；黑底/合成假设；约 1.5 min/帧；Xue 指非黑底时生成事件与真实偏差导致不鲁棒【论文已有结论】 |
| **Xue BMVC 2022** | EM：事件↔轮廓三角面关联；最大化视线与面的对齐（侧向/法向） | 静态背景；缺轮廓事件（背景同色）失败；轮廓**不足以同时解 R、t 与形状**（文末自述）；可漂移；非实时（MANO 缓冲约 8.76 s）【论文已有结论】Table 1：合成 MANO mean MPJPE **Ours 明显优于 Nehvi 11.61 mm**（约 2.5×） |
| **EventCap CVPR 2020** | 异步事件轨迹切片 + 强度图 CNN 2D/3D 位姿批优化 + 边界精修 | 依赖低帧率强度图抑漂；人体骨架非 MANO 手；批优化非因果 50 ms【论文已有结论】 |

**【推断】** 事件模型式跟踪一致要求：**显式残差 + 好初值 + 足够轮廓事件**。S37 学习式路由有「关联」的影子（事件→部件），但没有「预测轮廓 vs 事件」的比较项——与 Xue/Nehvi 的结构差在 M3，而不在「用不用事件」。

#### (e) CLIFF ≈2.6 mm vs HandDGP 幅度冲突 —— 已核实

1. **CLIFF「约 2.6 mm」**【论文已有结论】：Table 2，Human3.6M，**MPJPE**：full 81.4 vs w/o CI 84.0 ⇒ **Δ = 2.6 mm**。主文档写法正确；勿与 PA-MPJPE（Δ≈0.3）或 w/o CI&CS 的 3.8 mm 混用。
2. **HandDGP「2.7 mm」**【论文已有结论】：正文写 FreiHAND 上相对 A2（w/o DGP）**CS-MVE 降 2.7 mm**（49.0→46.3，Table 2）。同表 HO3D：**35.1 mm**；Human3.6M：**17.3 mm**。另有相对 CMR 的 **2.6 mm** 相机系改进叙述。
3. **「81.3→46.3」冲突来源**【推断】：跨表误读——CLIFF Table 1 里 HMR 在 3DPW 的 **PA-MPJPE 81.3**，与 HandDGP FreiHAND **CS-MVE 46.3** 无关。两代理读的是不同列/不同数据集。
4. **Venue**【论文已有结论】：截至本轮只核实 arXiv `2407.15844`（Valassakis, Garcia-Hernando）；**正式会议/期刊 venue 仍标待核**（与主文档 ✘ 一致）。

**净结论**：两个「约 2.6–2.7 mm」都真实，但 **指标与对象不同**（全身裁剪位置条件的 MPJPE vs 手部相机系顶点误差的 DGP 消融），**不能互相替代，也不能用来互相否定**。

### M4 手指父链 / 门

- **空白（本任务）**：故意不深读。ReFit 的每关节 GRU 是「部件更新解缠」旁证，细节交 C13。

### M5 编码器时间感受野

- **空白**：本批反馈/相机方法不检验事件 GNN 感受野。

### M6 表述（delta + 课程噪声）

- **弱反例**【论文已有结论】：CAPTRA / DeepIM / se(3)-TrackNet 都在小运动体制下回归相对量且成功——相对更新本身可工作；失败模式更常是 **缺少可比较的测量**（回到 M3）或训练分布（M8）。
- **空白**：无「GNN×绝对」格子的文献对应。

### M7 泛化

- **空白**：本批非 zgz / 非事件受试者迁移。

### M8 训练/测试 prev 误差分布

- **弱支持**【论文已有结论】：DeepIM 训练时多轮迭代以匹配测试迭代；se(3)-TrackNet 训练噪声 \(\sigma_t,\sigma_w\)；CAPTRA 在线加噪生成规范化点云。均承认 **训练时的状态误差分布要覆盖测试递推**。
- 【实验事实】S37 课程噪声独立、测试自相关（`00_CONTEXT` §3–4）——文献方向一致，无事件手直接数字。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A — 预测位置上的**比较量**进根头（主推，对 M3）

- **数学形式**【推断】：保留现有路由 \(a_{ij}\)。令事件相对指派顶点的图像偏移  
  \(\mathbf{r}_i = (p_i - \Pi(v^*_i(\mathrm{prev})),\; d_i)\)（或部件均值 \(\bar{\mathbf{r}}_j=\sum_i a_{ij}\mathbf{r}_i/\sum_i a_{ij}\)）。  
  根增量 \(\Delta root = F_\phi\big([\mathrm{feat};\, e_{0:15};\, \bar{\mathbf{r}}_{0:15};\, z_{\mathrm{prev}}]\big)\)，要求 \(\bar{\mathbf{r}}=0 \Rightarrow\) 该支路贡献接近 0（可选硬约束）。
- **接入点**【代码事实】：`semkine/routed_readout.py:route_front_vertex_lbs` 已返回 `d,v` 却在 `model/model.py` 调用处丢弃；池化后根头 `Linear(4624→6)`。最小改动：在 `pool_joint_evidence` 旁并行池化偏移，拼进根头输入。
- **成本量级**【推断】：零训练探针（拼已有 `d`）CPU；冻结主干训根头 ~10²–10³ 步；不引入第二遍渲染（避开 it2 爆炸模式）。
- **最小证伪**【待实验验证的假设】：  
  1. 同 checkpoint 把 `d`/偏移拼进根头 vs 现网：G2（10° 纠正增益）不升、G1 不降 → **「缺比较」不是根旋瓶颈**。  
  2. 若只做「按 GT 投影对齐再池化绝对特征」（纯 PyMAF 式、无 \(\mathbf{r}\)）也不升 → 需要**显式残差**而非仅对齐采样。  
  判据沿用 rootinnov 门 + RA ≥1.1 mm。

### 机制 B — 相机 / 裁剪几何条件（次优先）

- **形式**：\(I_{cam}=[c_x/f,c_y/f,b/f]\) 或已知 \(K\) 的归一化量 + prev 腕深，拼进根头（CLIFF CI）。
- **接入**：根头输入拼接；监督若改 2D，须在**全图相机系**（CLIFF CS），与当前参数空间 MSE 不同——成本更高。
- **证伪**：固定内参、几乎不裁剪的 DAVIS 上，只加 CI 类输入、两种子 RA 改善 < 1.1 mm → 主文档「根缺 K」优先级应下调（本批【推断】倾向于此）。

### 机制 C — 事件轮廓关联残差（高风险，对照用）

- **形式**：Xue 式视线–轮廓面能量，或 Nehvi 式渲染事件相关；作测试期优化或小新息头。
- **接入**：评测期旁路，不进主训练图；或冻结主干只对根做短程 LM。
- **证伪 / 风险**：仓库 it2 已表明**第二遍渲染反馈在递推下可爆炸**【实验事实】；Xue 自述轮廓不够解 R+t+形状。若从 GT 初值单缓冲优化根旋明显降角误差、但从闭环 prev 发散 → 证实「残差有信息但闭环不稳」，应先做机制 A 的一次前向比较量，而不是两遍 render-and-compare。

---

## 4. 限制与诚实声明

- 深读 12 篇 + 2 篇第一轮补视角；未宣称穷尽。CVF 部分页曾 403，以 arXiv HTML / 项目 PDF 为准。
- 所有 mm 数字来自原文表；**未**把文献 PA-MPJPE / CS-MVE 与 zgz 递推 RA 混比。
- HandDGP venue、STORM `2511.09771` 方法细节、Oberweger 全文本轮未补核。
- DeepIM 作者：Yi Li, Gu Wang, Xiangyang Ji, Yu Xiang, Dieter Fox（arXiv HTML）。
- ReFit：arXiv `2308.11184`；本轮读 CVF/项目 PDF。
- **最可能被推翻的结论**：见文末回复 ④——「对齐采样特征已足够、不必显式 \(\mathrm{obs}-\mathrm{pred}\)」可能被机制 A 的探针推翻或反过来被证伪。

---

## 附：重点问题短答索引

| 问题 | 结论标签 |
|---|---|
| (a) 预测处特征 vs 显式比较 | PyMAF Table 4：**对齐 ≫ 网格/全局**；鲜有「显式残差 vs 对齐特征」头对头 → S37「分组但不比较」缺的是比较信号，文献偏向先补对齐条件特征【论文+推断】 |
| (b) 闭环反馈不稳 | 仓库 it2 爆炸【实验事实】；DeepIM/Xue/RRTrack 承认漂移/需重初始化【论文】 |
| (c) 相机条件消融 | CLIFF CI：**+2.6 mm MPJPE**（H36M Table 2）【论文】；SPEC/Zolly 对变化焦距/俯仰更关键 |
| (d) 事件模型式跟踪 | Nehvi 相关能量；Xue 轮廓 EM；失败：坏初值、无轮廓事件、轮廓不够解满 6D+形状【论文】 |
| (e) 2.6 vs HandDGP | CLIFF 2.6 与 HandDGP FreiHAND 2.7 **皆真、不同指标**；81.3→46.3 为跨表误读【论文+推断】 |
