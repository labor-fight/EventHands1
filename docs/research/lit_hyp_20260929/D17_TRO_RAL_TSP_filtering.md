# D17：期刊文献——TRO / RA-L / TSP / T-IV 学习型滤波、学习协方差、事件跟踪

> 子代理 D17 · 2026-09-29 · 全部简体中文  
> 依据：`docs/research/lit_hyp_20260929/00_CONTEXT.md`；主文档 `docs/网络结构分析.md` §2 / 附录；避免重复 `docs/research/dir12_20260928/G11_G12_vo_slam_egomotion.md`（事件 VO/SLAM 几何）与 `C12_learned_filtering_gain.md`（ML 会议学习滤波）。  
> 纪律：未改仓库代码/配置/测试/`outputs/`；未删除文件；未训练；未用 GPU。  
> 与 C12 分工：C12 管 NeurIPS/ICML/CVPR 等；**本笔记管机器人与信号处理期刊**（及任务点名的结构相似旁证）。

---

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29 |
| 检索式（WebSearch） | `TLIO Tight Learned Inertial Odometry RA-L 2020`；`AI-IMU Dead-Reckoning T-IV 2020`；`KalmanNet TSP 2022 Revach`；`Split-KalmanNet Choi TVT`；`RNIN-VIO learned covariance`；`Latent-KalmanNet TSP`；`ESVO Zhou TRO 2021`；`Continuous-Time Visual-Inertial Odometry Event Cameras Mueggler TRO 2018`；`learned covariance ablation fixed Kalman VIO`；`KalmanNet colored noise temporally correlated`；`Towards Consistent Batch State Estimation Time-Correlated Measurement Noise` |
| 核实来源 | arXiv abs / ar5iv HTML；Crossref DOI；IEEE Xplore 书目；官方项目页（cathias.github.io/TLIO、KalmanNet GitHub、zju3dv/rnin-vio、HKUST ESVO）；HAL PDF（AI-IMU）；Weizmann PDF（Split-KalmanNet）；RPG / ifi.uzh PDF（连续时间事件 VIO） |
| 筛选规则 | 只保留与 `00_CONTEXT` §6 的 **M1–M8** 结构相似者：学习测量 ± 学习协方差 + 经典滤波器；学习 Kalman 增益 / 分裂协方差；测量误差时间相关；事件跟踪中增益与不确定度如何设定。按关键词命中但只是「几何 VO/BA、无滤波增益故事」的，交给 G11/G12，本笔记只抽不确定度设定。 |
| 排除 | 纯 ML 会议可微滤波（Backprop KF、HTTYDF、RKN）→ C12；事件立体深度数值 → G06/G07；无学习、无协方差消融的一般 VIO 综述。 |
| 覆盖缺口 | (1) IJRR 上「学习协方差 + 滤波」结构相似全文本轮未稳定取到（检索到 Barfoot 等稀疏变分推断等，但不贴 M1 消融）；(2) KalmanNet / Split-KalmanNet **未做有色测量噪声**专用实验；(3) 事件手部跟踪上无「学习 R + EKF」期刊先例；(4) ESVIO（Sensors / RA-L 旁支）事件残差常取单位协方差，作旁证不入深读计数。 |

深读篇数：**11**（其中 KalmanNet 标「第一轮已读·本轮核实」；ESVO 几何部分 G11 已读，本任务补不确定度视角）。另 3 篇结构旁证（RNIN-VIO、Wagstaff AIM、时间相关批估计）入候选表但 venue 不在指定清单或正式 venue 待核，深读计数中 RNIN 与时间相关文计入（任务点名或直接关 M2），Wagstaff 仅旁证不计入深读总数。

---

## 1. 候选目录表

| # | 标题 | 作者（一作 / 末位） | venue · 年 | 链接 | 读到 | 对应 M | 深读？ |
|---|---|---|---|---|---|---|---|
| 1 | TLIO: Tight Learned Inertial Odometry | Wenxin Liu / Jakob Engel | IEEE RA-L 2020 | https://arxiv.org/abs/2007.01867 · DOI [10.1109/LRA.2020.3007421](https://doi.org/10.1109/LRA.2020.3007421) · [项目](https://cathias.github.io/TLIO/) | ar5iv 全文 HTML | M1,M3,M8 | ✓ |
| 2 | AI-IMU Dead-Reckoning | Martin Brossard / Silvère Bonnabel | IEEE T-IV 2020 | https://arxiv.org/abs/1904.06064 · DOI [10.1109/TIV.2020.2980758](https://doi.org/10.1109/TIV.2020.2980758) | ar5iv / HAL 全文 | M1 | ✓ |
| 3 | KalmanNet: Neural Network Aided Kalman Filtering… | Guy Revach / Yonina C. Eldar | IEEE TSP 2022 | https://arxiv.org/abs/2107.10043 · DOI [10.1109/TSP.2022.3158588](https://doi.org/10.1109/TSP.2022.3158588) | ar5iv 全文 | M1,M3,M7 | ✓ 第一轮已读 |
| 4 | Split-KalmanNet… | Geon Choi / Namyoon Lee | IEEE TVT 2023（KalmanNet 后续；非 TSP） | https://arxiv.org/abs/2210.09636 · DOI [10.1109/TVT.2023.3270353](https://doi.org/10.1109/TVT.2023.3270353) | 官方 PDF | M1 | ✓ |
| 5 | Latent-KalmanNet… | Itay Buchnik / Nir Shlezinger | IEEE TSP 2024 | https://arxiv.org/abs/2304.07827 · DOI [10.1109/TSP.2023.3344360](https://doi.org/10.1109/TSP.2023.3344360) | ar5iv 全文 | M3,M5（弱） | ✓ |
| 6 | Event-Based Stereo Visual Odometry（ESVO） | Yi Zhou / Shaojie Shen | IEEE TRO 2021 | https://arxiv.org/abs/2007.15548 · DOI [10.1109/TRO.2021.3062252](https://doi.org/10.1109/TRO.2021.3062252) | ar5iv 全文 | M3（几何新息） | ✓ 不确定度视角；几何 G11 第一轮 |
| 7 | Continuous-Time Visual-Inertial Odometry for Event Cameras | Elias Mueggler / Davide Scaramuzza | IEEE TRO 2018 | https://arxiv.org/abs/1702.07389 · DOI [10.1109/TRO.2018.2858287](https://doi.org/10.1109/TRO.2018.2858287) | PDF 全文 | M3 | ✓ |
| 8 | ESVO2: Direct Visual-Inertial Odometry with Stereo Event Cameras | Junkai Niu / Yi Zhou 等 | IEEE TRO 2025（仓库/PDF 声明） | https://arxiv.org/abs/2410.09374 | PDF | M3 | ✓ 轻（权重设定） |
| 9 | Towards Consistent Batch State Estimation Using a Time-Correlated Measurement Noise Model | （作者见 arXiv） | arXiv:2303.06507；正式 venue **待核** | https://arxiv.org/abs/2303.06507 · HTML | 全文 HTML | M2 | ✓ |
| 10 | RNIN-VIO: Robust Neural Inertial Navigation Aided VIO… | Danpeng Chen / Guofeng Zhang | ISMAR 2021 / TVCG 特刊（**非** TRO/RA-L） | https://zju3dv.github.io/rnin-vio/ · PDF | PDF 全文 | M1,M3 | ✓ 任务点名旁证 |
| 11 | A Self-Supervised, Differentiable Kalman Filter for Uncertainty-Aware VIO | Brandon Wagstaff / Jonathan Kelly | IEEE/ASME AIM 2022（非指定期刊） | https://arxiv.org/abs/2203.07207 · DOI 10.1109/AIM52237.2022.9863270 | PDF | M1,M3 | 旁证，不计入 11 |

未入深读：Ultimate SLAM（RA-L 2018，G11 已读，本任务无新增不确定度消融）；ESVIO Chen 等（事件残差常取 \(P=I\)，仅作反例旁注）；FlexKalmanNet / 双层优化学协方差（ICRA/预印，结构弱相关）。

---

## 2. 按结构问题 M1–M8：支持 / 反对 / 空白

### M1 常数增益融合

**【论文已有结论 · 支持「学习测量 + 学习协方差 + 经典滤波」优于固定协方差】**

1. **TLIO（RA-L 2020，全文 HTML）**  
   - 问题：仅 IMU 行人 6-DoF；网络回归 3D 位移 **与** 不确定度，紧耦合进 stochastic cloning EKF。  
   - 相对最佳 RoNIN 速度拼接：平均 yaw / 位置漂移分别降 **27% / 33%**（摘要与 §I）。  
   - **消融（§VII-B1，Fig. 8）**：手调常数协方差 `tlio-fixcov` / `tlio-mse` **达不到** 用网络回归协方差的 `tlio`；常数协方差还有初始化失败导致 ATE/Drift CDF 到不了 100%。相对 `3d-ronin-mse`，TLIO 平均 yaw / 位置漂移再降 **27% / 31%**。【论文已有结论】  
   - 与 S37：**有显式测量** \(d\) 与 \(R(d)\)，增益由 EKF 算；S37 是 `F+G` 无门、无 \(R\)。【代码事实】`model/model.py` 根头线性 + `prev_mlp` 残差相加。

2. **AI-IMU（T-IV 2020，全文）**  
   - 问题：车载仅 IMU；IEKF + 伪测量（侧向/垂向速度≈0）；**CNN 动态适配伪测量噪声 \(N_n\)**。  
   - KITTI 平均平移误差 **1.10%**（摘要 / Table 1）。  
   - **消融（§V-D，seq.01）**：完整方法 \(t_{rel}=1.11\%\)；**无协方差适配** → **1.94%**；无外参对齐 → 1.65%。转弯时适配器把协方差放大约 **\(10^2\)**（Fig. 8）。【论文已有结论】  
   - 注意：适配出的 \(N_n\) **可为了滤波性能故意偏离真实统计**（文中明确）。

3. **KalmanNet（TSP 2022，第一轮已读·本轮核实）**  
   - 用 RNN 学 \(K_t\)，更新 \(\hat x = \hat x^- + K_t\Delta y\)；**前提是显式新息**。  
   - 线性状态演化失配：相对错配 MB-KF 约 **+3 dB**（Fig. 6(a)）；观测旋转失配仍可近 MMSE（Fig. 6(b)）。【论文已有结论】  
   - 合成非线性 + 部分信息：Table IV 等显示 KB 相对 EKF/UKF/PF 在失配下更稳。【论文已有结论】

4. **Split-KalmanNet（TVT 2023，全文 PDF）**  
   - 两个 RNN **分别**学先验协方差与新息协方差，再与 \(H_t\) 组成 \(K_t\)。  
   - 噪声异质 \(\nu\) 大时，KalmanNet MSE 超 perfect EKF，**Split 仍近 MMSE**（Fig. 3–4）；时变 \(R_t\) 时 Split 可跟踪慢变失配（Fig. 5）。【论文已有结论】

5. **RNIN-VIO（ISMAR/TVCG，旁证）**  
   - 学习位移 + log-对角协方差进 EKF；挑战场景相对纯 VIO ATE 明显下降（Table 5：如 Challenging02 **0.743 → 0.520 m**）。【论文已有结论】  
   - 训练技巧：先 MSE 再 NLL（否则难收敛）——与「协方差头可训性」相关。

**【反对 / 边界】**  
- TLIO Fig. 9：**高频更新仍改善 ATE/Drift，尽管测量时间相关**——说明在「短窗位移测量 + 正确相关建模（stochastic cloning）」下，滤波仍有增益；**不是**「任何相关误差都使滤波无用」。【论文已有结论】  
- 若缺少可写的 \(\Delta y\)，KalmanNet 类结构**无法直接移植**到 S37 的 `F+G`。【推断】

**【净结论】** **支持 M1 的「状态/测量相关增益优于常数」**，但期刊证据几乎一律要求 **显式测量 +（通常）学习 \(R\)**；反对「无新息的常数 F+G 加个标量门就等于 Kalman」。

### M2 逐包误差是时间相关系统偏差

**【论文已有结论 · 支持「相关测量噪声时白噪声 KF / 只调增益不足」】**

1. **Time-Correlated Measurement Noise（arXiv:2303.06507，全文；venue 待核）**  
   - 真实激光地标测量协方差**非块对角**（Fig. 1）；带宽 \(b=0\)（忽略相关）ergodic NEES / RMSE 最差；\(b\ge 1\) 显著改善（Fig. 2 仿真；Table I / Fig. 4 实验）。【论文已有结论】  
   - 直接支持：前端误差时间相关时，应学 **相关噪声模型**，而非仅对角 \(R\) 或仅 \(K\)。

2. **TLIO Fig. 9** 承认测量时间相关，但仍用滤波+克隆；改善来自正确处理相关与自适应 \(R\)，不是否认相关。【论文已有结论】

3. **KalmanNet / Split** 公式与实验均以 **白噪声 / 失配 \(Q,R\)** 为主；**未见**「加性慢变测量偏差 \(b_t\)」专用实验。【空白 → 对 M2 无直接支持】

**【与 S37】** 【实验事实】同包迭代不动点 ≈ 闭环误差（`00_CONTEXT` §4 §8）；TF 误差与运动/事件数无关 → 固定噪声底。【推断】期刊滤波文献**更支持「若偏差是慢色噪声，要增广/差分/批相关模型」**，与 C12 的 M2 读法一致；**不支持**「只换学习增益就能把系统偏差滤掉」。

**【净结论】** **支持 M2 作为与 M1 竞争的机制**（尤其时间相关测量模型文）；KalmanNet 线对「有色系统偏差」**空白**。

### M3 缺少观测减预测 / 几何条件

**【支持】** 所有成功期刊滤波都构造新息或几何残差：

| 工作 | 新息 / 残差形式 | 不确定度 |
|---|---|---|
| TLIO | 网络位移 vs 状态克隆预测 | 网络 \(R\) + NEES 一致性 |
| AI-IMU | 伪测量 \(v^{\mathrm{lat}},v^{\mathrm{up}}=0\) | 学习 \(N_n\) |
| KalmanNet / Split | \(\Delta y = y - h(\hat x^-)\) | 学 \(K\) 或 \(\Sigma,S\) |
| ESVO | 时间表面时空一致性 / 跟踪对 TS 负片 | Student-\(t\) 残差 → IRLS；深度融合用 \(t\) 滤波；跟踪用 Huber + LM |
| 连续时间事件 VIO | 事件重投影 + IMU 连续时间残差（B 样条） | NLS，传感器噪声模型（固定类） |
| ESVO2 | 视觉残差 + IMU 预积分 | IMU 用传播协方差加权；视觉侧见原文 |

**【ESVO 数字（Table III）】** Student-\(t\) IRLS vs 标准 LS：均值深度误差 **2.76 → 2.15 cm**，std **2.94 → 1.29 cm**（仿真三平面）。【论文已有结论】  
不确定度还用于剪枝不可靠深度（Fig. 14）。**【事件 VO 的「增益」不是学出来的 \(K\)，而是残差权重 / 鲁棒核 / 协方差传播。】**

**【与 S37】** 【代码事实】路由丢弃 \(d\)、顶点 id；根头不读 \(r_{\mathrm{prev}}\)、\(K\)、深度（`routed_readout.py`、`model.py` 根头）。完美路由下 \(z\) 对 prev 误差 \(R^2=0\)（`00_CONTEXT` §4 §9.2）。【推断】期刊一致要求**可减的预测**；无几何新息则无 TLIO/KalmanNet 可挂载点。

**【净结论】** **强支持 M3**。

### M4 手指头缺父链 / 证据混合 / 无逐关节门

**【空白】** 本批 TRO/RA-L/TSP/T-IV 滤波文均为刚体定位 / 标量或低维状态 SS 模型，**无关节手部**结构对应。不据此支持或反对 M4。

### M5 事件编码器时间感受野与层级

**【弱相关】** Latent-KalmanNet（TSP 2024）：高维观测先编码到潜空间再 KalmanNet；强调表征需「利于跟踪」。【论文已有结论】  
**【空白】** 无事件 GNN 感受野 2–7 ms 的期刊对照。不把 Latent-KNet 升格为 M5 证据。

### M6 表述（delta + 课程噪声 → 收缩）

**【弱旁证】** TLIO / AI-IMU / RNIN 的网络输出是**相对位移测量**，状态仍由滤波积分；不是「\(\Delta=\) 网络绝对回归 + 盲先验相加」。【推断】结构上更接近「测量模型」，反对 S37 把网络当完整后验更新。  
**【空白】** 无 MANO 轴角课程噪声期刊文。

### M7 泛化

**【支持弱】** KalmanNet 强调数据效率、短轨迹训练可测长轨迹（Table I 等）。【论文已有结论】  
TLIO / AI-IMU 跨序列留一训练（AI-IMU 明确 leave-one-sequence）。**【空白】** 无「事件手受试者迁移」期刊滤波文。

### M8 训练/测试 prev 误差分布不一致

**【弱相关】** TLIO 用 stochastic cloning 显式处理重叠窗相关；训练用 GT 位移监督网络。【论文已有结论】  
Wagstaff（AIM，旁证）用可微 EKF 在训练时暴露闭环不确定性。**【空白】** 无「课程独立噪声 vs 自生成相关误差」期刊直接消融。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A（优先）：学习测量协方差门控事件支路（TLIO / AI-IMU 型）

- **数学形式**：把根更新写成  
  \(\hat x^+ = \hat x^- + K(\hat R)\,(z - h(\hat x^-))\)，  
  或简化 \(\Delta_{\mathrm{root}} = \sigma(-s)\,F + \sigma(s)\,G\)，其中标量/对角 \(s=\mathrm{MLP}(\mathrm{coverage},|F|,|z-h|)\) 扮演 \(\log R\)。  
- **接入点**：`model/model.py` 根融合（`root_head` 输出与 `prev_mlp` 相加处，约 `_decode` / 残差相加）；可靠度特征来自 `semkine/routed_readout.py` 的 `coverage` 与（若恢复）距离 \(d\)。  
- **成本量级**：冻结主干，只训门/协方差头（千–万参）；前向多一次小 MLP，相对 0.827 G 可忽略。  
- **最小证伪**：  
  1. 零训练 oracle：每步用 GT 在 \(\{0,F,G,F+G\}\) 或网格 \(\alpha F+(1-\alpha)G\) 上选最优根旋；上界 \(\lt 1.1\,\mathrm{mm}\) → 放弃增益故事。  
  2. 固定 \(\alpha\) vs 学习 \(\alpha(\mathrm{coverage})\)：若学习门相对最佳常数 \(\alpha\) 改善 \(\lt 1.1\,\mathrm{mm}\) 且两种子不劣失败 → **反对「自适应增益必要」**。  
  3. 若门在干净 prev 上仍注入大角（G1 失败）→ 门只是整体缩小，等价常数 \(\rho\)。

### 机制 B：显式几何新息（ESVO / KalmanNet 前提）

- **数学形式**：\(r = \phi(\mathrm{events}) - \Pi(\mathrm{MANO}(\mathrm{prev}))\)（轮廓/时间面/投影残差），\(\Delta = K\,r\)。  
- **接入点**：恢复路由丢弃的 \(d\) / 顶点；或 `semkine` 渲染 prev 与事件比较（对照 mesh_query / rootinnov 臂设计）。根头输入从「绝对池化特征」改为「残差特征」。  
- **成本**：一次投影/距离场，CPU 探针级可先做；全训练再开。  
- **最小证伪**：冻结主干，只训「残差 → \(\Delta R\)」头；对照「同样容量读绝对 \(z\)」。若残差头在 TF 干净样本上仍 \(\ge 3.9^\circ\) 注入且闭环不降 → **表示瓶颈在编码器（M5），不是融合（M1）**。与 `00_CONTEXT` §9.2 手工剪影残差探针闭环。

### 机制 C（对 M2）：时间相关 / 有色误差模型（批相关噪声文 + 经典增广）

- **数学形式**：测量噪声 AR(1) 或带宽-\(b\) 逆协方差；或状态增广偏差 \(b_t\)。  
- **接入点**：评测闭环误差自相关诊断（已有 §8）；训练侧用相关噪声课程（`PREV_NOISE_MODE` 已有 correlated 档）对齐测试。  
- **最小证伪**：估计闭环根旋误差的滞后-1 相关；若 \(\rho_{1}\approx 0\) 而增益消融仍无效 → 弱化 M2；若 \(\rho_{1}\) 高且「增广 \(b\) / 差分新息」改善 \(\ge 1.1\,\mathrm{mm}\)、而仅学对角 \(R\) 无效 → **支持 M2 优先于 M1**。

---

## 4. 限制与诚实声明

1. **场景鸿沟**：期刊证据来自 IMU 行人/车辆、合成 SS、立体事件刚体 VO——**无单目事件 MANO 闭环**。  
2. **KalmanNet 对有色偏差空白**：模型失配实验 ≠ 慢变系统偏差；不可把 TSP 结论直接写成「M2 已被 KalmanNet 否定」。  
3. **Split-KalmanNet venue 是 TVT 非 TSP**；作 KalmanNet 后续收录，不冒充 TSP。  
4. **RNIN-VIO / Wagstaff** 不在指定期刊清单；RNIN 因任务点名深读，Wagstaff 仅旁证。  
5. **时间相关批估计文正式 venue 本轮未用 Crossref 钉死**（标待核）；数字来自 arXiv HTML。  
6. **ESVO 跟踪目标**是非线性最小二乘 + 鲁棒核，**不是**学习 \(K\)；与 S37 可微读出不是同一接口。  
7. **AI-IMU 适配协方差可系统性偏离真实统计**——学到的是「滤波器好用的权重」，不是校准似然。  
8. 未打开 IEEE 付费 PDF 终稿页时，以 arXiv/HTML/作者页与 Crossref 书目为准；页码/细表以 arXiv 版本引用。  
9. 与 C12：会议侧 HTTYDF「无遮挡时异方差几乎无用」与本笔记 TLIO「学 \(R\) 关键」并存——**关键是测量质量是否随样本剧变**；S37 闭环稳态更像慢偏差（M2）而非遮挡式异方差。

---

## 5. 对 S37 根融合（M1 / M2 / M3）最直接的启示

1. **M3 优先于 M1**：【论文已有结论】TRO/RA-L/TSP 成功滤波一律有 \(\Delta y\) 或几何残差；【代码事实】S37 根头无 prev、无 \(d\)、无投影残差。【推断】在补新息之前谈「KalmanNet 式增益」接口不成立。  
2. **有新息之后，学 \(R\) / 自适应增益才有期刊级消融支持**：TLIO Fig. 8、AI-IMU seq.01（1.11% vs 无适配 1.94%）、Split 异质噪声。【实验事实】S37 已有 coverage；可作最便宜的 \(R\) 代理，但须先有可减残差。  
3. **M1 不能自动打败 M2**：时间相关测量模型文 + S37 同包不动点现象 →【推断】若闭环误差高度自相关，应先做相关/增广探针；仅学对角门可能落回 K0/K1/K2 常数增益权衡线。  
4. **事件跟踪期刊的「不确定度」是鲁棒权重与深度融合，不是端到端学 \(K\)**：ESVO Table III / Fig. 14；连续时间 VIO 是固定噪声 NLS。把 ESVO 当成「学习滤波」会误读。  
5. **对主文档对照表**：建议新增 TLIO、AI-IMU、Split-KalmanNet、时间相关测量噪声文，并在 KalmanNet 行注明「需显式新息；有色偏差未测」。

---

## 附：核心论文卡片（深读 11）

### A. TLIO — Liu / Engel — RA-L 2020
- **DOI / 链接**：10.1109/LRA.2020.3007421 · https://arxiv.org/abs/2007.01867  
- **问题**：IMU-only 紧耦合位移测量滤波。  
- **对应 M**：M1, M3, M8（窗相关）。  
- **关键差异**：有 \(d,\hat R\)；S37 无。  
- **为何参考**：固定协方差 vs 学习协方差的直接消融（Fig. 8）。  
- **读到**：全文 HTML。  
- **数字**：相对 RoNIN 漂移 −27%/−33%；相对 3d-ronin-mse 再 −27%/−31% yaw/位置漂移。

### B. AI-IMU — Brossard / Bonnabel — T-IV 2020
- **DOI**：10.1109/TIV.2020.2980758 · arXiv:1904.06064  
- **对应 M**：M1。  
- **数字**：平均 \(t_{rel}=1.10\%\)；无适配 1.94%（seq.01）。  
- **读到**：全文 HTML/HAL。

### C. KalmanNet — Revach / Eldar — TSP 2022（第一轮已读）
- **DOI**：10.1109/TSP.2022.3158588 · arXiv:2107.10043  
- **对应 M**：M1, M3。  
- **数字**：状态失配约 +3 dB vs 错配 KF（Fig. 6(a)）。  
- **读到**：全文 HTML（本轮核实）。

### D. Split-KalmanNet — Choi / Lee — TVT 2023
- **DOI**：10.1109/TVT.2023.3270353 · arXiv:2210.09636  
- **对应 M**：M1（异质 \(Q/R\) 失配）。  
- **读到**：全文 PDF。  
- **要点**：分裂学 \(\Sigma\) 与 \(S\)；异质噪声下稳过 KalmanNet（Fig. 3–4）。

### E. Latent-KalmanNet — Buchnik / Shlezinger — TSP 2024
- **DOI**：10.1109/TSP.2023.3344360 · arXiv:2304.07827  
- **对应 M**：M3（高维→潜空间再滤波）。  
- **读到**：全文 HTML。

### F. ESVO — Zhou / Shen — TRO 2021
- **DOI**：10.1109/TRO.2021.3062252 · arXiv:2007.15548  
- **对应 M**：M3（几何残差 + 不确定度剪枝）。  
- **数字**：Table III Student-\(t\) vs LS：2.76→2.15 cm（均值）。  
- **读到**：全文 HTML（本任务不确定度视角；几何 G11 已读）。

### G. Continuous-Time Event VIO — Mueggler / Scaramuzza — TRO 2018
- **DOI**：10.1109/TRO.2018.2858287 · arXiv:1702.07389  
- **对应 M**：M3（连续时间残差融合）。  
- **读到**：PDF 全文。  
- **要点**：B 样条轨迹 + 事件/IMU NLS；增益=优化权重，非学习 \(K\)。

### H. ESVO2 — Niu 等 — TRO 2025
- **链接**：arXiv:2410.09374  
- **对应 M**：M3。  
- **读到**：PDF（轻读权重：IMU 预积分协方差加权）。

### I. Time-Correlated Measurement Noise — arXiv:2303.06507
- **对应 M**：M2。  
- **读到**：全文 HTML；venue 待核。  
- **数字**：忽略相关（带宽 0）NEES/RMSE 最差；\(b=1\) 起明显改善。

### J. RNIN-VIO — Chen / Zhang — ISMAR/TVCG
- **链接**：https://zju3dv.github.io/rnin-vio/  
- **对应 M**：M1, M3。  
- **读到**：PDF。  
- **数字**：Table 5 Challenging02 ATE 0.743→0.520 m（BVIO→RNIN-VIO）。

### K. （旁证，不计入深读）Wagstaff 可微 KF VIO — AIM 2022
- 学习异方差测量进可微 EKF；EuRoC 视觉退化仍稳。结构支持 M1+M3，venue 非指定清单。
