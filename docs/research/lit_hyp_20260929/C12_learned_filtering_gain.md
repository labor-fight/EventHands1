# C12：学习型滤波——状态/新息相关增益、新息表示、有色测量误差

> 子代理 C12 · 2026-09-29 · 全部简体中文  
> 依据：`docs/research/lit_hyp_20260929/00_CONTEXT.md`；主文档 `docs/网络结构分析.md` §2 / 附录已列 KalmanNet、Selective Sensor Fusion，本笔记补结构相似深读与 M1↔M2 判别。  
> 纪律：未改仓库代码/配置/测试/`outputs/`；未训练；未用 GPU。

---

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29 |
| 检索式（WebSearch） | `Backpropagation through Kalman filter Haarnoja NeurIPS 2016`；`Recurrent Kalman Networks Becker ICML 2019`；`Deep Variational Bayes Filters ICLR 2017`；`How to Train Your Differentiable Filter Autonomous Robots 2021`；`Differentiable Particle Filters RSS 2018`；`KalmanNet Revach TSP 2022 ablation innovation`；`Selective Sensor Fusion Chen CVPR 2019 soft hard mask ablation`；`differentiable filter colored noise temporally correlated measurement bias`；`Bryson Henrikson colored measurement noise measurement differencing Schmidt consider Kalman`；`learned uncertainty pose fusion CVPR`；`Structured Inference Networks Krishnan AAAI 2017`；`PIXIE Feng moderator 3DV 2021` |
| 核实来源 | arXiv abs/html（ar5iv）、NeurIPS proceedings、PMLR、RSS proceedings、CVF open access、Springer / DOI 页、OpenReview（DVBF） |
| 筛选规则 | 只保留与 S37 的 **M1–M8** 结构相似者：显式/学习增益、新息、学习观测+协方差、有色/时相关测量误差、闭环或分布外失效。关键词命中但结构是「整段 VIO / 纯生成 SSM / 期刊 IMU 紧耦合」的不深读。 |
| 排除 / 转交 | TLIO、AI-IMU 及 TRO/RA-L/TSP 导航滤波 → **D17**（最多列名）。主文档已深读的 KalmanNet / Selective Sensor Fusion：**第一轮已读**，本任务只核实消融数字。主文档标 ◐ 的 HTTYDF 本轮补全文。 |
| 覆盖缺口 | (1) 顶会里几乎没有「姿态相关系统偏差 + 学习增益」的直接对照；(2) Schmidt–Kalman 原文未取全文（标「基础」简述）；(3) 无事件手部闭环上的学习增益反例。 |

深读篇数：**12**（含 2 篇「第一轮已读·本轮核实」+ 2 篇标「基础」）。另列 4 篇轻量候选不进深读。

---

## 1. 候选目录表

| # | 标题 | 作者（一作 / 末位） | venue · 年 | 链接 | 读到 | 对应 M | 入深读？ |
|---|---|---|---|---|---|---|---|
| 1 | Backprop KF: Learning Discriminative Deterministic State Estimators | Haarnoja / Abbeel | NeurIPS 2016 | https://arxiv.org/abs/1605.07148 | 全文 HTML | M1,M3,M7 | ✓ |
| 2 | Differentiable Particle Filters… | Jonschkowski / Brock | RSS 2018 | https://arxiv.org/abs/1805.11122 · DOI 10.15607/RSS.2018.XIV.001 | 全文 PDF 抽文 | M1,M7 | ✓ |
| 3 | Recurrent Kalman Networks… | Becker / Neumann | ICML 2019 | https://arxiv.org/abs/1905.07357 · PMLR v97 | 全文 PDF | M1,M3 | ✓ |
| 4 | Deep Variational Bayes Filters… | Karl / van der Smagt | ICLR 2017 | https://arxiv.org/abs/1605.06432 · OpenReview | 全文 PDF 抽文 | M3（弱） | ✓ 轻 |
| 5 | Structured Inference Networks…（含 Deep Kalman） | Krishnan / Sontag | AAAI 2017 | https://doi.org/10.1609/aaai.v31i1.10779 · arXiv:1511.05121 | 摘要+DKF PDF | M3（弱） | ✓ 轻 |
| 6 | How to Train Your Differentiable Filter | Kloss / Bohg | Autonomous Robots 2021 | https://arxiv.org/abs/2012.14313 · DOI 10.1007/s10514-021-09990-9 | 全文 HTML | M1,M2,M8 | ✓ |
| 7 | KalmanNet… | Revach / Eldar | IEEE TSP 2022 | https://arxiv.org/abs/2107.10043 · DOI 10.1109/TSP.2022.3158588 | 全文 HTML | M1,M3,M7 | ✓ 第一轮已读 |
| 8 | Selective Sensor Fusion for Neural VIO | Chen / Trigoni | CVPR 2019 | https://arxiv.org/abs/1903.01534 · CVF | 全文 PDF | M1,M7 | ✓ 第一轮已读 |
| 9 | Kalman Filtering with Gaussian Processes Measurement Noise | Kurtz / Lin | arXiv 2019（venue 待核） | https://arxiv.org/abs/1909.10582 | 全文 HTML | M2 | ✓ |
| 10 | Estimation using sampled data containing sequentially correlated noise（基础） | Bryson / Henrikson | J. Spacecraft & Rockets 1968 | DOI 10.2514/3.29327（二手引用核实） | 仅二手/摘要链 | M2 | ✓ 基础 |
| 11 | Collaborative Regression…（PIXIE） | Feng / Black 等 | 3DV 2021 | https://arxiv.org/abs/2105.05301 | HTML 摘要+章 | M1（弱，静态门） | 轻列 |
| 12 | （D17 列名）TLIO / AI-IMU 等 | — | TRO/RA-L | — | 不深读 | — | 转 D17 |

未入表但检索到、结构不相似：OTAKNet（2025 预印）、DEM 有色 NCM 估计（控制会议）、通用 VIO 置信加权期刊稿——不凑数。

---

## 2. 按 M1–M8：文献支持 / 反对 / 空白

### M1 常数增益融合

**【论文已有结论】** 在「测量质量随状态变化」或「模型失配」时，**状态/新息相关增益优于常数增益**——但几乎总以**显式新息或显式测量**为前提。

- **Backprop KF（NeurIPS 2016，全文）**：CNN 同时输出 \(z_t\) 与 \(R_t\)，与 KF 端到端训练。遮挡盘跟踪 Table 1：BKF RMS **0.0537** vs piecewise KF **更高且几乎完全依赖动力学**（因 \(R\) 不依赖观测而学得过大）vs LSTM(64) **0.1407**。【论文已有结论】
- **HTTYDF（AutoRob 2021，全文）**：Table I，30 distractors，常数 \(R\) vs 异方差 \(R\)——dEKF RMSE **16.2 → 8.8**，NLL **14.0 → 10.7**，\(\mathrm{corr}(R,\text{可见像素})=-0.78\)；常数 \(R\) 时学到 \(\sigma_{r_p}\approx 25.4\)、几乎忽略观测。【论文已有结论】
- **KalmanNet（TSP 2022，第一轮已读·本轮核实）**：用 RNN 学 \(K_t\)，输入含新息差分等；状态演化失配时相对 MB-KF 约 **+3 dB**（Fig. 6(a) 叙述）。无显式 \(Q,R\) 时仍可达近 MMSE。【论文已有结论】
- **Selective Sensor Fusion（CVPR 2019，第一轮已读·核实）**：soft/hard mask 相对 direct 拼接；EuRoC Vision Degradation：Direct rot **0.0696** → Soft **0.0533**（Table 3）。干净数据上增益常很小（KITTI Normal Soft 平移甚至略差）。【论文已有结论】
- **RKN（ICML 2019，全文）**：因子化潜空间上 Kalman 增益由不确定度决定；摆锤高噪声 Table 2：RKN log-lik **≈6.18–6.25** > LSTM/GRU **≈5.65–6.05**。【论文已有结论】

**【反对 / 上限】** HTTYDF **KITTI-10 Table IV**：无遮挡 VO 任务上，异方差观测噪声相对常数噪声**几乎不降 RMSE**，甚至异方差 \(R\)+常数 \(Q\) 更差（作者写可能过拟合）。→ **增益自适应只在「瞬时可观质量变化」时划算**；测量误差若是慢变系统偏差，学 \(R(x)\) 不够。

**【与 S37】** 【代码事实】`Δroot = F(事件;a(prev)) + G(prev)` 无样本相关门（`model/model.py` `_decode_active` / `prev_mlp` 残差相加）。【实验事实】同包迭代收敛点 ≈ 闭环（`00_CONTEXT` §4 §8）→ 跨包平均未发生。文献支持「若存在可用新息，状态相关增益有用」；**不支持**「在无新息的 F+G 上只加标量门就能跨包滤波」。

### M2 逐包误差是时间相关系统偏差（与 M1 竞争）

**【论文已有结论 · 基础】** 测量噪声时间相关时，标准白噪声 KF（含只调增益）**次优**；经典处理：

1. **状态增广**（把有色噪声状态并入）——Bryson & Henrikson, *J. Spacecraft & Rockets* 1968（DOI 10.2514/3.29327；本笔记经 Kurtz/Lin、Chang 等二手核实，**未读 1968 原文 PDF**）。  
2. **测量差分 / 时间差分**——同一脉络；Chang (2014) 等证明与增广在线性情形等价。  
3. **Schmidt–Kalman consider**：不确定偏差作「考虑参数」不完全估计（教材级；本轮未核全文）。

**【论文已有结论】** **Kurtz & Lin, arXiv:1909.10582（全文 HTML，venue 待核）**：视觉 SLAM/AprilTag 类测量噪声**显著自相关**；白噪声 KF 非最优；用 GP 测量噪声扩展滤波可改进。直接支持「感知前端输出的误差 ≈ 有色/相关」时，**只改增益不够，要改噪声时间模型**。

**【学习方法怎么处理】** HTTYDF 的异方差/相关协方差针对的是**瞬时**噪声结构（遮挡、接触动力学），**不是**慢变姿态偏差的增广。KalmanNet 用 RNN 隐式记二阶矩，**仍假设创新可写 \(\Delta y = y - h(\hat x^-)\)** 且噪声近似可被增益吸收；对「每步同一方向的系统偏差」无定理保证能使稳态低于单次测量误差。

**【自适应增益的上限 · 推断】** 若信念误差在数百 ms 内高度自相关且近似加性偏差 \(b_t\)，则最优更新需估计 \(b\)（增广 / consider）或差分掉 \(b\)；**仅学 \(K_t\) 的上界是「把偏差当过程噪声」或「把增益压到近 0」**——对应 S37 现象：干净 prev 仍注入 6–8°（`00_CONTEXT` §4 §9），常数缩小 \(\rho\) 只能换权衡线（旧协议 K0/K1/K2 线索）。**【推断】M2 比 M1 更贴合「同包不动点≈闭环」**；M1 机制在「瞬时可靠度变化」（coverage、遮挡）上仍可局部有用。

### M3 缺少观测减预测 / 几何条件

**【支持】** 所有有效学习滤波都显式构造新息或学习测量：

| 工作 | 新息 / 测量形式 |
|---|---|
| KalmanNet | \(\Delta y_t = y_t - \hat y_{t|t-1}\)；特征 F2 等 |
| Backprop KF / HTTYDF | 学习 \(z_t = n_s(D_t)\)，\(R_t\)；再进 KF |
| RKN | 编码器输出观测均值与不确定度，Kalman 更新 |
| DVBF / Structured Inference | 生成式 \(p(x|z)\)，非实时判别滤波 |

**【空白】** 无一篇顶会工作在「丢弃投影残差 / 深度 / \(r_\mathrm{prev}\)」的线性读出上证明可恢复绝对朝向。→ 支持主文档 D2，不支持「只加门不加新息」。

### M4 手指头缺父链 / 无逐关节门

**【空白】** 本专题文献几乎不涉及树状关节读出。Selective Fusion / PIXIE 的门是模态或部位专家置信，不是父坐标系下的关节门。不从滤波文献外推。

### M5 事件编码器时间感受野

**【空白】** 学习滤波默认已有紧凑观测向量；不讨论事件图 2–7 ms 感受野。

### M6 表述（delta + 课程噪声 → 均值收缩）

**【弱支持】** HTTYDF：仅 MSE 训练会**高估过程噪声、过度信任观测侧的相反错误**；NLL 才能学准噪声。→ 损失与噪声模型耦合决定有效增益。【空白】无直接「delta+课程噪声 = Wiener 收缩」的顶会论文（该结论来自仓库 §9.1）。

### M7 泛化

**【支持】** DPF（RSS 2018）：相对端到端学习，算法先验使定位更 **policy-agnostic**，误差率降约 **~80%**（摘要/正文声明）；LSTM 换策略泛化差。【支持】KalmanNet Table I：纯 RNN 在更长轨迹相对 MMSE 差 **>50 dB**；结构化增益可转移。【弱相关】Selective Fusion：退化输入下 graceful；经典 OKVIS/MSCKF 在强退化下 abrupt fail（Table 5 叙述）。

### M8 训练/测试 prev 误差分布不一致

**【弱支持】** HTTYDF：训练序列长度 \(k=1\) 时 NLL 爆炸（Fig. 3：dUKF 47.4±3.9）；需多步展开学噪声。→ 开环单步噪声模型 ≠ 闭环相关误差。【空白】无顶会直接对比「训练 i.i.d. 注入噪声 vs 测试自回归误差」。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A — 异方差观测：\(R(\text{可靠度})\) 门控测量支路（偏 M1）

- **数学**：\(\hat x = \hat x^- + K(\rho)\,(z - H\hat x^-)\)，或简化 \(\Delta = \sigma(g(\rho))\odot F + \sigma(h(\rho))\odot G\)，\(\rho\in\{\mathrm{coverage},\,|d|,\,|F|,\,|G|\}\)。  
- **接入点**【代码事实】：`model/model.py` 根头与 `prev_mlp` 相加处；可靠度来自 `semkine/routed_readout.py` 的 coverage / 若恢复被丢弃的 \(d\)。  
- **成本量级**：冻结主干只训门与根头（与主文档 D1 同阶，千步级热启动）。  
- **何时文献说优于常数**：遮挡/可见质量变化（HTTYDF Table I；BKF Table 1）。  
- **最小证伪**：主文档 Step 0 oracle 门网格；若上界 < 1.1 mm → **H1-gain 出局**【待实验验证的假设】。另：门使 G1↓ 且 G2 同比例↓ → 只是常数 \(\rho\)。

### 机制 B — 显式新息 + 学习增益（KalmanNet 形，偏 M1∩M3）

- **数学**：构造 \(y = \phi(\text{事件},\,\mathrm{prev})\)（如部件平均偏移、带号距离），\(\Delta y = y - y(\mathrm{prev})\)，\(K_t = \mathrm{GRU}(\Delta y,\,\Delta x)\)，\(\Delta\mathrm{root}=K_t\Delta y\)。  
- **接入点**：已有 `ROOT_INNOVATION` 钩子（`model/model.py` / `semkine/root_innovation.py`）；需保留 G 并**用新息条件门控**，而非第一步那种「新息加项 + 删根回拉」。  
- **成本**：机制门通过后再两种子全量。  
- **证伪**：无几何新息时学增益 ≈ 常数门；+B 不提高 G2 → 瓶颈在表示（P3/M5）。

### 机制 C — 有色偏差增广 / 测量差分（偏 M2，决定 M1 vs M2）

- **数学（基础）**：\(v_{t}=\psi v_{t-1}+\varepsilon_t\) 则增广 \([x;v]\)，或 \(\bar z_t = z_t-\psi z_{t-1}\) 白化后再 KF。学习对应：慢状态 \(b_t\) 或差分残差。  
- **接入点**：在根旋转上维护 \(b\)（EMA / 小维状态），更新 \(\mathrm{root}\leftarrow\mathrm{root}+K(\Delta y-b)\)；或同包迭代不动点与闭环差作为「可滤波性」探针（已有 §8）。  
- **成本**：零训练探针优先；增广需改状态契约，慎重。  
- **证伪 M2**：若白化/增广后稳态 **明显低于** 单包信念（≥1.1 mm 且双种子），则 M2「增益无用」被削弱；若不动点仍钉死闭环 → **支持 M2，M1 上限到顶**【待实验验证的假设】。

**推荐判决序**：先做机制 C 的零训练探针（M1 vs M2），再决定是否投入 A/B。

---

## 4. 限制与诚实声明

1. 无一篇深读论文是「单目事件 50 ms 手部 MANO 递推」；迁移均为结构类比。【推断】  
2. Bryson & Henrikson、Schmidt–Kalman：**未读原书/原文 PDF**，结论经 Kurtz/Lin（arXiv HTML）与综述链核实。  
3. Kurtz & Lin (1909.10582) **venue 待核**（预印本核实存在）。  
4. HTTYDF Table I 数字来自 ar5iv HTML 解析（dEKF 16.2/8.8 等）；与 Springer 正式排版若有出入以 DOI 版为准。  
5. KalmanNet「+3 dB」来自正文对 Fig. 6(a) 的叙述，Table 单元格未在本笔记逐格抄全。  
6. DPF「~80%」为作者摘要声明；未重跑实验。  
7. 机器人期刊（TLIO、AI-IMU）按任务归 **D17**，未深读。  
8. 主文档附录已有 KalmanNet / Selective Sensor Fusion；本笔记避免重复叙事，只补消融与有色噪声视角。

---

## 附录 A：核心论文条目（交付要求字段）

### A1. Backprop KF — Haarnoja / Abbeel — NeurIPS 2016
- 链接：https://arxiv.org/abs/1605.07148  
- 问题：图像观测下可微判别式状态估计。  
- S37：M1（学 \(R_t\)）、M3（学测量 \(z\)）、M7（结构优于 LSTM）。  
- 关键差异：有 EKF 动力学与显式 \(z,R\)；S37 无。  
- 为何参考：证明**观测条件协方差**相对常数 \(R\)/piecewise 的消融（Table 1；遮挡 Fig. 5）。  
- 读到：全文 HTML。  

### A2. Differentiable Particle Filters — Jonschkowski / Brock — RSS 2018
- 链接：https://arxiv.org/abs/1805.11122  
- 问题：可微粒子滤波端到端学运动/测量模型。  
- S37：M7（算法先验 vs LSTM 策略敏感）。  
- 关键：粒子+显式测量似然；非 MANO。  
- 读到：全文 PDF 抽文。数字：误差率降 ~80%（作者声明）。  

### A3. Recurrent Kalman Networks — Becker / Neumann — ICML 2019
- 链接：https://proceedings.mlr.press/v97/becker19a.html · arXiv:1905.07357  
- 问题：高维特征空间因子化 Kalman 更新。  
- S37：M1（不确定度门控）、M3。  
- Table 2：RKN log-lik 高于 LSTM/GRU。读到：全文 PDF。  

### A4. DVBF — Karl / van der Smagt — ICLR 2017
- 链接：https://arxiv.org/abs/1605.06432 · OpenReview HyTqHL5xg  
- 问题：无监督学潜 SSM。与 S37 判别跟踪结构弱相似（M3 生成式对照）。读到：全文 PDF 抽文。  

### A5. Structured Inference Networks / Deep Kalman — Krishnan / Sontag — AAAI 2017 / arXiv 2015
- 链接：https://doi.org/10.1609/aaai.v31i1.10779 · https://arxiv.org/abs/1511.05121  
- 结构化变分推断；非实时增益。读到：DKF PDF + AAAI 摘要页。  

### A6. How to Train Your Differentiable Filter — Kloss / Bohg — AutoRob 2021
- 链接：https://arxiv.org/abs/2012.14313 · DOI 10.1007/s10514-021-09990-9  
- 问题：可微滤波实现与噪声模型消融。  
- **关键消融**：Table I hetero vs const \(R\)（见 §2 M1）；KITTI 上 hetero \(R\) 未必降 RMSE（§2 M1 反对）。有色/相关噪声可学但难（正文 VI-E3）。  
- S37：M1/M2/M8。读到：全文 HTML。  

### A7. KalmanNet — Revach / Eldar — TSP 2022（第一轮已读）
- 链接：https://arxiv.org/abs/2107.10043  
- 核实：新息特征必要；失配 +3 dB；纯 RNN 长轨迹失效。读到：全文 HTML（本轮）。  

### A8. Selective Sensor Fusion — Chen / Trigoni — CVPR 2019（第一轮已读）
- 链接：CVF / https://arxiv.org/abs/1903.01534  
- 核实：Table 1–3 soft/hard vs direct；退化时有用、干净时边际。读到：全文 PDF（本轮）。  

### A9. KF with GP Measurement Noise — Kurtz / Lin — arXiv:1909.10582
- 链接：https://arxiv.org/abs/1909.10582  
- 问题：时间相关测量噪声下的最优线性滤波。S37：**M2**。读到：全文 HTML。venue 待核。  

### A10. Bryson & Henrikson 1968（基础）
- DOI 10.2514/3.29327；状态增广处理序相关测量噪声。读到：仅二手核实。  

---

## 附录 B：D17 转交（不深读）

- TLIO（IMU 位移学习 + 滤波）  
- AI-IMU 死Reckoning  
- 其他 TRO/RA-L 学习协方差 VIO  

请 D17 从「紧耦合导航滤波」视角覆盖；本笔记不引用其数字。
