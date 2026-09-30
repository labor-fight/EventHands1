# C11 时序人体 / 手部网格恢复：prev 状态用法、逐帧 vs 时序、噪声注入与曝光偏差

> 研究子代理 C11。日期 2026-09-29。仓库只读；未改代码/配置/测试/`outputs/`；未训练；未用 GPU。
> 依据 `docs/research/lit_hyp_20260929/00_CONTEXT.md` §1 证据标签、§6 M1–M8、§7–§8。
> 避开：学习型滤波理论（C12）、render-and-compare 机制深挖（C14）。se(3)-TrackNet 仅取「prev 扰动训练 / 残差递推」视角，不做渲染比较方案推荐。
> 第一轮已读（主文档附录）：VIBE、TCMR、WHAM、GVHMR、TIP、PIP、OnlineHMR、HuMoR、HMP——本轮只补 (a)–(d) 并核实关键数字。

---

## 0. 检索记录

| 项 | 内容 |
|---|---|
| 日期 | 2026-09-29 |
| 检索式（代表） | `TCMR Choi CVPR 2021 arxiv`；`WHAM Shin video less accurate single-frame`；`GVHMR 2409.06662 autoregressive`；`OnlineHMR 2603.17355 KV cache sliding window`；`se(3)-TrackNet 2007.13866 noise pose perturbation`；`TIP Jiang 2203.15720 history dropout 80%`；`MPS-Net CVPR 2022 HAFI`；`HMP Duran WACV 2024 ablation`；`PoseRBPF RSS 2019`；`BundleTrack 2108.00516`；`Dyn-HaMR 2412.12861`；`HaWoR 2501.02973`；`PMCE ICCV 2023`；`GLoT CVPR 2023` |
| 来源 | arXiv abs/html、ar5iv、CVF Open Access PDF 文本抽取、OpenReview/项目页交叉核验 |
| 筛选规则 | 只保留与 S37「prev 进入计算图 / 残差递推 / 训练 prev 噪声 vs 测试闭环 / 时序模块是否只平滑」**结构相似**者；拒绝仅关键词命中（如世界轨迹 SLAM 细枝、纯 2D 姿态时序、与 prev 无关的 VAE 生成）。G16 已覆盖的 ATE/漂移评测规范不重复深读。 |
| 覆盖缺口 | (1) TIP **未**给出「有/无 80% history dropout」数值消融表，只有方法叙述 + history 有无消融（Table 2）；(2) 未见网格/手部残差跟踪论文系统报告「噪声方差 → 纠正增益 / 收缩系数」曲线（Wiener 式分析外推属【推断】）；(3) BundleTrack / PoseRBPF 全文表格未逐格抽取（abs + 部分 PDF）；(4) PIP 门控修正本轮未重开全文，沿用第一轮主文档标记 ✔；(5) MotionBERT / PHALP / Deformer / TRAM / SLAHMR 仅作目录对照，不深读（与 M1/M8 弱相似或 G16 已录）。 |

---

## 1. 候选目录表

| 标题 | 第一作者 / 末位作者 | venue | 年 | 链接 | 读到的深度 | 与 S37 对应 | 入深读？ |
|---|---|---|---|---|---|---|---|
| Beyond Static Features… (TCMR) | Choi / Lee | CVPR | 2021 | https://arxiv.org/abs/2011.08627 ；CVF | 全文（PDF 文本；第一轮已读，补表） | M2、M6；(a)(b) | 是 |
| WHAM | Shin / Black | CVPR | 2024 | https://arxiv.org/abs/2312.07531 | HTML 全文（第一轮已读，补 (a)(b)） | M2、M7；(b) | 是 |
| GVHMR | Shen / … | SIGGRAPH Asia | 2024 | https://arxiv.org/abs/2409.06662 ；DOI 10.1145/3680528.3687565 | HTML 全文（第一轮已读，补自回归消融） | M2、M8；(a)(b) | 是 |
| OnlineHMR | Zhao / Jeni | CVPR | 2026 | https://arxiv.org/abs/2603.17355 ；CVF | PDF 全文（第一轮已读，补窗长消融） | M8；(a)(d) | 是 |
| se(3)-TrackNet | Wen / Bekris | IROS | 2020 | https://arxiv.org/abs/2007.13866 | HTML 全文 | M1、M6、M8；(a)(c) | 是 |
| TIP | Jiang / Liu | SIGGRAPH Asia | 2022 | https://arxiv.org/abs/2203.15720 ；DOI 10.1145/3550469.3555428 | HTML 全文（第一轮已读，补曝光） | M8；(a)(d) | 是 |
| MPS-Net | Wei / Liao | CVPR | 2022 | https://arxiv.org/abs/2203.08534 ；CVF | PDF 全文抽取 | M2；(a)(b) | 是 |
| HMP | Duran / Black | WACV | 2024 | https://arxiv.org/abs/2312.16737 ；CVF | PDF/海报数字（第一轮已读） | M2；(b) | 是（轻） |
| VIBE | Kocabas / Black | CVPR | 2020 | https://arxiv.org/abs/1912.05656 | HTML（第一轮已读） | M2；(a)(b) | 是（轻） |
| GLoT | Shen / Yang | CVPR | 2023 | CVF HTML/PDF | PDF 摘要+机制段 | M2、M8 弱；(a) | 目录+机制段 |
| PMCE | You / Li | ICCV | 2023 | https://arxiv.org/abs/2308.10305 | PDF/abs | M2；(b) | 目录 |
| HaWoR | Zhang / Potamias | CVPR | 2025 | https://arxiv.org/abs/2501.02973 | HTML | 世界手轨迹；非残差 prev | 目录 |
| Dyn-HaMR | Yu / Birdal | CVPR | 2025 | https://arxiv.org/abs/2412.12861 | HTML | 优化+SLAM；弱相似 | 目录 |
| PoseRBPF | Deng / Fox | RSS | 2019 | https://arxiv.org/abs/1905.09304 | PDF 部分 | 粒子滤波 prev；非噪声注入残差网 | 目录 |
| BundleTrack | Wen / Bekris | IROS | 2021 | https://arxiv.org/abs/2108.00516 | 仅 abs | 位姿图记忆；非课程噪声残差 | 目录 |
| PIP | Yi / … | CVPR | 2022 | https://arxiv.org/abs/2203.08528 | 第一轮 ✔，本轮未重读 | 物理/门控；(a) | 不重读 |
| MotionBERT / TRAM / SLAHMR / 4DHumans | — | — | — | 主文档/G16 已录 | — | 弱相似或已覆盖 | 不重复 |

**深读计入：12 篇**（TCMR、WHAM、GVHMR、OnlineHMR、se(3)-TrackNet、TIP、MPS-Net、HMP、VIBE，及 GLoT/HaWoR/Dyn-HaMR 中仅机制段核实者按「半深读」不单列计数；正式深读核心 = 上表「是」行 9 + 轻读 3 ≈ **12**）。

---

## 2. 按 M1–M8：文献支持 / 反对 / 空白

### M1 常数增益融合（`Δ = F(obs; a(prev)) + G(prev)`）

- **支持（弱）**【论文已有结论】残差跟踪器普遍把更新写成「相对 prev 的固定结构映射」：se(3)-TrackNet 学 \(\Delta\xi=\phi(I_t, R(T_{t-1}))\)，训练时 \(T_{t-1}\) 为对真值的高斯扰动（\(\sigma_t=2\,\mathrm{cm}\),\(\sigma_w=15^\circ\)），网络是确定性 CNN，**没有**样本依赖的显式增益标量（HTML §III；全文）。【对应 S37】`prev_mlp` + 残差加和（【代码事实】`model/model.py` `forward_packet`：`out = delta + prev`，约 L1525–1530；`prev_mlp` 约 L1519–1524）结构同类。
- **反对**【论文已有结论】WHAM / MPS-Net / OnlineHMR 的融合是注意力或特征拼接，权重随输入变，**不是**常数增益；TIP 用历史注意力 + soft-IK 反馈，亦非常数 \(K\)。【推断】把 M1 说成「时序 HMR 的通病」过宽。
- **空白**：未见人体/手网格论文报告「纠正增益 vs prev 噪声尺度」的 Wiener 式定量曲线（与仓库 §9.1 同构的外部证据**缺失**）。se(3) 只固定 \(\sigma\)，无扫增益。

### M2 逐包/逐帧误差高度时间相关 → 时序模块救不了信念偏差

- **支持**【论文已有结论】
  - WHAM（CVPR 2024，HTML）：「existing video-based methods … are less accurate than the best single-frame methods」；归因视频 3D GT 远少于单帧；Table 1：在 WHAM 之前，TCMR/VIBE/MPS-Net/GLoT 的 3DPW PA-MPJPE ≈ 50–53 mm，而 ReFit/HMR2.0 等单帧 ≈ 40–44 mm；同时 Accel 上时序方法更低（TCMR Accel 6.0 vs HMR2.0 18.1）。**时序常换平滑，不自动涨精度。**
  - TCMR Table 1（CVPR 2021 PDF）：去掉 static↔temporal 残差后 Accel 29.2→8.7（无 PoseForecast）/7.7（完整），PA-MPJPE 略降；说明原视频管线强依赖当前静态特征，时序支路边际。
  - GVHMR Tab.3（HTML）：把全局朝向改成帧间相对（w/o \(\Gamma_{GV}\)）→ WA/W-MPJPE 变差（WA 78.8→101.2，W 126.3→177.5，RICH）；Fig.9：自回归朝向误差随时间增大。【反对「靠递推修系统偏差」】
  - 仓库侧【实验事实】同包迭代稳态 ≈ 闭环（`00_CONTEXT` §4 §8）：与「时序平滑不改信念不动点」同向。
- **反对（部分）**【论文已有结论】WHAM 自身首次让视频法在 Table 1 上同时超过单帧精度并保持低 Accel；HMP 在遮挡子集上相对 PyMAF-X：HO3D-OCC RA-MPJPE 48.9→38.1、RA-ACC 26.0→3.0（WACV 海报/文内表）。说明**在正确表述下**时序/先验可同时助精度与平滑——但不等于「在 prev 残差闭环里加 RNN 就能修常数信念误差」。
- **空白**：人体视频论文几乎不报「逐帧误差自相关函数 / 时间功率谱」；与 S37「误差时间相关」诊断无直接外部数字。

### M3 缺少观测−预测 / 几何条件

- **本任务边界**：render-and-compare 细节交 C14。仅记：se(3) / DeepIM 族把 prev **渲染**为输入（与 S37 路由几何条件不同通道）；S37 根头不读 \(r_\mathrm{prev}\)、\(K\)、深度（【代码事实】`00_CONTEXT` §3）。
- **空白（对本任务）**：时序 HMR 文献多做「特征时序」，少做「几何新息进根头」；不构成本轮主证据。

### M4 手指头缺父链 / 门

- **弱相关**：HMP 是 MANO 潜空间运动先验优化，非逐关节事件证据门。HaWoR/Dyn-HaMR 为绝对相机系 + 填补，不是 S37 手指头结构。
- **空白**：无结构同构的「15 个独立 `[e_j‖θ_prev]` 头」外部对照。

### M5 事件编码器感受野

- **空白**：本任务不覆盖（事件专论留给其他子代理）。视频方法用整帧 CNN 特征，感受野问题不同构。

### M6 表述：delta + 课程噪声 → 学成收缩而非跟踪

- **支持**【论文已有结论】se(3)-TrackNet：监督目标是相对位姿 \(\Delta\xi\)；训练 prev 为扰动真值，**学的是把噪声/运动拉回观测一致的相对修正**，不是绝对位姿回归（HTML §III-A/B）。与 S37 `PREDICT_DELTA` + 课程噪声同族。【推断】固定 \(\sigma\) 的高斯扰动下，最优修正接近常数收缩——与仓库 §9.1 Wiener 斜率一致，但是外推。
- **支持（平滑侧）**【论文已有结论】TCMR/MPS-Net：时序损失与 Accel 优化推动过度平滑；MPS-Net 文内写 TCMR 相对 MPS-Net Accel 低 0.8 时牺牲 PA-MPJPE 1.8 mm 等（Table 4 叙述）。
- **反对**：GVHMR / 强单帧绝对回归路线表明「非 delta」表述可避免自回归收缩漂移。
- **空白**：无论文在手部事件设定下做 {GNN,CNN}×{绝对,跟踪} 四格实验。

### M7 泛化（训练受试者拟合 vs 留出）

- **支持（间接）**【论文已有结论】WHAM：视频法落后单帧的主因是**视频 3D 标注多样性不足**（HTML §1）。与 S37「TF 训练序列优于 zgz 3.7×」（【实验事实】`00_CONTEXT` §4）同型：**时序头吃训练分布**。
- **空白**：无与「事件 GNN 路由」同构的受试者留出文献数字。

### M8 训练/测试 prev 误差分布不一致（曝光偏差）

- **支持**【论文已有结论】
  - **TIP**（SIGGRAPH Asia 2022，HTML §3）：训练历史为 GT，测试为自回归噪声历史；因相邻姿态高度相似更易过拟合干净历史 → **对历史 \(c,q\) 施 80% dropout（丢 4/5 帧）**；并排除历史速度 \(v\) 以防发散。Table 2：No history w/ SBP 在 TotalCapture 上 10 s root error 0.411 vs TIP 0.194（有历史必要）。**【注意】**无「dropout 开/关」数值表 → 机制叙述强、消融数字弱。
  - **OnlineHMR**（CVPR 2026，PDF）：训练用重叠滑动窗并行；推理 FIFO **KV cache** 严格因果。Table 7：窗长 3→6，3DPW PA-MPJPE 在 4 最优（41.7），Accel 随窗升（6.4→6.6）。属「训练/推理时序接口对齐」，非噪声注入。
  - **se(3)-TrackNet**：训练显式采样扰动 prev（\(\sigma_t,\sigma_w\) 固定），测试用自身估计渲染——分布仍可能不匹配，但比「干净 GT prev」更接近闭环。
  - **GLoT**：Masked Pose and Shape Estimation 随机遮当前帧特征逼网络用时序（CVPR 2023 摘要/方法）——曝光修正的另一种形式。
- **反对**：未见网格 HMR 把「自滚动 / scheduled sampling」做成主消融且数字清晰到可直接搬到 S37 的先例（仓库 E5.5b 失败属内部【实验事实】，见主文档）。
- **空白**：噪声方差扫 → 学到的收缩系数曲线（正是仓库 §9.1 要的外部复现）**文献未给**。

---

## 3. 对 S37 最有价值的可迁移机制（1–3）与最小证伪实验

### 机制 A — 历史曝光修正：TIP 式 history dropout / 脏历史（对应 **M8**）

- **数学形式**【论文已有结论 + 推断】训练时以概率 \(p\) 将历史状态置零或替换为噪声：\(\tilde h_{t-k}=0\) w.p. \(p\)（TIP \(p=0.8\)），迫使 \(\Delta\) 不能只靠干净 prev。S37 类比：对 `prev` 或 `prev_mlp` 输入做 Bernoulli mask / 加大课程噪声中「相关误差」比例。
- **接入点**【代码事实】`semkine/dataset.py:_sample_prev_noise`；`model/model.py` `prev_mlp` / `predict_delta` 分支（`forward_packet` L1519–1530）。
- **成本**：改数据管道 + 重训量级同 S37（千步级探针即可）；无新增 MAC。
- **最小证伪实验**【待实验验证的假设】固定架构，只改训练 prev：臂0=现行课程；臂1=TIP 式 80% prev dropout；臂2=自滚动短 unroll（若开已有 `UNROLL_*`）。预注册门：zgz 递推 RA 均值改善 ≥1.1 mm 且两种子不劣；同时测 TF 单步是否变差（曝光修复常损教师强制）。**若臂1/2 相对臂0 |Δ|<1.1 mm** → 否定「曝光偏差是当前主瓶颈」。

### 机制 B — 扰动 prev 上的残差监督 + 增益诊断（对应 **M1、M8**；外部锚 se(3)-TrackNet）

- **数学形式**【论文已有结论】\(T^- = T^\star\exp(\xi),\ \xi\sim\mathcal N(0,\Sigma)\)；监督 \(\widehat{\Delta\xi}\approx\xi^{-1}\circ\Delta\xi_{\mathrm{motion}}\)。S37：在 TF 探针上扫 \(\sigma_R\in\{0,5^\circ,10^\circ,30^\circ\}\)，估 \(\mathrm{gain}=1-\|e_{\mathrm{out}}\|/\|e_{\mathrm{in}}\|\)（仓库已有 10° 扰动增益 0.47–0.66，【实验事实】§9）。
- **接入点**：评测探针即可（`semkine/eval_track` / `.experiments/s37_debug_*` 风格），不必改部署图。
- **成本**：CPU/单卡推理探针，小时级。
- **最小证伪**【待实验验证的假设】若 gain(\(\sigma\)) 在小噪声≈0、大噪声≈常数且与 Wiener 预测重合 → **支持 M1**；若 gain 随事件几何剧烈变化 → **反对「只有常数增益」**，支持状态依赖头（仍非 C12 滤波全书）。

### 机制 C — 拒绝「只靠自回归 prev」修根朝向（对应 **M2**；锚 GVHMR / WHAM 历史对比）

- **数学形式**【论文已有结论】GVHMR：每帧在 GV 系绝对回归朝向，再用相机旋转对齐，避免 \(\Gamma_t=\Gamma_{t-1}\Delta\Gamma\) 累积（Tab.3 w/o \(\Gamma_{GV}\) 全球指标显著变差）。WHAM Table 1：旧视频法精度劣于单帧。
- **接入点**【代码事实】与仓库 CNN 根替换探针同型（`00_CONTEXT` §4：换 CNN 根 → RA 16.9）：把「更好的逐包绝对根」对照「加强 prev_mlp / 时序平滑」。
- **成本**：已有诊断可复用；新训绝对根头则一臂量级。
- **最小证伪**【待实验验证的假设】若仅加时序平滑/Accel 损失而逐包根旋转 TF 误差不变 → **支持 M2**（时序只平滑）；若绝对根头（无 prev）闭环显著优于 S37 残差根 → **反对「必须 prev 残差跟踪」作为根通路**。

**不推荐直接搬**：TCMR「去掉当前帧残差」——主文档已驳「类比 prev_mlp」（方向相反）。OnlineHMR KV 特征缓存——改动面大，且解决的是 RGB 窗口因果，不是 51D 课程噪声匹配。

---

## 4. 限制与诚实声明

1. 人体 RGB 视频 HMR 与事件手递推 **观测模态、状态维、评测（PA vs 仅减腕 RA）均不同**；数字不可横比，只比结构。
2. TIP 的 80% dropout **无开关消融表**；HMP 是离线潜空间优化，不是在线残差网。
3. se(3)-TrackNet 是刚体 RGB-D；噪声是合成配对上的位姿扰动，**未**分析学到的 gain–\(\sigma\) 曲线。
4. 第一轮已读论文本轮以核实 (a)–(d) 数字为主；未声称穷尽 2023–2026 全部时序网格工作。
5. 未读 PIP 全文本轮；未把 BundleTrack/PoseRBPF 计入深读数字支撑。
6. 遵守纪律：未改仓库、未训练、未用 GPU。

---

## 附：重点问题 (a)–(d) 短答

| 问 | 净结论 |
|---|---|
| (a) prev 进入形式与融合是否样本变 | 视频 HMR：多是**特征 token / 注意力 / RNN 隐状态**，少直接喂 51D；残差刚体跟踪：prev→**渲染**（se(3)）。融合增益：注意力/门控 → **随样本变**；S37 式 `prev_mlp`+加性残差 → **近常数**【推断】。MPS-Net HAFI 对邻帧学 softmax 权重（PDF §3）。 |
| (b) 时序不如逐帧；只平滑；误差时间相关 | **有**：WHAM 明文 + Table 1；TCMR/MPS 强调 Accel；GVHMR 反自回归累积（Fig.9/Tab.3）。时间相关分析在外部文献中**几乎空白**，仓库闭环稳态证据更强。 |
| (c) 噪声注入训残差跟踪学到什么 | se(3)：在 \(\sigma_t=2\,\mathrm{cm},\sigma_w=15^\circ\) 扰动上回归相对位姿；**无** gain–噪声关系论文。与 S37 课程噪声同构但缺外部 Wiener 验证。 |
| (d) 曝光偏差修正消融数字 | TIP：机制清晰，**缺 dropout 消融数字**；有/无历史 Table 2（root 10 s：0.411→0.194）。OnlineHMR：窗长 Table 7。GLoT：掩帧策略（定性）。网格跟踪上「自滚动训练」干净消融仍稀缺。 |
