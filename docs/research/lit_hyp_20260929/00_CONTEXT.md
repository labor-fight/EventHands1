# 第二轮结构问题调研：20 个子代理共用事实摘要（2026-09-29 晚）

> 所有子代理**先读完本文件**再工作。本文件是事实摘要；权威来源是文中列出的仓库文件，引用前请自行打开核对。
> 仓库根目录：`/data1/lyq/code/mesh/EventHands1`。Python：`/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python`。
> 主文档（本轮要补充的对象）：`docs/网络结构分析.md`（原名 `S37_LITERATURE_HYPOTHESES_20260929.md`）（第一轮：P1–P4、约 75 篇文献、H1–H3）。

## ⚠ 并行会话警告（21:20 补充，必读）

- 同一工作区还有**另外两个 Cursor 对话在同时工作**，与本轮调研无关：
  - 一个在 20:4x 按用户指令**删除了 9 份旧文档**：`docs/experiment_history.md`、`PLAN_SELECTION_VERDICT_20260825.md`、`ASYNC_SPARSE_SOTA_MASTER_VERDICT_20260826.md`、
    `debug_e55b_unroll_20260825.md`、`EVENT_GNN_SURVEY_20260829.md`、`docs/semkine/*.md`（4 份），并改写了账本与 S27 文档里的引用。
    这些文件仍在 git HEAD，**只读**方式：`git show HEAD:docs/<文件名>`；B8 已把副本导出到 `.experiments/lit_hyp_20260929/B8/extracted/`，可直接读。
    今天 19:56–20:32 的 6 份 `S37_XYZ_*` 文档已被合并为 `docs/S37_XYZ_CANDIDATE_20260929.md` 与 `docs/S37_XYZ_SCREEN_PREREG_20260929.md`。
  - 另一个正在开发 XYZ 候选：`semkine/xyz_mesh.py`、`configs/semkine/s37_xyz_c{0,1}_s3407.yaml`，并在 21:08 修改了 `model/model.py`（新增默认关闭的 `xyz_mesh` 分支，文件 1418 → 1556 行）。
- 规则：**不要读写、运行或评价**这些 XYZ 新文件；不要恢复被删文档到工作树；不要 `git checkout / restore / stash / reset`。
- 行号漂移：本文件 §3 的 `model/model.py` 行号是 1418 行版本的。当前版本的函数位置：`root_head`（routed 分支）约 863 行、`prev_mlp` 约 891 行、
  `_decode_active` 1155、`_project_prev` 1378、`_route_nodes` 1388、`forward_packet` 1422。引用时**以函数名为主**，行号以你读取时的当前文件为准。
  `semkine/routed_readout.py`、`event_gnn.py`、`eval_track.py`、`dataset.py`、`tools/make_s36_row.py` 自 09-28 未变，§3 的行号仍有效。

## 0. 本轮要回答什么

针对 `s37_routed`（当前臂）已经由代码确认的结构问题，结合**仓库已有实验**与**已发表文献**，判断真正的性能瓶颈，
并设计能够**证伪**假设的最小实验。不是刷指标、不是推荐新大模型、不是按论文罗列。
每个结论必须形成闭环：代码结构 → 实验现象 → 可能机制 → 文献是否支持 → 受控实验 → 什么结果能证明/否定。

## 1. 证据标签（强制）

每条陈述必须带一个标签，禁止把推断写成事实：

- 【代码事实】附 `file:line` 或函数名，自己打开核对过
- 【实验事实】附仓库文档 § 或产物路径（`outputs/semkine/*.json`、`.experiments/*/*.json`、`logs/*`）
- 【论文已有结论】附论文标题 + venue + 年份 + 链接，且注明你读到的是全文 / HTML / 仅摘要
- 【推断】你的推理
- 【待实验验证的假设】

## 2. 协议与指标口径（读数前必须知道）

- 单目事件相机（240×180）3D 手部跟踪。状态 = MANO 51 维 `[0:3]` 平移（米，相机系）、`[3:6]` global_orient（轴角）、`[6:51]` 15 个局部关节轴角。
- 训练：9 受试者 72 条序列（`splits_semkine.json`）。验证 = 选点 = 上报：留出受试者 zgz 的两条序列 `zgz_global`（1 段 69 s）+ `zgz_local`（90 段，平均 0.7 s，每段从 GT 起步），共 2590 帧，50 ms 一步。
  **选点集与上报集重合**，数字含选点增益。两种子 3407 / 3408。
- 四个精度列都是 RA：只减腕点、**不对齐旋转**（`semkine/eval_track.py:127-128`），全局旋转误差全额计入。
  "local / global" 指 zgz_local 与 zgz_global 两条序列，不是坐标系（`tools/make_s36_row.py:54`）。
- 统计边界（`docs/FAILURE_AND_CLEANUP_LEDGER.md` §1.2）：采纳需两种子均值改善 ≥ 1.1 mm 且逐种子不劣；|Δ| < 1.1 判打平。
  S36 两种子差 3.5 mm；S37 在 zgz_local 上两种子差约 6 mm（23.33 / 29.49）；网格中位数比选中点高 2–4 mm。
- 报告格式（`AGENTS.md`）：结果只用 `tools/report_table.py` 生成的表；没有主行（`outputs/semkine/<run>_main_row.json`）的臂不能作为结果报告。

## 3. S37 计算图（代码事实，已逐行确认；参数总数 733,830 = 主行 `params_total`）

| 模块 | 位置 | 输入 → 输出 | 要点 |
|---|---|---|---|
| 事件 token | `semkine/encoder.py:153-171` | (ΣN,5) → (ΣN,7) | `[x/W, y/H, 2p−1, t/Δt, log1p(Δt_ie/1µs), SAE_same/Δt, SAE_opp/Δt]`，在**全部**事件上计算后才抽样 |
| 抽样 | `semkine/event_gnn.py:127-146` | → (B,2048,7)+mask | 等步长抽样，至多 2048 节点 |
| 建边 | `semkine/event_gnn.py:149-176` | → idx (B,2048,8), dp (B,2048,8,3) | 只在**时间序前 32 个节点**里取 8 近邻（因果）；距离在 `(x/W, y/H, t_norm·t_scale)` 中算，`t_scale=1`：一个包长 = 一个画面宽 |
| embed + EdgeConv×3 | `semkine/event_gnn.py:57-83, 220-222` | h (B,2048,128) | `h ← h + mean_j ReLU(W[h_j−h_i; dp])`，无下采样、无层级、无全局回传；主干约 51.7K 参数却占 99.7% MACs |
| 全局读出 proj | `semkine/event_gnn.py:226-233` | mean‖max 256 → 512 → 512 | 394K 参数（54%），**只喂根头** |
| prev 几何 | `model/model.py:1242-1250` | FK + 投影（渲染内参 K×0.375），`no_grad` | |
| 路由 | `semkine/routed_readout.py:40-90`；调用 `model/model.py:1365` | a (B,2048,16) | 前表面最近顶点 ≤16 px 时取其 LBS 行，否则 0；**距离 d 与顶点 id 被丢弃** |
| 证据池化 | `semkine/routed_readout.py:93-118` | e (B,16,257) | `[LBS 加权均值 128 ‖ 硬分派 max 128 ‖ coverage 1]`；`coverage_j = Σ_i a_ij / N_live` |
| 根头 | `model/model.py:825, 1133-1141` | `Linear(4624 → 6)`，输入 `[feat 512 ‖ e_0..e_15]` | **单层线性，不读 prev 的任何分量，不读 K、深度** |
| 手指头 ×15 | `model/model.py:830-834, 1146-1153` | `[e_{k+1} 257 ‖ θ_k^prev 3] → 64 → 3` | 关节 k+1 局部轴角的唯一事件通路；契约测试钉死对其他证据梯度为零（`tests/test_s37_routed_readout.py:170-186`） |
| prev_mlp | `model/model.py:852-860, 1380-1386` | 51 → 64 → 51，加到 Δ | 看不到事件，末层零初始化 |
| 门 + 残差 | `model/model.py:1387-1392` | out = prev + Δ | 只在整包无事件时置零（没有逐关节门；对照 mesh_query 臂 `model/model.py:1270-1283` 有逐关节门）；**轴角直接相加** |
| MANO | `model/pose_repr.py:81-106`、`model/mano_layer.py:100-169` | 51 + betas(GT) → 778 顶点、21 关节 | 只在评测与路由里运行 |
| 损失 | `model/model.py:177-210` | log10[(450·MSE₄₅ + 60·MSE_R + 30000·MSE_t)/51] | 参数空间，不经 MANO |

- 训练样本（`semkine/dataset.py:330-387`）：窗口 W ~ 对数均匀 [30, 300] ms；target = GT(end)；prev = GT(start) + 课程噪声（`dataset.py:221-251`：
  50% 小噪声 5 mm / 0.05 rad / 0.05 rad，30% 大噪声 50 mm / 0.3 / 0.3，20% 相关/定向）。
- 增强（仅训练）：domrand roll 同步改事件、R_g、t；scale/shift **只改 K**（`semkine/domrand.py:193-195`）；keep 采样 0.25–1.0；热像素；
  全局极性翻转 + 逐像素随机极性交换（`dataset.py:358-362`）。**逐帧 CNN 的配置 `configs/semkine/s1_abs_domrand.yaml` 同样开了这两项极性增强**，不构成 CNN 与 S37 的差异。
- 评测（`semkine/eval_track.py:131-264`）：W = 50 ms；每段只在开头用一次 GT + 小噪声，之后 prev ← 上一步输出。
- `UNROLL_*`、`GAIN_REG_W` 有代码（`model/model.py:231-275` 附近），本配置未开。
- 每包事件数（zgz 实测）：zgz_global 中位 29,974 / 50 ms（91% 的包超过 2048，中位只有 7% 成为节点）；zgz_local 中位 660（p10 104），全部保留；训练 local 中位约 3000。
  按均匀事件率估算，前 32 个候选节点只跨 0.78 ms（global）/ 2.4 ms（local），三跳上界约 2.3 / 7.3 ms。
- 逐帧 CNN（`configs/semkine/s1_abs_domrand.yaml`）：ResNet18 看 LNES（`INPUT_MODE: legacy_lnes`），`PREDICT_DELTA: false`、不读 prev，绝对 51 维，同样的域随机化与极性增强。

## 4. 实验事实总表（全部在 zgz 协议下，除非标"旧协议"）

主行（可作为结果引用）：

| 臂 | 递推 RA 两种子均值（3407 / 3408） | 其它 | 来源 |
|---|---|---|---|
| EventHands-PCA6 baseline | —（MPJPE local 30 / global 10.99） | PCA6、`eval_abs` 协议、对 PCA6 投影 GT 计分，**与 S37 线不同协议** | `AGENTS.md`、`S37_ROTW_CNNROOT_PREREG.md` §5B 注 |
| S36 EventGNN（渲染 prev） | 23.17（21.42 / 24.92） | 7.04 ms, 0.827 G, 0.95 M | `S37_ROUTED_READOUT_PREREG.md` §2/§6 |
| **S37 routed（当前臂）** | **20.74（19.23 / 22.26）**；local 26.41 / global 15.82 | 7.72 ms, 0.827 G, 0.73 M；证据置零 → 28.29（+7.6） | 同上 §6 |
| S37 fkgraph | 22.10（22.64 / 21.56），abs −11.6 | 5.38 ms, 0.084 G, 0.23 M；观测置零 → 101；流速项无用 | `S37_FKGRAPH_PREREG.md` §6–§7 |
| S37 meshgraph | 24.89（20.48 / 29.30） | 不稳定全在根旋转（网格上 RA 与旋转相关 0.96 / 0.89，与手指 ≈ 0） | `S37_MESHGRAPH_PREREG.md` §6–§7 |
| S37 rootinnov 第一步 | 22.94（23.01 / 22.86） | 冻结主干 + 新息头 + 删 prev_mlp 根 6 行 + 1500 步热启动；G1 5.72 / 5.28°（门 ≤ 3）、G2 0.26 / 0.36（门 ≥ 0.5） | `S37_ROOT_INNOVATION_PREREG.md` §5–§6 |
| S37 rotw10（LAMBDA_R ×10） | 20.74（21.10 / 20.38）打平 | 根旋转不降；×30 单种子 19.10（诊断） | `S37_ROTW_CNNROOT_PREREG.md` §5A |
| S37 meshq | 有 checkpoint（`outputs/semkine/s37_meshq_s340{7,8}/`），**无主行**，prereg §6–§7 未填 | 设计含逐关节硬门、无 prev_mlp | `S37_MESHQ_PREREG.md` |

诊断（不是臂，不进结果表，只作机制证据）：

- **逐帧绝对 CNN**（`s37diag_cnnabs`，11.18 M 参数）：13.56（13.31 / 13.80），global 9.89 / 10.45，local 17.25 / 17.66；根旋转 8.1–9.3°；
  仅手指 RA global 4.4–4.9 / local 10.9–11.6（S37 为 6.0–8.8 / 16.8–17.2）。
  把 S37 的全局旋转每步换成 CNN 的：**16.93**；根整体换：16.79。剩余差距全在手指（`S37_ROTW_CNNROOT_PREREG.md` §5B）。
- `track_render51_dr_*`（CNN + 渲染 prev + 域随机化）：11.9–13.3，账本称"同协议，主表已有"（`FAILURE_AND_CLEANUP_LEDGER.md` 09-06 条），**出处与协议待核**。
- S37 探针（`S37_ROUTED_READOUT_PREREG.md` §6）：TF 单步 9.78 / 闭环 20.71，"放大率" 2.12（S36 2.58）；oracle 路由（GT prev 路由）→ 23.18（+2.5）；
  路由召回/纯度 @GT 0.95/1.00，@小噪声 0.95/0.76，@大噪声 0.67/**0.14**，@闭环 0.95/0.54；根头自身响应增益 平移 0.54、旋转 0.04/0.06/0.13；
  姿态更新比 0.98、方向余弦 0.26（S36 0.96/0.25）；TF 训练序列 → zgz 差 1.79（S36 0.48，3.7 倍）。
- §8 同包迭代（`.experiments/s37_reanalysis_20260928/`）：原地不动单步 2.43（global）/ 4.41（local）；保持段首 26.6 / 28.2；
  闭环 15.66 / 15.97（global）、23.33 / 29.49（local）；TF 单步 6.6 / 8.3、11.8 / 13.1；从 GT 迭代 8 次 13.4 / 16.1、19.1 / 27.6；
  从远起点 1→8 次：global 20.5→14.5、19.0→16.8，local 27.8→26.8、29.5→36.1；闭环每步关节位移是真实的 1.5×（global）/ 3×（local）；69 s 内误差不增长。
  §7 的"放大率 = 误差传播增益"读法已被 §8 修正为"闭环稳态 ≈ 逐包信念误差"。
- §9 分解（`.experiments/s37_debug_20260928/`）：每步根旋转换 GT → 12.4（合并；global 9.58 / 6.15，local 17.45 / 17.71）；手指换 GT → 15.1；
  根旋转冻结在段首：local 22.37 / 23.85（**优于模型**）；手指冻结在段首：global 13.43 / 15.21（**优于模型**）。
  TF 根旋转 6.0–7.9°，真实 50 ms 转动 1.6°（global）/ 3.0°（local）；更新与所需方向余弦 0.05–0.11；**TF 误差与运动量、事件数无关**（固定噪声底）。
  F（根头）与 G（prev_mlp）各转 15–25°、方向与所需几乎正交、相互抵消；去掉 prev_mlp 闭环发散（81–101）。
  10° 扰动的纠正增益 0.47–0.66，其中 prev_mlp 单独 0.27–0.63，事件只 0.01–0.20。
  损失份额：平移 45%、手指 39–40%、根旋转 15.5%；RA 灵敏度根旋转 55–95 mm/rad、手指每维 3.0 mm/rad。
  已排除：可观测性（面内 5.5–7.0° 与面外同量级）、朝向分布外、轴角 π 跳变、受试者迁移。
- §9.1：prev_mlp 根斜率与课程噪声下的 Wiener 收缩系数吻合（预测 −0.34 / −0.30 / −0.24，实测 −0.30 / −0.24 / −0.25 与 −0.33 / −0.32 / −0.31）。
  分工况最优增益：小噪声 0.02、大噪声 0.41、干净 0。
- §9.2 单包信息探针（ridge / MLP，°；保持 9.21）：S37 根输入 z（4624 维）8.02 / 8.44–8.62；z + r_prev 7.53–7.60；**完美路由 z = 9.20（R² 0）**；
  119 维手工 2D 剪影残差 7.69 / 7.08；×3D 杠杆臂 + r_prev 7.09 / 6.96；+ oracle 事件深度 5.30 / 4.51；干净样本上所有解码器仍注入 3.9–5.0°。
- §9.3 窗口 50→300 ms：两种子均值 20.34–20.90，根旋转 10.5–12.4° 不变；global 仅手指 RA 改善 1–1.8 mm。
- XYZ 支路 DEBUG（`S37_XYZ_BRANCH_DEBUG_PREREG_20260929.md`）：FK 雅可比局部数值秩 51；轴角 2π 等价（几何相同、加性残差差 2π）；
  多头可块稀疏精确合并；prev_mlp 不是死支路；mean 不能重建 max。
- 旧协议（5 受试者线，**不可与 zgz 数字比较**，只作机制线索）：域随机化 19.26 → 12.77；常数 δ-trust 0.5：19.26 → 18.73；
  两遍 render-and-compare 爆炸；KEG 学习路由回路增益 0.91；E5.5b 恒定 unroll 坍缩回绝对回归（"约 40 mm 档"，待核）；
  K0/K1/K2 课程臂"落在同一条常数增益权衡线上"（09-28 已删记录，结论见 `docs/research/dir12_20260928/00_CONTEXT.md` §4）。

## 5. 文献覆盖现状（不要重复做已做过的事）

- 第一轮主文档附录已列约 75 篇（事件手/人体、KalmanNet、Selective Sensor Fusion、PIXIE、PyMAF、Oberweger、se(3)-TrackNet、DeepIM、CAPTRA、CLIFF、
  HKMR、MAED/KTD、HybrIK(-X)、NIKI、KITRO、Hamba、Hand4Whole、PARE、HOPE-Net、HaMeR、IntagHand、METRO、Mesh Graphormer、WHAM、GVHMR、VIBE、TCMR、
  TIP、PIP、OnlineHMR、HuMoR、HMP、DAgger、Scheduled Sampling、AEGNN、RVT、SSM for events 等）。核验标记与已发现的读数冲突见主文档 §0–§1。
- `docs/research/dir12_20260928/` 另有 09-28 的深读笔记（繁体）：G01 单目手网格（约 27 篇）、G06/G07 事件双目与深度、G11/G12 VO/SLAM、
  G14 MANO 运动学与可见性、G16 局部/全局误差与漂移、G17 AEGNN 与增量图、G18 EventNet 与递归点集。
- **你的任务不是扩充数量**，而是：(1) 为具体结构问题找到结构上真正相似的工作；(2) 核实被用来支撑关键论断的数字；(3) 补齐期刊（TPAMI/IJCV/TIP/TMM/TCSVT/TNNLS/PR/TRO/RA-L 等）缺口；
  (4) 找反例（证明某机制不必要或有害的论文）。已读过的论文只在需要核实关键数字时重读。

## 6. 本轮要检验的候选机制（中性列出，允许你推翻）

- **M1 常数增益融合**：`Δroot = F(事件; a(prev)) + G(prev)`，只能表示常数增益。
- **M2 逐包误差是时间相关的系统偏差**（与 M1 竞争）：若逐包信念误差在数百 ms 内高度相关，任何增益/滤波都救不了，只有更好的逐包估计有用。
  线索：同包迭代收敛点 ≈ 闭环误差；K0/K1/K2 在同一权衡线上；TF 误差与运动量、事件数无关；CNN 根替换 −3.8 mm。
- **M3 缺少观测减预测 / 几何条件**：d、v*、g 被丢弃；根头看不到 r_prev、K、深度；完美路由下 z 对 prev 误差无信息。
- **M4 手指头缺父链坐标系 / 证据混合 / 无逐关节门**。
- **M5 事件编码器的时间感受野与层级**：节点只看 2–7 ms、3 跳、一阶池化 → 无法形成部件形状 → 绝对朝向不可从特征中恢复；窗口加长不变是因为感受野没变。
- **M6 表述（formulation）本身**：delta + 课程噪声 + prev_mlp 使网络学成"往均值收缩"，而不是绝对估计或跟踪；2×2 {编码器 GNN / CNN} × {绝对 / 跟踪} 缺"GNN 绝对"这一格。
- **M7 泛化**：S37 单步对训练受试者拟合更好但不迁移（TF 差距 3.7× S36）。
- **M8 训练/测试 prev 误差分布不一致**：训练为独立注入的课程噪声，测试为自身时间相关误差。

## 7. 纪律

- **不修改**仓库的代码、配置、测试、`outputs/`；不删除任何文件；不训练；不启动 GPU 作业。
  允许 CPU 轻量分析（读 JSON、统计），脚本放 `.experiments/lit_hyp_20260929/<你的编号>/`，运行时 `torch.set_num_threads(4)` 或更少。
- 文献：必须核实论文真实存在（arXiv abs/html、CVF open access、IEEE Xplore/ACM DL 的 DOI 页、OpenReview、官方项目页）。不编造标题、作者、数字、公式。
  读不到全文就写"仅摘要"。不把预印本写成已发表；venue 不确定就写"venue 待核"。优先原论文与官方页面，不用二手博客。
  WebFetch 读不了 PDF 二进制时，优先 `arxiv.org/abs/…`、`arxiv.org/html/…`、`ar5iv.labs.arxiv.org/html/…`、`openaccess.thecvf.com` 的 HTML。
- 网络工具：WebSearch / WebFetch（可能在 `cursor` 动态命名空间里，用 GetDynamicTools 查看后通过 CallDynamicTool 调用）。
- 09-28 已删的预注册（可从提交 `85d76a1` 取回）：**只有被明确分配的子代理**可以读，并且必须标"已删记录，未复核"。

## 8. 交付

- 写入 `docs/research/lit_hyp_20260929/<编号>_<主题>.md`（简体中文），结构由你的任务说明给出。
- 最后一条回复给主代理：交付文件路径；≤ 50 行的要点，每条带证据标签与出处；你认为**最可能被推翻**的一条结论；你没能核实的内容。
