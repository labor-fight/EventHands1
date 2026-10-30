# M6 预注册草案：运动学子空间相关谱 KSCS 与并格共享分解 GSCD（诊断，CPU，无 GPU 臂）

> 草案状态：由实现 agent 起草，**尚未冻结**；主会话把本文并入唯一预注册并以 SHA-256 冻结 `tools/mv2/m6_kscs.py` 之后才允许正式读数。
> 来源：计划 `/home/lyq/.claude/plans/enumerated-exploring-quilt.md` §3 M6；R3 条目 `P5-E5-R1`（KSCS）、`P1-E7-R1`（GSCD），见 `outputs/kbattle_20261025/v3/r2_selected_entries.md`。
> 工具：`tools/mv2/m6_kscs.py`（单文件，含 `decide(result)`）。冒烟：`python tools/mv2/m6_kscs.py --smoke`；正式：`python tools/mv2/m6_kscs.py --out outputs/mv2/m6 [--workers 16]`（先 `common.verify_protocol()`，冻结前会退出）。
> 解释器 `/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python`，`CUDA_VISIBLE_DEVICES=""`，`PYTHONDONTWRITEBYTECODE=1`；全前台，无后台进程。种子 20261026（所有随机操作由它派生）。

## 0. 共同项

**问题**：K★ KS2（`#kstar-results-20261025`，`outputs/kstar/s2_stale/result.json`）测得冻结 K7 顶点锚定 patch 头（`outputs/xvm/s2/p4/A_m600/heads.pt` 的 `patch_r2`，SHA `f81ca794…`）的残差跨顶点 hop 相关谱：local hop1 / 2 / 3 / 4 = 0.838 / 0.598 / 0.247 / −0.117（global 0.881 / 0.726 / 0.513 / +0.100）。本诊断问两件事：谱是否是运动学（FK / LBS 雅可比）误差子空间的影子；以及其中多少来自相邻顶点共享同一采样窗（并格共读）。**不求解、不融合、不去相关**；不能主张性能；谱来自 patch CNN 头，不是 S39 读出的谱。

**KS2 残差的精确定义**（`tools/kstar/head_probe.py::run_stale`）：`res = pred_R0 − tg["full"]`，`tg["full"] = e1 · (uv(x_t) − uv(x_0))`（像素，沿顶点像面法向 `e1 = R[..., :, 0]`），其中 `x_0` = end−50 ms 的 ±25 ms 箱平滑 GT（锚点），`x_t` = end 的平滑 GT；**根与平移同样在动**（不是 `art` 目标）；掩码 `vis & obs2 & cos<0.4`（`obs2` = 近 2 cm 内 50 ms 有事件）；hop 相关 = 对每样本掩码内 hop-h 顶点对 (i<j) 的 (res_i, res_j) 作全体汇总 Pearson（汇总均值，`Lin.stats`）。tools 里 `HopAcc` 即此估计量。

**读入 / 只读 import**：`tools/kstar/head_probe.py`（`StaleData`、`load_head`、`read_head`、`Lin`、`hop_pairs`）、`tools/xvm/*`、`tools/s44/evflow.py`（`Mano64`、`Seq`、`visible`）、`semkine/jacobian.py`、`lie.py`、`xvm.py`（`vertex_frames`、`projection_jacobian`）、`mrf.py::hop_matrix`、`xyz_mesh.vertex_normals`、`sba.py::make_grid`。不修改任何既有文件。不读 zgz。

**数据与受试者**：FIT7 = ch、lfz、lpc、lr、ly、lyh、lyq 的 BASE 录制 `<subj>_local` / `<subj>_global`（14 条，只读 GT pos51 / betas / K，不读事件）；SELECT = ycy、ylf。

**输出**：`outputs/mv2/m6/result.json`（`common.save_json` 独占写）、`decision.json`（`decide(result)`）、`receipt.json`、`cache/pe1.pkl`、`cache/select.npz`（阶段缓存，可 `--stage pe1|collect` 分段）。

## 1. PE′-1　模板预测谱（不碰数据，只读 FIT7 GT 位姿标签）

**目的**：检验 KS2 的 hop 谱能否由"16 关节运动学误差经 FK 雅可比与法向投影"在零拟合下预测，并与图拉普拉斯低通场及各向同性噪声两个零模型对比。

**数据**：FIT7 BASE 14 条序列，P4 样本网格（`tools/xvm/p4_data.py` 的端点定义）每 **2** 个取 1，共 **8,897** 个位姿（local 4,274 / global 4,623；步长 8 时为 2,230，仅用于估时）。锚点位姿 = end−50 的 ±25 ms 平滑 GT；betas / K（×0.375）取序列自身。掩码只能用 `vis & cos<0.4`（无事件，故没有 `obs2`）。

**统计量**：
- 切空间 ξ = [ρ(3) | φ(3) | dφ(45)]（右乘李代数，`semkine/jacobian.py::vertex_jacobian`，含 posedirs 项；单测对 `ManoLayer` 有限差分校验）。顶点行 `Jn_i = e1_iᵀ (∂π/∂X)_i ∂X_i/∂ξ`（px / 单位切量），`e1` 取 `xvm.vertex_frames` 在锚点位姿的值。
- 先验 δξ ~ N(0, diag σ²)，r = Jn δξ。零均值、各 DOF 独立 ⇒ KS2 估计量有闭式（无 Monte Carlo 噪声）：
  `ρ_h = Σ_k σ_k² cov_h[k] / sqrt( Σ_k σ_k² vx_h[k] · Σ_k σ_k² vy_h[k] )`，`cov_h[k] = Σ_poses Σ_{(i<j), hop h, 两端在掩码内} Jn_ik Jn_jk`，`vx_h[k]`、`vy_h[k]` 为对应的 i 侧 / j 侧平方和（`accumulate_jmodel`；单测对 600 次抽样的 Monte Carlo 谱一致）。
- σ 的四个预声明先验（均由 **FIT7 GT 的 50 ms 李代数增量 RMS** 定，不用 SELECT、不用账本谱、不拟合；按域分别定，local 用 local 位姿）：
  - `J_main`（主）：17 个部位（平移、根旋转、15 个指关节）各向同性，σ_part = 该部位三轴增量 RMS 的均方根；
  - `J_perdof`：逐 DOF 增量 RMS；
  - `J_equal_effect`：σ_k ∝ 1 / (J 第 k 列在掩码顶点上的 RMS)，每 DOF 等像素效应；
  - `J_fingers`：`J_main` 但平移与根旋转 σ = 0（KS2 `art` 目标约定；**注意**账本谱用的是 `full` 目标，见 §5 差异 1）。

**零模型**：
- (a1) `lap_scalar`：图拉普拉斯（1-ring 组合拉普拉斯 L）热核低通随机场，协方差 `exp(−τL)`，标量场；
- (a2) `lap_vector`：同核的 2-D 向量场（两分量独立同核），投影到 `e1` 后的标量；
- τ 由二分使**池化 hop-1 相关 = 账本 hop-1**（local 0.8377 / global 0.8805）标定，每个 FIT7 受试者与汇总各标定一次（零模型有 1 个自由参数，J 模型 0 个，对 J 不利）；
- (b) `iid`：各向同性逐顶点噪声，hop≥1 相关恒为 0。

**门（本域 = local，账本谱为 L_h）**：
- `T_h = max(2·SD_h, halfCI_h)`，`SD_h` = 7 个 FIT7 受试者各自 J_main 谱在 hop h 的样本标准差（ddof=1，"折间离散"；7 个留一汇总复制品的 SD 约小 6 倍，作为描述量 `sd_loso_replicates` 一并记录），`halfCI_h` = 账本 `r_ci` 半宽；`T = max_{h≤3} T_h`（替换计划的占位 0.10）。
- `T2 = max(2·SD(gap_s), rms_h(halfCI_h))`，`gap_s` = 受试者 s 上"三个零模型中 RMSE 最小者 − J_main 的 RMSE"（RMSE 取 hop 1–4 对账本；替换占位 0.03）。
- `pass_spectrum_match`：池化 J_main 的 `max_{h≤3} |ρ_h − L_h| ≤ T` 且 `ρ_4 < 0`；`pass_beats_null`：池化 `gap ≥ T2`。
- **通过** = 二者皆真；**杀死** = `gap < T2`（运动学解释对零模型无特异性）；其余 = 仅描述（含 σ 变体、global 域）。

**有效性条件（任一不满足 ⇒ `tool_defect`，决不记为通过）**：14 项单测全过（见 §4）；账本 JSON 与账本文字谱一致（±5e-4）；FIT7 白噪声机会份额均值 < 0.5（份额指标非空转，见 §2）。零模型二分未达到 hop-1 目标只记 note。

## 2. PE′-2　冻结 patch 头在 SELECT 上的真实残差在 span(J) 内的方差份额

**目的**：KS2 只存了矩累加器，没有逐顶点残差，所以用 `head_probe.StaleData` / `load_head` / `read_head` / `Geo` **原样重算**（只读 import，不改 head_probe.py），再把残差投影到 span(J)。

**数据**：SELECT（ycy、ylf）`subject_seqs` 的全部 16 条序列（BASE 与 `_v2–_v4` 变体）、步长 4、连续端点对，共 **10,140** 个样本（与账本 `n_samples` 相等；已核对），R0（600 ms 时间面，训练时形式）。主读数 = KS2 同一样本集（为的是复现账本谱）；BASE-only 4 条序列作敏感性。

**重现校验（工具有效性）**：重算的 local / global hop 1–4 谱 |Δr| ≤ 0.01 且对数 n 相对差 ≤ 1e-4，样本数 = 10,140；冒烟已在 32 行切片上证明与 `head_probe.run_stale` 的累加逐位一致（|Δr| < 1e-9，n 相等）。

**统计量**（每样本在自己的锚点 x₀、自己的掩码上）：
- `Jw = Jn · diag(σ_main)`（σ_main 域内 FIT7 `J_main`），SVD，秩 k = 奇异值 ≥ 1e-2 · s_max 的个数；份额 = ‖U_kᵀ r‖² / ‖r‖²；池化份额 = Σ‖PJ r‖² / Σ‖r‖²（按能量加权，"方差占比"）。k、掩码顶点数 n_mask、系数参与数 PR 作 n_eff 的描述量；1e-3 秩阈值作敏感性。
- 对照 / 零模型：
  - `chance`（机会份额）：残差在掩码内随机置换后的份额（每样本 4 次）；
  - `gauss`：独立高斯向量份额，应 ≈ k / n（单元检查）；
  - `wrong_pose`：J 取同序列另一位姿（偏移 97 个样本）、同顶点同法向；**描述量**（实测 J 的像面法向子空间对位姿很不敏感，不用作特异性门）；
  - `smooth_basis`：同维 k 的最低 k 个网格拉普拉斯模态限制到掩码行后的份额——**特异性对照**：运动学子空间是否强于任意同维光滑基；
  - `pos_control`：用本样本 J 模拟 r = Jw·z + 10 % 能量 iid 噪声，份额应 ≥ 0.85；
  - FIT7 白噪声机会（PE′-1 位姿 + 掩码 + σ_main 秩阈值的 k / n）：7 个受试者均值 μ_w 与 SD s_w（"FIT7 计算的份额基线"）。
- CI：分组 bootstrap（组 = 序列 × 1 s 块，KS2 同定义，1000 次，种子 20261026）；另报 ycy / ylf 各自份额（只有 2 个 SELECT 受试者，受试者块 bootstrap 无意义，见 §5 差异 4）。

**门（local，全 16 序列）**：
- **通过**：份额 s ≥ 0.6，且 s−chance 的 CI 下界 > 0，且 s > μ_w + 2 s_w，且 s−smooth_basis 的 CI 下界 > 0；
- **杀死**：s < 0.3，或 s−chance 的 CI 上界 ≤ 0；
- 其余：仅描述。
- 阈值来源：0.6 / 0.3 取自计划原文；"机会 / 光滑基 / FIT7 白噪声"三条是本草案新增的特异性条件，避免 ≥ 0.6 只来自"秩 51 / 掩码 ≈ 170"的维数占比或光滑性（见 §5 差异 5）。
- **有效性**：pos_control < 0.85、gauss 偏离 k / n 超 0.03、FIT7 白噪声机会均值 ≥ 0.5、重现校验失败 ⇒ `tool_defect`。

**描述量**：global 域及 BASE-only 同表；J 模型在 SELECT 真实掩码上的谱（含零模型）`pe2.j_on_select_masks`；账本复现谱及 CI。

## 3. GSCD　并格共享相关分解（local，主读数）

**目的**：同 hop 内，采样窗重叠大的顶点对，残差相关是否更高；即相关谱里"共读同一些粗像素"的份额。

**s_ij 的定义**（预声明，主）：patch_r2 头对顶点 i 读一个以 uv₀ᵢ 为中心、沿 `R`（e1,e2）旋转的 16×16 双线性采样窗，半宽 ρ = f·0.02 m / z（f = √(fx fy)，K×0.375，与 `PatchHead.forward` 同）；窗内 256 个采样点按 `grid_sample`（align_corners=True，零填充）展开成 240×180 粗像素权重图 aᵢ（单测与 `torch.grid_sample` 逐点一致）。设粗像素 p 的源面积增益 g_p（`semkine/sba.py::make_grid().gain`，9 / 6 / 4）。
`s_ij = Σ_p g_p aᵢ(p) aⱼ(p) / sqrt(Σ g aᵢ² · Σ g aⱼ²)`：粗像素事件数方差 ∝ g_p 的独立散粒噪声经两个线性窗读出的**精确相关系数**（单测：对 4000 次合成噪声的经验相关与 s_ij 差 < 0.06）。非加权版 `su`（g≡1）作敏感性。
描述量（不参与门）：S39 layer1–3 读出（步长 4 / 8 / 16，`common.sample_map` 几何）两顶点双线性单元支撑的直方图交集 `sL1/2/3` 的三分层差、共享比例。注意：S39 的单元不是 patch 头的机制，只用来给 M1 的"每格取一次"提供数量级。

**统计量**：对 hop 1–3、两端都在 KS2 掩码内的顶点对（i<j），按 hop 内 s_ij 的 1/3、2/3 分位（本集合内汇总）分三层；D_h = 最高层汇总 Pearson − 最低层汇总 Pearson；**D = D_1、D_2、D_3 的平均**。CI = 分组 bootstrap（组 = 序列 × 1 s 块，1000 次，分位点固定）。

**零模型 / 对照**：
- 置乱对照：在每个（样本，hop）段内随机置换 s_ij（保持 hop 分布与每段 s 的多重集，破坏 s 与顶点对的对应），分位点不变，200 次，报 |D_perm| 的 97.5 % 分位；
- `within`：层标签取（样本 × hop）段内 s 的秩三分位（去掉样本间尺度 / 深度混杂），段内对数 < 9 的段丢弃；
- `dist_matched`：hop 内按像面距离 / ρ 的 5 分位分箱，再在箱内按 s 三分层（去 hop 内几何距离混杂；检验力低，仅描述）；
- `control_syn`（功效 / 阳性对照）：合成残差 = 窗口权重图 × 散粒噪声 √g_p·z，同一 s 分层。该对照的相关**按构造等于 s_ij**，若 D_syn < 0.15 说明本工具即使机制成立也分不出来；
- `control_lap`（几何对照）：同点上的图拉普拉斯热核标量场（τ 由 SELECT 掩码标定到账本 hop-1），无任何像素机制；
- `control_iid`：独立高斯，D 应 ≈ 0。

**门**（0.15 = KS2 谱跨度 0.59 的 1/4，0.05 = KS2 时间置乱 −0.004 / −0.014 量级加余量；皆预注册约定、不是文献阈值，不由 FIT7 校准）：
- **通过**：D ≥ 0.15 且 CI 下界 > 0 且置乱 |D_perm| 97.5 % 分位 < 0.05，且 `control_syn` D ≥ 0.15（有功效），且 `within` CI 下界 > 0，且 `control_lap` D < 0.15；
- **作废**（计划："差 < 0.05 且 CI 含 0 → 本分解作废，M6 保留诊断身份"）：D < 0.05 且 CI 含 0，且有功效；
- 其余：仅描述（含"有 D 但 within 不成立 = 尺度 / 深度混杂""几何对照也分层 = 几何解释""功效不足 = 无结论"）。
- **有效性**：`control_iid` 的 |D| ≥ 0.01、置乱 97.5 % 分位 ≥ 0.05 ⇒ `tool_defect`。

**描述量**：global 域、BASE-only、非加权 s、各 hop 的 D_h、s 的 hop 内分位、s 与距离的 Spearman（共线性）、sL1/2/3。

## 4. 单元检查（冒烟与正式都先跑；全部合成 / 资产，不含效应）

hop 估计量恢复已知核谱；J 模型闭式 = Monte Carlo；切空间雅可比 = `ManoLayer` 有限差分（相对 < 1e-5）；span 投影（空间内 1、正交 0、高斯 ≈ k/n）；窗口重叠（相同 =1、单调、不交 =0、旋转 90° 不变、均匀增益时加权 = 非加权）；窗口点 = `torch.grid_sample`（含边界零填充）；散粒噪声相关 = 加权余弦；三分层估计恢复已知 D 与零 D；置乱保持段多重集；热核标定命中目标；李代数同态增量 = 0；`decide()` 的 20+ 个分支（pass / kill / descriptive / tool_defect）；池化估计 = `head_probe.Lin`；hop 点对 = `head_probe.hop_pairs`。冒烟另含：几何 = `Geo`、J 点 = FK 顶点、缓存往返、`result` 可 JSON 序列化、重算 = `run_stale`（32 行切片逐位）、合成 J 残差份额高且赢光滑基、合成光滑残差输给光滑基、合成窗口噪声 D ≥ 0.15、iid D ≈ 0。

## 5. 与计划 §3 的差异及理由

1. **残差口径 = `full` 目标，不是 `art`。**账本 hop 谱是 `pred − tg["full"]`（`head_probe.py::run_stale`：根与平移同动），任务说明中"art 目标"是 KS2 偏相关门用的另一个目标；为可比，主模型 `J_main` 含全部 51 DOF，`J_fingers` 为 `art` 口径变体（描述）。
2. **PE′-1 掩码缺 `obs2`**（无事件）；PE′-2 的 `j_on_select_masks` 用真实掩码补一份 J 模型谱。
3. **PE′-2 主样本集 = KS2 的 16 条 SELECT 序列（含 `_v2–_v4`）**，而不是通用规则的 "只用 BASE"：为的是复现账本 hop 谱（那份谱就是这 16 条的汇总）；BASE-only 作敏感性。patch 头只在 FIT7 上训练，SELECT 对它是留出的。
4. **CI**：计划写"9 条序列 bootstrap"、任务写"受试者块 bootstrap"，SELECT 只有 2 个受试者；改用 KS2 同口径的（序列 × 1 s 块）分组 bootstrap，另给两个受试者各自的点估计。
5. **阈值**：计划的 0.10 / 0.03 → `T`、`T2`（FIT7 折间 SD 与账本 CI 半宽的较大者，见 §1）；0.6 / 0.3 / 0.15 / 0.05 / 0.15 取计划原文；新增的是特异性 / 功效 / 混杂条件（机会份额、FIT7 白噪声份额、光滑基、置乱、within、几何对照、合成功效），它们只会把"通过"降为"仅描述"或 `tool_defect`，不改变计划的杀死条件。`T` 用"受试者间 SD"而非"留一复制品 SD"（后者小约 6 倍，更严），请主会话确认。
6. **s_ij 用 patch 头自己的窗**（256 点旋转方窗），不是 S39 layer 单元；后者作描述量。
7. 计划中的"n_eff 决定顶点卷积层数与采样密度"只报描述量（k、n_mask、PR），不作门。

## 6. 预计 CPU 时间（144 核；以切片实测外推）

单测 6 s；PE′-1 约 3 min（8,897 位姿，FK + 雅可比 + 稀疏 hop 求和，另含 σ 预扫描）；SELECT 重算 10,140 样本 16 worker 约 8 min；span / J 分析约 2–3 min；GSCD 点对（约 7.7 M 对）约 2 min、统计与 200 次置乱 + 1000 次 bootstrap × 约 10 个变体约 4–6 min。合计约 20–25 min，内存 < 20 GB，无 GPU。

## 7. 开发期读数披露

开发中只读取了：窗口重叠 s_ij 的 hop 内分位与合成对照（合成窗口噪声 D_syn ≈ 0.25，图拉普拉斯几何对照 D_lap ≈ 0.02–0.05，iid ≈ 0；800 个 SELECT 样本几何上跑出，真实残差未打印）；合成 J 残差与合成光滑残差上 span 份额与光滑基的判别力（J 合成份额 ≈ 0.93、赢光滑基 ≈ 0.08；光滑合成残差输 ≈ 0.37）；计时。**没有读取或写出任何真实残差的 hop 相关、份额、层间差或 J 模型对账本的偏差。**冒烟只写形状、计时与通过 / 失败（`outputs/mv2/smoke/m6/smoke.json`）。

## 8. 结果 JSON 结构（`decide` 读取的键）

`ledger{local,global}{r,ci,n}`、`unit_checks`、`pe1.{local,global}{sigma,spectra{J_main…,lap_scalar,lap_vector,iid}{pooled,per_subject},tau,white_chance,J_main_loso}`、`pe2{repro,share{local,global}{all,base},j_on_select_masks,…}`、`gscd{local,local_base,global,global_base}{primary,perm,within,dist_matched,control_syn,control_lap,control_iid,sens_unweighted,layer_cells,…}`。`decide` 输出 `kscs_spectrum` / `kscs_span` / `gscd` 三个裁决、`tool_defect`、`consequence`（通过 → M1 的 span(J) 投影配对臂、每格取一次去重汇聚臂；杀死 / 作废 → 诊断身份保留；均无 GPU 臂）。
