# EvHandTrack 实验设置说明

整理日期：2026-11-10；2026-11-13 按现行方法（先路由采样 + r_s2dx2）改写。

本文依据实际数据划分、训练配置、运行日志及评测代码，整理 EvHandTrack 的数据、评测、指标和训练设置。global/local 的动作定义采用用户在本次会话中补充的采集说明。

当前正式 EvHandTrack 为 **先路由采样（Route-First Sampling，RFS）+ 表面映射网络 r_s2dx2 + PMR D1（R）主干 + rol5**（用户 2026-11-12“全文统一成先路由”；2026-11-13“论文用 r_s2dx2，真实数据，仿真数据，和 EvRealHands 两个真实数据集都换成 r_s2dx2”）。仿真数据与 EvRealHands 上的训练种子为 **3407、3408**（用户 2026-11-13“种子2个就行”），运行名为 `pmr_rfsr_s<种子>`（从头 6000 步）和 `pmr_rfsr_rol5_s<种子>`（rol5 微调），臂名 `rfsr` → `rfsr_rol5`；真实 DAVIS346 上的主行已有 3407、3408、3409 三个种子，保持不变。主结果采用流式输入形式。表面映射网络的论文名称为 **Event Surface Mapping Network (ESMNet)**，论文使用它的 r_s2dx2 变体（宽度比 r_s2d 大，见 4.2）；代码中的历史类名、配置键和检查点名称保留原标识。

2026-11-10 定的 EvHandTrack（先采样再路由，路由器 r_s2d，D1 R + rol5，三个种子 `pmr_d1r_rol5_s3407/3408/3409`）现称 **原版 EvHandTrack**，只作记录。本文凡写到原版之处都会注明；没有注明的内容对两者相同。

当前进度（2026-11-13）：仿真数据和 EvRealHands 的 `rfsr` 两个种子均已训满 6000 步并通过验收，`rfsr_rol5` 在训练，读数待完成。成本（时延、MACs、整体参数）待成本块：全部训练与读数结束后，在一个安静的独占块里与原版同块测，本文不预填数值。真实 DAVIS346 的 `rfsr_rol5` 三个种子已完成，三位留出受试者合并、正常光下为 18.12 / 10.40 / 13.22 / 7.71 / RA 14.39（MPJPE-L / MPJPE-G / MPVPE-L / MPVPE-G / RA，毫米；记录见[账本](EXPERIMENT_RECORDS.md#real-rfs-20261112)）。运行与登记见[账本 RFSU 一节](EXPERIMENT_RECORDS.md#rfsu-20261113)。

身份文件：[outputs/evhandtrack/identity.json](../outputs/evhandtrack/identity.json) 目前仍是原版的身份。现行版的身份文件在仿真两个种子的读数与成本块完成后另写，原版身份届时归档。

## 1. 数据与划分

### 1.1 数据来源与参考标注

主实验使用 `hand_data51`。基础数据来自真实 RGB-D 手部录像，事件流由 RGB 视频经 v2e 合成。MANO 标注由 WiLoR 提供逐帧初始化，再结合 RGB-D 信息进行时序优化得到。因此，本实验的数据应描述为**由真实录像生成的合成事件数据，以及通过 RGB-D 优化获得的 MANO 伪真值**。

下文公式中的 GT 表示这套参考标注。该标注不等同于动作捕捉系统直接测量得到的真值。

原始数据包含 10 名受试者，每名受试者分别录制 global 和 local 两类运动，共 20 条基础录像。源 RGB 分辨率为 640 × 480；转换后事件输入的空间分辨率为 **240 × 180**。

状态标注插值到 **1 ms 时间网格**：平移采用 PCHIP，根旋转及手指关节旋转采用旋转样条；插值在连续有效标注区间内进行。每个状态包含平移 3 维、根旋转轴角 3 维和手指姿态轴角 45 维，共 **51 维**。

来源：

- [RGB-D 数据与 MANO 伪真值说明](/data1/lyq/data/hand_data/WiLoR/README_pseudo_gt.md)。
- [v2e 生产入口](/data1/lyq/data/hand_data/v2e/scripts/batch_bag_v2e.py)。
- [数据转换实现](../tools/prepare_hand_data.py)。

### 1.2 全局运动与局部运动的划分依据

根据采集时的运动类型，将序列划分为全局运动（global）和局部运动（local）：

- **全局运动：**由手臂带动手掌及手指产生的整体运动。
- **局部运动：**手臂保持静止，手指执行单个手指运动与多个手指运动的混合。

两类序列分别用于评估整体运动条件下和手指局部关节运动条件下的跟踪表现。程序根据序列已有类别读取对应数据。

上述定义由用户在本次会话中确认。local/global 是运动类别，与指标采用相机坐标误差还是平移对齐误差是两个独立概念。

### 1.3 状态更新网络的训练与评测划分

主实验采用按受试者隔离的划分。权威划分文件为：

`/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json`

| 用途 | 受试者 | 基础录像数 | 实际使用序列数 |
|---|---|---:|---:|
| 状态更新网络训练 | ch、lfz、lpc、lr、ly、lyh、lyq、ycy、ylf | 18 | 72 |
| 开发与最终报告 | zgz | 2 | 2 |

每名训练受试者的 global/local 基础录像均保留四个版本：

| 版本 | 内容 |
|---|---|
| 原始版本 | 基础 RGB 录像经 v2e 合成的事件 |
| `_v2` | 动态光照增强后重新生成的事件 |
| `_v3` | 正负对比度阈值扰动后重新仿真的事件 |
| `_v4` | 深度感知空间扰动后重新仿真的事件，并更新相应相机内参 |

因此，训练集包含 9 × 2 × 4 = **72 条序列**，global/local 各 36 条。这些序列来自 18 条基础录像及其增强版本，不能表述为 72 次独立采集。当前 `_v2` 已采用动态光照增强，历史材料中的均匀速度缩放描述不适用于当前版本。

评测使用未参与训练的受试者 zgz 的两条基础序列 `zgz_global`、`zgz_local`。

**当前 `val`、`val_core` 和 `test` 都指向同两条 zgz 序列，开发集与最终报告集重合，没有另设独立封存测试集。**训练受试者与 zgz 之间没有主体交叉；模型最终使用预先规定的末步检查点，具体步数见第 4 节。

来源：

- [实际划分文件](/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json)。
- [划分生成与协议说明](../tools/build_splits_semkine.py)。
- [V2 动态光照版本说明](/data1/lyq/data/hand_data/v2e/reports/v2_dynamic_illumination_gate.md)。
- [V3 对比度阈值增强说明](/data1/lyq/data/hand_data/v2e/reports/v3_threshold_augmentation.md)。
- [V4 空间扰动说明](/data1/lyq/data/hand_data/v2e/reports/v4_depth_aware_spatial_jitter.md)。

### 1.4 ESMNet 使用的训练子集

论文使用 ESMNet 的 r_s2dx2 变体（结构见 4.2）。它**每个数据集各训一个，用该数据集自己的训练集**，配方与该数据集上 r_s2d 的配方相同，只换网络结构。

- **仿真数据（hand_data51）：**从上述 9 名训练受试者中排除 ylf，实际使用 **8 名受试者、64 条序列**：ch、lfz、lpc、lr、ly、lyh、lyq、ycy。仅以种子 3407 预训练一次；状态更新网络的两个种子共用同一冻结 ESMNet 检查点。状态更新网络及 rol5 微调仍使用 9 人、72 条序列。
- **真实 DAVIS346：**用全部 12 名训练受试者，不留出受试者（划分见附录）。
- **EvRealHands：**用它自己的训练受试者；数据、划分与协议见[账本](EXPERIMENT_RECORDS.md#erh-prereg-20261111)。

原版 EvHandTrack 使用的 r_s2d 在仿真数据上同样是 8 人、64 条序列，三个种子共用同一检查点。

来源：[ESMNet 训练入口](../tools/pfo/router_train.py)、[留出受试者的数据过滤实现](../tools/s45sm/router_train.py)、[仿真 r_s2dx2 入口](../tools/rfsu/router_sim.py)、[真实数据与 EvRealHands 的 r_s2dx2 入口](../tools/real/router_variant.py)。

## 2. 评测协议

### 2.1 连续有效区间与初始化

正式主结果采用 PFO 引擎的 **E1 连续递推协议**。每条序列按标注文件中的 `valid_runs_ms` 划分为连续有效区间；每个能产生评测输出的有效区间独立初始化。

设有效区间为半开区间 [a, b)，段首初始状态为：

$$
\hat{\boldsymbol x}_a=\boldsymbol x_a^{GT}+\boldsymbol\epsilon.
$$

各状态分量的初始化扰动为：

| 分量 | 扰动分布 |
|---|---|
| 平移的三个分量 | 独立零均值高斯噪声，标准差 0.005 m，即 5 mm |
| 根旋转轴角的三个分量 | 独立零均值高斯噪声，标准差 0.05 rad |
| 手指轴角的 45 个分量 | 独立零均值高斯噪声，标准差 0.05 rad |

评测随机数种子固定为 0，并按相同顺序处理 global、local 序列，使不同模型训练种子的读数具有一致的初始扰动。该评测初始化与训练阶段的混合前态扰动分布应分别描述。

来源：[E1 引擎](../tools/pfo/engine.py)、[初始化噪声实现](../semkine/eval_track.py)。

### 2.2 固定手形与已知相机内参

固定手形指使用每条序列提供的 **10 维 MANO 形状参数 beta**，并在该序列中保持不变。模型预测的是姿态、根旋转和平移，形状参数不由状态更新网络估计。预测与参考标注的 MANO 解码均使用同一组 beta。

相机内参 K 同样来自序列标注文件。构造事件射线时，内参按 `RENDER_SCALE = 0.375` 缩放到 240 × 180 事件网格。评测属于已知个体形状和相机标定条件下的跟踪。

来源：[序列上下文](../tools/pfo/nets.py)、[现行版配置](../configs/rfsu_20261113/sim/pmr_rfsr_rol5_s3407.yaml)、[原版配置](../configs/pmr_20261109/pmr_d1r_rol5_s3407.yaml)、[前向与相机内参实现](../model/model.py)。

### 2.3 事件窗口与递推过程

观测窗口和状态更新间隔均固定为 **50 ms**。区间 [a, b) 内的输出端点为 a + 49、a + 99、a + 149 等，且端点必须小于 b；每个端点使用截至该端点的 50 ms 事件。

连续递推过程为：

$$
\hat{\boldsymbol x}_{k}
=\hat{\boldsymbol x}_{k-1}
+f_\theta\left(\mathcal E_k,\hat{\boldsymbol x}_{k-1};\boldsymbol\beta,\boldsymbol K\right).
$$

段首首次更新后，下一步直接使用上一步预测作为前态；区间内不再用标注状态修正预测。空事件窗口的状态增量为零，保持前态。每段首个输出也参与评分。

当无效标注间隔结束、进入新的有效区间时重新初始化。因此，连续递推的范围是**每个连续有效区间内**。

| 评测序列 | 原记录时长 | 能产生输出的递推段数 | 50 ms 评分端点数 |
|---|---:|---:|---:|
| `zgz_global` | 约 69.334 s | 1 | 1386 |
| `zgz_local` | 约 65.284 s | 41 | 1204 |
| 合计 | — | 42 | 2590 |

local 原始标注含 90 个有效区间，其中 41 个区间足以产生至少一次 50 ms 输出。

端点网格由 E1 引擎与序列决定，与方法无关，上表数目对原版和现行版相同。

来源：[E1 递推实现](../tools/pfo/engine.py)、[原版正式流式评测报告](../outputs/pmr/20261109/readings/pmr_d1r_rol5_s3407__str/e1/report.json)、两条序列的 `aux.npz`。现行版的流式读数由 [tools/rfsu/sim.py](../tools/rfsu/sim.py) 的 `read` 写入 `outputs/rfsu/20261113/sim/readings/pmr_rfsr_rol5_s<种子>__str/e1/report.json`（读数完成后出现）。

### 2.4 流式主行的执行范围

正式主行使用流式输入形式：采样由已到达事件的历史事件量决定，像素时间表面跨状态更新保留，并利用过去 50 ms 的历史。

现行版在流式采样之前先做路由（见 4.3）。读数时，[semkine/rfs.py](../semkine/rfs.py) 的 `install_stream` 替换流式实现的取包函数：

1. ESMNet 对该更新区间内的全部事件逐个判断是否在手上（手上概率 p ≥ 0.5）；该区间之前 50 ms 的历史事件作为独立一包做同样的判断；
2. 手上事件，加上离同一包内某个手上事件不超过 NEAR 像素（仿真 240 × 180 画布为 11 像素）的手外事件，构成可采事件；
3. 流式采样规则只在可采事件上运行：每个事件的取样步长由其到达时过去 50 ms 内的可采事件数决定，预算 2048，取样计数器的起点沿用整个事件流的计数。

7 维事件 token 仍按整个事件流计算，上一事件间隔与同像素历史都含手外事件；只有节点的选择限定在可采事件内。

当前实际前向仍每 50 ms 更新一次 MANO。ESMNet 的判断按完整更新区间进行，PFA 记录在更新时间确定，不是逐事件到达即判断。已有前缀与分块等价检验具有固定更新网格、路由器豁免这一边界；先路由使节点选择依赖路由器对整个更新区间的判断，因此同样不能声称完整路由器已经逐事件因果，或任意改变状态更新边界后整网输出仍然相同。原版的该检验回执在 `outputs/evhandtrack/receipts/`，是针对先采样再路由的流式实现做的；未见针对先路由版的同类回执。

正式流式主行由分包形式训练得到的检查点替换采样和 token 计算方式得到，没有额外进行流式形式重训，现行版与原版相同。

来源：[流式输入实现](../semkine/tpd_str.py)、[先路由采样实现](../semkine/rfs.py)、[输入形式读数入口（原版）](../tools/pmr/run_forms.py)、[现行版读数入口](../tools/rfsu/sim.py)。

## 3. 指标与多种子统计

### 3.1 相机坐标绝对误差

MANO 解码得到 **21 个关节、778 个顶点**。内部三维坐标单位为米，报告时统一换算为毫米。

设相机坐标系中的预测关节和标注关节分别为 $\hat{\boldsymbol J}_{t,j}$ 与 $\boldsymbol J_{t,j}$，预测顶点和标注顶点分别为 $\hat{\boldsymbol V}_{t,v}$ 与 $\boldsymbol V_{t,v}$。关节索引为 j = 0, …, 20，顶点索引为 v = 0, …, 777。对 T 个有效评分端点，定义：

$$
\operatorname{MPJPE}_{cam}
=\frac{1000}{21T}\sum_{t=1}^{T}\sum_{j=0}^{20}
\left\|\hat{\boldsymbol J}_{t,j}-\boldsymbol J_{t,j}\right\|_2,
$$

$$
\operatorname{MPVPE}_{cam}
=\frac{1000}{778T}\sum_{t=1}^{T}\sum_{v=0}^{777}
\left\|\hat{\boldsymbol V}_{t,v}-\boldsymbol V_{t,v}\right\|_2.
$$

两项均保留相机坐标系中的绝对位置误差，对应结果文件中的 `mpjpe_abs_mm`、`mpvpe_abs_mm`。

### 3.2 腕部相对 MPJPE

令下标 0 表示腕关节。腕部相对 MPJPE 分别减去预测与标注各自的腕关节位置：

$$
\operatorname{MPJPE}_{wrist}
=\frac{1000}{21T}\sum_{t,j}
\left\|
(\hat{\boldsymbol J}_{t,j}-\hat{\boldsymbol J}_{t,0})
-(\boldsymbol J_{t,j}-\boldsymbol J_{t,0})
\right\|_2.
$$

该指标只移除腕部平移，保留旋转、尺度和关节姿态误差，不进行 Procrustes 对齐。当前结果文件中的 `mpjpe_ra_mm` 使用这个定义。

### 3.3 当前 MPVPE 相对误差的实际对齐点

**当前正式评分代码的 `mpvpe_ra_mm` 对齐的是各自网格的顶点 0，而非腕关节。**

其实际定义为：

$$
\operatorname{MPVPE}_{vertex0}
=\frac{1000}{778T}\sum_{t,v}
\left\|
(\hat{\boldsymbol V}_{t,v}-\hat{\boldsymbol V}_{t,0})
-(\boldsymbol V_{t,v}-\boldsymbol V_{t,0})
\right\|_2.
$$

原因是评分对顶点数组调用 `root_align`，该函数统一减去输入数组的第 0 个点。MANO 顶点数组保留原始 778 个网格顶点，其第 0 个点不是额外插入的腕关节。

因此，现有主表中的 MPVPE 应描述为**顶点 0 平移对齐后的平均顶点误差**。

如果论文采用腕部相对 MPVPE，则应分别计算预测顶点减预测腕关节、标注顶点减标注腕关节：

$$
\operatorname{MPVPE}_{wrist}
=\frac{1000}{778T}\sum_{t,v}
\left\|
(\hat{\boldsymbol V}_{t,v}-\hat{\boldsymbol J}_{t,0})
-(\boldsymbol V_{t,v}-\boldsymbol J_{t,0})
\right\|_2.
$$

这是另一种评分定义，需要按该定义重新计分后使用对应数值。本文记录现有实现及其差异。

来源：[评分函数](../tools/xcl/evaluate.py)、[root_align](../semkine/eval_track.py)、[MANO 输出](../model/mano_layer.py)。

### 3.4 主表列名与聚合顺序

| 主表列名 | 实际含义 |
|---|---|
| MPJPE-local | local 动作序列上的腕部相对关节误差 |
| MPJPE-global | global 动作序列上的腕部相对关节误差 |
| MPVPE-local | local 动作序列上的顶点 0 对齐顶点误差 |
| MPVPE-global | global 动作序列上的顶点 0 对齐顶点误差 |
| RA-MPJPE（递推） | 两类序列全部递推端点汇总后的腕部相对关节误差 |

每个种子先在每个端点对关节或顶点取平均，再对对应序列的有效端点取平均。合并两类序列时，按端点数加权：

$$
E_{all}^{(s)}
=\frac{1204E_{local}^{(s)}+1386E_{global}^{(s)}}{2590}.
$$

EvHandTrack 行对其所含训练种子的对应指标等权平均。设该行的种子集合为 $\mathcal S$，则

$$
\bar E=\frac{1}{|\mathcal S|}\sum_{s\in\mathcal S}E^{(s)}.
$$

仿真数据与 EvRealHands 上现行版 $\mathcal S=\{3407,3408\}$，即 $\bar E=\left(E^{(3407)}+E^{(3408)}\right)/2$（用户 2026-11-13“种子2个就行”）；真实 DAVIS346 的主行 $\mathcal S=\{3407,3408,3409\}$。原版（仿真）为三个种子 3407、3408、3409，$\bar E=\left(E^{(3407)}+E^{(3408)}+E^{(3409)}\right)/3$。对比基线保持三个种子。

每个种子表示状态更新网络分别从头训练、再进行同种子 rol5 微调；同一数据集上各种子共用同一个冻结的 ESMNet（r_s2dx2 每个数据集各一份）。结果 JSON 中保留的历史键 `two_seed_mean`，在现行版仿真主行里恰好是两种子均值，种子数由 `n_seeds = 2` 标明；在原版主行里同一个键存的是三种子均值，由 `n_seeds = 3` 标明。

来源：[端点评分与总体聚合](../tools/pfo/engine.py)、[现行版出表与两种子均值](../tools/rfsu/tables.py)、[原版三种子均值生成](../tools/evhandtrack/identity.py)、[主表指标映射](../tools/evhandtrack/identity_b2rol5.py)。

## 4. 实现与训练

### 4.1 三阶段训练设置

下表为仿真数据上现行版的设置。

| 阶段 | 训练数据 | 种子 | 每次优化步数 | 峰值学习率 | 学习率策略 |
|---|---|---|---:|---:|---|
| ESMNet（r_s2dx2）预训练 | 8 人、64 条序列 | 3407 | 6000 | 0.001 | 500 步线性预热，之后恒定 |
| D1 R 状态更新网络（`rfsr`，先路由采样） | 9 人、72 条序列 | 3407、3408 | 6000 | 0.004 | 500 步线性预热，随后余弦衰减到峰值的 2% |
| rol5 闭环微调（`rfsr_rol5`，先路由采样） | 同上，连续五包样本 | 对应同种子 | 1500 | 0.001 | 100 步线性预热，随后余弦衰减到 0 |

三个阶段均使用 Adam、双卡 DDP、每卡批量 512、总批量 1024、BF16 混合精度，每个 rank 使用 16 个数据加载进程。rol5 的批量单位为五包展开样本。真实 DAVIS346 与 EvRealHands 上的状态更新网络改用单卡（每步 512 包、梯度累积 2，等效批量 1024）和每卡 6 个数据进程，见附录。

ESMNet 取固定第 6000 步权重；状态更新网络先取同种子第 6000 步权重作为微调初值，最终取 rol5 第 1500 步权重。

现行版的 `rfsr` / `rfsr_rol5` 配置与原版同种子的 D1 R / rol5 配置经逐行比对（种子 3407、3408）：除路由器检查点、运行名与输出目录、rol5 的热启动检查点路径、`RFS` 配置块（先路由采样的参数与 `MAX_ROWS`）之外，其余各项完全相同，包括学习率、预热、衰减、在线增强、热像素、批量、步数、损失、前态噪声与确定性设置。原版的种子为 3407、3408、3409。

来源：[ESMNet 优化器和训练入口](../tools/s45sm/router_train.py)、[现行版 rfsr 配置](../configs/rfsu_20261113/sim/pmr_rfsr_s3407.yaml)、[现行版 rfsr_rol5 配置](../configs/rfsu_20261113/sim/pmr_rfsr_rol5_s3407.yaml)、[原版 D1 R 配置](../configs/pmr_20261109/pmr_d1r_s3407.yaml)、[原版 rol5 配置](../configs/pmr_20261109/pmr_d1r_rol5_s3407.yaml)。

### 4.2 ESMNet 预训练

ESMNet 从随机初始化开始训练，以事件观测预测部位类别、静止 MANO 模板表面位置和手部事件有效性。输入采用事件图像特征和逐事件 token，使用像素重排与 CNN/FPN 提取图像上下文。

论文使用的 r_s2dx2 与原版的 r_s2d 同属 [semkine/pfo_router.py](../semkine/pfo_router.py) 的轻量 SRNet 族，流程相同：4 通道事件图像，经 2 × 2 无损像素重排加 1 × 1 卷积，四级残差块 CNN 与 FPN，在每个事件的整数像素位置取特征并拼接 7 维 token，再接部位、模板点、手上三个线性头。r_s2dx2 只改宽度：

| 设计 | 主干各级通道 | FPN 通道 | 逐事件头隐藏维 | 参数量 |
|---|---|---:|---:|---:|
| r_s2d（原版） | 16 / 24 / 48 / 64 | 32 | 64 | 137,252 |
| r_s2dx2（现行） | 32 / 64 / 96 / 128 | 64 | 128 | 580,116 |

其余设计（像素重排起点、FPN 之后的 3 × 3 平滑、各线性头）不变。参数量取自两个检查点的完成记录。

逐事件监督由标注 MANO 网格生成的表面伪标签提供，包含三角面、重心坐标和有效性。模板表面点由三角面顶点及重心坐标计算，部位类别由三角面的部位归属确定。

预训练损失为：

$$
\mathcal L_{ESM}
=\mathcal L_{part}^{CE}
+\mathcal L_{point}^{SmoothL1}
+0.5\mathcal L_{onhand}^{BCE}.
$$

部位和表面点损失作用于有效手部事件；手部有效性损失覆盖采样事件。表面点按 0.01 m 缩放，即在厘米尺度上计算坐标损失。

r_s2dx2 在每个数据集上用该数据集 r_s2d 的配方训练一次，种子均为 3407，只换网络结构，取最终（第 6000 步）检查点。后续状态网络训练和评测时，ESMNet 设置为评测模式且全部参数冻结。

- **仿真数据：**[tools/rfsu/router_sim.py](../tools/rfsu/router_sim.py) 原样调用 `tools/pfo/router_train.py`：S45 的训练数据与增强（depth 尺度模式、逐事件 XAG 源）、留出 ylf 的 8 名受试者、双卡 DDP × 512 包、Adam、学习率 0.001、500 步线性预热后恒定、BF16、6000 步。检查点 `outputs/rfsu/20261113/sim/routers/r_s2dx2/r_s2dx2_final.ckpt`，完成记录 [r_s2dx2_final.json](../outputs/rfsu/20261113/sim/routers/r_s2dx2/r_s2dx2_final.json)（SHA256 前 8 位 4d8d0b03，580,116 参数，2026-11-13 03:59 UTC 完成）。
- **真实 DAVIS346：**[tools/real/router_variant.py](../tools/real/router_variant.py)（`ROUTER_VARIANT=r_s2dx2`）在 `tools/real/evht_router.py` 的真实数据配方上只换网络：真实训练集的 12 名受试者、346 × 260 画布、真实数据的逐事件源，其余同仿真配方。2026-11-12 18:28 UTC 按事先写定的规则选入（[选择记录](../outputs/real/20261110/routers/chosen.json)）。检查点 `outputs/real/20261110/routers/r_s2dx2/r_s2d_final.ckpt`（文件名沿用 r_s2d）。
- **EvRealHands：**同一工具经 `tools/erh/run.py` 在 EvRealHands 的训练集上训练。检查点 `outputs/erh/20261111/routers/r_s2dx2/r_s2d_final.ckpt`（SHA256 前 8 位 db30a08d，2026-11-13 03:39 UTC 完成）。

原版的 r_s2d 仅在仿真数据上训练一次，种子为 3407，检查点 `outputs/pfo/20261103/routers/r_s2d/r_s2d_final.ckpt`。

来源：[预训练入口](../tools/pfo/router_train.py)、[表面映射网络实现](../semkine/pfo_router.py)、[监督与损失](../semkine/s45sm.py)、[原版预训练完成记录](../outputs/pfo/20261103/routers/r_s2d/r_s2d_final.json)、[真实数据路由器配方](../tools/real/evht_router.py)、[真实数据 r_s2dx2 完成记录](../outputs/real/20261110/routers/r_s2dx2/r_s2d_final.json)。

### 4.3 先路由采样、状态更新网络与从头训练

**先路由采样（RFS）。**对每个事件包，在选节点之前先做三步（[实现](../semkine/rfs.py)；规则的来历见[账本](EXPERIMENT_RECORDS.md#real-rfs-20261112)）：

1. 冻结的 ESMNet（r_s2dx2）对包内**全部**事件逐个判断，手上概率 p ≥ τ = 0.5 为手上；
2. 可采事件 = 手上事件，加上离同一包内某个手上事件不超过 NEAR 像素（方窗，以画布像素计）的手外事件；
3. 节点 = 原有的采样规则只在可采事件上运行：训练中为每包至多 2048 个节点的均匀步长规则，读数中为流式规则（见 2.4）。

NEAR 在仿真 240 × 180 画布上为 11 像素，在 346 × 260 画布上为 16 像素，两者都约等于 640 × 480 网格上的 30 像素。选节点不读前态、标签或相机内参 K。其余部分——事件 token、ESMNet 在节点处给出的模板面与重心、PFA、残差槽（未关联池）以及主干——都与原版相同。变化只有一点：残差槽现在只收手边的手外事件（路由器判手外而离手上事件不超过 NEAR 像素的事件，以及没有得到有效面地址的可采事件），远处背景不再进入；原版先在包内全部事件上采样，再由路由器判断，被判手外的节点都进残差槽。引入这一步的原因是真实 DAVIS346 的 local 序列只有约 15%–30% 的事件落在手附近，原版的采样预算大多花在背景上。

训练时路由器要判整批事件。为控制显存，配置里的 `RFS.MAX_ROWS` = 4,000,000 限制路由器分块补齐后的行数，超过就把分块对半拆；路由器逐事件的输出与分块无关（批归一化用推断模式，每包一张图，逐事件的头），只差浮点舍入。训练进程另设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。真实 DAVIS346 主行（`rfsr`，账本 #real-rfs-20261112 的臂）的配置未设这一上限。

**状态更新网络。**状态更新主干为 D1 R，输入沿用冻结 ESMNet 路由和 PFA 关联（节点由上面的先路由采样选出）。PFA 将事件的预测模板位置映射到前态 MANO 表面，结合事件射线形成几何残差。

主干包含 16 个 MANO 部位节点和 1 个手节点，隐藏维数为 128，在 62 条分型有向边上进行三轮运动学传播。边门由前态关节几何计算，同次前向的三轮共用；各轮消息映射独立。15 个手指节点分别读出自身的 3 维关节增量，手节点读出平移和根旋转增量，形成 51 维加性更新。

从头训练时，每个样本使用一个事件窗口以及窗口起点的标注状态加扰动，监督窗口末端状态。窗口在 30–300 ms 范围按对数均匀方式采样，接近有效段起点时可截短。

训练前态采用混合扰动：

| 概率 | 扰动类型 | 平移 | 根旋转与手指轴角 |
|---|---|---|---|
| 50% | 小尺度独立高斯扰动 | 各分量标准差 5 mm | 各分量标准差 0.05 rad |
| 30% | 大尺度独立高斯扰动 | 各分量标准差 50 mm | 各分量标准差 0.3 rad |
| 20% | 方向性扰动 | 随机方向，幅度 0–50 mm | 根和手指各自随机方向，向量幅度 0–0.3 rad |

方向性扰动中的幅度是对应分量组的向量范数，不能与逐分量高斯标准差混用。

状态损失为：

$$
\mathcal L_{state}
=\mathcal L_{rot}+\mathcal L_{trans}+2\mathcal L_{FK}.
$$

旋转项基于预测与标注四元数内积的平方，采用 `1 - dot²`；平移项为米单位的 Smooth L1；FK 项为腕部相对 MANO 关节的平均欧氏距离。优化时使用组合损失的 log10。配置中保留的旧 51 维加权 MSE 系数不应当作该主损失的权重。

来源：[先路由采样](../semkine/rfs.py)、[主干实现](../semkine/pmr_trunk.py)、[窗口与前态扰动](../semkine/dataset.py)、[状态损失与学习率实现](../model/model.py)。

### 4.4 rol5 闭环展开微调

每个种子从自身 D1 R（`rfsr`）第 6000 步检查点初始化，继续微调 1500 个优化器步。每个样本包含连续五个 50 ms 事件包，总跨度为 250 ms；训练进程同样装入先路由采样，五个包的节点都按 4.3 的规则选出。

第一步使用受扰动的标注前态，后续步骤使用前一步预测作为前态。梯度贯穿状态递推以及可微的 MANO、关联深度和几何统计计算。五步监督等权平均后取 log10：

$$
\mathcal L_{ROL}
=\log_{10}\left[
\frac{1}{5}\sum_{k=1}^{5}
\mathcal L_{state}(\hat{\boldsymbol x}_k,\boldsymbol x_k^{GT})
\right].
$$

同一个展开样本的五包共用几何增强参数及极性映射，前态噪声仅在展开起点采样。训练使用梯度检查点以控制展开的显存开销。

来源：[现行版 rol5 配置](../configs/rfsu_20261113/sim/pmr_rfsr_rol5_s3407.yaml)、[原版 rol5 配置](../configs/pmr_20261109/pmr_d1r_rol5_s3407.yaml)、[展开数据与训练实现](../semkine/tpd_rol.py)。

### 4.5 在线数据增强

| 增强项 | 设置 |
|---|---|
| 图像平面旋转 | ±15° |
| 尺度 | 0.8–1.25，采用 depth 模式 |
| 图像平移 | 每轴 ±14 px |
| 事件随机保留比例 | 0.25–1 |
| 极性增强 | 全局极性翻转、逐像素极性交换 |

ESMNet 预训练保留热像素注入，配置率为 0.0002；该率的单位为每像素、每极性、每毫秒的期望事件数。D1 R 状态更新网络及 rol5 的热像素注入率为 0。

这些在线增强与训练集中的 `_v2/_v3/_v4` 离线版本是两层不同的增强。

来源：[ESMNet 数据基础配置](../configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml)、[状态网络现行版配置](../configs/rfsu_20261113/sim/pmr_rfsr_s3407.yaml)、[状态网络原版配置](../configs/pmr_20261109/pmr_d1r_s3407.yaml)、[增强实现](../semkine/domrand.py)。

### 4.6 硬件与实现记录

机器共 8 张 NVIDIA L20（每张 46,068 MiB）。仿真数据上现行版的 ESMNet（r_s2dx2）、`rfsr` 和 `rfsr_rol5` 的每次训练均采用 **两张 L20**（双卡 DDP）：2026-11-13 种子 3407 在直连的 GPU 4、5，种子 3408 在经 gpu-borrow 借用的 GPU 0、1（训练日志记录了各 rank 的 `CUDA_VISIBLE_DEVICES`）。真实 DAVIS346 与 EvRealHands 上的状态更新网络用单张 L20（每步 512 包、梯度累积 2）。原版 EvHandTrack 的正式训练日志同样确认 ESMNet、D1 R 和 rol5 的每次训练均采用两张 L20。状态网络和 rol5 训练经确定性训练入口：启用确定性算法，关闭 cuDNN benchmark 和 TF32；现行版运行目录里的 `pmr_det.json` 记录了这些设置。

当前核对的运行环境为 PyTorch 2.1.0、PyTorch Lightning 1.9.5、CUDA 11.8；其中 PyTorch 2.1.0 也记录于正式成本测量启动文件。CPU 型号、主机内存容量与完整历史环境锁定信息未由本文列出的训练记录确认。

精度读数采用参考主干实现。`semkine/pmr_fast.py` 将统计与传播合并为较少的大算子，用于快速实现的成本测量；它与参考实现的对应关系记录于原版的正式身份文件。

**成本（待成本块）。**现行版的时延、MACs 和整体参数要在全部训练与读数结束之后，于 GPU 4–7 中一张无其他进程的卡上、一个安静的独占块里，与原版同块测（仿真用 [tools/rfsu/cost.py](../tools/rfsu/cost.py)，真实数据用 [tools/real/rfs_cost.py](../tools/real/rfs_cost.py)），本文不预填数值。测量用可部署形式：路由器每包只判一次（稠密主干一次、头对每个事件各一次），历史窗口的手上标记沿用上一包，节点处的路由直接取该次逐事件输出，节点选择算在每步之内；计时前先与读数实现（对包、历史窗口、节点各调用一次路由器）逐包核对。目前能确定的只有路由器本身：参数由 137,252 增至 580,116，整体参数相应由约 0.90 M 增至约 1.34 M（由路由器参数差推算）；真实画布上每包 2048 个事件时，路由器的乘加约由 1.38 亿增至 5.72 亿（账本 [#real-rfs-20261112](EXPERIMENT_RECORDS.md#real-rfs-20261112) 2026-11-13 02:21 UTC 条）。原版的成本为流式 17.45 ms / 0.102 G / 0.90 M（快速实现，同块 S43 为 10.90 ms，记录于 [账本](EXPERIMENT_RECORDS.md#evhandtrack-d1-cost-20261110)），只作记录。

来源：

- [原版 ESMNet 正式训练日志](../outputs/pfo/20261103/logs/p2r_train_r_s2d.log)。
- [原版 D1 R 正式训练日志](../outputs/pmr/20261109/driver1/logs/s1_d1_train_d1r.log)。
- [原版 rol5 正式训练日志](../outputs/pmr/20261109/driver1/logs/s3a_d1_train_d1r_rol5_s3407.log)。
- [现行版仿真 r_s2dx2 训练日志](../outputs/rfsu/20261113/logs/router_sim_r_s2dx2.log)。
- [现行版 rfsr 种子 3407 训练日志](../outputs/rfsu/20261113/logs/sim_rfsr_3407__train_rfsr.log)、[种子 3408](../outputs/rfsu/20261113/logs/sim_rfsr_3408__train_rfsr.log)。
- [现行版 rfsr 运行的确定性记录](../outputs/rfsu/20261113/sim/runs/pmr_rfsr_s3407/pmr_det.json)。
- [确定性训练入口](../tools/pmr/train.py)。
- [快速实现](../semkine/pmr_fast.py)。
- [正式成本测量环境记录](../outputs/evhandtrack/costs/evhandtrack_s3407__stream__pmrfast__default/launch.json)。

## 附录：真实 DAVIS346 数据的独立协议

新采集的真实事件数据采用独立于上述主实验的划分：

| 用途 | 受试者 | 使用序列数 |
|---|---|---:|
| 训练 | ch、dzy、lbh、lfz、lr、lyh、lyq、sbl、sm、wjh、wzh、zy | 192，含离线增强版本 |
| 调参 | wh | 4，基础录制 |
| 测试 | kqj、lqh | 8，基础录制 |

该数据包含 global/local 两类动作和正常光/暗光两类光照。论文只用正常光，并把三位留出受试者 kqj、lqh、wh 合并后报告（用户 2026-11-12）；两种光照合并的表只作记录。在该数据上训练并测试的方法为 EventHands、EvHandPose、EventHPE、FastMETRO-Event 和 EvHandTrack（账本 [REAL 修订 2](EXPERIMENT_RECORDS.md#real-amend2-20261110)、[修订 5](EXPERIMENT_RECORDS.md#real-amend5-20261111)）。本文主实验的 9 人/zgz 划分与此处的 12/1/2 划分应分别表述。

**EvHandTrack 在真实 DAVIS346 上的现行设置**（先路由采样 + r_s2dx2 + D1 R + rol5，臂 `rfsr` → `rfsr_rol5`，配置在 [configs/real_20261110/pmr/](../configs/real_20261110/pmr/pmr_rfsr_rol5_s3407.yaml)）：

- **画布与长度：**346 × 260 画布，同一虚拟针孔；原来按 240 宽画布以像素给出的长度（域随机化的平移、射线关联的带宽与前表面容差）按 346 / 240 放大。
- **路由器：**r_s2dx2 在真实训练集的 12 名受试者上训练（4.2），2026-11-12 18:28 UTC 按事先写定的规则选入。
- **先路由采样：**τ = 0.5，NEAR = 16 像素（约等于 640 × 480 网格上的 30 像素）；该主行的配置未设 `RFS.MAX_ROWS`。
- **状态更新网络与 rol5：**单卡，每步 512 包、梯度累积 2，每卡 6 个数据进程，不做验证轮，固定取最后一步；学习率、步数、损失与增强方式同仿真配方（带像素长度的增强参数已按上一条放大）。
- **种子：**主行 `rfsr_rol5` 为 3407、3408、3409 三个种子，已全部完成。图中消融里 `rfsr_rol1` 有三个种子，其余消融臂只训 3407、3408（2026-11-13）。
- **读数：**流式 E1，先测试序列再调参序列。

EvRealHands 的数据、划分与协议见账本 [#erh-prereg-20261111](EXPERIMENT_RECORDS.md#erh-prereg-20261111)；其上的 EvHandTrack 用该数据集自己的 r_s2dx2 与同一套先路由规则（NEAR = 16 像素），种子 3407、3408。

来源：[真实数据转换与主体定义](../tools/real/convert.py)、[真实数据划分](/data1/lyq/code/mesh/EventHands/data/hand_data_real_n346/splits_real.json)、[真实实验预注册](EXPERIMENT_RECORDS.md#real-prereg-20261110)、[真实数据 EvHandTrack 工具](../tools/real/evht.py)、[先路由臂工具](../tools/real/rfs.py)。
