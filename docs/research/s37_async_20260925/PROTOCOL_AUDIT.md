# 数据、指标、历史比较与延迟合同审计

证据截止 2026-09-25。状态标签：observed=本次读代码/产物或执行核查；reported=旧记录声明；inferred=推导；proposed=未执行方案。本文件不是新的模型精度结果。

## 1. 指标的实际定义（observed）

`model/mano_layer.py:100-169` 输出 778 mesh 顶点与 21 个 OpenPose 顺序关节；内部运动学/蒙皮使用 16 个 MANO 关节。15 个手指局部旋转、root 旋转、平移组成 51D 状态。关节不等于顶点；LBS 权重是运动学归属先验，不是当前事件对隐藏顶点的直接观测。

`semkine/eval_track.py:127,273-274` 中 MPJPE 为每帧把预测与 GT 各自减去 joint 0 后，21 点欧氏距离均值乘 1000。输入几何以米计，输出误差以 mm 计；只有平移对齐，不做旋转/尺度 Procrustes 对齐。相机系全局旋转仍影响根对齐误差。

`local/global` 由 `zgz_local/zgz_global` 动作序列划分，非坐标系名称。`tools/make_s36_row.py` 按帧数加权聚合，同种子的 local/global 必须出自同 checkpoint。两种子均值只能作为重复性摘要，不能把某种子 local 与另一种子 global 拼成满足门槛的模型。

MPVPE 沿用 `root_align(pv)`，实际减 mesh vertex 0。不能悄悄改为 wrist alignment 后沿用旧结果。目标的 MPJPE 定义不变；若额外分析腕对齐 mesh，仅在诊断中另名记录。

## 2. 截图与历史最好（observed + 可比性边界）

截图 `render + SO3/FK Loss + LBS + DomRand` 对应 `outputs/hand_data51/track_render51_dr_sem_rep2/eval_step1000/track_metrics_step50.json`，同一 checkpoint 的 local=15.104646682739258，global=10.028024673461914，overall RA=12.387967877774626。严格 `<15.1` 不通过；不能把显示的 15.1 当作已经达标。10.66 是新目标阈值，不是该 checkpoint 的 global。

`outputs/hand_data51/main_table.json` 保留的是多个已选代表点，包含单副本/不同配方，不是全体模型双种子同预算排名。S37 当前臂由用户指定，不等于所有路线历史最优。

截图 CNN 配方原件目前可读：`../EventHands/configs/eventhands_track_render51_dr_sem_rep2.yaml`；其训练元数据指向该文件。旧训练预算 3000 updates × 2048 batch，S37 为 6000 × 1024，样本呈现量相同但优化器更新次数、LR、warm start、损失不同。不能把架构差距全归因于图网络。

公平对照分层：

- **H（历史重现）**：固定历史权重/配置、两条 zgz、旧初始化和 shape/时间合同，仅核验已有数字。允许保留旧 CNN 输入；不用于新模型调参或完整系统合法性证明。
- **C（严格因果开发）**：训练原始事件，观测只到查询时刻，固定均值 shape 或由过去事件估计，因果初始化，无 GT valid-run 自动重置。候选间观察/监督/预算相同。
- **F（独立最终验收）**：冻结模型后首次打开新采集数据；与 S37/CNN 在同输入可用性/初始化合同下比较。无新数据时标 HOLD，不能把历史 zgz 改名 sealed test。

H/C/F 的指标公式一致，输入信息不同，不混表解释为方法效果。用户未确认新数据时，新开发集可来自旧训练受试者，必须重训对照以排除预训练接触开发集的优势；不得称新受试者泛化。

## 3. 时间、shape 与初始化（observed）

`dataset.py:267-281,365-369` 用 `[start_ms,end_ms+1)` 的事件，标签 `pos51[end_ms]`。`eval_track.py:193,242` 同样存在观测到标签之后 1 ms 的旧合同。

本次只对四条训练序列取固定窗口核验：`.research/s37_async_20260925/audit_contracts.json`，保留每窗标签之后事件计数、像素合法性、时间顺序和每 ms 事件负载。实际的正计数证明需要另立严格因果合同；不推断对精度的影响。

严格合同建议查询时刻 q 整数微秒，只消费 `t < q`，标签取 q 对应的 ms 样本；不对旧标签时间自行插值生成更有利真值。截止时刻无事件也由 wall-clock/watermark 触发输出，不等待未来事件证明过去已结束。

旧 `eval_track.py:177,193` 把序列 GT betas 与 GT+noise 起点输入模型。最终系统必须替换为明确先验/事件初始化，shape 来自固定训练均值或因果预测；GT betas 可作训练监督和 GT mesh 构造，不能读入预测 mesh 分支。有效 GT 缺失只控制评分，不能触发由标签告知的恢复重置。

## 4. 完整延迟合同（proposed，资源配额待确认）

设备候选：单张无其他计算进程的 NVIDIA L20；CPU 输入线程最多 4，模型 FP32 为精度对照，混合精度必须先过轨迹回归。显卡不能仅看 utilization=0，须检查进程和显存。

先用已授权训练数据冻结负载包络，再测未调参 replay。禁止事后把超限帧排出包络。至少覆盖静止、低事件率、突发、遮挡、恢复、全局刷新。原始事件全部接收；任何准入/采样另报丢弃量与对应精度控制。

为每个实际输出保留时间戳：最早尚未得到响应的事件接收、包触发、排队开始/结束、H2D、预处理、图/缓存、关联和读出、状态更新、MANO decode、最终 mesh 可用。定义：

`L_complete = t_mesh_ready - t_first_unserved_event_arrival`。

另记 query deadline 到 mesh_ready 的年龄，不用它替换完整延迟。没有新事件的定时输出测 deadline 到输出；warm start/recovery 也计入其自身输出记录。GPU CUDA events用于分解，完整 wall-clock 用单调时钟并正确同步；不以缩放数或 min-of-means 验收。

候选触发上限先定 1 ms，剩余完整服务预算至多 6 ms；这是设计约束而非保证。若服务速度低于输入/输出频率，排队会使 7 ms 不可能。长历史记忆不等于等待 50 ms；可以利用已经到达的过去 50 ms，但必须每个输出及时消费新到事件。

全部样本记录 P50/P95/P99/max、超限率、事件负载、吞吐和队列长度。任何实测 L_complete>7 ms 则该包络本轮门 FAIL；max≤7 ms 仅支持受测包络，不宣称硬实时证明。

旧 `make_s36_row.py` 预加载 GPU 的 forward_packet、固定 prev、均值最小值及 1.75 ms anchor 缩放不含这些阶段，继续只保留在历史主表，不能复用为新 7 ms 验收。

## 5. 当前未知和解阻条件

1. 独立新数据存在性、来源和许可：待用户说明；无则最终泛化 HOLD。
2. GPU 配额/训练预算：待用户说明；不将现存进程当作可抢占任务。
3. 严格因果与无 GT shape/init 下旧基线误差：未重测，不把历史数值直接搬过来。
4. 控制流到输出的真实成本和恢复能力：未测；单元测试通过不能替代。
5. shape/scale 的单目可观测边界：MANO 尺寸与训练统计是先验，不能把尺度估计当独立深度观测。

## 6. 审计产物身份

`.research/s37_async_20260925/source_identity.json` 哈希绑定本次读取的核心源码、配置、MANO、split 与两种子选中 checkpoint。初始 git status 另存，所有用户修改保留。大事件数组采用现存只读 mmap，此阶段只记尺寸、shape、路径；正式训练前需数据 manifest 的内容身份，不把路径/mtime 当完整内容哈希。
