# MV2 · M9 预注册草稿：前缀稳定查询合同 + 上下文年龄前沿 + 发射受限成本账（PSQ · PAS/CAF · SORL/LBA）

> 草稿由实现代理写成，供主会话合并进唯一的 `#mv2-prereg-20261026` 并冻结。本稿之前**没有任何效应读数**。实现阶段只运行了 `--smoke`：层一的合成单测（不碰数据）、各工具的 bit 级自检与 `decide()` 管线（合成表）。披露：`m9_psq.py --smoke` 在 FIT7 序列 `lyh_local` 上跑过一次 S39 栈的 phase-0 对 `evalx.run_sequence` 的逐位复现（只输出 identical=True，未输出 RA）；`m9_age.py` 在 FIT7 序列 `ch_local` 上量过吞吐（只记时间）；`m9_cost.py --smoke` 静默执行了 S39 / S39+rootanc / S39+SLR 的计数与图断裂路径（未打印任何计数）。
> 所有工具：解释器 `/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python`，`CUDA_VISIBLE_DEVICES=""`，正式模式先 `common.verify_protocol()`，写 `outputs/mv2/<probe>/result.json`（独占）与 `receipt.json`；`decide(result)` 在各模块内。种子固定 20261026（合成流）。

---

## 1. `tools/mv2/m9_psq.py` —— PSQ（来源 `P5-E9-R1`）

### 目的
检验"对任一查询 q，输出只依赖 t_cut ≤ q 之前写入的事件与已提交前态；同一事件流在不同包边界相位下 RA 的变化在种子噪声内"。层一检验写路径的前缀稳定（逐位），层二检验节拍不变性。

### 数据
- 层一：合成事件流（泊松计数、热斑、含空毫秒与同像素碰撞），不读任何数据文件；S39 架构用随机初始化的 `MNISTModel(config_resolved.yaml)`（不读 checkpoint）。
- 层二：SELECT（ycy、ylf）的 `_local` / `_global` 四条基础录制；冻结 S39 三种子 step 6000；CPU；每序列一次完整闭环（`valid_runs_ms` 内，1229–1434 步）。不读 zgz。

### 被测栈（S39，读源码确认）
`ET.build_lnes(events, offsets, end, w)`，w = min(200, end + 1 − 段首)（W200 裁段首，`evalx.window_of`）；Count50 = `offsets[end+1] − offsets[end−49]`；`AdaptiveFilter.from_spec(model, ramp1000.json)`（DT2 事件量自适应滤波 + 证据门 ramp1000，spec 标签 `af9f41a0` 由工具校验）。这一个模块同时是"滤波"与"门"。

### 统计量与公式
- 层一（每项 float32 `np.array_equal` / `torch.equal`，无阈值）：T1 `build_lnes` 对 t_cut 之后事件改动不变（W ∈ {50, 100, 200, 裁到段首}）；T2 Count50；T3 滤波+门（桩跟踪器）；T4 段首裁剪不读段首之前的事件；T5 ms 重基准对时钟原点平移不变；T8a 随机初始化 S39 + 滤波的 9 步闭环，t_cut 之前各步逐位相同；T8b 对照：t_cut 之后输出确实改变。附加描述：T6 `RingLNES`（环形缓冲 + 固定窗重基准，最小修复）逐位等于批量写端、对过去查询前缀稳定；T7 S43 的 SAE 平面写端在"已写入更晚事件"后查询过去时刻**不稳定**（描述 S43，不是 S39）；I1 CPU 前向的批组成是否逐位不变（B=4 与 4×B=1；已知为 False，决定后续探针必须保持相同批组成才能逐位比较）。
- 层二：闭环与 `evalx.run_sequence` 同（`closed_loop`，`phase=0` 逐位等于评测器，正式运行在 ycy_local 种子 3407 上复核）；包边界相位 φ：ends = a + 49 + φ + 50k。φ ∈ {0, 10, 25（受门）, 50（同网格对照）, 1（微移对照）}。每序列初态噪声种子固定（同序列跨 φ、跨种子配对）。RA(seed, 域, φ) = 该域两个受试者序列逐步 RA-MPJPE 的步数加权均值；ΔRA = RA(φ) − RA(0)。

### 对照与零模型
层一：C1 一毫秒前瞻的写端必须被判非稳定；C2 窗内改动必须被判改变；C3 前瞻 Count 必须被判改变；T8b。层二：φ = 50 与 φ = 1 同口径（不受门）；φ = 50 的网格与 φ = 0 相同，只差初态历史，用来估"回路历史噪声底"。

### 门
- 层一：任一写步骤非逐位前缀稳定 ⇒ 登记 S39 为"批处理评测"身份（`batch_evaluation_identity`），最小修复 = `RingLNES`（须等于批量写端）；全过 ⇒ `pass`。
- 层二（每域、每个受门 φ ∈ {10, 25}、三种子）：全部 |ΔRA| ≤ 0.03 mm ⇒ `invariant`；三种子 |ΔRA| 全 > 0.15 mm 且同号 ⇒ `cadence_dependent`；其余 ⇒ `descriptive`。
- 总判：层一 pass 且 4 个受门格全 invariant ⇒ `contract_holds`；任一受门格 cadence_dependent ⇒ `kill_frontier_contract`（只报 replay benchmark）；其余 `descriptive`。若某域的 φ = 50 对照本身三种子同号 > 0.15，则在 notes 标注"回路历史噪声底不低于门，节拍读数不可分离"（不改变裁决类别，由主会话解读）。

### 阈值来源
0.03 mm = S44 P0b 预注册的 h10 − h50 门；0.15 mm ≈ 账本已观测 h10 − h50 = +0.165 / +0.194 / +0.188（`#s44-p0b-results-20261020`）；种子 SD local 0.379 / global 0.261（KS0）仅作比较尺度。逐位相等无阈值。全部取自计划 §3 M9。

### 工具有效性（否则 `tool_defect`，不得判 pass）
层一四个对照全检出；层二 φ = 0 逐位等于 `evalx.run_sequence`（同进程、同线程数）。

### 预计 CPU 时间
层一 < 10 s；parity 一条 SELECT 序列 ~1 min；层二 60 个闭环（3 种子 × 4 序列 × 5 φ），`--workers 24 --threads 2` 约 3–5 min 墙钟（~2 CPU·h）。

### 与计划 §3 的差异及理由
1. 计划预期"S39 前端很可能不前缀稳定"。读源码后，`build_lnes` 只读 `offsets[start .. end+1]`、Count50 只读 `[end−49, end]`、滤波只依赖 (x, n)，**按构造对 t_cut 之后的事件逐位不变**；合成单测也如此（不是数据读数）。真正的前缀风险不在写端，而在 (i) 网格锚在带标签的有效段首 `a`（部署需要"跟踪起点"），(ii) 包等待（Count50 要整包），(iii) S43 的 SAE 平面写端对过去查询不稳定。层二的节拍读数承担"是否节拍相关"的判定。
2. 增加 φ = 50 与 φ = 1 两个不受门对照，落实风险栏的"与 50 ms 内等价平移对照"。
3. CPU 闭环数值与历史 CUDA 评测不逐位可比，故不与账本 RA 比，只做 φ 间配对差。
4. 种子 SD 不用于门，仅作尺度。

---

## 2. `tools/mv2/m9_age.py` —— PAS / CAF（来源 `P5-E5-R2`、`P5-E6-R1`）

### 目的
测冻结 S39 的 E3 单步误差对前态年龄 a_p 与证据年龄 a_e 的依赖，判"精度轴"是否存在，并得到零交叉年龄 a_0 与捕获年龄 a_cap。

### 数据
SELECT 四条基础录制；冻结 S39 三种子；`SeqSamples(every=8)`（共 ~640 个端点，公共子集约 540）；W200 窗，prev = GT(end − 50 − a_p)；CPU；批大小固定 64。公共子集：end − 50 − 200 − 17 ≥ 段首且窗口满 200（所有年龄臂用同一批样本，配对）。

### 统计量与公式
- 误差 ε = 根对齐 MPJPE（mm），按域（local/global）合并样本取均值。
- 臂 `aged`：a_p ∈ {0, 10, 25, 50, 100, 200} ms；臂 `extrap`：对 GT(end − 50 − a_p) 做 oracle 常速外推 a_p ms（平移线性、根 SO(3) 左乘、手指轴角线性；速度取该时刻前 17 ms 的 GT 差，即一个 60 Hz 源帧）——预测 end − 50 时刻状态的上界，不可部署；臂 `extrap_q`：外推到查询时刻（a_p + 50 ms），仅描述。
- 证据年龄：W200 内事件年龄 a_e = 199 − ms_rel；`late_X`（只留 a_e ≥ X，X ∈ {25, 50, 100}）与 `recent_X`（只留 a_e < X，X ∈ {25, 50, 100, 150}），保留原 200 ms 归一化；与 a_p 全交叉，描述（不设门）。
- 持续性误差 d(a_p) = 持有 GT(end − 50 − a_p) 的 RA 误差；增益 G(a_p) = d(a_p) − ε(a_p)；a_0 = G 首个过零年龄（线性插值，无则 > 200）。
- 精度轴增量 inc50 = ε(a_p = 50) − ε(a_p = 0)，每种子每域。
- a_cap(c)：GT 在 end 与 end − a 间 21 关节平均三维位移首次 ≥ c 的 a（5 ms 网格，≤ 400 ms），c ∈ {14, 19} mm（KS4 / AURA B 的捕获范围），取样本中位数（并报 p25 / p75 与截尾率）。
- 外推收回比 = (ε_aged − ε_extrap)/(ε_aged − ε_aged(0))（a_p > 0 且分母 > 0）。

### 对照与零模型
a_p = 0 臂必须逐位等于"独立构造的普通 E3"（`build_lnes` 直接构图、`pos51[end−50]`、相同分块批）；`extrap` 在 a_p = 0 同样逐位等于；恒等证据掩膜 `recent_200` 经 transform 路径逐位等于；普通 E3 重复一次逐位相同。

### 门
- 精度轴（每域，SD = 0.379 / 0.261 mm，KS0）：三种子 inc50 全 ≥ 2 SD（0.758 / 0.522 mm）⇒ `alive`；三种子 inc50 最大值 < 1 SD **或** 三种子符号不同 ⇒ `dead`；其余 `indeterminate`。总判：两域都 alive ⇒ `alive`；两域都 dead ⇒ `dead`（M9 永久降为纯系统合同、"两速率改善精度"永久不主张）；一域 alive/dead 另一域否 ⇒ `mixed`（描述，不触发永久杀）。
- CAF（每域）：三种子 G(50)/G(0) 全 ≥ 0.5 且 a_cap(19 mm) ≥ 100 ms（= 2 个包间隔）⇒ `frontier_ok`；三种子 G(a_p = 50) 全 ≤ 0（一个包间隔内过零）或 a_cap(19 mm) < 50 ms ⇒ `positional_channels_downgrade`（任一域触发即采用：M1 PA2 / M3 / M4 位置型通道只能作全局旁路辅助）；其余 `indeterminate`。
- 外推价值（描述）：a_p = 50 的外推收回比三种子全 ≥ 0.5。

### 阈值来源
SD、MDE：KS0（`#kstar-results-20261025`，规划近似）；2×/1× SD：计划 §3；0.5 倍与 2 倍包间隔：设计常数【待用户确认，计划标注 "待验证"】；包间隔 50 ms 由 W200 "4 倍重叠" 推出并已对源码核对（`evalx.run_sequence` 的 STEP = 50）；0.5 收回比：占位【待验证】；14/19 mm：KS4 / AURA B。

### 工具有效性
`a0_bit_exact`、`extrap0_bit_exact`、`identity_mask_bit_exact`、`plain_repeat_bit_exact` 对每个种子均 True；每域样本 ≥ 100。

### 预计 CPU 时间
约 0.5 s/样本/条件组（8 线程）；每种子 ~5 min；3 种子并行（`--workers 3 --threads 8`）约 5–8 min。

### 与计划 §3 的差异及理由
1. 计划写"a_p ∈ {0, 8/10, 17/25, 34/50, 100, 200}"，取任务书的 {0, 10, 25, 50, 100, 200}。
2. 计划只写"外推臂"，这里给两种：`extrap`（外推回 prev 的原时刻，a_p = 0 时必须等于普通 E3，满足"a_p = 0 逐位等于 E3"）与 `extrap_q`（外推到查询时刻，描述）。
3. 双域判定的合并规则（alive 要求两域；永久杀要求两域）是实现代理的选择：计划未写域如何合并。保守方向：不因单域否定而永久杀。主会话可改写此条。
4. a_cap 以"GT 关节平均三维位移"换算（KS4 的 14 mm 是像面平移锚点偏移，这里取同量纲的三维位移）；描述性定义，须在冻结前确认。
5. 公共子集排除每段前 ~5 步（需要 250 ms + 17 ms 的段内历史）。
6. 种子数 = 3，没有独立的"种子 SD"实测；门里的 SD 是 KS0 的 zgz 闭环 RA 种子 SD，与 E3 增量并非同一量，但这是计划指定的尺度。

---

## 3. `tools/mv2/m9_cost.py` —— SORL / LBA（来源 `P5-E10-R1`、`P5-E2-R2`）

### 目的
零数据静态账：S39 / S39+rootanc / S39+SLR 的 aten 发射数、主机同步、数据相关形状算子、dynamo 图断裂、CPU 2/8 线程前向延迟；判 M1（SLR）的 Latency 主张是否站得住。

### 数据
无。合成稀疏事件图（5 % 占据、固定种子）、近均值前态、betas 零、默认 K；随机初始化模型：`configs/dt/dt_dz_l3.yaml`（`MNISTModel`）、`configs/dt/dt_dz_l3_rootanc_2k.yaml`（`MNISTModel`）、`configs/mv2/mv2_slr.yaml`（`semkine.slr.SLRModel`）。不读事件、标签与 checkpoint。

### 统计量与公式
- `TorchDispatchMode` 计数：total aten 调用、view-only（schema 返回别名且非写）、alloc-only（empty*）、`launches` = total − view − alloc；主机同步 = `_local_scalar_dense` / `is_nonzero` / `equal`；数据相关形状 = `nonzero` / `unique*` / `masked_select` / `argwhere` + 布尔掩膜 `index` / `index_put`。
- `torch._dynamo.explain` 的 graph_break_count，`torch.compile(fullgraph=True, backend="eager")` 是否成功（仅追踪，不需 C++ 编译器）。
- 静态源码 grep（`.item()` / `.cpu()` / `.numpy()` / `.tolist()` / `nonzero` / `unique` / `np.`）前向路径函数。
- 两种形态各测一遍：`default`（评测器调用的 `model(x, prev)`）与 `deployed`（`model/deploy_fast.py::enable_fast_render`：DT2 部署形态，Python-int MANO 链 + 无布尔索引的 `RenderFast`）。
- CPU 前向 batch 1、`no_grad`，p50 / p99 于 2、8 线程；每个线程数一个固定亲和性的新进程；模型顺序轮转交错；A/A 对照 = 独立构造的 S39 孪生模型。

### 对照与零模型
玩具模块（(x·2+1).sum()：3 次发射、0 同步；含 nonzero + 布尔索引 + `.item()` 的模块：已知计数与图断裂）；S39 自身作基线；A/A 延迟对照。

### 门（作用于 S39+SLR 的 `deployed` 形态；其无法构建时读 `default` 并令 gate_deployable = False）
gate_deployable；gate_sync（同步 + 数据相关形状 = 0 且 `slr.py` 内无静态 `.item/.cpu/.numpy/.tolist`）；gate_dds（数据相关形状 = 0）；gate_breaks（图断裂 = 0 且 fullgraph 追踪成功）；gate_launches（launches ≤ 3 × max(S39, S39+rootanc 的 launches)）；gate_cpu8（8 线程 p50 相对 S39 增量 ≤ +50 %【提议值，须预注册，非账本阈值】）。`latency_claim_deleted`：任一同步 / 数据相关形状、不可部署，或 8 线程增量 > +100 %。A/A 对照差 > 10 % ⇒ 延迟门 `undetermined`，总判不得为 pass。p99 增量仅描述。

### 阈值来源
0 同步 / 0 数据相关形状 / 0 断裂：CUDA graph 可捕获的必要条件；3×：S43 稀疏 304 对稠密 216 kernel（偏离 9）的 1.4× 放宽；+50 % / +100 %：计划提议值与杀伤规则。

### 工具有效性
计数器与 dynamo 对照返回已知答案；S39 与 S39+rootanc 能构建并运行；A/A ≤ 10 %。

### 预计 CPU 时间
计数 + dynamo ~1 min；延迟 2 个线程配置 × 300 次 × 8 个模型，约 4–8 min（机器负载会拉长，负载写入 receipt 的 `loadavg`）。

### 与计划 §3 的差异及理由
1. **未实现**：查询年龄六项解析分解、合成突发队列 p50/p95/p99、41k 事件/包的批量 warp p99、"墙钟 ≈ a·发射数 + b·同步数 + c·MAC" 三参数回归。理由：批量 warp 属 M3（尚无实现）；分解与队列需要部署实测量；回归需账本 ≥ 5 个延迟点（只读账本、不属零数据探针工具，可由主会话另做）。任务书只要求前述计数 / 延迟项。
2. "M1 各梯阶变体"：目前只有一个 SLR 配置。冻结前若有其他梯阶配置，请加到 `VARIANTS`。
3. 已知源码事实（不是读数）：`semkine/aura.py::visible` 使用布尔掩膜 `index`（`[ok]`），`ManoLayer` 默认链用张量索引（每次前向 15 次 `.item()`，`deploy_fast` 文档所述）。故 SLR 的 gate_dds / gate_breaks 在没有静态重写前很可能不过——预注册应把它当作"预期会触发杀伤规则"，不当作失败。
4. CPU 发射计数与 L20 graph 墙钟只是代理；CPU 通过不保证 graph 墙钟通过。
