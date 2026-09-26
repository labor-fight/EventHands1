# Token / 邻居 TorchScript 执行诊断：独立合同审查

2026-09-25，`audit_s37`。本次只读当前代码、G006 JSON 和既有诊断文档，并用标准库解析已有文件；没有运行 GPU、训练、TorchScript 或数值反例，没有修改核心实现。以下为待执行方案的审查，不是已通过实验。

## 已确认的动机和范围

**Observed**：直接解析 `.research/s37_async_20260925/g006_c1_captured_4000.json`，除首次 append 外共有 34 次，`append_ms` 合计 `178.534378`；该输入共 482 个事件，微批周期 1000 us、query 周期 4000 us。固定解码图不足以解除这一版本的 append 服务瓶颈。

**Observed**：当前 `semkine/streaming.py` SHA256 为 `24350b44d0813fbc758d0c5e798278e15595692030cba0a04d1b41a4ed44110b`。纯 tensor token 在 `:181-220`、建边在 `:226-245`，所有输入/时间/版本检查先在 `:122-170` 和 append 入口执行，空 append 在调用 helper 前返回（`:250-253`）。

**Inferred**：在 scratch 内把这两个纯 tensor 阶段改由 TorchScript 调度，是有界且可证伪的执行诊断。它不保证减少 GPU 算子、排序复杂度、动态输出长度同步或整体时延。保留全部事件、1 ms 输入微批、50 ms horizon、W/K/N、已有 checks、输出接口与旧核心；不得把编译失败静默转回 eager，或为性能改变函数的数学定义。

## 参考独立性与容易伪等价的点

1. **原文件不改，不代表 reference 未变**。当前 `full_reference` 在 `streaming.py:323` 经 `self._tokens(...)` 调用 token 函数。若 scratch 替换同一 runner 的 `_tokens`，其“原 full reference”也会调用新实现。必须保留一个完全未替换的原 runner 作为 oracle，核对该实例 `_tokens` 的方法身份；新路径与原路径各用独立 state。不得用新 helper 同时生成两侧 token 后只比较后续网络。
2. **浮点表达式一致不保证执行一致**。`log1p(gap_f)` 不能改成 `log(1+gap_f)`；先 int64 相减再转浮点不能交换；`dp.square().sum(-1)` 的归约顺序、除法/乘倒数、融合 FMA、clamp 的位置都可能受优化影响。记录实际 JIT 执行图、fusion 配置和版本。改变距离末位可改变 topk 的邻居，而不只是让连续特征有微小误差。保留原 `topk(min(k,W), largest=False, dim=-1)` 及默认排序行为；不加 epsilon、tie-break 排序或 stable-neighbor 新规则来“修复”此诊断。
3. **dtype 和零也属合同**。索引/时间保持 int64，mask 保持 bool；按原逻辑 FP64 输入使用 FP64 时间计算，当前 GPU 候选可显式限制 FP32，而不能悄悄接收其他 dtype。`torch.equal` 把 +0 和 -0 当作相等；如报告“逐位”，须另外检查零的 signbit 或整数位视图。原 `-age.float()/H` 的同时间戳结果可能是 -0，不能未经记录就统一改成 +0。原 invalid 边由 `where` 清为零，不能把不同 invalid 浮点垃圾掩盖成普通 allclose 通过。
4. **SAE 依输入 ordinal 而非仅时间排序**。`packed=key*(M+1)+ordinal` 中，历史每 key 项 ordinal 为 0，当前事件为 1..M，`searchsorted` 用 left 边界并减 1（`:188-204`）。因此同时间戳仅能看到已接受前序，age=0 与 never 不同。不能改为按 timestamp 合并、右边界搜索、先全批更新 SAE 后取 age，或沿用别的块长的 multiplier。缺 key 时先用当前 ts 代替 prev 再相减，避免对无关旧时间做可能溢出的差值。这些都不能靠最终 h 的平均误差验收。
5. **动态形状与缓存**。使用 `script` 的动态参数，不能用单一示例 `trace` 冻结 M、旧 SAE 大小 P、是否有 last_timestamp、空缓存分支。保留 sparse sorted key/最后时间、每层最近 W、读出最近 N 的所有权和更新规则。M/P 变化造成新图编译或 fallback 必须可见并计时；不能为消除动态形状而引入完整稠密栅格、padding 伪事件、缓存过期规则或采样。

## 只需三类输入

### 1. 时间与 sparse SAE 的对抗小序列

同一像素重复事件、交替极性、缺失另一极性；相同 timestamp 的事件横跨 append 边界，包含首包与已有历史。时间差覆盖 `0, 1, H-1, H, H+1` 和一个长静止间隔，并对同一序列整体加 `9_000_000_000_000` us 原点偏移。穿插合法空 append，要求旧 state 身份与内容保持；空包不因 helper 的空 searchsorted 引入新行为。

核对原 token 七通道、SAE keys/timestamps、last_timestamp、事件计数逐项相同。将 first-ever 的 same/opp age=H、已见等 timestamp 的 age=0 与首事件 global gap=0 作为手算锚点，而不仅做两个实现互相对照。原输入拒绝 tests 仍由原 wrapper 执行；无需把 GPU预算花在复制所有既有非法输入排列。

### 2. 图并列与可变分块的小前缀

构造重复像素和对称位置的等距离前驱，包含相同 timestamp、窗口不足 W、刚好 W/超过 W、时间恰好 H/超过 H 的边。保持事件顺序，对同一前缀使用逐事件、跨 W 边界的变长块（例如 17/31/69），再附一个跨读出 N 淘汰的前缀。分块只是数值 Debug，不变更部署 1 ms 微批。

将局部索引映射到已接受事件全局 ID 后，先比较**有效**邻居 ID、排列、mask 和 dp，再比较各层 h/读出及原独立递推门。只比较无序邻居集合不足以保证后续浮点归约一致；并列点的历史特征也未必相同。invalid 邻居 ID 不具有观测关联语义，按原 mask 及零 dp 合同检查，不将任意 padding ID 当成真实连边。如合法邻居/顺序改变，不以 h allclose 掩盖它。

### 3. 已冻结真实负载与较大 sparse 字典

复用 G006 相同 1 ms 序列与原 8321 事件峰值：仍顺序分为 `8192+129` 全部接受，不截断，不增加微批等待。另在相同 helper 对照中保留一个较大的“实际已访问 key 集”历史状态，检查 P 增长后的排序/布尔压缩成本与正确性；不必构造新的大规模训练实验，也不将大字典合成准备成本混进真实输入服务。

先在相同历史/输入上比较原 helper 和 scripted helper，再在同一冻结模型下做集成前缀检查。端到端精度数值门保留原 `atol=rtol=1e-5`，不扩大；若额外声称 helper 逐位一致，应预先把该性质作为单独 gate，并明确检查 signed zero。完整推理服务的采样/排队/输出口径仍沿用已有协议。

## 判停与计时解释

- **语义判停**：任何 missing/zero-age、时间单位/原点、合法邻居 ID/顺序、mask、SAE 最后时间或事件计数不符，立即停止这一执行候选。任何 NaN/Inf、状态别名/静默回退、原 GPU 递推/梯度必需门失败，也不得继续正式服务宣称或筛查训练。保留最小失败输入和两侧中间值。若差异来自编译 fusion，仅能明确定位后作一次有依据的局部调度修订并重过相关 gate；不能换公式、放宽容差或同时修改 reference。
- **实现判停**：若目标环境不能 script 这些动态 tensor 操作，或只能依赖未记录 fallback/删除 checks/静态 trace 特化通过，则本诊断失败。不能把准备出的新计算路径当“同算子调度优化”继续推广。
- **成本判停**：在固定、非 profiler 的成对测量中，若 token+edge 及完整 append 均没有明确改善，或已知剩余 append 成本仍造成明确持续积压，停止扩大同版本回放与任何训练。已有 profile 的 CPU stage、CUDA time、嵌套 stage 不可相加或替代 wall time；减少 Python 调用/launch 数字本身不是速度证据。首次 script/优化/运行、动态形状重专化成本单列且冷服务保留，不能藏进不计时 warmup。小 helper 加速但完整服务失败，记录为局部工程证据，目标仍未达到。

主代理应在首次 GPU 执行前固定配对次数与无明确改善的停止口径；本审查不追加无限复测预算。剩余 Debug 额度支持一次判别，并不构成消耗额度或开训的理由。此时 screen/train 均尚未获得工程门通过。
