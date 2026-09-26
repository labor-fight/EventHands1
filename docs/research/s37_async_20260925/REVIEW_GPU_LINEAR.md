# C1 固定顺序 CUDA affine 的独立 Debug 审查

审查日期：2026-09-25。审查代理：`audit_s37`。本次只读实现、脚本、已落盘日志和 JSON，并用 CPU 文件哈希核对来源；**没有运行 GPU、训练或新数值反例，没有修改核心实现**。这是工程合同审查，不是精度、时延或新实验臂结果。

## 结论与适用范围

在已记录的 NVIDIA L20、PyTorch 2.1.0、FP32、TF32 关闭、无 AMP 的执行条件内，现有证据支持 C1 使用固定 tile 的 affine primitive 解决本轮发现的分批前向舍入差异。原先 `atol=rtol=1e-5` 没有放宽；独立完整前缀与缓存路径保留，原失败没有删除。现阶段没有从所读实现中发现仍可否定这些已测合同的确定缺陷。

这不等于任意长度、所有 GPU/编译器、所有参数值下的数学逐位保证。尚不能由此证明精度改善、7 ms、长程运动恢复、混合精度训练，或多步 MANO 训练梯度等价。

证据标签：**observed** 指本代理实际读取代码/文件；**reported** 指文件记载的他人执行结果，本代理未重跑；**inferred** 指代码推导；**proposed** 指下一阶段应执行的检查。

## 根因与最小修改是否匹配

- **observed/reported**：`.research/s37_async_20260925/gpu_graph_probe.json` 记录 token、有效全局邻居编号、mask、边偏移逐位一致；相同消息的均值归约亦一致。`gpu_debug_g001.json` 保留原递推失败，`gpu_numeric_probe.json` 保留 affine 分批诊断。这些证据支持先修数值运算顺序，而不是修改图或放宽递推门槛。
- **observed**：`semkine/streaming_linear.py:14-29` 固定 `16 × 32` 输出 tile、`32` 输入归约块；仅 rows 控制边界 mask，`do_not_specialize=['rows']`、固定 warps/stages，无依 batch shape 选择的 autotune，`tl.dot(..., allow_tf32=False)`。同一行被重新分块后保留相同输入维度归约布局，是分批稳定性的合理机制。
- **observed/inferred**：`semkine/streaming.py:94-100` 仅在显式 `fixed_fp32` 且 CUDA 时调用该函数；原 CPU `nn.Linear` 与旧模型模块未被替换。`streaming_tracker.py:83-85` 只为 C1 runner 选择它。参数对象仍来自原 EventGNN，不新增参数。
- **observed**：`streaming.py:323-358` 的 full reference 从完整前缀独立建图、收集全前缀消息；没有调用 append、读取增量缓存或复用缓存前向结果。两条路径共享已经独立检验的 affine 原语及 token 定义，符合本轮预先约定的参考边界。读出 projection 的 batch 恒为 1，继续使用原 `encoder.proj`。
- **observed**：backend 进入 state signature（`streaming.py:106-120`）；切 backend 后旧状态须重置，CPU 回归测试位于 `tests/test_streaming.py:200-215`。这避免把修复前的缓存混入修复后执行。

## 已存在的最小证据，不应重复开跑

1. **reported，独立核**：`gpu_fixed_linear_probe.py:15-36` / `gpu_fixed_linear_probe.json` 使用 257 行、输入维度 7 和 131、输出 128，比对整批与块长 1/7/17/31/64/128，记录前向逐位一致；与 FP64 affine 后转 FP32 的误差满足原容差。相同上游梯度下，输入、权重、bias 梯度与 PyTorch affine 比较通过。此梯度项本身是相同形状比较，不能单独代表不同分块反传等价。
2. **reported，反传内存**：同脚本 `:37-42` 测试 `8192 × 8 × 131` 消息输入的 backward，记录有限梯度及峰值显存 270829056 bytes。实现使用 `g @ W`、`g.T @ X`（`streaming_linear.py:54-61`），没有显式生成 `[rows, IN, OUT]` 逐行权重梯度。这支持避免已识别的 expanded-weight 巨量中间梯度风险；不是总体训练显存上界。
3. **reported，集成独立递推**：`gpu_debug_fixed.json` 记录原容差内通过，各层、读出、独立递推 pose/778 vertices/21 joints 最大差均为 0；包含真实 8321 事件突发、缓存实际 storage 检查、全模型有限梯度、优化器更新后旧状态拒绝和空状态重建。脚本 `gpu_debug.py:99-130` **只读取前 512 ms 真实事件，而查询到 1024 ms**：256 次查询中后 512 ms 没有新事件，不能称为 1024 ms 连续有事件更新；有事件的查询最多 128 次。
4. **reported，跨分块梯度**：`gpu_gradient_probe.py:29-56` / `gpu_gradient_probe.json` 补齐上述第 1 项不足。117 个合成事件的同一最终读出平方损失，完整前缀对比逐事件 append、17/31/69 分块；所有 encoder 参数梯度满足原 `atol=rtol=1e-5`，最大绝对差为 `embed.weight` 的 `3.814697265625e-6`。该输入跨越 window=32，但短于 50 ms，且未超过 readout max_nodes=2048；不可将之描述为任意长多次读出训练梯度保证。
5. **observed，证据来源**：当前 `gpu_debug.py`、`streaming.py`、`streaming_tracker.py`、`streaming_linear.py` 的 SHA256 均与 `gpu_debug_fixed.json.source_hashes` 一致。后两核来源同时与 `gpu_gradient_probe.json` 一致。本次实际读取的 `streaming_tests_v5.log` 是 `51 passed`；主代理另报告了 52 项回归，本审查尚未读取其对应日志，故不合并为已读计数。

## 必须保留的数值合同与最小后续检查

### A. 后向精度模式是外部合同，不能由文件头代替

**observed/inferred**：固定核 forward 硬编码关闭 TF32，并拒绝 CUDA autocast（`streaming_linear.py:35-39`）；但 backward 的两个 PyTorch matmul（`:58-59`）仍取决于执行 backward 时的全局精度设置。文件头的 “No TF32” 不足以保证 backward；也没有声明支持在另一个 autocast 上下文里反传。

**proposed，下一次训练/服务入口的必要检查**：显式固定并记录 FP32、`torch.backends.cuda.matmul.allow_tf32=False`、无 CUDA AMP；训练反传也在该合同内执行。现有 G001/G001-B 脚本已经正确设置 TF32off，因此无需为这个静态边界重复其 GPU 实验。若今后要接受其他精度模式，先添加精度模式拒绝/支持测试，重新做相应数值和梯度 gate，不能默认为兼容。固定 forward 不意味着 backward 对不同分块逐位相同，后者现有证据只是原容差通过。

### B. 正式短程训练应验证实际使用的 loss/unroll

**observed**：现有 integrated 梯度段为单次 query 的 pose+vertices 损失；跨分块等价检查为单个 encoder 最终读出损失。

**proposed**：若后续筛查训练使用多 query 递推损失，在其有预算 Debug 中加入同一小序列、同一观测/损失、独立完整前缀 vs 缓存路径的总损失与参数梯度比较，并验证一次优化器更新及显式 reset。只需覆盖实际拟使用的 unroll，不必额外建立无关高阶微分测试或扩大成正式训练。不能把本报告作为未实施训练协议的 Debug 通行证。

### C. 持续输入和部署成本属于下一阶段，不能被正确性 probe 替代

**proposed**：下一阶段 G002 按已冻结真实服务合同检查持续新事件、空事件、突发、查询等待/排队、传输、建图、MANO 输出及冷首输出。记录 Triton 版本、GPU 架构、CUDA/驱动及核 source hash；当前 artifact 已有 GPU UUID、PyTorch、source hash，不能由固定 tile 推断跨编译器逐位等价。首次编译与缓存状态应明确，不能将其隐式藏入 warmup 之后并称完整首次服务时延。

持续有事件的较长回放可暴露数值递推/状态问题，但无需现在无依据重复 G001；保留 512 ms 动态覆盖的准确表述。如新硬件、dtype、编译器、核代码或主要模型运算变更，再重跑相关最小 gate。

## 非阻断性边界

- 该 helper 面向当前 S37 的带一维 bias 的 `nn.Linear`、FP32 CUDA；没有宣称 bias-free、任意 malformed 参数、FP16/BF16、二阶导数或跨硬件逐位一致。现有模型不需要时，不应为这些功能扩大实现范围。
- `contiguous()` 支持常规非连续输入的逻辑数值，但现有独立核随机输入本身连续；后续如实际调用传入新的 stride 布局，可补一个 stride 对照。当前消息构造与 token 路径未发现因此失效的证据。
- 训练保留 autograd 历史，实际缓存 tensor 数量有界不代表训练图内存有界；核反传峰值通过也不消除长 unroll 的累计图成本。
- C1 的 token、采样/读出历史语义与旧 S37 不等价；固定 affine 只修复 C1 自身缓存/完整前缀数值一致性，不把旧权重自动变成已验证的新模型，也不改变 C0 对照的身份。

本审查支持在上述限定内继续已授权 G002 与后续有预算 Debug。精度与时延门槛仍待统一协议实测，历史 S37 当前臂保持不变。

## 后续补审：G004 后的两个固定形状 CUDA Graph 子段

本节仍为只读方案审查：读取 `gpu_mano_schedule_probe.py/json`、`model/mano_layer.py`、`model/model.py` 的 `_fk`/`_decode_active`/路由调用及当前 tracker；没有运行 GPU、训练或改实现。

**observed/reported**：G004 对 `transform_chain[self.parents[i]]` 做且仅做一次 AST 替换，host tuple 从当前 MANO parent buffer 读取；四个合成输入的 eager 前向/输入梯度逐位相同，20 次纯 MANO graph 输出与原 eager 逐位相同，cold capture 独立记录。这验证了一个局部调度机制，尚未验证捕获 `model._fk` 的 pose51 解码、十五个 decoder 或二者接入后的递推。

**审查意见**：对孤立 model 实例，在 scratch 中固定形状、FP32、eval、无梯度、明确冷准备成本，捕获 `_fk(pose51, beta)` 与 `_decode_active(feat, prev, evidence)`，两臂共用，保留 tracker 的因果/有限检查与原路径，**没有原则性否决点**。它没有减少 event append 工作量，不得据纯 MANO 调度结果声称端到端达标。临时替换 `mano.forward` 必须在 capture 成功和异常路径都恢复，不能变成模型的隐式全局补丁。

只增加以下两项最小合同，其他内容保持主代理已给定的严格验证范围：

1. **单实例调用顺序与输出所有权**。Static input/output 是可变资源，`copy → replay → clone` 整段须单调用线程、同一明确 CUDA stream 串行，或有等价互斥/依赖同步。仅验证参数版本不能阻止两个调用互相覆盖输入；最小诊断版可直接拒绝其他线程/stream，无需增加并发框架。做 A→B→A 调用并保留第一份输出，核验返回值在后续 replay 后保持逐位不变。特别注意一次 tracker query 已在路由的 `_project_prev` 调用 FK，末尾再次调用 FK（`model/model.py:1380-1409`、`streaming_tracker.py:164`）；不得让首次返回的 vertex view 指向第二次 replay 的静态输出。
2. **完整接入语义与失效检查**。分别对 C0、C1 比较原 eager 与两个捕获子段的短序列，事件输入相同、历史独立递推、包含空 query，检查 pose 与完整 mesh 逐位一致。G004 的纯 MANO 对照不能替代此项。补一个参数/或 buffer 变更拒绝与一个 Python 分支 flag 变更拒绝，验证 guard 在 replay 前执行。必须覆盖的静态分支包括 `pose_repr`、`ablate_joint_heads`、`mano.add_mean`/`num_joints` 及相关子模块的 training 状态；`_decode_active` 的 `evidence=None` 或非空 `root_extra` 不在当前固定合同，应明确报错而非忽略（`model/model.py:1193-1228`）。

`clone`、每次 guard、输入 copy 和必要同步都属于实际执行成本；两个子段及两臂的准备/capture 成本分别保留。通过这两项只表示所测推理调度接入没有改变输出，不授权把 inference-only wrapper 用于训练，也不替代后续真实负载时延验收。
