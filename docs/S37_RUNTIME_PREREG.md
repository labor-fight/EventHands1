# S37 运行时证据审计与下一实现的准入合同

2026-09-25，主代理。此文审核已存在的测量与代码，不是新性能实验预注册已执行，也不放行新训练。采用模型仍为S37 routed。来源身份、旧数值原样引用与当前GPU只读状态见 `research_state/audit/G24_runtime_resume_20260925.json`；新GPU消耗为零。

## 1. 已有三种不同的计时对象

**正式主表。** `tools/make_s36_row.py` 的历史S37行是包预置GPU、起点prev复用、多个轮次取最小均值并按完整CNN锚点缩放的forward口径。它与原H精度逐包用自身预测递推不相同，也不含完整服务等待/传输/mesh输出。保留历史行，不将旧JSON事后补成当时没有的现场元数据。

**G24-B1计算包络。** `scratch/goal_20260925/g24/bench_latency.py` / `bench_latency_b1.json` 已实际执行，固定checkpoint、训练序列、以GT为计时prev替身，无精度主张。其中A的 `timed(lambda: fwd(...))` 实际包含 `make_eval_packet`：不能只凭JSON字段名 `forward_packet_ms` 声称只有网络；所谓 `h2d_forward_d2h_ms` 再复制pose到CPU，未返回完整mesh。B每阶段独立同步，其和不能作完整运行时间。C在计时前完成token与采样，graph回放内复制的是已在GPU的固定shape输入；产生pose/vertices/joints，但测量返回clone只取pose和vertices，未包含完整host输出链。它验证了有限包上的pose parity与固定核心加速潜力，未验证所有空包、短包、overflow、模型变化及长递推。

因此，“固定核心很快”不证明整个S37需要的工作只有这些，也不能把所有eager与graph差额无条件归为一种原因。B的同步方式会改变调度；C的操作排布和输入准备范围也有区别。需要与实际服务绑定才能判断最终瓶颈。

**G006完整host回放。** `.research/s37_async_20260925/g006_c{0,1}_captured_4000.json` 是旧包重算控制与因果cache控制的固定小片段回放，真实host wall，按sensor timestamp模拟host arrival；没有测量实体相机transport。初次query包括在内，事先的模型加载/graph准备单独计时；有排队、输出deadline、full mesh+joint复制与空query输出记录。两个控制均明确旧门FAIL，未给精度收益。它们初始化/shape与H不同，checkpoint仅初始化，不可把其耗时与S37正式精度拼成一个方法。

`event_latency` 的样本单位是**每个非空输出服务到的最早pending事件**，不是所有事件加权的分位数；`deadline_latency`按每个输出请求，`empty_latency`按空请求；`all_response_latency`把非空最早事件年龄与空请求deadline延迟合并。最大最早事件年龄可约束该批被服务事件的最大等待，但分位数不能改称逐事件分布。原字段/数值保持，今后定义须显式。

## 2. 不存在的实现不能算交付

`research_state/debug/G24_fast_infer.md` 是旧未完成骨架，列出的 `semkine/fast_infer.py` 当前不存在，设计/CPU/GPU验证节仍待填。不把它当已实现 `StaticRoutedInfer` 或主线性能证据。真正可读的是上述scratch `StaticCore/StaticMano` 诊断、旧streaming实现与服务回放；它们尚未合并为正式快速模型。该骨架沿用旧scaled-forward硬门，已由GOAL §0的完整event→mesh门覆盖。

`semkine/streaming_tracker.py` 明确为新输入/runtime合同：arrival-final token、固定query间隔、显式外部state、完整MANO输出，任何pose/shape/K变化重算路由。当前原型默认中性pose/零beta；本项目原H合法初始条件仍单独保留。该原型不是旧S37权重的等价异步重放；短间隔不能静默缩放delta后宣称保持旧准确率。

## 3. 事件到mesh的时钟与调度下界

设事件真实接收时刻为a，首次包含其影响的完整mesh可用时刻为m，则event响应为m−a；输出请求deadline为q，请求响应为m−q。两者不同。若只每T时间服务一次，允许任意相位的事件到达，单是批等待的上确界已接近T；这不是GPU优化能消去的部分。固定50ms历史lookback本身不要求等待50ms，但“每50ms才更新一次状态并首次输出”会产生相应等待。

因此未来必须同时固定lookback、arrival batching、状态更新触发、输出节奏、丢弃/采样与过期规则。不能用50ms策略测精度，再用另一短周期策略测延迟。原H标签/初始化/指标继续保留；若内部状态更新策略改变，必须明确同一新策略如何在原评估时刻产出并接受验收，而非偷偷替换H逐包定义。当前未声称这一协议问题已经解决。

## 4. 下一正式实现的必要合同（proposed；未开运行）

1. **同函数或明确近似。** 固定checkpoint/代码/配置/资产/token/采样/算术精度，逐完整前缀比较原函数和优化函数。若要求bitwise则不可事后改allclose；近似必须事前冻结容差并报告长时状态漂移。有限GT计时替身parity不足以代表自身闭环。
2. **缓存依赖。** 图特征是否依赖窗口归一化、max赢家、过期事件、采样替换或重连，逐项列出；MANO变化使投影、visibility、路由及条件残差失效。常量offset、parent拓扑与动态coords不混同。换模型参数/设备/dtype/K/shape/stream/reset须拒绝旧cache或完整刷新。
3. **输出所有权。** CUDA graph静态buffer不能被下一次调用覆盖用户已拿到的pose/vertices/joints；返回拷贝或生命周期合同必须明确。包含host输出时不能只clone两个张量而漏第三个。跨线程/stream并发及动态shape须fail closed，不静默退到未经计时路径。
4. **服务负载。** 在无竞争L20上冻结事件率/突发、输出频率、history容量、初始化/恢复/全局刷新及采样策略；逐事件/最早pending/每query/空query分别保存分母和原始时间。必要等待、排队、接收后的preprocess/传输/建图/网络/MANO/输出均计入；实体transport未测则明确。不得以均值或P95代max，也不能将有限实测max当任意负载硬实时保证。
5. **停止条件与预算。** 先CPU数值/状态合同，后有界GPUDebug；任何数值、因果、所有权、overflow或parity失败停止，不按性能结果调门。GPU任务由已冻结budget runner预留命令+清理时间、身份快照、退出回执；没有必要性证据不为占卡启动作业。

这些要求可复用G17–20和G05的工程经验，但不是一个新科学候选。已有G007脚本化失败不能用“编译成功”覆盖；已有G006服务FAIL不能由B1较短核心时间改判。

## 5. 本轮核验与剩余证据

本轮实际读旧脚本/JSON、核对统计单位与已存在文件；不重跑G006/B1、不改主模型/旧回执。预算runner先前修复及20项CPU行为测试仍适用，G24最终功能/性能尚未全部完成。最新GPU只读快照无计算进程；GPU0继续排除，本轮未使用任何GPU。

当前最有用的工程交付是明确哪些旧证据可支持下一步，而非把诊断核心升级为已完成的快速推理模块。完整门仍未通过；新实现与科学候选需要在同一推理策略下闭合。

## 2026-09-25T19:26:15.221355+08:00 精确增量边界与F1限定执行

详research_state/debug/G24_exact_incremental_boundary_20260925.md：原H相邻50ms包不重叠；滑窗tnorm/SAE span/采样与过期可使全图失效，KNN实数时间平移相消不等逐位不变。F1已一次CPU核完整mesh输出FK复用，证据在docs/S37_GEOMETRY_REUSE_PREREG.md；仅消同输入重复几何，不改变图/路由/状态策略，不将其回填历史forward表或视为7ms通过。
