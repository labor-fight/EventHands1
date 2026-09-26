# F1：原S37完整mesh服务的FK重复计算合同

2026-09-25，执行前登记；C0r纯工程诊断，不是第四候选、不修改原模型、不改变输入/递推周期。唯一问题：本步最终MANO输出能否作为下一步同状态路由所需的MANO几何，避免同函数重复计算。原H相邻50ms包不重叠，不以此宣称旧事件图特征可缓存；改变时间token/采样仍是另一输入合同。

## 固定范围

- 原 `configs/semkine/s37_routed_s3407.yaml` 与R0冻结checkpoint；仅 `lyq_local` 原R0首段前8包。初始化直接取已冻结R0 `prev[0]`（原H合法GT+噪声初始化）；随后均自身递推。只比较CPU原实现与memo包装实现，不要求CPU对旧GPU逐位相同、不评分GT、不读zgz。当前事件/tsub/offset及初值文件身份冻结。
- 包含原 `forward_packet` 和输出 `_fk` 的完整局部函数；两分支FP32、eval、no_grad、autocast disabled、同包/shape/K。冻结前不运行输出/性能评分。
- 单CPU进程、最多4库线程；内部90秒、外层100秒+5清理；GPU0。失败即INCOMPLETE，保留失败定位；不改门/换片段重跑。不使用此CPU耗时外推L20毫秒。

## 缓存语义与必须通过的检查

缓存只有最后一组FK结果。键绑定pose/beta的device/dtype/shape/stride/storage offset/data pointer/版本，持原tensor强引用防地址回收；模型/MANO参数和buffer的identity/版本、pose_repr、add_mean、num_joints也绑定。FK和MANO forward的稳定函数身份、线程数、matmul精度/TF32/确定性/backend标志进入键；持参数/buffer及旧MANO强引用避免替换后的地址碰撞。只接受受控服务持有、通过普通PyTorch操作修改的张量；`.data`、NumPy/raw pointer写入或并发外部写没有版本保证，禁止用于本合同。inference_mode张量无版本，不支持，明确拒绝。图编译/AMP/多CUDA stream不在本次实现范围。

eval/no_grad/FP32及禁autocast为硬条件，违反即报错，不能悄悄返回缓存。返回值克隆保护内部所有权；缓存不保存投影，因此K变化每次重新投影，不能使用旧uv。stream切换/显式reset清缓存。主代码不植入猴子补丁；scratch测试在限定context内临时包装 `_fk` 并保证退出恢复。

启动环境清除 `S37_DBG/S37_DBG_CTX`，脚本再次检查；禁止框架全局module hooks，当前MANO局部forward/pre/backward hook非空则明确拒绝，并验证一次局部forward-hook拒绝。全局hooks与任意修改纯函数语义的外部代码不在本封闭执行合同内，不宣称检测全部副作用。

1. 固定8包原实现与包装实现的pose/vertices/joints逐位相同；原实现FK应为16次，包装实际FK应为9次且命中7次。若真实模型分支与此预期不符，停查定义，不改计数门。
2. 同状态空事件输出保持一致；重复FK返回相同数值但不共享对外输出storage。修改已经返回的vertices/joints不能污染下次路由或输出。
3. pose原地改变、beta原地改变、FK资产buffer改变、另一个流/显式reset均必须miss，且输出与原FK一致；值相同的不同tensor可以保守miss，不能错误hit。
4. K改变允许复用FK，但重新 `_project_verts` 后须与直接重算投影一致，并确实改变uv；不把K当FK几何依赖。
5. train/grad enabled、CPU autocast、inference_mode无版本tensor拒绝；异常退出恢复原model._fk。

只验证函数等价、缓存失效/所有权及实际重复计算数。没有速度通过门或训练通过门；即使全过，仍需另立无竞争GPU完整服务合同，并处理50ms输出等待和实际异步图输入。保持当前S37主行不变。

## 一次执行终态（2026-09-25T19:26:15.221355+08:00）

`f1_fk_memo_cpu_v1` 一次COMPLETED/exit0。外层wall `3.8201017739484087` 秒，内部 `0.9278119920054451` 秒，GPU0。执行前freeze SHA256 `4b80b38af918b2cfc251e1fe95c018759049eb09926b576aa536f903170395cf`，receipt SHA256 `61e6db09d3ecc0ee479e6c7577d536c2f5422767667aa6a14b991db0993fe16e`；22个源/模型/资产/配置/初值/脚本身份已冻结。原审查和prereg副本不改。

固定8包真实原事件数 `[482, 872, 582, 1667, 2950, 4372, 6408, 6396]`（含超2048采样cap包，不改变原采样）。两个分支各自从合法冻结初值递推，pose/778verts/21joints逐位相同；16次原FK对9次实际FK、7次命中，与执行前调用数门一致。全部48项记录通过，含空包保持、外部返回值修改保护、pose/beta/资产版本失效、stream/reset、K重投影及grad/train/autocast/inference/nonFP32/hook拒绝、异常恢复。

父方直接读取 `replay_outputs.npz` 独立核对8步三类完整数组、有限性及原始hash，未重跑模型。输入末端原H语义未更换，不与旧GPU输出做逐位门比较。父审及G24/真实几何提案来源身份在 `research_state/audit/F1_G24_RG_design_parent_20260925.json`。

**有限结论：限定CPU FP32/no_grad、受控ownership与所列变化下，复用完整输出的FK可消除下一路由的同输入重复计算。** 非空8包的完整输出有原始数组证据；控制分支记录为脚本断言回执，不冒认父重新执行。空包forward本身处于wrapper外，但当前空路径不调用FK；K控制是重投影，不是K变化的完整递推。没有逐项覆盖对象替换、不同storage布局、backend切换、GPU/AMP/并发/global hooks。输入的raw/.data外写禁止，不由版本计数检测。

这是scratch原型，未接入主工程，也不回填正式主表latency。未测GPU成本/完整服务，不宣称节省百分比或满足7ms；更不能解决原H50ms等待、旧C1 append瓶颈或图token全局失效。当前S37保留，三科学候选状态不变，不因F1开新训练。
