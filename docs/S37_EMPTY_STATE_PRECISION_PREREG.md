# EP0 空事件状态精度合同修复

登记：2026-09-26T03:39:20.147717+08:00，核心修改/新验证执行前。定位是目标§6空事件Debug必需工程修复，属于C0r hygiene，不是科学候选或新训练臂；不把它当精度提升、创新或完整7ms通过。

## 证据与范围

早期docs/research/s37_async_20260925/AUDIT_AGENT.md §7及.research/s37_async_20260925/audit/empty_autocast.json已证CPU BF16空包将FP32历史舍入。model/model.py::forward_packet当前EGM单独保留状态精度，S37旧分支对整个prev转out.dtype再相加；ZERO_EVENT_GATE只把delta清零，不能恢复被cast丢掉的位。既有FP32空包测试未覆盖它。不是本轮刚发现的科学根因，是尚未修完的已定位合同缺口。

只在predict_delta && zero_event_gate的最终返回处，用empty mask选择原始prev行；非空沿用旧计算顺序，再按torch共同dtype提升。不得将全部delta加法改为FP32后宣称非空等价，不加参数/新配置开关，不改数据、loss、事件支持、H50ms、输出时刻或checkpoint。使用torch.where不引入host条件同步。若prev为FP32、旧out为BF16/FP16，返回提升为FP32；非空数值仍为旧低精度计算的精确嵌入，并不是提高其计算精度。

## 固定验证与停止门

1. CPU合成真实tiny S37前向：全空和混合空/非空，FP32与BF16，非BF16可表达prev；空行应逐位等于输入prev，包含signed zero、重复空包；支持FP64/BF16 prev的合理测试如实现可用，不能写未覆盖。
2. 对照冻结model_before.py的旧算术：非空值、对应state/parameter梯度保持；空行对prev梯度为identity、对模型参数为0/None；无gate和非delta路径不改。新参数/键为0，旧checkpoint严格加载。
3. 既有8包真实train缓存/旧S37 checkpoint的FP32输出与旧封存张量、当前修改前代码均逐位一致；不重采样、不解码新数据、不做标签评分。缓存target仅为原batch必备字段，验证不使用loss/GT。单CPU进程4库线程，120秒外层上限。
4. 单GPU1 L20，budget_run Debug timeout90秒+10清理；同合成合同在CUDA FP32/BF16/FP16实际验证，不能只用CPU推断CUDA。若失败，保留原因，修实现问题需重新冻结具体修复，不改数值容差救场。GPU0排除，启动前复核占用。
5. 完整恢复合同继续绑定model/model.py源hash；旧完整checkpoint必须拒绝跨本次代码变更的精确resume，不能静默更新旧provenance。新代码内数学/优化连续恢复的原RC1通用机制不改，按独立审查判断必要新增验证，不无故重训旧X1。

本变更可能改变含空行的BF16训练loss及其整个batch的log-loss缩放，不能声称旧BF16训练轨迹逐位延续；旧FP32主评测路径须保持。没有新精度评估、选点或主行。本轮完成后只可称空状态合同修复，并保留科学主方案与最终三硬门未达。

## 身份与产物

执行前核心和既有回归产物SHA见scratch/goal_20260926/ep0/identity_before.json；原始model副本model_before.py。父仅改共享core/文档，resume_contract代理仅写tests/test_s37_empty_state_precision.py和EP0独立审查。旧工作树大量修改保留，不自动提交或推送。


## 2026-09-26T03:46:33.002988+08:00 CUDA v1测试fixture失败，固定修复后重验

CPU20通过/10显式CPU FP16跳过，原回归8通过，旧8包FP32三方逐位与实际complete checkpoint源码拒绝均通过。首个GPU作业ep0_cuda_precision_20260926_v1退出1：18项通过，12项在构造expected梯度的`expected[empty] = weights.to(state_dtype)`处触发Torch2.1 strict-deterministic CUDA Indexing.cu内部形状断言；失败发生在输出身份断言及实际autograd之后、梯度比较之前。因此不能写空行梯度GPU已通过，亦不是模型空行数值失败。

v1完整回执/XML/冻结测试/独立审查存scratch/goal_20260926/ep0/cuda_v1，原budget_run日志保留，实际7.631778687005863GPU秒，cleanup空。仅将期望值的广播布尔赋值换成明确outer-product `empty[:,None] * weights[None,:]`，与原逐元素定义相同；不改模型、数值门、样例、精度或deterministic设置。修正后重跑CPU矩阵及单GPU1同90秒+10清理，作为v2；不是科学参数搜索，不覆盖v1。


## 2026-09-26T03:49:20.935041+08:00 EP0终态：有限状态空包精度合同已修复

核心最终6行post-mask修复，未增加参数/配置。CPU新精度矩阵20通过、10项CPU不支持的FP16显式skip；既有S37/S2回归8通过。GPU1 L20/Torch2.1/CUDA11.8严格deterministic的FP32/BF16/FP16共30项通过，含混合空包、有限状态身份梯度、空行模型梯度0/None、非空旧算术及梯度逐值相等、5次空包反馈signed-zero位模式。v1 fixture失败保留，只有期望梯度赋值换为等价outer product，v2不降门。

实际旧S37 checkpoint严格加载、既有8包train缓存的FP32输出与修改前代码及旧封存tensor均逐位一致，参数733830/键90保持。真实X1c complete checkpoint仅改变model源码SHA时按原合同拒绝exact resume。没有重训、选点、GT评分或精度新主行；验证范围不是完整AMP训练轨迹相等或所有架构/CUDA负载覆盖。

边界：AMP时整个输出dtype可提升为FP32，非空值保持旧低精度算术，不自动提高非空精度；含空行训练loss及log-loss缩放可能改变。旧FP32普通状态回归保持，signed-zero复制是有意加强。旧complete checkpoint精确恢复须使用原源快照；不改旧权重contract。增加的最终where没有重新测主表Latency或完整event→mesh，不能把历史行冒称改后测量。

执行产物：scratch/goal_20260926/ep0/{cpu_tests.xml,cuda_tests.xml,cuda_receipt.json,real_regression.json,resume_guard.json,core_delta.patch,cuda_v1/}；作业ep0_cuda_precision_20260926_v1/v2分别FAILED/COMPLETED，均cleanup空、leader/runner已退出。本轮实际Debug收费14.499832835048437 GPU秒；全局预算{"debug": {"charged": 6891.220436390839, "reserved": 0.0}, "screen": {"charged": 37046.12625307795, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}。GPU0未使用。

裁决：保留此正确性修复；S37仍当前臂，C0r科学HOLD、固定C1r/R0 REJECT、C2r HOLD保持，三项硬目标与唯一创新主方案未达。LC0旋转复合提案另已静态关闭，不由本工程修复获得训练准入。
