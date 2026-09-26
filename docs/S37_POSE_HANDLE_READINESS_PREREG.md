# PR0：原调用接口的 pose 返回前沿

2026-09-26，数值执行前登记。AS0已关闭当前完整输出返回接口上的直接preview形式，但CC1未保存forward返回时刻。PR0只辨别：保持原forward算术与本次插桩调度，在forward返回后、输出MANO/FK与D2H之前发布pose，是否仍被调用方取得tensor的时刻阻断。不是新preview、流水线、GPU pose-ready测量或创新候选。

预执行修订：原after-call-only版本的时刻严格说是返回的上界，不能冒作下界。未启动任何数值；原稿和脚本保存在scratch/goal_20260926/pr0/pre_boundary_revision。最终版在隔离实例的原forward_packet最后return out之前插入一个host时钟赋值，PRE/POST夹住同线程调用方取得返回值的时刻；固定门改用PRE。

## 固定对象与最小改动

- CC1 optimized C，CC0两个500步checkpoint（3407、3408），每个ylf_global/ylf_local原序列首64个H帧；顺序3407 global/local、3408 global/local。原合法初值、betas、K、50ms输入和自身反馈保持；无GT/新精度评分、无zgz。
- 继承CC1所有冻结源码/输入身份与数据文件size/mtime；核CC0保存的对应256帧predictions/prev，仅对首64逐字节回归。CPU原事件前缀预载到RAM后开始计时，禁止提前构包；此负载不含物理sensor传输/磁盘读取。
- 原串行顺序：等待release → 构包/H2D → forward_packet → 输出FK → pose/vertices/joints CPU复制 → 原有synchronize → full_complete → 诊断 → prev=pose。
- host perf_counter标记：构包返回、forward最终return之前、调用返回之后。**不增加中途同步、CUDA events、hook、缓存或精度改变**。PRE是当前插桩执行下同线程调用方取得句柄的下界，POST是上界；均不是GPU-ready。代码内route_stats等仍原样执行。
- 不改core文件：AST读取当前原函数，仅在最后return out前新增`self._pr0_pre_return_marker = _pr0_clock()`赋值，删除此节点后AST必须与原函数完全相同；只绑定到本实例，globals用副本、无闭包。限定原routed非memory分支；每次调用前marker清空，并核`packet_return <= PRE <= POST <= complete`；finally恢复实例方法并移除marker。它改变的是插桩调度，不能把此时序当未插桩旧CC1的精确重建。
- 五次原首包预热后重置原合法初值；首正式包照计。每流单独anchor模拟H到达并保留排队；所有帧保留，无计时重试/挑选。

## 固定计算与决定

第i窗的输入到达时刻由原ev5时间与同一host anchor还原。若要用前窗返回的committed pose，纯调用层提前交接的乐观下界为：

`wait_handle(i,a) = max(0, pre_return_marker(i-1) - arrival(a))`。

该下界给PRE之后的return/交接、GPU未完成部分及新preview零成本；只对返回后才消费tensor的接口及本次插桩时序成立。首窗已合法初始化，就绪等待另记零，不进后续63窗主分母。原始输入分母不等于全部事件被2048采样保留或影响输出。

- 任一流出现 `wait_handle > 7 + 1e-5 ms`，且身份/原预测回归通过：`REJECT_RETURN_ONLY_EARLY_HANDOFF_ON_MEASURED_SCHEDULE`。保留普通>7与保守计数。这个门只否定单纯提前调用层交接的固定形式；不推出所有提前计算、算子融合或异步调度不可能。
- 若所有流无反例：`HOLD_GPU_READINESS_AND_PREVIEW_UTILITY_UNMEASURED`。不能宣布pose真正就绪、preview有效或完整7ms通过。
- 身份、有限性、时序、回归或计数失败：`INVALID`，无工程结论，不替换seed/片段或放宽标准。
- 四个人工算术夹具（理想、早事件延迟、晚事件已就绪、首窗）必须通过。理想发布在逻辑窗口起点的负对照仅容许1e-5ms浮点残差；它是公式控制，不是性能成绩。

报告每流构包返回、forward调用host时长、begin到PRE/POST、PRE至POST夹宽、forward返回到完整输出、完整service/query/raw event响应及状态等待下界P50/P95/P99/max/>7比例；它们写执行后附录，不改主表。host forward区间可能含GPU同步与dispatch空隙，不能叫纯GPU kernel耗时；输出尾段可能含尚未完成的forward，不能叫纯FK成本。

新时钟可受仪器和当前调度影响，不把CC1旧时钟与本次拆分值相减，不主张新增调用必然单调变慢。1e-5ms仅为浮点比较裕量，不是插桩误差上界；PRE/POST夹宽也不是未插桩时刻的校准。阳性只识别固定调用边界；未记录GPU-ready仍明确未知。

## 执行、身份和预算

父代理单次GPU1（UUID GPU-39a910a1-7942-a849-4d85-d5007ec25de6；执行前复核空闲），物理GPU0永禁。必须通过budget_run；Debug进程100秒+清理10秒，最多110 GPU秒；四库线程/单进程，不分片，不训练。FP32/no_grad/eval、TF32 off、deterministic保持CC1设置。

先静态审核脚本并冻结其SHA、本文及审查报告，保存完整身份和环境；输出目录run_v1禁止覆盖。完成后核原pose/prev逐字节及持有首份输出不被后续覆盖，再应用固定门。失败不自动重启；只有明确脚本错误且原门/数据未改才可另外登记修复。

S37当前臂、三个科学候选与SC0/CC0/CC1/AS0终态保持。原短周期反馈澄清尚未答；本次不执行该变更。后续实现需由本次真正排除/保留的依赖决定，不能借此自动启动preview或训练。
