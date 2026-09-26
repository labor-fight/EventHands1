# CD0：当前C执行器的同函数MANO/解码头捕获

2026-09-26，数值执行前登记。PR0在实际插桩调度上排除了仅提前调用层交接；下一步改变返回前的调度工作。已有G004/G005验证了固定MANO父索引/原头算序的捕获原语，G006在另一种4ms反馈服务上仍未过完整门。本轮只把这些原语做成可复用opt-in模块，并验证当前C原H闭环，不能把旧原语成功当本次已经成功。

## 唯一实现及不变项

- 新模块`semkine/captured_s37_decode.py`，显式`CapturedS37Decode(model)`；安装只覆盖隔离实例`_fk`与`_decode_active`。默认模型/训练入口不变，不依赖scratch或.research模块。
- 从当前原ManoLayer.forward取得源码，唯一AST替换`transform_chain[self.parents[i]]`为同值host父索引。逆替换后AST必须等于原函数，父拓扑/资产保持。只在捕获FK时临时绑定该实例mano.forward，异常也恢复；正式图执行复用原_fk和_decode_active算术，不合并15个head的矩阵乘法。
- 每次copy当前pose/betas或feat/prev/evidence到静态buffer，图replay后clone每份输出。输出FK同样调用本实例_fk，不把旧网格返回充当新mesh。动态输入、所有权、reset/空包/beta变化须验证。
- 单GPU/单线程/单stream、batch1、FP32、eval/no_grad、TF32 off、deterministic。拒绝AMP/grad/train、换设备/stream、参数或buffer变更、方法/flags/hook等不支持上下文；不得静默eager回退。完整运行守卫成本计时。
- 原event_tokens、采样、图边/三层、常量context、路由/证据池化、prev_mlp、delta和空包gate保持。无跨包图缓存、无事件丢弃变化、无短周期反馈或preview。

## 身份与范围

CC1 optimized的两个C@500 checkpoint（3407/3408），固定ylf_global/ylf_local原256帧/合法初始化/原beta/K；只读已用原输入与封存predictions/prev，不读新GT或zgz、不评分精度。继承旧CC1的81SHA与330数据stat，再冻结新模块、测试、harness、本文、静态审查。

CPU只做新增模块适用的语法/上下文测试；GPU作业首先执行新模块合成合同，然后对每seed每序列原256帧比较reference C与captured C，各自prev←自身pose，与旧C256行prediction/prev逐字节一致。原外部ManoLayer decode与model._fk完整vertices/joints亦须逐字节一致，固定FP32不可放宽；任何失败停止成本阶段，记录INVALID/不采用。

合成/适用合同至少覆盖：空包→非空、单事件、短包、2048上限外包；变化pose/beta/feature/evidence；同一首份返回值在后续图重放后不被覆盖；安装/恢复及异常恢复；grad/AMP/train/stream、模型参数版本变化应明确拒绝。测试不要求所有不支持架构成功。工程Debug不证明新科学信息或精度收益。

## 唯一成本回放及门

全等价通过后，固定顺序`reference3407 → captured3407 → captured3408 → reference3408`，各global/local原首64帧。原合法初态分别reset，五次首包预热后再reset；每条流独立anchor和50ms release，prev为自身预测。首正式请求保留。两分支均做实际CPU构包/H2D→完整forward→输出_fk→pose/778顶点/21关节CPU复制→末端同步，不在计时中作额外parity/profiler同步。

输入events/tsub前缀预置RAM但不提前构造下一包；构包、图、运行守卫、copy/replay/clone、输出FK和D2H均在service。保存真实队列、所有完整query和raw事件到达响应；旧50ms等待保持，不把service通过误称完整7ms。物理sensor输送/磁盘读取不在此次有限负载。setup/capture/暖机另报，不能藏进未披露的免费准备。

每seed合并global/local的128个service样本。**两seed的captured P50/P95/P99/max均≤对应reference，且P50严格降低**，才记`EXACT_CAPTURE_COST_REDUCTION_ON_FIXED_WORKLOAD`；否则`EXACT_CAPTURE_COST_GATE_FAILED`，保留全部尾部退化，不追加轮次/改统计/调守卫。任意身份/全等价/输出持有失败则INVALID，不解释速度。

此工程门只决定该opt-in实现是否在固定负载有同函数成本证据；不是C/L模型采纳、独立泛化、7ms硬实时或创新门。即便所有服务处理<7ms，事件等待/排队和中间输出效用仍分别未解决。不得取C精度配另一个服务策略的速度验收。

## 预算、停止与交付

单次budget_run，GPU1 UUID GPU-39a910a1-7942-a849-4d85-d5007ec25de6，启动前确认空闲；GPU0禁止。Debug执行240秒+清理10秒，≤250 GPU秒；单进程四库线程，无训练、分片或新科学候选。新模块/tests之外临时执行器留在scratch/goal_20260926/cd0。必需Debug失败不继续成本；失败只在明确执行器错误时另登记修复，不换seed/负载救场。

原core与SC0/CC0/CC1冻结源码不动、checkpoint键和值不变。S37仍当前臂，三候选门保持，完整目标未达。执行后附录记录命令/身份/全部分位数/最大值/超限/准备成本/内存/预算和独立终态复核；主表沿用已有S37行，不读取zgz生成工程诊断新行。
