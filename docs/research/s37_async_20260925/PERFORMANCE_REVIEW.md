# G002/G003 性能反方审查与最小机制探针

审查日期：2026-09-25。范围：只读现有代码和真实回放/trace，新增独占scratch诊断脚本及本文；本审查者没有启动GPU、训练或修改core。下列阶段计数和时间是工程诊断证据，不是新精度实验、main row或7ms验收结果。

## 当前证据支持的结论

G002四条回放均明确失败。C1的首次事件append有显著冷成本，但去掉这个成因也无法解释后续持续积压；C0与C1的共享MANO/heads路径及C1的1ms微批执行均已存在持续吞吐瓶颈。G003进一步支持“许多小算子的CPU派发/执行组织”是必要诊断方向；不支持把全部时间归罪于GPU计算量、`.item()`同步、Triton kernel算术或首次编译。当前不可开正式训练，也不可用提高节点裁剪、延长周期、丢掉冷输出或删检查宣布达标。

四条输入的raw xyp、int64 t和K哈希逐一相同，均为train `lyq_local[0,50)ms`、482事件，继续输出到100ms；完整response包含真实积压，四条超7ms率均为100%。证据是`.research/s37_async_20260925/g002_{c0,c1}_{2000,4000}.json`。C1/4ms实测最大response为1841.197104ms；C1/2ms为2173.16607ms；C0/4ms为729.3083ms；C0/2ms为1010.997925ms。这些是模拟host arrival口径的实测失败，不能称任意传感器输入硬实时结论。

用于根因分析的原始服务区间摘录如下，**不是重新裁掉冷启动后的合格延迟**：

- C1/4ms首append为1259.663447ms，其余34次append合计207.374855ms，中位5.7406585ms；非空query含输出copy中位19.654151ms，首次155.597543ms；空query含copy中位6.075374ms。
- C1/2ms首append为1274.067397ms，其余34次append合计215.237654ms；非空query含copy中位20.000984ms，空query中位5.8043505ms。
- C0/4ms原始buffer append中位1.282667ms；非空query含copy中位23.741448ms，首query402.282538ms；空query中位5.7363805ms。C0/2ms对应中位约1.452275/24.22483/5.500185ms。

因此，即便将C1首次append冷成本修为零，其后34次append所需服务已超过100ms逻辑片段的时长。空query单次成本也大于2/4ms输出周期，长期静止同样会积压。7ms响应门和2/4ms持续服务能力是不同约束，都不能靠forward均值证明。C0 raw buffer没有事件GNN前向，其append仍约1ms，说明检查/小kernel派发也需区分，但不能直接把C0/C1差值当某一个模块的纯成本：buffer、输入语义和节点准入不同。

## G003的可追溯定位

原始`.research/s37_async_20260925/gpu_profile.py`先执行3次诊断warmup，再以CPU+CUDA profiler、record_shapes和profile_memory运行一次117事件、两次append、一次非空query及一次空query。**该profile有显著侵入开销，不能拿stage CPU时间替代G002非instrumented延迟。** 各stage含嵌套调用，不能相加当总成本；CUDA event/设备时间也不能再与其CPU等待相加。

`gpu_profile.json`记录：两append CPU 35.797ms、CUDA 1.340ms；两query CPU 60.264ms、CUDA 1.207ms；8个fixed affine调用CPU 3.014ms、CUDA 0.086ms；两次token CPU 9.465ms、CUDA 0.521ms；两次构图CPU 11.268ms、CUDA 0.228ms。单次active-head解码CPU 11.296ms、CUDA 0.151ms。三个FK合计CPU 33.841ms、CUDA 0.508ms。全段1054个`cudaLaunchKernel`，92个`aten::item`和100个`cudaStreamSynchronize`；后者CPU总时长仅0.370ms。故“同步次数很多”本身不能解释几十毫秒CPU阶段时间。

本次独立解析`gpu_profile.trace.json`，按同pid/tid且完整落在`stage.model._fk`的时间区间统计，保存到`.research/s37_async_20260925/events/performance_trace_audit.json`（包含trace/源代码哈希及逐item前驱select）：

- 第1 FK区间起点1790273352291332µs，时长11800µs；第2起点1790273352323634µs，时长11500µs；第3起点1790273352337456µs，时长10543µs。
- **每次是16个item，不是15个**：15个`long int`的item均紧随输入维度`[[16]]`的`aten::select`，索引1至15，对应`model/mano_layer.py:138`的`transform_chain[self.parents[i]]`。三个FK共45个此类父节点取值。
- 每次还有1个`float` item，前驱select来自`[16,1,4]`的最后一列；对应`_transform_mat`里`pad[:,:,-1]=1.0`的fill路径，不能把它误称GPU父节点同步。
- 每个FK内恰有15次`cudaStreamSynchronize`，总时长分别48/50/48µs。MANO父索引优化的直接理由是消除设备标量进入Python索引这一**CUDA graph捕获障碍**，不应预先承诺靠它省掉整个5–6ms，更不能承诺7ms系统门。

调用路径已核实：`semkine/streaming_tracker.py:150`→`model._route_nodes`→`_project_prev`→`_fk(prev)`先得到旧mesh供路由；`streaming_tracker.py:169`再调用`_fk(pose)`输出更新mesh。非空query两个FK输入不同，不能简单删掉第二次；空query跳过路由/heads仍执行完整FK。固定旧pose几何缓存是否值得另测，当前不作为本轮第二个候选。

`model/model.py:1203`明确保留15个joint head的Python顺序，因为历史batched-baddbmm更改归约顺序后出现递推指标偏移。此次不能重演“公式相同即结果等价”。若未来捕获heads，应保留现有调用和算子顺序，依然过独立递推门。

## 未被现有证据解开的成因

1. **冷Triton边界**：C1第一次`_linear`才import `semkine.streaming_linear`。该模块有Triton import、JIT/cache lookup/模块装载及首launch，其他CUDA库亦可能首用。`streaming_linear.py:14`已`do_not_specialize=['rows']`、固定tile、无autotune；IN/OUT有7→128和131→128两个形状。1259ms不能未经trace就命名为“编译时间”，也不能把G003中0.086ms的8次稳态kernel成本外推为冷行为。若本轮纯FK判别之后仍需要，只做一次新进程首append的Python/cold API trace，分别记录import/cache/实际编译/首次launch，保留原G002冷首输出；不清理用户缓存或把提前编译悄悄移出服务SLA。
2. **SAE和构图**：`streaming.py:181–220`每append将历史访问过的pixel/polarity键与新键一起sort/searchsorted，随后布尔压缩；`226–245`构造M×W候选再topk。布尔索引和dynamic shape还会迫使主机得知输出长度。G003的token/graph GPU时间小，但短片段只有117事件，不能代表P逐渐接近86400键的长轨迹；不能把该片段“CPU派发主导”推广为所有负载都如此。
3. **输入与完整检查**：`streaming.py:150–166`包含有限性、坐标、极性、时间顺序等GPU→host判定；`streaming_tracker.py`还有时间水印及输出有限性；replay还有明确同步和CPU全输出copy。这些均在当前合同/计时内。可以在后续有证据时研究融合一次校验或合法host输入检查，但不能删除检查、跳过失败输入、异步返回尚未确认的mesh来降数。
4. **实际GPU kernel vs主机组织**：1054 launch包括MANO、15个heads、稀疏token/graph、检查/分配和pool。profile给出了派发问题的证据，却没有证明每个stage可捕获；动态节点数、SAE压缩、bool判断、int(t)等阻止直接整tracker capture。当前只选择纯MANO固定shape子段，避免整网编译与无依据多候选。

## 唯一执行候选：固定shape纯MANO CUDA graph，父索引改为明确host tuple

准备的`.research/s37_async_20260925/gpu_mano_schedule_probe.py`只在独立进程内解析当前原始`ManoLayer.forward`，把唯一`transform_chain[self.parents[i]]`索引表达式替为`transform_chain[_mano_parent_indices[i]]`；tuple从原MANO buffer一次复制到host。反向替换后的AST必须与原AST完全相同，替换数必须等于1，源文件不写入。`J[:, self.parents[1:]]`保留原GPU张量索引；16级拓扑、每次matmul、LBS、778顶点和21关节全不改。

这不是新网络，也不增加学习权重。实际持有显式parents tuple、稳定输入buffer、graph输出buffer、参数/buffer的id/version/device/dtype/shape签名；每次执行前检查签名，不兼容则报错，不能使用旧graph。训练autograd只用原/改后eager验证；本次捕获仅no_grad推理，不暗示CUDA graph backward已经支持。

预先冻结的最小门与预算：

1. 标准库AST-only检查：已运行，exit0，`AST_ONLY_PASS_NO_TORCH_OR_GPU`，证据`events/mano_schedule_ast_check.json`；只证明唯一替换与AST不变性，不证明GPU语义。
2. root审查后至多一次120s GPU job（脚本内部115s超时，失败落JSON且退出，不fallback）。四组固定seed=20260925的中性/随机pose/不同beta/translation，原eager与host-parent eager的完整mesh和四项输入梯度均要求finite且**bitwise相同**。失败时不继续capture，不放宽为allclose。
3. 通过后，明确记录capture所需3次side-stream warmup及第一次graph capture成本；它们是诊断准备成本，绝不从G002冷启动验收记录抹除。稳定B=1输入buffer capture原顺序的纯MANO；原eager、host-parent eager和graph各运行同一固定合成输入序列20次，轮换每轮的三臂执行顺序。
4. 每个测量wall区间包含context检查、相同输入D2D copy、完整778顶点+21关节输出CPU copy及同步；另列device timeline，明确其中包含主机派发造成的间隙，不能称纯kernel active time。每轮graph完整host输出仍须与原eager bitwise相同。此序列是**合成MANO输入，不是真实事件回放**，也不使用GT。
5. 即使纯MANO graph成功且更快，完整输入/编码/路由/heads/排队尚未包含，**不得放行训练或宣布7ms**。C1每1ms微批约6ms的瓶颈独立存在。若捕获失败，保留异常和已通过门，先定位失败算子，不自动扩展全网编译或消耗剩余预算。

候选额外成本：稳定GPU输入/输出及graph私有内存、capture准备与版本检查；模型权重相同但每个模型/设备/dtype/资产布局需独立有效context。将来若推广到正式runner，必须显式交付初始化/失效/恢复成本，重新跑独立递推/完整mesh/冷和稳态回放；不能让scratch AST成为正式推理或训练的隐式依赖。

拟由root执行的命令形状（GPU UUID、独占与总预算仍由root调度器设置）：

```bash
timeout 120s env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  .research/s37_async_20260925/gpu_mano_schedule_probe.py \
  --authorized-gpu --out .research/s37_async_20260925/gpu_mano_schedule_probe.json
```

本审查者未执行该GPU命令。脚本保存自身/原MANO/replay守卫/资产哈希与输入摘要；trace审计保存G002四条原日志哈希。根因证据和有限执行候选不等于精度机制有效，更不构成创新或SOTA声明。

## 后续事实：root执行G004，与G005适配交付

root已单次执行上述G004，实际文件`.research/s37_async_20260925/gpu_mano_schedule_probe.json`为`PURE_MANO_DIAGNOSTIC_PASS_NOT_LATENCY_ACCEPTANCE`。四组原/host-parent eager全mesh及输入梯度bitwise；20轮CUDA graph全mesh亦bitwise。纯MANO含相同输入D2D及完整输出CPU copy，原eager P50/max为3.649791/5.046273ms，host-parent eager为3.364996/3.952056ms，graph为0.2740665/0.370521ms。第一次capture为77.469571ms，capture前3次warmup为17.853982ms，均单列保留。结果支持“父索引适配可解除捕获障碍，单纯父索引eager优化有限”，不证明完整服务或冷启动满足7ms。

基于此证据，root明确授权只扩展两个固定shape子段的scratch适配；不增加事件encoder执行候选。本审查者准备：

- `.research/s37_async_20260925/decode_schedule_probe.py`，唯一公开类`DecodeScheduleProbe`。用法：在`torch.no_grad()`中`plan=DecodeScheduleProbe(model); plan.install()`；显式caller持有plan；报告`plan.metadata`；原bound oracle为`plan.original_fk`/`plan.original_decode`；最后`plan.restore()`。
- 仅捕获`model._fk([1,51],[1,10])`和`model._decode_active([1,512],[1,51],[1,16,257])`。MANO实例forward仅在FK准备/捕获时临时采用单AST替换函数，并在finally立即精确恢复；没有改class或旧core。安装的两个实例wrapper每次copy→replay→clone，不能把graph静态输出借给递推状态。
- 每次调用拒绝不同创建线程或CUDA stream，检查所有参数/buffer的身份、version、device、dtype、shape，所有子模块的training状态及身份；冻结pose_repr、ablate_joint_heads、mano.add_mean/num_joints。必须eval、FP32、TF32off、显式no_grad、无autocast和固定输入shape；evidence=None或root_extra非None明确报错。任何不相容都不fallback。
- `.research/s37_async_20260925/gpu_decode_schedule_parity.py`只有三组pose的完整FK/delta bitwise、固定feat/evidence；A→B→A持有旧返回的clone检查；flag和buffer各一次变更拒绝。完整C0/C1的独立递推与空query检查由root单独实现，不在此脚本扩张。

本审查者只做了标准库parse/compile语法检查，保存`events/decode_schedule_syntax.json`，没有import torch或运行GPU。helper的source hash在该JSON和未来`plan.metadata`中记录。GPU语义尚待root的G005；G004纯MANO成功不能替代此门，也不能解决1ms append约6ms的既存瓶颈。若G005过门，仍须按root预注册的G006做C0/C1完整4ms回放，包含初始化、capture与等待的准确口径，继续保留G002失败。

## 主代理执行补记：G005/G006终态

主代理已执行独立子段数值、输出所有权、buffer/flag失效，以及C0/C1分别独立递推；所有限定bitwise合同通过。完整4ms服务重测仍全部超限，详见DEBUG.md及`g006_*_captured_4000.json`。准备费显式包括在model startup，原G002冷失败继续保留，不将pure MANO speedup称为系统speedup。

结论收敛：固定解码调度机制有局部必要性和数值证据，但没有解决C1每1ms事件更新吞吐。即使令首append和所有query成本为零，后34个append仍需178.534378ms，长于100ms逻辑回放，构成当前版本的明确性能NO-GO。当前不提升scratch plan为正式默认、不训练、不继续相同实现跑更大场景；下一步需先作事件更新执行调度的新预注册和判别，不新增学习模块或把末N截断当免费速度。当前代数/因果正确不等于方法有效，目标未达成。
