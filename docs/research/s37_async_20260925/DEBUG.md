# Debug 登记

## D001：旧 S37 能否直接复用 prefix 缓存

预先定义：固定包时长 50 ms；前 128 个事件完全不变，再追加 32 个；如果可以直接复用旧节点，则旧节点 token 及采样身份不变（token atol 1e-6，身份精确）。这是结构反例诊断，不是精度实验，也不是对 S37 checkpoint 的评测。预期：SAE fallback 随 span 改变，均匀采样随总数改变，故合同不通过。

命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/audit_contracts.py
```

输入、完整读数和 source/checkpoint 哈希分别保存于 `.research/s37_async_20260925/audit_contracts.json`、`source_identity.json`。该脚本同时仅对四条训练序列做只读 timestamp/坐标/负载检查；不读取新模型验证分数、不修改训练文件。

实际结果：命令 exit=0。旧 prefix token 最大变化 0.06400001049041748；64 个旧采样身份中 38 个在追加后不再保留。直接复用缓存合同 FAIL，符合预期。四条训练序列的所选窗时间顺序/坐标检查完成；详情 JSON，不能据此声称全数据已通过。需要改变 token/准入定义才能建立精确异步参考，原 S37 精度结论不受此反例判决。

后续门（尚未执行）：冻结事件 token/邻接定义后，同观测全量/增量输出 fp32 atol=1e-5、rtol=1e-5；逐节点逐层而非只比较最终均值。GPU 累加不同顺序若不满足该门，判失败并定位，不在看结果后放宽。长期漂移另外量化，不能以局部数值近似推定闭环等价。空包保持、序列 reset、checkpoint/version/cache 不相容时报错、突发输入预算、optimizer 和小样本拟合均须独立通过。

## D002：CPU 回归及新合同

基础回归实际命令在 `.research/s37_async_20260925/baseline_contract_tests.log`，61 项通过；包含原S37读出、数据、原评测parity、packetizer。旧评测parity使用冻结旧checkpoint，未选择新模型或调整zgz超参数。耗时272.84s，CPU 4线程。

新 `tests/test_streaming.py` 与 `test_streaming_tracker.py` 初版23项通过，独立反方发现input buffer别名和时间静默取整后已修；新增输入覆盖、逐层、手算token、长gap零消息与短递推全mesh检查。v2连同原S37读出回归共35项通过，4.64s，18项依赖弃用警告；`.research/s37_async_20260925/streaming_tests_v2.log`。旧S37源码没有修改。该日志覆盖CPU，不证明GPU容差、完整系统Debug或精度。

## D003：预算内真实训练小样本拟合（执行前登记）

仅CPU、4线程、wall cap 120s，最多64个Adam steps、LR=0.001、clip norm=1。加载S37种子3407已选权重作为初始化，改后的流式token不是旧checkpoint等价输入。取原train的lyq_local/ch_local各首个长度≥108ms有效段，固定start=a+100ms的连续两个4ms包；全量事件不采样；第二包用第一包预测历史，初始化固定中性pose/z=0.45、beta=0。GT仅监督第二包末端51D状态，沿用原MSE51D；不把GT形状或GT历史送进模型。

门：所有损失/梯度有限、参数实际更新，64步后loss≤首步的10%；任何异常或120s超时记录失败，不增预算重试调参。本检查只是能否拟合两个训练样本及贯穿两个包的梯度，不支持泛化、冷启动有效或最终精度；GPU/混合精度与长期递推门仍独立待验证。新拟合参数不覆盖S37 checkpoint，也不保存为正式候选。

命令：`timeout 120s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/tiny_fit.py`。

实际：exit=0，64步完成；loss从1.825713038444519到0.07471268624067307，比值0.0409224695597985，参数确实改变，无非有限梯度；优化循环5.997s。两段训练输入包数分别59/21及175/126，无GT推理输入，无checkpoint写出。原始`.research/s37_async_20260925/tiny_fit.json`、`tiny_fit.log`。D003的有限小样本门PASS；不推广为恢复、泛化或指标达标。

## D004：实际CPU回放与计时记账（执行前登记）

不使用GPU、不训练、不读GT标签/shape/history。固定训练序列lyq_global、start=0ms、读取[0,50)ms全部事件，输出到100ms包含静止timer，周期4ms、微批1ms、CPU1线程。单命令wall cap 60s，不按结果换片段；目的为真实接口、排队、完整778顶点输出、读出裁剪记账验证。GPU性能门仍HOLD，CPU数值不能外推GPU。

预期：每个合法输入事件按原顺序接收一次；25个按固定deadline的mesh输出；不跳过滞后帧，排队随真实服务成本累积；无新事件仍输出mesh，未服务事件只有输出成功后才清零；超8192事件/append显式失败、不截掉输入。任何异常保留partial timeline，不提高容量重跑伪造PASS。固定epoch映射模拟host arrival而非实测相机驱动，因此即使全响应低于7ms也不算正式验收。

命令：`timeout 60s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/replay_latency.py --device cpu --threads 1 --sequence lyq_global --start-ms 0 --period-us 4000 --duration-ms 100 --out .research/s37_async_20260925/replay_cpu_debug.json`。

实际：exit=0，status=COMPLETED；1214个事件全部接收，25个完整mesh输出成功，13个有新事件输出+12个静止定时输出。计时范围为固定host replay时钟，非传感器驱动；CPU单线程累计排队，峰值1087事件，100ms逻辑时间实际消耗315.908ms。有事件完整响应P50/P95/P99/max分别73.352517/224.387098/237.5128996/240.79435ms，13/13超7ms；无事件deadline响应max236.959743ms，12/12超限。该CPU包络明确不满足7ms，但不据此判定L20性能。调度/完整输出记账门PASS，目标GPU完整延迟仍未测。

这次固定片段最高readout仅1214节点，未触发2048裁剪，不能据此宣称高负载信息完整。完整train offsets包络另立LOAD_ENVELOPE.md。后续head参数失效修复不改变固定权重回放的数值算子，仍需GPU正式性能计时覆盖其新增版本检查成本。

## R5：head参数变化未拒绝旧递推状态（独立审阅实际反例）

`.research/s37_async_20260925/audit/review_final_counterexample.json`：正规`no_grad.add_`修改root_head，旧state的encoder cache签名仍相同，query接受了旧预测历史。虽然事件h依然数学有效，但违反本方案“参数变更后显式reset”的整个递推系统合同。没有正式训练已启动。

最小修复：StreamTrackingState保存全模型参数的id/version/device/dtype，append/query含空输入均检查；权重改变时initialize重新开始。加入三类head × 空/非空输入旧状态六项回归；v3日志单列，不覆盖v2或审阅快照。

实际v3：exit=0，共41项通过、18项依赖弃用警告，5.38s；`.research/s37_async_20260925/streaming_tests_v3.log`。独立代理已只读复查实现、针对性测试和新日志，见REVIEW_FINAL_CPU.md的resolution部分。未重新做无关基础回归或拟合；该改动只增加状态有效性守卫。

## D006：已观察最大真实burst的分块合同（执行前登记）

仅CPU4线程、wall cap 30s、无训练/optimizer。固定全train offsets扫描找到的最大bin：ycy_global_v3的[14664,14665)ms，8321事件；读取该bin全部xyp、tsub及标定K，不读标签或GT shape。先检查微秒范围/非递减及core输入；冻结S37 seed3407权重只作初始化，tracker起点14664ms、query14668ms。调用D005保序外层分块，预期两次core append=8192+129，全部事件接收。

对照独立full_reference一次处理完整8321事件，原预注册FP32 atol=rtol=1e-5，比较所有五个readout输出；已有逐层小fixture测试不替代本次实际burst。预测必须输出有限完整778/21 mesh，单独标出最后2048直接读出。通过不证明长时递推、低读出保留的精度或GPU性能；失败/timeout不增预算重试。

命令：`timeout 30s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/burst_contract.py`。结果只写scratch JSON，不产生模型结果主行或checkpoint。

实际：exit=0，PASS_BOUNDED_CPU_CONTRACT；固定8321事件按8192+129分块，两次core调用、无前端事件丢弃，五项readout与完整重算最大绝对差均0（原门atol=rtol=1e-5）。成功输出778顶点和21关节，pending清零，最后2048节点直接读出，未宣称其余全部进入几何路由。脚本内部wall0.557s；CPU append服务237.773ms、query到host mesh19.617ms，仅为本次调试读数，未模拟持续到达排队，不能替代完整延迟分布。真实输入哈希、IDs、时序核验、子调用和输出证据保留在`burst_contract.json/log`。D006局部门PASS，GPU/长期/精度门仍HOLD。


## D005：真实训练突发负载的无损微批分块（执行前登记）

新证据：`.research/s37_async_20260925/geometry/train_load_manifest.json`完整扫描原train的72条序列offsets，1ms最大8321事件（`ycy_global_v3`，首个最大bin起点14664ms），两个bin超过8192。该证据只覆盖offsets定义的负载，未验证所有tsub内容。D004的固定片段及其结果原样保留；8192的单次core append上限不足以直接接收完整观测负载，不能通过截断输入或提高core上限掩盖。

预登记修改仅限scratch `replay_latency.py`：已到达的同一1ms微批按现有`max_events_per_append`保序切成必要子调用，全部事件编码，不新增query、不改deadline/epoch，不提高核心上限。预处理/H2D及全部子调用/同步成本仍计入该微批wallclock服务时间，记录`subappend_count`及`subappend_sizes`。中间递推state仅为局部变量；全部子调用及同步成功后才提交backend state；失败传播，调度器不得把部分事件标为accepted/served。

标准库自测预算：单次命令30s以内，不import torch、不读真实事件/GT、不跑GPU/训练。预期覆盖8321→[8192,129]及小容量边界，连续event ID完整且各一次；注入第二子调用失败，原backend state保持同一对象，accepted/served仍为0，失败显式传播。另保留既有调度边界/空流/积压/mesh确认自测。新JSON独立保存，不覆盖旧日志。标准库测试仅证明适配层调度与原子提交，不证明真实tracker数值chunk等价、真实突发耗时或7ms；真实burst接口检查由主代理另行预算登记。

预定命令：`timeout 30s env CUDA_VISIBLE_DEVICES='' python .research/s37_async_20260925/replay_latency.py --self-test --out .research/s37_async_20260925/events/replay_selftest_d005.json`。


### D005 实际记录（标准库适配层自测）

按上述预定命令执行，exit=0，`SELF_TEST_PASS_ONLY`，新JSON为`.research/s37_async_20260925/events/replay_selftest_d005.json`，其中保存工具SHA256与完整argv。没有import torch，也没有GPU/真实模型执行。6个容量边界案例通过；8321事件分成[8192,129]且连续ID恰好各一次。注入第二子调用失败时，原state仍为同一对象，整个微批accepted/served均为0，无query发生。原有定时边界、积压不重置、空流、读出cap及mesh确认检查一并通过。

成功微批的append记录新增`subappend_count`/`subappend_sizes`；全部子调用与最终同步计入原append服务区间。失败继续显式传播、保留STARTED记录，不提交中间state。此测试不证明真实tensor实现的chunk等价或7ms，D006另检真实8321事件及完整mesh。既有D004结果、旧自测JSON和核心代码均未改。

## D007：C0最小修复对照的算子忠实性（执行前登记）

上一goal turn有实际源码、审阅修复与检查证据，分类为progress。本轮GPU预算仍未落实，但C0公平比较的薄适配器是尚可独立完成的必要工作，不重复已通过C1检查。只在统一scratch加入C0适配与有限合同脚本，不改旧模型/配置/结果，不跑GPU或训练。CPU4线程、单验证命令wall cap60s。

C0复用SparseS37Tracker的因果水印、无GT固定初始化/beta、FP32累加、同query周期及完整MANO；替换事件encoder执行器为50ms已到达原始事件buffer，query时原样调用原EventGNN.forward。保持原token7、整窗均匀采样、EdgeConv与pad/mask。时间为t_rel=(int64_t-(q-50000us))*1e-6，delta_t=0.05；冷启动不伪造历史、不缩短时间归一化常数。原始buffer只保留可能参与后续query的50ms历史；容量500000个原始事件（高于已观察51个ms bin上界51×8321=424371），超限显式失败，不静默裁剪为末2048。该控制有原有的采样，不与C1末N读出混称同节点准入。

门：相同原始观测、FP32 prev/先验shape/K下，C0预测与独立构造EventPacketBatch后原model.forward_packet一致，pose/完整mesh atol=rtol=1e-5；相同流不同append分块保持相同C0结果；冷启动/空新事件不复用旧证据推进；buffer复用/历史过期/reset/变权/非法时序正确处理；C0与C1 API的K/shape/init/cadence相同。需同时检查原模型所有预期参数收到有限梯度（只backward，不optimizer），GT不进入推理。只在训练固定片段与合成fixture核验，不读取验证分数。

对照仅表明C0适配忠实，不能把原50ms训练权重称为有效2/4ms模型，也不能证明C0/C1精度高低、创新或7ms。其raw buffer与采样成本在未来正式性能测量必须全部计入；当前C1 replay的末N统计不得直接用于C0。转入正式训练前须明确提升并版本化该适配器，禁止成为隐式scratch依赖。

实现前补充反方意见：共享tracker此前在pending_events=0时仍调用encoder.readout后才keep pose。C1也有无效pool/proj，C0则会无效重算整个图，若照此计时会人为夸大缓存在静止时的收益。因此只在共享新runner里统一无新事件路径：检查全模型和encoder配置/stream版本，直接保持pose并输出MANO，不调用readout。旧模型forward不改。补必要回归：无新事件时装一个会报错的readout替身，query仍成功；此时改变encoder配置必须继续拒绝旧state。由于触及公共query，重跑受影响streaming合同，不重复无关数据回归或拟合。

实际：首条v4命令在测试收集阶段因Python3.9下`int | None`注解即时求值失败（exit2，3.48s），没有执行模型。只为新scratch适配补`from __future__ import annotations`，保留失败日志。修复后相同验证范围/预算v5：exit0，51项通过，5.47s；`.research/s37_async_20260925/streaming_tests_v5.log`。包括C0原算子/mesh独立构造parity、实际8321事件训练burst、C0/C1共享输入、空闲快路径和受影响C1合同。无optimizer、无GPU、无测试集分数。该修复不构成增加训练预算重试。

## D008：真实短连续序列的独立闭环重算漂移（执行前登记）

仅CPU4线程、wall cap60s、不训练，固定原train lyq_local的[0,512)ms全部事件；4ms query，继续512ms无新事件timer，总256次输出。固定S37 seed3407旧权重仅作C1初始化。预测从neutral/beta0先验开始，各自保留独立历史：cached支路用SparseS37Tracker；reference支路每次新事件query从完整已到达prefix用full_reference重算，再用其自身prev解码/递推；无新事件两边都keep并输出MANO。禁止reference拿cached pose当历史。只读取原始事件/offsets/tsub和K，不读GT标签/beta。

门：每次query的五项encoder输出、51D状态、778顶点及21关节沿用FP32 atol=rtol=1e-5；所有值有限，记录最大绝对差与发生时刻。每次append检查no_grad状态下cache tensor的真实storage bytes符合所保留suffix、各层≤W、readout≤N、SAE键≤2WH；没有隐藏全prefix特征storage。只有reference允许保留完整prefix供重算。60s超时/差异超门即FAIL，不减少片段或放宽容差补过。这里只覆盖1.024s输出、低事件率真实起点，不能称长时恢复/全负载/精度已证。

命令：`timeout 60s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/recursive_parity.py`；输出独立JSON/log，保留失败进度。

实际：exit0，PASS_BOUNDED_CPU_RECURRENCE，46.596s完成。固定512ms训练片段共有36510事件，继续静止至1024ms，256次完整输出；encoder五项、pose/vertices/joints的最大绝对差均0，使用各自独立递推历史而非单步共享prev。末端live节点数0。被检查的持久cache tensor storage峰值1207152 bytes，无保留整prefix的隐藏tensor storage；这不是进程峰值内存、临时workspace或GPU显存测量。原始JSON/log为`.research/s37_async_20260925/recursive_parity.*`，包含输入哈希、checkpoint、query和范围。该局部门PASS；分钟级漂移、真实恢复、GPU精度/性能及模型精度仍未验证。

## G001：首轮目标GPU数值与梯度合同（执行前登记）

用户确认资源预算后执行；最多一张空闲L20（当前物理7，CUDA_VISIBLE_DEVICES固定UUID），CPU4线程，作业总wall cap600s，计入1 GPU小时Debug额度。启动前复查设备进程/显存；不与其他本任务GPU作业并行，不抢占已有任务。FP32、TF32关闭，cublas确定性配置；未预注册混合精度误差容限，因此本轮先不用AMP。

冻结C128/L3/W32/K8/2048的S37已选seed3407权重只作数值初始化。检查：合成事件的逐事件/分块/完整prefix五项输出、各层cache与独立完整参考；原CPU预定atol=rtol=1e-5不放宽。真实最大8321事件分块/full-prefix读出与完整mesh；独立GPU递推复用D008固定lyq_local[0,512)ms输入+512ms静止、256query及其自身reference prev；缓存的持久storage边界。梯度/optimizer用单独重建的小模型或完整模型副本，所有预期梯度有限、更新后旧状态拒绝、空事件FP32保持；不得改S37 checkpoint。GT只允许作为CPU既有tiny-fit监督，G001不读GT标签/shape/history。

失败停止该job后续依赖门、保留已完成项目及错误，不静默fallback CPU，不重复启动完整任务；仅在明确代码或数值原因修复后重跑受影响检查。通过仅说明该设备/FP32/有限输入合同，不代表方法有效。实际完整服务性能由后续G002独立测量，C0接入及采样统计先单独核对。GPU混合精度是否需要，按实测瓶颈再决定，不能用省时理由免除相关Debug。

G001实际FAIL：9.1375 GPU秒，先通过合成分块/逐层/时间原点与真实8321 burst，但独立递推完成69次后下一query的pose[49]差异1.2725591659545898e-5超过原门。此前单步h最大差8.940696716308594e-8，mesh差仍在原门内；这些小量不能用来撤销pose失败或放宽容差。梯度/optimizer阶段未执行。原始`gpu_debug_g001.json`与`jobs/g001.json/log`保留；无训练启动。

G001-N数值根因判别（执行前登记）：GPU7单卡、wall cap120s，计入Debug额度。对相同固定FP32输入和原embed/三层linear参数，比较整批nn.Linear与按不同比例分块调用，隔离GEMM的M尺寸；另比较逐row bmm（1×K与K×C，expand共享权重，不增加参数）及FP64计算后转换FP32的参考。只有相同输入的算子结果随分块变化、固定row算子消除该变化的直接证据，才考虑最小执行核修复。记录误差和初步compute成本但不作完整延迟/性能结论；不改变任何core、原checkpoint或容差。此为诊断job，不重复完整G001；失败不自动重试。

G001-N实际：5.1764 GPU秒，固定相同输入下embed与所有edge linear均存在跨批量大小差异（约1e-7–1e-6）；row-bmm也未消除，故该替代被反证，不接入core。误差对照FP64同样保存，原始`gpu_numeric_probe.json`；其CUDA event均值不是完整延迟。下一判别G001-N2（单卡≤60s）核验token/有效邻居全局ID/mask/dp逐位相同，以及同消息tensor下两种mean实现逐位一致；不先归罪GEMM以外的模块。

G001-N2实际PASS诊断范围：2.9237 GPU秒，token/有效邻居全局ID/mask/dp及同消息mean逐位相同，`gpu_graph_probe.json`。结合已观察embedding起始差异与固定输入Linear反例，支持先修线性算子执行顺序，而不改图或mean。row-bmm候选已排除，也不采用可能产生[R,I,O]巨量反向中间量的展开权重训练实现。

G001-F最小修复预注册：保留所有学习参数和矩阵乘加公式，GPU FP32事件embedding/消息linear使用固定16×32×32 tile、固定K循环、无autotune且禁止TF32的Triton执行核，批量M只影响grid/mask，不决定归约路径。当前环境实际为Triton2.1.0，`tl.dot(...,allow_tf32=False)`；[官方矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)仅作实现参考，API按本地签名核实。CPU仍用原Linear；C0原算子不变。新增核无参数，backward使用标准两次矩阵乘与bias求和，避免权重展开的巨大梯度中间量。未验证AMP不静默启用或回退。

先做隔离GPU核Debug（≤180s）：原embed/edge维度、1/7/17/31/64/128等分块与整批应逐位同；对FP64→FP32 oracle及PyTorch forward/backward沿用atol=rtol=1e-5，有限梯度与峰值内存记录。通过才接入C1完整prefix/cached并重跑受影响G001（≤600s），原失败仍为FAIL、不放宽门，不让reference共享cached的prev。full_reference必须保留独立全prefix建图，允许只共享已独立验证的线性primitive；逐层输出显式返回作Debug，不靠修改C0模型hook掩盖差异。若固定核成本存在结构瓶颈，继续性能门失败，不为了数值PASS直接放行训练。


## G000-adapter：共享回放器的C0/C1直接读出记账（执行前登记）

只修改scratch `replay_latency.py`，增加显式`--control c0`并导入同目录已存在`packet_control.PacketS37Tracker`，默认`c1`仍为`SparseS37Tracker`。同一原始训练片段、1ms接收微批、2/4ms完整mesh输出周期、固定epoch、无GT中性初始化、checkpoint初始化、无warmup、原子分块与wallclock计时边界保持一致；不运行性能/训练或GPU。C0按原EventGNN `_sample`的float32均匀stride选择整50ms窗，C1仍为该窗最后max_nodes的缓存节点，不能套用相同准入规则。

直接读出IDs、跨度、比例和未直接读出的新事件数分别统计。公共tracker在无pending事件时跳过encoder readout，故这类输出实际直接读出计数为0，另列窗口内潜在节点数。诊断成员表在replay epoch前由确定性deadline及各自过去前缀离线构造，不输入模型、不改变处理序列；该诊断开销单独记录，避免给C0计入重复采样诊断而给C1免除。模型本身的C0采样和buffer成本仍在实时append/query内完整计时。报告control定义及adapter/core源码SHA256。

门与预算：标准库共享调度自测覆盖C0/C1同accepted/deadline而不同IDs、空新事件0实际readout、50ms过期及新事件遗漏计数；随后至多一条30s CPU1线程命令，将记账采样IDs逐项对照原`EventGNN._sample`，含0/1/cap边界、8321及500000原始事件计数。只调用采样计数算子，不运行网络/mesh/性能测量，无GT/真实事件读取。保存新JSON，旧D004/D005与其它日志保留。任何不一致显式失败，不能静默回退C1规则或放宽容差；采样身份要求完全一致。


### G000-adapter 实际记录

两条命令均exit=0：

```bash
timeout 30s env CUDA_VISIBLE_DEVICES='' python .research/s37_async_20260925/replay_latency.py --self-test --control c0 --out .research/s37_async_20260925/events/replay_selftest_g000_adapter.json
timeout 30s env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python .research/s37_async_20260925/events/test_replay_c0_counts.py --out .research/s37_async_20260925/events/replay_c0_counts_g000.json
```

标准库共享调度自测PASS且未import torch。CPU算子计数对照PASS_EXACT_IDS_CPU_ONLY，共140例（12个边界/容量例+seed=20260925的128个随机计数），每个采样ID及非零片段ID偏移均与原`EventGNN._sample`完全一致；实际命令约2.23s。仅运行CPU采样索引算子，没有网络forward、mesh、GPU、真实数据、GT、训练或延迟测量。对应JSON保存完整命令、版本、各例ID摘要及源码SHA256，旧日志未覆盖。

默认`--control c1`保持，C0通过`--control c0`显式选择，同一入口/clock/原子分块规则。每输出记录clip内`direct_readout_event_ids`（加input.raw_event_id_range[0]可得到原序列ID）、first/last ID、时间跨度、比例、pending_not_direct_readout。无新事件则`readout_performed=false`且实际direct count=0，`eligible_readout_count`只代表可能参与读出的窗口节点数。这个记账修正不改变tracker输出；旧C1日志里的direct count是潜在保留节点口径，不能与新实际空闲读出计数直接混用。

C0诊断float32均匀stride副本只用于CPU离线成员表；真实模型仍直接运行原采样算子。成员表在epoch前构造、开销单列`offline_readout_audit_ms`、不作为模型输入；所有真实C0 buffer/采样/建图/路由/完整MANO仍计入同实时区间。C0原始buffer容量500000不隐式提高，超限继续显式失败；C1仍为每事件编码、末N读出。CPU身份对照不证明目标GPU的算子数值/完整延迟，后续GPU门及公平性能比较仍由root执行。

源码身份（本次验证快照）：

- replay_latency.py: `1b5f4d9ffa3e0ee345ee3ddb9f61ddd5900011c814385b5a0c30c5f9800b9e16`
- .research/s37_async_20260925/events/test_replay_c0_counts.py: `370ad41665836fd64b644170e058a7fdd9c693b861d1be6325bfc0b6eee3b8b4`
- semkine/event_gnn.py: `bf4cf521186c31c44c145587881bfb25afcc826e525391285dbccb03fef62feb`
- .research/s37_async_20260925/packet_control.py: `6b0faa065857c8b8faf59e90eb86492d9e873baf0ae4b9ab55d4958fdb407cfe`
- semkine/streaming.py: `068ca5b33ce20e4e082b6fc49984467237b5b22b393241fc4020d1aeef22e656`
- semkine/streaming_tracker.py: `eb470f1b37444b0d76bd1c6592d62cfbe7c508f912e14811d17e10f707c50ec5`

G001-F实际：隔离核`gpu_fixed_linear_probe.json` PASS，6.78194 GPU秒；固定batch schedule在已测分块逐位相同，FP64前向及原Linear梯度通过原容差，8192×8×131 backward峰值270829056 bytes。接入后的`gpu_debug_fixed.json` PASS，14.10085 GPU秒：256次独立递推、逐层与全部mesh/pose最大差均0；峰值GPU allocated724337152 bytes，持久cache峰值1207152 bytes。仅新增C1 runner选择fixed_fp32，原模型Linear和C0未改；CPU/C0关联回归52项通过，6.17s。原失败脚本另存`jobs/g001_source.py`，失败JSON不覆盖。

G001-B反向分块补充（执行前登记）：一张空闲GPU，≤120s。同一seed251的C128/L3/W32/K8/2048 encoder，117个合成严格因果事件，比较完整prefix与逐事件、17/31/69分块的readout同损失，对全部encoder参数梯度原atol=rtol=1e-5；所有梯度有限且存在。独立forward affine oracle已有，但不能代替跨append的autograd路径/累加检查。无optimizer、无训练/GT、无精度结论。失败显式停后续训练门。

G002完整GPU服务回放（执行前登记）：G001集成/反向均通过后，统一scratch回放C0/C1，各2/4ms query，1ms接收微批，同train lyq_local[0,50)ms输入及50ms静止，共100ms固定deadline、冷首输出不丢；每job≤60s，一卡顺序，4线程，FP32/TF32off。测排队/等待/H2D/append/query/MANO/host output，原7ms最大值门。初始资产加载另列，模拟sensor时间映射host arrival仍非真实driver验收。此次先确认低负载是否已结构失稳；如失败先分解，不直接浪费预算全包络或训练。通过才补固定高burst片段、恢复和持续负载。两周期都不能通过时不选一个冒充达标，也不调节点量或跳帧压低延迟。

G001-B实际PASS，`gpu_gradient_probe.json`，9.63951 GPU秒。逐事件/三段同损失全部encoder参数梯度满足原1e-5容差；不代表未预注册的AMP合法。CPU52项输出在会话tool session17835，概要凭据`streaming_tests_v6_receipt.json`明确注明是实际stdout摘要而非原始日志。

G002实际4个job均完成，但C0/C1×2/4ms全部FAIL_OBSERVED_OVER_7MS；共150次输出均超限（每个job独立JSON保存P50/P95/P99/max与比例）。C1 4ms冷append1259.877ms、随后多次约6ms；有事件query冷155.598ms、随后约19ms；总排队不能从单阶段均值移除。C0也明显失败，故不能把cache数值一致当作服务成本可行。输入仅lyq_local[0,50)ms 482事件、接续50ms静止；现不扩大高负载跑数、不选周期、不训练。根因仍需profile区分launch/同步/计算，不能把瓶颈预先归为数学架构。

G003性能根因profile（执行前登记）：一张空闲L20，≤120s，4线程，FP32/TF32off。固定S37权重与无GT中性初始化，117合成事件、两段append+query；先3次固定shape机械预热仅为分解steady算子成本，单列其时间且不替代G002冷完整延迟。一次torch.profiler记录CPU/CUDA算子及stage范围（输入验证/稀疏SAE/图/linear/读出/路由/heads/MANO），导出top CPU/CUDA成本、kernel/同步次数与trace。同步扰动及profiler成本只用于定位，不写入7ms结果。只有证明现有执行调度而非算法计算是主瓶颈，才试一项语义保持的最小执行优化；所有cold记录保留。

G003实际完成，7.38192 GPU秒；trace约1054次cudaLaunchKernel、100次cudaStreamSynchronize、92次aten::item。3次MANO FK合计CPU33.841ms、CUDA归属0.508ms；2次append合计CPU35.797ms/CUDA1.340ms。此处是带record_shapes/memory及range的诊断，不与未instrumented G002时延混用。trace逐FK范围实际各16次aten::item；其中15次与model/mano_layer.py:138把CUDA parents scalar用作Python list索引相符，另1次尚需定位，不能写成全部15次已逐源码归因。GPUparents转host带来的串行边界是事实，但其profile item总耗时不足以单独解释整query，应同时承认大量逐算子dispatch。可尝试仅固定不可训练拓扑的Python索引以移除这一捕获障碍，保持数值顺序，再隔离纯FK CUDA graph执行证明是否有收益；不能先改全模型或拿graph纯compute替代端到端。

执行调度参考来源：PyTorch官方 [CUDA Graphs说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) 与 [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) 明确CPU/GPU同步和动态shape不允许直接捕获，稳定地址/shape及side-stream准备需由调用者保证。此为执行机制参考，非研究创新或本机性能证据；本地torch2.1需按实际API验证，不能直接依赖最新文档新增接口。拟议试验只捕获安全子段，所有必需输入检查、输出复制、初始化/失效开销继续记录。

G004纯MANO调度判别（执行前登记）：≤120 GPU秒，root统一budget_run，最多1张空闲L20。只在scratch读取原ManoLayer.forward并替换一个AST父索引表达式为从当前buffer冻结的host tuple；反向AST恢复必须与原完全一致，核心源码不改，不复制整模型。四组固定合成pose/beta/translation先核完整778/21输出及四输入梯度bitwise；如失败立即停止。然后原eager/host-parent eager/CUDAgraph各相同20输入，host tuple/所有MANO参数buffer版本变更拒绝，完整输出copy计入各臂wall。初始化/捕获必要warmup/cold capture另列，均计入GPU预算。只证明capture-safe子段的执行必要性，绝不替代G002输入等待/排队/全模型/恢复门；append瓶颈仍存在，不能据纯FK通过开训。源码`.research/s37_async_20260925/gpu_mano_schedule_probe.py`经主代理审查。

G004实际PASS，3.37259 GPU秒；4组原/host-parent eager完整顶点/关节及输入梯度逐位相等；20次同输入的graph/eager输出亦逐位相等。含GPU输入copy/完整host输出copy的纯MANO原eager P50=3.649791ms、max=5.046273ms，host-parent eager P50=3.364996ms、max=3.952056ms，graph P50=0.2740665ms、max=0.370521ms。capture prerequisite warmup17.853982ms、cold capture77.469571ms另存且已计费。仅父索引缓存收益小；graph消除大量逐算子dispatch的收益可见。没有队列/事件/heads，因此本节不是完整Latency结果或7ms通过证据。

G005受控调度集成前判别（执行前登记）：在scratch提供一个显式执行计划，依G004单AST拓扑替换仅准备model._fk固定51D/beta形状graph，及G003已定位固定heads输入的_decode_active graph；捕获后立即恢复MANO实例forward，禁止全局类monkeypatch。只该孤立模型实例的两个调用由显式计划包装为输入copy→graph replay→输出clone；无state/geometry旧结果缓存。所有参数/buffer身份、版本、设备、dtype及模型flags变化拒绝；training、启用grad、AMP、TF32、变shape均显式拒绝，无静默fallback。原tracker时间/finite/noGT检查全部保留。各3组固定输入与原FK/delta bitwise及输入buffer复用不改已返回输出，版本失效拒绝；失败不进完整回放。GPU≤120s，seed固定，初始化/capture单列计费。该scratch不成为正式训练隐式依赖。

G006完整服务重测（仅G005通过后）：各臂4ms固定周期、原G002同lyq_local[0,50)ms+50ms静止、同1ms接收/输入/权重/FP32/线程及完整计时。两臂共用显式graph计划，准备费作为model startup单列（与原资产加载同一既定setup边界），传感器epoch内任何首次encoder工作仍计入，不跳首输出或重置落后clock。仍保存G002无graph的全部冷启动失败，不以setup外移称未改冷成本。此次只能判已准备执行计划的服务范围；最终完整声明还需明确ready前启动/恢复成本和真实driver。任一超7ms或持续队列积压仍FAIL；不会因纯FK收益重跑2ms、高负载或开训。各job≤60s，仍计入Debug1h。

G005-B接入合同补充（执行前）：不是共享prev的单步对照。C0和C1分别加载相同旧初始化的两个独立model实例，eager/reference与显式graph计划各自递推；同G002真实train lyq_local[0,50)ms完整482事件、1ms微批、4ms query到100ms，共25查询，包含空尾段。每一步pose/完整vertices/joints要求bitwise，持有首份返回值并检查后续replay没有别名改写。原始模型文件与MANO不改，只在各自孤立model实例安装helper；结束恢复方法。单作业≤120s。原G002回放入口快照保存为jobs/g002_replay_source.py，新入口默认eager不变，仅显式--decode-schedule captured启用诊断计划。

G005实际：孤立3pose/FK/heads、A→B→A持有输出、flag及buffer失效拒绝全部PASS，5.22549 GPU秒；`gpu_decode_schedule_parity.json`。G005-B两控制独立eager/graph历史各25次（13次新事件、12次静止）全pose/778/21逐位相同，持有首输出不被覆盖，7.73487 GPU秒；`gpu_decode_recurrence.json`。MANO原实例方法在准备成功/失败finally均恢复，测试结束也恢复原model方法；无核心文件变化、训练或权重写出。

G006实际：两个4ms控制均完成，但仍FAIL_OBSERVED_OVER_7MS，25/25输出全部超限。C1完整response P50/P95/P99/max=454.555073/476.2562912/481.50081308/482.997653ms；C0=341.538582/365.73508/371.05643652000003/372.574023ms。原始完整时间线`g006_c1_captured_4000.json`与`g006_c0_captured_4000.json`保存，分别计6.58415/6.68198 GPU秒。显式model startup（含graph计划）分别907.196258/932.795447ms，不冒充已计入服务max。

C1有事件query含全输出copy中位7.218008ms，空query中位1.3388855ms；C0为11.631481/1.211084ms。这些分段仅用于根因，不能取代完整排队时间。最强停止证据：C1除首append外的其余34次append仍耗178.534378ms，超过50ms输入支持及100ms整个逻辑回放长度；即使把首次冷成本完全消掉并免除所有query，仍无法在这个输入序列/1ms微批合同下持续服务。故只继续预热、重复测时或扩大GPU训练预算无法修复本版本必要门。

**本次受控执行优化判别结束：当前C1运行器性能NO-GO；G004/G005调度机制保留为可追溯scratch，不提升为默认正式路径。** 不再重测2ms/更大负载、不启动screen/train、不宣称精度或创新有效。下一实质工作若继续，应先单独预注册稀疏token/邻居/缓存元信息更新的执行调度设计，保持同事件/同节点/同算子，证明去掉逐算子host开销能达到微批吞吐，再重过全prefix/独立长递推/梯度/缓冲失效与完整cold+恢复门；不能仅删除检查或将接收延迟移出统计。此项可能需要重做执行器，已超出本轮“只捕获固定解码子段”的单一修复，不伪称已实施。

## G007：事件更新纯tensor调度最小判别（2026-09-25新goal turn，执行前登记）

上一turn为progress：新增固定算序核数值/梯度证据、完整服务失败与局部解码候选反证；不是无进展或活作业等待。本turn实际检查handoff_gpu_stage所有已登记文件哈希不变；旧16个GPU job均终态，GPU当前0占用、1–7看似空闲，但仍逐job守卫，最多2卡授权未变。继续目标，不重跑G006原实现。

G003/G006支持先判别事件token/边张量运算的Python派发成本。最小候选为只把这两个无参数纯tensor函数移入TorchScript调度，在独占scratch定义与原函数逐项对应的实现；不改core/reference/checks、参数、全事件准入、SAE稀疏访问键、W/K/节点/1ms接收/2或4ms输出。TorchScript是执行机制而非研究创新；[PyTorch官方JIT概述](https://github.com/pytorch/pytorch/blob/main/torch/csrc/jit/OVERVIEW.md)说明fusion有运行时shape依赖及后端差异。本机2.1.0源码torch/jit/_fuser.py已读；不用已弃用nvFuser开关、不换环境，记录真实优化IR，不能因为调用script就宣称kernel已融合。

范围：先CPU compilation/语义≤30s、4线程；通过再单GPU≤120s。只接受FP32 xyp/int64 ts。固定seed20260925；长度1/2/7/17/31/64/117及真实8321 burst，含空/非空旧SAE、同像素极性ties、相反极性0龄/never、H边界、长原点shift、W边界。与当前CausalEventEncoder原函数分别比较token/SAE键时间/邻居索引/mask/dp；离散结构要求逐位同，浮点token/dp先用更强bitwise门避免无证改变边与递推；失败即记录并不安装。不能改topk tie规则或signed-zero来制造通过。

只在相同输入的预热诊断重复20次测阶段wall（含同步），另记首次调用/编译。此非完整时延；warmup和static-input repetitions不能替代真实冷回放。若这项小调度仍有明显不足，不自动叠多个后端/调参；记录残留成本，按预先设计的稀疏固定容量执行方案决定下一判别。完整实现与正式训练仍依赖原始full-prefix、独立递推、梯度、异常/状态、长期/恢复及完整7ms门。

G007实际FAIL，4.675830196007155 GPU秒。CPU单fixture编译/语义通过；GPU第二个case `n1_old37_shift0.tokens0`出现bit pattern差异，数值最大2.3283064365386963e-10。该门在运行前已冻结为包含signed-zero的bitwise，故不把“小误差”改判通过，也不安装script helper、不让它污染full_reference、不重跑同设置期望不同结果。默认TorchScript的执行优化不能在此次范围内被称为原语义保持的替代；具体差异算子尚未定位，不能武断归因某一fuser。剩余未执行case和阶段成本不填推测值。原JSON/job日志完整保留；额外tensor源码仅为scratch反证，无core修改。
