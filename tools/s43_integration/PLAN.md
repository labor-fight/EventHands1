# S43 B-R 隔离接入：最小改动与验证计划

2026-10-27，CPU-only。用户在上一轮停止扩大B-R之后明确授权工程接入、小训练、完整流式回放；本轮重新预注册。B-R仅通过原型实现契约，未通过精度门，不称已验证改进。

## 最小修改

所有生产源码与旧原型保持SHA不变。新文件限tools/s43_integration/、configs/s43_integration/、tests/test_s43_integration.py及实验记录。通过独立wrapper加载旧S43 checkpoint，旧配置/未开启开关时直接调用原forward，逐值对照；不修改model/model.py等冻结文件。

新增MODEL.S43_EVENT_REP=legacy|enhanced|br。legacy默认原last LNES。enhanced为last/count/first，通过必要6→2通道1x1适配再调用原S43。BR读取原phase核的K3相对证据及公开候选的预测LNES模板，学习小的候选混合/2→2适配，输出180×240×2事件图；仍走原conv1、稀疏ResNet、prev render/MLP、51D输出、SO3/MANO解码和_so3_fk_loss。BR不读取原LNES作旁路（除empty flag），不新增主干、MANO头或监督项。训练原conv1是否放开是独立增量消融。

原型参数：W50/1ms、低分辨24×32、rootx候选(-15,0,+15)mm、3个共享contrast(.8,1,1.2)、δ.12、ap=am2、cap8、alpha.9、nu.02、证据温度1000（上一轮FIT锁定）。stored polarity未认证，分数只称假设模型证据。CPU C++常色三角面覆盖加速须与原NumPy provider逐项验证，保留全部51个时间点，不降采样，不换似然核；这是等价实现优化，不是主干/损失新臂。

## 训练与在线共同合同

SequenceBank显式reset/ingest/readout/release，seq+segment隔离，max16 sequences/100000源每epoch；padding不进入状态。同时间戳multiset、source ID幂等、完整ms watermark、半开区间。ingest先更新受影响计数/SAE；phase完整bin推进仍是全KMPQ计算；readout可缓存但不提交手态。MANO候选FK/三角光度、CNN前向仍全局。

Tracker独立保存手态，query只输出，commit才更新手态。BR窗内H在reset时由历史状态冻结，前态改变须重建/同窗重放；连续S43输出一般不在3候选内，因此固定50ms commit后显式conditional_restart/uniform phase开启下一epoch，绝不继承不合法旧候选phase或声称全流Bayes。legacy/enhanced的原事件时钟独立保留。目标旧label[end]、事件cut=end+1ms，明确1ms fixed lag；历史起点的1ms hold近似沿原型，三臂共同。

批训练逐clip通过相同reset/ingest/readout函数。两步clip、TBPTT=1，每次手态/几何条件detach，状态不进state_dict、不跨batch或seq保图；padding/重启/segment reset显式处理。历史预热只读原先公共初态下两次因果S43预测，GT只作监督与评分。训练与在线比较逐prefix表示，而非仅比较最终预测。

## 冻结的小规模运行

seed3407、Adam lr=.001、batch2、每clip2步、100 optimizer steps，梯度范数裁剪5，BN固定eval。除表示/必要适配器和输入conv1 trainability，其余权重冻结；原loss取均值后使用原log10配置。先做真实冻结S43/真实事件的forward/backward smoke、旧路径精确parity、adapter梯度/empty/状态管理检查；未修的实现问题阻止科学裁决。

臂：A0原冻结S43；A1 legacy输入conv1小训练；A2 enhanced适配+conv1小训练；A3 BR适配+conv1小训练；A4 BR只训适配器（必要适配增量消融）。训练与评价顺序、实际clip索引与初值保存；全部100步，不据中途成绩改步数/seed。FIT使用ch/lfz/lpc/lr现有缓存可延续的前4窗口每录制组成clip（最多32clip）；SELECT ycy/ylf只评价不训练。若某尾窗无后继，仅按run长度跳过并记清单。

完整流式回放：至少完整ycy_local/global及zgz_local/global的所有valid run，hop50ms；按相同语料与提交时钟比较A0–A4。使用原评价指标实现（MPJPE/MPVPE root-align等），分开报告公共初态+100ms因果历史预热的可部署流与旧标准评价GT-assisted初始化；不得把新冷启动协议冒充正式主表。优先完整可部署流；若另复刻标准口径，使用原rng/初始化/窗口/滤波栈并单独标记。未跑的正式口径写未测，不挪用历史行。

前态影响：固定SELECT事件及目标，历史rootx偏移0/5/15mm，比较BR重新构造与旧摘要冻结的表示/输出；对legacy/enhanced事件表示应完全不变，但原prev render仍随手态变。纯query不得反馈造成伪恢复。至少一个完整录制另做同一时刻重复query对照，固定commit下预测/状态应一致。

## 验收与报告

必须：旧配置/checkpoint输出逐值兼容，batch/online同一前缀表示相同；混seq/padding/梯度截断/empty/reset有界检查；真实端到端forward/backward；所有小训练保存checkpoint和loss/梯度/实际参数与步数；完整录制无丢包/重放重复消费且实际已处理所有valid run；记录局部ingest、全局几何/phase/CNN/query/commit及内存成本，C++冷编译另列。

训练成功先指finite与非零可用梯度，loss是否下降及完整递推效果照实报告；不因进程返回0宣布有效。前态域差、conditional phase重启、固定K3覆盖、未做全6000步/多seed等剩余问题明确列出，不追加未授权大训练。

<!-- S43-INTEGRATION-PREREG:END -->
