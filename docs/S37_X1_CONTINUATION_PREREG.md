# X1c 明确非精确优化器续接筛查预注册

登记时间：2026-09-26T02:47:06.666369+08:00。科学问题与门先冻结；此时未读X1任何新精度分数，未启动GPU。

## 问题与准入

旧X1 `x1_screen_s3407` 超时，原2500 checkpoint与TIMEOUT/INCOMPLETE永不覆盖。该网络直接编码原始事件，无prev输入，是内部测量候选筛查，不能单独成为用户要求的跟踪方法。R0的固定S37 64D读出阴性并未测量本分支效用；X1不因超时归为精度失败。

只回答：既有层级事件GNN经原2500优化器状态起步、一次固定新随机流补500步后，event-only预测与固定S37融合是否达到旧G-a/G-b效用门。它不证明事件测量独立于历史先验、不证明prev-free必要性、不证明时序创新，也不能放行昂贵主方案训练。沿现有C1r观测前提的未完成诊断，不增加第四主候选；其他候选HOLD/REJECT及S37保持。

独立准入报告：research_state/debug/X1_continuation_admission_20260926.md（完成后父复核；未完成不得视为通过）。

## 明确的新分支合同

- 原配置 configs/semkine/x1_hier_abs_s3407.yaml、原source outputs/semkine/x1_hier_abs_s3407/x1_hier_abs_s3407-step=2500.ckpt；源模型、Adam、scheduler及global_step=2500保持。源epoch/采样位置不认作已知的恢复点。
- 新分支名 x1c_hier_abs_s3407_b13407，独立输出 outputs/semkine/x1c_hier_abs_s3407_b13407；源文件不修改。
- **NONEXACT_LEGACY_OPTIMIZER_BRANCH**：旧Python/NumPy/torchCPU/CUDA RNG与sampler不可恢复，明确新设runtime/sampler随机流seed=13407，每rank确定性初始化，从新排列开头采样。数据cfg.SEED=3407不变，故(seed,index)增强不变；不得用改cfg.SEED代替runtime seed。
- 同2 GPU×512、BF16、Adam LR .004、warmup既已完成、相同训练集和损失。第一新优化器步2500→2501，总终点3000，最多新增500更新；不重置optimizer/scheduler，不增加epoch式隐性额外训练。
- 完整新checkpoint记录源SHA/globalstep/epoch/branchseed/provenance、每rank随机态和真实消费sampler游标；之后该分支内部支持精确恢复。启用RC1 strict deterministic执行，因而这也不是旧非确定性数值轨迹。模型架构与损失不改。
- 每100 optimizer步完整保存（相对原500仅提高安全保存频率，无择优），只评价终点3000，不从中间checkpoint选择。保留原validation cadence=1000；其日志不得用于选择重启/停止或调参。
- 新入口先通过legacy分支CPU实际优化测试及X1真实两卡BF16合同检查；普通complete resume仍拒绝legacy。任何缺失、数值不有限、OOM、不一致或资源超限立即停止，保留失败，不自动降低batch或精度。

## 必需 Debug（任何500步筛查之前）

CPU：实际legacy模型/Adam/scheduler/globalstep载入后首步前一致；新分支自身中间完整checkpoint恢复与连续轨迹逐位相同（输入/RNG/参数/optimizer/scheduler）；错误flag混用/源覆盖拒绝。保留RC1已验证行为回归。

X1 GPU：同源2500、同runtime13407、2×512/BF16/workers12，原dataset增强；Debug配置仅终点2504、保存每2步、独立output/log；训练步骤2500..2503。逐rank追踪step、原输入摘要、随机态、有限loss/梯度、首步前模型/Adam/scheduler身份。再从2502完整checkpoint恢复至2504，与此Debug分支连续轨迹逐位一致；最后模型/optimizer/scheduler与采样/RNG相同，严格确定性无unsupported-op错误，峰值每卡<44GiB。原validation interval=1000，2504不触发validation；不能把这4步转入科学分支。严格门失败则不启动X1c500步。

本次不是重做已过D0–D6，而是检查新增legacy分支与strict-deterministic执行对实际X1兼容性。Debug预算最多480GPU秒（timeout230秒×2，另cleanup10秒×2）；科学screen单次timeout2200秒×2，含cleanup4420GPU秒；评价screen单卡timeout600秒，含cleanup610GPU秒。合计分别受现有stage预算约束；不跨阶段挪用。启动前再次检查授权GPU1–7，排除GPU0。

## 固定评价与停止

只在新分支回执COMPLETED/exit0、source lineage正确、globalstep=3000、500次更新且完整checkpoint存在时，运行旧 `scratch/goal_20260925/main/x1_fixed_screen_eval.py` 同款G-a/G-b逻辑，唯一权重路径切换为新分支终点。需保存脚本diff/哈希；不得将2500或中间点替代终点。

- G-a：X1单独原H逐窗绝对预测，global≤13 mm且local≤17 mm。
- G-b：固定S37 seed3407@2500与新X1c@3000共享事件/共同递推prev，alpha=.5，根SO(3)最短插值、finger线性；X1 finger先加MANO mean再canonical、再减mean。原S37初始化、50ms窗口/评分不变。global≤11.5 mm且local≤17.5 mm；strong门global<10.66且local<15.1。
- 不做网格选择、alpha/seed/主体扫参，不因单动作收益忽略另一个动作。只一次原预注册zgz诊断，无进一步zgz驱动修改。
- 若G-a和/或G-b通过：仅保留这一个明确新分支为后续允许输入观测效用证据，仍须训练集/开发集上的机制、跟踪历史必要性、创新/成本门；不可采纳X1或双模型为最终主方案。
- 若失败：终止此固定测量分支/固定融合，不用新随机seed或延长训练救回；结论限于此网络/训练预算/非精确分支，不能外推原始事件没有有效测量信息。
- 失败/超时：保留INCOMPLETE，无精度结论，不无条件重试。

## 报告边界

全部内部效用数值、检查门、资源回执保存在本文件/ledger；不生成单种子伪主行。面向用户只按AGENTS生成baseline+当前S37九列表与简短裁决。历史表Latency不等完整event→mesh延迟，主目标仍未完成。

## 执行记录

待实际执行，不能以本登记充通过。


## 2026-09-26T02:55:10.047053+08:00 Debug v1失败及等价显存修复登记（实施前）

实际v1两卡首步前model/optimizer/scheduler/global_step继承均逐位通过，但严格确定性反向在首次optimizer更新完成前CUDA OOM；未生成新checkpoint，resume阶段未启动。回执FAILED/1，58.78554308204912GPU秒、cleanup完整，无残留；源2500 SHA不变。科学500步screen未启动，原v1失败保留。不能以此作科学X1效用阴性。

下一有界工程假说：EdgeConv与SetAbstraction的特征gather将每条边索引expand到C通道；strict CUDA backward需要大额索引/排序暂存。将特征采样改为flatten(B*N,C).index_select(batch_offset+node_id)可保留同一图、权重、激活和函数，降低索引暂存。只是计算实现修复，不变batch/精度/损失/网络层宽或科学门。

实施范围：共享EdgeConv增加默认关闭的compact_gather选项（S37默认原路径保留），仅X1内部EdgeConv启用；X1 SetAbstraction同用紧凑特征采样。3D位置索引无可学习梯度，原样。所有state_dict键/形状与参数量不变。

验证先于重启：CPU重复节点/混合批primitive与X1前向/梯度等价，float64 atol1e-12/rtol1e-10、float32 atol1e-5/rtol1e-4，输出采样逐位同；已有X1定义测试与S37入口回归。一个GPU小形状的原/新特征gather前向逐位、FP32反向atol1e-5/rtol1e-4，BF16前向逐位（不冒称旧/新BF16 reduction数值轨迹相同）；实际X1两卡同2×512 BF16运行v2仍必须原逐位分支恢复及<44GiB门全过。不得降低原科学batch/精度或放宽实际恢复标准。

v2 Debug预留单次230秒两卡+10秒cleanup，共480GPU秒（仍受总Debug预算）；小GPUprimitive检查并入该v2进程、同资源预算。只允许针对已定位实现问题的一次修复后v2，不因超时/未过门无限重试；v2仍失败则科学screen保持未启动。若通过，screen配方/终点/原G-a/G-b保持原登记。

实施前数值补充：官方strict index_select的宽slice kernel可能每次重复写回BF16，故紧凑helper对FP16/BF16先提升特征到FP32、选行后恢复输出dtype；反向重复行累加在FP32，最后只cast一次，与原gather的opmath累加方式保持一致。新旧小形状BF16梯度增加逐位检查（固定高重复index、固定seed），不只检查forward；若不逐位则记录差异且不据此降低逐位门。额外FP32临时边值如仍导致v2 OOM则停止，不再改batch或精度。

CPU实现验证（2026-09-26T02:58:53.392492+08:00）：`tests/test_x1_compact_gather.py tests/test_x1_event_hier.py tests/test_s37_routed_readout.py` 单次38 passed，日志scratch/goal_20260926/x1c/compact_cpu_v1.log；不重做拟合。新增legacy分支的15项CPU测试也已通过，日志scratch/goal_20260925/legacy_optimizer_branch_cpu/validation.log。v2真实两卡Debug已启动，未将启动当通过；科学screen尚未开始。


## 2026-09-26T03:00:44.210117+08:00 X1c必要Debug通过，固定screen准入

CPU legacy分支15项、compact/原X1/S37回归38项通过。v1首步OOM保留；v2仅修紧凑特征索引及低精度FP32累加，实际2×512 BF16 reference2500..2503、resume2502..2503两rank均完整。首步前模型/Adam/scheduler/step继承逐位，后续输入/RNG/loss trace及最终模型/Adam/scheduler/采样/RNG逐位相同；无validation发生。实际各rank峰值18.2497–18.3251GiB，原<44门通过。GPU小primitive前向两dtype逐位；FP32梯度maxabs0.000335693359375且原atol1e-5/rtol1e-4通过，BF16本固定例梯度逐位；不外推所有旧/新反向逐位。所有实际恢复门仍原标准。

两次Debug累计188.13862717803568GPU秒；全Debug6876.72060355579/12600，剩余5723.27939644421。v2回执COMPLETED/0 cleanup空，源码冻结核查通过。依据预注册启动唯一X1c screen：原2500→3000、新runtime13407、原cfg数据seed3407、2×512 BF16，checkpoint每100、原batch/LR/validation interval。只用新源2500，Debug优化未进入科学分支。此为固定筛查准入，不是创新/主方法采纳。


## 2026-09-26T03:16:17.255172+08:00 X1c固定500步终态、唯一评价准入

`x1c_screen_20260926_v1` COMPLETED/0，wall885.1768604159588s，1770.3537208319176GPU秒，cleanup无残留。父check_screen.py通过：两rank恰500次更新及梯度检查、原model/Adam/scheduler继承、globalstep3000、Adam各参数step3000、LR.004、source SHA未变、新完整checkpoint两rank各消费256000样本且RNG完整、源码/配方冻结。未以中间loss/ckpt选择。现在仅执行原固定G-a/G-b终点一次评价；结果未读取前门不变，GPU1由budget_run复核空闲，timeout600s。审计scratch/goal_20260926/x1c/screen_audit.json。


## 2026-09-26T03:19:40.809579+08:00 X1c终态：STOP_FIXED_X1C_JOINT_GATES_FAILED

一次固定终点评价已完成，COMPLETED_DIAGNOSTIC_ONLY。原G-a/G-b各自global子门通过，但各自local子门未通过，联合ga/gb/strong门均false。不能写成完全无效或原始事件没有测量信息；也不能以global收益抵销local失败。保留S37，关闭此固定新分支及固定融合，不扫seed/alpha/节点数/训练长度/中间checkpoint，不自动进入动态增益训练。

内部未舍入值（mm，仅本文/原artifact；非两种子正式主行）：
- G-a absolute：global 12.609850883483887 / local 24.587575912475586；门≤13/≤17，联合失败。
- G-b alpha=.5：global 11.285038948059082 / local 21.79408073425293；门≤11.5/≤17.5，联合失败；strong<10.66/<15.1也失败。
- 两策略均global1386帧、local1204帧。初始化/输入/50ms/H评分与旧固定脚本保持。

父核：训练终点3000完整状态、每rank恰500新步/样本游标/Adam步数/原LR、源2500 SHA未变；评价输入hash、终点SHA与训练审计一致；可执行源码冻结一致，重算联合门与原artifact一致。旧评价只存汇总，不声称做了保存逐帧数组的独立再评分。正式主行未生成，不能把本单种子/复用zgz筛查当独立泛化或新SOTA。

原X1 TIMEOUT/INCOMPLETE保持；新X1c是明确非精确optimizer continuation（新runtime/sampler13407、数据3407），不是补回原3000轨迹。新分支内完整恢复及compact gather工程通过不构成科学创新。C0r HOLD、固定C1r/R0 REJECT、广义观测前提HOLD、C2r HOLD保持；用户三项硬目标和唯一创新主方案仍未达。表中历史Latency不是完整端到端证明。

资源终态：本轮Debug188.13862717803568 GPU秒、screen 1843.499812023947 GPU秒。全任务Debug 6876.72060355579/12600、screen 37046.12625307795/50400、train 0.0/201600，所有预留归零。本轮四作业均按实际失败/成功保留且cleanup完整，PID/runner均消失，nvidia-smi此刻无compute进程；GPU0未使用。

后续科学问题：在local稀疏手指事件下，允许历史状态提供静止结构，如何构造可验证的局部修正而不过度覆盖历史；本次绝对估计/固定整手融合不能完成此要求。若提出新机制，须首先说明与已终止X1c固定规则、R0/R1、SU2、CI1等的明确区别和新增允许信息，再在训练/开发材料上预注册判别；不能从当前zgz结果反调此分支或把常见自适应增益称新颖。此处是未解问题，不是已获准新训练。短周期更新澄清仍待答，原H未变。

证据：scratch/goal_20260926/x1c/{eval_v1.json,screen_audit.json,debug_v2/receipt.json}、research_state/audit/X1C_terminal_parent_20260926.json及本轮4个budget_run回执。

独立终态复核：scratch/goal_20260926/x1c/terminal_review.md，同意上述裁决；同时补核Debug各rank inherited恰1条、gradient步集合完整。未运行额外实验，未改变阈值或候选准入。
