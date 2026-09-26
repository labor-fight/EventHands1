# TM0 / TM1：更早自身预测的条件纠偏信息

2026-09-25；父代理。TM0 为静态必要性/旧覆盖审；TM1 为一次固定缓存的内部经验诊断。**已按冻结合同完成一次CPU诊断，终态见§7；没有新主方案或训练臂获准。** 执行前全文保存在 `scratch/goal_20260925/tm1/prereg_frozen.md`。

## 1. 问题、最近对照与裁决边界

S37 当前函数只显式读一帧 prev，跨包回读 `prev←pred`；没有额外 pose lag。它仍是递推模型，不能称完全无历史。既有 E3 比较自身误差与随机噪声，R0 Bctx 只读当前 prev/context；旧 S12/G4 有实际 pose+velocity 滤波，EGM 有事件特征记忆。因此“加速度状态/记忆/滤波”不是空白，更不是新颖性证明。

本合同只问：**在原 S37 固定自身递推轨迹上，更早的自身预测相对明确的当前缓存表示，是否有额外、可读且对任务误差有用的纠偏信息？** 阳性最多为 C0r 下一项独立闭环必要性/Debug 提供依据，不推翻固定 R0 失败，不证明唯一瓶颈、独立观测、条件互信息或主创新。阴性停止本次固定历史读出形式，不换 lag/维度/正则/主体救场。

历史差分包含真实运动与估计误差变化，不是测得速度；直接加动量可能把原 head 已预测的运动再加一次。一般 pose 非 Markov 反例只证明存在可能性，不能代替本项目实际效用。本合同不要求先有普遍可辨定理、外观真值或 flow GT。

## 2. 固定数据与可用信息

只用原 `scratch/goal_20260925/r0/extract_v1` 的 18 个缓存：ch、lfz、lpc、lr、ly、lyh、lyq、ycy 的 local/global 共 16 fit 序列；ylf 的两序列为 internal dev。每条第一 valid run 的 128 步，原 S37 seed3407 step2500，自身回读，50 ms 间隔；原 contract/receipt/NPZ/provenance 身份必须一致。

backbone 训练见过全部九主体；ylf 前缀已被 R0 等评估、自适应复用。本合同的一次 dev 评估**不是新鲜留出、独立测试或泛化确认**。不访问 zgz，不增加序列或延长前缀。labels 只用于拟合/评分，不进入部署候选输入。

记 `p_k=prev[k]`、`z_k=pred[k]`、`y_k=target[k]`。A 阶段只解码 core、prev、pred、n_events、end_ms、betas；核 128 步、有限/类型/shape、`p[k]=z[k−1]` 逐位一致、end_ms 每步50、原首 run 内连续、core 末三维对应本 joint 的 prev。配置 flags 与原源码只核身份，不加载 checkpoint。

近期历史 `D_k=p_k−p_(k−1)`；k≥2 才保证两端均为自身预测。固定陈旧历史对照 `D_(k−8)`，比近期历史旧400 ms；统一只在 k≥10 且当前包非空时拟合/应用修正。所有输出前10步及空包均保持原 z，不删除评分分母。陈旧对照严格使用过去、不跨 run 或 sequence；不是保持 marginal 的随机置乱，只检验近期性，不作为独立因果识别工具。

## 3. 固定算法：先消除 ridge 重复特征混淆

仅改 45 维 finger 输出，原 root 六维保持逐位相同。每个 finger j 独立拟合三维存储角残差，使用共同 fit 时间样本；不把弧度平方损失称为 mm、SO(3) 距离或 mesh 风险。

当前表示 `X_j=[core_j(260),p(51),z(51)]` 共362维，加截距；这是原 finger 直接输入和完整当前姿态/预测，不包括全部原始事件或所有 GNN 节点，不能叫全部当前信息。fit-only 以 float64 求逐列均值/标准差，零标准差列尺度置1；dev 只应用冻结参数。

固定四输出：O 为原 z；C 为当前表示残差基线；H 为 C 加近期历史残差修正；S 为 C 加陈旧历史残差修正。C 的目标为 `y_j−z_j`，固定 ridge 最小化 `mean ||Xb−r||² + 0.01 ||b_nonintercept||²`，均值只按样本、三坐标平方求和，闭式正则为 `N×0.01`，截距不惩罚，不调 λ。

直接拼 lag 会通过重复当前列改变 ridge 有效惩罚，等参数或同 λ 无法排除。故对 H/S 各自先标准化51维 D（仅 fit 统计），使用同一 X 的非正则 SVD 投影 `P=X⁺D`，固定相对截断 `rcond=1e−10`，形成 `S_fit=D−XP`；dev 用同一 P。只相对于保留的数值列空间主张正交，不把有限截断充数学完全信息消除。

历史残差按 fit 标准差缩放，std<1e−6（已标准化 D 的单位）列固定置零，防止放大数值残屑。C 权重冻结，H/S 只对其剩余 target 残差拟合同样 λ=0.01 的51维 ridge，不加截距；不得重新分配 C 权重。不做多次 seed、lag、λ、方向、主体或特征搜索，不夹紧预测角度或挑子组。保存 SVD 保留秩/截断方向数、投影正交余量和历史有效列；不看数值再改截断。阳性只限相对固定线性当前基线的历史可读性，不排除更强当前非线性读出足以解释。

## 4. 阶段顺序与资源

1. 只读代码/旧 JSON，TM0 两独立报告；核当前合同与旧 R0/S12 的不同及局限。
2. 纯人工定义检查：重复当前特征不能产生有效历史残差；已知额外历史方向可读；冻结 C；未来后缀修改不改变较早的近期/陈旧 lag；序列/reset、warmup、空包、root 固定。此时不解码真实缓存或 target。
3. A：全部18序列输入合同、严格过去配对与完整 mask 封存；fit-only 归一化、SVD/历史残差和控制支持封存，再打开 target。任何失败停止，不改变门重跑。
4. B：只读16 fit 的 target，拟合一次 C/H/S；保存所有权重、normalizer、投影及全18序列预测并封存，之后才能打开两 dev target。
5. C：原 MANO 解码 dev 的 O/C/H/S 与 GT，joint0 平移对齐的21关节平均距离，所有128步计分；另保存参数风险与每步输出作内部解释。不得把被动固定历史成绩写作递推主行。

一次 CPU 数值 worker，库线程≤4，禁止多进程分片；GPU不可见、GPU计费0。外层 wall 上限180秒，随后最多10秒进程组清理；达到期限停止，不续跑、不重试。采用已验证的 budget runner 清理函数，单独 CPU launcher 不挪用 GPU 阶段预算。真实执行前再核资源；GPU0不涉及。资产仅 `assets/mano_right.npz` 与既有 MANO 解码代码，不读取原事件、完整模型权重或额外 GT 文件。

labels 的 access_started 必须在尝试读取前落盘，read_confirmed 在成功后落盘，分 fit/dev；超时不能误报“未读标签”。保存 A 封印、B 权重/预测封印、命令、身份、日志、wall/RSS、进程清理终态。

## 5. 固定门与否决

A 支持门：对 fit/dev×local/global 四组分别、逐 joint 计算；该 joint 须同时满足近期历史残差有效数值秩≥1（fit 退化列置零后，固定相对容差1e−10）以及近期/陈旧标准化残差至少50%的 eligible rows 不同（该行 float64 max abs>1e−6）。每组至少12/15个 joint 同时满足；不允许跨 joint 合并差异掩盖不支持的部分。不满足为 INCONCLUSIVE_HISTORY_SUPPORT，不读标签；不改阈值。保留所有 joint/行，不只评分通过支持的部分。

经验效用门（internal dev，非泛化显著性）：H 的 local MPJPE 至少比 O 低1.1 mm、比 C 低0.5 mm；H 的 global 相对 O 和 C 均不恶化超过0.3 mm。数值门沿用原 R0 所需的最小有用量级，执行后不放宽。若正收益成立，S 相对 H 的 local 恶化至少达到 H 对 O 收益的一半，才称近期性对照支持；无正收益不能单独把 S 更差当机制通过。

所有门通过仅为 `ALLOW_FURTHER_CAUSAL_HISTORY_DEBUG`；任何模型采纳、训练主方案、完整闭环与最终主行均未通过。效用失败为 `NO_FIXED_HISTORY_READOUT_GAIN`；效用过而近期性不支持为 `INCONCLUSIVE_RECENCY`。数值/资源/文件异常单列 `FAILED_OR_INCOMPLETE`，不冒充科学阴性。不无限重试、挑 dev 最好点或更换读出救场。

## 6. 执行前检查记录（原草案状态保留）

此草案尚待两独立 TM0 报告、对本数值合同的定向反方意见、实现、定义检查和源身份冻结；尚未有 TM1 数值结果。与用户仍待答的更短周期更新无依赖：不修改原50ms协议或反馈历史。当前 S37 及 C0r/C1r/C2r 裁决保持。


## 7. 2026-09-25T22:38:40.854620+08:00 一次执行终态

`tm1_temporal_history_v1`：launcher COMPLETED/exit0，worker COMPLETED_WITH_LABELS，**NO_FIXED_HISTORY_READOUT_GAIN**。A的四组均15/15 joint有支持；不是无支持或数值失败。16 fit的固定拟合完成，全部18序列预测封存后才读取两dev标签；每条128步完整评分，前10步和空包原样保留。实际原缓存没有空包，empty合同是人工定义覆盖与空集检查，不能声称本次真实空包已验证。

下表由 `run_v1/stage_c.json` 生成，仅说明固定历史上内部开发前缀的端点误差；不是候选自身递推，不是新鲜留出，不进入正式主表：

| 固定输出（内部端点诊断） | local RA-MPJPE (mm) | global RA-MPJPE (mm) |
|---|---:|---:|
| O 原S37预测 | 42.124414829798 | 25.430504490308 |
| C 当前线性基线 | 36.614422273378 | 37.556231678787 |
| H 近期历史残差 | 36.390516791372 | 39.753833342295 |
| S 陈旧历史残差 | 36.622224328242 | 37.749499098605 |

H对O的local改善为5.733898038426 mm，但对C仅改善0.223905482006 mm，未达0.5门；global相对O/C分别恶化14.323328851987/2.197601663508 mm，均超过0.3门。不能把局部单项改善拼接成联合通过，也不能说完全没有局部改善。C本身在global明显退化，H没有解决它；该结果不证明历史无条件信息或所有更强当前表示/历史方法不可能。

陈旧S相对H的local差为0.231707536870 mm；由于联合效用未过，按原合同标 **INCONCLUSIVE_NO_UTILITY**，不是近期性通过。停止本固定ridge/lag/残差化形式，不改λ、lag、warmup、主体或按动作切换规则救场。S12/G4、EGM、R0的旧失败继续保留。

实际资源：worker 4.181599915959 s，外层 4.662058702903 s，RSS 641.515625 MiB，单CPU worker/4库线程，GPU0秒；10次MANO批解码、0次tracker forward，没有更改任何反馈历史。PID 2801137 已不在，进程组清理前后无残留。这是诊断耗时，不是推理Latency或完整7ms测量。

## 8. 已完成检查与可续接证据

- TM0数学/最近邻与历史覆盖两份真实独立报告：`research_state/debug/TM0_temporal_state_necessity_20260925.md`、`TM0_temporal_history_coverage_20260925.md`；父核54条读时来源记录，含2个旧归档JSON成员，见 `audit/TM0_parent_sources_20260925.json`。新原文计数为0。
- 执行前9项人工定义检查一次PASS；独立17份来源静态终审与7关键源live身份绑定通过；62文件执行身份 `identity.json`，SHA `29cd456c621ada7ca35ee9ac6915730bb0131f77e6fcaac5cdfeb36ceecd8688`。不是62项算法测试。
- A/B/C各封存34/21/3文件；父保存输出审核全部18序列root/warmup/empty身份、两dev逐帧保存关节的评分与原门，见 `research_state/audit/TM1_terminal_parent_20260925.json`。没有重拟合、重跑FK或重读原target；参数风险仅核保存数组与聚合，不假称重新生成。
- 独立终态解释审：`scratch/goal_20260925/tm1/terminal_review.md`；报告SHA `a625be5b680c59f9c94f275cf43ccc98e652c37cfbbfc5d0407edf0526ecb896`，来源清单 `scratch/goal_20260925/tm1/terminal_review_sources/manifest.json`。执行前doc随后追加本节，旧runtime身份仍以frozen副本和封印为准。

保留S37。C0r/C2r HOLD，固定C1r/R0 REJECT，未产生第四候选或新的训练准入；完整目标仍未实现。此处关闭的是一个固定经验形式，不能用“历史不是独立测量”禁止其他具备独立必要性证据的经验研究。
