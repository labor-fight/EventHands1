# MP1 — 状态尾部精度配对短训（执行前合同）

2026-09-26。MP0 的固定整梯度保真门通过，但 3408 objective 差和 3407 最大输出差退化；这不是训练效用。MP1 只回答一个工程问题：在匹配的新优化轨迹内，BF16 网络最后的 FP32 状态累加是否改善训练后的原 H 自身递推。它属于 C0r 最小修复对照，不是第四候选、创新通过或正式训练准入。默认 S37 保持。

## 固定比较与数据

两个种子 3407、3408，各自从原 S37 同 seed step2500 **仅载入权重**。A 是对应未短训的源权重；L 是原 BF16 尾部；P 只设 `MODEL.STATE_ACCUM_FP32=true`。模型全部参数可训练；不加模块/损失/状态噪声课程。沿用原 51D 加权 MSE 的 FP32 log10 objective、数据增强、30–300ms 训练窗、图/路由/空包策略。L/P 每 seed 同一索引及增强、初始权重、优化器和数据顺序。

新轨迹固定每卡 batch16、单卡、无梯度累积，fresh Adam LR=0.0000625（原 0.004×16/1024 的显式线性缩放，仅是本诊断的设计选择，不保证与原轨迹等效），warmup0、constant LR、BF16、500 次更新、仅最终 step500 为科学端点。不是原 batch1024 的精确续训，不能把与 A 的差单独归因于精度；核心因果比较是 P 对 L。源 optimizer/RNG/sampler 均不冒充恢复。禁止在结果后改 seed、batch、LR、步数、端点或选最佳中间 checkpoint。

训练采用原 train split 排除全部 ylf，64 序列/8主体；zgz 不读取。ylf 已被源 backbone 和此前诊断使用，明确是重复使用的开发诊断，不是新 holdout/泛化证明。原种子驱动 pure(seed,index) 数据增强；显式 `FixedSampler` 用同 seed 的 torch.randperm，恢复后按 committed cursor 续接。NUM_WORKERS=0，单 CPU 数值 worker、4 库线程；不做 CPU 分片。两臂保存每批输入摘要及完整 sampler 身份。

排除 ylf 后数据 index 会重新编号，增强因此不保证与历史九主体训练逐位相同，只保证本次同seed L/P配对。输入摘要包含全部实际输入张量（含时间间隔、序列和段边界），不把仅 events/prev 摘要称为完整packet身份。

## Debug 准入（无开发集评分）

MP0 已执行的值/梯度/空混合/精度矩阵不无故重复。MP1 新增以下实际 GPU 合同，任何失败停止 screen：

1. seed3407 的训练集固定首16个 randperm 索引，L/P 同一增强包；各16次 BF16/fresh Adam 更新。原目标有限、所有训练参数梯度非空且有限、总梯度非零、参数确实更新，最终同包 objective 严格低于更新前。空包更新后仍精确保留 prev。
2. 两臂分别在同一冻结新源码/配置内，真实16样本loader，BF16 连续4次与2次停止+2次恢复。逐位比较模型、optimizer、scheduler（此设计为空）、step、RNG、sampler 和续接 loss/input trace。源/config 合同不匹配必须拒绝。旧 complete guard 不变。
3. FP32 起始同权重 L/P 输出等价；相同输入的 BF16 输出与 MP0 同尾部公式合同兼容。统计 Debug 时间/显存，只作本作业资源准入，不能报为推理延迟。
4. 固定一次 Debug command timeout600s（预算含清理另10s）；训练每臂 timeout900s+10s、最终原H开发评估 timeout900s+10s。按 Debug 实测16样本优化步中位时长×500 必须≤720s、峰值allocated<40GiB，才能发起screen。Debug失败不能静默缩batch或改变LR救场。

## 自身递推评估与联合门

先冻结 ylf_global、ylf_local **全部 valid_runs_ms 的所有50ms帧**及身份，无 prefix筛选。各 seed 的 A/L/P 调用原 `semkine.eval_track.track_sequence`，FP32、window=step=50ms、noise_scale1、原 GT+噪声段初值/GT betas/末端+1ms事件窗、init RNG=0 且固定 global→local 顺序。每个模型独立反馈自己的预测；相同段初值需逐位核对。空包/低事件/失败帧不删，所有原有效帧为完整分母，非有限输出直接失败。保存每帧 prediction/prev/target/ends/initials 并从保存预测复核原指标。

冻结联合门（未舍入毫米）：P 的 local 两 seed均值较 L 和各自 A 的两seed均值均改善≥1.1；每个 seed 的 P local 严格优于该seed L和A；P global 两seed均值相对 L、A 的退化均≤0.3。各 seed 各动作结果原样记录，不能用 local收益隐藏global差。若过门，仅准入后续原H主协议与完整延迟验证，不直接替换当前臂；未过则关闭这项固定精度短训，不改诊断超参救场。

所有流完成并封印预测后才计算联合门。无中途开发集选点，无 zgz 调参。A/L/P同时使用FP32推理，所以本次验证训练轨迹效用；不把训练AMP设置偷换成新的推理策略。主表保持基线+S37；本诊断无zgz main row。完整端到端7ms、创新性和异步计算更新仍为独立未过门。

## 冻结与续接

`scratch/goal_20260926/mp1/prepare.py` 先只读split/valid-run元数据，登记全部源码、原权重、MANO、split、数据size/mtime和完整帧清单，生成四份配对配置及 contract。同一冻结源码内 CompleteCheckpoint 每100步保存，实际恢复须保留模型/optimizer/precision适用状态/RNG/sampler/step。任务经 budget_run，GPU0禁止，启动前检查GPU1–7空闲，不干扰他人；不会为占卡重复实验。MP1终态不得重跑，失败修正只可针对明确工程错误且保留首失败，不改科学门。

每个完整checkpoint同时携带截至已提交step的input/loss前缀，不能仅在正常训练结束写sidecar。恢复只依赖checkpoint恢复此追踪，评估从最终checkpoint读取0..499全部输入身份；Debug中停止sidecar移至审计路径后再恢复，并逐位比对完整追踪。四条screen任务串行执行以遵守单CPU数值worker；到达固定timeout为不完整终态，不擅自延长预算或重复启动。
