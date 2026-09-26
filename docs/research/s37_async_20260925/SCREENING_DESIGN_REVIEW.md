# C0/C1 短程筛查数据与训练入口审阅

日期：2026-09-25；审阅代理：`/root/literature_geometry`。依据 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) 最新授权：最多 2 张空闲 GPU，Debug/筛查/正式训练分别 ≤1/4/16 GPU 小时；当前 GPU7 的 G001/G002 由主代理独占调度。本审阅没有运行 GPU、模型、优化器或训练，也没有修改主代码。

**结论：现有 `semkine.train` 不能直接作为严格 C0/C1 筛查入口。** 可以复用 S37 模型、原损失、MANO 与新的 tracker，但必须新增很薄的严格流式样本/训练驱动，并把 C0 scratch 适配提升为显式版本化实现。已有 warm-start tiny-fit 仅证明局部优化路径，不证明从随机权重、中性状态启动的两个控制已通过 Debug。必需 Debug、完整性能门或以下数据合同有一项未通过，筛查保持 HOLD；本文件不是放行记录。

## 1. 已查实的数据和入口事实

- `outputs/semkine/s37_routed_s3407/training_metadata.json` 与 seed3408 对应文件都列出原九主体 `ch, lfz, lpc, lr, ly, lyh, lyq, ycy, ylf`；已选 S37 checkpoint 见过其中任何一个拟议开发主体。不能共同 warm-start 后把原 train 的子集称为未见开发。历史选点又使用 `zgz_global/zgz_local`，zgz 不能变成独立新测试。
- `semkine/train.py:99–101,129–136` 默认构造原 `train/val_core`，支持 `INIT_FROM`、teacher、resume；`SemKineDataset.__getitem__`（`semkine/dataset.py:334–379`）以 `pos51[start]` 作为 prev、带噪声后送前向，`_pack` 又携带序列 GT beta。它按 `valid_runs_ms` 建窗口索引并产生起止标记；直接复用会破坏严格推理 reset/历史输入合同。
- 同一旧 loader 用事件 `[start,end+1) ms` 预测 `pos51[end]`；新合同应在整数 query `q_ms` 只接收 `t<1000*q_ms`，监督 `pos51[q_ms]`。不能只把 WINDOW_MS 改为 2/4。标签有效条件为 `0<=q_ms<n_ms and valid_ms[q_ms]`；恰到 `q_ms=n_ms` 可输出但无该行标签，不能改用 `n_ms−1` 行。
- `tools/prepare_hand_data.py:42–105,259–338` 将原始有效、非插值注释的平移做 PCHIP，root 与 15 个局部旋转分别做 RotationSpline，产生 1 ms `.meta` 和 `valid_ms`。无效行初始化为零，**零行不是合法目标**。原始注释以事件首包 `header_stamp_ns` 为时标原点；translation 来源为 `camera_transl_right_canonical_m`。插值标签可合法用于离线监督，不能宣称每 2/4 ms 有一次独立物理测量。
- 本次额外只读检查了全部 72 条 train 的 aux 字段以及 `.meta` 的 4-byte header/文件大小，未读取 pose 数值、训练分数或验证/测试文件：每条 `.meta` 为 51 维且行数、`valid_ms` 长度、`n_ms` 一致；全部 canonical right、beta shape `(10,)`、K shape `(3,3)`；布局为 `[t3,R_root3,hand_pose_residual_aa45]`。原始注释时间差的逐序列中位数范围约 16.738415–16.778469 ms，说明 2/4 ms 主要是插值监督。本次未全量核验 pose 的有限性、magic bytes、标定数值、事件坐标和微秒时间内容；这些仍是新 loader 的 Debug 门。
- `semkine/streaming_tracker.py:87–157` 已显式从 `[0,0,0.45,0,...]`、beta=0、空图初始化；每次查询以自身预测递推，参数版本改变拒绝旧状态。`model/pose_repr.py:85–101` 给 residual 加一次 `hands_mean`，`model/model.py:1005` 的 MANO `add_mean=False`；所谓中性初始化实际是 MANO residual 原点/均值姿态，**不是 15 个完整关节轴角全部为零的平手**。新筛查应保留现有定义，避免同名换初态。
- `.research/.../tiny_fit.py` 从已选 S37 权重加载、挑有标签段、只递推两个 4 ms 包再监督末状态。它没有 GT 历史推理输入，但不能替代本轮随机初始化、固定时网格、较长 unroll 的 Debug。旧 `semkine.eval_track.track_sequence` 也不能复用为 C 开发评测：它从 GT+noise 初始化、按 valid-run reset，并给预测分支传 GT beta。

## 2. 唯一建议的 C 开发划分

采用在新指标产生前可重现的规则：**将原 train 主体字符串排序的最后一个 `ylf` 整体作为 C-dev，其余八主体作为 C-train**。这是新的、明确命名的开发协议，不覆盖原 `splits_semkine.json`，不恢复已退役旧划分。母清单和原训练权重只用于审计，训练入口禁止它们的默认回退。

精确清单可用以下笛卡尔积无歧义生成，并在机器 manifest 中展开为全部名称：

- C-train 主体：`ch, lfz, lpc, lr, ly, lyh, lyq, ycy`；每个主体全部 `{global,local} × {空后缀,_v2,_v3,_v4}`，共 64 条。例如 `ch_global, ch_global_v2, ch_global_v3, ch_global_v4, ch_local, ch_local_v2, ch_local_v3, ch_local_v4`；其他主体同式，名称必须与母清单精确相等。
- C-dev 八条：`ylf_global, ylf_global_v2, ylf_global_v3, ylf_global_v4, ylf_local, ylf_local_v2, ylf_local_v3, ylf_local_v4`。均保留完整流；不挑窗口或有利变体。
- C/final-test 清单留空：没有独立新增数据，F 保持 HOLD。原 `zgz_*` 不进入新训练、调参、选 checkpoint 或最终独立测试宣称。

本次 aux 计数：C-train 原始总时长 4,144,360 ms、标签有效 4,109,324 ms；C-dev 490,272 ms、有效 490,088 ms。固定原点、`q<n_ms` 时，C-train 的 2/4 ms 查询有效标签数分别为 2,054,612/1,027,292；C-dev 为 245,040/122,520。它们是可监督查询数，**不是独立样本数**。

`v2–v4` 不能当作额外独立受试者。此次发现 ylf 同类别四个变体的注释数量、时长及时间差中位数相同；旧 `docs/experiment_history.md:1137` 也把 v2–v4 称为变体。尚未核实每种变换的来源与完整内容哈希，所以以最保守的 `(subject,base_category)` 家族分组：全家族只在一侧，开发不对 8 条或几十万时点做独立样本 bootstrap。一个开发主体不足以支持跨主体泛化显著性声明。

## 3. 初始化、shape 和标签分离

每个 seed（预定 3407、3408）先 `manual_seed(seed)` 后 `MNISTModel(cfg)` 随机构建一次，保存其 state_dict 内容哈希；C0/C1 从这份共同随机初始化各自复制。两个 seed 彼此独立。禁止 `INIT_FROM`、`load_from_checkpoint`、`resume`、distillation teacher、已拟合权重、旧优化器状态及来自开发主体的预训练缓存。MANO 固定资产是双方共同模型先验，不视为本次受试者训练。

预测分支始终用 beta=0；GT beta 只属于 target/evaluator 的独立对象。建议最小筛查继续用现有 51D weighted-MSE：`model._compute_loss(pred, gt_pose)`，权重 `450/60/30000`、NORMALIZER=51，**该损失本身不用 beta**。不新增 FK/mesh loss，避免同时改变多个因素。若将来确需几何监督，应显式 `MANO(pred,beta0)` 对 `MANO(gt_pose,beta_gt)`；现有 `_so3_fk_loss`（`model/model.py:121–147`）给两支同一个 beta，不能直接用于这一合同。

评测也必须预测 mesh 使用 beta0，目标 mesh 使用合法 GT beta；不能把 GT 也改成 beta0 来消除 shape 误差。GT-free shape 可能形成不可消除的形状偏差，这是结果限制，不是改标签/指标的理由。两个 mesh 都加且只加一次 MANO mean。FK 的 16 个运动学节点与最终 MANO 输出的 21 个评分点、778 个顶点继续区分。

训练/开发服务接口只接受 `xyp,int64_timestamp,K,explicit_state`；不向模型传 `target/valid/beta_gt/j0_gt/sequence annotations`。监督器在输出后用 `(trial,query_us)` join 到标签，断言一一对应；替换整个标签文件、valid mask 与 GT beta 必须不改变同权重推理输出。相机 K 来自标定，调用路径不通过 `set_hand_context(gt_beta,K)`；为 `_ctx_betas` 放毒值的已有 NoGT 负控制应覆盖新训练驱动。

## 4. 最小训练流程与公平预算冻结

先由 G002 的纯工程结果选择共同 query 周期 `p∈{2,4} ms`，两者都不能满足已冻结性能门则不得开筛查。两臂微批 1 ms、50 ms horizon、相同 append 分包、相同事件源，C0 原均匀采样与 C1 末 N 的差异保留并记账；不能声称是同节点消融。

建议最小筛查采用 **64 ms 固定网格训练 episode**：每条 C-train 的 episode 起点为 `a=64k`，末尾不足 64 ms 的最后片段单独登记，不用 GT valid-run 选起点。每个 episode 从上述中性状态、空缓存独立启动，逐 1 ms append/逐 p ms query，以自身预测历史连续递推，监督其中所有有效 query。它是明确的有界训练课程，不是声称完整序列已验证；开发则每条从 t=0 只初始化一次，完整流无 GT reset，标签缺失只屏蔽评分。

64 ms 内无 detach、无 teacher forcing、无优化器更新；一个 episode backward 完成后释放所有 state，再 optimizer.step。这样不需要新增 TBPTT/cache 重计算抽象，符合当前“变权旧 cache 拒绝”合同。不可仅把 `_version` 改成新版本来沿用旧权重 cache。若随机初始化下 64 ms 已不稳定/OOM，先显式失败并研究最小共同修复；不临时缩短其中一臂或改初态。

按主体/序列轮转覆盖全部 64 条、每条的固定网格 episode 用 seed 排列，先生成相同的 episode ID 顺序并哈希，再喂两臂。训练 clip 可重叠 source family 的变体，但不跨 dev 主体；没有随机 window 切出“验证集”。零有效标签 episode 在清单中保留并记为无 loss，不执行 optimizer；optimizer 更新表提前从有监督条目生成并另存。无事件且有标签仍保留，无法取得梯度的纯空 episode 必须预登记为无更新并记账，不能训练中隐式补采易例。

为保持新增代码最小，第一轮双方均关闭 speed/domrand/polarity/hot-pixel augmentation；此设置是共同筛查合同，**不是复现原 S37 增强训练，更不能对历史增强 S37 作公平精度排序**。若主审计选择保留增强，则必须在开训前把同一 episode 的几何/K/时间变换及随机数流实现为共享预处理并补 Debug，不能训练途中开启，也不能以这项设计备选另开实验分支。

建议固定：单卡 FP32、每次 1 episode、无梯度累积；Adam LR=0.001、前 32 optimizer updates 线性 warmup 后恒定、global grad norm clip=1；episode 内有效 query 的现有 raw loss 先平均，再按原 `LOG10=true` 对均值取 log10；非正或非有限 loss 显式失败，不静默 clamp/跳过。LR、loss、batch、clip、augmentation 是双方完全相同的新 C 合同，不继承原 1024 effective-batch 的“同预算”名义。最多 1024 updates/臂/seed；实际共同 U 必须依吞吐、按下式在读取任何开发指标之前冻结。

预算的可执行冻结方式（均为上限，不要求耗尽）：

1. 全部 Debug 通过后，在**筛查 14,400 GPU秒内部**给无持久更新的 C-train 吞吐/内存校准最多 180 GPU秒。固定全 64 条各一个网格 episode，额外包含由 counts 决定的最大 64 ms 负载 episode；不看 loss 好坏挑样，不读取 dev 标签。两臂相同次序。测完整 forward/backward/update/IO 成本；临时校准权重全部丢弃、正式初始化重新加载共同随机 state_dict。
2. 4 个必要作业是 `seed3407×{C0,C1}` 与 `seed3408×{C0,C1}`。每作业预留训练 ≤2100 s、完整 C-dev 评估 ≤1200 s、加载/编译/写盘 ≤150 s；四作业共 ≤13,800 s，剩余 600 s 含前述 180 s 校准及有限控制。开发耗时未实测，1200 s 只是硬上限；超限即未完成，不能缩成有利子集。
3. 设校准中两个臂每 episode 完整 update 的最大成本为 `c_max`，取 `U=min(1024,64*floor(2100/(1.25*c_max*64)))`，在机器配置中写入该整数、测量 ID、共同 ordered episode IDs 与 LR schedule。若 `U<64`，此次预算不足以覆盖所有训练序列，筛查 HOLD，不开残缺一臂。该公式是预先约定预算规则，不是观察精度后改训练长度；它无法保证所有后续成本受界，wall cap 仍独立执行。
4. 某臂 OOM/NaN/超时或需实质修复：保存失败与已用 GPU秒，停止依赖；不因另一臂更快而让它多训，也不从正式训练 16 h 借额度。四个完整配对作业未完成不宣称双种子可重复改善。单卡顺序固定 `C0/3407,C1/3407,C1/3408,C0/3408`；真有第二空闲卡才由主调度分配，不改变 effective batch、U 或样本表。
5. 每次训练只保留固定终点 U 为主 checkpoint（可另留恢复 checkpoint，但不得选其中最优）；开发指标仅在该终点计算一次。后续机制控制仅在主差异出现后、剩余额度可负担且控制合同预先登记时执行；不足则科学归因 HOLD，不能由主对照胜出自动认定异步机制成立。

这是有预算约束的建议方案，尚未测得新随机初始化训练吞吐/完整开发成本，因而不能保证 4 h 内完成。约束不足时正确结论为 INCOMPLETE，而不是只报完成得最好的一对。

## 5. 开发指标、门与信息泄漏防护

C 开发仍按现有指标数学定义：21 点关节 root-align 后的误差在 `*_local*` 与 `*_global*` **动作类别**内分别汇总；这两个名称不是局部坐标/全局坐标误差。MANO 顶点与关节使用各自原定义，不额外做 Procrustes/旋转/尺度对齐。现有 `root_align` 对输入减其第 0 点（包括顶点指标）这一事实须通过 evaluator parity 固定，不能顺手更改历史公式。只把预测支路与 target 的 beta 输入合同明确分开；H 历史分数不能直接与 C 排名。

所有 C-dev 全流预测先只按 K/events/state 生成，然后由独立评分器加载 valid/GT。记录所有 query，包括冷启动、低事件、无效标签、静止和恢复；只在有效 q 上评分，不因误差过大删除帧。不依据 GT valid-run reset、裁段、warmup 丢帧，或重新用 GT 纠正历史。模型失败输出不能当 missing label 忽略。`local/global` 主指标用有效帧数加权，逐 trial 原始值保留；同一次 checkpoint 两者同时判断。

最小短程门建议在开训前冻结为：两个 seed 均完整完成、全 dev 流无数值/状态失败；相对共同初始化有训练可学习证据；C1 相对 C0 的 local 和 global 在两个 seed 均严格降低才记为“开发方向性通过”。不以四个数字里挑两个拼接过门。此门只允许进一步评估价值，不等于 15.1/10.66/7 ms 达标，不等于显著泛化、H1 归因或 SOTA。一个开发主体、两个 source families 的支持很弱，不提供伪精确独立样本置信区间。

需写入且校验的冻结 manifest 字段：母 split SHA256；展开后 train/dev 名称与主体/家族交集为空；raw events/offsets/tsub/meta/aux 的内容 SHA256、长度及路径；MANO/source/config/adapter SHA256；seed、初始 state_dict SHA256、optimizer 初态；query schedule、episode 表、loss mask hash、U、实际有效 query/event 数、完整训练与开发耗时、失败原因。完整 events hash 可能涉及大量 IO，本次未执行；可在有预算的准备步骤顺序读取并缓存，不能用 header/stat 代称内容 hash。不得修改共享原数据或母 manifest。

## 6. 真正必要的最小新增代码和 Debug

1. 一个显式 C split/episode manifest 生成器或现有工具的小适配：不默认回退 `val_core`；固定主体/家族、时网格与哈希。可以先放统一 scratch 检查，但训练正式依赖须版本化。
2. 一个小流式 dataset/reader：复用 `_read_meta51` 和原 memmap 逻辑，按 query 严格构造 int64 时间，输入、标定与监督对象分开；禁止 `SemKineDataset.__getitem__` 的 GT prev/beta 包复用。标签构造函数可以单独单元测试；不复制整套原 dataset。
3. 将 `.research/.../packet_control.py` 显式提升至版本化控制模块，并给 `tools/train_streaming.py`（建议名）一个 `--arm c0/c1` 分支；共享训练循环、loss/optimizer/episode 次序、tracker 初态与输出。它只需普通 PyTorch bounded unroll，不必为此重构 Lightning 原训练或复制模型。
4. 评分器小适配：共同 full-stream driver + 现有指标公式，pred beta0/target beta_gt 两条显式 MANO 调用；先输出预测再独立加载标签，禁止调用旧 `track_sequence`。

新增相关 Debug 必须在正式筛查前通过：2/4 ms 边界、q=n_ms、无效标签零值、不同主体/变体隔离、meta finite/magic 和 shape/单位；原始 tsub 范围/非递减；标签/GT beta 毒化不改前向；双方同随机权重哈希；从中性初态的 64 ms 前向/全部预期梯度/Adam 更新有限；empty/低事件/8321 burst；optimizer 后旧状态被拒绝且新 episode reset；new loss 的 log10/平均次序；现有功能与 C0 packet parity；新随机初态的小样本拟合与完整开发只读入口。只根据具体失败补最小修复并重测相关项，不用完整训练来完成这些 Debug。

## 7. 尚未验证的限制

开发划分和训练配方是本审阅建议，还没有机器配置冻结、运行或选中；唯一写入是本审阅文档。未运行新标签解析器、全部 raw 时间/坐标验证、随机初始化梯度/拟合或评测 parity。未查清变体生成机制和跨主体内容重复，需哈希/来源家族核验；若发现跨主体同源，应整体分组后在看分数前重新冻结。固定 beta0 的 shape 误差、中性 0.45 m 的捕获范围、64 ms 训练能否支持分钟级递推均未知。最终独立数据仍缺失；不能把这个完整开发主体重新命名为 test。

## 8. 候选机器合同登记（先登记，后执行）

主代理确认 G002 四条性能试验均未通过，当前仍在执行调度诊断；**不训练、不新增 loader/训练代码**。本项只允许生成 `.research/s37_async_20260925/screening_contract_candidate.json` 与 `screening_contract_candidate_checks.json`，状态固定 `PROPOSED_NOT_TRAINING_READY`。依本文件已声明的名字规则展开 ylf 全主体留出 64/8 清单、以 `(subject,local/global)` 分组全部四个变体，记录标签/query 边界、每 seed 两臂共同随机初始化及 INIT_FROM/teacher/resume 禁用约束、最多 1024 updates 和共同 U 的预注册预算规则；不会根据模型分数选名单。

只用 CPU 标准库、至多两个 CPU、单次 wall cap 60 s；计算 split/config/代码/原训练元数据/MANO 与 72 条 `.meta`、`_aux.npz`、`_offsets.npy` 内容 SHA256。events/tsub 大文件只读文件 stat 和 NPY header，明确内容 SHA 未计算、内容未验证；meta/aux 哈希读取全部字节，但不计算姿态误差、不调用任何模型。标准库检查 64/8 数量、不交、并集等于原 train、主体/变体 family 不跨、名称展开一致，另保存检查 JSON。失败保留显式失败状态，不自动反复运行；任何校验通过仅为数据身份合同，不是性能/训练放行。
