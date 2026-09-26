# MP0终态：固定训练批次的AMP梯度保真门通过

唯一CUDA job `mp0_state_accumulation_20260926_v1`已COMPLETED/0。终态`AMP_GRADIENT_FIDELITY_IMPROVED_ON_FIXED_BATCH`，仅指两份固定权重在同一既存8样本训练缓存上，保精度尾部的完整参数梯度比旧BF16尾部更接近完整FP32参考。不是新递推精度、训练收敛、所有数值一致改善或模型采用。

## 实现及实际验证

`model/model.py`新增白名单配置`MODEL.STATE_ACCUM_FP32`，默认false；true限定routed event_gnn的delta路径。只把最终state addition提升到至少FP32，FP64输入保FP64，原最终empty where保持。原头部、prev_mlp输入和头部求和仍可能低精度；没有新增观测、参数、权重键或默认配置变更。已有EGM/streaming类似算术与MP0原S37单变量问题分开，非首次发明。

CPU最终38 pass、18显式CPU FP16不支持skip（初批30/14，新增dtype/optimizer8/4）；CUDA56 pass、无skip/failure。覆盖同AMP raw delta的值和真实autograd参考、非空零/微小增量及五次累加、空/混合包、FP32非线性目标梯度回归、FP16/BF16/FP64状态dtype、原损失人工一次SGD及参数确实更新、absolute配置拒绝，另重测EP0默认合同。新开关下signed-zero位测试没有单独执行，不把默认EP0该项成绩扩大为新路径执行证据；最终where的保留另有源码依据。

既存8包CPU FP32：default/off、opt-in/on、原封存输出三者逐字节相同，输出SHA `c3904c1af153f81cce1f466ec55a1edb1249603462a245268a631e5ce3cf5789`。参数733830、state_dict键90保持，旧权重strict load通过。CUDA只比较同backend的F/F_flag，输出及所有参数梯度逐值相同，不宣称CPU/CUDA逐位相同。

实际probe为同训练批次、两个seed均step2500的F/L/P，原MSE→log10反向、全参数梯度非None/finite，未做真实optimizer step，所有state_dict值保持原权重。P与L各自尾部严格等于保存的同AMP raw delta加法参考；以此隔离了最后累加，不能消除更早网络误差。缓存target用于原训练监督梯度，不能说“没有读取标签”；没有开发/测试评分或新训练。

## 内部保真数值与反证边界

- seed3407：L/P相对F的梯度L2距离分别`0.997887647943305 / 0.8837196487206898`，相对L2为`0.05783033689432186 / 0.05121398697637035`；严格门通过。
- seed3408：L/P距离`1.0626413167415352 / 0.8868522537641071`，相对L2为`0.06622306861768612 / 0.0552680163376848`；严格门通过。
- 不一致改善必须保留：3408的log10损失相对F绝对误差从L `0.003215223550796509`变成P `0.00649300217628479`；3407最大51D输出差从L `0.05648994445800781`变成P `0.0568007230758667`。51D混合米/弧度，不能称这一最大值为mm或完整姿态误差。这些不是事后修改联合梯度门的理由，也不允许宣称所有数值更准。
- 门比较整个梯度，不保证每一参数块改善。保存全向量、names/sizes和父分块复核在`mp0/gradient_blocks.json`。更接近FP32梯度不保证更好统计泛化或自身闭环表现；8条旧训练样本不是独立留出数据。

## 执行、身份与恢复

执行前28文件identity SHA `57384217dd0c74e8dc5621812ca40719b18f1c8862c80b5abdd7b31fe6f51353`。早期冻结脚本引用了不存在的3408独立YAML，实际读取数组前修成原始共享3407配置+SEED覆盖；没有改变任何权重/候选/保真门。CPU阶段identity和prereg副本另留`identity_cpu_v1.json`/`prereg_cpu_v1.md`；CUDA前仅把历史parity的CPU backend及非线性测试目标措辞写清，科学门不变。

GPU1 L20、Torch2.1、strict deterministic、TF32关闭、4CPU线程；GPU0未用。预算runner实际收费`10.243856586981565` GPU秒；worker内部wall`7.354625948006287`秒，峰值allocated`1896976384`bytes。CPU历史回归wall`1.2786620379192755`秒。上述是整个Debug执行耗时，不是Latency或完整event→mesh7ms测量。

worker3072958与runner3072947已退出，cleanup前后活进程为空、无timeout。父单CPU从保存CPU张量重新核28身份、F/F_flag完整值/梯度、L/P范数、尾部舍入、56 CUDA及CPU XML和终态，未复跑模型/反向/拟合；审计`research_state/audit/MP0_terminal_parent_20260926.json`。独立前检`MP0_PRECISION_CONTRACT_REVIEW_20260926.md`与终态文本复核分别保留，不称独立数值复跑。

本轮core与测试的diff/源快照在`mp0/core_delta.patch`、`mp0/model_before.py`，既有dirty修改保留，未提交/推送。旧complete checkpoint仍因源码/config fingerprint变化按既有合同拒绝精确恢复；本轮未重做恢复试验。若要训练，应在新的冻结源码/配置内核对恢复，不改旧合同、不重启旧X1或其他已终态实验。

## 下一判别与采用决定

保留default-off的数值修复，S37及主表保持。当前C0r只有一个有实际保真支持的最小工程假说，仍无训练/闭环效用；固定C1r/R0 REJECT、C2r HOLD、创新门不变。

下一项有明确决策价值的工作是MP1配对短训：同一原S37初始权重、数据顺序/增强、batch/LR/步数/BF16，只改state accumulation；两seed各自原H FP32自身递推，与同预算旧尾部和原S37比较两个动作的完整分母。必须先另冻训练预算、恢复/优化合同和联合采用门，不把本次梯度门替代其精度门，也不能因后续失败改seed/步数/学习率救场。此项属于最小修复对照，不能据此宣称新的观测机制或CVPR级创新。

原H、短反馈未答项、完整延迟口径保持。完整目标active未达。

## 独立终态归档 2026-09-26T10:41:49.727646+08:00

独立终态审查 `research_state/debug/MP0_TERMINAL_REVIEW_20260926.md`（SHA256 e364b8b53f0e409723d0150de09ad56599e2867717d49076a67b3cb61fda5962）已绑定；只读回执/XML/文字与父audit一致性，不声称第二次张量数值复算。审查所读旧父audit已保存快照，再向现audit附审查身份。两项次要误差退化及零真实优化步限制完整保留，MP0无需重跑。
