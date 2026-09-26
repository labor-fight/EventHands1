# SC0终态附录 — 池化前远程上下文配对诊断

记录时间：2026-09-26T04:17:55.360113+08:00。本文件是**实际执行后的终态记录，不是新预注册或更改门槛**。原预注册 `docs/S37_SPATIAL_CONTEXT_PREREG.md` 保持字节不变，完整预执行版本及全部源码见 `research_state/snapshots/SC0_frozen_sources_20260926/`。

## 裁决

`STOP_FIXED_ADAPTER_UTILITY_FAILED`。F比A改善，但未超越同容量L的预注册local门，且一个seed不优于L。因此关闭固定远程适配形式，不能称远程信息已证明有效；不继续扫seed、索引间隔、宽度、LR、训练步数或解冻重训。S37仍当前臂，C0r HOLD，不增第四完整候选。

必须保留有限阳性：L/F共同适配在该开发前缀上都改善A，**不是“所有适配没有utility”**。这不能分清特征校准、额外参数优化、更新幅度变化或事件依赖交互；也不是独立泛化、主表精度、科学创新或端到端≤7ms证据。不能自动采纳L。下一动作仅先核对共同收益与旧P1/CI1/R0等最简单解释的区别，决定是否存在新的最小判别；没有准入新训练或救场。

## 固定终点实际读数（仅诊断，禁止混入主表）

源模型同一S37 seed3407@2500，适配器两seed各L/F500步、batch16、FP32/Adam0.001；原参数精确冻结。ylf_global和ylf_local各原H前128帧，自己的预测闭环，各臂同初始化。ylf曾被backbone训练和既往开发使用。local/global为动作类别，数字为joint0平移对齐MPJPE，不是坐标系。

- A: global 25.42074966430664 mm; local 42.287445068359375 mm.
- L_3407: global 16.12363052368164 mm; local 39.01381301879883 mm.
- F_3407: global 16.298173904418945 mm; local 39.75185012817383 mm.
- L_3408: global 18.18001937866211 mm; local 39.436195373535156 mm.
- F_3408: global 18.439098358154297 mm; local 38.191734313964844 mm.

L均值：local39.22500419616699/global17.151824951171875mm；F均值：local38.971792221069336/global17.36863613128662mm。F对L的local改善仅0.25321197509765625mm，低于预注册1.1mm；seed3407 F比L更差。门原样：`{"local_mean_vs_A": true, "local_mean_vs_L": false, "local_all_seeds": false, "global_vs_A": true, "global_vs_L": true}`。所有帧有限，全部五流预测先封存后计算门；没有选最好seed或中间checkpoint。

原 `semkine.eval_track.track_sequence` 只经scratch固定valid-run前缀封装，无core评测更改。每个流同rng0、global→local次序和初值，逐run重置、逐帧prev←自身pred；全部初值数组逐位相等。保存pred/prev/GT/end/betas/支持，独立从保存预测作MANO FK重算十个stream×sequence读数，均与原评分≤1e−6mm一致。五个模型源及实际checkpoint SHA见 `eval_v1/summary.json` identities。

## Debug与恢复证据

CPU adapter8项和既有CompleteCheckpoint15项通过；CUDA8项新图测试通过。实际旧checkpoint和既存八个真实训练包验证A/L/F初始FP32/BF16完全相同。两臂新的真实train16样本各拟合16次，原所有tensor冻结、仅4新增tensor可训、末层首步有梯度及首层第二步有非零梯度、末端loss小于初始、空包原prev保持。固定来源及度数边界/少事件/截断/原g合同均检查。

实际SC0回存恢复：4步连续 vs2步停止+2步恢复，停止点/终点断言、恢复段loss轨迹、最终model/Adam/scheduler/global_step/RNG/sampler均完全相同。Debug workers0；正式诊断workers4预取边界另有既有CompleteCheckpoint测试，不能将这次说成新4-worker恢复数值验证。每100步完整grid+终点checkpoints包含实际order、RNG和loops；每对L/F完整order SHA及前两batch全部events/prev/target/K SHA相同。旧weights-only source无原Adam继续，新optimizer从0起。

`training_metadata.json` 中通用入口默认写的“按val_core选点”标签在终态更正为实际固定step500无验证/选点，原文件完整保留为 `training_metadata_original.json`。真实val_check_interval10000且max_steps500，无dev/zgz选择。诊断manifest过滤后idx重编号，改变与旧训练相比的具体噪声实例；两臂严格配对，原增强分布规则保持，非旧轨迹续接。

## 实施、修复、资源

仅新增scratch代码，核心S37/默认数据划分/主行均未改。静态审阅发现NumPy bool不能JSON序列化、需显式MANO hash与初值身份比较，在任何screen启动之前修复并更新eval-only hash；两版合同/配置/评测源码与修复记录均保留。模型/训练/科学门未改，实际GPU Debug无失败重试。

六个budget_run任务全COMPLETED/0，cleanup完整；Debug 10.140244781039655GPU秒，screen 191.4592664471129GPU秒，SC0总201.59951122815255/1800GPU秒，GPU0未使用。总预算实际debug6901.360681171878/12600，screen37237.58551952506/50400，train0/201600，预留0。未借阶段额度、未续训或取更多dev窗口。

父终态审计 `research_state/audit/SC0_terminal_parent_20260926.json`；独立终态报告 `research_state/debug/SC0_terminal_review_20260926.md`。主表继续历史S37，不为诊断读取zgz制造main row；所有目标尚未完成。
