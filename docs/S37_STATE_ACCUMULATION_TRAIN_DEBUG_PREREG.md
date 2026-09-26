# MP1 配对短训Debug终态与screen准入

2026-09-26T10:55:31.502301+08:00。唯一Debug COMPLETED/0，worker3082149/runner3082140已退出，cleanup空。GPU1 L20收费18.911669617984444秒，GPU0未用。

固定16包各16次全参数BF16更新；L原log10objective -0.6581183671951294→-0.955927848815918，P -0.6550718545913696→-0.9580127596855164。仅小样本拟合，不是MPJPE/开发集效用。全部grad存在有限、整体非零，参数更新、空包恒等、初始FP32旗标输出等价。

双臂连续4与2+2恢复逐位相同：模型、Adam、空scheduler、global_step、RNG、sampler、完整0..3input/loss前缀和续接后两步记录。正常stopped sidecar移至审计路径，恢复仅靠checkpoint；没有做OS kill故障注入。source/config拒绝为直接callback负例。父重新加载保存checkpoint逐项核对，无新模型或反向执行：`research_state/audit/MP1_debug_parent_20260926.json`。

固定同包计算中位步时×500=77.22346996888518秒；峰值L=3412979712、P=3418885120字节。两资源必要门通过，但不包括全训练CPU读取增强/摘要/保存开销，不证明900秒整任务必完成，更不是完整推理延迟。

按冻结合同准入四条串行screen：L3407、P3407、L3408、P3408，各500步batch16/LR6.25e-5/BF16/freshAdam，唯一最终端点，每条timeout900+cleanup10秒。完成后六模型A/L/P×seed全部ylf原H有效帧评估；不读取zgz、不修改科学门、原S37不自动采用。预算{"debug": {"charged": 7162.782417368726, "reserved": 0.0}, "screen": {"charged": 37363.602386470026, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}。完整目标active未达。

独立只读Debug终态审查已归档：`research_state/debug/MP1_DEBUG_TERMINAL_REVIEW_20260926.md`，SHA256 489718cd66e7a5d22e32d2be5546dd24f20bc615fbe4083ff2d5a292e6da5917。其只核seed3407工程Debug，未做OS kill、张量复算或开发评分；父保存checkpoint复核独立列出。
