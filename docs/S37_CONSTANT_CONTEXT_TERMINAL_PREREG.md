# CC0终态附录 — 常量特征校准的有限效用与比较边界

2026-09-26T04:38:39.994739+08:00。这是执行后的事实记录，**不是新预注册或事后修改门**。原 `docs/S37_CONSTANT_CONTEXT_PREREG.md` 保持字节不变，源码/合同快照 `research_state/snapshots/CC0_frozen_sources_20260926/`。

## 终态与实际含义

`CONSTANT_UTILITY_WITHOUT_MATCHING_L`。U(C)和U(L)都通过：固定后续片段上，常量bias与已保存局部消息块分别相对原A有两seed local改善和global不退化。C的local均值更好，但global相对L的均值及一个seed超出预注册容差，**常量完全足够的联合门未过**。不因差距接近容差而放宽，不取C的local与L的global拼接、不选最好seed，也不能把本终态缩写成“bias无效”“L全面更优”或“邻居交互已证明必要”。

原固定SC0 F仍关闭。C是128参数、SC0 L的精确数学嵌套校准子模型；C/L带来的不同取舍保留为C0r的有限工程证据。它没有新增事件信息，不是常量姿态输出，无法倒推L实际用了什么。原L也可能通过自身特征重编码/额外优化获益，本次没有排除这些解释；不自动追加unary/affine/新头/解冻/更长训练。S37继续当前臂，不采纳新主方法，完整三项目标与科学创新仍未达。

## 冻结后缀的未舍入读数（内部诊断，非main row）

每个流都从原合法起点自行反馈跑完256个原H50ms帧，仅评分第129–256帧；两类序列和同一模型/策略固定。local/global是动作类别，MPJPE均为joint0平移对齐。没有在帧129重置GT、换入A状态或分动作选模型。

- A: global 23.45029067993164mm; local 40.33618927001953mm.
- L_3407: global 20.926959991455078mm; local 38.322086334228516mm.
- L_3408: global 13.456564903259277mm; local 38.87397003173828mm.
- C_3407: global 21.084667205810547mm; local 38.2958984375mm.
- C_3408: global 13.984333038330078mm; local 35.44363784790039mm.

C两seed均值local/global=36.869768142700195/17.534500122070312mm，L=38.5980281829834/17.191762447357178mm。C−L global均值=0.34273767471313477mm，高于原0.3mm工程容差；seed3408的global差也超过0.3。C local满足两seed及均值0.5mm容差。全部门保留：

```json
{
  "U(C)": {
    "passed": true,
    "gates": {
      "local_mean_vs_A": true,
      "local_both_seeds_positive": true,
      "global_mean_vs_A": true
    },
    "local_mean": 36.869768142700195,
    "global_mean": 17.534500122070312
  },
  "U(L)": {
    "passed": true,
    "gates": {
      "local_mean_vs_A": true,
      "local_both_seeds_positive": true,
      "global_mean_vs_A": true
    },
    "local_mean": 38.5980281829834,
    "global_mean": 17.191762447357178
  },
  "C_vs_L": {
    "local_both_seeds": true,
    "local_mean": true,
    "global_both_seeds": false,
    "global_mean": false
  },
  "C_sufficient": false
}
```

这些容差不是统计等价置信区间。后缀按序号预先固定，未用首128分数挑段；但它与前缀相邻、同一ylf序列、原backbone曾训练ylf且项目以前复用ylf，不是独立泛化或从未见过数据的证明。

## 真实性与数值合同

- CPU/CUDA各6项常量测试通过；SC0 W2=0,b2=b嵌套前向FP32/BF16精确一致，bias梯度按事前固定归约舍入界核验。FP32主训练不使用BF16梯度容差作为科学放行门。
- 实际原S37@2500和旧8真实train包，C零初始化输出与A在FP32/BF16相同；原g/坐标/mask合同、空/少节点、不读取额外h/dp内容、仅bias可变、原事件路径仍可变均有测试。新增128参数，不改变旧733830参数。
- 原SC0同16样本/同index/order的小拟合16步，首loss−0.6582049131393433、末−0.7248752117156982，有限有效梯度/更新，所有旧tensor逐位不变、空包prev保持；仅证明训练路径可工作。
- 实际4步连续 vs2+2恢复：model/Adam/空scheduler列表/step/RNG/sampler/恢复段loss全相同。该Debug为单卡workers0，不冒称两个正式500步作业实际中断过；原完整恢复机制及每100步checkpoint照常保存。
- 两个C正式fit均500步、batch16、FP32 Adam0.001且仅bias128更新，旧L/F完全未重训。C与对应L的完整train index、完整sampler order、前两batch实际events/ptr/prev/target/K哈希相同；其余批配对由同纯(seed,index)增强与固定order保证，不声称存了每批完整输入。
- 五流完整且有限；A/L三个流各前128预测和prev与已封存SC0 NPZ作dtype/shape/连续原始字节比较，通过signed-zero层面的回归。所有旧seal、六prefix NPZ、旧contract/summary身份在启动先校验。两段global/local原valid-run数量相同的预先断言通过，因此原跨序列init RNG消费保持；所有初值字节相同。
- 每个未跨合法run的时刻，prev字节等于本臂前一步pred；评分前保存全部pred/prev/GT/end/betas并封存。保存预测独立FK路径复算全部256帧分数与原evaluator≤1e−6mm一致，然后按固定slice评分128帧；复用同MANO，不称独立几何实现。
- 直接检查两个正式checkpoint的validation batch_progress总completed=0，见`validation_absence.json`。通用trainer的默认“按val_core选点”文本保留在training_metadata_original.json，当前metadata更正实际固定step500/无validation/无suffix选点；没有zgz参与。

## 实施、资源与下一决定

本轮仅新增scratch模块/测试/入口，核心和SC0所有源码及旧L checkpoint SHA保持。启动GPU前静态审查补齐GPU synthetic设备硬断言、旧封存身份、字节回归及run-count一致检查；预冻结版本保留cc0/pre_identity_revision，不追写成从未改动。无GPU失败重试、无科学门变化。

四个budget_run任务全部COMPLETED/0，PID/runner退出，cleanup无存活进程；GPU0未使用。CC0总134.67438118404243/640GPU秒；其中Debug8.657514239079319、screen126.01686694496311。全局累计debug6910.018195410958/12600，screen37363.602386470026/50400，train0/201600，预留0。未借阶段额度、续训、换seed或前缀。

父审计 `research_state/audit/CC0_terminal_parent_20260926.json`，独立终态复核 `research_state/debug/CC0_terminal_review_20260926.md`。本固定比较已经完成，不需追加探针来决定是否满足其原门。下一决策若推进C0r，应基于本次保留的真实工程效用，另预定义统一精度与完整成本的验证问题，不能把这次容差未过改成通过、也不能自动扩大模型。科学创新与最终验收单列；当前无新训练准入。
