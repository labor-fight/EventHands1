# MP1 — 固定状态累加精度配对短训终态

时间：2026-09-26T11:11:17.445953+08:00。裁决 **STOP_FIXED_PRECISION_TRAIN_UTILITY_FAILED**。四条固定500步BF16训练及唯一全ylf原H评估均COMPLETED/0，不重启。两seed配对数据身份成立，但P相对L的local收益未达到冻结量级，且相对源权重A的global显著退化；不采用P，不改当前S37，不开正式训练。

## 原合同与范围

`docs/S37_STATE_ACCUMULATION_TRAIN_PREREG.md` 在真实训练前冻结，contract SHA256 fb3893aa9d1f3ccccb02c10ca04a8c78db5a3f0f023a161a3030bb2442703dda。两seed各自原S37 step2500仅权重热启动，fresh Adam、batch16/LR6.25e-5/warmup0/500updates/BF16、全参数；L旧尾部与P FP32尾部是唯一配对变量。这是新的小batch优化轨迹，不是原batch1024精确续接。八主体64训练序列排除ylf和zgz；ylf已被backbone和先前诊断使用，仅为重复开发诊断。

Debug两臂真实16包拟合、BF16 4与2+2完整恢复、输入追踪随checkpoint保存、身份拒绝/资源必要门通过。细节 `docs/S37_STATE_ACCUMULATION_TRAIN_DEBUG_PREREG.md`；独立源码/Debug审查均闭合，未做OS kill故障注入。四条科学训练每条实际500更新、每100步完整保存、全部500批L/P输入摘要一致。原trainer metadata通用selection_policy仍有“RA选点”旧字符串；本轮专属合同/receipt/eval只取final500，实际max500<val_interval10000、sanity0，没有运行选点工具或中途开发评分；原metadata保留不改。

每模型原H自身反馈、FP32推理、init RNG0且global→local、window=step50ms、GT+噪声段初值/GT betas/末端+1ms事件窗不变。全部有效global1226/local1222帧，共六模型14688输出，无截prefix/选择有效子群或丢失败帧。A/L/P均重新跑同完整分母，未拼旧prefix分数；所有目标帧、段初值、betas一致，段内prev严格等于该模型上一步预测。预测保存并封印后才计算联合门。

## 内部数值（不是zgz主行）

| seed | mode | local RA-MPJPE mm | global RA-MPJPE mm | local MPVPE mm | global MPVPE mm |
|---|---|---|---|---|---|
| 3407 | A | 27.895286560058594 | 24.849411010742188 | 21.63584327697754 | 18.723291397094727 |
| 3407 | L | 28.08826446533203 | 28.039756774902344 | 21.834575653076172 | 19.991735458374023 |
| 3407 | P | 28.05435562133789 | 27.792098999023438 | 21.777528762817383 | 19.843191146850586 |
| 3408 | A | 32.03074645996094 | 24.428482055664062 | 26.395692825317383 | 18.43570899963379 |
| 3408 | L | 29.306541442871094 | 32.15815353393555 | 24.19383430480957 | 21.333969116210938 |
| 3408 | P | 29.22153663635254 | 32.199920654296875 | 24.09075927734375 | 21.29106330871582 |

均值（未舍入）：

```json
{
  "A": {
    "local": 29.963016510009766,
    "global": 24.638946533203125
  },
  "L": {
    "local": 28.697402954101562,
    "global": 30.098955154418945
  },
  "P": {
    "local": 28.637946128845215,
    "global": 29.996009826660156
  }
}
```

P对L的local均值仅改善0.059456825256347656 mm，未过≥1.1mm门。P对A的local均值改善1.3250703811645508 mm，但3407 P local反而差于其A；不能用3408的变化代替逐seed门。P global均值相对A退化5.357063293457031 mm，超出≤0.3mm；P对L的global均值改善0.10294532775878906 mm，但3408 P global仍比L更差。所有个体值保留，不把mean通过扩写为全部seed不退化。

冻结五门为 `{"local_mean_vs_A": true, "local_mean_vs_L": false, "local_all_seeds": false, "global_vs_A": false, "global_vs_L": true}`。这拒绝本次固定精度短训的联合效用，不证明任何量化修复永远无用，更不把两个科学seed当一般分布结论。MP0的固定梯度更接近FP32仍成立；更接近FP32梯度并没有在这项匹配训练中给出足够的任务收益。

## 执行、审计和资源

唯一debug18.911669617984444 GPU秒；四训练合计520.523057546001 GPU秒；唯一eval203.23968954198062 GPU秒。均GPU1 L20，GPU0未用；六job cleanup空且worker/runner已退出。eval内部worker200.3150036300067秒与wrapper收费起止不同，不互换。无新推理延迟测量；表中历史scaled-forward不能解释为完整7ms。

原eval从保存预测重新MANO解码核原RA-joint/vertex结果；父随后加载保存checkpoint和NPZ，复核51冻结源码/四配置、500步/完整trace/参数变化、全部12个流的封印/分母/目标初值/自身反馈及五门，没有重跑模型、MANO或训练：`research_state/audit/MP1_terminal_parent_20260926.json`。父audit CPU0.803272349992767秒。独立终态文字/JSON复核已闭合，不冒称独立张量复跑。

累计预算：`{"debug": {"charged": 7162.782417368726, "reserved": 0.0}, "screen": {"charged": 38087.36513355801, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}`；无RUNNING/预留。源码/配置/权重/数据size-mtime/作业命令和封印均落盘；数据size-mtime不等于大文件内容SHA。主核心保持MP0 default-off实现，无新核心修改、提交或推送。

## 关闭与下一边界

固定MP1关闭；不得换seed、LR、batch、步数、端点或按动作选权重救场。MP0 opt-in可保留工程实现，当前采用臂仍是S37。C0r HOLD、固定C1r/R0 REJECT、C2r HOLD，未新增第四候选、主方案或创新通过。原zgz主指标及完整event→mesh7ms仍未同时达成，完整目标active未达。

下一准入问题先限于定义：M4的同观察绝对姿态等价变换是否同样改变可用于tracking的相对增量；若其只保留常量偏置而不改变增量，则仍须寻找会改变增量的同观察反例或明示过去信息如何约束该自由度。先对照已做NF2/PA1和原作近邻区分是否有新可改变的实现决定，不能把“相对”换名直接准入新模块/数据/训练。该问题目前未形成数值合同，不改变原H或未答的短反馈边界。

## 独立终态闭合 2026-09-26T11:13:18.162205+08:00

`research_state/debug/MP1_TERMINAL_REVIEW_20260926.md` SHA256 b6c7322bdb5630e8184850e0d6ce98b192e936a468f7197b871e3b5217644e59 已实际读取并绑定父audit；审查所读旧audit快照保留，审查仅文字/JSON/标量身份，不加载checkpoint/NPZ/GT或运行数值。确认固定T/F/F/F/T门和少量P对L改善同时保留，不把未达实用门写成每项都退化。原通用selection_policy遗留措辞由专属合同/实际final500身份解释，原证据未改。
