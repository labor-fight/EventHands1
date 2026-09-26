# PA1终态：固定事件轨迹在local不足以支持配对诊断

唯一run `pa1_past_anchor_track_v1` 已退出0；终态 `COMPLETED_GATE_STOP / INCONCLUSIVE_FIXED_TRACK_SUPPORT`。真实二维前端和固定伪锚候选计算完成，联合支持门未过，因此没有读取target、计算姿态误差或启动B。不是精度阴性实验，也不是模型采用、材料身份、创新或完整7ms通过。

## 实际执行与身份

- 冻结合同：`docs/S37_PAST_ANCHOR_TRACK_PREREG.md`；副本`pa1/prereg_frozen.md`。49文件identity SHA256 `19c8b2b294bf5bd653e686a409d1444234a01dff3717f5476a62fe73595d7331`，包括代码、输入、MANO和18个官方来源文件；官方commit及MIT许可保留在vendor。
- 唯一命令：`python scratch/goal_20260926/pa1/launch_once.py`。worker PID3066845已退出；单CPU数值worker、4库线程，C++顺序运行且父等待，无CPU分片。worker wall4.173991453950293秒，launcher4.613799230894074秒，峰值RSS398312KiB，GPU0秒。无timeout，cleanup前后活进程均为空。
- A0两个真实native首秒的种子/事件输入/全接受ID链和所有快照先封印（9文件）；之后A1才读R0白名单自身历史、MANO、K，保存32查询候选/评分/选择再封印（5文件）。无新S37 forward、权重更新、GT历史替代或候选自身反馈。
- `run_v1/b`不存在，worker/launcher均`labels_read=false`；符合静态核对的门路径。身份核对会哈希NPZ文件原始压缩字节，不能与反序列化target混称。没有系统调用追踪证据，不把文本一致性当独立系统访问审计。

## 人工Debug及独立复核

最终10项前端接口和6项完整评分人工检查通过，包括193事件中间reference、同时间行身份、前缀、重置、origin、空/稀疏、局部候选、插值、配对共同常数、共同非法投影删行重配、完整分母和RA单位。没有模拟真实材料、背景/遮挡、长期漂移；这些检查不能证明跟踪质量或HASTE近似评分等于另一全量算法。

保留两处执行前修正。初次集成toy误把P最小候选期望写为q0；独立二次式导数证明应为q−，修正测试而未改评分政策，失败记录在`integration_debug.json`。独立G23发现非空种子前缀但后续无事件时完整性断言误报，修成`end>first`才期待末行、否则−1，并加人工例。K原生标定、全部固定源maxτ保守界、B整字段解压范围均在真实运行前明确，未按真实支持率改合同。

独立报告`research_state/debug/PA1_INTERFACE_REVIEW_20260926.md`最终覆盖pipeline SHA `6744c89e717182f1d7012b2a199f1405978faf19a7fc57c78e4718f85fc8c528`，确认执行前缺陷关闭。终态报告`PA1_TERMINAL_REVIEW_20260926.md`实际读取保存JSON并确认worker退出；它是独立文本/源码复核，没有独立重跑数值。

父`audit_terminal.py`已对49冻结身份、A0/A1封印、全接受ID链、32候选保存数组、投影残差重算、相对候选margin、选择/退回和门进行保存工件核验，`research_state/audit/PA1_terminal_parent_20260926.json`为`SAVED_ARTIFACT_AUDIT_CLOSED`。没有复跑tracker/MANO或读取标签。

## 内部支持数据与范围

固定200ms源锚：local25种子，2条running，ray-hit命中1/未命中1；global29种子，25条running，命中14/未命中11。所有32查询都保存，没有挑帧。

local每个查询共同源均为1，支持0/16，未达到8/16门；P一元素循环必为恒等，16个Main/P相对候选margin差均为0，Main/P全部回q0。部分查询Main本身仍有非零候选spread，故不能声称local完全没有几何敏感性。保存参与轨迹相对cut的reference年龄为22.588–278.901ms，明确是观测reference的年龄，不是实测推理延迟或因果输出回填。

global每个查询共同源均为14，支持16/16，候选spread和P相对margin差均非零；该动作覆盖门通过。参与轨迹reference年龄为0.272–38.139ms。没有B标签，不能推断选择优于q0/Hold/CV/P，更不能用这一动作的支持代替local门或认证材料对应。

上述差异只限定所冻结的活动种子、193接受事件前端、200ms锚、自身mesh首交和index6±5°政策。拒绝将其扩成“所有过去对应无效”“单目local不可观测”或通用信息下限。HASTE是现有算法，原生全事件额外信息/成本也不同于S37采样输入；即便有支持也不自动成为新完整GNN方案。

## 裁决与续接

关闭这一次固定PA1，不为global单独开B、不降低配对支持门、不调种子数/193窗长/源时刻/轴/幅度或延长首秒救场。不存在可恢复的在跑任务。新的实质机制只有在说明区别于PA1/NF2/M4/HE0、信息来源及可改变的实现决定后才能获得独立诊断准入；预算未花完不是理由。

S37 `s37_routed`保留；C0r HOLD、固定C1r/R0 REJECT（广义观测问题HOLD）、C2r HOLD不变。主表无新臂，无新zgz main row；表内Latency仍是历史scaled-forward，不能当作本次CPU测量或完整≤7ms。原H50ms/短反馈待答项不变；完整精度、服务及创新目标active未达。
