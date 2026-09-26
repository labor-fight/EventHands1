# AS0：窗口内预览与端点回灌的实际状态就绪合同

2026-09-26，保存时钟数组读取前登记。本轮为既有CC1工件的单CPU依赖分析，无模型执行、训练、GT评分、GPU或新服务策略实施。

## 问题与边界

完整目标要求同一checkpoint/输入/状态策略同时具有原H精度和完整event→mesh≤7ms。当前H每50ms才回灌；CC1已证明其固定完整输出成本仍不满足7ms。一个候选解释是：固定窗口起点的committed pose，窗口内从同一起点状态与已到达前缀产生provisional mesh，H端点才commit，不将预览反馈。它在数学上可保持端点状态链，但端点相等既不证明中间姿态正确，也不证明实际调度可行。

本轮只判断：**在CC1当前同步、完整输出返回后才向调用方发布下一pose的接口上，直接附加该类预览是否能满足所有输入事件7ms响应？** 不改H、不给预览算精度、不把它作为新的低延迟主方法，不借此绕过未答的短周期反馈问题。新异步pose发布/mesh解码解耦、跨stream流水线、历史外推或旧状态预览是不同合同，不在本否定范围。

前提必须明确：CC1实测service结束时间是完整pose/vertices/joints CPU-ready，**不是单独pose的GPU-ready时刻**。仅对原串行返回接口，committed pose的发布不早于这个完成时刻，故可把它作为最乐观的发布下界；不能据整体service反推任意重构流水线的pose-ready下界。源代码verify.replay在完整输出及少量诊断记录后才执行prev=pose，因此真实发布还可能更晚。

## 冻结计算

输入为`cc1/run_v1/summary.json`、原8个定时NPZ、原冻结contract、verify.py与父终态审计。先核对父audit已封存NPZ SHA与所有原source SHA，无新数据或checkpoint读取。8个流均分析，不按seed/场景筛选。

NPZ的每行有固定H end、事件数、release/begin/complete时刻；逐事件保存了当前完整输出的response age。对每一行按事件计数从flattened response数组取对应片段，重建该事件的模拟到达时刻：

`arrival_ms = current_complete_ms - saved_event_response_ms`。

核对事件总数、数组有限性、50ms release间隔、begin≥release、complete≥begin，以及每个重建arrival位于该窗`[release-50,release)`；允许仅用于保存时钟浮点减法的1e-5 ms舍入边界。不放宽模型parity或科学精度门。

对第j≥1窗口需要的起点状态x_j，以`previous_complete_ms`作为原同步发布时刻的乐观下界，即使假设新预览的所有计算都耗时0，也至少等待：

`wait_lower_bound_ms = max(0, previous_complete_ms-arrival_ms)`。

首窗口初始化原本已就绪，lower_bound定义为0；主判别分母为后续255窗口的所有原始输入事件，另保存首窗口数量，不以首窗口稀释后续依赖。记录每流n、P50/P95/P99/max及>7ms比例、受影响query数和最大反例所在窗口。记录正等待和超过7ms的事件数，不声称每个raw事件被2048节点采样保留或必然影响预测。

必要反例门：任一后续输入事件的上述依赖下界>7ms，则拒绝“当前同步返回接口直接加预览即可兑现完整7ms”的形式；无需实际执行预览。若没有，依赖门只记未否定，仍不准入训练/采纳，真实处理与中间精度未验证。

同到达时间的机制负对照：把状态发布人为设成理想的上一逻辑端点（当前窗start），所有wait下界应为0，至多1e-5ms舍入；这是**不真实的零耗时发布反事实**，不能当优化后的性能结果。它验证本分析的非零量来自实际发布依赖，而非错误把50ms lookback当固定等待。

另执行几个固定人工例：端点零耗时发布、发布延迟10ms的早/晚事件、前一计算越过下一release的排队、第一窗口已初始化；预期写在代码中先于真实数组读取。不是重新跑CC1计时、不是模型性能增益实验。

## 决策与限制

若拒绝固定形式，下一真正实现问题是明确pose-ready/mesh-ready的任务依赖图及允许的状态/输出语义；不能把整体service直接等同pose-ready，也不能仅将旧CPU完成标号换名变成早期状态。中间输出必须另有准确率、因果支持与同一策略的验证合同，不能把H端点分数配预览延迟。

该分析不声称所有preview/commit设计物理不可能，不论证科学创新，不增加第四候选，不重开已有SC0/CC0/CC1/G006/G007或改H。固定输入表示的prefix全图失效已由旧G24审查覆盖，不新增相同反例或以预览名义重复训练。

CPU单计算进程、4库线程，60秒上限；仅numpy保存数组/时钟，无GPU。当前上一goal turn属于progress：CC1独立推理实现、真实parity和固定host回放已全部完成并独立复核。完整目标仍active。报告与身份落research_state/audit，机会图只添加有范围的依赖事实。
