# CC1：固定常量校准模型的同函数执行与成本归因

2026-09-26，数值执行前登记。父执行、真实独立代理只作静态审查/限定文件实现；不是新训练或第四科学候选。

## 问题与实际决定

CC0 的 C 与 L 在固定开发后缀均有有限效用，但 C 未满足相对 L 的全部 global 容差；终态永久保留。当前 C 仍沿用诊断 wrapper：原 EventGNN 全局池化/投影被随后覆盖，而且 ConstantContext 不读取的额外 context idx/dp 仍被计算。旧 B1/G006/F1/F2 没测过这个固定 C 路径。

本轮只决定是否交付**同一 C checkpoint 的独立、可严格加载的最小推理实现**，并测量 C 自身 wrapper→精简路径的成本变化。不比较 L 的成本，不挑 C/L/seed/scenario 赢家，不把 C 采纳为当前模型。S37 仍当前臂，无新训练准入。创新性、精度硬门和完整服务门均不能由本工程交付代替。

## 固定实现

只新增 opt-in `semkine/constant_context_infer.py`，原 EventGNN/model、SC0/CC0 已冻结源码不改。复用原 EventGNN.forward；在附加时关闭其 readout 标志但保留原 proj 参数，在外层按原公式仅池化一次。保留原三层 W32/k8 图、路由、状态更新、MANO 和所有权重键。

仅删掉 C 不读取的额外 context 邻居索引/几何。`previous_count = cumsum(mask)-mask`，degree 为 min(8,previous_count) 且只对 live query 生效；按原 8 槽生成 emask。保留 bias 的 autocast dtype、broadcast→multiply→sum→divide、`(h+residual)*mask` 的顺序，不能用实数等价 q*b 替换。全空事件轴补原 dtype 的零 out；混合空行、稀疏 mask、单节点保持原语义。模块不得依赖 scratch 训练入口，默认模型不自动接入。

## 身份和正确性门

只用两份 CC0 seed3407/3408 的已存 step500 checkpoint；源 S37 同为3407@2500。配置、MANO、数据与已封存 CC0 预测身份在运行前写入 `scratch/goal_20260926/cc1/contract.json`，启动逐项核 SHA/size/mtime。无 checkpoint/seed/片段选择，无新标签评分或 zgz 读取。

先执行 CPU4库线程的合成合同，再同一空闲 L20 上执行 GPU 合成合同。FP32/BF16、非零 bias、空/混合/单节点/稀疏掩码、严格 state_dict 与一次 proj 调用、返回值持有不被下次覆盖：前向值必须 dtype/shape/字节完全一致；不放宽 allclose，不验证训练梯度等价。

随后按 CC0 原两个序列顺序、原起点、原256帧与H50ms，分别用精简 C 连续回灌自己的状态；两 seed 全部 prediction 与 prev 必须和 CC0 已封存数组逐字节一致。旧 wrapper 同输入 pose/完整 mesh/joints 的对照也必须逐字节一致。起点用已封存的合法原初始化，不能在中途重置。只读取 sealed prediction/prev/initials，禁止以 target 重算或选新精度。任何正确性失败停止成本测试，不改变精度或容差门。

## 固定成本测量

正确性门通过后，参考/精简 C 对各自两 seed 都在原两个256帧前缀自身递推。固定顺序 reference3407→optimized3407→optimized3408→reference3408，序列 global→local。各序列五次同一初始包预热后重置回原合法初态，正式首 query 仍计入。计时前只预载不可变 events/offset/tsub、K/betas与权重；不预构 packet/token/图/mesh。模型/资产加载与准备时间另存，磁盘和实体传感器 transport 不在回放测量内。

这是**按记录时间戳模拟 host arrival 的有限负载回放**：序列时钟从第一窗起点开始，query release 为原 end_ms+1，前后相差50ms。按真实墙钟等待 query，前次尚未完成会排队；不把每次 anchor 重新置零来隐藏队列。服务从实际开始到完整 pose/vertices/joints CPU输出就绪，包含窗口构包、H2D、原图/网络/路由、MANO、完整D2H与同步。等待/排队/处理及输出响应分别保存。自身状态仍是模型上一步GPU输出。

每个原始输入事件按时间戳到完整输出计算年龄，保存逐帧完成时刻、query时刻、计数、最早/最晚事件响应及所有事件响应统计；空请求用 query 响应独立分母。报告 P50/P95/P99/max/>7ms 比例、原始事件量与2048采样节点数、固定20Hz输出、实际最长包与发生位置。原始输入事件被服务不代表每一个被网络保留或对预测有可识别因果影响。没有覆盖的实体传输、任意更大负载、突发/恢复不声称通过。

此固定策略的等待本身接近50ms；不能以计算变快声称最终≤7ms。正式主表 historical scaled-forward 不重写。不以C精简路径对L未精简wrapper选模型。本轮只报告两个 C 实现相同输入/状态下的成本，原CC0效用由严格预测身份承接，绝不拼成新正式主行。

交付门：全部正确性合同过门即可交付显式 opt-in 同函数执行模块；速度改善只在两 seed 的各自序列合并 service P50、P95均不高于reference时标记 observed cost reduction，否则记未确认成本改善，不选最优重复、不追加扫参。此门不包括模型采纳或7ms通过。

## 资源与停止

CPU单计算进程、4库线程；GPU只用启动前确认空闲的GPU1，GPU0始终排除。单个budget_run Debug作业 timeout240秒+10秒清理，总上限250 GPU秒，不借用screen/train额度。拟一个合成/实际parity/回放作业；不可因性能阴性增加重复。实现故障可保留失败后只修确切bug，但任何再执行须另记原因与同阶段预算，不能改判据。

运行前登记源码/config/checkpoint/封存预测/数据身份并保留快照；运行结束保存日志、receipts、输出数组和完整时间统计，立即更新STATE/JOURNAL/RUNS与失败账本。上一goal turn属于progress：实际完成CC0独立复核和证据图归档，无活任务需续跑。本轮完整目标active未达。
