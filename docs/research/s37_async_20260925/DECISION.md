# 研究决策与最小接入设计（实现前冻结）

日期：2026-09-25。当前采纳的实验臂仍为 S37。以下决策只授权进入最小实现与 CPU Debug，不是精度采纳、GPU训练放行或创新性确认。完整目标保留在 STATE.md。

## 1. 决策依据与问题重述

锚点：S37 已从 LNES 迁移到原始事件，但当前 token、采样和图每包重算，没有可审计的异步 runtime。其独立状态无关事件证据与关节路由仍是有效接口；现有三维图/记忆和精确关联干预没有证明应替换主干。最新 georoot stage0 四个原始 JSON 已完成，但未过其自身准入门；不得因为旧文档“待填”而重新启动它。

主挑战：如何在不引入额外传感器或真值历史的前提下，把只到当前时刻的稀疏观测持续编码为可重用证据，支持三维手状态的及时校正，同时检验跨包证据是否真的改善低事件率跟踪？

- 抽象上移：把“证据记忆”和“假设依赖的几何关联”分开，避免历史预测反复被算作新测量。
- 下移：追加事件是否改变旧特征？姿态改变是否只要求重算路由？相同 prefix 的全重算和增量输出是否一致？
- 反转：更频繁更新可能只是更频繁注入噪声；跨包记忆可能固化错误关联，必须有空观测与等信息负控制。

## 2. 20 视角交叉审查汇总

这是三个真实代理分组的二十视角，不是二十个独立代理。完整证据/最强反例/最小试验见 AUDIT_AGENT.md、LITERATURE_EVENTS.md、LITERATURE_GEOMETRY.md 及源 ledger。

| 视角 | 决策证据 | 最强反例 | 最小判别 |
|---|---|---|---|
| ① S37 | token/span与采样非prefix稳定 | 因果边仍需全重算 | 固定prefix追加未来事件 |
| ② 失败史 | xyz图/记忆及georoot已有反证 | 改名继续失败机制 | 先审已留JSON及原门 |
| ③ 单目几何 | 一像素一射线，无唯一z | 同投影不同尺度/运动 | 投影不变反例 |
| ④ 事件生成 | 亮度变化非几何边界标签 | 内部纹理、闪烁也发事件 | 保count破坏时间/极性 |
| ⑤ MANO | 16 FK/778 surface/21评分点不同 | LBS不是观测置信度 | 同指/跨指权重扰动 |
| ⑥ 可见性 | 顶点front规则有近似风险 | 精确ray不保证闭环改善 | 固定状态/节点的factorial诊断 |
| ⑦ 异步GNN | AEGNN及官方BN有近似边界 | 忽视归一化/删点失效 | 全prefix逐层等价 |
| ⑧ EventNet | 解析历史递推已有先例 | decay本身不能证明新信息 | 同计算量无时序对照 |
| ⑨ 稀疏CNN | 活跃更新需依赖传播 | 空间扩散成本随层增长 | 统计真实更新节点数 |
| ⑩ 稀疏Transformer | token少不等于实际更快 | routing/sort/launch主导 | 完整wall-clock分解 |
| ⑪ mesh/姿态 | EvHandPose已有visible mesh flow | 几何先验冒充测量 | 相同观测下破坏关联 |
| ⑫ mono3D | learned depth受尺度先验约束 | 汽车尺寸假设不迁移手 | shape/depth等投影对 |
| ⑬ RGB stereo | 可借对应失效逻辑 | 偷用第二视图可见性 | 去第二视图后重写合同 |
| ⑭ event stereo | TESNet/EMatch的第二视图越界 | 过去帧非已知baseline | 独立手指运动对应反例 |
| ⑮ flow/seg | normal flow只约束一方向 | 对称邻域偷未来、aperture | past-only+time毁坏控制 |
| ⑯ tracking | 过去特征能保留信息 | 平滑改善均值但增加lag | 急停/反向/长期递推 |
| ⑰ VO/SLAM | 静态场景前提不适用全手 | 手运动被错误当ego | 相机/手指运动独立合成 |
| ⑱ 不确定性 | 观测信息与先验分开 | damping满秩不代表可观测 | 空观测不能产生新信息 |
| ⑲ 性能 | 路由和mesh刷新不能漏计 | 只报forward平均造成功 | 完整replay尾部与队列 |
| ⑳ 公平/反方 | GT init/shape和复用zgz有边界 | 新协议低分与旧协议混比 | 同预测评测parity+新封存集 |

生成透镜为异常、跨领域因果DAG类比、反转、实验优先四类。各视角提出的种子已经按机制聚类；不把二十视角变成二十模型分支。

## 3. 最多三个候选与淘汰/保留理由

**C0 必需最小修复对照。** 保持 S37 网络，冻结严格因果观测、固定 shape/init 与开发分割，并修复经 CPU 证实的 autocast 空包舍入问题。历史原路径原样保留，新运行器使用 fp32 状态累加。MSE vs 现成 SO3/FK 是后续单变量配方控制；既有弱绝对平移梯度反例仍有效，不作为本次默认“免费提升”。C0 是确认公平比较的工具，不主张创新。

**C1 主方案：可精确重算的因果事件证据缓存 + S37 几何路由。** 只改已证明不满足 streaming 合同的 token/准入/runtime，不替换 EdgeConv、宽度、关节 heads、MANO 或 51D 接口。先证缓存正确和预算可行，再检验相同观察量下跨包证据是否改善精度。科学状态 HOLD；工程实现可 proceed。不得称 AEGNN/周期聚合为新发明，EV009 已有近邻。

**C2 唯一条件备选：关联不确定性/可见性最小修复。** 只在 train-only、使用模型自预测历史的成对实验中，固定事件/节点/时间支持，发现错误直接关联导致可重复的轨迹损害且精确/拒绝关联能消除该损害时启动。以相同节点数/开销负控制隔离。当前混合符号的旧ray干预未达到该门，因此停车；禁止同时换成 xyz mesh 图。normal-flow新支路也停车，不能用新名称重跑未过门的georoot。

选择依据是依赖关系和反证，不是投票或分数。C1 解决最终系统必需且已经证实的缺陷；C2 的主要必要性尚无证据。C1若不满足性能门，先定位输入/路由瓶颈；不得无限改模块追分。最终若精度未过，记录失败并在预算内停止，不宣称目标完成。

## 4. C1 数学定义、状态和缓存

事件按 `int64` 微秒时间和输入顺序到达。固定 H=50000 us，W=32，K=8，保留 S37 C=128、L=3。首次 Debug 不采样：每个事件进入编码器一次；读出最多 N=2048 个最近且年龄≤H的节点。读出裁剪是一项明确的记忆容量限制，不能称所有历史信息完整保存。

7维 arrival-final token 为：

`φ_i=[u_i/W_img, v_i/H_img, 2p_i−1, min(Δt_i/H,1), log(1+Δt_i/1us), min(age_same/H,1), min(age_opp/H,1)]`。

无历史时 age=H，首个全局event Δt=0。同时间戳以接收顺序为因果tie-break；两极性年龄为0时必须与“从未观测”区分。稀疏 pixel/polarity→last timestamp 字典只保存实际访问键，不构造完整图像特征或稠密LNES。绝对时基仅用于差分；任意统一时间平移不改变 token/边。

取前 W 个已接收事件中真实年龄≤H的 K 个最近者，距离使用 `(Δu/W_img, Δv/H_img, Δt/H)`；整数时间先相减后转浮点。时间不是深度。这个最大年龄限制在实现前经性能视角反方加入：长静止后不允许一条极长 dt 边回连陈旧证据；完整重算参考使用相同规则。每层沿用

`h_i^0=ReLU(embed(φ_i))`

`h_i^l=h_i^(l−1)+mean_{j∈N_i} ReLU(W_l[h_j^(l−1)−h_i^(l−1); Δp_ji])`。

微批内新节点也必须读取同批更早节点的上一层特征。保存各层最近 W 的特征与时间/坐标，旧特征从不被重新定义。较早影响已经写入后续 h；参考算子是完整因果prefix，不是截掉prefix后重新建图。读出缓存另存最近N个末层特征和观测位置。参数更新、事件标定/坐标变换或stream身份变化必须报错或显式reset；不跨序列隐式续接。

在 query q：剔除年龄>H的读出节点；重新 MANO FK、投影和所有live节点路由；复用 S37 `pool_joint_evidence`、global proj、`_decode_active`、prev_mlp。不缓存跨pose的关联和max。默认几何仍是S37近顶点规则，C2未通过前不添加新几何模块。

状态更新先保留51D接口，`s_next = s + Δ` 在fp32完成；空新事件保持状态，不让旧记忆/prev支路凭空多次推进。连续短间隔不能未经训练直接复用50ms delta：旧权重只作初始化和数值Debug，短interval训练/状态更新合同是筛查前必需项。

最终预测shape必须是声明的固定先验或仅过去事件估计，不能读GT shape。第一版Debug固定beta=0（MANO模型的形状原点，不是本数据人群均值）、中性pose和z=0.45m；不假装已能可靠冷启动。后续若改用训练集估计的beta均值，须先冻结产物、对照共享，并重新过相关Debug，不能静默更换。初始化与恢复未过门，完整系统不得报告成功。

## 5. 伪代码与触发

```text
on_events(xyp, int64_t, state):
    validate(stream, monotonic_t, pixels, polarity, weights_version)
    tokens, sparse_SAE = arrival_final_tokens(xyp, t, state.SAE)
    graph_state = append_only_EdgeConv(tokens, predecessor_layer_cache)
    update(last_W_layer_cache, last_N_readout_cache)

on_query(now, predicted_pose, shape_prior, K, graph_state):
    if no_new_events: keep pose, produce requested mesh
    else:
        live = last_N with now-H <= timestamp < now
        mesh_prev = MANO(predicted_pose, shape_prior)
        evidence = reroute_and_pool(live, mesh_prev, K)
        delta = original_S37_heads(evidence, predicted_pose)
        pose_next = fp32_state_update(predicted_pose, delta)
    return pose_next, MANO(pose_next, shape_prior), explicit_state
```

事件接收不等待整50ms窗；局部更新试验微批timeout上限1ms。3D state和完整mesh的周期在GPU真实服务成本测完后冻结，先比较2ms/4ms两个工程负载点，不看精度选周期；如果两者均不能满足完整7ms并维持队列稳定，判该实现性能门失败。只有已到达历史可被复用，flush不得靠未来事件；停止事件流也要用真实时钟触发deadline。

上述cadence是性能判别范围，不是已满足7ms的声明。正式精度实验只使用性能可行后预先冻结的一个周期；拒绝在精度结果后挑有利周期。

## 6. 成本、反证与预注册门

增量编码每event O(W + LKC²)，working cache O(LWC)，readout O(NC)；稀疏SAE上界为实际访问像素极性键数。几何刷新仍有 O(N×778) 距离/topk与O(N×16×C)池化，MANO decode和十五heads也须计时。未假设GPU kernel launch或数据传输免费。

Debug必须：

- 完整prefix vs 任意分块/逐event：每层和读出 fp32 atol=rtol=1e-5；ID、reset和empty状态精确。统一时间平移、重复时间戳、未来改写不影响已发输出。
- 严格异常：乱序、非法坐标/极性、NaN/Inf、不相容权重/stream显式报错；不默默clamp、sort或丢包。
- 合成刚体几何、前后表面、跨指关联反例；当前S37近似未修部分仍登记风险，不用“缓存正确”覆盖它。
- backward所有预期参数有有限梯度；optimizer确实更新；CPU小样本拟合下降。该拟合仅Debug，不作精度证据。
- 旧接口回归，FP32状态保持；GPU混合精度/长期递推/真实吞吐另门。

精度假设 H1：同原始观察支持和预算，跨包prefix证据在低事件率下改善跟踪。负控制：在相同坐标/极性/事件数量下打乱合法时间间隔与顺序关系；去跨包历史但保持同节点预算。若改进在这些控制中保持，不能归因时序信息。相同新输入的full-prefix和cached模型必须同预测，缓存本身不应“提高精度”。

创新 gate：EV001/EV002/EV009 和 G01/G02/G03 覆盖关键组件。只有能证明新的非刚体观测/状态耦合机制，且反方控制支持，才能升级科学主张；工程速度改善最多先作系统贡献。当前novelty状态 `differentiated in constraints, scientific novelty unverified`。

预算门：GPU Debug、筛查、正式训练数额等待用户确认。CPU当前仅允许小fixture与基础回归；不在CPU偷偷跑正式训练。每次实质修改重跑受影响合同。若必需Debug失败，停止训练升级并记录失败。

## 7. 实现后新增证据与决策修订（不覆盖原预注册）

CPU bounded合同已过局部门；反方发现的buffer别名、非整数时间及全模型变权失效问题修复并通过回归，见DEBUG.md和两份审阅。这支持实现正确性范围，不将科学HOLD改为PASS。

原train全部72条offsets普查发现1ms峰值8321，超过单append的8192；因此原容量不能视作完整训练包络。最小修复是在外层将同一已到达微批顺序分成若干≤8192子调用，保持全部事件和query截止，计入全部成本；不增加模型容量、不丢事件。core的超限报错仍保留。此前低负载CPU回放未覆盖此情况，相关新Debug须另记。

同一普查证明末2048读出并未完整覆盖所有到达事件：2/4ms周期下，约17.2%/41.4%的事件永远不直接进入一次读出。图/SAE可能保留间接影响，因此计数既不能等同前端丢包，也不能直接推出精度损失。但C1的信息保留和高负载收益现在是明确待验项，正式开训前不能仅引用低负载拟合PASS。不得为绕过这一发现悄悄改cap、挑低事件率验收或改名为“所有事件完整融合”。

本轮仍只保留C1一个主方案及条件C2；先验证串行分块处理burst，再在已授权GPU包络内判断2/4ms可行性和受控读出影响。若原末N机制被证实造成关键精度损害，再做有证据的最小读出修正，先更新合同后实现；当前不未经验证扩充注意力、三维图或深度模块。
