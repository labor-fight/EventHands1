# S37 异步研究：事件机制、稀疏编码与性能证据

审计日期：2026-09-25（Asia/Shanghai）。责任范围：④事件生成、⑦异步 GNN、⑧EventNet/点集递归、⑨稀疏 CNN、⑩稀疏 Transformer、⑲CPU/GPU 性能。本文由一个真实子代理完成六个视角审查；六个视角不是六名独立代理。本文没有运行 GPU、训练或精度实验，所有方法收益均未验证。

## 决策框架与已读材料

研究锚点：S37 已经使用稀疏事件节点，但“稀疏节点”“因果边”“可精确增量运行”是三个不同命题。问题是能否在保留 MANO 状态和读出接口的条件下，让事件证据及时更新，并避免陈旧几何把事件误路由到背面或邻指。最终指标、数据协议与硬件验收由主审计冻结，本文不另立指标或允许 RGB 推理。

实际读取了用户指定的两个 SKILL.md、generative-lenses、evaluation-and-output、research-playbook、graph-schema、scoring-rubric，以及 Event4D 失败知识库和 DSEC-3DOD worked case/challenge/graph。此处仅迁移其证据规则：保持同一观测/预测的口径一致性、机制负控制、无效实现不算科学失败、cheap falsifier 先于训练。DSEC 双目结论不移作本任务结论。

方法采用四种生成视角：异常（稀疏但不快）、结构类比（递推充分统计量）、反转（缓存何时比重算更错）、实验优先（固定事件前缀和 MANO 后定位第一处偏差）。不是以堆叠 graph/memory/attention 作为创新依据。

检索边界：2023-01-01 至 2026-09-25 的公开工作为主，补充四个指定基础工作及事件成像基础。检索 CVF、arXiv、作者实验室/官方 GitHub、Nature、CCF 官网；实际执行 33 个 query，见 [sources_events.json](sources_events.json)。本分工覆盖事件表示与增量推理，不覆盖全部 3D/SLAM/手部文献。未访问 JCR/中科院分区授权数据库，不能给未核验的“一区 Top”标签。CVPR 主会按 [EV015 CCF 官方 A 类表](https://www.ccf.org.cn/Academic_Evaluation/AI/)标记；workshop、WACV、ECCV 不冒充 CVPR 主会 A 类。Nature 的发表事实已核验，具体分区版本未核验。不是穷尽系统综述。

## 最关键的代码事实

以下为读取代码得到的 **observed**，不是训练推断。代码快照 SHA-256 在 sources_events.json 的 local_code_snapshots 中。

1. **S37 主路径没有 BatchNorm。** `semkine/event_gnn.py:111–121` 是 Linear/ReLU，`model/model.py:908–953` 的 root/joint heads 与 prev_mlp 也是 Linear/ReLU。不能为了“BN 会破坏缓存”给这个主干添加无必要改动。历史其他模型/可选支路需要另查。
2. **节点集合不是前缀稳定的。** `EventGNN._sample`（`semkine/event_gnn.py:127–147`）以整包 counts / min(counts,max_nodes) 决定 uniform stride；超过 cap 后追加事件会改变被保留的旧事件。即使不改变 graph 函数，旧节点身份已经不同。
3. **token 可能回写旧事件。** `event_tokens`（`semkine/encoder.py:145–165`）含包内 t/Δt 和 SAE/Δt；`sae_times` 的首次同/异极性 fallback 使用本包最后—最早事件 span（约 `encoder.py:88–95,138–141`）。只固定 nominal Δt 不能消除实际 span 对早期 token 的依赖。
4. **包边界改变历史定义。** `inter_event_dt` 将每包首事件间隔设零，`sae_times` 以 packet/pixel/polarity 分组。跨包缓存这些值不是当前算子的等价运行。
5. **边按前驱序号有界，不按真实物理半径有界。** `_edges` 在前 window=32 个保留节点中取 k=8，距离包含归一化时间。事件率、采样与时间尺度改变都会改变邻居。它是有界搜索启发式；代码注释“远时间不会邻近”不构成所有负载下等同完整半径图的证明。
6. **几何关联缓存与事件特征缓存要分开。** `route_front_vertex_lbs`（`semkine/routed_readout.py:38–94`）对每个读出节点搜索投影顶点，再在近邻中按深度选前面顶点；MANO pose/K/beta 改变会使 responsibility、hard max 归属和 coverage 失效。但 S37 `PREV_RENDER=false`，事件 h 本身不依赖 MANO，不能因此全局重算 h。
7. **当前路由是可见性近似。** 选前 k 个最近投影顶点并在最近距离+3px 内取最小 z，不是三角形 z-buffer/射线首交，也没有证明能拒绝所有背面和邻指关联。该缺陷是否为精度主因须由根因探针判定。
8. **池化有不同更新代数。** 全局 sum/count 可在稳定节点集合上增删；max 插入容易、删除当前极值需保留次极值/树或重算。MANO 变动使每关节加权 mean/max/coverage 全部可能改变，不能只更新新事件贡献。

以上足以否定“直接给现有 S37 加一个 cache 即等价”的主张，但不足以证明任何新架构更准。

## 文献证据矩阵

状态：**reported** 表示作者论文报告，**observed** 表示本次查到公开正文/代码；迁移和失效判断为 **inferred/proposed**。所有外部指标不与本项目毫米误差比较。

| ID / 发表状态 | 已核验机制与定位 | 最小可迁移思想 | 单目非刚性手的反例 / 新增成本 | 最小反证实验 |
|---|---|---|---|---|
| EV001 [AEGNN](https://arxiv.org/pdf/2203.17149)，CVPR 2022；原文 §4.1–4.2 | 图局部变化只传播至受影响层；插入、pooling 和旧节点退出都须维护 | 缓存不受新事件影响的特征，维护精确依赖 | S37 的采样/token/几何并非不可变；删除和重连扩大无效集合。成本是图索引、各层 h 和更新列表 | 同一不可变事件前缀，逐次插入与完整重算逐层比较；另做删除/重连对照 |
| EV002 [EventNet](https://arxiv.org/pdf/1812.07045)，CVPR 2019；§2.4 Eq.3–5 | 特定复数时间编码与 max 允许递推；有限静态输入表能替代 MLP | 时间变化必须有可递推代数；解耦事件特征与查询时刻 | 全局 max 会丢局部指位；动态 MANO query 不能全部变为静态 LUT。新增时间状态与查表存储 | 相同事件集打乱保留短时统计的时间块；比较有/无空间局部关联，若效果不变则不支持时间机制 |
| EV003 [AsyNet](https://arxiv.org/pdf/2003.09148)，ECCV 2020；§3 Eq.2,5–9 | sparse recursive representation 和 active-site rulebook 递推；新激活/失活位置特殊处理 | 仅重算受输入差分影响的 active sites，保持同步参考算子 | 当前任务禁止退回完整稠密栅格主路径；sparse conv 邻域膨胀、索引成本可能占主导 | 固定 sparse 算子逐事件/整包同输入等价；独测新激活、消失与 max 删除 |
| EV004 [SAST](https://arxiv.org/pdf/2404.01882)，CVPR 2024；§3、Table 1/2/4 | scene-adaptive window/token co-selection 与 masked attention | 借“证据价值而非 event count”分配计算预算；先做无学习机制诊断 | 低活动的细手指可能被删；selection 会变更 key 集并使旧 attention 缓存失效；排序/装填增开销 | 同保留数 uniform/随机/importance 三组；错位 importance 负控制；需要端到端实际速度收益 |
| EV005 [RVT](https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html)，CVPR 2023；abstract | 卷积位置先验、局部/全局 attention、时间递推 | 可用作“稀疏是否真的更好”的 dense 思想对照 | 主方法是 dense 参考，完整迁移违反当前最小修改及稀疏要求；未知低事件手指泛化 | 公平输入/状态/输出频率固定后比较，不能拿其他任务 forward 数字当延迟结论 |
| EV006 [HMNet](https://arxiv.org/pdf/2305.17852)，CVPR 2023；§3 Fig.2 | 多速率层级记忆；event sparse cross-attention 写入 dense memory | 将事件接收、特征、状态、完整输出频率分开设计 | dense memory 不符合最终原始稀疏主路径；快指局部信号被慢上下文抹除。多级状态难失效 | 固定观察前缀，快速反向运动在刷新前后比较；检查错误旧记忆是否持续传播 |
| EV007 [SSM events](https://openaccess.thecvf.com/content/CVPR2024/html/Zubic_State_Space_Models_for_Event_Cameras_CVPR_2024_paper.html)，CVPR 2024；abstract、[官方代码 SSM-ViT](https://github.com/uzh-rpg/ssms_event_cameras) | 有物理时间尺度的状态空间更新用于频率变化 | 明确 Δt，避免包宽度变化被误当动作变化 | 只换 decay 不能产生缺失空间/深度信息；官方骨干仍是层级 ViT。增加状态与频率标定 | 保持同事件时间戳，仅改变 chunk 划分；若输出仍强依赖切包则不支持 rate robustness |
| EV008 [DAGr](https://www.nature.com/articles/s41586-024-07409-w)，Nature 2024；Methods | 深异步图与图像支路，Methods 明确 detection adding 在无事件时提供 RGB 下界 | 借有向图和更新抑制思想，严格去掉 RGB 依赖 | 静止/遮挡的优势不能移用到单目 event-only；时间图中的第三维不是物理深度。需 CUDA graph indexing | event-only 低事件/停止场景，禁 RGB 和 GT history，独测正确恢复与先验锁死 |
| EV009 [HUGNet2+PA](https://openaccess.thecvf.com/content/CVPR2025/papers/Dampfhoffer_Graph_Neural_Network_Combining_Event_Stream_and_Periodic_Aggregation_for_CVPR_2025_paper.pdf)，CVPR 2025；§1–2 | 无累积异步分支 + 过去数据周期上下文；原文称微秒能力针对异步硬件 | 保留快局部响应，以过去全局上下文补充 | 已占据“快局部+慢全局”结构新意；光流不等于非刚性 3D 校正；上下文陈旧、分支额外成本 | 时间反向/突然加速，破坏快速分支时间仍保留统计；若主改善不变，时间主张失败 |
| EV010 [ACGR](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Asynchronous_Collaborative_Graph_Representation_for_Frames_and_Events_CVPR_2025_paper.pdf)，CVPR 2025；§4.1，supp A.2 | frame-event graph；明确八类主表与两类另表不同 | 借 protocol 警示与训练监督/推理输入分离 | 完整输入含 RGB，禁止直接做最终路径；跨表换类别能虚增比较 | 所有 comparator 同输入模态和同评测集合；禁以 RGB 模型效果证明 event-only |
| EV011 [SpTopoNet](https://openaccess.thecvf.com/content/CVPR2026/html/He_Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection_CVPR_2026_paper.html)，CVPR 2026；官方 abstract | 稀疏卷积和事件轨迹拓扑约束；只核到官方摘要，正文直链 403 | 仅作为反查线索，不实施未读机制 | 事件轨迹连接不等同手 mesh 拓扑；相交/遮挡会产生错误连通；成本待正文审计 | 先取得原文；交叉手指、遮挡合并场景对照错误连通，未通过前不加 loss |
| EV012 [eGSMV](https://openaccess.thecvf.com/content/WACV2026/html/Verma_Event-based_Graph_Representation_with_Spatial_and_Motion_Vectors_for_Asynchronous_WACV_2026_paper.html)，WACV 2026；abstract | 分离空间与 motion-vector 图 | 作为“空间/时间双图”新颖性反例 | 空间近邻仍能跨手指，motion vector 的可靠性未验证；双图增索引和内存 | 固定 graph 总边数比较，运动向量打乱控制；不能按模块名字宣称 3D |
| EV013 [Event-based Vision survey](https://arxiv.org/abs/1904.08405)，TPAMI 2022；摘要与作者页面 | 事件记录像素亮度变化 | 将 event likelihood 与 MANO 几何先验分开 | 纹理、光照变化、传感器噪声均可触发，mesh 边不一定亮度边；无事件不等于静止且位置正确 | 亮度扰动与几何运动分离、零运动带噪/低纹理运动两种对照 |

公开代码核验：AEGNN、AsyNet、SAST、DAGr、SSM 均核到作者仓库；此次只对 AEGNN BN 文件作源码审计，其余仓库不声称完成复现。EventNet 搜到的 zhangchushu/EventNet 明确为非官方，未充当官方代码。论文中的性能为作者报告，本文未独立重现。

## 三个必须保留的反证

**AEGNN 并非可直接套用的等价证明。** [EV014 官方 BN 实现](https://github.com/uzh-rpg/aegnn/blob/d96e13b2f80f3c7515a65baf966544e0914d068d/aegnn/asyncronous/batch_norm.py) 的 `__graph_processing` 注释明确承认追加节点改变分布，以初始分布近似；本次保存 raw 文件 SHA-256 与源码副本在统一 scratch。严格等价必须对实际算子证明；`eval()` 下固定统计 BN 才是逐节点 affine，训练 batch 统计不具这个性质。S37 当前无 BN，故这是借用代码的风险，不是 S37 现存 BN bug。

**AEGNN 文献有计算量修订。** arXiv v3（2022-11-01）封面明确修订 §5 的 computation accounting，旧 CVF 文案与更新版摘要不是同一口径；本文不引用旧“200×”宣传作为速度承诺。

**SAST 的少 FLOPs 不保证更快。** Table 2 的 SAST runtime 高于其 RVT baseline，Table 1 的 A-FLOPs 排除了卷积。以 attention FLOPs 优势推导完整 mesh ≤7ms 不成立。只讨论这张表的方向性反例，不将其硬件数字与本服务器合并。

## 六个专家视角的交叉质疑

| 视角 | 证据 | 最强反例 | 最小验证实验 / 什么结果会改判断 |
|---|---|---|---|
| ④事件生成 | EV013、EV003 §3.1；本地 token 和历史路由 | 一个事件可能来自纹理/光照/噪声；历史表面不是当前测量。t 是时间而非 z | 固定事件计数、空间、极性边际，破坏时间顺序；额外做 GT 几何仅用于诊断的前/后表面干扰定位，系统结论使用预测历史 |
| ⑦异步 GNN | EV001/EV008/EV014；本地采样和时间依赖 | 单个追加事件能改所有旧 token/选点，理论局部图并不等于本算子局部 | 同一 prefix 严格比较 tokens、node IDs、edges、每层 h、readout、最终 mesh；首处偏差确定责任，不能只看最终接近 |
| ⑧EventNet/点递归 | EV002 §2.4；本地早期 recurrence 已存在于 encoder.py | 精确递推的全局统计可以很快却不含相邻手指区分信息；复杂 query 破坏 LUT | 同模型预算、同事件输入，用几何关联和 joint-ID 打乱控制判断局部性必要性；不要重跑仓库已否决 recurrence 直到有新机制证据 |
| ⑨稀疏 CNN | EV003 §3.2；本地 EventGNN 已有 3 层 | 低 N 不代表 low latency，active sites 膨胀与 rulebook 重建可吃掉收益；重新稠密化违约 | 只在真实节点负载与可复用邻域确定后做 operator microbenchmark；完整传输/预处理/mesh 计时与该 microbenchmark 分开 |
| ⑩稀疏 Transformer | EV004 Table 2；EV005–007 | pruning 忽略无事件但状态不确定的手指；改变 token set 使已有 attention 更新 | 相同保留事件数下对比 uniform 与 importance；注意力只作为触发条件备选，若收益只来自丢事件则重新命名主张 |
| ⑲CPU/GPU 工程 | 本地 argsort/searchsorted、topk、route 的 nonzero/host bool；EV001 timing、EV004 Table 2 | 单事件 GPU kernel 启动与同步成本可远大于 MAC；突发事件队列失稳，即使平均 forward 很低 | 无竞争目标卡 causal replay；按 event load、query trigger、冷启动/refresh 分层，记录等待/队列/H2D/preprocess/graph/readout/MANO/output 的 P50/P95/P99/max/超限率；不可把 max 当无限负载保证 |

## 缓存参考算子的数学边界

设不可变已接收节点为 e_i，a_i 为在接收时确定且不依赖未来包终点的特征，前驱集 `N(i) ⊂ {i-W,…,i-1}`。只在这些条件成立时，

\[
h_i^0=\phi(a_i),\quad h_i^{\ell+1}=h_i^\ell+\frac1{|N(i)|}\sum_{j\in N(i)}\psi_\ell(h_j^\ell-h_i^\ell,\,r_{ij})
\]

在收到 e_i 时只算新节点即可和**整个因果前缀参考**一致。每层保存前 W 个节点的状态即可供后续读取；更早影响已留在 h 中。这不等价于“把滑动窗口外节点删除后从零重算窗口”，后者边界节点及其下游会变，必须传播删除失效。

在查询 t_q 只对活跃集合 A_q 读出：

\[
V_q^- = \operatorname{MANO}(s_q^-),\quad a_{ij}^{(q)} = \mathcal R_j(e_i,V_q^-,K),\quad E_j^q=\operatorname{Pool}_{i\in A_q}(a_{ij}^{(q)},h_i),\quad s_q^+=s_q^-+D(E^q,\operatorname{Pool}(h),s_q^-).
\]

V 是 mesh 顶点，j 是运动学关节；LBS 描述先验影响，不证明某个观测由该关节独立产生。上述 a 为关联责任而非独立深度测量。MANO 更新仅使 route/pool 失效，h 不失效；若未来又把 prev render 加入 token，则这个性质消失。

潜在执行合同（**proposed，未实现、未测**）：

```
on_events(causal_microbatch):
    check timestamps and sequence id; reject disorder or explicit reset
    admit by frozen causal rule; record every dropped event and reason
    for new admitted nodes in time order:
        build immutable token from causal state and fixed physical time scale
        select at most W previous admitted nodes, compute k edges
        compute each graph layer using predecessor cache; commit cache
    add h to active-readout storage; expire readout entries by declared rule

on_query(deadline):
    read only events <= query time; snapshot latest predicted MANO state
    recompute route and joint pooling for every active readout node
    run unchanged output heads; decode complete MANO; timestamp output
    save state for next query; do not use ground-truth history

on_reset(sequence/calibration/weights/token-contract change):
    invalidate token/graph/all layer/readout/state caches as applicable
```

事件接收可连续进行，CPU/GPU 传输按固定上限 microbatch；局部 h 更新由新事件触发；三维状态和完整 mesh 输出必须是明确且受预算约束的 query。只更新局部 h 不能宣称每事件都有完整 mesh 输出。若保留未来跨度为50ms的**非重叠收包再输出**，其等待纳入延迟；若维护过去50ms滑窗并高频读出，50ms是历史支持长度，不自动等于50ms新增等待。两者必须在 runtime 中区分。

成本推导：EventGNN graph 搜索 O(NW)，每层 message O(NkC²)，图激活 O(LNC)，route 朴素 O(|A_q|×778)，pool O(|A_q|×16×C)。append 更新把图计算 N 换成新增 m，但不能据此把每次 route 的 |A_q| 换成 m；完整 mesh 也不能省略。3 层局部缓存只解决重复 encoder 计算，不保证最终端到端预算。

## 三个候选及淘汰边界

这只是本分工提交主代理的候选审查；主方案须结合失败记录与根因实验才定。主指标仍是同 checkpoint、同协议的两个阈值及完整延迟，不以诊断指标替代。

| 候选 | 假设 / 最小变化 | 最近邻 / 新颖性状态 | 最小测试及负控制 | 暂定门 |
|---|---|---|---|---|
| C0 最小修复对照 | S37 最大损失来自可验证的观测关联/实现缺陷；保持训练、encoder、heads，先定位再仅修那一项 | S37 本身；不把 bugfix 当研究创新 | frozen predictions/evaluator parity；同 packet 和 prev 下逐环节定位；对任何可见性修复做邻指/背面分离负控制 | HOLD：等待主审计根因；必要控制，优先便宜诊断 |
| C1 不可变因果证据 + 每次完整几何读出 | 固定物理时间尺度和 causal admission 可消除旧 h 回写；保留 S37 heads/MANO，用 query 重新关联 | AEGNN、EventNet、DAGr、HUGNet2+PA 已覆盖大部分；只可能在非刚性几何与缓存正确性/观测机制形成区别 | prefix/full parity；切包不变性；MANO 改变后 route 正确失效；时间块破坏和相同采样数控制 | HOLD：需证实改 token/采样不会先损失 S37 能力；不能宣称当前 checkpoint 数值等价 |
| C2 非 GNN 因果点递归替代 | 若局部 graph 索引/launch 经测量主导成本，而局部关系负控制显示关系不必要，使用轻递推统计替换 encoder | EventNet / SSM / 既有本地 recurrence；likely duplicative，且须审查历史否决 | 同输出频率和总输入量，递推 vs graph 容量对照；空间打乱负控制 | PARK：只有 C1 图工程超预算且图关系机制无效时触发，禁止同时堆入主方案 |

按 research-opportunity rubric，C0/C1/C2 的暂定 (importance,gap,feasibility,novelty,information gain,leverage) 为 (5,2,4.5,1,5,4)/(5,2.5,3,2,4.5,4)/(4,1,3,1,3,3)。按技能权重计算为 C0=73.0、C1=70.5、C2=51.0（0–100）。评分仅排最小研究价值，均不代表性能；C0 强在便宜排除误因，C1 强在可复验算子，C2 新意低且有已失败路线风险。所有门均 HOLD/PARK，不能凭算术评分开训；gap/feasibility 各 ±1 的敏感性足以改变 C0/C1排序，故先做能共同否定两者的协议/缓存探针。

逐项评分理由（依据分别为本地代码事实、EV001/002/008/009 与本文六视角）：

- C0：importance=5，因为错误评测/关联会使全部后续结果失去可解释性；gap=2，因为已知工程疑点尚非明确科学空白；feasibility=4.5，因为固定输入分析无需正式训练；novelty=1，因为修复不等于新机制；information gain=5，因为首个错误环节可排除整类候选；leverage=4，因为冻结协议和探针可复用。
- C1：importance=5，因为真正因果低延迟是用户硬目标；gap=2.5，因为非刚性读出边界尚需验证而通用缓存已有工作；feasibility=3，因为 token/参考算子需修改且 GPU 预算待定；novelty=2，因为 AEGNN/HUGNet2+PA 近邻很强；information gain=4.5，因为 chunk parity 与负控制能直接否定机制；leverage=4，因为显式状态/失效规则可支撑多种未来读出。
- C2：importance=4，因为 graph 瓶颈若属实需非 GNN 备选；gap=1，因为 EventNet/SSM 及本地已有路线占据主要机制；feasibility=3，因为可复用代码但历史失败需先澄清；novelty=1，因为只是 encoder 替换；information gain=3，因为能测 graph 关系必要性但无法单独解释3D关联；leverage=3，因为可复用作低成本对照。

条件式收敛建议：先 C0 的固定输入证据审计；若异步缓存缺陷经测量成立且协议允许，C1 可作唯一主路径工程候选；C2 只作有触发条件的备选。若背面干扰并未定位到路由，就不先加三维图/可见性模块。不会因论文“更近”就替换有效 S37。

## 证据图（文本邻接，避免把图漂亮误当证据充分）

- N1 `S37 token/采样非前缀稳定` (observed; local encoder/EventGNN) → limited-by → N2 `直接套用 AEGNN 的精确缓存` (proposed; EV001)。
- N3 `S37 h 与 MANO 无关` (observed; config PREV_RENDER=false) → supports → N4 `h 缓存与 route 刷新分离` (inferred; EV001)。
- N5 `前顶点路由不是完整可见面证明` (observed; routed_readout) → motivates → N6 `固定观测根因分解` (proposed; C0)。
- N7 `快局部/慢全局已有文献` (reported; EV006/EV009) → overlaps → N8 `异步 mesh 方案创新主张` (proposed; C1)。
- N9 `稀疏 FLOPs 不蕴含实际 latency 降` (observed table relationship; EV004) → motivates → N10 `完整 causal replay 性能验收` (proposed; perspective 19)。
- N11 `RGB 静止补偿有效但违背输入合同` (reported; EV008/EV010) → limited-by → N12 `纯事件初始化/静止恢复` (proposed; remains open in this corpus)。

## 下一步与严格 Debug 责任边界

1. 主代理已有 frozen-input prefix 审计时复用其原始产物，不另造重复脚本。此阶段只需证实节点身份/token/edge 第一个不一致点。
2. 一旦定义新的参考算子，先在 CPU 合成前缀固定 tolerance：建议 float64 单步 atol 1e-10/rtol 1e-8；float32 GPU 按实际算子及 reduction 顺序预登记（例如 atol 1e-5/rtol 1e-4），不能在失败后放宽。比较每层、route、输出与长轨迹，不只平均误差。
3. 空事件、每像素同时间不同极性、超 cap、长静止、突发、重置、变 K/beta/pose、window expiration 全部覆盖。追加与删除是不同测试。实时新 token 不得读取 final packet span 或未来 count。
4. 正式训练前还需 optimizer/gradient、finite、小样本拟合和未变路径回归；这些不在本子代理执行范围内，也不能被本报告代替。
5. 正式性能必须在无竞争目标卡测；不能用论文 GPU、THOP MAC、单次 forward、异步硬件潜力或 P95 代替完整≤7ms验收。等待 budget/hardware/output-frequency 冻结后才能设计正式运行负载。

未完成的文献缺口：EV009 直链403且未定位作者代码；EV011 仅官方摘要且原文未读；EventNet 官方代码未找到；2026 未被搜索引擎索引的公开工作及 NeurIPS 2026 在执行日前实际公开状态未穷尽；本分工不完成手部任务最近邻或全部期刊分区核验。代码和研究结论的证据成熟度保持分开。

## 对主代理 C1 最小实现提案的补充反方审查

提案为固定 H=50ms 的 tokens7、复用 S37 所有学习权重、前32事件图、每层缓存、只读 last2048 且 age≤H、每 query 重做全读出集路由。该方案的工程必要性已可定位；其新颖性和精度机制仍为 HOLD。下述约束应在实现前冻结：

- “无输入采样”仍包含 readout cap 截断。在密集流中只直接读最后2048个节点，其历史时长小于50ms。记录每 query 的原始/编码/读出事件数、有效历史时长，最终以公平消融量化影响。不能称读取了完整窗口。
- 前32邻居没有最大物理 age 时，长静止后第一事件可连接很久之前事件，edge dt/H 可任意大。仅 clip token 不能限制 edge。应指定最大 edge age≤H 或在长 gap 时明确刷新图，并将规则同时施于完整前缀参考。若保留无限 age，必须把长 gap 作为可能杀死路线的测试。
- 每像素时间字典需要显式 unseen 哨兵；合法相同 timestamp 不能被误判成从未发生，也不能读取同时间的未来次序事件。
- prefix h 的跨包支持和旧 S37 不同；必须重新训练才能判断精度。缓存 detach 后的训练是截断 BPTT，不是完整前缀反传等价；短程 Debug 应检查定义的梯度合同而非悄悄接受 detach。
- 合法负载内不能只靠异常拒绝来满足延迟；超过 maxinputbudget 可以 fail-loud，但该范围必须在验收前固定并作为系统能力边界。
