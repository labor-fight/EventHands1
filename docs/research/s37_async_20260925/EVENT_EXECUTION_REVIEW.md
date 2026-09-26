# 事件前端固定容量执行审查（设计，未实现）

2026-09-25；只读审查者：literature_events。本文件不代表新core、GPU实验、精度结果或7ms通过。唯一推荐是**条件性执行原型**：固定512事件块、仅保存已访问键的固定容量稀疏SAE、固定W32逐层缓存，捕获append的静态工作区；query继续使用原读出压缩、归约和路由。先做无SAE改动的padding/邻居/逐层判别，失败就停止，不先写整套哈希执行器。

## 事实边界与改变范围

`semkine/streaming.py:181–220`是当前稀疏SAE：合并历史键与当前事件、按key+接收序号排序、分别查同/异极性前驱，最后布尔压缩最新值；`226–278`是W32/K8构图、fixed_fp32 affine和逐层/读出缓存；`291–307`是当前动态有效节点压缩与原mean/max/proj。

G006已准备decode graph，仍不能持续服务：`g006_c1_captured_4000.json`中首append之后34次append合计178.534378ms，超过100ms逻辑回放；全部25输出超7ms。这不是解码子段进一步预热能解决的问题。G007按root刚完成的判别，默认TorchScript token在n=1/old37的第2例已有2.3283e-10差异，未过其bitwise门、未安装；不能因绝对差很小而改门。

保持的合同：原7维arrival-final token及H=50000µs、事件接收序、W=32/K=8、原距离与topk、C128/L3固定顺序linear、所有真实事件编码、最后N=2048事件的时间窗口读出、原投影/heads/MANO、1ms接收与2/4ms输出deadline。B只是计算buffer容量：m=1也按既定1ms截止立即处理，不等到凑满512；末块padding不是采样，不生成新观测，不能推进时钟、计数或历史状态。

本设计不改变学习参数，也不把时间当深度；MANO改变仍只影响全部live节点的读出路由。它不声称稀疏哈希或CUDA graph是科学新意。

## 只冻结一个块大小：B=512

先用已知原始counts比较两个执行容量，不看精度、不跑GPU调参：

- G006原始输入482事件、35个非空1ms bin、最大bin72。B128和B512都需35次块执行；B128物理处理4480行（9.2946058091倍），B512处理17920行（37.1784232365倍）。B512这里没有减少dispatch，消息计算约为B128的4倍，块内接收序匹配矩阵为16倍。**这是本选择最强反例，不能隐藏。**
- 完整train offsets记录最大8321事件/bin；沿用原8192+129两个合法append，B128是64+2=66次执行，B512是16+1=17次执行。填充行数分别8448/8704，约1.0153/1.0460倍真实事件。必须测17次连续replay、全部输入copy、状态更新及两个append原子边界的累计成本，不能拿一次512行耗时代替burst。
- 全train的1ms counts P50/P95/P99=92/2109/3084。对这些计数，B128分别1/17/25次块执行，B512为1/5/7次。8321是已观察最大值而非任意传感器负载上限。原manifest未给出全部ceil(count/B)的总和，不从分位数捏造全train均值。

选择512的唯一理由是：已证实主机派发过多，较大块减少中高负载的重复graph提交，特别是66→17的真实burst分解。代价是低事件率多算很多，实际收益未知。因此只冻结512进入最早bitwise门，不同时实现128分支，也不因后续失败偷偷换块大小。若低负载或burst累计服务不改善，该设计NO-GO；重选执行容量须另写有新证据的决策，而非把它称已经验证的备选。

## 固定形状如何表示真正稀疏的SAE

推荐工作区是**一维显式key/value哈希字典**，不是以像素坐标直接索引的完整timestamp图，更不是dense learned feature map：

- `keys[S] : int64`，EMPTY=-1；`last_t[S] : int64`；`P_active`为设备上的已占槽计数。只有实际收到的键`k=((y*240+x)*2+p)`可以占槽，空槽的timestamp没有观测意义。使用带碰撞处理的哈希，不将key变换为无碰撞的完整pixel-address表冒充稀疏。
- 暂定容量`S=2^18=262144`，合法键域最多`2*240*180=86400`，所以实际占用率上界约0.3296。容量大于P_active；**分配内存是O(S)，不是O(P_active)**，不能用“稀疏”掩盖预留空间。key/value都是整数元数据，无图像栅格卷积/稠密token/每像素神经特征。
- hash函数、探测顺序和初值必须冻结。每次lookup/insert可在设备kernel内做有界探测，不需要把P_active或unique个数取回Python。不同分块可能导致不同物理槽布局，等价检查应比较canonical的key→last_t映射，而不是要求hash slot编号等于原排序数组位置。
- 对合法域S大于最多86400个键；采用完整最多S次探测能保证空槽仍存在，但最坏探测成本很差。不能擅自只探测32次后丢键，不能伪称hash操作最坏O(1)。记录实际最大/分位probe长度及冲突键；有限probe仍找不到位置是显式失败，整个append不提交。若合法输入碰撞导致持续服务失败，应否决容量/执行方案，不能事后排除该输入或无限扩表重试。
- 不做SAE时间淘汰。当前参考保留所有已访问键到reset，且`_validate_input`的最旧时间溢出检查也依赖它们。虽然超过H的age会被clip到H，将其改成missing可能在普通token上相同，也不能未经新证据修改字典/错误合同。

尚未实测：GPU哈希insert在当前软件栈中的capture支持、原子写入吞吐、任何probe分布或等价性。静态lookup只是设计，不能继承G001的“已验证”标签。

## 两阶段稀疏查找，杜绝包内未来泄漏

对一个真实块`m≤512`，device标量m只用于mask，真实序号`i=0..m-1`保持原接收顺序；补齐行`i≥m`永远无效。执行顺序必须是：

1. **只读旧字典快照**：每个有效i查询old(k_i)和old(k_i xor 1)，得到存在位与旧timestamp。此阶段没有SAE写入。
2. **包内前驱**：在固定B×B序号关系上，分别找`j<i AND j<m AND k_j=k_i`、`j<i AND k_j=(k_i xor 1)`中最大的j。若存在则选t_j，否则选旧字典timestamp；两者都缺失才使用missing=H。该匹配只比较整数key和接收ordinal，不按t重新排序、不把同timestamp视为无历史。
3. 原样计算`clip((t_i-t_prev)/H)`及全局相邻事件dt/log1p等token，保留原float32运算顺序。missing时先选择t_i再做差，避免对无意义sentinel做减法溢出。首个全局事件dt=0，后续第一个块事件使用真正的上一事件时间。
4. **写新字典**：每个key只由当前块最后一次出现的有效ordinal代表更新，value=t_last。代表mask也用固定B×B序号关系形成，无动态unique压缩；同key只有一名writer。新键用CAS占槽，既有键更新timestamp；所有old lookup早已结束，因此不会读到本块未来事件。

不能将步骤1/4合成“每线程先atomicMax(timestamp)再读”：那会让早事件看到晚事件；同timestamp还会把“对侧首次出现”和“已见对侧但age=0”混淆。代表唯一后，不需要让多个同key线程争抢最后写入值；不同key碰撞只影响物理存放位置，不得影响token。

B×B ordinal匹配是简单可审计的起点，B512的262144位置比较代价必须计入；不是声称最优GPU算法。该设计不再叠加第二个排序/扫描候选。

## W32/K8、缓存与topk的严格边界

固定recent buffer每层保存32个真实历史节点及有效长度，读出buffer保存2048个真实节点及有效长度；真实global event ordinal是主要身份。设备buffer可用环形地址，但候选必须恢复原逻辑顺序：i-1,…,i-32；不足32时负global ID无效，长gap仍按原`age≤H`判断，不能把过期节点移掉后重新定义前32个邻居。

每层计算均读取旧recent层快照和本块该层的真实新节点；直到所有新h完成才更新缓存。环绕写入必须保证每slot只有最终那个真实event writer：m>32时不能用无序scatter让多事件写同一recent slot。readout的覆盖也只因真实event计数推进，padding不得挤掉节点。长期影响继续编码在h中，不能将它换成“最后50ms全量重算”的模型。

**topk ties是首个必须证伪的危险点。** 原代码`distances.topk(8, largest=False)`没有声明lexicographic tie rule；相同像素/时间、距离对称或FP32舍入都可能产生同距，而同距邻居的h不一定相同。固定B可能触发不同kernel组织。必须逐项检查真实节点的有效邻居global ID及K维顺序，不能只比距离/无序集合，也不能加epsilon或换stable sort修饰成“确定性”后宣称原算子等价。

仍调用原torch.topk、保持候选宽度32；fixed_fp32 linear保持当前16×32×32 tile和K循环。padding值先设为安全有限值并mask，避免未使用的sentinel溢出/NaN污染；fake节点即使经过bias产生非零h，也不能进入SAE、候选、缓存、任何计数或读出。

**query不纳入本轮append graph。** 先按真实global ID把末2048历史还原为原顺序，再按`q-H≤t<q`做原动态压缩，调用原mean/max/proj与路由。用2048个补零节点的sum/有效数替代原n_live长度的mean，会改变FP32归约路径；不能默认等价。这样仍有query动态shape/同步成本，G006的非空query本身约7.218ms，故append成功也不自动解决整个7ms门。

## 原子提交、显式所有权与capture失效

固定mutable工作区不能直接当作返回state。否则下一次replay会覆盖用户持有的旧状态，也无法满足D005“第二子调用失败不提交第一部分”的合同。

推荐单一规则：每次合法append（真实M≤原8192上限）先把显式输入state复制到context工作区，在同一创建线程/stream顺序执行`ceil(M/512)`个图；全部输入检查、token/层输出有限性、容量及kernel状态检查通过后，**clone-out**生成独立新state才返回。失败丢弃工作区，原state不变。整个1ms微批若为8321，则沿用已登记8192+129两次合法append；外层ModelBackend仍在全部成功后才提交self.state。此规则不偷偷把8192改大，也不允许中途输出mesh。

固定工作区不做隐式训练history：这是no_grad推理执行计划。模型参数/buffer版本、dtype/device、所有static配置、线程/stream变动都要拒绝并显式重建；reset重新清空实际visited keys、所有有效计数与stream ID。形状固定并不代表checkpoint更新后旧graph仍有效。

原有限性/单位/像素/极性/单调时间/水印/整数溢出检查不得删除。可以在固定graph中累计error bits，最后一次同步按冻结优先级显式报错，但错误状态不能发布。极端int64差分必须先用无溢出的比较判非法，再对运算输入作安全mask；不能先让减法overflow再解释成合法age。任何训练/梯度路径仍需独立设计与Debug，不能把本执行计划自动接入optimizer。

## 内存、算量与反超条件

以C128/L3/W32/N2048、int64 key/time、FP32 h估算（推导，不是实测）：

- SAE 262144×16 bytes =4MiB；recent四层h=64KiB，readout h=1MiB，recent/readout xyp+t约40.625KiB；每份完整state约5350016 bytes，即5.10217MiB。工作区、旧state和成功返回clone同时存在时，约15.30652MiB，尚未含临时消息、allocator/capture pool和保留的更多旧状态。
- 每append copy-in+clone-out约10.20435MiB；1000次非空append/s约10GiB/s复制流量。这是固定容量的实价，不是原O(P_active)紧凑存储。长期保留旧state会线性增加内存，不能只报工作区常驻大小。
- 当前embedding+三层消息的MAC/event近似`7*128 + 3*8*131*128 = 403328`，不含邻居/聚合/SAE。B512约0.206504G MAC/block；G006的小片段仅真实0.1944G，但padding后约7.2276G。只有实际主机派发收益大于这些补齐/内存/字典成本才值得继续。
- 真实8321 burst为17个B512图、8704个物理行，约3.5106G MAC消息+embedding，另加17次字典/ordinal/缓存更新及两个append提交成本。至少有17次host graph提交及输入更新；不预先假设一次graph=一个便宜kernel。

反超最可能出现在：每bin只有1–2事件但表P_active已接近域上限；高冲突hash键；CPU线程调度抖动；高burst的17次累计提交；clone-out造成allocator/内存压力。P_active增长不改变神经token数量，但会增加hash冲突和状态容量成本。空bin应跳过append而保持完整timer输出，不得为了性能停止mesh输出。

## 分阶段最小判别与停止条件

这些是待root预注册拆分的门；本审查者没有执行GPU或新增core。

**E0：先只测padding、原topk和fixed affine。** 不实现hash；token仍来自当前原SAE参考。用B512和真实m=1/7/31/32/33/127/128/129/511/512，history有效数0/1/31/32，覆盖同timestamp、同像素重复、对称同距、全无效邻居及±1µs的H边界。比较有效候选mask/global ID/有序K、每层真实h和缓存；先小合成fixture，再原训练8321 burst的17段与独立完整prefix。当前执行替换判别要求bitwise，不借较早通用1e-5门为它降级；任一token/ID/真实层h差异就停止，不构建hash、不跑精度。

**E1：只做整数稀疏状态机。** E0通过后才在scratch实现两阶段key/time机制；对照原排序字典的canonical映射与完整7维token。覆盖跨块同key/异极性、同timestamp先后、超过H、最大/最小合法int64时基及失败输入；P_active从空到稠密访问的86400键，并构造hash碰撞键。要求键/存在位/ordinal完全一致、token bitwise；禁止将“给定静态lookup”当已通过此门。若原dtype顺序相同仍不一致，定位并停止，不能悄悄启用新的token浮点实现。

**E2：捕获最小append并量化累计成本。** 仅E0/E1通过才构建固定工作区，先验证A→B→A保留旧state、后续块失败不提交、reset、版本/stream拒绝及环绕；再按已冻结G006全35个非空bin和真实8321 bin检查整个调用时间，包含copy、padding、所有17次提交、字典更新、检查和clone-out。预留capture/编译/首次使用开销，冷与已准备执行状态分开报告。

E2的必要吞吐门：在与原基线相同低负载counts上，除冷成本外的全部append服务不能仍超过输入支持时长；真实burst必须按其所在完整输出周期检验排队，不以单块均值验收。通过只是继续完整cold/恢复/持续负载回放的必要条件。最终所有原始事件和同checkpoint/协议的精度门、完整7ms最大值/超限率仍独立，不能因子段变快开训。

E0任一bitwise失败、E1任何因果或容量失败、E2旧state被覆写/失败提交/吞吐仍结构失稳，均NO-GO本冻结设计。不连续堆新模块、不转dense pixel map、不减节点/丢事件、不使用未来事件或GT历史；下一选择只能基于新增失败证据重新登记。
