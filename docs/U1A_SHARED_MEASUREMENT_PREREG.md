# U1a：共享节点头是否损伤测量

2026-10-02，正式 U1a 训练之前固定。当前项目基线仍为 S37；本实验的表征与测量参照是 S38 v2。

## 问题与可归因的比较

固定 S38 sparse_pyramid encoder **权重及 BN 统计**、输入支持、旋转约定、滤波增益、数据和损失，只比较一个共享节点 MLP 与 17 个不共享 MLP。暂不加入状态消息、图消息、学习增益或新的 encoder。此阶段是冻结已训练表征的 head probe，不能以失败否定联合训练。

17 槽为 translation、root、15 个 MANO finger residual。两臂输入打包、字段 mask、type/joint embedding、逐节点 LayerNorm、隐藏宽度 64、三维输出完全一致。translation 读取原 feat/evidence/prev51 支持；root/finger 只读 state-independent pooled，屏蔽其他字段。S38 的 root 为 Exp(raw) R_ref；finger 为 MANO residual45，hands_mean 只在 decode 时加一次；root/finger 保留增益 0.5 的因果滤波，空包保持 prev51。

为了使用统一的 3D 节点接口，translation 将原 root_head 与 prev_mlp 的组合重新参数化为一个槽；两臂同时执行。因此相对旧 S38 的差异包含平移重参数化，只有 shared 对 untied 的配对差异能归因于共享。没有状态消息；translation 的原 prev 输入及固定滤波是保留的 tracking 接口。

同 seed 从该 seed 的 S38 6000-step last 加载 encoder。untied 的全部 17 个 MLP、embedding 和 norm 初值逐 tensor 复制 shared，零输出初始化；非零输出时的函数/Jacobian 等价由测试验证。独立 DataLoader generator 固定相同 seed；各 rank 初始两批完整 packet 字段 hash 必须相同。冻结状态、真实损失能下降、混合空包和 DDP 均须先 debug。

宽度一致不代表参数量一致：shared 约 0.349M 可训练参数，untied 约 5.923M；总参数分别约 1.896M、7.471M。若只有 shared 失败，必须先检查梯度、量纲和容量，不能直接断言共享不可行。

## 固定预算与数据协议

唯一只读划分 `/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json`，SHA256 `2a4769f33ab86a02f6124ddf0096f445adaa4702e7798b2f73c01b1fb474b405`。训练 9 人 72 序列；zgz_global/local 同时为开发/测试，未另设受试者测试。seed 3407、3408，所有超参数在结果前固定。

每臂 2 卡 × batch512 × accumulation1，有效 batch1024；Adam 4e-3，warmup500，bf16，cosine 到 2%。2k 是筛查，6k 是最终预算；分别从相同 prepared init 重启，不能把 2k 压缩 cosine checkpoint 接着训练成 6k。这里 2k/6k 是 **额外 head-only** 优化步，encoder 已先训练 6k；不宣称与 S38 从头训练总预算相同。两卡浮点归约与 S38 单卡累积不同，但 U1a 两臂内部配方相同。

四个 paired run 分占 GPU [0,1]/[2,3]/[4,5]/[6,7]，各 18 个物理核及其 SMT siblings，14 loader workers/rank。只使用空闲 GPU，不改其他用户任务。debug 使用 batch16 的 20 步两卡短测。

## 评估与判定门

只评 explicit last，不选点。固定 `evalx.py eval --ckpt last --split val_core --controls --tf --perturb`，GPU batch1、50ms、原初始化与扰动协议。2k 仅定位，不作为最终采用决定。6k 以同 seed paired shared-minus-untied 为主比较；同时对该 seed S38 6k last 检查测量与 tracking 守护。

共享不损伤的操作判据：两个种子 RA 差均不超过 +1.1mm，平均 MPJPE-local 差不超过 +1.1mm，绝对平移误差比不超过 1.1；两个种子的递推根 geodesic overall/global 及 raw 根 geodesic overall/global 均不劣于 +1.0°。raw/teacher-forced root/finger 与递推结果并报在诊断中，不能把滤波收益误认为测量收益。两臂相对 S38 的可用守护：平均 RA 和 local 不劣于 +1.1mm，平移比不超过 1.1，两个种子的递推及 raw 根 overall/global 均不劣于 +1.0°；共享门通过但两臂均未过参照守护，只能称“共享未额外损傷，但接口尚不可用”。只有两种子，结论为当前数据的工程筛查，不宣称总体统计显著。

两者失败：先检查目标切片、mean、左右乘、单位、路由 mask、梯度、冻结和配对数据流，再判断接口重参数化或不足训练预算。只有共享失败：先检查每类梯度方向/幅值、量纲和容量；宽度/参数匹配控制在独立 prereg 更新后执行。U0 合并48D可作低成本接口诊断，不作为最终局部结构。本轮不扩展到消息传播。

## 结果来源

DDP 初次短测中独立头在 GPU 2/3 的 NCCL 初始化停滞，未进入首批数据。已保留 attempt1 产物，仅终止本任务核验过的进程；统一设置 NCCL_P2P_DISABLE=1 后两臂 20 步均完成。此为系统通信排障，不作为网络结构的失败证据。

`outputs/u1a/initialization.json` 记录 anchor 与 split hash；`debug_contract.json` 记录完整接线；`pair_stream_rank*.json` 记录配对数据流；各 run 的 resolved config、training metadata、last checkpoint 和 eval JSON/NPZ 保留。主表通过 `tools/u1a/report.py` 写 main_row，再由 `tools/report_table.py` 生成；不得手填或借用 S38 成本冒充 U1a 成本。
