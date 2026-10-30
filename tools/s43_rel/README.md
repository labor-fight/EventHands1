# S43 有向事件关系记忆

本轮最终裁决是 NO_GO_S43_DEFAULT_REPLACEMENT：精度大体保持，但尚未证明相对简单增强/核心消融的独特任务价值；β0完整软件mesh输出P50为57.62ms，配对S43为9.79ms。保留原S43默认，新表示仅作可复现实验分支。完整结果见 docs/EXPERIMENT_RECORDS.md#s43-rel-doc-20261027（原 docs/S43_REL_实现训练与流式结果_20261027.md，2026-10-28 并入账本）、outputs/s43_rel_20261027/mean_shape/results.json 和 final_decision.json。

这是原 S43 的独立 opt-in 分支。事件写入与 MANO 无关，查询时才读取 MANO 耦合方向与事件加权支持。原生产源码、旧 B-R 和 S43/S39 默认指针不改。

本轮是完整的配对微调协议：四组同 seed3407、同 canonical step6000 初值、同事件/标签排程，已各完成6000次实际优化更新、192000个输出，并通过全程配对来源核验。该输出预算为历史6.144M的1/32，不能称相同预算的从头训练。实际结果和验收请以 outputs/s43_rel_20261027/ 下的 completion/report 为准；过程中的 loss 不是最终递推精度。

## 环境

~~~bash
cd /data1/lyq/code/mesh/EventHands1
export PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=2
~~~

需要现有 EventHandsTrain 环境、g++、MANO asset、原 S43 检查点及原 splits_semkine.json。只使用物理GPU4–7；0–3不使用。CPU kernel 按源码和编译标志身份缓存构建（core在临时目录、几何在outputs/s43_rel/build或指定目录），冷编译成本单列。

配置入口：

- configs/s43_rel/*.yaml：R0 points 读出，保留初始负结果。
- configs/s43_rel/r1/*.yaml：最终 R1 joint_gram 读出。
- 四组为 s43、enhanced、no_order、full。

四组保留原 LNES/render 通路。增强有真实 last/first/count/4-bin/EMA；full/no_order 有真实6通道A和同一typed读出；no_order在固定p/q槽内作空间反向Z平均。full不是将候选模板伪装成LNES，也不是独占替换全部LNES的模型。

外供条件沿原 S43：240×180事件坐标、原640×480 K 映射一次、序列β。现有回放β来自原aux/sequence_gt，精度属于给定个体形状的条件协议；真实部署需外部标定β或另行评估均值形状，不能称无需标定的纯事件重建。姿态GT、真实事件面身份及未来轨迹不进入观测构造或历史状态。

## 共用核心

- core.py / core_cpu.cpp：完整同刻组的正滞后、双极性、固定指数核 A/Z；有序来源防重合同、水位、纯查询、数值状态导出。
- geometry.py / geometry_cpu.cpp：可见网格轮廓、解析局部SO(3)作用、稀疏关系读取；近π不使用固定log chart。
- joint_reader.py：方向f，以及由(a,lambda)得到的观测加权C和diagG；不将Gram作为Fisher/协方差。
- data.py / training.py：连续run lanes、自身预测反馈、TBPTT1；不在optimizer步或普通clip边界reset尾迹。
- replay.py：独立 ingest/query/commit 在线调度，使用同一core/geometry/model；W50读取和查询间隔分离。

原始事件格式为 int64 N×4，列顺序 x, y, integer_timestamp_ms, stored_polarity_0_or_1。真实同payload多触发保留重数；不依据同ms数组次序制造运动。source_range与packet_ordinal来自有序唯一来源；乱序/迟到违反合同会明确拒绝，不暗中丢事件。

数学与状态合同见 docs/网络结构与可观测性诊断研究.md#s43-redesign-20261027（原 docs/S43_三种表征方案与主方案定义_第二轮重设计_20261027.md，2026-10-28 并入研究稿附录 O）；R1读出修正见 outputs/s43_rel_20261027/correction_R1_prereg.json。指数记忆有连续历史尾部，不是精确W50滑窗。W50原始输入缓冲另有262144事件硬界，超域在所有状态写入前整包拒绝；不裁count。W50原始payload最多8MiB；A/Z核心另占18,616,960B/序列，共享拓扑5,510,720B，模型/workspace/日志/轨迹另算。

## 验收与机制实验

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q \
  tests/test_s43_rel_core.py tests/test_s43_rel_geometry.py \
  tests/test_s43_rel_model.py tests/test_s43_rel_replay.py \
  tests/test_s43_rel_joint_reader.py tests/test_s43_rel_geometry_fast.py \
  tests/test_s43_rel_online.py tests/test_s43_rel_deployment.py
~~~

各机制入口先用 --help 查看当前参数。读数前协议、输入清单、固定读出及源SHA在各结果目录保存，禁止覆盖已读分数。

~~~bash
"$PY" -m tools.s43_rel.mechanisms --help
"$PY" -m tools.s43_rel.support_experiment --help
"$PY" -m tools.s43_rel.readouts_revision --self-check
~~~

E1/E2主结果在 mechanisms/e12/summary.json；E3在 mechanisms/e3/summary.json；唯一读出修正结果在 mechanisms/e12/R1/summary.json。合成、现有v2e数据、纯代数验收分开。静态闪烁反例说明非零支持不是可靠运动置信度；TORE4同核对照保留。R1后现有数据E1双读出联合门仍未过，E2任务收益不一致，合成与TORE4同核对照相当；E3静态闪烁反例禁止把支持当置信度。不得用公式一致或toy成功替代任务收益。

完整重跑命令如下，使用新的输出目录保留原始结果。R1读出明确复用已交付的冻结 E12 特征和原头分数，不重新选择原始证据或基线：

~~~bash
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_rel.mechanisms \
  --out outputs/reproduce_rel_e12 --stage all --device cuda:0 --workers 8
"$PY" -c 'from pathlib import Path; from tools.s43_rel.readouts import summarize; summarize(Path("outputs/reproduce_rel_e12"))'
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_rel.support_experiment \
  --out outputs/reproduce_rel_e3 --duration-ms 1000
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_rel.support_experiment \
  --existing-cache outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz \
  --target-unit mm --out outputs/reproduce_rel_e3_task
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_rel.readouts_revision --run \
  --prereg outputs/s43_rel_20261027/correction_R1_prereg.json \
  --device cuda:0 --out outputs/reproduce_rel_e12_R1
~~~

## 训练

首次实际运行使用已交付的 protocol.json 及阶段 authorization JSON。授权文件记录的是本轮用户指令和流程门，不是新增用户审批；所有配置绑定SHA。已存在结果拒绝覆盖，复现使用新tag。

~~~bash
"$PY" -m tools.s43_rel.orchestrate launch \
  --stage small --tag reproduce_small_r1 --config-dir configs/s43_rel/r1
"$PY" -m tools.s43_rel.orchestrate launch \
  --stage full --tag reproduce_full_r1 --config-dir configs/s43_rel/r1
~~~

单卡入口示例：

~~~bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m tools.s43_rel.training train \
  --config configs/s43_rel/r1/full.yaml --stage full \
  --out-dir outputs/reproduce_rel_full \
  --device cuda:0 --protocol outputs/s43_rel_20261027/protocol.json \
  --authorization outputs/s43_rel_20261027/authorization_full_r1.json
~~~

小阶段120×B4×clip2；全阶段6000×B8×clip4，Adam1e-4，full warm500/cosine，BN统计冻结、原模型参数可训练。优化目标为各phase原log10-loss平均，curve.loss记录各phase原始loss平均，两者不混称。clip为梯度累积/自反馈段，TBPTT=1。训练从每个eligible run公共初态开始，无GT前态；评估的共同canonical两窗预热另列。small的SELECT为ycy/ylf；full纳入全部九人，故最终ycy回放是训练域诊断，不能继续称未见个体SELECT。

last.ckpt保存模型、优化器、调度器、RNG、lane游标、完整A/Z及自身预测状态。固定step=*.ckpt仅存权重供固定检查，不是完整resume。6000步只评价最后权重，不选择最好checkpoint。短于50ms的run不作为训练样本；已选run的末短尾在下一次换run时消费，不制造额外监督。

## 完整流式评估

~~~bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m tools.s43_rel.replay \
  --config configs/s43_rel/r1/full.yaml \
  --checkpoint outputs/s43_rel_20261027/train/full_r1/full/step=6000.ckpt \
  --recording zgz_local --hop 50 --duplicate-query \
  --out outputs/reproduce_rel_full_zgz_local --device cuda:0
~~~

--hop 可为25/50/100，原LNES观测始终W50，A/Z消费全部新事件。--hop 25 --fixed-commit-ms 50用于检验额外纯query：50ms处输出应与固定50ms策略一致；不同commit日程属于不同策略，分开评价。

默认所有valid run、共同预热与尾事件都消费，并核验计数和顺序摘要。仅完整推理结束后读姿态GT评分；valid-run外事件未被冒称评分。结果包括JSON、预测/GT/end/run的NPZ、同步姿态输出分位时延、吞吐、CUDA/RSS峰值及分项维护成本。完整录制回放的service止于51D姿态，MANO网格仅在离线评分时构建。独立资源基准和外供客户端实际包含末端MANO解码及网格传输，作为sensor-to-mesh成本。所有service不包含物理传感器等待，1ms目标滞后及watermark等待单列；不复用历史缩放GPU时延。

~~~bash
CUDA_VISIBLE_DEVICES=7 "$PY" -m tools.s43_rel.benchmark \
  --config configs/s43_rel/r1/full.yaml \
  --checkpoint outputs/s43_rel_20261027/train/full_r1/full/step=6000.ckpt \
  --recording ch_local --hop 50 --queries 64 \
  --out outputs/reproduce_rel_resource.json --device cuda:0
~~~

--synthetic-events-per50 可选10000/50000/150000，压力输入和真实前缀分开报告。benchmark不读姿态GT；资源包含共同canonical预热副本，冷加载/编译/预热另列。最终资源基准使用weights-only step=6000.ckpt，避免加载8条训练lane与Adam状态；last.ckpt用于完整恢复。v2_surface的主latency包括原base._fk及778顶点/21关节D2H，pose51_latency另列。单batch训练吞吐不作为在线batch1延迟。

最终完整权重和nominal hop50报告齐备后可运行 robustness.py：固定前态扰动、fresh/stale条件、GT只作评分的局部手指运动分层。脚本会拒绝用未完成6000步或不匹配权重的轨迹作正式结论。

## 解释边界

新增typed结构和A通道的作用须与增强及no_order对照分开。前态无关写入只保证声明有限核的重读，不保证任意历史成像似然、全局捕获范围或遮挡身份恢复。当前网络仍在查询时执行全局几何与原主干前向；只有观测记忆更新是局部增量。

geometry_fast.py是额外等价工程候选，保留全部LBS权重并验收数值一致；它不进入本轮训练或默认评估，也不是第二科学修正。单FK/J速度不能替代整链性能。

## 外供事件流客户端

`online.py` 是可独立运行的无标签客户端。它只读取外供事件 `.npy`、标定 JSON、模型配置和权重，不实例化数据集 reader，不打开 target/meta，也不要求提供个体 GT 形状。它复用同一个 `CausalRelationTracker`，不改变上述训练、科学修正或正式评分协议。

事件文件必须是可 mmap 的整数 `N×4` 数组，列为 `[x,y,t,p]`：坐标为240×180，存储极性为0/1。`--time-unit ms` 使用整数毫秒；`us` 在每个有界包中执行 `t//1000`，不利用同毫秒内顺序。跨毫秒组乱序会明确报错，不排序或丢事件。时间列只保留 mmap view，按固定65536行块校验，再用 `searchsorted` 定位每个1ms包，不复制全流时间轴。

标定 JSON 必须提供原640×480网格的 `K_cam[3,3]`。可选 `betas[10]` 和 `initial_state[51]`；省略时分别使用全零均值形状和公共初态 `[0,0,0.6,0,…]`。例如：

~~~json
{"K_cam": [[1000,0,320],[0,1000,240],[0,0,1]]}
~~~

`--start-ms`、`--stop-ms` 声明完整半开覆盖区间，并须包括文件内全部事件；省略时取首事件所在bin到末事件bin结束。无事件的边界区间应显式提供，空流必须提供两项。客户端先执行两次共同 canonical W50 预热，再按25/50/100ms查询并提交，始终以W50读取原LNES、连续维护A/Z；最终短尾全部消费。每条JSONL结果包含51维姿态、米制相机坐标的778顶点和21关节、证据截止和 `t_query=cutoff−1ms`。超过W50的262144事件上限时整包拒绝；输出日志是另列的O(T)磁盘产物。

已有hand_data51事件文件前350ms无标签工程样例，不提供GT β，也不评分：

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_rel.online \
  --events outputs/s43_rel_20261027/engineering/external_client/ch350/events.npy \
  --time-unit ms \
  --calibration outputs/s43_rel_20261027/engineering/external_client/ch350/calibration.json \
  --config configs/s43_rel/r1/full.yaml \
  --checkpoint outputs/s43_rel_20261027/train/full_r1/full/step=6000.ckpt \
  --hop 50 --start-ms 0 --stop-ms 350 --device cpu \
  --out outputs/reproduce_external_stream.jsonl
~~~

使用GPU时设 `CUDA_VISIBLE_DEVICES=4`（或5/6/7）及 `--device cuda:0`；6000步最终权重已经完成，并实际通过同一350ms样例的五次完整mesh输出检查。客户端生成JSONL和同名 `.summary.json`，保存源文件字节数/SHA、标定身份、全事件计数/摘要、尾部计数及简单service统计。工程样例只证明均值形状条件下该无标签接口能执行，不证明均值形状精度或跨个体效果。

规则多边形轮廓的角点、自遮挡切换和退化分支有近似边界；branch_regular已记录，当前神经读出未将其作为可靠性门。no_order仅抹空间反向有向分量，仍保留正滞后、极性及age时间核信息；匹配审计见audits/spatial_order_intervention.json。原项目标题或历史实时结果不代表此新分支已满足实时目标。

## 无 GT 手形输入的部署验证

上面的原 S43 兼容回放沿用 aux/sequence_gt β，是给定手形的条件诊断。独立的 mean_shape_* 路径把十维 β 固定为公开均值0；推理禁止打开姿态 .meta 或读取 aux betas 字段，只使用事件、外供 K、公共初态和声明的序列区间。全部预测先落盘，然后评分阶段用 β0 解码预测网格、真实评分 β 解码 GT 网格。训练仍含个体形状条件监督，不将均值形状测试称作形状免标定精度保证。

冻结扩展为 outputs/s43_rel_20261027/mean_shape_deployment_extension.json；资源与准确率只在原 postprocess_complete 及 finalization_complete 两门都完成后执行。五组为 canonical 原冻结 S43，加 s43/enhanced/no_order/full 四个6000步微调模型，使用同一 β0 协议。以下命令使用新输出避免覆盖：

~~~bash
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_rel.mean_shape_benchmark \
  --arm full --device cuda:0 \
  --out outputs/reproduce_rel_mean/resources/full_h50.json
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_rel.mean_shape_replay \
  --config configs/s43_rel/r1/full.yaml \
  --checkpoint outputs/s43_rel_20261027/train/full_r1/full/last.ckpt \
  --recording zgz_local --out outputs/reproduce_rel_mean/replays/full_zgz_local_h50 \
  --device cuda:0 --threads 2
~~~

其余微调臂按同样方式替换 arm/config/checkpoint；canonical 采用 s43.yaml 并省略 --checkpoint。补齐五组资源与五组×两条 local/global 流后：

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_rel.mean_shape_report \
  --root outputs/reproduce_rel_mean
~~~

正式队列为 tools.s43_rel.mean_shape_queue，执行顺序是五组 GPU4 串行资源、十条 GPU4–7 并行回放、严格汇总；已有正式目录会拒绝重复写入。实际结果保存在 outputs/s43_rel_20261027/mean_shape/。独立 mean-shape 验收使用 tests/test_s43_rel_mean_shape_replay.py 与 tests/test_s43_rel_mean_shape_tools.py；最终统一测试命令为：

~~~bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q tests/test_s43_rel_*.py
~~~

原项目四指标沿用 J0（关节腕点）/V0（顶点0）各自对齐，V0不等同腕点对齐MPVPE。β0结果额外保存共同关节腕点对齐MPVPE与绝对J/V误差。局部手指分层仅复用既定GT事后mask，按完全相同(run_id,end_ms)对齐已保存误差；无样本会明确记缺证据。Torch allocator峰值、NVML设备20Hz观测基线与增量分别记录，不把包含设备reserved的NVML raw used称为精确进程显存。
