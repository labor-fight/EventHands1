# S43 第三轮表征研究（opt-in）

从已完成的 S43-REL 继续，旧默认分支、数据和结果只读。研究入口是 [REL归因](../../docs/网络结构与可观测性诊断研究.md#s43-part-09)、[B原型结果与A问题](../../docs/网络结构与可观测性诊断研究.md#s43-part-10)，[三案定义](../../docs/网络结构与可观测性诊断研究.md#s43-third-round-design-20261028) 与 [100单元交叉审查](../../docs/网络结构与可观测性诊断研究.md#s43-third-round-100-20261028)。

本目录中的原子63D关节代理/10类合成读出不等于完整51D MANO方法。是否训练独立前端由实际实验与明确协议判断，不从文件名或代码存在推出方法成功。

## 已完成归因的复现

仓库根执行，使用新输出目录，不覆盖正式读数：

```bash
export PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
export ROUND=outputs/s43_revisit_20261028
export REPRO=outputs/reproduce_s43_revisit
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.representation_audit \
  --self-check --out "$REPRO/arithmetic"
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.rep_readout_probe \
  --prereg "$ROUND/stage1_protocol.json" --out "$REPRO/reader_ladder" --threads 2
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.rep_direction_probe \
  --prereg "$ROUND/stage1_direction_extension.json" --out "$REPRO/direction" --threads 2
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.interface_audit \
  --protocol "$ROUND/stage1_protocol.json" --out "$REPRO/interface"
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.interface_optimization \
  --protocol "$ROUND/stage1_protocol.json" \
  --extension "$ROUND/stage1_optimization_extension.json" --out "$REPRO/checkpoints"
```

`freeze.py` / `freeze_stage2.py` 是已执行的一次性注册入口，已存在时拒绝覆盖，不能拿重跑预注册充当新实验。`report_attribution.py`、`compile_battle.py`只从已保存结果重建指定文档节；`validate_records.py`仅核已冻结依赖、100格身份与新增链接，不训练或重评分。

## 原型核心与新检查

- `witness_core.py`：原生ms计数、固定封槽、有限片段选择、完整coarse支撑与缺测标识；不使用LNES/A/Z。来源防重和同payload真实重数分开；批次lane身份不参与表示随机选择。
- `witness_geometry.py`：公开/部署MANO的有限运动核及受控合成生成器，核不称似然。此源被80条合成生成记录冻结。
- `witness_probe.py`：统一源身份硬门、10组控制、错误前态选择、共享适配CNN及RBF。必须工程验收与实际sourcefreeze后才抽取/拟合；现有288窗继承旧给定形状历史，只算条件信息诊断。
- 运行时缓存修正使用单独新文件和R1身份；原资源失败保留。数学局部加法不等于当前Python封槽/网络整链局部或实时。

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q \
  tests/test_s43_witness_core.py tests/test_s43_revisit_witness_geometry.py
```

正式资源、任务结果及完整前端状态以 `outputs/s43_revisit_20261028/WORK_STATE.md` 与各阶段回执为准。尚未实际跑过的训练、mesh时延或指标不填写成已完成数字。

## Stage2 已完成的负结果

[任务解释与完整预算](../../outputs/s43_revisit_20261028/witness_atomic_v1/atomic_interpretation.md)、[完成验收](../../outputs/s43_revisit_20261028/witness_atomic_v1/completion_validation.json)、[真实流表示成本](../../outputs/s43_revisit_20261028/witness_resources_v2/results.md)。B在两读出、三种选择前态均未超过完整2px场；原始失败和一次读数前缓存修正均保留。独立合成审计 `witness_synthetic_diagnosis.py` 只重验原生证据／已保存预测，没有重新训练或修改门。

Stage2复现分离观测与拟合，使用新路径；冻结文件校验的是原依赖与输入身份，不允许先改旧源再沿用冻结：

```bash
export B_FREEZE="$ROUND/witness_probe_engineering/actual_sourcefreeze.json"
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.witness_probe extract \
  --domain synthetic --queries 50 --out "$REPRO/B_synthetic_features" --source-freeze "$B_FREEZE"
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.witness_probe fit \
  --domain synthetic --features "$REPRO/B_synthetic_features" \
  --out "$REPRO/B_synthetic_readers" --source-freeze "$B_FREEZE" --device cuda:0
```

既有数据使用 `--domain existing`，extract不加合成专用queries。该入口每域／每折混合所有sampler，不能冒充每组都独立2000步。

## Stage3 共享时空场与适配读出

- `field_core.py`：有界native-ms计数环、完整来源合同和pending同刻组，numeric4,408,000B；同一core可用于训练构造与在线读出，未保存学习latent。
- `field_query.py`：五种因果查询坐标，15指骨段×3锚×2符号×8时间，固定骨向局部框架。generic轨迹与SO(3)路径共享精确端点；hand-box只给ROI。查询不改变原始记忆。
- `field_query_probe.py`：九组分别独立CNN与RBF；错误anchor同时作用于坐标、context、残差基点；corrected视图只作评价，排除FIT。当前仍是10类／63D代理，不是完整MANO网络。

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q \
  tests/test_s43_time_field_core.py tests/test_s43_field_query.py \
  tests/test_s43_revisit_field_query_probe.py
```

[Stage3协议](../../outputs/s43_revisit_20261028/stage3_field_query_protocol.json)与实际首任务sourcefreeze各自记录。`prepare` 只在实际工程门通过、源码和输入冻结后运行；`fit` 明确选择domain和arm，每组／每折2000步。不要重新覆盖原freeze或已评分目录，正式45份预测齐后才统一评分。完整训练的预算、β0部署与资源结果须以后续实际回执为准。

```bash
export A_FREEZE="$ROUND/field_query_engineering/actual_sourcefreeze.json"
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.field_query_probe prepare \
  --domain synthetic --out "$REPRO/A_features/synthetic" --source-freeze "$A_FREEZE"
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.field_query_probe fit \
  --domain synthetic --arm anatomical --features "$REPRO/A_features/synthetic" \
  --out "$REPRO/A_readers/synthetic/anatomical" --source-freeze "$A_FREEZE" --device cuda:0
```

以上仅复现一个arm；完整对照需按协议运行两域九组后统一 `score`。实际运行脚本在 [run_field_atomic_v1.sh](../../outputs/s43_revisit_20261028/field_query_engineering/run_field_atomic_v1.sh)，绑定原输出路径且已经启动，不要再次执行它；复现时复制为新脚本并替换所有输出路径，保留原工件。首实际sourcefreeze为 `03f285756fef89d939b5dcc228551ddfdcc8284abe006b0d98d2030ddf39717f`。固定预测归档、完整源码末检与结果报告各自记录，运行中不读取部分SELECT指标。

## Stage4：identity-init MANO 读出与外供流客户端

Stage4 已完成74份固定2000步拟合，比较63D关节输出与51D右乘SO(3) MANO解码，损失固定10mm标量尺度、末层零初始化。结果和验收在 [field_mano_atomic_v1](../../outputs/s43_revisit_20261028/field_mano_atomic_v1/)。这是小型读出研究；完整S43网络与充分训练归Stage5。

```bash
export S4_FREEZE="$ROUND/field_mano_engineering/actual_sourcefreeze.json"
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.newfield_mano_probe fit \
  --domain existing --arm hand_box --reader mano51 --regime nominal_guard \
  --features "$ROUND/field_mano_atomic_v1/features/existing" \
  --source-freeze "$S4_FREEZE" --device cuda:0 --out "$REPRO/S4_handbox_guard"

# 外供integer Nx4[x,y,t,p]，p只能0/1；β0和公共初态，无GT评分。
# 此处是已训练的小型atomic head，不是Stage5完整RN权重。
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.field_mano_online \
  --events "$ROUND/../s43_rel_20261027/engineering/external_client/ch350/events.npy" \
  --calibration "$ROUND/field_mano_online_engineering/ch350/calibration.json" \
  --checkpoint "$ROUND/field_mano_atomic_v1/readers/synthetic/static/mano51/fold0/head.pt" \
  --serving-contract "$ROUND/field_mano_engineering/serving_contract.json" \
  --time-unit ms --shape mean0 --hop 50 --device cpu --out "$REPRO/S4_stream.jsonl"
```

Stage4全部拟合的实际队列脚本为 `field_mano_engineering/field_mano_fitqueue.py`，固定指向已完成目录，不能再次原地执行。单份重现实验写新目录。两种reader、全部fold及4个nominal guards的预测必须齐备后才统一 `score`；单份新拟合不能代替原74份结论。

## Stage5：完整S43、R1工厂和充分训练

截至2026-10-28工程交付，full尚未完成，最终准确率和独占GPU资源不填为已完成。当前五臂为 `s43/field_box/field_box_collapse/field_static_collapse/field_static`，每臂6,144,000个实际监督输出、B64×clip4、seed3407，从共同canonical重新开始；pilot权重不接着当full初值。ycy/ylf属于全量训练域，后续主评中的这些录制只作训练诊断。βinput固定0，真实β只用于训练监督和保存预测后的评分；显式known-β诊断另列。

attempt0有限预测导致ROI越过近面，失败源和轨迹保留。R1只为有限无效几何加入同场全图uniform9×10查询，association标志置0；不clip/hold/reset姿态。模型、原loss与预算保留。冻结身份为：

- R1 pilot实际sourcefreeze：`s43_full_field_R1/engineering/actual_sourcefreeze.json`，SHA `fa2fb3d7a1f6159b96279f21db2d106d014809192dcf5a719faed57087b91249`。
- full执行sourcefreeze：`s43_full_field_R1/engineering/full_execution_sourcefreeze.json`，SHA `2d0969ce7de66fb393cc9fc8fa0d2c5285c85b5df0501301a83016d312c19d98`。CPU整数core及R1网络不变，直接native-ring读出避免复制；旧pilot通过显式审计桥接，full仍fresh canonical。
- full权重必须是 `s43_field_full_training_v1` / `execution_variant=native_ring_readout_full_v1`；工厂严核R1协议、`invalid_geometry_uniform_r1`模型和 `native_ring_bins_v1`读出合同。pilot旧R1格式只用于显式工程客户端；旧失败格式不会自动加载。

已启动的外层dispatcher会等五pilot门、运行五full、完成配对审计，再自动进入唯一collector。不要另开同目录队列。以下是对应的直接入口，供核对命令或在明确恢复既有作业时使用：

```bash
export S5="$ROUND/s43_full_field_R1"
export E5="$ROUND/s43_full_field_R1_evaluation_v1"
export FULL_FREEZE="$S5/engineering/full_execution_sourcefreeze.json"

# sourcefreeze及显式pilot转交记录齐备后，full-only队列入口。
# 当前已有dispatcher；不要重复运行。
"$PY" -m tools.s43_revisit.s43_field_full_queue --source-freeze "$FULL_FREEZE"

# 查看训练进度，不打开主评GT。
tail -n 1 "$S5/train/full/field_static/curve.jsonl"
ls "$S5"/all_final_weights_complete.json "$S5"/audits/full.json

# 只有确认旧controller/worker已退出且恢复身份已处理后，单臂才用同源last.ckpt续跑。
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.s43_field_full_training train \
  --config "$S5/configs/full_fastreadout/field_static.json" \
  --source-freeze "$FULL_FREEZE" --out "$S5/train/full/field_static" \
  --resume "$S5/train/full/field_static/last.ckpt" --device cuda:0
```

恢复只接受完成optimizer更新、输入/预测/commit均结束的checkpoint：模型、Adam、调度器、RNG、源游标、整数core与有界历史一起恢复。`final_weights.ckpt`只供推理，没有完整训练状态；pilot checkpoint与不同source/执行variant不能混作resume。恢复不重置时间尾、不将预算从头计数。实际配置路径以full freeze中 `configs` 清单为准。

## Stage5：predict-only、评分门和资源

```bash
# 无GT短流工程，原生S43不维护无用field；使用新的输出前缀。
CUDA_VISIBLE_DEVICES="" "$PY" -m tools.s43_revisit.s43_field_replay predict \
  --arm canonical --recording ch_local --engineering --stop-ms 350 \
  --native-s43 --threads 2 --device cpu --out "$REPRO/S5_canonical_ch350"

# 正式predict-only也要求全部五full末权重和配对审计；这里不评分。
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.s43_field_replay predict \
  --arm field_static --recording zgz_local \
  --checkpoint "$S5/train/full/field_static/final_weights.ckpt" \
  --device cuda:0 --threads 2 --out "$REPRO/S5_fixedlast_zgz_local"

# 唯一正式collector：外层dispatcher会自动调用；手动核对时不要再开第二份。
"$PY" -m tools.s43_revisit.s43_field_collect --wait \
  --train-root "$S5" --out "$E5"
```

collector固定120条语义唯一轨迹：36主流、12known-β条件、36cadence、36一次性前态扰动。36条β0主流保存齐后，显式提供known-β的真实形状JSON；全部120条预测与来源、地址、warmup、commit、条件和权重核验后才读GT pose。known-β是GT辅助条件诊断，不能称为无GT部署精度。主表J0对齐joint0腕，V0对齐vertex0；绝对J/V和真腕对齐V保留。其他query间隔比较同时报告100ms共有地址交集。新field固定W50，S39/W200栈没有被强行套用。

资源在训练和准确率worker结束后逐模型独占GPU4，CPU2、FP32/TF32关闭。每个case前只读nvidia-smi并实际排队等固定GPU空闲，不杀他人进程、不选另一张快卡；执行中观察到foreign进程则保留失败，不采用其性能。主时延来自无内部hooks的第一遍；第二遍独立分类、第三遍独立统计NN主要Conv/Linear MAC×2，各遍都核同51参数、mesh和source。masked dense卷积按完整实际执行shape计；occupancy理论单列，geometry/CPU/FK不塞入网络FLOPs。原0.101G只保历史口径。输入mmap/打包计入服务，输出序列化、冷加载与source预/后哈希在外；预哈希会暖OS缓存。

线程以 `threadpoolctl.threadpool_info()` 实测记录：此环境NumPy用MKL，`OPENBLAS_NUM_THREADS=1`不能据此声称BLAS1，`torch.set_num_threads(2)`也影响共享MKL。资源/新客户端在固定设置后记录实际所有BLAS/OpenMP pool；不为这项元数据改训练或推理数值设置。

失败不伪造hold/clamp输出：保留真实部分轨迹、未消费数量及重置要求。失败资源没有完整流吞吐。canonical与新训S43的原生时间戳服务和额外shared-audit成本分列。所有六模型均统计已解码输出的近面/负root-z，warm/steady、query/commit分母分开；field输入前态回退计数另列。

## Stage5：独立外供流与时延

`s43_field_online.py`直接读用户的 `events.npy`（整数N×4 `[x,y,t,p]`，p为0/1）和calibration JSON；不实例化DatasetReader、不读aux/meta或目标。JSON给 `K_source`640×480或 `K_event`240×180之一，后者反算source时只除一次.375；模型查询转回event时只缩放一次。可给 `boundary_ms/stop_ms` 或 `valid_runs_ms`，声明不能丢弃输入。us时间只在当前包floor到ms；没有全时间戳副本或亚毫秒顺序特征。默认β0和公共初态 `[0,0,.6]+48zero`；提供外部 `beta_input` 必须显式 `--shape provided` 并标条件来源。

```bash
CUDA_VISIBLE_DEVICES=4 "$PY" -m tools.s43_revisit.s43_field_online \
  --events /path/to/events.npy --calibration /path/to/calibration.json \
  --checkpoint "$S5/train/full/field_static/final_weights.ckpt" \
  --time-unit ms --shape mean0 --hop 50 --device cuda:0 --threads 2 \
  --out "$REPRO/S5_online.jsonl"
```

每真实run先两次自身50ms预热，再按25/50/100ms查询；`--hop 25 --commit-ms 50`保留额外纯查询。JSONL每条含51D状态、778顶点、21关节和warm/commit标识；summary有分项服务时间、来源身份、实际checkpoint预算和输出深度诊断。核心/历史有界，输入逻辑包上限默认1,048,576事件（最多32MiB materialization，超域拒绝不截断），O(T)输出/计时日志另列。客户端工程时延不能代替正式独占GPU资源测量，运行成功也不表示方法科学门通过。

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m pytest -q \
  tests/test_s43_field_replay.py tests/test_s43_field_benchmark.py \
  tests/test_s43_field_operations.py tests/test_s43_field_online.py \
  tests/test_s43_field_resource_wait.py tests/test_s43_field_perturb.py
```

正式完成以 `$E5/evaluation_complete.json`、`report/results.json`、`resources_complete.json` 和 `actual_evaluation_sourcefreeze.json` 为准；文件未齐时保持未完成，不从pilot或部分ZGZ结果选模型。
