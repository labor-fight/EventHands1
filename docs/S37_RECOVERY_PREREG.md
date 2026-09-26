# RC1 — 完整训练断点恢复工程合同

2026-09-26。对应用户目标§9，属于工程修复，不是新的科学候选或精度实验。

## 缺口与最小实现

实际 S37 checkpoint 保存模型、优化器、scheduler 和 Lightning loop 进度，但无主训练进程 RNG、shuffle order 与已消费游标。证据：`research_state/audit/CHECKPOINT_RECOVERY_20260926.json`。

官方 [Lightning LTS 文档](https://lightning.ai/docs/pytorch/LTS/clouds/fault_tolerant_training_basic.html) 将 FT 标作实验功能；本机1.9.5源码的 automatic 验证拒绝当前 RandomSampler/乱序DistributedSampler，不能直接设置环境变量并宣称完成。独立代理核库源码，父实现并实际运行。

`semkine/checkpointing.py` 保留 Trainer：使用原 PyTorch RandomSampler 与原 Lightning DistributedSamplerWrapper 的乱序，保存每rank完整permutation，只在已消费训练批次后提交游标。预取不推进保存位置；当前 SemKineDataset 的增强是(seed,index)纯函数，不依赖 worker 的持久随机流。

保存模型/优化器/scheduler/适用scaler仍交给Lightning。附加每rank Python/NumPy/Torch CPU/当前CUDA RNG和采样状态。固定步网格延到下一批执行前保存，序列化副本清除未执行批的ready/last标记，并编码真实data epoch；不改运行态计数。恢复RNG晚于初始化/loader启动，完整epoch之后则早于下一次shuffle生成。

早停可能触发额外的非计划验证；完整模式下仅为这种额外验证保存/恢复训练RNG，正常验证维持原行为。限定当前模型的训练/epoch hooks合同；epoch结束不会改权重或执行epoch scheduler。当前S37为Adam和step LambdaLR。若引入这类额外状态更新，须重新验证，不能将本结果外推到任意Lightning模型。

## 使用与边界

新训练显式加 `--complete-checkpoints` 或配置 `TRAIN.COMPLETE_CHECKPOINTS: true`；已有旧训练默认路径保留。从新checkpoint恢复自动启用完整模式。新模式拒绝旧checkpoint及不匹配的代码/配置/设备拓扑/数据索引/加载参数；旧 `--resume` 明示仅恢复已有optimizer等状态，缺RNG/sampler，不伪称仅权重或完整恢复。

支持同world size、同执行合同的完成optimizer step。梯度累积中途的grad未保存，不认证任意异常文件；周期checkpoint保留安全更新边界。不自动启动旧X1或扩训练预算。当前实现对Lightning1.9.5版本绑定；后续升级须复测。

恢复合同含有效配置、源文件SHA256、MANO/划分/索引SHA256、事件文件大小/mtime、torch/Lightning/CUDA版本。大事件文件大小/mtime是身份检查而非全内容hash；完整恢复另要求数据内容不变，不能声称检测到保留mtime的任意原位篡改。

## 验证与固定判据

CPU单训练进程、4库线程，worker=0/2用于实际loader预取，非CPU分析分片。连续运行与从保存文件新建Trainer恢复，比较后续全部样本ID、epoch/batch_idx/step、Python/NumPy/Torch随机数、loss、完整参数、optimizer、scheduler、最终sampler与RNG；要求逐位一致。覆盖epoch末批前保存、跨epoch、验证重合、累积2批、完整/未完成epoch的last。拒绝旧checkpoint/不同seed。

CPU v1实际发现未执行最后批被is_last_batch跳过；修正序列化副本。v3/v4分别暴露早停额外验证的RNG和完整epoch恢复问题；修正后v5全部8项通过。原失败日志保留在 `scratch/goal_20260926/rc1/`。没有降容差或略去失败场景。

下一CUDA合同：GPU1+2跑2-rank FP32 DDP；GPU3跑单卡FP16和BF16，独立作业可并行，各最多180 wall秒（预算按卡数×全子进程wall收费）。每case13个toy optimizer step，step4恢复；累积2、workers2及验证重合。所有rank的样本、随机数、loss、参数/optimizer/scheduler/完整恢复字段逐位一致；FP16另核scaler，BF16明确无scaler。固定三case，不以CPU代CUDA通过。仅合成数据，无真实训练/zgz/模型指标。

代码、CPU日志、合同冻结后才启动GPU作业。错误须定位修复、复测受影响范围；不得把仅权重加载或Debug通过写成方法有效。GPU未通过不得认证该精度/并行范围；当前科学训练准入门不因工程通过改变。


## 2026-09-26T02:17:23.984616+08:00 合成CUDA终态与实际入口Debug合同
RC1合成CUDA：2-rank FP32 DDP与单卡BF16全部逐位通过；FP16原job在最后checker查询obsolete scaler key时失败，训练保存数据未失配。根据实际PL1.9.5的MixedPrecisionPlugin键重新核保存artifact，全部逐位通过且scaler确认存在；不重跑FP16、不改写原FAILED回执。三个budget job全部终态/cleanup无残留，累计Debug6548.858925030567/12600 GPU秒。证据gpu_ddp_v1/receipt.json、gpu_mixed_v1/16/verification_corrected.json及gpu_bf16_v1/receipt.json。
独立源码审后的入口修复：仅rank0写metadata，源身份覆盖全部semkine/model Python，teacher绑定内容；legacy_lnes合法缺失tsub记录None，raw仍由原loader拒绝。下一唯一有界真实入口检查：S37原架构/增强/划分/LR，明确Debug batch2×2GPU、workers2、4step、save2，BF16；从step2新进程恢复自动识别完整模式，比较每rank所有后续输入tensor hash、Python/NumPy/Torch/CUDA RNG、loss及完整checkpoint逐位一致。使用原train入口，scratch wrapper仅记录，不改模型数学；val_interval1000且max4不评分zgz。不得写main row/方法有效。GPU1/2共180wall秒上限、无效不调容差。


## 2026-09-26T02:18:54.367363+08:00 RC1入口启动参数错误及明确修正
实际入口v1未训练即失败：父命令误重复--gpu，argparse只保留最后一张，PL明确拒绝两卡请求。收费5.63061250501778 GPU秒，原FAILED回执/日志保留，无checkpoint或评分。修正为逗号分隔两UUID、输出新entrypoint_v2目录；脚本仅新增输出参数，冻结v1脚本并登记v2身份。核心模型/训练/恢复代码、4step/逐位判据完全不变。


## 2026-09-26T02:20:51.814238+08:00 RC1真实图训练的数值失败与修复验证
入口v2实际执行完连续4step与从2恢复的2step；每rank输入tensor hash/所有RNG/逐step loss、完整恢复字段、scheduler、step/epoch全部逐位一致，但最终embed.weight与embed.bias及其Adam状态不逐位一致，故原判据FAILED不降容差。差异证据entrypoint_v2/parameter_differences.json（仅这两参数，最大3.129243850708008e-07），不是认证通过。推断待验：事件图反向gather/累加的非确定性。安装PL1.9.5 accelerator_connector.py:222–234表明deterministic=True使用torch严格确定性算法和CUBLAS_WORKSPACE_CONFIG=:4096:8；新完整模式显式开启并关闭benchmark，写进恢复合同。旧默认路径未改变。下一v3保持原数据/4step/两卡BF16/逐位门，只复测受影响实际入口。若不支持确定性算子则显式失败，不warn_only，不以放宽门代修复。入口v2源码冻结，原job FAILED/cleanup保留。


## 2026-09-26T02:23:31.952443+08:00 最终证据与交付
RC1完整训练恢复工程完成，范围限定当前Lightning1.9.5、确定性(seed,index)数据、同执行合同及已完成optimizer step。新增semkine/checkpointing.py，接入semkine/train.py的--complete-checkpoints；新checkpoint自动识别完整恢复，旧checkpoint明确缺RNG/sampler。新模式保存每rank Python/NumPy/Torch CPU/CUDA RNG、原shuffle顺序与实际消费游标，保留Trainer模型/optimizer/scheduler/适用scaler；严格校验源码/配置/数据索引/加载设置。预取不推进保存游标，固定网格保存处理已提交loop状态。仅完整模式启用严格确定性算法，原默认路径保留。
CPU8项通过，合成2-rank FP32、FP16（含scaler）、BF16连续/恢复逐位通过。实际S37两卡BF16入口从step2恢复至4，两个rank后续输入tensor hash、随机状态、loss及最终完整模型/Adam/scheduler/step/epoch/采样状态全部逐位一致（entrypoint_v3/receipt.json）。checkpoint内验证batch完成数为0，未评测zgz、未形成科学训练或新main row。详docs/S37_RECOVERY_PREREG.md、audit/RC1_terminal_parent_20260926.json。
失败保留：CPU早期暴露未执行批/早停验证RNG/epoch边界问题，均修复且不降标准；FP16checker旧键错误只核保存artifact，不重跑；实际入口v1传卡参数错误无训练；v2两embed参数/Adam因非确定性数值不同而FAILED，v3严格确定性修复后按同一逐位门通过。所有job终态、cleanup完成，nvidia-smi无计算进程。RC1累计194.38110592600424 GPU秒；全任务Debug6688.5819763777545/12600，剩余5911.4180236222455。screen/train预算不变。

### 使用入口

新获准训练在既定config/预算上加 `--complete-checkpoints`；从它保存的 `last.ckpt` 或固定step文件恢复时，`--resume <checkpoint>` 自动识别，不必重复此开关。保持原有效配置、world size、数据、加载参数及源码；分支调试请用新的 `--output-dir`。本工程测试不授予尚未过创新门的科学训练准入。

CPU复现：`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python -m pytest -q tests/test_complete_checkpoint.py`。真实入口的逐字命令、设备UUID、预算和回执见 `research_state/jobs/rc1_s37_entrypoint_20260926_v3.json`；所有源码身份已冻结，不自动重跑。

完整模式的确定性设置可能改变吞吐；这里不声称训练速度或推理精度收益。实际S37只验证了当前两卡BF16小batch恢复；任意新模型/epoch有状态hook/设备拓扑/库版本不在认证范围内。GPU0未使用。源/数据内容保持一致是恢复前提，大事件文件大小/mtime检查不等同于全内容哈希。Lightning的通用mid-epoch警告仍原样输出，未屏蔽；本模式的采样恢复另有实际逐位证据。
