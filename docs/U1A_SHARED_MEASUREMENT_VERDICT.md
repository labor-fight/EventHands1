| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S37（统一3种子 last 参照） | 28.75 | 19.55 | 24.13 | 14.20 | 23.83（23.56 / 22.95 / 24.98） | 8.04 ms | 0.827 G | 0.73 M |
| S38 spmeas（匹配2种子 last） | 22.12 | 11.97 | 18.51 | 9.18 | 16.69（15.59 / 17.80） | 7.30 ms | 0.234 G | 1.77 M |
| U1a shared 6k（冻结 S38） | 39.96 | 27.28 | 32.26 | 20.72 | 33.18（39.61 / 26.74） | 7.28 ms | 0.240 G | 1.90 M |
| U1a untied 6k（冻结 S38） | 57.39 | 32.12 | 42.29 | 22.53 | 43.87（47.87 / 39.86） | 8.57 ms | 0.240 G | 7.47 M |

6k判定：共享门通过；shared 与 untied 均未通过 S38 可用守护，共享未额外损伤并不代表当前接口可用。

不符合本轮共享候选采用门；任一 raw 或递推根守护失败都不能由 RA 均值掩盖。

# U1a 共享测量接口 verdict

## 协议与预算

训练固定9人72序列，zgz_global/local同时为开发与测试，仅使用种子3407/3408和 explicit last；S37表行为统一3种子last参照，不是本轮配对对照。
2k与6k各自从同一 prepared init 重新训练，metadata.resumed_from=None；6k不是从2k压缩cosine checkpoint续训。
encoder来源是各seed已经训练6000步的S38，权重及BN统计冻结；2k/6k为额外 head-only 步数，不是与S38从头训练等总预算。
shared与untied保留同输入字段、mask、type/joint embedding、逐槽LN、hidden64和3D输出；同seed的17个untied heads从shared逐tensor复制，配对rank数据流一致。
与原S38相比，两臂同时改了translation槽（原routed T+prev_mlp组合重参数化）、norm支持范围与finger表达；shared相对untied才是本轮主要配对比较，宽度相同也不代表参数量匹配。

## 固定源码与历史输入隔离

本轮后处理采用已封存的 bubblewrap 命名空间：源码原提交 code_origin_commit=a6e59946b35a1227a5b3b8d3c56224c36a57f244，snapshot_root=/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994；在 logical_repo=/data1/lyq/code/mesh/EventHands1 下执行，outputs共享原训练产物。
隔离元数据记录并逐文件核验16份原提交源码、6份历史输入只读绑定；53条历史引用保持原SHA。并发main迁移保留在宿主目录，未重写过去2k/U0产物；current_host_sha256只是封存时宿主版本记录，隔离执行实际读取的是sha256指定的历史字节。snapshot后续support封存提交不冒充源码原提交。
历史S38成本源 s38_spmeas_tf_pert_main_row.json 精确SHA256=d82bfedf76f0bbecfaef0a43bae52e36ba8e736641aca59208d2106165283871；S37统一3seed last源 rt_s37_3seed_tf_pert_main_row.json 精确SHA256=0afdf6f1b8117fe5b5bc144d277a4a263c5c4a3c4f863efdb8ae2750c1e75e99。manifest、launcher、原副本和隔离目标的实际SHA均列于末尾证据表；旧引用仍按原SHA严格校验。

## 保留的2k筛查结果

| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S37（统一3种子 last 参照） | 28.75 | 19.55 | 24.13 | 14.20 | 23.83（23.56 / 22.95 / 24.98） | 8.04 ms | 0.827 G | 0.73 M |
| S38 spmeas（匹配2种子 last） | 22.12 | 11.97 | 18.51 | 9.18 | 16.69（15.59 / 17.80） | 7.30 ms | 0.234 G | 1.77 M |
| U1a shared 2k（冻结 S38） | 29.32 | 24.25 | 24.68 | 19.01 | 26.60（27.16 / 26.04） | 7.31 ms | 0.240 G | 1.90 M |
| U1a untied 2k（冻结 S38） | 46.92 | 34.22 | 35.46 | 24.10 | 40.13（50.23 / 30.02） | 8.67 ms | 0.240 G | 7.47 M |

2k仅为筛查：共享门通过；shared 与 untied 均未通过 S38 可用守护，共享未额外损伤并不代表当前接口可用。

2k不能作最终采用决定。

以上2k失败结果原样保留；6k新结果不能覆盖筛查证据或把筛查结果改称最终通过。

## Debug、DDP与源合同

主代理已执行：python -m pytest tests/test_u1a_readout.py tests/test_s38.py tests/test_s37_routed_readout.py -q --disable-warnings；stdout记录45 passed、18 warnings、5.15s、exit0。
当时未单独重定向测试日志；这是主代理执行记录，末尾测试源码SHA256是当前源码证据，不冒充独立测试执行artifact，也未为生成文档重复运行测试。
debug_contract验证完整forward、非零head下root/finger对prev独立、有限梯度、原mean只加一次、混合空包保持和真实S38损失下降。
两臂各20步双卡DDP退出0；debug及正式phase的last/frozen/finite/paired审计均通过，rank0/1的前两批完整packet字段指纹匹配。
首次DDP通信停滞attempt1已保留，统一NCCL_P2P_DISABLE=1后完成；通信排障不作为网络结构失败证据。

## U0 full接口等价检查及范围

U0在原S38两块输入支持与hidden子空间内折叠root64与finger256为48D块对角读出；保留原T、prev_mlp、MANO、encoder及BN，未新增优化步骤。
两个S38 last seed均覆盖全部2590个固定50ms窗口，原模型与U0每次收到同一GT/noised prev；另含384个非零随机pool的FP32/FP64代数检查及empty/mixed/train-mode forward检查。
通过仅说明当前CPU FP32、禁用TF32、同输入forward的数值等价；独立FP64随机代数亦通过。没有AMP等价、梯度或optimizer等价、闭环递推轨迹/精度等价的证明；train-mode forward不等于训练等价。
U0不是学到的17槽共享head或新准确率臂；它不能证明padding/LN、容量、translation重参数化或训练预算中的任一因素是失败原因。

| seed | full窗口 | 随机FP32 raw48最大差(rad) | 随机FP64 raw48最大差(rad) | 固定窗口raw48最大差(rad) | full51最大差 | 根旋转最大差(°) | T最大差(m) |
|---|---|---|---|---|---|---|---|
| 3407 | 2590 | 1.144e-05 | 2.309e-14 | 5.066e-06 | 2.533e-06 | 7.348e-05 | 0.000e+00 |
| 3408 | 2590 | 1.335e-05 | 2.487e-14 | 4.292e-06 | 2.146e-06 | 8.031e-05 | 0.000e+00 |

## 6k守护与辅助诊断

### 根守护诊断（角度单位 °，正差表示候选更差）

global 指 zgz_global 序列；root 为相机坐标下的 MANO global orientation，不代表世界坐标系追踪。
raw 来自全部固定50ms窗口的 nonempty 测量 Exp(raw) R_ref；teacher-forced gain0.5输出不能替代 raw。

| 比较 | seed | 范围 | raw 候选 | raw 参照 | raw 差 | 递推候选 | 递推参照 | 递推差 | raw门 | 递推门 |
|---|---|---|---|---|---|---|---|---|---|---|
| shared/untied | 3407 | overall | 26.9719 | 33.5306 | -6.5587 | 25.9819 | 31.4083 | -5.4265 | 通过 | 通过 |
| shared/untied | 3407 | global | 18.8488 | 22.5707 | -3.7219 | 18.5441 | 21.8167 | -3.2726 | 通过 | 通过 |
| shared/untied | 3408 | overall | 15.1038 | 33.0217 | -17.9179 | 14.6906 | 30.8468 | -16.1562 | 通过 | 通过 |
| shared/untied | 3408 | global | 15.7156 | 16.8795 | -1.1639 | 15.6695 | 16.2000 | -0.5305 | 通过 | 通过 |
| shared/S38 | 3407 | overall | 26.9719 | 10.0288 | +16.9431 | 25.9819 | 9.0309 | +16.9510 | 失败 | 失败 |
| shared/S38 | 3407 | global | 18.8488 | 8.5687 | +10.2801 | 18.5441 | 8.2942 | +10.2499 | 失败 | 失败 |
| shared/S38 | 3408 | overall | 15.1038 | 10.9898 | +4.1140 | 14.6906 | 10.1286 | +4.5620 | 失败 | 失败 |
| shared/S38 | 3408 | global | 15.7156 | 8.7484 | +6.9672 | 15.6695 | 8.4003 | +7.2692 | 失败 | 失败 |
| untied/S38 | 3407 | overall | 33.5306 | 10.0288 | +23.5018 | 31.4083 | 9.0309 | +22.3775 | 失败 | 失败 |
| untied/S38 | 3407 | global | 22.5707 | 8.5687 | +14.0020 | 21.8167 | 8.2942 | +13.5225 | 失败 | 失败 |
| untied/S38 | 3408 | overall | 33.0217 | 10.9898 | +22.0319 | 30.8468 | 10.1286 | +20.7182 | 失败 | 失败 |
| untied/S38 | 3408 | global | 16.8795 | 8.7484 | +8.1311 | 16.2000 | 8.4003 | +7.7996 | 失败 | 失败 |

每个种子、overall/global 的 raw 与递推根差均须不超过 +1.0°；不允许跨种子平均掉根失败。

### TF、hold、noevents与扰动定位（overall辅助，不进入主表）

| arm | seed | raw root° | TF root° | 递推root° | raw finger RA-mm | TF RA-mm | hold RA-mm | noevents RA-mm | 10° retention k1/k2 |
|---|---|---|---|---|---|---|---|---|---|
| shared | 3407 | 26.9719 | 13.6051 | 25.9819 | 13.3705 | 20.9538 | 27.2844 | 27.2844 | 0.500346/0.250375 |
| shared | 3408 | 15.1038 | 7.7959 | 14.6906 | 13.9730 | 14.3818 | 27.2844 | 27.2844 | 0.500071/0.250069 |
| untied | 3407 | 33.5306 | 16.8580 | 31.4083 | 13.3182 | 26.3006 | 27.2844 | 27.2844 | 0.501063/0.251169 |
| untied | 3408 | 33.0217 | 16.6094 | 30.8468 | 18.3908 | 22.7949 | 27.2844 | 27.2844 | 0.500974/0.250958 |
| S38 | 3407 | 10.0288 | 5.3042 | 9.0309 | 9.1640 | 9.0278 | 27.2844 | 27.2844 | 0.500202/0.250189 |
| S38 | 3408 | 10.9898 | 5.7504 | 10.1286 | 9.6268 | 10.0515 | 27.2844 | 27.2844 | 0.500141/0.250193 |

TF仍经过gain0.5固定滤波；raw finger articulation按probe独立的root-relative FK合同计量，TF/递推RA并非纯finger误差。
hold与noevents对照检验空包保持。retention接近0.5、0.25主要反映固定滤波与prev-independent测量合同，不能当作根测量合格或新的动态恢复机制证据。

### 平移与非根守护

| 比较 | 判据 | 观察值 | 上限 | 结果 |
|---|---|---|---|---|
| sharing | s3407/recursive_RA_delta_mm | -8.257822 | 1.100 | 通过 |
| sharing | s3408/recursive_RA_delta_mm | -13.122763 | 1.100 | 通过 |
| sharing | mean/recursive_local_RA_delta_mm | -17.427701 | 1.100 | 通过 |
| sharing | mean/absolute_translation_ratio | 0.000365 | 1.100 | 通过 |
| shared_vs_S38 | mean/recursive_RA_delta_mm | 16.484134 | 1.100 | 失败 |
| shared_vs_S38 | mean/recursive_local_RA_delta_mm | 17.837945 | 1.100 | 失败 |
| shared_vs_S38 | mean/absolute_translation_ratio | 0.942711 | 1.100 | 通过 |
| untied_vs_S38 | mean/recursive_RA_delta_mm | 27.174427 | 1.100 | 失败 |
| untied_vs_S38 | mean/recursive_local_RA_delta_mm | 35.265646 | 1.100 | 失败 |
| untied_vs_S38 | mean/absolute_translation_ratio | 2585.440298 | 1.100 | 失败 |

T 比为候选/参照的两种子平均绝对平移误差比；不是旋转或 mesh 测量精度的替代指标。

| arm | seed | 递推T overall-mm | 递推T local-mm | TF T overall-mm | 递推绝对MPJPE overall-mm |
|---|---|---|---|---|---|
| shared | 3407 | 51.5152 | 55.3791 | 20.8481 | 62.0585 |
| shared | 3408 | 73.7505 | 94.9356 | 19.9872 | 85.2812 |
| untied | 3407 | 174731.0180 | 375818.1875 | 35.8837 | 174745.7180 |
| untied | 3408 | 168817.5425 | 363100.5312 | 25.2307 | 168833.4413 |
| S38 | 3407 | 55.8668 | 64.3344 | 45.4769 | 59.8000 |
| S38 | 3408 | 77.0114 | 92.2837 | 41.4801 | 82.7995 |

本6k untied/S38递推T比=2585.440298；shared/untied比=0.000364623。后者极小来自untied的递推绝对平移漂移放大了分母，不能解释为共享机制优越。上表同时保留TF、递推T及绝对MPJPE；RA按根对齐，不能显示整体平移漂移。
该T判据按原预注册规则保留，不因异常分母重算或修改gate。共享门通过仅表示本轮shared相对untied满足既定守护；两臂对S38的根与局部精度守护仍须独立判定，也不能由T比指定重参数化、归一化或容量中的单一失败原因。
TF T仍为几十mm而untied递推T剧增，提示T反馈映射的闭环失稳；这是TF/递推差异支持的定位线索，不是已证明的具体参数或归一化因果。
raw probe采用每个固定窗口起点的GT上下文，仅用于routing/T，无噪声或递推反馈；root测量本身已验证独立于prev。当前shared、untied在该上下文下仍未过S38 raw根测量守护，因此T反馈映射失稳迹象与raw根测量退化是并存的两类问题，应分别定位。

### 梯度方向与幅值（独立训练单 batch）

| phase | arm | seed | 参数组 | 坐标数 | norm T | norm root | norm finger | cos(T,R) | cos(T,F) | cos(R,F) |
|---|---|---|---|---|---|---|---|---|---|---|
| 6k | shared | 3407 | shared_head_with_LayerNorm | 348409 | 7.465942 | 0.786207 | 2.742404 | -0.006085 | 0.018702 | -0.128667 |
| 6k | shared | 3407 | shared_MLP_Linear_only | 337859 | 7.448592 | 0.786053 | 2.735715 | -0.006029 | 0.018759 | -0.128925 |
| 6k | shared | 3407 | identity_embeddings | 280 | 0.489863 | 0.004690 | 0.032589 | -0.135862 | 0.000000 | 0.000000 |
| 6k | shared | 3408 | shared_head_with_LayerNorm | 348409 | 5.496052 | 0.856056 | 2.363386 | -0.001681 | 0.025952 | -0.211373 |
| 6k | shared | 3408 | shared_MLP_Linear_only | 337859 | 5.489848 | 0.855188 | 2.355519 | -0.001913 | 0.025623 | -0.212025 |
| 6k | shared | 3408 | identity_embeddings | 280 | 0.945095 | 0.007486 | 0.025591 | -0.057372 | 0.000000 | 0.000000 |
| 6k | untied | 3407 | untied_bank_head_with_LayerNorm | 5922953 | 14.185231 | 0.629244 | 3.638563 | 0.000000 | 0.000000 | 0.000000 |
| 6k | untied | 3407 | untied_bank_MLP_Linear_only | 5743603 | 14.151391 | 0.627161 | 3.630631 | 0.000000 | 0.000000 | 0.000000 |
| 6k | untied | 3407 | identity_embeddings | 280 | 0.208643 | 0.000112 | 0.000565 | 0.081710 | 0.000000 | 0.000000 |
| 6k | untied | 3408 | untied_bank_head_with_LayerNorm | 5922953 | 12.419267 | 0.821814 | 3.099160 | 0.000000 | 0.000000 | 0.000000 |
| 6k | untied | 3408 | untied_bank_MLP_Linear_only | 5743603 | 12.403991 | 0.820080 | 3.091227 | 0.000000 | 0.000000 | 0.000000 |
| 6k | untied | 3408 | identity_embeddings | 280 | 0.185412 | 0.000166 | 0.000907 | -0.144717 | 0.000000 | 0.000000 |

每个向量是其加权任务项对 grad(log10(total)) 的贡献，使用同一 detached 1/(ln10·total)，不是分别对 log10(各项) 求导。
这是每个 checkpoint 的一个固定、增广训练 batch（16样本），不是总体梯度统计或因果证据；不同参数坐标数与输出尺度使 norm 不能直接代表训练质量。
untied bank 的任务坐标互不重叠，其零 cosine 属于结构定义；identity embedding 的耦合单独列出。

### 输出尺度（rad；T delta 为 m）

| phase | arm | seed | 上下文 | root分量RMS | root范数RMS | GT root范数RMS | finger分量RMS | GT finger分量RMS | finger范数RMS | T delta RMS |
|---|---|---|---|---|---|---|---|---|---|---|
| 6k | shared | 3407 | zgz全nonempty窗口 | 0.334152 | 0.578768 | 0.302598 | 0.390890 | 0.361150 | 0.677041 | - |
| 6k | shared | 3407 | 独立训练单batch | 0.162564 | 0.281569 | 0.347074 | 0.299482 | 0.331297 | 0.518718 | 0.024152 |
| 6k | shared | 3408 | zgz全nonempty窗口 | 0.115313 | 0.199728 | 0.302598 | 0.360931 | 0.361150 | 0.625151 | - |
| 6k | shared | 3408 | 独立训练单batch | 0.054913 | 0.095112 | 0.266230 | 0.297551 | 0.331297 | 0.515373 | 0.032588 |
| 6k | untied | 3407 | zgz全nonempty窗口 | 0.440824 | 0.763530 | 0.302598 | 0.432925 | 0.361150 | 0.749847 | - |
| 6k | untied | 3407 | 独立训练单batch | 0.196686 | 0.340670 | 0.347074 | 0.319498 | 0.331297 | 0.553386 | 0.023762 |
| 6k | untied | 3408 | zgz全nonempty窗口 | 0.482541 | 0.835785 | 0.302598 | 0.541984 | 0.361150 | 0.938745 | - |
| 6k | untied | 3408 | 独立训练单batch | 0.144317 | 0.249964 | 0.266230 | 0.315419 | 0.331297 | 0.546321 | 0.032252 |
| 6k | S38 | 3407 | zgz全nonempty窗口 | 0.177426 | 0.307311 | 0.302598 | 0.320463 | 0.361150 | 0.555058 | - |
| 6k | S38 | 3408 | zgz全nonempty窗口 | 0.170073 | 0.294575 | 0.302598 | 0.340609 | 0.361150 | 0.589953 | - |

root 是相对 R_ref 的旋转残差；finger 是 residual45，hands_mean 只由原 FK/decode 加一次。尺度差、norm 支持域、容量和 T 重参数化均是待隔离因素，不能据本表指定单一原因。

## 2k辅助诊断（保留原筛查失败）

### 根守护诊断（角度单位 °，正差表示候选更差）

global 指 zgz_global 序列；root 为相机坐标下的 MANO global orientation，不代表世界坐标系追踪。
raw 来自全部固定50ms窗口的 nonempty 测量 Exp(raw) R_ref；teacher-forced gain0.5输出不能替代 raw。

| 比较 | seed | 范围 | raw 候选 | raw 参照 | raw 差 | 递推候选 | 递推参照 | 递推差 | raw门 | 递推门 |
|---|---|---|---|---|---|---|---|---|---|---|
| shared/untied | 3407 | overall | 14.7287 | 31.1554 | -16.4267 | 14.3535 | 29.9140 | -15.5605 | 通过 | 通过 |
| shared/untied | 3407 | global | 14.2837 | 25.2452 | -10.9614 | 14.3044 | 24.7877 | -10.4833 | 通过 | 通过 |
| shared/untied | 3408 | overall | 16.7455 | 20.4770 | -3.7315 | 16.0704 | 18.6261 | -2.5557 | 通过 | 通过 |
| shared/untied | 3408 | global | 13.7873 | 13.8361 | -0.0488 | 13.5603 | 13.3532 | +0.2071 | 通过 | 通过 |
| shared/S38 | 3407 | overall | 14.7287 | 10.0288 | +4.6999 | 14.3535 | 9.0309 | +5.3227 | 失败 | 失败 |
| shared/S38 | 3407 | global | 14.2837 | 8.5687 | +5.7151 | 14.3044 | 8.2942 | +6.0103 | 失败 | 失败 |
| shared/S38 | 3408 | overall | 16.7455 | 10.9898 | +5.7557 | 16.0704 | 10.1286 | +5.9418 | 失败 | 失败 |
| shared/S38 | 3408 | global | 13.7873 | 8.7484 | +5.0390 | 13.5603 | 8.4003 | +5.1599 | 失败 | 失败 |
| untied/S38 | 3407 | overall | 31.1554 | 10.0288 | +21.1266 | 29.9140 | 9.0309 | +20.8831 | 失败 | 失败 |
| untied/S38 | 3407 | global | 25.2452 | 8.5687 | +16.6765 | 24.7877 | 8.2942 | +16.4935 | 失败 | 失败 |
| untied/S38 | 3408 | overall | 20.4770 | 10.9898 | +9.4872 | 18.6261 | 10.1286 | +8.4976 | 失败 | 失败 |
| untied/S38 | 3408 | global | 13.8361 | 8.7484 | +5.0878 | 13.3532 | 8.4003 | +4.9529 | 失败 | 失败 |

每个种子、overall/global 的 raw 与递推根差均须不超过 +1.0°；不允许跨种子平均掉根失败。

### TF、hold、noevents与扰动定位（overall辅助，不进入主表）

| arm | seed | raw root° | TF root° | 递推root° | raw finger RA-mm | TF RA-mm | hold RA-mm | noevents RA-mm | 10° retention k1/k2 |
|---|---|---|---|---|---|---|---|---|---|
| shared | 3407 | 14.7287 | 7.6216 | 14.3535 | 14.1641 | 14.5209 | 27.2844 | 27.2844 | 0.500060/0.250049 |
| shared | 3408 | 16.7455 | 8.5852 | 16.0704 | 13.0271 | 14.0866 | 27.2844 | 27.2844 | 0.500183/0.250184 |
| untied | 3407 | 31.1554 | 15.6661 | 29.9140 | 12.6893 | 26.5382 | 27.2844 | 27.2844 | 0.500524/0.250455 |
| untied | 3408 | 20.4770 | 10.3901 | 18.6261 | 13.4282 | 16.9344 | 27.2844 | 27.2844 | 0.500548/0.250538 |
| S38 | 3407 | 10.0288 | 5.3042 | 9.0309 | 9.1640 | 9.0278 | 27.2844 | 27.2844 | 0.500202/0.250189 |
| S38 | 3408 | 10.9898 | 5.7504 | 10.1286 | 9.6268 | 10.0515 | 27.2844 | 27.2844 | 0.500141/0.250193 |

TF仍经过gain0.5固定滤波；raw finger articulation按probe独立的root-relative FK合同计量，TF/递推RA并非纯finger误差。
hold与noevents对照检验空包保持。retention接近0.5、0.25主要反映固定滤波与prev-independent测量合同，不能当作根测量合格或新的动态恢复机制证据。

### 平移与非根守护

| 比较 | 判据 | 观察值 | 上限 | 结果 |
|---|---|---|---|---|
| sharing | s3407/recursive_RA_delta_mm | -23.064016 | 1.100 | 通过 |
| sharing | s3408/recursive_RA_delta_mm | -3.980801 | 1.100 | 通过 |
| sharing | mean/recursive_local_RA_delta_mm | -17.601677 | 1.100 | 通过 |
| sharing | mean/absolute_translation_ratio | 0.924230 | 1.100 | 通过 |
| shared_vs_S38 | mean/recursive_RA_delta_mm | 9.911403 | 1.100 | 失败 |
| shared_vs_S38 | mean/recursive_local_RA_delta_mm | 7.192730 | 1.100 | 失败 |
| shared_vs_S38 | mean/absolute_translation_ratio | 0.723796 | 1.100 | 通过 |
| untied_vs_S38 | mean/recursive_RA_delta_mm | 23.433811 | 1.100 | 失败 |
| untied_vs_S38 | mean/recursive_local_RA_delta_mm | 24.794407 | 1.100 | 失败 |
| untied_vs_S38 | mean/absolute_translation_ratio | 0.783134 | 1.100 | 通过 |

T 比为候选/参照的两种子平均绝对平移误差比；不是旋转或 mesh 测量精度的替代指标。

| arm | seed | 递推T overall-mm | 递推T local-mm | TF T overall-mm | 递推绝对MPJPE overall-mm |
|---|---|---|---|---|---|
| shared | 3407 | 53.9463 | 54.6581 | 18.6120 | 62.8268 |
| shared | 3408 | 42.2304 | 28.9359 | 17.9890 | 54.7250 |
| untied | 3407 | 54.6824 | 62.3962 | 19.1108 | 70.8362 |
| untied | 3408 | 49.3791 | 43.8442 | 16.9653 | 64.3757 |
| S38 | 3407 | 55.8668 | 64.3344 | 45.4769 | 59.8000 |
| S38 | 3408 | 77.0114 | 92.2837 | 41.4801 | 82.7995 |

### 梯度方向与幅值（独立训练单 batch）

| phase | arm | seed | 参数组 | 坐标数 | norm T | norm root | norm finger | cos(T,R) | cos(T,F) | cos(R,F) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | shared | 3407 | shared_head_with_LayerNorm | 348409 | 8.347072 | 0.506887 | 2.372779 | 0.000886 | 0.022327 | -0.148747 |
| 2k | shared | 3407 | shared_MLP_Linear_only | 337859 | 8.333277 | 0.506495 | 2.370868 | 0.000728 | 0.022504 | -0.149408 |
| 2k | shared | 3407 | identity_embeddings | 280 | 0.746725 | 0.001276 | 0.024748 | 0.251806 | 0.000000 | 0.000000 |
| 2k | shared | 3408 | shared_head_with_LayerNorm | 348409 | 7.561452 | 0.587134 | 1.776752 | -0.066965 | 0.193302 | -0.117528 |
| 2k | shared | 3408 | shared_MLP_Linear_only | 337859 | 7.559198 | 0.586807 | 1.775864 | -0.066916 | 0.193493 | -0.117678 |
| 2k | shared | 3408 | identity_embeddings | 280 | 0.859816 | 0.001510 | 0.012251 | -0.040071 | 0.000000 | 0.000000 |
| 2k | untied | 3407 | untied_bank_head_with_LayerNorm | 5922953 | 15.350127 | 0.487837 | 4.020991 | 0.000000 | 0.000000 | 0.000000 |
| 2k | untied | 3407 | untied_bank_MLP_Linear_only | 5743603 | 15.300003 | 0.487497 | 4.016423 | 0.000000 | 0.000000 | 0.000000 |
| 2k | untied | 3407 | identity_embeddings | 280 | 0.275850 | 0.000253 | 0.000820 | -0.454900 | 0.000000 | 0.000000 |
| 2k | untied | 3408 | untied_bank_head_with_LayerNorm | 5922953 | 12.629476 | 0.962298 | 3.220277 | 0.000000 | 0.000000 | 0.000000 |
| 2k | untied | 3408 | untied_bank_MLP_Linear_only | 5743603 | 12.621628 | 0.961442 | 3.216021 | 0.000000 | 0.000000 | 0.000000 |
| 2k | untied | 3408 | identity_embeddings | 280 | 0.106201 | 0.000271 | 0.001497 | -0.055536 | 0.000000 | 0.000000 |

每个向量是其加权任务项对 grad(log10(total)) 的贡献，使用同一 detached 1/(ln10·total)，不是分别对 log10(各项) 求导。
这是每个 checkpoint 的一个固定、增广训练 batch（16样本），不是总体梯度统计或因果证据；不同参数坐标数与输出尺度使 norm 不能直接代表训练质量。
untied bank 的任务坐标互不重叠，其零 cosine 属于结构定义；identity embedding 的耦合单独列出。

### 输出尺度（rad；T delta 为 m）

| phase | arm | seed | 上下文 | root分量RMS | root范数RMS | GT root范数RMS | finger分量RMS | GT finger分量RMS | finger范数RMS | T delta RMS |
|---|---|---|---|---|---|---|---|---|---|---|
| 2k | shared | 3407 | zgz全nonempty窗口 | 0.103975 | 0.180090 | 0.302598 | 0.407752 | 0.361150 | 0.706248 | - |
| 2k | shared | 3407 | 独立训练单batch | 0.092100 | 0.159522 | 0.347074 | 0.288554 | 0.331297 | 0.499789 | 0.023716 |
| 2k | shared | 3408 | zgz全nonempty窗口 | 0.203651 | 0.352733 | 0.302598 | 0.396557 | 0.361150 | 0.686858 | - |
| 2k | shared | 3408 | 独立训练单batch | 0.120467 | 0.208655 | 0.266230 | 0.286776 | 0.331297 | 0.496711 | 0.032495 |
| 2k | untied | 3407 | zgz全nonempty窗口 | 0.401347 | 0.695153 | 0.302598 | 0.402001 | 0.361150 | 0.696286 | - |
| 2k | untied | 3407 | 独立训练单batch | 0.190759 | 0.330404 | 0.347074 | 0.320181 | 0.331297 | 0.554570 | 0.023950 |
| 2k | untied | 3408 | zgz全nonempty窗口 | 0.296951 | 0.514333 | 0.302598 | 0.409310 | 0.361150 | 0.708945 | - |
| 2k | untied | 3408 | 独立训练单batch | 0.145292 | 0.251653 | 0.266230 | 0.314934 | 0.331297 | 0.545482 | 0.031782 |
| 2k | S38 | 3407 | zgz全nonempty窗口 | 0.177426 | 0.307311 | 0.302598 | 0.320463 | 0.361150 | 0.555058 | - |
| 2k | S38 | 3408 | zgz全nonempty窗口 | 0.170073 | 0.294575 | 0.302598 | 0.340609 | 0.361150 | 0.589953 | - |

root 是相对 R_ref 的旋转残差；finger 是 residual45，hands_mean 只由原 FK/decode 加一次。尺度差、norm 支持域、容量和 T 重参数化均是待隔离因素，不能据本表指定单一原因。

## 下一步范围

两臂均未过S38守护：先预注册三组单因素对照，每次只改一项，其余读出、训练和评估合同固定；各项通过后再整合恢复方案。

1. norm支持域：单独恢复S38读出输入的归一化支持范围，保持finger表达与T路径固定。
2. finger表达：单独比较hidden256联合45维读出与hidden64逐槽3维读出，保持norm支持域与T路径固定；这是一项容量/输出组织联合因素，不能再拆称已确认的纯宽度效应。
3. T闭环：保留同一R/F测量与滤波，单独比较原S38 routed T+prev_mlp与完整node ΔT反馈路径，保持norm支持域与finger表达固定。

分别定位GT上下文下的raw根/手指测量与递推T稳定性，再整合通过的恢复；不能一次恢复三因素后将改善归给其中一项。
这些是后续受控对照建议，不是已确认的padding-LN因果解释；当前失败也不足以直接否定共享机制。

两个种子、一个开发/测试受试者、冻结表征head probe仅支持本设置的工程判定；不能宣称最优架构、总体显著、真实异步执行、局部mesh更新或NeurIPS/TPAMI级创新。
本工具不修改AGENTS.md的当前臂；所有推荐均为后续待检验机制。

## 输入与SHA256

下表包含生成时实际读取的主表、gate全inputs及递归来源、状态、诊断、checkpoint/config/manifest/init、配对DDP指纹、debug、U0与代码来源。未读取selection文件。

| 路径 | 证据用途 | SHA256 |
|---|---|---|
| [/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json](/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json) | 2k/S38_sources/provenance/3407/manifest; 2k/S38_sources/provenance/3408/manifest; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/manifest; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/manifest; 2k/shared/row_sources/baseline_reference/provenance/3407/manifest; 2k/shared/row_sources/baseline_reference/provenance/3408/manifest; 2k/shared/row_sources/provenance/3407/manifest; 2k/shared/row_sources/provenance/3408/manifest; 2k/untied/row_sources/baseline_reference/provenance/3407/manifest; 2k/untied/row_sources/baseline_reference/provenance/3408/manifest; 2k/untied/row_sources/provenance/3407/manifest; 2k/untied/row_sources/provenance/3408/manifest; 6k/S38_sources/provenance/3407/manifest; 6k/S38_sources/provenance/3408/manifest; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/manifest; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/manifest; 6k/shared/row_sources/baseline_reference/provenance/3407/manifest; 6k/shared/row_sources/baseline_reference/provenance/3408/manifest; 6k/shared/row_sources/provenance/3407/manifest; 6k/shared/row_sources/provenance/3408/manifest; 6k/untied/row_sources/baseline_reference/provenance/3407/manifest; 6k/untied/row_sources/baseline_reference/provenance/3408/manifest; 6k/untied/row_sources/provenance/3407/manifest; 6k/untied/row_sources/provenance/3408/manifest; U0/3407/manifest; U0/3408/manifest | 2a4769f33ab86a02f6124ddf0096f445adaa4702e7798b2f73c01b1fb474b405 |
| [docs/U1A_SHARED_MEASUREMENT_PREREG.md](/data1/lyq/code/mesh/EventHands1/docs/U1A_SHARED_MEASUREMENT_PREREG.md) | U0/source_unchanged; code_or_prereg_source | a193dc919d21de9ebc0b5ad67d88b0865807c38a9aa4660d32afc0df465911c2 |
| [model/model.py](/data1/lyq/code/mesh/EventHands1/model/model.py) | U0/source_unchanged; code_or_prereg_source; isolation/active_source | 794dd39b1fa81f7e54f3ff077b27a3f6712f29a4835d36ca9bcb01f2bd186b91 |
| [outputs/semkine/rt_s37_3seed_tf_pert_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/rt_s37_3seed_tf_pert_main_row.json) | 2k/evaluation_sources/baseline_sources/s37_row; 2k/evaluation_sources/table/rows/rt_s37_3seed_tf_pert; 2k/table_row/rt_s37_3seed_tf_pert; 6k/evaluation_sources/baseline_sources/s37_row; 6k/evaluation_sources/table/rows/rt_s37_3seed_tf_pert; 6k/table_row/rt_s37_3seed_tf_pert; isolation/active_read_only_historical_target | 0afdf6f1b8117fe5b5bc144d277a4a263c5c4a3c4f863efdb8ae2750c1e75e99 |
| [outputs/semkine/s38_spmeas_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/config_resolved.yaml) | 2k/S38_sources/provenance/3407/config; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/config; 2k/shared/row_sources/baseline_reference/provenance/3407/config; 2k/untied/row_sources/baseline_reference/provenance/3407/config; 6k/S38_sources/provenance/3407/config; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/config; 6k/shared/row_sources/baseline_reference/provenance/3407/config; 6k/untied/row_sources/baseline_reference/provenance/3407/config; U0/3407/config | c45f4116d651bb69ac32dd99ebec723ca0f06676c14f5e5f890e1a8eb2c90dae |
| [outputs/semkine/s38_spmeas_s3407/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/evalx_val_core_last_tf_pert.json) | 2k/S38_sources/provenance/3407/evaluation; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/evaluation; 2k/shared/row_sources/baseline_reference/provenance/3407/evaluation; 2k/untied/row_sources/baseline_reference/provenance/3407/evaluation; 6k/S38_sources/provenance/3407/evaluation; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/evaluation; 6k/shared/row_sources/baseline_reference/provenance/3407/evaluation; 6k/untied/row_sources/baseline_reference/provenance/3407/evaluation | 2498b62b547cc127a63d67973a82e4b5e277ad78857e78d2a7bae72fe5640705 |
| [outputs/semkine/s38_spmeas_s3407/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/evalx_val_core_last_tf_pert.npz) | 2k/S38_sources/provenance/3407/diagnostic_arrays; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/diagnostic_arrays; 2k/shared/row_sources/baseline_reference/provenance/3407/diagnostic_arrays; 2k/untied/row_sources/baseline_reference/provenance/3407/diagnostic_arrays; 6k/S38_sources/provenance/3407/diagnostic_arrays; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/diagnostic_arrays; 6k/shared/row_sources/baseline_reference/provenance/3407/diagnostic_arrays; 6k/untied/row_sources/baseline_reference/provenance/3407/diagnostic_arrays | 66d974a6b591c28de8c28dfa3317e98af20c13389bd52def4a7638e28e29e18b |
| [outputs/semkine/s38_spmeas_s3407/last.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/last.ckpt) | 2k/shared/row_sources/provenance/3407/frozen_s38_source; 2k/untied/row_sources/provenance/3407/frozen_s38_source; 6k/shared/row_sources/provenance/3407/frozen_s38_source; 6k/untied/row_sources/provenance/3407/frozen_s38_source | 1e4b923ca37da968bd379d4505e1ebb3fcebe6841a4b681aa6edf3f1392c6b44 |
| [outputs/semkine/s38_spmeas_s3407/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/probe_raw_last.json) | 2k/S38/3407/probe; 2k/gate_inputs/8; 2k/raw_sources/S38/3407; 6k/S38/3407/probe; 6k/gate_inputs/8; 6k/raw_sources/S38/3407 | 6ebb5421db09fae1665f12b789972361e89ccb9196884f8b2029bae03bdeb322 |
| [outputs/semkine/s38_spmeas_s3407/s38_spmeas_s3407-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/s38_spmeas_s3407-step=6000.ckpt) | 2k/S38_sources/provenance/3407/checkpoint; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/checkpoint; 2k/shared/row_sources/baseline_reference/provenance/3407/checkpoint; 2k/untied/row_sources/baseline_reference/provenance/3407/checkpoint; 6k/S38_sources/provenance/3407/checkpoint; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/checkpoint; 6k/shared/row_sources/baseline_reference/provenance/3407/checkpoint; 6k/untied/row_sources/baseline_reference/provenance/3407/checkpoint; U0/3407/checkpoint | 1e4b923ca37da968bd379d4505e1ebb3fcebe6841a4b681aa6edf3f1392c6b44 |
| [outputs/semkine/s38_spmeas_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3407/training_metadata.json) | 2k/S38_sources/provenance/3407/training_metadata; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/training_metadata; 2k/shared/row_sources/baseline_reference/provenance/3407/training_metadata; 2k/untied/row_sources/baseline_reference/provenance/3407/training_metadata; 6k/S38_sources/provenance/3407/training_metadata; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3407/training_metadata; 6k/shared/row_sources/baseline_reference/provenance/3407/training_metadata; 6k/untied/row_sources/baseline_reference/provenance/3407/training_metadata | d8e46b7c2ed388424bb5f0b6c415e5f6d14cc344df33b465d5eec30150ec9bc4 |
| [outputs/semkine/s38_spmeas_s3408/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/config_resolved.yaml) | 2k/S38_sources/provenance/3408/config; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/config; 2k/shared/row_sources/baseline_reference/provenance/3408/config; 2k/untied/row_sources/baseline_reference/provenance/3408/config; 6k/S38_sources/provenance/3408/config; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/config; 6k/shared/row_sources/baseline_reference/provenance/3408/config; 6k/untied/row_sources/baseline_reference/provenance/3408/config; U0/3408/config | 01cc569eb6d669c0707d163ed12f858d73257b0c91c3e12018506df9ef78fbd8 |
| [outputs/semkine/s38_spmeas_s3408/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/evalx_val_core_last_tf_pert.json) | 2k/S38_sources/provenance/3408/evaluation; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/evaluation; 2k/shared/row_sources/baseline_reference/provenance/3408/evaluation; 2k/untied/row_sources/baseline_reference/provenance/3408/evaluation; 6k/S38_sources/provenance/3408/evaluation; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/evaluation; 6k/shared/row_sources/baseline_reference/provenance/3408/evaluation; 6k/untied/row_sources/baseline_reference/provenance/3408/evaluation | f3077f9578e3159b1d6984ea5ed46fad125c559e307242b7fd8a3b8e09ef8a9d |
| [outputs/semkine/s38_spmeas_s3408/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/evalx_val_core_last_tf_pert.npz) | 2k/S38_sources/provenance/3408/diagnostic_arrays; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/diagnostic_arrays; 2k/shared/row_sources/baseline_reference/provenance/3408/diagnostic_arrays; 2k/untied/row_sources/baseline_reference/provenance/3408/diagnostic_arrays; 6k/S38_sources/provenance/3408/diagnostic_arrays; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/diagnostic_arrays; 6k/shared/row_sources/baseline_reference/provenance/3408/diagnostic_arrays; 6k/untied/row_sources/baseline_reference/provenance/3408/diagnostic_arrays | aba87c0e8a2c8bbcb9986223477636c9c3fe4969f287ab109161ef14ed1900c7 |
| [outputs/semkine/s38_spmeas_s3408/last.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/last.ckpt) | 2k/shared/row_sources/provenance/3408/frozen_s38_source; 2k/untied/row_sources/provenance/3408/frozen_s38_source; 6k/shared/row_sources/provenance/3408/frozen_s38_source; 6k/untied/row_sources/provenance/3408/frozen_s38_source | cbcbc8fa54146fbb63ac333964b1f7d6597af34d8be2e22d5ad68c8b7d3062a0 |
| [outputs/semkine/s38_spmeas_s3408/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/probe_raw_last.json) | 2k/S38/3408/probe; 2k/gate_inputs/9; 2k/raw_sources/S38/3408; 6k/S38/3408/probe; 6k/gate_inputs/9; 6k/raw_sources/S38/3408 | b3e2a27ade8368293703256ab20f090a343a5315cc91f14606ef55b08343a488 |
| [outputs/semkine/s38_spmeas_s3408/s38_spmeas_s3408-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/s38_spmeas_s3408-step=6000.ckpt) | 2k/S38_sources/provenance/3408/checkpoint; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/checkpoint; 2k/shared/row_sources/baseline_reference/provenance/3408/checkpoint; 2k/untied/row_sources/baseline_reference/provenance/3408/checkpoint; 6k/S38_sources/provenance/3408/checkpoint; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/checkpoint; 6k/shared/row_sources/baseline_reference/provenance/3408/checkpoint; 6k/untied/row_sources/baseline_reference/provenance/3408/checkpoint; U0/3408/checkpoint | cbcbc8fa54146fbb63ac333964b1f7d6597af34d8be2e22d5ad68c8b7d3062a0 |
| [outputs/semkine/s38_spmeas_s3408/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_s3408/training_metadata.json) | 2k/S38_sources/provenance/3408/training_metadata; 2k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/training_metadata; 2k/shared/row_sources/baseline_reference/provenance/3408/training_metadata; 2k/untied/row_sources/baseline_reference/provenance/3408/training_metadata; 6k/S38_sources/provenance/3408/training_metadata; 6k/evaluation_sources/baseline_sources/s38_reference_per_seed/3408/training_metadata; 6k/shared/row_sources/baseline_reference/provenance/3408/training_metadata; 6k/untied/row_sources/baseline_reference/provenance/3408/training_metadata | 23e69f9ca5e6833d83c54e3688dafcf6513943006569a76140855ebdb197254f |
| [outputs/semkine/s38_spmeas_tf_pert_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_spmeas_tf_pert_main_row.json) | 2k/S38_sources/cost_provenance; 2k/evaluation_sources/baseline_sources/s38_cost_row; 6k/S38_sources/cost_provenance; 6k/evaluation_sources/baseline_sources/s38_cost_row; isolation/active_read_only_historical_target | d82bfedf76f0bbecfaef0a43bae52e36ba8e736641aca59208d2106165283871 |
| [outputs/semkine/s38_u1a_fixed_reference_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/s38_u1a_fixed_reference_main_row.json) | 2k/evaluation_sources/fixed_s38_reference; 2k/evaluation_sources/table/rows/s38_u1a_fixed_reference; 2k/table_row/s38_u1a_fixed_reference; 6k/evaluation_sources/fixed_s38_reference; 6k/evaluation_sources/table/rows/s38_u1a_fixed_reference; 6k/table_row/s38_u1a_fixed_reference | a9fe3b3640f63bf061983299790ce7d860c4a5f4b737b9dbcd7f459b04ccf187 |
| [outputs/semkine/u1a_shared_2k_extended.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_extended.json) | 2k/gate_inputs/1; 2k/shared/extended; isolation/active_read_only_historical_target | a45904022786e7be6304df78565eb567ddee783aa783d86fcf274fec9c2df893 |
| [outputs/semkine/u1a_shared_2k_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_main_row.json) | 2k/evaluation_sources/table/rows/u1a_shared_2k; 2k/gate_inputs/0; 2k/shared/row; 2k/table_row/u1a_shared_2k; isolation/active_read_only_historical_target | 1b614db25ebd1b1f81bde8ccfc49b82c9f00e6557c5aeae03f7d74cd324ab4f5 |
| [outputs/semkine/u1a_shared_2k_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/config_resolved.yaml) | 2k/shared/3407/config; 2k/shared/row_sources/cost_provenance/config; 2k/shared/row_sources/provenance/3407/config | 1675e49998ee410f3c64fdfc151d49f9757fe293160c169a8b5076d2a06439ca |
| [outputs/semkine/u1a_shared_2k_s3407/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/evalx_val_core_last_tf_pert.json) | 2k/shared/row_sources/provenance/3407/evaluation | 1660895846d2b2a23bf5b93f8bfd422107b953e911f1bfe9b66d01f7d0a6d676 |
| [outputs/semkine/u1a_shared_2k_s3407/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/evalx_val_core_last_tf_pert.npz) | 2k/shared/row_sources/provenance/3407/diagnostic_arrays | 03b599a9c119ebec53e702d15625c3e8a5045221c602fb425cfb711b8095e703 |
| [outputs/semkine/u1a_shared_2k_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/pair_stream_rank0.json) | 2k/shared/3407/rank0_stream | 08ee164d9247f6780e217d8b34edd6a2f957dd94f4a36fc8c7422190c972569c |
| [outputs/semkine/u1a_shared_2k_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/pair_stream_rank1.json) | 2k/shared/3407/rank1_stream | 16baf16737a9f11a739ec869a4095a990eb589afb6b1be9ecfe6373c0b49468e |
| [outputs/semkine/u1a_shared_2k_s3407/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/probe_raw_last.json) | 2k/gate_inputs/4; 2k/raw_sources/shared/3407; 2k/shared/3407/probe | f447e8deef56ce1852e5a477b827ce756e3b784a505ad4f02fc88fdc07db4321 |
| [outputs/semkine/u1a_shared_2k_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/training_metadata.json) | 2k/shared/3407/metadata; 2k/shared/row_sources/provenance/3407/training_metadata | 51f743d9176e78054a3a6410a6b00a4a5f8ea16c3effa0053930bcb9f14e441e |
| [outputs/semkine/u1a_shared_2k_s3407/u1a_shared_2k_s3407-step=2000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3407/u1a_shared_2k_s3407-step=2000.ckpt) | 2k/shared/row_sources/cost_provenance/checkpoint; 2k/shared/row_sources/provenance/3407/checkpoint | 2ef69f3ee41b50600f8c9fb5b471c54bb76d4ee24c75545fe99a32986507a2f1 |
| [outputs/semkine/u1a_shared_2k_s3408/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/config_resolved.yaml) | 2k/shared/3408/config; 2k/shared/row_sources/provenance/3408/config | 00d8b1ea9edf9f25db3392a398e507c864afb0b0a2f4e6b70bfcb161d3e640f8 |
| [outputs/semkine/u1a_shared_2k_s3408/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/evalx_val_core_last_tf_pert.json) | 2k/shared/row_sources/provenance/3408/evaluation | c8c93b12f248f211b408fe4ef663ddaf96637c71fb127c07c8f2461049d8e888 |
| [outputs/semkine/u1a_shared_2k_s3408/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/evalx_val_core_last_tf_pert.npz) | 2k/shared/row_sources/provenance/3408/diagnostic_arrays | 06a7ca7bcc8d74f0fb301211a8282ecc4df6daa8bb31daa1f7513ddb1334710e |
| [outputs/semkine/u1a_shared_2k_s3408/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/pair_stream_rank0.json) | 2k/shared/3408/rank0_stream | eb7d94a4c1c96708d49c6043ea63b26cdfcabd67115f6662223da8145d01b30c |
| [outputs/semkine/u1a_shared_2k_s3408/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/pair_stream_rank1.json) | 2k/shared/3408/rank1_stream | ab5c79a48c4eb5a13e4f205f3da44336218d0747afbb93d7dd4a65293aef6af3 |
| [outputs/semkine/u1a_shared_2k_s3408/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/probe_raw_last.json) | 2k/gate_inputs/5; 2k/raw_sources/shared/3408; 2k/shared/3408/probe | 552a422472150a40db9e8b92217ee4e88a98fdfcf6a76021ed36f262f501bd77 |
| [outputs/semkine/u1a_shared_2k_s3408/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/training_metadata.json) | 2k/shared/3408/metadata; 2k/shared/row_sources/provenance/3408/training_metadata | f99b53a27b618c8925e3736d581c37d9069587505ae81ea2723ee55c6a826285 |
| [outputs/semkine/u1a_shared_2k_s3408/u1a_shared_2k_s3408-step=2000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_2k_s3408/u1a_shared_2k_s3408-step=2000.ckpt) | 2k/shared/row_sources/provenance/3408/checkpoint | 8a9264800b90c1b0c731c8848f607917363a8478968ee878116e91d580023bc4 |
| [outputs/semkine/u1a_shared_6k_extended.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_extended.json) | 6k/gate_inputs/1; 6k/shared/extended | fb62b44406ddf7445fc4d74920f2ee08dbf784315d86543e582d26070b75693b |
| [outputs/semkine/u1a_shared_6k_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_main_row.json) | 6k/evaluation_sources/table/rows/u1a_shared_6k; 6k/gate_inputs/0; 6k/shared/row; 6k/table_row/u1a_shared_6k | 55b2231c1d55716efccc38e13d8ba113910ef93e755396c32404616b9b426f45 |
| [outputs/semkine/u1a_shared_6k_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/config_resolved.yaml) | 6k/shared/3407/config; 6k/shared/row_sources/cost_provenance/config; 6k/shared/row_sources/provenance/3407/config | 379529762ca430115675ed86afca400cfd85e52e80f69d571b33bccf8df3908f |
| [outputs/semkine/u1a_shared_6k_s3407/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/evalx_val_core_last_tf_pert.json) | 6k/shared/row_sources/provenance/3407/evaluation | 6470a6d77d4ffa36552186dba3742048d14590ed35ec87ebf6ca184e5d958040 |
| [outputs/semkine/u1a_shared_6k_s3407/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/evalx_val_core_last_tf_pert.npz) | 6k/shared/row_sources/provenance/3407/diagnostic_arrays | b3c3e4560a4d2fc4014841544f9d1d00b97cbe3673028d14ed845e8ac80367cf |
| [outputs/semkine/u1a_shared_6k_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/pair_stream_rank0.json) | 6k/shared/3407/rank0_stream | 08ee164d9247f6780e217d8b34edd6a2f957dd94f4a36fc8c7422190c972569c |
| [outputs/semkine/u1a_shared_6k_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/pair_stream_rank1.json) | 6k/shared/3407/rank1_stream | 16baf16737a9f11a739ec869a4095a990eb589afb6b1be9ecfe6373c0b49468e |
| [outputs/semkine/u1a_shared_6k_s3407/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/probe_raw_last.json) | 6k/gate_inputs/4; 6k/raw_sources/shared/3407; 6k/shared/3407/probe | 55f4710b63903d38d2f323a90b4772526ceae6fd29c443f1b9040bd3022a5bd5 |
| [outputs/semkine/u1a_shared_6k_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/training_metadata.json) | 6k/shared/3407/metadata; 6k/shared/row_sources/provenance/3407/training_metadata | 89084f48ecd3a2e7c424c55001cc353498988765399be76a0eb78ec44acae4eb |
| [outputs/semkine/u1a_shared_6k_s3407/u1a_shared_6k_s3407-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3407/u1a_shared_6k_s3407-step=6000.ckpt) | 6k/shared/row_sources/cost_provenance/checkpoint; 6k/shared/row_sources/provenance/3407/checkpoint | 05d9916acfbddd4d93606ad93a6ac1e25f40dcf23abe9f0c732edfd3dac163b8 |
| [outputs/semkine/u1a_shared_6k_s3408/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/config_resolved.yaml) | 6k/shared/3408/config; 6k/shared/row_sources/provenance/3408/config | d7bd0fec15c9204ae511fc13566e7a954cd6e1974f20dac536c3881cc3572de4 |
| [outputs/semkine/u1a_shared_6k_s3408/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/evalx_val_core_last_tf_pert.json) | 6k/shared/row_sources/provenance/3408/evaluation | 36eff09e5f6cc0ca6a28ffe2bb646f53aee9c385a2ee36adbb20f9b939c4506d |
| [outputs/semkine/u1a_shared_6k_s3408/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/evalx_val_core_last_tf_pert.npz) | 6k/shared/row_sources/provenance/3408/diagnostic_arrays | 03ec60f43cadb2bb3cdc58eb6ee4f0daacbd43f8a23d15ab101e107460049232 |
| [outputs/semkine/u1a_shared_6k_s3408/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/pair_stream_rank0.json) | 6k/shared/3408/rank0_stream | eb7d94a4c1c96708d49c6043ea63b26cdfcabd67115f6662223da8145d01b30c |
| [outputs/semkine/u1a_shared_6k_s3408/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/pair_stream_rank1.json) | 6k/shared/3408/rank1_stream | ab5c79a48c4eb5a13e4f205f3da44336218d0747afbb93d7dd4a65293aef6af3 |
| [outputs/semkine/u1a_shared_6k_s3408/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/probe_raw_last.json) | 6k/gate_inputs/5; 6k/raw_sources/shared/3408; 6k/shared/3408/probe | b9a0f1921462b0b1690ff716e16620c9eb84e9c1eb296776acc2a2555245b94b |
| [outputs/semkine/u1a_shared_6k_s3408/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/training_metadata.json) | 6k/shared/3408/metadata; 6k/shared/row_sources/provenance/3408/training_metadata | 54f7460eacad342d155ebf582b7aa4fa53ad40c02342231a0ed7eeb9e73db05a |
| [outputs/semkine/u1a_shared_6k_s3408/u1a_shared_6k_s3408-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_6k_s3408/u1a_shared_6k_s3408-step=6000.ckpt) | 6k/shared/row_sources/provenance/3408/checkpoint | 6c6125664d4e8e666b74018fbfd1e6f27d75eeba4ed59db9c7d98b6d7c20812a |
| [outputs/semkine/u1a_shared_debug_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_debug_s3407/config_resolved.yaml) | debug/shared/config | e9ed1ee7a444d576f58a37aaa109e6e034136194e535523a080a997174a19d10 |
| [outputs/semkine/u1a_shared_debug_s3407/last.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_debug_s3407/last.ckpt) | debug/shared/last_checkpoint | 0391a87b91b8284b1b195a9f19523d96d92e93e864de79c8539dcdb9c400954f |
| [outputs/semkine/u1a_shared_debug_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_debug_s3407/pair_stream_rank0.json) | debug/shared/stream | 05015cf70e147736eb28916faa52e1bed39b6a93fb42358c0c0d926f379bef86 |
| [outputs/semkine/u1a_shared_debug_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_debug_s3407/pair_stream_rank1.json) | debug/shared/stream | fc58bcf74db18a488dad13484064b3eb9cdcd910ce525fd3a856ef1b5b31d7c4 |
| [outputs/semkine/u1a_shared_debug_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_shared_debug_s3407/training_metadata.json) | debug/shared/actual_metadata | 05f20f0758f8117f7d26cb0644643c36ad352f9334d6f1e6d11c813cb3b25644 |
| [outputs/semkine/u1a_untied_2k_extended.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_extended.json) | 2k/gate_inputs/3; 2k/untied/extended; isolation/active_read_only_historical_target | 151d733e9dfdce50bfb67f01643a91a3f63a2b66c4b4c2b456358f35a07e452f |
| [outputs/semkine/u1a_untied_2k_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_main_row.json) | 2k/evaluation_sources/table/rows/u1a_untied_2k; 2k/gate_inputs/2; 2k/table_row/u1a_untied_2k; 2k/untied/row; isolation/active_read_only_historical_target | 55f7e7d76f83c7a2ed90179578ad16a2abb6ab93fbf28d34535867c206d9d4e3 |
| [outputs/semkine/u1a_untied_2k_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/config_resolved.yaml) | 2k/untied/3407/config; 2k/untied/row_sources/cost_provenance/config; 2k/untied/row_sources/provenance/3407/config | db7166a88f24641e029297195202dc69a5a46b198b8fe3b21d4be91ef47ff413 |
| [outputs/semkine/u1a_untied_2k_s3407/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/evalx_val_core_last_tf_pert.json) | 2k/untied/row_sources/provenance/3407/evaluation | 3a7bec914dae5606b86b53ca1c443cef3e18ab360b0612483eb143ed2db1620a |
| [outputs/semkine/u1a_untied_2k_s3407/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/evalx_val_core_last_tf_pert.npz) | 2k/untied/row_sources/provenance/3407/diagnostic_arrays | 805f2a6e347fa9bbda92cb954197a588bdb32df9ab796b3f0f4103b657965c66 |
| [outputs/semkine/u1a_untied_2k_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/pair_stream_rank0.json) | 2k/untied/3407/rank0_stream | 08ee164d9247f6780e217d8b34edd6a2f957dd94f4a36fc8c7422190c972569c |
| [outputs/semkine/u1a_untied_2k_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/pair_stream_rank1.json) | 2k/untied/3407/rank1_stream | 16baf16737a9f11a739ec869a4095a990eb589afb6b1be9ecfe6373c0b49468e |
| [outputs/semkine/u1a_untied_2k_s3407/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/probe_raw_last.json) | 2k/gate_inputs/6; 2k/raw_sources/untied/3407; 2k/untied/3407/probe | ba5e7daebb16b0f6848a94bf37d340aada76b352c3a177db317efda1bc897d08 |
| [outputs/semkine/u1a_untied_2k_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/training_metadata.json) | 2k/untied/3407/metadata; 2k/untied/row_sources/provenance/3407/training_metadata | dce7fcd458ebcd0eb052720c7e28429962cd0de0e450b953576938832dc412ef |
| [outputs/semkine/u1a_untied_2k_s3407/u1a_untied_2k_s3407-step=2000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3407/u1a_untied_2k_s3407-step=2000.ckpt) | 2k/untied/row_sources/cost_provenance/checkpoint; 2k/untied/row_sources/provenance/3407/checkpoint | 208839262034513cf33eb83ad8a4da13b543efd14db5d4f5f7367a574a2f825b |
| [outputs/semkine/u1a_untied_2k_s3408/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/config_resolved.yaml) | 2k/untied/3408/config; 2k/untied/row_sources/provenance/3408/config | e33c766cabfa3e5f88171f114af7e6f756cf2453c1b034a47a4178f9f68cff1b |
| [outputs/semkine/u1a_untied_2k_s3408/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/evalx_val_core_last_tf_pert.json) | 2k/untied/row_sources/provenance/3408/evaluation | 4df5c07db75c0ce0755ee492bece9a567c97e1a18389e9db1b1ce9bf88af466a |
| [outputs/semkine/u1a_untied_2k_s3408/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/evalx_val_core_last_tf_pert.npz) | 2k/untied/row_sources/provenance/3408/diagnostic_arrays | d97bfdf3b8bd7e1e50d0400ee978aff39c8098ea6de7736df2e13f6580e66bd7 |
| [outputs/semkine/u1a_untied_2k_s3408/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/pair_stream_rank0.json) | 2k/untied/3408/rank0_stream | eb7d94a4c1c96708d49c6043ea63b26cdfcabd67115f6662223da8145d01b30c |
| [outputs/semkine/u1a_untied_2k_s3408/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/pair_stream_rank1.json) | 2k/untied/3408/rank1_stream | ab5c79a48c4eb5a13e4f205f3da44336218d0747afbb93d7dd4a65293aef6af3 |
| [outputs/semkine/u1a_untied_2k_s3408/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/probe_raw_last.json) | 2k/gate_inputs/7; 2k/raw_sources/untied/3408; 2k/untied/3408/probe | 79d0e9d1765668e25fb4f3c9040aa4459d50271a16e5ff545713b4433f558896 |
| [outputs/semkine/u1a_untied_2k_s3408/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/training_metadata.json) | 2k/untied/3408/metadata; 2k/untied/row_sources/provenance/3408/training_metadata | 7a724342af2700682d3300383f22824bbeb9d1dc07142fed7c5e2a358c05f66a |
| [outputs/semkine/u1a_untied_2k_s3408/u1a_untied_2k_s3408-step=2000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_2k_s3408/u1a_untied_2k_s3408-step=2000.ckpt) | 2k/untied/row_sources/provenance/3408/checkpoint | b1e44fc821f61c0cfcaacdbf0b9a3fa0588e0d3c1768891ca4b29993b51b6609 |
| [outputs/semkine/u1a_untied_6k_extended.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_extended.json) | 6k/gate_inputs/3; 6k/untied/extended | 855f918804998c693949c0a285c0cfe74db18ccee433becad478f99f6dc8a8c3 |
| [outputs/semkine/u1a_untied_6k_main_row.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_main_row.json) | 6k/evaluation_sources/table/rows/u1a_untied_6k; 6k/gate_inputs/2; 6k/table_row/u1a_untied_6k; 6k/untied/row | 44863cf526c1a26a09fd98dc60aa8111e0cbaca49062e84691afb78955f1e151 |
| [outputs/semkine/u1a_untied_6k_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/config_resolved.yaml) | 6k/untied/3407/config; 6k/untied/row_sources/cost_provenance/config; 6k/untied/row_sources/provenance/3407/config | 4a06af336884520d605514e35f6377a5b363acb1544921574d644b1151079c10 |
| [outputs/semkine/u1a_untied_6k_s3407/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/evalx_val_core_last_tf_pert.json) | 6k/untied/row_sources/provenance/3407/evaluation | 7c73230c9d5ac67daa55f931e14a19f252cfcd192f3ba370069cb770b85b45f8 |
| [outputs/semkine/u1a_untied_6k_s3407/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/evalx_val_core_last_tf_pert.npz) | 6k/untied/row_sources/provenance/3407/diagnostic_arrays | 38ac8ae6a94a80e38776f8244e07f015e0962280f31d574f46df2ad46f21ed34 |
| [outputs/semkine/u1a_untied_6k_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/pair_stream_rank0.json) | 6k/untied/3407/rank0_stream | 08ee164d9247f6780e217d8b34edd6a2f957dd94f4a36fc8c7422190c972569c |
| [outputs/semkine/u1a_untied_6k_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/pair_stream_rank1.json) | 6k/untied/3407/rank1_stream | 16baf16737a9f11a739ec869a4095a990eb589afb6b1be9ecfe6373c0b49468e |
| [outputs/semkine/u1a_untied_6k_s3407/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/probe_raw_last.json) | 6k/gate_inputs/6; 6k/raw_sources/untied/3407; 6k/untied/3407/probe | 24c6f9bc685735071f8a6ff6b00f7fcab4e5f63bcb59b4e3b3dbc7d0f7fdcb8e |
| [outputs/semkine/u1a_untied_6k_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/training_metadata.json) | 6k/untied/3407/metadata; 6k/untied/row_sources/provenance/3407/training_metadata | f9d0c09bbee5f5cb6e458c92c22b5864564eaedc9174ea639498972809d925d9 |
| [outputs/semkine/u1a_untied_6k_s3407/u1a_untied_6k_s3407-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3407/u1a_untied_6k_s3407-step=6000.ckpt) | 6k/untied/row_sources/cost_provenance/checkpoint; 6k/untied/row_sources/provenance/3407/checkpoint | 96987981a5bfafbf143e4407f732576b85e5618a4cc7e77987291b7fcbc9b34c |
| [outputs/semkine/u1a_untied_6k_s3408/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/config_resolved.yaml) | 6k/untied/3408/config; 6k/untied/row_sources/provenance/3408/config | 65b2c97a815997a49d17fb77530f8c6756a0312efada3b1582f14f7d874345e0 |
| [outputs/semkine/u1a_untied_6k_s3408/evalx_val_core_last_tf_pert.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/evalx_val_core_last_tf_pert.json) | 6k/untied/row_sources/provenance/3408/evaluation | addf0b663bb6540153a01a1dee92f45ee2fc749430547124d02e8fdac66d7b33 |
| [outputs/semkine/u1a_untied_6k_s3408/evalx_val_core_last_tf_pert.npz](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/evalx_val_core_last_tf_pert.npz) | 6k/untied/row_sources/provenance/3408/diagnostic_arrays | b9ccc7d6acd2a84c1b8b55b9da96919419fe16d6edb17fd4a7b514d53cbdb52b |
| [outputs/semkine/u1a_untied_6k_s3408/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/pair_stream_rank0.json) | 6k/untied/3408/rank0_stream | eb7d94a4c1c96708d49c6043ea63b26cdfcabd67115f6662223da8145d01b30c |
| [outputs/semkine/u1a_untied_6k_s3408/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/pair_stream_rank1.json) | 6k/untied/3408/rank1_stream | ab5c79a48c4eb5a13e4f205f3da44336218d0747afbb93d7dd4a65293aef6af3 |
| [outputs/semkine/u1a_untied_6k_s3408/probe_raw_last.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/probe_raw_last.json) | 6k/gate_inputs/7; 6k/raw_sources/untied/3408; 6k/untied/3408/probe | 318e0f6877cf0a912578b3d1798243d5de0d802bcaaf73a67dae26c2ef627e71 |
| [outputs/semkine/u1a_untied_6k_s3408/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/training_metadata.json) | 6k/untied/3408/metadata; 6k/untied/row_sources/provenance/3408/training_metadata | d7eb3fff453322de0112179a6cdbacee3f2ef4a05918bfaedaeca63afb3beeb7 |
| [outputs/semkine/u1a_untied_6k_s3408/u1a_untied_6k_s3408-step=6000.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_6k_s3408/u1a_untied_6k_s3408-step=6000.ckpt) | 6k/untied/row_sources/provenance/3408/checkpoint | 311c2895f9385e7bb40bbaaadb6c9e9e96360c528fbf2c9a3cf987ff91918cdf |
| [outputs/semkine/u1a_untied_debug_s3407/config_resolved.yaml](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_debug_s3407/config_resolved.yaml) | debug/untied/config | ad3a141b0d48d98c483503054f3743b18da84723757d5d95c9ef49942539519a |
| [outputs/semkine/u1a_untied_debug_s3407/last.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_debug_s3407/last.ckpt) | debug/untied/last_checkpoint | 8971daf3d0870a71245eb909f732bb9a39d2121614126639ddf870cf3a4cc5b2 |
| [outputs/semkine/u1a_untied_debug_s3407/pair_stream_rank0.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_debug_s3407/pair_stream_rank0.json) | debug/untied/stream | 05015cf70e147736eb28916faa52e1bed39b6a93fb42358c0c0d926f379bef86 |
| [outputs/semkine/u1a_untied_debug_s3407/pair_stream_rank1.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_debug_s3407/pair_stream_rank1.json) | debug/untied/stream | fc58bcf74db18a488dad13484064b3eb9cdcd910ce525fd3a856ef1b5b31d7c4 |
| [outputs/semkine/u1a_untied_debug_s3407/training_metadata.json](/data1/lyq/code/mesh/EventHands1/outputs/semkine/u1a_untied_debug_s3407/training_metadata.json) | debug/untied/actual_metadata | 3be9dc47b530b4f02704681dd85f0637efd4c642cc8b2adb5ab946eafa9ad4a1 |
| [outputs/u1a/2k_evaluation_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_evaluation_state.json) | 2k/evaluation | 05e22d3b44e2631af2e62c68bb97acc1cb21ecedd815316509945efd53dab407 |
| [outputs/u1a/2k_gate.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_gate.json) | 2k/gate | 2bdb8b97f45f46477cc2ebdd943743f77146c75bc37d5efda1d86051f8682de7 |
| [outputs/u1a/2k_raw_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_raw_state.json) | 2k/raw_state | 16770e8deb1eddc6a72a6bd57f94bf92b9297c0af3ba7a418d48904bb35bae02 |
| [outputs/u1a/2k_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_state.json) | 2k/training | 5b7895a3f7a316b8476f0da742345132af19f0b6b2f356eb584dc28246ecbff3 |
| [outputs/u1a/2k_table.md](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_table.md) | 2k/evaluation_sources/table; 2k/standard_table | 1f48cc43db7274d2a9561b3b694e9883d24d2a7c9f1a68ed5da6f17599029b14 |
| [outputs/u1a/2k_verification.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/2k_verification.json) | 2k/evaluation_sources/verification; 2k/verification | e8f63926ff40413d836a18a2af9e5416149f5de2974b1125e31d3fe13cdf3822 |
| [outputs/u1a/6k_completion_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_completion_state.json) | 6k/completion | 59818c4efe65067c7865ab9e922d1b3f0efaf58ad27dffc5e7a306d282464dde |
| [outputs/u1a/6k_evaluation_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_evaluation_state.json) | 6k/evaluation | 47ca401a11b57e9816ef7b78da8870b74d5b03f7c60448bc73ec6ee04c13870d |
| [outputs/u1a/6k_gate.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_gate.json) | 6k/gate | 7deee2d5f6a3b0ce048f3032e4af08a059b079ec63df431e8005f9d7d5ba4563 |
| [outputs/u1a/6k_raw_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_raw_state.json) | 6k/evaluation_sources/raw_barrier; 6k/raw_state | 1c86acf4c38a5edefb4ba5013b7e0781110de9f9043a93eb68d07e1f36745400 |
| [outputs/u1a/6k_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_state.json) | 6k/training | 36331cd6b3e4344d576469cbd63c94c425d3a8c858ff4bb57de37c5c4e2c415e |
| [outputs/u1a/6k_table.md](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_table.md) | 6k/evaluation_sources/table; 6k/standard_table | 3c31f3644b1397c64d6e339bab6e350352e271d5206ea762af03a7150e7e31ce |
| [outputs/u1a/6k_verification.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/6k_verification.json) | 6k/evaluation_sources/verification; 6k/verification | 5a7f16288a082c432f56e85a7db748655793285a4fbbb4886000a35cc71867b4 |
| [outputs/u1a/debug_contract.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/debug_contract.json) | debug/forward_gradient_FK_hold | e0155cb0bfded7f60afb46c2f4481c8f60710f57c3570712d3765dbd01724859 |
| [outputs/u1a/debug_state.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/debug_state.json) | debug/DDP_state | fbd3d4195ada5bfac14537c697fc009c92fe06023fe99fa2614bc1c28267d34f |
| [outputs/u1a/debug_verification.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/debug_verification.json) | debug/last_frozen_paired | 18abbd39723198750dc6d94ebe64bb422b01cfea27f0563cd0dea6e04ebed07b |
| [outputs/u1a/execution_isolation.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/execution_isolation.json) | sealed_execution_isolation_metadata | ce3e61edbf762648123b9e87ec121d82ee4c2e09820cd361e0ccf341b2489802 |
| [outputs/u1a/init/shared_s3407.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/u1a/init/shared_s3407.ckpt) | 2k/shared/row_sources/provenance/3407/initialization; 6k/shared/row_sources/provenance/3407/initialization | 42fc45ef3fcd85c0daeb317be412857c90a5acf3d74fcd0bc9f55d0fbfabc648 |
| [outputs/u1a/init/shared_s3408.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/u1a/init/shared_s3408.ckpt) | 2k/shared/row_sources/provenance/3408/initialization; 6k/shared/row_sources/provenance/3408/initialization | 9088239a47ddf119ade433d110a3119cf3ef33234b12ebbb39c5696b97af2627 |
| [outputs/u1a/init/untied_s3407.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/u1a/init/untied_s3407.ckpt) | 2k/untied/row_sources/provenance/3407/initialization; 6k/untied/row_sources/provenance/3407/initialization | 7cc7b130e689d4e6fae032a079632f29805c1f46b8b205d7ad42b8e3137dea31 |
| [outputs/u1a/init/untied_s3408.ckpt](/data1/lyq/code/mesh/EventHands1/outputs/u1a/init/untied_s3408.ckpt) | 2k/untied/row_sources/provenance/3408/initialization; 6k/untied/row_sources/provenance/3408/initialization | 72b1f12b4acdafa23bc66f90a408cf1a4dc3b2d1546ad1ef9f27812a7ba006a5 |
| [outputs/u1a/initialization.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/initialization.json) | 2k/shared/row_sources/provenance/3407/initialization_registry; 2k/shared/row_sources/provenance/3408/initialization_registry; 2k/untied/row_sources/provenance/3407/initialization_registry; 2k/untied/row_sources/provenance/3408/initialization_registry; 6k/shared/row_sources/provenance/3407/initialization_registry; 6k/shared/row_sources/provenance/3408/initialization_registry; 6k/untied/row_sources/provenance/3407/initialization_registry; 6k/untied/row_sources/provenance/3408/initialization_registry; prepared_initialization_registry | 036a4cf4544cb8835ce0fff68884fded857a5f06f75065b73fb43a0f1c573200 |
| [outputs/u1a/u0_equivalence_full.json](/data1/lyq/code/mesh/EventHands1/outputs/u1a/u0_equivalence_full.json) | U0/full_equivalence | eb311b7fffcfb526eac48292a4ecdcd64ce8294b5105cf13e14abad5b4f27d6c |
| [outputs/u1a/u1a_shared_debug_s3407.log.attempt1](/data1/lyq/code/mesh/EventHands1/outputs/u1a/u1a_shared_debug_s3407.log.attempt1) | debug/preserved_first_DDP_attempt | 7c9570e06c235224ed68eeafeb518f2edd0b45b3404c557b807ad9d66c055daa |
| [outputs/u1a/u1a_untied_debug_s3407.log.attempt1](/data1/lyq/code/mesh/EventHands1/outputs/u1a/u1a_untied_debug_s3407.log.attempt1) | debug/preserved_first_DDP_attempt | 21cfeb37f65b12a5ad4d407f34883aa95682607f3f10dfc595348a3f5cf6c9de |
| [semkine/dataset.py](/data1/lyq/code/mesh/EventHands1/semkine/dataset.py) | isolation/active_source | 7c7495c1eda4e7278a7cdbd0faec284542826e5b1e88ff062e908c3a1a8c59f8 |
| [semkine/eval_track.py](/data1/lyq/code/mesh/EventHands1/semkine/eval_track.py) | isolation/active_source | fca4c4934a09adc5f465a63653ddf17d513321efbee198a498c60231dd3b63d3 |
| [semkine/events.py](/data1/lyq/code/mesh/EventHands1/semkine/events.py) | isolation/active_source | 5f63b32693ae8a81b8888c0d50295a7b416c35e660d98672bee6e0e3ee86c27e |
| [semkine/lie.py](/data1/lyq/code/mesh/EventHands1/semkine/lie.py) | isolation/active_source | 6bb7cd626e64d2f9960884c45f98480e066a9bd0ebed99a7453d7e889db7e065 |
| [semkine/sparse_pyramid.py](/data1/lyq/code/mesh/EventHands1/semkine/sparse_pyramid.py) | isolation/active_source | 9b272a3f6d4d1bd06c6528886149fd4b874affa1ccec4fe9666173e0312571c3 |
| [semkine/train.py](/data1/lyq/code/mesh/EventHands1/semkine/train.py) | code_or_prereg_source; isolation/active_source | 3b80ded8765dc867f4b73eee37fea7e80d415e9e6f86d22f8068448ca0ac1f41 |
| [semkine/u1a_readout.py](/data1/lyq/code/mesh/EventHands1/semkine/u1a_readout.py) | U0/source_unchanged; code_or_prereg_source; isolation/active_source | 2d12dfe29addcb3d17fea76fae1f1063142f97a0179f723d5546e99aaf7a6bf0 |
| [tests/test_s37_routed_readout.py](/data1/lyq/code/mesh/EventHands1/tests/test_s37_routed_readout.py) | code_or_prereg_source | 072e08ca19a06d233083a7dcedbd78a51df6bc4c8bcf959df0f27f3680085844 |
| [tests/test_s38.py](/data1/lyq/code/mesh/EventHands1/tests/test_s38.py) | code_or_prereg_source | a76a1ad80a0b049e9c42a3e13e9abedef42a862d75539b5a86de3d72b5cb672e |
| [tests/test_u1a_readout.py](/data1/lyq/code/mesh/EventHands1/tests/test_u1a_readout.py) | code_or_prereg_source | 9881130fa7c84ca722cf11eba15cfe8941253876688f49a931ca65f59d808f37 |
| [tools/report_table.py](/data1/lyq/code/mesh/EventHands1/tools/report_table.py) | code_or_prereg_source; isolation/active_source | d431c6dc6c1c12229843b966a8e311406fac2cd22d9220ec0c79d9d0f55976b4 |
| [tools/u1a/debug.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/debug.py) | code_or_prereg_source | 42953b737221756d81438181cb6827ec838dafd8b3f207fd8680c5b1e2a27958 |
| [tools/u1a/evaluate_phase.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/evaluate_phase.py) | code_or_prereg_source | d0818fc809e810ce78225abab82b520739585e0c94a0eb7402a08da5e934a0a8 |
| [tools/u1a/finish_6k.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/finish_6k.py) | code_or_prereg_source | cfb520894fcb09b8acd6868b3ddaef3616f18993adf33e53e486d5e1776d1c55 |
| [tools/u1a/gates.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/gates.py) | code_or_prereg_source; gate executable; isolation/active_source | 3636fbe5a9eb867be601f768a1acb96ae4725509d0602d147167a4b08b3504d8 |
| [tools/u1a/make_verdict.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/make_verdict.py) | code_or_prereg_source | 7d522a44f5890d0f00224e03d93d0615c319a4b8290fd107c4040dfb038996cf |
| [tools/u1a/prepare.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/prepare.py) | code_or_prereg_source; isolation/active_source | 92a6b88de14584c73552633792364dcac710a78faf0e7102c6cc4f7cb9f80ef0 |
| [tools/u1a/probe.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/probe.py) | code_or_prereg_source; isolation/active_source | 20e39b487d830c40cdadd5162506f5926f08b3100031de62db9a2d738bfbc0e6 |
| [tools/u1a/report.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/report.py) | code_or_prereg_source; isolation/active_source | 34e75c3aceb2eda6eefb0390606c324cfa2516f5ca616cce46e6bebed73df423 |
| [tools/u1a/run_raw.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/run_raw.py) | code_or_prereg_source | 142ff50cd5c142581509c7fa32dfe6780d1e34b1542673e303ff11d1849bda22 |
| [tools/u1a/u0_equivalence.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/u0_equivalence.py) | U0/executable; code_or_prereg_source; isolation/active_source | 438e04bf07387d5adb4caf025f27affc1b02061f0e7d7414eae7dca3a1b1a691 |
| [tools/u1a/verify.py](/data1/lyq/code/mesh/EventHands1/tools/u1a/verify.py) | code_or_prereg_source; isolation/active_source | 5b35cac657c9e96ae8ed772a9c0f5539293e95ed2fe019cbfcb48567fd0d000a |
| [tools/x1001/evalx.py](/data1/lyq/code/mesh/EventHands1/tools/x1001/evalx.py) | code_or_prereg_source; isolation/active_source | c7a38f26502b5ef76fa3a1784bf55884eff6126426bea52fafbbcabb3561ca09 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/model/model.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/model/model.py) | isolation/protected_origin_copy | 794dd39b1fa81f7e54f3ff077b27a3f6712f29a4835d36ca9bcb01f2bd186b91 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/rt_s37_3seed_tf_pert_main_row.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/rt_s37_3seed_tf_pert_main_row.json) | isolation/exact_historical_copy | 0afdf6f1b8117fe5b5bc144d277a4a263c5c4a3c4f863efdb8ae2750c1e75e99 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/s38_spmeas_tf_pert_main_row.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/s38_spmeas_tf_pert_main_row.json) | isolation/exact_historical_copy | d82bfedf76f0bbecfaef0a43bae52e36ba8e736641aca59208d2106165283871 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_shared_2k_extended.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_shared_2k_extended.json) | isolation/exact_historical_copy | a45904022786e7be6304df78565eb567ddee783aa783d86fcf274fec9c2df893 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_shared_2k_main_row.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_shared_2k_main_row.json) | isolation/exact_historical_copy | 1b614db25ebd1b1f81bde8ccfc49b82c9f00e6557c5aeae03f7d74cd324ab4f5 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_untied_2k_extended.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_untied_2k_extended.json) | isolation/exact_historical_copy | 151d733e9dfdce50bfb67f01643a91a3f63a2b66c4b4c2b456358f35a07e452f |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_untied_2k_main_row.json](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/pinned_outputs/semkine/u1a_untied_2k_main_row.json) | isolation/exact_historical_copy | 55f7e7d76f83c7a2ed90179578ad16a2abb6ab93fbf28d34535867c206d9d4e3 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/dataset.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/dataset.py) | isolation/protected_origin_copy | 7c7495c1eda4e7278a7cdbd0faec284542826e5b1e88ff062e908c3a1a8c59f8 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/eval_track.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/eval_track.py) | isolation/protected_origin_copy | fca4c4934a09adc5f465a63653ddf17d513321efbee198a498c60231dd3b63d3 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/events.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/events.py) | isolation/protected_origin_copy | 5f63b32693ae8a81b8888c0d50295a7b416c35e660d98672bee6e0e3ee86c27e |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/lie.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/lie.py) | isolation/protected_origin_copy | 6bb7cd626e64d2f9960884c45f98480e066a9bd0ebed99a7453d7e889db7e065 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/sparse_pyramid.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/sparse_pyramid.py) | isolation/protected_origin_copy | 9b272a3f6d4d1bd06c6528886149fd4b874affa1ccec4fe9666173e0312571c3 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/train.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/train.py) | isolation/protected_origin_copy | 3b80ded8765dc867f4b73eee37fea7e80d415e9e6f86d22f8068448ca0ac1f41 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/u1a_readout.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/semkine/u1a_readout.py) | isolation/protected_origin_copy | 2d12dfe29addcb3d17fea76fae1f1063142f97a0179f723d5546e99aaf7a6bf0 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/report_table.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/report_table.py) | isolation/protected_origin_copy | d431c6dc6c1c12229843b966a8e311406fac2cd22d9220ec0c79d9d0f55976b4 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/gates.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/gates.py) | isolation/protected_origin_copy | 3636fbe5a9eb867be601f768a1acb96ae4725509d0602d147167a4b08b3504d8 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/isolated_exec.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/isolated_exec.py) | isolation/launcher | 396a41e2819dc4a59783928b4596b3de5ed57a12b7e76e4755709f2d487eb1ed |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/prepare.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/prepare.py) | isolation/protected_origin_copy | 92a6b88de14584c73552633792364dcac710a78faf0e7102c6cc4f7cb9f80ef0 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/probe.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/probe.py) | isolation/protected_origin_copy | 20e39b487d830c40cdadd5162506f5926f08b3100031de62db9a2d738bfbc0e6 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/report.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/report.py) | isolation/protected_origin_copy | 34e75c3aceb2eda6eefb0390606c324cfa2516f5ca616cce46e6bebed73df423 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/u0_equivalence.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/u0_equivalence.py) | isolation/protected_origin_copy | 438e04bf07387d5adb4caf025f27affc1b02061f0e7d7414eae7dca3a1b1a691 |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/verify.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/u1a/verify.py) | isolation/protected_origin_copy | 5b35cac657c9e96ae8ed772a9c0f5539293e95ed2fe019cbfcb48567fd0d000a |
| [/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/x1001/evalx.py](/data1/lyq/code/mesh/EventHands1_U1a_frozen_a6e5994/tools/x1001/evalx.py) | isolation/protected_origin_copy | c7a38f26502b5ef76fa3a1784bf55884eff6126426bea52fafbbcabb3561ca09 |
