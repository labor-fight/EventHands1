# CC1 同函数执行与成本终态附录

这是实际执行后的附录，不是追写预注册。原合同 `docs/S37_CONSTANT_EXECUTION_PREREG.md` 与启动快照保持不变。

时间：2026-09-26T05:02:48.649754+08:00。终态 `EXACT_EXECUTION_OBSERVED_COST_REDUCTION`。固定两seed合并序列的service P50/P95门通过；但尾部退化，完整7ms失败。仅交付显式opt-in的同函数C执行器，不更换S37、不采纳C为主模型、不比较L成本。

## 实际交付与正确性

`semkine/constant_context_infer.py::enable_constant_context_inference` 可安装到同配置新建EventGNN，严格加载旧CC0 state_dict，独立于scratch训练wrapper。保留原W32/k8三层图、所有原参数与context.bias、八槽浮点归约和原状态更新；省去额外context idx/dp与第一次被覆盖的全局pool/proj。原EventGNN/model、SC0/CC0源码未改，参数总量733958不变。默认模型没有自动接入。

CPU30项及CUDA30项均0失败/0跳过，覆盖FP32/BF16非零bias、空/混合/单节点/稀疏mask、严格键/对象身份、一次proj和原g/坐标/掩码。两seed×两序列各256帧FP32自身递推：全部prediction与prev对旧CC0封存数据逐字节一致，原wrapper同输入pose/vertices/joints亦逐字节一致。先通过所有parity再计时；计时流再核自身预测与prev全部一致、持有CPU输出未被后续调用覆盖。BF16是合成同设备对照，未声称有BF16实际长轨迹验证。

未读取NPZ的targets或数据.meta端点标签，未评分新精度、选点或训练。既有CC0效用与C−L容差失败原样保留；本次不能作为新的独立泛化证据。

## 固定负载与计时边界

GPU1 L20，Torch2.1/CUDA11.8，FP32、TF32关闭、确定性算法、单进程4库线程；GPU0未用。原两个ylf前256帧、20Hz H50ms，原合法初值/K/betas，原末端完整1ms bin。每条序列五次首包预热后重置到合法初态，正式首query计入；固定reference3407→optimized3407→optimized3408→reference3408，global→local。

事件/offset/tsub必要前缀先物化到拥有内存的数组；权重/资产/上下文加载在准备阶段，计时不显式读磁盘。按记录时间戳模拟host arrival，固定一个时钟anchor，实际等待release并保留超时队列。service含窗口构包、H2D、token/图/网络/路由、MANO及完整pose/mesh/joints D2H同步。query响应=队列+service；事件响应另含原50ms触发等待。没有物理sensor transport、OS收包、内存锁页保证或任意负载硬实时测量。

每个实现每seed的global包含4885700原始事件、519838采样节点，最大包51005事件；local包含718262原始事件、359475采样节点，最大包11924事件。两者均256请求，没有自然空请求；合成空包正确性不能冒称自然空请求延迟证据。原节点cap2048和采样保持，事件分母为原始输入被服务数量，不代表所有事件均保留或有可识别的预测影响。

## 原值成本与尾部退化

下表仅存于文档，不是正式主表。单位ms；不按更好的序列或seed选取。

| C执行器/seed | service P50 | P95 | P99 | max | >7ms比例 |
|---|---|---|---|---|---|
| reference_3407 | 26.09108097385615 | 26.5398918883875 | 26.740997083252296 | 26.874214061535895 | 1.0 |
| optimized_3407 | 25.62892501009628 | 26.262720627710223 | 36.58471661503423 | 47.39080904982984 | 1.0 |
| optimized_3408 | 25.619168009143323 | 26.120492652989924 | 26.405044642742723 | 47.46602801606059 | 1.0 |
| reference_3408 | 26.099560491275042 | 26.59041660372168 | 26.829875838011503 | 28.60682096797973 | 1.0 |

固定门只涉及两seed各自合并两个序列后的service P50/P95，两者都通过。optimized3407的P99及两个optimized的max高于对应reference；global3407的中位处理成本亦未改善。原因尚未定位，不能未经证据归为噪声/系统抖动，不能称所有分位、场景或最坏情况更快。没有因尾部阴性追加重复或改门。

| C执行器/seed/序列 | 全部原始输入事件响应 P50 | P95 | P99 | max | >7ms比例 |
|---|---|---|---|---|---|
| reference_3407/ylf_global | 50.81669751089102 | 73.39601791463801 | 75.37424201145714 | 76.84569697030419 | 1.0 |
| reference_3407/ylf_local | 51.1089874431491 | 73.62599694170058 | 75.81320794997737 | 76.52528191974817 | 1.0 |
| optimized_3407/ylf_global | 50.84931617602706 | 73.49007006268948 | 75.4957390250639 | 76.72135998727737 | 1.0 |
| optimized_3407/ylf_local | 50.21217961329949 | 72.92450808454198 | 75.38048607530068 | 97.20152007066662 | 1.0 |
| optimized_3408/ylf_global | 50.6490907166155 | 73.28104302287097 | 75.40023904293759 | 97.01996010262536 | 1.0 |
| optimized_3408/ylf_local | 50.53271576762208 | 73.10523591004304 | 75.29308798257262 | 76.96979398897375 | 1.0 |
| reference_3408/ylf_global | 50.92474166303873 | 73.44213798642141 | 75.4139290787862 | 76.94474512245452 | 1.0 |
| reference_3408/ylf_local | 51.10285494010897 | 73.62621307838691 | 75.86236898787303 | 77.32800701705855 | 1.0 |

完整事件响应全部超过7ms，处理时间本身也全部超过7ms；50ms触发等待仍在。此策略没有通过最终延迟门。上表不是实体相机端到端测量；不覆盖更多突发/恢复、任意事件率或通用硬实时上界。P50/P95下降不抵消P99/max退化。原始逐query时钟/计数/pose/prev与逐原始输入事件响应均保存在run_v1 NPZ；各序列service/query/queue/最早/最晚/空请求全部分位数见同目录summary.json。

## 资源、复现与下一动作

唯一作业cc1_exact_execution_20260926_v1 COMPLETED/0，实际144.20186586596537/250 GPU秒，cleanup空、leader/runner均退出，无重跑。累计预算：`{"debug": {"charged": 7054.220061276923, "reserved": 0.0}, "screen": {"charged": 37363.602386470026, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}`。81文件启动SHA保持，快照research_state/snapshots/CC1_frozen_sources_20260926。

运行入口：`scratch/goal_20260926/cc1/verify.py`；冻结合同`contract.json`；测试`tests/test_constant_context_infer.py`。旧作业已终态，不重复启动。安装示例：同cfg构造MNISTModel并eval，调用enable_constant_context_inference(model.event_encoder)，然后strict加载原CC0完整state_dict，再冻结全模型/eval；不要把普通MNISTModel strict load成功误当已安装执行器。

父审计research_state/audit/CC1_terminal_parent_20260926.json；独立终态报告另存research_state/debug/CC1_terminal_review_20260926.md，不改已冻结的harness审查。工程交付可作今后固定校准对照的显式执行器，不能据它自动部署、主训或改科学候选。下一动作应处理允许的状态更新/输出调度与事件→mesh等待合同；原短周期反馈澄清尚未答，不擅自变更H。精度与科学创新仍未完成，不能以继续微调此常量分支替代。
