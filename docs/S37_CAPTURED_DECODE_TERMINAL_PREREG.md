# CD0：原生同函数捕获执行终态附录

本页是实际执行后附录；原预注册、实现、执行器和九文件合同均保持冻结。

终态：`EXACT_CAPTURE_COST_REDUCTION_ON_FIXED_WORKLOAD`。两个seed各合并global/local后的P50/P95/P99/max门全部通过；但3408 local的最大服务耗时退化，完整7ms门失败。当前臂仍为S37，未采纳C为主模型。

## 交付与正确性

`semkine/captured_s37_decode.py`是显式安装到隔离模型的复用模块。只捕获原MANO/FK与原15个解码头的算术；唯一MANO父索引AST改写可逆，其余AST不变。每次复制动态输入、重放图、clone全部输出；正式计时包含全部运行守卫，未修改token、采样、图边、路由、反馈、参数或精度。原core/default训练入口未改，运行不依赖scratch。

最终CPU8/8通过（1.173秒），覆盖AST与安装所有权等静态合同，不冒称CUDA测试。唯一GPU任务通过14条合同记录：构造异常恢复、grad/inference/AMP/非严格确定性/train/换stream/错误dtype及shape/hooks/buffer变更/参数version变更拒绝、异常退出恢复。注入的是Python捕获调用异常，不证明任意设备故障后的CUDA恢复；.data/raw storage/外部CUDA写入与无协调并发仍在隔离所有权合同之外。

合成0→1→7→2051→0事件序列及中途beta变更通过。两个C@500 checkpoint×两序列各256帧，1024次自身闭环：reference/captured pose与prev对旧封存C结果逐字节一致，完整mesh/joints与原外部MANO逐字节一致；保留GPU输出在后续重放中不被覆盖。全部通过后才开始8流×64正式计时；计时后512次pose/prev仍对旧轨迹逐字节一致、保留CPU输出不变。父单CPU另读保存NPZ复算全部摘要分位数、计数、时钟关系、配对历史及成本门。

未读取新GT标签或zgz、未评分精度、未训练或选checkpoint。C仍是已用开发样本上的有限工程对照，不能当独立泛化或科学创新。

## 工作负载与时间边界

GPU1 NVIDIA L20、Torch2.1/CUDA11.8，FP32、TF32 off、严格deterministic、eval/普通no_grad、单stream、单进程4库线程及1 interop。GPU0未用。固定顺序reference3407→captured3407→captured3408→reference3408，各global→local原首64帧；每流五次首包预热后合法reset，正式首请求保留。原H50ms/20Hz及末端完整1ms事件bin保持。

原始event/tsub已先物化RAM；实际CPU构包、H2D、完整forward、输出FK、pose+778顶点+21关节D2H及同步均计入service，含守卫/copy/replay/clone。query响应另计排队；raw事件响应还含原50ms触发等待。无未来构包、无中途同步profiling、无preview或短反馈。物理sensor传输、OS收包、磁盘加载不在此回放计时；未覆盖所有突发/恢复或任意事件率。

每seed每实现global共1163690原始事件、127972采样节点，最大包50631；local共163834原始事件、88777采样节点，最大包10299；各64请求，均无自然空包。空包只有合成正确性证据。原cap2048保持；raw分母不是全部事件均被保留或具有可识别预测影响。

## 预注册合并服务成本门

| C执行器/seed | n | P50 ms | P95 | P99 | max | >7比例 |
|---|---|---|---|---|---|---|
| reference_3407 | 128 | 25.495886569842696 | 25.96122365212068 | 27.05469513428398 | 28.06356700602919 | 1.0 |
| captured_3407 | 128 | 23.11763452598825 | 24.603863595984876 | 25.08151171146892 | 25.48922400455922 | 1.0 |
| captured_3408 | 128 | 23.46567704807967 | 24.56663222401403 | 24.850952959386632 | 27.940941974520683 | 1.0 |
| reference_3408 | 128 | 25.526327488478273 | 28.893242857884612 | 29.510902155889198 | 30.389657011255622 | 1.0 |

门要求两seed各自合并128请求的四统计量均不退且P50严格下降；父从保存数组复算一致。合并门通过不等于每个序列每个尾部都改善：3408 local服务P99由25.941702499985695升至25.950360555434592 ms，最大值由26.004406972788274升至27.940941974520683 ms，raw事件响应最大值由75.79062397562653升至77.86650291818664 ms。原因未定位，不能归咎于噪声后剔除。两seed的local正式首请求都更慢，以下原值保留。未重复测量、调整守卫或事后改门。

## 各流全部延迟统计

| 执行器/seed/序列/边界 | n | P50 ms | P95 | P99 | max | >7比例 |
|---|---|---|---|---|---|---|
| reference_3407/ylf_global/queue_ms | 64 | 0.09750796016305685 | 0.10735723190009594 | 0.1091835112310946 | 0.10940397623926401 | 0.0 |
| reference_3407/ylf_global/service_ms | 64 | 25.24196798913181 | 26.068770640995353 | 27.09951577009633 | 28.06356700602919 | 1.0 |
| reference_3407/ylf_global/query_response_ms | 64 | 25.33935708925128 | 26.157760148635134 | 27.205032979836684 | 28.16306205932051 | 1.0 |
| reference_3407/ylf_global/all_raw_input_event_response_ms | 1163690 | 50.09078700095415 | 72.79319709632537 | 75.02692299894976 | 78.1430620132596 | 1.0 |
| reference_3407/ylf_global/oldest_event_per_nonempty_query_ms | 64 | 74.55430853296997 | 75.91241935653669 | 77.03446298181363 | 78.1430620132596 | 1.0 |
| reference_3407/ylf_global/newest_event_per_nonempty_query_ms | 64 | 25.94408163568007 | 27.554402271052833 | 27.915783277479868 | 28.231062716804423 | 1.0 |
| reference_3407/ylf_global/empty_query_response_ms | 0 | None | None | None | None | None |
| reference_3407/ylf_local/queue_ms | 64 | 0.09860744467005134 | 0.11382902739569545 | 0.1350993139203637 | 0.1570410095155239 | 0.0 |
| reference_3407/ylf_local/service_ms | 64 | 25.52131103584543 | 25.812499213498086 | 26.36300995480269 | 27.247529942542315 | 1.0 |
| reference_3407/ylf_local/query_response_ms | 64 | 25.610118987970054 | 25.89959186152555 | 26.493410782422867 | 27.40457095205784 | 1.0 |
| reference_3407/ylf_local/all_raw_input_event_response_ms | 163834 | 49.62405429687333 | 72.71486304234709 | 75.18505393527431 | 76.82657092809686 | 1.0 |
| reference_3407/ylf_local/oldest_event_per_nonempty_query_ms | 64 | 75.12802696728605 | 75.68334036441212 | 76.22726074485394 | 76.82657092809686 | 1.0 |
| reference_3407/ylf_local/newest_event_per_nonempty_query_ms | 64 | 26.1467641452327 | 26.922170007601242 | 27.64875649590985 | 28.472569910809487 | 1.0 |
| reference_3407/ylf_local/empty_query_response_ms | 0 | None | None | None | None | None |
| captured_3407/ylf_global/queue_ms | 64 | 0.10960194049403071 | 0.12360656983219086 | 0.15487171476706862 | 0.15520688612014055 | 0.0 |
| captured_3407/ylf_global/service_ms | 64 | 23.80875451490283 | 24.77350991102867 | 25.279287850717083 | 25.48922400455922 | 1.0 |
| captured_3407/ylf_global/query_response_ms | 64 | 23.927197966258973 | 24.884514301083982 | 25.396401309408247 | 25.599753949791193 | 1.0 |
| captured_3407/ylf_global/all_raw_input_event_response_ms | 1163690 | 48.72650941833845 | 71.44909196067606 | 73.54875393211846 | 75.11875395430256 | 1.0 |
| captured_3407/ylf_global/oldest_event_per_nonempty_query_ms | 64 | 73.21787044638762 | 74.49990760309444 | 74.99541130341807 | 75.11875395430256 | 1.0 |
| captured_3407/ylf_global/newest_event_per_nonempty_query_ms | 64 | 24.604350887238947 | 25.501557890092762 | 26.022373750573035 | 26.374187483452218 | 1.0 |
| captured_3407/ylf_global/empty_query_response_ms | 0 | None | None | None | None | None |
| captured_3407/ylf_local/queue_ms | 64 | 0.10865950025618076 | 0.13068512198515234 | 0.14318687259219587 | 0.14638097491115332 | 0.0 |
| captured_3407/ylf_local/service_ms | 64 | 22.892909008078277 | 24.078448017826304 | 24.294330378761515 | 24.483973043970764 | 1.0 |
| captured_3407/ylf_local/query_response_ms | 64 | 23.00996903795749 | 24.179997493047267 | 24.389414915349334 | 24.578510085120797 | 1.0 |
| captured_3407/ylf_local/all_raw_input_event_response_ms | 163834 | 47.50123342964799 | 70.20780295133578 | 72.82813698402602 | 74.4065100618172 | 1.0 |
| captured_3407/ylf_local/oldest_event_per_nonempty_query_ms | 64 | 72.66450854949647 | 73.68309460871396 | 74.19969174414273 | 74.4065100618172 | 1.0 |
| captured_3407/ylf_local/newest_event_per_nonempty_query_ms | 64 | 23.64119748817761 | 24.778419488575075 | 25.286447311518803 | 25.34635798074314 | 1.0 |
| captured_3407/ylf_local/empty_query_response_ms | 0 | None | None | None | None | None |
| captured_3408/ylf_global/queue_ms | 64 | 0.10800850577652454 | 0.11985541786998509 | 0.13204252696596078 | 0.15012791845947504 | 0.0 |
| captured_3408/ylf_global/service_ms | 64 | 23.54928501881659 | 24.40335854771547 | 24.681223118677735 | 24.876719107851386 | 1.0 |
| captured_3408/ylf_global/query_response_ms | 64 | 23.645356006454676 | 24.503814888885245 | 24.790290753589943 | 24.987826007418334 | 1.0 |
| captured_3408/ylf_global/all_raw_input_event_response_ms | 1163690 | 48.54958781506858 | 71.42694699577979 | 73.53188004344702 | 74.26587197696777 | 1.0 |
| captured_3408/ylf_global/oldest_event_per_nonempty_query_ms | 64 | 72.9461419599829 | 73.98843612085331 | 74.12840534736448 | 74.26587197696777 | 1.0 |
| captured_3408/ylf_global/newest_event_per_nonempty_query_ms | 64 | 24.237121641635937 | 25.381462259683616 | 26.000044595682958 | 26.397376065142364 | 1.0 |
| captured_3408/ylf_global/empty_query_response_ms | 0 | None | None | None | None | None |
| captured_3408/ylf_local/queue_ms | 64 | 0.11208048090338707 | 0.12841249117627737 | 0.15565271256491536 | 0.17253786791116 | 0.0 |
| captured_3408/ylf_local/service_ms | 64 | 23.176024027634412 | 24.64468159014359 | 25.950360555434592 | 27.940941974520683 | 1.0 |
| captured_3408/ylf_local/query_response_ms | 64 | 23.28176994342357 | 24.78861401323229 | 26.05974811362102 | 28.039502911269665 | 1.0 |
| captured_3408/ylf_local/all_raw_input_event_response_ms | 163834 | 48.104502446949525 | 71.15364288911196 | 73.72771591180927 | 77.86650291818664 | 1.0 |
| captured_3408/ylf_local/oldest_event_per_nonempty_query_ms | 64 | 72.88350996550425 | 74.42181110745871 | 75.88017529020719 | 77.86650291818664 | 1.0 |
| captured_3408/ylf_local/newest_event_per_nonempty_query_ms | 64 | 24.30212615290661 | 25.516543933190302 | 26.734238826669674 | 28.57350390404467 | 1.0 |
| captured_3408/ylf_local/empty_query_response_ms | 0 | None | None | None | None | None |
| reference_3408/ylf_global/queue_ms | 64 | 0.09912007953971624 | 0.11116034002043307 | 0.12226821854710578 | 0.12654403690248728 | 0.0 |
| reference_3408/ylf_global/service_ms | 64 | 25.52802296122536 | 29.274043795885518 | 29.879344984656196 | 30.389657011255622 | 1.0 |
| reference_3408/ylf_global/query_response_ms | 64 | 25.622993533033878 | 29.384880006546155 | 29.98951941495761 | 30.499963089823723 | 1.0 |
| reference_3408/ylf_global/all_raw_input_event_response_ms | 1163690 | 50.35164239816359 | 72.79764194972805 | 74.98688413761556 | 79.76896306499839 | 1.0 |
| reference_3408/ylf_global/oldest_event_per_nonempty_query_ms | 64 | 74.75644555597677 | 78.94192399136955 | 79.46830938872883 | 79.76896306499839 | 1.0 |
| reference_3408/ylf_global/newest_event_per_nonempty_query_ms | 64 | 26.16341810207812 | 29.907885493012138 | 30.62647801195272 | 31.078961817547686 | 1.0 |
| reference_3408/ylf_global/empty_query_response_ms | 0 | None | None | None | None | None |
| reference_3408/ylf_local/queue_ms | 64 | 0.09783392306417227 | 0.1183052605483681 | 0.12946761096827686 | 0.13240589760243893 | 0.0 |
| reference_3408/ylf_local/service_ms | 64 | 25.52336099324748 | 25.800191721646115 | 25.941702499985695 | 26.004406972788274 | 1.0 |
| reference_3408/ylf_local/query_response_ms | 64 | 25.616295519284904 | 25.90016121394001 | 26.03204694809392 | 26.0783500270918 | 1.0 |
| reference_3408/ylf_local/all_raw_input_event_response_ms | 163834 | 49.63762443512665 | 72.6526238955556 | 75.26112497434934 | 75.79062397562653 | 1.0 |
| reference_3408/ylf_local/oldest_event_per_nonempty_query_ms | 64 | 75.1899434428196 | 75.62647512910173 | 75.78285230172317 | 75.79062397562653 | 1.0 |
| reference_3408/ylf_local/newest_event_per_nonempty_query_ms | 64 | 26.14171347813687 | 26.86723543796695 | 27.24268665164708 | 27.28634916711581 | 1.0 |
| reference_3408/ylf_local/empty_query_response_ms | 0 | None | None | None | None | None |

所有正式service、query响应、raw事件响应均超过7ms。原50ms等待仍然存在；即使服务加速也不能宣称完整事件→mesh达标。此处测得最大值不构成任意负载硬实时保证，不能与其他checkpoint精度或另一输出策略拼接。

## 准备、捕获、暖机与首请求

| seed | 加载/捕获/全回归 s | 捕获准备总计 ms | FK预热 ms | FK捕获 ms | heads预热 ms | heads捕获 ms |
|---|---|---|---|---|---|---|
| 3407 | 16.23704423592426 | 157.866039 | 8.977518 | 64.645029 | 10.152254 | 54.557128 |
| 3408 | 15.615685319993645 | 195.650168 | 9.405523 | 101.66055 | 10.180347 | 54.18002 |

| 执行器/seed/序列 | 五次预热总计 s | 正式首请求service ms（已计入分布） |
|---|---|---|
| reference_3407/ylf_global | 0.08039274904876947 | 16.284959972836077 |
| reference_3407/ylf_local | 0.08045614697039127 | 16.81584003381431 |
| captured_3407/ylf_global | 0.05091045598965138 | 9.849683032371104 |
| captured_3407/ylf_local | 0.086240514065139 | 18.04970297962427 |
| captured_3408/ylf_global | 0.085956078954041 | 18.185715074650943 |
| captured_3408/ylf_local | 0.08628900803159922 | 18.04183900821954 |
| reference_3408/ylf_global | 0.14288752397987992 | 28.98779499810189 |
| reference_3408/ylf_local | 0.0804690399672836 | 16.594941029325128 |

捕获总计是实例准备；加载/捕获/全回归包含Debug工作，二者不能相加解释部署冷启动。预热含构包和最后同步；首请求不删除。没有独立冷启动分布或稳态普适性结论。

进程Torch峰值allocated=285854208 bytes（含多模型与Debug，不是单模型部署峰值，非reserved/NVML整卡内存）。参数733958不变。进程内部elapsed=59.20672589389142秒；budget_run完整初始化/退出/清理收费62.146028652903624 GPU秒。

## 资源、身份及决定

唯一job `cd0_captured_decode_20260926_v1` COMPLETED/0，实际62.146028652903624/250 GPU秒，cleanup无残留，leader/runner均退出。源码前后9+81 SHA及330数据stat保持。累计预算：`{"debug": {"charged": 7133.62689116376, "reserved": 0.0}, "screen": {"charged": 37363.602386470026, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}`。原作业已终态，不重启。

准确命令和设备在research_state/jobs/cd0_captured_decode_20260926_v1.json；冻结合同scratch/goal_20260926/cd0/contract.json、快照research_state/snapshots/CD0_frozen_sources_20260926；全部逐query/逐事件时钟与pose/prev在run_v1，父复算执行器record_terminal.py不加载模型、不访问GPU。

使用要求：先同配置构造并安装constant-context执行器、strict加载对应C权重、全模型冻结/eval，再在普通torch.no_grad中创建并使用CapturedS37Decode上下文；只适用于文档化的隔离FP32实例。不自动接入当前S37、部署或训练。

保留该opt-in工程实现与固定负载成本证据；不删除守卫以追逐速度，不重跑不利尾部。C0r HOLD、固定C1r/R0 REJECT、C2r HOLD及S37当前臂保持，完整目标未达。下一判别须同时面对事件触发等待和有用的因果状态校正，先利用已有原始事件/状态证据界定新增信息，不能继续以纯执行微优化替代创新与精度。原短周期反馈未获新答复，不擅改H。

父audit：research_state/audit/CD0_terminal_parent_20260926.json；独立终态审查另存research_state/debug/CD0_terminal_review_20260926.md，父读取最终SHA后关闭复核。summary写在finally之前，本结论额外要求了实际clean exit及恢复无异常，不仅凭summary判定。
