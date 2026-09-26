# MP0：legacy S37非空AMP状态累加

2026-09-26，实际诊断前登记。PA1已经支持停止，不改其窗口/种子/轴/门或开标签。此项属于C0r最小数值修复，不是第四完整候选、创新通过或训练配方改进结论。

## 已定位边界

`forward_packet`的legacy非空路径先将FP32历史转成head dtype，再作低精度delta相加，最终EP0的where可把结果扩回FP32但不能恢复丢掉的位。EP0有意锁定这个非空旧算术；EGM和SparseS37Tracker已经有保状态精度累加的先例。这里不宣称首次发现或用尾部算术解释全部漂移。混合精度网络内部、prev_mlp输入和route仍可有量化差异，不能把尾部修复叫完整FP32网络。

最小opt-in `MODEL.STATE_ACCUM_FP32=false`保持默认。true仅准入原始event_gnn+routed+predict_delta，最终delta和prev在至少FP32的共同dtype相加（FP64输入保留FP64），空包最终where继续精确拷贝prev；没有参数/旧权重键新增。FP32默认评测应逐位相同；AMP训练的损失、梯度和最终权重允许改变，但必须重新验证，不能冒称旧complete checkpoint精确恢复。旧源码已保存在`mp0/model_before.py`，旧EP0/RC1合同不改。

## 固定Debug与一次实际保真比较

人工真实tiny S37测试先覆盖CPU FP32/BF16和授权CUDA FP32/BF16/FP16：flag关闭的legacy合同；flag开启同一次AMP raw delta的显式提升加法参考；非空微小增量不被历史舍入吞掉；空/混合行状态及梯度；FP32值及非线性平方目标梯度逐位回归；FP16/BF16/FP64状态的至少FP32提升；人工输入上原MSE/log10的一次SGD更新有限且参数确实改变；错误架构配置明确拒绝。新测试验证算术合同，不把“等于自己的定义”作为效用证据。必要继承EP0默认回归，只在代码有实质修改时重测。

实际数据固定为已归档`scratch/goal_20260925/x1/s37_bitwise/packet.pt`全部8条训练样本，不换样、不新增数据/zgz、不拟合；缓存中target是原训练监督，本次允许用于原损失的梯度保真测量，不是验证精度。两个原S37 seed3407和3408均固定step2500，对每个seed比较：F=完整FP32默认、L=BF16旧尾部、P=BF16保FP32状态尾部。模型同权重/同输入，无optimizer step；使用原51D损失及log10，全部参数梯度拼接后比较F，保存完整输出、梯度与损失。数值严格deterministic、TF32关闭、autocast只改变指定模式。

同时保存同一AMP raw delta，验证L尾部等于旧算术、P尾部等于提升算术；这是隔离归因检查。所有参数键/值执行前后保持，FP32 flag开关须逐位输出及梯度一致；3407的历史8样本输出由独立CPU脚本逐字节匹配；GPU只核同一CUDA backend的F/F_flag开关等价，不要求CPU/CUDA跨后端逐位一致。混合精度头内部误差和尾部舍入分开保存，不计算新的递推精度行。

**保真门事前固定：**每seed P相对F的整个梯度L2距离严格低于L相对F；两个seed都满足才`AMP_GRADIENT_FIDELITY_IMPROVED_ON_FIXED_BATCH`，否则`NO_JOINT_GRADIENT_FIDELITY_GAIN`。这不是任务效用/收敛保证，也不要求每一个参数或每样本都改善；完整分块梯度和输出差异仅作解释，不能事后改门。若L与F梯度本来逐位相同，保真门未通过而不是除零/宣称改善。记录原损失差、梯度cosine、相对L2、参数块及RA几何变化（若解码则明确仅输出差异而非GT精度），不用它们择seed/配置。

若合同失败，修复明确实现错误并保留失败，不放宽标准；若实际固定保真门失败，不扫precision、改样本/step/seed来救场。人工通过只能保留opt-in实现，是否开展后续配对短训须有保真信号及单独固定真实闭环筛查合同；本轮不自动开训练，也不改默认S37或主表。

## 资源和归档

CPU单数值worker、4库线程，人工测试每次≤120秒；CUDA只用实时确认空闲的GPU1–7之一，GPU0禁用，单job经budget_run Debug计费、总300秒+10秒清理，含CUDA人工和实际两seed比较。不为占卡并行。执行前固定core/prereg/tests/harness/cache/checkpoints/依赖SHA与命令，结果存在拒绝重复启动。实际source/参数/输出/梯度和终态落盘，父核保存工件，独立复核范围如实记录。无新Latency测量、主行、模型采用或完整目标达成。
