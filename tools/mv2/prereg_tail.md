### §3 GPU 阶梯

- **只开过了自己探针的机制臂**：M1（放行条件见 M1 节）；M2 只作为 M1 子类内的因子臂；M3（PB 过且 M1 过）、M4（PC 过）、M5（PD 过）作为 M1 上的单变量条件臂；M6、M7、M8（除非 SRCA 放行训练内自监督）、M9、M10 无 GPU 训练臂。各臂独立过自己的门，互不绑死；条件臂的具体配置在其门通过后、开训前以协议修订登记（不改门）。
- **S3 debug**：单 seed 3407，`MAX_STEPS 6000` 跑到 2000 即停（与 S39 同一 cosine 学习率进度）；停止条件（只停不放行，`tools/mv2/s3_check.py`）：SELECT 上 E3 RA ≤ S39@2000 + 0.3 mm、carry（根 8° / 手指 N(0, 0.10)）≤ S39@2000 + 0.05、输出有限。
- **S4**：同一运行续到 6000 步，与 S39 seed 3407 配对；zgz 一次性报告：`tools/mv2/slr_entry.py evalx row … --variant w200cfg` 生成主行，`tools/report_table.py` 出九列表（含 baseline 与 S39 行）；方向支持 = local ≤ −0.3 mm 且 global ≤ +0.2 mm（S42 单种子 6k 门，`#s42-verdict`，描述性）；效应 ≥ 1.5 × MDE_local（1.72 mm）才补 seed 3408 / 3409 并按种子单位确证。另报 E1 / E2 / E3、根旋转 / 平移 / 手指、绝对腕点、段龄、carry、amp（非主表）。
- **资源**：物理 GPU 2–7，单卡单运行（S39 配方：单卡、批 512 × 累积 2，约 75 min / 6000 步）；评估与训练不挤同一卡。

### §4 阈值来源

各方案节内逐项写明；跨方案通用的：KS0 种子 SD 与 MDE（`#kstar-results-20261025`）；S42 单种子 6k 门（`#s42-verdict`）；Holm α = 0.05；本预注册新设的约定值（PA1 零模型容差 −0.05 mm、PA2 天花板 0.90、M9 CPU 增量 +50 % 等）在各节标为"约定 / 提议"。没有一个门取自 zgz 读数。

### §5 停止规则

- 失败分两类：工具缺陷（有效性条件或对照不过 → 该门降为描述；允许一次重设计，须新预注册并保留本次失败）；结果失败（对照全过而主门不过 → 按各节默认动作终止）。不得二读 zgz、不得加长、换 seed 或改门。
- 任何续做需用户另行决定，并作为新研究重新预注册。

### §6 冻结

`tools/mv2/freeze.py` 写 `outputs/mv2/protocol.json`：本节（`#mv2-prereg-20261026` 至 `MV2-PREREG:END`）文本的 SHA-256；`tools/mv2/*.py`、`configs/mv2/*.yaml`、`tools/sba/*.py`、`semkine/slr.py`、`tests/test_slr.py` 与所 import 冻结模块的 SHA；S39 三种子 step 6000 与 step 2000 检查点的 SHA；划分文件 SHA；git HEAD 与 `git status` 摘要（工作树未提交）；FIT7 + SELECT 基础录制数据文件的 (size, mtime_ns)。修订只进 `amendments`。
