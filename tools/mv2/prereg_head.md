## MV2 预注册：十方案 v2.1 的零训练探针与 GPU 阶梯（2026-10-26，写于任何新读数之前）

### §0 来历、范围与披露

- **来历**：用户 2026-10-26 指令"GPU2-7，全量cpu现在开始执行方案"，执行十方案 v2.1（`/home/lyq/.cursor/plans/mano_顶点关节算子十方案_battle_27767ea2.plan.md` §B；登记 `#mv2-decisions-20261025`、`#mv2-r3-result-20261025`、`#mv2-stage0-20261025`、`#mv2-exec-auth-20261026`）。只用物理 GPU 2–7，CPU 全量。
- **范围**：本预注册登记 M1–M10 的全部零训练探针（工具在 `tools/mv2/`，判读函数为各工具的 `decide()`，汇总 `tools/mv2/decide.py` 写 `outputs/mv2/decisions.json`）与 GPU 阶梯（S3 debug、S4 6000 步配对、S5 条件臂）。M7 的 Stage 0 已完成（`#mv2-stage0-20261025`），其后果作为本预注册的约束写入各方案。
- **不变**：S39 不变；划分只用 `splits_semkine.json`；`model/model.py`、`semkine/xyz_mesh.py`、`semkine/train.py`、`dataset.py`、`eval_track.py`、`model/config.py`、`tools/make_s36_row.py`、`tools/report_table.py` 与 R2 / MRF / AURA / K★ 冻结文件不改。新代码只在 `semkine/slr.py`（`SLRModel(MNISTModel)`，`MODEL.SLR*`，opt-in，`tests/test_slr.py` 7 项：默认路径与 S39 逐位相同、参数 2,802,645、开启后零初值时与 S39 函数相同且不消耗全局随机数、梯度到达、错键拒绝、入口替换）、`tools/mv2/`、`configs/mv2/`。训练与评估经 `tools/mv2/slr_entry.py` 替换类后调用原入口。
- **已有读数披露（写本节时已知，不得用于设定或调整任何门）**：
  - K★ 阶梯（`#kstar-results-20261025`）：KS1 理想增量 −2.43 / −1.12 mm、r\* 质量 −0.156；KS2 冻结 patch 头偏相关 0.46 / 0.50、残差跨顶点相关 hop1 0.838 / hop2 0.598 / hop3 0.247 / hop4 −0.117；KS3 r\*\_E3 中位 0.959（真实剖面描述性 0.970、GLS > 0.99）；KS4 14 mm 新息斜率 0.054 / 0.051。
  - Stage 0：事件侧 σ\* 录制级缺口（zgz_local 0.943 … lpc_local 1.013）、σ_mask\* 0.998–1.028、betas ±1 SD → Δσ\* ±0.05、标签 > 30 Hz 功率 ≤ 1.7 %。
  - R3 battle：72 位评审只读文档与源码，未读数据（`#mv2-r3-result-20261025`）。
  - 工程预检：`configs/mv2/mv2_slr_precheck.yaml` 60 步训练（GPU 2，`outputs/mv2/precheck/`），只验收能否训练、峰值显存 10 GB、约 2.65 batch/s；日志里打印的训练 / 验证损失不作任何读数。各工具 `--smoke` 只打印形状、计数与耗时；S3 检查器在预检权重上的冒烟只打印字段名。
  - 账本既有的 DT2 `rootanc` / `rooth`（`#dt2-verdict`）、S42 carry（`#s42-verdict`）、XDI、SBA、AURA、MRF、AGG778 读数，各探针的先验引用都标出锚点。

### §1 数据、单位与统计

- 受试者：FIT7（ch、lfz、lpc、lr、ly、lyh、lyq）用于拟合与标定；SELECT（ycy、ylf）用于门。只用基础录制 `<subj>_local / _global`，不用 `_v2/_v3/_v4` 训练变体。zgz 只在 S4 主行（一次性报告）与 M5 之外的明确条目中出现；探针一律不读 zgz。
- S39 在 9 人全量上训练，SELECT 对 S39 是样本内、对 FIT7 上拟合的读出是留出；所有门的解读按此限定。
- 区间：（序列, 1 s）块自助 2,000 次，种子 20261026；受试者只有 2 名 SELECT，不对受试者做自助；种子以"三种子各自同号"作附加条件。zgz 上的模型比较以种子为单位（KS0：种子 SD local 0.379 / global 0.261 mm；3 对 3 MDE local 1.149 / global 0.793 mm，是规划近似、非门槛）。
- 每个工具在任何读数前调用 `tools/mv2/common.py::verify_protocol()`（文件、检查点、本节文本的 SHA），结果文件独占写入。

### §2 各方案的探针与门

（下列各节为各工具的预注册；"门"一律在 FIT7 / SELECT 上判；阈值来源写在各节。）
