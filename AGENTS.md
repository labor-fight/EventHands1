# Agent conventions for EventHands1

## 代码清理（用户 2026-11-01"这批代码有点乱，你清理删除一下，屎山代码你也清理一下"）

- **第一步已做（2026-11-01 07:33–07:55 UTC）：删除已结案研究的代码，最终 1,067 个文件（约 21.3 万行），61 个顶层目录整个消失。**
  - 判定方法：静态引用图，包括导入（含函数内导入、相对导入、importlib 字符串）和代码字符串、.sh、.json、.yaml 里的文件路径；不计 docstring 与注释。
  - 删的是从下列活代码根都走不到的文件：`model/`；核心 semkine（train / dataset / eval_track / xyz_mesh）；S45 线（`tools/s45*`、`semkine/s45*`、`configs/s45*`）；C1 / C1SF；`tools/make_s36_row.py` 等出表工具；`deploy/`；共用驱动在跑队列里的全部路径。
  - 具体包括：工具、配置、测试；35 个 semkine 模块（xkf*、sba*、rgq、slr、mv2_rootself 等）；`.experiments/`；三个 S43 C++ 源文件。
  - 另删了两个清理前就坏掉的测试：`tests/test_xag_diagnostics.py`、`tests/test_xa_assoc.py`。它们导入 10-19 已删的 `tools.xag` / `configs/xa`。
  - 删除后重跑保留的测试，又发现静态图看不到的依赖：在 `sys.path` 插入后用裸名导入（`tools/dt`、`tools/tracking`），以及分段拼出的配置路径（`ROOT / "configs" / "dt" / …`）。
    - 已从归档逐字节恢复 160 个文件：S39 的推理栈（`tools/tracking/evalx.py`、`tools/dt/filter_eval.py` 及其辅助）；`configs/dt`、`dt2`、`rt`、`semkine`、`semkine_recipes`；`tools/rt/make_configs.py`、`tools/config_utils.py`、`tools/select_checkpoint.py`；`tools/xcl/sources/step1_source_labels.py`。
    - 另删了只测已结案 XVM 的 `tests/test_xvm_ladder.py`。
  - 验收：保留的 76 个测试文件删除前后逐项相同：前 1155 过 / 65 跳过 / 10 失败，后 1173 过 / 65 跳过 / 10 失败，没有任何用例改变结果。
    - 那 10 个失败清理前就在：test_xag_model ×5、test_dt_model2 ×2、test_dt_model4 ×1、test_dt_render ×1、test_xcl_depth ×1。
    - 多出的 18 个通过是对比脚本按类名归档的口径差，两次都在、都通过。
- **保留：**
  - 所有活代码字节不变：S45-TR / RT / AT / SM 的冻结清单仍可校验，第 12–14 轮照常跑。
  - 参考基线配置：S39 的 `configs/dt` / `dt2` / `rt`、S43 `configs/s43/s43n_sp.yaml`、`configs/eventhands_*.yaml`。
  - 数据 / MANO / 视频 / 配置工具，`evcreader/`、`transfer_check/`、`docs/`。
  - `outputs/` 与根目录 `logs/`：账本引用其中的日志作证据。
- **恢复：**
  - 归档在 `outputs/deleted_20261101/code_cleanup/`：`inventory.json` 记逐文件 SHA256，`files/` 是副本，另有 `deletion_receipt.json`。
  - 被跟踪的文件也可以 `git checkout 6fe3ca7 -- <路径>` 恢复。`.experiments/` 里 17 个 git 忽略的日志 / `.pt` 只有归档副本。
- **下文各节里已删的路径**（如 `tools/kstar/`、`tools/aura/`、`tools/xkf*/`、`configs/dt/` 的其它配置）是历史引用。C1 复现、XCL、XAG-paired、C1SF 的冻结协议在清理前就已与当前树不符（107 / 2 / 107 / 5 个文件），要复现须检出各自的提交。
- **第二步已做（2026-11-01 10:56–12:10 UTC，`#code-cleanup-step2-20261101`；用户选择"批次跑完后做"与"不提交"）：活代码去屎山。**
  - 备份：`outputs/deleted_20261101/live_code_backup_20261101T105636Z/`（806 个文件，逐文件 SHA，已复核）。不做 git 提交。
  - 核心：删去约 40 组已结案研究的可选功能与调试 agent log，13 个 semkine 模块、18 个测试、145 个配置随之删除；`model/model.py` 3050 → 985 行。S45 线运行时打补丁要用的接口与属性全部保留。
  - 守卫：`tests/test_core_fingerprint.py`（九个必须保留的配置在 CPU 上的逐位指纹）、`tests/conftest.py`。以后改核心代码先跑它。
  - 冻结哈希：已结案 S45 线研究的协议不改；核心文件的新 SHA 记在 `tools/frozen_amendments.json`，各研究的校验经 `tools.frozen.accepts`。**以后再改这些核心文件，须在该 JSON 里追加旧 → 新 SHA 并附逐位等价证据**，否则那些研究的工具会拒绝运行。
  - `tools/s45sm/`：65 个版本脚本合并为 20 个模块，入口 `python tools/s45sm/cli.py <动词>`（旧脚本 → 动词对照见 `tools/s45sm/legacy.py`）。旧脚本归档 `outputs/s45sm/20261031/legacy_code/`，代码身份 `manifests/p1_core.json`。研究汇总用 `cli.py decide --as-of-round 14`（r15 / r16 是工程验收轮，不是研究轮）。
  - 验收全部通过：
    - CPU 指纹逐位相同；全部测试 812 过、0 失败；31 个已发布读数由新工具重算一致。
    - GPU：S45 50 步预检、pfat0pdz 50 步预检（经新 `train`）与原记录逐位相同；S45 与 xt0_srdwa 的主行重评精度数组完全相同。
  - 验收中发现并修正新 `train` 入口把动词留在 argv 里的错误（`#s45sm-tools-cutover-amend1`）。

## ACU · 对应偏差 × 更新规则 × 因果异步（用户 2026-11-04 /goal，约 15:25 UTC）

- **目标：**
  - 两条线分别定位、设计、验证，最后合成严格因果、支持异步事件输入的统一方案：A = 事件—MANO 对应的结构性偏差；B = 状态更新纠错不足、闭环累积。
  - 精度：五项全面优于 S43（引擎三种子 12.81 / 8.02 / 10.77 / 6.64 / 10.24），并提升错误前态恢复与长序列稳定性。
  - 成本：时延、FLOPs、参数（含路由器或其替代）与 S43 同量级。
  - 资源：只用 GPU 2–7、CPU 全量。
- **写定约束：**
  - 部署候选必须因果异步：事件 e_i 的编码、关联、更新只用它自己、过去事件的缓存与当时的 MANO 状态；
  - 须测"前缀不变"与"分包不变"；旧 r_s2d 与标签路由只作诊断对照；
  - 2×2（原 / 新观测 × 原 / 新更新）须适配重训、匹配预算；测试时替换只作诊断；
  - **满足因果异步或改变观测接口若须大改，先调研并给出最小方案、公式、收益依据、成本与风险，征得用户同意再做。**
- **已做（`docs/EXPERIMENT_RECORDS.md#acu-diag-20261104`，零训练、CPU、描述，未登记臂）：**
  - 工具 `tools/acu/diag_*.py`（D1 逐步分解、D1b 证据量、D2 接口信息、D3 因果路由、D4 可观测方向、D5 投影 vs 缩小）；产物 `outputs/acu/20261104/diag/`。
  - 要点：
    - 所有网络（含 B2-gt、S43）从正确前态出发都乱改，闭环每步只纠回自身误差的 3–13 %，闭环误差 ≈ 同包反复的不动点；
    - 只用身份、由前态轮廓关联（CA），路由器带进的偏差少一半到四分之三（D2）；
    - r_s2d 改成只看过去事件的滑窗图，身份精度只降 1–5 个百分点（D3）；
    - 按可观测方向挡住学到的更新不如均匀缩小（D5，负结果）。
  - 审计出 B2 部署路径的三处未来信息：整包事件图路由；`stride_sample` 依赖包总数；token 的 SAE 回填用包跨度，包边界清零。
- **用户决定（`#acu-decisions-20261104`）：** 身份来源 = PR 点式异步图路由器（完全不用事件图 CNN）；批准 CA；B 线只筛 B1；登记后自动推进。
- **已登记、已冻结（`outputs/acu/20261104/manifests/`）：**
  - s1a：a1 = B2-gt + CA，a2 = B2 + CA（`#acu-s1-prereg-20261104`）；
  - s1b：b1 = B2 + B1（`#acu-s1b-prereg-20261104`）；
  - s1c：PR 路由器（`#acu-s1c-prereg-20261104`）；
  - s2：异步 2×2 {PFA, CA} × {原, B1}，PR 身份，因果管线（`#acu-s2-prereg-20261104`，门 = s1c `PR_OK`）。
- **第 1a 阶段结果（`#acu-s1a-results-20261104`，18:52 UTC）：**
  - a1 `IFACE_LOSES_INFO`：CA + 标签身份 RA 10.73，对 B2-gt 5.98；
  - a2 `CA_NOT_EFFECTIVE`：CA + r_s2d 为 12.75，对 B2 3407 11.47；
  - 单步几乎不变，损失在闭环：按前态最近轮廓关联，测得的残差偏向前态，纠错弱，从均值手型出发不收敛；
  - 不做允许的那次接口修订，2×2 照登记跑。换观测接口须先征得用户同意。
- **第 1b 阶段结果（`#acu-s1b-results-20261104`，21:03 UTC）：** b1 `B1_NOT_EFFECTIVE`，RA 11.92 对 B2 11.47。
  - 单步注入变小，E3 好 0.15 mm；但运动增益一起变小，闭环纠回份额不变（根 0.03、手指 0.06），放大 2.05 → 2.19。
  - 2×2 的"新更新"格照登记仍用 B1。B 线换别的机制须用户决定。
- **代码：**
  - `semkine/acu_ca.py`（CA）、`acu_b1.py`（B1）、`acu_pr.py`（PR + 因果步长）、`acu_hist.py`（50 ms 历史）、`acu_causal.py`（因果逐事件管线）；
  - `tools/acu/`（arms*、train*、run*、jobs*、decide*、router_pr、chain_r2、freeze）；
  - 测试 `tests/test_acu*.py`，每个文件单独进程跑：全局钩子同进程会残留。
- **驱动：** `tools/acu/chain_r2.py`（`setsid nohup … run`，16:31 UTC 起，pid 记在 `outputs/acu/20261104/chain.pid`）。
  - `policy.json` 的 `gpus` = [4, 5, 6, 7]，`train_pairs` = [[4, 5], [6, 7]]（`#acu-amend1-20261104`）：
    - 另一用户（lmx）占着 GPU 0–3（18:41–21:06 UTC 离开过 GPU 3，那段时间 GPU 3 只跑单卡读数）；
    - lmx 再离开 GPU 3 时可以加回 3（只跑单卡读数），并在账本该节登记。
  - 作业按阶段 `chain_r2.py add`；失败保留、不重试；fresh-only；DRAIN / STOP 只在紧急时用。
- **注意：**
  - 新阶段的文件放新文件；已冻结的 `tools/acu/*.py` 不改。
  - 第 2 阶段训练依赖 PR 检查点（`outputs/acu/20261104/routers/pr/pr_final.ckpt` 及 `receipts/pr_formal.json`）。
- **工程修订 2（`#acu-amend2-20261104`）：**
  - 故障：PR 用 float64 `cumsum`，CUDA 确定性模式下报错。`pr:pre` 失败，连带 s1c / s2 / s3 全部跳过。
  - 修正：`semkine/acu_pr_det.py` 改用 int64 定点前缀和。
  - 凡运行 PR 的作业都经包装入口：`tools/acu/router_prd.py`、`train_s2d.py`、`python -m tools.acu.detrun <模块>`、规划器 `jobs_s3r2.py`。
  - 作业以 `r2:` 前缀重新登记（`jobs/amend2.json`，冻结 `manifests/amend2.json`），失败现场在 `preflight_failed/`。
  - 新的 GPU 入口冻结前先在 GPU 上、确定性模式下跑通一次。CPU 冒烟发现不了 CUDA 的确定性报错。
- **第 1c 阶段结果（`#acu-s1c-results-20261104`，22:49 UTC）：** PR `PR_WEAK`（等价 PASS；zgz_local 部位 / 手指组精度 0.456 / 0.625，r_s2d 0.703 / 0.858；ylf_local 同样）。主线第 2 阶段按规则关闭，PR2 修订放行。
- **PR 修订 PR2（`#acu-pr2-prereg-20261104`，22:10 UTC，写于任何 PR / PR2 读数之前）：**
  - 起因：PR 训练日志部位精度 0.56，r_s2d 同步 0.79，`PR_WEAK` 几乎确定。
  - PR2 = PR + 因果多尺度形状上下文（`semkine/acu_pr2.py`，3 个尺度 × 5 × 5 格的计数与固定特征，40 ms，int64 前缀和），成本与 PR 持平（50,020 参数）。
  - 设计试点在 `outputs/acu/20261104/pilot_pr2/`：第 500 步训练部位精度 0.54，PR 0.42。
  - 冻结工具在命名空间 q 下原样运行：`tools/acu/ns_q.py` 把根改到 `outputs/acu/20261104q`、`configs/acu_20261104q`（与研究根同级），PR → PR2。
  - 入口：`router_pr2.py`、`train_s2q.py`、`python -m tools.acu.qrun <模块>`、`jobs_s3q.py`。命名空间入口必须先 `ns_q.enter()`，再 import 其它 tools.acu 模块。
  - 作业 `jobs/q.json`（键前缀 `q:`，50 个），冻结 `manifests/pr2.json`。
  - `q:gate0` 只在主 s1c 为 `PR_WEAK` 时放行；PR2 也 `PR_WEAK` 时第 2 阶段关闭，须问用户身份来源。
- **PR2 停止（`#acu-stop-pr2-20261105`；用户 2026-11-05 约 01:40 UTC："现在停止训练吧，默认PR_WEAK，现在重新分析"）：**
  - 01:41:23 UTC touch `outputs/acu/20261104/STOP`，01:42:26 驱动记 STOPPED_PRESERVED；PR2 训练在第 3119 步被终止，没有检查点。
  - PR2 按用户决定视作 `PR_WEAK`（不是读数），第 2、3 阶段未跑，45 个 `q:` 作业未开始。
  - 不重启驱动、不追加作业；STOP 文件只在用户下令时移除。之后转入重新分析（`#acu-reanalysis-20261105`）。
- **重新分析（`#acu-reanalysis-20261105`，零训练、描述，未登记臂）：**
  - 工具 `outputs/acu/20261104/analysis/reanalysis/router_reliability.py`（同 router_breakdown 的节点与解码，加置信度与"包 × 部位"相干偏移）。
  - 置信度挑得出部位错误（r_s2d 前 30 % 部位精度 0.95–1.00），挑不出部位内相干偏移：新受试者 5.4–6.1 mm，路由器训练受试者 3.4 mm，σ* 不是原因。
  - PR 欠拟合：在自己的训练受试者上部位精度也只有 0.48–0.49。
  - A、B 是同一问题：证据有相干偏差，主干学到低增益。主干用九人训练，其中八人的路由是样本内质量，比测试时干净（推断）。
  - 建议先做训练—测试一致的相干偏移剂量—反应（标签 + 合成部位内位移场 0 / 3 / 6 mm，三训练约 5 h），据此决定逐事件对应 + 异步路由器这条路是否继续。未登记，待用户决定。
- **注意：** 等待脚本不要用 `pgrep -f` / `grep` 匹配会出现在等待命令自身里的字符串；本次 D4 的等待循环自匹配过一次，已停掉。

## B2UB · B2 的分层理想实验与上限（用户 2026-11-04："U1 + U2，连 U3 一起做；沿用 GPU 2–7"）

- **范围：** B2 = F + gshare（PFO2 三种子）。U1 测试时理想路由（零训练，B2 三种子）；U2 闭环逐自由度钳位（零训练，新任务 `e1c`）；U3 B2-gt = B2 配方改为标签路由训练（只训种子 3407）。只用 GPU 2–7、CPU 全量；标签 / GT 只作诊断；全部读数是描述，不改 PFO / PFO2 / EVF 的判读（"路由器线永久关闭"不重开）。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104`（写定读法、K1–K4 逐位工程验收、差距分解、上限表）。冻结 `outputs/b2ub/20261104/manifests/r1.json`（`python -m tools.b2ub.freeze write|verify --stage r1`）；已冻结文件不改，新行为放新文件。
- **代码：** `tools/b2ub/`（common、arms、train、engine（e1c）、run（u1 / f1label / u2 / accept / u3 / u3c / check）、analyze、jobs、chain、freeze）、`tests/test_b2ub.py`、`configs/b2ub_20261104/`；产物 `outputs/b2ub/20261104/`。CPU 测试设 `CUDA_VISIBLE_DEVICES=""`。
- **注意：** `tools/pfo/nets.py` 的混合路由（label_loc / label_part / label_onhand）默认用 sr1 当路由器；`tools/b2ub/run.py` 在本进程里把 `_router_for` 重新绑定到配置的部署路由器（B2 为 r_s2d）。B2-gt 训练时 `semkine.xag.sample_sources` 保持原生（读包里的 XAG 伪源），关联钩子与 B2 训练相同（`semkine/s45sm.py` 的 PFA + 空射线关联）。
- **驱动：** `tools/b2ub/chain.py`（EVF 调度器原样；`setsid nohup … run`；`drain` / STOP 仅紧急；fresh-only；失败保留不重试）。作业 `python -m tools.b2ub.jobs r1`；判读 `python -m tools.b2ub.analyze all`（最终作业，只算一次）。
- **结果（`#b2ub-results-20261104`，2026-11-04 04:53 UTC COMPLETE）：** 118 个作业全过，K1–K4 逐位 PASS，U3 训练验收 PASS，无修订。
  - U1 `TEST_ROUTE_INSUFFICIENT`：B2 + 测试时标签路由三种子 13.70 / 7.97 / 11.68 / 6.28 / RA 10.63，local 仍高于 S43；收益全在事件落到哪个面（部位 + 部位内定位），手上判定无用。
  - U2：部署 B2 的闭环损失（自身闭环 11.57 对前态全钳位 5.49）约一半根旋转、一半手指，平移约 10 %，近似可加。
  - U3 `BOUND_BELOW_S43`（单种子）：B2-gt + 标签路由 8.05 / 4.17 / 6.75 / 3.55 / RA 5.98，五项比 S43 低 3.1–4.8 mm，比 pfagtdz 低 0.5–0.8 mm；配部署路由 r_s2d 为 RA 15.44（暴露偏差 +3.97）。
  - 差距分解（3407，RA 11.47 → 4.18）：测试时路由 1.10（15 %）、训练时路由 4.39（60 %）、剩余闭环 1.80（25 %）。P0-B 只量了测试时那一块；路由器线的关闭不因此改，是否重新评估由用户决定。不自动开新臂。
  - 事后分析（`#b2ub-why-20261104`，描述）：闭环放大几乎只由训练时路由决定（r_s2d 训练 2.05 / 1.93，标签训练 1.43 / 1.43，不论测试路由）。可部署设计（S43、B2、pfasrdz、EVF）都落在"单步准 ↔ 放大小"的同一条权衡上，RA 都在 10.2 以上；改主干只能沿这条线挪，要越过须改进入主干的信息。

## PFO · 以 pfasrdz 为对象的瓶颈定位与定向小改（用户 2026-11-03 /goal）

- **目标：** 保持 pfasrdz 框架（SRNet 事件观测 → PFA 关联 → 事件—顶点关系编码与聚合 → MANO MeshGNN → LBS 槽池化 → 状态增量更新），先定位精度与效率瓶颈，再做有针对性的小改动，使五项精度（MPJPE / MPVPE local、global、RA）全面优于 S43（三种子 12.80 / 8.01 / 10.77 / 6.65 / 10.24），参数、端到端时延、FLOPs（含路由器）与 S43 同量级。
  - /goal 即执行授权：只用 GPU 2–7、CPU 全量，各阶段预注册、冻结、验收后自动推进；关键结论用配对多种子；标签 / 真值只作诊断；路由不依赖前态。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103`（协议、预算、第 1 阶段零训练诊断）、`#pfo-p2a-prereg-20261103`（基础臂 B 与 MeshGNN 对照）。冻结清单 `outputs/pfo/20261103/manifests/p1.json`、`p2a.json`；已冻结文件不再改，新行为放新文件。
- **预算（写定）：** 与 S43 主表实现（稀疏执行器）同块、同标志、同 600 个 lyq 包重测；参数、实际执行 MACs、时延（主行口径与端到端 p50 / p99）倍率分别 ≤ 1.5 为"接近"，≤ 2.0 为"同量级"。时延的确定性 / 默认两种标志都测，比较时两边同一种。
- **代码：** `tools/pfo/`（common、nets、engine、cost、check、jobs、chain；第 2 阶段 arms、train、run2、jobs2）、`semkine/pfo_diag.py`（测试时干预与阶段截取）、`semkine/pfo_trunk.py`（MeshGNN 变体 base / nomp / self / rewire）、`tests/test_pfo*.py`、`configs/pfo_20261103/`；产物 `outputs/pfo/20261103/`。
- **驱动：** `tools/pfo/chain.py`（`setsid nohup … run`，队列 `queue.json`、状态 `status.json`、日志 `logs/`）；作业只用 `chain.py add` 追加；失败保留不重试；停机 `touch outputs/pfo/20261103/STOP`。训练分两条链（一次一个训练），留一对卡给读数；成本块是 GPU 4 独占作业。
- **训练：** `tools/pfo/train.py --config configs/pfo_20261103/pfo_<arm>_s<seed>.yaml`（semkine/train.py + sr1 路由钩子 + 主干变体 + 确定性标志）；种子改 `SEED`（初值与数据流），余弦衰减；读数经 `python -m tools.pfo.run2 eval|cost|accept`。
- **已知（写入前已见）：** pfasrdz 的单步（E3 overall 5.75）比 S43 教师强制（8.29）准，闭环放大 2.17 对 1.24，差距在闭环。成本烟测：pfasrdz 的 MACs 约为 S43 执行器的 3 倍（MeshGNN 153 M、路由器稠密主干 165 M），时延约持平；MANO FK 在两者中都约 3 ms。
- **工程修订 1（`#pfo-amend1-20261103`）：** G0 的 S43 闭环逐位判据不适用（evalx 用非确定性内核，闭环放大舍入差），改为段首 + 单步 + 三种子均值判据（G0b PASS）。之后与 S43 的比较一律用引擎重算的三种子数（12.81 / 8.02 / 10.77 / 6.64 / 10.24）。
- **第 1 阶段结果（`#pfo-p1-results-20261103`）：**
  - 瓶颈是闭环纠错增益偏低：local 每步保留约一半的根 / 手指注入误差，S43 0.08–0.16。
  - global 的差距是根旋转长时漂移。
  - 起作用的路由误差是部位内定位（尤其手掌）与部位，不是手上判定或阈值。
  - 训练时的路由质量同时决定主干能用多好的证据。
  - 探针没看到未读出的线性信息。
- **第 2 阶段登记：** `#pfo-p2a-prereg-20261103`（B + MeshGNN 对照：nomp / self / rewire）、`#pfo-p2r-prereg-20261103`（低成本 SRNet：r_s2d / r_s2dns / r_s2dw，sr1 配方只改结构）、`#pfo-p2b-prereg-20261103`（noise / bank / t0ptrain / noprev / pos）、`#pfo-p2c-prereg-20261103`（nohot）。
  - 新臂只加 `tools/pfo/arms_<阶段>.json`，入口 `train3.py`、`run5.py`、`jobs4.py`。
  - 最终行由 `tools/pfo/final.py` 生成（`outputs/semkine/pfo_<名>_main_row.json`，可直接用 `tools/report_table.py` 出表）。
- **第 2k 阶段（`#pfo-p2k-prereg-20261103`，零训练、只用 CPU；用户 11-03 下令，由另一会话登记）：** 显式几何更新的杀死测试。在 sr1 / T0P / 标签路由的 PFA 重投影残差上解一步高斯—牛顿，对照 B。代码 `tools/pfo/gn.py`、`tests/test_pfo_gn.py`，产物 `outputs/pfo/20261103/gn/`，作业键前缀 `p2k:`。写定规则：sr1 路由下一步 E3 > B 的 E3 均值，或 zgz_local R15 / F15 保留率 ≥ 0.45 → 不做这个大改。
  - 结果（`#pfo-p2k-results-20261103`）：**KILL**。sr1 下 local 保留率 R15 0.86、F15 0.99；48 个网格点里没有一个能既纠错又不被带偏。标签路由下根 / 平移一步几乎纠完（0.21 / 0.09），手指在任何设置下都 ≥ 0.68。不做显式几何更新。
- **最终阶段（`#pfo-p3-plan-20261103`、`#pfo-p3-prereg-20261103`）：**
  - F = `f1`：B 去掉消息传递（trunk4 "nomp"）+ 训练不注入热像素（`nohot`）+ 路由器 `r_s2d`，三种子。逐一去掉 `f1hot`（两种子）。
  - 附加臂（每个相对 F 只改一处，两种子与 F 配对，规则按五项判据，`tools/pfo/analyze3.py rules`）：
    - `#pfo-p3b-prereg-20261103`：`f1bank`、`f1w50`；
    - `#pfo-p3c-prereg-20261103`：`f1self`（逐顶点深度，部署走 `semkine/pfo_trunk5.py` + `tools/pfo/run9.py`，先过等价检查）、`f1noise`。
  - 部署路由器零训练替换（`#pfo-p3d-prereg-20261103`）：`r_s2dkd` / `r_s2dw` / `r_wide` / `r_s2d_nh` 换进 F 三种子，按五项判据决定。
  - 最终成本块：`tools/pfo/jobs6.py cost --after <最后的作业>`（GPU 4 独占，两种标志）。等最后一批训练开训后再追加，避免它就绪后挡住其它 GPU 作业。
  - 闭环检查表：`tools/pfo/analyze3.py checks`；主行：`tools/pfo/final.py row` / `compare`。
  - 2026-11-03 04:55 UTC（`#pfo-p3e-prereg-20261103`，写于 F 种子 3408 / 3409 读数前）：F 种子 3407 RA 11.64 比同种子 nohot（10.63）差 1.0 mm，而 F 只多两处成本驱动的结构决定；登记逐一恢复臂 `f1mp`（恢复消息传递）与 `f1sr1`（路由器回 sr1），各两种子对 F 配对，规则"恢复后好 ≥ 0.3 mm 且两种子同向才撤销"；后果（低成本传播 l1 / gshare 替代、p3d 候选重判、补三种子）已写定。读数工具 `tools/pfo/analyze4.py rules`；`tools/pfo/arms_p3e.json` 冻结于 `manifests/p3e.json`。
  - 2b / 2c 筛选读齐：noise、bank、t0ptrain、noprev、pos 都不晋级，只有 nohot 晋级；逐一去掉消融 f1hot 两种子确认 nohot 留在 F（RA −0.34 / −0.94，收益全在 local）。
  - **2026-11-03 10:01 UTC 用户指令"现在停止goal"（`#pfo-stop-20261103`）：** 用驱动自己的 STOP 停止，10:02 记 STOPPED_PRESERVED；f1self 3407 / 3408（第 5500 步）与 f1noise 3407（第 4000 步）被终止、现场保留、不评测；96 个作业未开始（f1self / f1noise 评测、f1noise 3408、p3e 两臂、f1bank 3409、r_s2dw × 3409、最终成本块）。不再登记、不追加；续做须用户下令、先写工程修订并换新驱动（chain.py 是 fresh-only），被终止的臂 fresh 重跑。停止时判读状态与交付物状态见该节。
  - 附加臂 f1bank 两种子按写定规则采纳（`#pfo-p3b-bank-20261103`，08:50 UTC；只靠 global 8.00 < 8.02 的 0.02 mm 边际，local 反而差 1.5 mm，如实标注）；种子 3409 已排队（`p3b2:*`）。路由器替换（p3d）在最终配置确定后再对其三种子追加。
  - **F 三种子结果（`#pfo-p3-f-results-20261103`，06:15 UTC）：15.09 / 9.16 / 12.20 / 7.49 / RA 11.92，五项判据 0 / 5，不达标**；保留率与 B 相同（local R15 0.44–0.46），差距仍在闭环饱和水平。最终配置等附加臂 / 路由器替换 / 恢复臂读完后按写定规则选。
  - 最终成本块 `p3:cost:*`（`jobs/p3_cost.json`）已于 03:45 UTC 追加，挂在 18 个训练验收之后。模块贡献 / 成本表：`python -m tools.pfo.modtable`（`analysis/modules.md`）。
- **2026-11-03 09:3x UTC 用户指令"现在能不能停止goal，你根据目前的结论和进展重新设计goal"（`#pfo-stop-20261103`）：** 停止 PFO goal。驱动 STOP 由脱离会话的等待进程在三个在跑训练（f1self 3407 / 3408、f1noise 3407）退出后触发（最迟 12:30 UTC），驱动记 STOPPED_PRESERVED；之后不登记新轮、不追加作业、不重启驱动。未跑：f1self / f1noise 评测、f1bank 3409、p3e、r_s2dw 第三种子、成本块；最终配置未选。停止时最好的三种子可部署行 F + r_s2dkd 14.77 / 9.02 / 11.82 / 7.38 / RA 11.69（0 / 5），目标未达成。chain.py 是 fresh-only，后继研究要用新驱动目录或新状态文件。

## EVF · 事件点 × 前态虚拟帧的闭环跟踪（用户 2026-11-03 /goal，约 16:10 UTC）

- **目标：** 三种子五项均值全部低于 S43 引擎重算 12.81 / 8.02 / 10.77 / 6.64 / 10.24，成本与 S43 同量级（逐项倍率，≤ 1.5 接近、≤ 2.0 同量级）。
  - 原始事件点直接进网络（不用路由器 / PFA / LNES / 事件图像）；前态只经两条路进网络：虚拟帧（前态 MANO 渲染，节点按自己的像素读取，显式"事件减预测"）与可选 prev48。不得用前态硬性决定事件归属。
  - 损失一字不改（so3_trans_fk 1 / 1 / 2 …），其余沿用 B2 配方。须先征得用户同意：改损失、改数据或划分、测试时用标签、改输入方案。驱动重启须用户下令。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104`（输入实现、A / anovf、判据、轮次与后果、避坑、交付）。冻结 `outputs/evf/20261104/manifests/r<k>.json`（`python -m tools.evf.freeze write|verify --stage r<k>`，列表 `tools/evf/manifest_r<k>.json`）；已冻结文件不改，新行为放新文件。
- **规则：** 过筛（种子 3407，2000 步，对 B2 3407 第 2000 步）= zgz_local R15 与 F15 都低 ≥ 0.10 且 E3 不差 > 1.0 mm；过筛才训满 6000（门作业，退出码 3 = 规则关闭）；单种子有效 = 对 B2 3407 第 6000 步 RA 与 G 都好 ≥ 0.5 mm；有效后才加 3408 / 3409；采纳 = 三种子 G 好 ≥ 0.5 且无种子 RA 差 > 0.1；不用"低于 S43 的项数"。
- **代码：** `semkine/evf_render.py`（精确光栅化、轮廓、融合 Triton JFA）、`evf_nodes.py`（前缀稳定节点、特征）、`evf_net.py`（A 与投影平移头）、`evf_model.py`（EVFModel，BaseModel 损失不变）、`evf_async.py`（异步运行时）；`tools/evf/`（common、arms + `arms_r<k>.json`、train、nets（引擎的 evf 网络类型）、run、replay、decide、jobs、chain、cost、freeze）；`tests/test_evf.py`。运行 CPU 测试时设 `CUDA_VISIBLE_DEVICES=""`（否则会用到 GPU 0）。
- **驱动：** `tools/evf/chain.py`（PFO2 调度器逻辑；`setsid nohup … run`；`drain` / `policy.json` / STOP 仅紧急；fresh-only；作业按阶段追加）。训练用 `tools/evf/train.py --config configs/evf_20261104/evf_<arm>_s<seed>[_<phase>].yaml`（screen = 同配方第 2000 步停；Trainer deterministic；expandable_segments）。读数 `python -m tools.evf.run eval|evalb2|accept`。
- **注意：** 训练包装器经 `--max-steps` 改步数，DDP 子进程会用改过的 argv 重跑包装器（已处理）；手动起 DDP 时每个进程用不同的 MASTER_PORT。
- **结案（2026-11-03 19:05 UTC，停止点 1；交付 `#evf-close-20261104`）：** 第 1 轮 A（点版本）、anovf（对照），第 2 轮 b（图版本）、ait2（两次迭代）、alarge（加大扰动），种子 3407 第 2000 步都不过筛（对 B2 第 2000 步 R15 / F15 0.48 / 0.48、E3 5.78）。
  - EVF 臂单步最好（E3 4.95–5.43，E3 − 恒等 1.6–2.1），但保留率 0.56–0.67，高于 B2，闭环放大 3.2–3.7。
  - f1 显示它们反复喂同一包会越走越远（均值手型起点 C 1.18–1.19 > 1；B2 0.85，S43 0.71）：学到的是"事件中的运动"，不是"预测与事件对齐"。
  - 去掉虚拟帧：E3 9.36，平移完全不纠。
  - 成本（同块）：A 参数 0.07×、MACs 0.21× / 满载 1.27×、时延 1.6×；b 时延 3.1–3.3×。A 的异步—同步等价 PASS（2.7e-7）。
  - 驱动 COMPLETE（19:39）。续做（运动外推通道、去时间特征的诊断臂、不动点训练）须用户下令、换新驱动。

## PFO2 · 以 F 为基底的闭环增益（用户 2026-11-03 /goal，约 10:30 UTC）

- **目标：** 以 PFO 的 F（pfasrdz 配方 + 余弦、训练不注入热像素、r_s2d）为基底，保持原框架，MeshGNN 以单轮 l1 或共享门控 gshare 保留；三种子五项均值全部低于 S43 引擎重算的 12.81 / 8.02 / 10.77 / 6.64 / 10.24；参数、执行 MACs、端到端时延对 S43 稀疏执行器的倍率逐项报（≤ 1.5 接近、≤ 2.0 同量级），同块同标志同 600 个 lyq 包重测。
  - /goal 即执行授权：只用 GPU 2–7、CPU 全量，各阶段预注册、冻结、验收后自动推进；大改（换表示 / 关联范式 / 主干 / 状态更新框架）先论证、征求同意；驱动重启须用户下令。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103`（协议、写定规则、第 0 批训练、探针判据、第 1 阶段结构、停止点、交付）；探针实现细节 `#pfo2-p0-prereg-20261103`（在任何探针读数前写）。冻结清单 `outputs/pfo2/20261103/manifests/s<k>.json`（`python -m tools.pfo2.freeze write|verify --stage s<k>`），已冻结文件不改，新行为放新文件。
- **规则要点：** 采纳 / 替换 = 五项正差距和改善 ≥ 0.5 mm 且无种子 RA 变差 > 0.1；禁止"低于 S43 的项数"判据；单种子只用于杀死，晋级 / 采纳 / 承重要两种子配对同向，达标要三种子。
- **代码：** `tools/pfo2/`（common、chain、arms + `arms_<stage>.json`、train、run、replay、decide、jobs、freeze）、`semkine/pfo2_*.py`、`tests/test_pfo2*.py`、`configs/pfo2_20261103/`；产物 `outputs/pfo2/20261103/`。PFO 的引擎 / 成本工具 / F 与 S43 读数只读沿用（`tools/pfo2/run.py` 把引擎输出根改到 PFO2），PFO 目录不写入。
- **驱动：** `tools/pfo2/chain.py`（`setsid nohup … run`；`chain.py drain` = 只停派发、在跑作业跑完后退出；STOP 仅紧急；`policy.json` 与作业 `pairs` 只决定训练用哪对卡；失败保留不重试；fresh-only）。作业只用 `chain.py add` 追加；`chain.py summary` 看状态。门作业（`decide gate`）退出码 3 = 被写定规则关闭，不是故障。
- **第 0 批：** f1mp 两种子（两种子都好 ≥ 0.3 mm → 基底 gshare，否则 l1）→ 基底 B2 三种子（验收：预检与每种子前 500 步与单项变体实现 rl1 / rgs 逐位相同）；f1self（第 5500 步）/ f1noise（第 4000 步）是 PFO 停止时截断的权重，只评测、只作描述。
- **工程修订 1（`#pfo2-amend1-20261103`）：** semkine/train.py 建 Trainer 时不传 `deterministic`，Lightning 会调 `torch.use_deterministic_algorithms(False)`，所以 **PFO 的全部训练（含 F）都是非确定性内核**（同种子只共享初值与数据流）。f1mp 保留 F 的程序（`tools/pfo2/train.py`）以便配对；B2 与之后所有 PFO2 臂用 `tools/pfo2/train_det.py` / `train_s1.py`（Trainer `deterministic=True`，逐位可复现）。F 复现门改为程序等价检查（`tools/pfo2/equiv.py`）。以后任何新训练入口都要给 Trainer 传 deterministic=True。
- **工程修订 2（`#pfo2-amend2-20261103`）：** 同一检查点的多个读数并发写同一 spec 文件会竞争；新入口一律用 `tools/pfo2/specs.py`（`run2.py`、`run_s1.py`）。
- **已定（`#pfo2-s0-results-20261103`）：** B2 = F + gshare（`fgs`；f1mp 对 F −0.51 / −0.67 mm）。P0-A FAIL、P0-B σ* 4.42 mm → 路由器线永久关闭、P0-C ρ_feat 0.525 → A1 / A2 / A3 都不开；P0-D 待 B2 三种子（`p0d2:*`）。
- **第 1 阶段（`#pfo2-s1-prereg-20261103`）：** GNN (a) 位移头（解析 LBS 雅可比最小二乘）与 (c) 差分消息，各 full / self / rewire 两种子（`tools/pfo2/arms1_s1.json`，注册表 `reg1.py` 读 `arms1_*.json`——**不要**把新臂文件命名成 `arms_*.json`，冻结的 `arms.py` 会去读它）；规则 `decide_s1.py --plan tools/pfo2/plan_s1.json`。P0-D 不过则停止点 (i) 触发，立即 `chain.py drain`。
- **调度注意：** 驱动不把评测放在有训练的卡上，训练队列一直有就绪作业时评测会饿死；要读数时用 `policy.json` 暂时拿掉一对训练卡。大批描述性 GPU 作业会挡住排在后面的关键作业（队列顺序派发）。

- **修订 3（`#pfo2-amend3-20261103`，用户 11-03 约 15:45 UTC："尽量不要跑那么多种子，确定有效果以后再多种子验证"）：** 15:48 UTC 起 `policy.json` 训练对为空（不开新训练，读数与 P0-D 照常）。新臂先只训种子 3407 筛选（RA 比 B2 3407 好 ≥ 0.3 mm，或 local R15 / F15 都降 ≥ 0.05，且 E3 不差 > 0.3 mm）才加种子 3408；三种子只给最终配置。第 1 阶段是否继续，待用户在 PFO2 与新方案之间决定。

- **PFO2 结案（`#pfo2-close-20261103`，17:25 UTC）：** 按修订 3 与停止点 (i)（P0-D CLOSED，一阶增益 −0.49）收尾；17:13 drain，17:19 DRAINED；B2 三种子 15.00 / 8.58 / 12.38 / 6.92 / RA 11.57（G 5.98，目标未达成）；gdisp 两种子只作描述（RA −0.63 / +0.73，无效应）。PFO2 驱动不再重启；B2 读数只读沿用为 EVF 对照。

## T1A-UB · S45-T1a 框架的上限估计与预算内逼近（用户 2026-11-02 目标）

- **目标：** 保持 S45-T1a 的框架（事件输入—事件与 MANO 交互—网格特征处理—状态增量更新），估计其可达上限，并在参数、时延、FLOPs 三项各自不超过 T1a 的前提下优化内部设计。
  - 区分"标签辅助诊断上界"与"相同部署输入及预算下的经验可达上限"。
  - 允许模块重排、替换、合并与预算重分配；不另起大型架构，不加部署输入，部署时不用真值。
- **授权（AskUserQuestion）：**
  - "GPU 2–7，自动推进"：每轮预注册、冻结、预检后自动开训、评测、读数，出结果后分析并登记下一轮，不逐轮等确认；GPU 0–1 不用。
  - "XAG 伪标签可以，只用在训练"：可作监督或蒸馏；测试时用标签路由的结果只算诊断上界。
- **预算（T1a `s45tr_t1a_s3407` 第 6000 步）：** 参数 494,323（含测试路由器）；MACs 732,061,888（thop + 函数式乘积 + 路由器）；时延 ≤ 同块重测的 T1a（发布值 15.59 ms）。训练时读的路由不部署、不计入。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#t1aub-r1-prereg-20261102`；每轮 `tools/t1aub/registry_r<k>.json` + `outputs/t1aub/20261102/manifests/r<k>.json`，写于该轮任何读数之前；评测 / 判读工具在第一个读数前另冻结。已冻结文件不再改，新行为放新文件。
- **代码边界：** `semkine/t1aub.py`（opt-in 子类：读出宽度重建、标签位置特征 oracle、EdgeConv 重参数化）、`tools/t1aub/`（common / train / accept / evaluate / decide）、`tests/test_t1aub.py`、`configs/t1aub_20261102/`；路由沿用 S45-SM 的路由器与 PFA（`semkine/s45sm*.py`，不改）。
- **驱动：** 共用 `tools/s45rt/chain_r2.py`，键前缀 `ub:`；声明内存：读 XAG 源的训练 260、其余训练 130、评测 15 GiB。训练加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。失败保留、不重试；STOP 会连带其它研究，先问用户。
- **第 1 轮：**
  - 上限阶梯：gt（标签路由 + PFA）、gtf（标签位置作特征）、t0p（T0P 路由 + PFA）、dz；
  - 预算可行性：t0ps16、t0pnoeg、t0pnoeg20；
  - 零训练单元：候选 A = t0ps16_srdwa（491,534 参数），候选 B = t0pnoeg20_srdwa（491,290 参数）。
  - 前提：t1areplay 须逐位复现 T1a 的 50 步预检，否则不开正式训练（17:16 UTC 通过）。
  - 补充 R1P（`#t1aub-r1p-prereg-20261102`）：零训练信息保留探针（事件编码级、槽级 f4），`tools/t1aub/probe.py`。
- **第 2 轮（`#t1aub-r2-prereg-20261102`，零训练）：** 前态在路由器前两名部位内仲裁关联（`semkine/t1aub_arb.py`；point σ = 2 px / ray 门控 / 部位优先参照），六个单元；第 2 轮起的单元由 `tools/t1aub/r2.py`、`evaluate2.py`、`registry2_r<k>.json` 解析，冻结 `manifests/rnd2.json`。
- **第 3 轮（`#t1aub-r3-prereg-20261102`）：** T1a 在 s45samp 采样流上的配对复现 t1arep，并写定最终配对比较规则。
- **第 4 轮（`#t1aub-r4-prereg-20261102`）：** 候选 B 主干上的训练侧机制 b4cos / b4n2 / b4dz / b4match，每臂只改一处；N2 经 `tools/t1aub/train2.py` + `train2_r4.json`。除 b4cos 外，正式训练排在 gt 正式训练之后。
- **第 5 轮（最终轮，工具已写、未登记）：** `tools/t1aub/final.py` 按写定规则生成：X = 候选 B 主干 + `summary_r4_amend1.json` 的采纳机制，两条采样流（legacy 主干 / `x5rep`），部署单元 x5l / x5d + srdwkda；诊断单元 x5g（主干 + 测试时标签路由，`tools/t1aub/r5.py`、`evaluate5.py`）；候选 A 在预算内且有非 N2 采纳机制时另训 a5。判读 `decide_final.py`（第 3 轮配对规则 + 最佳可部署 + 差距分解）。预注册模板在会话 scratchpad，等 `ub:r4:decide:amend1` 后填入账本、`final.py register`、`final.py jobs --block b5`、`chain_r2.py add`。
- **第 6 轮（`#t1aub-r6-prereg-20261102`，05:31 UTC 登记）：** gtf 为 FEATURE_SUFFICES（ρ_feat 0.83–0.84，C 0.56），按第 1 轮写定的下一步训 rf6 = 候选 B 主干 + srdwkda 预测的表面位置作输入特征 + S45 自身射线关联（`tools/t1aub/r6.py`、`train6.py`、`evaluate6.py`、`breakdown6.py`、`decide6.py`、`round6.py`）。对 b4match 判 FEAT_BETTER / NULL / WORSE / MIXED，不改第 5 轮的最终配置判读。b6 块与判读作业由 `final.py jobs` 随第 5 轮一并生成（经 evaluate7 / breakdown7，依赖 `ub:r5:decide`）；不要用冻结的 `round6.py block`（它写的是 evaluate6）。
- **工程修订 1（`#t1aub-amend1-rows-20261102`，07:4x UTC）：** 07:28 b1 第一个带路由器的行（t0p）在尾延迟一步失败，连带 b1 / b2 / b4 各行与第 1、2、4 轮判读被跳过。原因是 S45-SM 的路由器缓存 `_net` 用原地的 `nn.Module.to`，`make_main_row` 在 CPU 上数 MACs 时把 CUDA 条目指向的那个路由器搬走了。修复 `tools/t1aub/devfix.py`（按设备各一份拷贝，数值不变）+ `evaluate7.py` / `breakdown7.py`；b1 第一次的三行归档到 `outputs/t1aub/20261102/rows_attempt1/b1/`，整块以原名重测；重登记的键 `:row2:` / `:bd2:` / 判读 `:r2`（`jobs/amend1_rows.json`）。轨迹与读出不受影响。以后冻结主行工具前，先用带路由器的网络把整条主行跑通一次。
- **2026-11-02 11:21 UTC 用户指令"现在停吧"（`#t1aub-stop-20261102`）：** 用驱动自己的 STOP 停止，11:22 记 STOPPED_PRESERVED；驱动里无其它研究的作业。第 5 轮未登记，rf6 未训练，第 1、2、4、6 轮写定判读未计算；b1 重测完成到 gtf 行（dz 行被终止）。不再登记新轮、不追加作业；续做须用户下令，并换新驱动（chain_r2 是 fresh-only）且先写工程修订。描述性小结见账本该节。
- **已有设计分析（ylf，不读 zgz）：** `outputs/t1aub/20261102/analysis/design_ylf/`。前态正确时它分部位强于事件；换成闭环状态后，任何前态仲裁都比路由器单独差。
- **备用代码（未登记）：** `semkine/t1aub_router.py`（dwt / dwr / dweg 路由器变体）与 `tools/t1aub/router.py`（srdwkd 配方训练）。
- **加新轮注意：**
  - `registry_r<k>.json` 必须与其清单在同一进程里写入，否则 `verify_protocol` 会让所有作业失败；第 5、6 轮的登记先写本轮的文件清单 `r<k>x.json`（round k），再写登记，最后写轮清单，任何时刻都不会出现"已登记未冻结"；
  - 冻结清单把 `tools/frozen_amendments.json` 也冻住，所以本研究的文件不能走旧 → 新 SHA 修订，新行为一律放新文件；
  - 260 GiB 的标签训练会被 130 GiB 的作业饿住；
  - 已就绪的 exclusive 作业会挡住新的 pair / gpu 作业，且只在没有任何 GPU 作业时开始，所以行块要依赖上一块的末尾，并尽量依赖本轮自己的全部读数（否则会等别的训练跑完）；
  - `chain_r2.py add` 要求依赖键已存在；依赖尚未登记轮次的作业要等那一轮登记时再追加。

## S45 · XYZ-C1 + SO(3)+平移+FK损失（用户2026-10-29固定的后续研究基底）

- **用户明确采用：** `C1SF so3（seed3407）` 固定命名为 **S45**，本课题后续修改以它为基底与对照。不是DZ/ZR组合，不改变原XYZ-C1的关联、φ、778聚合、MeshGNN或51D加性更新。原C1／S39／S43工件保留。
- **唯一身份：** `outputs/s45/baseline.json`；来源 `c1sf_so3_s3407`，固定step6000，权重 `outputs/c1sf/20261029/runs/c1sf_so3_s3407/c1sf_so3_s3407-step=6000.ckpt`，SHA256 `d0164d55660c31184e8fd0b70f22f56cf9a7a064a216073862c9c56653d7dce0`。不是另一个文件 `repro_final.ckpt` 的SHA。配置来源 `configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml`，快照 `outputs/s45/20261029/config_snapshot.yaml`；主行别名 `outputs/semkine/s45_main_row.json`。
- **配方：** `LOSS.TYPE=so3_trans_fk`，ROT／TRANS／FK权重1／1／2；原双卡2×512、Adam .004、warmup500后constant、6000步、BF16、legacy_none、原数据和噪声合同。新臂须新run／输出目录，并区分fresh配对重训与从S45微调，不覆盖S45或重选末权重。
- **已完成读数：** local／global／递推RA为23.51／9.37／15.94，单seed描述性；采用是用户的研究基底决定，不是新的可靠性通过。历史撤回的 `s45_meshrootjoints` 只按原run名辨识，不与本S45混用。
- **继续分析范围：** 前态条件关联／关系φ→51D纠错更新。允许核对已结案C1SF和已有缓存的事后描述；本次没有授权新增训练或新模型评测。原C1上的归因数值不能直接迁移到S45；后续机制实验须以S45固定身份设计、预注册并遵循已批准资源范围。
- **2026-10-29 S45-LOC（用户"我要具体知道问题出在哪里。是我的仿真event的问题还是event和mano的关联问题还是后续的7维以及13维特征问题"）：** 零训练、冻结 S45 的定位读数，不训练、不改冻结文件。预注册 `docs/EXPERIMENT_RECORDS.md#s45loc-prereg-20261029`（写于读数前，披露了无模型的录制事件率 / 深度统计与工程烟测），冻结 `outputs/s45loc/20261029/protocol.json`（`tools/s45loc/freeze.py`），串联 `tools/s45loc/chain.py`（状态 `status.json`），只用 GPU 4–7（每卡两进程）+ CPU。33 个任务：zgz 基线须逐位复现 C1SF 缓存；训练样本内 E3 / E1；zn 平移与虚拟变焦移植；σ* 事件缩放；事件稀疏；token 时间列分位映射；XAG 伪源 oracle 清洗；E3 扰动保留率。工具 `tools/s45loc/`，测试 `tests/test_s45loc.py`。S39 / S43 / C1 / S45 不变。
- **2026-10-29 S45-LOC 结果（`#s45loc-results-20261029`，附录 V；修订 1 = GT 姿态 / 平移钳位，写于主读数后，`#s45loc-amend1-20261029`）：** zgz 基线逐位复现 C1SF 缓存，上一轮数字全部复核一致。
  - 单步：超出 X = 6.11，其中 29 % 结构性（训练样本内同样超出，"假更新"）；zgz 特有 4.33 的 60 % 来自 13 维里的 zn 绝对深度越界（zgz_local 0.725 m 对训练 ≤ 0.67 m；只平移 zn −2.62，镜像 +3.67）。仿真事件（稀疏模拟 +0.31、σ* 修正 −0.03、token 时间列分位映射 +0.21 无帮助）与关联错误（oracle 清洗反而 +1.7）都可忽略。
  - 闭环：local 闭环差是结构性的（训练样本内 E1 中位 22.9 ≈ zgz 23.5），由前态姿态漂移驱动（GT 姿态钳位去掉 87–93 %，GT 平移钳位约 0）；关联塌缩与平移漂移是伴随现象。
  - 下一步（深度通道去越界 / 抑制假更新 / 自身闭环状态训练 + 记录携带不一致量）待用户决定，须重新预注册。
  - "深度通道去越界"已由 C1SF 的 dz / zr 在 C1 原损失上做过（用户 2026-10-29 提醒；只读缓存对照记在结果节末尾与附录 V.6）。
    - 单步有效：E3 15.35 → 8.59 / 9.80，深度跳变消失，几何通道转为有用。
    - 闭环无效：三者都约 24 mm。
    - 没有任何 S45（SO3）+ DZ / ZR 的组合权重。
- **2026-10-29 S45-DEEP（用户"我需要你找到深层且更本质的原因"）：** 零训练冻结 S45，只做闭式 ridge 探针。预注册 `docs/EXPERIMENT_RECORDS.md#s45deep-prereg-20261029`，冻结 `outputs/s45loc/20261029/deep/protocol.json`，工具 `tools/s45loc/deep.py`，GPU 4–7。
  - 结果（`#s45deep-results-20261029`，附录 W）：S45 的手指更新是学到的姿态先验，即向训练均值收缩（单步 10 % / zgz 17 %）加去掉不合理姿态，事件项极弱。
  - 依据：同一组正确事件反复送入，仍从 GT 漂向均值（RA 4.9 → 21 mm，k 0.10 → 0.50，按规则记"中间"）；依赖事件的部分只跟踪真实手指运动 5 %；流形内误差不由事件纠正（ρ 0.59–0.67，换错事件不变）；闭环状态的槽特征对真实手指姿态只比前态好 9 %（表示瓶颈）。闭环塌向均值姿态（k 0.47–1.01，RA ≈ 均值姿态 17–21 mm）。
  - 由此 S45 + DZ / ZR 只能修 zgz 的 zn 越界那一层；本质要同时改表示（记录携带不依赖前态的观测—预测残差）和目标（自身回放前态 + 干净前态），待用户决定并重新预注册。

## S45-TR · 在 S45 上只改一步的八个训练（用户 2026-10-30 指令"T0 改成用一对卡，然后现在进行训练GPU2-7，CPU全量"）

- **范围：** `TRAIN.md` 的 T0 / T1a / T1b / T1d / N1–N4，每个 S45 臂只改一处、fresh 6000 步、S45 严格合同，单 seed 3407 描述性。只用物理 GPU 2–7（三对），CPU 全量；GPU 0–1 不用。S39 / S43 / C1 / S45 不变。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030`；协议 `outputs/s45tr/20261030/protocol.json`（`tools/s45tr/freeze.py protocol`），后写的代码在各自预检前冻结为 `outputs/s45tr/20261030/manifests/*.json`（`freeze.py manifest`）。已冻结的文件（含 `tools/s45tr/common.py`、`chain.py`、`train.py`、`tests/test_s45tr.py`）不要再改；新臂的测试放新文件。
- **代码边界：** `semkine/s45tr_*.py`（opt-in 钩子：N1 子类、N3 / N4 前态采样器、N2 训练步、T1a 事件图 token、T1b T0 路由）、`tools/s45tr/`、`tests/test_s45tr*.py`、`configs/s45tr_20261030/`；不改 `model/model.py`、`semkine/xyz_mesh.py`、`semkine/dataset.py`、`semkine/train.py` 等冻结文件。
- **驱动：** `tools/s45tr/chain.py run`（后台，状态 `outputs/s45tr/20261030/status.json`，日志 `logs/`）；新作业只用 `chain.py add <jobs.json>` 追加，不改不删；`touch outputs/s45tr/20261030/STOP` 停止。前提 P0：`s45replay` 50 步逐位复现 S45 第 1–3 步，不过则不开任何正式训练。失败保留、不重试。
- **排期：** 实际顺序 N1 / N3 / N2 → T0 → T1a → T1b（T0 在留出受试者 ylf 上过门：模板误差中位 ≤ 8 mm 且优于 S45 射线关联）→ T1d → N4（状态库 + N2 / N3 读数后）。主行时延最后在 GPU 4 串行测。
- **工程修订 1–5**（`#s45tr-amend1-20261030` … `#s45tr-amend5-20261030`，都写于各自读数之前）：T1d / T1b 用 `tools/s45tr/train_xag.py`（退役 `tools.xag` 审计的替身）；N2、T1a、N4 进程加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（N4 是修订 5：正式训练第 2500 步后显存碎片，fresh 重跑 `formal:n4:r2`）；T1a / T0 的事件图改按行取邻居 + 逐轮重算（数学不变）；T0 面分类 dtype。失败现场在 `outputs/s45tr/20261030/preflight_failed/` 与 `formal_failed/`；队列重排过两次（只改先后，`queue.json` 的 `reorders`）。
- **T0 门（2026-10-30 05:09，读一次）：FAIL**——留出受试者 ylf 上模板误差中位 14.90 mm（门 8.0 mm；S45 射线关联 15.01 mm），按预注册 T1b 不训练（作业跳过，T0 权重与 `t0/gate.json` 保留）。

## S45-CT · 对应扰动训练 σ16 / π（用户 2026-10-30 指令"沿用 GPU 2–7，开始训练"）

- **范围：** 在 S45-TR T1d 上只改面路由要替换的事件的（面，重心）：σ16（换成模板测地 8–24 mm 的随机面，即 P2 noise16）、π（包内循环打乱，即 P2 shuffled），各 fresh 6000 步 + `t1dreplay` 50 步复现；区分"T1d 的收益来自对应信息（A）"与"标签伪影（B1 精确 / B2 GT 占据泄露）"，决定做不做可部署的对应预测器 T0′。两臂用标签、不可部署。只用物理 GPU 2–7（训练一对卡，评测 / 读数每卡至多两个进程，主行 GPU 4 串行）+ CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 不变。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#s45ct-prereg-20261030`；协议 `outputs/s45ct/20261030/protocol.json`（`tools/s45ct/run.py freeze`）；驱动 `tools/s45ct/run.py run`（状态 `status.json`），失败保留、不重试；`touch outputs/s45ct/20261030/STOP` 停止。第 0 步验收（T1d 复现、扰动审计、代理复现、评测复现、区分能力）不过不开正式训练。代码只在 `semkine/s45ct_route.py`、`tools/s45ct/`、`tests/test_s45ct.py`、`configs/s45ct_20261030/`；冻结后不再改。
- **结果汇总：** 仓库根目录 `Results.md`（TRAIN.md 全部训练的结果）。
- **2026-10-30 结果（`#s45ct-results-20261030`）：判 B2。**
  - 样本内收益保留比例 R：σ16 0.99、π 0.88；两臂从均值手型出发都收敛；zgz local 三个晚期点都优于 S45。
  - T1d 的收益主要来自标签给出的 GT 占据（每包事件落在哪些 GT 面上），逐事件配对只占约 0.1–0.3，且主要用在单步精度上。
  - 按预注册：停 T0′，不再把 T1d 当作对应信息的上界。
  - 下一步零训练探针"事件能否预测 GT 占据"：须新预注册，待用户决定。
  - S45-RF P2 的冻结代理在 π 上严重低估训练后的保留（I 0.074 对 R 0.88），不能当作训练后耐受度的预测。
  - σ16 / π 主行时延在 pxm 训练同时进行时测得，不作解读。
- **2026-10-30 结论分级（`#s45-grading-20261030`，事后描述，判读不改）：**
  - π 确实拆掉了逐事件配对（换面中位 119 mm，88 % 跨部位），B2 站得住；"配对占 0.1–0.3"只作描述。
  - 单 seed 噪声底：zgz local ≈ 5 mm、global ≈ 8 mm；样本内 ≈ 2–3 mm（近乎空改动的 pxc 三个检查点都稳定好约 1–2 mm）。低于噪声底的差只作描述。
  - 三类分级（可下结论 / 只作描述 / 不作结论）见 `Results.md` 的"结论分级"一节。

## S45-PX · 前态顶点位置进记录（用户 2026-10-30 指令"反正现在卡也空着，你再给我训一个版本，S45，但是单个event从7维到13维的时候也把上一个状态的MANO的x，y，z也当作已知信息放在维度里，你看看怎么设计。"）

- **范围：** 在 S45 上只改一处：每条（事件, 顶点）记录 13 → 16 维，多出关联顶点在前态网格上的位置。pxc 用相机 XYZ（字面版本；顶点相机坐标本可由像素 + zn + dXYZ 算出，只是改成线性可得）；pxm 相对前态腕点（MANO 系去平移，记录原本没有的"顶点在手上的哪里"）。经零初值、无偏置的 3 → 64 映射接到 φ 第一层（XVM sign 接口），初始函数与 S45 逐位相同。各 fresh 6000 步、S45 严格合同、单 seed 3407 描述性、可部署。只用物理 GPU 2–7（S45-CT 运行期间只用 6–7、一次一个训练、可用内存 ≥ 160 GiB 才开训、不测主行），CPU 全量；GPU 0–1 不用。S39 / S43 / C1 / S45 不变。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#s45px-prereg-20261030`；协议 `outputs/s45px/20261030/protocol.json`（`tools/s45px/run.py freeze`）；驱动 `tools/s45px/run.py run`（状态 `status.json`），失败保留、不重试；`touch outputs/s45px/20261030/STOP` 停止。预检须逐位复现 S45 第 1 步、第 2 步离开 S45，不过不开正式训练。判读：zgz XCL 单 seed 规则 + 样本内三个晚期点 ≥ 2.5 mm 且 ≥ 7 / 9 录制，判读表 A_MANO / A_CAMERA / B_SAMPLE_ONLY / E_ZGZ_ONLY / C_NULL / REVERSE 及下一步预先写定（`tools/s45px/decide.py`）。代码只在 `semkine/s45px.py`、`tools/s45px/`、`tests/test_s45px.py`、`configs/s45px_20261030/`；冻结后不再改。
- **工程修订 1（`#s45px-amend1-20261030`，写于读数之前）：** S45 的 `xyz.fc2` 初值为零，第 1 步上游梯度全为零，所以臂到第 3 步才离开 S45；验收判据改为"第 1、2 步逐位等于 S45、第 3 步离开"（`accept_r1.py`，回执 `receipts_r1/`），正式训练入口 `train_r1.py`，续跑驱动 `run_r2.py`（状态 `status_r2.json`），冻结清单 `manifests/amend1.json`；原 FAIL 回执保留，两个预检训练沿用。
- **2026-10-30 结果（`#s45px-results-20261030`）：判 C_NULL。**
  - 两臂都在噪声内。pxc 的 zgz 与 S45 持平，样本内 +0.9–2.1 mm。
  - pxm 6000 步 zgz local −3.7、样本内 −4.6 mm，但 5500 步 zgz local 反而差 4.7 mm，样本内 5000 步差 1.0 mm。
  - 机制读数没有出现 H_PX 预期的三项同时下降。
  - 按预注册：缺顶点位置不是瓶颈（与 S45-DEEP 一致），停止这条线。
  - 主行时延 14.15 / 14.28 ms 为单独测量。

## S45-RT · 可部署路由（用户 2026-10-31 指令"你把方案1和方案二整理一下，整理到train里。先部署着"）

- **范围：** `TRAIN.md` 第 8 节。路由器只看事件、不看前态，是 S45 方法自己的模块。用户强调 S45 方法新，不用 S39 等外部姿态网络。
  - 方案一 = 重开 S45-TR 的 T1b（冻结 T0 路由，τ 0.5），S45-TR 的文件与工具原样，产物在 S45-TR 目录。
  - 方案二 = 新路由器 T0P（CNN 看整包事件图，按约 12 mm 半径、不跨部位的 279 个表面块分类，置信度筛选），加 T1p（S45 + 冻结 T0P 路由）。
  - 单 seed 3407，描述性。只用物理 GPU 2–7 + CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 与已有判读不变。
- **预注册与冻结：**
  - 预注册 `docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031`（含对 S45-TR"T0 不过门不训 T1b"的偏离登记）。
  - 协议 `outputs/s45rt/20261031/protocol.json`（`tools/s45rt/chain.py freeze`，2026-10-30 16:16 UTC）。
  - 方案二代码与配置冻结于 `outputs/s45rt/20261031/manifests/plan2.json`（`tools/s45rt/freeze_plan2.py`，16:24 UTC，块表 SHA `a36382d8…`）。冻结后不再改。
- **代码边界：** `semkine/s45rt_router.py`、`tools/s45rt/`、`tests/test_s45rt.py`（5 项 CPU 测试通过）、`configs/s45rt_20261031/`；不改任何冻结文件。
- **门与判读：**
  - T0P 门：ylf 上跨部位比例比 T0 低 ≥ 0.05，且覆盖率 ≥ 0.5；不过门不训 T1p。
  - 两臂对 S45 的判读：R（三个晚期点）、C、zgz XCL 规则 → DEPLOYABLE / IN_SAMPLE / WEAK / NULL，由 `tools/s45rt/decide.py` 写 `summary.json`。
- **驱动：** `tools/s45rt/chain.py run`（后台，状态 `outputs/s45rt/20261031/status.json`，日志 `logs/`）。
  - 作业只用 `chain.py add` 追加；失败保留、不重试；`touch outputs/s45rt/20261031/STOP` 停止。
  - 训练作业开训前要求主机可用内存 ≥ 280 GiB，且离上一个训练开训 ≥ 20 min。
  - 主行在 `barrier:all` 之后于 GPU 4 串行测。
  - S45-TR 的旧驱动（pid 95963）空转，不再往里加作业。
- **2026-10-31 00:51 停止**（`#s45rt-stop-20261031`）：
  - 用户的"先部署着"本意是先准备、不是开训。以后只在用户明确说"开始训练 / 开训"时才启动驱动。
  - 我对驱动发 SIGSTOP 想暂停派发新作业。会话后台任务因此结束，系统发 SIGTERM，驱动连带终止了 T1b 正式训练（约 500 步，现场保留）。T0P / T1p 未开始。
  - 重跑须用户下令，并先写工程修订。长驱动一律用 `setsid nohup` 与会话脱离启动，暂停 / 停止只用驱动自己的机制，不用 SIGSTOP。
- **第五批计划：** `TRAIN.md` 第 9 节，S45 误差归因 F1–F5 与条件追加项。只是计划，未预注册、未部署。
- **2026-10-31 01:00 工程修订 1 与重启**（`#s45rt-amend1-20261031`；用户："现在GPU2-7全部开始训练，全范围部署"）：
  - 被终止的 T1b 现场归档到 `outputs/s45tr/20261030/formal_failed/s45tr_t1b_s3407_r1_20261031/`。
  - 新驱动 `tools/s45rt/chain_r2.py`（`queue_r2.json` / `status_r2.json` / `logs_r2/`）用 `setsid nohup` 启动。
  - 按声明的主机内存准入：读 XAG 源的训练 260、其余训练 130、评测 15 GiB；合计 ≤ 440 GiB。
  - `tools/s45rt/barrier_r2.py` 只等显式列出的作业，S45-RT 的判读不被同队列的 S45-AT 推迟。
  - 主行在两个研究的全部读数之后测。冻结于 `manifests/amend1.json`。
- **2026-10-31 结果（`#s45rt-results-20261031`，`Results.md` 第 9 节）：T0P 门 FAIL → T1p 未训练；T1b 判 WEAK。**
  - T0P 只有覆盖率不过：0.400 < 0.5。被路由的事件跨部位 0.080（T0 0.432），模板误差中位 5.4 mm（T0 14.9）；不筛选时跨部位 0.203、面测地中位 9.4 mm，好于 σ16 锚点（描述）。
  - T1b：R 0.23 / 0.19 / 0.18（样本内好 2.8–3.6 mm），C 1.33 不收敛（S45 1.39），zgz 无支持；6000 步 local 25.41 / global 11.29。
  - 冻结代理 P2 给 T0 的 0.195 与训练后 R 一致（π 上则低估）。
  - 写定的下一步没有覆盖"T1b WEAK + T1p 未训练"；`TRAIN.md` 9.4 的"先做长窗路由器"前提与门数值不符。待用户决定，未追加作业；改阈值或训 T1p 都是偏离，须另行登记。
  - T1b 主行（10:16，GPU 4，无其它 GPU 作业）：local 25.41 / global 11.29 / RA 17.86，时延 20.58 ms；与 S45 只换数据顺序的 s45samp（27.00 / 12.11 / 19.03）同在噪声带内。

## S45-AT · S45 误差归因（用户 2026-10-31 指令"现在GPU2-7全部开始训练，全范围部署"，范围 `TRAIN.md` 第 9 节）

- **范围：** 五个 fresh 6000 步训练，各自相对基线只改一处，S45 严格合同，seed 3407，单 seed 描述性：
  - `dz` = S45 + 深度增强；
  - `t1ddz` = T1d + 深度增强；
  - `t1dnoylf` = T1d 训练去掉 ylf；
  - `s45samp` = S45 只换采样流（严格合同硬性要求 seed 3407，所以不换 seed）；
  - `w200` = 200 ms 证据窗口，前态在窗口末端前 50 ms，评测仍用 50 ms 网格、窗口在段首裁短。
  - 只用物理 GPU 2–7 + CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 不变。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031`；协议 `outputs/s45at/20261031/protocol.json`（`tools/s45at/freeze.py`，01:09）；队列清单的出处记录在 `outputs/s45at/20261031/manifests/jobs.json`。
- **代码边界：** `tools/s45at/`（`train.py` 的 ylf 过滤与 dedicated 采样都是 opt-in；`common.py install_window_patch` 只在 w200 的评测进程里生效）、`tests/test_s45at.py`（7 项通过）、`configs/s45at_20261031/`。不改冻结文件。
- **判读：** `tools/s45at/decide.py` 计算 depth_s45 / depth_t1d / interaction / window / heldout 五个效应，噪声取 |S45 − s45samp|。门槛为 max(5 / 8 / 2.5 mm, 噪声)，分类 detected / suggestive / not detected，结果写 `summary.json`。
- **驱动：** 共用 S45-RT 的 `chain_r2.py`。XAG 两臂等 `formal:t0p` 结束，其余三臂等 T1p 预检验收结束，让 S45-RT 先行。失败保留、不重试。
- **2026-10-31 06:35 工程修订 1**（`#s45at-amend1-20261031`，写于任何 w200 读数与判读之前）：
  - w200 正式训练第 213 步显存不足：物理 GPU 7 已分配 23.99 GiB、已保留未分配 16.04 GiB，与 S45-TR N2 / N4 同为分配器碎片；200 ms 窗口每批事件约为 S45 的 1.7 倍。
  - 失败现场在 `outputs/s45at/20261031/formal_failed/s45at_w200_s3407_attempt1/`。
  - 处理：加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，`formal:at:w200:r2` fresh 重跑；验收、评测、读数、判读、主行以 `:r2` 重登记，命令不变。
  - 原 `final:s45at` 会因缺 w200 报告预期失败；r2 的评测排在 `barrier:s45at` 与已登记主行之后。判读只由 `final:s45at:r2` 算一次（已被修订 2 取代）。
  - 冻结清单 `outputs/s45at/20261031/manifests/amend1.json`。
  - 核验（`#s45at-amend1-check-20261031`，描述）：r2 07:41 在 GPU 4–5 开训，前 212 步两个 rank 的输入、事件包与损失指纹与第一次逐位相同，即这个设置与换卡都不改数值。
- **2026-10-31 07:55 工程修订 2**（`#s45at-amend2-20261031`，写于任何 s45samp 读数与判读之前，按修订 1 的预先登记）：
  - s45samp 正式训练第 3701 步显存不足（物理 GPU 6：已分配 23.45 GiB、已保留未分配 15.71 GiB），同一种分配器碎片；每批事件数与 dz 同量级。
  - 失败现场（3700 步记录、第 500–3500 步检查点）在 `outputs/s45at/20261031/formal_failed/s45at_s45samp_s3407_attempt1/`，不评、不读。
  - 处理同修订 1：`formal:at:s45samp:r2` 加 expandable_segments fresh 重跑，依赖作业以 `:r2` 重登记，命令不变。
  - 判读改由 `final:s45at:r3` 算一次：s45samp r2 的 e3 评测等 `final:s45at:r2` 结束后才跑，所以 `final:s45at` 与 `final:s45at:r2` 都按预期因缺输入失败、不写任何东西。
  - 冻结清单 `outputs/s45at/20261031/manifests/amend2.json`。
  - 核验（`#s45at-amend2-check-20261031`，描述）：r2 08:10 在 GPU 2–3 开训，前 3700 步两个 rank 与第一次逐位相同，第 500–3500 步检查点逐字节相同。原 `final:s45at` 08:42 按预期因缺 s45samp 报告失败，未写出任何东西。
- **2026-10-31 结果（`#s45at-results-20261031`，`Results.md` 第 10 节；`final:s45at:r3` 10:49 算一次）：第二层是一般的新受试者迁移。**
  - 噪声（S45 − s45samp）：zgz local 3.48 / global 2.74 / 样本内 2.24 mm，门槛保持 5 / 8 / 2.5 mm。
  - heldout 检出：ylf 留出罚分 4.2–4.9 mm（T1d 下 ylf_local 7.2–7.8 → 11.7–12.1 mm）；去掉 ylf 不影响 zgz。
  - depth_t1d（+4.3 / +1.6 / +2.1 mm）与 depth_s45 都未检出；dz 只改善单步（E3 好 2.9 mm），与 C1 上同型。interaction 只在样本内 suggestive。
  - window 变差（zgz local −2.3 / −13.2 / −5.9 mm，suggestive），w200 训练不稳。
  - 从均值手型出发：t1ddz / t1dnoylf 收敛（C 0.50 / 0.51），dz / s45samp / w200 不收敛（C 1.48 / 1.14 / 1.15）。
  - 主行：t1ddz 12.82 / 6.58 / RA 9.48（不可部署，S45 系最好）；s45samp 27.00 / 12.11 / 19.03，S45 已发布的行落在偏好的一侧。
  - 写定解读不规定新训练。训练受试者固定 9 人，可动的只有受试者层面的增强。下一步待用户决定；未追加作业，共用驱动空转。

## S45-SM · 只看事件的表面路由器替换"对着前态的关联"（用户 2026-10-31 目标"S45的大框架不变……在延迟和参数略微上升的情况下，误差能降低至接近12.82 / 6.58 / RA 9.48……框架具有普适性……至少一个 Nature 级创新点"）

- **授权：** 用户确认"GPU 2–7 开训"：预注册、冻结、预检通过后自动串联训练与评测；每轮结果出来后继续分析并追加下一轮，不逐轮等确认。成本上限"参数 ≤ +30 %，时延 ≤ +2 ms"：总参数（含路由器）≤ 0.6 M，主行时延比同期重测的 S45（`s45ref`）多 ≤ 2 ms。只用物理 GPU 2–7 + CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 与已有判读不变。
- **2026-11-01 06:45 UTC 用户指令"这批训练完，先停止goal。"**（`#s45sm-stop-20261101`，写于第 12–14 轮任何读数之前）：已排队的第 12–14 轮（含第 14 轮还没开训的 pfat0pdz_rep_cos）照常跑完，评测、读数、判读照常。之后不再登记新轮、不追加作业，上面"每轮结果出来后追加下一轮"的授权到此为止；各轮写定的后续只记为待用户决定。共用驱动之后空转，不 touch STOP。
- **2026-11-01 代码清理后（`#code-cleanup-step2-20261101`）：** 本节以下提到的 `tools/s45sm/*.py` 旧脚本（train3、accept4、decide3、kind_run2、read_run 等）已归档到 `outputs/s45sm/20261031/legacy_code/`。现在一律用 `python tools/s45sm/cli.py <动词>`，对照见 `tools/s45sm/legacy.py`；新轮的登记、冻结与作业分别用 `cli.py freeze round`、`cli.py jobs standard`。
- **内容：** S45 大框架不变，只把面分配的来源换成只看当前事件包的路由器（不读前态、K、betas）。路由器：t0p（S45-RT T0P 冻结，1.6 M，超预算，只作桥接）、sr1（SRNet，137,028 参数，16 部位 + 模板坐标）。机制：face_route（T1d 原样）与 pfa（跳过射线求交，所有手上事件放到路由面，深度取路由点在前态网格上的深度）。第 1 轮四臂 t0pdz / pfat0pdz / srdz / pfasrdz，都 = S45-AT dz + 路由器。
- **预注册与冻结：** `docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031`；`outputs/s45sm/20261031/manifests/r1.json`（`tools/s45sm/freeze.py`）。每轮新登记 `tools/s45sm/registry_r<k>.json` + 新预注册小节 + `manifests/r<k>.json`，写于该轮任何读数之前；已冻结文件不再改，新行为放新文件。
- **代码边界：** `semkine/s45sm.py`（opt-in：`install` 改 `semkine.xag.sample_sources`，pfa 另改 `route_association` 与 `semkine.xyz_mesh.associate_rays`，`uninstall` 还原）、`tools/s45sm/`、`tests/test_s45sm.py`、`configs/s45sm_20261031/`；不改冻结文件。验收要求各臂的样本索引、初值与每步训练包与 S45-AT dz 逐位相同。
- **驱动：** 共用 `tools/s45rt/chain_r2.py`（`add` 追加，键前缀 `sm:`；声明内存：读 XAG 源的路由器训练 260、各臂 130、评测 15 GiB）。训练作业加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。失败保留、不重试；`touch outputs/s45rt/20261031/STOP` 停止整个共用驱动（会连带其它研究，先问用户）。
- **判读：** `tools/s45sm/decide.py --round <k>`：AT_TARGET / NEAR_TARGET / MECHANISM / WEAK / NULL；Params 列须加上路由器参数（主行工具只数模型内参数），FLOPs 加路由器 MACs。
- **2026-10-31 第 1 轮结果（`#s45sm-r1-results-20261031`）：** PFA 在两种路由器下都好于 face_route。
  - pfat0pdz（超预算）14.40 / 7.91 / 10.93，NEAR_TARGET，C 0.67。
  - 预算内 pfasrdz 15.88 / 9.54 / 12.48，WEAK，C 0.814，global 差 2.96；原实现时延 +5.3 ms。
- **第 2 轮（`#s45sm-r2-results-20261031`）：** 增强版 SR 路由器无改善，沿用 sr1。0.6 M 内路由器不能再加宽。
- **第 3 轮（`#s45sm-r3-prereg-20261031`，12:18 UTC 起）：** 在 pfasrdz 上做零参数改动：
  - spfasrdz：软 PFA，`semkine/s45sm_soft.py`；
  - pfasrbank：pfasrdz 自身闭环状态库，`tools/s45sm/bank.py`；
  - spfasrbank：两者合并。
  - 第 3 轮起的入口：`train2.py`、`accept2.py`、`evaluate2.py`、`readouts2.py`、`evaluate_fast.py --block`、`decide2.py`。
  - 主行用快速实现，时延与同一独占块里重测的 S45（`s45ref_<块>`）比。
  - 冻结：`manifests/r3.json`、`r3amend0.json`。
- **第 3 轮结果（`#s45sm-r3-results-20261031`，15:12 UTC）：三臂都是 WEAK，效应都在噪声内。**
  - pfasrbank 16.30 / 8.54 / 12.15：差距都在噪声底内，只因 C 0.84 不进 NEAR_TARGET。
  - 软 PFA 方向一致变差；状态库只让第一步变好，不动点不变。剩下的差距看路由质量。
  - 部署时延（快速实现，块 r3a / r3b）：PFA 臂与 S45 持平（13.96–14.50 对 13.99–14.21 ms），只剩参数预算约束。
- **第 4 轮（`#s45sm-r4-prereg-20261031`；结果 `#s45sm-r4-results-20261031`）：** srkd = sr1 同结构，用冻结 T0P 蒸馏（`router_train3.py`）。16:59 UTC `select_r4.py` 判未入选（ylf_local 跨部位 28.98 % 对 29.37 %，块级占据 0.450 对 0.467），沿用 sr1。
- **第 5 轮（`#s45sm-r5-prereg-20261031`，冻结 `manifests/r5.json`）：**
  - pfagtdz = t1ddz 的标签面路由 + PFA，不可部署的上界（`semkine/s45sm_oracle.py`，`tools/s45sm/*oracle*.py`）。
  - pfakdbank = pfasrbank 换 srkd，只在 srkd 入选时追加作业；srkd 未入选，不训练（`jobs/r5b.json` 只追加 pfagtdz 主行与判读）。
  - 判读 `decide3.py --round 5`。块 r5 与判读作业在选择之后追加（作业文件 `jobs/r5a.json`，后续 `r5b.json`）。
  - 登记里 `"oracles"` 键只由 `tools/s45sm/oracle.py` 读；`"conditional": "select_r4"` 的臂未入选时不得训练。
  - 结果（`#s45sm-r5-results-20261031`，20:46 UTC）：pfagtdz 8.77 / 5.00 / 6.75，越过目标，C 0.49，上界 AT。pfasrdz 离目标的差距全在路由（local 路由 +7.11、关联 −4.05）。写定 line = ROUTER，repeat 无。
- **第 6 轮（`#s45sm-r6-prereg-20261031`，冻结 `manifests/r6.json`，写于 srkd 与第 5 轮任何读数之前）：只训路由器 srpkd。**
  - srpkd = T0P 的块分类网络缩到 137,787 参数，加 T0P 蒸馏（`semkine/s45sm_srp.py`、`tools/s45sm/router_train4.py`）。
  - 依据 D1：T0P 的优势在部位内定位（块占据），跨部位只差 2–3 个百分点。
  - 冻结入口经 `tools/s45sm/srp_run.py`（环境变量 `S45SM_ENTRY`）注册路由器类型 srp 后运行。
  - 选择 `select_r6.py`：ylf_local 块占据 ≥ 参照 + 0.02，且跨部位 ≤ 参照 + 0.5 个百分点；参照为第 5 轮所用路由器。
  - 臂在第 5 轮判读后按 line 登记为第 7 轮。
  - 结果（20:47 UTC）：srpkd 未入选。ylf_local 占据 0.510 对 0.467 过门，跨部位 30.4 % 对 29.4 % 超出 +0.5 个百分点；路由器仍为 sr1。srpkd 的上下文探针也不用上下文。
- **R6C 路由器上下文探针（`#s45sm-r6c-prereg-20261031`，冻结 `manifests/r6c.json`；结果 `#s45sm-r6c-results-20261031`）：** 零训练。所有路由器都在 30–300 ms 包上训练，探针让同一组权重在推理时看最近 100–300 ms 的事件（`semkine/s45sm_ctx.py`、`tools/s45sm/router_ctx.py`、`select_r6c.py`）。
  - sr1 / t0p 都未选出窗口：local 几乎不变，global 变差。不登记推理时上下文变体；srpkd 的探针在其 D1 后照常读。
  - 推理变体的评测包装 `tools/s45sm/ctx_run.py`（测试 `tests/test_s45sm_ctxrun.py`）已写好但未冻结，只在某路由器选出窗口时随那一轮冻结。
- **第 8 轮（`#s45sm-r8-prereg-20261031`，冻结 `manifests/r8.json`，写于 srpkd、srdwkd 与第 5 轮任何读数之前）：只训路由器 srdwkd。**
  - srdwkd = T0P 的块分类网络，主干 3 × 3 卷积换成逐通道 + 1 × 1，宽度 32 / 64 / 96 / 128、FPN 40、头 72，137,291 参数；配方与蒸馏同 srpkd（`semkine/s45sm_dw.py`、`tools/s45sm/router_train5.py`）。
  - 需要 dw 的冻结入口经 `tools/s45sm/kind_run.py` 运行（先注册 srp，再注册 dw，两者可同时在场）。
  - 选择 `select_r8.py`：参照为第 6 轮之后的路由器，门槛同第 6 轮。臂在其选择后按当时写定的 line 登记（第 9 轮）。
  - 排在 srpkd 正式训练之后（一次只有一个读 XAG 源的训练）。
- **第 7 轮（`#s45sm-r7-prereg-20261031`，冻结 `manifests/r7.json`，写于任何交换读数之前）：主干 × 测试路由器交换，零训练。**
  - 依据：训练时读的路由不进部署参数；pfasrdz / pfat0pdz / pfagtdz 三个主干（训练时用 sr1 / T0P / 标签路由）结构相同，配置只差路由段。
  - 七个单元登记为第 7 轮的臂（登记键 `"trunk"`）：预算内 xsr_srp、xt0_sr1、xt0_srp、xgt_sr1、xgt_srp；超预算（描述）xsr_t0p、xgt_t0p。
  - 包装 `tools/s45sm/swap_run.py`（`S45SM_ENTRY` = evaluate2 / readouts2 / evaluate_fast）：注册 srp、dw，把单元解析到主干第 5000 / 5500 / 6000 步的检查点（按正式回执核对 SHA）与单元自己的冻结配置；回执 `swap_receipts/`。
  - 判读 `decide3.py --round 7`（经 kind_run），之后 `tools/s45sm/swap_table.py` 读一次。写定的主干选择 T*：预算内 C ≤ 0.8 中主行 RA 最低者的主干。T* 由特权路由训练时，以后的路由器改动先与 T* 零训练配对；否则照旧训练臂。
  - 偏离登记：第 5 轮写定的 ROUTER 候选 1（长窗上下文，训练形式）与 2（九人路由器）顺延。
- **第 10 轮（`#s45sm-r10-prereg-20261031`，冻结 `manifests/r10.json`，写于这些路由器的任何读数与第 7 轮判读之前）：路由器退火与部位优先解码。**
  - 退火 sr1a、srpkda、srdwkda（`tools/s45sm/router_train7.py`）：基础路由器配方原样，从其最终权重继续 2000 步，学习率 100 步升到 1e-3 后线性降到 0；结构与参数不变。串行占读 XAG 源的训练位。
  - 部位优先解码 srpkdh、srpkdah、srdwkdh、srdwkdah、t0ph（`semkine/s45sm_hier.py`，零参数）：按块概率之和选部位，再在部位内取最大块。`tools/s45sm/decode_variant.py` 放指向基础权重的符号链接与完成记录；需要这些类型的冻结入口经 `tools/s45sm/kind_run2.py` 运行。
  - 读数：D1（经 kind_run2）与 `tools/s45sm/d1_table.py`。网络层面在第 7 轮的表出来后登记为第 11 轮：T* 由特权路由训练时，七个预算内路由器各与 T* 零训练配对；否则只按写定规则训一个臂。
  - `tools/s45sm/swap_run2.py`（第 9 轮起的交换包装：任意有正式回执的主干，另注册部位优先类型）随第 9 / 11 轮冻结。
- **第 7 轮结果（`#s45sm-r7-results-20261031`，23:08 UTC）：预算内没有 NEAR / AT。**
  - 最好的是 xt0_srp（pfat0pdz 的主干 + srpkd）15.86 / 8.50 / RA 11.92：差距都在噪声底内，但 C 0.803 比门槛高 0.003，记 WEAK。
  - 暴露偏差：标签路由训练的主干配 sr1，local 比 sr1 路由训练的主干差 5.09 mm，E3 local 14.71 对 6.98。T0P 路由训练的主干没有这个问题。
  - 同一主干下 srpkd ≥ sr1，第 6 轮的 D1 代理门比网络层面严。
  - 写定 T* = pfagtdz 的主干（预算内 C ≤ 0.8 的只有两个标签主干单元），line = ROUTER，repeat 无。
  - 第 8 轮：srdwkd 未入选（跨部位 31.1 % 对 29.4 %），zgz_local 与 srpkd 几乎相同。
- **第 9、11 轮（`#s45sm-r9-prereg-20261031`，冻结 `manifests/r9.json`、`r11.json`，写于这些路由器的任何网络层面读数之前）：零训练配对。**
  - srdwkd（第 9 轮）与第 10 轮的七个预算内路由器（第 11 轮），各与 T* 配对（写定），并与 pfat0pdz 的主干 T2 配对（扩展，偏离登记：预算内 RA 最低的 xt0_srp 的主干）。十六个单元都在预算内。
  - 登记键 `"pair_trunk"`、`"pairing"`（tstar / t2）。冻结的 `tests/test_s45sm_r7.py` 只允许第 7 轮带 `"trunk"`。
  - 入口 `tools/s45sm/swap_run2.py`；描述表 `tools/s45sm/swap_table2.py`；测试 `tests/test_s45sm_r11.py`（5 项）。
  - 块 r11 在全部读数之后；`decide3.py --round 11`（经 kind_run2）一并判读第 9、11 轮（第 9 轮不单独判读），再 `swap_table2.py`。
  - 以后的路由器轮次照旧与 T* 和 T2 零训练配对。
- **第 12 轮（`#s45sm-r12-prereg-20261031`，冻结 `manifests/r12.json`，写于这些路由器的任何读数与第 9、11 轮网络层面读数之前）：九人训练的块路由器（ROUTER 候选 2）。**
  - srpkd9 = srpkd 的配方原样，训练受试者九人（`router_train4.py`，heldout = null）；srpkd9a 退火（`router_train7.py`）；srpkd9h、srpkd9ah 部位优先解码。四个都是 137,787 参数。
  - 八个单元（xgt_ / xt0_ × srp9、srp9a、srp9h、srp9ah），经 `swap_run2.py`；块 r12 依赖 `sm:r11:decide`；`decide3.py --round 12`（经 kind_run2），再 `tools/s45sm/swap_table3.py`。测试 `tests/test_s45sm_r12.py`（3 项）。
  - srpkd9 排在 srdwkda 退火之后（一次只有一个读 XAG 源的训练）。
  - 结果（`#s45sm-r12-results-20261101`，07:34 UTC）：九人效应都在噪声内（zgz_local 占据 +0.012–0.017）；T2 四个单元 NEAR_TARGET（RA 11.78–11.94），没超过 xt0_srdwa；写定 repeat = xt0_srdwa（即第 13 轮），line = ROUTER 按停止指令不登记。块 r12 的时延与代码清理的 CPU 测试重叠，只作描述。
- **第 9–11 轮结果（`#s45sm-r11-results-20261101`，2026-11-01 03:58 UTC）：预算内首次出现 NEAR_TARGET，但不稳。**
  - 五个 T2 单元（pfat0pdz 的主干 + srdwkda / srdwkdah / srpkda / srpkdah / srpkdh）为 NEAR_TARGET；最好的 xt0_srdwa 15.47 / 8.49 / RA 11.73，C 0.793，596,350 参数，时延比同块 S45 少 0.18 ms。
  - 这些单元主行彼此在噪声内（RA 11.52–11.92）；xt0_srdw / xt0_srdwh 因 C 0.815 / 0.808 记 WEAK。NEAR 与否只看 C 落在 0.8 的哪一侧。
  - 退火、部位优先、可分离卷积在网络层面都在噪声内；T2 比 T* 好 2.3–4.7 mm local（暴露偏差重现）。
  - 写定 line = ROUTER，repeat = xt0_srdwa。
- **第 13 轮（`#s45sm-r13-prereg-20261101`，冻结 `manifests/r13.json`）：写定的重复。**
  - pfat0pdz_replay：经 `tools/s45sm/train3.py` 把 pfat0pdz 的 50 步预检逐位复现（`replay3.py` 比较日志、梯度、全部状态快照），证明新入口不改数值；不过则不训重复。
  - pfat0pdz_rep：pfat0pdz 的配方只换采样流（S45-AT s45samp 的流），fresh 6000 步，`accept3.py` 验收；xt0rep_srdwa = 它的主干 + srdwkda（零训练）。
  - 判读 `tools/s45sm/repeat_read.py`（不跑 decide3 --round 13）：xt0rep_srdwa 在预算内且 NEAR / AT → REPRODUCED，否则 NOT_REPRODUCED；两行都进主表。块 r13 在 `sm:r12:decide` 之后。
  - 以后 decide3 的 repeat 落在 xt0_srdwa / pfat0pdz_rep / xt0rep_srdwa 上记"已重复"；落在其它 T2 单元上只需把 pfat0pdz_rep 的主干与该路由器零训练配对。
  - 登记键：`"replay_of"` + `"phases"`（只预检的工程复现）、`"sampler": "dedicated"` + `"repeat_of"`（重复臂）；第 13 轮起的训练入口 `train3.py` / `accept3.py` / `make_configs3.py`。
  - 结果（`#s45sm-r13-results-20261101`，10:00 UTC）：NOT_REPRODUCED。xt0rep_srdwa 15.21 / 9.83 / RA 12.33（C 0.788），差异都在噪声内，但 global 越出 NEAR 门槛，记 MECHANISM；预算内 NEAR 记为边界结果，不再重复。pfat0pdz_rep 仍为 NEAR（超预算）。
  - 工程修订 1（`#s45sm-r13-amend1-20261101`）：`repeat_read.py` / `cosine_read.py` 必须经 `tools/s45sm/read_run.py` 运行（先注册路由器种类），第一次读数作业因此失败、未写出任何东西；第 14 轮主行块与读数以 `:r2` 重登记。
- **第 14 轮（`#s45sm-r14-prereg-20261101`，冻结 `manifests/r14.json`；扩展，偏离 line 规则登记）：T2 主干的学习率余弦衰减。**
  - pfat0pdz_cos / pfat0pdz_rep_cos = pfat0pdz / pfat0pdz_rep 只加 `TRAIN.LR_SCHEDULE: cosine`（登记键 `"lr_schedule"`、`"schedule_of"`）。前 500 步因子与恒定计划相同，验收（`accept4.py`，`common4.verify_run4`）要求预检与基础臂逐位相同、正式训练前 500 步逐位相同。入口 `train4.py`（train3 原样，配方核对换成 `expected_raw4`）。
  - 单元 xt0cos_srdwa / xt0repcos_srdwa（+ srdwkda）；`cosine_read.py`：ADOPT / REJECT / MIXED。单元配置不带 `TRAIN.LR_SCHEDULE`（训练专用键，评测不读）。
  - 结果（`#s45sm-r14-results-20261101`，10:43 UTC）：ADOPT。xt0cos_srdwa 14.15 / 8.67 / RA 11.22（C 0.761，预算内 RA 最低）、xt0repcos_srdwa 15.43 / 8.60 / 11.78（C 0.738），都是 NEAR_TARGET；没有 AT（超预算 pfat0pdz_cos 只差 global 0.08 mm）。写定：以后训练主干用余弦计划。这批（第 12–14 轮）全部完成，按停止指令不再登记。

## S45-GRADE · 结论分级与样本内晚期检查点补读（用户 2026-10-30 指令"给 N2、N4、T1a 补第 5000、5500 步的训练录制读数：只评测已有权重"）

- **范围：** 不训练，只读 N2 / N4 / T1a 已有权重的 `tl_e1`（9 条训练 local 录制上的自身闭环）。新读第 5000、5500 步，按写定规则把 `#s45-grading-20261030` 里这三臂的 B 类样本内增益升为 A、留在 B 或降为 C。只用物理 GPU 2–7（每卡一个进程）+ CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 与已有判读不变。
- **预注册与冻结：**
  - 预注册 `docs/EXPERIMENT_RECORDS.md#s45grade-prereg-20261030`，写于读数之前，披露了已知的第 6000 步值；T1a 按规则最多为 weak。
  - 协议 `outputs/s45grade/20261030/insample/protocol.json`（`tools/s45grade/insample.py freeze`）。
  - 代码只在 `tools/s45grade/`；S45-TR 读数代码原样调用，只换加载器。
- **驱动与验收：**
  - 驱动 `tools/s45grade/insample.py chain`（状态 `status.json`）。失败保留、不重试。
  - 先用同一加载器重读三臂第 6000 步，须与 S45-TR 已发布数组逐位相等，才开新读数。
- **2026-10-30 结果（`#s45grade-results-20261030`）：** 验收逐位 PASS，14:31 UTC 完成。
  - N4 → A：训练录制三个检查点都好 3.1–5.4 mm、8–9 / 9。但它的状态库就来自这些训练序列，属分布内拟合；zgz 仍在噪声内。
  - N2、T1a → B：三个检查点方向一致，但幅度在门槛附近。
（用户 2026-10-30 指令"先做 P1，再做 P2"）

- **范围：** 会话分析 S45-TR 后提出的两项零训练冻结探针。P1：N4 在 S45 闭环状态上的手指纠错是不是事件驱动的（换成晚 250 ms / 1 s 的事件包做线性拉力分解，T1d 作阳性对照）；P2：冻结 T0 的路由按部位聚合后有没有无偏、有信息的新息（另写预注册，写于 P2 读数之前）。零训练，只读 S45-TR 冻结权重；只用物理 GPU 2–7（每卡至多两个进程）+ CPU；GPU 0–1 不用。S39 / S43 / C1 / S45 不变。
- **预注册与冻结：** P1 `docs/EXPERIMENT_RECORDS.md#s45rf-prereg-20261030`（含冻结前功效压测 `outputs/s45rf/20261030/p1/power.json`），协议 `outputs/s45rf/20261030/p1/protocol.json`（`tools/s45rf/p1.py freeze`）；驱动 `p1.py chain`，失败保留、不重试。代码只在 `tools/s45rf/`、`tests/test_s45rf.py`，以只读方式导入 S45-TR / XCL 的冻结模块；冻结后不再改 `tools/s45rf/p1.py`。
- **P1 结果（`#s45rf-p1-results-20261030`）：不确定（线性分解不成立）**——复现全过；T1d 对照在 250 ms 自检不过、在 1 s 过（份额 1.05）；N4 在 1 s 份额 0.53 [0.38, 0.73] 但自检不过 → 不据此开训练。描述：所有网络（含不纠错的 S45）都跟随换入事件，手指跟随 T1d > N4 > 其余；N4 的 E3 变差不是状态库滞后造成的（ΔL −0.05 mm）。
- **P2 预注册（`#s45rf-p2-prereg-20261030`，写于任何 ylf / zgz 读数之前）：** 冻结 S45 关联 + 冻结 T0，ylf（T0 留出受试者）决定、zgz 描述；按部位聚合的新息信息量 I 与锚点（部位路由 = XCL A1 无增益，面路由 = A2 / T1d 有增益）及剂量—反应（测地噪声 4–64 mm、40 % 换部位）比较。冻结前在训练受试者 ch 上只跑锚点（两轮，第一轮归档在 `p2/archive_dry_r1/`）。协议 `outputs/s45rf/20261030/p2/protocol.json`（`tools/s45rf/p2.py freeze`）。
- **P2 结果（`#s45rf-p2-results-20261030`）：T0 无用（边界）**——有效性过（锚点差 0.749）；ylf loop 前态 I(t0) 0.195 ≈ 部位路由 0.201（XCL 已知无增益）≪ 面路由 0.950；r* = 32 mm（随机误差）。按预注册不训 T1b；对应路线要靠以前态为条件的对应预测器（T0′）。描述：误差结构比逐事件精度重要（随机 16 mm 噪声仍 0.911，40 % 换部位只剩 0.234，T0 落在后一档）；S45 自己的关联在自己的闭环状态上只有 0.094；T0 在正确前态上虚假新息最高（F 0.51）。

## C1SF · XYZ-C1 训练受试者 GT 前态单步 + 三个单因素配对重训（用户 2026-10-29 指令"这三部分能不能安排上。第一步进行完，第二部和第三步分别每次只改一个，然后双卡训练。GPU4-7"）

- 来历：会话诊断 C1 的问题集中在"记录里没有位移测量 + zn 绝对深度越界 + 读出先验"，用户要三步：①零训练，冻结 C1 在训练受试者上做 GT 前态单步，与 zgz 对照，分"受试者偏差 / 结构本身有害"；②单因素配对重训 DZ（`SCALE_MODE: depth`）与 ZR（zn 相对前态根深度），各只改一处；③SO3（损失改 S39 的 SO(3)+平移+FK）。同日先按用户要求停掉 GPU 4–7 上的 S43 Stage 5 全训（15:46，产物保留，未登记续跑）。
- 预注册 `docs/EXPERIMENT_RECORDS.md#c1sf-prereg-20261029`（写于任何 C1SF 读数之前），冻结 `outputs/c1sf/20261029/protocol.json`（`tools/c1sf/freeze.py`）；配对前提 P0-E（当前代码重评 C1 与 XCL trace 逐数组相等）与 P0-T（c1replay 50 步回放与 C1 逐位相等）不过即停、向用户报告。
- 只用物理 GPU 4–7、CPU 全量；串联驱动 `tools/c1sf/chain.py`（状态 `outputs/c1sf/20261029/status.json`），结果由 `tools/c1sf/decide.py` 写 `step1.json` / `decision.json` / `report.md`。单 seed 3407 描述性，不补种子、不加长、不改门；失败保留、该臂终止。
- 代码边界：新代码只在 `tools/c1sf/`、`semkine/c1sf_zrel.py`（opt-in，替换构造用的 `XYZMeshNet` 类，不改 `semkine/xyz_mesh.py`）、`tests/test_c1sf.py`、`configs/c1sf_20261029/`；不改 `model/model.py`、`semkine/xyz_mesh.py` 与其它冻结文件。S39 / S43 / C1 不变。
- 2026-10-29 Z1 结果（`#c1sf-z1-results-20261029`）：评测配对与 v1 训练回放均逐位 PASS；local 训练受试者 Δ 中位 +2.07 对 zgz +10.94（r 0.19，以受试者偏差为主），global +1.76 对 +2.39（r 0.74，结构本身有害）；冻结干预在训练受试者与 zgz_local 上反号。
- 2026-10-29 用户"把GPU2-3也用上……调整一下训练参数"→ 协议修订 1（`#c1sf-amend1-20261029`，`protocol_r2.json`）：GPU 2–7 三对、三臂同时训练；只改数据加载资源（worker 16、prefetch 2，增强按样本播种，不改任何输入，c1replay r2 逐位核验）；trace 评测每卡两个；batch / LR / 步数 / 精度 / 判读不变。r2 驱动状态 `outputs/c1sf/20261029/status_r2.json`。
- 2026-10-29 第二、三步结果（`#c1sf-results-20261029`，单 seed 描述性，全部验收 PASS）：RA local / global / 递推 C1 27.99 / 21.06 / 24.28，dz 24.53 / 12.98 / 18.35，zr 24.01 / 13.89 / 18.60，so3 23.51 / 9.37 / 15.94；XCL 规则 local 三臂噪声带内，global dz、so3 方向支持（多半来自 C1 自身 5000→6000 步 global 13.3→21.1 的退化，只有 so3 在 5500 / 6000 两点都低于 C1 晚期最好值）。机制判"越界解释得到支持"：dz / zr 的 zgz_local E3 降 6.8 / 5.5 mm，几何置零从 C1 的 −3.3 变成 +2.5 / +5.1（3D 通道转为有用）；so3 E3 local 降 4.8 但几何置零仍 −2.0。E3 local 仍约为恒等 4.41 的两倍；腕点绝对误差 dz local 73→29 mm，三臂 global 都变差 8–19 mm。不补种子、不加长、不组合，续做待用户决定。S39 / S43 / C1 不变。

## K-Battle 与 K★ · MANO 顶点/关节算子方向（用户 2026-10-25 指令"执行这个 plan，先 CPU debug，再训练；卡能挤就挤，不能挤就排队"）

- **计划与 Battle**
  - 计划：`/home/lyq/.cursor/plans/mano_顶点关节算子十方案_battle_27767ea2.plan.md`。
  - 60 + 12 专家两轮 battle 已完成：子代理为 `claude-sonnet-5-5` high，并发 4。产物在 `outputs/kbattle_20261025/`，综合见 `docs/网络结构与可观测性诊断研究.md#k-battle-20261025`。
  - 用户方向「顶点/关节卷积 + FK 点溅射逆深度」两轮全票不能作主创新：算子层是 GLS / PLIKS 改写，溅射逆深度 ≡ 沿射线平移。
  - K★ 整包第二轮 no-go。按计划默认选择，只保留 K7（顶点锚定时间面测量）一线。
- **预注册**
  - 预注册 `docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025`，写于任何阶梯读数之前，并披露第一轮专家已做的预注册前读数。
  - 冻结在 `outputs/kstar/protocol.json`（`tools/kstar/freeze.py`）。
  - 阶梯：KS0 A/A → KS1 融合天花板（单向杀）→ KS2 陈旧性 / 持续性 → KS3 相关噪声 r\* → KS4 PA-fit → KS5 功效。全部为 CPU，门由 `tools/kstar/decide.py` 写 `decision.json`。
- **GPU 与停止**
  - 只有 KS1–KS4 全过才开 KS6 GPU 段：C1-lite debug，单 seed 3407，只用物理 GPU 4–7。
  - 挤卡条件：不占 MRF / AURA D 的时延窗口，且该卡在本作业启动后仍有 ≥ 20 GiB 空闲；否则排队。
  - 失败默认动作是终止：不二读 zgz、不加长、不改门。续做须用户另行决定并重新预注册。
- **代码边界**：新代码只在 `tools/kstar/`，只读 import 冻结模块，不改 `model/model.py`、`semkine/xyz_mesh.py` 与 R2 / MRF / AURA 冻结文件。S39 不变。
- **2026-10-25 阶梯结果（`#kstar-results-20261025`）**：KS1 杀 B（理想 GT 增量融合到 S39 只省 zgz_local 2.43 / global 1.12 mm，r\* 质量增量 0.156 mm）、KS2 过（冻结头在测当前运动，偏相关 0.46 / 0.50）、KS3 local 死（r\*\_E3 中位 0.959）、KS4 信息受限（14 mm 偏移新息斜率 0.05）、KS5 无功效 → **K★ / K7 线终止，不开 GPU，S39 不变**。两处工具键路径修订（`decide.py ks4`、`corr_rstar.py measured_profile`）登记于 `outputs/kstar/protocol.json` 的 `amendments`（`tools/kstar/amend.py`）；KS3 的真实剖面补跑只作描述（`outputs/kstar/s3_corr_rstar_measured/`）。
- **2026-10-25 用户改向**：「把十方案重新包装成专用于 mesh / MANO 卷积网络的 Nature 级创新点，结合 docs 优化计划，根据实验记录的问题提出自己的创新；初期方案仍为十个；文献 / 数据调研由用户亲自做，agent 只负责包装与提出」。与已发表方法等价（GLS / PLIKS）**不是否决理由**，只是要管理的包装风险。重设计产物 `outputs/kbattle_20261025/v2/`（简报 `00_brief.md`、阶梯读数 `01_kstar_ladder_readings.md`、挖掘 / 设计 / 聚类 / 反驳 JSON），新计划写回原计划文件（旧版快照 `outputs/kbattle_20261025/plan_snapshot.md`）。任何新读数须先新预注册。
- **2026-10-25 十方案 v2 计划已批准**（`/home/lyq/.claude/plans/enumerated-exploring-quilt.md`；用户决定登记 `docs/EXPERIMENT_RECORDS.md#mv2-decisions-20261025`）：十方案按身份分四类——机制 M1 表面提升读出 / M3 材料点事件积累 / M4 遮挡轮廓一维测量 / M5 掌部刚体先对齐；因子 M2 规范表面框架 / M8 mesh 物理一致自监督；诊断 M6 相关精度；证据 M7 投影规范归因（Stage 0）/ M9 查询时刻前沿 / M10 真实传感器 + 第二关节体。批准后只执行：R3 两轮生成式 battle（R1 60 + R2 12，`claude-sonnet-5-5` high，否定即替代，产物 `outputs/kbattle_20261025/v3/`，综合写附录 M）→ 覆盖写回原 cursor 计划文件 → Stage 0 A1–A3 只读核查（用户授权；`outputs/mv2/stage0/`，只读、不改数据 / K / 标签）→ 登记决定。Stage 1–6（预注册、CPU 探针、GPU debug、训练）另待用户指令。原 v2 设计工作流在会话重启时中断，设计 / 反驳阶段由 R3 取代。
- **2026-10-25 R3 结果**（附录 M `docs/网络结构与可观测性诊断研究.md#r3-battle-20261025`，产物 `outputs/kbattle_20261025/v3/`）：R1 60 位专家、R2 12 位反驳者全部交卷并过 schema 校验，`data_read` 全为 none；**M2 规范表面框架被替换**为「射线规范商读出 + 度量尺度协变邻域 RGQ·MSCN」（R1 探针轴 37 / 60、R2 12 / 12 no-go）；**M6 相关精度汇聚被替换**为「运动学子空间相关谱 + 并格共享分解 KSCS·GSCD」（诊断，无 GPU 臂；R2 探针轴 9 / 12 no-go）；M9 探针重写为「前缀稳定查询合同 + 上下文年龄前沿 + 发射受限成本账」；M1 两轮无 no-go，60 条条件并入计划；M3 / M4 / M5 / M8 / M10 探针按少数否定意见修订。偏离：子代理实际为会话默认模型（自定义 sonnet 代理类型未被识别）。替代方案里标【占位 / 待验证】的阈值须在预注册前以 FIT7 或账本先验定值。十方案 v2.1 已覆盖写回 cursor 计划文件（副本 `outputs/kbattle_20261025/v2/03_plan_v2.md`）。
- **2026-10-25 Stage 0 结果（`#mv2-stage0-20261025`，`outputs/mv2/stage0/`，只读、CPU、描述）**：A1 部分闭合——「约 6 % 尺度剖面差」**不是**标签—RGB-D 尺度错误（σ_mask\* 0.998–1.028）、不是 K 倍率 / 主点（Δσ\* ≤ 0.011）、不是并格相位（≤ 0.013，且 σ\* 工具不经该约定）或光栅化（≈ 0）；是事件侧 σ\* 相对标签 / RGB 的录制级缺口（zgz_local 0.943 对掩膜侧 1.015、ycy_local 0.946、zgz_global 0.961、lyq_local 0.984、lpc_local 1.013，20 录制中位 0.98，与录制深度 Spearman −0.49），并非 zgz 特有也非全局 6 %；betas 第一主成分 ±1 SD（手长 ∓4.7 %）是唯一量级够的敏感度（Δσ\* ±0.05）但未被识别为原因；v2e 事件边缘响应 / SBA σ 滋扰 / 手尺寸无法分摊，历史 runtime（refined NPZ、subject_shape、raw WiLoR、v2e run 记录）缺件，**不可闭合**。A2：1 ms 标签 > 30 Hz 功率占比 ≤ 1.7 %（样条镜像），60 Hz 样本自身 15–30 Hz 带 17–22 %（真运动与抖动不可分）。A3：无 v2e manifest / 命令 / 日志，仅 AEDAT 头 / MP4 标签 / 训练变体 sidecar 的二手声明。后续凡引用"6 %"处证据级别以此为准；M2 的 σ 扰动族形状与 M7 的慢状态主张受此约束。

## KA · 运动学图册（关节卷积与整图 LNES 卷积分开）第一步：等变性检验（用户 2026-10-28 指令"先写第一步的预注册并冻结，然后在 CPU 上跑"）

- 来历：用户要"只对相关关节做卷积 / 算子、与整图 LNES 卷积分开、为 mesh 重建设计的新方法"；agent 提出把卷积定义在前态 FK 给出的 MANO 运动学图册上（主张对深度 / 根旋转 / 关节角干扰等变），分三步检验：①零训练等变性检验 ②同算量信息探针 ③受试者学习曲线。本次只授权①。
- 预注册 `docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028`（写于任何 KA 读数之前），冻结 `outputs/kin_atlas/protocol.json`（`tools/kin_atlas/freeze.py`）；工具 `tools/kin_atlas/`（`common.py`、`synth.py`、`real.py`、`decide.py`），测试 `tests/test_kin_atlas.py`，串联 `outputs/kin_atlas/run_ka.sh`。**只用 CPU**（`nice 10`、48 进程），不读 zgz、不训练、不碰 Stage 5 五臂全训（GPU 4–7）。门：G1 合成 oop + combined atlas_kin − 较优 box ≥ 0.05；G2 真实 18 条训练录制 ≥ 0.03（下界 > 0）；G3 atlas_kin − atlas_rigid 下界 > 0 决定关节部分是否必要。G2 不过即停止此线。S39 / S43 不变。
- **2026-10-28 结果（`#ka-results-20261028`）：inconsistent**——V1–V3 过；G1 合成不过（oop + combined atlas_kin 0.959 对 box16 0.951，+0.008；box 已到 0.951、余量 0.049 < 门 0.05，冻结前未压测余量，是预注册设计缺陷，已披露、未改判）；G2 真实过（0.662 对 box24 0.604，+0.058 [0.051, 0.065]）；G3 过（对 atlas_rigid +0.026）。描述：扰动前态（10 mm + 5°）下关节优势消失（+0.003，CI 含 0），刚性 3D 提升对手框仍 +0.032；"姿态指纹"混杂未分离。按预注册只作描述，第二步不自动进入，待用户决定（结案 / G1 余量缺陷重设计 / 补姿态指纹零模型后再定）。S39 / S43 不变。

## OBS · 时间对齐审计 + 个体标定上界 + 诊断研究（用户 2026-10-26 指令"现在进行1，2，4"）

- 来历：MV2 终止后 agent 提出五个方案，用户选 1（事件—标签时间对齐审计，OBS-A）、2（个体标定上界，OBS-C）、4（把已有证据写成诊断研究，OBS-W）。**只用 CPU**（`CUDA_VISIBLE_DEVICES=""`），不用 GPU。
- 预注册 `docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026` 写于任何 OBS 读数之前，冻结 `outputs/obs/protocol.json`（`tools/obs/freeze.py`）；工具 `tools/obs/`（`oa_clock.py`、`oa_shift.py`、`oc_calib.py`，测试 `tests/test_obs.py`），产物 `outputs/obs/`，串联 `outputs/obs/run_obs.sh`。只读 import 冻结模块，不改任何冻结文件、数据、标签或 K。OBS-C 的门在 zgz 上判、读一次，训练录制只作描述。数据修正、重训或评测协议的任何改动都交用户决定。S39 不变。
- OBS-W：`docs/网络结构与可观测性诊断研究.md#obs-h-001`，由工作流从账本抽取并逐条核对证据后起草，OBS-A / OBS-C 结果出来后补入。
- 2026-10-26 结果（`#obs-verdict-20261026`）：**OBS-A** 事件与标签共用真实帧时钟——OA1 原版因 U 族零模型不对称记工具缺陷，按 §4 做唯一一次重设计 OA1-R2（读数前用合成压测修订过统计量，`#obs-a1r2-prereg-20261026`），判 20 / 20 基础流与 54 / 54 变体流在真实时钟上，OA2 旁证一致；TBC 的 e(n) 时基错位不存在，数据无需修正。**OBS-C** 中间：σ 环内校正在闭环里无效（≤ 0.09 mm）；录制级根 + 手指常偏上界 local 约 0.5、global 约 1.1 mm；zgz_local 段级常偏约 2 mm（41 段，逐段变化）；同一人另一条录制的离线标定反而变差。S39 不变。 **OBS-W** `docs/网络结构与可观测性诊断研究.md#obs-h-001` 已完成（工作流抽取 + 逐条核对 + 三视角批评与修订 + 最终核对）。
- 2026-10-26 事后机制分析（`#why-ten-schemes-20261026`，用户指令"深刻分析为什么这些方案没有效果"；全部描述，脚本 `outputs/obs/posthoc/why_*.py`）：S39 是逐帧绝对回归器（任意幅度前态扰动一步保留率 0.05–0.07，平移 0.13–0.19）；误差是新受试者上的逐帧映射偏差——E3 从完美前态出发比恒等差 4.1 mm（训练录制 +0.9），70–80 % 误差能量在 > 1 s 尺度，闭环各幅度箱 λ 0.92–0.94 是偏置而非记忆，姿态条件性（远姿态近邻余弦 0.35–0.75）并向训练先验收缩（根收缩系数 0.18–0.24）。十方案都在解"Δ 的测量"，而问题是"映射的泛化 + 环路不用前态"；探针在样本内 / E3 下恰好测不到。S39 不变。
- **2026-10-26 用户指出当前 baseline 是 S43**：`outputs/semkine/s43n_sp_tf_pert_main_row.json`（稀疏渲染–比对事件跟踪，**W50 原始协议、无滤波**：12.80 / 8.01 / 10.77 / 6.65 / RA 10.24 / 7.80 ms / 0.101 G / 2.82 M；同一网络在 S39 栈下 9.21）。此后的分析以该行为参照，并同时报 W50 原始与 S39 栈两口径；本文件的 S39 规则（主表含 S39 行、S39 不变、替换须过门）未经用户另行指示不改。机制对照 `#why-s43-baseline-20261026`：S43 的四项签名与 S39 相同（一步保留率 0.03–0.09；单步 − 恒等：训练录制 −0.16 mm、zgz +4.8 mm；慢变能量 65–87 %；向先验收缩，local 根 0.29–0.33），只是平移保留率更高（0.26–0.39）且 W50 无滤波的环路放大更大（local 1.3）。zgz 12.8 ≈ 恒等 4.3 + 单步映射偏差 4.9 + 环路放大 3.2。
- 2026-10-28 综合（`#why-s43-vertex-joint-20261028`，无新读数）：S43 上"顶点 / 关节卷积"类方案六次检验（S44 P0.4、REL、反证片段 B、场查询 A Stage 3 / R1 / Stage 4）无一超过同容量简单控制（手框 / 配额 / 恒等前态）；原因：关节语义相对"手在哪"无额外任务信息（static 对 hand-box ±0.2 mm）、50 ms 窗内没有关节级证据（null 对、SRCA）、锚定在前态上的读出被前态误差污染、S43 的误差是逐帧映射的受试者偏差。"FK 点溅射逆深度"本就是 S43 的输入。Stage 5 五臂全训（`outputs/s43_revisit_20261028/s43_full_field_R1/`）仍在跑，不读、不预判。

## MV2 · 十方案 v2.1 执行（用户 2026-10-26 指令"GPU2-7，全量cpu现在开始执行方案"）

- 授权范围：执行十方案 v2.1（`/home/lyq/.cursor/plans/mano_顶点关节算子十方案_battle_27767ea2.plan.md` §B；工作副本 `/home/lyq/.claude/plans/enumerated-exploring-quilt.md`）的 Stage 1–6：预注册 + 冻结 → CPU 零训练探针 → M1 GPU debug → 6000 步配对 → 条件臂 → 证据包。**只用物理 GPU 2–7（六卡）**，GPU 0–1 不用；CPU 全量。
- 规则沿用计划 §4：预注册（`docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026`）写于任何新读数之前并冻结于 `outputs/mv2/protocol.json`（`tools/mv2/freeze.py`）；门只在 FIT7 / SELECT 上判，zgz 只作一次性报告；各臂独立过自己的门；GPU 臂只开过了自己探针的机制臂；失败默认终止，续做须用户另行决定。
- 代码边界：新代码 opt-in——`semkine/slr.py`（`SLRModel(MNISTModel)` 子类，键 `MODEL.SLR*`）、`tools/mv2/`（探针、包装入口、freeze / decide）、`tests/test_slr.py`；不改 `model/model.py`、`semkine/xyz_mesh.py`、`semkine/train.py`、`dataset.py`、`eval_track.py`、`model/config.py`、`tools/make_s36_row.py`、`tools/report_table.py` 及 R2 / MRF / AURA / K★ 冻结文件。训练与评估经 `tools/mv2/` 包装入口 monkeypatch。S39 不变。
- 2026-10-26 02:52 正式 CPU 探针串联启动（`outputs/mv2/run_probes.sh`，日志 `outputs/mv2/logs/chain.log`，`decide.py` 写 `outputs/mv2/decisions.json`）；M1 放行后自动接 `outputs/mv2/run_gpu_m1.sh`（GPU 2 训练、GPU 3 在 2000 步跑 S3 停止规则、续到 6000 步后 S39 栈评测与九列表）。执行记录 `docs/EXPERIMENT_RECORDS.md#mv2-run-20261026`。
- 2026-10-26 探针（`#mv2-results-20261026`）：M1 PA1 / PA2 通过 → M1 放行（协议修订 1：M9 的"整模型 0 图断裂"门因冻结 S39 自身就有 6 处断裂而判工具缺陷、降为描述；M1 不作时延主张）；M2 RGQ weak、MSCN 杀 → 协议修订 2 登记 RGQ 因子臂（`semkine/rgq.py`，`outputs/mv2/run_gpu_rgq.sh`，GPU 4 / 5，等 M1 的 S3 判 CONTINUE）；M6 无 GPU 臂；M9 年龄轴死、PSQ 工具缺陷（描述）；M10 迁移三档全关。
- 2026-10-26 GPU 结果：M1 S3 CONTINUE（E3 6.16 对 S39@2000 7.12）；S4（`#mv2-m1-s4-20261026`，seed 3407 配对）local 10.32 对 10.68、global 7.78 对 7.38、RA 8.96 对 8.91 → global +0.40 不满足 S42 单种子门，local 效应 0.36 < 1.72 不补种子，M1 线终止。RGQ 因子臂在 2000 步被 S3 停止（根 carry 0.232 > 0.215）。M4 杀；M10 SAD 工具缺陷；M10 SDG 封锁真实传感器零样本主张。S39 不变。
- 2026-10-26 11:33 全部完成（`#mv2-verdict-20261026`）：M5 RCA 过 → 协议修订 4 登记"根块限定自身前态训练"臂（`semkine/mv2_rootself.py`，GPU 6 / 7，`--num-workers 40` 工程重启），2000 步被 S3 停止（E3 8.03 > 7.42）；M3 FBW 描述、SMG 工具缺陷、EXPO 杀；M8 SRCA / TBC 杀、PF 工具缺陷。没有机制臂通过 GPU 门，MV2 按失败默认终止；续做须用户另行决定并重新预注册。S39 不变。

## AURA · XYZ-C1「关联」（事件射线 × 手部外包围）隔离检验（用户 2026-10-24 批准计划）

- 用户 2026-10-24 要单独检验关联（射线–外包围相交、判断事件是否在手周围、外包围顶点作 GNN 节点，参考 Aura Mesh），批准 `/home/lyq/.claude/plans/cosmic-launching-cupcake.md`。只用物理 GPU 4–7，CPU 全量；D 段单 seed3407。A、B 立即在 CPU 上做；GPU 段（C、C2、D）排在 AGG778 R2（含其 D）与 MRF（B / C / D）全部结束之后，由 `tools/aura/run.sh gpu`（`tools/aura/orchestrate.py`）等两者的完成标记后自动执行。
- 预注册 `docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024`（写于任何 B / C / D 读数之前，披露了工程冒烟与读数前的设计细化：P2 改为全包置换 shufG、D 触发集去掉 split / contour、A3 的 float32 容差），冻结 `outputs/aura/protocol.json`（`tools/aura/freeze.py`）。阶梯 A（实现，门）→ B（算子：检测 / 对应 / 捕获范围 / 方向，描述）→ C（R2 测量工具下 155 个探针头，门按余量 H = 恒等 − T6@c1 缩放，6 个主对比 Holm 校正）→ C2（背景注入，描述）→ D（总是 none 与 shuffle，至多 1 个按选择受试者触发的可部署变体）。工具 `tools/aura/`，产物 `outputs/aura/`；新键 `MODEL.XYZ_AURA`（`semkine/aura.py`，opt-in，C1 默认路径逐位不变）。
- `semkine/xyz_mesh.py` 与 `model/model.py` 被 R2 / MRF 冻结：c1 在 `semkine/aura.py` 里重写并逐位等于库函数；`MODEL.XYZ_AURA` 接线是四处精确插入（`tools/aura/wiring.py`，接受 R2 或 MRF 接线后两种插入前状态），只在 R2 与 MRF 都完整结束后由 `tools/aura/d_stage.py apply-wiring` 应用。S39 不变。
- 2026-10-24 Stage A：A1、A2、A4–A6 通过；A3 有 2 / 2,404,491 个事件未解释——同一像素的两个外包围事件，两个候选顶点角距相对差 1.5e-6、float32 把次序反了，加权记录集合相同、q 差 1e-8（审计的平局清单缺"前 16 名内部的次序平局"，不是实现缺陷）。按预注册"A 失败即停止"：B 未读、GPU 段未启动，待用户决定是否登记偏离（`docs/EXPERIMENT_RECORDS.md#aura-a-result-20261024`）。
- 2026-10-24 用户选择"修审计后重跑 A 再继续"：A3 平局清单补入前 16 名内部的次序平局（`tools/aura/a_audit.py` 修订，协议 `amendments` 记录，原协议与 A 第一次运行已归档），正式重跑 A 通过（`#aura-a2-result-20261024`）。
- 2026-10-24 Stage B（描述，`#aura-b-result-20261024`）：工具链精确复现 P1（6.312 % / 80.564 %）。c1 在 E3 保留全部手部事件、部位一致 0.81 / 0.88（local / global），闭环 E2 降到 0.48 / 0.65；外包围对手 / 非手的区分几乎只来自距离；local 的内 / 外占据几乎不编码运动方向（AUC 0.53，σ\* 校正后 0.64，2D 邻近 + 剪影位 0.70）；度量壳 aura10 把跨指串扰从 12–15 % 降到约 2 %。GPU 段（V6 → C → C2 → D）由 `tools/aura/orchestrate.py` 后台等待 R2 与 MRF 完成后执行，状态见 `outputs/aura/orchestrator.json`。
- 2026-10-24 23:44 协议修订 2（工程修订，`#aura-wait-amendment-20261024`）：编排器判断"R2 / MRF 无进程"改为只认 argv 里的脚本路径。原来的 `pgrep -f` 会被 R2 会话中自匹配、永不退出的守望 shell 挡住。编排器在 wait 步停止后重启，未启动任何 GPU 作业。
- 2026-10-25 C / C2 / D 结果（`#aura-results-20261025`，10:35–20:26 全部完成，原排期，无修订 3）：
  - E3 与 noise 两体制有效。
  - 任务价值几乎全在占据（occ ≈ M@c1；P1 global 实质 +0.72 mm）；绑定 global 实质（+0.66），local 未定。
  - 3D 几何（P3）与部位级错派（P6）可忽略。
  - 噪声前态下 C1 像素外包围带在 zgz_local 实质优于只用表面（P4 +7.4 mm）与 Aura Mesh 式度量壳（P5 +4.9 mm）；壳节点、可见性、带宽剂量无增益；背景注入 30 % 影响 ≤ 0.04 mm。
  - 无可部署变体触发。D 的 none / shuffle 单 seed 都在噪声带内；none 的闭环绝对腕点在 zgz_global 发散（1.78 m，描述）。主表 `outputs/aura/main_table.md`。
  - 裁决：C1 关联无可部署改进，C1 / S39 不变。`model/model.py` 含 MRF + AURA 两处 opt-in 接线。

## MRF · XYZ-C1「提升感受野」（MeshGNN）隔离检验（用户 2026-10-24 批准计划）

- 用户 2026-10-24 批准 `/home/lyq/.claude/plans/xyz-c1-meshgnn-receptive-field-mrf.md` 的排期："A 可以马上在 CPU 上做；B 排在 R2 之后，并先看 R2 的有效性结果；A–C 合计约 37 GPU·h。这部分可以排。D 的种子数就一个就行。"只用物理 GPU 4–7，CPU 全量；D 段单 seed3407（XCL 单 seed 规则 + H3=0 护栏）。
- 预注册 `docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024`（写于任何读数之前，披露了 `outputs/mrf/precheck_20261024/` 的描述性预检），冻结 `outputs/mrf/protocol.json`（`tools/mrf/freeze.py`）。阶梯 A（CPU，实现正确性）→ B（AGG778 R2 测量工具下的隔离审计：消息传递 L3np − N0d、外推 L3 − L3np、尺寸 K12np − L3np 与 Gg − Gl、能力格、B-vertex）→ C（冻结模块截断 / 池化 / 稀疏度，描述）→ D（只按选择受试者触发：总是 `self3`，按 B 加 `noprop` / `ring4np` / `globalloc`，≤3 臂）。工具 `tools/mrf/`（启动 `tools/mrf/run.sh a|b|c|d`），产物 `outputs/mrf/`；新键 `MODEL.XYZ_RF`（`semkine/mrf.py`，opt-in，C1 默认路径逐位不变）。S39 不变。
- `model/model.py` 被 AGG778 R2 的协议冻结：`MODEL.XYZ_RF` 接线（`tools/mrf/wiring.py`，三处精确插入，协议记录前后 SHA）只在 R2 完整结束后由 `python tools/mrf/d_stage.py apply-wiring` 应用；在此之前不要改 `model/model.py`，B / C 也必须在应用接线之前跑完。
- 2026-10-24 A 通过（`#mrf-a-result-20261024`）。2026-10-25 01:28 读过 R2 有效性（`outputs/mrf/r2gate.json` 四个条件均可读）后启动 B → C → D 串联 `outputs/mrf/run_bcd.sh`（日志 `outputs/mrf/logs/chain.log`，pid 存 `chain.pid`）：每阶段队列完整才进下一阶段，B / C 汇总在下一阶段 GPU 作业旁跑，D 前自动 `apply-wiring`，任一失败即停。AURA 的编排器在 MRF 读数全部完成且 GPU 4–7 空闲后才接上。
- 2026-10-25 B 结果（`#mrf-b-result-20261025`）：四个条件有效性全过；H-U、H-P、H-S（4 跳）可忽略，能力格 L3 ≈ N0d，B-vertex 外推特征远不如无参数近邻；M.Gg / M.Gl 在真实特征上 12 个种子中 8 个塌缩为常数输出（V5 未覆盖），Gg − Gl 不可解读。D 只触发 `mrf_self3_a0_s3407`。S39 不变。
- 2026-10-25 C 结果（描述，`#mrf-c-result-20261025`）：冻结 C1 / A3F0 截断消息或固定支持会明显变差（A0 闭环 global 发散），重学探针无此依赖 → 共适应；事件稀释到 1/4 时 L3 − N0d 不变。08:36 D 自动开始，`model/model.py` 已应用 MRF 接线（SHA `f5716a1b…`）。
- 2026-10-25 D 与总裁决（`#mrf-d-result-20261025`）：`mrf_self3_a0_s3407` 预检、正式训练与配对核验均 PASS，10:30 全部完成，主表 `outputs/mrf/main_table.md`。
  - XCL 单 seed 规则：RA global 方向支持（E1 −9.28 mm），但几乎全在闭环（E2 −1.65、E3 −0.15；C1 自己在 5000 → 6000 步时 global 从 13.3 升到 21.1）；RA local 与腕部噪声带内，腕部绝对误差 E1 变差 16–19 mm。
  - 裁决：实现正确；有效感受野约 1 跳；消息传递、外推、尺寸无实质作用；给 GT 输入也不会利用邻域；冻结干预只反映共适应。不替换 C1 / S39。
  - `model/model.py` 保留 MRF 接线（AURA 接线接受此状态）。注意力两臂的低 LR 重跑待用户决定。S39 不变。

## AGG778 · XYZ-C1「聚合 778」隔离检验（用户 2026-10-24 批准计划）

- 用户问"单独看看聚合 778 这个模块到底有没有问题、怎么做理想实验才能充分证明"，批准计划 `/home/lyq/.claude/plans/xyz-c1-778-docs-inherited-hamming.md`，选择 A–C 后自动进入 D（D 段单 seed3407，描述性），只用物理 GPU 4–7，CPU 全量。
- 预注册 `docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024`（写于任何读数之前），冻结 `outputs/agg778/protocol.json`（`tools/agg778/freeze.py`）。阶梯 A（实现正确性）→ B（冻结 C1 / A3F0 的任务层 + 算子层信息审计，3 探针种子）→ C（稳健性，描述）→ D（只换聚合的配对重训，按选择受试者效应自动选 ≤3 臂）。工具 `tools/agg778/`（启动 `tools/agg778/run.sh ab|c|d|summary`），产物 `outputs/agg778/`；新键 `MODEL.XYZ_AGG`（`semkine/agg778.py`，opt-in，legacy 逐位不变）。S39 不变。
- 2026-10-24 结果：A 通过（聚合实现正确；工具链逐位复现 XCL E1、精确复现 E3 与 P4）。B 的有效性前提部分不过：A0 天花板 E3 5.27 未低于恒等 4.41（噪声锚点训练的探针下游在干净锚点上做虚假纠偏）；"仅计数"在 C1 关联下携带方向信息（0.26），供体事件在噪声锚点下携带锚点误差（0.16–0.18）。按预注册停止，不出裁决，**未进 D**（`d_stage.py select` 不自检有效性标志，人工不启动）。描述性读数：A3 下 C1 池化形式几乎不丢逐顶点信息，任务层缺口集中在训练所得特征迁移到 zgz_local（T2 11.8 对固定均值 5.2）；A0 下小运动的信息损失在分箱。是否登记偏离继续 D、或重设计，待用户决定。详见 `docs/EXPERIMENT_RECORDS.md#agg778-results-20261024`。
- 2026-10-24 用户选择"重设计一轮（需新预注册）"，批准 R2 计划（同一计划文件的 R2 版）。修正后的测量工具：探针下游不读 prev48，残差槽置零（桥接臂除外），干净锚点体制为主，A3 另跑噪声锚点体制复现。试点只用选择受试者，T0s 等于恒等，T6 低于恒等约 44 %，通过。正式检验：H-F 池化形式（M13 − T4）；H-T 迁移（T2 / M13i / M13 / T2abs / Mabs / Mcore）；H-B 分箱（M13 − M13fan）。有效性按条件 × 体制判定；D 臂预先固定（H-T → noabs 在 A3 与 A0；H-B → fanout 在 A0），单 seed，自动执行。预注册 `#agg778-r2-prereg-20261024`，冻结 `outputs/agg778/r2/protocol.json`，启动 `tools/agg778/r2_run.sh probes|d`；R1 被改动的冻结文件归档于 `outputs/agg778/r1_source_archive/`。新变体 `XYZ_AGG: noabs`（opt-in）。S39 不变。
- 2026-10-25 R2 结果（`docs/EXPERIMENT_RECORDS.md#agg778-r2-results-20261025`）：A3/E3、A0/E3、A3/noise 有效性全过；H-F 可忽略（C1 的 φ + 池化不比固定均值丢任务信息），H-T、H-B 未支持（R1 的迁移缺口在修正工具下不复现，扇出无增益）→ D 选择为空，未训练。描述：A0 与 A3 的 local 之差约 1.0 mm，指向关联而非聚合。事后诊断（不改裁决）：探针读出层激活随输入信息量单调（T6 约 27 %，A3 记录约 12 %，A0 各种聚合约 1.3 %，无信息时为 0）；A0 global 两格"未定"来自 Mabs 种子 2 的读出塌缩。A0_e3_g3 两次 OOM 现场保留。R2 结案，AGG778 无待办 GPU 作业。S39 不变。

## SBA · 扫掠边界对齐（用户 2026-10-22 指令"Execute the plan"）

- 执行 `/home/lyq/.claude/plans/pasted-content-id-ca61-reanalyze-s43-iterative-mccarthy.md`：把 MANO 当作事件生成器——可见遮挡轮廓沿轨迹扫过的面积给出逐像素期望事件数，Poisson 似然含负证据，精确运动学 Fisher 打分求解。只用物理 GPU 4–7，CPU 全量。
- 预注册在任何读数之前写入 `docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022`，源码 SHA 冻结于 `outputs/sba/protocol.json`。阶梯严格按序 S0a → S0b → S0c → P0 → P1 → P2 → P3，每级按预注册门可停止；本轮零训练，P4 只在 P0–P3 全过时执行；常数与选择只在训练受试者上定，zgz 只报告。产物 `outputs/sba/`，工具 `tools/sba/`，核心 `semkine/sba.py`（opt-in，生产路径不导入）；S39 默认路径逐位不变，S39 不变。
- 2026-10-22 结果：S0a 合成对照与 S0b 训练充分性通过（每事件偏差增益 local 3.3 / global 5.5 nats，κ 跨受试者 CV 0.24）；S0c 不过（zgz_local 事件率为训练标定的 1/2.4，改用剖面 κ）；P0 在 zgz_local 被否定（z 线最优偏 +45 mm；手指 ±20° 优先率全部域 0.58–0.68，整手平移 / 根 0.92–0.97）→ 按预注册停止，P1–P3、P4 未执行，无主行。S39 不变。详见 `docs/EXPERIMENT_RECORDS.md#sba-verdict-20261022`。
- 2026-10-22 追加授权（用户）：分析 P0 停止原因（D 轮，`#sba-d-prereg-20261022`），据此提出一个针对性创新点并训练，同时执行 P4；只用物理 GPU 4–7（四卡）。原 P0 裁决保留不改；新机制另行预注册后才读数。前臂胶囊关节索引勘误：OpenPose 序下掌指关节为 (5, 9, 13, 17)。
- 2026-10-22 D 轮：+45 mm 是 zgz 标签—图像约 6% 的二维尺度失配（绕腕缩放 σ\*=0.94 复现；S39 存档预测同向偏远 +30～+50 mm），不是跟踪失败；手指扫掠在单一发射率下被高估约 2 倍；明暗极性不可预测。第二轮（`#sba-p4-prereg-20261022`）：冗余量正交的生成式事件推断 + P4 发射模型（GPU 4–7 四卡数据并行），Q0 门通过才进入 P1′–P3′。
- 2026-10-22 第二轮结果：P4 四卡训练完成（留出受试者每事件 +0.066 nats；中 / 无名 / 小指自遮挡边发射率远低于拇 / 食 / 掌）；Q0 在 zgz_local 被否定（手指 ±20° 优先率 0.570 < 0.60；global 0.644，训练与留出受试者 global 0.77 / 0.74）→ 按预注册停止，P1′–P3′ 未执行，无主行。S39 不变。详见 `docs/EXPERIMENT_RECORDS.md#sba-r2-verdict-20261022`。
- 2026-10-22 停止原因复核（事后描述，不改裁决；`#sba-why-20261022`，`tools/sba/p0_visibility.py`，CPU）：P0 / Q0 的手指 ±20° 扰动（XDI 定义，局部 x 轴）在本 MANO 模板中是绕骨轴的扭转，可见变化中位 16–27 像素²（根 / 平移 450–1800），关节误差约 0.5 mm；优先率随可见变化单调升到 0.76–0.85，所以手指门不能证明与 MPJPE 相关的手指误差不可识别。深度偏置是录制级的标签—事件尺度差 × 距离（σ\* 0.93–1.01，训练录制 ycy_local 0.93；z\* ≈ 深度 × (1/σ\* − 1)，Pearson 0.72），不是 zgz 特有，也不是事件率失配。
- 2026-10-22 第三轮（用户指令"参考 docs 的研究结论重新设计实验"，批准计划 `/home/lyq/.claude/plans/please-refer-to-the-logical-hollerith.md`）：只用物理 GPU 2–7（2–3 被占用时退到 4–7），CPU 全量。只执行零训练地形，检验按真实 FK 关节误差定尺度的解剖手指扰动（单指屈曲 5 / 10 mm、去根协同 5 mm）在单窗内能否识别，以及 ≤ 200 ms 跨窗积累（事件只用一次）是否为杠杆。条件段（求解器修复、固定滞后、闭环）只预注册，需另行授权。预注册 `docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022`；冻结 `outputs/sba/r3/protocol.json`；代码 `tools/sba/r3_*.py`（opt-in），产物 `outputs/sba/r3/`，启动 `outputs/sba/run_r3.sh`。S39 不变。
- 2026-10-22 第三轮前置门：训练 local 两项未过（捐赠 synrc5 0.605 > 0.60；规范 R 0.722 < 0.75），查实现无缺陷（95 % 区间跨门槛，源于逐包尺度抖动与屈曲 / 伸展的足迹偏好）。用户决定读 zgz，全部冻结门不改，偏离公开登记（`#sba-r3-pregate-20261022`，`outputs/sba/r3/e1/deviation_pregate.json`）；续跑 `outputs/sba/run_r3_continue.sh`。训练受试者上 10 mm 手指误差单窗可识别（0.71–0.92）。
- 2026-10-22 第三轮结果（附前置门偏离；8.7 GPU·h）：E1 zgz_global 通过（单指屈曲 5 / 10 mm 0.72 / 0.85，去根协同 5 mm 0.93）；zgz_local 部分（0.65 / 0.76 / 0.72），捐赠 ≈ 0.5 有效。E8 跨窗积累为偏差限制：200 ms 信息是 50 ms 的 4–6 倍，优先率只升 0.01–0.03（zgz_local 部分、zgz_global 否定）。判读树：global → 条件段模式 1 需另行授权；local 的部分 + 部分没有预注册路线，需用户决定。第二轮 Q0 的捐赠数值受未收敛剖面影响。无主行，S39 不变。详见 `docs/EXPERIMENT_RECORDS.md#sba-r3-verdict-20261022`。

## XVM · 顶点锚定的事件运动测量（用户 2026-10-20 批准计划，2026-10-21 授权执行）

- 用户指令"用 GPU 4–7（共四卡）执行整个计划，包括所有实验与完整训练"：执行 `/home/lyq/.claude/plans/pasted-content-id-315d-has-the-tingly-moler.md`。只用物理 GPU 4–7；GPU 0–3 有其他用户进程，不用。CPU 全量可用。
- 预注册在任何读数之前写入 `docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021`（分析判断见 `#xvm-analysis-20261021`）。阶梯严格按序：C.0 信息上限 → C.1 oracle → C.2 可学习性（P4、C1+sign、P5、P6）；P4 与 P5 都通过才启动 C.3 两臂 × 三种子完整训练。超参数与尺度只在训练受试者上选，zgz 只报告。产物 `outputs/xvm/`，工具 `tools/xvm/`；新代码 opt-in，S39 / C1 / XCL 默认路径逐位不变。S39 不变，未经新门不替换。
- 2026-10-21 结果：信息上限允许单目逐顶点二维测量（P2 R_art 0.28），但学到的测量在 local 上不足（选中点集核 zgz_local 法向相关 0.34，参照时间面 patch 0.64，门槛 r* 0.83，相邻误差相关约 0.7）；P4、P5、C1+sign（local 29.86 / global 11.65 / RA 20.12，单 seed）均不过，按预注册未启动完整训练；与 S39 的朴素信息融合使 RA 升到 39–66。S39 不变。新增 opt-in 键 `MODEL.XYZ_SIGN`（零初值旁路，C1 默认逐位不变）。详见 `docs/EXPERIMENT_RECORDS.md#xvm-verdict-20261021`。

## S44 · EvTree（用户 2026-10-19 23:4x 授权执行计划）

- 用户指令"现在执行plan，GPU2-7，6卡，CPU全量可以使用"：执行 `/home/lyq/.claude/plans/pasted-content-id-fa8a-https-github-com-elegant-alpaca.md`。只用物理 GPU 2–7；GPU 0 有其他用户 OVGGT 进程，GPU 0–1 不用。
- 预注册在任何结果之前写入 `docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019`。先做阶段 0 零训练探针 P0.0–P0.10，按其 GE0 决策再决定是否训练；超参数只在训练受试者上选，zgz 只报告。产物 `outputs/s44/`，工具 `tools/s44/`；新代码 opt-in，S39/S43 默认路径逐位不变。S39 不变，未经新门不替换。
- 2026-10-20 结果：阶段 0 两轮快路径（预注册尺度与大尺度）P0.1 与终门 P0.10 均作废（local 变化 ≤ 0.03 mm），oracle 归属及给定 GT 刚体速率时仍解不出手指增量，瓶颈在事件证据而非关联；按规则转架构 D（S39 + S42-B 三种子 6000 步），local 10.25 / global 7.47 / RA 8.76，三门不过。S39 不变。详见 `docs/EXPERIMENT_RECORDS.md#s44-d-result-20261020`。

## XCL 阶梯控制实验（用户 2026-10-19 执行计划）

- XCL最终结果状态：COMPLETE_SCIENTIFIC_REVIEWED；五个新训练验收、十模型134项评估及两次fresh2590包逐数组一致全部完成；结果`outputs/xcl/20261019/report/main_table.md`，复核`outputs/xcl/20261019/postprocessing/scientific_review.json`，账本`docs/EXPERIMENT_RECORDS.md#xcl-result-20261019`。单seed3407描述性比较，oracle不可部署，S39不变。

- 用户指定执行 `/home/lyq/.cursor/plans/xyz关联gnn控制变量复盘_7ce02791.plan.md`，训练仅物理GPU2–7：2–3固定A2F0→A3F0，4–5固定C0→A0FΩo，6–7固定A0FΩd；每臂seed3407、fresh6000、2×512、BF16、legacy_none、worker12/prefetch1及原损失/Adam/LR/调度，不新增seed或择优重训。
- 复用严格C1及已完成的A0F1/A1F0/A1F1/A2F1共五个模型，均绑定既有6000步checkpoint与初值/实际输入流；失败A2F0现场保留，新A2F0另开目录fresh，不resume。入口`tools/xcl/execute.py`，新协议/状态`outputs/xcl/20261019/{protocol,status}.json`，配置`configs/xcl_20261019/`；禁止重启旧协调器。
- `/dev/shm`已恢复tmpfs+1777，实际共享内存创建/读写/释放通过；正式训练前须五个新cell全部通过50步真实双卡、完整验证126批/每rank64094样本并返回训练、C1共有37个state tensor初值与共有输入/实际索引验收。预检只作工程验收。
- A3仅在全部74序列depth sidecar完成、逐毫秒旧来源parity与全文件SHA通过后使用；新增depth identity在A3预检前冻结为`depth_manifest.json`及`depth_binding.json`，正式接受文件再次绑定该SHA。FΩ两臂共享独立RNG初始化的投影层；上游rel/vin/edge/GNN冻结，原残差槽数值与零事件门保留，仅投影和下游槽/读出训练。
- 评估仅GPU4–7，每卡一进程；全部训练验收后先串行含Latency主行，再执行两次fresh完整2590包轨迹及E2/E3/E4/E5与冻结权重干预。单seed描述性阈值不替代可靠性；oracle不可部署，S39不变。
- 旧XAG/XA按计划先归档清单与关键数字；源码迁移及新运行时独立验收后再删除旧目录，清理已完成，旧pointer已归档删除；新结果仍须统一评估完成后发布。原C1/paired协议、源码归档、权重及失败保留。详见`docs/EXPERIMENT_RECORDS.md#xcl-20261019`。

## XAG 与新C1严格配对重训（用户 2026-10-18 追加授权）

- 用户已明确选择“五组各重训一次seed3407，形成严格配对比较”。新研究位于`outputs/xag_paired/20261018/`，入口`tools/xag_paired_20261018/`，配置`configs/xag_paired_20261018/`。五格为`a0f1/a1f0/a1f1/a2f0/a2f1`，各fresh6000一次；A0F0直接复用已完整复现的`xyz_c1_stable_s3407`，不新增baseline训练、seed3408或同seed择优复跑。
- 五格使用新C1的`legacy_none`实际采样与严格确定性数值合同；共享模型初值、每rank实际数据索引与共有事件/增强/前态输入须与C1核验。新增39个F+参数在三格共享初值；科学臂学习后的权重、梯度与自身递推不要求相同。仍为2×512、BF16、worker12/prefetch1、原损失/Adam/LR/调度、原数据划分及完整2590包评价。
- 正式队列固定：GPU0–1 A0F1，2–3 A1F1，4–5 A2F1，6–7先A1F0再A2F0；不按中途成绩改变顺序、步数或保留项。先进行真实双卡/完整验证/返回训练的工程验收，再发布独立新P/A/E与源码冻结。旧186/213文件冻结、原协议、失败、权重及已重核旧行全部保留。
- 用户2026-10-18追加指定：后续统一评估只使用物理GPU4、5、6、7四卡，每卡同时一个评估进程，待最后GPU6–7训练结束并通过全五格训练验收后启动。先串行生成含Latency测量的五主行，按4/5/6/7/4轮换，期间不并发本任务trace；随后四卡并行执行每臂两次fresh完整递推。原冻结协调器的GPU0评估安排被此指令取代；通过独立执行修订与接管入口落实，不修改926冻结文件/P/A、训练身份、6000步权重、评分与完整两次2590包轨迹合同。GPU0–3不再用于本轮后续评估。
- 四卡资源接管已于北京时间2026-10-18 15:57生效：入口`tools/xag_paired_20261018/continue_gpu4567.py --override outputs/xag_paired/20261018/execution_gpu4567/override_v3.json`，当前进度仍读原`status.json`，接管与执行回执在`execution_gpu4567/`。新入口只监督既有最后训练、随后调用原冻结验收/评估/评分；不重启训练。原`execute.py`是fresh-only历史入口，不要重启或恢复其暂停进程以免重新派发GPU0。
- 同日16:02因主机`/dev/shm`挂载消失、root0755目录导致worker SHM权限错误，最后A2F0在1437批丢失待收结果；该错误早于资源接管约162秒。按原P的fail-closed/no-resume停止并保留失败，当前`status.json=FAILED_PRESERVED`，四个已完6000训练保留、五主行仍0。原协调/最后训练/新协调进程均已退出，未启动评估、未更新旧metrics pointer。需管理员恢复tmpfs+1777；不得据step1000 checkpoint宣称严格续训（缺RNG/worker/采样器状态），不得重跑其它四臂或把失败臂当完成。后续评估的GPU4–7指令继续有效。证据`outputs/xag_paired/20261018/audit/postfreeze_shm_failure_20261018/final_receipt.json`。
- 新五格正式完成且完整训练/主行/两次2590包轨迹及独立评分核验通过后，才原子更新`outputs/xag/current_metrics.json`并归档旧pointer。工程预检不作精度结果，旧五格不能冒充新配对结果。部位/表面GT仍是不可部署oracle；全表为单seed3407描述性比较，不据此宣称跨seed可靠性。详见`docs/EXPERIMENT_RECORDS.md#xag-paired-c1-20261018`。

## XYZ C1 可复现基线与指标重核（用户 2026-10-18）

- 用户要求查清历史C1差距、建立稳定可复现C1 baseline，并重新确认其余五格指标；仍只用seed3407，不做多seed随机训练或择优。
- 原XAG六格、原协议/186文件冻结及全部历史产物保持。新的C1复现研究另存`outputs/xyz_c1_repro/20261018/`，入口`tools/xyz_c1_repro/`：预先指定历史None采样的C1为canonical；完全相同的fresh重放只作工程核验，不当第二seed、不平均、不择优；另一个固定3407的dedicated采样仅作单因素对照。三角色均固定原6000步、2×512、模型/增强/损失/Adam/BF16，确定性执行与资源设置另行显式封版。
- 只有完整fresh6000重放、参数/Adam/数据索引记录及完整zgz预测均通过，才能把新C1标成“已验证可复现baseline”。短程PASS不能替代该结论；若分数低于历史，照实保留，不换用更好对照。
- 原五格使用既有step6000权重重核。固定权重F+默认评估有数值非确定性，新统一严格评估与新主行另存`outputs/xag/reviews/reconfirm_metrics_20261018/`，旧行不覆盖。不得把旧训练重新标成与新C1确定性配对的训练。S39保持不变。
- 2026-10-19按XCL计划归档删除旧XAG产物：原`outputs/xag/current_metrics.json`及六格重核主行已归档至`outputs/deleted_20261019/xag_xa_legacy/files/`，不再是当前入口。原`outputs/semkine/xag_*_main_row.json`保留历史默认运行时结果，不能再把它们当作本次更新后的指标。旧矩阵内五格的配对参考仍是同合同A0F0；新确定性None基线属于另一训练合同。
- 新C1已完成完整fresh6000复现及2590包输出核验，正式研究基线入口为`outputs/xyz_c1_repro/baseline.json`（canonical=`xyz_c1_stable_s3407`，legacy_none，step6000），主行别名`outputs/semkine/xyz_c1_baseline_main_row.json`。最终实物审查`outputs/xyz_c1_repro/20261018/audit/final_release_independent_readme_v2_20261018/receipt.json`及独立NumPy评分`outputs/xyz_c1_repro/20261018/audit/c1_metric_definition_final_20261018/receipt.json`均PASS；详见`docs/EXPERIMENT_RECORDS.md#xyz-c1-repro-result-20261018`。固定环境内的工程复现不等于跨seed可靠性或恢复历史全部精度；不得因dedicated对照总体分数更好而替换预指定None基线。
- 基线权重、完整复现证据及其SHA必须随身份保留，不能仅保留主行/日志而清理权重。完整复现入口是`tools/xyz_c1_repro/reproduce.py`，说明见同目录README；原生`semkine/train.py`不会自动启用本次严格数值合同。此独立包装入口只做过CPU guard/schema验收，完整物理重放由同一冻结train/evaluate/trace协调器执行；不能冒称另跑过一次入口全训。原postprocessing manifest保留历史身份，README文案修订的当前补充为`outputs/xyz_c1_repro/20261018/postprocessing_manifest_readme_v2.json`；213训练冻结及P/A/E不变。

## XAG 当前训练范围（用户 2026-10-17 修订）

- XAG 现仅执行六格 `a0f0/a0f1/a1f0/a1f1/a2f0/a2f1`，每格只保留一次 **seed3407** 训练，共 **6 个 run**；不再派发 seed3408 或同 seed 独立复跑。修订前已产生的训练、失败、OOM、检查点及审查证据完整保留，不删除、不覆盖。
- 六格均保持原 fresh step6000、双卡 2×512、模型/数据/增强/损失/优化器及完整 zgz 评价合同；已完成或正在执行的保留项沿用其原训练身份，不按中间成绩选 seed 或重置训练。数据划分和 S39 不变。
- 原 `outputs/xag/protocol.json`、186 文件源码冻结及归档不改；范围修订见 `outputs/xag/single_seed/protocol.json`，新执行入口在 `tools/xag_single_seed/`，当前状态以 `outputs/xag/single_seed/status.json` 为准。旧18项执行状态只作历史记录。
- 此轮结果按单 seed 描述性比较：表内明确 seed3407，精度与 RA 不写双 seed 均值或双 seed 括号；保留规定九列及 baseline/S39 行。原双 seed 同向条件及依赖同 seed 复跑差的噪声门不再适用，不据单 seed 宣称可靠性 PASS、稳定优势或主因已确定。详见 `docs/EXPERIMENT_RECORDS.md#xag-single-seed-20261017`。

## 数据协议固定，不再改动（用户 2026-10-01）

- 训练：9 人（ch、lfz、lpc、lr、ly、lyh、lyq、ycy、ylf），72 条序列。
- zgz（`zgz_global`、`zgz_local`）既是开发集也是测试集：选点、调参、上报都在这两条序列上。不另设封存测试集。
- **2026-10-25 修订（用户决定，十方案 v2 的 M10）**：EvRealHands（服务器 `/data1/lyq/code/mesh/EventHands/data/evrealhands12/`，10 人 80 条，DAVIS346 346×260，15 Hz 标签、快动作只有 2D 真值）解禁为**只作最终测试**：不选点、不调参、不进 zgz 主行，单独一表报告；51 维转换、2D / 3D 真值分档与 K / 像素合并 / 时钟清单另行计划并登记后才能读数。zgz 的开发 = 测试地位、9 人训练与 `splits_semkine.json` 不变。登记见 `docs/EXPERIMENT_RECORDS.md#mv2-decisions-20261025`。
- 唯一的划分文件是数据根下的 `splits_semkine.json`（由 `tools/build_splits_semkine.py` 生成，文件只读）。
  配置的 `DATA.SPLITS_MANIFEST` 要么不写（默认就是它），要么写它的绝对路径；不要新建或换用别的划分。
- 退役的 8 人划分和旧封存测试（ly）已作废；按 8 人划分训练的 run 不进主表。

## 实验编号（用户 2026-10-04）

- **S39 唯一指 `dt3_dz_l3_w200cfg`**：`dt_dz_l3`（种子 3407 / 3408 / 3409）+ 证据窗口 W = 200 ms（裁段首）+ DT2 事件量自适应滤波 + 证据门 ramp1000。
  主行 `outputs/semkine/dt3_dz_l3_w200cfg_main_row.json`（RA 8.71（8.91 / 8.53 / 8.69）），出表用 `--label dt3_dz_l3_w200cfg="S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"`；来历见 `docs/EXPERIMENT_RECORDS.md#dt3-verdict`。
- 09-21 的 `s39_covmap`（S37 routed + 根读表面覆盖图，09-28 已删）不再叫 S39，只用 run 名 `s39_covmap` 指代。

## Results are reported in one table, with these columns only (user's format, 2026-09-21)

Every result reported to the user — a training arm, an ablation, a probe — is headlined by this
table and nothing else: same columns, same order, same units. No extra metric columns (abs MPJPE,
rotation angle, amplification, grid median, mechanism-gate numbers, ...) in the report; those
stay in the arm's pre-registration section of `docs/EXPERIMENT_RECORDS.md` for whoever asks.

```
| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 * | 10.27 | 7.36 | 8.94 | 6.25 | 8.71（8.91 / 8.53 / 8.69） | 4.09 ms | 1.252 G | 2.80 M |
```

- Generate it, do not type it: `python tools/report_table.py <run> [<run> ...] --label <run>="<显示名>"`.
  It reads `outputs/semkine/<run>_main_row.json` (written by `tools/make_s36_row.py` on the zgz
  protocol), so the table cannot drift from the artifacts.
- Accuracy cells are the two-seed mean in mm; the RA cell carries the two seeds in parentheses.
  Latency in ms (full-model anchor 1.75 ms scaling), FLOPs as MACs of the whole `forward_packet`
  in G, params in M.
- Always include the baseline row and the current arm's row.
- A one- or two-sentence verdict (pre-registered gate pass / fail, adopt / keep the current arm)
  may follow the table. The *why* goes to the arm's pre-registration and failure/cleanup sections in `docs/EXPERIMENT_RECORDS.md`.
- An arm without a main row is not reported as a result; run `tools/make_s36_row.py --run <arm>` first.

- 用户2026-10-06明确暂停原持续实验 goal，代码已恢复并保留 S39（`dt3_dz_l3_w200cfg`）。自动派发已停止；未经用户重新授权，不继续诊断或训练。正式S39使用原三种子step6000权重，显式W200裁段首、Count50、DT2+ramp1000离散spec；恢复与失败归档见 `docs/EXPERIMENT_RECORDS.md#tracking-stop-s39-20261006`。
- 用户2026-10-06晚重新授权新一轮（S42，GPU4–7）：候选 A/B/C 与消融均已完成，最好的 S42 候选（S39 配方 + 部署环路自身前态训练 B + resnet18_l3w192）local 9.90 / global 7.28、参数 1.94 M、MAC 1.098 G，但按预注册门可靠性、local 滞后与端到端时延未过，**未替换 S39**；结论、主表、门与复现见 `docs/EXPERIMENT_RECORDS.md#s42-verdict`。S42 的源码改动全部 opt-in（`MODEL.GLIMPSE*`、`DATA.CLIP_*`、`TRACK.SELF_*`、`MODEL.DISTILL_CKPT` 列表），默认路径与 S39 逐位相同。
- 用户2026-10-09授权 S37 XYZ 关联三层验证（零训练来源分析、单检查点探针、配对重训 xa_c1/xa_c1r/xa_c0/xa_orc/xa_ctr，GPU2–7）：预注册、门与结果见 `docs/EXPERIMENT_RECORDS.md#xyz-assoc-20261009`。新增键 `MODEL.XYZ_ASSOC_MODE` / `XYZ_CONTOUR_GAP` / `XYZ_ASSOC_PREV` 全部 opt-in，默认路径与 C1 逐位相同；xa_orc 用真值分配，只作诊断上界，不可部署。
  结果（2026-10-10）：G4 复现成立（末步 22.55 对历史 21.92）；噪声底 δ0 = 3.1 mm；真值分配上界 overall −10.6 / local −15.7 mm（含泄漏）；替代关联 contour 两种子同向改善 2.5 / 4.4 mm 但在噪声底内，φ ≤ 0.25；匹配 C0 在 local 优于 C1 6.9 mm（超 τ、同号）；L2 两项判据不支持。裁决：关联规则是次要缺陷，主瓶颈是闭环前态误差与关联互相放大的结构及 local 深度分布外；见 `#xyz-assoc-20261009` 裁决区。S39 不变。
- 用户2026-10-14授权 XDG（GPU2–7 六卡训练、全部 CPU）：检验"显式绝对平移 t × 训练深度支持"造成的 local 深度钳位。XDG-0（冻结特征头重拟合）与 XDG-1（完整重训 `xdg_q1t0` / `xdg_q1dz` / `xdg_q1r`）机制门均 PASS：去掉 t 或改 depth 尺度后 local 偏置从 −65～−73 降到 +3～+4 mm；但 ±50 mm 纠回仍弱，q1t0 主表无改善，q1dz local 19.54 / global 14.41 / RA 16.79（单 seed、语义总效应）。新键 `MODEL.XDL_STATE_T` 是 opt-in，默认路径与 Q1 逐位相同。见 `docs/EXPERIMENT_RECORDS.md#xdg-20261014`。S39 不变。
- 用户2026-10-14批准 XDI（不改关联、无参数事件→轮廓新息只学增益）：Phase 0 零训练可行性四门在 local 全 FAIL（GT 前态被拉偏 45 mm，GT 不是真实事件下轮廓代价的最小点），按停止规则未接入模型、未训练；`semkine/xyz_innovation.py` 仅作诊断工具。见 `docs/EXPERIMENT_RECORDS.md#xdi-20261014`。S39 不变。
- 用户2026-10-14授权 XKF（GPU4–7 四卡、全部 CPU）：给 XYZ 加一条不读前态的事件直接测量分支（长窗口 LNES + HandDGP 平移），XYZ 关联与 MANO 图不变、改训为运动模型，二者用残差统计得到的常数增益融合（E-3DPSM 思路）。预注册与冻结清单见 `outputs/xkf/protocol.json`，训练由用户运行 `tools/xkf/launch.sh`；新代码全部 opt-in（`MODEL.XKF*`、`EventPacketBatch.lnes_window_ms`），XDL 默认路径逐位不变。见 `docs/EXPERIMENT_RECORDS.md#xkf-20261014`。S39 不变。
- 用户2026-10-15更新 XKF 授权：由 agent 直接使用 GPU2–7 六卡执行，必须先 debug 到效果门通过，再开始全量训练。按三组 2×512 分配 seed3407→GPU2–3、3408→GPU4–5、3409→GPU6–7；先200步非正式调试，不以进程成功或有限性替代学习效果。`tools/xkf/launch.sh` 默认仅 debug，全量需三种子学习检查通过的记录及匹配源码清单；原协议与 Stage0 F1 失败保留。修复/进度见 `docs/EXPERIMENT_RECORDS.md#xkf-debug-20261015`。
- 用户2026-10-15要求立即排查local递推：三seed debug已到2000步并以`STOPPED_DEBUG_GATE_FAIL`停止；未全量。冻结权重的全2590包D分解和112片段×9初态×6模式诊断发现D二维几何/姿态深度偏差、I正确前态漂移及联合扰动t/root误差抵消同时存在，未发现腕点坐标或DGP求解错误。原门/冻结训练源码不变；`xkf_debug2000_main_row.json`为明确标记非正式step2000的完整序列诊断，不能充当step6000或readiness。见`docs/EXPERIMENT_RECORDS.md#xkf-local-recovery-20261015`。S39不变。
- 用户2026-10-15要求立即修复：XKF R2三seed同LR配对续训2000→2200及全部评估已完成（GPU2–7）。`XKFRepairModel`的FP32累加、真实GT运动标签及训练专用D几何辅助均opt-in，旧冻结源码不变；25项实现测试、20项分析器测试及136项独立训练审计通过。I的local clean漂移减少，但D全local深度三个seed均差于对照，local融合clean及恢复门FAIL；`outputs/xkf/repair_20261015_r2/decision.json`为`STOPPED_DEBUG_GATE_FAIL`，未全量、不原样加长该候选。主行`xkf_r2_control_step2200`与`xkf_r2_repair_step2200`均非正式；代码正确性不等于效果过门。统一表、完整配对证据和后续边界见`docs/EXPERIMENT_RECORDS.md#xkf-repair-r2-20261015`。S39不变。
- 用户同日再次明确要求“local融合修好”：R3继续执行（GPU2–7、debug先于全量），不是因R2失败而停止任务。独立测量标定、训练原始事件BN统计、融合策略及真实own-loop验证见`docs/EXPERIMENT_RECORDS.md#xkf-fusion-r3-20261015`和`outputs/xkf/fusion_20261015_r3/protocol.json`；保留旧失败与全部原门，不挑seed/片段，不以冻结平移或代码测试通过代替恢复效果。旧冻结源码及S39不变。

## IDP · 身份 / 点位阶梯（用户 2026-11-05 "现在直接进行，开训。"）

- **来历：** 对关联问题"第一个原因"的零训练归因（会话 scratchpad，未登记）：路由器的沿骨误差由数据决定（r_s2d / sr1 / T0P 的包级偏移相关 0.67–0.85），集中在透视缩短比 < 0.6 的手指（三分之一节点贡献约六成均方误差，T0P 也不救）；路由器读的是手指的可见范围（去掉指尖事件整指预测移 7–15 mm）。结论：标签上界 5.98 不可达，要先分开"身份"与"点位"的责任。
- **三臂（B2 配方、种子 3407）：** l1 = 训练与测试都用 `label_part`（标签身份 + r_s2d 点位）；l2 = l1 + `semkine/idp_feat.py` 的记录特征（前态手指透视缩短比与屈曲角，零初值映射进 φ）；l3 = B2 + 同一特征（可部署）。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105`；冻结 `outputs/idp/20261105/manifests/r1.json`（`python -m tools.idp.freeze write|verify --stage r1`）。规则：L1 `POINT_ERROR_LIMITS` / `IDENTITY_IS_LEVER` / `MIXED`；L2 `FEAT_HELPS` / `FEAT_NULL`；L3 按修订 3 筛选，补种子须用户下令。
- **代码：** `tools/idp/`（common、arms、train、run、analyze、jobs、chain、freeze）、`semkine/idp_feat.py`、`tests/test_idp.py`（CPU，`CUDA_VISIBLE_DEVICES=""`）、`configs/idp_20261105/`。只读沿用 PFO 引擎、PFO2 的 B2、B2UB 的 label_part 读数、EVF 的 B2 成本块。
- **驱动：** `tools/idp/chain.py`（B2UB 驱动逻辑；`setsid nohup … run`；fresh-only；失败保留不重试；DRAIN / STOP 仅紧急）。作业 `outputs/idp/20261105/jobs/r1.json`（31 个）。读 XAG 源的训练声明 260 GiB，l1 / l2 串行。
- **注意：** `tools/pfo/nets.py` 的混合路由默认用 sr1；本研究与 B2UB 一样在进程里把 `_router_for` 重新绑定到配置的 r_s2d。训练时 label_part 混合经 `tools.pfo.nets.mixed_sources` 装在 `semkine.s45sm.install` 之上。
- **结果（`#idp-results-20261105`，08:58 UTC COMPLETE，31 个作业全过）：** L1 **MIXED**（RA 9.69，五项单种子全低于 S43；配 r_s2d 11.55，无暴露崩塌）；L2 **FEAT_NULL**（9.49，对 L1 −0.19）；L3 **SCREEN_FAIL**（11.74）。
  - 事后零训练（描述）：B2 → 标签上界 5.5 mm ≈ 手指级身份 1.4 + 指节 0.4 + 沿骨点位 3.7；误差集中在图像里很短的手指；加大路由器、置信度门控、固定规则前态仲裁都无效。
  - L1 式训练（标签部位只在训练时）部署结果与 B2 同档，但能兑现身份改善（1.86 mm 对 B2 主干 0.39），可作后续主干基底；补种子须用户下令。

## PCR · 前态条件路由器（用户 2026-11-05 "……让路由器这一段学会用三维先验来解开弯曲手指的歧义……这部分为什么不进行呢"）

- **来历：** IDP 与零训练读数显示 B2 到标签上界的 5.5 mm ≈ 手指级身份 1.4 + 指节 0.4 + 沿骨点位 3.7，误差都集中在图像里很短（弯曲 / 朝向相机）的手指上；加大路由器、置信度门控、固定规则的前态仲裁都无效。用户已放开路由器段的异步要求，并决定让路由器读前态（偏离 PFO 的"路由不依赖前态"，只限本研究）。
- **两臂（只训路由器，r_s2d 配方，种子 3407）：** `pcr` = r_s2d + 前态 16 通道部位投影图（零初值 `prior_in`），训练前态 = 数据集 mixed 噪声；`pcrb` = 同上 + 一半前态换成 pfasrdz 闭环状态库（N4）。零训练换进冻结的 L1 / B2 主干读闭环。
- **预注册：** `docs/EXPERIMENT_RECORDS.md#pcr-prereg-20261105`；冻结 `outputs/pcr/20261105/manifests/r1.json`。规则：d = RA(L1 + R) − RA(L1 + r_s2d)，≤ −0.85 `PCR_WORKS`，≥ −0.30 `PCR_NULL`，其间 `PCR_WEAK`；R15 / F15 升 ≥ 0.05 标 `PRIOR_COLLAPSE`。重训主干与补种子须用户下令。
- **代码：** `semkine/pcr_router.py`、`tools/pcr/`、`tests/test_pcr.py`（CPU，`CUDA_VISIBLE_DEVICES=""`）、`configs/pcr_20261105/`。新路由器经评测配置的 S45SM 块接入（kind "pcr"，`register()` 让 `FAST.install` 装上读前态的采样器）。
- **驱动：** `tools/pcr/chain.py`（GPU 4–7，B2UB 驱动逻辑，fresh-only）。作业 `outputs/pcr/20261105/jobs/r1.json`（36 个）。
- **结果（`#pcr-results-20261105`，10:18 UTC COMPLETE，36 个作业全过）：** pcr / pcrb 都是 **PCR_NULL**（RA 11.85 / 11.76 对 L1 + r_s2d 11.55），pcrb 标 **PRIOR_COLLAPSE**。
  - 前态正确时单步更准（E3 local 8.36 → 7.38 / 6.78），闭环里路由跟着前态走：R15 0.34 → 0.38 / 0.42，C 0.86 → 0.90 / 0.96。闭环前态下手指级准确率只多 0.6–0.8 个点。
  - 按预注册：单包事件图路由 → 逐事件对应框架的精度线到此为止，下一方向待用户决定。

## TPD · B2 三问题分解求解（用户 2026-11-05 /goal，约 14:30 UTC）

- **目标：** 以 B2 为起点分别解决手指级身份（约 1.4 mm）、指节与沿骨点位（约 4.1 mm）、框架自身闭环损失（约 1.8 mm），再合成；最终可部署形式三种子五项均值全部低于 S43 引擎值 12.81 / 8.02 / 10.77 / 6.64 / 10.24、每种子 RA < 11.0。/goal 即执行授权（每阶段预注册、冻结、验收后自动推进），只在改数据协议 / 划分、放宽约束、部署行用标签时停下来问；只用 GPU 2–7。约束 4（闭环网络不得用稠密前态表示，稀疏 FK 投影允许）的边界待用户确认，新增前态投影的臂（EXT）登记前先问。
- **登记：** `docs/EXPERIMENT_RECORDS.md#tpd-prereg-20261105`（目标、约束、阶段、通用规则、停止点）；各阶段另有小节。
- **代码：** `semkine/tpd_*.py`、`tools/tpd/`、`tests/test_tpd*.py`（CPU，`CUDA_VISIBLE_DEVICES=""`，每个文件单独进程跑）、`configs/tpd_20261105/`；产物 `outputs/tpd/20261105/`。
- **驱动：** `tools/tpd/chain.py`（IDP 调度逻辑，允许 GPU 2–7 上有 ≤ 6 GiB 的他人轻负载；`setsid nohup … run`；fresh-only；失败保留不重试；DRAIN / STOP 仅紧急）。作业按阶段 `chain.py add`；条件作业经 `tools/tpd/maybe.py`（规则未放行时为空操作，exit 0）。
  - 现役是驱动 5：`TPD_DRIVER_DIR=driver5 tools/tpd/chain4.py`（= `chain3.py`，只用 GPU 4–7，训练对 (4,5) / (6,7)；状态 `outputs/tpd/20261105/driver5/`）。用户 2026-11-06 起 TPD 只用 GPU 4–7（GPU 0、3 有别人的进程；修订 6 `#tpd-amend6-20261106`）。新作业一律 `TPD_DRIVER_DIR=driver5 python tools/tpd/chain4.py add`；会自己追加作业的工具要设这个环境变量（`run_rolk.py` 之类默认写驱动 4）。
  - 驱动 4 已 DRAIN（只让在跑的 c5 3407 训练跑完），它的 27 个待跑作业原样迁到驱动 5，排在 `d5:wait:c5` 后面。`chain2.py` 每次遍历只判一次训练间隔，规则已关闭的条件作业改作 CPU 作业跑、不占卡；`chain3.py` 另加 `TPD_START_BUSY_OK`。驱动 1–4 状态保留（修订 1–3、6）。
  - 分工（2026-11-06）：本研究有两个会话。STR 与驱动由一个会话负责；合成阶段的结果、ERH 表、成本块由另一个会话负责。账本只在末尾追加，追加前先互相通知。
  - /dev/shm（252 GB）是瓶颈：K ≥ 5 的展开批次很大（带 XAG 源约 2.5 GB），用预取 1（`tools/tpd/ns_amend3.py`，配置 `configs/tpd_20261105/amend3/`，入口 `tools/tpd/a3.py`）；c0 式的可变窗 XAG 训练单独约占 110 GB，不和两个 K = 5 训练同时跑（修订 3、5）。
  - 新的包装入口先做双卡 DDP 冒烟：Lightning 用改写后的 `sys.argv` 重启 rank 1（修订 4）。
- **阶段 ROL（`#tpd-rol-prereg-20261105`）：** B2 的 K 包闭环展开微调（`semkine/tpd_rol.py`：展开数据集、可微前态几何、逐包梯度检查点、余弦到 0），四臂 rol1 / rol3d / rol3 / rol5（种子 3407，从 B2 3407 第 6000 步微调 1500 步，峰值 1e-3）；入选臂自动补种子 3408（从 B2 3408）。入口 `tools/tpd/train_rol.py`、读数 `python -m tools.tpd.run_rol`、判读 `python -m tools.tpd.analyze_rol plan|all|all2`、作业 `python -m tools.tpd.jobs_rol rol`、冻结 `python -m tools.tpd.freeze write|verify --stage rol`。
- **注意：** autocast 会缓存 no_grad 下做的权重低精度副本，训练步里任何 no_grad 主干前向之后要 `torch.clear_autocast_cache()`，否则该步梯度到不了权重（ROL 冒烟发现）。GPU 冒烟入口 `tools/tpd/smoke_rol.py`（`TPD_SMOKE_DIR`、`TPD_SMOKE_STEPS`）。
- **ROL 结果（`#tpd-rol-results-20261105`）：** 选中 rol5（3407 RA 10.49）；种子 3408 RA 9.48，对 B2 3408 `SCREEN_PASS` → `ADOPT`。增益全在单步（E3 降），放大不降。
  - ROL-c（`#tpd-rolc-prereg-20261105`，描述）：rol5 补种子 3409，三个种子各自零训练换成因果形式读数，表 `analysis/rolc.md`；入口 `tools/tpd/train_rolc.py`、`run_rolc.py`（经 `ns_seed.py` 把种子表扩到 3409）。
- **成本块工具：** `python -m tools.tpd.cost run|jobs`（PFO 成本工具原样；因果形式的逐包步长 k 在取窗时算出、随包携带，测试 `tests/test_tpd_cost.py`）。按 `#tpd-syn-prereg-20261105`，等最终配置确定后再登记（GPU 4 独占，同块重测 S43，两种标志）。
- **ROL-c 结果（`#tpd-rolc-results-20261105`）：** rol5 部署形式三种子 12.86 / 7.75 / 10.66 / 6.28 / 10.12（只有 MPJPE-L 高于 S43 0.05，但不异步）；零训练换成因果形式 13.37 / 7.99 / 10.97 / 6.48 / 10.49。
- **ROL-k（`#tpd-rolk-prereg-20261106`，描述）：** rol5 在因果形式下训练（`tools/tpd/ns_rolk.py`、`train_rolk.py`、`run_rolk.py`），种子 3407 对零训练因果形式的 rol5 3407 按修订 3 筛选；通过时筛选作业自己往驱动 4 加种子 3408 / 3409（`jobs/rolk_s23.json`）。不是目标写定的最终配置（从头、L1 式）。结果 `SCREEN_FAIL`（RA 10.57 对 10.78，门 −0.3；`#tpd-rolk-results-20261106`），阶段停止。
- **EvRealHands（`#tpd-erh-prereg-20261106`，只报）：** 转换 `tools/tpd/erh_convert.py`（右手 42 条 → `outputs/tpd/20261105/erh/data/`；K / 时钟 / MANO 约定的核对见 `erh/convert_report.md`），读数 `tools/tpd/erh_eval.py`（3d 层；loop 与 h200；保持起点基线；GT 轮廓诊断）。手在 0.8–1.06 m，超出训练深度；冒烟里所有网络（含 S43）在 200 ms 内发散。表 `erh_eval table` 在最终配置读数之后运行一次。
- **问题 2 EXT（`#tpd-ext0-prereg-20261105`）：** 零训练可行性 `KILL`（r = 0.28 < 0.3），不训练。
- **问题 1 LC（`#tpd-lc-prereg-20261105`）：** `semkine/tpd_lc.py`（r_s2d + 包前 200 ms 历史通道，零初值）与对照 lc0，`tools/tpd/router_lc.py` 微调 3000 步；两者 `ID_SCREEN_FAIL`（zgz_local 手指级 0.866 / 0.856，门 0.90），不读闭环。读数入口 `python -m tools.tpd.run_lc`；判读表修订 1 用 `tools/tpd/analyze_lc2.py`。杠杆 (b)–(d) 不开（理由见 `#tpd-syn-prereg-20261105`）。
- **异步形式（`#tpd-causal-prereg-20261105`）：** B2 原样不过前缀检验（节点按包总数均匀取样；token 的"未发放"填包实际跨度）。`semkine/tpd_causal.py`（因果步长 k = ceil(n_prev / 2048)、"未发放"填 Δt）通过 `tools/tpd/asynccheck.py`；B2 零训练换成因果形式 RA +0.26 → `TRAIN_CAUSAL`：最终配置在因果形式下训练（`semkine/tpd_final.py`、`tools/tpd/train_final.py`、`tools/tpd/run_final.py`）。
- **合成（`#tpd-syn-prereg-20261105`）：** c0（L1 式、因果、B2 可变窗）与 ROL 选中臂对应的展开臂（c3d / c3 / c5），从头 6000 步；条件作业经 `tools/tpd/maybe_syn.py`；判读 `python -m tools.tpd.analyze_syn screen|keep|final`。读 XAG 源的训练实测约 200 GiB PSS，同时最多两个。
- **流式形式 STR（`#tpd-str-prereg-20261106`）：** `semkine/tpd_str.py`。
  - 节点按过去 50 ms 的事件率逐事件稀疏，不截包末；像素时间面跨包保持。路由器 r_s2d 仍按包、豁免，所以不能说"事件到达即写入"。
  - 入口：`train_str.py`、`run_str.py`、`asynccheck_str.py`（含分包检验）、`jobs_str.py`，冻结 `manifests/str.json`。
  - 零训练换入（`#tpd-str-ref-results-20261106`，描述）：rol5 三种子 12.95 / 7.96 / 10.70 / 6.46 / 10.28。
- **2026-11-06 04:52 UTC 用户指令"先停止训练吧"（`#tpd-stop-20261106`）：** 用驱动自己的 STOP 停了驱动 4、驱动 5（都记 STOPPED_PRESERVED）。
  - 被终止的训练现场保留，不评测：
    - STR 3407，约第 950 步，检查点到第 500 步；
    - c5 3407，约第 5250 步，检查点到第 5000 步。
  - 没跑：合成筛选、STR 筛选、EvRealHands 表、成本块。目标未达成。
  - 不重启、不追加；STOP 文件保留。续做须用户下令、先写工程修订、换新驱动目录，被终止的臂 fresh 重跑。
