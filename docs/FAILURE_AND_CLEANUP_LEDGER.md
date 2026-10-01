# EventHands1 失败证据与工作树清理账本

> 日期：2026-08-23  
> 目标：用一份文档替代散落的失败结论；清理前先区分“已证伪”“尚未实测”“必须保留的负对照”。

## 0. 证据优先级

遇到相互矛盾的历史描述，按以下顺序裁决：

1. 同一 checkpoint 的推理期消融、逐样本/逐序列递推评测；
2. 双副本或三 seed、固定 checkpoint 网格、配对统计；
3. teacher-forced probe、LOSO、反事实/数据审计；
4. 单次独立训练之间的差值；
5. 早期定性判断或后来已被修正的文档。

早期“LNES 使运动方向在网络前完全消失”的判断已被后续 probe 与 LOSO 修正。它不能作为新论文的当前结论。

> **协议声明（2026-09-05 起）。** 训练一律用默认划分 `splits_semkine.json`（9 受试者 72 条序列），
> 验证、选点、上报**统一只用留出受试者 zgz 的两条序列**（`zgz_global` + `zgz_local`；`val` = `val_core` = `test`）。
> 2026-08-22 → 09-05 期间的 5 受试者 / lr-lyq 选点线（S1–S46）的训练产物与日志已全部清除（见文末条目），
> 本文 §1–§2 中的具体数字都是在那条线上测的，**只作机制记录，不能与 zgz 协议下的数字直接比较**。
> 2026-09-06 起原 S37 及其后的全部实验（S37–S46）连同代码、配置、文档一起清除（文末条目），编号从 S37 重新开始：
> 2026-09-07 的 **S37 路由读出**（`S37_ROUTED_READOUT_PREREG.md`）在 zgz 上对 S36 −2.43 mm，已采纳，**当前臂与后续最优起点为 S37 路由读出**；
> 同日用户定义的 **S37 FK 图**（`S37_FKGRAPH_PREREG.md`）−1.07 mm 打平，成本 1/10，作对照与成本参照保留。
> 2026-09-18 用户手绘的 **S37 网格图**（`S37_MESHGRAPH_PREREG.md`，整张 FK mesh 作图 + LBS pooling）两种子分裂
> （20.48 / 29.30），不采纳；不稳定全在 root 旋转读出，手指读出稳定，保留作对照。
> EventGNN 线上仍有效的文档是上述 prereg；文献以 `docs/research/dir12_20260928/` 为准。

## 1. 当前可信事实

### 1.1 保留的基线机制

- `track_delta51` 的纯开环积分会严重漂移，不能作为主方法；
- `track_render51` 的 previous-state geometry + `prev + delta` 是必须保留的闭环机制；
- 零事件 identity、每段一次初始化、sequence reset、MANO 51D 状态和 recursive evaluator 必须冻结；
- checkpoint 选择应使用 recursive RA，而非 `val_loss`；
- PCA-6 不是当前主要瓶颈，Full-51 只带来很小变化；
- 低事件率域覆盖是实测的大瓶颈，域随机化双副本把历史递推 RA 从约 19.26 mm 改善到 worse-of-two 约 12.77 mm；
- 因此任何新 raw-event/GNN 模型都必须使用同一增强并对 12.77 mm 控制负责。

### 1.2 统计边界

历史相同配置的训练散布足以淹没 0.3–0.5 mm 的单点差异。主方法保留门槛应满足至少一项：

- pooled/paired 改善约 1.1 mm 或更大；
- paired bootstrap 95% CI 不跨 0；
- 精度在 2% 内，但真实 p50/p95 延迟或操作数至少改善 20%。

不保留仅在单 seed、单 best checkpoint 上好看的模块。

## 2. 失败/修正路线总表

| 路线 | 实测状态 | 根因 | 清理裁决 |
|---|---|---|---|
| 纯 `track_delta51` | FAIL，递推 RA 曾达到约 80–93 mm，绝对误差持续漂移 | 从 2D 事件弱观测量裸积分 51D；无每步几何锚定 | 保留最小负对照、指标和一个可复现配置；删除冗余 checkpoint/log |
| SPA/KSGN 加性语义图头 | FAIL/打平；同 checkpoint 清零图头只改变约 0.16 mm | 强 `fc -> 51D` 主路已完成任务，零初始化旁路没有动力长大 | 不在新架构中复活；若本机仍有孤立实现且无引用，删除源码/config/output |
| EDD 自由学习 evidence damping | FAIL | clamp/初始化导致死区；修正后 gate 又被 teacher-forcing 目标主动压制 | 删除旧 gate 实现；新方法只用单调固定信息阻尼起步 |
| J3D 辅助 FK loss | FAIL/无收益 | 高权重时梯度压倒参数损失；平衡后收益消失；单步目标未触及递推分布 | 删除专用实验脚本/config；通用 FK loss 函数可保留 |
| CMN 单轮闭环匹配噪声 | FAIL，local 退化 | 用当前策略残差建库把训练扰动从约 48 维收窄到 34–37 维，覆盖反而变差 | 删除 CMN 训练臂和产物；保留结论 |
| 旧“99.4% 正交=输入方向完全消失” | 已推翻 | 闭环差分把相关重估计噪声变成伪正交；TF probe 仍有方向信息 | 从主叙事删除；只留修正记录 |
| `jitter_static` 主指标 | 作废 | val 几乎没有满足静止阈值的帧，统计无效 | 从 Gate/主表删除；可换为速度分桶或 prediction acceleration |
| SO(3)+root-relative FK | 方向略好但严格 Gate 未过；absolute 明显退化 | 米制 SmoothL1 在厘米误差下梯度太弱，root-relative FK 不约束绝对平移 | 旧配方删除；保留稳定 quaternion/SO(3) 工具；新 loss 必须加 absolute FK 与 beta≈0.01 m |
| 独立 7ch/LBS 语义或 semsil | 未过严格精度 Gate；原 MSE 上低风险低收益，SO3FK 上不利 | 输入信息没有形成任务唯一通路；与 loss 存在交互 | 不作为主贡献；只保留 renderer/单元测试中通用语义查询 |
| 两遍 iterative render-and-compare (`it2`/`sem_it2`) | 明确撤回；absolute MPJPE 和 jitter 显著爆炸，延迟近翻倍 | 把第一遍预测再次反馈到第二遍，在递推 rollout 放大状态误差 | 若本机仍残留，源码/config/output 全删；仅在本文保留失败原因 |
| 16-node KinematicGNN（复制同一 global feature） | 尚不足以构成主方法 | 没有 event-to-joint 局部证据；若作为加法头会重复 KSGN 失败 | 不直接删除通用 tree layer；重构为 routed evidence 的唯一解码通路 |
| `AEGNNLite` 全包 `torch.cdist` | 结构性 NO-GO，不是有效异步实现 | \(O(N^2)\)、未来边风险、无 evolving state、global pooling 丢局部性 | 替换，不在主配置中保留；可留 tiny reference 测试或移入 archive |
| raw recurrent scan | 尚未完成公平 battle | B×T padding、逐事件 Python loop 在高事件率下风险大，但思想未被精度证伪 | 不误删；先作为 F0/raw control，优化 scan 后再裁决 |
| sparse cell frontend | 尚未完成公平 battle | 可能重新离散为粗网格，局部几何信息不足 | 保留作效率对照，不作为默认主方案 |
| domain randomization | PASS | 覆盖低事件率/采集域，双副本改善大且方向一致 | 必须保留并对所有新旧模型公平使用 |
| 08-22 起的全部训练：数据划分被静默替换 | 协议错误，非方法失败；08-22 后每次训练只用了 5/10 受试者，43% 数据被闲置在评测集 | `dataset.py` 划分解析是"优先 `splits_semkine.json`，否则回退 `splits.json`"，而前者在 08-22 被写入且切法完全不同；config 不记录划分，日志不打印受试者 | 划分已统一回采集时的 9 训练受试者 / zgz 留出；旧划分退役为 `_retired_splits_semkine_5v2v3.json`；训练元数据强制记录 `train_subjects`；`select_checkpoint.py` 加受试者泄漏断言。**所有 08-22–08-26 的数字必须标注为"5 受试者训练"协议，不可与新数字混用** |
| E5.5b 恒定 UNROLL_P=0.5(无退火) | FAIL,TF RA 42.5 mm / 递推 144.5 mm(step 3500 探针) | 从 step 0 就有一半样本以近随机的自预测做条件,网络最优解是无视 prev,坍缩回绝对回归(历史绝对回归正是 ~40 mm 档) | 产物留在 `outputs/semkine/s22_keg_halo_unroll_s34xx` 作负对照;修复=UNROLL_RAMP [500,2000] 线性退火(scheduled sampling 原文的 curriculum,Bengio et al. NeurIPS'15),重启臂名 `s22r_keg_halo_unroll` |
| 原 S37 FK 直连（`PREV_FK_DIRECT`，已不存在） | 5 受试者线上一次实验（08-30）：精度门槛过（RA −1.95）、**延迟门槛否**（6.01 → 6.37 ms）、abs +2.69，却只凭精度采纳为基座；09-04 以三条理由撤回。09-06 在 zgz 协议下重跑过一次，**已按用户指令作废并清除，不保留数字** | 撤回理由（09-04）：GT prev 的逐事件到关节/顶点解析距离+逆深度可能是把目标几何标注到事件上的闭环捷径；每包 MANO FK + 逐节点 top-k 近邻比光栅化更慢；方法退化为 MANO 先验驱动跟踪。流程错误：预注册门槛被当读数 | 09-06：`PREV_FK_DIRECT` / `_fk_extra` 路径、config、测试、工具、产物、日志、文档全部删除（文末条目）；叠在它上面的 S38–S42（读出 ×2、loss ×2、FK 方向 ×1）连同 S43–S46 同日清除 |
| **S37 FK 图**（`ENCODER: fk_graph`，`PREV_RENDER: false`，09-07，用户定义的形态） | **打平**（精度门未过 0.03 mm，两种子反向）。zgz 递推 RA 22.10（22.64 / 21.56）对 S36 23.17；abs 67.4（−11.6）；观测置零 +79 mm；延迟 5.38 ms（−22% 同会话）、0.084 G、0.23 M。H1–H6 见 `docs/S37_FKGRAPH_PREREG.md` §6–§7 | 渲染比较的稀疏对偶：在 prev 表面点上读"事件落在我周围哪里"，与 S36 信息量相当（TF 9.05 对 9.17）。赢在 root 平移（200 个节点的偏移场按序进 root 头，S36 的池化把它平均掉了）；输在手指（稀疏事件下 8 维汇总丢局部形状）。流速项在 50 ms 内无信息；旋转分量在训练分布里仍不可读。后半网格退化（27–31），最佳点都在前 2000 步 | 保留 checkpoint 与全部 JSON 作对照与成本参照；流速项可删；手指精度与 root 旋转为后续变量。**当前臂仍为 S37 路由读出** |
| **S37 路由读出**（`ROUTED_READOUT`，`PREV_RENDER: false`，09-07） | **PASS**。zgz 递推 RA 20.74（19.23 / 22.26）对 S36 23.17，两种子都更好，网格整体更好；abs 75.8；证据置零闭环 +7.6 mm；延迟 7.72 ms（+10%）。H1–H5 探针见 `docs/S37_ROUTED_READOUT_PREREG.md` §6–§7 | 增益是闭环性质：TF 单步差 0.6，放大率 2.58 → 2.12；每个关节只从自己附近的事件取证据，一处状态误差不再经共享池化向量污染全部关节。路由是固定几何（LBS 行），课程噪声无法把它摊平——这是与 KEG（学习路由进图结构，回路增益 0.91）和 S38（学习 σ 被摊到 365 px）的本质区别；oracle 路由反而更差，路由误差不是瓶颈。大噪声课程下路由纯度 0.14，但没有让头忽略证据 | **采纳，当前臂 S37**。保留 checkpoint、选点 JSON、主行与探针 JSON。剩余短板与 S36 共有：root 旋转盲（增益 ≤ 0.13）、更新方向余弦 0.26、TF 单步 9–10 mm > 真实步进 5.8 mm |

## 3. 源码/产物分类

### 3.1 必须保留

```text
model/mano_layer.py 及 MANO assets
现有 recursive evaluator
track_render51 / track_render51_domrand 的控制配置
低事件率域随机化与相机一致增强
semkine/events.py 中微秒时间戳与 ragged packet 数据合同
SO(3)/Lie 数值稳定工具及其测试
固定 checkpoint 网格、per-sequence 指标和 paired 统计工具
```

负对照也不是“垃圾”：`track_delta51`、普通 raw recurrent、global pooling AEGNN 等应保留一个最小可复现配置和最终指标，论文消融需要它们；冗余中间 checkpoint、TensorBoard、临时脚本和重复报告才可删除。

### 3.2 可自动删除的生成物

以下不包含科研证据，可由清理工具自动列入候选：

```text
**/__pycache__/
**/*.pyc, **/*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
**/*.tmp, **/*.swp, **/*~
nohup.out
空的临时目录
```

普通 `*.log` 可能包含唯一训练证据，因此默认只报告，不自动删除；只有相应指标、config、commit、checkpoint hash 已写入合并账本后，才显式加入删除清单。

### 3.3 只允许显式删除的研究产物

不得对整个 `outputs/` 做递归盲删。需要逐个传给 cleanup tool，例如：

```text
outputs/.../it2
outputs/.../sem_it2
outputs/.../cmn
outputs/.../edd
outputs/.../ksgn_rebuild_failed
```

实际目录名必须以本机 `find` 结果为准，不能根据文档猜路径。每个目录删除前至少保留：

```text
config 或完整 config diff
起止 commit
selected checkpoint 名与 sha256（若论文还需复现）
最终 per-sequence metrics
Gate verdict
失败原因
```

若该失败臂仍要作为论文负对照，保留一个最终 checkpoint；其他 checkpoint 可删。

### 3.4 失败 `.py` 的删除条件

`.py` 源码只有同时满足以下条件才能删除：

1. 已由本文表格明确归类为 retired；
2. `git grep`、config、shell script、test 和 import graph 中引用为 0；
3. 不包含仍被主线复用的通用数学/数据函数；
4. 已提交到 Git，或先生成 patch/commit hash 作为可恢复证据；
5. 使用 cleanup tool 的 `--allow-source-delete` 显式授权。

不能因为文件名带 `test`、`probe` 或 `debug` 就删除；很多判别性 probe 比模型代码更有科研价值。

## 4. 建议的本机清理流程

```bash
cd /data1/lyq/code/EventHands1

git branch --show-current
git rev-parse HEAD
git status --short

# 1) 默认只审计，不删除
python tools/cleanup_research_tree.py \
  --root . \
  --report artifacts/cleanup/cleanup_audit.json

# 2) 查看现有实验目录；不要按文档猜路径
find outputs -maxdepth 3 -type d | sort
find . -type f \( -name '*.log' -o -name '*.tmp' -o -name '*.pyc' \) | sort

# 3) 对明确失败目录做第二次 dry-run
python tools/cleanup_research_tree.py \
  --root . \
  --failed-dir outputs/真实失败目录A \
  --failed-dir outputs/真实失败目录B \
  --report artifacts/cleanup/cleanup_failed_arms.json

# 4) 检查报告和 git diff 后才执行
python tools/cleanup_research_tree.py \
  --root . \
  --failed-dir outputs/真实失败目录A \
  --failed-dir outputs/真实失败目录B \
  --apply \
  --report artifacts/cleanup/cleanup_applied.json

# 5) 源码单独处理；必须显式授权
python tools/cleanup_research_tree.py \
  --root . \
  --failed-file path/to/retired_experiment.py \
  --allow-source-delete \
  --apply \
  --report artifacts/cleanup/cleanup_source.json

# 6) 清理后回归
git status --short
python -m pytest tests -q
```

清理工具默认拒绝：

- `.git/`、`data/`、MANO asset；
- 未显式给出的 checkpoint；
- 源码 `.py`；
- repo root 外路径；
- symlink 指向的外部目录；
- `main/master` 上的 `--apply`，除非显式 `--allow-protected-branch`。

## 5. 文档地图（2026-09-29 再收敛）

活跃入口：

```text
docs/FAILURE_AND_CLEANUP_LEDGER.md          # 失败证据、协议、清理；本文
docs/S37_ROUTED_READOUT_PREREG.md           # 当前臂
docs/S37_FKGRAPH_PREREG.md / S37_MESHGRAPH_PREREG.md / S37_MESHQ_PREREG.md
docs/S37_ROOT_INNOVATION_PREREG.md / S37_ROTW_CNNROOT_PREREG.md
docs/S37_XYZ_CANDIDATE_20260929.md        # XYZ 候选：结构、信息检查、训练准入
docs/网络结构分析.md                        # S37 结构问题、文献对照、下一轮实验（两轮）
docs/research/dir12_20260928/               # 当前文献
```

仍有唯一数字、不并入本文：

```text
docs/GNN_ARMS_ARCHIVE_20260828.md           # 已删 KEG/CellGNN 臂的网格，无法从别处重建
docs/POSITIONING_VS_E3DPSM.md               # 常数增益融合不能当创新点
docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md  # 保持率惩罚干预失败
```

2026-09-29 删除（结论已在本文，或被 `docs/research/dir12_20260928/` 重读覆盖）：`experiment_history.md`、`PLAN_SELECTION_VERDICT_20260825.md`、`ASYNC_SPARSE_SOTA_MASTER_VERDICT_20260826.md`、`debug_e55b_unroll_20260825.md`、`EVENT_GNN_SURVEY_20260829.md`、`semkine/{EXPERIMENT_LOG,CLAIM_MATRIX,FAILURE_CASES,ARCHITECTURE_AUDIT}.md`。

2026-09-24 已删（内容已进本文，无独立数字）：`EVENT_KINEGRAPH_MASTER_PLAN.md`、`EVENT_KINEGRAPH_EXECUTION_LOG.md`、`EXPERIMENT_SYSTEMATIC_SUMMARY.md`、`debug_closed_loop_diagnosis.md`、`semkine/FINAL_REPORT.md`。

## 6. 新路线的 kill rules

为避免再次堆叠方法，EventKineGraph 也必须可被否证：

- causal hash graph 若不能通过线性内存/高事件率 p95 Gate，停止，不改成更大图；
- LBS router 若 oracle 都不能比 global pooling 改善 active-joint error ≥5%，停止 kinematic GNN；
- GNN 若不优于 matched-parameter shared MLP，删除 GNN，论文保留 router + MLP；
- evidence damping 若只改善异常 low-event 序列、却损害 normal bucket >2%，回退；
- incremental runtime 若与同步前缀不等价，不能写 asynchronous inference；
- 最终若只超过 19.26 而不能超过同增强 12.77 控制，不允许写 accuracy SOTA；
- 未有公开同协议测试时，不允许把不同数据集上的 EvHandPose/Ev2Hands 数字直接并表声称超过。

## 7. 本分支没有做的事

- 没有删除本机 `/data1` 的任何内容；
- 没有运行 CUDA、DDP、完整数据集或训练；
- 没有读取两个本地 research skill；
- 没有把结构推导当成已验证增益；
- 没有把 60 角色模拟评审冒充真实专家实验。

这份账本的作用是让下一次 Codex 执行先清理证据和协议，再实现最小 E1/E2，而不是重复启动已经被否证的完整大网络。

## select_checkpoint.py 会静默选中发散的 checkpoint（2026-08-26 修）

`best = min(rows, key=lambda r: r[KEY])`：NaN 与任何数比较都返回 False，
所以当网格里第一个 checkpoint 发散成 NaN 时，`min` 永远不会替换它，直接把
坏模型选出来。运行时证据（`logs/seed2_final_1p05.log`）：

```
  step=500    RA=     nan  abs=     nan
selected step=500  RA=nan  .../s24_lowg_1p05_bnfix_s3408-step=500.ckpt
```

正确答案是 step=5500 / RA=19.6107。触发条件是 λ ≥ 1 的 LowG arm 早期不稳定
（同一网格上还有 step=1500 RA=105.17、step=6000 RA=71.36 这类尖峰），
所以任何高 λ / 高学习率的 run 都可能中招，而且**不报错**。

已修：先按 `math.isfinite` 过滤，全部非有限则 `SystemExit`，
过滤掉的 step 会打印出来。受影响的 `s24_lowg_1p05_bnfix_s3408/selection_*.json`
已按存档 grid 重算修正（无需重跑 checkpoint 评估）。

教训：任何"取最小值选模型"的地方都要先过滤非有限值——
发散产生的 NaN 会伪装成"最优"。

## 数据划分被静默替换，一半训练受试者消失了 5 天（2026-08-27 修）

**这是本项目至今影响面最大的坑，而且它不报错、不警告、不留痕。**

`semkine/dataset.py` 的划分解析是"优先文件名，否则回退"：

```python
mp = Path(manifest_path) if manifest_path else root / "splits_semkine.json"
if mp.exists():
    m = json.loads(mp.read_text())
    if split in m:
        return [(s, m[split]["legacy_dir"][s]) for s in m[split]["trials"]]
legacy = json.loads((root / "splits.json").read_text())   # 永远走不到
```

数据根目录里同时存在两个划分文件，对**同一批 74 条序列**给出完全不同的切法：

| 文件 | 日期 | train | 留出 |
|---|---|---|---|
| `splits.json`（采集时划分，作者本意） | 08-11 | **9 受试者 / 72 序列** | zgz / 2 序列 |
| `splits_semkine.json` | 08-22 | **5 受试者 / 40 序列** | lr,lyq（val）+ ycy,ylf,zgz（test） |

`splits_semkine.json` 存在，于是 `if mp.exists()` 命中，legacy 那行**永远执行不到**。
08-22 之后的每一次训练都只用了 5 个受试者，而 config 里没有任何一行提到划分，
作者的认知和 `splits.json` 一致（10 人分出 1 个），日志也从不打印受试者列表。

运行时证据（`outputs/semkine/s1_track_domrand_s3407/training_metadata.json`）：

```
train_sequences = 40 条
  -> 受试者(5) = ['ch', 'lfz', 'lpc', 'ly', 'lyh']
train_samples = 2491120
```

**后果不止是数据少了。** 08-22 的重划分本意是给出 subject-disjoint 的三段划分，
让选点集和上报集互不重叠（`_leakage` 记录 `clean`，动机是合理的），
代价是把 lr、lyq、ycy、ylf 这 4 个受试者、32 条序列（**全数据集的 43%**）
从训练搬到了评测。于是：

- §12 测出的 9.2 mm"跨受试者泛化差距"，是在**训练受试者只有一半**的条件下测的；
- 项目门面数字 16.638 mm 是在 `val_core` = lr,lyq 上测的，
  而这两人在作者本意的划分里**属于训练集** —— 也就是说这个数字一直是
  "5 受试者训练 + 跨受试者评测"，比作者以为的协议严格得多，两者不可直接比较；
- §12.4 预注册的"留一受试者"闸门因此是多余的：不需要重新留出，
  4 个受试者的数据本来就闲置在评测集里。

已修：

1. `splits_semkine.json` 现在就是采集时划分（9 受试者训练 / zgz 留出），
   由 `tools/build_splits_semkine.py` 生成，5/2/3 那份退役为
   `_retired_splits_semkine_5v2v3.json`（仅供追溯 docs 里已有的旧数字）。
   因为改的是**默认路径**，20+ 个工具和所有 config **一行都不用改**就统一了。
2. `semkine/dataset.py` 新增 `splits_manifest(cfg)`：config 可用
   `DATA.SPLITS_MANIFEST` 显式钉住划分，不设则用默认。
3. `semkine/train.py` 在 `training_metadata.json` 里记录 `splits_manifest`
   和 `train_subjects`，并在启动时打印划分文件名与受试者列表。
4. `tools/select_checkpoint.py` 断言被评测的受试者不在该 run 的训练集里
   （`--manifest` 可覆盖划分，用于拿旧 checkpoint 跨协议评测）。
5. `tools/run_s26_9subj.sh` 启动前校验"9 训练受试者 + 留出 zgz"，不符就拒跑。

**教训（两条，都比这个 bug 本身更通用）**：

- **"优先文件名，否则回退"的解析顺序必须把选中结果打印出来。**
  任何一段 `if path.exists(): use it` 的默认查找逻辑，都是一个可以被
  凭空出现的文件劫持的开关；不打印就等于没有。
- **不记录划分的 config 是不可审计的 config。** 训练元数据里必须有
  受试者列表这种"人一眼能看出不对"的字段。这个 bug 存活 5 天的唯一原因，
  是没有任何一处输出会让人发现受试者从 9 个变成了 5 个 ——
  `train_samples = 2491120` 这种数字没人能看出异常。

遗留待办：`data/hand_data51/buckets/{train,val,test}_step50.json` 是 08-22
按旧划分生成的（`val_step50.json` 里是 lr+lyq，不含 zgz）。
`masks_for_sequence` 对找不到的序列返回 `{}` 而不报错，
所以这是**同一类静默降级**。它不影响选点（`track_sequence` 的
`bucket_manifest=None`），但按新划分做分桶分析前必须重新生成。

## 内联 `bash -c` 等待器里的 `pgrep -f` 匹配到了自己，白等 5 小时（2026-09-05）

为了省掉 `finish_s44.sh` 对已选过的种子 3407 的重复选点，我用一行
`nohup bash -c 'while pgrep -f "semkine/train.py --config configs/semkine/s44_meshroot" ...; do sleep 60; done; ...'`
替换了它。训练 03:05 结束，选点直到 08:33 都没有开始：`pgrep -f` 匹配完整命令行，
而这段模式串就写在等待器自己的 `bash -c` 命令行里，于是永远能匹配到一个"活着的训练进程"——它自己。
脚本文件里的 `finish_s44.sh` 没有这个问题，因为模式串在文件内容里、不在进程命令行上。

教训：**用 `pgrep -f` 做存活检测的循环，模式串绝不能出现在检测者自己的命令行上**；
写进脚本文件，或者用 `pgrep -f "[s]emkine/train.py ..."` 这种自排除写法，
并且等待器要打印"检测到 N 个进程"这类心跳——静默的 `sleep 60` 循环和挂死没有区别。
没有产生错误结果，只是浪费了半天：3408 的选点改为直接启动。

同一天第二次踩中：用 `pkill -f "tools/run_s46.sh"` 停训练时，模式串在执行它的那条 shell 命令行里，
`pkill` 先杀掉了正在执行的 shell 自己，后面的命令没有运行、也没有任何输出。
`pkill -f` 与 `pgrep -f` 是同一个坑，同一个解法：`pkill -f "tools/[r]un_s46"`。
`finish_s46.sh` 起已经改为自排除写法并每 5 分钟打印心跳。

同日第三件小事：`finish_s46b.sh` 由 `finish_s46.sh` 做字串替换（`s46_rotcur` → `s46b_rotcur_s44base`）生成，
选点日志路径 `logs/s46_select_s${seed}.log` 里没有这个子串，没被替换，S46b 的选点输出覆盖了 S46 的选点文本日志
（选点 JSON 在各自 run 目录下，未受影响，网格无损失）。教训：**用字串替换派生脚本后，对生成物 `rg` 一遍所有路径**，
凡是带"前一代名字"的路径都要逐个核对，而不是只看主名字替换成功。

## 清除 5 受试者 / lr-lyq 选点线，验证统一为 zgz（2026-09-05，用户指令）

**为什么。** 08-22 的重划分把 lr、lyq、ycy、ylf 共 32 条序列（43% 的数据）从训练搬到评测，S1–S46 全部在 5 受试者上训练、
在 lr/lyq 的 8 条 `val_core` 上选点；主表（`tools/make_main_table.py`）的所有行则是 9 受试者训练、zgz 上验证与上报。
两套协议的数字互不可比（S46 在 val_core 上对 S36 的 −5.6 mm 到 zgz 上只剩 −0.2 mm，其中含在 val_core 上选点的增益 2.4–4.7 mm）。
用户决定：只保留一套协议——**训练 9 受试者，验证/选点/上报只用 zgz**。

**删了什么。** `outputs/semkine/` 下 68 个目录 + 134 个散落 JSON（62.4 GiB，含 S1–S46 全部 checkpoint、TensorBoard、
选点 JSON、探针/主行 JSON、S22 失败对照、mainline/s0/s7/s17 产物），`logs/` 下全部 115 个文件，本次调试会话日志。
保留：`outputs/semkine/s26_abs_9subj_*`（9 受试者、仅元数据）、`data/hand_data51/_retired_splits_semkine_5v2v3.json`（仅供追溯 docs 数字，已被守卫拒绝加载）。
删除清单：`outputs/_archive/semkine_5subject_line_20260905_deletion_report.json`；
文本类产物（495 个 json/log/md，13 MB）打包在 `outputs/_archive/semkine_5subject_line_20260905.tar.gz` 作最低限度的证据保全
（§3.3 的要求：config、选中 checkpoint 名、最终 per-sequence 指标、verdict 都在里面），不需要可直接删。

**改了什么。**
- `semkine/dataset.py::sequences_for_split`：文件名以 `_retired` 开头的划分一律拒绝加载（`ValueError`），
  除非显式设 `EVENTHANDS_ALLOW_RETIRED_SPLITS=1`（只用于给归档产物补分）。`tests/test_s1_dataset.py` 新增两项：
  默认划分 train 为 9 受试者 72 条、`val`/`val_core`/`test` 都恰为 zgz 两条；退役划分被拒绝且 `configs/` 下无任何钉定。
- `configs/semkine/*.yaml` 共 14 个：删除 `DATA.SPLITS_MANIFEST: _retired_splits_semkine_5v2v3.json`，S36/S43–S46b 头部加协议注记。
- `tools/make_s36_row.py`、`tools/profile_s36_latency.py`、`tools/probe_s36_nodes.py`：不再写死退役划分与选点文件名，
  从配置取划分、按 `selection_val_core_step50*.json` 找选点文件。`select_checkpoint.py` 的默认 `val_core` 现在就是 zgz。

**后果。** 这条线上所有已发表在 docs 里的数字（S36 31.67 / 45.65，S43–S46 的每一个门槛与探针读数）失去了可复现的 checkpoint，
只剩文本记录。新协议**选点集与上报集重合**（都是 zgz 两条序列，2590 帧），数字会含选点增益且噪声大，
门槛与种子差的解读要相应放宽——这是 08-22 重划分本想解决而现在被明确放弃的问题。

## zgz 协议下的第一批数字：S36 重训（2026-09-06 12:42 → 17:18，`tools/run_zgz_protocol.sh` 的 09-05 版）

9 受试者 72 条序列训练（`train: 72 seqs / 4544300 samples / 9 subjects`），固定网格，zgz 上递推 RA 选点，两种子。
主表格式（`tools/make_s36_row.py`，`outputs/semkine/s36_eventgnn_main_row.json`，延迟在空机上测、Full 锚点 1.415 ms）。
同一批次还重训了当时的网格关联臂，其行与读数已随 09-06 的清除一并删去；`logs/run_zgz_protocol_outer.log` 仍是这批训练的时间记录。

| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | abs | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|---|
| S36 EventGNN（渲染 prev） | 28.43 | 18.60 | 24.26 | 15.00 | **23.17**（21.42 / 24.92） | 79 | 7.04 ms | 0.827 G | 0.95 M |
| **S37 路由读出**（状态无关图 + FK 路由逐关节证据，09-07，`docs/S37_ROUTED_READOUT_PREREG.md`） | 26.41 | 15.82 | 21.40 | 12.39 | **20.74**（19.23 / 22.26） | 76 | 7.72 ms | 0.827 G | 0.73 M |
| **S37 FK 图**（用户设计：FK 是图、事件是观测、无 token 无几何支路，09-07，`docs/S37_FKGRAPH_PREREG.md`） | 30.92 | **14.43** | 26.47 | **11.59** | **22.10**（22.64 / 21.56） | **67** | **5.38 ms** | **0.084 G** | **0.23 M** |
| S37 网格图（用户手绘：整张 FK mesh 作图、面 1-ring 边、LBS pooling 到 16 关节，09-18，`docs/S37_MESHGRAPH_PREREG.md`） | 36.32（3407 **25.80** / 3408 46.85） | 14.96 | 27.63 | 11.69 | 24.89（**20.48** / 29.30） | 59 | 11.99 ms | 0.314 G | 0.44 M |
| 参照：`track_render51_dr_*`（CNN + 渲染 + 域随机化，同协议，主表已有） | 13.3–17.8 | 9.3–12.6 | 11–15 | 7–10 | **11.9–13.3** | 58–97 | ≈1.75–2.5 ms | 1.65 G | 11.2 M |

读法：

- 在完全相同的协议下，事件 GNN 线（S36 23.2，S37 20.7）比主表里已有的 CNN+渲染+域随机化基线（11.9–13.3）**差 8–10 mm**。
  §1.1 早就写了"任何新 raw-event/GNN 模型都必须对 12.77 mm 控制负责"，S36 以来的全部比较都在 lr/lyq 上做、
  从未对这条控制负责过——这是 08-22 重划分留下的最大盲区。
- S36 两种子差 3.5 mm（21.42 / 24.92），zgz 只有两条序列、选点集与上报集重合：任何 1 mm 量级的单点差异都在噪声里。
- S37 路由读出对 S36 **−2.43 mm，两种子都更好**（−2.19 / −2.66），整张网格中位数也低 2.3 mm，证据置零后闭环 +7.6 mm（承重）。
  增益全在闭环放大率（2.58 → 2.12），TF 单步反而差 0.6 mm；oracle 路由（GT prev 路由）让闭环更差 2.5 mm，
  KEG 式"路由误差回灌"不成立。**采纳，当前臂 S37。**
- S37 FK 图（用户定义的形态）对 S36 **−1.07 mm，两种子方向相反**（3407 +1.22，3408 −3.36），按 |Δ| < 1.1 判**打平**；
  abs −11.6（root 平移解开：腕误差 64 → 46 mm，root 平移响应增益 0.32 → 0.90），global −4.2 但 local +2.5；
  观测置零闭环 +79 mm（承重），oracle 归属反而崩（观测是相对 prev 量的差，不是捷径）；流速项 (b_u, b_v) 对结果零贡献（H6）；
  root 旋转仍盲（增益 ≤ 0.19）。成本：同会话延迟 −22%，FLOPs 1/10，参数 1/4，显存 1/5，无节点上限。
- S37 网格图（用户手绘图）对 fk_graph **+2.79 mm，两种子分裂**（3407 −2.16 → 20.48，3408 +7.74 → 29.30），**不采纳**。3407 的 TF 单步 8.07 是全线最低；
  两种子的手指（旋转对齐后）误差相同（闭环 17.9 / 18.4，fk_graph 24.7 / 26.2；整张网格上 20.4 / 22.9 对 21.0 / 22.2，打平），差别全在
  **root 旋转**：闭环全局旋转中位数 15.6° 对 28.2°，整张网格 RA 与旋转误差相关 0.96 / 0.89、与手指误差相关 ≈ 0，网格 spread 14 / 26 mm（fk_graph 9 / 6）。
  原因是图里只有面 1-ring 边（3 跳 ≈ 1–2 cm），root 头从 16 个各自局部的 LBS 均值里线性拼全局旋转，没有 fk_graph 的关节节点 + 运动学树那条长程通路。
  观测置零 +22.6 / +27.4（承重）。延迟 11.99 ms（B = 1 时跳跃泛洪与 z-buffer 的小核启动），FLOPs 0.314 G，参数 0.44 M。

结论：**这条线的下一个问题仍是为什么整条事件 GNN 线在 zgz 上落后 CNN+渲染基线 8–10 mm**。任何后续臂的采纳门槛都应改为
对 `track_render51_dr_*` 的 12–13 mm 负责。S37 的探针指出三个共有短板：root 对旋转近乎全盲（响应增益 ≤ 0.13）、
姿态更新方向与所需方向的余弦只有 0.26、TF 单步误差（9–10 mm）大于真实每步运动（5.8 mm）。

## 清除 S37 之后的全部实验（S38–S46）；S37 的 zgz 重跑随后也作废清除（2026-09-06，用户指令）

**为什么。** FK 直连只做过一次实验（S37）；S38–S42 是它被采纳后叠在上面的五轮读出/loss/FK 方向实验，S43–S46 是 S36 基座上的
网格关联线。zgz 协议下 S46 对 S36 只差 0.57 mm（小于种子差），旧线上的 −5.6 mm 没有迁移。用户决定：只重跑 S37 这一个实验
（两种子、对现有 S36 zgz run），并把"S37 后面所有的实验内容、结果、代码"清除。
**同日晚，用户判定 09-06 的 S37 重跑有问题，指令连同其代码、日志、输出全部清除，不保留数字**；`model/model.py` 里的
`PREV_FK_DIRECT` / `_fk_extra` 路径也随之整块删除，仓库只剩 S36 的渲染条件化路径。

**快照决定。** 09-02 之后的全部工作（含 S43–S46 的代码与文档）在清除时**未提交**；用户明确选择**不留快照直接删**。
这偏离本文 §3.4 第 4 条（删源码前须有 commit 或 patch 作为可恢复证据），在此如实记录：S37–S42 的 config/test/tool
仍可从 `e9c4435` 恢复，S43–S46 的代码、配置、探针与两份文档不可恢复。

**删了什么。**
- 代码：`semkine/mesh_assoc.py`；`model/model.py` 里的 `ENCODER_JOINT_READOUT`（S38）、`FK_DIRECTIONAL`（S40）、`ENCODER_ATTN_POOL`（S41）、
  `mse_51d` 上的 `ABS_FK_WEIGHT` 嫁接（S42）、全部 `MESH_ASSOC*`（S43–S46）分支与键；`semkine/event_gnn.py` 的 `joint_queries` /
  `attn_pool` / `return_nodes`；`semkine/frontends.py` 的对应转发；`semkine/dataset.py` 的 `PREV_NOISE_P_ROT` 旋转课程（S46）
  及 `TRACK_KEYS` 里的四个键。`so3_trans_fk` 路径上的 `TRANS_BETA` / `ABS_FK_WEIGHT`（S4 实现，`tests/test_s4_loss.py`）保留。
- 测试：`tests/test_s43_mesh_assoc.py`，`tests/test_s1_dataset.py` 的两项 S46 测试（09-05 的划分测试保留）。
- 工具：`tools/probe_event_mesh_assoc.py`、`probe_s36_prev_offset.py`、`probe_s43_root_decomp.py`、`probe_root_rotation.py`、
  `probe_root_rotation_linear.py`、`probe_root_rotation_shrinkage.py`；`run_s43/44/45/46/46b.sh`、`finish_s43/44/45/46/46b.sh`。
- 配置：`configs/semkine/s43_meshassoc_s3407.yaml`、`s44_meshroot_s3407.yaml`、`s45_meshrootjoints_s3407.yaml`、`s46_rotcur_s3407.yaml`、
  `s46b_rotcur_s44base_s3407.yaml`；S38–S42 的 config/test/tool 维持 09-04 的 `git rm`。
- 文档：`docs/S43_MESH_ASSOC_PREREG.md`、`docs/S44_S45_ROOT_EVIDENCE.md`、`docs/S37_S42_FKDIRECT_RETRACTION.md`；
  `docs/EVENT_GNN_SURVEY_20260829.md` 删去 S38–S42 的设计、预注册与结果，只留三个角度的机制判定。
- 产物：`outputs/semkine/s46_rotcur_s340{7,8}/`（280 MB）、`s46_rotcur_main_row.json`、`logs/s46_rotcur_*`、`logs/row_s46_zgzproto.log`；
  归档包 `outputs/_archive/semkine_5subject_line_20260905.tar.gz` 重新打包，剔除 91 个 S43–S46b 条目（495 → 404 个文件），
  删除清单 JSON 作为记录保留。
- S37 重跑（同日作废）：`configs/semkine/s37_fkdirect_s3407.yaml`、`tests/test_s37_fk_direct.py`（`git rm`）、为它新建的
  `tools/profile_latency_ab.py`、`tools/probe_s37_shortcut.py`、`tools/make_s37_figure_simple.py`、`docs/S37_FKDIRECT_ZGZ_PREREG.md`、
  `docs/assets/s37_fkdirect_simple.png`，两种子的 run 目录（326 MB）、主行/探针/延迟 JSON 与全部 `logs/*s37*`；
  `model/model.py` 的 `PREV_FK_DIRECT` 键、`prev_fk` 分支、`_fk_extra` 与 `FK_*` 常量。

**改了什么。** `tools/run_zgz_protocol.sh` 改为 `run_zgz_protocol.sh <arm> [seeds]`（同一臂两种子并行、各自选点；现有 S36 run 的出处）。
清除后 `python -m pytest tests -q` 全部通过，`rg` 在 `semkine/ model/ tools/ tests/ configs/` 下对 S37–S46 的键与名字零命中。

**流程教训。** 09-04 的撤回文档第 60 行把 S37 的数字写成"9 训练受试者 / zgz 留出协议"，而删除前的 config（`e9c4435`）钉着退役的
5 受试者划分——协议标签写错比数字写错更难被发现，任何数字旁边的协议标签都应从 `training_metadata.json` 的 `train_subjects` 抄，不凭记忆写。

## S37 路由读出：方案 A 的学习式实现在 zgz 协议下过门（2026-09-07 11:25 → 13:48）

用户提出的方案 A：上一状态不再渲染，事件图状态无关；节点特征经消息传递后按上一状态 MANO FK 的 LBS 几何路由到
16 个关节，逐关节回归 Δθ。实现为 `MODEL.ROUTED_READOUT`（`semkine/routed_readout.py`，7 项契约 `tests/test_s37_routed_readout.py`），
对 S36 恰两处不同（`PREV_RENDER: false` + 路由读出）。两种子并行训练 1 小时 45 分，固定网格 zgz 选点。

结果与判读见 `docs/S37_ROUTED_READOUT_PREREG.md` §6–§7；zgz 表与 §2 已更新。要点：−2.43 mm、两种子一致、网格整体更好、
证据承重（置零 +7.6 mm）；增益来自闭环放大率而非单步；oracle 路由反而更差，路由误差不是瓶颈；大噪声课程下路由纯度 0.14
但头没有学会忽略证据。**采纳为当前臂。**

流程记录：训练前 40 分钟与用户自己的另一个 7 卡任务共用 GPU 4–7（每卡 5–8 GB、26–30% 利用率），显存 44/46 GB 偏紧但未 OOM，
只影响步时（1.3 → 1.0 s/it）。探针工具的 oracle 路由通过模型上一个只供探针用的 `route_prev_override` 钩子实现，默认 `None`。

## S37 FK 图：用户定义的形态（FK 是图、事件是观测、无 token、无几何支路）在 zgz 上打平（2026-09-07 20:13 → 23:30）

实现为 `MODEL.ENCODER: fk_graph`（`semkine/fk_graph.py`，9 项契约 `tests/test_s37_fk_graph.py`）：16 关节 + 192 顶点 + 背景共 209 个节点，
边 = 静止姿态网格 kNN / LBS top-2 / 运动学树（边特征只有边类型）；全部事件按投影就近归属到节点，每节点 8 维观测
（计数、偏移、离散度、时间、极性、局部流速）；EdgeConv×3；手指头读自己关节节点，root 读 16 关节 + 背景。
结果与 H1–H6 判读见 `docs/S37_FKGRAPH_PREREG.md` §6–§7；zgz 表与 §2 已更新。

三条工程记录：

- **DDP 在验证边界卡死。** 两种子在 step 3378 同时挂住，30 分钟后 NCCL watchdog 杀进程（两个独立进程同一步，无 OOM）。
  给 `semkine/train.py` 加了 `--resume`（Lightning `ckpt_path`），从 `last.ckpt`（step 3000）以单卡 batch 1024 续训到 6000
  （有效 batch 与 LR 不变）。step ≤ 3000 与 ≥ 3500 的 checkpoint 来自两个阶段，元数据两份都保留。
- **float32 原子加不可复现。** `index_add_` 的求和顺序随机，~1e-7 抖动被递推放大到 0.0755 mm，超过 0.05 mm 复现门。
  改为 float64 累加再转 float32（`_segment_sums`），5/5 逐位相同，无额外开销；重选后选中 step 不变，复现漂移 0。
  **教训：任何进入递推评测的 scatter/index_add 都必须 float64 累加**，否则选点与主行永远对不上。
- 流速项 (b_u, b_v) 是"变化信息"的字面实现，H6 证明它在 50 ms 包内没有信息（真实每步运动 5.8 mm ≈ 几个像素，单节点内的时间回归
  被噪声淹没）；有用的"变化"只存在于 prev → 事件的偏移里，而偏移项已经给出了。

## S37 网格图：用户手绘图（整张 FK mesh 作图、LBS pooling 到关节）在 zgz 上两种子分裂，不采纳（2026-09-18 11:28 → 15:30）

实现为 `MODEL.ENCODER: mesh_graph`（`semkine/mesh_graph.py`，12 项契约 `tests/test_s37_mesh_graph.py`）：prev 经 FK 的 778 个顶点 + 背景共 779 个节点，
边 = 网格面 1-ring；背面剔除 + 点溅 z-buffer 决定可见顶点，跳跃泛洪查找表把事件分给最近可见顶点（≤ 16 px）；每顶点 6 维观测（流速项按 fk_graph H6 去掉）；
EdgeConv×3；固定蒙皮权重把顶点特征池成 16 个关节证据 `[mean ‖ max ‖ cov]`，手指头读自己的证据 + prev 角，root 读 16 个证据平铺 + 背景。
状态与 Δ 是 51D MANO 参数（图里的"关节点坐标"改读为关节参数：FK 要旋转，关节位置里没有 twist）。对 fk_graph 只差节点 / 边 / pooling 三处。
结果与 H1–H4 判读见 `docs/S37_MESHGRAPH_PREREG.md` §6–§7；zgz 表与 §2 已更新。

要点：递推 RA 20.48 / 29.30（均 24.89）对 fk_graph 22.10；**手指读出成立且两种子一致**（TF 手指误差 −14%，闭环网格均值与 fk_graph 打平），
**root 旋转读出不稳定**（网格 RA 与旋转误差相关 0.96 / 0.89，与手指误差 ≈ 0；旋转 sd 7–9° 对 fk_graph 2–4°）。原因：图里只有 1-ring 边，
root 头没有长程通路；fk_graph 的关节节点 + 运动学树边正是这条通路。下一个单变量：加回关节节点 / 全局节点，其余不动。

三条工程记录：

- **顶点法向的 float32 原子加**（`facing_camera`）与 fk_graph 的 `_segment_sums` 同源：掠射角顶点的可见性可能在两次运行间翻转。训练中改为 float64 累加，
  选点与主行都在修复后运行，复现漂移 0.0000 mm（两种子）。
- **B = 1 延迟 11.99 ms** 不是 FLOPs（0.314 G）而是小核数量：跳跃泛洪 5 轮 × 8 邻域、z-buffer `max_pool2d`、779 节点 EdgeConv。训练吞吐（1.5 it/s @ 2 × 512）不受影响。
  若保留此形态，事件数小时退回一次 `cdist` 或用 CUDA graph 固化。
- 探针脚本里一处自己的错误值得记：`Tensor.norm(-1)` 是 p = −1 的范数（标量），不是 `dim=-1`；中间一版探针因此全零，靠"TF 比闭环还差"的反常读数发现。
  另一处：短于一步的 `valid_runs_ms` 段没有窗口，构造 prev 索引时必须跳过，否则 prev 与窗口错位。`tools/probe_s37_meshgraph.py` 已含两处修正。

## 2026-09-28：按用户指令清除 S37 之后的全部实验与记录，回到 S37 起点

用户指令"S37 之后的实验以及记录全部删除，重新分析"。保留 S37 家族四臂（`2649ea0`：routed / fkgraph / meshq / meshgraph），**不留快照直接删**；
已提交内容仍可从 `85d76a1` 取回，未提交部分（下列 `.experiments/` 等目录、8 份 terminal prereg、本文 09-26 之后的追加段）不可恢复。

- 删：S38 三维网格图 / georoot、S39、778 顶点事件引导图与记忆臂、09-24 深度 / xyz 探针、09-25–26 异步研究线（约 60 份 `S37_*` 后续 prereg、
  `docs/research/s37_async_20260925/`）及 x1 / x1c / k0–k2 / c0 / sc0 / cc0 / mp1 各臂的代码、配置、测试、工具、run 目录（≈ 3.1 GB）与日志；
  `.experiments/`、`.research/`、`research_state/`、`scratch/`（≈ 6.6 GB）；本文 S38 起的全部条目与表行。
- 恢复到 `2649ea0`：`model/model.py`、`semkine/{event_gnn,frontends,mesh_graph,fk_graph,routed_readout,train,eval_track}.py`、`tests/test_s37_mesh_graph.py`、
  `tools/{make_s36_row,make_s37_figure_simple,run_closed_loop_probe}.py`、四臂 prereg 与结构图。
- 保留：`AGENTS.md` 与 `tools/report_table.py`（统一结果表）；09-24 的 docs 收敛与旧 evsim / live_demo 代码的删除（不是实验）。

清除后 `pytest tests` 346 通过、2 跳过。当前臂与后续对照：S37 路由读出（`s37_routed`）。

## 2026-09-28：S37 路由读出重新分析（无训练）

同包迭代 + 原地不动 / 保持段首两条参照线（`docs/S37_ROUTED_READOUT_PREREG.md` §8）：闭环误差等于逐包信念的收敛点（global 13–17 mm，69 s 无漂移），
不是递推累积；local 上信念对手指基本无信息，闭环 23.33 / 29.49 对"保持段首"28.2。瓶颈在逐包绝对估计，不在递推。当前臂不变。

## 2026-09-28（下午）：S37 误差归因——最大单因是根旋转（无训练）

`docs/S37_ROUTED_READOUT_PREREG.md` §9，探针 `.experiments/s37_debug_20260928/`。闭环每步把根旋转换成 GT：global 15.8 → 7.9（低于基线 10.99），
local 26.4 → 17.6；换手指：13.4 / 17.1。根旋转在 GT 初始化后 250 ms 内就到 ~10°；TF 单步注入 6–8°（真实 1.6–3.0°），方向与所需几乎无关。
机制：根头是不读 prev 根旋转的单层线性层，prev 只经不看事件的 `prev_mlp` 进入（根斜率 −0.24…−0.64，往均值拉），两项各 15–25° 相互抵消；
大噪声课程与 51 维 MSE 的分配（根旋转 15.5%、平移 45%，相对 RA 低配 ~170×）是成因。local 另有一项手指误差（仅手指 16.8–17.2 mm，事件最稀的序列）。当前臂不变。

## 2026-09-28（晚）：根旋转新息的信息探针；用户方向 A / B 的判断（无训练）

`docs/S37_ROUTED_READOUT_PREREG.md` §9.1–9.2，`.experiments/s37_debug_20260928/probe_info.py`。`prev_mlp` 的根斜率与课程噪声下的 Wiener 收缩系数吻合（预测 −0.34 / −0.30 / −0.24，
global 实测 −0.30 / −0.24 / −0.25、−0.33 / −0.32 / −0.31）：可加可分的根只能取常数增益；K0 / K1 / K2 课程臂落在同一条权衡线上。
单包解码所需根旋转（zgz，°）：保持 9.21，S37 自身 7.93–8.21，z 8.02–8.04，z + r_prev 7.53–7.60（只加先验回拉），完美路由 = 保持（R² 0），
119 维 2D 剪影残差 7.08–7.69，× 3D 杠杆臂 6.96–7.09，+ oracle 事件深度 4.51–5.30。判断：A 作为"约束路由"无效（路由是根证据感知 prev 误差的唯一通道，
历史上 S38a 26.90、778 顶点记忆臂 78.78）；prev 3D 几何只值得作残差的雅可比（+0.3–0.5°）。B 只有每部位深度误差达到约 1 cm 才有用，这与估计旋转同难；
prev 给的深度是循环量。深度噪声扫描（`probe_dz_sweep.py`）：每部位深度误差 σ = 0 / 1 / 2 / 3 / 5 cm 时比无深度好 1.8–2.5 / 1.2–1.6 / 0.8–0.9 / 0.6 / 0.3–0.4°，
大 σ 端含未加噪的剪影覆盖通道，偏乐观。当前臂不变。

## 2026-09-29：S37 根新息第一步（冻结骨干、只训根）不过

`docs/S37_ROOT_INNOVATION_PREREG.md`。根加一条读"事件相对 prev 投影剪影的残差、prev 杠杆臂只乘残差"的新息头（1.4 K 参数），`prev_mlp` 根 6 行置零，
冻结事件图 / 手指头 / `prev_mlp`，从 S37 同种子选中点热启动训 1500 步。递推 RA 22.94（23.01 / 22.86）对 S37 20.74，global 18.74 对 15.82；
G1 TF 旋转 5.72 / 5.28°（门 ≤ 3），G2 纠正增益 0.26 / 0.36（门 ≥ 0.5），G3 过（新息置零 +86 mm），G4 不过；延迟 20.66 ms（SDF 未优化）。
新息承重但单包噪声大，去掉先验回拉后纠正不足，global 变差。不进入全量训练；当前臂仍为 S37 路由读出。

## 2026-09-29：证据窗口 50→300 ms 不改变 S37 根旋转（零训练）

`docs/S37_ROUTED_READOUT_PREREG.md` §9.3。窗口与步长解耦、输出仍在 50 ms 网格：两种子均值 RA 20.34–20.90（W = 50 为 20.74），根旋转 10.5–12.4° 不变，
global 手指随窗口变好 1–1.8 mm。根旋转是逐包绝对估计的上限，不是证据量、递推或增益结构单独能改的。磁盘上无绝对 CNN checkpoint，"CNN 根接入 S37"的零训练测试做不了。

## 2026-09-29（中午）：绝对 CNN 的根接入 S37（诊断 B，零训练融合）

`docs/S37_ROTW_CNNROOT_PREREG.md` §5B。S26 配方的逐帧绝对 CNN（51 维、域随机化，两种子重训）在同一 zgz 协议、同一批帧上 RA 13.31 / 13.80（均值 13.56，
global 9.89 / 10.45，local 17.25 / 17.66），比 S37 20.74 低 7.2 mm。把 S37 的全局旋转每步换成 CNN 的：16.93（−3.81，过预先定的 1.1 mm 规则）；
根整体换：16.79。剩下的差距在手指：仅手指 RA local CNN 10.9–11.6 对 S37 16.8–17.2。S37 的逐包估计在根和手指上都落后于逐帧 CNN。臂 A（旋转权重）训练中。

## 2026-09-29：旋转损失权重打平；逐帧绝对 CNN 的根接入 S37 有效，CNN 本身好 7 mm

`docs/S37_ROTW_CNNROOT_PREREG.md`。A：S37 只把 `LAMBDA_R` 60 → 600 从零重训，递推 RA 20.74（21.10 / 20.38）与 S37 打平，根旋转不降；×1800 单种子 19.10 打平
（global 旋转 11.3° → 7.6°，local 不动）。B：S26 配方的 51 维绝对 CNN 重训两种子（13.31 / 13.80，global 9.9–10.5、local 17.3–17.7），
把它的全局旋转接进 S37 闭环 20.74 → 16.93（两种子都改善），换整个根 16.79；CNN 自身 13.56，手指也全面好于 S37。当前臂不变；下一步待用户在
"轻量绝对根分支"与"把逐帧绝对估计做轻、递推只作平滑"之间选择。


## 2026-09-29：XYZ 支路信息 DEBUG（恒等式与反例，无训练/精度评估）

按用户要求对支路冗余作可执行检查，详 `docs/S37_XYZ_CANDIDATE_20260929.md` §3；脚本、JSON、固定输入、两个 S37 已选 checkpoint 的等价读出重写与 SHA256 保存在 `.experiments/xyz_branch_debug_20260929/`。CPU 执行，来源运行期间不变，未修改模型或训练。

区分了信息冗余与计算/归纳偏置：已知 K 与完整网格时，射线、深度假设和相对 XYZ 为确定性变换；同一批事件特征上的关节 mean/max/mass 加未匹配统计可重建全局统计。两个真实 checkpoint 的 root/手指/prev_mlp 可代数合并，但保留原输入且块稀疏，不能据此证明任意小头等价或删历史支路不影响精度。

反例显示未匹配信息、时空联合关系、最大值与关节槽位不能被简单均值替代；FK 局部状态自由度与轴角全局等价表示也必须区分。修订前稿“单小头/均值足够/历史角可直接删”的过强建议；XYZ 候选仍待实现训练，当前臂保持 `s37_routed`，不生成伪造候选主行。

## 2026-09-29：XYZ 候选算力投入判断（仅评估，未开训练）

用户询问刚 DEBUG 的网络是否值得训练。复核本次信息检查、09-24 历史 XYZ 文档、已训练 `s37_meshgraph` 和近期 S37 归因后，建议**暂不投入完整双种子训练，最多先做有预算上限的小规模机制验证**。详见 `docs/S37_XYZ_CANDIDATE_20260929.md` §4。

依据：信息通路成立不代表新网络已经可训练或能改善闭环；相近的旧网格臂未通过采纳门；`z_prior` 不增加独立深度观测；候选的逐事件关系编码、三维边和共同非线性读出仍有未验证的学习价值。建议先量真实完整包成本，再以候选及同构去显式三维通道的匹配训练作有限预算筛选，出现一致信号后另行登记完整比较。短训失败只能说明本轮不追加资源，不能直接判架构无效。本轮无模型改动、无 GPU 作业、无新精度结果；当前臂仍为 `s37_routed`。

## 2026-09-30：EdgeConv 6 层与根头逐关节融合均打平，不采纳

两臂双种子均完成 6000 步训练及 12 点 zgz 固定网格选点。09-30 上午在空闲 GPU 2 依次补齐主表，四次复测 RA 与选点记录的漂移均为 0.0000 mm（日志显示精度）。
结果与判读见 `docs/S37_EDGE6_ROOTFUSE_PREREG_20260929.md` §7–§8；主行产物为 `outputs/semkine/s37_{edge6,rootfuse}_main_row.json`。

edge6：递推 RA 21.38（22.61 / 20.16），相对 S37 +0.64 mm；rootfuse：20.29（20.52 / 20.06），相对 S37 −0.45 mm。
两臂均在预注册 ±1.1 mm 打平区间，均值未到 ≤19.64，且 3407 均退化、3408 均改善，故两种子不退化条件也不满足。
edge6 改善 local、退化 global；rootfuse 改善 global、退化 local；新增计算或参数尚无满足采纳门的收益。

训练 H6 的采样梯度判据两臂均通过；根旋转、oracle 根旋转及仅手指机制诊断本轮未补跑，不据主表作机制归因。
保持当前臂 `s37_routed`，本轮未改模型、未重训、未删除产物。

## 2026-09-30 补记：XYZ 三维关系候选已完成单种子短训，未过追加资源门

实际训练与评估于 09-29 晚完成：C1 / C0 均为 seed 3407、1500 步，固定 500 / 1000 / 1500 网格，选中 1000 / 500；尚无完整双种子训练。
主行与判据补入 `docs/S37_XYZ_SCREEN_PREREG_20260929.md` §5，并更正设计稿页首仍称“未训练”的旧状态；本次只复核既有产物与补记文档。

C1 是保留 778 顶点 XYZ、射线关联的事件深度先验与三维相对关系、固定网格边和共同读出的版本；C0 仅屏蔽学习模块的显式三维通道。
C1 递推 RA 21.0400，C0 22.1533，差 1.1133 mm，三点逐点比较均优于 C0；但选中点 TF 根旋转 6.8361° > 5.6095°，H4 整体未过。
对 S37 同种子、同预算网格最好点 19.5541，C1 落后 1.4860 mm，H5 判为同预算落后；证据置零劣化 26.2777 mm，H6 过。
依预注册规则，本轮证据不足以追加资源，保持 `s37_routed`；该结论不是完整训练采纳判定，不宣布架构无效。

## 2026-09-30：按用户新指令启动 XYZ C1 完整双种子训练

用户在查看短训结果后明确要求“现在做完整双种子训练”，覆盖此前默认不自动扩训练的安排。
09:20:49 启动 `s37_xyz_c1_full`：3407 使用 GPU 2,3，3408 使用 GPU 4,5，均从零训练 6000 步；除种子、步数上限与产物路径外，配置与原 C1 一致。
新配置与原配置的结构化比较通过；72 条训练序列/9 受试者、zgz 验证与种子均由训练元数据核对。
独立后台流程自动衔接 12 点选点、双种子主行复测和统一表，不覆盖短训的 C1/C0 产物。登记、状态及最终结果见 `docs/S37_XYZ_FULL_PREREG_20260930.md`。
本条是启动记录；当前臂保持 `s37_routed`，尚无本次完整训练结果。


## 2026-09-30：XYZ C1 完整双种子训练完成

未通过预注册精度采纳门；均值差 -0.7508 mm（打平）。详见 `docs/S37_XYZ_FULL_PREREG_20260930.md` §5；主行为 `outputs/semkine/s37_xyz_c1_full_main_row.json`。保留原短训产物，当前臂配置未自动改动。
