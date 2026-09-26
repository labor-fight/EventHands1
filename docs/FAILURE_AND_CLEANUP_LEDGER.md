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

早期“LNES 使运动方向在网络前完全消失”的判断已被后续 probe 与 LOSO 修正（`experiment_history.md` §10–11）。它不能作为新论文的当前结论。

> **协议声明（2026-09-05 起）。** 训练一律用默认划分 `splits_semkine.json`（9 受试者 72 条序列），
> 验证、选点、上报**统一只用留出受试者 zgz 的两条序列**（`zgz_global` + `zgz_local`；`val` = `val_core` = `test`）。
> 2026-08-22 → 09-05 期间的 5 受试者 / lr-lyq 选点线（S1–S46）的训练产物与日志已全部清除（见文末条目），
> 本文 §1–§2 中的具体数字都是在那条线上测的，**只作机制记录，不能与 zgz 协议下的数字直接比较**。
> 2026-09-06 起原 S37 及其后的全部实验（S37–S46）连同代码、配置、文档一起清除（文末条目），编号从 S37 重新开始：
> 2026-09-07 的 **S37 路由读出**（`S37_ROUTED_READOUT_PREREG.md`）在 zgz 上对 S36 −2.43 mm，已采纳，**当前臂与后续最优起点为 S37 路由读出**；
> 同日用户定义的 **S37 FK 图**（`S37_FKGRAPH_PREREG.md`）−1.07 mm 打平，成本 1/10，作对照与成本参照保留。
> 2026-09-18 用户手绘的 **S37 网格图**（`S37_MESHGRAPH_PREREG.md`，整张 FK mesh 作图 + LBS pooling）两种子分裂
> （20.48 / 29.30），不采纳；不稳定全在 root 旋转读出，手指读出稳定，保留作对照。
> 2026-09-20 的 **S38 三维网格图**两臂（`S38_MESH3D_PREREG.md`）：prev 几何进顶点特征 / 边（S38a）26.90 **不过**，闭环回灌（放大率 4.26）；
> 只给 root 部件杠杆臂（S38b）21.76 过对 fk_graph 的精度门、消除了网格图的种子分裂，但对 routed 仍 +1.02，TF 旋转未改善，**不换臂**。
> EventGNN 线上仍有效的文档：`EVENT_GNN_SURVEY_20260829.md`（文献与机制判定）与上述 prereg。

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
| **S38a 三维网格图**（`ENCODER: mesh_graph` + `MESH_GRAPH_EDGE_GEO` + `MESH_GRAPH_RIGID_NODE`，09-20） | **FAIL**。zgz 递推 RA 26.90（30.45 / 23.34）对 meshgraph 24.89、fk_graph 22.10；3407 整张网格 ≥ 41，选在 step 500；TF 7.07 全线最低、放大率 4.26 全线最高；单步 RA@26 mm 18.68（门 16.9）；观测置零 +65 / +18，abs 11 m。`docs/S38_MESH3D_PREREG.md` §8–§9 | prev 的 FK 几何以 `relu(W_h h_i + W_r r_i + b)`、`X_j − X_i` 进顶点特征与边——`W_r r` 项只依赖 prev、不管事件说什么都在，网络把它当先验：TF 里 prev = GT + 噪声，先验有信息；闭环里回灌。"只以相对量进入"不够：16 个部件相对根关节的相机系位置就是姿态，1-ring 边向量就是局部表面朝向 | **不采纳**。checkpoint 与 JSON 保留作负对照。教训写进契约：证据为零的节点对输出的贡献必须恰为零，不论几何；几何只允许乘证据（力矩 `r × δ`、骨局部系观测），不允许加性项 |
| **S38b 部件杠杆臂**（`ENCODER: mesh_graph` + `MESH_GRAPH_ROOT_LEVER: 64`，09-20） | **过对 fk_graph 的预注册精度门，不换臂**。RA 21.76（22.05 / 21.48）< 22.10 且逐种子不劣；对 routed +1.02；网格图的种子分裂消失，闭环旋转 p50 19.7° / 17.0°、网格 sd 5.2 / 4.7（meshgraph 7.4 / 8.9，门 ≤ 4 未过）；手指 19.6 ± 1.3 / 18.8 ± 1.9 全线最稳；abs 77（meshgraph 59）；网格中位 30–32（fk_graph 26.8）；TF 旋转 9.3° / 6.0° 未改善；观测置零 +109 / +28 | 杠杆臂只进 root、64 维瓶颈、`seen_j` 门住，回路没炸（放大率 2.47 / 3.13 与 meshgraph 同）；收益是 root 读出的一致性 / 正则，不是"旋转变得可读"——四个 S38 run 与 meshgraph 的 TF `val_rot_loss` 都平在 0.002–0.004、2k 后不降（H5）。瓶颈在训练信号 / 课程（0.3 rad 旋转噪声总与 50 mm 平移、大幅手指噪声同时出现），不在表示 | **不换臂，当前臂仍为 S37 路由读出**。保留作对照。下一个单变量按预注册 H5 应是课程，不是再改图 |

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

## 5. 文档地图（2026-09-24 收敛后）

活跃入口：

```text
docs/FAILURE_AND_CLEANUP_LEDGER.md          # 失败证据、协议、清理；本文
docs/S37_ROUTED_READOUT_PREREG.md           # 当前臂
docs/S38_GEOROOT_PREREG.md                  # 进行中的下一臂
```

仍有唯一数字、不得再拆的收据：

```text
docs/S37_FKGRAPH_PREREG.md / S37_MESHGRAPH_PREREG.md / S37_MESHQ_PREREG.md
docs/S38_MESH3D_PREREG.md                   # 09-20 退役 S38
docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md     # 778 顶点记忆臂（含初版+debug）
docs/S37_XYZ_EVENT_DEPTH_ANALYSIS_20260924.md
docs/GNN_ARMS_ARCHIVE_20260828.md           # 已删 KEG/CellGNN 臂的数字
docs/EVENT_GNN_SURVEY_20260829.md
docs/experiment_history.md                  # 旧协议 §1–14
docs/semkine/{EXPERIMENT_LOG,CLAIM_MATRIX,FAILURE_CASES,ARCHITECTURE_AUDIT}.md
docs/PLAN_SELECTION_VERDICT_20260825.md
docs/ASYNC_SPARSE_SOTA_MASTER_VERDICT_20260826.md
docs/POSITIONING_VS_E3DPSM.md
docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md
docs/debug_e55b_unroll_20260825.md
```

2026-09-24 已删（内容已进本文或记忆臂预注册，无独立数字）：`EVENT_KINEGRAPH_MASTER_PLAN.md`、`EVENT_KINEGRAPH_EXECUTION_LOG.md`、`EXPERIMENT_SYSTEMATIC_SUMMARY.md`、`debug_closed_loop_diagnosis.md`、`semkine/FINAL_REPORT.md`、以及事件引导网格拆开的三份（`EVENT_GUIDED_MESH_PREREG.md` / `_DEBUG_` / `_TRACKING_DESIGN_`）。

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
| S38a 三维网格图（网格图 + 1-ring 边带 3D 相对向量 + 带杠杆臂的刚体节点，09-20，`docs/S38_MESH3D_PREREG.md`） | 37.39（41.25 / 33.53） | 17.78 | 30.06 | 13.48 | 26.90（30.45 / 23.34） | 80 | 13.07 ms | 0.327 G | 0.47 M |
| S38b 部件杠杆臂（网格图 + root 读 `relu(W[e_j ; r_j])`，图不动，09-20，`docs/S38_MESH3D_PREREG.md`） | 30.15（29.39 / 30.90） | 14.48 | 23.96 | 11.01 | 21.76（22.05 / 21.48） | 77 | 12.83 ms | 0.315 G | 0.46 M |
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
- S38a 三维网格图（网格图 + 1-ring 边带 3D 相对向量 + 带杠杆臂的刚体节点）对 fk_graph **+4.80 mm，不过**（30.45 / 23.34；3407 整张网格 ≥ 41，
  选在几乎没训的 step 500）。TF 单步 7.07 全线最低而放大率 4.26 全线最高：prev 的几何以 `W_r r` 加性项进了顶点特征，被当先验用，闭环回灌
  （与"原 S37 FK 直连"撤回理由、KEG 回路增益 0.91 同一机制）。观测置零 +65 / +18，abs 飞到 11 m。
- S38b 部件杠杆臂（图不动，root 额外读 `relu(W[e_j ; r_j])`）对 fk_graph **−0.34 mm，过预注册精度门**（22.05 / 21.48，逐种子不劣），
  网格图的种子分裂消失，闭环旋转 p50 19.7° / 17.0°、网格 sd 5.2 / 4.7（meshgraph 7.4 / 8.9），手指 19.6 ± 1.3 / 18.8 ± 1.9 全线最稳；
  但对当前臂 routed **+1.02**，网格中位 30–32（fk_graph 26.8），abs 77（meshgraph 59），**TF 旋转不改善**（`val_rot_loss` 四个 run 都平在 0.002–0.004、
  2k 后不降——H5：瓶颈在课程不在表示）。观测置零 +109 / +28。**不换臂**。两臂 checkpoint 与 JSON 保留作对照。

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

## S38 三维网格图：prev 几何进证据通路的两种放法，一种回灌、一种只稠一致性；都不换臂（2026-09-20 17:45 → 21:06）

起点是用户的问题"778 个顶点是 3D 的，2D 的网络改成 3D 的可不可以"。对话里把"2D → 3D"拆成五个单变量，只做直接对应网格图失败点（root 旋转两种子分裂）的两个，
不做动态 kNN 建边。预注册、门槛、预测在训练前写定（`docs/S38_MESH3D_PREREG.md` §4–§5），两臂对 S37 网格图各只改一处（`tests/test_s38_mesh3d.py` 17 项契约锁定 diff），
全关 = S37 网格图逐位相同（S37 checkpoint `strict` 加载无缺失、无多余键）。训练在另一项目让出 GPU 4–7 后由 `tools/run_s38.sh` 自动起，`tools/finish_s38.sh` 训完自动跑主行、
旋转 / 手指分解、机制门（新工具 `tools/probe_s38_gate.py`）、闭环 + prev 噪声扫描，并对照门槛汇总（`logs/finish_s38_summary.log`）。复现漂移四个 run 都是 0.0000 mm。

- **S38a**（1-ring 边特征 = 3D 相对向量 + 带杠杆臂的刚体节点）：RA 26.90（30.45 / 23.34），**不过**。TF 单步 7.07 全线最低、闭环放大率 4.26 全线最高，
  3407 整张网格 ≥ 41 mm、选在 step 500；观测置零后 abs 飞到 11 m。几何以 `W_r r` 加性项进了顶点特征，被当先验用——TF 下有信息、闭环下回灌。
- **S38b**（图不动，root 额外读 `relu(W[e_j ; r_j])`，16 部件共享 W）：RA 21.76（22.05 / 21.48），**过对 fk_graph 的预注册精度门**（−0.34，逐种子不劣），
  网格图的种子分裂消失，闭环旋转 p50 19.7° / 17.0°、网格 sd 5.2 / 4.7，手指 19.6 ± 1.3 / 18.8 ± 1.9 全线最稳；但对当前臂 routed +1.02、网格中位 30–32、abs 77，
  **TF 旋转没有改善**（`val_rot_loss` 0.002–0.004，2k 后不降）。**不换臂。**

判读（详见 prereg §9）：预注册 H1 的前半——"旋转读不出是因为证据通路里没有相机系杠杆臂（双线性）"——**不成立**：杠杆臂进来了，TF 旋转不动。
成立的是 H5：四个 run 的 TF 旋转损失与 meshgraph 同一水平且随训练不降，表示层的两种改法都没碰到它，瓶颈在训练信号 / 课程。
放开"几何不进特征"这条法则的条件，"只以相对量进入"**不够**——相对量仍是状态；正确的条件是几何只能**乘**证据（证据为零则贡献恰为零）。
这条写进后续任何几何臂的契约。

下一个单变量按 H5：课程（把 root 旋转噪声与平移 / 手指噪声解耦，或降大噪声比例），其余不动。

**清理（09-21 16:10，用户指令："踩坑和实验内容留在 docs 里，其余的记录删除"）**：保留 `docs/S38_MESH3D_PREREG.md`（设计、门槛、结果、判读、产物登记含选中 checkpoint sha256）
与本文条目；代码回退到 S37 提交 `2649ea0`（`semkine/mesh_graph.py`、`semkine/fk_graph.py`、`model/model.py`、`tools/make_s36_row.py`），
`configs/semkine/s38_*`、`tests/test_s38_mesh3d.py`、`tools/{run_s38.sh,finish_s38.sh,probe_s38_gate.py}` 删除，`outputs/semkine/s38_*`（≈ 360 MB）与 `logs/*s38*` 删除。
**快照提交 `bf4f0ca`** 含删除前的全部代码、config、测试、工具与文档，任何一件 `git checkout bf4f0ca -- <path>` 可取回；checkpoint 按 prereg §6 重训可复现。
与 S38 无关而同期加入的 `tools/report_table.py` 与 `AGENTS.md`（统一结果表）保留。

## 2026-09-23：按用户澄清实现完整 778 顶点的事件引导图（待训练）

用户指出原手绘要求事件参与三维构图，进一步明确保留全部 778 个顶点，用上一状态之后到来的事件包驱动修改。
原 `s37_meshgraph` 是固定面片 1-ring 加节点观测，不能直接等同于这次明确的结构，也不能把旧主行当成本次实现的结果。

新增 `ENCODER: event_guided_mesh` 与独立配置 `configs/semkine/event_guided_mesh_s3407.yaml`：
每节点的三维近邻候选由事件观测距离重排，三层有证据支持的三维消息卷积，固定 LBS 均值汇到 16 个关节，再仅由这些特征预测 Δ。
全部节点保持原 MANO 编号；未观察节点可接收传播，几何在无任何证据时不能生成特征。
该分支去掉 prev MLP、头部 prev 角及背景直达 root；旧臂实现和已有权重接口保留。

实现、具体构图公式、时序和验证记录见 `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md` §1。新契约与旧臂回归通过，默认配置 CUDA bf16 前反向、优化一步和权重保存加载通过；
审查修复了 bf16 零更新舍入 prev 及几何预处理受 autocast 影响两处问题。尚未启动完整训练，没有新的精度/成本主表行或采纳结论。

## 2026-09-23：完整顶点事件图 debug——缺观测相似性阻隔传播，暂不作精度结论

按用户“先 debug 分析”检查了代码、实际训练包、MANO 模板边界反例及默认单卡批量；完整记录在
`docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md` §2，原始读数与复现脚本在 `.experiments/egm_debug_20260923/`。

- 事件确实改变邻居集合，但 `mean((o_i-o_j)^2)` 将无观测当真实零观测。空节点之间分数更低，可能排除有证据发送者；
  后续仅允许支持节点发消息，于是产生传播阻隔。合成单源与实际 MANO 单源都复现；真实训练包相对几何对照的支持范围也收缩。
  这是新公式的设计风险，不能解释成旧 S37 的已知失败原因。下一次最小修改优先处理缺观测比较，不同时换 LBS、头或课程。
- LBS 固定全顶点分母会减弱稀疏支持；极小正权重能打开个别完整关节头门。未支持“一个点打开全部 16 门”的过强猜想。
  三维近邻跨指与连续几何被 no_grad 截断均确认存在；准确率影响尚未评估。
- 修复了继承的 `assign_events_by_lut` 边界错误：越界/非有限原始坐标原来会 clamp 到手边缘，现在送背景，合法输入逐位不变。
  非有限背景行不会污染 EGM 丢弃背景后的前景或其他包。未改 jump-flood 的种子碰撞近似和非默认大 band 局限。
- 数据准备的姿态标签时刻与事件窗口右端、独立训练与后续递推条件之间有 1 ms 差异。沿用旧协议，纠正文档与图中的严格时间表述，
  用 k 表示递推步号；目标张量仍不参与构图/前向。
- 本次增加两项输入边界回归；EGM 与旧 mesh/FK/routed/S39 共 54 项通过。CUDA bf16 默认单卡批量的前向、损失、反向及一步 Adam 通过；
  Linear 使用 bf16，归约提升 fp32 符合实际 autocast 行为。未运行多卡完整训练，不报新主表行，不据此换臂。

## 2026-09-23：无观测保持的跨包顶点记忆底层（待明确 mesh 输出契约）

用户要求无观测顶点保持原状态并结合论文实现 tracking。核对 Mesh Graphormer/METRO 的空间重建、EvRGBHand 的历史特征融合和 E-3DPSM 的事件驱动状态后，
明确这些论文不提供“每个未观测 MANO 顶点绝对坐标冻结”的保证。局部潜状态保持与相机系坐标冻结对应不同输出路径，已询问用户区分；
具体文献、公式和接口见 `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md` §3。

完成公共模块 `semkine/mesh_memory.py`：caller-owned GRU 顶点记忆，只更新当前实际有观测的节点，未观测节点逐位保持历史，单独返回当前创新量；
不缓存训练批次，不把历史本身当新运动。`tests/test_mesh_memory.py` 的 7 项测试通过。此模块尚未接入模型、连续包训练或递推评估，
现有 encoder 的行为未变；不能将底层通过称为完整 tracking 已实现或已具备训练后性能。

## 2026-09-23：局部记忆 tracking 已接入并启动双种子训练

用户确认保留局部顶点记忆、允许随整手和关节运动，并要求开始训练。新增 `event_guided_mesh_memory` 独立配置，
保留全部 778 节点、图卷积、固定 LBS 到 16 关节与 51D Δ 主干；缺观测点的 GRU 持久状态逐位保持。
构图采用 4 个最近几何来源与 4 个当前观测引导来源，不足补几何。姿态和顶点记忆都在连续两包训练中显式传递，
main 不使用 GT prev；推理在每个 valid run 重置并在段内持续传递，不把状态偷偷缓存到随机训练 batch 之间。

启动前两项修订：

- 当前 Torch 2.1 的 CUDA fused GRUCell 不支持 bf16，真实双包 smoke 暴露该错误；仅循环单元关闭 autocast、保持 fp32，图卷积和头继续混合精度。
- 只池化 `H_next-H_prev` 在 GRU 的不同非零固定点处都会变成零，可能丢失持续运动信息。最终改为池化当前观测门控的 `H_next`，
  差值只作诊断；没有当前观测的历史不能直接产生 Δ。新增固定点测试锁定此语义，不将这项推理当作训练后精度改善。

最终 96 项契约与旧分支回归通过，真实双卡每卡 512 连续包对的两步训练通过；无观测 hold、空包恒等、两包梯度与完整 MANO 重建均覆盖。
2026-09-23 15:15:59 启动种子 3407（GPU0、1）与 3408（GPU2、3），各 6000 步，后续自动按固定 checkpoint 网格递推选点。
启动器 PID 1359394，日志和源码/配置快照登记于 `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md`。
训练刚启动，无新主表行，不报告性能门通过或采纳。

## 2026-09-23：778 顶点局部记忆臂训练完成，精度门失败

`event_guided_mesh_memory` 双种子均完成 6000 步；12 点固定网格递推选点于 19:12:07 全部结束，均选 step=500。
主表复评与选点数值完全复现，双种子递推 RA 均值 78.7832559328 mm，高于冻结 S37 routed 对照，且两种子均未通过各自门槛。
本次配置不采纳，保留 S37 routed；全部训练产物保留，没有追加训练。

末期 checkpoint 的递推误差比早期选中点更高，不能把较低训练/验证损失等同于跟踪改善。
当前结果不足以在事件构图、固定分母 LBS、局部记忆、两包训练与长序列状态分布之间归因，需独立控制变量诊断。
详细选点、权重哈希、成本口径和自动生成的正式表见 `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md`。

报告前修正 `tools/make_s36_row.py`：仅对 memory 臂用 `track_packet` 连续传递预测姿态和顶点记忆，
valid run / warmup / 计时 pass 各自重置；完整 MACs 路径计入 GRUCell，写明 latency_protocol。
旧臂保留原计时行为，故新旧耗时的状态条件不同，不能声称严格相同状态下的速度消融。

## 2026-09-24：讨论探针——"MANO 不经投影直接建图、给 2D 事件加深度维"能否提升 S37 的 global（无训练、无代码改动）

用户问：S37 路由读出若让 MANO 不经投影直接在 3D 建图、给 2D 事件加深度维度，能否不损失深度信息、提升 global。只做分析，不做设计。
探针 `.experiments/depth_lift_20260924/probe_depth_axes.py`，产物同目录 `s37_s340{7,8}.json`、`baseline.json`、`geom.json`。
zgz 协议，`default_rng(0)` 按序列顺序共用，与 `make_s36_row.py` 相同；无 oracle 的闭环逐位复现主行（3407：15.664 / 23.334；3408：15.972 / 29.491），
基线 `abs_pca12` step 11000 复现 10.986 / 30.003。

| 读数（zgz_global；S37 为闭环，3407 / 3408） | S37 routed | 逐帧基线 EventHands-PCA6（对完整 GT） |
|---|---|---|
| RA | 15.66 / 15.97 | 11.91 |
| 去掉相机 z 分量后的 RA（只剩像面内 x/y） | 11.59 / 12.35 | 7.08 |
| 每关节平均 abs(e_z)（深度分量） | 9.03 / 8.25 | 8.75 |
| 误差平方中 z 的占比 | 0.38 / 0.33 | 0.63 |
| 根旋转误差 p50：像面内（绕光轴）/ 像面外（改变深度的倾斜） | 5.3° / 8.1°；3.8° / 8.9° | 2.4° / 7.0° |
| 同上，TF 单步 | 1.8° / 5.4°；1.6° / 5.3° | — |

闭环 oracle（每步把该部分换成 GT；RA global / local，3407；3408）：无 15.66 / 23.33；15.97 / 29.49。根深度 z：15.52 / 26.89；18.21 / 29.77。
x/y 平移：14.10 / 23.42；14.70 / 28.35。整个根旋转：9.58 / 17.45；6.15 / 17.71。只去像面外误差：16.02 / 19.84；11.96 / 24.07。
只去像面内误差：12.84 / 22.56；12.05 / 26.56。根深度 oracle 下 abs 大降（3408 global 75.4 → 21.1，local 89.1 → 40.3）。

"给事件补深度"的实测：用 prev 前表面给每个事件补深度（即 S37 路由已选出的那个顶点的 z），对同一像素下 GT 前表面的深度误差 p50
global 27 / 76 mm、local 99 / 81 mm；逐步中位误差与 prev 根深度误差的相关 global 0.96 / 0.98、local 0.88 / 0.74。
图像一阶可观测性（GT 网格，扣掉其余五个根自由度后的独有 RMS 像素位移，zgz_global 0.56 m）：像面外旋转 0.10–0.12 px/°、像面内 0.36 px/°；
深度平移 0.042 px/mm、横向 0.12–0.14 px/mm。GT 每 50 ms 步运动 p50：像面外 1.1°、像面内 0.5°、横向 5.3 mm、深度 1.3 mm（每步深度信号约 0.06 px）。

判读：
- 事件是 (x, y, t, p)，没有可"保留"的深度；S37 投影的是 prev 网格，z 已用于前表面测试。
  补上的深度维只能来自 prev——实测它就是状态自身的深度误差（相关 0.74–0.98）——或来自单目学习。
- 完美绝对深度不改善 RA：根深度 oracle global −0.15 / +2.24、local +3.56 / +0.28 mm。深度是 abs 的主要来源，不是 RA 的。
- global 上 S37 比逐帧基线多出的误差全在像面内：深度分量与基线相同（8.3–9.0 对 8.75），像面内分量 12.0 对 7.1；像面外旋转与基线同量级（7–9°），
  多出的是像面内旋转（3.6–5.3° 对 2.4°）。TF → 闭环，像面内旋转涨 2–3 倍（1.7° → 3.6–5.3°），像面外涨 1.5 倍：闭环漂在 2D 看得见的方向上。
- 深度相关的像面外旋转在 zgz_local 更重（只去它 −3.5 / −5.4，只去像面内 −0.8 / −2.9）；在 global 上两种子不一致（+0.35 / −4.0）。
- 同族历史：S38a（3D 边 + 刚体节点）26.90 不过；778 顶点 3D KNN 记忆臂 78.78 不过；原 FK 直连（逐事件 prev 逆深度）撤回。几何以 prev 先验进入、闭环回灌。
- 口径附注：主表基线行 10.99 / 30 是对 PCA6 投影后的 GT 算的；对与 S37 相同的完整 GT 为 11.91 / 31.94（帧也不同：标注帧 + 100 ms 窗口，S37 为 50 ms 递推网格）。

结论：该方向对 global 没有运行时证据支持；不开臂、不改代码。这条线在 global 上的缺口是闭环中像面内旋转 / 平移的漂移，不是深度丢失。

## 2026-09-24：按用户指令删除 S39 实验记录，恢复 S37 起点

用户要求删除 S39 记录，最优仍从 S37 开始。删除覆盖图臂的预注册与结果文档、账本结果条目、专属训练产物（含 checkpoint）、主行、探针产物及日志；报告规则与示例同步恢复为 S37 路由读出（`s37_routed`）。当前臂及后续实验对照均为 S37 路由读出。此次仅清理记录与产物，覆盖图实现、配置及测试保留，不进行训练。

## 2026-09-24：完整 FK 三维点构图的 DEBUG 静态分析

按用户要求只作 DEBUG 分析，不启动正式训练。代码核对与数学边界写入 `docs/S37_ROUTED_READOUT_PREREG.md` §8：完整 XYZ 可保留并参与网格图计算；S37 routed 当前实际是事件图加 FK 路由，改用顶点图会改变节点载体。三维 kNN 邻接本身对刚体运动不变；应区分建边、坐标特征与二维事件关联。不投影顶点可以改用事件射线关联，但仍需相机几何与遮挡处理，不新增深度观测。本次没有新增性能结果、运行探针或修改模型实现，当前臂保持 S37 路由读出。

## 2026-09-24（晚）：完整 FK 三维点构图 / 事件深度通道的运行时 DEBUG（无训练，不改仓库代码）

用户以 S37 为起点再问两点：MANO FK 保留完整 3D 点、不投影、直接 xyz 建图；异步事件流保留深度通道引导 prev 的 3D 建图。上一条静态分析的 DEBUG 清单在两个种子的选中 checkpoint 上执行，
探针 `.experiments/xyz_graph_depth_20260924/probe_xyz_graph_depth.py`，产物同目录 `s37_s340{7,8}.json`、运行输出 `s37_s340{7,8}.log`；闭环逐位复现主行（15.664 / 23.334；15.972 / 29.491）。
逐条判定与数字见 `docs/S37_XYZ_EVENT_DEPTH_ANALYSIS_20260924.md` "运行时 DEBUG"，§8 附结论摘要。

- 投影没有丢深度：S37 在 32–46% 的路由事件上用 z 选前后层；不投影的点到射线关联在网格外事件上与 S37 一致 97–99%。偏差来自顶点散点加 3 px 前表面测试（local 上 26–30% 的网格上事件关节与精确射线交点不同），推理期换成精确射线关联后闭环 RA 变化在 ±0.34 mm 内、符号不一。
- 当前姿态 xyz-kNN 图：global 上 60% 就是网格边；zgz_local 最差 10% 的帧 29% 的边短路，闭环 prev 建的图与同刻 GT 图最差只重合 53–55%。应保留固定拓扑或规范空间邻接。
- 事件深度：按 3 px 容差只有 10–25% 的事件在有确定深度的表面内部；prev 给的深度即 prev 的深度误差（闭环 p50 25–100 mm，相关 0.75–0.98，斜率约 1），异步只占 1–1.5 mm；一个 50 ms 包内的逼近和视差信号只有 0.02–0.11 px。
  用 prev 的 z 改建事件图会动 20–31% 的手上边，而闭环下判跨层边的精确率约 0.5、召回约 0.4。
- 误差与手的距离：只有 3408 在 zgz_global 上正相关（ρ 0.44），3407 无，序列之间无关，判不确定。

结论：两个方向都不开臂，当前臂保持 S37 路由读出。文献核实（Pixel2Mesh、PyMAF、DynamicFusion、CMR、RootNet、I2L-MeshNet、BEVDepth、EMVS、Ev2Hands、EvHandPose）同文。

## 2026-09-24：docs 收敛

用户要求删除可删、合并重合。未动预注册收据或账本里的唯一数字。

删：`EVENT_KINEGRAPH_MASTER_PLAN.md`（08-23 合同，已被 S37 线取代）、`EVENT_KINEGRAPH_EXECUTION_LOG.md`（从本账本抄回）、`EXPERIMENT_SYSTEMATIC_SUMMARY.md`（08-25 总览，职责归本账本）、`debug_closed_loop_diagnosis.md`（核心结论已被 `experiment_history.md` §10 推翻）、`semkine/FINAL_REPORT.md`（`EXPERIMENT_LOG` + `CLAIM_MATRIX` 已覆盖）。

合：事件引导网格四份并入 `EVENT_GUIDED_MESH_MEMORY_PREREG.md`（初版公式、debug 读数、记忆臂契约、训练主行）。

资产：只留当前臂 / S36 / 网格图 / 记忆臂用到的图；未引用的 CellGNN、模板草稿、重复 PNG/SVG 删除。`.experiments/` 快照未动。

## 2026-09-25：原始因果事件目标的审计与最小CPU实现，未开正式训练

按用户新goal先审计、再研究决策、再有限Debug。完整续接入口：[STATE](research/s37_async_20260925/STATE.md)；协议/历史身份：[PROTOCOL_AUDIT](research/s37_async_20260925/PROTOCOL_AUDIT.md)；二十视角与最多三个候选：[DECISION](research/s37_async_20260925/DECISION.md)；实际命令、门和未过项：[DEBUG](research/s37_async_20260925/DEBUG.md)；预算/公平验收：[EXPERIMENT_PLAN](research/s37_async_20260925/EXPERIMENT_PLAN.md)。两份原文/官方代码文献矩阵与source ledger同目录，未证明新颖性或SOTA。

本次证实旧SAE fallback/均匀采样不具prefix稳定性；加入独立流式runner，复用S37学习算子与路由，显式无GT状态、FP32累加和完整MANO输出。旧源码/基线权重/结果未覆盖。独立审阅发现并修复输入buffer别名、时间静默取整、head变权未使旧状态失效；基础CPU回归61项、最新合同41项通过。64步CPU小样本拟合通过其预注册门，无新checkpoint，不能视为模型有效性实验。

CPU回放真实保留积压并明确不满足7ms；不外推为L20结论。原train全量负载检查与2048读出截断另留证据，不能用低负载fixture支持全包络。GPU数值/完整性能、长期漂移、恢复、严格协议精度及独立新数据尚有未过项；未启动GPU或正式训练，当前臂保持S37，不生成虚构C1主表行。既有georoot/xyz图/mesh记忆失败记录继续约束后续，不重启已失败路线。

所有一次性脚本、JSON和日志仅在`.research/s37_async_20260925/`；保留反例和修复前后证据，没有删除旧产物或恢复用户已删除文档。后续额度/数据答复到达后按STATE继续，不把阶段交付当目标达成。

后续自动续接补充：C0 scratch控制通过独立原forward_packet数值对照，公共无新事件路径不再计算未使用证据，避免人为夸大缓存效率；受影响合同与C0共51项CPU测试通过，首次Python3.9注解收集失败及修复日志保留。D008在1.024s输出范围内、两条各自持有历史的闭环中比较cached/full-prefix并通过，未使用GT历史或标签。详见同目录DEBUG的D007/D008。GPU和训练仍未启动，剩余科学/性能门与原目标不变。

### 2026-09-25 S37严格异步阶段GPU记录

- G001原CUDA Linear跨M舍入差在独立递推放大，pose原1e-5门失败；逐步证实token/邻居/mean相同而Linear不同。row-bmm替代同样失败，已排除、未接入。
- 新固定算序FP32核仅服务新C1 runner，先隔离数值/梯度/内存后集成，原容差和独立reference历史保留；限定范围复查通过。旧18个审计文件/资产/checkpoint哈希未变化。
- G002 C0/C1×2/4ms低负载完整回放全部失败；没有降低延迟口径、丢首输出、跳帧、缩节点或开训。G003 profile用于定位执行成本，后续只试有证据的最小调度机制。
- 所有GPU均经空闲检查，任务wall×卡数保守计入已确认Debug上限。scratch新文件为诊断/控制证据，不是正式训练的隐式依赖；没有新增候选权重、废弃训练分支或修改历史main row。源/命令/设备/耗时和所有失败保留于本轮research目录。

同轮G005/G006收尾：唯一固定解码调度候选保持已测bitwise语义，但完整服务仍FAIL；采纳其瓶颈诊断证据，暂不提升为正式默认、不训练。后续同配置计时/更多暖机/扩大训练不构成新证据，停止重复；事件更新执行器若重设计需新预注册及全部相关Debug。所有16个GPU job已退出，Debug仅用115.18286187993363 GPU秒；其余预算未挪用。无废弃训练权重可清理，scratch源码/失败/trace/预算凭据全部是必要复现证据，保留；没有删除用户文件。handoff_gpu_stage.json汇总本阶段源与终态（由实际文件生成）。

## 2026-09-25T16:08:25.753096+08:00 S37课程短程与方向诊断续接（不新增采用臂）

K1/K2种子3407固定六点筛查与选点完成，两臂均未过预登记双精度门和标准网格中位改善门；最佳点E2/E3继续其原队列，K0仍运行。`select_checkpoint.py`偶数网格记录的是上中位数，续接已用原grid重算标准median并保留旧JSON，最佳点身份未改，无需重跑。FK/mesh方向探针没有达到原方向改善门，仅否定该受测表示/训练组合，不外推全部几何/流机制。详细未舍入读数、注册时间与证据身份在 `S37_CURRICULUM_PREREG.md`。全部训练产物保留，当前采用臂仍S37，单种子诊断没有生成正式main row。

## 2026-09-25T17:02:46.910896+08:00 K0与R0固定判别完成，保留S37

K0原配置完整网格出现非有限闭环，best/terminal的tf0仍未优于保持，H3′未支持；原选择finite-only统计不代表完整稳定性。详见S37_CURRICULUM_PREREG.md。R0全部5臂×3种子固定拟合完成，Bres未过相对B0/Bctx门；有效支持的控制因没有正收益可消除而INCONCLUSIVE，不能据此否定所有方向几何。详见S37_READOUT_PREREG.md与启动前research_state/debug/R0_readout_probe_prereg.md。两者均不采纳，无新正式主行；不改标准、筛样本或无限续训。所有必要反例/日志/权重保留，无用户文件清理。

### 2026-09-25T17:50:09.228638+08:00 R1固定手指旁路移除：STOP，无清理/重跑

仅使用R0已冻结自身历史2048个fit帧。去prev_mlp手指分量相对原预测和等角度幅度对照均未通过收益门，global退化；实际FK位移幅度未配平，方向解释INCONCLUSIVE。原门要求fit不过不读dev，已遵守，未产生dev干预评分。独立回执/算术父审完成，CPU约2.787秒、GPU0秒。见 docs/S37_BYPASS_PREREG.md 与 research_state/audit/R1_terminal_audit_20260925.json。不重调旁路比例，不复活R0，不裁剪/改原S37权重。失败只限此checkpoint直接移除，不能证明一切历史先验必要。

## 2026-09-25T18:42:40.226037+08:00 M3定义合同完成，科学采用门未执行

M3仅固定模板/深度/匀速族的原始触阈等式与nuisance投影；一次CPU合同通过、低速空支持保留，详S37_EVENT_OBSERVATION_PREREG.md。未知phase/C的退化与错配后仍非零的局部量不能改判成真实纠偏。没有新主行、没有架构/训练通过；R0/R1 REJECT及C0r/C2r HOLD不变。原始事件/receipt/freezes保留，不清理用户文件或重跑参数网格。

## 2026-09-25T19:09:32.847738+08:00 D1输入终态与M4必要性裁决

D1一次COMPLETED/两PARITY_PASS，父核16文件、两序列在线字典计数及源stat均匹配；M4两报告41来源匹配。原生身份可恢复，缩放格合并memory并不证明精度原因。M4数值DEFERRED，继续oracle profile不闭合未知外观必要性；纯推导明确已知phase不等同已知reference。图50节点44关系；三候选/预算不变，无GPU或训练放行。下一真实代理只读审等价增量边界与真实几何竞争合同。

## 2026-09-25T19:26:15.221355+08:00 F1归档与下一真实几何合同

F1一次COMPLETED，原预设8包完整输出bitwise/调用数及48项记录通过，父核已保存数组；限CPU受控纯函数缓存，不代表GPU或完整7ms。G24原H无事件跨窗复用，RG1最终64包被动几何排序设计/反方归档；后者尚在实现，未评分。不重开R0/R1/georoot，不新加候选，不放行训练。

## 2026-09-25T19:51:25.753315+08:00 RG1固定几何排序终态

A→seal→B一次COMPLETED，64/64训练包；主项NOT_LOWER_BOTH_BASELINES。C_time两类均劣于原pred；时间控制有数值支持但无已证正收益可消除，空间控制全饱和退化为tie回pred。只否定固定被动scorer，不是新递推RA/新观测不可能证明；不修改模型、不重试、不放行训练。18定义检查及32项人工算术检查已完成；父核75冻结文件+64输入artifact及独立保存行算术通过。CPU外层5.557777795009315s、GPU0。详docs/S37_GEOMETRY_SCORE_PREREG.md、research_state/audit/RG1_terminal_parent_20260925.json、scratch/goal_20260925/rg1/terminal_science_review.md。

原代码/输入/回执/全部固定包保留；未清理用户文件。仅新scratch诊断与证据文档，无新增主臂、GPU训练或预算改变。

## 2026-09-25T20:18:34.349701+08:00 NF1静态审查与TEGBP补读终态

法向约束消去自由速度后可能保留条件姿态信息，不能以速度块或零一阶秩作普遍否定；物理材料速度和随姿态重关联的ray-hit导数须分开。当前future-event forecast不能独立认证物体法向，照明ramp有合法静止反例，故仅该认证变体DEFERRED；没有运行数值NF1。缺独立flow GT不禁止另行公平检验经验姿态效用，也不准许复活R0加头搜索。TEGBP补齐一个关键近邻方法族，已覆盖aperture不确定性与图传播；作者flow核心时间不能作为本项目完整时延。

详docs/S37_NORMAL_FLOW_PREREG.md及四份引用报告；父归档research_state/audit/NF1_TEGBP_parent_archive_20260925.json核验153条文件身份，不是153项算法测试。无新真实数组、FK、forward、拟合、合成fixture或GPU动作。C0r HOLD、固定C1r/R0 REJECT（广义前提HOLD）、C2r HOLD不变；三候选，没有唯一通过创新门的主方案。原短周期状态更新问题仍未答，未改原H策略；完整目标active。

## 2026-09-25T20:44:47.324645+08:00 F2逐步完整输出FK缓存

GPU精确递推与上下文门过，限定处理P50/P95双门未过，DO_NOT_ADOPT_PROTOTYPE。首次包保留，所有8包/4次重复均纳入，不缩分母或重试。保留scratch原型及证据，不接主模型。非完整7ms结果，非原评估器批解码加速；详S37_GPU_FK_REUSE_PREREG.md。


## 2026-09-25T21:08:58.749323+08:00 NF2实际支持门终态

NF2一次实际CPU八锚完整，固定共同支持不足，终态INCONCLUSIVE_SUPPORT_OR_PROFILE；B未启动/标签未读。四个无支持锚未算profile，不能写成数值失败；另四锚数值稳定/控制非恒等，但不能据无标签分数断言姿态效用。按原门停止，不降N门或改窗/方向/数据重试。35项人工定义检查与父45输入/22封印/8锚保存artifact核验完成，CPU外层8.565081570995972秒、GPU0。详docs/S37_NORMAL_FLOW_PROFILE_PREREG.md、audit/NF2_terminal_parent_20260925.json及独立terminal_review。S37与三候选裁决保持，唯一主方案/新训练/完整7ms仍未过。


## 2026-09-25T21:30:50.891733+08:00 SU1稀疏更新静态审查终态

三份真实独立审查及父训练合同完成，新增Zhang2016/Huang2010两基础原作、八原作明确复用；父核190条文件身份（不是算法测试）。P1 hard-count=0不等soft支持为零，head ablation后仍加G；旧失败不覆盖全部局部保持，但源码路径也未证明危害。通用投影/退化方向保持已有直接先例；限制速度或选择null-space代表先验，非自动新增测量。局部未知静态外观构造只在明示无回访等条件下成立，不是实际手部零信息证明。

当前不实现投影、不重扫rank、不复活P1/R1/R0/NF2固定形式；经验最小修复仍可凭具体诊断和公平train-only效用准入，不要求普遍可辨定理或flow/外观真值。下一先冻结既有R0 soft-support精确零的train-fit样本合同，再判是否值得分析其更新行为；尚未读新数组或运行。详docs/S37_SPARSE_UPDATE_PREREG.md、research_state/audit/SU1_parent_archive_20260925.json。0GPU/模型/FK/新标签/数值实验；原三候选与S37不变，完整目标active。短周期递推问题仍未答，未擅改策略。


## 2026-09-25T21:53:39.035253+08:00 SU2原更新必要性诊断终态

SU2一次固定train-fit缓存诊断完成，COMPLETED_WITH_LABELS / NO_TWO_ACTION_PARAMETER_HARM_SIGNAL。零soft支持时原更新确实存在，但两动作该子组的行加权参数风险均值均降低，仍有相反主体/行；空包0行仅空集通过，不称实证覆盖。没有保持干预、模型/FK或闭环新结果；不据此增加soft-zero门、挑有害行或扫低支持阈值。父核47输入/17封存/32输出数组/624汇总（非算法测试），独立终态审完成；CPU外层0.709282886935398秒，GPU0，PID/cleanup无残留。详docs/S37_ZERO_SOFT_SUPPORT_PREREG.md。

S37与三候选门不变，完整目标active。原短周期递推问题未答，未擅改协议。下一先核旧证据是否已经覆盖路由分支切换的具体状态扰动问题；无新的实验或训练准入。


## 2026-09-25T22:09:22.036903+08:00 DR1离散路由静态审查终态

DR1两份真实独立静态审查完成，父核51条来源记录（含复用，不是算法测试）。当前S37固定离散路由分支内e为常量，最终仍有base、finger own-angle与G的直接历史路径；no_grad不等前向独立，分支切换不等任务危害。旧root_inplane已经分离真实root头/历史旁路的有限响应，E3同seed的3/6尺度已经是同方向半幅对；P2/A2/R0/georoot原反证保留。缺少自身q0与显式分支标签的完整组合，不足以改变C0r决策，终态STOP_NO_DISTINCT_DECISION_CHANGING_DIAGNOSTIC。不运行DR1分支普查、幅度扫描或新头；无新数组/GT/权重/模型/FK/数值/GPU动作。详docs/S37_ROUTE_REGULARITY_PREREG.md与audit/DR1_parent_archive_20260925.json。

DR1已静态终态，没有数值作业待启动。下一科学动作须提出相对R0/R1/RG1/NF2/SU2/DR1不重复、能改变候选决策的允许输入经验假说，并先写两种相反结果各自改变什么决定；当前没有预注册的新实验。不因剩余预算或尚缺分支日志重跑旧探针，不要求普遍可辨定理/flow或外观真值作为所有经验方法的前提。


## 2026-09-25T22:38:40.854620+08:00 TM0/TM1终态归档

TM0两独立静态审与TM1一次固定CPU经验诊断完成。原S12/G4旧归档及EGM失败已核，TM1不同于其泛泛滤波/记忆提案；但实际终态NO_FIXED_HISTORY_READOUT_GAIN。A四组均15/15有支持，H local相对原预测有改善但相对当前C未达门，global明显退化；联合效用失败、近期性INCONCLUSIVE_NO_UTILITY。不调λ/lag/warmup/主体/动作规则重试，不把固定历史端点诊断作候选自身闭环或新鲜验证。父核62输入/34A+21B+3C封存与保存预测/评分，独立前检/终态审完成；CPU外层4.662058702902868秒、GPU0，PID/cleanup无残留。详docs/S37_TEMPORAL_HISTORY_PREREG.md。

S37与原三候选门不变；无通过创新门的新主方案或待启动训练。下一判别须有与已终止形式不同的具体信息来源或已定位实现缺陷，并预先明确能改变哪个决定；不能把剩余预算、未记日志或单动作局部收益当新实验依据。原短周期递推问题仍未答，未改50ms反馈。


## 2026-09-25T22:53:10.031330+08:00 AC0增强坐标与C0历史归档

AC0两份独立静态审查及父源码/旧JSON核查完成，终态STATIC_CLOSE_NO_DECISION_CHANGING_DIAGNOSTIC。事件/K/MANO增强传递及主投影代数一致；独立principal log可产生近2π坐标跳变，但实际训练暴露和危害未证，不为计数重开数值。恢复旧C0 SO3FK真实训练与C0c规范化反馈失败到当前摘要：不是未尝试路线。旧canonical_aa任意finger residual包角不保证完整旋转不变，但保存14记录finger超π比例全0，不能据此撤销旧实测。旧CNN也有raw prev MLP和加法，撤回完整CNN天然包角不变的过强解释，保留经验结果。父核58来源记录含复用及一个归档JSON成员；这些是文件身份，不是算法测试。无新真实数组/GT/模型/FK/数值/GPU；主模型、配置、数据、历史主行不改。详docs/S37_AUGMENTATION_COORDINATE_PREREG.md。

S37与C0r HOLD、固定C1r/R0 REJECT、C2r HOLD保持；没有新主方案或完整目标通过。原短周期更新问题仍未答，未改H反馈。


## 2026-09-26T01:51:50.431503+08:00 NG1终态
NG1已一次完整64包固定权重/自身历史实验。严格邻域遗漏存在，A/R0与A替换逐位通过、路由不变，但政策效用与全历史特异性均未过。STOP_FIXED_POLICY_NO_UTILITY；D支持不等创新。关闭此固定替换，不扫描/重训；S37与原三候选裁决保持。证据docs/S37_NEIGHBORHOOD_PREREG.md及research_state/audit/NG1_terminal_parent_20260926.json。


## 2026-09-26T02:23:31.952443+08:00 RC1完整恢复工程终态
RC1完整训练恢复工程完成，范围限定当前Lightning1.9.5、确定性(seed,index)数据、同执行合同及已完成optimizer step。新增semkine/checkpointing.py，接入semkine/train.py的--complete-checkpoints；新checkpoint自动识别完整恢复，旧checkpoint明确缺RNG/sampler。新模式保存每rank Python/NumPy/Torch CPU/CUDA RNG、原shuffle顺序与实际消费游标，保留Trainer模型/optimizer/scheduler/适用scaler；严格校验源码/配置/数据索引/加载设置。预取不推进保存游标，固定网格保存处理已提交loop状态。仅完整模式启用严格确定性算法，原默认路径保留。
CPU8项通过，合成2-rank FP32、FP16（含scaler）、BF16连续/恢复逐位通过。实际S37两卡BF16入口从step2恢复至4，两个rank后续输入tensor hash、随机状态、loss及最终完整模型/Adam/scheduler/step/epoch/采样状态全部逐位一致（entrypoint_v3/receipt.json）。checkpoint内验证batch完成数为0，未评测zgz、未形成科学训练或新main row。详docs/S37_RECOVERY_PREREG.md、audit/RC1_terminal_parent_20260926.json。
失败保留：CPU早期暴露未执行批/早停验证RNG/epoch边界问题，均修复且不降标准；FP16checker旧键错误只核保存artifact，不重跑；实际入口v1传卡参数错误无训练；v2两embed参数/Adam因非确定性数值不同而FAILED，v3严格确定性修复后按同一逐位门通过。所有job终态、cleanup完成，nvidia-smi无计算进程。RC1累计194.38110592600424 GPU秒；全任务Debug6688.5819763777545/12600，剩余5911.4180236222455。screen/train预算不变。


## 2026-09-26T02:40:25.291545+08:00 CI0/CI1终态
CI0独立数学/旧失败审与CI1一次固定权重CPU干预完成。H(e,q)−H(0,q)确实区别于R1仅减G，但实际完整16fit×128帧的中心化未过三项预注册效用门，终态STOP_FIXED_NULL_CENTERING。关闭该固定形式，不扫系数/参考点/支持子组、不进入dev或自身闭环。所有预测先封存后target解码，A/R旧评分及CPU head/MANO重构合同通过；327个自然零证据关节行全部保留，0个真实空包（不称实证空包覆盖）。数值与限制见docs/S37_CENTERED_UPDATE_PREREG.md；审计audit/CI1_terminal_parent_20260926.json。
CI0还推导了起点噪声、终点监督时的条件/无条件差包含事件解释的运动项；当前混合噪声、log-batch loss、未训练missing-condition及状态依赖路由不支持likelihood-score主张。相减本身已有Classifier-Free Guidance先例，不升级C2r或创新主方案。
单CPU计算进程4库线程，worker2.5882992499973625秒、launcher2.9623382809804752秒，退出0/cleanup无残留；GPU0秒，Debug累计仍6688.5819763777545/12600。没有改网络/配置、训练、重算encoder/路由、读取dev/zgz或新main row。S37与C0r HOLD、固定C1r/R0 REJECT、C2r HOLD保持；目标active且未达。


## 2026-09-26T03:01:38.033354+08:00 X1c Debug与固定新分支启动

旧X1 TIMEOUT/INCOMPLETE保持。新增明确非精确optimizer continuation工程，CPU15项及实际2-rank优化状态/新分支内续接逐位通过；strict旧gather v1首步OOM失败保留，compact索引+FP32累加等价修复后v2原2×512通过（CPU38项）。两个Debug均终态cleanup空，188.13862717803568GPU秒；没有改batch/损失/精度或科学门。现仅预注册500步新分支screen运行中，尚无精度结论。细节docs/S37_X1_CONTINUATION_PREREG.md。


## 2026-09-26T03:19:40.809579+08:00 X1c终态：STOP_FIXED_X1C_JOINT_GATES_FAILED

一次固定终点评价已完成，COMPLETED_DIAGNOSTIC_ONLY。原G-a/G-b各自global子门通过，但各自local子门未通过，联合ga/gb/strong门均false。不能写成完全无效或原始事件没有测量信息；也不能以global收益抵销local失败。保留S37，关闭此固定新分支及固定融合，不扫seed/alpha/节点数/训练长度/中间checkpoint，不自动进入动态增益训练。

内部未舍入值（mm，仅本文/原artifact；非两种子正式主行）：
- G-a absolute：global 12.609850883483887 / local 24.587575912475586；门≤13/≤17，联合失败。
- G-b alpha=.5：global 11.285038948059082 / local 21.79408073425293；门≤11.5/≤17.5，联合失败；strong<10.66/<15.1也失败。
- 两策略均global1386帧、local1204帧。初始化/输入/50ms/H评分与旧固定脚本保持。

父核：训练终点3000完整状态、每rank恰500新步/样本游标/Adam步数/原LR、源2500 SHA未变；评价输入hash、终点SHA与训练审计一致；可执行源码冻结一致，重算联合门与原artifact一致。旧评价只存汇总，不声称做了保存逐帧数组的独立再评分。正式主行未生成，不能把本单种子/复用zgz筛查当独立泛化或新SOTA。

原X1 TIMEOUT/INCOMPLETE保持；新X1c是明确非精确optimizer continuation（新runtime/sampler13407、数据3407），不是补回原3000轨迹。新分支内完整恢复及compact gather工程通过不构成科学创新。C0r HOLD、固定C1r/R0 REJECT、广义观测前提HOLD、C2r HOLD保持；用户三项硬目标和唯一创新主方案仍未达。表中历史Latency不是完整端到端证明。

资源终态：本轮Debug188.13862717803568 GPU秒、screen 1843.499812023947 GPU秒。全任务Debug 6876.72060355579/12600、screen 37046.12625307795/50400、train 0.0/201600，所有预留归零。本轮四作业均按实际失败/成功保留且cleanup完整，PID/runner均消失，nvidia-smi此刻无compute进程；GPU0未使用。

后续科学问题：在local稀疏手指事件下，允许历史状态提供静止结构，如何构造可验证的局部修正而不过度覆盖历史；本次绝对估计/固定整手融合不能完成此要求。若提出新机制，须首先说明与已终止X1c固定规则、R0/R1、SU2、CI1等的明确区别和新增允许信息，再在训练/开发材料上预注册判别；不能从当前zgz结果反调此分支或把常见自适应增益称新颖。此处是未解问题，不是已获准新训练。短周期更新澄清仍待答，原H未变。

证据：scratch/goal_20260926/x1c/{eval_v1.json,screen_audit.json,debug_v2/receipt.json}、research_state/audit/X1C_terminal_parent_20260926.json及本轮4个budget_run回执。


## 2026-09-26T03:32:59.886746+08:00 NC0历史覆盖复核（非新增数值实验）

标识：`NC0_STOP_CMN_MECHANISM_ALREADY_TESTED_NO_NEW_NECESSITY`。独立审查见 `research_state/debug/NC0_noise_geometry_admission_20260926.md`；父直接复读 `.experiments/s37_s38_fk_20260909/docs/debug_closed_loop_diagnosis.md` §2及现失败账本CMN行。旧CMN用SPA自身训练集闭环51D残差建库，按类别采样并径向拉伸匹配手工噪声幅度，双副本筛查local退化。因此“保持尺度、改真实误差方向/关节相关性”不是此前未测的机制；当前没有不同的必要性证据，不以中心化协方差、精确RMS或换到S37作为自动重开理由。

边界：这是旧SPA/CNN及历史划分的证据，原执行代码和权重已删除；本轮没有复现旧CMN，也不能证明其精确逐块RMS合同或将旧数字与当前zgz主行比较。PCA参与维数不是精确秩，原Gaussian mixed有full support，不能把“覆盖收窄”解释为严格零支持；旧“单轮DAgger注定失败”不作普遍定理。DART已有相应匹配噪声先例，匹配covariance本身不构成新创新。

仅恢复遗漏反证，关闭NC0提案；未读取新真实数组/标签/权重，未执行模型、数值探针或训练，GPU收费0。K0/K1/K2、S22等原终态保持，C0r HOLD、固定C1r/R0 REJECT（广义观测前提HOLD）、C2r HOLD保持，S37仍当前臂。下一动作仍须先有独立定位且区别于已失败形式的具体缺陷；不能用误差谱重算来代替有效判别。


## 2026-09-26T03:43:48.733835+08:00 LC0旋转复合提案静态终态

裁决LC0_STOP_NO_INDEPENDENT_NECESSITY；独立审查research_state/debug/LC0_rotation_update_admission_20260926.md。C0端点SO3FK损失、C0c反馈主值化不是复合增量训练，但“不等同”不证明新训练必要。S37输出被训练为端点坐标差，没有声明为角速度或群切空间量，因此prev+delta并不是把物理角速度积错。正确把同一端点转换为相对旋转再复合只保持原旋转；直接把旧delta当左/右群增量改变了语义，不能据效果归因到积分修复。旧S3/S12已有Lie retraction实施，EventHPE已有直接复合先例。

局部MANO必须先加hands_mean，左右扰动要区分父/自身坐标；旧SE(3)root retraction还会改变平移/pivot合同，不能当纯旋转消融。当前无独立故障定位，关闭提案，不开旧头compose、曲率计数或重训练；不宣称所有复合更新无用。0GPU、无新真实数组或数值。EP0空状态精度修复是另一个已定位的实现合同，不由LC0准入，不冒充旋转模型创新。


## 2026-09-26T03:49:20.935041+08:00 EP0终态：有限状态空包精度合同已修复

核心最终6行post-mask修复，未增加参数/配置。CPU新精度矩阵20通过、10项CPU不支持的FP16显式skip；既有S37/S2回归8通过。GPU1 L20/Torch2.1/CUDA11.8严格deterministic的FP32/BF16/FP16共30项通过，含混合空包、有限状态身份梯度、空行模型梯度0/None、非空旧算术及梯度逐值相等、5次空包反馈signed-zero位模式。v1 fixture失败保留，只有期望梯度赋值换为等价outer product，v2不降门。

实际旧S37 checkpoint严格加载、既有8包train缓存的FP32输出与修改前代码及旧封存tensor均逐位一致，参数733830/键90保持。真实X1c complete checkpoint仅改变model源码SHA时按原合同拒绝exact resume。没有重训、选点、GT评分或精度新主行；验证范围不是完整AMP训练轨迹相等或所有架构/CUDA负载覆盖。

边界：AMP时整个输出dtype可提升为FP32，非空值保持旧低精度算术，不自动提高非空精度；含空行训练loss及log-loss缩放可能改变。旧FP32普通状态回归保持，signed-zero复制是有意加强。旧complete checkpoint精确恢复须使用原源快照；不改旧权重contract。增加的最终where没有重新测主表Latency或完整event→mesh，不能把历史行冒称改后测量。

执行产物：scratch/goal_20260926/ep0/{cpu_tests.xml,cuda_tests.xml,cuda_receipt.json,real_regression.json,resume_guard.json,core_delta.patch,cuda_v1/}；作业ep0_cuda_precision_20260926_v1/v2分别FAILED/COMPLETED，均cleanup空、leader/runner已退出。本轮实际Debug收费14.499832835048437 GPU秒；全局预算{"debug": {"charged": 6891.220436390839, "reserved": 0.0}, "screen": {"charged": 37046.12625307795, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}。GPU0未使用。

裁决：保留此正确性修复；S37仍当前臂，C0r科学HOLD、固定C1r/R0 REJECT、C2r HOLD保持，三项硬目标与唯一创新主方案未达。LC0旋转复合提案另已静态关闭，不由本工程修复获得训练准入。

## 2026-09-26T04:17:55.360113+08:00 SC0真实配对短训终态
固定L/F×3407/3408各500步已完成；原S37冻结，原H开发前缀自身闭环。F对A改善但未胜L预注册门：STOP_FIXED_ADAPTER_UTILITY_FAILED。共同适配收益是有限工程现象，不是远程必要性、独立泛化或新方法采纳证据。不自动解冻/延训/采纳L。详docs/S37_SPATIAL_CONTEXT_TERMINAL_PREREG.md、audit/SC0_terminal_parent_20260926.json。SC0总201.59951122815255GPU秒，全部cleanup完成；主行和默认split/core未改。

## 2026-09-26T04:38:39.994739+08:00 CC0真实嵌套校准对照终态
两bias-only128参数fit各500步完成；A/L/C各自从合法起点递推256，固定后128评分。U(C)与U(L)均过，但C对L的global均值/一个seed容差不通过：CONSTANT_UTILITY_WITHOUT_MATCHING_L。保留局部与全局取舍，不把C local与L global拼接、不放宽门、不说bias无效或邻居机制必要。A/L首128字节回归、配对index/order、旧tensor冻结/完整保存和0 validation实证通过；4job cleanup完。CC0总134.67438118404243GPU秒，无core/旧L/F修改或新mainrow。详docs/S37_CONSTANT_CONTEXT_TERMINAL_PREREG.md、audit/CC0_terminal_parent_20260926.json。C0r有有限工程效用证据仍HOLD，科学/完整验收另列，无自动续训准入。


2026-09-26T04:44:53.060337+08:00 CC0终态独立复核已关闭：见research_state/debug/CC0_terminal_review_20260926.md。认可固定CONSTANT_UTILITY_WITHOUT_MATCHING_L；保留两种有限效用，不采纳C/L、不改变原容差或旧F关闭决定。四任务均清理完成，无待恢复任务；本次归档未启动数值作业。


## 2026-09-26T05:02:48.649754+08:00 CC1同函数执行与成本终态
EXACT_EXECUTION_OBSERVED_COST_REDUCTION；CPU30/CUDA30、两seed完整ownH字节回归通过。只交付opt-in同函数执行模块；固定pooled P50/P95门过但P99/max存在退化、完整7ms失败。无新训练/精度或C/L选择。详docs/S37_CONSTANT_EXECUTION_TERMINAL_PREREG.md与audit/CC1_terminal_parent_20260926.json。唯一job已COMPLETED/0并清理，实际144.20186586596537GPU秒；主方案与S37当前臂不变，目标未达。


## 2026-09-26T05:14:08.685784+08:00 AS0串行状态发布依赖终态
REJECT_DIRECT_PREVIEW_ON_CC1_SERIAL_RETURN；单CPU1.8340502460487187秒，GPU0。既有CC1时序下直接挂preview即使零处理也有>7ms依赖反例；只对原同步完整输出返回接口，不把full-ready冒充pose-ready、不否定所有异步调度。保存工件分析非新latency测量。详docs/S37_ASYNC_OUTPUT_CONTRACT_TERMINAL_PREREG.md、audit/AS0_terminal_parent_20260926.json。S37与科学候选保持，完整目标未达。


## 2026-09-26T05:34:31.788687+08:00 PR0 return-only交接形式关闭
实际插桩原C调用的PRE下界已在真实下一窗事件上给出>7ms反例；固定256帧old-history回归通过。仅交接位置前移不足于该保存调度，需返回前计算变化；未测GPU-ready、不否定其他调度、不重测救场或自动准入preview。详docs/S37_POSE_HANDLE_READINESS_TERMINAL_PREREG.md。S37/三候选保持，完整目标未达。


## 2026-09-26T06:00:10.605442+08:00 CD0同函数捕获固定负载终态
EXACT_CAPTURE_COST_REDUCTION_ON_FIXED_WORKLOAD；CPU8/8、14GPU合同记录、5合成包、1024原H字节回归及512成本帧通过，父已复算NPZ。两seed pooled四分位门过，3408 local max仍退化且完整7ms失败。仅保留opt-in原函数捕获，无训练/精度/模型采纳。唯一job 62.146028652903624GPU秒且清理完。详docs/S37_CAPTURED_DECODE_TERMINAL_PREREG.md与audit/CD0_terminal_parent_20260926.json；科学候选及S37保持，目标未达。


### 2026-09-26T06:02:35.205819+08:00 CD0独立终态复核与证据图关闭
父已读取debug/CD0_terminal_review_20260926.md，最终SHA d9b0823383ddb1427d240ed4b7408ba426b0dc9112c74cad99aa1270a7a1b8e2已绑定audit。认可限定pooled四统计量成本门，保留3408 local尾部及两local首请求退化、完整7ms失败。父已单CPU复算保存NPZ；复核是静态/文本审查，非独立数值重算。图新增L122–L126/N95–N96（96节点94关系），C0r HOLD/novelty0及其他候选保持。无运行/待恢复任务，无新训练或精度，完整目标active未达。前版存research_state/snapshots/CD0_review_close_20260926T060235+0800。


## 2026-09-26T06:31:22.393706+08:00 HE0固定历史事件特征可读性终态
唯一CPU诊断完成：NO_FIXED_PAST_EVIDENCE_READOUT_GAIN；15joint四组控制支持有效，但H相对C的local及相对O的global未过联合门，控制解释INCONCLUSIVE_NO_UTILITY。人工13项、58冻结身份、A/B/C封印与父保存NPZ/字节/评分/门复算完成。只淘汰固定被动读出，不证明历史普遍无用或新的闭环结果。GPU0费、worker/launcher已退出且cleanup空；S37/三候选保持。详docs/S37_HISTORY_EVIDENCE_TERMINAL_PREREG.md、audit/HE0_terminal_parent_20260926.json；独立终态交叉核验待关闭。完整目标active未达。


### 2026-09-26T06:34:25.633632+08:00 HE0独立终态复核与证据图关闭
父已读取debug/HE0_terminal_review_20260926.md，最终SHA 4456162dcf96438a0723c51438ea0bc87dbf7dae7c5309536f1eea4bf5599366绑定audit。认可固定线性历史证据的联合效用FAIL与pairing INCONCLUSIVE_NO_UTILITY；保留复用开发数据、固定反馈及非普遍历史无信息的范围。父已复算保存数组；独立审查为文本/源码/标量，不冒称独立数值复跑。补充覆盖核查2304真实帧当前空包0，空包证据仅来自人工合同；180 warmup行与自然历史零支持存在。图新增L127–L132/N97–N99（99节点97关系），候选门/分数不改，无运行或待恢复作业。S37仍当前臂，完整目标active未达。前版存research_state/snapshots/HE0_review_close_20260926T063425+0800。


## 2026-09-26T09:56:52.432342+08:00 PA0有界语义审查与人工终态
五项定义检查完成、14身份核验、worker退出；两份独立报告和父fallback已归档。新增真实来源是固定HASTE源码的reference/availability区别；新增代数界限是配对错置可只加所有候选共同分数。无真实数据/模型/GT/GPU/效用或时延。详docs/S37_PAST_ANCHOR_TERMINAL_PREREG.md；图102节点100关系，S37/三候选保持。下一为PA1具体合同与接口正确性，未准入真实运行。上一turn分类progress，完整目标active未达。


## 2026-09-26T10:21:04.553666+08:00 PA1固定事件轨迹诊断终态

唯一run pa1_past_anchor_track_v1 COMPLETED/0，INCONCLUSIVE_FIXED_TRACK_SUPPORT。local固定源命中1，0/16支持，P恒等；global命中14，16/16支持，但联合门未过，B不存在/target未解压。不能把global支持作效用或local支持不足作普遍不可观测。禁止改种子/窗口/锚时刻/候选/门救场。

前端10项+集成6项人工检查最终通过，初始P测试期望错误与空流ID修复均保留。49冻结身份、A0九文件/A1五文件封印、32保存候选/残差/选择/门父核完成，独立前检与终态文字复核已关闭。CPUworker4.173991453950293秒/launcher4.613799230894074秒，GPU0费，PID3066845退出、cleanup无残留。详docs/S37_PAST_ANCHOR_TRACK_TERMINAL_PREREG.md、audit/PA1_terminal_parent_20260926.json。

图新增L139–L145/N103–N104（104节点102关系），不改三个候选门/分数；S37仍当前臂、原H/短反馈待答项保持，完整目标active未达，无可恢复数值任务。下一实质方案须区别于已关闭形式并先冻结可改变的实现决定，现无新训练准入。


## 2026-09-26T10:37:01.530900+08:00 MP0原S37非空AMP状态累加终态

新增MODEL.STATE_ACCUM_FP32默认false、限定routed event_gnn+delta、无参数/键变化。CPU38pass/18不支持skip、CUDA56全过；既存8包CPU默认/开启/历史输出逐字节相同。两seed固定8训练样本相对完整FP32的全部参数梯度L2都降低，AMP_GRADIENT_FIDELITY_IMPROVED_ON_FIXED_BATCH；3408 log10损失差和3407最大51D输出差反而增大，不宣称所有数值/精度改善。真实optimizer0，无新自身闭环/开发评分/训练。

唯一job mp0_state_accumulation_20260926_v1 COMPLETED/0、10.243856586981565GPU秒、GPU1 L20，worker/runner均退出且cleanup空；父28身份及保存张量/范数/尾部/测试回执核验完成。详docs/S37_STATE_ACCUMULATION_TERMINAL_PREREG.md与audit/MP0_terminal_parent_20260926.json；独立终态文本复核待最后绑定。

保留opt-in实现，S37/三候选采用门保持。下一MP1为只改这一算术的同预算两seed配对BF16短训与原H各自递推效用，须先固定训练/恢复/评估合同，不拿保真门冒充精度采用。新core使旧complete fingerprint拒绝恢复，旧源快照保留；不重启旧任务。预算{"debug": {"charged": 7143.870747750741, "reserved": 0.0}, "screen": {"charged": 37363.602386470026, "reserved": 0.0}, "train": {"charged": 0.0, "reserved": 0.0}}，无预留/运行，GPU0禁用，完整目标active未达。

## MP0 independent terminal closure 2026-09-26T10:41:49.727646+08:00
research_state/debug/MP0_TERMINAL_REVIEW_20260926.md SHA256 e364b8b53f0e409723d0150de09ad56599e2867717d49076a67b3cb61fda5962 bound in parent audit, with reviewed pre-binding audit snapshotted. Text/XML/receipt consistency only, no independent numerical rerun. Graph106 nodes/103 relations; C0r/O10 remains HOLD, no accuracy adoption. MP1 paired short-training contract remains next distinct decision.

## MP1 Debug terminal 2026-09-26T10:55:31.502301+08:00
Both-arm fixed BF16 fitting and complete recovery gates passed. Saved checkpoint parent audit closed. Details docs/S37_STATE_ACCUMULATION_TRAIN_DEBUG_PREREG.md. Four fixed paired short screens admitted, no utility or accuracy adoption. No repeated MP0/MP1 Debug.

## MP1 fixed paired training terminal 2026-09-26T11:11:17.445953+08:00
MP0 independent terminal review is closed. MP1 strict Debug and four matched BF16500-step new trajectories complete; full ylf own-H joint utility FAIL. Fixed500 all-input matching and six-model full denominators verified. No hyperparameter/seed/endpoint rescue; retain S37 and default-off numerical option, candidate gates unchanged. Details docs/S37_STATE_ACCUMULATION_TRAIN_TERMINAL_PREREG.md; parent audit research_state/audit/MP1_terminal_parent_20260926.json; independent terminal review awaiting final binding. Total MP1 GPU742.674416705966 seconds. Goal active/unachieved. Next admission is bounded relative-increment equivalence definition, no numerical/training admission.

### MP1 independent terminal and graph closure 2026-09-26T11:13:18.162205+08:00
research_state/debug/MP1_TERMINAL_REVIEW_20260926.md SHA256 b6c7322bdb5630e8184850e0d6ce98b192e936a468f7197b871e3b5217644e59 bound in audit; reviewed prior audit snapshotted. No remaining review discrepancy. Graph108 nodes105 edges, C0r/C1r/C2r gates/scores unchanged. Fixed MP1 utility failure only, no universal precision impossibility or partial-metric adoption; no running/resumable task, whole goal active/unachieved.


## RI0 terminal 2026-09-26T11:35:37.661130+08:00
唯一六项有限定义检查和父保存有理记录审计已通过，三个静态审查实际完成。关闭仅相对重参数化的信息增益论证，保留条件可辨识及分布先验；无新估计器/候选/训练。详docs/S37_RELATIVE_INCREMENT_TERMINAL_PREREG.md、research_state/audit/RI0_terminal_parent_20260926.json。无数值作业待恢复，S37/原H保持，完整目标active未达，本轮progress。


### RI0 evidence graph closure 2026-09-26T11:39:11.080282+08:00
三份独立静态审查、一次CPU定义执行及父保存有理记录审计闭合。图111节点108关系，候选门/分数保持；RI0不准入估计器/训练。快照research_state/snapshots/RI0_closed_20260926T113911。核心/14冻结身份不变，无RUNNING或待恢复任务，GPU费用0。归档辅助误写核心路径已纠正，保留ABORTED记录，不是实验失败或重跑。完整目标active未达，本轮progress；当前无新的已准入实验。
