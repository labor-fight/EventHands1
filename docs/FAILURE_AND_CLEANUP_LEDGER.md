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

`docs/debug_closed_loop_diagnosis.md` 中“LNES 使运动方向在网络前完全消失”的核心判断已被后续 probe 与 LOSO 修正。它只能作为历史排查记录，不能作为新论文的当前结论。

> **协议声明（2026-09-05 起）。** 训练一律用默认划分 `splits_semkine.json`（9 受试者 72 条序列），
> 验证、选点、上报**统一只用留出受试者 zgz 的两条序列**（`zgz_global` + `zgz_local`；`val` = `val_core` = `test`）。
> 2026-08-22 → 09-05 期间的 5 受试者 / lr-lyq 选点线（S1–S46）的训练产物与日志已全部清除（见文末条目），
> 本文 §1–§2 中的具体数字都是在那条线上测的，**只作机制记录，不能与 zgz 协议下的数字直接比较**。
> 2026-09-06 起原 S37 及其后的全部实验（S37–S46）连同代码、配置、文档一起清除（文末条目），编号从 S37 重新开始：
> 2026-09-07 的 **S37 路由读出**（`S37_ROUTED_READOUT_PREREG.md`）在 zgz 上对 S36 −2.43 mm，已采纳，**当前臂 S37**；
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

## 5. 文档收敛方案

活跃 `docs/` 最终只保留三份入口：

```text
docs/EVENT_KINEGRAPH_MASTER_PLAN.md      # 当前方法、公式、代码与 Gate
docs/FAILURE_AND_CLEANUP_LEDGER.md       # 修正后的失败证据与清理策略
docs/REPRODUCIBILITY.md                  # 环境、manifest、命令、最终结果
```

现有大文档先不要立刻物理删除。完成本机引用审计后：

1. 把仍有唯一 receipt 的内容压缩到 `REPRODUCIBILITY.md`；
2. 对已经被本文完整覆盖、且无脚本引用的历史报告执行 Git 删除；
3. Git 历史本身保留原文，不需要在活跃工作树继续维护互相矛盾的多份 Markdown；
4. 资产图片只保留最终主表、关键 failure plot 和方法图。

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
