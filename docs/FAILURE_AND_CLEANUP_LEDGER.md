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
| E5.5b 恒定 UNROLL_P=0.5(无退火) | FAIL,TF RA 42.5 mm / 递推 144.5 mm(step 3500 探针) | 从 step 0 就有一半样本以近随机的自预测做条件,网络最优解是无视 prev,坍缩回绝对回归(历史绝对回归正是 ~40 mm 档) | 产物留在 `outputs/semkine/s22_keg_halo_unroll_s34xx` 作负对照;修复=UNROLL_RAMP [500,2000] 线性退火(scheduled sampling 原文的 curriculum,Bengio et al. NeurIPS'15),重启臂名 `s22r_keg_halo_unroll` |

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
