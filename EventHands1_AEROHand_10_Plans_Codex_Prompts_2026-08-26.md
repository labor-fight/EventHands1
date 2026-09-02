# EventHands1：从 LNES Tracking 到真异步稀疏手部 Mesh 重建

## AERO-Hand 主方案、十条可证伪研究路线与完整 Codex 实施提示词

**仓库：** `labor-fight/EventHands1`  
**审计分支：** `research/event-kinegraph-20260823`  
**审计日期：** 2026-08-26  
**目标任务：** 单目事件相机、连续递归 tracking、51D MANO 状态、严格因果、无中途 GT 重置的手部 mesh 重建  
**首选论文定位：** event-native continuous hand mesh tracking，而不是“把某个检测网络改成手部回归器”

---

## 0. 先说明结论边界

1. 本文中的“100 位专家 battle”是由 100 个**角色化审稿/研究视角**构成的系统性红队推演，不是 100 位真实外部专家投票。它的作用是暴露方案的失效条件，而不是制造虚假的共识。
2. 本会话可以读取私有 GitHub 仓库和已进入文件库的历史设计/日志；但无法直接挂载你本机的：
   - `/data1/lyq/code/skill/creative-thinking-for-research`
   - `/data1/lyq/code/skill/research-opportunity-graph-skill`
   - 仅存在于本机而未提交、未上传的日志
   因此，本文没有假装读取这两个本地 skill。本文实际采用了它们通常对应的三类研究原则：**先发散再收敛、机会图中的机制迁移、claim—evidence—failure 三联审计**。
3. 没有任何方案能够在训练前诚实保证“一定 SOTA”或“一定中 CVPR/NeurIPS”。本文给出的是：
   - 当前代码和日志证据下，成功概率最高的主方案；
   - 每条路线的硬门槛、止损点和反证实验；
   - 只有跨协议、跨种子、公开数据集结果成立后，才允许写 SOTA claim。
4. 仓库里不同阶段使用过不同协议。`19.25760436702419 mm`、S20/S21/S22 的 `20–31 mm` 不能直接混为一张表。任何实验必须同时记录 checkpoint、split、window/query policy、初始化规则和 evaluator commit。

---

# 第一部分：代码与日志真正说明了什么

## 1.1 当前最重要的五个事实

### 事实 A：LNES 的确有表示瓶颈，但当前主要精度瓶颈不是“缺一个更强 backbone”

LNES 会把原始事件压成固定二维表面，丢失：

- 微秒级事件顺序；
- 同一像素多次触发；
- 任意查询时刻；
- 稀疏计算和事件到达即更新的能力。

但是当前实验又显示，密集 LNES 在严格递归 tracking 上仍明显优于 KEG。也就是说：**先把 LNES 换成稀疏图，不会自动提升精度。**

### 事实 B：KEG 的单步观测已经接近 LNES，闭环才是主要失败源

现有日志的量级结论是：

- 单步 teacher-forced：KEG 约 `11.1–11.6 mm`，LNES 约 `10.95 mm`；
- 严格递归：KEG 约 `29–32 mm`，LNES 约 `20–23 mm`；
- KEG 的误差增益斜率约 `0.29–0.33`，LNES 约 `0.15`；
- halo routing 能改善约 `2.6–4.8 mm`，但无法消除递归差距；
- KSSF 被静默后 KEG 会严重崩溃，说明 KSSF 不是无用模块；
- S23 residual unroll 已有代码/配置，但在审计时还没有形成训练后证据。

所以最合理的诊断不是“KEG 表达不够”，而是：

> KEG 的观测本身过度依赖上一时刻姿态，导致上一帧误差既进入状态，又进入事件—手部关联，形成双重回声。

### 事实 C：当前 KEG 还不能称为 AEGNN/FARSE 意义上的真异步网络

`semkine/keg.py` 当前做的是包内统计和包级重算；`semkine/frontends.py` 中的 AEGNNLite 仍是 packet-wide `torch.cdist`；没有 caller-owned、跨调用持久的激活状态。因此它可以称为：

- raw-event frontend；
- sparse/ragged input；
- packet-recursive aggregation；

但还不能严谨声称：

- event-by-event asynchronous inference；
- affected-node-only recomputation；
- arbitrary-query persistent event state。

### 事实 D：`AdaptivePacketizer` 有必须先修的流式合同问题

`semkine/packetizer.py` 会把尾部数据放进 `_buf`，但下一次 `push()` 没有将 `_buf` 与新输入重新拼接。结果是跨调用事件可能被丢失，而且 `_buf` 同时混放 events、timestamp、mask、group，`_stats()` 对它的解释也不稳定。

在这个问题修复之前，不允许基于该 packetizer 写“完整事件守恒”或“真流式”claim。

### 事实 E：仓库已经有很强的几何资产，不能再平行重写一套

现有分支已经包含：

- `semkine/lie.py`：`SE(3) × SO(3)^15` 状态、Log/Exp、51D retraction；
- `semkine/filter.py`：Lie error-state filter、速度状态、Q、Joseph update、NIS；
- `semkine/jacobian.py`：MANO 投影解析 Jacobian、Fisher 信息；
- `semkine/gn.py`：active-set GN/LM；
- `semkine/anchor.py`：流形上的绝对 anchor 融合与触发器；
- `semkine/mainline.py`：routing/filter/anchor 的组合；
- `semkine/eval_track.py`：一次 GT 初始化后严格递归的 evaluator。

因此最佳新框架应该是**把这些资产接成一个统一可训练、真流式的观察器**，而不是再写一套独立 EKF、MANO Jacobian 或后处理器。

---

## 1.2 当前递归失败的第一性原理推导

令真实状态为 `X_k`，预测状态为 `X̂_k`，流形误差为：

\[
 e_k = \operatorname{Log}(X_k^{-1}\hat X_k) \in \mathbb{R}^{51}.
\]

当前 KEG 的一步映射可抽象为：

\[
\hat X_k = \Phi(E_k,\hat X_{k-1})
          = \operatorname{Retr}\Big(\hat X_{k-1},
             f\big(H(E_k,\hat X_{k-1}),\hat X_{k-1}\big)\Big),
\]

其中：

- `E_k` 是当前事件包；
- `H` 是 KSSF、可见性、LBS、SDF、hard/halo routing 等姿态条件化观测；
- `f` 是事件编码器和 51D head。

在真实轨迹附近线性化：

\[
 e_k \approx A_k e_{k-1}+\epsilon_k,
\]

而闭环 Jacobian 可拆为：

\[
A_k \approx
\underbrace{\frac{\partial \Phi}{\partial \hat X_{k-1}}\Big|_{H\ \mathrm{fixed}}}_{\text{状态递归的直接项}}
+
\underbrace{\frac{\partial \Phi}{\partial H}
\frac{\partial H(E_k,\hat X_{k-1})}{\partial \hat X_{k-1}}}_{\text{姿态条件化观测的回声项}}.
\]

LNES 是状态无关输入，所以第二项近似为零；KEG/KSSF 的第二项不为零。halo 只是把 `H` 对状态的变化变平滑，不能令其为零，因此能改善但不能根治。这与日志中“单步接近、递归差大、halo 有效但不足”完全一致。

### 必须引入一个状态无关观察器

若增加绝对/恢复分支：

\[
X_k^A = g_A(M(E_{\le k})),
\]

其中持久事件记忆 `M` 的更新不使用 `X̂_{k-1}`，则该分支的观测误差不会通过 KSSF 再次复制上一帧误差。把状态无关分支与条件化分支融合，闭环敏感度可近似写为：

\[
A_{\mathrm{fused}} \approx W_C A_C + W_A A_A,
\]

其中状态无关绝对分支的 `A_A` 显著小于完全 KSSF 条件化分支，因而只要 `W_A` 不是零，就有机会降低闭环谱半径，而无需丢掉 KEG 对局部手指运动的精确性。

---

# 第二部分：最佳主方案——AERO-Hand

## 2.1 名称与一句话定义

**AERO-Hand：Asynchronous Error-state Recovery Observer for Hand Mesh**

> 用一个完全不依赖上一帧姿态的真异步稀疏事件记忆产生绝对/恢复证据，用现有 KSSF–KEG 产生局部运动学创新，再在 `SE(3) × SO(3)^15` 的误差状态空间中以信息形式融合，并用真实闭环 rollout 训练其收缩性。

这不是“AEGNN + MANO head”，而是一个事件专用观察器。

## 2.2 结构总览

```text
raw x,y,t,p
    │
    ├── Pose-free Persistent Async Memory M_free
    │      ├── two active-cell scales
    │      ├── exact exponential decay / chunk invariance
    │      ├── 16 joint queries + root query
    │      └── absolute state X_A + block information Λ_A
    │
    ├── Pose-conditioned KSSF/KEG M_kin
    │      ├── keep current KSSF semantics and halo
    │      ├── only predict local tangent innovation μ_C
    │      └── analytic Fisher / calibrated information Λ_C
    │
previous posterior X⁺, v⁺, P⁺
    │
    ├── Lie process prediction → X⁻, v⁻, P⁻
    │
    └── blockwise robust information fusion
           δ* = argmin prior + absolute + local innovation
           X⁺ = Retr(X⁻, δ*)
```

## 2.3 真异步事件记忆

对尺度 `s` 的活动单元 `c`，定义持久状态：

\[
m^s_c(t)\in\mathbb{R}^{d_s}.
\]

事件 `e_i=(x_i,y_i,t_i,p_i)` 到达其单元 `c_s(e_i)` 时：

\[
m^s_c(t_i^+)=
\exp[-\lambda_s (t_i-t_i^-)]\odot m^s_c(t_i^-)+B_s\phi(e_i).
\]

查询时刻 `t_q`：

\[
\bar m^s_c(t_q)=\exp[-\lambda_s(t_q-t_c^{\mathrm{last}})]\odot m^s_c(t_c^{\mathrm{last}}).
\]

这个更新满足半群性质：

\[
U(E_2,U(E_1,m_0))=U(E_1\cup E_2,m_0),
\]

只要事件顺序一致，因此同一流一次送入、随机分块送入、逐事件送入，查询状态必须一致。这是“真异步”最重要的可测试合同。

**建议的最小实现：**

- 两个尺度：8 px 与 24 px；
- 隐层：48 或 64；
- 4 个正的衰减时间尺度；
- 存储可以是规则网格 tensor，但更新计算必须只触碰活动 index；
- 不使用事件级 `kNN`，不使用 `torch.cdist`；
- 空事件只做时间衰减和 filter predict。

## 2.4 16 个关节查询，而不是全局 mean/max pool

令活动单元特征为 `h_c`，根与 15 个 MANO 关节查询为 `q_j`：

\[
a_{jc}=\operatorname{softmax}_{c\in\mathcal A}
\left(\frac{(W_q q_j)^T(W_k h_c)}{\sqrt d}+b_j(x_c,y_c)\right),
\]

\[
z_j=\sum_{c\in\mathcal A}a_{jc}W_vh_c.
\]

然后仅做 1–2 层 MANO tree message passing：

\[
z'_j=\sigma\left(W_s z_j+\frac1{|\mathcal N(j)|}
\sum_{u\in\mathcal N(j)}W_n z_u\right).
\]

这借用了 StreamPETR/Sparse4D 的“对象查询随时间传播”和 Hamba 的“少量关节 token + 图结构扫描”，但没有把大规模检测 transformer 生搬进来。

## 2.5 两类观测

### 状态无关绝对观测

\[
X_A = g_A(\{z'_j\}_{j=0}^{15},z_{global}),
\]

它的输入不包含 `prev_state`、KSSF 或上一帧投影。相对于 process prior：

\[
\mu_A=\operatorname{Log}((X^-)^{-1}X_A).
\]

### 条件化局部创新

保留当前 KSSF/KEG，但令它只输出：

\[
\mu_C=g_C(E_k,\operatorname{KSSF}(X^-))\in\mathbb R^{51}.
\]

最终状态不再使用 `prev + delta`，而是：

\[
X_C=\operatorname{Retr}(X^-,\mu_C).
\]

## 2.6 信息融合，不使用自由学习的 sigmoid gate

每个块分别处理：

- root translation：3D；
- root rotation：3D；
- 15 个 joint rotation：每块 3D。

对第 `b` 块，先得到：

\[
\Lambda_{P,b}=(P_b^-)^{-1},\quad
\Lambda_{A,b},\quad
\Lambda_{C,b}.
\]

两个网络分支高度相关时，直接相加信息会过度自信。首版使用块级 covariance intersection：

\[
\Lambda_{F,b}(\omega)=\omega\Lambda_{A,b}+(1-\omega)\Lambda_{C,b},
\]

\[
\eta_{F,b}(\omega)=\omega\Lambda_{A,b}\mu_{A,b}
+(1-\omega)\Lambda_{C,b}\mu_{C,b},
\]

\[
\omega_b^*=\arg\min_{\omega\in\{0,0.125,\ldots,1\}}
\log\det(\Lambda_{F,b}(\omega)^{-1}).
\]

然后：

\[
\delta_b^*=(\Lambda_{P,b}+\Lambda_{F,b})^{-1}\eta_{F,b},
\]

\[
X^+=\operatorname{Retr}(X^-,\delta^*).
\]

分支明显冲突时，用对称 Huber 信息缩放：

\[
d_b=(\mu_{A,b}-\mu_{C,b})^T
(\Sigma_{A,b}+\Sigma_{C,b})^{-1}
(\mu_{A,b}-\mu_{C,b}),
\]

\[
w_b=\min\left(1,\sqrt{\tau_b/(d_b+\epsilon)}\right),
\quad \Lambda_{F,b}\leftarrow w_b\Lambda_{F,b}.
\]

这会在两者冲突时回退到 process prior，而不是让一个任意 gate 把错误分支放大。

## 2.7 训练必须改成真实闭环分布

当前二窗口 residual perturbation 不能代表几十步后相关、非高斯的 recursive state。训练单元必须是连续 run：

\[
\hat X_0=X_0^{GT},\qquad
\hat X_k=F_\theta(E_k,\hat X_{k-1}),\ k=1\ldots K.
\]

除 `k=0` 外，不得把 GT 状态再喂回网络。使用 TBPTT：

- Stage 1：`K=4`；
- Stage 2：`K=8`；
- Stage 3：`K=16`；
- 最终稳定性：随机 `K∈{8,16,32}`。

损失：

\[
\mathcal L=\sum_{k=1}^{K}\gamma^{K-k}
\left(
\lambda_J\mathcal L_J+
\lambda_V\mathcal L_V+
\lambda_R\mathcal L_{SO(3)}+
\lambda_T\mathcal L_T+
\lambda_N\mathcal L_{NLL}+
\lambda_C\mathcal L_{contract}
\right).
\]

其中：

\[
\mathcal L_{SO(3)}=\sum_j
\|\operatorname{Log}(R_{j,gt}^TR_{j,pred})\|_1,
\]

\[
\mathcal L_{NLL}=\frac12 r^T\Lambda r-\frac12\log\det\Lambda,
\]

收缩正则用随机方向有限差分或 JVP：

\[
g=\frac{\|e_k(\hat X_{k-1}\boxplus \delta)-e_k(\hat X_{k-1})\|}
{\|\delta\|+\epsilon},
\]

\[
\mathcal L_{contract}=\max(0,g-g_{target})^2.
\]

它不是要求每一步强收缩，而是防止网络学出日志中 `0.30+` 的误差回声。

---

# 第三部分：2023–2026 机会图——借机制，不搬网络

以下不是“所有论文的书目穷举”，而是覆盖用户指定方向的**机制完整代表图**。

| 方向 | 代表工作 | 可借机制 | 不应照搬的部分 | 在 EventHands1 中的落点 |
|---|---|---|---|---|
| 真异步图 | AEGNN | 新事件只重算受影响节点；同步训练/异步推理 | 事件级全局 kNN 图 | 固定活动单元图、缓存每层激活 |
| 递归点事件 | EventNet | 事件到达即更新、常数内存 | 全局 PointNet pooling 直接回归 51D | 单元级时间编码与递归统计 |
| 稀疏异步 CNN | FARSE-CNN | recurrent sparse state、跨尺度 compression | 完整检测 backbone | 快/慢两级手部记忆 |
| 低延迟记忆 | RVT、HMNet | 持久时间记忆、不同速率层级 | 稠密窗口 token | coarse slow root + fine fast fingers |
| 自适应稀疏 | SAST | 按场景证据选择计算 | 检测 window token 筛选器 | 按关节 Fisher 触发 query/update |
| Event SSM | State Space Models for Event Cameras | 正时间常数、跨频率泛化 | 把整段事件先栅格化再扫 | exact decay bank / arbitrary query |
| 事件手部 | EvHandPose | contrast maximization、hand-edge sparse supervision | 手流表示作为唯一输入 | 训练期 event likelihood 辅助损失 |
| 单目事件双手 | Ev2Hands | 左右手消歧、碰撞约束 | 双手专用大结构 | 可选的 self-collision/part ownership |
| Event+RGB hand mesh | EvRGBHand | 模态互补与退化建模 | 推理时依赖 RGB | 用其教师/退化思想做预训练，不在主模型用 RGB |
| 动态背景手部 | EventEgoHands | 手区域分离，抑制 ego/background events | 单纯二值 mask 后再密集回归 | Plan 9 border ownership/ego decouple |
| 连续事件人体 | Continuous-Time Human Motion Field | 时间隐式姿态、任意时刻查询 | 大人体 latent 直接迁移 | Plan 6 的 Lie spline oracle |
| RGB 手部基础 | HaMeR | 大数据、强绝对手先验 | 大 ViT 作为事件主干 | 仅作 absolute teacher/预训练 |
| 轻量手 mesh | Efficient Hand Mesh Baseline | 少量判别 token + mesh regressor | RGB token generator | 16 joint query 的轻量设计 |
| 图与 SSM | Hamba | 关节图引导少 token 扫描 | 双向非因果视频扫描 | query 时 MANO tree causal/structural scan |
| 一阶段手重建 | HandOS | detection/2D/3D 一体化，避免级联错误 | 大型 frozen detector | 事件 foreground、joint、mesh 共享 query |
| 动态相机手 | Dyn-HaMR、HaWoR | 相机/手运动解耦、world grounding | 离线多阶段大优化 | Plan 9 的条件分支 |
| 不确定手 mesh | MaskHand | 分布、置信度、遮挡恢复 | 生成式多样本采样作为在线主干 | block precision 和 recovery trigger |
| 在线人体 mesh | OnlineHMR | 严格因果、cache、在线一致性 | 人体 SLAM 全系统 | caller-owned state 与在线协议 |
| 物理手运动 | PAD-Hand | 物理残差作为虚拟观测、方差 | diffusion 主干 | Plan 10 的轻量 virtual observation |
| 流式 3D 检测 | StreamPETR | 对象查询跨时间传播 | 数百检测 query | 16 MANO joint queries |
| 稀疏 4D | Sparse4D/DetAny4D | 稀疏实例状态、时序一致性 | BEV/box decoder | 关节级状态与 sequence consistency |
| stereo/depth | IGEV-Stereo | 强初值 + 小步迭代修正 | dense cost volume | process/absolute anchor + local innovation |
| 3D 点跟踪 | SpatialTracker | 在 3D/低维运动域跟踪、ARAP | 稠密像素轨迹 | MANO 低维状态与骨架约束 |
| Event 3D tracking | 3D Feature Tracking via Event Camera | motion compensation、双极性关联 | 双目几何 | 单目局部 motion compensation/association |
| Event stereo 3D | Ev-Stereo3D | 连续时间检测、语义与几何双滤波 | stereo/3D box 结构 | 语义记忆与几何创新解耦 |
| 统一 4D | Any4D | ego/allo 因子解耦、模块化 4D 状态 | 大规模多视图 transformer | 相机与手运动因子分离 |
| Event VO | 5-point event relative motion、Full-DoF egomotion | 事件流形、线/边界运动几何 | 场景线假设直接套手 | Plan 9 camera motion probe |
| Event SLAM | EN-SLAM | 连续事件差分约束、tracking+BA | 隐式场景重建 | event residual 进入 error-state estimator |
| 动态分割 | Moving Border Ownership | 边界所属侧、遮挡即建模 | 通用场景完整分割网络 | 手轮廓 ownership 与背景抑制 |
| 动态 BA | HumanBA | 动态人体也可成为几何约束 | 离线全局 BA | 手/相机分量的可靠性加权 |
| 数据蒸馏 | EventHub | 无昂贵 GT 的 proxy event/label factory | stereo foundation model | RGB/仿真预训练 pose-free async branch |

---

# 第四部分：100 角色专家 battle

## 4.1 角色构成

每组 10 个角色，共 100 个：

1. 事件表示与传感器物理；
2. AEGNN/动态图；
3. 稀疏 CNN/Transformer/SSM；
4. MANO/手部 mesh/运动学；
5. tracking/filter/control；
6. scene flow/trajectory/continuous time；
7. stereo/depth/3D detection；
8. VO/SLAM/ego-motion；
9. 优化、不确定性与数值稳定；
10. CVPR/NeurIPS 审稿、复现和系统评测。

## 4.2 每个方案的评分维度

- 根因覆盖：25；
- 闭环稳定性：20；
- 真异步性：15；
- 代码复用/复杂度：10；
- 创新性：15；
- 可证伪性：10；
- 公开 benchmark 适配：5。

## 4.3 共识排序

| 排名 | 方案 | 综合判断 | 最强支持理由 | 最大反对理由 |
|---:|---|---|---|---|
| 1 | P1 AERO-Hand 双观察器 | **主线** | 同时解决姿态回声、真异步、流形更新和 rollout mismatch | 训练和校准步骤多，必须严格分阶段 |
| 2 | P4 KineQuery16 | 高潜力 | 去掉 global pooling，直接形成关节级持久证据 | query 若被 prev pose 条件化会重新产生回声 |
| 3 | P3 Persistent Cell-AEGNN | 高可行 | 最直接修正当前 `cdist` 假异步问题 | 单独换 frontend 未必改善闭环 |
| 4 | P7 NeuMANO-FG Lite | 高创新 | 网络学关联、解析几何解状态，解释性强 | 错误 correspondence 会让 GN 精确地走错 |
| 5 | P2 Lie-IEKF-Net | 最稳妥控制臂 | 改动最小，能快速验证 dynamics/retraction 假设 | 单独作为论文主贡献不够新 |
| 6 | P5 FARSE-Kine | 中高 | 快慢时间尺度天然适合 root/finger | 工程复杂度高于 PC-AEGNN |
| 7 | P10 Event-Physics/Domain | 中高附加贡献 | 解决标注稀缺与 sim2real | 不能替代主观察器，只能后加 |
| 8 | P8 信息触发查询 | 中等 | 很适合事件相机的 rate/latency claim | 对 RA 精度的直接增益不确定 |
| 9 | P9 EgoHand-Decouple | 条件高价值 | 动态相机数据上可能是关键瓶颈 | 固定相机数据上没有必要 |
| 10 | P6 CT-KineFlow | 高风险高收益 | 理论上最 event-native 的时间表示 | 现有 5–50 ms 日志不支持“窗口离散化是主因” |

## 4.4 Battle 中所有组都同意的六条否决线

1. **没有跨调用持久状态和 chunk-invariance test，不得称真异步。**
2. **继续 `pred = prev + delta` 直接相加 51D，任何 SOTA claim 都站不住。**
3. **只做 teacher-forced 单步提升，不足以证明 tracking 有效。**
4. **同一个 KSSF 条件化分支再叠更多 GNN/Transformer，不能消除姿态回声。**
5. **学习一个无约束 sigmoid gate/variance head，而不做 NLL、coverage、NIS 校准，会被认为是任意融合。**
6. **只在自有数据上称普适 SOTA，不足以支撑 CVPR/NeurIPS 级 claim。**

---

# 第五部分：十个方案总表与实施顺序

| 方案 | 核心变量 | 新增核心文件上限 | 首个 GPU 前置诊断 | 进入正式训练的门槛 |
|---|---|---:|---|---|
| P1 AERO-Hand | pose-free observer + info fusion | 2 | memory chunk parity、absolute oracle | 绝对分支单步不差于 KEG 1.5 mm 以上；g 方向正确 |
| P2 Lie-IEKF | retraction/process/filter | 0–1 | frozen inference paired eval | recursive 至少改善 2 mm，NIS 不发散 |
| P3 PC-AEGNN | active-cell affected update | 1 | sync/async parity、op count | 单步不差 current KEG 0.5 mm 以上 |
| P4 KineQuery16 | 16 joint queries | 1 | attention entropy/part ownership | per-finger oracle 比 global pool 明显更好 |
| P5 FARSE-Kine | recurrent sparse hierarchy | 1 | state equivalence、compression loss | 参数/ops 门槛内且单步持平 |
| P6 CT-KineFlow | continuous pose field | 1 | GT trajectory oracle | oracle 至少改善 1.5 mm 或 event residual 20% |
| P7 NeuMANO-FG | learned ownership + GN | 1 | GT correspondence oracle | oracle GN 明显优于 learned direct head |
| P8 Info Packetizer | Fisher-triggered query | 0–1 | event conservation、matched-rate curve | 同 RA 下 query/latency 降低 ≥25% |
| P9 Ego Decouple | camera/background motion | 1 | camera-motion contamination probe | moving-camera subset有显著 gap，固定场景不退化 |
| P10 Event Physics | weak/self supervision | 1 | 每个 loss 单独 zero/train probe | 单一 loss 在 real val 改善且不破坏 recursive |

**推荐执行顺序：**

```text
F0 流式合同修复
  → P2 冻结网络的 Lie-filter 控制臂
  → P1 pose-free memory + dual observer（主线）
  → P3 或 P4，选一个加强主线
  → P7（若解析 correspondence oracle 成立）
  → P8 做 latency/rate 论文点
  → P10 做 sim2real/弱监督
  → P9 仅在动态相机诊断通过后做
  → P6 仅在 continuous-time oracle 通过后做
```

---

# 第六部分：统一实验协议与硬门槛

## 6.1 两套基线必须分开记录

### Protocol A：历史 frozen `track_render`

- 参考值：`19.25760436702419 mm`；
- 只有完全相同 split/evaluator/checkpoint/window 时才可比较；
- 按历史跨训练噪声地板，论文级胜出门槛建议：`< 18.16 mm`，即超过约 `1.1 mm`。

### Protocol B：当前 S20/S21/S22 matched protocol

- 每一个事件窗口和随机种子都同时跑 LNES 与 candidate；
- 以 paired difference 为主，不把另一个 protocol 的 19.26 混进来；
- 最终门槛：三种子平均至少优于同协议 LNES `1.1 mm`，且每个种子不允许灾难性失败。

## 6.2 必报指标

- RA-MPJPE、MPJPE、MPVPE；
- root translation、root rotation geodesic；
- 每根手指/每个关节组误差；
- P50/P90/P95/P99；
- jitter、acceleration error、最大连续漂移；
- strict recursive failure rate；
- g-curve slope、fixed-point error、echo excess；
- zero-event 行为；
- latency/query、latency/event、active cell/node 数、峰值显存；
- query frequency generalization：5/10/20/50 ms 和信息触发；
- uncertainty：NLL、coverage、NIS、ECE 或 reliability diagram。

## 6.3 统一工程合同

- 所有一次性 probe 放在 `/tmp/eventhands_<plan>_*.py`；
- 不新增 Gate JSON；结果追加到：
  - `docs/semkine/EXPERIMENT_LOG.md`
  - `docs/FAILURE_AND_CLEANUP_LEDGER.md`
- 每个训练臂只改变一个研究变量；
- quick probe → 单种子 early-stop → 两种子 → 三种子 final；
- checkpoint 依据 strict recursive metric 选，不依据 `val_loss`；
- raw events 保留 `x,y,t,p`、`ptr`、`sequence_id`、绝对 `t_start_us/t_end_us`；
- 不做 `N_max` 全批 padding；
- 不跨 sequence 传播任何 state；
- 不使用 GT prev，除每个 valid run 的唯一初始化；
- 零事件包合法，必须只 propagate process/decay；
- 任何“异步”实现必须通过 full-stream、random-chunk、event-by-event 三路等价测试；
- 新增依赖前先证明现有 PyTorch scatter/index 操作做不到；
- 禁止在 forward 中重建 MANO model 或反复加载 checkpoint；
- 禁止 `torch.cdist(events, events)`；
- 禁止把十个想法一次性组合训练。


---

# 第七部分：F0——所有方案之前必须完成的流式合同修复

F0 不是论文创新点，而是所有“异步、连续、任意查询”实验的可信前提。它不允许启动完整训练，只修合同并做 parity test。

## 7.1 修复 `AdaptivePacketizer` 的事件守恒

当前 `AdaptivePacketizer.push()` 在调用结束时把余量写入 `_buf`，但下一次调用没有把 `_buf` 拼回，因此应重构为具名 buffer：

```python
@dataclass
class StreamBuffer:
    events_xypt: np.ndarray
    t_us: np.ndarray
    contour_mask: np.ndarray
    group_ids: Optional[np.ndarray]
```

下一次 `push()` 首先执行：

\[
B_{new}=\operatorname{concat}(B_{old}, S_{new}),
\]

然后只在合并流上找切点。必须保证：

\[
N_{in}=N_{emitted}+N_{buffered},
\]

并且输出包按绝对时间严格单调、无重复、无遗漏。

### 必须新增/扩展的测试

- 两次 `push` 与一次拼接 `push` 的所有包逐项一致；
- 随机 2–50 次 chunking 与 full-stream 结果一致；
- `flush()` 明确定义：结束时输出尾包或保留尾包，不得静默丢弃；
- reset 后 buffer、时间起点、统计量全部清空；
- `contour_mask/group_ids=None` 时合同仍成立；
- 同一时间戳事件保持输入稳定顺序。

## 7.2 把 51D 欧氏加法改为流形组合

当前 tracking 语义保留：

```text
[translation 3, root rotation axis-angle 3, 15 local joint axis-angle × 3]
```

但最终更新改为：

\[
\hat X_k=\operatorname{Retract}_{51D}(\hat X_{k-1},\delta_k),
\]

调用仓库已有的 `semkine.lie.retract_51d`。为了避免一次性改变主结果，先新增 config：

```yaml
MODEL:
  UPDATE_RULE: euclidean_control   # frozen control
# or
  UPDATE_RULE: lie_retraction      # candidate
```

硬门槛：

- `delta=0` 时逐位 bitwise identity；
- 小角度 `||δ||<1e-5` 时与欧氏加法一阶一致；
- 大旋转连续多步不产生 NaN、无 axis-angle 跳变导致的 mesh 爆炸；
- old control config 数值不变。

## 7.3 建立 caller-owned stream state

新接口必须显式传入和返回状态，禁止藏在全局模块属性中：

```python
state = model.init_stream_state(
    sequence_id=...,
    x0_51d=...,
    betas=...,
    camera_K=...,
    t0_us=...,
)
state = model.update_events(state, events_xypt, t_us)
pred, state = model.query_pose(state, query_t_us)
state = model.reset_stream_state(state)
```

状态至少包含：

```text
sequence_id
last_event_t_us
last_query_t_us
pose posterior / previous pose
persistent event memory
per-cell last-update time
optional velocity and covariance
betas and camera_K references
```

### F0 Gate

F0 只有在以下全部通过后才结束：

1. packetizer 随机 chunk parity：`max_abs_diff == 0`，事件守恒精确；
2. stream state 不跨 sequence 泄漏；
3. zero-event query 合同明确并通过；
4. Lie retraction 单元测试通过；
5. frozen Euclidean control 的旧结果可复现；
6. 不启动超过 100 step 的训练。

---

# 第八部分：十个方案的详细设计

## P1：AERO-Hand——状态无关异步恢复观察器 + KSSF 局部观察器

### P1.1 研究假设

当前 KEG 的主要缺陷不是 KSSF 无效，而是所有观测都依赖上一姿态。增加一个 pose-free 持久事件观察器后：

\[
\left\|\frac{\partial \mu_A}{\partial \hat X_{k-1}}\right\|\approx 0,
\]

再与 KSSF 局部观测融合，可以同时保留：

- pose-free 分支的全局恢复能力；
- KSSF 分支的局部手指精修能力；
- 原 tracking 的每段一次初始化和严格递归。

### P1.2 模块

**模块 A：PoseFreeAsyncMemory**

- 8 px / 24 px 两尺度活动单元；
- 每尺度 4 个正衰减常数；
- 事件只更新命中的 cell；
- query 时仅 decay 活动 cell，不重新读取整个历史；
- 绝不访问 `prev_state`、render、KSSF、MANO 投影。

**模块 B：KineQuery16**

- root + 15 MANO joint query；
- 每个 query 对活动 cell 做一次小型 cross-attention；
- 一层 MANO tree message passing；
- 输出绝对 `X_A` 与每块 precision 参数。

**模块 C：KSSF innovation**

- 复用当前 `KinematicEventGraph`、KSSF、halo；
- 输出局部 tangent innovation `μ_C`；
- 复用解析 Fisher `Λ_C` 或对其做单调标定。

**模块 D：Lie prediction/fusion**

- 复用 `semkine/filter.py` 的 process prior；
- 复用 `lie.py` 的 retraction；
- 首版使用 blockwise covariance intersection；
- 只允许 17 块：root translation、root rotation、15 joints。

### P1.3 信息参数化

每个 3D block 不预测完整任意矩阵，先预测 Cholesky 对角与下三角：

\[
L_b=
\begin{bmatrix}
\operatorname{softplus}(a_1)+\epsilon&0&0\\
b_1&\operatorname{softplus}(a_2)+\epsilon&0\\
b_2&b_3&\operatorname{softplus}(a_3)+\epsilon
\end{bmatrix},
\qquad
\Lambda_b=L_bL_b^T.
\]

这样 precision 必定正定。绝对分支训练使用：

\[
\mathcal L_A=
\sum_b\left[
\frac12 r_{A,b}^T\Lambda_{A,b}r_{A,b}
-\frac12\log\det\Lambda_{A,b}
\right].
\]

为防止网络把所有 precision 压到零，加入弱范围先验：

\[
\mathcal L_{prec}=\sum_b
\left(\max(0,\ell_{min}-\log\det\Lambda_b)^2+
\max(0,\log\det\Lambda_b-\ell_{max})^2\right).
\]

### P1.4 代码修改

| 文件 | 修改 |
|---|---|
| `semkine/async_memory.py`（新） | 持久 active-cell memory、chunk-equivalent update/query |
| `semkine/aero.py`（新） | 16 query、absolute/local observation、CI fusion、stream state |
| `semkine/keg.py` | 暴露 KSSF observation，不重写已有实现 |
| `model/model.py` | 新 encoder/observer 分发；`lie_retraction`；保留 control |
| `semkine/dataset.py` | contiguous sequence chunks、随机 chunk boundaries、rollout targets |
| `semkine/train.py` | TBPTT rollout curriculum、state detach、recursive validation hook |
| `semkine/eval_track.py` | caller-owned state 与任意 query path；旧 path parity |
| `configs/semkine/p1_aero_hand.yaml` | 独立 config，所有新 key fail-fast |

新增核心文件上限为 2。其余必须扩展现有入口。

### P1.5 分阶段 Gate

**P1-A：异步记忆合同，无训练**

- full stream / random chunks / event-by-event：`max_abs_diff <= 1e-6`；
- 每次 event 更新 touched cell 数与理论一致；
- 内存不随 stream 长度线性增长；
- zero event 只 decay，不制造新证据。

**P1-B：pose-free absolute oracle，单步训练**

- 绝对分支禁止读取 `prev_state`；
- teacher-forced RA 不得比 current KEG 差超过 `1.5 mm`；
- 对输入 `prev_state` 做随机替换，`X_A` 必须 bitwise/浮点容差内不变；
- 每指 query attention 不能全部退化成相同分布：平均 pairwise cosine < 0.95，且消融某 query 主要影响对应手指。

**P1-C：单种子短闭环**

- recursive RA `<= 26 mm`；
- g-slope at 26 mm `<= 0.24`；
- no divergence；
- absolute branch precision 与 realized block error Spearman `>=0.25`。

**P1-D：两种子主 Gate**

- 两 seed 都优于 matched KEG；
- worse-of-two recursive RA `<=22.5 mm`；
- paired improvement 对 matched LNES 至少开始接近 0，不允许仍差 5 mm 以上；
- 5/10/20/50 ms 全部稳定。

**P1-E：三种子论文 Gate**

- 同协议平均优于 LNES `>=1.1 mm` 或 paired bootstrap 95% CI 完全低于 0；
- long-horizon、low-event、fast-motion 三桶均不退化；
- p95 latency 或 event operation 至少有一个比 LNES 改善 `>=20%`；
- calibration coverage 合理。

### P1.6 必要消融

```text
A1 pose-free only
A2 KSSF only/current KEG
A3 pose-free + KSSF, naive average
A4 pose-free + KSSF, learned sigmoid gate
A5 pose-free + KSSF, block CI（主方法）
A6 CI without process prior
A7 no rollout
A8 2-window residual rollout
A9 true K-step rollout
A10 Euclidean update vs Lie retraction
```

### P1.7 论文贡献边界

可以主张：

- 第一种面向单目事件手 mesh tracking 的 pose-free/pose-conditioned 双观察器；
- caller-owned true asynchronous memory；
- 信息形式的流形融合与闭环收缩训练；
- 低延迟任意时刻 mesh query。

不能在 Gate 前主张：

- 一定超过所有 RGB/event 方法；
- 神经形态硬件能耗优势；
- event-by-event latency 已优于 GPU LNES，除非真实测量。

---

## P2：Lie-IEKF-Net——最小改动的闭环动力学控制臂

### P2.1 研究假设

即使不改 backbone，当前 `prev + delta`、缺少显式速度/协方差和固定更新权重，也可能贡献了一部分递归误差。用已有 LieFilter 把网络输出当观测，可先回答：

> 仅修正流形动力学和证据权重，能否把当前 KEG/LNES 的递归误差明显拉低？

这是一条**低成本因果诊断**，不是最终主论文。

### P2.2 模型

状态：

\[
s_k=(X_k,v_k,P_k),\qquad X_k\in SE(3)\times SO(3)^{15}.
\]

预测：

\[
X_k^-=X_{k-1}^+\boxplus(\Delta t_k v_{k-1}^+),
\]

\[
v_k^-=e^{-\gamma\Delta t_k}v_{k-1}^+,
\qquad
P_k^-=\Phi_kP_{k-1}^+\Phi_k^T+Q(\Delta t_k).
\]

网络观测：

\[
z_k=\operatorname{Log}((X_k^-)^{-1}X_k^{net}).
\]

观测协方差先不用新 learned head，而采用：

\[
R_k=(\alpha\Lambda_k+\beta I)^{-1},
\]

其中 `Λ_k` 是已有解析 Fisher。没有 `Λ` 的 LNES control 使用 event count/rate 的单调标定，只作为公平对照。

更新：

\[
K_k=P_{xx}^-(P_{xx}^-+R_k)^{-1},
\quad
X_k^+=X_k^-\boxplus K_kz_k,
\]

协方差用 Joseph form。

### P2.3 代码

优先只改：

- `semkine/mainline.py`；
- `semkine/eval_track.py`；
- `configs/semkine/p2_lie_iekf_<arm>.yaml`。

禁止另写第二套 filter。必要时仅新增一个薄 adapter。

### P2.4 实验臂

```text
F0 raw network output
F1 Lie retraction only
F2 constant-gain Lie filter
F3 Fisher-weighted Lie filter
F4 Fisher-weighted + velocity
F5 F4 + calibrated process noise
```

每次只打开一个变量。

### P2.5 Gate

- 完全 frozen checkpoint、paired inference；
- recursive RA 至少改善 `2.0 mm` 才值得进入 P1；
- NIS 中位数/coverage 不明显过度自信；
- filter 不得用 GT 选每帧 gain；
- 5/10/20/50 ms 都不发散；
- 若所有臂改善 `<1.1 mm`，结论是 dynamics 不是主要项，停止单独 filter 扫参。

---

## P3：PC-AEGNN——Persistent-Cell Affected Graph Neural Network

### P3.1 研究假设

像素事件 kNN 既昂贵又不稳定；但 AEGNN 的真正价值不是“图”本身，而是：

> 新事件到达时，只更新其影响邻域，并缓存其余节点的层级激活。

把事件图换成固定多尺度 active-cell graph，可保留受影响节点更新，同时将邻域与复杂度固定。

### P3.2 图定义

尺度 `s` 的 cell 节点：

\[
c=(\lfloor x/s\rfloor,\lfloor y/s\rfloor).
\]

固定 8-neighbor + parent/child cross-scale edges。一个事件首先更新源 cell：

\[
h_c^{(0)+}=D_{\Delta t}h_c^{(0)}+\phi(e_i).
\]

第 `\ell` 层只更新受影响集合：

\[
\mathcal A_0=\{c(e_i)\},\qquad
\mathcal A_{\ell}=\mathcal A_{\ell-1}\cup
\mathcal N(\mathcal A_{\ell-1}).
\]

对 `v∈A_l`：

\[
h_v^{(\ell)+}=\sigma\left(
W_s h_v^{(\ell-1)+}+
\sum_{u\in\mathcal N(v)}W_m h_u^{(\ell-1)+}
\right).
\]

未受影响节点的缓存保持不变。同步 full recompute 必须与 affected update 对齐。

### P3.3 与原 AEGNN 的区别

- 不建立事件—事件全局 kNN；
- 不用 `torch.cdist`；
- 图节点上界由传感器网格决定；
- 事件只更新常数大小邻域；
- 可与 P1 的 pose-free 观察器直接组合；
- GNN 是表示机制，不承担整篇论文的所有贡献。

### P3.4 代码

- 新增 `semkine/persistent_cell_gnn.py`；
- 修改 `semkine/frontends.py` 注册新 frontend；
- 修改/复用 `semkine/async_memory.py` 的 stream state；
- 新增 `tests/test_pc_aegnn.py`；
- config：`p3_pc_aegnn_abs.yaml` 和 `p3_pc_aegnn_keg.yaml`。

### P3.5 Gate

1. 同一权重下 synchronous full recompute 与 affected update：`max_abs_diff <=1e-5`；
2. operation count 与事件数近似线性，不随历史长度增长；
3. 5 万事件包不能出现 O(N²) 内存；
4. 单步 RA 不得比 current KEG 差超过 `0.5 mm`；
5. 若单独换 frontend 后 recursive 仍差 LNES `>5 mm`，不继续扩大 GNN；转回 P1 双观察器。

---

## P4：KineQuery16——16 个持久关节查询代替全局池化

### P4.1 研究假设

当前 raw scan、AEGNNLite、许多 PointNet 路径最终都做 global mean/max。对手部而言，这把小拇指局部事件和腕部大轮廓混为一体。MANO 已知只有 16 个运动学节点，因此用 16 个持久 query 是最小而必要的结构先验。

### P4.2 查询不能依赖上一姿态做 hard association

查询向量由：

- 可学习 joint identity；
- canonical MANO parent/depth/finger ID；
- pose-free event memory；

构成，不得用上一姿态投影决定哪些事件归哪个 query。允许把 `prev_state` 只送入最终 dynamics head，但做消融。

\[
q_j^0=E_{joint}(j)+E_{finger}(f_j)+E_{depth}(d_j).
\]

单层 sparse cross-attention：

\[
z_j=\sum_{c\in\mathcal A}a_{jc}Vh_c.
\]

再做树卷积：

\[
\tilde z_j=z_j+\operatorname{TreeConv}(z)_j.
\]

输出：

\[
\delta_{root}=g_0(\tilde z_0),\qquad
\delta_j=g_j(\tilde z_j),\ j=1\ldots15.
\]

共享 `g_j` 加 joint embedding，防止 15 个独立大 MLP。

### P4.3 代码

- 新增 `semkine/kine_query.py`；
- 复用 `semkine/gnn.py` 的 MANO tree edge；
- `model/model.py` 增加 unique-path head；
- 不允许复制同一个 global feature 到 16 节点；
- config：`p4_kinequery16_posefree.yaml`。

### P4.4 前置 oracle

用 GT 2D/3D joint 或 GT mesh 投影，只在诊断中给每个 event 一个 soft part label，比较：

```text
global pool head
vs
GT-part pooled 16-node head
```

若 GT part oracle 对 per-finger RA 改善 `<1.5 mm`，说明局部归属不是当前瓶颈，P4 停止。

### P4.5 Gate

- oracle 先通过；
- 学习 query 的 attention entropy 不能全均匀，也不能单 cell collapse；
- silence query `j` 的主要影响应落在其关节/后代，cross-finger leakage 有上限；
- 参数数在 global head ±10% 或明确报告；
- 两 seed recursive 至少有一个超过 1.1 mm 门槛，否则作为 P1 的结构消融而非主贡献。

---

## P5：FARSE-Kine——快/慢层级的全异步稀疏手部记忆

### P5.1 研究假设

手部运动的时间尺度不一致：

- 腕部/整体平移需要较长上下文；
- 指尖和局部关节需要短延迟；
- 低事件率时需要慢状态保持；
- 高速局部运动时需要快速更新。

借用 FARSE-CNN 的 recurrent sparse hierarchy 与 compression，但只保留两层：

```text
fine 8 px / fast decay → finger evidence
coarse 24 px / slow decay → root/global evidence
```

### P5.2 递归与压缩

fine state：

\[
f_c^+=D_f f_c+\phi_f(e_i).
\]

每当 fine cell 累计信息超过阈值，压缩到 coarse parent：

\[
q_{p(c)}^+=D_q q_{p(c)}+C[f_c,\log(1+n_c),\Delta t_c].
\]

重要的是 compression 是**流式汇聚**，不是每 50 ms 重新把 fine feature 做 dense downsample。

查询时：

\[
z_j=\operatorname{Attn}(q_j,\{f_c\}\cup\{q_p\}).
\]

### P5.3 代码

- 新增 `semkine/farse_kine.py`；
- 复用 F0 stream state；
- 不导入完整 FARSE repo；
- 参数与 PC-AEGNN 对齐；
- config：`p5_farse_kine.yaml`。

### P5.4 Gate

- random chunk parity；
- compression 前后事件贡献守恒/可追踪；
- fine-only、coarse-only、two-level 三臂；
- two-level 单步必须优于或持平 fine-only；
- recursive 至少改善 `1.1 mm` 或在 2% 精度内 p95 latency/ops 改善 `>=25%`；
- 若只增加复杂度而无精度/效率价值，停止，不再加第三层。


---

## P6：CT-KineFlow——连续时间 Lie 运动场（高风险路线）

### P6.1 研究假设

事件没有天然帧边界。若固定 5/10/20/50 ms 的离散查询本身造成明显误差，应该直接估计连续时间状态：

\[
X(t)=X_{a}\boxplus \xi(t),
\qquad
\xi(t)=\sum_{m=1}^{M}B_m(t)c_m,
\]

其中 `B_m(t)` 是局部支撑的 cubic B-spline basis，`c_m∈R^{51}` 是 Lie tangent control coefficients。

事件 `i` 的轮廓残差：

\[
r_i(c)=n_i^T\left(u_i-\pi(X_i(X(t_i)))\right).
\]

控制量 Jacobian：

\[
\frac{\partial r_i}{\partial c_m}
=
\underbrace{\frac{\partial r_i}{\partial \xi(t_i)}}_{\text{现有 MANO/event Jacobian}}
B_m(t_i).
\]

因此可复用 `semkine/jacobian.py`，而不是重写 MANO 微分。

### P6.2 为什么先做 oracle

当前日志已经说明 5–50 ms 变化未必是主因，所以 P6 不能直接训练大模型。先用 GT trajectory/高频标签拟合 spline：

1. 只用可见 GT pose 拟合 `c_m`；
2. 在任意时刻重建 GT mesh；
3. 比较 piecewise-constant、constant-velocity、Lie spline；
4. 测 event residual、interpolation MPJPE 和 latency。

若 oracle 都不显著优于离散模型，学习版不会凭空有效。

### P6.3 最小模型

网络不输出每时刻完整 pose，而输出局部 control increment：

\[
\Delta c_k=g(M(E_{(t_{k-1},t_k]}),s_{k-1}).
\]

每次只保留最近 4 个 control knots，保证因果与常数内存。查询 `t_q` 只评估局部 basis。

### P6.4 代码

- 新增 `semkine/ct_kineflow.py`；
- 修改 `eval_track.py` 支持 arbitrary timestamp query；
- 复用 `lie.py`、`jacobian.py`；
- 临时 oracle 脚本放 `/tmp/eventhands_p6_oracle.py`；
- config：`p6_ct_kineflow.yaml`。

### P6.5 Gate

**Oracle Gate：** 至少满足一个：

- arbitrary-time RA 改善 `>=1.5 mm`；
- event contour residual 降低 `>=20%`；
- 同精度下 query latency/频率优势 `>=25%`。

**学习 Gate：**

- no future knot/event；
- chunk/query invariance；
- knot boundary 无跳变；
- 5/10/20/50 ms 与非整数 query 都稳定；
- 若 oracle 未过，P6 立即归档，不投入 GPU 长训练。

---

## P7：NeuMANO-FG Lite——学习事件归属，解析求解姿态

### P7.1 研究假设

直接 51D 回归把两个问题混在一起：

1. 哪些事件属于手、轮廓、哪根手指；
2. 给定这些观测，MANO 状态应如何改变。

仓库已经有正确的解析 Jacobian 和 GN/LM，因此新网络可以只学习：

- hand/background ownership；
- contour probability；
- normal/orientation；
- part distribution；
- residual reliability。

状态由解析因子图更新，形成“learned correspondence + analytic state update”。

### P7.2 事件因子

网络对事件 `e_i` 输出：

\[
o_i\in[0,1],\quad c_i\in[0,1],\quad
\pi_i(j),\quad n_i\in S^1,\quad w_i>0.
\]

在当前预测 `X^-` 上，KSSF 给出候选 surface/face。残差：

\[
r_i(X)=n_i^T\left(u_i-\pi_K(S_i(X))\right).
\]

优化目标：

\[
\min_{\delta}
\sum_i o_ic_iw_i\rho\big(r_i(X^-\boxplus\delta)\big)
+\|\delta\|_{\Lambda_P}^2
+\lambda_{lim}\mathcal L_{joint-limit}.
\]

一次或两次 LM：

\[
(J^TWJ+\Lambda_P+\lambda I)\delta=-J^TWr.
\]

不得超过 2 次迭代作为主路径，否则实时性和稳定性风险过大。

### P7.3 关键创新边界

这不是传统 model fitting 的重复，关键在于：

- event-native、异步持久 ownership memory；
- 解析 MANO posedirs-aware Jacobian；
- activity/Fisher 选择 active blocks；
- 只优化可观测关节；
- 与 learned direct observer 做信息融合。

### P7.4 前置 oracle

至少做三臂：

```text
O1 GT foreground + GT part + GT normal + GN
O2 GT foreground + KSSF part/normal + GN
O3 learned ownership + KSSF/learned normal + GN
```

只有 O1/O2 在相同初始误差下比 direct KEG 更新明显更好，才训练 O3。

### P7.5 代码

- 新增 `semkine/neumano_fg.py`；
- 复用 `semkine/gn.py`、`jacobian.py`、`kssf.py`；
- 修改 `model/model.py` 暴露 event factor head；
- config：`p7_neumano_fg.yaml`。

### P7.6 Gate

- oracle GN 对 5/10/20/40 mm 初始误差均有 basin；
- 正确观测时 residual 单调下降；错误 correspondence 的 kill test 不得无界爆炸；
- active block 之外的坐标严格不动；
- learned factors 的 ownership/normal 有独立指标；
- 两 seed recursive 至少改善 `1.1 mm`；
- 如果 correspondence oracle 都无优势，停止，不把 GN 作为装饰性模块。

---

## P8：Fisher-Triggered Event Query——按可观测信息而非固定窗口输出

### P8.1 研究假设

固定 50 ms 是 LNES 的历史选择，不是事件测量的物理规律。事件流中：

- 高速清晰轮廓可能 1–5 ms 已足够；
- 静止/遮挡 50 ms 也没有新信息；
- 不同手指的可观测性不同。

因此 query/packet 触发应由信息增量决定，而不是只由时钟决定。

### P8.2 信息累积

对于事件残差 Jacobian `J_i` 与噪声 `σ_i^2`：

\[
\Lambda(t)=\sum_{t_i\in(t_{last},t]}J_i^T\sigma_i^{-2}J_i.
\]

便宜版不每事件构建完整 51×51，而维护 17 块或对角代理：

\[
I_b(t)=\sum_i \|J_{i,b}\|_2^2/\sigma_i^2.
\]

触发条件之一：

\[
\Delta\log\det(\Lambda_P+\Lambda(t))\ge\tau_I,
\]

或：

\[
\max_b I_b(t)\ge\tau_b,
\]

同时有最小/最大等待：

\[
\Delta t\in[t_{min},t_{max}].
\]

### P8.3 不能把 rate 改变当精度提升

所有比较都做 matched-rate curve：

- 固定 5/10/20/50 ms；
- event-count threshold；
- contour-count threshold；
- Fisher threshold；
- 相同平均 query/s 下比较 RA；
- 相同 RA 下比较 query/s、p95 latency 和 operations。

### P8.4 代码

- 首先完成 F0 packetizer；
- 修改 `semkine/packetizer.py` 加 block-information accumulator；
- 修改 `eval_track.py` 支持 variable query times；
- 复用 `jacobian.py` 的 cheap block norm；
- 不新增复杂 scheduler framework；
- config：`p8_info_packetizer.yaml`。

### P8.5 Gate

- 事件精确守恒；
- trigger 只看过去；
- max wait 时即使零事件也允许 process query；
- 同 RA 下 query/s 或 p95 latency 降低 `>=25%`；或同 rate 下 RA 改善 `>=1.1 mm`；
- 若只通过降低 query 频率“看起来更平滑”，但 absolute drift 变差，则 FAIL。

---

## P9：EgoHand-Decouple——相机/背景运动与手运动的条件解耦

### P9.1 研究假设

只有在数据存在明显相机运动、动态背景或身体运动时，背景事件才会污染手部 tracking。固定相机场景上直接加 VO/SLAM 模块是不必要复杂化，因此必须先诊断。

### P9.2 零训练 contamination probe

构造以下统计：

- GT/高质量手 mask 内外事件率；
- 手外 dominant optical/event flow；
- 背景事件与 root update 的互信息/相关性；
- camera-moving subset 与 static-camera subset 的误差差；
- 背景事件随机删除对预测的影响；
- 只保留背景事件时的虚假更新幅度。

若静态相机为主且差异不显著，P9 不实施。

### P9.3 最小双运动模型

事件 ownership：

\[
\alpha_i=P(z_i=\text{hand}\mid e_i,M),
\qquad 1-\alpha_i=P(z_i=\text{background}).
\]

背景全局运动 `\theta_{ego}` 用低维 2D affine/homography 或 rotation-only proxy，禁止一开始上完整 SLAM：

\[
u_i^{bg}\approx W(u_i;\theta_{ego}).
\]

补偿后手事件：

\[
\tilde u_i=u_i-W(u_i;\hat\theta_{ego}),
\]

或者不 warp 坐标，只把背景运动 embedding 与 ownership 送给 pose-free memory。

总观测分解：

\[
E=E_{hand}\cup E_{bg},
\qquad
\mathcal L_{own}=\operatorname{BCE}(\alpha,\text{pseudo/GT mask})+
\lambda_{cons}\mathcal L_{motion-consistency}.
\]

### P9.4 机制来源

借用 event ego-motion、EN-SLAM、Moving Border Ownership、Dyn-HaMR/HaWoR 的**运动因子分离**思想；不把完整 VO/SLAM 或 world-coordinate optimizer 搬进当前单目手模型。

### P9.5 代码

- 新增 `semkine/ego_decouple.py`；
- 对 pose-free memory 增加 `ownership`/`ego embedding` 可选输入；
- 原 static-camera config 完全不变；
- config：`p9_ego_decouple.yaml`。

### P9.6 Gate

- contamination probe 必须先显示 moving-camera subset 至少比 static subset差 `>2 mm` 或背景消融产生明显变化；
- 背景-only 输入的更新幅度下降 `>=50%`；
- static subset 不退化超过 `0.5 mm`；
- moving subset 改善 `>=1.5 mm`；
- 未过诊断则停止，不能以“理论上更完整”为理由保留。

---

## P10：Event-Physics/Domain——事件生成物理与弱监督的附加路线

### P10.1 研究假设

自有数据和仿真数据的主要风险之一是：网络可能拟合事件率、对比度阈值、噪声与背景，而不是可迁移的运动几何。事件物理监督应作为 P1/P3/P4 的附加训练信号，而不是再造一个主干。

### P10.2 候选损失必须逐个验证

**A. Motion-compensated event contrast**

根据预测状态把事件 warp 到参考时刻，形成 IWE：

\[
I(u;X)=\sum_i p_i k\left(u-W(u_i,t_i;X)\right).
\]

对手区域最大化锐度、对背景不强制：

\[
\mathcal L_{IWE}=-\operatorname{Var}(I\odot M_{hand}).
\]

**B. KSSF contour likelihood**

\[
\mathcal L_{edge}=\sum_i \rho\left(
\operatorname{SDF}(u_i;X(t_i))
\right)\cdot w_i.
\]

**C. Event polarity consistency**

若有渲染亮度/teacher image，可用预测 motion 与图像梯度的符号约束；无可靠亮度时禁止伪造该 loss。

**D. Time-reversal consistency**

在没有未来泄漏的训练诊断中，把局部 event segment 反序并反转时间/运动标签，检查 inverse update：

\[
\delta(E^{rev})\approx -\operatorname{Ad}\,\delta(E).
\]

它仅作训练增强/一致性，不进入在线推理未来信息。

**E. Sensor-domain randomization**

随机化：

- contrast threshold proxy；
- event dropout/rate；
- refractory/dead pixel/hot pixel；
- timestamp jitter；
- polarity flip/swap；
- background event mixture。

必须保持几何与标签同步。

### P10.3 数据使用

- 自有单手数据：主监督；
- Ev2Hands-R/S：只在协议和 MANO/hand-count 适配后做外部泛化；
- EvHandPose 等有事件手标签数据：可做 3D joint/pose transfer，不伪造 mesh GT；
- RGB hand datasets/HaMeR teacher：只用于预训练 absolute prior，不进入 event-only test；
- v2e/仿真：用于覆盖极端速度、亮度和噪声，但必须在真实数据单独报告。

### P10.4 代码

- 新增 `semkine/event_physics.py`；
- 每个 loss 都有独立 config 开关和梯度单元测试；
- 一次只开一个 loss；
- 临时可视化写 `/tmp`；
- config：`p10_<loss_name>.yaml`。

### P10.5 Gate

- loss 对正确 pose 的值/梯度优于明确扰动 pose；
- 不允许 loss 在 all-zero output 下取得伪最优；
- 单一 loss 先做 500–1000 step scout；
- real val recursive RA 改善 `>=1.1 mm` 或 sim-to-real gap 统计显著收窄；
- 一旦损害 strict recursive，立即移除；
- 最终只保留最多 1–2 个互补 loss，禁止 loss soup。

---

# 第九部分：数据集、baseline 与评测设计

## 9.1 数据样本从独立窗口改为连续 chunk

新的训练 item 应是同一个 valid run 内的连续事件片段：

```text
SequenceChunk
  sequence_id
  run_id
  t0_us
  query_times_us[K+1]
  events_xypt (ragged over whole chunk)
  event_ptr_per_step[K+1]
  x0_51d
  target_51d[K]
  betas
  camera_K
  augmentation_state
```

要求：

- `x0_51d` 是 chunk/run 起点唯一条件；
- 后续 target 只用于 loss；
- 同一个 chunk 的相机几何增强必须一致；
- event dropout/hot pixel 可随时间变化，但要可复现；
- chunk boundary 随机化，防止模型记固定 50 ms 相位；
- DataLoader 不持有跨 batch 的模型 state，state 由每个 chunk 显式构建；
- 真在线 evaluator 则由 caller 跨调用持有 state。

## 9.2 必须保留的 baseline

| Baseline | 用途 |
|---|---|
| frozen `track_render51_domrand` | 历史最可靠闭环控制 |
| current LNES matched protocol | 当前公平精度控制 |
| raw_scan | 无图 raw-event control |
| sparse_cell | 简单稀疏 control |
| AEGNNLite | 证明 packet-wide cdist 的局限，仅作负对照 |
| current KEG hard | KSSF 条件化 control |
| current KEG halo | 当前最佳 KEG control |
| Euclidean vs Lie update | 更新几何消融 |
| constant delta-trust | 零参数 filter/gate control |
| P2 frozen IEKF | dynamics control |

## 9.3 外部数据集路线

### 主结果层

自有 EventHands 数据，维持现有训练/val/test 与 subject-disjoint 协议。必须给出：

- 采集设备与分辨率；
- event preprocessing；
- GT/伪 GT 来源；
- 每段长度、动作、光照、速度、相机运动；
- split 是否 subject-disjoint；
- 标签插值比例和不确定性；
- 所有许可边界。

### 外部验证层

根据可获得标注逐层验证：

1. **Ev2Hands-R/S**：双手任务可先单手 crop/可见手子集，不把双手 SOTA 与单手模型硬比；
2. **EvHandPose / EventHands 类数据**：报告 joint/pose transfer；
3. **EventEgoHands 或动态背景手数据**：验证 P9，不要求 mesh GT 时报告 segmentation/pose consistency；
4. **仿真 event hand**：验证 rate、噪声和任意 query 泛化；
5. 没有 mesh GT 的数据，不得把 pseudo label 当 measured mesh GT。

## 9.4 公开结果表建议

### 表 1：主精度

```text
Method | Input | True async | Causal | Reset | RA-MPJPE | MPJPE | MPVPE | P95
```

### 表 2：闭环稳定

```text
Method | 5ms | 10ms | 20ms | 50ms | g@6.6 | g@26 | max drift | failure rate
```

### 表 3：效率

```text
Method | events/s | query/s | p50/p95 latency | active ops/event | memory | params
```

### 表 4：机制消融

```text
Pose-free | KSSF | Lie filter | CI | rollout K | recursive RA | calibration
```

### 表 5：跨域

```text
Train domain | Test domain | low light | fast motion | low event rate | moving camera
```

## 9.5 SOTA claim 的最低证据

至少需要：

- 同一 test split、同一 MANO/joint convention；
- 三个独立 seed；
- 逐序列 paired bootstrap；
- 完整 recursive，而非 teacher-forced；
- 对 LNES/current KEG/最强公开 event hand baseline 的公平复现或官方结果；
- 真异步 efficiency claim 有真实 incremental runtime；
- 自有数据结果之外至少一个外部验证；
- 公开代码、config、最终 checkpoint 选择策略和失败日志。


---

# 第十部分：Codex 执行提示词

## 10.0 使用方法

下面包含一个共同前置合同、一个 F0 提示词和十个方案提示词。执行原则：

```text
每次只把“共同前置合同 + 当前一个 Prompt”交给 Codex。
当前 Prompt Gate 没过，不允许自动进入下一 Prompt。
十个 Plan 是竞争路线，不是十个模块都要装进最终模型。
```

---

## 共同前置合同：每个 Prompt 前必须附上

```text
你正在 Ubuntu 本地仓库 /data1/lyq/code/EventHands1 中工作。

总体任务：
在不破坏现有 EventHands/LNES/KEG 可复现性的前提下，研究单目事件相机的真异步、严格因果、连续递归手部 MANO mesh tracking。51D 语义固定为：
[translation 3, root axis-angle 3, 15 local joint axis-angle × 3]。
每个 valid run 只允许起点一次 GT/标注初始化；之后严格 prev <- pred，GT 只能进入 loss 和 metric。

开始前必须执行并记录：
cd /data1/lyq/code/EventHands1
pwd
git remote -v
git fetch origin --prune
git branch --show-current
git rev-parse HEAD
git status --short

分支规则：
1. 以 research/event-kinegraph-20260823 及当前本地更新为证据基线。
2. 若工作树有用户修改，先记录并保护，禁止 reset --hard、clean -fd、强制 checkout、覆盖未提交文件。
3. 不在 main/master 上直接改；为当前 Prompt 建独立 research 分支，若同名分支已存在则审计后复用。
4. 不自动 push，不 force push，不 rebase 用户未授权分支。
5. 默认不自动 commit；只有用户明确要求时才 commit。

必须先阅读：
- docs/EXPERIMENT_SYSTEMATIC_SUMMARY.md
- docs/EVENT_KINEGRAPH_MASTER_PLAN.md
- docs/FAILURE_AND_CLEANUP_LEDGER.md
- docs/debug_e55b_unroll_20260825.md（若存在）
- docs/semkine/FINAL_REPORT.md（若存在）
- docs/semkine/EXPERIMENT_LOG.md
- 当前 Prompt 涉及的源码、config、tests、trainer、evaluator

若以下本地 skill 存在，读取它们，但仅用于候选假设、反例、kill test 和机会图；不能把 skill 输出当实验事实：
- /data1/lyq/code/skill/creative-thinking-for-research
- /data1/lyq/code/skill/research-opportunity-graph-skill

强制代码审计：
- 用 rg/find 追踪真实调用链，禁止根据提示词猜文件名。
- 若本文文件名与本地真实代码不同，以本地代码为准，先写 symbol/file:line 映射。
- 优先扩展已有通用实现，禁止创建功能重复的第二套 Lie、MANO、filter、Jacobian、evaluator。
- 每个 config 新键必须加入显式 schema/白名单；未知或未读取的 key 要 fail-fast，禁止 silent no-op。

冻结的科研合同：
1. 不允许未来事件、未来标签或后续 GT 泄漏。
2. sequence/run 边界必须清空 event memory、hidden、packet buffer、filter、cache；betas 和 camera_K 不跨 sequence。
3. 零事件段合法；行为必须明确定义并测试。
4. 局部旋转使用 SO(3) Exp/Log；最终候选更新使用仓库已有 lie.retract_51d。Euclidean update 只能作为冻结 control。
5. 禁止 packet-wide torch.cdist(events, events) 成为候选主路径。
6. 禁止 N_max 全批事件 padding；保持 ragged ptr 或 active-cell state。
7. 没有 caller-owned persistent state、chunk-invariance 和 affected-update 测试，不得写“真异步”。
8. 不以 val_loss 选择 checkpoint；按固定 step grid 报告 strict recursive val 指标。
9. 不把不同 protocol 的 19.2576 mm 与 S20/S21/S22 数值混表。
10. 一次只改变一个研究变量；禁止把十个 Plan 一次堆进网络。
11. 不增加 Gate JSON 框架；一次性诊断脚本放 /tmp/eventhands_*，最终可复现工具才进入 tools/。
12. 不删除 checkpoint、日志或失败代码；失败结论追加到现有 Markdown ledger。
13. 不伪造测试、指标、显存、延迟或 PASS。
14. 首阶段禁止完整长训练，只允许 unit/smoke/tiny/scout；Gate 通过后才给出正式训练命令。

统一判据：
- 小于 1.1 mm 的跨训练差异不声称精度胜出，除非 paired bootstrap 95% CI 不跨 0。
- 精度在 2% 内时，真实 p95 latency、operations/event 或 query rate 至少改善 20–25% 才可声称效率胜出。
- 所有训练路线：机制 probe -> 单 seed scout -> 两 seed -> 三 seed final。
- 所有最终比较必须 strict recursive；teacher-forced 仅作诊断。

每个 Prompt 完成时必须输出：
1. start/end HEAD 与 branch；
2. dirty_before/after；
3. 阅读过的关键文件与真实调用链；
4. changed files 和每个 file:symbol 的作用；
5. 公式到代码的映射及 tensor shape；
6. 精确执行的 tests/commands 与 pass/fail 数；
7. baseline parity；
8. GPU/CPU、runtime、VRAM（实际测到才写）；
9. 所有指标，不能只报最好点；
10. Gate verdict：PASS / FAIL / BLOCKED；
11. 若 FAIL，说明被证伪的假设并停止，不自动进入下一 Prompt。
```

---

## Prompt 0：F0 流式合同、packetizer 与 Lie update 修复

```text
本阶段名称：F0_STREAM_CONTRACT。
目标不是提升精度，而是修复真流式研究必须满足的基础合同。禁止启动完整训练。

先审计并记录真实 file:line/symbol：
- semkine/packetizer.py::AdaptivePacketizer
- tests/test_s15_packetizer.py
- semkine/events.py::EventPacket/EventPacketBatch/collate_packets
- semkine/lie.py::retract_51d/local_coordinates_51d
- model/model.py 中 pred/delta/prev 的最终组合位置
- semkine/eval_track.py 的 strict recursive prev <- pred 路径
- config schema/loader

任务 A：修复 packetizer 跨调用事件守恒。
1. 不要继续用混合 List[np.ndarray] 表示 _buf。建立具名、类型明确的内部 buffer；优先放在 semkine/packetizer.py，不新增核心文件。
2. 每次 push 开始必须把旧 buffer 与新 slice 合并，再统一切包。
3. 明确定义 flush(end_t_us=None, emit_tail=True/False)。不得静默丢尾部。
4. 保持输入稳定时间顺序；同 timestamp 以输入顺序为 tie-break。
5. 增加计数器：n_input、n_emitted_events、n_buffered_events；必须满足精确守恒。
6. reset 清空所有 buffer、t0、计数器和统计。

任务 B：增加 caller-owned stream-state 最小合同。
1. 先不要实现新网络，只建立轻量 dataclass/interface，放入最合适的已有模块；若必须新增文件，最多一个 semkine/stream_state.py。
2. 接口至少包括 init/update/query/reset 所需字段：sequence_id、时间、prev pose、betas、K、event memory placeholder、optional filter state。
3. state 必须显式传入/返回；禁止 module-global mutable state。
4. sequence_id 不同或时间倒退必须 fail-fast。

任务 C：加入 Lie update candidate，同时保留 Euclidean control。
1. config 增加 MODEL.UPDATE_RULE: euclidean_control | lie_retraction。
2. euclidean_control 的旧 config 与旧 checkpoint 数值行为保持不变。
3. lie_retraction 调用仓库已有 semkine.lie.retract_51d；不要重写 Exp/Log。
4. zero delta 必须严格 identity；小角一阶一致；大旋转无 NaN。

先写 tests，再写实现。至少包含：
- two_push_equals_one_push
- random_chunking_preserves_every_event（固定 seed，2–50 chunks）
- flush_and_reset_contract
- same_timestamp_stable_order
- sequence_reset_no_leakage
- time_regression_rejected
- zero_delta_lie_identity
- small_delta_first_order_match
- repeated_large_rotation_finite
- euclidean_control_parity

测试要求：
- 运行现有 packetizer/events/lie/eval 相关 tests；
- 新增 tests 全部通过；
- 如完整测试环境受 CUDA/MANO 依赖阻塞，至少 CPU 合同测试必须真实通过，并准确报告阻塞项；
- 只允许最多 100 step smoke，默认不训练。

F0 Gate：
PASS 需要：
- 事件计数精确守恒；
- random chunk 与 full stream 输出逐包相同；
- no sequence leakage；
- Lie tests 通过；
- Euclidean control parity 通过；
- 没有 silent config key。
否则 FAIL/BLOCKED，并停止。

最终把 F0 结论追加到 docs/semkine/EXPERIMENT_LOG.md；若发现历史结果可能被 packetizer bug 影响，追加到 docs/FAILURE_AND_CLEANUP_LEDGER.md，但不要篡改旧记录。
```


---

## Prompt 1：P1 AERO-Hand 主方案

```text
本阶段名称：P1_AERO_HAND_DUAL_OBSERVER。
独立研究分支建议：research/p1-aero-hand-20260826。

总假设：
当前 KEG 单步接近 LNES、严格递归却多约 10 mm，主因是 KSSF/route 让观测依赖 prev pose，上一帧误差既进入状态，又进入事件—手部关联。构建一个完全不读取 prev pose 的持久异步绝对观察器，再把它与现有 KSSF 局部创新在 Lie error-state 信息空间融合，应降低 closed-loop error gain，同时保留局部手指精度。

本 Prompt 必须分 P1-A/P1-B/P1-C 三个小阶段执行；前一 Gate 未通过不得进入后一阶段。本轮默认只完成 P1-A 和 P1-B scout，不自动启动最终训练。

一、先审计并输出真实调用链
必须阅读/定位：
- semkine/events.py
- semkine/encoder.py
- semkine/frontends.py
- semkine/keg.py，特别是 KSSF query、routing、node stats、readout
- semkine/kssf.py
- semkine/gnn.py
- semkine/lie.py
- semkine/filter.py
- semkine/jacobian.py
- semkine/mainline.py
- model/model.py 的 raw-event forward、PREDICT_DELTA、unroll、loss
- semkine/dataset.py、train.py、eval_track.py
- configs/semkine/s20/s21/s22/s23 相关 configs
- tests/test_s18_keg.py、test_s12_filter.py、test_mainline.py

把真实链写成：
raw event -> dataset/collate -> current frontend -> KSSF/KEG -> delta/readout -> state update -> MANO -> recursive evaluator。
明确列出哪些 tensor 当前依赖 prev_state，哪些不依赖。

二、P1-A：实现 pose-free persistent async memory 合同
最多新增两个核心文件：
1. semkine/async_memory.py
2. semkine/aero.py
不要再创建平行的 lie/filter/jacobian。

A.1 数据/状态结构
实现 caller-owned：
- PoseFreeMemoryState
- AEROStreamState
状态至少包含：
  sequence_id
  last_event_t_us
  last_query_t_us
  per-scale cell feature
  per-scale last_update_t_us
  active cell mask/index
  previous posterior pose
  optional velocity/covariance
  betas/camera_K
所有状态显式传入/返回，不在 module 属性里偷偷缓存 sequence-specific tensor。

A.2 两尺度 active-cell memory
默认：cell sizes 8 和 24；hidden 48 或 64；4 个正 decay rates。
事件更新公式：
  m_c(t_i+) = exp(-softplus(lambda) * dt) * m_c(t_i-) + B phi(e_i)
事件 token至少使用：
  x_norm, y_norm, polarity_sign, log_dt_ie, same/opp SAE 或仓库已有稳定 token
但 token 计算不能读取 prev pose/KSSF/render。
只更新命中 cell；禁止 packet-wide event-event cdist；禁止每次 query 重放历史事件。

A.3 query
query(t_q) 对 active cells做 lazy decay并输出 ragged/packed cell features：
  cell_xy, scale_id, feature, age, count/support
不得分配 dense HxW feature map作为主计算；为了测试可构建同步 reference，但 reference 不进入候选 runtime。

A.4 必写合同测试
先写以下 tests，再实现：
- full_stream_equals_random_chunks
- full_stream_equals_event_by_event
- same_prefix_same_state
- query_does_not_mutate_past_evidence_except_defined_lazy_decay
- pose_free_output_independent_of_prev_state
- reset_clears_all_scales
- no_sequence_cross_contamination
- zero_event_query_decay_only
- touched_cells_bounded_per_event
- no_cdist_in_candidate_call_graph（静态 rg/monkeypatch kill test）
容差：float32 max_abs <=1e-6 或给出数值合理的更严密容差；不能只比较均值。

P1-A Gate：
- 所有合同测试通过；
- memory footprint 对流长度近似常数，仅与 active grid上界有关；
- operations/event 不依赖历史事件总数；
- prev_state 随机改变不改变 pose-free memory/readout输入；
- 若不通过，FAIL并停止。

三、P1-B：16 joint query absolute observer
在 semkine/aero.py 实现：
- root + 15 MANO canonical joint queries；
- query identity来自 joint index、finger id、tree depth/canonical metadata，不来自上一帧投影；
- 对 active cell features做一层轻量 cross-attention；
- 一层 MANO tree message passing，优先复用 semkine/gnn.py 的 edges/TreeConv；
- shared joint head + joint embedding；root head独立；
- 输出 absolute 51D state X_A 或相对于固定 canonical/reference的状态，然后通过 lie.local_coordinates_51d 转成相对 process prior的 mu_A；
- 输出 17 个 3x3 SPD precision block，使用 Cholesky参数化，不能输出未约束矩阵。

严格禁止：
- 把同一 global pooled feature复制16次；
- 用 prev pose/KSSF决定 query-cell hard assignment；
- 用 GT part label进入推理；
- 先 dense raster再ResNet作为 candidate；
- 一个巨大 transformer。

先做两个零训练/小训练 probe：
B.1 GT-part oracle：只作为诊断，把事件按GT/高质量投影 part聚合，比较 global pool vs 16 part pool。脚本放 /tmp/eventhands_p1_part_oracle.py，不提交。
B.2 learned query scout：最多1000 steps、1 seed，固定其余协议。

损失：
- MANO joint/vertex loss复用现有定义；
- SO(3) geodesic/root translation；
- precision Gaussian NLL：0.5*r^T Lambda r - 0.5 logdet Lambda；
- weak precision range prior；
- 不要同时加入P10物理loss。

P1-B Gate：
- GT-part oracle per-finger/overall至少显示结构价值；若改善<1.5 mm，记录FAIL，停止把query当主创新；
- learned absolute observer teacher-forced RA不得比current KEG差>1.5 mm；
- 改变prev_state时X_A保持不变；
- query pairwise attention不能全部相同：报告attention cosine/entropy；
- precision SPD、finite；预测std与realized block error Spearman至少0.20（scout可用趋势门槛）；
- zero event不能凭空制造高precision；
- 参数新增目标<2M，报告真实数值。

四、P1-C：接入KSSF局部观察器和Lie信息融合
只有P1-A/B PASS才做。

C.1 process prior
复用 semkine/filter.py 的 Lie process；不要复制代码。明确root metric采用R3 x SO(3)还是SE(3)，与现有loss/anchor保持一致。

C.2 local observer
复用current KEG/KSSF/halo作为mu_C来源。必要时给keg暴露observation接口，但保持旧forward control数值不变。
Lambda_C优先来自现有analytic Fisher；若尺度不匹配，只允许一个单调calibration scale/floor/ceil，不能自由 learned gate。

C.3 block covariance intersection
17个block分别在固定omega grid {0, .125, ..., 1}上选使fused covariance logdet最小的omega。
如果两个branch disagreement Mahalanobis d超过阈值，用Huber信息缩放；冲突时回退process prior。
实现必须保证SPD、对称、finite。

C.4 state update
最终：
  delta = solve(Lambda_P + Lambda_F, eta_F)
  X_plus = lie.retract_51d(X_minus, delta)
禁止 pred=prev+delta candidate路径。

C.5 rollout数据与训练
扩展现有dataset，不另造数据复制：
- 同valid run连续chunk；
- query K=4起步；
- only x0作为输入初值；
- k>0严格使用模型posterior；
- 同chunk geometric augmentation一致；
- TBPTT detach按config明确；
- 后续可K=8/16，当前scout只K=4。
新增config：configs/semkine/p1_aero_hand_k4.yaml。

C.6 contraction诊断
复用现有g-curve/echo诊断；实现随机tangent perturbation有限差分或JVP，报告：
- g@small
- g@26mm附近
- fixed-point error
- echo excess
不要一开始把contract loss权重设很大；先测后决定。

P1-C单seed scout Gate：
- strict recursive RA <=26 mm；
- g-slope <=0.24；
- 对matched KEG显著改善且无sequence divergence；
- 5/10/20/50ms全部finite；
- CI优于naive average或learned sigmoid control至少一个稳定性指标；
- precision/NIS不明显过度自信。
未过则停止，不进入2/3 seed。

五、必须做的控制/消融config
- p1_abs_only
- p1_kssf_only_control
- p1_naive_average
- p1_learned_sigmoid_negative_control
- p1_block_ci
- p1_block_ci_no_process
- p1_block_ci_euclidean_control
- p1_block_ci_lie
- p1_k1_teacher_forced
- p1_k4_rollout
每个config只能改变所命名变量。

六、训练命令策略
不要猜conda环境/设备。先从现有README、脚本、history中找真实可运行命令并复用。
先给出：
- CPU/unit命令
- 1 batch forward/backward smoke
- <=1000 step scout命令
只有P1-C PASS后，才输出K=8/16和2/3 seed命令，不自动执行。

七、结果记录
把每个子阶段的假设、命令、全量网格点、指标、失败原因追加到docs/semkine/EXPERIMENT_LOG.md。
任何结构被否决，追加到FAILURE_AND_CLEANUP_LEDGER.md，禁止删除旧实现。
最终明确：当前证据是否只支持“best hypothesis”，还是已经支持“beats matched LNES”。
```


---

## Prompt 2：P2 Lie-IEKF-Net 控制臂

```text
本阶段名称：P2_LIE_IEKF_CONTROL。
独立分支建议：research/p2-lie-iekf-20260826。
目标：不训练或仅标定极少参数，使用仓库已有Lie filter/process/Jacobian验证“流形动力学与证据加权能解释多少递归差距”。这不是重写EKF，也不是最终主网络。

先审计：
- semkine/filter.py全部数学与dtype
- semkine/mainline.py如何调用filter/router/anchor
- semkine/jacobian.py的Lambda来源和量纲
- semkine/eval_track.py state_hook/active_policy路径
- model/model.py输出语义
- tests/test_s12_filter.py、test_mainline.py
- 当前LNES、KEG hard、KEG halo的selected checkpoints/configs

先做严格parity：
1. use_filter=false/use_router=false时，SemKinePolicy输出与raw network逐位一致。
2. frozen checkpoint与事件流只forward一次；所有filter arms共享相同网络输出，做paired inference。
3. 不允许用GT逐帧调gain、Q、R或trigger。

实现/配置五臂，每次只变一个变量：
F0 raw network output
F1 Lie retraction only
F2 constant-gain Lie filter
F3 Fisher-weighted Lie filter
F4 F3 + velocity process
F5 F4 + calibrated process noise/floor/ceil

公式必须映射到已有实现：
X- = X+_{k-1} boxplus(dt*v)
v- = exp(-gamma*dt)*v
P- = Phi P+ Phi^T + Q(dt)
nu = Log((X-)^-1 X_net)
R = (alpha*Lambda + beta*I)^-1
K = Pxx(Pxx+R)^-1
X+ = X- boxplus K*nu
P+使用Joseph form。

如果LNES没有解析Lambda：
- 只允许event count、contour count、rate等单调标定control；
- 不能给LNES一个更强learned uncertainty head而让比较失真。

标定策略：
- 只在val_core做小网格，固定预注册范围；
- 报告每个网格点，不隐藏best-of-N；
- 选定后锁死，在test不调参；
- 至少报告gamma、sigma_a、info_divisor、floor、ceil。

必须测试：
- P始终对称PSD；
- zero measurement只predict；
- dt变化时Q尺度正确；
- sequence reset；
- root/joint local coordinates一致；
- float64 filter与float32 network边界正确；
- no-GT leakage静态/运行时kill test。

评测：
- LNES和KEG各自做paired arms；
- strict recursive 5/10/20/50ms；
- RA/abs/MPVPE/jitter/long-drift；
- NIS、coverage、trace、gain统计；
- 每序列paired difference。

P2 Gate：
PASS进入P1融合研究需要：
- 至少一个frozen arm比对应raw checkpoint改善>=2.0mm；
- 不以明显增加jitter或abs error换RA；
- 5/10/20/50ms无发散；
- NIS/coverage不是完全过度自信；
- 结果在至少两个init-noise seed方向一致。
若最佳改善<1.1mm，判定“filter不是主要单独解法”，停止继续扫参；保留为论文control。

不要新增核心文件，除非真实调用链无法用薄adapter接入；若新增，最多1个且解释为何不能扩展mainline.py。
最终把全量表和裁决追加到EXPERIMENT_LOG.md。
```

---

## Prompt 3：P3 Persistent-Cell AEGNN

```text
本阶段名称：P3_PERSISTENT_CELL_AEGNN。
独立分支建议：research/p3-pc-aegnn-20260826。
目标：实现AEGNN真正有价值的“affected-node-only update”，但不用事件级全局kNN，不用packet-wide cdist。该阶段先验证异步等价与单步表示，不承诺解决闭环；若单独frontend不解决递归，必须回到P1双观察器。

先审计：
- semkine/frontends.py::AEGNNLite/SASTLite/FARSELite
- semkine/encoder.py raw_scan/sparse_cell
- semkine/events.py ragged contract
- semkine/keg.py输入输出接口
- 当前测试中对AEGNN/cdist/内存的记录
- AEGNN论文官方机制：同步图层与异步受影响节点更新的等价条件

新增最多一个核心文件：semkine/persistent_cell_gnn.py。
优先复用F0/P1已有stream state；若P1未执行，在本文件内定义最小state但不得复制packetizer/lie。

图设计：
- scale 8px与24px，可先只做8px单尺度control；
- 节点是固定cell，不是每个event；
- 每尺度8-neighbor固定边；跨尺度child-parent边可作为第二变量，首臂关闭；
- 每节点缓存h^(0..L)与last_update_time；
- event token只写入源cell h0；
- affected set A0={source}; A_l=A_{l-1} union neighbors(A_{l-1})；
- 只重算A_l内节点，其他缓存不变；
- max layers=2，hidden=64起步。

必须同时实现一个只用于测试的synchronous_reference：
- 对当前全部节点/边full recompute；
- 相同权重、相同event顺序、相同decay；
- candidate runtime不调用reference。

先写tests：
- one_event_sync_async_parity
- random_stream_sync_async_parity
- random_chunk_async_parity
- only_affected_nodes_mutate
- operation_count_bounded
- no_event_event_cdist
- reset_and_sequence_isolation
- zero_event_decay
- backward_gradient_parity（小图）
- very_large_packet_no_quadratic_memory

集成臂：
P3-0 current raw_scan
P3-1 current sparse_cell
P3-2 PC-AEGNN single scale
P3-3 PC-AEGNN two scale（只有P3-2 PASS才做）
P3-4 P3-2作为pose-free absolute frontend
P3-5 P3-2接current KSSF KEG
不要把P3-4和P3-5同时融合，那属于P1。

训练：
- 先one-batch overfit/1000-step scout；
- 参数数与current raw/Keg head对齐，报告；
- 训练数据、loss、head、update rule与control一致；
- 同步训练/异步推理前必须证明数值等价；若训练使用异步实现，也要报告速度。

Gate：
1. sync/async max error<=1e-5；
2. ops/event近似常数，不随历史长度增长；
3. 大包不出现O(N^2)内存；
4. teacher-forced RA不差current KEG超过0.5mm；
5. strict recursive若仍差matched LNES>5mm，明确结论“frontend replacement不解决echo”，停止加深GNN；
6. 若精度2%内且实际operations或p95 latency改善>=25%，可作为效率贡献保留。

输出每层affected-node数量、缓存内存、events/s、query latency；未真实测硬件时不要声称能耗。
```

---

## Prompt 4：P4 KineQuery16

```text
本阶段名称：P4_KINEQUERY16。
独立分支建议：research/p4-kinequery16-20260826。
目标：验证global pooling是否掩盖局部手指证据。使用root+15 MANO canonical queries和一层树消息传递；query-event关联不得由prev pose/KSSF hard routing决定。

先审计：
- semkine/encoder.py的pool_packets/global feature
- semkine/keg.py node readout和当前16节点语义
- semkine/gnn.py::KinematicGNN/TreeConv/mano_edges
- model/model.py active head、joint decoders
- MANO parents/joint ordering
- 每指/每关节现有metric工具

先做零训练GT-part oracle，脚本放/tmp/eventhands_p4_part_oracle.py：
1. 用GT mesh/pose投影或现有可靠label为event生成soft MANO part responsibility，仅用于oracle。
2. 使用相同event token和相近参数量，比较global pool head vs 16 part-pooled head。
3. 报告overall RA、per-finger、distal joints、fast-motion/occlusion桶。
4. 不把oracle label写入正式dataset或推理。

Oracle Gate：
- overall或关键local finger至少改善1.5mm；
- 若没有，P4 FAIL，停止训练learned query。

若通过，新增最多一个核心文件semkine/kine_query.py：
- 16 learnable canonical query embeddings；
- joint index、finger id、tree depth等固定metadata；
- 对active cells/raw tokens做一层cross-attention；
- masking只基于有效cell/event和因果时间，不基于GT/prev projection；
- 一层TreeConv，禁止>2层；
- shared joint output head + joint embedding；root独立6D head；
- 支持absolute pose-free版和local innovation版，但本次只训练一个：优先pose-free absolute。

严禁：
- 把一个global featureexpand到16节点再称joint query；
- 15个独立大MLP导致参数暴涨；
- 双向使用未来query/state；
- 用prev_state作为attention key位置而不做pose-free control。

必写tests/diagnostics：
- query permutation/joint-order correctness
- root/joint output slices正确
- silence query j主要影响对应joint/descendants
- cross-finger leakage matrix
- attention entropy与pairwise cosine
- empty event行为
- causal mask
- parameter-count parity
- pose-free independence from prev_state

训练臂：
Q0 global pool control
Q1 16 query no tree
Q2 16 query + one tree layer（主候选）
Q3 Q2 + prev_state in final head only
Q4 Q2 + prev-conditioned attention（负对照，只有必要时）
每个臂单独config，不能一起扫隐藏维度。

Gate：
- Q2 teacher-forced/per-finger优于Q0且没有整体退化；
- 两seedstrict recursive改善>=1.1mm或paired CI<0；
- attention不是全部相同/全均匀/单cell collapse；
- 参数在control±10%或明确做容量匹配；
- 若Q4更好但g-slope变坏，支持“prev-conditioned association产生echo”，不得选Q4为主线。

最终把oracle、learned query、leakage/attention图和Gate写入日志。
```


---

## Prompt 5：P5 FARSE-Kine 两层快慢异步记忆

```text
本阶段名称：P5_FARSE_KINE_HIERARCHY。
独立分支建议：research/p5-farse-kine-20260826。
目标：借用FARSE-CNN的recurrent sparse hierarchy/compression思想，做手部专用的fine-fast finger memory与coarse-slow root memory。不要导入或照搬完整检测backbone，不加第三层。

先审计：
- semkine/frontends.py::FARSELite是否只是wrapper/近似
- semkine/encoder.py sparse_cell/raw_scan
- P3/P1若存在的active-cell state
- 当前root/finger每组误差与事件支持统计
- 数据中窗口事件数、速度、低事件率分布

先做机制诊断：
1. 按每步GT或当前预测运动，把误差分为root translation/root rotation/每指。
2. 测不同历史长度2/5/10/20/50/100ms对各组单步oracle/readout的影响。
3. 只有root/global明显受益于长上下文且finger受益于短上下文，才进入实现。

新增最多一个核心文件semkine/farse_kine.py；优先复用active-cell基础。

结构：
- fine scale: 8px, hidden 48/64, fast positive decay bank；
- coarse scale: 24px, hidden 64, slow positive decay bank；
- event先更新fine命中cell；
- 当fine cell累计count/support或时间达到阈值，执行一次compression到parent coarse cell；
- compression输入为fine feature、log count、age/support；
- compression后fine state是保留、部分重置还是残差保留，必须作为明确单变量，首版选择论文机制最简单的一种并记录；
- query同时读取fine/coarse active states；root query偏coarse，finger query可读两者，但不做hard mask。

公式到代码：
f_c+ = D_f(dt) f_c + phi(e)
q_parent+ = D_q(dt) q_parent + C([f_c, log(1+n_c), age_c])
z_j = Attn(q_joint_j, fine union coarse)

必写tests：
- full/random-chunk parity
- compression_boundary_parity
- no_event_decay
- reset/sequence isolation
- coarse parent index correctness
- contribution trace：每个输入event要么在fine buffer，要么已进入coarse，不得丢失
- bounded memory/ops
- no dense downsample in candidate
- gradients through compression

实验臂：
H0 PC-AEGNN/fine-only control
H1 coarse-only
H2 fine+coarse，固定compression
H3 H2不同decay separation（只有H2过Gate才做）
不要同时改query head/CI/filter。

Gate：
- 时间尺度诊断先成立；
- H2单步不差H0超过0.5mm；
- root与distal finger至少一个维度显示互补，而不是同一feature重复；
- recursive改善>=1.1mm，或精度在2%内operations/p95 latency改善>=25%；
- 参数目标<2M新增；
- 如果H2只增加计算、没有精度或效率收益，FAIL并停止，不增加第三层/更多compression。

记录fine/coarse active counts、compression rate、state memory、每关节误差和query latency。
```

---

## Prompt 6：P6 CT-KineFlow 连续时间 Lie 运动场

```text
本阶段名称：P6_CT_KINEFLOW_ORACLE_FIRST。
独立分支建议：research/p6-ct-kineflow-20260826。
目标：先证伪/证实“固定离散窗口是主瓶颈”。必须先做GT trajectory oracle；oracle不过，禁止实现学习网络和长训练。

先审计：
- 原始GT时间分辨率、插值方式、valid_runs
- semkine/dataset.py中ms索引与sub-ms timestamps
- eval_track.py的固定step查询
- lie.py retraction/local coordinates
- jacobian.py event residual/Jacobian
- 当前5/10/20/50ms实验日志

阶段6-A：连续时间oracle
临时脚本/tmp/eventhands_p6_ct_oracle.py，不提交核心实现。
对每个valid run构造：
O0 piecewise constant
O1 constant velocity on Lie tangent
O2 causal cubic B-spline/Lie local spline，最多最近4个knots
O3 noncausal spline只作为upper bound，必须明确不能进入主方法

状态：X(t)=X_a boxplus sum_m B_m(t)c_m。
使用真实GT标签拟合/插值，报告：
- 标签时刻reconstruction error
- 标签间任意时刻（若有高频GT/渲染GT）
- event contour residual或KSSF residual
- knot boundary continuity
- root/joint/per-finger
- 5/10/20/50ms与随机query times

Oracle Gate：至少满足一个：
- causal O2相对O1 RA改善>=1.5mm；
- event residual降低>=20%；
- 同精度下query/representation效率有明确>=25%优势。
若不满足：写FAIL，归档P6，停止。

阶段6-B：最小学习版（仅oracle PASS后）
新增最多一个核心文件semkine/ct_kineflow.py。
- caller-owned最近4 knots/control coefficients；
- 网络只预测新control increment，不预测未来knots；
- 每个event/query只依赖t<=t_query；
- 使用lie.retract_51d组合；
- 复用jacobian：dr/dc_m = dr/dxi(t_i)*B_m(t_i)；
- query arbitrary timestamp；
- 常数内存，旧knots边缘化/丢弃规则明确。

必写tests：
- no_future_basis_support
- causal_vs_offline_prefix parity
- knot_boundary_C0/C1 continuity（按所选spline定义）
- random_query_order rejection或明确支持
- random event chunk parity
- reset
- constant-pose/constant-velocity exact cases
- finite gradients

训练只做<=1000 step scout，control为同一frontend/参数量的离散Lie update。
Gate：
- arbitrary-time精度或event residual达到oracle方向；
- strict recursive不差control；
- 5/10/20/50ms都稳定；
- 非整数query可运行；
- 若学习版无法接近oracle收益50%，停止，不扩大神经网络。
```

---

## Prompt 7：P7 NeuMANO-FG Lite 学习因子 + 解析GN

```text
本阶段名称：P7_NEUMANO_FACTOR_GRAPH_LITE。
独立分支建议：research/p7-neumano-fg-20260826。
目标：把“事件归属/可靠性学习”和“MANO状态求解”拆开。网络预测event factor，状态更新复用现有KSSF、analytic Jacobian、active-set GN/LM。先做GT correspondence oracle，oracle不过不训练。

先审计：
- semkine/kssf.py fields/query/face/bary/SDF/normal
- semkine/jacobian.py posedirs-aware analytic Jacobian
- semkine/gn.py LM solve、active mask、iteration
- semkine/router.py Fisher/active groups
- current KEG event routing与KSSF silencing结果
- 是否有GT mesh/part/foreground/normal可生成oracle label

阶段7-A：oracle
脚本/tmp/eventhands_p7_factor_oracle.py。
构造：
O0 current direct KEG update
O1 GT foreground+GT part+GT contour normal -> GN
O2 GT foreground+KSSF part/normal -> GN
O3 KSSF only current state -> GN
在5/10/20/40mm初始扰动、不同手指/遮挡/事件率上测：
- 一步RA改善
- residual before/after
- convergence basin
- inactive coordinate变化
- failure/divergence rate

Oracle Gate：
- O1/O2至少一个在多级扰动上显著优于O0，且residual/pose大多数同时改善；
- 若没有，FAIL，停止，不把GN当装饰。

阶段7-B：factor head
新增最多一个核心文件semkine/neumano_fg.py。
每个event/active-cell输出：
- foreground ownership o_i
- contour probability c_i
- part distribution pi_i over root+15
- 2D unit normal n_i（归一化）
- positive reliability w_i（softplus/clamp）
输出不能直接绕过GN回归完整51D，除非作为明确direct-head control。

状态求解：
min_delta sum_i o_i*c_i*w_i*rho(r_i(X- boxplus delta))
          + delta^T Lambda_P delta
          + optional joint-limit prior
最多1–2次LM；active groups来自Fisher/router；inactive坐标严格为0。
使用仓库lie.retract_51d。

鲁棒性：
- Huber/Tukey选择一个，首版固定；
- correspondence不确定时降低w，不允许负权；
- LM damping有固定预注册小网格，报告全量；
- 求解失败返回process prior/直接observer，不返回NaN。

必写tests：
- analytic/autograd/finite-difference existing tests仍过
- factor shapes/ranges/normal unit norm
- zero weights -> zero measurement update
- inactive blocks unchanged
- correct synthetic residual decreases
- adversarial wrong correspondence bounded
- SPD normal equations/damped fallback
- max two iterations
- no GT in inference
- no future events

训练阶段每次只加一个监督：
F1 foreground only
F2 + contour
F3 + part
F4 + normal/reliability
F5 full factor+GN
不得一次开全后无法归因。

Gate：
- factor独立指标优于简单KSSF heuristic；
- GN success rate高、无爆炸；
- 两seedstrict recursive改善>=1.1mm或paired CI<0；
- p95 latency可接受，报告GN占比；
- 若learned correspondence让GN比O2下降过大且无法用简单监督修复，停止，不增加大Transformer。
```


---

## Prompt 8：P8 Fisher-Triggered Adaptive Query

```text
本阶段名称：P8_INFORMATION_TRIGGERED_QUERY。
独立分支建议：research/p8-info-query-20260826。
目标：修复F0 packetizer后，以事件对MANO状态的可观测信息触发query，而不是固定50ms。主贡献是精度—延迟—更新率Pareto，不允许用降低query频率掩盖漂移。

前置条件：F0 PASS。若packetizer事件守恒未通过，本Prompt直接BLOCKED。

先审计：
- semkine/packetizer.py current proxy/counters/buffer
- semkine/jacobian.py Fisher/row/block norm
- semkine/router.py group information
- eval_track.py固定step assumptions
- filter.py zero-measurement process propagation
- 当前5/10/20/50ms结果、事件率分布

实现原则：
1. 首版不每event构造完整51x51矩阵；维护17个block scalar/3x3 proxy。
2. 对event i，累积I_b += ||J_i,b||^2 / sigma_i^2，或复用已有cheap Fisher block。
3. trigger满足：
   - dt >= min_dt；且
   - block information、delta-logdet或support达到threshold；
   - 或dt >= max_dt强制process/query。
4. zero-event到max_dt时允许无measurement process step。
5. 所有事件只消费一次；trigger后清空本packet accumulator但不清空长期async memory。
6. trigger只看过去，不看GT/当前真实error。

配置：
PACKETIZER.MODE: fixed_time | event_count | contour_count | fisher_block
MIN_DT_US
MAX_DT_US
INFO_THRESHOLD
MIN_GROUPS
所有key fail-fast。

测试先行：
- event_conservation_across_variable_queries
- trigger_prefix_causality
- same_stream_random_chunk_same_query_times
- max_wait_zero_event
- reset
- threshold_monotonicity
- no_double_consumption
- long_term_memory_not_reset_on_query
- fixed_time_control_parity

实验必须画matched-rate/Pareto曲线：
A fixed 5/10/20/50ms
B event count thresholds
C contour count thresholds
D Fisher thresholds
对每个点报告：
- avg/p50/p95 query interval
- queries/s
- events/query
- RA/abs/MPVPE/jitter/max drift
- p50/p95 latency和operations
- low/high event-rate buckets

不能只选择最好的一个threshold；预注册小网格并报告全部。

P8 Gate：
PASS需至少一个：
- 在相同平均query/s下RA改善>=1.1mm；
- 在相同RA±2%下query/s、operations或p95 latency降低>=25%。
同时：
- max drift/failure不能变坏；
- 低事件率段不能因长时间不更新而失控；
- fixed-time control parity通过。
若只让轨迹更平滑但absolute/long drift变差，FAIL。

不要新增核心文件，优先扩展packetizer.py/eval_track.py；若必须新增信息accumulator，最多一个小模块且解释原因。
```

---

## Prompt 9：P9 EgoHand-Decouple

```text
本阶段名称：P9_EGO_HAND_DECOUPLE_DIAGNOSTIC_FIRST。
独立分支建议：research/p9-ego-decouple-20260826。
目标：只有在动态相机/背景事件确实污染手tracking时，才加入最小运动分解。禁止直接搬完整VO/SLAM。

阶段9-A只做零训练contamination probe，不改主模型。
先审计：
- 数据集中相机是否固定、是否有IMU/camera pose、背景类型
- event coordinates、hand mask/mesh projection可用性
- sequence metadata和category
- current model在static/moving/background复杂度桶的误差
- 是否存在EventEgoHands/自采移动相机外部数据

临时脚本/tmp/eventhands_p9_contamination.py，输出每序列：
- hand-mask内外events/s
- background event fraction
- 背景dominant motion proxy
- background-only输入产生的root/joint update norm
- 随机删除背景事件后的预测变化
- static vs moving-camera RA/abs/g-slope差异
- 背景事件与root delta相关性

诊断Gate：至少满足一个：
- moving-camera/complex-background subset比static差>2mm；
- background-only产生明显虚假更新；
- 删除/补偿背景事件显著改善。
若不满足，P9 FAIL并停止，记录“不必要复杂度”。

阶段9-B最小实现（仅诊断PASS）：
新增最多一个核心文件semkine/ego_decouple.py。
模型只做：
- event/cell hand ownership alpha；
- 低维background motion theta_ego，首版2D affine或rotation-only proxy，不能上完整SLAM；
- pose-free memory输入增加ownership或ego embedding；
- static-camera config旁路保持完全一致。

两种候选只选一个先做：
E1 soft ownership weighting，不warp events；
E2 低维background compensation后再写入memory。
不要E1/E2同时做。

监督来源按优先级：
1. GT/mesh projected hand mask；
2. RGB teacher/SAM等离线pseudo mask，但标注为pseudo；
3. motion consistency weak supervision。
推理必须event-only，除非明确做多模态control。

损失：
BCE/soft ownership + background motion consistency；不加入P10其它loss。

测试：
- static bypass parity
- ownership range/normalization
- background-only update suppression
- no GT/RGB inference leak
- causal motion estimate
- reset
- zero-event
- compensation coordinate/time consistency

训练/评测：
- static subset、moving subset、background complexity buckets分开报告；
- matched parameter/control；
- background-only adversarial test；
- 外部动态背景数据只报其可支持指标，不伪造mesh GT。

P9 Gate：
- background-only update norm下降>=50%；
- moving subset RA改善>=1.5mm；
- static subset退化<=0.5mm；
- overallstrict recursive不发散；
- 若只在pseudo-mask指标好而pose无改善，FAIL。
```

---

## Prompt 10：P10 Event-Physics / Domain Weak Supervision

```text
本阶段名称：P10_EVENT_PHYSICS_SINGLE_LOSS_SCREENING。
独立分支建议：research/p10-event-physics-20260826。
目标：给已通过机制Gate的P1/P3/P4主干增加最多1–2个event-physics或domain loss，改善real泛化。禁止loss soup；每个loss必须先做正确性probe，再单独scout。

先审计：
- 当前训练/val真实与仿真比例
- event simulation/v2e参数与真实相机差异
- KSSF SDF/normal和可微性
- 是否有可靠RGB/亮度teacher；没有则禁止polarity photometric loss
- 当前augmentations：dropout、hot pixel、polarity flip/swap、time jitter
- loss分发和SO3/FK loss

新增最多一个核心文件semkine/event_physics.py，所有loss纯函数/小module；不重写frontend。

候选loss必须按以下顺序独立筛选，每次只做一个：

L1 KSSF contour likelihood
  L_edge = sum_i w_i rho(SDF(u_i; X(t_i)))
先用GT pose、扰动pose做loss landscape/gradient方向probe。

L2 motion-compensated event contrast/IWE
  I(u;X)=sum_i p_i k(u-W(u_i,t_i;X))
  L_IWE=-Var(I * M_hand)
必须证明all-zero/极端warp不是伪最优；只在hand区域计分。

L3 time-reversal local consistency
  delta(E_rev)约等于正确群作用下的inverse delta
只作训练增强；在线推理不读取未来。必须处理root/joint SO3 inverse，不可简单51D取负。

L4 sensor-domain randomization expansion
  contrast/rate/dropout/refractory/hot pixel/timestamp jitter/background mixture
必须保持label和几何一致；每种增强单独开关。

L5 polarity/brightness consistency
只有存在可靠亮度梯度/teacher时允许；否则标记BLOCKED，不伪造。

每个loss前置测试：
- correct_pose_better_than_perturbed
- gradient_points_toward_small_known_correction（数值局部）
- finite/no_nan
- zero-event behavior
- no trivial collapse
- causal/no future
- geometry augmentation consistency
- config off parity

筛选流程：
1. zero-training landscape probe；
2. one-batch overfit；
3. 500–1000 step single-seed scout；
4. real val strict recursive；
5. 只有改善>=1.1mm或real-sim gap显著收窄才进入2 seed。

每次保持主干checkpoint/init、数据、optimizer一致，仅改变一个loss/augmentation。
报告：
- sim val、real val、low-light、fast-motion、low-rate、noise buckets
- RA/abs/MPVPE/g-slope
- loss magnitude/gradient norm
- training stability
- 不同domain的paired gap

Gate：
- 单个loss对real strict recursive有可重复正效应；
- 不损害closed-loop稳定；
- 不依赖测试标签；
- 最终最多保留两个互补项，必须再做each/combined消融；
- 若所有项<1.1mm且CI跨0，结论为无有效辅助loss，保持主模型简单。

数据许可/标签边界：
- pseudo label不得写成measured GT；
- 没有mesh GT的外部数据只报其支持的joint/consistency指标；
- RGB teacher只用于训练时要在论文中明确，event-only推理单独验证。
```


---

# 第十一部分：100 个角色化专家 Battle 的逐项记录

这 100 个角色是系统性模拟的研究/审稿视角，不是真实人员。Battle 分三轮：第一轮独立指出最可能有效机制；第二轮每个角色给出一个致命反例；第三轮按“根因覆盖—可证伪—复杂度—论文价值”投票。

| # | 专家角色 | 首选路线 | 一票否决/关键反对意见 |
|---:|---|---|---|
| 1 | 事件传感器阈值物理专家 | P10 | 没有真实 contrast/refractory 标定时，物理 loss 可能只拟合模拟器。 |
| 2 | 事件噪声与 hot-pixel 专家 | P1 | pose-free memory 若无 noise-aware decay，会把坏点永久写入状态。 |
| 3 | 时间戳与同步专家 | F0/P1 | sub-ms 时间、同 timestamp 稳定顺序和跨 chunk 守恒不成立，所有异步 claim 作废。 |
| 4 | 事件生成模型专家 | P10 | IWE/极性损失存在退化解，必须做 loss landscape 与 trivial-collapse kill test。 |
| 5 | 低照度事件专家 | P1 | 绝对恢复分支应在低纹理/低亮度中证明比 LNES 更稳，而非只看平均值。 |
| 6 | 高速运动事件专家 | P8 | 高速段应由信息触发缩短查询，而不是固定 50 ms。 |
| 7 | 稀疏静止段专家 | P2 | 无事件不等于状态确定；协方差应增长而不是 bitwise freeze。 |
| 8 | 传感器域迁移专家 | P10 | rate/dropout/threshold randomization 必须单变量验证。 |
| 9 | 事件极性专家 | P10 | 无可靠亮度梯度时禁止强行加入 polarity photometric consistency。 |
| 10 | 神经形态硬件专家 | P3 | GPU 上 sparse 代码不等于低功耗；没有硬件测量不得声称能耗。 |
| 11 | AEGNN 理论专家 | P3 | 核心是 affected-node equivalence，不是把 event 建成昂贵 kNN 图。 |
| 12 | 动态图算法专家 | P3 | 动态图拓扑若随预测姿态变化，会重新引入误差回声。 |
| 13 | PointNet/EventNet 专家 | P4 | global max/mean 对局部手指证据过粗，需 joint queries 或 part-aware pooling。 |
| 14 | FARSE-CNN 专家 | P5 | 层级 compression 必须真流式，不能每包做 dense downsample。 |
| 15 | RVT 专家 | P1 | 持久 memory 值得借用，但大 transformer 不适合本项目。 |
| 16 | HMNet 专家 | P5 | 快慢 memory 应对应真实运动时间尺度，先做诊断再实现。 |
| 17 | SAST 专家 | P8 | 自适应计算应由场景/信息驱动，但 selection 不能吞掉小指事件。 |
| 18 | Event SSM 专家 | P1 | 正时间常数与频率泛化很适合；必须验证 5–50 ms。 |
| 19 | 状态缓存专家 | P1 | caller-owned state 和 sequence reset 是系统级因果的核心。 |
| 20 | 流式 API 专家 | F0 | 隐藏在 module 属性的 sequence state 会让并行 batch 和复现失败。 |
| 21 | 稀疏卷积系统专家 | P3 | 报告真实 active ops，不能只报 dense FLOPs。 |
| 22 | CUDA scatter 专家 | P3 | 固定 cell index/scatter 足够时，不应引入难维护依赖。 |
| 23 | 内存复杂度专家 | P3 | 任何 O(N²) cdist 在 29k events 窗口不可接受。 |
| 24 | 低延迟系统专家 | P8 | 必须同时报 per-event update 与 per-query latency。 |
| 25 | 实时部署专家 | P1 | 异步更新和 MANO query 的调度必须分开测。 |
| 26 | 数值精度专家 | P2 | filter covariance 用 float64 有依据，但边界转换要严格测试。 |
| 27 | 批处理专家 | P1 | ragged batch 与 caller state 必须兼容多 sequence，不得交叉污染。 |
| 28 | 编译与依赖专家 | P3 | 先用 PyTorch 原语实现 reference，再决定是否需要 custom kernel。 |
| 29 | 吞吐评测专家 | P8 | 不同 query rate 的吞吐不可直接比较，必须画 Pareto。 |
| 30 | 嵌入式部署专家 | P5 | 两层结构已足够；第三层会让论文和工程都失控。 |
| 31 | MANO 运动学专家 | P1 | 状态更新必须在 SE(3)×SO(3)^15 上，而不是 51D 欧氏相加。 |
| 32 | 手指局部运动专家 | P4 | distal joints 需要独立证据，不应被腕部大轮廓淹没。 |
| 33 | LBS/posedirs 专家 | P7 | 解析 Jacobian 必须包含 pose blendshape 路径。 |
| 34 | 手部遮挡专家 | P7 | 错误 correspondence 会让 GN 精确走错，必须有可靠性与 fallback。 |
| 35 | 手部碰撞专家 | P10 | 单手主线不要过早加复杂碰撞；仅在交互数据需要时加入。 |
| 36 | 绝对手先验专家 | P1 | pose-free 分支需要绝对恢复能力，但不应变成 HaMeR 大模型迁移。 |
| 37 | 视频手重建专家 | P1 | 训练必须连续 rollout，二窗口扰动不代表长时闭环分布。 |
| 38 | 手部形状专家 | P1 | betas 应序列固定，不应由每个事件包反复预测。 |
| 39 | 相机模型专家 | P9 | root/camera 解耦必须明确坐标系和 metric。 |
| 40 | 手 mesh 评测专家 | P1 | RA 之外必须报 absolute、MPVPE、per-finger 与 tails。 |
| 41 | 非线性控制专家 | P1 | 关键指标是闭环 Jacobian/谱半径方向，不是单步 loss。 |
| 42 | 误差状态滤波专家 | P2 | 网络输出应作为 tangent measurement，更新用 retraction。 |
| 43 | Invariant EKF 专家 | P2 | right/left convention 不一致会造成隐蔽系统误差。 |
| 44 | 协方差校准专家 | P1 | learned precision 必须用 NLL、coverage、NIS 验证。 |
| 45 | 鲁棒统计专家 | P1 | 分支冲突时应降信息/回退 prior，不应由自由 sigmoid 放大。 |
| 46 | 系统辨识专家 | P2 | Q/R 只在 val 锁定小网格，test 不可重调。 |
| 47 | 观测器设计专家 | P1 | pose-free 与 pose-conditioned 双观察器直接命中当前根因。 |
| 48 | 稳定性证明专家 | P1 | contract regularizer 只能辅助，最终仍需实测 g-curve。 |
| 49 | 故障恢复专家 | P1 | 需要低频绝对恢复，而不是每帧强行 blend。 |
| 50 | 闭环测试专家 | P1 | 一次 GT 初始化后的 strict rollout 是唯一主指标。 |
| 51 | 连续时间轨迹专家 | P6 | 先用 causal spline oracle 证明窗口离散化确实是瓶颈。 |
| 52 | Scene Flow 专家 | P7 | 局部运动残差可借鉴 flow，但不应先估密集 scene flow 再回归手。 |
| 53 | 3D trajectory 专家 | P1 | 状态与速度应作为低维轨迹，而非独立帧预测。 |
| 54 | 点跟踪专家 | P7 | learned association + analytic geometry 比全局回归更可解释。 |
| 55 | 运动分割专家 | P9 | foreground ownership 只有动态背景诊断通过才值得做。 |
| 56 | 遮挡边界专家 | P9 | border ownership 比简单二值 mask 更能处理轮廓两侧归属。 |
| 57 | 任意时刻查询专家 | P6/P8 | 任意 query 需要持久状态或连续 field，不能重放窗口。 |
| 58 | 长时记忆专家 | P5 | root 与 fingers 的历史长度应由数据证实。 |
| 59 | 短时运动专家 | P8 | 高速局部事件应触发早查询。 |
| 60 | 轨迹平滑专家 | P2 | 更平滑不等于更准，必须同时看 absolute drift。 |
| 61 | 双目深度专家 | P1 | 可借“强先验+局部迭代”机制，不应构造单目手的 cost volume。 |
| 62 | IGEV-Stereo 专家 | P7 | 迭代更新有效的前提是几何 residual 正确。 |
| 63 | StreamPETR 专家 | P4 | 跨时间查询可迁移为 16 joint queries，而非数百 detection queries。 |
| 64 | Sparse4D 专家 | P4 | 稀疏实体状态适合关节，但 BEV 框架完全不必要。 |
| 65 | 4D detection 专家 | P1 | 时序状态要显式持久，不能把多窗口堆通道。 |
| 66 | Event stereo 专家 | P1 | 语义记忆与几何观察分离值得借用。 |
| 67 | 3D object tracking 专家 | P2 | process/measurement 分离能提升可解释性。 |
| 68 | 迭代检测专家 | P7 | 每次解析更新必须有 basin 与 fallback。 |
| 69 | 多尺度检测专家 | P5 | 只保留快/慢两尺度，避免 FPN 式过度设计。 |
| 70 | 小目标检测专家 | P4 | 小指事件类似小目标，global pooling 最易丢失。 |
| 71 | Event VO 专家 | P9 | 背景全局运动只在动态相机数据必要。 |
| 72 | Event SLAM 专家 | P9 | 完整地图优化对当前单手任务过重。 |
| 73 | Ego-motion 专家 | P9 | 先做 background-only 虚假更新 probe。 |
| 74 | 动态 SLAM 专家 | P9 | 手与相机运动因子需解耦，但静态相机应旁路。 |
| 75 | Bundle Adjustment 专家 | P7 | 解析因子图可借 BA 思想，但在线只做 1–2 次局部 LM。 |
| 76 | 世界坐标手重建专家 | P9 | world grounding 是独立 claim，不要与 camera-relative 主结果混淆。 |
| 77 | IMU 融合专家 | P9 | 没有同步 IMU 就不要假设可用。 |
| 78 | 坐标系审计专家 | P1 | camera/MANO/root pivot 约定必须贯穿数据、Lie、Jacobian。 |
| 79 | 漂移诊断专家 | P1 | silhouette 外事件可作 gross drift trigger，但不能检测毫米级误差。 |
| 80 | 在线 SLAM 系统专家 | P1 | reset、cache、latency、faithfulness同等重要。 |
| 81 | Gauss–Newton 专家 | P7 | active-set、damping、rank deficiency必须显式处理。 |
| 82 | LM 数值专家 | P7 | Cholesky-only 对奇异可观测性不稳，需安全 fallback。 |
| 83 | Fisher 信息专家 | P8 | 信息只代表局部可观测性，不等于网络精度。 |
| 84 | Covariance Intersection 专家 | P1 | 相关分支未知时 CI 比直接相加更保守。 |
| 85 | Bayesian 深度学习专家 | P1 | 任意 uncertainty head 没有 calibration 就没有意义。 |
| 86 | 鲁棒 M-estimator 专家 | P7 | 权重与 ownership 错误时必须限制 influence。 |
| 87 | 优化 basin 专家 | P7 | oracle 要覆盖多级初始误差，而非只测接近 GT。 |
| 88 | 几何微分专家 | P7 | analytic/autograd/finite-difference 三重验证不可省。 |
| 89 | 统计检验专家 | P1 | 逐序列 paired bootstrap 比单个平均数可信。 |
| 90 | 消融设计专家 | P1 | 每个实验只改变一个机制，避免组合增益无法归因。 |
| 91 | CVPR 方法审稿人 | P1 | 最强故事是 event-native observer，不是 AEGNN 换 backbone。 |
| 92 | NeurIPS 理论审稿人 | P1 | 若投 NeurIPS，需要更强 observer/uncertainty/semigroup 分析。 |
| 93 | CVPR 系统审稿人 | P1/P8 | 真异步必须有真实 incremental runtime 与 arbitrary query。 |
| 94 | 数据集审稿人 | P10 | 自有数据必须公开 split、GT 来源、插值与许可。 |
| 95 | 可复现性主席 | F0 | 旧/新 protocol 不分开会使所有结论无效。 |
| 96 | 负结果审稿人 | P2 | 失败 control 也有价值，但必须诚实记录。 |
| 97 | 开源维护者 | P1 | 新增核心文件少、复用已有 Lie/Jacobian 才可维护。 |
| 98 | 统计审稿人 | P1 | 三 seed 与 paired CI 是 SOTA claim 底线。 |
| 99 | 伦理与声明审稿人 | P10 | pseudo GT、teacher、外部数据边界必须写清。 |
| 100 | 领域主席 | P1 | 最终只保留主机制+一个必要增强，十个想法不能全进论文。 |

## 11.1 Battle 汇总

最终的角色投票不是“方法正确性证明”，只是资源分配优先级：

| 路线 | 支持强度 | Battle 裁决 |
|---|---:|---|
| P1 AERO-Hand | 最高 | 唯一同时覆盖当前误差回声、真异步、局部几何和闭环训练的主线 |
| P4 KineQuery16 | 高 | 作为 P1 pose-free readout 的首选结构；oracle 不过即删除 |
| P3 PC-AEGNN | 高 | 解决异步实现真实性与效率；不能单独承担精度故事 |
| P7 NeuMANO-FG | 中高 | 高创新、高风险；必须 oracle-first |
| P2 Lie-IEKF | 中高 | 最先执行的低成本控制臂；适合作为因果证据 |
| P5 FARSE-Kine | 中 | 仅在时间尺度诊断支持时使用 |
| P10 Event Physics | 中 | 主线通过后最多保留 1–2 个辅助项 |
| P8 Information Query | 中 | 更可能形成效率/延迟贡献，而非主要精度贡献 |
| P9 Ego Decouple | 条件性 | 只在动态相机/背景污染被量化后做 |
| P6 CT-KineFlow | 最低先验 | 理论吸引力高，但当前日志不支持；先做 oracle 决定生死 |

所有角色的共同否决是：**只换 backbone、继续依赖上一姿态构造全部事件特征、只做单步 teacher forcing、以及没有 chunk parity 却声称真异步。**

---

# 第十二部分：最终建议的论文形态

## 12.1 推荐标题

首选：

> **AERO-Hand: Asynchronous Error-State Observers for Continuous Event-based Hand Mesh Tracking**

备选：

> **Pose-Free Meets Kinematics: A Dual Observer for Asynchronous Event Hand Mesh Tracking**

不推荐把标题写成：

- “AEGNN for Event Hand Mesh Reconstruction”：容易被认为只是换 backbone；
- “Fully Asynchronous Hand Mesh”但没有 incremental runtime/chunk parity；
- “Continuous-Time”而实际仍只固定 50 ms；
- “Physics-Aware”而物理只是一个很小 loss。

## 12.2 最终主网络只应保留三项贡献

### 贡献 1：Pose-Free Persistent Asynchronous Memory

- raw events 直接、持久、受影响单元更新；
- exact decay 与 chunk semigroup；
- 16 个 MANO joint queries；
- 无上一姿态条件化，因此提供 recovery/absolute evidence。

### 贡献 2：Kinematics-Conditioned Local Observer

- 保留 KSSF/KEG 对局部手指运动的优势；
- 只输出 tangent innovation 与 Fisher；
- 不让它单独控制闭环。

### 贡献 3：Information-Form Lie Error-State Fusion + Rollout

- process prior、absolute/local observation 在 17 个块上保守融合；
- `SE(3) × SO(3)^15` retraction；
- strict causal rollout 与 contraction diagnostics；
- 任意查询频率和真实流式评测。

P3/P4 是贡献 1 的实现候选，P2 是关键控制，P8 可成为效率扩展，P10 可成为泛化扩展。P5/P6/P7/P9 只有通过自己的 oracle/Gate 才进入论文，且最终最多再保留一个。

## 12.3 CVPR 与 NeurIPS 的不同叙事

### 更自然的 CVPR 叙事

- 首个 event-native continuous hand mesh observer；
- 强实际结果：严格闭环、低延迟、低照、高速、外部数据；
- 清楚可视化：pose echo、绝对恢复、per-finger attention、任意时刻 mesh；
- 完整 benchmark/protocol 与代码。

### 要达到 NeurIPS 主会风格，还需额外强化

- 对 async memory 半群/同步—异步等价给出更一般的理论；
- 对双观察器闭环误差给出可验证稳定性界或局部收缩分析；
- 对 correlated observations 的 CI/robust fusion 做更系统不确定性研究；
- 在不止手部一个任务上验证 observer 原理，或给出更普适的学习框架。

因此，**在现有项目基础上，CVPR 方法+系统论文的成功路径更短；NeurIPS 需要把观察器与异步状态更新提升为更一般的方法学。** 这只是投稿定位判断，不是录用保证。

## 12.4 Claim—Evidence 矩阵

| 计划 claim | 最低证据 | 反证/必须避免 |
|---|---|---|
| 真异步 | caller-owned state；full/chunk/event parity；affected ops | 包级重算、隐藏缓存、`torch.cdist` |
| 比 LNES 更精确 | 同协议 3 seed；paired bootstrap；strict recursive | teacher-forced、不同 split/checkpoint 混表 |
| 闭环更稳定 | g-curve、fixed-point、echo excess、long drift | 只报平均 RA |
| 局部手指更好 | per-finger/distal metrics；query ablation | 只报 root-aligned overall |
| 任意时刻 query | 非固定 timestamp；频率泛化；不重放历史 | 只换 10/20/50 ms window |
| 更高效 | 真实 per-event/per-query p95、active ops、memory | 只报参数/FLOPs或理论能耗 |
| 不确定性有效 | NLL、coverage、NIS、reliability | 任意 confidence 可视化 |
| 泛化更强 | 外部真实数据、sensor/rate/noise buckets | 只在仿真或自有 test 上 |
| event-only | test 时没有 RGB/depth/未来帧 | teacher/辅助模态未披露 |
| 连续时间 | 状态函数 `X(t)` 与 arbitrary query | 固定窗口模型更名 |

## 12.5 最小完整实验树

```text
E0 protocol audit + frozen baseline replay
E1 F0 stream/packet/Lie contracts
E2 P2 frozen Lie-filter causal control
E3 P1 pose-free memory contract
E4 global pool vs KineQuery16 oracle + learned scout
E5 pose-free absolute branch
E6 KSSF-only vs absolute-only vs naive fusion vs CI
E7 K=1 vs two-window vs true K=4/8/16 rollout
E8 5/10/20/50ms + arbitrary query
E9 low-event / fast-motion / low-light / occlusion / long-run buckets
E10 one efficiency extension: PC-AEGNN or information-triggered query
E11 one generalization extension: best single P10 loss/domain randomization
E12 external dataset transfer and final 3-seed test
```

## 12.6 失败时的止损决策树

```text
F0不通过
  └─ 不做任何async claim，先修合同

P2改善<1.1mm
  └─ filter不是主解；仍可作为control，资源转P1

pose-free absolute branch单步差>1.5mm
  ├─ GT-part oracle过：改P4 readout/监督
  └─ GT-part oracle也不过：pose-free绝对证据不足，重新审视数据可观测性

P1单步好但recursive仍差
  ├─ 检查branch是否偷偷依赖prev
  ├─ 检查rollout是否仍GT回灌
  ├─ 检查precision/CI是否过度自信
  └─ 检查memory noise/stale-state

P3效率好但精度无增益
  └─ 作为异步实现/效率贡献，不把它写成精度创新

P7 oracle不过
  └─ 删除因子图路线，不继续训练correspondence

P8只让曲线平滑、absolute drift变坏
  └─ FAIL，不用低query率掩盖误差

所有方法都未超过LNES 1.1mm
  └─ 不写SOTA；最诚实成果可能是机制负结果或数据/协议论文，继续定位可观测性而非堆网络
```

---

# 第十三部分：参考论文与机制索引

下列文献用于支撑“为什么借这个机制”，不是要求把所有网络组合。主设计优先依赖官方论文页面/作者代码；最终投稿前应由 BibTeX 工具统一核验作者、页码和版本。

## 13.1 事件异步、稀疏与状态记忆

- **[R1]** Schaefer et al., *AEGNN: Asynchronous Event-Based Graph Neural Networks*, CVPR 2022. 机制：受影响节点异步更新、同步/异步等价。  
  https://openaccess.thecvf.com/content/CVPR2022/html/Schaefer_AEGNN_Asynchronous_Event-Based_Graph_Neural_Networks_CVPR_2022_paper.html
- **[R2]** Sekikawa et al., *EventNet: Asynchronous Recursive Event Processing*, CVPR 2019. 机制：逐事件递归、常数历史内存。
- **[R3]** Gehrig and Scaramuzza, *Recurrent Vision Transformers for Object Detection With Event Cameras*, CVPR 2023. 机制：低延迟持久递归记忆。  
  https://openaccess.thecvf.com/content/CVPR2023/html/Gehrig_Recurrent_Vision_Transformers_for_Object_Detection_With_Event_Cameras_CVPR_2023_paper.html
- **[R4]** Hamaguchi et al., *Hierarchical Neural Memory Network for Low Latency Event Processing*, CVPR 2023. 机制：快慢层级记忆。  
  https://openaccess.thecvf.com/content/CVPR2023/html/Hamaguchi_Hierarchical_Neural_Memory_Network_for_Low_Latency_Event_Processing_CVPR_2023_paper.html
- **[R5]** Peng et al., *Scene Adaptive Sparse Transformer for Event-based Object Detection*, CVPR 2024. 机制：场景自适应稀疏计算。  
  https://openaccess.thecvf.com/content/CVPR2024/html/Peng_Scene_Adaptive_Sparse_Transformer_for_Event-based_Object_Detection_CVPR_2024_paper.html
- **[R6]** Zubic et al., *State Space Models for Event Cameras*, CVPR 2024. 机制：可学习时间尺度与跨推理频率泛化。  
  https://openaccess.thecvf.com/content/CVPR2024/html/Zubic_State_Space_Models_for_Event_Cameras_CVPR_2024_paper.html
- **[R7]** Santambrogio et al., *FARSE-CNN: Fully Asynchronous, Recurrent and Sparse Event-Based CNN*, ECCV 2024. 机制：全异步递归稀疏层级与 compression。  
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7037_ECCV_2024_paper.php
- **[R8]** Sui et al., *Adaptive Spatial-Temporal Window: Unlocking the Potential of Event Cameras in Heterogeneous Velocity Scenarios*, CVPR 2026. 机制：异质速度下自适应时空分组。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.html

## 13.2 事件手部、mesh 与运动

- **[R9]** Jiang et al., *Complementing Event Streams and RGB Frames for Hand Mesh Reconstruction* (EvRGBHand), CVPR 2024. 机制：事件/RGB互补、事件前景稀疏与背景溢出诊断。  
  https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_Complementing_Event_Streams_and_RGB_Frames_for_Hand_Mesh_Reconstruction_CVPR_2024_paper.html
- **[R10]** Millerdurai et al., *3D Pose Estimation of Two Interacting Hands from a Monocular Event Camera* (Ev2Hands), 3DV 2024. 机制：单目事件双手、左右消歧、碰撞约束、真实/仿真数据。  
  https://github.com/Chris10M/Ev2Hands
- **[R11]** EvHandPose, *Event-based 3D Hand Pose Estimation with Sparse Supervision*, TPAMI 2024（以最终正式版本为准）。机制：事件运动补偿/稀疏监督。
- **[R12]** Wang et al., *Continuous-Time Human Motion Field from Event Cameras*, ICCV 2025. 机制：从事件估计连续时间人体运动场。  
  https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Continuous-Time_Human_Motion_Field_from_Event_Cameras_ICCV_2025_paper.html
- **[R13]** Kang et al., *Event6D: Event-based Novel Object 6D Pose Tracking*, CVPR 2026. 机制：高速事件姿态 tracking、任意时刻重建与时序一致性。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Kang_Event6D_Event-based_Novel_Object_6D_Pose_Tracking_CVPR_2026_paper.html

## 13.3 RGB/视频手部结构先验

- **[R14]** Pavlakos et al., *Reconstructing Hands in 3D with Transformers* (HaMeR), CVPR 2024. 机制：强绝对手先验与大规模数据；本方案仅借 teacher/absolute prior。  
  https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html
- **[R15]** Dong et al., *Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba*, NeurIPS 2024. 机制：少量关节 token、图引导状态扫描。  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/03e9a69e5b686c316a07d73f0cf5e225-Abstract-Conference.html
- **[R16]** Chen et al., *HandOS: 3D Hand Reconstruction in One Stage*, CVPR 2025. 机制：2D/3D joint与mesh一体化，减少级联错误。  
  https://openaccess.thecvf.com/content/CVPR2025/html/Chen_HandOS_3D_Hand_Reconstruction_in_One_Stage_CVPR_2025_paper.html
- **[R17]** Yu et al., *Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera*, CVPR 2025. 机制：动态相机下手/相机因子分离。  
  https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html
- **[R18]** Zhang et al., *HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos*, CVPR 2025. 机制：相机轨迹与手运动解耦。  
  https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.html
- **[R19]** Zhao et al., *OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery*, CVPR 2026. 机制：系统级因果、cache、在线一致性和效率。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_OnlineHMR_Video-based_Online_World-Grounded_Human_Mesh_Recovery_CVPR_2026_paper.html
- **[R20]** Ismayilzada et al., *PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery*, CVPR 2026. 机制：物理残差作为虚拟观测及其方差；本方案只借轻量观测思想。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html

## 13.4 3D/stereo/tracking/SLAM 的可迁移机制

- **[R21]** Wang et al., *StreamPETR: Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection*, ICCV 2023. 机制：跨时间持久 object queries；迁移为 16 joint queries。
- **[R22]** Xu et al., *IGEV-Stereo: Iterative Geometry Encoding Volume for Stereo Matching*, CVPR 2023. 机制：强初始化 + 小步迭代几何修正，不迁移 dense volume。
- **[R23]** SpatialTracker, CVPR 2024. 机制：在 3D 运动域进行长期点追踪与局部刚性约束。
- **[R24]** *Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Object Detection*, ICCV 2025. 机制：连续事件语义与几何观测分离。
- **[R25]** *A 5-Point Minimal Solver for Event Camera Relative Motion*, ICCV 2023. 机制：事件运动几何与极小约束。
- **[R26]** EN-SLAM, 2024. 机制：连续事件约束进入 tracking/mapping，而不是只形成事件帧。
- **[R27]** Bartolomei et al., *EventHub: Data Factory for Generalizable Event-Based Stereo Networks without Active Sensors*, CVPR 2026. 机制：以 proxy event/annotation 构造可泛化训练数据。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Bartolomei_EventHub_Data_Factory_for_Generalizable_Event-Based_Stereo_Networks_without_Active_CVPR_2026_paper.html
- **[R28]** Fu et al., *One-Shot Flow, Any-Time Frame: A Bidirectional Warping Framework for Event-Based Video Frame Interpolation*, CVPR 2026. 机制：一次构建运动表示、任意时刻查询。  
  https://openaccess.thecvf.com/content/CVPR2026/html/Fu_One-Shot_Flow_Any-Time_Frame_A_Bidirectional_Warping_Framework_for_Event-Based_CVPR_2026_paper.html

## 13.5 参考文献使用规则

- 引用某论文，只引用其**必要机制**；不要写“我们组合了 AEGNN+FARSE+RVT+Mamba+EKF+GN”。
- 主文方法相关工作可分为：event representation、event hand reconstruction、online mesh tracking 三组；stereo/SLAM 放在“mechanism inspiration”或附录。
- 2026 工作在最终投稿前再次核对正式 proceedings、代码和发布日期。
- 对无法公平复现的方法，使用官方结果并清楚标注，不把不同输入模态直接声称同赛道 SOTA。

---

# 第十四部分：最终执行结论

当前证据下，最合理的资源分配不是同时实现十个 Plan，而是：

```text
第一优先：F0 → P2 → P1
第二优先：P4 与 P3 二选一，依据 oracle/异步 parity
第三优先：P8 或 P10 选一个形成效率/泛化扩展
条件路线：P7 oracle 过才做；P9 contamination 过才做；P6 continuous-time oracle 过才做
```

**主线最终应保持简单：**

```text
2-scale pose-free async memory
+ 16 MANO queries
+ current KSSF local innovation
+ existing Lie process/filter/Jacobian
+ blockwise conservative fusion
+ true K-step recursive rollout
```

它比“直接把 ResNet/LNES 换成 AEGNN”更可能同时提高精度、稳定性和论文创新性，因为它对准了仓库日志已经暴露的主要矛盾：

> 当前稀疏事件分支不是看不见手，而是看手的方式过度依赖已经漂移的上一姿态。

文件中的所有门槛都应被视为预注册决策规则。达到门槛才进入下一阶段；达不到就记录 FAIL 并停止。只有最终在相同协议、严格递归、三种子、外部验证和真实增量效率上成立，才能诚实写“新 SOTA”。
