# EventKineGraph：单目事件手部 Mesh 的异步运动学图跟踪主方案

> 仓库：`labor-fight/EventHands1`  
> 审计日期：2026-08-23；**2026-08-24 修正（S20 实测 + skill 复审）**  
> 本文状态：**研究与实现合同，不是已取得的 SOTA 结果**

## 0.1 2026-08-24 必须执行的修正（深夜第二次修订：软路由已被否证）

S20 闭环探针已经把“换成 GNN 会丢信息”这件事量清楚了：KEG 单步 11.59 mm vs LNES 10.95 mm（在 1.1 mm 训练分辨率内打平），递推 30.39 vs 20.87 mm，放大倍数 x2.62 vs x1.91。**当晚的软路由重训（两个 seed）已经验收：FAIL**（`closed_loop_probe_soft_50ms.json`：x2.66 / x2.47，gate ≤ x2.10，单步还倒退 0.6–1.4 mm）。修正如下：

1. **禁止把像素 AEGNN / causal hash event graph 当作 LNES 的替换入口。** 不变。`probe_gnn_info_loss.py` 补充了实测：像素 kNN 邻域跨手指率 0.30–0.80。AEGNN 只借“新事件只更新被触及的子图”；子图必须是 16 个 MANO 节点。
2. **“闭环 Lipschitz 软路由”这一押注已被运行时数据否证，不得再投入。** 机制解释已测出：在闭环真实误差尺度（1 cm / 0.1 rad）下，软 top-4 的路由迁移 TV=1.46–1.78（满分 2），与硬 argmax 相等——蒙皮支撑面整体迁移时凸组合随之整体迁移，Lipschitz-at-zero 买不到闭环稳定。同时 drift 曲线证明闭环是收缩的（x1 与 x4 初始噪声轨迹重合），缺口是**每步误差注入的稳态平衡点太高**，不是误差复利。
3. **当前最大概率 SOTA 变量是“证据不删除”：visibility 硬门是被忽略的主嫌疑。** 实测：GT 状态下 `keep=vis` 就删掉 ~74% 事件（vis_frac≈0.26），被删事件中位 |sdf| 4–9 px、41–97% 在 12 px 轮廓带内——事件在运动轮廓两侧成对出现，外侧一半被整体清零丢进 BG；状态误差增大时删除率单调上升（7/8 窗口，低事件率序列 x2.5–3.5），构成稠密对照没有的正反馈。修复方向 = 把硬 vis 门换成 SDF 连续带权 `w_geo = exp(-max(sdf,0)/κ)`，轮廓外事件保留几何通道（sdf、normal 本身就是纠错所需的失配几何）并路由到最近轮廓段的关节。是否需要叠加短闭环 unroll 训练，由 `closed_loop_sensitivity_50ms.json` 的不动点分析（R2）裁决：若独立噪声 g 曲线的不动点 ≈ 实测递推稳态，表示层修复优先；若远低于稳态，训练分布修复（unroll）必须同时上。
4. **论文主张再次收窄并换正确的词。** 可发表贡献不是“Lipschitz 路由”，而是“**证据完备的状态条件事件–运动学路由**（deletion-free state-conditioned routing）+ LTI 时间聚合 + 树更新”，负对照链完整：像素图（跨手指混合）、硬门（删证据）、软化路由（错误的连续性目标）三个失败机制全部有测量支撑，这本身是论文 §analysis 的骨架。
5. 下文 §2 的三条机制维持上一版裁决：机制 1（因果局部事件图）为负对照；机制 3（16 节点树）相对 routed MLP 仍要过 1.1 mm Gate。新增：任何修复不得再引入“把证据置零/丢弃”的路径——零事件恒等约定除外。

## 0.2 2026-08-24 深夜第三次修订：不动点裁决完成，双支柱定案（debug 3eaac2 收口）

`closed_loop_sensitivity_50ms.json` 四臂扫描（keg_soft / 同 ckpt 硬切换 / keg_hard / lnes ×
teacher-forced、递推、三档实测条件误差 6.6 / 26.1 / 50.0 mm）把 §0.1 留下的裁决全部关闭，
详细数字见 `docs/semkine/EXPERIMENT_LOG.md` 的 S20 adjudication 节。结论压缩成四条：

1. **g 曲线定量（R1 CONFIRMED）**：KEG 单步误差对条件状态误差的斜率 ≈0.32，LNES ≈0.15，
   2.2–2.8 倍。这是状态条件前端自己的敏感度，视觉证据删除率（vis_frac 0.62→0.32 随误差单调降，
   递推稳态 0.47）与它同向（R3 CONFIRMED）。
2. **不动点远低于稳态（R2 CONFIRMED，裁决词生效）**：独立噪声不动点 KEG 14.3 mm / LNES
   12.0 mm，实测递推稳态 32.6 / 20.9 mm——**两臂都有 x1.7–x2.3 的“相关误差回声”超额**，
   按 §0.1 第 3 条的裁决词：**表示修复与训练分布修复必须同时上**。递推里两臂每步更新坍缩到
   需要量的 11%（0.069 vs 0.61 rad；LNES 同样 0.071）,是 exposure bias 的教科书指纹
   （DAgger, Ross & Bagnell AISTATS'11；scheduled sampling, Bengio et al. NeurIPS'15）——
   在 teacher-forced 加噪下 update_ratio 0.77–0.93,训练分布里从未有过自生漂移(R4 在 TF 域
   REJECTED、在闭环域 CONFIRMED)。
3. **路由形式正式出局（R5 REJECTED）**：同 ckpt 硬/软切换递推差 0.8 mm、单步差 0.12 mm，均在
   1.1 mm 复制底噪内；独立训练的 keg_hard 30.4 mm 也在 seed 方差内。连同软路由重训 FAIL，
   路由形式的全部变体（硬/软/重训软）都有测量负结果，写进论文负对照链，不再投入。
4. **双支柱执行序（已启动）**：
   - **E5.5a `s21_keg_halo`（2026-08-24 夜已启动，seed 3407/3408，GPU 6/7）**：deletion-free
     halo 路由。`w_geo = exp(-[sdf]_+/6px)` 连续可见性（SoftRas 式把二值覆盖换成 SDF 单调函数，
     Liu et al. ICCV'19）；最近点投影 `π(x)=x-(d+1)n`（SDF 性质 ∇d=n，窄带内一阶精确）读产生
     该事件的轮廓段的 LBS；路由测度 `w_geo·Σŵ_k δ_{j_k} + (1-w_geo)·δ_BG` 总质量恒 1。
     内侧事件与硬门版逐位一致（合同测试已过），单变量 = 被删那一半证据的去向。
     Gate：G-a 单步 ≤11.59+1.1；G-b g 斜率@26mm <0.20；G-c 放大 ≤x2.10 或递推 ≤21.9 mm。
   - **E5.5b unroll/scheduled-sampling（第一次 FAIL 已修正，退火版 2026-08-25 16:32 重启，
     seed 3407/3408 双卡并行）**：短闭环展开训练打两臂共享的回声超额。算术上限已知：只修回声 →
     KEG 落回不动点 ~14.3 mm；只修表示 → 残留 x1.74–x2.28 回声。E5.5a 验收：G-a PASS（单步
     12.11/11.16），G-b FAIL（斜率 0.24/0.27 未平），G-c FAIL（递推 27.8 vs 21.9）——但递推双
     seed 一致改善 2.6–4.8 mm（机制=断删除正反馈，回声超额 x2.28→x1.85–1.95），halo 保留。
     E5.5b 实现 = 数据集出 `(前导窗, 主窗)` 相邻对（与递推 evaluator 铺贴一致，前导 end = 主窗
     start-1），训练时用前导窗自预测（stop-gradient）替换主窗的 GT+噪声条件
     （scheduled sampling, Bengio et al. NeurIPS'15；DAgger on-policy 分布, Ross & Bagnell
     AISTATS'11）。**第一次训练用恒定 UNROLL_P=0.5 从 step 0 生效——FAIL（step3500 探针
     TF RA 42.5 mm / 递推 144.5 mm）：早期自预测近随机，一半条件是垃圾，网络学会无视 prev，
     坍缩回绝对回归档（账本已记）。修复 = TRACK.UNROLL_RAMP [500, 2000] 线性退火（Bengio 原文
     的 curriculum 正是防此事故），前 500 步纯 teacher-forced（等价 s21），2000 步后满 0.5。
     退火版 30 分钟处 train_loss 0.507 / val_loss 0.125（constP 同期 39 / 40.9，差两个量级），
     健康。**合同测试 `tests/test_s22_unroll.py` 4 项已过（含退火曲线）；单变量 vs s21 = 训练
     条件分布。Gate：G-a 单步 ≤ s21+1.1；G-b 递推 ≤21.9 或放大 ≤x2.10；G-c 递推每步更新量脱离
     坍缩（pred_step_pose > 0.2 rad,坍缩值 0.069）。
   - 预算：E5.5a 已占 GPU 6/7 各 ~3h；E5.5b 实现 + 双 seed 另计。s20b（硬化噪声表）与 E5.5b
     同属训练分布修复，s20b 结果可作为 E5.5b 的下界参考，不重复投入。


## 0. 证据边界

本轮能够审计的是 GitHub 当前快照、仓库内 `docs/experiment_history.md`、`docs/debug_closed_loop_diagnosis.md`、`docs/semkine/`、`semkine/` 源码，以及此前保存在文件库中的 SemKine 计划。当前会话没有挂载：

```text
/data1/lyq/code/EventHands1
/data1/lyq/code/skill/creative-thinking-for-research
/data1/lyq/code/skill/research-opportunity-graph-skill
```

因此本文不会声称已经读取两个本地 skill，也不会声称已经删除 Ubuntu 本机上的 untracked 日志、输出、checkpoint 或失败 `.py`。本机清理由 `tools/cleanup_research_tree.py` 以 dry-run 开始。

## 1. 当前代码与实验的关键裁决

### 1.1 必须保留的已验证闭环

现有可信主线不是纯增量积分，而是：

```text
当前事件证据
+ 上一时刻 51D MANO 状态
+ 上一状态几何反馈
-> 预测相对上一状态的 delta
-> prev + delta
-> 递推到下一时刻
```

51D 状态布局保持：

\[
x=[t\in\mathbb R^3,\;\omega_0\in\mathbb R^3,\;\omega_{1:15}\in\mathbb R^{45}].
\]

以下合同不得破坏：

1. 每段只在起始处使用一次初始化状态；其后严格 `prev <- pred`。
2. 零事件包严格输出恒等更新。
3. MANO、相机内参、beta、sequence reset 和递推 evaluator 保持原语义。
4. 新模型仍输出 MANO 状态，而不是直接回归 778 个顶点。
5. checkpoint 必须按递推验证指标选择，不能按单步 `val_loss` 选择。

### 1.2 不能继续沿用的两个“伪 GNN”入口

#### `semkine/frontends.py::AEGNNLite`

当前实现对每个事件包调用全量 `torch.cdist`，复杂度和显存为 \(O(N^2)\)。仓库统计中高事件率窗口可接近 3 万事件/50 ms，这一路径在真实包上不可扩展。它还存在：

- 整包 kNN 可能把未来事件连到过去事件，违反因果性；
- 没有持久化 evolving graph；
- 没有显式 \(\Delta x,\Delta y,\Delta t\) 与极性边特征；
- 最后全局 mean/max pooling，手指局部证据全部混合；
- 与 MANO 当前状态、可见性和关节支撑无关。

结论：**借 AEGNN 的“只更新受新事件影响的局部子图”思想，但不能直接保留当前实现，也不应原样移植官方代码。**

#### `semkine/gnn.py::KinematicGNN`

当前 16 节点树图把同一个 global feature 复制给全部关节，再拼接上一角度。它没有回答最重要的问题：

> 哪些事件约束拇指、哪些约束食指、哪些只解释根节点运动？

历史 KSGN/SPA 加性图头已经证明：若保留强大的全局 `fc -> 51D` 旁路，零初始化图头会变成惰性旁路，推理期清零只改变约 0.16 mm。新方法必须让局部事件证据经过图路由成为关节更新的**唯一通路**，不能再做“ResNet 主头 + 小 GNN 加法头”。

### 1.3 不能再使用的旧叙事

旧文档曾写“LNES 丢掉事件顺序，所以方向信息在进入网络前已不存在”。后续 teacher-forced probe、LOSO 与域随机化已经修正该结论：

- LNES 确实丢失事件重复次数与窗口内精细时序，也造成固定窗口和固定计算；
- 但当前最强的已测精度修复来自**低事件率域覆盖**，不是单纯换网络；
- `zgz_local` 的极低事件率曾支配总指标；相机一致几何增强、事件随机丢弃和热像素增强把递推 RA 从约 19.26 mm 改善到双副本 worse-of-two 约 12.77 mm。

因此新论文的正确命题是：

> 解除 LNES 的固定帧化、重复覆盖和查询频率限制，并用当前 MANO 状态把稀疏事件路由到真正可观测的关节；同时保留已经验证的低事件率域随机化。

新 GNN 若不能在同一域随机化下超过 12.77 mm，只能证明部署形式更异步，不能证明精度更高。

## 2. 最终主方案

建议方法名：

**EventKineGraph: State-Conditioned Asynchronous Kinematic Graph Tracking for Monocular Event Hand Mesh Recovery**

只保留三个核心机制：

1. **Causal Local Event Graph（因果局部事件图）**：不用 LNES，不构造全量距离矩阵，只连接有限时间和空间邻域内的历史事件。
2. **Skinning-Observability Bipartite Router（蒙皮—可观测性双部路由）**：利用上一 MANO 状态的投影、可见性、LBS 权重与局部投影 Jacobian，把事件送到 16 个 MANO 更新节点。
3. **Kinematic Lie Update Graph（运动学 Lie 更新图）**：在 MANO parent-child 树上做两层消息传播，输出 root 与 15 个局部关节的切空间增量，再递推组合。

不在第一版同时加入 Transformer、SNN、Gauss-Newton、Kalman、diffusion 或绝对大模型。SAST 只借“按证据分配预算”的思想；EventNet/SSM 只借“递归状态与真实时间间隔”的思想；AEGNN 只借“局部受影响子图更新”的思想。

## 3. 严格数学定义

### 3.1 状态与输入

事件：

\[
e_i=(\mathbf u_i,t_i,p_i),\qquad \mathbf u_i=(x_i,y_i),\quad p_i\in\{-1,+1\}.
\]

递推前状态：

\[
X^-_k=(\mathbf t^-_k,R^-_{k,0},\ldots,R^-_{k,15},\beta),
\]

其中 \(R_{k,0}\) 是根旋转，\(R_{k,1:15}\) 是 MANO 局部关节旋转，\(\beta\) 在同一序列内固定。

事件以 1–5 ms 或固定 64–512 events 的 micro-packet 输入；网络允许在任意查询时刻输出状态，但训练初期仍以现有 50 ms 标注间隔监督。

### 3.2 因果哈希事件图

把 \((x,y,t)\) 放入离散哈希单元：

\[
\mathbf c_i=\left(\left\lfloor\frac{x_i}{s_x}\right\rfloor,
\left\lfloor\frac{y_i}{s_y}\right\rfloor,
\left\lfloor\frac{t_i}{s_t}\right\rfloor\right).
\]

只从当前单元和相邻单元查询历史节点，定义：

\[
\mathcal N_i^-=
\left\{j:\;0<t_i-t_j\le \tau,\;
\left\|S(\mathbf u_i-\mathbf u_j)\right\|_2\le r\right\},
\qquad |\mathcal N_i^-|\le K.
\]

必须满足 `t_j < t_i`，因此不存在 future edge。推荐首版：`hidden=64`、`K=16`、两层 message passing、节点寿命 \(\tau=5\sim15\) ms。

边特征：

\[
\epsilon_{ji}=\left[
\frac{\Delta x}{W},\frac{\Delta y}{H},
\log\left(1+\frac{\Delta t}{\tau_0}\right),
 p_i,p_j,p_ip_j
\right].
\]

事件更新：

\[
m_{ji}^{(l)}=\phi_l(h_j^{(l)},h_i^{(l)},\epsilon_{ji}),
\]

\[
\bar m_i^{(l)}=\sum_{j\in\mathcal N_i^-}\alpha_{ji}^{(l)}m_{ji}^{(l)},
\qquad
h_i^{(l+1)}=\operatorname{GRU}_l(h_i^{(l)},\bar m_i^{(l)}).
\]

图构造和消息数应为 \(O(NK)\)，禁止 `torch.cdist(N,N)`。

### 3.3 状态条件几何查询

首个可训练版本复用现有 MANO renderer，但只在事件坐标查询上一状态的几何属性；**不再把事件变成 LNES 图输入 CNN**。几何稳定后，再用 778 个投影顶点的二维哈希查询替代稠密几何图，作为效率消融。

由上一状态生成：

\[
V^-=\operatorname{MANO}(X^-_k),\qquad
\hat{\mathbf u}_v=\pi_K(V^-_v).
\]

对事件 \(e_i\)，查询其附近可见顶点或三角面，得到：

\[
g_i=[d_i,\;\mathbf n_i,\;\Delta z_i,\;\nu_i,\;\bar{\mathbf w}_i],
\]

其中 \(d_i\) 是事件到当前投影手表面的有符号/近似轮廓距离，\(\mathbf n_i\) 是局部残差法向，\(\Delta z_i\) 是深度差，\(\nu_i\) 是可见性，\(\bar{\mathbf w}_i\in\mathbb R^{16}\) 是由 MANO LBS 权重插值得到的关节支撑。

若查询到三角面顶点 \(v_1,v_2,v_3\) 与重心权重 \(b_{ir}\)：

\[
\bar w_{iq}=\sum_{r=1}^{3}b_{ir}W_{v_rq}.
\]

### 3.4 事件—关节双部路由

事件对关节 \(q\) 的局部投影敏感度：

\[
s_{iq}=\left\|\mathbf n_i^\top J_{iq}(X^-_k)\right\|_2^2,
\qquad
J_{iq}=\frac{\partial\pi_K(V_i)}{\partial\delta\omega_q}.
\]

路由权重：

\[
a_{iq}=\frac{
\nu_i\,\bar w_{iq}\,\exp(\gamma s_{iq})
}{
\epsilon+\sum_{r=0}^{15}\nu_i\,\bar w_{ir}\,\exp(\gamma s_{ir})
}.
\]

每个关节聚合自己的事件证据：

\[
z_q=\frac{\sum_i a_{iq}\,\psi(h_i,g_i)}{\epsilon+\sum_i a_{iq}},
\qquad
I_q=\sum_i a_{iq}.
\]

\(I_q\) 是该包对关节 \(q\) 的证据/信息量。它同时用于更新阻尼和审计，不需要额外人工标签。

第一版可先只用 LBS、可见性和距离构成 \(a_{iq}\)，待单元测试与 oracle route 通过后再加入 Jacobian 项；禁止一次性把所有复杂度堆入首个 scout。

### 3.5 MANO 树 GNN

16 个可更新节点：root + 15 articulation joints。21-joint 输出中的 fingertip 只作只读观测/损失节点，不作为独立旋转状态。

初始节点：

\[
q_q^{(0)}=\left[z_q,\log(\epsilon+I_q),\operatorname{Log}(R^-_{k,q}),e_q,h^-_q\right],
\]

其中 \(e_q\) 是 joint-id embedding，\(h^-_q\) 是持久化小状态。

只在 MANO parent-child 边上传播两层：

\[
q_q^{(l+1)}=q_q^{(l)}+\Phi_l\left(
q_q^{(l)},
\sum_{r\in\mathcal N_{\rm MANO}(q)}
\Psi_l(q_r^{(l)},q_q^{(l)})
\right).
\]

输出：

\[
(\delta\mathbf t,\delta\omega_0,\ldots,\delta\omega_{15},\log\sigma_0^2,\ldots,\log\sigma_{15}^2).
\]

网络保持简单：事件图两层 + 双部聚合一层 + MANO 树两层；不做 778 顶点在线 GNN，不做全连接 16 节点 attention。

### 3.6 证据阻尼与 Lie 递推

历史失败说明，teacher forcing 下自由学习的保守 gate 容易被关闭。首版采用单调、可解释的信息阻尼：

\[
g_q=\operatorname{clip}\left(\frac{I_q}{I_q+\lambda_q},g_{\min},1\right),
\qquad \lambda_q>0.
\]

推荐先固定 \(\lambda_q\) 为训练集该关节信息量的低分位数；只有固定规则通过后，才允许学习 `softplus(lambda_q)`。

更新：

\[
\mathbf t_k=\mathbf t^-_k+g_t\delta\mathbf t,
\]

\[
R_{k,q}=R^-_{k,q}\operatorname{Exp}(g_q\delta\omega_q).
\]

根平移与根旋转采用 \(\mathbb R^3\times SO(3)\) 解耦组合，不用完整 SE(3) 左乘把旋转误差耦合成平移偏差。

零事件包：\(I_q=0\)，强制 \(g_q=0\)、\(X_k=X^-_k\)。

### 3.7 损失

首版监督目标：

\[
\mathcal L=
\lambda_R\mathcal L_{SO(3)}+
\lambda_t\mathcal L_t+
\lambda_J\mathcal L_J+
\lambda_V\mathcal L_V+
\lambda_D\mathcal L_{distill}+
\lambda_U\mathcal L_{uncertainty}.
\]

旋转流形损失：

\[
\mathcal L_{SO(3)}=
\frac1{16}\sum_q
\rho\left(\left\|\operatorname{Log}\left((R_q^{gt})^TR_q\right)\right\|_2\right).
\]

平移必须有厘米量级有效梯度：

\[
\mathcal L_t=\operatorname{SmoothL1}_{\beta=0.01m}(\mathbf t-\mathbf t^{gt}).
\]

FK 关节与顶点同时保留 root-relative 和 absolute 项，避免历史 SO(3)+FK 配方把非对齐 MPJPE 从约 63.7 mm 推到约 83.6 mm：

\[
\mathcal L_J=
\rho(J-J^{gt})+\eta\rho((J-J_{root})-(J^{gt}-J^{gt}_{root})).
\]

教师蒸馏只在训练初期使用，且蒸馏 tangent/FK，不直接对原始 51D axis-angle 做 MSE：

\[
\mathcal L_{distill}=\sum_q
\left\|\operatorname{Log}\left((\Delta R_q^{T})^T\Delta R_q\right)\right\|_1
+\eta_t\|\Delta\mathbf t-\Delta\mathbf t^T\|_1.
\]

可选事件一致性必须等主监督版本稳定后再加入：用预测局部运动把每个事件沿其真实时间戳 warp 到查询时刻，最小化其到预测轮廓的鲁棒距离；不先生成 IWE 图。

## 4. 训练方案

### 4.1 公平基线

所有网络臂必须使用相同：

- subject-disjoint split；
- 相机/极性/时间单位；
- 低事件率域随机化；
- 事件 keep ratio、热像素与几何增强；
- 初始化噪声；
- 递推 evaluator；
- 训练步数、checkpoint 网格和 seed。

至少冻结两条控制：

1. `track_render51_domrand`：当前精度控制；
2. `raw_scan_domrand`：不含 event graph 和 joint router 的原始事件控制。

### 4.2 三段训练

#### A. 单步可学习性

- teacher-forced previous state；
- 只训练 50 ms 查询；
- 验证 event graph、router、GNN 能否学到正确 delta；
- 100–500 个样本 overfit 必须先达到明显低误差。

#### B. 短闭环

- 4–8 个 micro-packet unroll；
- 逐步把 previous state 从 GT 换成 prediction；
- loss 在每个状态查询点计算；
- hidden/state 在 sequence 边界清空，TBPTT 时显式 detach。

#### C. 多频率与真异步

- 随机输出间隔 5/10/20/50 ms；
- 同一事件前缀，batch 同步实现与 incremental 实现必须数值等价；
- 训练时不读取查询时刻之后的事件。

## 5. 代码落地地图

建议新建独立 namespace，保留旧模型不动：

```text
semkine/event_graph.py          # causal hash graph + bounded lifetime
semkine/geometry_router.py      # event-coordinate MANO/LBS/Jacobian query
semkine/event_joint_graph.py    # sparse event -> 16 joint bipartite aggregation
semkine/kinematic_tracker.py    # 2-layer MANO tree GNN + Lie update
semkine/async_runtime.py        # persistent graph/state and prefix queries
semkine/losses_eventkine.py     # SO(3), abs/root-relative FK, distillation
```

修改：

```text
semkine/events.py
  - 保留微秒时间戳；
  - 输出按 sequence 排序的 micro-packet；
  - 增加 sequence_id/query_time/reset 标志；
  - 不做 B x T 大 padding。

semkine/train.py
  - 增加 multi-step unroll sampler；
  - checkpoint 仍固定网格保存；
  - validation 主指标使用 recursive RA；
  - teacher 只在明确阶段启用。

model/model.py
  - 不把新架构继续塞进 MNISTModel 的大量 if/else；
  - 仅保留共享 MANO、renderer、loss/eval adapter；
  - 新建 AsyncKinematicTracker 作为独立模型类。
```

第一批合同测试：

1. `no_future_edges`：所有边满足 `t_src < t_dst`；
2. `bounded_degree_and_linear_edges`：边数不超过 \(NK\)；
3. `hash_vs_bruteforce_tiny`：小样本哈希半径图与暴力 reference 一致；
4. `zero_event_identity`：状态逐位不变；
5. `sequence_isolation`：graph/hidden 不跨序列；
6. `router_normalization`：每事件路由权重和为 1 或 0；
7. `lbs_support_counterfactual`：单关节扰动只影响其正确蒙皮子树；
8. `sync_async_prefix_parity`：相同事件前缀输出一致；
9. `lie_roundtrip`：`Exp(Log(R))` 与长序列组合稳定；
10. `no_global_bypass`：清零 joint evidence 必须让局部 delta 消失，不能由全局 FC 偷走任务。

## 6. 分阶段 Gate

| 阶段 | 唯一主要变量 | 必须通过的 Gate | 失败动作 |
|---|---|---|---|
| E0 | 清理与基线冻结 | 本机复现控制；评测协议、manifest、hash 完整 | 不改网络 |
| E1 | raw micro-packet 数据 | 事件逐元素一致；时间单调 100%；零事件合法；DataLoader p50 ≤1.25× legacy | 只修数据 |
| E2 | causal hash event graph | 无 future edge；边数 \(O(NK)\)；高事件率不 OOM；tiny reference 一致 | 删除 graph 实现 |
| E3 | LBS event-to-joint router | oracle route 比 global pooling 的 active-joint error 至少低 5%；root-only false routing 降 30% | 保留 raw encoder，不进 GNN |
| E4 | 2-layer kinematic GNN | 相对 matched-parameter shared MLP，local/active error低 ≥5%，overall RA 不恶化 >0.10 mm；或同精度延迟低 ≥20% | 回退 shared MLP |
| E5 | evidence damping | low-event bucket 提升 ≥10%；normal bucket 不退化 >2%；零事件 drift=0 | 回退固定 trust |
| E6 | 4–8 step closed loop | 递推 RA 相对 raw control 至少改善 1.1 mm；两副本方向一致；global 退化 <0.5 mm | 停止扩模型，诊断 |
| E7 | true async runtime | prefix parity 通过；5/10/20/50 ms 均可查询；p95 随事件数近线性；精度退化≤2% | 保留同步训练版本 |
| E8 | 最终统计 | 3 seeds；paired bootstrap 95% CI；test 前冻结；同协议比较公开基线 | 未过不得写 SOTA |

统计噪声已有约束：独立训练散布可达到毫米级，不能用单个 best checkpoint 或单 seed 的 0.3–0.5 mm 差异写贡献。建议最小可发表精度效应门槛仍采用约 1.1 mm，或 paired CI 明确不跨 0。

## 7. 60 角色对抗评审的合并裁决

这里的“60 位专家”是 10 组 × 6 个专业角色的结构化模拟评审，不冒充真实外部评审：

1. 事件表示与传感器；
2. GNN 与动态图；
3. 稀疏 CUDA/系统；
4. MANO/手部 Mesh；
5. Tracking/滤波；
6. 几何优化；
7. Event flow/depth/stereo；
8. 稀疏 3D 检测；
9. 训练、泛化与统计；
10. CVPR/NeurIPS 审稿与复现。

候选裁决：

| 候选 | 主要优点 | 致命问题 | 裁决 |
|---|---|---|---|
| 整包普通 AEGNN + global pooling | 最容易写 | 非因果风险、\(O(N^2)\)、关节局部性丢失 | 淘汰 |
| 稀疏卷积网格 | 工程成熟 | 重新离散成固定网格，异步更新与关节路由不足 | 仅效率对照 |
| 完整 SAST | 自适应 token 预算 | 仍是窗口化 dense token pipeline，模型过重 | 只借 scorer 思想 |
| EventNet/PointNet 全局递归 | 简单、因果 | 不知道事件属于哪个关节，遮挡时混淆 | raw control |
| 778 顶点 Mesh GNN | 表面细 | 在线节点过多，观测与状态维数不匹配 | 淘汰 |
| **事件—MANO 双部图 + 16 节点树 GNN** | 观测支撑、运动学和异步计算统一 | router 正确性与实现是主要风险 | **主方案** |

三轮对抗后的共识不是“把 AEGNN 装进 EventHands”，而是：

> 用上一 MANO 状态建立事件到关节的任务条件图；事件图解决局部时空编码，双部图解决证据归属，MANO 树图解决关节耦合。三者分别对应三个不可替代问题。

## 8. 审稿人最可能攻击的点

### “这只是 AEGNN + MANO”

回答必须依靠双部路由消融：普通 event graph/global pooling、LBS-only router、LBS+observability router、无 kinematic edges、错误随机 topology。主贡献是**状态条件 event-to-joint graph**，不是 GNN 层本身。

### “你以前的 GNN 头已经失败”

明确展示结构差异：以前是强全局 FC 主路 + 零初始化加性图头；现在删除局部 45D 的全局旁路，关节 delta 必须由 routed joint evidence 产生。加入 `no_global_bypass` 测试和推理期 evidence ablation。

### “LNES 并不是已证实的精度瓶颈”

同意并收窄主张：目标是事件原生频率泛化、按事件量缩放和关节局部可观测性；不声称仅去掉 LNES 必然提升精度。精度控制必须使用同一低事件率域随机化。

### “单目事件不能恢复绝对深度”

任务是有初始化的递推 tracking，不是每次从零绝对重建；previous MANO state 提供尺度/深度先验。仍需分别报告 root-relative 与 absolute MPJPE/MPVPE，低证据时阻尼更新，长时漂移单独报告。

### “所谓异步只是把 50 ms 包换成图”

必须提供：事件前缀查询、sync/async parity、1/5/10/20/50 ms 输出、每千事件延迟、p95 高事件率延迟、边/节点实际更新数。没有这些证据不能写 asynchronous deployment。

## 9. 可发表贡献的正确表述

只有 E8 通过后，论文才可主张：

1. 一种不把事件压成 LNES/voxel frame、直接从因果事件图递推 MANO 状态的单目事件手部 Mesh tracker；
2. 一种由上一 MANO 投影、LBS 支撑和局部观测 Jacobian构成的 event–joint bipartite router；
3. 一种可同步训练、按事件前缀异步执行的 16 节点运动学图更新器，并在多查询频率和低事件率条件下验证；
4. 一个严格区分 normal/low-event、global/local、root-relative/absolute、精度/延迟的评测协议。

在本次检索范围内，已有事件手工作主要采用事件图像、全局点云/特征融合、手流/IWE 或分割增强；未发现与“状态条件 event-to-MANO-joint 双部图 + 真前缀异步 Lie tracking”完全同构的方法。但“first”必须在论文提交前再做一次系统检索和作者代码核验。

## 10. 明确禁止

- 不把 `AEGNNLite` 的 `torch.cdist` 当最终实现；
- 不先构造完整图再乘 active mask；
- 不保留局部 45D 的强 global FC 旁路；
- 不把 778 顶点作为在线状态图；
- 不直接 axis-angle 相加；
- 不用不同域增强、不同初始化或不同 evaluator 比较新旧模型；
- 不用单 seed/best checkpoint 声称 SOTA；
- 不在首版同时加入 GNN、SAST、SNN、GN、滤波和绝对锚；
- 不因为方法“看起来高级”而保留未过 Gate 的组件。

## 11. 立即执行顺序

1. 在本机运行 cleanup dry-run，冻结 `track_render51_domrand` 与 raw data manifest；
2. 只实现 E1/E2：micro-packet + causal hash graph，不接 MANO；
3. 通过线性复杂度与因果测试后，实现 LBS-only router；
4. 先比较 routed shared MLP，再增加两层 MANO GNN；
5. 只有 GNN 相对 matched MLP 过 Gate，才加入 Jacobian observability 和 evidence damping；
6. 最后做 multi-step closed loop 与 incremental runtime；
7. 所有失败臂写入一个失败账本并从主配置、主表和主方法图删除。
