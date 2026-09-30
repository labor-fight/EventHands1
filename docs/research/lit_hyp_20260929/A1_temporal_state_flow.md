# A1：时间信息流与 prev / 递推状态（深度审计）

> 子代理 A1 · 2026-09-29 · 对照 `00_CONTEXT.md` §1/§7/§8  
> 范围：S37 routed（`configs/semkine/s37_routed_s3407.yaml`）  
> 不改仓库代码；未跑 GPU；未写 CPU 分析脚本（本题可由代码 + 已有 JSON 闭合）。

---

## 0. 符号与输出合成（总览）

【代码事实】S37 一步输出（有事件时）：

```
Δ_root(6)  = root_head([feat_512 ‖ e_0..e_15])          # 不读 prev 参数
Δ_finger_k = joint_head_k([e_{k+1} ‖ θ_k^prev])          # k=0..14
Δ_raw      = concat(Δ_root, Δ_finger_*)
Δ          = Δ_raw + prev_mlp(prev)                        # PREVPOS_EMBED
out        = prev + Δ                                      # PREDICT_DELTA；轴角直接相加
```

出处：`model/model.py:1155-1192`（`_decode_active`）、`:1514-1527`（`prev_mlp` + 门 + 残差）、`:1521-1526`（`ZERO_EVENT_GATE` 仅整包）。

空包：`counts<=0` → `Δ=0` → `out=prev`（bitwise）。【代码事实】`model/model.py:1523-1526`；encoder 侧空包 feat/evidence 亦为零（`event_gnn.py:193-200`、`model.py:1469-1472`）。

---

## 1. prev 到达输出的每一条路径

| # | 路径 | 入口 | 可微？ | no_grad？ | 影响输出维 | 训练时 prev | 测试时 prev |
|---|---|---|---|---|---|---|---|
| P1 | **路由 a** | `_route_nodes` → `route_front_vertex_lbs` → `pool_joint_evidence` → `e` | 否（对 prev）：几何与 a 在 `no_grad`；对 `h` 是 | **是**（FK/投影/a） | 经 `e` 间接影响全部 51：根头读全部 `e_0..e_15`；手指头 k 只读 `e_{k+1}` | GT(start)+课程噪声；路由用该 noisy prev | 段首 GT+小噪声，其后自身输出；闭环纯度↓ |
| P2 | **手指自身角 θ_k^prev** | `joint_heads[k]([e_{k+1}, prev[6+3k:9+3k]])` | **是** | 否 | 仅手指 3 维 `Δ_finger_k`（契约：对其它证据梯度为 0） | 同上 noisy GT(start) | 闭环自生成 |
| P3 | **prev_mlp** | `prev_mlp(prev)` 加到 `Δ_raw` | **是** | 否 | 全部 51（末层零初始化起步） | 同上；学 Wiener 式收缩 | 闭环；关掉则发散 |
| P4 | **残差加法** | `out = prev + Δ` | **是**（对 prev 为恒等通道） | 否 | 全部 51；轴角**加性**非 SO(3) 复合 | 同上 | 同上 |
| P5 | （对照，本臂关）prev_render | `query_render` 进 token | — | no_grad 渲染 | — | S37 `PREV_RENDER: false` | — |

细节核对：

- **P1 路由**：`model/model.py:1384-1392`、`:1499-1500`；`routed_readout.py:40-90`。`d` 与顶点 id **丢弃**，只留 LBS 行×band 门。【代码事实】`routed_readout.py:10-17,86-87`。
- **P2**：`model/model.py:1184-1191`；头结构 `Linear(257+3 → 64 → 3)`：`:868-871`。根头 **不**拼接任何 prev 分量：`:1174-1179`、`:863`。
- **P3**：`:891-898`、`:1514-1520`。配置 `PREVPOS_EMBED: true`（`s37_routed_s3407.yaml:55`）。
- **P4**：`:1521-1526`。轴角加性与 MANO 几何 2π 等价性冲突已在 XYZ 支路 DEBUG 排除表中记录（本任务不展开）。

【推断】梯度回传：训练损失对 prev **本身**无监督（prev 是条件），但对 P2/P3/P4 的参数路径可微；P1 切断对 FK/路由参数的梯度（本臂无学习路由）。

---

## 2. 训练 vs 测试 prev 误差分布

### 2.1 训练：课程噪声混合（逐维写法）

【代码事实】仅 `self.train` 时注入：`dataset.py:353-356`，`prev = prev + noise`（**加性轴角/平移，非 SO(3) 左乘/复合**）。

采样 `dataset.py:221-251` + 配置 `s37_routed_s3407.yaml:76-86`：

| 档 | 概率 | 平移 [0:3] | 根旋 [3:6] | 手指 [6:51] | 块内相关？ | 块间？ |
|---|---|---|---|---|---|---|
| 小 | 50% | iid `N(0, 0.005 m)` | iid `N(0, 0.05 rad)` | iid `N(0, 0.05 rad)` ×45 | 独立 | 独立 |
| 大 | 30% | iid `N(0, 0.05 m)` | iid `N(0, 0.3 rad)` | iid `N(0, 0.3 rad)` ×45 | 独立 | 独立 |
| 相关/定向 | 20% | 随机单位方向 × `U(0, 0.05)` | 同，mag∈`U(0,0.3)` | 同，45 维方向 × `U(0,0.3)` | **块内完全相关（单方向）** | **三块独立** |

混合期望量级（由公式，非实验）：transl RMS ≈ **50 mm**；根旋向量 RMS ≈ **0.30 rad ≈ 17°**；单指维 RMS ≈ **0.17 rad ≈ 9.7°**。【推断】由混合方差算出。

**无时间相关**：每个样本独立抽噪声；窗口起止是 GT(start)/GT(end)，prev 误差与上一包网络输出无关。【代码事实】`dataset.py:337-356`。`UNROLL_PAIR` 本配置未开（§4）。

### 2.2 测试：段首 + 闭环

【代码事实】`eval_track.py:155-195`：每 valid run 一次 `prev = GT(a) + sample_init_noise(..., scale)`；默认 `--init-noise-scale 1.0`（`:364`）→ **仅小噪声档** σ=(5 mm, 0.05 rad, 0.05 rad) iid；之后 `prev ← pred`。

闭环量级（已有产物）：

| 量 | 3407 | 3408 | 出处 |
|---|---|---|---|
| 递推 RA（合并） | 19.26 | 22.15 | `outputs/semkine/closed_loop_s36_s37routed.json` |
| wrist mm p50/p90 | 56.2 / 119 | 75.1 / 102 | `outputs/semkine/probe_s37_route.json` `closed` |
| rot_deg p50/p90 | 10.7 / 17.2 | 11.5 / 18.9 | 同上 |
| 闭环 rot 均值 °（global/local） | 11.3 / 11.1 | （decomp 3408: 11.6 / 12.8） | `.experiments/s37_debug_20260928/decomp_*.json` / `rot_*.json` |
| 手指 RA 分量 local | ~16.8 | ~17.2 | `00_CONTEXT` §9；decomp `ra_art` |

### 2.3 结论：支撑内？时间相关？

- **边际幅值**：测试稳态根旋 ~11°（≈0.19 rad）落在训练小档 σ=0.05 与大档 σ=0.3 之间；腕偏 p50 ~56–75 mm 与训练混合 transl RMS ~50 mm **同量级**，p90 ~100–120 mm 已超过大档一维 σ=50 mm 的典型实现。【实验事实】+【推断】
- **结构不在训练支撑内**：训练误差 = **包间独立、加性、与事件/运动无关的注入**；测试误差 = **自回归、时间相关、与路由几何耦合**（闭环路由纯度 0.54 vs GT 1.0 / 小噪声 0.76，`probe_s37_route.json`）。【实验事实】
- **时间相关性在训练中未出现**：代码路径无 scheduled sampling / residual unroll（本臂）；历史开启过的 unroll 属旧协议负对照（§4）。对应候选机制 **M8**。【代码事实】+【推断】

【待实验验证的假设】H-A1-M8：闭环误差的时间自相关（滞后 1–8 步）显著高于「从训练课程噪声独立重采样」的对照；若把测试 prev 每步替换为「GT + 独立课程噪声」而保持事件，RA 应接近 TF 而非闭环。  
**最小证伪（零训练）**：在已有 ckpt 上跑三轨——(i) 真闭环；(ii) TF；(iii) 每步 `prev=GT+独立 mixed 噪声`（同分布重采样）。若 (iii)≈(i) 则推翻「时间相关是主因」；若 (iii)≈(ii) 则支持 M8。

---

## 3. 时间归一化与尺度歧义

### 3.1 Token（包内相对 + 一条绝对）

【代码事实】`encoder.py:153-170`：

| 维 | 公式 | 时间基准 |
|---|---|---|
| t_norm | `t / Δt_packet` | **包相对** |
| log_dt_ie | `log1p(Δt_ie / 1e-6)` | **绝对**（秒级间隔，ε=1 µs，`EPS_DT`） |
| SAE_same/opp | `(t_since_last_pol / Δt).clamp(0,1)` | **包相对**（未触发时 fallback=包跨度） |

事件时间戳本身：窗内相对起点的 µs（`dataset.py:276-280`；评测 `eval_track.py:69-76`），再 /`delta_t_s`。

### 3.2 建图

【代码事实】`event_gnn.py:212-217`：图坐标第三维 = `t_norm * t_scale`，配置 `ENCODER_T_SCALE: 1.0` → **一个包长 = 一个画面宽**。近邻窗是时间序上前 32 个**节点**（`:149-176`），不是固定毫秒。

### 3.3 对速度/运动特征的含义

- 训练窗对数均匀 **30–300 ms**，测试固定 **50 ms**（yaml DATA/EVAL）。同一物理角速度：在 300 ms 包里 `Δt_norm` 更小、图上时间距离被压扁；在 30 ms 包里被拉长。【推断】
- EdgeConv 的 `dp[...,2]` 与 SAE/Δt 都随包长伸缩 → **运动快慢与窗长纠缠（尺度歧义）**。【推断】
- **绝对时间线索仅 `log1p Δt_ie`**（及由其驱动的任何隐式速率）。它不随 `WINDOW` 显式归一化，但依赖包内事件率。【代码事实】`encoder.py:167`
- §9.3 窗口 50→300 ms 递推几乎不动、根旋不变：与「感受野/归一化已按包伸缩、加长窗不增加绝对时间信息」一致。【实验事实】`00_CONTEXT` §9.3

【待实验验证的假设】H-A1-scale：把 `t_norm`/`SAE` 改为绝对秒（或固定 50 ms 参考）而保持图拓扑，TF 根旋应变；若不变则推翻「尺度歧义是根旋瓶颈」。  
**最小证伪（零训练优先：特征探针）**：固定 ckpt，对同一批事件分别用 Δt=50/150/300 重算 token（只改归一化分母），看 root 输入 z 与 TF Δ_rot 的变化；再可选 ridge 解码。

---

## 4. 跨包状态与 UNROLL

### 4.1 跨包只有 51 维

【代码事实】评测环：`prev_t = pred`（`eval_track.py:195`）。`forward_packet` 无 LSTM/GRU/scan 隐状态；EventGNN 包间独立（`event_gnn.py:178-245`）。`route_stats` 仅为诊断字典。MANO betas/K 是序列常数上下文，不是递推状态。  
⇒ **除 51 维 `prev_state` 外无跨包可学习状态。**

### 4.2 UNROLL / GAIN_REG：代码有、本臂未开

【代码事实】`model/model.py:231-275` `_maybe_unroll`：

- 数据集需 `UNROLL_PAIR` 返回 `(lead, main)`（`dataset.py:382-387,546`）。
- 以概率 `UNROLL_P`（可 ramp）把 main 的 `prev_state` 换成：  
  - **replacement**：`lead` 前向（`no_grad`+detach）的预测；  
  - **residual**：`(GT+课程噪声) + (lead_pred - lead_target)`。
- `GAIN_REG_W`：对条件扰动的保留率惩罚（`:324+`）；S37 yaml 无此键 → 0。

S37 配置无 `UNROLL_*` / `GAIN_REG_*` → `unroll_p=0`，`unroll_pair=False`。【代码事实】`model.py:640-658`；`s37_routed_s3407.yaml`。

### 4.3 为何没开（历史）

【实验事实】`docs/FAILURE_AND_CLEANUP_LEDGER.md` 表内 E5.5b：恒定 `UNROLL_P=0.5` 无退火 → TF RA ~42.5 / 递推 ~144.5，坍缩回绝对回归（~40 mm 档）。  
【代码事实】docstring `model.py:245-254`：replacement 让 TF 变好但递推变差（网络学「prev 可信」）；residual 形式为 E5.5c 补救设计。  
【实验事实】`docs/POSITIONING_VS_E3DPSM.md`：`s22_keg_halo_unroll` 单步最好、G 与递推最差。  
`docs/debug_e55b_unroll_20260825.md` **已删**（账本 §224）；结论以上述账本/docstring 为准，标「已删原文未复核」。

---

## 5. 评测段首与 zgz_local

【代码事实】噪声量 = `TRACK.PREV_NOISE_{T,R,POSE}` × `init_noise_scale`（默认 1.0）→ **5 mm / 0.05 rad / 0.05 rad** iid（`eval_track.py:118-124,159`）。**不是**训练的 30% 大噪声 / 20% 定向。

【实验事实】zgz_local：**90 段、均 ~0.7 s**，每段从 GT+小噪声起步（`S37_ROUTED_READOUT_PREREG.md` §8；`00_CONTEXT` §2）。global：**1 段 69 s**。

对 local 列的影响：

- 短段频繁近 GT 复位 → local 数字含大量「段首附近」帧（prereg：33% 帧在初始化后 1 s 内）。【实验事实】prereg §9
- 「保持段首」local RA **28.2**；闭环 23.3/29.5——3408 **差于**不更新。【实验事实】`.experiments/s37_reanalysis_20260928/regime_*.json`；prereg §8
- abs：保持段首 local **27.9** vs 闭环 89–110 → 模型在乱动平移；主表 RA 列掩盖此点。【实验事实】prereg §8
- 【推断】主表 local 优于 baseline 的部分增益来自**协议短段复位**，不是长时跟踪能力；与 global 不可混读（prereg 已写）。

---

## 6. 空包门 vs 单关节未路由

【代码事实】`ZERO_EVENT_GATE` **只看整包** `counts<=0`（`model.py:1523-1525`）。无逐关节门（对照 mesh_query 有 coverage 门，本臂无）。

单关节无节点：`pool_joint_evidence` 中 `wsum=0` → mean 置零、peak 置零、`coverage=0`（`routed_readout.py:103-118`）。  
手指头仍算：`Δ_k = f(0_{257}, θ_k^prev)`，再加 `prev_mlp` 对应维，再 `+ prev`。【代码事实】`model.py:1190-1191,1514-1526`。  
⇒ 用户表述 **「f(0, θ_k) + prev_mlp」再经残差加 prev** 正确（残差不可省）。

**频率**：`outputs/semkine/probe_s37_route.json` 仅有节点级 recall/purity（`tf.routing`），**无**「每关节 coverage=0」频率。【未能核实】需 CPU 探针：对 zgz_local 每包统计 `(count[:,1:]==0)` 均值（`route_stats["route_joints_hit"]` 只给均值命中关节数，训练日志若未落盘则现算）。

粗界【推断】：local 中位 ~660 事件全保留、band 16 px；多数包应命中多数关节，但稀有关节/遮挡/腕偏大时 coverage=0 仍可能发生——**无数字不写进结论**。

【待实验验证的假设】H-A1-miss：local 上指尖关节 `coverage=0` 的包占比 ≥10%，且这些包上该指 TF 误差主要由 prev_mlp+残差解释。  
**最小证伪**：现有 ckpt + 统计 `count==0` 频率；对 miss 子集比较 `joint_head` 输出范数 vs `prev_mlp` 范数。

---

## 7. 信息缺口清单（进入/融合/训测不一致）

| ID | 现象 | 标签 | 【待实验验证的假设】 | 最小证伪（优先零训练） |
|---|---|---|---|---|
| G1 | 根头不读 `r_prev`/`t_prev`/K/深度 | 代码 `model.py:1174-1179` | 根旋误差主因是缺几何条件而非事件不足（M3） | 已有 `info_*.json`：`z+r_prev` 仅 7.5–7.6° vs hold 9.2；若再拼 oracle 深度才到 ~5°——与「只缺 r_prev」不完全吻合。**追加**：强制 root_head 输入拼 `prev[3:6]` 的线性探针（冻主干） |
| G2 | 路由丢弃 d、vid | `routed_readout.py:10-17` | 距离/深度残差未进头导致常数增益融合（M1/M3） | 已有完美路由 z 的 R²≈0（`info` zroute）；拼 2D 残差可降。重复该探针即可，不必训练 |
| G3 | 事件证据进根头但旋转纠正极弱（增益 0.04–0.13） | `probe_s37_route` root_gain | 事件路径对根旋近于噪声，prev_mlp 主导 | TF 上 `heads` vs `pmlp` vs `noev`（decomp 已有）：noev≈full，heads 单独更差 → **已强支持**；可用「置零 feat 保留 e」再拆一次 |
| G4 | F 与 G 大、近正交、相消 | `rot_*.json` tf_step_heads/pmlp | 融合是对抗抵消而非观测−预测 | 记录 `cos(Δ_F, Δ_G)` 与 `‖Δ_F+Δ_G‖/‖Δ_need‖`；若相关化（同向）应升——需改结构才测，零训练只能确认抵消统计 |
| G5 | 手指头看不到父链/兄弟证据 | `model.py:1190` + 契约测试 | 缺父链导致局部轴角歧义（M4，A3 深挖） | 本文件不展开内部；交叉：coverage=0 时是否更糟（G6） |
| G6 | 无逐关节门；miss → 仍输出非零 Δ | §6 | miss 关节被 prev_mlp 乱推 | 见 H-A1-miss |
| G7 | 训练独立噪声 vs 测试时间相关误差 | §2 | **M8** 是闭环−TF 差距主因 | 见 H-A1-M8 三轨 |
| G8 | 包相对时间归一化 | §3 | 尺度歧义限制运动特征（M5 相关） | 见 H-A1-scale |
| G9 | 未开 residual unroll | §4 | 暴露偏差可经 residual unroll 缓解 | 旧协议曾失败；**新协议零训练无法证伪**，属训练实验。可先测「用闭环 prev 分布的离线噪声模型」重放（仍零训练） |
| G10 | local 段首协议美化主表 | §5 | local 列不测长时跟踪 | 已有 hold/copy/闭环对照；**按 elapsed 分桶**（eval 已写 drift）看 local 是否随段内时间变差——零训练重跑 eval 即可 |
| G11 | 轴角加性残差 | `out=prev+Δ` | 大误差时加性更新与几何不一致 | XYZ DEBUG 已部分排除 π 跳；可在闭环大腕偏子集上比「加性 Δ」与「SO(3) 复合」oracle 重写输出（后处理，零训练） |

---

## 8. 路径可微性一览（答问 1 压缩）

```
prev ──► [no_grad] FK+投影+a ──► e ──► root_head / finger_heads     (P1，条件不可微)
     └─► θ_k ─────────────────────► finger_heads                     (P2，可微)
     └─► prev_mlp ────────────────► +Δ                               (P3，可微)
     └─► +Δ ──────────────────────► out                              (P4，可微恒等)
整包空 ──► Δ:=0 ──► out=prev                                         (门，不可学习)
```

---

## 9. 未核实 / 边界

1. zgz_local **逐关节** `coverage=0` 频率：现有 JSON 无此字段。  
2. `debug_e55b_unroll_20260825.md` 已删；细节以账本 + `model.py` docstring 为准。  
3. 训练混合噪声的 RMS 为解析近似，未对 dataloader 实证直方图。  
4. 未重跑任何 eval；数字均引自既有产物路径。
