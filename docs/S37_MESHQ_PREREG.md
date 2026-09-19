# S37 网格查询（用户设计）：prev 只以 FK 网格进入，网格读事件图、驱动关节——预注册

> 2026-09-07。这是用户明确要求的 S37 设计（"prev 生成 MANO FK 就足够了；事件建图得到节点和边信息，再去驱动 FK 变化；
> 逐事件 token 去掉"）。之前同日的 `s37_routed`（事件按 LBS 路由到关节、prev 还走 prev_mlp 与关节头）不是用户的设计，
> 作为中间记录保留在 `S37_ROUTED_READOUT_PREREG.md`。门槛与假设在训练启动前写定；§6 结果与 §7 判读训完后填。
>
> 协议：9 受试者 72 条序列训练（默认 `splits_semkine.json`），验证、选点、上报只用留出受试者 zgz 的两条序列（2590 帧），
> 50 ms 步长，固定 500 步网格，`tools/select_checkpoint.py` 递推 RA 选点，两种子 3407 / 3408。对照：现有 S36 zgz run
> 与同日的 `s37_routed`，都不重训。

---

## 1. 设计（`semkine/mesh_query.py`，`MODEL.MESH_QUERY`）

- **prev 只以 FK 网格进入。** `PREVPOS_EMBED: false`（无 prev_mlp），关节头不读 prev 角，节点上无渲染通道。网络里出现 prev 的唯一地方
  是 MANO FK 之后的 778 个顶点及其投影。
- **事件直接建图，无 token。** 节点属性只有事件本身 `(x, y, p, t)`（`ENCODER_NODE_ATTRS: raw4`；S36 的 token 还有 log 事件间隔与两个
  SAE 派生量），因果 k-NN（前 32 里取 k=8，≤2048 节点）、EdgeConv×3 与 S36 相同；不建全局池化读出（DDP 不允许无梯度参数）。
  消息传递后得到节点特征 `h_i`（128）和每节点**边摘要** `g_i = [mean(dx, dy, dt), mean|dp|]`——图自己对局部运动的读数。
- **网格读图。** 192 个固定查询顶点（每关节 12 个，按主导蒙皮权重分层选取，确定性），投影到事件帧；每个**可见**查询顶点
  （被 3 px 内更近的顶点遮挡即剔除）用固定核 `w_qi = softmax(−d²/2·8²)`，只在 16 px 带内的活节点上，聚合
  `v_q = [Σw h_i ‖ Σw g_i ‖ Σw (p_i − u_q)/16 ‖ mass_q]`（137 维）。核宽是常数：S38 的可学习 σ 被课程摊到 365 px，
  KEG 的学习路由把回路增益推到 0.91，这里没有任何参数可被优化器用来放宽。
- **驱动关节。** 蒙皮权重把顶点证据带到关节：`e_j = Σ_q W[q,j]·has_q·v_q / Σ_q W[q,j]·has_q`，`cov_j = 该关节表面被看见的比例`；
  关节头 k 只读 `e_{k+1}`（138 维，含 cov），输出乘 `1[cov_{k+1} > 0]`——**表面没有事件的关节不动**（逐关节的零事件恒等）；
  root 头读 `[网格聚合 138, e_0..e_15 按关节顺序平铺]`，乘 `1[任一关节被看见]`。空包 → 一切为零 → 逐位返回 prev。
- 与 S36 的差异（`test_s37_meshq_config_diff_against_s36`）：`PREV_RENDER`、`PREVPOS_EMBED`、`ENCODER_NODE_ATTRS`、`MESH_QUERY`（+3 旋钮）。
  其它一切（采样、图、EdgeConv、loss、课程、网格、DDP）S36 逐字相同。
- 契约（`tests/test_s37_meshq.py`，8 项）：查询集每关节等量；核只在带内且剔除被遮挡顶点；未被看见的关节证据与覆盖恰为零；
  解码器门控（未观测关节 Δ 恰为零）、无旁路（关节 k 只对 e_{k+1} 有梯度，root 读全部）；prev 只经 FK 进入（无 prev_mlp、头不读 prev 角、
  节点 4 维、无池化读出）；空包逐位返回 prev；`ablate_evidence`（"网格什么都没看见"）退化为输出 = prev。
- 冒烟（单卡 512 包 60 步）：峰值显存 23.9 GiB（S36 24.1），约 0.97 s/it。

```mermaid
flowchart LR
  prev[prev 51D] --> fk[MANO FK 778 顶点 投影]
  fk --> q[192 个可见查询顶点]
  ev["异步事件流 (x,y,t,p)"] --> graph["建图 kNN + EdgeConv x3 无 token 无渲染"]
  graph --> h[节点特征 h_i]
  graph --> g["边摘要 g_i = mean(dx,dy,dt)"]
  h --> gather["固定核聚合 v_q = [h, g, 偏移, mass]"]
  g --> gather
  q --> gather
  gather --> lbs["蒙皮权重 -> 关节证据 e_j, 覆盖 cov_j"]
  lbs --> heads["关节头 k 读 e_k+1 x 1[cov>0]; root 读网格聚合 + 16 证据"]
  heads --> delta[Δθ]
  delta --> out[x_k = prev + Δ]
```

## 2. 预注册门槛（zgz，两种子）

参照：S36 递推 RA **23.17**（21.42 / 24.92），abs 79.0；`s37_routed` **20.74**（19.23 / 22.26），abs 75.8。

- **精度**：对 S36 采纳需两种子均值 RA ≤ 22.07 且逐种子不劣于 S36 同种子，abs ≤ 81。与 `s37_routed` 的比较只记录，|Δ| < 1.1 判打平。
- **机制门**：`ablate_evidence`（网格什么都没看见 → 输出 = prev，即"不动"基线）的闭环 RA 必须比正常闭环差 ≥ 1.5 mm。
- 无论结果如何写入本文 §6 与账本；同时报告与 CNN+渲染基线（11.9–13.3）的距离。

## 3. 命令

```bash
python -m pytest tests/test_s37_meshq.py -q
nohup tools/run_zgz_protocol.sh s37_meshq > logs/run_s37_meshq_outer.log 2>&1 &
python tools/make_s36_row.py --run s37_meshq
CUDA_VISIBLE_DEVICES=6 python tools/run_closed_loop_probe.py --split val_core --out outputs/semkine/closed_loop_s37meshq.json \
    --arm s37m_3407=<ckpt>:configs/semkine/s37_meshq_s3407.yaml --arm s37m_3408=<ckpt>:configs/semkine/s37_meshq_s3407.yaml
CUDA_VISIBLE_DEVICES=6 python tools/probe_s37_route.py --runs outputs/semkine/s36_eventgnn_s3407 outputs/semkine/s36_eventgnn_s3408 \
    outputs/semkine/s37_meshq_s3407 outputs/semkine/s37_meshq_s3408 --out outputs/semkine/probe_s37_meshq.json
```

## 4. 变更记录

- 09-07 下午：实现、8 项契约、冒烟；本文写定；训练启动时间见 §6。

## 5. 本质原因：预注册假设与判别探针（`tools/probe_s37_route.py`，对本臂同样适用）

| 假设 | 测什么 | 成立的判据 |
|---|---|---|
| **H1** 课程大噪声下网格看不到正确的节点，头学会少动 | 节点→关节责任（注意力质量按蒙皮权重分配）的召回/纯度在 prev = GT / 小噪声 / 大噪声 / 闭环 四种条件；证据置零（= 不动基线）与换事件后的位移 | 大噪声召回 < 0.5，且正常闭环对"不动"基线的优势 < 1.5 mm |
| **H2** 网格对图的状态条件读取在闭环自放大 | 放大率对 S36（2.58）与 `s37_routed`（2.12）；oracle 几何（GT prev 做网格、其余自预测） | oracle 收回对 S36 差距的一半以上，或放大率 > 2.6 |
| **H3** root 盲 | 腕平移 / 全局旋转分解；prev 平移 36 mm、旋转 10° 时 root 头自身响应增益（本臂无 prev_mlp，响应全部来自网格读数） | 旋转增益 < 0.3 且腕误差 ≥ S36 + 5 mm |
| **H4** 不跨受试者 | 4 条训练序列与 zgz 的 TF 单步差 | 差 ≥ S36 的 1.5 倍 |
| **H5** 过冲 | `update_ratio_pose` / `alignment_pose` | update_ratio > 1.3 或 alignment < S36 |

额外要回答的（本臂特有）：去掉 prev_mlp 与关节头的 prev 角之后，root 与手指的更新是否仍能表达"prev 的姿态先验"——若闭环旋转 p50 明显
劣于 S36 的 12°，读作先验缺失；若 TF 单步差不多而闭环更差，读作"没有 prev_mlp 的阻尼"。

## 6. 结果（待填）

## 7. 判读（待填）
