# 778 顶点事件引导图 + 局部记忆

2026-09-23。这一条线的唯一入口：无记忆初版（未训练）→ debug → 用户确认局部记忆 tracking → 双种子训练失败。
臂名 `event_guided_mesh_memory`；无记忆配置 `event_guided_mesh` 只作实现对照，没有主行。
旧 `s37_meshgraph` 的主行不代表本结构。对照仍为 S37 路由读出。

网络概览：[PNG](assets/event_guided_mesh_memory_overview.png) / [SVG](assets/event_guided_mesh_memory_overview.svg)。

---

## 1. 无记忆初版（`ENCODER=event_guided_mesh`，未训练）

配置 `configs/semkine/event_guided_mesh_s3407.yaml`。prev 只经 MANO FK 进入；事件包驱动 778 顶点图；固定 LBS 均值到 16 关节再预测 Δ。
没有 `prev_mlp`、头部 prev 角、背景直达 root 或节点 ID 嵌入。`target` 不参与构图或前向。

```text
s(k-1) -- MANO FK --> V(k-1): [778,3] ----+
                                        |
E_k: (x,y,time,polarity) --> 2D归属 --> 事件引导的完整778顶点图
                                        |
                              3D图卷积 x3: [778,128]
                                        |
                         固定LBS归一化池化: [16,128]
                                        |
                     root头 + 15个关节头 --> Δ: [51]
                                        |
                         s(k) = s(k-1) + Δ
```

构图（初版公式，debug 后记忆臂已改邻接规则）：全部 778 节点保持 MANO 编号；投影 + 背面剔除 + 点 z-buffer 归属；每点 6 维观测；最近 32 候选上

`score(i,j) = ||V_i-V_j||²/(0.02 m)² + mean_c((o_ic-o_jc)²)`，

取 8 个来源。初始特征 `ReLU(Linear(o_i))*has_event_i`；仅有支持的来源发消息。LBS 分母是关节在全部 778 顶点上的固定权重总量。空证据时特征与 Δ 恰为零。FK / 归属 / 离散邻接 / 相对几何 `no_grad`。

物理时间沿用旧协议，不是严格 `(t-1,t]`：事件窗 `[start_ms, end_ms+1)`，标签在 `end_ms`，窗右端与独立训练 / 递推条件各差 1 ms。图中的 `k` 是递推步号。

契约：`semkine/event_guided_mesh.py`、`tests/test_event_guided_mesh.py`。初版 + LUT 边界共 54 项与旧臂回归通过；未启动该配置的完整训练。

---

## 2. Debug（无记忆版，无精度结论）

产物 `.experiments/egm_debug_20260923/`。真实包取 `lyq_local` / `lyq_global` 各 24 个 50 ms 窗 × clean/small/large，共 144 例；prev = 窗起点 GT + 受控噪声，不是模型递推。

**缺观测被当成真实零观测。** `mean((o_i-o_j)²)` 让空节点之间分数更低，随后 `supported_edges` 又禁止空来源发消息。合成直线 34 点、仅顶点 0 有观测：默认规则三层支持 `1→1→1→1`，关掉事件重排则 `1→5→9→13`。接收点 1 上，有事件的最近点 0 分数约 0.16917，较远空点 9 约 0.16000。MANO 模板仅源点 304 有观测时默认仍只支持它自身，几何对照可到 36 点。

真实包 clean：`lyq_local` 初始支持 245.92 → 三层 659.46（几何对照 751.17）；`lyq_global` 359.33 → 726.13（761.79）。large：local 372.46 对 485.88，global 455.67 对 549.38。事件确实改边（clean 有向边替换约 14.08% / 16.58%；约 58.97% / 68.05% 的接收点邻居集合改变）。建议的下一次单变量：仅两端都有观测时用事件差；缺观测接收点走几何通路。记忆臂已按此改邻接，见 §3。

**LBS 固定分母稀释稀疏证据；二值门不表达强度。** 顶点 226 对关节 3 原始权重约 `1.03868e-5`，归一化系数约 `1.89849e-7`，门仍全开。large 下平均每包 1.92 / 1.54 个关节支持质量 <1% 但门开。每点只覆盖 1–6 个关节列，不是“一点打开全部 16 门”。

**欧氏图可跨指。** 接收点 394 与来源 473 距约 9.95 mm，跨不同主导手指组、间隔 8 个网格边跳。clean 跨非根手指组边约 4.60% / 1.57%。

**LUT 越界已修。** `assign_events_by_lut` 原先 round+clamp，越界/非有限坐标可能落到边缘手顶点；现非法坐标进背景，合法输入逐位不变。jump-flood 仍是近似：同一圆整像素只留最低可见顶点 ID；clean 每包约 14.33 / 9.67 个可见种子碰撞，抽检归属不一致约 2.92% / 1.14%。

**工程。** L20 / Torch 2.1 / bf16；B=512 峰值 allocated ≈ 9.14 GiB。Linear 为 bf16，EdgeConv 归约在 autocast 下为 fp32。未跑 DDP 或完整训练，不报主表行。

---

## 3. 局部记忆 tracking（已训练臂）

用户确认方案 A：778 顶点局部记忆，允许随整手和关节运动。文献（Mesh Graphormer / METRO / EvRGBHand / E-3DPSM）提供空间建模或历史融合的启发，**不**提供“未观测 MANO 顶点绝对坐标冻结”。局部 hidden 保持 ≠ 相机系 xyz 冻结；最终网格仍由 `s(k)` 的完整 MANO 重建。

配置 `configs/semkine/event_guided_mesh_memory_s3407.yaml`。状态：姿态 `s:[B,51]` 与 `H:[B,778,128]`、`seen:[B,778]`，均由调用者持有。

邻接（相对初版的单变量）：每点排除自身后取最近 32 候选；保留最近 4 个几何来源，再补 4 个当前有观测来源；不足用未选几何补齐。无观测 query 不用全零 `o` 当真实测量。`EGM_EVENT_WEIGHT=0` 定义为纯几何最近邻，是消融开关。

```text
proposal_i   = GRUCell(F_i, H_prev_i)
H_next_i     = where(m_i, proposal_i, H_prev_i)
innovation_i = where(m_i, H_next_i − H_prev_i, 0)   # 诊断，不进 LBS
seen_next_i  = seen_prev_i OR m_i
readout_i    = where(m_i, H_next_i, 0)
```

`m_i=1` 当且仅当本包事件计数通道 > 0。只池化当前观测门控的 `H_next`：GRU 在不同非零固定点处差值都趋零，只读差值会丢持续运动。空包 / 证据消融时姿态与记忆逐位保持；有观测驱动运动时，未观测顶点随蒙皮走。

`TRACK.UNROLL_PAIR=true`：`lead→main` 传递预测姿态与记忆，不 detach，只监督 main 末端；main 的 GT prev 不进前向。Torch 2.1 CUDA fused GRUCell 不支持 bf16，仅循环单元关 autocast。Δ 头最后一层权重 ×`1e-3`。接口：`model.track_packet(packet, node_state)`；每个 valid run 重置。

公共记忆层 7 项 + 整网 / 两包 / 重建契约共 96 项通过。`semkine/mesh_memory.py`、`tests/test_mesh_memory.py`。

---

## 4. 训练登记

精度门在启动前固定：对照 `outputs/semkine/s37_routed_main_row.json`（SHA-256 `dd9e7b3de73dbf804766b7491ebd0d2945be1ba541ca3c462940bf0329acaff8`），双种子递推 RA 均值 < 20.7432807742 mm，且 3407 ≤ 19.2296450331、3408 ≤ 22.2569165152。

- 种子 3407 / 3408；每种子 2 卡 × 512 连续包对；Adam 0.004，warmup 500，共 6000 步；固定 500 步网格、zgz 递推选点。
- 启动 **2026-09-23 15:15:59 +08:00**，launcher PID `1359394`；3407 用 GPU 0–1，3408 用 GPU 2–3。
- 产物：`outputs/semkine/event_guided_mesh_memory_s340{7,8}/`（含 `config.yaml`、`source_sha256.txt`、`source_snapshot.tar.gz`）；日志 `logs/event_guided_mesh_memory_*.log`；工程目录 `.experiments/egm_memory_20260923/`。
- 真实 B512 两包约 38.1M 事件，峰值 allocated 18.19 GiB。选点于 19:12:07 全部结束。

两种子均选 step=500，不是最后一步：

- 3407：选中 RA 64.2484913182 mm（step=6000 为 95.2638521762）；权重 SHA-256 `d1bc4521bc0042801d99b94cc7fe2700be42c5faf8208e1ea842759a920fab44`。
- 3408：选中 RA 93.3180205474 mm（step=6000 为 102.8028294744）；权重 SHA-256 `b4f4f5fe53fbe581192b6d020c40e5d90070de7bbbff66c67bdf1353e25ab691`。

末期递推比最早保存点更高。两包训练 vs 长序列评估的状态分布差异未做受控归因。

---

## 5. 正式主行与判定

表由 `python tools/report_table.py s37_routed event_guided_mesh_memory` 读取主行生成。准确率 mm，双种子均值；RA 括号为 3407 / 3408。

| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S37 路由读出（对照） | 26.41 | 15.82 | 21.40 | 12.39 | 20.74（19.23 / 22.26） | 7.72 ms | 0.827 G | 0.73 M |
| 778 顶点图 + 局部记忆（本次） | 58.87 | 96.08 | 50.84 | 88.96 | 78.78（64.25 / 93.32） | 20.31 ms | 0.392 G | 0.41 M |

判定：预注册精度门失败，不采纳；当前臂仍为 S37 路由读出。主表复评与选点逐位一致，漂移 0。

- 主行：`outputs/semkine/event_guided_mesh_memory_main_row.json`。
- 记忆臂耗时在 valid run 内传递姿态和顶点记忆；对照臂未重测，状态条件不同，不能当严格速度消融。
- THOP 全模型 MACs 含 GRUCell，不含离散构图 / 投影归属 / scatter / MANO FK。
