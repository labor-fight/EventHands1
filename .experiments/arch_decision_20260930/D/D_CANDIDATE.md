# D：抽样/邻域/时间感受野 — 唯一候选

## 结论

**候选**：`ENCODER_WINDOW` 32→256（只改因果候选窗；k=8、层数=3、抽样=等步长≤2048 不变）。

**可证伪假设 H**：因果窗过窄使 kNN 被迫在同毫秒内取远邻，三跳时间 RF≈2 ms≪50 ms；扩大窗到 256 后，同 k 能选到更近空间、跨毫秒的前驱，三跳时间跨度升到 ≥5 ms，从而改善递推 RA（尤其 global）。

## 证据位置

| 来源 | 要点 |
|---|---|
| A4 `a4_encoder_audit.json` | hop3 时间 1.87 ms(g)/4.64 ms(l)；边 dt 中位 0；保留率 g 0.074 / l 1.0 |
| `docs/S37_EDGE6…§5.1/§7` | 加深到 6 层 RF 仅→4 ms；递推 RA 21.38 打平（种子分裂） |
| 本目录 `nbr_window_probe.json` | **W256 时 61.8% 选中边 offset>32**；h3_span 1.87→7.06 ms；边长 13.4→6.3 px |
| `docs/网络结构分析.md` §10 P-B / E8 | 已登记「WINDOW 32→256」为感受野单变量 |

## 为何不选其他（一次一因素）

| 备选 | 否决理由 |
|---|---|
| 加深 EdgeConv | edge6 已打平；层数≠候选窗宽度 |
| 体素均匀抽样 (EV-VGCNN) | 改节点集（E9），与邻域正交；local 已全保留，不解释「62% 真邻被 W32 挡掉」 |
| 物理半径邻域 | 同时改度量与选边规则，非单变量；W256 已让现有 (x,y,t) kNN 偏好近邻 |
| 层级池化 | 新模块+多跳抽象，非单键；X1c 曾混变量 |

## 数据流与张量（相对 S37 仅 `_edges` 的 `window`）

```
events (E,≥4) + ptr (B+1)
  → _sample → src (B,N≤2048), mask (B,N)
  → p (B,N,3) = (x/W, y/H, t_norm·t_scale)
  → _edges(window=W): cand (N,W) → d (B,N,W) → topk k=8
       → idx (B,N,8), dp (B,N,8,3), emask (B,N,8)
  → 3× EdgeConv → h (B,N,128) → 路由读出（不变）
```

## 代码改动（不碰 `model/model.py`）

- **已有键** `MODEL.ENCODER_WINDOW`（默认 32，`model.py:896` / `EventGNN.window`）。新臂 yaml 写 `256`；S37 保持 32 → **逐位不变，无需新键、无需改 model.py**。
- 触及函数：仅 `semkine/event_gnn.py::_edges` 读 `self.window`（行为随配置变；实现可不改）。契约：`window=32` 与现权重前向 max|Δ|=0。
- 新配置：`configs/semkine/s37_win256_s340{7,8}.yaml`（其余字段 = s37_routed）。

## 匹配对照

同日程 6000 步、种子 3407/3408、同数据顺序/课程/损失/选点网格；对照 `s37_routed` 主行不重训。采纳门同 edge6：均值 RA≤19.64 且两种子均不劣于同种子 S37。

## 机制指标（训前/训后，CPU 可复用 `nbr_window_probe.py`）

1. zgz_global：`frac_edge_off_gt32`（W256 期望 ≈0.6）；`hop3_dt_ms` 中位 ≥5 ms（现 1.87）。
2. 边长中位下降（探针 13.4→6.3 px）。
3. 主表 + 可选：闭环根旋 / oracle 根后手指 RA（不写进报告表）。

## 成本估计

| 项 | S37 (W=32) | W=256 |
|---|---|---|
| EdgeConv MACs | ≈0.824 G（k 不变） | **同** |
| `_edges` 距离搜 | O(N·32) | O(N·256) ≈ **×8**（thop 通常不计） |
| 报告 FLOPs/step | 0.827 G | ≈0.827 G |
| Latency | 7.72 ms | 预计 +0.3–1.5 ms（建边） |
| 显存 | ~24 GiB @512 | cand 缓冲 ×8；峰值或需 batch 256×accum2（同 edge6 预案） |
| Params | 0.73 M | **不变** |

## 训练预算

2 种子 × 6000 优化器步，2 卡/任务，约 2×~2 h（若显存吃紧则微批同 edge6）。先 H4 逐位契约 + H5 冒烟再开训。

## 淘汰条件

- 机制门失败：W256 下 hop3_dt 中位仍 <4 ms，或 `frac_edge_off_gt32`<0.2 → **图假设错，停训/不采纳**。
- 主表：均值打平（\|Δ\|<1.1）或变差，或任一种子相对同种子 S37 退化 ≥1.1 mm → **淘汰，保持 s37_routed**。
- 显存无法在有效 batch=1024 下稳定 → **改微批后再判；仍过不了成本则淘汰该窗宽**。

## 最小实验（唯一）

`s37_win256`：仅 `ENCODER_WINDOW: 256`，其余同 S37；双种子全量 + `make_s36_row` + `report_table.py`。机制探针复用本目录 `nbr_window_probe.py`（已出 W32/W256 基线 JSON）。
