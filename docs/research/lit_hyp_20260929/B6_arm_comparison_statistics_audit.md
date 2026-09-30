# B6：跨臂比较与统计稳健性审查（2026-09-29）

> 研究子代理 B6。只读 JSON / 配置 / 文档 / 日志；CPU 分析脚本在
> `.experiments/lit_hyp_20260929/B6/`（`analyze_grids.py` → `grid_stats.json`）。
> 不改仓库代码、不训练、不用 GPU。证据标签见 `00_CONTEXT.md` §1。

---

## 0. 范围与产物

| 臂 | 主行 | 选点网格 | 预注册 |
|---|---|---|---|
| S36 EventGNN | `outputs/semkine/s36_eventgnn_main_row.json` | `s36_eventgnn_s340{7,8}/selection_val_core_step50.json` | **无独立 prereg**（设计写在 S37 routed §1） |
| S37 routed（当前臂） | `s37_routed_main_row.json` | `s37_routed_s340{7,8}/…` | `docs/S37_ROUTED_READOUT_PREREG.md` |
| S37 fkgraph | `s37_fkgraph_main_row.json` | `s37_fkgraph_s340{7,8}/…`（另有 `*_fp32atomics` 旧选点） | `S37_FKGRAPH_PREREG.md` |
| S37 meshgraph | `s37_meshgraph_main_row.json` | `s37_meshgraph_s340{7,8}/…` | `S37_MESHGRAPH_PREREG.md` |
| S37 rootinnov | `s37_rootinnov_main_row.json` | `s37_rootinnov_s340{7,8}/…`（6 点网格） | `S37_ROOT_INNOVATION_PREREG.md` |
| S37 rotw10 | `s37_rotw10_main_row.json` | `s37_rotw10_s340{7,8}/…` | `S37_ROTW_CNNROOT_PREREG.md` §A |
| S37 meshq | **无主行** | `s37_meshq_s340{7,8}/…` | `S37_MESHQ_PREREG.md`（§6–§7 空） |
| CNN 诊断 `s37diag_cnnabs` | 无主行（诊断） | `s37diag_cnnabs_s340{7,8}/…`（8 点） | 同上 §B |
| rotw30 | 无主行（单种子诊断） | `s37_rotw30_s3407/…` | 同上 |

统计复算产物：`.experiments/lit_hyp_20260929/B6/grid_stats.json`。

---

## 1. 比较表：变量隔离与预算

选点方式（各臂共用，除另注）：`tools/select_checkpoint.py` 在固定网格上取 **递推 `mpjpe_ra_mm` 最小**；
`val_core` = zgz 两条 / 2590 帧 / 50 ms；**选点集 = 上报集**。
评测段首噪声：`TRACK.PREV_NOISE_{T,R,POSE}=0.005/0.05/0.05`（S36 / S37 / CNN yaml 相同）；
主行复现 `rng = np.random.default_rng(0)`（`tools/make_s36_row.py:81`）。

| 比较 | 改变的变量（YAML 实测） | 参数 / FLOPs / 延迟（主行） | 种子 / 步数 / 网格 | 是否隔离声称模块 |
|---|---|---|---|---|
| **S37 vs S36** | `PREV_RENDER True→False` **且** `ROUTED_READOUT` + band/k | 0.73 M / 0.827 G / 7.72 ms vs 0.95 M / 0.827 G / 7.04 ms | 3407/3408；6000 步；500…6000×12 | **否：两处同时改**（prereg 自称「恰为两处」，不是单因子） |
| **fkgraph vs S36** | 整编码器 `event_gnn→fk_graph`；去掉事件图旋钮与 routed；加 FK_GRAPH_* | 0.23 M / 0.084 G / 5.38 ms vs 0.95 M / 0.827 G / 7.04 ms | 同网格；**训练中途 DDP 卡死→单卡续训**（prereg §4） | **否：整图形态替换**；另有训练协议扰动 |
| **meshgraph vs fkgraph** | `fk_graph→mesh_graph` + MESH_GRAPH_*（节点/边/可见性/pooling；流速关） | 0.44 M / 0.314 G / 11.99 ms vs 0.23 M / 0.084 G / 5.38 ms | 同网格 | **部分是**（同族三处结构差；prereg 契约锁定）；成本差 2×+ |
| **rootinnov vs S37** | `ROOT_INNOVATION` + `PREV_MLP_ROOT:false` + 冻结骨干 + 热启动 + **MAX_STEPS 1500 / SAVE 250** | 0.74 M / 0.827 G / **20.66 ms** vs 0.73 M / 0.827 G / 7.72 ms | 同种子热启动；**6 点网格 250…1500** | **否：多变量捆绑**（新息头 ⊕ 去根回拉 ⊕ 短训 ⊕ 冻结） |
| **rotw10 vs S37** | **仅** `LOSS.LAMBDA_R 60→600` | 同参/同 FLOPs；延迟 8.00 vs 7.72（同量级） | 同网格两种子 | **是：单因子** |
| **CNN vs S37** | 骨干 ResNet18/LNES、绝对 51 维、不读 prev、batch/卡数不同等 **≥20 项** | 11.18 M（诊断） vs 0.73 M | CNN 实际网格 500…4000×8；S37×12 | **否：整表述对照**（诊断，非臂） |
| **meshq vs S37**（无主行） | `MESH_QUERY` + `PREVPOS_EMBED:false` + `ENCODER_NODE_ATTRS:raw4` + 去掉 routed | （无主行） | 同网格两种子 | **否：门控 ⊕ 无 prev_mlp ⊕ 查询读出 ⊕ raw4 多因子**；设计上是「逐关节门+无 prev_mlp」的现成对照 |

【代码事实】配置差分：`.experiments` 未改仓库；用 `tools/diff_configs.py` + 扁平键对比（见上表「YAML 实测」）。
【实验事实】主行数字：各 `*_main_row.json` 的 `two_seed_mean` / `params_total` / `macs_forward_packet` / `latency_ms_scaled_full1p75`。

---

## 2. 统计：网格矩、配对差、选点增益、1.1 mm 检出能力

### 2.1 各臂固定网格递推 RA（mm）

【实验事实】源：各 `selection_val_core_step50.json` 的 `grid[].mpjpe_ra_mm`；复算
`.experiments/lit_hyp_20260929/B6/grid_stats.json`。

| 臂（种子） | 选中 step / RA | 网格 n | 均值 | 中位 | 标准差 | 选点相对均值增益 |
|---|---|---|---|---|---|---|
| S36 3407 | 4000 / **21.42** | 12 | 25.34 | 25.18 | 2.26 | −3.92 |
| S36 3408 | 5000 / **24.92** | 12 | 27.61 | 26.70 | 2.74 | −2.69 |
| S37 3407 | 2500 / **19.23** | 12 | 22.25 | 22.79 | 2.47 | −3.02 |
| S37 3408 | 5000 / **22.26** | 12 | 25.02 | 23.95 | 2.57 | −2.76 |
| fkgraph 3407 | 2000 / 22.64 | 12 | 26.50 | 25.49 | 3.15 | −3.87 |
| fkgraph 3408 | 1500 / 21.56 | 12 | 25.52 | 26.32 | 1.97 | −3.96 |
| meshgraph 3407 | 3000 / 20.48 | 12 | 26.42 | 26.30 | 4.08 | −5.94 |
| meshgraph 3408 | 5000 / 29.30 | 12 | 39.50 | 37.56 | 6.30 | **−10.20** |
| rootinnov 3407 | 1000 / 23.01 | **6** | 26.79 | 25.59 | 4.06 | −3.78 |
| rootinnov 3408 | 1250 / 22.86 | **6** | 24.41 | 23.99 | 1.44 | −1.55 |
| rotw10 3407 | 6000 / 21.10 | 12 | 25.11 | 25.59 | 2.58 | −4.00 |
| rotw10 3408 | 4000 / 20.38 | 12 | 24.46 | 23.99 | 3.47 | −4.08 |
| meshq 3407 | 2500 / **43.34** | 12 | 65.79 | 65.23 | 17.39 | −22.45 |
| meshq 3408 | 4000 / **41.83** | 12 | 68.44 | 52.02 | 26.44 | −26.61 |
| CNN 3407 | 4000 / 13.31 | 8 | 17.90 | 16.53 | 4.53 | −4.59 |
| CNN 3408 | 3500 / 13.80 | 8 | 18.43 | 16.04 | 5.77 | −4.63 |

两种子选中 RA 差：S36 **3.50**；S37 **3.03**；meshgraph **8.82**；fkgraph 1.07；rotw10 0.72；CNN 0.49；rootinnov 0.15。
【实验事实】与 `00_CONTEXT.md` §2 / 账本 §「S36 两种子差 3.5 mm」一致。

### 2.2 同 step 配对差（臂 A − 臂 B；负 = A 更好）

【实验事实】`grid_stats.json` → `paired` / `summary_pairs`。

| 比较 | 两种子选中差均值 | 两种子网格均值差 | 逐种子选中是否都更好 | 备注 |
|---|---|---|---|---|
| S37 vs S36 | **−2.43** | **−2.84** | 是（−2.19 / −2.66） | 网格均值优势 **大于** 选中点；选点未夸大 S37 |
| fkgraph vs S36 | −1.07 | −0.47 | **否**（+1.22 / −3.36） | 选中点比网格均值更乐观约 0.6 mm |
| meshgraph vs fkgraph | +2.79 | **+6.95** | 否 | 选点大幅掩盖网格灾难（尤其 3408） |
| rootinnov vs S37 | +2.19 | +0.87（仅共同 step 500/1000/1500，n=3） | 否 | 网格集合不同，配对弱 |
| rotw10 vs S37 | **+0.00** | **+1.15** | 否（+1.87 / −1.88） | **选中打平，但整网更差 ~1.2 mm** |
| CNN vs S37 | −7.19 | −5.30（共同 8 点） | 是 | 诊断；选点增益两边相近 |

选点相对网格均值的「乐观幅度」：S37 均 −2.89，S36 均 −3.31 —— S37 **没有**靠更大选点增益赢 S36。

### 2.3 1.1 mm 门槛的检出能力（两种子 + 选点=上报）

账本门槛【实验事实】`docs/FAILURE_AND_CLEANUP_LEDGER.md` §1.2：采纳需约 ≥1.1 mm 且不宜只靠单 seed；
S37 操作化【实验事实】`S37_ROUTED_READOUT_PREREG.md` §2：两种子均值 ≤ 对照−1.1 **且逐种子不劣**。

【推断】把同臂两种子选中 RA 差当作 \(|X_1-X_2|\)，在正态假设下
\(\mathbb{E}|X_1-X_2|=\sigma\cdot 2/\sqrt{\pi}\)，得 \(\hat\sigma_{\mathrm{S36}}\approx 3.11\)、\(\hat\sigma_{\mathrm{S37}}\approx 2.68\)；
单种子臂差标准差 \(\hat\sigma_\Delta=\sqrt{\hat\sigma_{\mathrm{S36}}^2+\hat\sigma_{\mathrm{S37}}^2}\approx\mathbf{4.10}\,\mathrm{mm}\)。
Monte Carlo（5×10⁴，脚本内）：真实改善 \(\delta\)、严格门（均值≥1.1 **且** 两种子都不劣）的检出率：

| 真实 δ (mm) | 严格门 power | 仅均值≥1.1 |
|---|---|---|
| 0（假阳性） | **0.23** | 0.35 |
| **1.1** | **0.35** | 0.50 |
| 2.43（S37 实测量级） | 0.50 | 0.68 |
| 3.0 | 0.57 | 0.74 |
| 5.0 | 0.78 | 0.91 |

【推断】在「选点=上报、仅两种子」下，**1.1 mm 严格门对真效应 1.1 mm 的检出能力约 ⅓**；对真零的误过门率仍约 23%。
账本另列的 paired bootstrap 95% CI【实验事实】§1.2 在本批 zgz 臂上**未见执行记录**。
用「全臂种子差」估的 \(\hat\sigma\approx 2.14\)（含 meshgraph 8.8 与 rootinnov 0.15）会抬高 power，**不宜**作为 S37↔S36 的默认噪声模型。

---

## 3. meshq：网格 RA 与是否值得补主行

【实验事实】`outputs/semkine/s37_meshq_s340{7,8}/selection_val_core_step50.json`：

- 3407：选中 step 2500，递推 RA **43.34**（global 43.12 / local 43.58）；网格均值 65.79、中位 65.23、std 17.4。
- 3408：选中 step 4000，RA **41.83**（43.16 / 40.30）；网格均值 68.44、中位 52.02、std 26.4。
- 两种子均值选中 ≈ **42.6**，对 S37 20.74 约 **+22 mm**；对「保持段首」量级（~28，见 S37 §8）仍明显更差。

**无主行 → 按 `AGENTS.md` 不得作为结果表行。**

【推断】作为「逐关节硬门 + 无 prev_mlp」的现成负对照，**选点 JSON 已足够否证「该形态在 zgz 上可用」**；补主行只多延迟/FLOPs/params，
**不值得为刷结果表优先补**。若后续要写进账本失败条或做成本对照，再跑 `make_s36_row.py --run s37_meshq` 即可（低优先级）。

---

## 4. 协议一致性

### 4.1 zgz 臂之间

| 项 | 结论 | 出处 |
|---|---|---|
| 评测代码路径 | 共用 `semkine/eval_track.track_sequence`；RA = 只减腕、不对齐旋转 | 【代码事实】`eval_track.py:127-128,224` |
| 段首噪声 | S36/S37/CNN yaml 的 `TRACK.PREV_NOISE_*` 相同 | 【代码事实】各 yaml `TRACK` 段 |
| 主行 rng | `default_rng(0)`，`noise_scale=1.0`；与选点漂移断言 `<0.05` | 【代码事实】`make_s36_row.py:81-95` |
| 主行复现 | 全部已有主行臂 drift **0.0000 mm** | 【实验事实】`logs/row_*_zgzproto.log` |
| 选点准则 | 网格 `argmin(mpjpe_ra_mm)` | 【代码事实】`select_checkpoint.py:112` |
| fkgraph 特例 | 曾因 float32 atomics 复现漂移 0.0755；改 fp64 后重选点，step 不变 | 【实验事实】`S37_FKGRAPH_PREREG.md` §4 |
| rootinnov | 评测协议同；**训练预算与网格不同** | 【实验事实】prereg §1、选点 JSON `steps` |

### 4.2 PCA6 基线行 vs S37 线

【实验事实】`S37_ROTW_CNNROOT_PREREG.md` §5B 注 / `00_CONTEXT.md` §4：

- PCA6：**12 维 PCA**、`eval_abs` 协议、对 **PCA6 投影 GT** 计分；表内 local 30 / global 10.99；**无递推 RA 列**。
- S37 线：51 维、zgz 递推、RA 不对齐旋转、选点=上报。
- 诊断 CNN（`s37diag_cnnabs`）与 PCA6 **不是同一模型**；与 S37 同 zgz 递推协议时 CNN≈13.56。

【推断】**不能把表内 PCA6 与 S37 的 mm 做因果比较**；表中并列仅作历史锚点/成本参照。

---

## 5. 放大率（TF / 递推）在 §8 之后还能不能跨臂比闭环性质？

【实验事实】§7 原读法（`S37_ROUTED_READOUT_PREREG.md` §7）：S37 增益来自放大率 2.58→2.12（闭环传播改善）。
【实验事实】§8 修正（同文档 §8；`00_CONTEXT.md` §4）：闭环稳态 ≈ 逐包信念误差；放大率 **不是**误差传播增益，只是
\(\mathrm{闭环RA}/\mathrm{TF\,RA}\)。S37−S36 的 −2.43 应读作信念更准，不是传播方式改变。

【实验事实】原始探针 `outputs/semkine/closed_loop_s36_s37routed.json`：

| 种子 | S36 amp | S37 amp |
|---|---|---|
| 3407 | 2.14 | 2.14 |
| 3408 | **3.01** | 2.10 |
| 均 | 2.58 | 2.12 |

fkgraph amp ≈ 2.44 / 2.43；meshgraph 2.51 / **2.86**（`closed_loop_s36_s37fkgraph.json`、`closed_loop_s37meshgraph_vs_fkgraph.json`）。

【推断】§8 之后：

1. **不能**再用「放大率下降 ⇒ 闭环传播更好」做跨臂因果结论（§7 机制叙事已被作者自修正）。
2. 放大率仍可作**描述性比值**，但跨臂比较脆弱：S36 两种子 amp 已差 0.87；比值同时揉进 TF 分母与闭环分子；选点=上报使两端都偏乐观。
3. 若要比「闭环性质」，§8 给出的可操作量是：**同包迭代收敛点、相对保持段首/原地不动、误差是否随时间增长**——不是 amp 标量。

---

## 6. 结论清单（支持 / 相关 / 证据不足）

### 6.1 有实验支持（可保留，带标签）

1. 【实验事实】S37 选中点两种子均值 20.74 vs S36 23.17，逐种子都更好（−2.19 / −2.66）；**整网网格均值也更好**（−2.84）。出处：主行 + `grid_stats.json`；prereg §6。
2. 【实验事实】证据置零闭环 +7.6 mm（机制承重）。出处：`S37_ROUTED_READOUT_PREREG.md` §6。
3. 【实验事实】rotw10 与 S37 选中点打平（|Δ|<0.01），且网格均值 **更差** +1.15 → 「只加重旋转损失不能稳定赢」有支持。出处：主行 + 配对网格。
4. 【实验事实】rootinnov G1/G2/G4 不过、递推 22.94；「冻结骨干+新息头第一步」未过预注册门。出处：`S37_ROOT_INNOVATION_PREREG.md` §5–§6。
5. 【实验事实】meshgraph 两种子分裂、均值劣于 fkgraph；网格 std/选点增益异常大。出处：主行 + 网格表。
6. 【实验事实】fkgraph 对 S36 |Δ|=1.07、种子反向 → 按门 **打平**。出处：主行 + prereg §6。
7. 【实验事实】同协议下 CNN 诊断 13.56，fuse_R 16.93（对 S37 −3.81）。出处：`S37_ROTW_CNNROOT_PREREG.md` §5B。
8. 【实验事实】meshq 选中 ~42–43 mm，形态失败。出处：meshq selection JSON。
9. 【实验事实】§8：global 闭环不随 69 s 漂移；同包迭代收敛点贴近闭环。出处：S37 prereg §8。
10. 【代码事实】+【实验事实】zgz 臂主行复现 rng 协议一致（drift 0）。

### 6.2 仅相关 / 描述性（勿写成因果）

1. 【推断】S37 与 S36 的 −2.43 **相关于**去掉渲染 prev + 路由读出的联合改动；不能拆成「只因路由」。
2. 【推断】放大率 2.58→2.12 与采纳 **同时出现**，但 §8 后不再支持「传播机制改善」因果。
3. 【实验事实】fkgraph abs −11.6 与 RA 打平并存 → abs 改善与 RA 门通过无关。
4. 【推断】meshgraph 3407 单种子好看与 3408 崩坏，和 root 旋转探针相关（prereg §6–§7），但是否「整臂不稳只因 root」仍依赖分解探针，不是主行单独能证。
5. 【推断】CNN 全面优于 S37（根+手指）是 **表述/编码器双重对照**，不能直接证「只要换根头就够」。

### 6.3 当前证据不足以支持该结论（逐条）

1. **「S37 增益来自闭环误差传播方式改善（放大率下降）」**  
   当前证据不足以支持该结论：作者 §8 已用同包迭代推翻 §7 读法；−2.43 应读信念误差。出处：`S37_ROUTED_READOUT_PREREG.md` §8:173-175。

2. **「S37 vs S36 干净隔离了路由读出模块」**  
   当前证据不足以支持该结论：YAML 同时改 `PREV_RENDER` 与 `ROUTED_READOUT`。出处：配置差分 §1。

3. **「1.1 mm + 两种子门保证统计可靠采纳」**  
   当前证据不足以支持该结论：在 \(\hat\sigma_\Delta\approx4.1\) mm 下真效应 1.1 的严格门 power≈0.35，假阳性≈0.23；未见 bootstrap CI。出处：§2.3；账本 §1.2。

4. **「选中点优势可代表整条训练轨迹」**（尤其 meshgraph / rotw10）  
   当前证据不足以支持该结论：meshgraph 选中差 +2.8 对网格均值差 +7.0；rotw10 选中打平但网格 +1.15。出处：§2.2。

5. **「fkgraph 与 S36 信息量相当故形态打平可归因于观测充分性」**  
   当前证据不足以支持该结论：编码器整换 + DDP 中断续训；打平落在门边缘（差 0.03 mm 到 1.1）。出处：`S37_FKGRAPH_PREREG.md` §4；主行。

6. **「rootinnov 否证了『新息几何条件』本身」**  
   当前证据不足以支持该结论：同时去掉根 prev_mlp、冻结骨干、只训 1500 步；失败可归因于去回拉或欠训。出处：prereg §1、§6。

7. **「加重 LAMBDA_R 对根旋转无效」的强因果（跨剂量）**  
   当前证据不足以支持「×30 也无效」：×30 仅单种子；×10 打平支持「×10 非杠杆」，外推剂量需第二种子。出处：`S37_ROTW_CNNROOT_PREREG.md` §5A。

8. **「表内 PCA6 基线说明 S37 仍差 8+ mm」**  
   当前证据不足以支持该结论：协议不同（PCA6/`eval_abs`/投影 GT）。同协议参照应是 CNN 诊断 13.56。出处：§4.2。

9. **「meshq 证明『无 prev_mlp + 逐关节门』有害」**  
   当前证据不足以支持该结论：meshq 还改了 raw4、查询聚合、去 routed；失败未分解到门或 prev_mlp。出处：`S37_MESHQ_PREREG.md` §1；网格 ~42 mm。

10. **「跨臂比较放大率可排序闭环品质」**  
    当前证据不足以支持该结论：见 §5；S36 种子间 amp 已剧烈波动。

---

## 7. 简要判读（给主文档用）

- 唯一较干净的结构正增益仍是 **S37 vs S36（−2.43，两种子+整网同向）**，但 **不是单因子**；§7 的放大率机制叙事应弃用，改用 §8。
- **统计上**本协议对 ~1 mm 效应欠功率；对 ≥2.5 mm 且两种子同向的效应（S37、CNN）更可信；对打平/边缘（fkgraph、rotw10）应用网格均值与种子符号三重核对。
- meshq **不必优先补主行**；已是强负对照。
- 放大率标量 **不宜**再作跨臂闭环性质的主证据。
