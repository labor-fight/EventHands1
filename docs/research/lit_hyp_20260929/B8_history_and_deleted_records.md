# B8：历史证据与已删记录的重叠检查（2026-09-29）

> 研究子代理 B8。只读检索；未改仓库代码/配置/测试/`outputs/`；未训练；未用 GPU。  
> 提取缓存（只读副本）：`.experiments/lit_hyp_20260929/B8/extracted/`。  
> **工作树状态注记**：审计时 HEAD=`85d76a1`，所列 Part1 文档多已从工作树暂存删除（`git status` 呈 ` D`/`D `），内容以 `git show 85d76a1:<path>` 核对；Part2 预注册同属该提交上的已删/待删记录，一律标「已删记录（85d76a1），未复核」。

---

## 第一部分：工作树谱系证据（可作【实验事实】；须标旧协议 / zgz）

### 1.1 文档可用性

| 文档 | 工作树现况 | 读源 |
|---|---|---|
| `docs/GNN_ARMS_ARCHIVE_20260828.md` | 在 | 工作树 |
| `docs/POSITIONING_VS_E3DPSM.md` | 在 | 工作树 |
| `docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md` | 在（有未提交修改） | 工作树 |
| `docs/experiment_history.md` 等其余 Part1 列表项 | 暂存删除 | `85d76a1:` |
| `docs/semkine/*.md` | 暂存删除 | `85d76a1:` |
| `docs/research/dir12_20260928/00_CONTEXT.md` §4 | 在 | 工作树 |

### 1.2 与六问直接相关的工作树事实

**（A）「事件图编码器 + 绝对输出（不读 prev）」是否训过**

- 【实验事实｜旧协议机制线索】E5.5b **不是**刻意的「GNN 绝对」臂：KEG-halo + 恒定 `UNROLL_P=0.5` 使网络学会无视 prev，**坍缩回绝对回归行为**；step 3500 探针 TF RA **42.5 mm** / 递推 **144.5 mm**。「约 40 mm 档」= 文档把 42.5 对齐到历史绝对回归量级。出处：`docs/debug_e55b_unroll_20260825.md:32-33`；账本 `docs/FAILURE_AND_CLEANUP_LEDGER.md:71`。
- 【实验事实｜zgz 筛查，见 Part2】真正「稀疏事件编码器、绝对 51D、forward 不读 prev」的训练臂是 **X1 / `event_hier`**（见第二部分）。非分层的 `event_gnn` + `PREDICT_DELTA=false` 干净格点，工作树未见正式主行。
- 【实验事实｜zgz】逐帧绝对 **CNN**（非 GNN）`s37diag_cnnabs`：RA 13.56（13.31/13.80）。`docs/S37_ROTW_CNNROOT_PREREG.md` §5B；`00_CONTEXT.md` §4。

**（B）`track_render51_dr_*` 的 11.9–13.3**

- 【实验事实】账本引用：`docs/FAILURE_AND_CLEANUP_LEDGER.md:409`（09-06 zgz 表参照行）写 RA **11.9–13.3**，「同协议，主表已有」。
- 【实验事实｜评测集 = zgz 两序列、2590 帧】权威明细：`outputs/hand_data51/report_domrand_2x2.md:3`  
  Protocol：`eval_track.py`，50 ms，GT+噪声初始化，`val = zgz_global + zgz_local`，按递推 RA 选点。  
  选中 worse-of-two（同文件 §2）：`dr_sem` **11.89/12.39**，`dr_so3fk` **12.25/12.19**，`dr_both` **13.01/12.97**，`dr_semsil` **13.66/13.95**。账本 11.9–13.3 ≈ 前三臂选中带。
- **训练日**：产物时间戳约 **2026-08-27**（`training_metadata.json` / `report_domrand_2x2.md` mtime）。配置族 `configs/eventhands_track_render51_dr_*.yaml`；CNN + 渲染 prev + 域随机化；副本名 `*_rep2`（非 semkine 3407/3408）。
- **是否「9 受试者 + 现代 zgz 上报协议」**：评测侧与现行 zgz 留出一致（两序列、2590 帧）。训练元数据**未记录** `train_subjects`；`train_samples=4544300` 与早期 `track_render51` 相同，属 hand_data51 线，**不是** `tools/run_zgz_protocol.sh` / `make_s36_row.py` 的两种子主行。不宜无注记地写成「与 S37 完全同训练协议」。
- **checkpoint**：**仍在** `outputs/hand_data51/track_render51_dr_{both,so3fk,semsil,sem}{,_rep2}/`（含选中 step 的 `.ckpt` 与 `eval_step*/track_metrics_step50.json`）。

**（C）旧协议 S7 常数 δ-trust / S13 绝对锚定**

- 【实验事实｜旧协议，冻结 `track_render51` step1000】δ-trust 0.5：RA **19.258 → 18.730**（约 −0.53 mm）；定位为补丁非解法。`docs/experiment_history.md` §10（约 L1148–1155）；S7 扫表复现 18.7300：`docs/semkine/EXPERIMENT_LOG.md` S7 节（约 L226）。结论：可实现增益几乎已被全局常数收缩拿走（oracle 最优约 18.52）。
- 【实验事实】S13 触发式绝对锚：**单元/融合合同 PASS，跟踪精度跑未完成/被否决不必跑**。等预算间歇锚已判为最差组合。`docs/POSITIONING_VS_E3DPSM.md:190,203-209`；`docs/semkine/EXPERIMENT_LOG.md` S13 行（约 L16）。
- 【实验事实｜旧协议】域随机化双副本 worse-of-two：**19.26 → 12.77**（历史控制线，常与 δ-trust 旧代码线纠缠，主表注记提醒不可与现行表混比）。`docs/experiment_history.md` §11；账本 §1.1。

**（D）dir12 §4 对已删臂的摘要（工作树可引）**

- 【实验事实｜二次摘要】`docs/research/dir12_20260928/00_CONTEXT.md` §4：K0/K1/K2「落在同一条常数增益权衡线上」；778 顶点记忆 **78.78**；旧 S7 δ-trust 19.26→18.73；S13 类模块「从未在闭环跟踪中评测完」。细节以 Part2 原文为准。

**（E）其它 Part1 要点（机制，旧协议为主）**

- `GNN_ARMS_ARCHIVE` / `PLAN_SELECTION` / `S27`：稠密 `s1_track_domrand` 在 **5 受试者**线上递推约 **16.64 mm**；G 与递推强相关；压 G≠锚融合。不可与 zgz 主行直比。
- `EVENT_GNN_SURVEY`：09-06 清除后只留机制判定；当前臂 S37。
- `ASYNC_SPARSE_*`：异步/稀疏 SOTA 裁决文档（Part1 列表），本轮六问无新增独立数字，略。

---

## 第二部分：已删记录（`git show 85d76a1:<path>`）

每项格式：设计一句话｜主要数字｜判定｜是否单变量。全部标 **已删记录（85d76a1），未复核**。

### SPATIAL_CONTEXT（+ TERMINAL）

- **设计**：冻 S37@2500，在 layer3 后、池化前加残差消息 MLP；L=局部邻域，F=远程邻域。  
- **数字**（ylf 前缀自身闭环，mm）：A local/global 42.29/25.42；L 均值 39.23/17.15；F 均值 38.97/17.37；F−L local 仅 **0.25**。  
- **判定**：`STOP_FIXED_ADAPTER_UTILITY_FAILED`（未超 L 的 1.1 mm 门）。  
- **单变量**：是（相对 A 只加适配器；L/F 成对）。  
- 已删记录（85d76a1），未复核 — `docs/S37_SPATIAL_CONTEXT_{,TERMINAL_}PREREG.md`

### NEIGHBORHOOD

- **设计**：固定历史一步，用邻域替换策略相对原读出。  
- **数字**：B−A local **−0.03 mm**；漏邻居支持率 0.82。  
- **判定**：政策效用未过；停止固定邻域替换。  
- **单变量**：是（固定 ckpt 上替换邻域）。  
- 已删记录（85d76a1），未复核 — `docs/S37_NEIGHBORHOOD_PREREG.md`

### TEMPORAL_HISTORY

- **设计**：在 R0 缓存上对 finger 存角做近期历史残差 ridge 读出。  
- **数字**：H 对 O local 改善约 **5.73**，对 C 仅 **0.22**（门 0.5）；global 相对 O/C 恶化 14.3/2.2。  
- **判定**：`NO_FIXED_HISTORY_READOUT_GAIN` / 联合效用失败。  
- **单变量**：是（固定读出形式）。  
- 已删记录（85d76a1），未复核 — `docs/S37_TEMPORAL_HISTORY_PREREG.md`

### HISTORY_EVIDENCE（+ TERMINAL）

- **设计**：历史残差头 H vs 共同条件基线 C vs 代数控制 N。  
- **数字**：O−H local **4.76** PASS；C−H local **−0.85** FAIL；H−O global **+15.3** FAIL。  
- **判定**：`NO_FIXED_PAST_EVIDENCE_READOUT_GAIN`。  
- **单变量**：是。  
- 已删记录（85d76a1），未复核 — `docs/S37_HISTORY_EVIDENCE_{,TERMINAL_}PREREG.md`

### STATE_ACCUMULATION（含 TRAIN / DEBUG / TERMINAL）

- **设计**：BF16 尾部状态累加改 FP32（`STATE_ACCUM_FP32`）；MP0 梯度保真 → MP1 配对短训。  
- **数字**：MP0 两 seed 梯度 L2 门 **PASS**；MP1 ylf 自身递推 P 对 L local 均值仅 **+0.059 mm**，对 A global 退化约 **5.36 mm**。  
- **判定**：MP0 保真过；MP1 `STOP_FIXED_PRECISION_TRAIN_UTILITY_FAILED`。  
- **单变量**：是（仅尾部精度）。  
- 已删记录（85d76a1），未复核 — `docs/S37_STATE_ACCUMULATION_*PREREG.md`

### NORMAL_FLOW（+ PROFILE）

- **设计**：法向流/forecast 认证与 NF2 root-Y profile。  
- **数字**：NF2 A 支持门未过（local 1/4 eligible）；**未读标签**，无姿态 mm。  
- **判定**：forecast 变体暂缓；NF2 `INCONCLUSIVE_SUPPORT_OR_PROFILE`。  
- **单变量**：诊断合同，非训练臂。  
- 已删记录（85d76a1），未复核 — `docs/S37_NORMAL_FLOW_{,PROFILE_}PREREG.md`

### RELATIVE_INCREMENT（+ TERMINAL）

- **设计**：澄清相对增量定义与构造，不重跑 NF2/PA1。  
- **数字**：无任务 mm；定义/构造合同。  
- **判定**：定义澄清完成；无新训练准入。  
- **单变量**：合同检查。  
- 已删记录（85d76a1），未复核 — `docs/S37_RELATIVE_INCREMENT_{,TERMINAL_}PREREG.md`

### CENTERED_UPDATE

- **设计**：手指头对 null 证据做中心化经验更新（相对失败 R1）。  
- **数字**：内部门评分（例 A mpjpe≈25.08）；B 相对 A/C 未过 0.5 mm 级效用门。  
- **判定**：`STOP_FIXED_NULL_CENTERING`。  
- **单变量**：是（固定中心化形式）。  
- 已删记录（85d76a1），未复核 — `docs/S37_CENTERED_UPDATE_PREREG.md`

### CURRICULUM（K0 / K1 / K2）

- **设计**：只改 prev 噪声课程（K1 全小噪声；K2 放大噪声占比；K0 训练噪声置零）。架构=S37。  
- **数字**（单种子 3407 筛查，非两种子主行）：  
  - K1@1000：local **25.54** / global **16.84**，双硬门与 grid 改善门均 **FAIL**。  
  - K2@2500：local **23.56** / global **14.89**，两门 **FAIL**。  
  - K0：仅 step500 有限递推 local **55.1** / global **91.0**；其后 global **NaN**；tf0 不优于 keep-prev，cos≈0.24–0.28。  
  - 割线斜率：K1≈1.0（弱收缩）、K2≈0.44–0.48（强收缩）——支撑「同一权衡线」叙事。  
- **判定**：拒绝该固定课程为目标解；K0 停诊断。  
- **单变量**：是（仅噪声混合/尺度）。  
- 已删记录（85d76a1），未复核 — `docs/S37_CURRICULUM_PREREG.md`；配置 `configs/semkine/k{0,1,2}_*.yaml`

### EVENT_OBSERVATION

- **设计**：合成单关节阈值事件的局部可观测性/J 合同（非 MANO 全手）。  
- **数字**：无 mm 主行；局部合同 PASS，G22 机制门 NOT_TESTED。  
- **判定**：完成定义合同，不放行训练。  
- **单变量**：合成合同。  
- 已删记录（85d76a1），未复核 — `docs/S37_EVENT_OBSERVATION_PREREG.md`

### PAST_ANCHOR（+ TERMINAL / TRACK）

- **设计**：HASTE 式二维轨迹伪锚 → 姿态候选。  
- **数字**：PA1 local 支持 **0/16**（门 8/16）；global 16/16 但未开 B 标签。  
- **判定**：`INCONCLUSIVE_FIXED_TRACK_SUPPORT`；关闭固定 PA1。  
- **单变量**：固定前端/候选政策。  
- 已删记录（85d76a1），未复核 — `docs/S37_PAST_ANCHOR*.md`

### CONSTANT_CONTEXT（+ TERMINAL）

- **设计**：冻骨干，只拟合池化前常数 bias b∈R^128，对照已存局部消息块 L。  
- **数字**：C local/global 均值 **36.87/17.53**；L **38.60/17.19**；C−L global **+0.34**（容差 0.3）。  
- **判定**：`CONSTANT_UTILITY_WITHOUT_MATCHING_L`；不采纳。  
- **单变量**：是（仅 b）。  
- 已删记录（85d76a1），未复核 — `docs/S37_CONSTANT_CONTEXT_{,TERMINAL_}PREREG.md`

### x1_hier / event_hier

- **设计**：`ENCODER: event_hier`（整包空间 kNN + 三级 set-abstraction）；**绝对 51D，不读 prev**（`PREDICT_DELTA/PREVPOS_EMBED/ROUTED_READOUT/ZERO_EVENT_GATE=false`）。相对 S37 多处同时改（编码器+输出契约+loss 块+步数），**不是**「只改绝对/跟踪」单格。  
- **数字**（单种子续训 X1c@3000，zgz 筛查）：G-a abs global **12.61** / local **24.59**（门 ≤13/≤17，local FAIL）；G-b α=0.5 与 S37 融合 global **11.29** / local **21.79**（联合 FAIL）。  
- **判定**：`STOP_FIXED_X1C_JOINT_GATES_FAILED`；原 X1 曾 TIMEOUT。  
- **单变量**：**否**（多因子相对 S37）；但是「GNN/层级编码器 × 绝对」这一格的最接近实测。  
- 已删记录（85d76a1），未复核 — `docs/S37_X1_CONTINUATION_PREREG.md`；`configs/semkine/x1_hier_abs_s3407.yaml`

### k0 / k1 / k2

- 见 CURRICULUM。产物路径曾写在预注册（`outputs/semkine/k*_s37_*`）；**当前 `outputs/semkine/` 无这些目录**（已清或未保留）。

### c0（`c0_s37_so3fk`）

- **设计**：S37 路由 + SO(3)/FK 损失块（配置注释为 C0 损失臂）。  
- **数字**：本重叠集的预注册正文未附闭环主行；未见保留 `outputs/semkine/c0_*`。  
- **判定**：配置存在于 85d76a1；**结果未在本批文档核实到数字**。  
- **单变量**：相对 S37 主要为 loss（配置声称）。  
- 已删记录（85d76a1），未复核 — `configs/semkine/c0_s37_so3fk_s3407.yaml`

### s39_covmap

- **设计**：S37 路由上 root 头另读表面 coverage map（配置注释称单变量相对 S37）。  
- **数字**：配置内引用 debug：换 GT 根旋 19.26→13.47；ridge 覆盖图降旋转误差等——**属设计动机，非正式主行**。工作树无 `S39_COVMAP_PREREG` 结果节；`outputs/semkine/` 无 s39 目录。  
- **判定**：**训练结果未核实**（仅配置+动机）。  
- **单变量**：设计上是。  
- 已删记录（85d76a1），未复核 — `configs/semkine/s39_covmap_s3407.yaml`

### event_guided_mesh / event_guided_mesh_memory

- **设计**：778 顶点事件引导图；记忆臂加顶点 GRU + unroll 对。  
- **数字**：记忆臂两种子选 step500，RA **78.78（64.25/93.32）**；local/global MPJPE 58.87/96.08。无记忆版仅 debug（支持/边替换），无精度结论。  
- **判定**：精度门失败，不采纳。  
- **单变量**：记忆邻接相对初版有单变量改动；整臂相对 S37 多因子。  
- 已删记录（85d76a1），未复核 — `docs/EVENT_GUIDED_MESH_MEMORY_PREREG.md`；配置 `event_guided_mesh{,_memory}_s3407.yaml`

---

## 必须回答的六问

### 1. 是否训练过「事件图编码器 + 绝对输出（不读 prev）」？E5.5b ~40 mm 指什么？

- **有**：X1/`event_hier` 绝对臂（Part2）。结果：绝对筛查 global≈12.6 尚可、**local≈24.6 未过门**；与 S37 固定融合亦未过门。已删记录（85d76a1），未复核。  
- **没有**核实到「非分层 `event_gnn` + 绝对」的干净 2×2 格点主行。  
- **E5.5b 的 ~40 mm**：指恒定 unroll 坍缩后 TF **42.5 mm**，文档称为历史「绝对回归」档；**不是**成功的 GNN-abs 成绩，而是「学会不看 prev」的失败对照。出处：`docs/debug_e55b_unroll_20260825.md:32-33`；`FAILURE_AND_CLEANUP_LEDGER.md:71`。【实验事实｜旧协议】

### 2. `track_render51_dr` 11.9–13.3 出处

| 项 | 内容 |
|---|---|
| 出处文档 | `outputs/hand_data51/report_domrand_2x2.md`；账本压缩引用 `FAILURE_AND_CLEANUP_LEDGER.md:409` |
| 训练 | 约 **2026-08-27**，`track_render51_dr_{both,so3fk,sem,semsil}{,_rep2}`，CNN+渲染+domrand |
| 评测 | **zgz_global+zgz_local**，2590 帧，50 ms，递推 RA 选点 |
| 数字 | 选中带约 **11.89–13.95**（账本写作 11.9–13.3） |
| 9 受试者+现代 zgz 主行协议 | **评测对齐 zgz；训练受试者未在 metadata 钉死；非 3407/3408 主行** |
| checkpoint | **仍在** `outputs/hand_data51/track_render51_dr_*` |

### 3. 分层/整包 kNN 事件编码器（X1 / event_hier）

- **测过**（已删记录）。绝对输出；G-a/G-b 联合失败（见上）。相对 S37 **非单变量**（编码器层级+kNN 定义+绝对头+loss+预算一并改）。

### 4. K0/K1/K2

- 见 CURRICULUM：K1/K2 闭环精度门 FAIL；K0 发散/NaN；机制上支持课程只移动「保持 vs 修复」权衡（dir12 §4 摘要）。单种子筛查，非主行。checkpoint 目录现已不在 `outputs/semkine/`。

### 5. 特征记忆 / 历史证据 / 法向流 / 相对增量 / 中心化更新

| 方向 | 结果摘要 |
|---|---|
| 特征记忆（EGM memory） | RA **78.78**，不采纳 |
| 历史证据 / 时间历史 | 固定读出效用门失败 |
| 法向流 | 支持不足，未进标签评分 |
| 相对增量 | 仅定义合同，无精度 |
| 中心化更新 | `STOP_FIXED_NULL_CENTERING` |
| 状态累加训练 | 梯度保真过，任务效用失败 |
| 空间/邻域/常数上下文 | 固定适配未证明远程/邻域必要 |

### 6. 旧协议 S7 δ-trust 与 S13 绝对锚定

- **S7 δ-trust 0.5**：19.26→**18.73**；oracle 显示再挖空间极小。  
- **S13**：触发锚跟踪实验**未作为精度结果跑完**；等预算对照已判不宜跑。

---

## 总表：候选方向 × 证据状态

| 候选方向 | 工作树证据 | 已删记录（85d76a1） | 是否已测 | 是否单变量 | 对下一轮的含义 |
|---|---|---|---|---|---|
| GNN/层级绝对（不读 prev） | E5.5b 坍缩≈40mm 档（旧协议）；CNN 绝对 13.56（zgz） | X1/event_hier G-a/b FAIL | 是（X1）；非分层 GNN-abs 未干净测 | X1 否 | 填 2×2 需**单变量**绝对 GNN；勿把 42.5 当 GNN-abs SOTA |
| CNN+渲染+domrand 控制 | 账本 11.9–13.3；`report_domrand_2x2.md`；ckpt 在 | — | 是 | 臂内 loss/通道消融 | GNN 线仍落后该控制 8–10 mm；引用须注明 hand_data51/非 3407 主行 |
| 课程噪声 K0/K1/K2 | dir12 §4「权衡线」 | CURRICULUM 数字与 FAIL | 是（3407 筛查） | 是 | 再调课程难破常数增益；勿当主行 |
| 空间远程上下文 | — | SPATIAL_CONTEXT FAIL | 是（固定适配） | 是 | 短训远程消息未超局部；不证明重训更大图无用 |
| 邻域替换 | — | NEIGHBORHOOD FAIL | 是 | 是 | 固定 ckpt 扩邻域无用 |
| 时间/历史证据读出 | — | TEMPORAL/HISTORY FAIL | 是 | 是 | 被动历史残差头未过门 |
| 状态 FP32 累加 | — | MP0 PASS / MP1 FAIL | 是 | 是 | 数值修补≠精度 |
| 法向流 | — | NF 无 mm | 合同/支持不足 | 诊断 | 需新可观测合同，勿复活未过支持门的 profile |
| 相对增量 | — | RI0 定义 only | 未测任务 | — | 无精度结论可引用 |
| 中心化更新 | — | STOP | 是 | 是 | 关闭该固定形式 |
| 过去伪锚（HASTE） | — | PA1 支持不足 | 部分 | 固定政策 | local 支持失败；勿当材料对应已证 |
| 常数上下文 bias | — | 未过 L 联合门 | 是 | 是 | bias≠完整消息块 |
| 事件引导 mesh 记忆 | dir12 §4：78.78 | EGM MEMORY 78.78 | 是 | 整臂否 | 强负对照；几何加性记忆高风险 |
| covmap / c0 loss | — | 仅配置/动机 | **结果未核实** | 设计上是 | 下一轮若重做须重跑并写主行 |
| δ-trust / 触发锚 | experiment_history / POSITIONING | EXPERIMENT_LOG S7/S13 | δ-trust 是；S13 跟踪否 | δ-trust 是 | 常数收缩已挖尽；触发锚勿作创新主线 |

---

## 诚实边界

- Part2 一律**未复核**执行哈希/重跑。  
- 若干 `outputs/semkine/k*`、`x1*`、`event_guided*`、`s39*` 目录**现已不在盘上**；数字只来自 85d76a1 文档。  
- `track_render51_dr` 的训练受试者列表未在 metadata 钉死。  
- 「K0/K1/K2 同一权衡线」在 dir12 为摘要；定量斜率在已删 CURRICULUM，非正式两种子主行。
