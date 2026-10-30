### M1 表面提升读出 SLR（工具 `tools/mv2/pa_probe.py pa1 | pa2 | pa3`；GPU 臂 `semkine/slr.py`）

**共同样本**：冻结 S39 三种子（`outputs/semkine/dt_dz_l3_s{3407,3408,3409}` step 6000）；FIT7 + SELECT 的 18 条基础录制（不含 `_v2/_v3/_v4` 训练变体），评测器 50 ms 网格上每段第一步之后每 8 个端点取一个（FIT7 2,204 / SELECT 635 个端点 / 种子）；W200 事件图（`semkine.eval_track.build_lnes`，裁到段首）；前态 prev = GT(end − 50 ms)（E3 单步，`tools/trk6/carry_probe.py` 的构造）。`tools/mv2/common.py::s39_forward_taps` 取 layer1–3 特征，并在每次运行中与 `model(X, P)` 逐位核对。S39 在 9 人全量上训练，所以 SELECT 对 S39 是样本内；所有读出只在 FIT7 上拟合、在 SELECT 上判，它们回答的是"训练分布内冻结特征图是否还有 GAP 之外、与 mesh 锚定有关的线性信息"，不回答新受试者泛化。

**PA1 可兑现空间证据**
- 前态：prev = GT(end − 50) ⊕ S39 自己的训练前态噪声（`TRACK.PREV_NOISE_MODE: mixed`，按 `SemKineDataset._sample_prev_noise` 逐样本以种子 (20261026, 序列哈希, 端点序号, 99) 抽取）。目标 Δ = GT(end) − prev（51 维）。
- 读出臂（GAP 块恒在场，各臂都是 GAP ⊕ 臂块）：`gap`（layer3 全局平均池化，256）；`rootanc21`（rung 0，DT2 `rootanc` 的复现：layer2 在 FK(prev) 21 关节处双线性采样 + 归一像面坐标 + 界内标志，2,751）；`tile`（按 FK 腕点配准的 9×9 规则网格、半径 = 可见顶点到腕点投影的最大距离，在 layer1–3 采样，36,288，非 mesh 空间读出对照）；`lift_mean`（可见界内顶点的 layer1–3 采样均值 + 覆盖率，449）；`lift_vert`（逐顶点单点采样，348,544）；`lift_win`（逐顶点 5 点窗：中心与 ±ρ 沿 u、v，ρ = fx·10 mm / z，窗内平均）；`lift_ring`（lift_win 的面 1-ring 可见加权平滑）；`lift_lbs`（lift_win 按 LBS 权重 × 可见性汇聚到 16 关节 + 覆盖率，7,184；主臂，与 GPU 臂结构对应）。对照：`ctl_vshuffle`（每样本随机置换顶点身份后 LBS 汇聚）、`ctl_shift10`（采样点整体沿随机方向偏 10 mm 等效像素）、`ctl_tshuffle`（lift_lbs 特征换成同序列另一样本的）、`ctl_noise`（同维白噪声）。
- 标准化与求解：各维按 FIT7 行 z-score，块按 1/√d 缩放（每块对核的迹贡献相同）；对偶（核）岭，中心化，λ_rel ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1, 10} × tr(K)/n，以留一 FIT7 受试者的平均 RA（prev + 预测的根对齐关节误差，mm）选择；在 SELECT 上报 RA 与平移 / 根 / 手指三块的 R²。df 配平对照：`lift_lbs` 取使有效自由度等于 `tile` 的 CV 最优 df 的 λ（二分）。
- **主对比（唯一）**：每个 SELECT 样本 err(lift_lbs) − err(tile)（三种子的均值），local 与 global 各一，（序列, 1 s）块自助 2,000 次，单侧 p = 自助均值 ≥ 0 的比例，两域 Holm（0.025 / 0.05）。
- **门**：某域通过 = Holm 显著、均值 < 0、95 % CI 上界 < 0、三种子各自均值 < 0、df 配平版均值 < 0；且无一域显著变差（CI 下界 > 0）。工具有效性：`ctl_tshuffle − gap` 与 `ctl_noise − gap` 的均值都 ≥ −0.05 mm（零模型不得优于 GAP），否则 PA1 记 tool_defect（描述）。mesh 特异性：`lift_lbs − ctl_vshuffle` 与 `lift_lbs − ctl_shift10` 的均值都 < 0。**杀**：两域都不通过。功效说明：1.5 × MDE_local = 1.72 mm 是三种子确证所需；PA1 通过只放行探索性单 seed GPU debug。其余各臂相对 gap / tile 的差、rung 0、df、R² 只作描述。
- 阈值来源：0.025 / 0.05（Holm，α = 0.05）；−0.05 mm 零模型容差为本预注册约定（KS0 种子 SD 0.379 的约 1/8）；MDE 来自 KS0（`#kstar-results-20261025`）。

**PA2 捕获范围（位置型信息）**
- 前态条件：clean、像面平移 5 / 10 / 14 / 19 / 25 mm（垂直于腕射线的随机方向）、沿腕射线 ±45 mm、根旋转 5° / 11°（随机轴、相机系左乘），逐样本以 (20261026, 序列哈希, 端点, 条件) 播种；**每个条件都重跑冻结 S39 前向**（S39 输入含前态渲染）。FIT7 增广：每样本以 0.8 概率随机赋一个偏移条件、否则 clean；SELECT 每样本全部 9 个条件。读出 `gap` / `tile` / `lift_lbs`，同 PA1 的标准化、求解与 λ 选择。
- 统计量：b_off = 1 − carry，carry = ⟨J(out_c) − J(out_0), J(prev_c) − J(prev_0)⟩ / ‖J(prev_c) − J(prev_0)‖²，out = prev + 读出预测；平移 / 深度偏移用绝对关节（含平移），旋转偏移用根对齐关节；三种子平均；S39 自身输出的 1 − carry 作上限参照。
- **门（t14，SELECT 全部）**：diff = b_off(lift_lbs) − max(b_off(gap), b_off(tile))；过 = diff ≥ 0.10 且（序列, 1 s）块自助 95 % CI 下界 > 0；**天花板规则**：若 max(b_off(gap), b_off(tile)) ≥ 0.90，PA2 记 ceiling（不可判别，不作门，M1 的放行只看 PA1）；否则杀。其余条件、分域、按录制 σ\*（Stage 0 / `outputs/sba/p0v`）分层只作描述。阈值来源：0.10 与 0.30 线（KS4 `#kstar-results-20261025` 的冻结 patch 头 b_off 0.054 / 0.051）、0.90 天花板为本预注册约定（S39 自身 carry 根 0.05 / 0.13、手指 0.04 / 0.10，`#s42-verdict`）。

**PA3（描述）**：SELECT 每个手指子树，因果"被运动照亮的轮廓"代理（可见顶点像面法向速度 |n·v|，v 取 GT(end − 100) → GT(end − 50) 的 FK 投影差）与子树 2 px 内事件数，作为"S39 单步使根对齐关节误差下降"的预测量的 AUC；不设门。

**M1 GPU 放行**：PA1 = pass 且 PA2 ∈ {pass, ceiling} 且 PA1 工具有效且 mesh 特异，且 M9 的 SORL / LBA 静态账对 SLR 通过（0 主机同步、0 数据相关形状算子、0 图断裂）。否则 M1 不开 GPU，降为 M3 / M4 的读出载体并记录。

**GPU 臂（放行后）**：`configs/mv2/mv2_slr.yaml` = S39 配方（`configs/dt/dt_dz_l3.yaml`：单卡、批 512 × 累积 2、bf16、Adam 4e-3、warmup 500、cosine、6000 步、同增强与前态噪声）+ `MODEL.SLR*`（layer1–3、投影 32、隐藏 64、1 轮 1-ring、度量半径 10 mm、软可见性、零初值末层；新增 86,826 参数），经 `tools/mv2/slr_entry.py train` 训练。S3 debug：seed 3407，`MAX_STEPS 6000` 跑到 2000 即停；停止条件（只停不放行）：SELECT 上 E3 RA ≤ S39@2000 步 + 0.3 mm 且 carry（根 / 手指）≤ S39@2000 + 0.05，且无发散；S39@2000 用 `dt_dz_l3_s3407-step=2000.ckpt`。S4：继续到 6000 步，与 S39 seed 3407 配对，zgz 一次性报告（`tools/mv2/slr_entry.py evalx row … --variant w200cfg`，主行 + `tools/report_table.py` 九列表）；方向支持 = local ≤ −0.3 mm 且 global ≤ +0.2 mm（S42 单种子 6k 门，描述性）；效应 ≥ 1.5 × MDE（local 1.72 mm）才补 seed 3408 / 3409。

### M2 射线规范商读出 + 度量尺度协变邻域（工具 `tools/mv2/pa_probe.py rgq | mscn`）

**PA-RGQ**：同 M1 的样本（FIT7 + SELECT，E3 前态）。前态沿腕射线刚性移动使腕点到 w / σ（σ ∈ {0.94, 0.97, 1.03, 1.06}，像面尺寸约 ×σ、绕腕），重跑冻结 S39；读数 = 输出的根对齐关节相对 σ = 1 输出的平均位移（mm，对平移盲 = 非 t_z 通道的串扰）与 t_z 响应比（沿腕射线的输出修正 / 所需修正）。第二种扰动形状（Stage 0 要求）：渲染前态时把 betas[0] 改为使手长约 ×σ（Stage 0：+1 SD ≈ −4.7 % 手长），只作描述。**门**（σ = 0.94，SELECT 全部，三种子各自均值）：全部 ≥ 1.149 mm（local MDE）→ pass；全部 ≥ 0.379 mm（local 种子 SD）→ weak（只许 GPU 单 seed 描述）；否则 kill（S39 对规范冗余不敏感，商读出无可收，深度只能报告不能纠正）。阈值来源 KS0。

**PA5-MSCN**：FIT7（门）与 SELECT（描述）基础录制。事件绕渲染主点缩放 s ∈ {0.94, 0.97, 1.03, 1.06}（240×180 网格上最近邻），前态渲染与顶点投影用 fx, fy × s 的 K；读三种逐顶点特征（layer1–3 各用固定种子正交投影到 64 维）：`fixedwin`（layer2，±6 px 五点窗）、`fixedlayer`（layer2 中心）、`metric`（ρ = fx·10 mm / z 五点窗，层级按 log2(ρ / 3 px) 在 layer1–3 间连续插值）；对照 `metric_zperm`（逐顶点深度置换后的度量窗）。漂移 D_x(s) = Σ‖f_x(缩放) − f_x(原)‖ / Σ‖f_x(原) − f_x(A/A)‖，A/A = 同一前态、事件窗提前 10 ms。R = D(metric) / D(fixedwin)，R_z = D(metric_zperm) / D(fixedwin)，（序列, 1 s）块自助。**门**（FIT7，s = 0.94 与 1.06，每个种子）：R 的 95 % CI 上界 ≤ 0.80，且 R_z 的 CI 含 1（度量收益不是平滑），且 D(fixedwin) ≥ 1.5（探针敏感）→ pass；任一 R ≥ 0.95 → kill；不敏感 → untestable（M2 判为不可检验，终止）；其余 → descriptive。注意：图像缩放与沿射线平移只差约 1.1 px 视差，判读写"手变大"而非"深度协变"。阈值来源：0.8 / 0.95 来自计划 §3 M2（20 % 降幅为可见效应下限），1.5 来自 KS0 A/A 经验。

**M2 GPU 放行**：只在 M1 放行之后，作为 M1 子类内的单变量因子臂：PA-RGQ ∈ {pass, weak} → RGQ 读出分解 + prev σ 扰动训练臂；PA5 = pass → `MODEL.SLR_SCALE: metric` 对 `fixed`（6 px）的因子臂。σ 扰动族形状受 Stage 0 约束（绕腕缩放与小手两种都报）。
