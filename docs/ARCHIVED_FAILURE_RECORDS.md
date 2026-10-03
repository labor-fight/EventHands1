# 已关闭路线的实测归档（合并）

2026-10-03 把两份"无法从别处重建的唯一数字"合并成这一份，原文原样保留，只有三处机械改动：标题降一级并加 `[编号]` 前缀；份与份之间的链接改成本文内的链接；其他文档里对这些文件名的引用改成 `docs/ARCHIVED_FAILURE_RECORDS.md [编号]`。没有改动任何数字或判定。

| 编号 | 原文件 | 日期 | 内容 |
|---|---|---|---|
| [S27] | `S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md` | 2026-08-27 | 保持率能给递推误差排序但不能被干预：`GAIN_REG` 线关闭（英文） |
| [GNN-ARMS] | `GNN_ARMS_ARCHIVE_20260828.md` | 2026-08-28 | 已删 KEG / CellGNN / EventGNN 各臂的实测网格 |

---

## [S27] S27: retention ranks recursive error but cannot be intervened on

> 原文件：`docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md`（2026-10-03 并入本文，原文原样保留；取回：`git show 4fbb8e9:docs/S27_RETENTION_IS_NOT_A_CONTROL_VARIABLE.md`）


**Date:** 2026-08-27
**Status:** hypothesis falsified by intervention. The `GAIN_REG` line is closed on both of its two bullets.
**Protocol:** `val_core` under `_retired_splits_semkine_5v2v3.json` (8 sequences, 10452 frames,
subjects `lr`/`lyq` held out), 50 ms, strict recursion with one ground-truth initialisation per
valid run, checkpoint chosen on the fixed 500-step grid by that same recursive metric.

---

### [S27] 1. The reference numbers, re-derived under the current code

| arm | seed 3407 | seed 3408 |
|---|---:|---:|
| `s1_track_domrand` (warm-started dense tracker, no penalty) | **16.6375** | **17.7440** |

These reproduce the `16.64 / 17.74` quoted in the master verdict to four decimals, so the evaluator,
the split and the selection policy used below are the repository's own.

Two facts about that grid are worth carrying forward, because they bound what any claim can mean:
the s3407 grid runs `21.81 / 18.00 / 21.94 / 16.97 / 18.11 / 107.01 / 16.64 / 18.10` with a median
of 18.11, so selection is worth 1.47 mm on its own, and one grid point diverges outright.

### [S27] 2. What the retention penalty optimises, and what the loop actually meets

`TRACK.GAIN_REG_W` penalises `E_delta[ ||f(x+delta) - f(x)||_J^2 / ||delta||_J^2 ]` with `delta`
drawn from the isotropic curriculum noise. For a 51-D input that expectation is a Frobenius
quantity, `(1/51) sum_i sigma_i^2`. The recursion applies the same Jacobian to its own error every
step, so the error concentrates on the dominant direction and the amplification it meets is
`sigma_max`. These are not the same number and need not move together.

`tools/tmp_probe_spectral_gain.py` measures three gains on a frozen checkpoint, all at the amplitude
the loop is currently carrying and all in root-aligned joint space: `g_rand` (one isotropic
curriculum draw), `g_self` (the model's own accumulated error direction), and `g_spec` (a power
iterate `u <- A u`, warm-started along the recursion -- not an analogy for the loop but literally
the loop's own update on its error).

Nine frozen checkpoints (`outputs/semkine/spec_gain_{lnes,lowg,keg}.json`):

| ranks recursive RA at | all nine arms | dense family only (n=7) |
|---|---:|---:|
| `g_rand` -- what the penalty optimises | +0.367 (p=0.33) | **-0.357** |
| `g_self` | +0.533 (p=0.14) | +0.036 |
| `g_spec` | **+0.883 (p=0.0016)** | **+0.786 (p=0.036)** |

And over the S24 lambda sweep the penalty is a perfect controller of the wrong number:

| lambda | 0.03 | 0.11 | 0.32 | 3.16 | Spearman(lambda, .) |
|---|---:|---:|---:|---:|---:|
| `g_rand` | 0.219 | 0.188 | 0.148 | 0.085 | **-1.000** |
| `g_spec` | 0.726 | 0.633 | 0.743 | 0.718 | **+0.000** |
| `b` (single-step, mm) | 15.86 | 17.33 | 18.17 | 19.75 | **+1.000** |
| recursive RA (mm) | 21.27 | 19.44 | 20.45 | 21.18 | -0.200 |

The `g_spec / g_rand` ratio grows 3.3 -> 8.4 across the sweep. The mechanism is legible: the
cheapest way to shrink an average over 51 directions is to flatten the fifty that do not matter, so
the optimiser answers the penalty by making the operator *more* anisotropic, leaving the one
direction that governs the loop untouched, and pays for it in bias.

### [S27] 3. The intervention, and why it failed

S27 re-points the same penalty along a persistent power iterate
(`TRACK.GAIN_REG_DIR: spectral`), one warm-started iteration per optimiser step advanced from the
displacement the penalty forward already produces, so the cost is unchanged.

Two implementation contracts had to be added, both discovered by running rather than by reasoning:

1. **The iterate must stay out of the metric's null space.** Root translation is *exactly* the null
   space of root-aligned FK -- a 0.05 m shift moves the root-aligned joints by 2e-5 mm, against
   29 mm for root rotation and 22 mm for pose. An isotropic draw is safe by accident, because only
   3 of its 51 components are null and the other 48 dominate the denominator. Power iteration
   maximises the ratio by construction, so the null space is its global optimum: the first launch
   reported retention 30-38 within twenty steps, which is a division by ~zero, not a spectral
   radius. Pinned by `test_power_iterate_never_enters_the_metrics_null_space`.
2. **Splatting a reversed view does not reverse duplicate-write order** (found later, in the S28
   encoder, same class of bug: an operation whose ordering is undefined silently returning the
   wrong quantity).

Results, same protocol:

| arm | direction | lambda | RA mm |
|---|---|---:|---:|
| `s1_track_domrand_s3407` (control) | none | - | **16.638** |
| `s27_ws_spec_0p03_s3407` | spectral | 0.032 | 17.510 |
| `s24_lowg_0p03_bnfix_s3407` (scratch control) | isotropic | 0.032 | 19.823 |
| `s27_spec_0p01_s3407` | spectral | 0.010 | 19.945 |
| `s27_spec_0p03_s3407` | spectral | 0.032 | 23.552 |
| `s27_spec_0p11_s3407` | spectral | 0.105 | 31.161 |
| `s27_barrier_s3407` / `s27_ws_barrier_s3407` | spectral, barrier | - | 90.15 / 81.17 |

Recursive error rises **monotonically** with the strength of the spectral penalty.

### [S27] 4. Why, measured rather than argued

Re-probing the trained arms (`outputs/semkine/spec_gain_postfix_*.json`) separates two candidate
explanations and rejects the obvious one:

| arm | `g_rand` | `g_spec` | b (mm) | selected RA |
|---|---:|---:|---:|---:|
| `s1_track_domrand_s3407` | 0.241 | 0.449 | 15.42 | 16.64 |
| `s27_ws_spec_0p03_s3407` | 0.219 | **0.452** | 17.94 | 17.51 |
| `s27_spec_0p01_s3407` | 0.264 | **0.768** | 15.66 | 19.95 |
| `s27_spec_0p11_s3407` | 0.455 | **1.007** | 18.85 | 31.16 |

The penalty **did not reduce `g_spec` at evaluation at all** -- 0.452 against the control's 0.449 --
even though the retention it logged during training fell from 1.02 to 0.13-0.34
(`.cursor/debug-*.log`, `hypothesisId: H1-fix`). The two are different quantities. Training reads
the Jacobian along one persistent direction shared across the batch and advanced once per step; the
probe re-derives the locally dominant direction per sample. A single slowly-moving global vector is
cheap to be insensitive to, and the logged `u . u_prev` of 0.7-0.88 says it had all but stopped
moving. The network nulled the probe vector, not the operator.

So S27 is not evidence that `sigma_max` is the wrong target. It is evidence that **a single global
power iterate is not an estimator of it**, and that the penalty attached to that estimator costs
real accuracy (b rises 15.4 -> 17.9 mm at the mildest setting that moved anything).

### [S27] 5. The consequence that redirects the project

From the same probe, on the *selected control* checkpoint:

```
b (single-step, teacher-forced)  15.42 mm
recursive                        17.17 mm      ->  the loop contributes 11%
```

The entire retention programme is competing for 11% of the error. `b` is the other 90%. This is
consistent with, and explains, both null results: S24 moved its own target 2.6x and recursion did
not follow; S27 moved recursion, in the wrong direction, by damaging `b`.

Two bullets were pre-registered for the G line and both have now been fired. The line is closed for
accuracy. What remains is `b`, and there the defect is already measured
(`outputs/semkine/lnes_capacity.json`): at 50 ms, LNES puts 23474 events into 2367 slots and
**discards 73.8% of them by overwriting**, onto a surface that is then 97.3% empty. That is a
capacity defect in the state-independent half of the model -- the one half that, by the
architecture law in 3.5 of the master verdict, can be widened without feeding the loop. S28 tests
it.

### [S27] 6. What is kept, and what was reverted

The arm was falsified, so its code is gone rather than left switched off: `TRACK.GAIN_REG_DIR`,
`TRACK.GAIN_REG_FORM`, the `gain_u` power iterate, the observable-subspace mask and the barrier
branch have all been removed, and `configs/semkine/s27_*.yaml` deleted. `TRACK.GAIN_REG_W` is back
to exactly its pre-S27 behaviour. A config key that survives its own refutation is how this project
previously trained two arms that were silent copies of their control.

One compatibility shim remains and is load-bearing: `on_load_checkpoint` drops a `gain_u` key if it
finds one, because the S27 and S28 checkpoint grids were written while the buffer existed and
Lightning loads strictly.

Kept:

* `tools/tmp_probe_spectral_gain.py`: `g_spec` is the best available *ranking* statistic for
  recursive error even though it is not an intervention target, and it is the only probe that
  distinguishes "the penalty worked" from "the penalty's estimator was nulled". It is also what
  measured `b` against `rec`, which is the finding in section 5.
* Claim discipline: `g_spec` ranks (Spearman +0.883) but does **not** predict magnitude --
  `b/(1-g_spec)` over-predicts with a 9.1x spread across arms, against 1.24x for `b/(1-g_rand)`.
  The loop error is not fully aligned with the top singular vector. Neither number may be quoted as
  a predicted steady state.

---

## [GNN-ARMS] 稀疏/图前端各臂实测归档（KEG、CellGNN、EventGNN）

> 原文件：`docs/GNN_ARMS_ARCHIVE_20260828.md`（2026-10-03 并入本文，原文原样保留；取回：`git show 4fbb8e9:docs/GNN_ARMS_ARCHIVE_20260828.md`）


> 日期：2026-08-28
> 目的：`outputs/semkine/` 下 keg / cellgnn 各臂的 checkpoint、selection JSON、训练日志与相关代码
> （`semkine/keg.py`、`semkine/gnn.py`、`CellGNN`、`AEGNNLite`）在本日删除。删除前把这些文件里
> **无法从别处重建的数字**全部誊到这里。此后 `event_gnn` 是仓库里唯一的稀疏臂。
>
> 引用规则不变：本文档的每一个数字都来自被删文件里的实测结果，不是复述计划或推测。
> 若与旧文档冲突，以本文为准——旧文档写于这些臂跑完之前。

---

### [GNN-ARMS] 0. 一句话结论

三条"图"路线走完了。**只有 `event_gnn` 是真正的图神经网络，而它证明了"是不是真的图"根本不是瓶颈。**
稀疏臂整体停在 ~32 mm 递归 RA，稠密 LNES 臂是 17.19 mm，差 14.5 mm。瓶颈已定位到把整包压成
一个向量的全局池化读出，不是图算子、不是容量、不是事件采样率——三者都有实测排除。

---

### [GNN-ARMS] 1. 主对照表（`val_core`，`step_ms=50`，manifest `_retired_splits_semkine_5v2v3.json`）

递归 RA（mm，越低越好）。"中位数"是固定 checkpoint 网格的中位数，"选中"是网格上最优点。
选择从不看 `val_loss`——S0 审计发现 `val_loss` 与递归误差**反相关**。

| 臂 | 前端本质 | 前端参数 | 中位数 3407 | 中位数 3408 | 选中 3407 | 选中 3408 | 选中均值 | 种子极差 |
|---|---|---|---|---|---|---|---|---|
| `s1_track_domrand` 稠密 LNES | 稠密表面 + ResNet18 | 11.21 M | 18.11 | 19.40 | 16.64 | 17.74 | **17.19** | 1.11 |
| `s31_cellgnn` | 抽头绑定的 3×3 卷积 | 1.05 M | 39.34 | 53.33 | 34.41 | 39.39 | 36.90 | 4.97 |
| `s32_cellgnn_nohead` | 同上，去掉 active head | 1.05 M | 76.75 | 40.10 | 35.89 | 32.41 | 34.15 | 3.48 |
| `s33_cellgnn_render` | 同上 + 渲染栅格 | 1.05 M | 85.59 | 49.00 | 39.04 | 39.53 | 39.29 | 0.49 |
| **`s36_eventgnn`** | **真图：事件为节点** | **0.50 M** | 38.70 | 34.05 | 31.97 | 31.36 | **31.67** | **0.61** |

**S36 相对 S31：选中 RA 好 5.23 mm，中位数好 9.96 mm，跨种子极差从 4.97 收到 0.61（8 倍稳定性提升）。**
用一半的前端参数达成——所以这不是容量换来的。

> 注意：早期只看种子 3407 会得出"差 0.64 mm，是噪声"的错误结论。3407 单独看确实是平局
> （39.34 → 38.70），是 3408 把结论翻过来的（53.33 → 34.05）。**单种子不足以判定，这条记下。**

---

### [GNN-ARMS] 2. 完整 checkpoint 网格（被删 JSON 的原始数据）

#### [GNN-ARMS] 2.1 CellGNN 系（manifest `_retired_splits_semkine_5v2v3.json`，12 点网格）

| 臂 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s31_cellgnn_s3407 | 40.34 | 36.87 | 44.19 | 44.90 | 38.47 | 39.34 | 37.63 | 35.35 | **34.41** | 35.52 | 40.93 | 46.63 |
| s31_cellgnn_s3408 | **39.39** | 43.40 | 67.83 | 60.79 | 41.88 | 50.31 | 53.33 | 50.99 | 57.27 | 53.87 | 55.88 | 49.74 |
| s32_nohead_s3407 | **35.89** | 36.48 | 39.67 | 62.62 | 89.15 | 89.29 | 76.75 | 99.59 | 73.09 | 83.49 | 85.11 | 75.73 |
| s32_nohead_s3408 | 39.47 | 36.63 | 42.84 | 41.94 | 38.92 | 41.93 | 35.78 | **32.41** | 34.18 | 40.10 | 40.19 | 50.93 |
| s33_render_s3407 | **39.04** | 42.38 | 49.40 | 67.34 | 75.07 | 102.60 | 85.59 | 121.68 | 96.69 | 64.18 | 88.16 | 91.08 |
| s33_render_s3408 | 39.66 | **39.53** | 41.20 | 44.75 | 41.30 | 53.92 | 50.81 | 54.25 | 49.00 | 49.62 | 70.93 | 43.42 |

#### [GNN-ARMS] 2.2 EventGNN（S36，本次新增，12 点网格）

| 臂 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 4500 | 5000 | 5500 | 6000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s36_eventgnn_s3407 | 42.20 | 42.34 | 34.86 | 37.43 | 36.42 | 38.70 | 39.24 | 45.23 | **31.97** | 35.30 | 34.51 | 39.27 |
| s36_eventgnn_s3408 | 33.88 | 33.90 | 34.05 | 37.39 | 34.01 | 33.31 | 34.34 | **31.36** | 34.88 | 33.59 | 36.68 | 35.44 |

网格离散度：3407 为 13.26 mm，3408 为 6.02 mm。**选择增益（2.69–6.73 mm）小于离散度，说明选中点里
有相当比例是选择噪声，报数时必须把中位数一并给出。**

#### [GNN-ARMS] 2.3 KEG 系（manifest `splits_semkine.json`，8 点网格，与上表不同协议，不可直接跨表比较）

| 臂 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 | 3500 | 4000 | 中位数 | 选中 |
|---|---|---|---|---|---|---|---|---|---|---|
| s20_keg_mixed_s3407 | 32.85 | **29.32** | 32.63 | 34.37 | 31.97 | 30.13 | 34.67 | 38.24 | 32.85 | 29.32 |
| s20_keg_mixed_s3408 | 126.54 | 35.39 | 32.32 | 31.05 | 33.22 | **28.68** | 30.07 | 29.67 | 32.32 | 28.68 |
| s20_keg_mixed_s3407/hardroute | 34.34 | 33.58 | **30.39** | 31.35 | 34.60 | 32.39 | 31.88 | 33.47 | 33.47 | 30.39 |
| s20_keg_mixed_s3408/hardroute | 38.29 | 34.72 | **30.92** | 32.69 | 39.23 | 35.64 | 40.71 | 38.18 | 38.18 | 30.92 |
| s20_keg_mixed_s3407/prenorm | 36.45 | 68.40 | 34.68 | **33.54** | 43.51 | 38.29 | 35.01 | — | 36.45 | 33.54 |
| s20_keg_mixed_s3408/prenorm | 34.61 | 37.70 | **34.47** | 34.97 | 35.13 | 36.11 | 39.77 | 35.22 | 35.22 | 34.47 |
| s20b_keg_hardened_s3407 | 82.62 | **29.65** | 31.51 | 30.84 | 31.25 | 29.71 | 31.20 | 32.14 | 31.25 | 29.65 |
| s20b_keg_hardened_s3408 | 34.13 | 34.52 | 31.35 | 31.32 | 31.62 | **31.10** | 33.91 | 33.79 | 33.79 | 31.10 |
| s21_keg_halo_s3407 | 28.48 | 29.71 | **27.77** | 29.23 | 34.99 | 35.94 | 37.53 | 36.56 | 34.99 | 27.77 |
| s21_keg_halo_s3408 | 36.14 | 30.32 | **27.91** | 36.05 | 35.79 | 35.16 | 37.41 | 35.19 | 35.79 | 27.91 |
| s22_keg_halo_unroll_s3407 | 37.90 | **34.22** | 34.32 | 34.44 | 34.59 | 34.75 | 34.94 | 35.14 | 34.75 | 34.22 |
| s22_keg_halo_unroll_s3408 | 31.51 | 31.61 | 31.50 | 31.42 | 31.30 | 31.20 | 31.14 | **31.08** | 31.42 | 31.08 |

---

### [GNN-ARMS] 3. KEG 的死因：闭环增益（`spec_gain_*.json`）

这是 KEG 被判死的**决定性证据**，不是 RA 数字。`g_spec` 是闭环谱增益，> 1 意味着误差被自身放大。

| 臂 | g_self | g_rand | **g_spec** | 单步 teacher-forced 误差 | 递归误差 |
|---|---|---|---|---|---|
| `lnes_domrand_s3407`（稠密，赢家） | 0.280 | 0.241 | **0.449** | 15.42 mm | 17.17 mm |
| `lnes_domrand_s3408` | 0.245 | 0.227 | **0.481** | 17.25 mm | 19.35 mm |
| `keg_halo_s3407` | 0.703 | 0.578 | **0.910** | 12.43 mm | 26.59 mm |
| `keg_hardened_s3407` | 0.718 | 0.534 | **0.967** | 12.92 mm | 26.10 mm |
| `KEG_hardroute` | 0.714 | 0.582 | **0.921** | 12.28 mm | 26.28 mm |
| `KEG_statefree_enc`（消融 KSSF） | 0.902 | 0.674 | 0.932 | 18.11 mm | **68.98 mm** |

**读法：KEG 的单步精度比稠密臂更好（12.43 vs 15.42 mm），递归精度却差得多（26.59 vs 17.17 mm）。**
差距全部来自闭环增益 0.910 vs 0.449。这是本仓库最重要的一条方法论教训：

> **单步 teacher-forced 误差与递归误差可以反向。只报单步数字等于什么都没证明。**

`KEG_statefree_enc` 那一行同样关键：把状态从编码里拿掉后，单步误差涨到 18.11 mm，递归误差
爆到 68.98 mm（部分序列发散成 NaN）。说明 KEG 的精度**依赖**状态条件化的路由，而正是这个路由
把闭环增益推到 0.91。**这是结构性的两难，不是调参能解决的**——因此 KEG 线终止。

#### [GNN-ARMS] 3.1 KEG 的融合与占空比曲线（`fusion_keg.json` / `duty_keg.json`）

以 `s21_keg_halo_s3407-step=1500.ckpt` 为追踪臂，`s1_abs_domrand_s3407-step=3500.ckpt` 为绝对臂：

| 融合权重 w | RA | abs | jitter |
|---|---|---|---|
| 0.00（纯 KEG） | 27.77 mm | 50.35 mm | 1.64 mm |
| 0.25 | 20.18 mm | 40.10 mm | 2.65 mm |
| 0.50 | **18.75 mm** | 44.16 mm | 4.06 mm |
| 0.75 | 18.82 mm | 49.11 mm | 5.60 mm |
| 1.00（纯绝对） | 19.32 mm | 54.27 mm | 7.40 mm |

占空比锚定版（`duty_keg.json`）：w=0.25 → RA 20.25 mm / jitter 8.13 mm；w=0.5 → RA 19.38 mm / jitter 9.57 mm。
**融合最优点 18.75 mm 仍然打不过稠密臂单独的 17.19 mm，且 jitter 高一个量级。**

---

### [GNN-ARMS] 4. 三条被实测排除的假设（S36 阶段）

S36 的价值不在于它赢了 5 mm，而在于它把三个候选瓶颈依次排除，把问题收敛到读出。

#### [GNN-ARMS] 4.1 「不是真图神经网络」——排除

S31–S35 的 `CellGNN` 自称 GNN，S35 用数值验证证明它等价于**抽头绑定的 3×3 卷积**
（与 `F.conv2d` 逐位吻合到 5e-7）：邻接是固定格点，聚合核是常数 buffer，全部 690 个 tile 无论
占用与否都被卷积。S36 换成真图——邻接由数据决定、边携带相对几何 `(dx, dy, dt)`、算子各向异性、
边严格因果。结果：选中 RA 36.90 → 31.67 mm。**有效，但只有 5 mm，仍差稠密臂 14.5 mm。**

#### [GNN-ARMS] 4.2 「容量不够」——排除

S36 前端 0.50 M 参数，`CellGNN` 是 1.05 M。**一半的参数赢了 5 mm。** 容量不是瓶颈。

#### [GNN-ARMS] 4.3 「丢掉了 95% 的事件」——排除（最有信息量的一条）

`event_gnn` 每包只保留 `ENCODER_MAX_NODES` 个事件。训练时因 44 GiB 显存上限只能取 2048，而一个
包平均约 3 万事件、尾部到 788,817 个——**只保留了 4.4%–5.8%**。这是最显眼的嫌疑。

用 `tools/probe_s36_nodes.py` 做推理期扫描（同一个已训好的 checkpoint，只改推理节点数，无需重训）：

| 推理节点预算 | 约占事件 | 递归 RA |
|---|---|---|
| 512 | ~1.3% | 32.07 mm |
| 1024 | ~2.5% | 32.01 mm |
| 2048（训练所用） | ~5% | **31.97 mm** |
| 4096 | ~10% | 31.99 mm |

**8 倍的证据量变化，RA 只动 0.09 mm。** 采样率完全不是瓶颈。

#### [GNN-ARMS] 4.4 结论指向：全局池化读出

512 个节点经 mean+max 全局池化后携带的信息已经和 4096 个一样多——池化把空间细节抹平了，
喂多少节点、用多好的图算子都一样。这与文档中已记录的"全局池化 AEGNN 丢失关节局部性"吻合。

**下一步必须动读出，而且必须是状态无关的空间读出。**KEG 已经证明用姿态去路由会把闭环增益
推到 0.91（见第 3 节），这条路封死了。

---

### [GNN-ARMS] 5. 踩坑记录（避免重复）

#### [GNN-ARMS] 5.1 `AEGNNLite`：全包 `cdist` 是结构性 NO-GO

在整个 packet 上做 `torch.cdist` 建 k-NN 图，29,000 个事件就是 8.4e8 对距离，实测尾部包能到
788,817 个事件。**代价随事件率平方增长，高事件率必 OOM。** 该臂已删除。

`event_gnn` 的替代方案：邻居只在**时间窗内**搜（每个节点看它之前 `window` 个事件），代价
`O(N × window)`，且只要真实邻居半径落在窗内就是精确的——事件在时间上离得远就不可能是时空邻居。
运行时实测：`pairs_scored` 恒为常数 33,554,432，同期全包 cdist 需要 20.5–21.0 亿对，**61 倍差距
且完全不随事件率波动**（同批次单包事件量在 257,994 与 788,682 之间摆动，开销纹丝不动）。

#### [GNN-ARMS] 5.2 状态不得进入图的**拓扑**

KEG 的失败已在第 3 节量化。设计律：**证据的结构必须状态无关，状态只允许以"值"的形式进入节点
特征和读出。** `event_gnn` 的邻接只由 `(x, y, t)` 决定；上一帧渲染以逐事件像素取值的两个通道
进入节点特征（与真正能追踪的稠密臂同一种耦合方式：比较式，而非索引式）。

#### [GNN-ARMS] 5.3 `val_loss` 不可用于选 checkpoint

S0 审计：`val_loss` 与递归追踪误差**反相关**。S36 种子 3407 的 `val_loss` 轨迹
`0.179 → 0.172 → 0.156 → 0.203 → 0.184 → 0.242`（0.156 触底后回升），而 RA 最优点在 step=4500。
固定网格保存（`SAVE_EVERY_N_STEPS: 500`，`monitor=None`）正是为此。

#### [GNN-ARMS] 5.4 单种子不足以判定

见第 1 节注记。S36 vs S31 在种子 3407 上是平局（39.34 → 38.70，小于网格离散度 13.26），
在 3408 上是大胜（53.33 → 34.05）。**任何"是噪声"的判断都必须等双种子齐了再下。**

#### [GNN-ARMS] 5.5 本机 NCCL：GPU 0↔1 的 PCIe P2P 声明支持但不通

双卡首跑挂死 900 秒，前向一次都没被调用。`tools/probe_ddp.py` 定位：进程组 1.3 秒建好，
但一个 1024 元素的 `all_reduce` **118 秒不返回**；加 `NCCL_P2P_DISABLE=1` 后同一探针 5 秒内跑完。

**本机任何双卡训练都必须设 `NCCL_P2P_DISABLE=1`**，已固化在 `tools/run_s36.sh`。
复现方式见 `tools/probe_ddp.py` 的 docstring。

#### [GNN-ARMS] 5.6 消息张量宽度是 `event_gnn` 的显存天花板

逐边张量 `(B, N, k, ·)` 是峰值来源。最初消息取 `[h_i ; h_j - h_i ; dp]`（宽 `2C+3`），
`MAX_NODES=3072` 即 OOM。去掉 `h_i`（自身信息本就由残差携带）后宽度降到 `C+3`，
3072 变为可跑（1.80 s/it），同时参数减半。实测天花板：

| MAX_NODES | 2048 | 2560 | 3072 | 4096 |
|---|---|---|---|---|
| s/it（2×L20，512 包/卡） | 1.43 | 1.61 | 1.80 | OOM |

配置定在 2048 是为长跑留余量。注意稳态峰值只有 24.60 GiB / 44.39 GiB——OOM 发生在**瞬时连续块
分配**（需 12.14 GiB），不是稳态占用。

---

### [GNN-ARMS] 6. 被删除的内容清单

**代码**：`semkine/gnn.py`（KinematicGNN/TreeConv，死代码，无人 import）、`semkine/keg.py`
（KinematicEventGraph）、`semkine/frontends.py` 中的 `CellGNN` 与 `AEGNNLite`。

**配置**：`configs/semkine/` 下 s19–s23、s30（keg 系）与 s31–s34（cellgnn 系）。

**产物**：`outputs/semkine/` 下所有 keg / cellgnn 目录（215 个 checkpoint，约 1.4 GB）
与 `duty_keg.json`、`fusion_keg.json`、`spec_gain_keg*.json`；`logs/` 下对应训练与选择日志。

**保留**：`semkine/event_gnn.py`、`configs/semkine/s36_eventgnn_s3407.yaml`、
`outputs/semkine/s36_eventgnn_s340{7,8}`、`tools/run_s36.sh`、`tools/finish_s36.sh`、
`tools/probe_ddp.py`、`tools/probe_s36_nodes.py`。tracking 那一圈
（`PREDICT_DELTA`、`PREVPOS_EMBED`、`PREV_RENDER`、`ZERO_EVENT_GATE`、S10 active heads、域随机化）
全程未动。
