# Agent conventions for EventHands1

## 数据协议固定，不再改动（用户 2026-10-01）

- 训练：9 人（ch、lfz、lpc、lr、ly、lyh、lyq、ycy、ylf），72 条序列。
- zgz（`zgz_global`、`zgz_local`）既是开发集也是测试集：选点、调参、上报都在这两条序列上。不另设封存测试集。
- 唯一的划分文件是数据根下的 `splits_semkine.json`（由 `tools/build_splits_semkine.py` 生成，文件只读）。
  配置的 `DATA.SPLITS_MANIFEST` 要么不写（默认就是它），要么写它的绝对路径；不要新建或换用别的划分。
- 退役的 8 人划分和旧封存测试（ly）已作废；按 8 人划分训练的 run 不进主表。

## 实验编号（用户 2026-10-04）

- **S39 唯一指 `dt3_dz_l3_w200cfg`**：`dt_dz_l3`（种子 3407 / 3408 / 3409）+ 证据窗口 W = 200 ms（裁段首）+ DT2 事件量自适应滤波 + 证据门 ramp1000。
  主行 `outputs/semkine/dt3_dz_l3_w200cfg_main_row.json`（RA 8.71（8.91 / 8.53 / 8.69）），出表用 `--label dt3_dz_l3_w200cfg="S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"`；来历见 `docs/EXPERIMENT_RECORDS.md#dt3-verdict`。
- 09-21 的 `s39_covmap`（S37 routed + 根读表面覆盖图，09-28 已删）不再叫 S39，只用 run 名 `s39_covmap` 指代。

## Results are reported in one table, with these columns only (user's format, 2026-09-21)

Every result reported to the user — a training arm, an ablation, a probe — is headlined by this
table and nothing else: same columns, same order, same units. No extra metric columns (abs MPJPE,
rotation angle, amplification, grid median, mechanism-gate numbers, ...) in the report; those
stay in the arm's pre-registration section of `docs/EXPERIMENT_RECORDS.md` for whoever asks.

```
| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 * | 10.27 | 7.36 | 8.94 | 6.25 | 8.71（8.91 / 8.53 / 8.69） | 4.09 ms | 1.252 G | 2.80 M |
```

- Generate it, do not type it: `python tools/report_table.py <run> [<run> ...] --label <run>="<显示名>"`.
  It reads `outputs/semkine/<run>_main_row.json` (written by `tools/make_s36_row.py` on the zgz
  protocol), so the table cannot drift from the artifacts.
- Accuracy cells are the two-seed mean in mm; the RA cell carries the two seeds in parentheses.
  Latency in ms (full-model anchor 1.75 ms scaling), FLOPs as MACs of the whole `forward_packet`
  in G, params in M.
- Always include the baseline row and the current arm's row.
- A one- or two-sentence verdict (pre-registered gate pass / fail, adopt / keep the current arm)
  may follow the table. The *why* goes to the arm's pre-registration and failure/cleanup sections in `docs/EXPERIMENT_RECORDS.md`.
- An arm without a main row is not reported as a result; run `tools/make_s36_row.py --run <arm>` first.

- 用户2026-10-06明确暂停原持续实验 goal，代码已恢复并保留 S39（`dt3_dz_l3_w200cfg`）。自动派发已停止；未经用户重新授权，不继续诊断或训练。正式S39使用原三种子step6000权重，显式W200裁段首、Count50、DT2+ramp1000离散spec；恢复与失败归档见 `docs/EXPERIMENT_RECORDS.md#tracking-stop-s39-20261006`。
- 用户2026-10-06晚重新授权新一轮（S42，GPU4–7）：候选 A/B/C 与消融均已完成，最好的 S42 候选（S39 配方 + 部署环路自身前态训练 B + resnet18_l3w192）local 9.90 / global 7.28、参数 1.94 M、MAC 1.098 G，但按预注册门可靠性、local 滞后与端到端时延未过，**未替换 S39**；结论、主表、门与复现见 `docs/EXPERIMENT_RECORDS.md#s42-verdict`。S42 的源码改动全部 opt-in（`MODEL.GLIMPSE*`、`DATA.CLIP_*`、`TRACK.SELF_*`、`MODEL.DISTILL_CKPT` 列表），默认路径与 S39 逐位相同。
- 用户2026-10-09授权 S37 XYZ 关联三层验证（零训练来源分析、单检查点探针、配对重训 xa_c1/xa_c1r/xa_c0/xa_orc/xa_ctr，GPU2–7）：预注册、门与结果见 `docs/EXPERIMENT_RECORDS.md#xyz-assoc-20261009`。新增键 `MODEL.XYZ_ASSOC_MODE` / `XYZ_CONTOUR_GAP` / `XYZ_ASSOC_PREV` 全部 opt-in，默认路径与 C1 逐位相同；xa_orc 用真值分配，只作诊断上界，不可部署。
  结果（2026-10-10）：G4 复现成立（末步 22.55 对历史 21.92）；噪声底 δ0 = 3.1 mm；真值分配上界 overall −10.6 / local −15.7 mm（含泄漏）；替代关联 contour 两种子同向改善 2.5 / 4.4 mm 但在噪声底内，φ ≤ 0.25；匹配 C0 在 local 优于 C1 6.9 mm（超 τ、同号）；L2 两项判据不支持。裁决：关联规则是次要缺陷，主瓶颈是闭环前态误差与关联互相放大的结构及 local 深度分布外；见 `#xyz-assoc-20261009` 裁决区。S39 不变。
