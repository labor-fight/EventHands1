# Agent conventions for EventHands1

## 数据协议固定，不再改动（用户 2026-10-01）

- 训练：9 人（ch、lfz、lpc、lr、ly、lyh、lyq、ycy、ylf），72 条序列。
- zgz（`zgz_global`、`zgz_local`）既是开发集也是测试集：选点、调参、上报都在这两条序列上。不另设封存测试集。
- 唯一的划分文件是数据根下的 `splits_semkine.json`（由 `tools/build_splits_semkine.py` 生成，文件只读）。
  配置的 `DATA.SPLITS_MANIFEST` 要么不写（默认就是它），要么写它的绝对路径；不要新建或换用别的划分。
- x1001 的 8 人划分（`splits_x1001_dev_v1.json`）和封存测试（ly）已作废；按 8 人划分训练的 run 不进主表。

## Results are reported in one table, with these columns only (user's format, 2026-09-21)

Every result reported to the user — a training arm, an ablation, a probe — is headlined by this
table and nothing else: same columns, same order, same units. No extra metric columns (abs MPJPE,
rotation angle, amplification, grid median, mechanism-gate numbers, ...) in the report; those
stay in the arm's pre-registration doc under `docs/` for whoever asks.

```
| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |
|---|---|---|---|---|---|---|---|---|
| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |
| S37 路由读出（当前臂） | 26.41 | 15.82 | 21.40 | 12.39 | 20.74（19.23 / 22.26） | 7.72 ms | 0.827 G | 0.73 M |
```

- Generate it, do not type it: `python tools/report_table.py <run> [<run> ...] --label <run>="<显示名>"`.
  It reads `outputs/semkine/<run>_main_row.json` (written by `tools/make_s36_row.py` on the zgz
  protocol), so the table cannot drift from the artifacts.
- Accuracy cells are the two-seed mean in mm; the RA cell carries the two seeds in parentheses.
  Latency in ms (full-model anchor 1.75 ms scaling), FLOPs as MACs of the whole `forward_packet`
  in G, params in M.
- Always include the baseline row and the current arm's row.
- A one- or two-sentence verdict (pre-registered gate pass / fail, adopt / keep the current arm)
  may follow the table. The *why* goes to `docs/S3x_*_PREREG.md` and `docs/FAILURE_AND_CLEANUP_LEDGER.md`.
- An arm without a main row is not reported as a result; run `tools/make_s36_row.py --run <arm>` first.

- 当前最优起点与当前臂为 S37 路由读出（`s37_routed`），按 2026-09-24 用户指令恢复；后续实验以它为对照。
