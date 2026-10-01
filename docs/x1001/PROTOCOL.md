# x1001 研究协议（2026-10-01 开跑前写定）

本文件在任何 x1001 训练结果出现之前写定；之后只追加"变更记录"，不改正文。

## 1. 代码与运行

- 基线 commit：`3a86459`（main）。所有 x1001 改动在分支上提交，不动主工作树：
  - `x1001/base`：协议配置、run provenance、调度器、评测与诊断工具；
  - `x1001/e7`、`x1001/e8`、`x1001/e9`：各候选一个分支、一个 worktree（`EventHands1_x1001/wt/<arm>`）。
- 每个 run 有独立 run_id（`x1001_<arm>_s<seed>`），产物在 `EventHands1_x1001/runs/<run_id>/`，不覆盖历史 `outputs/semkine/`。
- 训练元数据记录 git commit、分支、dirty 标志、配置 SHA256、argv、有效 batch；调度器为每个作业写 run manifest
  （`sched/manifests/<id>.json`：GPU、CPU 切片、环境、起止时间、GPU 小时、GPU 利用率均值/峰值、显存峰值、CPU 秒、读字节、GNU time）。

## 2. 数据划分（固定）

| 用途 | 受试者 | 序列 | 文件 |
|---|---|---|---|
| 训练 | ch, lfz, lpc, lr, lyh, lyq, ycy, ylf（8 人） | 64 | `protocol/splits_x1001_dev_v1.json`（sha256 `0168fad7…f645`） |
| 开发（选点、调参、诊断） | zgz | zgz_global, zgz_local（2590 个 50 ms 包） | 同上（val = val_core = test） |
| 封存测试 | ly | 8 条（4 global + 4 local） | `protocol/splits_x1001_sealed_v1.json`（sha256 `ab8f3134…a9ea1`，只读） |

- zgz 自 2026-09-05 起一直用于选点与上报，故归为开发用途。
- 封存受试者按预先写定的规则抽取：对 9 名原训练受试者取 `sha256("EventHands1/sealed-test/v1/2026-10-01:" + s)` 最小者，结果为 ly。
  开发 manifest 中不含 ly 的任何序列；诊断探针的训练集也不含 ly；封存 manifest 只在最终评估时打开一次。
- 历史 checkpoint（9 受试者训练，含 ly）只用于第一阶段诊断（在 zgz 上），永不在 ly 上评估。

## 3. 训练预算（所有臂相同）

单卡 × 每步 512 × 梯度累积 2 = 有效 batch 1024；6000 个优化器步（6.14M 样本，约 1.5 遍 4.06M 训练样本）；
Adam，峰值 LR 4e-3，warmup 500 步，cosine 衰减到 2%；bf16；每 500 步存一个 checkpoint（12 点网格）；val loss 不参与任何选择。
配对种子 3407 / 3408；胜出者追加 3409；小收益（1.1–2 mm）追加到 3411。
变更理由（相对历史配方）：一卡一实验需要 512 × 2 替代 2 卡 × 512（BN 批大小不变）；cosine 取代常数 LR，
因为历史网格在常数 LR 下同一 run 内波动达 7 mm、选中点比网格中位数乐观 2.1–4.3 mm。

## 4. 选点与指标

- 选点：`tools/select_checkpoint.py`，开发集 zgz，12 点网格，50 ms 递推 RA，rng 种子 0，初始化噪声尺度 1.0。
  同时报告不选点的最后一步（6000）与网格中位数。
- 评测：`tools/x1001/evalx.py`，逐步复现协议闭环（与选点结果差 < 0.05 mm 才继续）。
  报告：根对齐 MPJPE / MPVPE（global、local、帧加权总体）、绝对 MPJPE、根旋转测地角误差、绝对平移误差、
  事件率分桶、连续跟踪失败（同一段内连续 ≥ 1 s RA > 50 mm 或根旋转 > 30°）、10 s 时间块 bootstrap 95% CI。
- 对照（同一批步）：`hold` 保持段首（初始化状态不动）、`noevents` 空包闭环。
- 初始化：协议为每个有效段 GT + 噪声（5 mm / 0.05 rad 量级）；凡结论涉及"普通推理"，另用训练集均值姿态冷启动（不含 GT）。

## 5. 判定

- 主行与统一表：`tools/x1001/evalx.py row` 写 `<base>/outputs/semkine/<arm>_main_row.json`，由 `tools/report_table.py` 生成表。
- 候选相对基线：同种子配对差的均值 ≤ −1.1 mm 且各种子方向一致 → 进入追加种子；|Δ| < 1.1 mm 记打平；≥ +1.1 mm 记变差。
  种子是独立训练重复；不得用大量相关事件包代替种子数。
- 先复现并追赶同协议 CNN（x1001_cnn），再与公开强基线比较。不把预算内最优写成 SOTA。

## 6. 封存测试

只对最终候选（基线、同协议 CNN、胜出者）各种子的开发集选中点评估一次；报告与第 4 节相同的指标，
外加对照与配对差（按序列、连续 10 s 时间块的 bootstrap）。

## 7. 延迟验收（与精度分开）

单卡、固定 CPU 配额（同一 NUMA 节点 4 个物理核）、fp32 推理；完整处理链（事件到达 → 打包 / LNES 或 token → H2D →
前向 → 输出回传）；按 5 ms 间隔到达、7 ms 截止回放真实事件流；报告 P50 / P95 / P99、积压与丢包。
测量时机器上不得有其他 GPU 作业；不得用多卡离线吞吐代替单流实时性能。
