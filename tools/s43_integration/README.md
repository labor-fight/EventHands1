# S43 B-R 隔离接入

本目录在冻结 S43 前增加表示核心与必要输入适配。B-R 上一轮通过的是实现契约，效果门未过；本分支用于工程和小训练验证，默认仍是 legacy。没有修改生产 `model/model.py`、原始 MANO、原损失或旧训练/评测入口。

## 最小改动与配置

新 `MODEL.S43_EVENT_REP` 取 `legacy`（默认）、`enhanced`、`br`。wrapper 消费自己的表示、输入层训练、温度和读出训练四个 MODEL 键后，把原有键交给原模型严格校验，不放宽原模型的未知键检查。

| 配置 | 事件表示与必要适配 | 训练范围 |
|---|---|---|
| `a0_frozen.yaml` | 原始 LNES | 全冻结；旧分支参照 |
| `a1_legacy.yaml` | 原始 LNES | 仅原输入 conv1 |
| `a2_enhanced.yaml` | last/count/first → 6→2 的 1×1 适配 | 适配器和 conv1 |
| `a3_br.yaml` | 有限轨迹证据 → 候选预测 LNES 混合 → 2 通道 | 适配器和 conv1 |
| `a4_br_adapter.yaml` | 与 A3 相同 | 仅适配器；增量消融 |
| `a5_br_readout.yaml` | 与 A3 相同 | 适配器、conv1与原rn.fc；事后读出适配消融 |

全部保留原稀疏 ResNet、prev-render/MLP、51D 输出、MANO 解码和 `so3_trans_fk`。没有增加主干或监督项；A5仅解冻已有线性读出权重。B-R 没有旁路读取原始观测 LNES，只有真实 empty 标记；模板是公开候选的预测，不是 GT 图像。连续头可输出候选外状态。

旧配置可由 `load_integrated(old_config)` 加载，未设置新键即 legacy。原 checkpoint 从 `INTEGRATION.BASE_CHECKPOINT`（默认原 seed3407 step6000）加载，旧命令继续使用原入口。`--checkpoint` 专供新 schema 的适配训练权重，显式校验 mode/temperature/K，避免把新权重静默当 LNES。新 checkpoint 不含运行中序列状态，恢复训练需重新按确定性 clip/seed 开始；没有声称精确续训 RNG/optimizer 状态。

## 共享核心与状态接口

```python
bank = SequenceBank(mode="br")
bank.reset(seq_id, start_us, history51, betas10, K_raw_640x480,
           segment_id=segment, history_time_us=start_us-1000)
bank.ingest(seq_id, events_absolute_us_xy_p, source_ids, watermark_us,
            padding_mask=padding)  # True = ignore
representation = bank.readout(seq_id, sealed_cutoff_us)
bank.release(seq_id)
```

`batch_from_windows` 直接使用相同 reset/ingest/readout，支持混序列、独立 clip、显式相对时间和 `valid=False` padding。无效行不读事件或历史、不建立状态。`batching.collate_windows` 提供有限零值 padding 状态与单独目标；`masked_original_loss` 只把有效行交给原损失，NaN padding 标签不会进入监督。

事件时间为绝对 us，表示只消费完整 1ms 组；同刻 multiset，不利用存储顺序。source ID 去重，真实同像素/极性/时间多触发保留；物理 ON/OFF 映射未认证，原 stored 槽不擅自翻转。读出必须对应当前 sealed prefix，历史查询需另建快照，未来 staging 不进入当前图。

每序列保存只读 c0、source 账和计数/phase。`reset/release` 分离 sequence/segment，默认最多16序列，每 epoch 最多262144 source（首次100000上限被完整 global 高峰触发，已公开修订）。原生计数缓存约17.28MB/epoch；该实现有界但不是最小内存。物理 provider 不支持近裁剪面或固定旋转 chart 的边界时，Tracker 明确 hold 手态并记录 unsupported，同时继续消费事件；不静默 clip，也不用原 LNES 旁路救 B-R。

```mermaid
flowchart LR
  E[新事件] --> I[ingest: 触及槽和source账]
  I --> A[完整ms推进: 全KMPQ phase]
  H[epoch历史手态快照] --> G[全MANO候选与三角光度]
  G --> A
  A --> R[readout: R与预测模板]
  R --> N[输入适配与原S43完整前向]
  N --> Q[query: 只输出预测]
  Q --> C[显式commit: 更新手态]
  C --> H
```

`S43StreamTracker.query()` 不更新手态/version或事件证据；`commit()` 仅接受固定50ms末的最新预测。关闭的 epoch 不能再次提交。手态与所有历史条件 detach，不在跨序列/batch保留计算图。B-R 窗内候选由 epoch 开头历史冻结；任意连续头输出通常不属于三候选，所以 commit 后是明确的 **conditional restart + uniform phase**，不是合法的旧候选 phase Bayes handoff。

## 真正增量与全局工作

| 操作 | 实际执行 |
|---|---|
| 原生 ingest | 仅更新触及的原生/低分辨槽与 source 账；普通 live 分支不扫描全缓存 |
| 完整 bin 的 phase | 仍全 K×M×P×Q，空 bin 也有代价 |
| reset/commit 后新 c0 | 全 MANO/FK、三角面光度、所有51时间点、模板前缀重建 |
| readout | 原生全图归约、证据查询及诊断快照复制 |
| query | 适配器、原 S43 全前向、原 prev-render；不是逐事件神经特征更新 |

C++ renderer 是原常色光度模型的等价特化，六组工程对照的状态/FK/整数光度路径全部一致；没有降时间采样。当前网络执行器是原 `forward` 的 masked-dense 调试实现，未接旧部署 `SparseExecutor` / CUDA graph。CPU replay 时间不等于 GPU 部署延迟，更不等于 sensor-to-mesh 延迟。日志和导出轨迹随长度增长，与有界在线表示状态分开计。

## 实际运行与复现

```bash
cd /data1/lyq/code/mesh/EventHands1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
INT_PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python

"$INT_PY" -m unittest discover -s tests -p 'test_s43_integration.py'
"$INT_PY" -m tools.s43_integration.data prepare \
  --out-dir outputs/s43_integration_20261027/data
"$INT_PY" -m tools.s43_integration.train smoke \
  --configs configs/s43_integration/a1_legacy.yaml \
            configs/s43_integration/a2_enhanced.yaml configs/s43_integration/a3_br.yaml \
  --out outputs/s43_integration_20261027/new_smoke.json
"$INT_PY" -m tools.s43_integration.train train \
  --config configs/s43_integration/a3_br.yaml \
  --out-dir outputs/s43_integration_20261027/new_train_a3
```

实际四臂各100步、seed3407、batch2、两步clip、TBPTT=1、Adam .001，梯度裁剪5；BN固定。32 FIT/16 SELECT clip，前两窗由公开初态的冻结 canonical S43 预热，GT仅监督。lead 表示可缓存固定条件，第二窗必须从本臂当前预测重新构造，不能缓存成教师状态。初末 FIT loss 仅固定前8 clip，SELECT为全部16；它们不是完整递推指标。

```bash
"$INT_PY" -m tools.s43_integration.replay \
  --config configs/s43_integration/a3_br.yaml \
  --checkpoint outputs/s43_integration_20261027/train/a3_br/last.ckpt \
  --seq zgz_local --threads 1 --init public --duplicate-query \
  --out outputs/s43_integration_20261027/replay/new_a3_zgz_local

"$INT_PY" -m tools.s43_integration.batch_replay --workers 8
"$INT_PY" -m tools.s43_integration.perturb \
  --config configs/s43_integration/a3_br.yaml \
  --checkpoint outputs/s43_integration_20261027/train/a3_br/last.ckpt \
  --out outputs/s43_integration_20261027/new_perturb.json
```

已有产物拒绝覆盖；重跑使用新 `--out` / `--out-dir`，或在独立输出目录复制同配置。完整回放不提供 maxsteps 快捷方式：所有 valid run、预热和尾事件均处理，source 总数和有序 UID digest 必须完全一致。valid-runs 外没有有效目标的事件单独标记，不混称已评分。

公开初始化采用无 GT 的两窗 canonical 预热；目标为 event cutoff 前1ms，各臂相同。`standard` 是旧 GT-assisted 初态诊断，与公共部署流分开，短于50ms段按原规则不抽初始化噪声。单条 CLI 的 rng0 不自动等于原多录制共享 RNG 的整套正式评价；S39 W200/filter 栈未在本固定 W50 接入中验证，冲突窗口配置明确拒绝，不生成新的正式主行。

## 结果入口与剩余问题

主报告、逐录制 JSON/NPZ、loss/梯度曲线、修订与失败现场在 `outputs/s43_integration_20261027/`。效果以该目录的实际结果为准，不能由 smoke 或训练 loss 下降宣布替换 baseline。

当前限制包括：有限 root-x 候选、未校准常色光度/极性、每 epoch phase 条件重启、固定 chart/近裁剪面、短 clip与长闭环历史分布差、尚未接稀疏部署执行器及W200正式栈。新分支默认不替换旧 S43。
