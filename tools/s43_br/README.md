# S43 B-R 最小原型与三组对照

本目录实现固定少量 MANO 轨迹的有限事件证据，不改生产 S43/S39。输入为相同 W50 / 1ms 完整计数前缀；原始 LNES、简单增强和 B-R 使用同一空间采样、历史、β/K、候选与目标。

- `carrier.py`：正规阈值 phase 核、共享 η、bounded epoch、只读 query、refresh/handoff。
- `geometry.py`：只读 MANO FK、三角面 z-buffer 和公开假设光度。
- `representations.py`：三种表示及保持 LNES/简单增强完全不变的细时间干预。
- `readouts.py`：共同 RBF 核岭读出、图像 CNN / 证据 MLP 适配读出。
- `data.py`、`cache_data.py`：现有数据只读、公共初态因果 S43 历史、pickle-free 缓存。
- `synthetic.py`：真实 MANO 几何、已知有限成像条件下的独立 context 控制。
- `experiment.py`：构造与评分，默认只评 validation/SELECT。
- `correction.py`：唯一一次修正，FIT 标定全窗极性解释与证据温度。
- `diagnostics.py`、`check_existing_contract.py`：前态/重放/歧义/生命周期及实际事件流契约。

预注册为 `preregistration.md`，冻结身份在 `outputs/s43_br_20261027/protocol.json`；结果、修订和最终判断位于同一输出目录。早期工程修订产物保留，不参与最终比较。报告以最终 `report.md` 和 `decision.json` 为入口。

## 环境与契约

```bash
cd /data1/lyq/code/mesh/EventHands1
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BR_PY=/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python
```

默认 Python 缺少冻结 S43 所需的 torchvision，使用上述已有环境。无 pip 安装、GPU 使用或大网络重训。

```bash
"$BR_PY" -m unittest discover -s tests -p 'test_s43_br*.py'
"$BR_PY" -m tools.s43_br.check_existing_contract
```

已有科学产物拒绝覆盖。下面列的是本轮各阶段的实跑命令形状；复跑请把 `--out` 改为新的文件名，并为生成缓存指定新目录。`freeze_stage` 会把当前源码及冻结模型身份写入 sources。

## 生成与构造

```bash
"$BR_PY" -m tools.s43_br.synthetic \
  --out outputs/s43_br_20261027/synthetic --workers 6
"$BR_PY" -m tools.s43_br.cache_data \
  --output-prefix outputs/s43_br_20261027/data/existing_fit_select \
  --subjects ch lfz lpc lr ycy ylf --samples-per-recording 8 --threads 4

"$BR_PY" -m tools.s43_br.experiment features \
  --kind synthetic --source outputs/s43_br_20261027/synthetic --axis root_x \
  --out outputs/s43_br_20261027/features/replay_synthetic_root_x.npz --workers 6
"$BR_PY" -m tools.s43_br.experiment features \
  --kind existing --source outputs/s43_br_20261027/data/existing_fit_select --axis root_x \
  --out outputs/s43_br_20261027/features/replay_existing_root_x.npz --workers 6
```

对 `index_y` 重复对应命令。当前构造已直接给三臂显式 β/K；历史 `*_v1.npz` 使用 `upgrade_context.py` 得到 `*_v1c.npz`，新构造不再需要升级。

## 共同读出、适配读出及修正

```bash
"$BR_PY" -m tools.s43_br.experiment evaluate \
  --source outputs/s43_br_20261027/features/synthetic_root_x_v1c.npz \
  --out outputs/s43_br_20261027/results/replay_root_validation.json \
  --eval-split validation --steps 400

"$BR_PY" -m tools.s43_br.correction components \
  --cache outputs/s43_br_20261027/data/existing_fit_select --axis root_x \
  --out outputs/s43_br_20261027/features/replay_root_components.npz --workers 6
```

两个轴 components 均完成后，只以 FIT 标签拟合共享校准参数：

```bash
"$BR_PY" -m tools.s43_br.correction fit \
  --features outputs/s43_br_20261027/features/existing_root_x_v1c.npz \
             outputs/s43_br_20261027/features/existing_index_y_v1c.npz \
  --components outputs/s43_br_20261027/features/existing_root_x_components_v2.npz \
               outputs/s43_br_20261027/features/existing_index_y_components_v2.npz \
  --out outputs/s43_br_20261027/replay_calibration.json
"$BR_PY" -m tools.s43_br.correction transform \
  --features outputs/s43_br_20261027/features/existing_root_x_v1c.npz \
  --components outputs/s43_br_20261027/features/existing_root_x_components_v2.npz \
  --calibration outputs/s43_br_20261027/calibration_v2.json \
  --out outputs/s43_br_20261027/features/replay_existing_root_v2.npz
```

同一 `evaluate` 命令对修正后的三臂再训练同样 400 步，不加长、不换 seed。共同核读出精确求解，适配头记录训练 loss、拟合率与实际参数数。B-R 解析概率另列，不冒充同架构学习结果。

## 最终报告数据

校准/读出日程锁定于 `final_lock.json` 后才评 synthetic `--eval-split test`。zgz 仅在同一锁定后缓存一次：

```bash
"$BR_PY" -m tools.s43_br.cache_data \
  --output-prefix outputs/s43_br_20261027/data/zgz_report \
  --subjects zgz --samples-per-recording 8 --threads 4 --allow-zgz-report-only
```

zgz 的 features/components/transform 与上面相同，随后使用 FIT 的特征作为训练源：

```bash
"$BR_PY" -m tools.s43_br.experiment evaluate \
  --source outputs/s43_br_20261027/features/zgz_root_x_v2.npz \
  --train-source outputs/s43_br_20261027/features/existing_root_x_v2.npz \
  --out outputs/s43_br_20261027/results/replay_zgz_root.json \
  --eval-split test --steps 400
```

## 解释边界

1. 本轮是 24×32、三候选机制任务，未运行完整递推，不是 S43 原主表替换；绝对根平移收益不能写成 RA/手指收益。
2. 现有数据沿 v2e 来源和伪标签，不能称真实传感器测试。GT 只在监督和评分中使用；候选最优误差是明确的 oracle coverage 下界，绝不补入候选。
3. 历史从公共固定初态做两次冻结 S43 推理，非 GT 初始化；历史时刻比 epoch 起点早 1ms，三臂使用相同 held-state 近似；标签仍为 cutoff 前 1ms。
4. 合成中知道公开成像核。B-R 的优势可能来自归纳偏置/样本效率；E1 观测破坏不是合法轨迹同码证明，E2 错绑不是 MANO 独有性证明。
5. 逐 bin prior-remix 保留 phase，是故意不一致的近似，不是另一套已校准传感器生成模型。
6. 计时为 CPU 回放，含构造/geometry/phase/query；synthetic 缓存读取的 render_seconds=0 不是免费几何。query 还复制完整诊断 phase，实际 O(KMPQ)。
7. 当前只检验了有限候选与局部支持；对全 51D 充分性、真实相机校准和跨人泛化没有认证。
