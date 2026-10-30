# C1 完整复现入口

`reproduce.py` 用固定的 seed3407、原 C1 模型和采样器 `generator=None`，从头训练 6000 步。它调用已经冻结的 `train.py`，没有新的训练实现；双卡每卡 batch512、12 个 worker、prefetch1、学习率、损失、增强、划分和严格数值策略均沿用 canonical 协议。

运行前查询 `outputs/xyz_c1_repro/20261018/status.json`。只有该文件同时记录 `COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS` 和 `baseline_ready: true`，而且完整模型、Adam、实际数据批次、检查点、主行及 2590 包递推输出的核验均通过，入口才允许启动。训练完成进程或一个中间检查点不能代替这个条件。

## 运行

在原机器、原仓库位置和原 `EventHandsTrain` 环境中运行；所选两张卡须空闲：

```bash
cd /data1/lyq/code/mesh/EventHands1
/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xyz_c1_repro/reproduce.py \
  --output-dir outputs/xyz_c1_repro/20261018/replays/my_replay_01 \
  --gpus 0,1
```

输出目录必须尚不存在，且是上述 `replays/` 的直接子目录。名称仅允许字母、数字和下划线，最多 56 个字符，首字符必须是字母或数字。可选 GPU 对只有原预检的 `0,1`、`2,3`、`4,5`；不接受交换顺序、单卡、更多卡、其他 seed、修改步数或续训参数。每次调用只启动一项 fresh 训练。

入口首先核验原协议的 SHA256、213 个冻结文件、数据大小与修改时间、E→P→A 绑定和完整 canonical 放行证据，再检查 Python、软件版本、CUDA/cuDNN/NCCL、全部 GPU UUID/型号及驱动与冻结环境完全一致。环境或硬件变化时会拒绝启动；这里不承诺跨环境逐位复现。

## 产物与判定

新目录中的 `config.yaml` 只改变 `TRAIN.RUN_NAME`、`TRAIN.OUTPUT_DIR`、`EVAL.OUTPUT_DIR`。新 `protocol.json` 和 `launch_acceptance.json` 引用原始证据并绑定新配置，保留原模型、学习配方、数学策略和数据协议。它们是已通过完整复现的原协议的路径变更记录，不表示重新执行了预检。

训练期间写入 `status.json`、日志以及 `run/` 内的双 rank 输入批次带、输入指纹、状态快照和检查点。持续低于 16 GiB 可用主机内存、出现新的 OOM、冻结输入改变或任何子进程失败时，入口停止自己启动的进程并保留失败产物。

完成后按顺序调用原封入口：

1. `verify_final.py` 比对 canonical 与新 run 的全部 6000 批、每 500 步模型/Adam/调度器/RNG 快照、实际检查点和最终状态。
2. `evaluate.py` 在严格数值策略下生成新主行与新成本；使用原完整 zgz 协议，不挑 seed、检查点或片段。
3. `trace.py` 保存两条 zgz 序列全部 2590 包的 incoming/prediction；逐包字节与 canonical 一致，主行全部精度指标精确一致，才写入 `COMPLETE_EXACT_FRESH_REPLAY` 和 `baseline_ready: true`。

新主行在 `<output-dir>/main_rows/`，完整证据和 SHA256 清单在该目录内。`outputs/semkine/xyz_c1_user_<目录名>_s3407` 是新建的 run 定位链接；旧实验和旧主行不覆盖。工程复跑不作为第二个独立 seed，也不与 canonical 平均。成本来自本次严格评估，时延无需与 canonical 逐位相同。

完整正式训练及后续核验的状态以 `outputs/xyz_c1_repro/20261018/status.json` 为准；新单项复跑也必须执行完整训练和后续核验，估时不能代替完成证据。

本入口随附的验收记录只覆盖 CPU guard/schema 与代码流审查，未记录通过 `reproduce.py` 另行完成 6000 步物理训练。原 canonical、工程复跑与采样对照由 `execute.py` 协调同一冻结的 `train.py`、`verify_final.py`、`evaluate.py`、`trace.py` 执行完整物理复现；其完成与否只由上述状态门及绑定的实物证据确认。

## CPU 检查

```bash
PYTHONDONTWRITEBYTECODE=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  -m unittest discover -s tests -p test_xyz_c1_reproduce_guards.py -v
```

检查使用临时目录和合成元数据覆盖拒绝启动、路径保留、配方不变、E→P→A 绑定及环境差异等边界，不执行 GPU 运算或训练。
