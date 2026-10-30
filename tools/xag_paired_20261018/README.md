# XAG：与确定性C1配对的五格训练

用户2026-10-18明确授权其余五格各从头训练一次seed3407。A0F0复用已经完成完整6000步工程复现的`xyz_c1_stable_s3407`；本入口不重新训练基线，也不增加其他seed或挑选检查点。

## 运行身份

新产物：`outputs/xag_paired/20261018/`。五格配置：`configs/xag_paired_20261018/`。原XAG及C1协议、冻结、权重、失败和旧指标保持原身份。

| cell | 关联 | GNN | 固定物理GPU |
|---|---|---|---|
| a0f1 | 原关联 | 增强 | 0,1 |
| a1f0 | 部位GT候选约束（oracle） | 原 | 6,7，先执行 |
| a1f1 | 部位GT候选约束（oracle） | 增强 | 2,3 |
| a2f0 | 表面GT对应（oracle） | 原 | 6,7，a1f0结束后执行 |
| a2f1 | 表面GT对应（oracle） | 增强 | 4,5 |

每格fresh6000、2×512、BF16、seed3407、历史None采样、原Adam/LR/调度/增强/前态扰动/损失；采用已完成C1的严格CUDA/BLAS/NCCL/TF32数值合同。原生`semkine/train.py`不会自行启用本次执行合同，必须使用此目录的冻结包装器。9人72训练序列与原划分不变。

## 验收与入口

先用独立目录做真实双卡预检，覆盖两次完整验证及返回训练。正式协议`protocol.json`、接受记录`launch_acceptance.json`只在独立预检证据全部PASS后由`freeze.py`创建。默认freeze只核验并写指定审查计划，加`--commit`才发布新协议及源码归档；不会启动训练。

正式唯一队列入口（须已封版，五个正式输出目录均不存在）：

```bash
PYTHONHASHSEED=3407 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  tools/xag_paired_20261018/execute.py \
  --protocol outputs/xag_paired/20261018/protocol.json \
  --acceptance outputs/xag_paired/20261018/launch_acceptance.json
```

该命令不接受resume或覆盖原目录；不能对已执行的身份再次运行。所有子进程及GPU对由协调器管理，状态写入`outputs/xag_paired/20261018/status.json`。源码/数据身份变化、新OOM、连续低内存或子进程失败会停止本队列拥有的进程组，并保留失败现场。

`verify_final.py`核验实际6000步索引/共有输入、共有初值、F+新增初值、两rank同步、参数/Adam/调度和固定检查点。各科学臂训练后的权重、梯度和自身递推无需相同。

`evaluate.py`调用原主行生成器；`trace.py`按原完整2590包递推，两次fresh load验证输入与输出逐包一致。`report.py`再次核验训练、行身份、双轨迹及独立NumPy评分，再调用原九列报告工具，完整发布到`release/`后才原子更新旧`outputs/xag/current_metrics.json`；其旧字节归档保留。未完成的臂不会成为结果行。

这是一轮单seed描述性配对，工程确定性不代表跨seed可靠性；A1/A2使用GT，不能部署。完整科学登记见`docs/EXPERIMENT_RECORDS.md#xag-paired-c1-20261018`。
