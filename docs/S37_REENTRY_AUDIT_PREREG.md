# RA1 — 当前请求的基线与证据续接核验

日期：2026-09-26。执行前登记的是**保存工件审计**，不是给已见结果的 NP1 实验补做科学预注册。NP1 的原合同仍是 `docs/S37_NATIVE_COORDINATE_PREREG.md`，本页不改变其阈值。

## 授权、预算与停止

本轮用户授权审计、基线核验与有预算的判别。只读已有源码、两种子选点/权重文件身份、旧主行、划分、有效段元数据及 NP1 保存输入/输出。一个 CPU 工作进程、数值库最多四线程、120 秒墙钟上限、GPU 0 秒、训练 0 步、新增文件上限 20 MiB。不重新执行 NP1、不读取新的测试预测或启动 `make_s36_row.py` 重测；已有 S37 主行可直接生成表。

通过条件：历史主行的两种子与 selection 身份、帧分母和全部汇总一致；旧身份清单的权重/配置/MANO/划分仍匹配；NP1 冻结输入、预测前封存及终态封存完整；64 个保存 A 输出与原参照逐字节相同；从保存 geometry 独立复算全部诊断评分与原门一致。汇总代数容差 1e-10 mm；GPU float32 评分和 CPU float64 几何复算容差 1e-4 mm，仅用于审计浮点归约，不改变科学收益门。任一失败或超时即记录并停止，不能覆盖旧数据或改变容差重试。哈希、代数复算不是新训练或新的独立预测复现。

最新请求明确：初始化/恢复也只能用事件、标定和因果历史。旧文档中 GT+noise 初始化授权只能描述历史 H；不能用于本次完整系统有效性证明。H 主行继续保留，C（严格因果部署）成绩当前未知；不同信息合同的精度不能相减归因于网络。已有短反馈待答项不阻塞本次保存工件审计，也不授权偷偷更改 H 的 50 ms 反馈。

## 审计范围与运行命令

`scratch/goal_20260926/ra1/audit_saved.py` 只用 NumPy/标准库；不导入模型、不执行 FK、不创建 CUDA 上下文。预先已读 NP1 receipt/stage_b，知道其停止裁决；此次目的是检验保存证据是否支撑裁决并修复 STATE 的过期续接点。审计拒绝覆盖自己的 receipt。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  timeout --signal=TERM --kill-after=5 120 \
  /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python \
  scratch/goal_20260926/ra1/audit_saved.py
```

## 结果与研究决策

执行后追加。禁止由本审计自动放行正式训练。
