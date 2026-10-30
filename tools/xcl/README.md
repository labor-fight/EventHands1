# XCL 2026-10-19

独立单seed3407阶梯控制研究。用户计划：`/home/lyq/.cursor/plans/xyz关联gnn控制变量复盘_7ce02791.plan.md`。完整预注册见`docs/EXPERIMENT_RECORDS.md#xcl-20261019`。

当前协调器已经启动，勿再次执行fresh入口；进度读`outputs/xcl/20261019/status.json`，日志`outputs/xcl/20261019/coordinator.log`及`logs/`。原配对/C1协调器禁止重启。

资源固定：GPU2–3 A2F0→A3F0，4–5 C0→A0FΩo，6–7 A0FΩd。先五cell各50步双卡预检（25/50完整验证、26返回训练），全部接受后再各fresh6000，2×512/BF16/legacy_none/worker12/prefetch1不变。GPU4–7只在全训练验收后做统一评价；Latency十主行串行，之后每卡一个进程执行E1两次fresh2590包及E2–E5/冻结权重干预。

A3的74序列深度构建只用CPU，日志`outputs/xcl/20261019/depth_build_full.log`。完整验收时发布独立`depth_manifest.json`和`depth_binding.json`并绑定正式预检接受文件。部分缓存不可用于A3。FΩ投影独立RNG初始化，两臂共初值，上游冻结，原残差数值与零事件门保留。Oracle均不可部署。

活动源码SHA与源码副本由`protocol.py`生成，发布后不修改。最初两次启动注册问题均在任何GPU训练更新之前停止并保留于`registrations/`，分别为历史stat包混入4个源码身份、同一路径多条合法tmpfs挂载记录。原C1/paired冻结从未修改。

环境：`/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python`，默认base环境缺lightning。新目录完整fresh启动命令仅为复现说明，当前状态存在时会拒绝：

```bash
PYTHONHASHSEED=3407 PYTHONDONTWRITEBYTECODE=1 /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/xcl/execute.py --protocol outputs/xcl/20261019/protocol.json
```

`verify_run`逐rank核对完整实际索引、共有packet、初始37state、最终及500步checkpoint模型/Adam/调度、rank同步和CPU RNG轨迹，效果不参与调度。失败stop-owned、保存、无resume。工程预检和单元测试不构成效果结论。

旧XAG/XA归档在`outputs/deleted_20261019/xag_xa_legacy/`。独立清理执行记录`outputs/xcl/20261019/postprocessing/`，仅全部5cell真实预检PASS后才删除授权范围；不会修改活动训练源码/协议/配置/数据。保护C1/paired权重、冻结归档、失败及原来源缓存。历史入口依赖可从归档复原，新入口独立于退役tools/xag。

完整报告由`tools/xcl/report.py`调用原九列formatter生成；三读数、晚期检查点及机制指标另存报告文件，不把旧行或工程预检伪装成新配对结果。最终只作单seed描述性解释，S39不变。
