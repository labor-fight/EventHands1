# XJ：位置偏移与当前运动联合的 MANO 关联读出

本目录是隔离、opt-in 的新训练工具。实验预注册见 `docs/EXPERIMENT_RECORDS.md#xj-prereg-20261029`；本次运行入口由 `outputs/xvm_joint/latest.txt` 指向。

本次首先训练局部能力阶段，不将其称为完整联合网络训练。canonical S43 的 W50 原始分支冻结，提供合法的全手绝对状态提议；前态来自本作业上一输出，段首采用公开初态及两次无标签 warmup。新增局部网络实际训练，不使用 GT 前态作输入。

数据流：

1. 原生事件构造公共 F16：两极性 × 三种 latest-age 衰减，以及当前50 ms的五箱 × 两极性计数。缓存保留 raw count，模型 patch 入口才显式 log1p。
2. 当前计数产生最多24个传感器侧支撑和 Sobel 图像法向；每个支撑保留最多三个不同 LBS 部位的射线角近邻顶点候选。
3. 双时间 F16、两张真实三角光栅轮廓距离图、局部源坐标及曝光掩膜组成37通道16×16 patch。
4. 共享 CNN 分别预测参考偏移 a、真实50 ms位移 d 的法向几何代理及联合误差。相同来源的候选共享关联质量，保留 unknown；纠偏残差大不自动表示错误身份。
5. 固定参考的一次联合102维线性解输出起点／末态修正，使用 physical51 加性轴角 Jacobian。末态修正加到全手提议，之后提交为下一步自身前态。

本阶段目标是 beta0 的同顶点投影几何代理，不声称已知真实事件的材料身份或永久轮廓对应。局部输出依赖候选采样，不是独立传感器；源权重和误差也不自动成为校准后验。当前仅支持 canonical `root_compose=add`，不静默切换旋转更新约定。

局部均值残差零初始化，初始 a=0、d为全手名义投影位移，联合更新严格为0。辅助参考扰动只作用局部查询，不修改全手分支的输入或提交历史。GT／实际形状只在全部前向输入和预测构造后进入独立监督槽。

当前预算为 seed3407、Adam1e-3、B4、12000次真实优化器更新，仅训练 FIT7；SELECT(ycy/ylf)用于预注册的最终能力检查，zgz不读。位置／运动 NLL、validity BCE和末态21关节MSE共同训练局部读出／关联／更新；固定来源组权重不会让 learned alpha 通过归零逃避监督。

先运行对应的新 CPU 契约测试，再冻结实际源／配置／协议／canonical权重／唯一划分。实际输入的前向反向工程检查为零优化器更新，只有该检查对同一 sourcefreeze 通过后才允许启动训练。CPU测试与实际工程检查均不构成机制效果通过。

```bash
/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python -m pytest -q \
  tests/test_xvm_joint_model.py tests/test_xvm_joint_data.py \
  tests/test_xvm_joint_geometry.py tests/test_xvm_joint_train.py

python -m tools.xvm_joint.control freeze --config RUN/config.json --out RUN/sourcefreeze.json
python -m tools.xvm_joint.control launch --engineering --config RUN/config.json --source-freeze RUN/sourcefreeze.json
python -m tools.xvm_joint.control launch --config RUN/config.json --source-freeze RUN/sourcefreeze.json
```

`RUN` 替换为具体运行目录。启动器将授权的物理GPU映射为逻辑cuda:0，并拒绝占用旧作业所在卡、GPU0–1、重复启动和源身份变化。本次优先物理GPU2；16 GiB进程上限与20 GiB空闲余量检查不等于已验证显存需求。

训练输出包括 `manifest.json`、`trace.jsonl`、`status.json`、完整checkpoint和失败现场。checkpoint保存局部权重、Adam状态、随机状态、输入游标、两快照／latest状态、已提交自身前态、绝对分支历史，以及实际输入与监督的哈希链。恢复要求源码／配置身份一致；不以一个权重文件冒充完整续训状态。

本工具不修改 `semkine/encoder.py`、`model/model.py`、`semkine/xyz_mesh.py`、已有训练作业、事件、标签、K或划分。最终是否解冻全手分支和进入完整配对训练，取决于冻结协议中的位置新息及净收益门；本阶段训练启动不表示这些门已经通过。
