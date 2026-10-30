"""Render the new attribution section from immutable measured probe results."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_revisit_20261028'


def read(name):
    return json.loads((OUT/name).read_text())


def render():
    p2 = read('rep_p2/summary.json')
    direction = read('rep_direction/summary.json')
    optimization = read('interface_optimization/results.json')
    interface = read('interface_audit/results.json')
    assert p2['execution_complete'] and all(p2['source_files_unchanged'].values())
    assert not p2['new_zgz_opened'] and not optimization['zgz_accessed']
    assert optimization['all_predictions_saved_before_pose_GT']
    rows = []
    for arm in p2['metrics']['rbf']:
        a, b = (p2['metrics'][head][arm]['SELECT'] for head in ('rbf','mlp'))
        rows.append(f"| {arm} | {a['abs_mm']:.3f} / {a['ra_mm']:.3f} | {b['abs_mm']:.3f} / {b['ra_mm']:.3f} |")
    short = []
    for recording, models in optimization['results'].items():
        values = []
        for checkpoint in ('canonical','s43_step500','s43_step2000','s43_step6000'):
            v = models[checkpoint]['whole_scored_prefix']
            values.append(f"{v['MPJPE_abs_mm']:.2f} / {v['MPJPE_J0_mm']:.2f}")
        short.append(f"| {recording} | " + ' | '.join(values) + ' |')
    return '''<a id="s43-part-09"></a>
### 第九编｜REL 完成后的失败归因与新问题（2026-10-28）

**本轮最有证据的结论是：方向统计并非完全无信息，但主要观察到小幅绝对位置条件收益，尚未建立稳定的局部关节运动价值；旧接口确实删掉了一部分新证据路径，而更完整的末级Gram读出没有解决任务问题。** 这三件事要分开，不能统称“新表示没用”，也不能由修好接口就预告新方法会成功。

本轮从[完整REL结果](EXPERIMENT_RECORDS.md#s43-rel-doc-20261027)继续，没有重跑四臂训练、54条最终完整回放或改写旧裁决。源码/配置/逐帧结果、相关历史OPKAC/CATA、SBA、MV2/OBS及表征稿按问题追踪；目录索引不等于全部历史逐字复核。新结果在 `outputs/s43_revisit_20261028/`，旧数据、K、标签、默认模型及REL交付包只读保留。

#### 9.1 本次小实验与判读边界

[Stage1协议](../outputs/s43_revisit_20261028/stage1_protocol.json)在新任务读出分数前冻结；[先验披露](../outputs/s43_revisit_20261028/stage1_prereg_disclosure.md)明确：旧结果/源码/曲线已知，六项无标签代数工程检查在冻结前已做，不伪称前瞻任务结果。方向拆分和检查点诊断是随后提出、各自读数前登记的扩展，不能改原门或据此择checkpoint。

1. P1：实际调用冻结读出核心的代数碰撞、正控及零证据检查。2×2 cell alias、法向一二阶矩、m→f零空间、每槽G的非对角项都可能碰撞；正控同时证明某些m损失不改变线性二次任务最优解。数学非单射不等于当前自然数据的MANO任务信息损失。[回执](../outputs/s43_revisit_20261028/rep_audit/audit.md)。
2. P2：复用原E12缓存288个50ms片段，FIT224/SELECT64、共同prev/K/β/target；八包×RBF/MLP。RBF带宽只用FIT，MLP固定JL512、128×2、300步，规范化与块预算预先登记。SELECT只有ycy/ylf两人，而且已被canonical和完整FT训练见过，只是本次头的留出；不是新测试集、完整流或硬件数据。[结果](../outputs/s43_revisit_20261028/rep_p2/results.md)。
3. 方向拆分：这是已知P2之后提出的条件诊断，保留同一full支持C/diagG；full沿用主P2已重新训练的头并绑定文件SHA，扩展另训f-only、support-only、空间无向f三包×两头，共六次拟合；不是冻结删支路后评价适配失败。[登记](../outputs/s43_revisit_20261028/stage1_direction_extension.json)、[结果](../outputs/s43_revisit_20261028/rep_direction/results.md)。
4. 接口/优化：相同FIT工程前缀的实际梯度与路径贡献（不读target），以及四条固定训练域首2s×四检查点共16条短流，公共初始化、同CPU执行，全部预测落盘后才评分。[接口](../outputs/s43_revisit_20261028/interface_audit/results.json)、[优化登记](../outputs/s43_revisit_20261028/stage1_optimization_extension.json)、[短流](../outputs/s43_revisit_20261028/interface_optimization/results.json)。没有新zgz选分，没有新GPU训练。

#### 9.2 有向Z到底提供了什么；读出是否是主瓶颈

以下是有限头的SELECT绝对/J0对齐关节误差mm，**不是完整S43主表**：

| 读入包 | RBF abs / RA | MLP abs / RA |
|---|---:|---:|
''' + '\n'.join(rows) + '''

更完整的slot-Gram相对生产包RBF反差0.240mm、MLP只省0.049；point-factor只省0.138/0.087mm且录制方向混合。均未达到两头≥max(0.25mm,2%)、3/4录制同向且无>0.5mm恶化的预定门。因此**当前不支持把C/diagG这一级压缩定位成失败主因**。这里的“生产包”指A/f/C/diagG统计包接本次共同有限头，并未复制实际S43适配器的75→32逐行MLP与均值池化；不能据此排除该32维池化的额外损失。缓存不含m/selected-Z，也不能排除更早的选边、法向投影和m→f损失；JL本身、有限样本与有限优化也保留。

共同支持上加入有向f，两头绝对误差分别从26.062→25.077、28.441→27.831mm，四录制均同向；RA仅从10.340→10.225、8.951→8.772mm，分别只有2/4和3/4录制同向。无向f近零，与support-only保存预测逐位相同。**方向有小幅条件预测价值，位置分量更一致；尚不能称其为可靠的局部手指测量。** 裸几何a在RBF上甚至达到10.146mm RA；支持包包含几何先验及活动分布，不能将它相对简单输入的全部优势归给事件对。f又不等于速度：纹理/极性、重复与反向运动、闪烁、遮挡及光流孔径仍可能给同一矩。

#### 9.3 旧主路与训练究竟限制了什么

实测功能前向与生产前向逐位一致。在已训full/ch工程输入中，旧mask一像素halo以外975个非零A条目到输出的Jacobian严格为零，改单个far像素输出不变；mask内和halo有非零梯度。**不是所有mask外输入都无效**：3×3卷积能看一像素halo，必须保留这个例外。

合法旧事件前缀后附**明确声明的合成100ms无事件后缀**，历史f/λ仍非零、typed门前范数0.01759，但旧空LNES门令输出=prev且新支路梯度全零；工程性去门恢复非零。它证明旧接口不能完全表达新记忆的声明语义，不证明空窗应该总更新或去门一定提高精度，也不提供自然数据中这种情况的发生率。新方案必须独立定义历史证据、事件新增与纯查询。

零初始化也不是死支路：初始化时typed隐藏层梯度为零，但输出权重梯度98.72、sensor权重梯度5.24，存在起步路径；这些范数来自固定seed随机标量投影的工程Jacobian，并非任务loss梯度。训练后删sensor/typed/both使FK输出分别移动约2.275/2.922/2.553mm。这里没有GT评分，三者还会相互抵消，**输出变化不是精度收益**。

16个固定短流的绝对/RA mm如下，这是**auxβ给定的条件诊断，非β0部署**；每格38个已评分输出，首100ms为公共canonical预热；完整与后半段、根/各指读数均保存：

| 训练域录制 | canonical | step500 | step2000 | step6000 |
|---|---:|---:|---:|---:|
''' + '\n'.join(short) + '''

6000步相对较早检查点总体改善，说明这一小预算并非已经不再学习；这些片段上也不支持单调早停叙述。四条都属于完整训练域，不能以FIT/SELECT名称把差异解释成新人过拟合。结合旧zgz没有超起点，只能说拟合/递推初始化与泛化权衡仍混合；没有隔离足够证据将其唯一归为LR、BF16、BN或核心假设失败。canonical的LR0.004与FT0.0001相差40倍，但有效batch1024对32，LR/batch比只约0.8；单看LR数值不足以判“优化太弱”，这个比也不能使Adam/截断/批噪声的两种优化等价。旧FT192k输出仅历史6.144M的1/32，旧BN冻结且新输入域改变，因此更不能把本轮当独立新前端的充分训练。

[权重与训练配置审计](../outputs/s43_revisit_20261028/interface_training/results.json)还确认：四臂输入摘要逐位配对；45个BN running buffer与预训练初值逐位一致；S43/full主干权重相对变化3.44%/3.67%，不是未更新。梯度norm>10的比例S43/enhanced/no_order/full分别约49.4%/55.0%/55.3%/72.5%，full更频繁触及统一clip。它支持后续检查梯度尺度与适配，不单独证明梯度裁剪导致负结果；不得看完分数再替原门辩护。

#### 9.4 哪些成本必要，哪些属于当前实现

部署β=0完整服务既有实测mean59.38ms：ingest25.45、几何20.34、读出4.23、表示0.85、网络及观测传输6.73、末端mesh1.39ms，其他传输/提交另列；不能重复相加父子项。REL固定核心18.62MB/序列，另有拓扑、raw窗口和workspace，不以进程RSS代表示字节。

- 对**既定显式Z状态**，每次事件维护有限邻边与各τ/γ槽、保存极性和正滞后，是所选统计的真实成本，不能全归咎Python/CPU。局部写入不意味着低常数或整体低延迟。
- 当前wrapper每timestamp仍用`bincount(minlength=2C)`及active扫描，含O(C)工作；数学局部不等于实际实现每一步均严格局部。这是可以独立优化的成本，区别于邻边多槽维护本身。
- 对**单目MANO任务**，全边Z、每query完整表面轮廓/J以及当前CPU实现不是原理必需。可换观测对象或几何近似，但必须重新证明信息和任务作用；不能把尚未实现的GPU化收益当测量。
- MANO几何/状态解释、网络与778顶点/21关节输出各有全局部分；当前均未实现逐事件局部神经执行。查询变密也会重复这些成本。
- 已有geometry_fast是数值工程备选，微基准FK约2.08倍、完整bind仅1.30倍，不等于整链训练/递推已验证更快。形状/拓扑缓存可以复用；姿态改变后的投影、可见性和读出依赖不自动命中缓存。[逐项成本来源](../outputs/s43_revisit_20261028/cost_audit/results.json)。

#### 9.5 新文献真正改变的判断

新增[18篇事件/对应/运动方法卡](../outputs/s43_revisit_20261028/literature/event_motion_review.md)含IJCV2024、TPAMI2023/2024三篇顶刊与2023—2026正式主会条目，附[逐页/片段读取范围](../outputs/s43_revisit_20261028/literature/read_scope.json)和出版核查；另有[mesh/单目检测/事件双目与深度补充](../outputs/s43_revisit_20261028/literature/root_geometry_transfer.md)、[稀疏异步补充](../outputs/s43_revisit_20261028/literature/root_sparse_transfer.md)。这是范围综述，不宣称穷尽所有论文；摘要核查和方法阅读分级，不把下载PDF当全文精读。

| 问题方向 | 实际可迁移思想 | 必须舍弃的过强推论 |
|---|---|---|
| mesh/姿态与单目检测 | 区分2D观测、几何假说、形状/尺度nuisance；数据与训练机会独立计账 | MANO先验不能创造额外测量；大RGB模型精度不能认证事件信息 |
| RGB/事件双目3D、深度 | 在歧义尚未解除时保留几何假设分布；对应身份与任务运动可以分流 | 双目基线/其他模态提供的度量证据不可移给纯单目 |
| flow、轨迹、运动分割 | 联合xy-t关系、时间上的运动变化与多运动模式；身份不变描述与方向证据区分 | 单normal-flow、高计数支持或kymograph边际不是完整运动/深度 |
| VO/SLAM、补偿、逆渲染 | 用可部署历史解释观测、明确静态/外观/阈值假设，背景与运动噪声分开 | 静态场景/已知帧锚/准确姿态不能暗中成为单目手部部署输入 |
| 事件轮廓关节、像素—表面 | 软对应/连续面、遮挡生成点变化、null/未解释观测 | 轮廓永久材料身份、当前面命中就是物理来源 |
| 概率关联/多假设 | 不把多峰均值当真实手态；coverage、排序、校准分开验证 | Gram支持、混合尺度、top-K分数不是自动校准后验 |
| 稀疏异步/增量 | 依赖图与缓存失效、批/在线等价、更新传播域及实际完整成本 | 局部sensor写入不是逐事件局部网络；跨权重版本状态不自动有效 |

直接限制重设计的近邻包括[低秩非刚体轮廓跟踪](https://is.mpg.de/publications/xue2024evnonrigid)、[DCT-Flow](https://doi.org/10.1109/TPAMI.2024.3361671)、[ETAP](https://openaccess.thecvf.com/content/CVPR2025/html/Hamann_ETAP_Event-based_Tracking_of_Any_Point_CVPR_2025_paper.html)、[DAGr](https://www.nature.com/articles/s41586-024-07409-w)。因此软face关联、沿候选轨迹查相关体、骨管、top-K或异步图本身不能再作为主创新。相对这些工作的新增命题必须在当前手部观测中被直接区分。

#### 9.6 收敛到的科学问题与复用边界

**核心科学问题：在根运动、形状/尺度和外观的共模混淆下，能否用有界的单目事件表示保留足以区分局部关节轨迹、并能反驳错误前态的证据，而不是只更准确地重现已有姿态先验与活动分布？**

这要求同时证明：保留的差异确实影响MANO所需目标，部署所得运动学历史确实改变证据表达，增量维护的总成本有实际收益。不能只证明时间被保存、GT tube有效或某一小头的总误差略降。当前最有证据的是接口约束与方向/位置分量差异；早期m/选边压缩的真实任务代价、局部高频物理信息、独立新人/真实传感器泛化仍未闭合。

可直接复用的成果是原生来源/同刻组/半开前缀、reset-ingest-readout-commit分离、训练与在线一致性、MANO与原监督/J0-V0评分、完整流与成本测量、对照配置和已经通过的107项旧测试。后续只新增机制要求的检查，**不把继承测试数当本轮新方法验证数**。三案重设计与100视角交叉反驳另行登记，当前草案不冒称通过原型或已成Nature级方法。
'''


def main():
    content = render()
    # 2026-10-28: docs/S43表征调研.md was merged into the research document (附录 N); same markers.
    path = ROOT/'docs/网络结构与可观测性诊断研究.md'
    start, end = '<!-- S43_REVISIT_STAGE1_BEGIN -->', '<!-- S43_REVISIT_STAGE1_END -->'
    text = path.read_text()
    section = start + '\n' + content + '\n' + end
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + section + text[b:]
    else:
        text += '\n\n---\n\n' + section + '\n'
    path.write_text(text)
    (OUT/'attribution_report.md').write_text(content)
    print('rendered Stage1 attribution; old sections preserved')


if __name__ == '__main__':
    main()
