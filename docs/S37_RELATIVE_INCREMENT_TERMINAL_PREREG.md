# RI0 — 相对增量有限定义检查终态

2026-09-26。原合同：[S37_RELATIVE_INCREMENT_PREREG.md](S37_RELATIVE_INCREMENT_PREREG.md)。唯一 `ri0_definition_v1` 已 COMPLETED / exit 0，六项均为 `DEFINITION_EXPECTATIONS_CONFIRMED`。这是固定代数及理想事件构造符合预期，不是新估计器、真实 MANO 效用、创新或训练准入。

## 改变的判断

M4 的常量角偏置不改变标量相对转角，不能用它直接否定所有增量观测；但仅将输出改为增量也不能保证未知外观问题消失。两个有限平面反例具有相同位置及速度初态、相同非空事件与全部连续 silence，却有不同增量。关闭“仅相对重参数化就新增观测信息/唯一恢复”的论证，保留有额外空间结构、独立标定或数据先验时的条件增量估计。

本轮未得到一个区别于 normal-flow、CMax、EPBA 或既有关联路线的新实现决定。S37 仍当前臂，C0r HOLD、固定 C1r/R0 REJECT（宽泛观测前提 HOLD）、C2r HOLD；不重开 NF2、PA1 或 MP1，不新增第四候选。

## 固定检查与保存证据

1. 常量偏置 `1/7` 保持角序列差分 `[1/10,1/5,1/5]`，初角不同。SO(3) 固定整数矩阵检查同时确认空间增量不变、体坐标增量共轭。后者只是坐标变换律，不是任意 MANO 的物理 gauge，也不是原 head 轴角相减的许可。
2. 三个 ramp 世界各有完整 16 像素、每像素三个负事件，共 48 个事件。平方触发时间精确为 `[1/4,1/2,3/4]`；终点平方时间 `49/64`。两组两世界事件列表与逐段连续 silence 证书完全相同，非有限采样近似。
3. 同外观 aperture 对的初始 reference 相同；终点位移分别 `[49/64,0]` 与 `[49/64,49/64]`。未知固定增益对的终点位移为 `[49/64,0]` 与 `[49/128,0]`；后者只保证相同初始 phase，绝对 reference 不同，两个世界的纹理也不同但各自时间不变。两组都同 `q0`、同 `qdot0=0`、同 C、恒定照明。
4. 各 silence 段在事件右端记录触发前 `-C` 极限，下一段在同一时刻左端为 reset 后 0；末段 horizon 闭合，最后 residual 为 `-1/64`。单调多项式系数、连续分段边界与 reset 证明整段无额外触发，抽查点不替代该证明。整个逆投影轨迹位于固定有限材料片内部，未事后裁去不利边界。
5. 已知增益且限定二次轨迹族时，首触发可分别恢复系数 1 与 1/2。它不说明事件独立提供了增益或任意触发之间的轨迹。移动理想 step 的速度 1 与 1/2 分别给到达时间 `[1,2,3]` 和 `[2,4,6]`；两固定 log 幅度不改变同速度到达时间。每像素 reset 后残差分别 `-1/8`、`-3/16`，绝对值均小于 C，之后无第二触发。
6. 固定循环时间错配保留位置/时间/极性/计数边际，得到不一致的 `x/t=[1/2,2/3,3]`，因此不符合这个恒速 step 族。它不是被认证的另一物理世界，也不是对真实运动特异性的控制结论。第 6 项的两速度初速本来不同，不继承 ramp 同初速声明。

完整精确记录：`scratch/goal_20260926/ri0/run_v1/exact_records.json`。全部分数按 numerator/denominator 保存，事件时间按正根的有理平方保存。无浮点阈值扫描，无真实事件数组、标签、网络、MANO 或优化。

## 执行与审计

冻结 identity SHA256 `3487641da6faa8ccece7e1b00e3578293ed38c8ec12ecdb0f29f2479f31ec657`，含 14 份源码/合同/独立静态报告/必要旧证据和缓存原作。源码 SHA `ecddde28f4e2be901438cf78f95229e476dd2183ba103ea17621b0d8d726a3af`，合同 SHA `ff9bb3e43c58e734bad51af26d443f9f4acfeacff4e1d708249ee0d912454b37`。

实际调用：`/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python scratch/goal_20260926/ri0/launch.py`。launcher 设置 CUDA 空、OMP/MKL/OpenBLAS/NumExpr 线程各 4、Python assert 启用，再调用 timeout30 秒 + kill5 秒的唯一 worker。只有一个数值 worker，未分片。所有启动/运行/终态登记在 RUNS.jsonl；已有 launch_receipt 或输出目录即拒绝重复运行。

worker 3097148、timeout 3097147、launcher 3097145 均已退出；worker 壁钟 `0.0435389990452677` 秒，外层 `0.0805364929838106` 秒，GPU 费用 0。这些是人工检查运行耗时，不是推理延迟。日志、退出码和精确记录已保存。

父审计 `research_state/audit/RI0_terminal_parent_20260926.json` 重新读取已保存有理数记录，复核身份、完整事件、连续区间、初态/位移、正控制及 PID 退出；没有再执行 worker 或模型。receipt SHA `68d7dd3b1ed7bc9eec9d8c3e7d158d7ef435984f6f065d46ed3b69240061e386`；精确记录 SHA `e6edb5c3e0b608644049a2dc0a9bb1d8a7600bf299a275984d2a8d917899904a`。

三份真实独立报告已在运行前完成并冻结：

- G22/G13：`research_state/debug/RELATIVE_INCREMENT_EQUIVALENCE_REVIEW_20260926.md`，SHA `973db2f0917b8fdc783332d3de774bd9beb3b41c558105fe0b0b15e174f3a845`。数学与原作边界；未运行数值。
- G17/G10：`research_state/debug/RELATIVE_INCREMENT_MECHANISM_BOUNDARY_20260926.md`，SHA `769be81280e2211bca0671b8a9785ab5d5d582649589c4cccc3151bd1dacd3f1`。既有机制、原作笔记和接口；未运行数值。
- G23：`research_state/debug/RI0_CONTRACT_REVIEW_20260926.md`，SHA `dc543659f6bd91d3af620cf15f4bc50e04d7377fa49ba3233b9e8325180edef1`。实际合同/实现静态可执行；未认证 launcher 或运行终态，执行与保存记录由父核。其初始指出的同初速范围、reset 端点及错配范围已在冻结前修复。

## 文献与推论的边界

父实际重读缓存 [CMax2018 原作](https://openaccess.thecvf.com/content_cvpr_2018/html/Gallego_A_Unifying_Contrast_CVPR_2018_paper.html) §2/§2.1 Eq1–3；其事件轨迹对齐已直接使用相对运动和时间。独立报告复核既有 normal-flow、Secrets、EPBA 原文或笔记，访问范围各自列明。HTTP403 与缓存访问均记录于 `research_state/lit/RI0_SEARCH_LOG_20260926.md`，不增加已深读篇数，不声称新全面检索。

可行集合中的两个精确 witness 不代表任意外观/动力学先验下边缘 likelihood 或后验权重相同；学习方法仍可能在任务分布上有效。平面有限 FOV、理想阈值和 step 控制都不构成完整可见 MANO 的普遍不可辨定理，也不要求经验方案先证明普遍唯一性。

本轮得到定义澄清与固定构造执行证据，属于 progress；完整精度/7ms/创新目标尚未达成。原 H50ms、短反馈待答项和用户 111 的既有解释不变。下一准入必须给出当前允许输入下、与已关闭形式不同的具体信息来源或已定位缺陷，并预先说明正反结果会改变哪个实现决定；当前 RI0 自身不提供该新实验。无待恢复数值任务，不因仍有预算而复跑终态或新增无决定作用的探针。
