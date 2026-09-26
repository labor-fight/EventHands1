# E5.5b unroll 训练三轮 debug 复盘(2026-08-25,session 3eaac2)

> 一句话:闭环展开(scheduled sampling)训练在本任务上连败两版——恒定混合率直接崩溃、
> 退火后"替换式"混合又稀释误差课程使递推恶化——两版失败同根:**把带噪声课程的条件状态
> "替换"成近乎干净的自预测,教会网络错误的"prev 可信"先验**。替换式正式判死;残差式
> (s23,已实现未训练)是该家族最后一个值得测的成员。本文记录全部测量、根因链与踩坑。

对应文档:门与最终数字见 `docs/semkine/EXPERIMENT_LOG.md`(S22 节 real-attempt verdict);
失败裁决入账 `docs/FAILURE_AND_CLEANUP_LEDGER.md`。本文是 debug 过程复盘,自包含。

---

## 1. 背景与目标

S20 闭环裁决确立双支柱:E5.5a halo 路由(修"证据删除",已完成:递推 30.4-32.6 → 27.8 mm,
保留)与 E5.5b unroll(修"训练分布缺自生漂移")。E5.5b 的动机是硬测量:两个前端在递推里
每步更新坍缩到需要量的 11%(0.069 vs 0.61 rad),而 teacher-forced 加噪下能走 0.77-0.93
——教科书式 exposure bias(Ross & Bagnell AISTATS'11;Bengio et al. NeurIPS'15)。
实现:数据集出 `(前导窗, 主窗)` 相邻对(铺贴与递推 evaluator 一致),训练时以概率
UNROLL_P 用前导窗的 stop-gradient 自预测替换主窗的条件状态。

验收门(双 seed,val_core,50 ms):
G-a 单步 ≤ s21+1.1 mm;G-b 递推 ≤ 21.9 mm 或放大 ≤ ×2.10;G-c 递推每步 pose 更新 > 0.2 rad
(坍缩值 0.069)。G-c 是机制门,读的是缺陷本身而非其后果。

## 2. 三轮实验与测量

### 第一轮:恒定 UNROLL_P=0.5(constP)—— 灾难性 FAIL

- 训练面相:train_loss 停在 ~31-33(健康臂 ~0.4,差 75 倍),val_loss 40.9(val 是
  teacher-forced 无噪的,跨臂可比,此值=真崩溃);ckpt step500/1000 在 val_core 上 NaN。
- step=3500 提前探针:**TF RA 42.5 mm / 递推 144.5 mm**。42.5 恰是历史"绝对回归"档
  ——网络的最优解是无视 prev,任务退化为从单窗事件做绝对回归。
- 根因:从 step 0 就有一半样本的条件来自未训练网络的输出(与位姿无关的噪声),
  "读 prev"在这一半样本上是负收益,网络学会不读。Scheduled sampling 原文的退火
  curriculum 正是防这个,第一版漏了。
- 产物留档 `outputs/semkine/archive/s22_constP_fail_s340*` 作负对照。

### 第二轮:UNROLL_RAMP [500, 2000] 线性退火 —— 训练健康,门 FAIL

修复:p_eff = UNROLL_P × ramp(global_step),step≤500 纯 teacher-forced(与 LR warmup
重合),2000 步后满 0.5。训练立刻恢复健康:30 分钟处 train_loss 0.507 / val_loss 0.125
(constP 同期 39 / 40.9,差两个量级)。双 seed 16:32-20:09 训完,网格完整。

期间另有一坑(用户侧发现并修):某两次尝试的 `TRACK.UNROLL_*` 键**无任何代码读取**,
训练成了 s21 的静默复制(no-op 臂)。用户重实现(`unroll_p_now`/`_maybe_unroll`)并加
`MNISTModel.TRACK_KEYS` 白名单,使未实现的 TRACK 键从"静默忽略"变为**构建期报错**。
真臂的行为学指纹:递推每步更新 0.105-0.192 rad vs s21 的 0.062(3 倍差,远超复制方差),
确认 unroll 真实生效。

### 21:44 四臂探针终审(`closed_loop_sensitivity_unroll_50ms.json`)

| 臂 | TF RA | 递推 RA | 放大 | TF@6.6mm | TF@26mm | 递推每步更新(rad) | 回声超额 |
|---|---|---|---|---|---|---|---|
| unroll s3407(选 step1000) | 10.92 | **34.22** | ×3.13 | 11.68 | 19.91 | 0.192 | ×2.22 |
| unroll s3408(选 step4000) | 10.26 | **31.03** | ×3.02 | 11.02 | 18.36 | 0.105 | ×2.26 |
| s21-halo 锚 | 12.11 | 27.76 | ×2.29 | 12.55 | 18.26 | 0.062 | ×1.85-1.95 |
| LNES 锚 | 10.95 | 20.87 | ×1.91 | — | — | 0.071 | ×1.74 |

门判定:**G-a PASS**(且优于 s21 达 1.2-1.9 mm);**G-b FAIL**(31-34 vs ≤21.9,比要改进
的 s21 还差 3.2-6.4 mm);**G-c 半解除**(0.105-0.192,s3407 擦线)。选点网格:s3407 从
step1000 起单调恶化(34.22→35.14),s3408 平缓改善(31.51→31.08),网格内趋势 ±1 mm
属底噪,绝对水平才是信号。

### 根因链(三个门合起来才看清)

1. 前导窗自预测离 GT 仅 1-2 mm(TF 单步本来就准)——**替换**进来的条件几乎是干净的。
2. 于是一半样本的 6.6-26 mm 噪声表课程被顶掉:**误差课程稀释**。测量:g 斜率(条件误差→
   单步误差)从 0.29 陡化到 0.38-0.42;小误差端变好(G-a 改善与此同因),大误差端变差。
3. 递推稳态坐在大误差区:halo 已压到 ×1.85-1.95 的回声超额回弹至 ×2.22-2.26,递推恶化。
4. G-c 的解除(更新量 3 倍)说明网络确实"敢动了",但动的质量差——**学会了动,没学会往哪动**,
   因为它同时学到了"prev 近乎干净"的错误先验。一个为对抗 exposure bias 设计的训练,
   反而拉大了 exposure gap。

**判决:替换式 unroll(恒定/退火两变体)正式判死,入账本。** 文献机械搬运的教训:Bengio
原文替换的是**干净 teacher token**,我们的 teacher 本来就带噪声课程(合成 DAgger),
同一操作在不同基线上语义相反。

## 3. 已实现未训练:s23 残差式(该家族最后一员)

`TRACK.UNROLL_RESIDUAL: true`(config `configs/semkine/s23_keg_unroll_resid.yaml`):
条件 = `(GT + 课程噪声) + (前导自预测 − 前导 target)`——课程保全,只注入自误差的时间相关
**结构**。合同测试 `test_residual_unroll_adds_the_lead_error_on_top_of_the_noise` 已过。
可证伪判据:若 s23 递推不破 s21-halo 的 27.8 mm,两窗训练分布家族整体关闭(其 1-2 mm 的
自误差幅度本就无法代表 27 mm 的稳态),下一个单变量转 **E5.6:lift 内 state-free fallback
通道**(g 斜率本身,R1 根因——12 个状态条件通道在漂移状态下"值错误"的问题)。

## 4. 踩坑清单

### 实验设计坑
1. **scheduled sampling 不退火 = 崩溃**。退火不是超参装饰,是防"早期自预测=噪声"的必要
   curriculum(constP 的 42.5/144.5 是代价)。
2. **"替换"与"叠加"在带噪声课程的基线上是两个方向相反的操作**(本轮核心教训,见 §2 根因链)。
3. **两窗 unroll 的覆盖上限**:自预测误差 1-2 mm,闭环稳态 27 mm+,差一个量级——单步展开
   可能根本采不到目标分布,设计时就该核对幅度(这也是 s23 的预注册死刑条款)。
4. **config 键无代码读 = 最危险的静默失败**:两次训练成了对照臂的复制品还带着实验臂的名字。
   修复范式:键白名单(`TRACK_KEYS`),未实现的键直接构建报错。

### 判读坑
5. **Lightning 进度条 total 含 val 批次**:曾据此误判"max_steps 失效、优化器步频减半"。
   硬证据是 ckpt 内的 `global_step`/`loops` 计数(torch.load 直接读),不是进度条。
6. **loss 跨条件分布不可比**:unroll 臂 train_loss 高可以是语义性的("从漂移恢复"更难)。
   可跨臂比的是 val_loss(teacher-forced 无噪、同分布)——constP 的 val_loss 40.9 才是
   崩溃实锤,而退火版 train_loss 也偏高但 val_loss 0.125 健康。
7. **选点网格 ±1 mm 内的趋势是底噪**(双 seed 方向都能相反),复制底噪 1.1 mm 之内的
   单点差异一律不解读。

### 基础设施坑(多人共享机)
8. **`pkill -f <串>` 会匹配自己所在 shell 的命令行**,把自己连带杀掉,输出全丢。
9. **`cd X && cmd1 & cmd2` 的作用域**:`&` 只把 `cd X && cmd1` 放后台,`cmd2` 在旧 cwd
   执行——两次日志重定向落错目录。后台链一律用绝对路径。
10. **文件/进程会被并行的人工操作竞态**:两次归档失败 run 目录,均在 1-2 分钟内被移回原路径,
    重启进程被连根杀——当时像"神秘恢复机制",真相是用户本人在实时调度(恢复目录、清 GPU、
    自己重启训练)。教训:动共享产物前先确认没有人在并行操作;绕不过就换全新实验名
    (旧路径原地留作负对照),不要对抗。
11. **GPU 归属要查证**:`nvidia-smi --query-compute-apps` + `ps -o user,cmd` 确认每张卡上
    是谁的任务;本机 GPU 在多用户间小时级流转,启动前的空卡几分钟后就可能被占。
12. **环境变量不会自动继承给用户手动启动的进程**:调试插桩以 `EVENTHANDS_DEBUG_SESSION`
    为门,用户自启的训练没带它,插桩静默——不是 bug,但取证时要先确认插桩链路是否激活
    (本轮靠 keg 侧日志的时间戳反推出三次启动各自的身份)。

### 有效实践(下轮沿用)
13. **早鸟探针**:训练中途拿 step=3500 ckpt 跑单臂探针只花几分钟,提前 1.5 小时定生死
    (constP 因此少烧数小时 GPU)。
14. **行为学指纹辨臂**:怀疑臂是静默复制时,不查代码历史,直接比行为(每步更新量 3 倍差
    即实锤),比考古快且硬。
15. **ckpt 是最好的黑匣子**:`hyper_parameters` 存 config 快照、`loops` 存精确步数,
    比日志和进度条都可信。
16. **失败立即入账本**(`FAILURE_AND_CLEANUP_LEDGER.md`),负对照产物留档不删,
    防止复投已死方向。

## 5. 资产与状态(截至 2026-08-25 22:00)

保留的代码(均有测量或合同测试支撑):
- E5.5a halo 路由(`semkine/keg.py:_halo_channels`):递推 -2.6~-4.8 mm,双 seed 一致。
- `UNROLL_RAMP` 退火 + `UNROLL_RESIDUAL` 残差开关(`model/model.py`);s22/s23 config;
  合同测试 5 项(`tests/test_s22_unroll.py`)+ halo 2 项(`tests/test_s18_keg.py`),
  连同既有测试 27 项全过。
- 探针工具的 delta 统计与敏感度扫描(`tools/run_closed_loop_probe.py`)已成常驻功能。

已清理:全部 debug 插桩(keg.py / model.py / 探针内 NDJSON)、一次性静态探针
`tools/probe_gnn_info_loss.py`、debug 日志文件;无残留后台进程。

下一步(按裁决树,未启动):训练 s23 双 seed → 若递推 < 27.8 则家族存活继续,
否则关闭训练分布方向,转 E5.6 state-free fallback 通道。
