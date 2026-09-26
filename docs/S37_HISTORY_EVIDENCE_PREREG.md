# HE0：冻结S37历史事件证据的条件可读性

2026-09-26，真实缓存解码/拟合前登记。数学/源码审查和13项人工CPU定义检查完成后，以identity.json及源快照冻结全文；执行后只另写附录。CD0工程终态已经关闭。本项属于C0r/C1r共同前提的有限诊断，不增加第四完整候选，不重开EGM、TM1、LowG或S27。

## 问题、旧覆盖与研究视角

实际S37每步把过去观测压成51D姿态，下一窗重新编码事件。R0缓存保留了原joint head实际输入：前128维为当前分指mean，随后128为max，256为coverage，末3维为对应prev角。过去这些特征来自已经到达的另一事件窗，但编码/路由依赖当时的自身姿态，不是独立传感器或未加工的原始事件。

TM1新增的是过去51D姿态差，没有测试过去的256D事件池化特征。EGM已实训顶点GRU+两包不detach传递，且失败，但同时改变encoder、路由、读出与训练，不能隔离本问题。LowG/S24/S27及旧ABC-C还覆盖了稳定性/展开，不能称这些机制从未训练。旧未知外观的同事件反例也不因把profile改称marginalization而消失。

本次唯一问题：**在原S37固定自身轨迹上，给定当前读出输入、当前完整prev/pred、生成过去特征的完整旧prev和活动量，过去分指池化特征是否具有额外、可线性读取且对任务有用的纠偏信息？** 阳性仅支持对这个固定条件表示的可读性；不证明51D状态的信息论非充分性、原始事件物理可辨识性、独立新测量、通用记忆有效或新颖性。

四种视角：异常反推（TM1 pose lag失败但未隔离实际旧观测）；抽象上移（当前状态是否足以保留后续纠偏信息）；跨领域类比（状态估计的历史充分统计量，但学习特征不自动是posterior）；实验反转（保留条件空间与二阶尺度，破坏历史残差与当前误差的有符号配对）。近期/陈旧特征对照作为未选变体留档，本次不测recency；未知外观积分、追加收缩正则不进入数值队列。

## 固定数据、观测和分母

只用R0已封存18×128缓存，8主体ch/lfz/lpc/lr/ly/lyh/lyq/ycy的16序列拟合，ylf的global/local为internal dev；原S37 seed3407 step2500自身回读、原50ms H窗口和合法初始化。backbone见过这九主体，ylf前缀反复被旧研究使用，不是新鲜留出或泛化验证。禁止读zgz、原生新事件、checkpoint或其他GT。

阶段A只读取core、prev、pred、n_events、end_ms、betas与旧JSON/provenance。核原R0 extraction receipt、NPZ SHA、128帧/50ms、同一valid run、prev[k]=pred[k−1]字节相同、core末3维=prev本joint、零coverage对应前256维全零。末端标签只在固定后续阶段读取。

逐序列定义k−1 lag，不跨序列、不循环wrap、不读取未来。eligible固定k≥10且当前包非空；前10步与当前空包所有输出保持原pred。每条128步全部计分，历史zero support、无效特征joint都不删除分母。主/控制用完全相同缓存、原历史和共同基线，不写回新反馈。

对joint j：

`X_j=[core_k,j(260),prev_k(51),pred_k(51),prev_(k−1)(51),log1p(n_k),log1p(n_(k−1)),coverage_(k−1),j]`，共416维再加截距。

`Z_j=core_(k−1),j[:256]`，不夹带旧末3维角或coverage；`M_j=1[coverage_(k−1),j>0]`。旧输出pred_(k−1)=当前prev已在X中。归一化与投影只使用16 fit序列的eligible行，不用dev拟合统计量。

## 唯一线性读出与匹配控制

共同基线C：X逐列fit均值/标准差，零std置1，截距不惩罚，固定ridge λ=.01预测target_finger−原pred_finger。原root六维不改，15 joint各自三输出。拟合后C权重冻结；额外历史头只拟合其剩余误差，不重新分配C权重。

历史残差：Z仅在有支持的fit行拟合均值/标准差；使用带支持掩码的共同条件设计做SVD投影`P=(M X)^+ M Z`，`S=M(Z−XP)`，再按全部eligible fit行的残差std缩放。std<1e−6列置零。SVD相对截断rcond=1e−10，正交性仅针对保留数值列空间；报告被截断方向。缺历史支持的S严格为零，不能先全体投影再mask却宣称原正交关系不变。所有eligible fit行，包括M=0行，仍保留在ridge损失分母。

直接逐行符号翻转`D S`虽保留各行幅度与Gram，却可能重引X方向（例如X含x、S=h、d=xh可使DS=x），不能作为匹配条件零假设。最终控制按以下固定形式：

1. 每序列/帧/joint由固定SHA256命名空间`HE0/20260926`生成±1符号；不依赖标签、数值或未来帧。符号随行变化，全局列翻转可被读出吸收，不算控制。
2. fit SVD：S=U_h diag(σ_h)V_hᵀ，r按S的固定截断得到，`H=S V_h[:r]`。只翻转实际保留的H方向，不让已丢弃方向重新进入null。
3. `N0=M(DH−XQ)`，Q在同一有支持fit设计上拟合，去掉重新引入的保留X方向；M在左侧逐行作用。N0的fit SVD为U_n diag(σ_n)V_nᵀ；null秩阈值为`1e−10×max(σ_h,max,σ_n,max)`，不能只按投影后可能全是舍入误差的σ_n,max计算秩。若其保留秩<r则控制不支持，停止不读标签。`N=N0 V_n[:r] diag(σ_h[:r]/σ_n[:r])`，最大重标定增益事前限1e6；超限同样停止，不降低r救场。
4. dev/all只应用固定fit投影、右变换和因果行符号。由此fit HᵀH与NᵀN匹配为diag(σ_h²)，且二者均在保留X列空间的正交补。**最终N不保逐行幅度或全部联合分布**；它是条件线性读出的代数配对控制，不是物理无事件世界。

H/N都无额外截距，维数r、ridge λ=.01及全eligible分母相同。r=0的joint额外校正固定零，不为它选择别的特征。数值门：保留X左奇异向量对H/N的相对投影余量≤1e−8，fit Gram相对误差≤1e−8；归一化分母用max(范数,1)，规则先固定。没有PCA维度、λ、lag或主体搜索。

输出O原pred，C共同基线，H=C+历史校正，N=C+控制校正。所有输出float32、只改finger45维，current empty/warm与root字节保持。历史M=0时H/N额外项相对C严格零；C仍可使用许可当前信息，因此不把“没有历史”误写成整个状态必须原样不动。

## 阶段、门和停止

先运行纯人工合同：当前重复列不能造历史残差、已知独立历史方向可读、zero support保持、fit统计冻结、未来后缀不改已有lag/sign、DS重引X反例与最终控制修复、Gram及正交关系、same-readout/current权重不被再拟合。人工失败不读真实缓存，明确实现错误可另记修复，不能因真实门失败重试。

A：全部18序列身份/因果/掩码封存；各joint H/N表示与变换、rank、活动列、控制差异及数值余量封存，先于任何target访问。每个fit/dev×local/global组，至少一个joint有r≥1且满足：H与N在至少一个eligible有历史支持行不同（max abs>1e−6），并且`||HHᵀ−NNᵀ||F / sqrt(max(||HHᵀ||F²+||NNᵀ||F²,1)) > 1e−6`。仅逐行不同不够：N=−H可被线性头完全吸收，必须作为人工无效null被拒绝。用r×r乘积算上述平方差，避免NxN矩阵；负浮点余量在64×eps×max(两个平方范数之和,1)内裁零，超出则数值失败。全部joint保留。任何需要的数值门/秩匹配失败，或某组完全无控制支持，终态`INCONCLUSIVE_HISTORY_EVIDENCE_SUPPORT`，不读标签，不换样本救场。

B：只读16 fit target，拟合C、H、N；保存全部18序列输出、权重和变换，封存后才可打开dev target。target访问开始/成功各自先后落盘，失败不能误称未读。根/初段/空包及M=0的额外校正字节/零值检查通过后进入C。

C：原MANO CPU解码O/C/H/N及两dev GT，按joint0平移对齐21关节欧氏误差mm，完整128步均分；不是新候选自身递推或正式主行。效用门沿用TM1量级：H local比O至少低1.1mm、比C至少低.5mm；H global相对O及C都不能恶化超过.3mm。记G=C_local−H_local，控制N相对C的local收益不得超过G/2。只有联合效用通过才解释控制：

- 效用失败：`NO_FIXED_PAST_EVIDENCE_READOUT_GAIN`；控制记INCONCLUSIVE_NO_UTILITY。
- 效用过、控制未过：`INCONCLUSIVE_HISTORY_EVIDENCE_PAIRING`。
- 两者都过：`ALLOW_FURTHER_FIXED_HISTORY_EVIDENCE_DEBUG`，仅下一项独立闭环必要性研究准入，不是训练/模型/创新采纳。

固定门失败后不改正则、基线、rank、符号种子、warmup、lag、joint/主体选择或动作规则；不把local收益与其他arm global拼接。运行异常另记FAILED_OR_INCOMPLETE，不能当机制阴性。

## 资源与可续接

新代码仅scratch/goal_20260926/he0/，复用TM1纯数学/输入合同和现MANO。一次CPU worker、4库线程、CUDA不可见，不分片；180秒执行+最多10秒进程组清理。GPU费用0，不训练tracker。全部源/合同/缓存provenance/资产及独立审查在启动前冻结；旧任务不重启。

执行后保存命令、身份、人工合同、A/B/C封印、完整失败及数值记录、wall/RSS/cleanup和独立终态审查，追加docs/FAILURE_AND_CLEANUP_LEDGER.md与research_state续接。仍以S37为当前臂；C0r HOLD、固定C1r/R0 REJECT、C2r HOLD。完整目标包含未达的精度、完整延迟及科学贡献，不由本被动缓存诊断缩减。
