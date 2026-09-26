# S37 单目事件 MANO：几何与运动文献审查

审查人：真实子代理 `/root/literature_geometry`。执行日与文献截止日：2026-09-25（Asia/Shanghai）。负责用户要求的③⑪⑫⑬⑭⑮⑯⑰八个视角；这是一个代理的八视角审查，不是八名独立专家，也没有执行新的 GPU 实验。来源索引、精确检索词、阅读深度和代码版本见同目录 `sources_geometry.json`。

## 决策框架与证据边界

决策是：在保留 S37 事件编码与 MANO 输出合同的条件下，新增哪一种观测约束值得进入训练前判别实验。主要验收量仍是仓库定义的 MPJPE-local/global，必须在同 checkpoint、同协议同时满足用户阈值；本文不据外部论文不同数据集的分数给 S37 排名。

已实际读取用户指定两个 skill 的 SKILL.md、创意透镜、idea-card/评分规则、研究检索规则、图 schema/评分规则和 Event4D 失败账本及案例相关部分。跨项目账本仅用于防止“工程通过=科学有效”“投影虚拟线=物理事件边”等推理错误，不把 StereoDETR 的失败当作本手部项目已执行实验。

检索使用 web 搜索、CVF/ECVA/NeurIPS 官方论文页、arXiv 原文、作者项目页和官方 GitHub。优先 2023–2026-09-25 公开研究，并纳入 MANO/EventHands 基础工作；未声称穷尽所有论文。CVPR/ICCV/NeurIPS 的 CCF-A 身份由 CCF 人工智能目录核验；TPAMI 属该目录 A 类期刊。ECCV、3DV、ICIP、IEEE Access 和 workshop 作为紧邻机制例外纳入，**不写成 CCF-A 主会**。本轮没有取得指定年份的中科院大类/小类一区 Top 授权目录，也没有逐项核对 JCR 学科 quartile，故不声称任何期刊的“中科院一区 Top”；该分类缺口保留。2026 新论文均有实际可打开的原文或官方发表页面，没有从年份猜造。

证据状态：论文机制为 `reported`，在源码中看到具体算子为 `observed`，迁移推论为 `inferred`，下述实验为 `proposed/not_run`。若只读到作者摘要/项目页，矩阵明确标记，不能用于原文未确认的细节。CVF 多个 PDF 打开出现 403；有 arXiv HTML 的改读作者原文，没有替代全文的只保留摘要级结论。

## 最影响方案的结论

1. **“事件驱动状态机 + delta + 不确定性融合”已是近邻工作。** E-3DPSM（CVPR 2026）有 direct/delta 预测和 Kalman 式融合，但原文输入是 LNES。官方代码 `base.py:101–105` 直接实例化 `bidir=True`，`s5_model.py:324–325` 的逐步 RNN 拒绝 bidirectional；因此其实时因果模式必须单独核验，不能搬 headline 充当严格因果系统证据。其 `learnable_kf.py:12–13,75,109` 是学习的状态维噪声参数，不能自动解释成按当前事件质量校准的观测不确定性。[G01]
2. **“MANO 可见表面 + mesh flow”已有直接手部先例。** EvHandPose（TPAMI 2024）§IV-A1 使用相邻 MANO 插值、射线第一交点及重心插值形成 mesh flow；源码 `supervision.py:167–184` 读取 `pix_to_face[...,0]` 和第一层 barycentric 权重。它的 dense LNES/flow 表示不符合本任务最终主路径，但机制不能被重新命名为新意。[G02]
3. **原始事件点集也不是新意。** Ev2Hands（3DV 2024）用事件 cloud、PointNet++ 与分割引导 attention 输出 MANO。其 `(x,y,t)` 的第三维是时间，不能解释为物理深度。EventEgoHands++（2026-09-15 公开，作者报告 IEEE Access 已接收）仍是 event frame、YOLO 系列手检测与可见性控制的两手 attention，其 §VI-E 自陈稀疏事件漏检及逐帧抖动。[G03,G04]
4. **normal flow 是受限观测，不是完整 3D 速度。** ICCV 2025 的 VecKM-flow 给出局部事件估计 normal flow 的机制；原文 Eq.3 是对称时空邻域，官方 `estimator.py:158–159` 在 slice 内按空间邻域构图，均没有逐事件因果保证。其 egomotion solver 需要 IMU。只迁移一维运动残差思想，不能直接依赖原推理 API 或其 IMU 后端。[G05]
5. **稀疏几何表征不会创造第二视角。** TESNet、EMatch stereo、3D Feature Tracking 的深度来自双目条件；SpatialTracker 的 lifted depth 来自单目深度模型；DEVO/单目 event depth 依赖静态场景及时间视差。非刚体手指可把深度变化与自身运动混淆，须保留可观测边界。[G08–G16]
6. **本仓库已有反证必须优先。** `docs/S37_ROUTED_READOUT_PREREG.md` §8 和指向的 `S37_XYZ_EVENT_DEPTH_ANALYSIS_20260924.md` 声明已做 exact-ray/xyz-kNN 诊断，未支持直接开新几何臂。此处只核对文档和当前路由源码；原始探针产物由主审计核验。不能因读了新论文便绕过这条内部反证。

## 文献证据矩阵

每行最后两项（迁移/反证）均是本次推断或实验建议，不是作者已经在 S37 上验证。

| ID / 方向 / 发表状态 | 原始假设与已读定位 | 最小可迁移思想 / 最近邻 | 在本任务上的失败条件 / 额外成本 / 最小反证 |
|---|---|---|---|
| G01 E-3DPSM；pose/state；CVPR 2026 | 原文 §3–4、Fig.2与主文causal说明；LNES、S5、direct+delta、learned fusion；官方代码已读 | 把增量与anchor角色区分；近邻 S37、HaPTIC | dense输入、bidir、长期自反馈；S5+第二head+filter成本；固定事件，仅替换未来后输出必须逐位一致；若仅平滑便保留同收益，不能归因异步观测 |
| G02 EvHandPose；hand/flow；TPAMI 2024 | 原文 §III-B/IV-A1、Eq.4–9；LNES+hand flow+ConvGRU；代码确认frontmost face | MANO投影运动可作合法训练约束；近邻 S37路由、G05 | 外观变化/遮挡切换/差历史姿态；插值MANO与raster开销；time shuffle保留增益则时序机制不成立 |
| G03 Ev2Hands；point hand；3DV 2024（紧邻例外） | 原文 §3.2–3.4；同像素聚合event cloud、PointNet++、feature attention；官方model代码 | 不必预测event唯一z即可输出MANO；近邻EventHands | cloud并不等于逐event更新；两手域/采样损失；FPS/kNN成本；打乱时间且保持计数判断到底用了多少时间信息 |
| G04 EventEgoHands++；mesh/visibility；IEEE Access接受状态由作者页确认 | 原文 §III、VI-E；官方代码YOLO-based crop、dense LNES参数；代码revision锁定 | 只有有效对象才交换观测；近邻EventEgoHands/双手attention | whole-hand可见性不等于surface可见性；检测+encoder+attention成本；指级遮挡和单手条件下检查有无虚假跨指收益 |
| G05 Learning Normal Flow Directly From Events；ICCV 2025 | 原文 §2.3、3.1 Eq.3、3.5–3.6；point encoding+normal-flow constraint；官方源码 | 观测只约束image-gradient方向；近邻EvHandPose/TMA | 对称邻域、aperture、corner、噪声；现API cdists+ensemble+CPU transfer；past-only重训与等算力time-shuffle比较 |
| G06 HaMeR；mesh；CVPR 2024 | CVF原始摘要、作者项目输入输出；大型ViT及多数据监督 | 数据/监督多样性应与编码器变量隔离；近邻HaPTIC | RGB外观与大模型不满足主路径；大ViT成本；同训练数据预算下检查收益是否来自数据量 |
| G07 EvRGBHand；mesh；CVPR 2024 | CVF原始摘要+官方repo训练说明；event/RGB互补、degrader | 训练时dropout和噪声增强的“信息可靠性”机制；近邻EvHandPose | RGB补静态信息不可在推理保留；双encoder成本；严格event-only重新训练才可比较 |
| G08 MonoDETR；monocular 3D；ICCV 2023 | CVF原文摘要/引言片段+官方repo；foreground depth guidance、object queries | query相关观测与先验分支分开；近邻S37 joint evidence | learned depth仍是先验，汽车尺寸/场景语义不适用于手；depth head+attention；同投影不同尺度的反例检验虚假depth证据 |
| G09 DynamicStereo；RGB stereo；CVPR 2023 | 原文HTML与官方repo说明；跨时空/双目特征交互 | 历史特征要先关联对齐再累积；近邻TESNet | 第二视角不可借、跨窗口未来上下文不可借；cost volume/temporal transformer；保留同token与算力但破坏对应关系 |
| G10 Selective-Stereo；RGB stereo；CVPR 2024 | CVF原始摘要与论文引言；frequency-aware recurrent update | 小邻域保边、大邻域抗稀疏需按证据选择；近邻DynamicStereo | 不能复制disp cost volume；多频支路成本；与同参数固定核对照并量化跨指边界污染 |
| G11 TESNet；event stereo/depth；ECCV 2024（近邻例外） | 原文 §3.2–3.4；warped past cost volume、stereoscopic flow；官方repo | stale memory须运动对齐且能失效；近邻DynamicStereo/EMatch | stereo+刚性深度不是单目观测；dense3D cost；破坏warp但保留feature/count，若等效则memory不是几何机制 |
| G12 EMatch；event flow/stereo；ICCV 2025 | 原文方法+CVF录用页、官方repo；时间/空间matching共享表示 | correspondence feature可复用而非两个大网；近邻TESNet/TMA | spatial对应默认有另一相机；pixelwise dense correlation成本；mono只保时间match，遮挡交叉时测关联错误 |
| G13 Active Event-based Stereo；CVPR 2025 | CVF原始摘要；双目+红外投影主动纹理 | 仅借“低纹理时观测稀缺”作为边界；近邻TESNet | 双目和主动光均越界，主路径拒绝；没有可用硬件条件；去掉第二相机/投影后深度不应仍宣称独立测量 |
| G14 On-Device Monocular Depth from Only Events；CVPR 2025 | 原文 §3.2明确static/no occlusion；§3.3 denseConvGRU；官方repo | 限定运动模型后才把flow约束解读为depth；近邻DEVO/Dynamo-Depth | 手部独立关节运动违反static；dense head+CMax成本；给同2D flow的不同depth/3D motion组合检验不可辨识 |
| G15 SpatialTracker；3D tracking；CVPR 2024 | 原文方法+CVF；单目depth lifts、triplane、ARAP | 跟踪关联/部件约束与表示维度区分；近邻HaPTIC | learned depth误差与全手rigid假设；depth network+triplane+迭代；depth偏差注入是否被错误固化 |
| G16 3D Feature Tracking via Event Camera；CVPR 2024 | CVF原始摘要明确stereo events输入 | temporal correspondence可帮助高速跟踪；近邻TESNet | 双目三角化不可移植；双目匹配+trajectory成本；左目单独时必须撤销公制3D可观测主张 |
| G17 RPEFlow；flow/sceneflow；ICCV 2023 | CVF原始摘要；RGB+LiDAR点云+events融合 | 2D与3D motion需有明确联系；近邻EvHandPose/G05 | LiDAR pointcloud提供了缺失z，禁止照搬；三模态网络成本；去辅助模态重新训练，不能以原文scene flow作为单目证据 |
| G18 TMA；event optical flow；ICCV 2023 | CVF原始摘要；细时段split、motion lookup、pattern aggregation | 在保持同事件集合下验证时序是否提供信息；近邻EMatch | dense/flow aperture，额外时间切片和lookup成本；顺序毁坏保留分布的negative control |
| G19 EV-LayerSegNet；motion segmentation；CVPRW 2025（workshop例外） | CVF原始摘要；分层affine flow+mask、自监督deblur、affine模拟集 | 分离不一致运动，不以全手刚体消背景；近邻Dynamo-Depth | articulated非affine、分割退化；额外mask+flow；同像素量下关节反向运动synthetic case |
| G20 DEVO；VO；3DV 2024（直接机制例外） | 原文 §3.1–3.2、§4.1；event voxel+learned sparse patch+DBA；repo | 只在有可追踪证据处构残差；近邻CMax-SLAM | 静态背景及尺度不确定；voxel+迭代BA；手充满FOV且独立变形应报不可辨识，不输出伪ego |
| G21 CMax-SLAM；continuous rotation；TRO 2024 | 原文/官方repo；rotation-only BA、Lie B-spline，输入events+calibration | 在事件时间评估连续轨迹；近邻DEVO | 纯旋转场景假设不能给手部translation/deformation；窗口BA成本；独立手指运动反证全局rotation拟合 |
| G22 Dynamo-Depth；mono/motionseg；NeurIPS 2023 | NeurIPS原文p.1摘要/引言、官方repo说明；motion-depth歧义、初始化运动分割 | 用成对不可辨识样本检查depth claim；近邻G14/DEVO | segmentation并未创造独立depth；depth+flow+mask成本；等投影motion-depth配对，不允许模型先验冒充测量 |
| G23 HaPTIC；hand trajectory；2025原文公开，发表状态未额外确认 | 原文 §3.1–3.4；相对首帧depth change、global context、跨帧attention | root轨迹与局部MANO状态分开建模；近邻E-3DPSM/HaMeR | RGB、跨帧非因果、初始depth先验；多ViT+attention；仅past的相同观测合同检查是否仍有收益 |
| G24 RPEP；event hand supervision；ICPR 2026作者页，例外 | 原文方法+作者实验室发表页；标注RGB用于预训练，推理event-only | 合法训练监督与推理输入严格区分；近邻EvHandPose/EvRGBHand | 伪event domain gap与预算不公平；训练生成开销；相同训练监督/步数才可归因新主干 |
| G25 MANO；基础；TOG/SIGGRAPH Asia 2017 | 作者官网模型定义/版本信息，S37资产合同需本地核验 | FK关节、surface vertices、LBS有不同作用 | 先验不等于测量，错误shape会影响scale；MANO解码成本；同投影不同shape/depth组合 |
| G26 EventHands；基础；ICCV 2021 | 官方repo/项目说明；本仓库原baseline来源 | 保留dense对照、输出合同与一致MANO资产 | dense LNES不进入最终主路径；不能把原论文数字与本仓库holdout直接比较；固定checkpoint/observations重评 |

## 八视角审查、最强反例与最小验证

### ③ 单目几何

证据：G02/G05/G14/G22。校准像素给射线 `r=K⁻¹[u,v,1]ᵀ`，未知深度 `λ>0`；历史 `X(θ⁻)` 是状态假设。针孔 Jacobian

`Dπ(X) = [[fx/Z,0,−fx X/Z²],[0,fy/Z,−fy Y/Z²]]`

满足 `Dπ(X)X=0`，单点沿射线位移的一阶投影为零。整体缩放 `X→sX` 不改投影；固定已知公制 MANO shape 可以约束尺度，但这是模型先验条件，不是每事件测得深度。时间 `t` 是观测发生时刻，不是 `Z`。

最强反例：对完全相同的静止event流，多个遮挡关节配置合法；运动表面也可通过同时改变depth和3D velocity维持image motion。最小实验：纯synthetic几何fixture，构造投影保持的状态/运动对；检查任何新association/更新器没有凭空“测”到唯一z。该实验不使用真实test GT历史，不是系统精度实验。

### ⑪ mesh/姿态

证据：G02/G03/G04/G06/G25。mesh surface决定哪些部分可发可见观测；FK joints决定运动链；LBS weights说明某顶点由哪些关节驱动，却不是事件一定来自哪个关节的标签。遮挡顶点可由同关节的可见证据和运动学间接更新。

最强反例：肌理/光照变化在face内部发event，而几何silhouette不动；把每个event都当mesh边会错。最小实验：同状态、同event坐标，分别随机打乱LBS joint labels与保持同指内部结构；若二者同样有效，不能主张语义运动学关联。另需固定same prediction对照joint regressor、FK joints与mesh vertex指标，防止评价对象错位。

### ⑫ 单目 RGB 3D 检测

证据：G08/G22。depth-aware query说明条件化读出有用，但由单目网络预测depth不能作为独立几何证据。最强反例：汽车平均尺寸先验可提供米制depth，迁至不同shape/尺度的手会崩；新增depth head还可能复制上一状态。

最小实验：数据分布不变、冻结同event与prev，给候选depth输入施加同投影尺度偏差；区分event-inferred correction与简单回归训练均值。只有对手shape/相机标定变化稳定才支持迁移；无需新增完整depth head做首轮诊断。

### ⑬ RGB stereo

证据：G09/G10。可迁移的只有“对应可信度”和“历史支持是否仍有效”，而不是stereo depth本身。最强反例：正确的stereo occlusion mask依赖第二视图；本任务没有。历史同一相机由于手指独立运动，不能当作已知外参的虚拟右目。

最小实验：同事件序列、相同采样与缓存寿命，比较不对齐缓存、按历史模型对齐缓存和打乱关联缓存；都必须仅past。若打乱仍保留收益，判为平滑/容量收益而非对应机制。采纳前再做长期漂移及刷新峰值成本。

### ⑭ Event stereo/depth

证据：G11/G12/G13/G14。时间对应并不能替代基线长度。G14原文假设静态且无遮挡；G13还使用主动投影，均不能借来为单目非刚体唯一depth背书。

最强反例：同一像素连续事件可以来自前后不同表面/不同手指，时间接近不代表同物理点。最小实验：可见面/背面先后交替的synthetic事件，标明只用于association Debug的已知surface identity，检查直接更新只进入可见关联；真实闭环仍使用预测历史。若只能GT历史下过，机制无有效性证据。

### ⑮ Optical flow / scene flow / motion segmentation

证据：G05/G17/G18/G19。可使用 normal-motion scalar 建立MANO运动约束，但不应承诺完整flow或3D scene flow。定义图像法向单位向量 `n̂_i` 与法向速度 `vⁿ_i`；对历史表面点候选 `v`，`r_iv=n̂_iᵀ Dπ(X_v) J_v ξ−vⁿ_i`，`J_v=∂X_v/∂q`、`ξ` 是局部pose速度。该式为本次推导，尚未实现/运行。

最强反例：角点、多运动重叠、低event率时估计的法向本身不稳；current state bias使关联Jacobian方向错。最小实验：固定S37采样节点，past-only local plane/flow提示与时间毁坏对照，测其是否提供超出当前 `dx,dy,dt` 特征的方向信息；不要把另一个大flow网络作为默认依赖。前置因果检查：追加/修改未来events不改变已发出的flow或pose。

### ⑯ 3D Tracking / 轨迹与连续时间状态

证据：G01/G15/G16/G23。continuous-time 是在真实 `t` 上定义状态及其演化，不是把fixed window结果插值后改名。historical MANO/fused pose可保留不可见的合理拓扑；其不确定性需要在无观测时不被错误压缩。

最强反例：平滑器改善抖动但增加lag，急停/反向动作使平均误差看似改善而峰值恶化；history error自证式重复进入measurement可导致过度自信。最小实验：同条past-only序列做空包、恒速→急停、片段reset；“完整重算”和缓存更新在预定容差内比较，MANO状态变更造成association变化时必须使几何缓存失效。用相同参数EMA/固定gain作反方control。

### ⑰ VO / SLAM / ego-motion compensation

证据：G20/G21/G22。相机运动与手运动只在有足够背景证据/正确运动假设时可分；S37 fixed external camera不需要为标题添加SLAM。最强反例：手填满视野、背景静止但无texture，VO会把手运动当camera运动；rotation-only模型拟合translation/parallax会错。

最小实验：合成相机rigid motion与手指local motion独立组合，保持相同事件率，验证全局补偿不会抹除手部关节motion；没有相机运动的真实部署合同中，先证明ego支路必要，否则拒绝引入。加入ego模型会增加时序状态、鲁棒估计、几何刷新成本。

## 可观测性与缓存的统一推导

本节是推导和提案，不是已执行结果。若选择从可见表面关联得到的稀疏运动残差，局部信息矩阵为

`H_obs = Σ_i w_i J_iᵀ J_i, J_i=n̂_iᵀ Dπ(X_vi) ∂X_vi/∂q`。

`H_obs` 的小特征值方向没有充分当前观测，不能因为MANO prior正则使总H满秩就声称这些方向被event观察。root与关节项应在同一 `q=[translation,root rotation,finger rotations]` 中先检查耦合；分别最小化两支路可能互相解释同一投影motion。`w_i`若来自历史预测不确定性，应说明与measurement相关，不能重复当独立信息相加。旋转更新在SO(3)采用复合，保留原接口时只在内部转换，避免在axis-angle跨π附近直接混合造成错误。

事件前端的状态无关局部缓存与几何后端缓存分开：新event只失效因果邻接受影响的节点；节点删除、pool最大值赢家失效、BatchNorm/归一化统计变更需额外更新。MANO pose变化导致 `uv,z,visibility,J_i` 变化，不能缓存它们并声称与全重算等价。仅在预先定义的几何位移界内可采用近似缓存，必须记录实际漂移和定期刷新尾延迟。不能通过不给mesh解码/延迟发出结果把≤7ms“测通过”。

## 从独立透镜到不超过三个候选

透镜A（异常优先）：仓库exact-ray替换未显著帮助，但闭环状态偏差仍大，提示“观测关联形式”未必是首要缺陷。
透镜B（跨领域结构类比）：normal flow只约束可见运动方向，对应手部MANO的低维切空间约束，不对应每event独立depth。
透镜C（反转）：无event时保留状态可能是正确行为，强制高频更新反而注入误差；要区分静止和不可见。
透镜D（experiment-first/null world）：先固定event/prev artifact，再毁坏时间或关节身份但保持count/compute，检验哪个信息通道真正承重。

这四条产生的idea seeds聚类为以下三项，编号仅属本分支的审查映射，不与主方案文档的候选编号强行同名。科学机制均 `hold`，必须先通过冻结数据/评测parity。

| 候选 | 科学问题 / 最小修改 | 最近邻与差异边界 | proceed / kill / cost |
|---|---|---|---|
| C0 最小修复对照（必须保留） | 若迁移/因果/缓存/单位合同有实际缺陷，只修该缺陷，保留架构、输出、训练预算 | 不主张新方法；先排除工程问题 | Debug确定失败且修复可复现才proceed；无缺陷就保持S37而非创造修复；开销应近零 |
| C1 仅past的局部motion evidence | 保留S37图与head，只在证实既有事件特征欠缺方向信息后加入廉价normal-motion提示或辅助监督；不独立输出event depth | G02/G05/G18已覆盖flow表征；差异仅可能在观测可识别性、严格因果增量和缓存合同联合验证 | 固定观测控制优于time-destroyed且跨seed同方向才proceed；收益由平滑/更多参数解释则kill；邻域复用 O(Nk)，新增统计/MLP须实测 |
| C2 关联不确定性驱动的有限更新 | 在证明跨指/遮挡association污染后，保留多个局部表面候选及reject，而非硬派唯一depth；观测弱时减少直接更新，运动学间接更新仍合法 | G01/G04/G05已有uncertainty/gating，G02已有可见面；需比min repair与固定gate多出可识别收益 | 若exact visibility probe或association负control不改变结果则kill；K候选 O(NK) 与小head；不得扩大成全mesh GNN |

**交叉质疑：** 几何视角质疑C1的法向/association不可辨识；flow视角要求past-only失效案例；mesh视角要求C2区别于LBS本有软权重；历史反方指出已有exact-ray反证；性能视角反对ensemble flow、densecost与每event MANO全解码；评测反方要求先排除test反复选点。C1/C2若不能独立说明信息增量，不应同时堆入主路径。可选择一个主方案，另一个仅以失败根因已验证为触发条件，未触发保持停车。

**与主审计决策的映射（2026-09-25）：** 主路径为必要的 exact causal-prefix cached S37，使已接收因果前缀的缓存计算与完整重算在预注册容差内等价；这是工程使能，科学增益保持 `hold`。本表 C0 对应必要最小修复；C2 只作为最多一个 visibility/association 备选，且必须由独立根因证据触发。本表 C1 normal-flow 停车，不作为另一条活动备选；不能把已失败 georoot 的同一观测信息改名为“法向残差”或“first-hit”后重新启动。法向残差公式只保留为可观测性反方审查工具，绝不作为已获验证的新方法。

## 后续队列与检索缺口

1. 主审计核对现有稀疏迁移与09-24原始探针，冻结正式可比协议；这是比新论文更高优先级的依赖。
2. 先pure synthetic几何/因果/缓存合同，后train-only机制诊断；GT surface identity只能做诊断上限，绝不能进入正式模型历史。
3. 只对幸存一个候选做小样本梯度/拟合/短程；现阶段外部源码只读，没有复制进主工程、装依赖或运行GPU。
4. 查实论文前须继续针对选定完整机制做反向/前向引文检索。目前局限：RGB stereo主要覆盖matching而非全部stereo 3D detector；RGB monocular detection主要MonoDETR；2026仅对直接相关手/pose进行了明确检索；TRO/IJCV的分区版本未核；部分CVF PDF 403；HaPTIC最终发表状态及RPEP官方代码未核。不能据本轮宣称全面SOTA/首次。
5. 若未来独立新证据支持重启停车的法向运动路线，必须先加查“articulated tracking normal-flow Jacobian observability”，并说明相对已失败 georoot 的新增观测信息；若visibility备选触发，加查“probabilistic event-to-mesh association / EM hand tracking”并打开Ev2Hands参考中的Xue等原文。这是尚未完成的novelty gate，而非已确认空白。

## 可续接资产

- 主交付：本文件与 `sources_geometry.json`；其中source IDs稳定。
- 只读代码快照：`.research/s37_async_20260925/geometry/`，5个官方仓库的tree/commit date及选读源文件，均在截止日前；未执行这些代码。
- 没有新增训练/评测结果；不生成或修改主行，不占用GPU，不覆盖已有S37文档。
