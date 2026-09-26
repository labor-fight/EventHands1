# RG1：两个已有姿态的被动几何排序

2026-09-25，执行前登记。此项是既有C0r诊断，不是第四候选、不构成闭环新臂。科学合同为 `research_state/debug/REAL_GEOMETRY_NEXT_CONTRACT_20260925.md`（SHA256 `4ce173e264441dd194b6a50d3890cc77dab0273957d83bccbb916fde208d6087`）；独立反方为 `REAL_GEOMETRY_COUNTER_REVIEW_20260925.md`（SHA256 `035cc0e573fda10873f9368351f2d4ec0f92d32f811cf3eec799af124c94524b`）。下文固定其可执行细节；不以事后解释替换原合同。

## 数据和问题

固定八训练主体 `ch,lfz,lpc,lr,ly,lyh,lyq,ycy`，按此顺序，各global/local，取原R0首段0-based索引15/47/79/111，共64包。不使用ylf/zgz、不补选空或困难片段。候选仅原S37自己的prev（A，保持）与pred（B，原提案）。仅复用合法段起点初始化后形成的原历史，不应用本评分选择，不重新产生后续状态。

阶段A只读原NPZ白名单prev/pred/betas/K/end和原events/offsets/tsub，核对R0 input/provenance及原FP32采样；不读取target/pos51/RGB/depth。阶段A写全64条评分、支持、候选身份与幅度；其输出整体hash封存后才启动独立阶段B读取原target，评价21关节joint0对齐的mm误差。不训练网络或调用encoder/checkpoint forward。

原H输入末端额外1ms保留并披露。候选B从prev到pred在51D存储坐标线性插值，alpha为0/.25/.5/.75/1，最近时片并列取早；事件alpha=clip((原窗口相对秒+0.001)/0.05,0,1)。这只是窗末回顾性评分轨迹，不是逐事件时刻已经能输出的因果姿态。

## 几何与固定分数

复用已存在G14/G15纯几何rasterizer，不读取真实深度。其整三角z>1e−3规则并不提供近面裁切：实现包装先要求全部候选verts/K有限，全部顶点z>1e−3；否则整次 `INCOMPLETE_UNSUPPORTED_NEAR_PLANE_GEOMETRY` 停止，不删除面/包，不当作空观察。相机投影/scale沿原源码。

外轮廓为图内四邻域前景–背景分界的前景像素中心，禁止以图外zero padding将FoV截边当手轮廓。这是栅格启发式，不称精确亚像素轮廓。到外轮廓距离为C，到填充投影距离为F；所有原采样事件共用分母，成本min(distance,16px)。空轮廓/空mask的非空事件成本16；满mask C=16/F=0。空事件记两候选同分。

固定计算C/F的原时片配对time、保持时间频数的marg、静态末端end，及seed20260925的极性内时间错配、xy半视野(120,90)循环移位。错配是否改变实际成本/候选差值必须记录，饱和无支持不作机制通过。不能用沿轨迹min距离替代marg，也不能对两候选各自筛点。

选择记录严格为 `S_A < S_B−1e−12 px` 才选A，否则B。空支持/tie回原B；全部保留分母。主项只C_time，其余为已冻结竞争解释，不事后挑最好项升主。

## 固定评价及可作结论

阶段B按原pose_repr/MANO/betas解码A/B/target，比较关节对齐误差；先四包每主体平均，再八主体等权，local/global分开。GT好坏tie容差1e−6mm，保留好/坏两类混淆、regret及各主体方向；不以裸accuracy>50%或事件样本数作证据。

机械描述性标签：C_time在local/global中均比always-A和always-B的主体等权均值少超过1e−6mm，才标 `DESCRIPTIVE_LOWER_BOTH_BASELINES`，否则 `NOT_LOWER_BOTH_BASELINES`。这不是有意义效应量、统计显著性、训练或采纳门。另按阶段A的joint RMS更新幅度，在每动作类取两种单阈值方向、唯一幅度相邻中点及两端常量，计算同样本GT选择最优的描述性oracle；不可部署，不作为严格否定门，不扩展分类器。

被动逐包选择不改变后继prev，因此其汇总不是新递推RA、更不是原H成绩。不因阳性复活georoot/R0/R1/P2/A2，阴性只否定这个有限scorer。没有新科学候选或训练放行。

## 运行前门与资源

先静态代码复核与一次纯合成定义检查：共同事件重排、已构造轨迹交换、零更新/恒定hold、time与marg关系、空/满mask/FoV边界、有效控制和非法输入失败。该检查不得读取真实序列/GT/模型数值。失败先停止定义合同，不进入真实阶段。

随后冻结所有代码/原模型几何/旧rasterizer/资产/输入身份及本预注册副本，一次执行A→score seal→B。CPU单进程工作、最多4库线程，不分片、不GPU；阶段A内部120s、阶段B内部30s，总运行由外层165s+5s清理限制（含两个顺序Python启动及hash封存），不并行工作、不自动重试。该总额明确含额外隔离评价进程，不伪装成原建议的135s单阶段上限。任何超时、身份/因果/数值/采样/几何合同失败，整个运行INCOMPLETE，不改变帧数、节点、时片、score或门重跑；A不完整时绝不启动B。

当前仅代码实现中，未执行合成检查或真实评分。本次原型不接主工程，不生成新架构主行；全部内部数值只记本文与失败账本。

## RG1一次终态 — 2026-09-25T19:51:25.753315+08:00
Run `rg1_passive_geometry_v1`：COMPLETED / exit 0；64/64 固定记录完整，A 无标签评分终态后封存，再由独立 B 读取标签。主项机械标签 **NOT_LOWER_BOTH_BASELINES**。不采纳该固定轮廓选择规则，不扩大网格或重试；C0r 仍 HOLD，固定 C1r/R0 REJECT、C2r HOLD。该标签不是新的统计或创新否定门。
先执行一次18项合成定义检查，全部通过，外层 1.9677353659644723 秒；独立B算术32项人工检查已归档，未读取真实标签或执行FK。实际A→B总外层CPU 5.557777795009315 秒，GPU 0；这不是模型推理延迟。父核75份冻结文件和64份输入artifact、全部封存hash以及保存标量的均值/regret/混淆/主体差值/幅度oracle通过，无FK或评分重算。见 `research_state/audit/RG1_terminal_parent_20260925.json`。
### 固定被动风险（仅本诊断内部mm；不是原H成绩或新递推RA）
| 固定选择 | local：8主体等权 | global：8主体等权 |
|---|---|---|
| always_hold | 26.221919387578964 | 20.823938816785812 |
| always_pred | 25.803830921649933 | 19.94657975435257 |
| C_time | 26.15774643421173 | 20.24284240603447 |
| C_marg | 26.10263964533806 | 20.24284240603447 |
| C_end | 26.18762969970703 | 20.2170567214489 |
| F_time | 26.251460880041122 | 20.28845104575157 |
| F_marg | 26.193948417901993 | 20.297459945082664 |
| F_end | 25.98000779747963 | 20.352847680449486 |
| C_time_perm | 26.229441612958908 | 20.24284240603447 |
| C_time_shift | 25.803830921649933 | 19.94657975435257 |
| F_time_perm | 26.14056369662285 | 20.297459945082664 |
| F_time_shift | 25.803830921649933 | 19.94657975435257 |

上述选择均保留同一原S37历史，不反馈到后续状态。评分既没有生成新姿态，也没有验证被选姿态自身递推后的分布。不能与正式zgz双种子主行直接比较。

- local：原pred有益/有害/评价同分分母为 {'pred_better': 19, 'hold_better': 13, 'evaluation_tie': 0}；C_time混淆为 `{"pred_better": {"A": 9, "B": 10}, "hold_better": {"A": 5, "B": 8}, "evaluation_tie": {"A": 0, "B": 0}}`。平均regret 1.2015200555324554 mm；相对always_pred，主体改善/退化/其余为 3/4/1。评价同分类为空，任何该类召回为NA，不能记0。各主体未舍入数值保留于label_metrics.json。
  同样本GT择优幅度oracle：small_pred，阈值 8.49585048854351，均值 25.67833262681961 mm。仅有限规则族的同样本描述，不可部署，不证明幅度已充分解释误差，不能据此否定所有事件几何方法的必要性。

- global：原pred有益/有害/评价同分分母为 {'pred_better': 20, 'hold_better': 12, 'evaluation_tie': 0}；C_time混淆为 `{"pred_better": {"A": 6, "B": 14}, "hold_better": {"A": 3, "B": 9}, "evaluation_tie": {"A": 0, "B": 0}}`。平均regret 1.09306700527668 mm；相对always_pred，主体改善/退化/其余为 2/3/3。评价同分类为空，任何该类召回为NA，不能记0。各主体未舍入数值保留于label_metrics.json。
  同样本GT择优幅度oracle：small_pred，阈值 9.964671451598406，均值 19.649690687656403 mm。仅有限规则族的同样本描述，不可部署，不证明幅度已充分解释误差，不能据此否定所有事件几何方法的必要性。

### 控制与否定范围
C/F的时间错配在全部64包改变配对成本和候选分差，数值支持存在；但C决策只在local的3包翻转，global无翻转。C_time相对C_marg在local更差、global相同；C_time本身两动作类均劣于always_pred，没有已证正收益可由时间控制消除。数值支持不等于有益时间信息或有效纠偏。
C/F空间错配在64/64包均让双方成本完全达到16px上限，固定tie规则回B，因此空间控制的汇总精确等于always_pred。全部记INCONCLUSIVE_SUPPORT；其较低误差不是“破坏空间证据后性能仍好”这一机制结论，也不能声称该控制排除了事件位置。
64包均非空，采样节点范围197–2048，47包原节点数超过上限；沿用原采样，没有丢包或子集筛选。记录2400个采样节点落于原H末端额外1ms，这一历史时间支持保留披露，不能用此次回顾性轨迹评分声称严格在线因果。近面、数值、输入/节点身份合同均未触发失败。
该结果只否定固定五时片、16px截断、像面轮廓/填充距离对这两个已有候选的预定选择效果；没有全面否定三维关联、事件观测信息、学习型状态校正或其他评分。也没有发现允许直接修改S37的结构性必要性，不因这次阴性复活georoot/R0/R1或添加第四候选。
### 身份及独立解释审查
- `scratch/goal_20260925/rg1/identity.json`：`36319e85a930a8cf4a32ec1177d50633d6f1c13ec7c8e226c9b901b7b926ab02`
- `scratch/goal_20260925/rg1/definition_receipt.json`：`9db9497332679690331fbbb1ea8aedc6998d165a543618cfae7456ea15b75523`
- `scratch/goal_20260925/rg1/review_static.md`：`1c1444911b9f304185ffa19c4e3c7b7ecaf3f81743d30b2852b10c496f532d62`
- `scratch/goal_20260925/rg1/launcher_receipt.json`：`ddbf2a605f2807e3666a0e3a91661a92eede03f9fc4411f1baa3d9304cd044f6`
- `scratch/goal_20260925/rg1/receipt.json`：`2a7d717473e9fd0891f4f5a3123ffb3835e9e213c6a74bb369760c5eb409758d`
- `scratch/goal_20260925/rg1/run_v1/receipt.json`：`c0a2d910a1cc936b05cfe8289de412e4fd9ee4eda7153ef94fa1ee42499fc61b`
- `scratch/goal_20260925/rg1/run_v1/stage_a_seal.json`：`92226144beffc7d0c45b0bbbe1d7bf06d282be2602ba75286a64ae199227f42e`
- `scratch/goal_20260925/rg1/run_v1/score_records.json`：`488d3d2f5393c011509ba7a96d7fa72a9fe4a98ff2e9daba63eaa0f30701968f`
- `scratch/goal_20260925/rg1/run_v1/label_receipt.json`：`4dca8aceb31fa5f5f024c1ca8b163cade0480d95265e7b9d33f34d09b7ecad91`
- `scratch/goal_20260925/rg1/run_v1/label_metrics.json`：`30fa3f256123eb188eb51e07d2b6350f7cb9bdf87429906f21dd14acf39315dc`
- `research_state/audit/RG1_terminal_parent_20260925.json`：`ad2230134cbe4ea1e0be4f7d64b6f4ec0ee8f72e80818e66424b070e4e357283`
- `scratch/goal_20260925/rg1/terminal_science_review.md`：`b14e1d5d0a8c5b50dd74b248c7d038754c8c2a699eaaefe351ca3e2d7c45a25f`

独立终态科学审查只读取上述结果/原合同，未重跑模型、真实FK或评分；其有限解释与父核一致。下一新机制必须先给出允许信息下可区分简单竞争解释的必要性证据；目前没有新训练准入。完整目标仍active、尚未达到联合精度/端到端延迟门。
