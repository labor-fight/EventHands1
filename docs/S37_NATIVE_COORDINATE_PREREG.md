# NP1 — 固定 S37 的原生细坐标与格中心对照

2026-09-26，执行前合同。属于 C0r 最小修复诊断，不新增完整候选或科学创新。RI0/NG1/NF2/PA1/MP1 等固定终态保持。

## 必要性与决定

D1 已证明原 native 像素到存储行可恢复；`tools/prepare_hand_data.py:179–180` 对 xy 做 floor 后只存 uint8(x,y,p)。现 S37 `event_tokens`、`event_gnn` 和 `route_front_vertex_lbs` 连续读取 xy，旧 NG1 固定原 xy/token 只换邻域候选，未覆盖这一输入政策。源码中的 coarse SAE 是区域活动特征，不是承诺逐 sensor pixel 的物理亮度似然；不能把 D1 mixing 直接说成模型错误。

本次只决定是否值得继续验证**旧权重上的坐标重定位**：细位置信息是否胜过 coarse 格本身可计算的固定中心偏移。主假说是 floor 量化遗漏的细位置会影响当前模型的有用事件—几何关系；竞争解释是共同偏移、整数训练分布或无效路由扰动。没有先验保证影响有益。

沿用 creative challenge 的四个视角：从物理 sensor identity 转到模型实际读取的坐标值；用已知量化映射的条件中心作测量学对照；反向检验恢复精度可能破坏整数输入训练分布；先固定打乱细坐标配对而保留实际选中节点边际的控制。两种可变决定为廉价格中心重定位与依赖原始坐标的 retrofit，均须效用通过才进入后续自身闭环 Debug。

## 数据、状态及四臂

- S37 seed3407 step2500，配置与 R0/NG1 相同；FP32 eval，无训练。原权重、原事件行/次序/时间/极性、原 H50ms、原 K/beta/历史状态、2048 stride 上限、W32/k8/时间尺度不变。MP0新增算术默认关闭；A须与NG1/R0保存预测逐位相同。
- 固定八训练主体 ch/lfz/lpc/lr/ly/lyh/lyq/ycy，各 local/global，R0 首段128步的零起始索引31/63/95/127，共64包。只用NG1 A缓存中输入/状态白名单，不读target。反事实输出不反馈；不称新递推结果或新鲜holdout。
- A：原 coarse 整数 xy。
- B：xy=3/8×native整数坐标，保持240×180单位；floor(Bxy)必须逐行等于Axy，其他列逐位不变。
- C：每 coarse 坐标u的所有合法 native 整数 preimage（ceil(8u/3)至ceil(8(u+1)/3)-1，与sensor边界相交）之均值×3/8；x/y独立。它是coarse信息的确定性重定位，不使用事件分布或标签，不统一加0.5，不改变相机half-pixel约定。
- D：B副本，仅在同一 stride 已选live原行的同(coarse x,y,p)组中，按原行升序将二维native residual tuple循环左移一位；未选行保持B，singleton保持。保存source-row映射，区分映射改变与实际xy改变。保留实际选中节点和全包的坐标多重集、均值、coarse/p分组以及时间/极性/计数；不认证物理合法的另一事件世界。

四臂均使用同一coarse pixel key计算SAE；B/C/D只改变token前两列，inter-event、SAE、极性和时间token须逐位相同。节点src/mask同一；图边、dp、hidden和几何路由允许自然改变。没有native SAE恢复，没有新增维度/模块；同事件不等于同图，路由变化本身也不代表纠错。

## 原始坐标准备与分阶段封存

唯一 CPU `np1_native_prepare_v1`，外层180秒+kill5，单worker/库线程4/CUDA不可见。先通过人工政策测试和人工token/采样合同，再顺序读取16个授权AEDAT4事件流至6.4秒，只保存四个固定50ms窗；允许解码最后覆盖边界的包，但不保存越界行。校验640×480、时间单调、source size/mtime执行前后不变。

每窗native行按原稳定ms分桶次序，逐行核floor xy、p和相对微秒时间与既存完整ev5一致；不得只比总计数。旧NG1 A封印与R0 provenance input hash也必须相同。保存native行/原A输入/四臂/选中行及D映射，逐文件封印。source记录路径、大小、mtime及选中语义行hash；不是全AEDAT内容hash。不得改 data 文件、窗口、事件集合或标定来凑parity。准备失败/超时即停止，不删输出重试。

唯一 GPU `np1_fixed_coordinates_v1`，Debug预算180秒+10清理，空闲非GPU0 L20，经budget_run执行。身份冻结包含当前实际源码与配置/权重/资产、旧封印/白名单、准备输出；旧R0源指纹只作历史来源，不能冒称与当前新增default-off源码相同。GPU阶段先核四臂token/src/原A预测逐位，保存全部输出及控制支持，封印后才读target。若身份、parity、有限性或超时失败，不读后续标签，不改容差或端点。

人工检查覆盖精确三相位中心及边界、空/单例/重复位置、选中子集的二维tuple边际和不跨极性/格、无效输入拒绝；另直接验证当前event_tokens/stride在四臂中的合同，并验证未选行fine xy对节点token/图/路由输入无影响。不用真实数据选择控制参数。

## 固定效用及解释门

全部64包保留，目标为原joint0对齐每包MPJPE(mm)，每主体四包均值后八主体等权，local/global分开。MPVPE去vertex0仅解释性保存。阶段输出封印后解码16序列完整128×51 target字段，只评分64端点，如实记录；无dev/zgz读数。

对B和C各自：相对A local改善≥1.1mm且global回退≤0.3mm才算政策效用通过。原生细位置的额外门：B相对C local改善≥0.5mm且global回退≤0.3mm；D在至少一半local包实际改变≥10%的选中live坐标，并在至少一半local包改变最终输出（平移max差>1e-6m或角坐标max差>1e-5rad）；最后D相对B local退化须≥B相对A改善的一半。没有B效用时，配对机制为INCONCLUSIVE_NO_UTILITY，即使D改变输出也不证明有效机制。

固定分支：B政策效用、B对C额外门和D门全部通过才准后续native坐标自身闭环Debug；否则若C政策效用通过，只准廉价格中心自身闭环Debug；否则停止这次固定权重坐标retrofit。B仅胜A但不胜C/有效D，不能准入native身份机制。任何阳性不等于采用、训练、新颖性、原主行或完整7ms达标。

阴性仅关闭此次固定checkpoint直接替换，不推出重新训练后的所有native表示无用；训练几何增强原来也会rint，B/C属于旧权重未见的fractional输入。禁止在本次结果后扫比例、偏移、k、窗口、种子、主体或错配顺序救场。GPU费用/CPU准备耗时分别记录，均不是部署延迟。
