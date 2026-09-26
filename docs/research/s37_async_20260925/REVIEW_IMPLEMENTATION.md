# C1 最小实现的独立反方审阅

日期：2026-09-25。范围：已冻结 DECISION.md、PROTOCOL_AUDIT.md，新增 streaming_tracker.py、tests/test_streaming.py，并只读补查审阅期间刚出现的 streaming.py 与实际调用的 model.py。本文只做静态源码/合同审阅；未运行模型、CPU 拟合、GPU 或训练，未修改实现和测试。以下四项区分真实代码边界缺陷与尚未满足的系统验证门，不把缺少性能测量说成已测性能失败。

结论：最小复用路线与冻结方向一致，但当前快照不宜宣称完整 Debug PASS。先修 R1/R2，再补 R3；R4 是正式性能和系统宣称的前置门。准确率与创新仍为 HOLD。

## R1 — 缓存借用了输入缓冲区，缓冲复用可静默改变历史（实际代码缺陷，优先修）

证据：`semkine/streaming.py:208–210` 的 `_cat` 在旧缓存为空时直接返回 `new`；`:255–261` 对 xyp/timestamps 切片后保存为 recent/readout cache。第一次非空 append 后，这些成员是调用方输入 tensor 的 view。CPU/GPU 接收缓冲区常会复用：调用方覆盖原 xyp/timestamps 时，缓存中的坐标/时间跟着变，但 h、SAE、last_timestamp 没有一起重新计算。`dataclass(frozen=True)` 不能阻止 tensor storage 改变。状态注释只要求不修改 state tensors，并没有让接收端明确放弃原输入 buffer 所有权。

影响：之后 query 可把旧 h 路由到新像素，或把历史节点提前过期；后续 append 的边会从已改坐标构建。parameter version 无法检测这种失效。大块输入的尾部 view 还会保留整个输入 storage，张量形状有界不等于存储严格按尾部大小有界。

最小修正：仅为最终保留的 recent/readout xyp 与 timestamp 后缀建立拥有独立 storage 的快照，或有可执行的、明确移交所有权接口；无需复制所有新算出的 h。建议做前者，减少运行器隐式要求。

最小回归：append 后覆盖原输入缓冲，之前 state 的 readout 和该 state 上的后续 append 必须与覆盖前保存的参考完全一致；包含第一包、先空后非空包和大于 readout cap 的输入。此项本审阅未执行，不把静态证明写成已跑测试。

## R2 — tracker 静默转换时间类型，且允许 cadence 超过证据寿命（实际接口缺陷，优先修）

证据：`streaming_tracker.py:66–69` 先用 `frame_period_us<=0` 验证，再 `int(frame_period_us)`；输入 `0.5` 会合法通过检查后变为零。`:83` 对 start_us/stream_id、`:99` 对 now_us 静默 int 转换，布尔值或非整数浮点可能被吞掉。这比底层 `CausalEventEncoder._query_time` 的 Integral/int64 检查更弱；无 graph 时 query 完全跳过底层时间检查。

另一独立边界：constructor 不限制 period 与 horizon。若 period>horizon，可在合法 watermark 后接收事件，随后第一次 query 时这些新事件全部过期，`:111` 不更新 pose，`:129` 却无条件清 pending_events。此运行并不违反代码的 cadence 检查，却静默丢掉尚未响应的观测。

最小修正：在任何转换之前统一验证非 bool 的 Integral，start/query 限于 int64；query 的加法也检查溢出。对本阶段仅支持的运行方式约束 `0<period<=horizon`，或对未服务事件过期显式报错/输出 drop accounting，不能仅清 pending。不要修改冻结 horizon 或时间排除边界。

最小回归：浮点/布尔 period、start/query，int64边界、period>horizon，以及正常2/4ms cadence；验证非法配置在接收事件前失败。

## R3 — 现有测试尚不能证明 tracker 的 NoGT、全 mesh 和逐层合同（验证缺口，Debug 不能放行）

证据：`tests/test_streaming.py` 只导入 EventGNN/CausalEventEncoder，未导入 SparseS37Tracker。`check_equal` 只比较最终 feat/h/pixels/mask；DECISION §6 要求每层等价和完整状态/mesh 合同。目前没有对应 tracker 测试、全 MANO 闭环回归或初始化/重置测试。`test_empty_gap_and_no_ancient_edge` 只是与共享年龄规则的 full_reference 比较，`fresh` 仅检查非 None，没有单独证明长 gap 时消息确实为零。

本次正向核查：tracker 的 `_route_nodes` 与 `_fk` 都显式传入 state.betas/K，调用路径不经过 `_resolve_betas_K`，当前源码没有读取 GT 或 `_ctx_betas/_ctx_K`。final `_fk` 位于空新事件分支之外，因此按调用契约每次 query 都生成完整 mesh。这些静态事实值得保留，但不能替代回归证据。`zero beta = training-model mean shape` 的注释也应改为“MANO beta 原点/模板均值”，除非另有真实训练人群均值产物。

最小修正/测试：

1. 增加有限 CPU tracker fixture：无新事件 pose 位级不变但每 query 仍输出 `(1,778,3)` vertices、`(1,21,3)` joints；有新事件只推进一次；固定 cadence、水印、query等时事件排除及跨序列 reset。
2. 给模型 `_ctx_betas/_ctx_K` 填入明显不同/非法值，结果仍只依赖传入标定和冻结 beta；oracle route override 应拒绝。此为 NoGT 隐式上下文负控制，不给系统输入 GT。
3. 在相同事件/state下 cached 与 full-prefix 接到同一 tracker 解码路径，比较 pose、vertices、joints及连续短 unroll。检查真实 routing的 no_grad 是既有离散 conditioning合同，不误称为完整可微关联。
4. 每层 recent cache 对齐完整参考相应后缀；为 token/首次与等时极性年龄、长gap无边写少量手算期望，避免两个实现共享同一个错误仍通过 parity。

## R4 — 7ms 定义清楚，但接收/调度/服务记账还不存在（系统验收缺口，不是测量失败）

证据：PROTOCOL_AUDIT §4 正确要求 `t_mesh_ready - t_first_unserved_event_arrival`；tracker state 只有 sensor timestamp、pending count和query时间，没有host monotonic arrival、first-unserved事件标识、队列/丢弃记账或deadline timer。外部必须在append之前按下一query deadline切分batch；如果提前append跨越下一deadline的事件，query会因last_event>=now拒绝，不能靠回退/排序/改query时间补救。静止时输出也完全依赖外部timer调用。API当前是核心算子，不是完整实时运行器。

最小修正：在正式GPU测试前增加一个薄的、显式状态的因果replay/scheduler，而不改网络。记录每个事件的sensor时间与host arrival，按已定deadline先服务旧query再接受越界事件；停止事件流仍触发timer。对每输出保存 first-unserved ID/arrival、队列、microbatch等待、传输、预处理、encoder、全live route、heads、两次必要FK/最终mesh和可用时刻。只在输出确认后标记事件已服务。readout cap/expiration应留明确计数，不能将未贡献/丢掉事件自动伪装成已响应。

延迟确认还需：正式硬件无竞争；正确GPU同步；2/4ms负载点仅作实现判别，选定周期后冻结精度协议；空事件mesh、恢复和全局刷新不能跳过；不得以拒绝合法包络内事件、漏输出或P95替代max/超限率。旧 `_route_nodes` 在 `S37_DBG` 环境变量打开时会执行额外几何/host同步和日志I/O，运行manifest应记录或显式关闭这个调试状态，不准测后解释掉慢样本。

## 正向边界与未扩大范围

- 本方案确实保留 S37 学习模块/51D 接口，MANO变化重算所有 live route，事件 h 仍可独立缓存；无需凭空增加 xyz 图或 BN 修复。
- 最大边年龄已经写入 streaming.py 的 append/full-reference，避免长静止古老边，这是与新参考算子一致的必要规则。
- 同一时刻输入顺序作 tie-break、查询 `[now-H,now)`、旧权重仅初始化、固定无 GT shape/pose先验、短周期需重新训练均在文档显式声明。不存在本次审阅发现的隐藏 RGB/第二视角输入。
- 冷启动精度、恢复机制、低事件率改进和科学新颖性未证成；这些已经 HOLD，本审阅不重复将它们列为实现 bug。
- 未访问测试集、未搜索新增论文、未覆盖或编辑任何用户源码。

## 审阅快照

下面身份绑定本次意见；文件继续变更时应据此区分“原问题已修”和“审阅时不存在问题”，不能覆盖历史意见。

```
DECISION.md          815e45bcbd567ab440a042948f780f87cbec32ae0076a9ef6f7042fd31a18527
PROTOCOL_AUDIT.md    7cbfad0328cc04d66ced72ae92d00114e4094af103a007ee5bfbbb9c4d6e4bac
streaming_tracker.py 0c7c9888d1f7f87035a8d473ac4588cfdb845af221021585ad935d876ecf8995
streaming.py         92038d31e31fbdf51f9f9f7573ba433162b8feba5db5a1996b373f22c89b9f3a
test_streaming.py    f5a9b1c8be4beb7a1ffdbd7686cfcb115ef9dc6119e3b1e8ec88c6509f49fda8
```

## 后续处置记录（不覆盖上述审阅快照）

主代理已修R1输入缓冲别名和后缀存储保留问题，R2采用整数/int64时间合同并限制period<=horizon，补齐R3的逐层、手算token、长gap和tracker完整mesh/NoGT控制。实际v2共35项通过，见DEBUG.md及原日志。

R4薄scheduler已由本审阅代理实现于统一scratch，标准库模拟时钟只验证调度；主代理随后完成固定训练片段的真实CPU回放，排队和超7ms如实保留。该工具仍不是实测相机接收链路，也没有GPU验收。另一个代理独立发现全模型权重版本失效缺口，修复与v3共41项通过见REVIEW_FINAL_CPU.md。完整Debug并未因这些局部门通过而自动放行。
