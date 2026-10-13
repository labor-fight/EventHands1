# F 反证交付（只读，2026-09-30）

**结论**：训练无 zgz 泄漏；主行选点乐观约 2–4 mm；§16 三条承重结论过断言；「同编码器绝对根+SO(3)融合」不宜作创新首选。

**证据**：(a) metadata/splits：9×72，zgz∉train；(b) `selection_*` 对网格中位 +2.1–4.3 mm，val=test=zgz；(c) `eval_track.py:159` 仅段首 GT 复位；(d) ridge/MLP 用 train 尾留出，但 H1 的 κ 在 zgz 上选（probe:367–374），滤波在冻结不动点上，非闭环。§12.3.4：16.93 混路由。

**否决**：融合已被 E-3DPSM、EventHands+Kalman、X1c α=0.5（分析.md:512、85d76a1）覆盖；「同图 prev 无关池」未证实，§9.2 R²≈0 高风险。

**最小实验**：冻 S37，训池→绝对 R + 闭环 α 扫描；并 E3 拆 CNN 路由/输出。
