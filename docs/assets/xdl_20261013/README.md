# XDL 正式文档附件（2026-10-13）

这里保存 `outputs/xdl/` 中正式图表、生成主表和 figure manifest 的逐字节副本，供 GitHub 和克隆后的文档直接阅读。本次文档同步未重新训练、推理或重算科学结果。

图表范围为五个新臂 F0/Q0/F1/Q1/N1 与旧 C0/C1 seed3407；收缩图保留全部冻结 large 探针，恢复图保留每序列24片段、21状态及真实49/50ms时钟。所有比较均为单 seed 描述，不给 CI、p 值或跨 seed 稳定性结论。完整口径见 [综合诊断](../../S37_XYZ_ASSOC_DIAGNOSIS.md#xdl)。

`xdl_figures_manifest.json` 是原始来源快照：其中的绝对路径仍指向生成时的本地产物和源码位置，未改成附件路径。图文件与快照内的 SHA 完全一致；该文件保留原始审计、冻结 bank、14份 raw 来源及曲线数值，不是独立可重放的数据包。`final_main_table.md` 是规范 `tools/report_table.py` 输出的原件副本。

原始审计 SHA：`c5ba02a7b473906fd508e4dfaa10d9aa08aa1c72fb028b01cd94791e573d47e1`。原始权重、数据和完整 raw 保留在本地，继续由 Git 忽略；CPU 复算前提与命令见 [复算入口](../../S37_XYZ_ASSOC_DIAGNOSIS.md#xdl-reproduction)。

附件文件 SHA256：

| 文件 | SHA256 |
|---|---|
| [xdl_large_bank_rho_gt.png](xdl_large_bank_rho_gt.png) | `b657569adc9544b98fdc4749c36ac81c146a33948c4d9c621fc11c9bb8eb878f` |
| [xdl_large_bank_rho_gt.svg](xdl_large_bank_rho_gt.svg) | `01f96888344773a55313563fd29fbe5f42ceb2094af6ba1e6900202192a20910` |
| [xdl_natural_recovery_gt.png](xdl_natural_recovery_gt.png) | `082c70f153ad0cb3d5962171626c74744e25007de5c89be93d5d31a9e8e07749` |
| [xdl_natural_recovery_gt.svg](xdl_natural_recovery_gt.svg) | `33a8e5392dfbd54b823d0bebe2b5d29a9cb36d0dd7d0bb367b0539c3bb92707a` |
| [xdl_figures_manifest.json](xdl_figures_manifest.json) | `5cd38fcddb2fbe2689186c8c02548a1496c35f28355c444b36cbdc8d89c6c895` |
| [final_main_table.md](final_main_table.md) | `c72d3a1dc8d8f552cce90c8fd35f2af5bc63eab04065bf8ff2b24c2d52058740` |
