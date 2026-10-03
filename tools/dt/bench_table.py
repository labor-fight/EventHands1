#!/usr/bin/env python3
"""Markdown table of a tools/dt/bench_dense.py result (parameters, MACs, eager and CUDA-graph latency).

    python tools/dt/bench_table.py outputs/dt/reports/bench_dense.json
"""
import json
import sys
from pathlib import Path

res = json.loads(Path(sys.argv[1]).read_text())
env = res["env"]
print(f"测量环境：GPU {env['gpu']}，CPU 固定在 {len(env['affinity'])} 个逻辑核，负载 {env['load_avg'][0]:.1f}，测量时间 {env['time']}；"
      f"全模型锚点 {res['anchor_ms'][0]:.2f} / {res['anchor_ms'][1]:.2f} ms（换算到 1.75 ms 标尺）。其他计算进程：{len(env['other_compute_processes'].splitlines()) if env['other_compute_processes'] else 0} 个。")
print()
print("| 臂 | 参数量 | MACs | eager 前向 ms（lyq_local / zgz_global，原始） | eager 换算到 1.75 ms 标尺 | 相对 dt_base | 闭环每步（含读回）ms | CUDA graph 前向 ms | graph 与 eager 逐位相同的包 | graph 闭环每步 ms |")
print("|---|---|---|---|---|---|---|---|---|---|")
ref = next(iter(res["arms"]))
for name, a in res["arms"].items():
    e1, e2 = a["lyq_local"]["eager"], a["zgz_global"]["eager"]
    g = a["lyq_local"].get("graph"), a["zgz_global"].get("graph")
    ratio = a[f"ratio_to_{ref}"]["lyq_local"]["eager"]
    gtxt = f"{g[0]['forward_raw_median']:.2f} / {g[1]['forward_raw_median']:.2f}" if g[0] else "不适用"
    gl = f"{g[0]['loop_raw']:.2f}" if g[0] else "-"
    eq = a.get("graph_bit_identical_packets_of_40")
    print(f"| `{name}` | {a['params'] / 1e6:.3f} M | {a['macs'] / 1e9:.3f} G | {e1['forward_raw_median']:.2f} / {e2['forward_raw_median']:.2f} | "
          f"{e1['forward_scaled_median']:.2f} / {e2['forward_scaled_median']:.2f} | {ratio:.3f} | {e1['loop_raw']:.2f} | {gtxt} | "
          f"{'-' if eq is None else f'{eq}/40'} | {gl} |")
