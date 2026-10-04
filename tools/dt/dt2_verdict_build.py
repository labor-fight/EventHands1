#!/usr/bin/env python3
"""DT2: build docs/DT2_VERDICT.md from its template and the generated tables (numbers are not copied by hand).

    python tools/dt/dt2_verdict_build.py --template outputs/dt2/reports/DT2_VERDICT.tmpl.md --out docs/DT2_VERDICT.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REP = REPO / "outputs/dt2/reports"
PY = sys.executable
ARMS2 = ("dt_dz_l3_nfh dt_dz_l3_nfk dt_dz_l3_w50 dt_dz_l3_w100 dt_dz_l3_lf dt_dz_l3_nosw dt_dz_l3_ema dt_dz_l3_wd dt_dz_l3_reg "
         "dt_dz_l3_rootw4 dt_dz_l3_rooth dt_dz_l3_rootanc dt_dz_l3_r32 dt_dz_l3_wwide dt_dz_l3_s12k dt_dz_l3_aug2 dt_dz_l3_roll "
         "dt_dz_l3_nzlo dt_dz_l3w128 dt_dz_l3w128_kd dt_dz_l3w128s48 dt_dz_l3w128b1").split()
MAIN = [("dt2_dz_l3_tf_pert", "dt_dz_l3（裸跟踪器）"), ("dt2_dz_l3_dt2filt", "dt_dz_l3 + 预注册常数滤波"),
        ("dt2_dz_l3_dt2afilt", "dt_dz_l3 + 事件量自适应滤波"), ("dt2_dz_l3_deploy", "dt_dz_l3 + 自适应滤波（部署形态）"),
        ("dt2_dz_l3w128_tf_pert", "dt_dz_l3w128（layer3 128 通道）"), ("dt2_dz_l3w128_kd_tf_pert", "dt_dz_l3w128_kd（+ 稠密蒸馏）"),
        ("dt2_dz_l3w128_kd_dt2afilt", "dt_dz_l3w128_kd + 自适应滤波"),
        ("dt2_dz_l3w128_kd_deploy", "dt_dz_l3w128_kd + 自适应滤波（部署形态）")]


def run(*args) -> str:
    return subprocess.run([PY, *args], cwd=REPO, check=True, capture_output=True, text=True).stdout


def table_only(md: str) -> str:
    return "\n".join(l for l in md.splitlines() if l.startswith("|"))


def bench_table(path: Path, title: str) -> str:
    d = json.loads(path.read_text())
    lines = [f"{title}（`{path.relative_to(REPO)}`，{d['run']}，{d['packets']} 包 × {d['rounds']} 轮，原始 ms，p50 / p99）：", "",
             "| 变体 | 20 Hz global | 20 Hz local | 背靠背 global | 背靠背 local |", "|---|---|---|---|---|"]
    for v, t in d["table"].items():
        cells = [f"{t[k][m]['p50']:.2f} / {t[k][m]['p99']:.2f}" for m in ("hz20", "b2b") for k in ("global", "local")]
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lat = d.get("latency_model") or {}
    for k, x in lat.items():
        lines.append("")
        lines.append(f"- 主表口径 `evalx.latency_model`（{k}）：raw {x['raw_ms']:.3f} ms，锚点 {x['anchor_ms']:.3f} ms，"
                     f"折算 {x['scaled_ms_full1p75']:.3f} ms")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(REP / "DT2_VERDICT.tmpl.md"))
    ap.add_argument("--out", default=str(REPO / "docs/DT2_VERDICT.md"))
    a = ap.parse_args()
    rows = [(r, l) for r, l in MAIN if (REPO / "outputs/semkine" / f"{r}_main_row.json").exists()]
    main_tab = run("tools/report_table.py", *[f"--label={r}={l}" for r, l in rows], *[r for r, _ in rows])
    cmp2 = run("tools/dt/dt2_compare.py", "--base", "dt_dz_l3", "--arms", *ARMS2, "--seeds", "3407", "3408",
               "--json", str(REP / "compare_2seed.json"))
    cmp3 = run("tools/dt/dt2_compare.py", "--base", "dt_dz_l3", "--arms", "dt_dz_l3w128", "dt_dz_l3w128_kd", "dt_dz_l3_r32",
               "--seeds", "3407", "3408", "3409", "--json", str(REP / "compare_3seed.json"))
    filt = run("tools/dt/dt2_filter_table.py", "--arms", "dt_dz_l3", "dt_dz_l3w128", "dt_dz_l3w128_kd", "dt_dz_l3_nzlo")
    bench = bench_table(REP / "D_bench_idle.json", "dt_dz_l3 + 自适应滤波，空闲 L20（GPU 4）")
    bw = REP / "D_bench_idle_w128.json"
    benchw = bench_table(bw, "w128 骨干的部署计时（dt_dz_l3w128_s3407，与 w128_kd 结构相同），空闲 L20（GPU 4）") if bw.exists() else ""
    s = Path(a.template).read_text()
    for k, v in {"<<DATE>>": time.strftime("%H:%M"), "<<MAIN>>": main_tab.strip(), "<<CMP2>>": table_only(cmp2),
                 "<<CMP3>>": table_only(cmp3), "<<FILT>>": filt.strip(), "<<BENCH>>": bench, "<<BENCHW>>": benchw}.items():
        s = s.replace(k, v)
    left = [k for k in ("<<MAIN>>", "<<CMP2>>", "<<CMP3>>", "<<FILT>>", "<<BENCH>>", "<<BENCHW>>", "<<DATE>>") if k in s]
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    Path(a.out).write_text(s)
    print(f"wrote {a.out} ({len(s.splitlines())} lines; main rows: {', '.join(r for r, _ in rows)})")


if __name__ == "__main__":
    main()
