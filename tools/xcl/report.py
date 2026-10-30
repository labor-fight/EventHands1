#!/usr/bin/env python3
"""Generate the original nine-column main table and XCL three-readout artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xcl.evaluate import path, save_new, sha


def direction_support(deltas, late, threshold):
    """Descriptive registration: E1/E2/E3 same sign; >=2 late steps same sign."""
    if len(deltas) != 3 or len(late) != 3:
        return "缺读数"
    if any(value is None for value in (*deltas, *late)):
        return "缺读数"
    sign = 1 if deltas[0] > 0 else -1 if deltas[0] < 0 else 0
    same = sign != 0 and all(value * sign > 0 for value in deltas)
    late_same = sum(value * sign > 0 for value in late) >= 2
    return "方向支持" if abs(deltas[0]) >= threshold and same and late_same else "噪声带内"


def collect(root, runs):
    wanted = {r["name"] for r in runs}
    records = {}
    for filename in sorted(Path(root).rglob("report.json")):
        row = json.loads(filename.read_text())
        if row.get("run") not in wanted or row.get("status") != "COMPLETE" or row.get("intervention", "none") != "none":
            continue
        if row.get("n_endpoints") != 2590 or row.get("n_failed") != 0 or row.get("seed") != 3407:
            raise ValueError(f"Incomplete registered evaluation: {filename}")
        state = filename.parent / "states.npz"
        if not state.is_file() or sha(state) != row["states_sha256"]:
            raise ValueError(f"Changed evaluation states: {filename}")
        key = (row["run"], row["mode"], row["step"])
        # The second full E1 repeat is required to match; retaining both is valid.
        if key in records:
            prior = records[key][1]
            if row["metrics"] != prior["metrics"] or row["checkpoint_sha256"] != prior["checkpoint_sha256"]:
                raise ValueError(f"Conflicting fresh evaluations for {key}")
            from tools.xcl.evaluate import compare_traces
            compare_traces(records[key][0].parent / "states.npz", state)
            prior["_equivalent_repeats"] = prior.get("_equivalent_repeats", 1) + 1
        else:
            row["_equivalent_repeats"] = 1
            records[key] = (filename, row)
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--protocol", required=True); p.add_argument("--evaluations-root", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    protocol_file = path(args.protocol)
    protocol = json.loads(protocol_file.read_text())
    runs = protocol["runs"]
    records = collect(path(args.evaluations_root), runs)
    output = path(args.output_dir)
    if output.exists():
        raise FileExistsError("Preserve earlier reports")
    missing = [(r["name"], m, step) for r in runs
               for m, step in (("row", 6000), ("e1", 6000), ("e2", 6000), ("e3", 6000),
                               ("gt_pose", 6000), ("gt_transl", 6000), ("e1", 5000), ("e1", 5500))
               if (r["name"], m, step) not in records]
    missing += [(r["name"], "second_fresh_e1", 6000) for r in runs
                if records.get((r["name"], "e1", 6000), (None, {}))[1].get("_equivalent_repeats", 0) < 2]
    if missing and not args.allow_partial:
        raise ValueError(f"Missing registered readouts: {missing}")
    row_paths = {}
    labels = {}
    for run in runs:
        key = (run["name"], "row", 6000)
        if key in records:
            item = records[key][1]["main_row"]
            filename = path(item["path"])
            if sha(filename) != item["sha256"]:
                raise ValueError("Main-row SHA256 changed")
            row_paths[run["name"]] = filename
            cell = run.get("cell", run["name"])
            oracle = cell.startswith(("a1", "a2", "a3")) or cell in ("a0fod", "a0foo")
            labels[run["name"]] = f"XCL {cell}（seed3407）" + (" [oracle]" if oracle else "")
    # Reuse the required original formatter without installing aliases/pointers.
    from tools import report_table as original
    class Rows:
        def __truediv__(self, filename):
            prefix = str(filename).removesuffix("_main_row.json")
            return row_paths[prefix]
    old = original.OUT
    original.OUT = Rows()
    try:
        main_table = [original.HEADER, original.RULE, original.BASELINE]
        s39 = "dt3_dz_l3_w200cfg"
        row_paths[s39] = ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"
        main_table.append(original.row(s39, "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"))
        main_table.extend(original.row(r["name"], labels[r["name"]]) for r in runs if r["name"] in labels)
    finally:
        original.OUT = old
    table_lines = ["| 模型 | 域 | E1 自身递推 RA | E2 共同前态 RA | E3 GT前态 RA |", "|---|---|---|---|---|"]
    for run in runs:
        for domain in ("local", "global", "overall"):
            cells = []
            for mode in ("e1", "e2", "e3"):
                record = records.get((run["name"], mode, 6000))
                cells.append(f"{record[1]['metrics'][domain]['mpjpe_ra_mm']:.4f}" if record else "待完成")
            table_lines.append(f"| {run.get('cell', run['name'])} | {domain} | " + " | ".join(cells) + " |")
    baseline = next((r for r in runs if r.get("cell") in ("a0f0", "c1")), None)
    decisions = {}
    if baseline:
        for run in runs:
            if run == baseline:
                continue
            decisions[run["name"]] = {}
            for domain, metric, threshold in (("local", "mpjpe_ra_mm", 5.), ("global", "mpjpe_ra_mm", 8.), ("local", "wrist_mm", 10.)):
                def delta(mode, step):
                    a = records.get((run["name"], mode, step)); b = records.get((baseline["name"], mode, step))
                    return a[1]["metrics"][domain][metric] - b[1]["metrics"][domain][metric] if a and b else None
                d = [delta(m, 6000) for m in ("e1", "e2", "e3")]
                late = [delta("e1", step) for step in (5000, 5500, 6000)]
                decisions[run["name"]][domain + "/" + metric] = {"deltas_E1_E2_E3_mm": d, "late_deltas_mm": late,
                    "threshold_mm": threshold, "interpretation": direction_support(d, late, threshold)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "main_table.md").write_text("\n".join(main_table) + "\n")
    (output / "three_readouts.md").write_text("\n".join(table_lines) + "\n\n单位 mm；仅 seed3407 描述性比较；oracle 不可部署。完整部位/绝对指标见 report.json。\n")
    save_new(output / "report.json", {"status": "PARTIAL" if missing else "COMPLETE", "missing": missing,
        "protocol_sha256": sha(protocol_file), "main_table_sha256": sha(output / "main_table.md"),
        "readouts": {"|".join(map(str, k)): {"report": str(v[0]), "report_sha256": sha(v[0]), "metrics": v[1]["metrics"]} for k, v in records.items()},
        "descriptive_decisions": decisions, "single_seed_no_reliability_claim": True})
    print("\n".join(main_table))
    print("Report artifacts:", output)


if __name__ == "__main__":
    main()
