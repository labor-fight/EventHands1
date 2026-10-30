#!/usr/bin/env python3
"""R3 (v3) battle JSON schema check and aggregation.

    python tools/kbattle/aggregate_v3.py check <file.json>
    python tools/kbattle/aggregate_v3.py r1          # aggregate outputs/kbattle_20261025/v3/r1_P*_E*.json
    python tools/kbattle/aggregate_v3.py r2          # aggregate r2_P*_{defend,prosecute}.json
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "outputs" / "kbattle_20261025" / "v3"
SLOTS = [f"M{i}" for i in range(1, 11)]
AXES = ("packaging", "probe")
VERD = {"go", "conditional", "no-go"}
INFO = {"position", "increment", "bias_source", "system", "evidence"}
REPL_KEYS = {"slot", "code", "name_zh", "name_en", "claim", "new_object", "mechanism", "fk_role", "problem_ids",
             "evidence", "info_type", "not_rename", "probe", "gpu_arm", "nature_framing", "cannot_claim", "risks"}
PROBE_KEYS = {"what", "data", "gate", "threshold_source", "kill_reading", "cpu_zero_training"}


def check_r1(d: dict) -> list[str]:
    errs = []
    for k in ("panel", "expert", "domain", "verdicts", "replacements", "first_falsification", "rank", "data_read"):
        if k not in d:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    v = d["verdicts"]
    if set(v.keys()) != set(SLOTS):
        errs.append(f"verdicts keys {sorted(v.keys())} != {SLOTS}")
    need_repl = set()
    for s in SLOTS:
        e = v.get(s, {})
        for a in AXES:
            if e.get(a) not in VERD:
                errs.append(f"{s}.{a} invalid: {e.get(a)!r}")
            if e.get(a) == "no-go":
                need_repl.add(s)
        for k in ("packaging_reason", "probe_reason", "strongest_rebuttal", "conditions"):
            if not isinstance(e.get(k, None), str):
                errs.append(f"{s}.{k} missing or not str")
        if (e.get("packaging") == "conditional" or e.get("probe") == "conditional") and not e.get("conditions", "").strip():
            errs.append(f"{s} conditional without conditions")
    repl_slots = Counter()
    for r in d["replacements"]:
        missing = REPL_KEYS - set(r.keys())
        if missing:
            errs.append(f"replacement {r.get('code')} missing {sorted(missing)}")
            continue
        if r["slot"] not in SLOTS:
            errs.append(f"replacement slot {r['slot']} invalid")
        repl_slots[r["slot"]] += 1
        if r["info_type"] not in INFO:
            errs.append(f"replacement {r['code']} info_type invalid")
        pm = PROBE_KEYS - set(r["probe"].keys())
        if pm:
            errs.append(f"replacement {r['code']} probe missing {sorted(pm)}")
        if not isinstance(r["evidence"], list) or not r["evidence"]:
            errs.append(f"replacement {r['code']} needs evidence list")
        for ev in r.get("evidence", []):
            if not (isinstance(ev, dict) and "number" in ev and "anchor" in ev):
                errs.append(f"replacement {r['code']} evidence entries need number+anchor")
    for s in need_repl:
        if repl_slots[s] != 1:
            errs.append(f"slot {s} judged no-go needs exactly one replacement, got {repl_slots[s]}")
    ff = d["first_falsification"]
    for k in ("target", "experiment", "threshold", "threshold_source", "cpu_zero_training", "est_hours"):
        if k not in ff:
            errs.append(f"first_falsification missing {k}")
    if sorted(d["rank"]) != sorted(SLOTS):
        errs.append("rank must be a permutation of M1..M10")
    if not isinstance(d["data_read"], str) or not d["data_read"].strip():
        errs.append("data_read must be a non-empty string ('none' if nothing)")
    return errs


def check_r2(d: dict) -> list[str]:
    errs = []
    for k in ("panel", "role", "verdicts", "replacement_scores", "second_replacements", "data_read"):
        if k not in d:
            errs.append(f"missing key {k}")
    if errs:
        return errs
    if d["role"] not in ("defend", "prosecute"):
        errs.append("role must be defend|prosecute")
    v = d["verdicts"]
    if set(v.keys()) != set(SLOTS):
        errs.append("verdicts must cover M1..M10")
    for s in SLOTS:
        e = v.get(s, {})
        for a in AXES:
            if e.get(a) not in VERD:
                errs.append(f"{s}.{a} invalid")
        if not isinstance(e.get("reason", None), str):
            errs.append(f"{s}.reason missing")
    for r in d["replacement_scores"]:
        for k in ("code", "slot", "packaging", "probe", "reason"):
            if k not in r:
                errs.append(f"replacement_scores entry missing {k}")
        if r.get("packaging") not in VERD or r.get("probe") not in VERD:
            errs.append(f"replacement_scores {r.get('code')} invalid axis verdict")
    for r in d["second_replacements"]:
        missing = REPL_KEYS - set(r.keys())
        if missing:
            errs.append(f"second replacement {r.get('code')} missing {sorted(missing)}")
    return errs


def agg_r1():
    files = sorted(glob.glob(str(V3 / "r1_P*_E*.json")))
    per = {s: {a: Counter() for a in AXES} for s in SLOTS}
    pool = defaultdict(list)
    ranks = defaultdict(list)
    ff = Counter()
    bad = []
    for f in files:
        d = json.loads(Path(f).read_text())
        e = check_r1(d)
        if e:
            bad.append((f, e))
            continue
        for s in SLOTS:
            for a in AXES:
                per[s][a][d["verdicts"][s][a]] += 1
            ranks[s].append(d["rank"].index(s) + 1)
        for r in d["replacements"]:
            pool[r["slot"]].append({"code": r["code"], "name_zh": r["name_zh"], "claim": r["claim"][:200],
                                    "info_type": r["info_type"], "by": f"{d['panel']}-{d['expert']}"})
        ff[d["first_falsification"]["target"]] += 1
    out = {"n_files": len(files), "n_valid": len(files) - len(bad), "invalid": [b[0] for b in bad],
           "per_slot": {s: {a: dict(per[s][a]) for a in AXES} | {"mean_rank": sum(ranks[s]) / max(1, len(ranks[s])),
                            "n_replacements": len(pool[s])} for s in SLOTS},
           "first_falsification_targets": dict(ff), "replacement_pool": pool}
    (V3 / "r1_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    lines = ["| 槽位 | 包装 go/cond/no | 探针 go/cond/no | 平均排名 | 替代数 |", "|---|---|---|---|---|"]
    for s in SLOTS:
        p = per[s]["packaging"]; q = per[s]["probe"]
        lines.append(f"| {s} | {p['go']}/{p['conditional']}/{p['no-go']} | {q['go']}/{q['conditional']}/{q['no-go']} | "
                     f"{out['per_slot'][s]['mean_rank']:.2f} | {len(pool[s])} |")
    (V3 / "r1_scores.md").write_text("\n".join(lines) + f"\n\n有效 {out['n_valid']} / {len(files)}；无效：{out['invalid']}\n")
    print("\n".join(lines)); print("invalid:", bad)


def agg_r2():
    files = sorted(glob.glob(str(V3 / "r2_P*_*.json")))
    per = {s: {a: Counter() for a in AXES} for s in SLOTS}
    rs = defaultdict(lambda: {"packaging": Counter(), "probe": Counter(), "slot": None})
    bad = []
    for f in files:
        d = json.loads(Path(f).read_text())
        e = check_r2(d)
        if e:
            bad.append((f, e)); continue
        for s in SLOTS:
            for a in AXES:
                per[s][a][d["verdicts"][s][a]] += 1
        for r in d["replacement_scores"]:
            rs[r["code"]]["slot"] = r["slot"]
            for a in AXES:
                rs[r["code"]][a][r[a]] += 1
    out = {"n_files": len(files), "invalid": [b[0] for b in bad],
           "per_slot": {s: {a: dict(per[s][a]) for a in AXES} for s in SLOTS},
           "replacement_scores": {c: {"slot": v["slot"], "packaging": dict(v["packaging"]), "probe": dict(v["probe"])}
                                  for c, v in rs.items()}}
    (V3 / "r2_scores.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out["per_slot"], ensure_ascii=False, indent=1)); print("invalid:", bad)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        d = json.loads(Path(sys.argv[2]).read_text())
        errs = check_r2(d) if "role" in d else check_r1(d)
        print("OK" if not errs else "FAIL\n" + "\n".join(errs))
        sys.exit(0 if not errs else 1)
    {"r1": agg_r1, "r2": agg_r2}[cmd]()
