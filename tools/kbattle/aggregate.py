"""Aggregate the K-Battle expert score files (outputs/kbattle_20261025/r1_P*_E*.json).

    python tools/kbattle/aggregate.py r1        # validate + aggregate round 1
    python tools/kbattle/aggregate.py r2        # same for round 2 revised scores (r2_P*_*.json)

Writes <round>_scores.json and <round>_scores.md next to the inputs. Scores are expert opinion,
not evidence of validity (plan section 10).
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics

OUT = "outputs/kbattle_20261025"
ITEMS = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "Kstar"]
KEYS = ["necessity", "reachability", "novelty", "feasibility", "nature", "equiv_risk"]
GO = {"go", "no-go", "conditional"}


def total(s):
    return sum(s[k] for k in KEYS[:-1]) + (5 - s["equiv_risk"])


def validate(d, name):
    errs = []
    sc = d.get("scores", {})
    for it in ITEMS:
        s = sc.get(it)
        if not isinstance(s, dict):
            errs.append(f"{name}: scores.{it} missing")
            continue
        for k in KEYS:
            v = s.get(k)
            if not isinstance(v, (int, float)) or not 0 <= v <= 5:
                errs.append(f"{name}: scores.{it}.{k}={v!r}")
    rank = d.get("rank", [])
    if sorted(rank) != sorted(ITEMS):
        errs.append(f"{name}: rank is not a permutation of the 11 items: {rank}")
    for it in ITEMS:
        if d.get("go", {}).get(it) not in GO:
            errs.append(f"{name}: go.{it}={d.get('go', {}).get(it)!r}")
    return errs


def load(round_):
    pat = os.path.join(OUT, "r1_P*_E*.json" if round_ == "r1" else "r2_P*_*.json")
    files = sorted(glob.glob(pat))
    docs, errs = {}, []
    for f in files:
        name = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))
        except json.JSONDecodeError as e:
            errs.append(f"{name}: invalid JSON ({e})")
            continue
        e = validate(d, name)
        errs += e
        if not e:
            docs[name] = d
    return docs, errs


def aggregate(docs):
    per_item = {it: collections.defaultdict(list) for it in ITEMS}
    per_panel = collections.defaultdict(lambda: collections.defaultdict(list))
    borda = collections.Counter()
    first = collections.Counter()
    go = {it: collections.Counter() for it in ITEMS}
    verdict = collections.Counter()
    kdel, kadd = collections.Counter(), []
    for name, d in docs.items():
        panel = re.match(r"r\d_(P\d)", name).group(1)
        for it in ITEMS:
            s = d["scores"][it]
            for k in KEYS:
                per_item[it][k].append(s[k])
            per_item[it]["total"].append(total(s))
            per_panel[panel][it].append(total(s))
            go[it][d["go"][it]] += 1
        for pos, it in enumerate(d["rank"]):
            borda[it] += len(ITEMS) - 1 - pos
        first[d["rank"][0]] += 1
        verdict[d.get("user_direction_verdict", {}).get("verdict", "?")] += 1
        ke = d.get("kstar_edit", {})
        kdel[ke.get("delete", "?")] += 1
        if ke.get("add") and ke.get("add") != "none":
            kadd.append((name, ke["add"]))
    n = len(docs)
    table = []
    for it in ITEMS:
        row = {"item": it, "n": n}
        for k in KEYS + ["total"]:
            v = per_item[it][k]
            row[k] = round(statistics.mean(v), 2) if v else None
        row["total_sd"] = round(statistics.pstdev(per_item[it]["total"]), 2) if n else None
        row["borda"] = borda[it]
        row["first_place"] = first[it]
        row["go"] = dict(go[it])
        row["panel_total"] = {p: round(statistics.mean(v[it]), 2) for p, v in sorted(per_panel.items())}
        table.append(row)
    table.sort(key=lambda r: (-r["total"], -r["borda"]))
    return {
        "n_experts": n,
        "table": table,
        "user_direction_verdict": dict(verdict),
        "kstar_delete": dict(kdel),
        "kstar_add": kadd,
    }


def markdown(agg):
    panels = sorted({p for r in agg["table"] for p in r["panel_total"]})
    head = ["项", "合计/30", "SD", "Borda", "第1名", "必要", "可达", "新颖", "可行", "Nature", "改写风险",
            "go/cond/no"] + panels
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in agg["table"]:
        g = r["go"]
        cells = [r["item"], r["total"], r["total_sd"], r["borda"], r["first_place"]] + [r[k] for k in KEYS] + [
            f"{g.get('go', 0)}/{g.get('conditional', 0)}/{g.get('no-go', 0)}"] + [r["panel_total"].get(p) for p in panels]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    lines.append("")
    lines.append(f"专家数 {agg['n_experts']}；用户方向裁决 {agg['user_direction_verdict']}；K★ 删除建议 {agg['kstar_delete']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("round", choices=["r1", "r2", "check"])
    ap.add_argument("file", nargs="?", help="check: one expert JSON file")
    a = ap.parse_args()
    if a.round == "check":
        errs = validate(json.load(open(a.file)), os.path.basename(a.file))
        print("OK" if not errs else "\n".join(errs))
        raise SystemExit(1 if errs else 0)
    docs, errs = load(a.round)
    agg = aggregate(docs)
    agg["errors"] = errs
    json.dump(agg, open(os.path.join(OUT, f"{a.round}_scores.json"), "w"), ensure_ascii=False, indent=1)
    md = markdown(agg)
    open(os.path.join(OUT, f"{a.round}_scores.md"), "w").write(md + "\n")
    print(md)
    if errs:
        print("\nERRORS:\n" + "\n".join(errs))


if __name__ == "__main__":
    main()
