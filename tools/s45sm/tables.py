"""Descriptive tables over the summaries (CPU only, run once after the round's `decide`; single seed 3407).

  table 7    trunk x test-router table of round 7 (rows = the three trunks with the routes they were trained with, columns
             the test routes sr1 / srpkd / T0P / labels); reads summary_r7.json and summary_r5.json (the oracle pfagtdz);
             trunk and router effects flagged "beyond" (|difference| > the noise floor 3.5 / 2.75 / 3.1 mm) or "within";
             the written trunk choice T*: among in-budget networks with C <= 0.8 the lowest main-row RA, else the lowest in
             budget; T* is that network's trunk, "privileged" when its training routes are not an in-budget router
  table 11   rounds 9 + 11 (reads summary_r11.json): every network whose trunk is pfasrdz / pfat0pdz / pfagtdz placed by trunk
             and test router; families anneal, part_first, structure, trunk; the in-budget networks with the lowest RA
  table 12   round 12 (reads summary_r12.json): the same grid with the nine-subject routers and one more family, `nine`
  table 10   the D1 table of the annealed and part-first routers against their bases (reads readouts_d1/<router>.json)
Each writes <results>/swap_table_r<k>.json / .md (`d1_table_r10.*`) exclusively.
"""
from __future__ import annotations

from .paths import NOISE, now, read, require, results_dir, save
from .protocol import verify_protocol
from .registry import registry, trunk_of

KEYS = ("local", "global", "ra")
C_MAX = 0.8

# ---- round 7
TRUNKS7 = {"pfasrdz": "sr1", "pfat0pdz": "t0p", "pfagtdz": "label"}
ROUTES7 = ("sr1", "srpkd", "t0p", "label")
IN_BUDGET_ROUTERS = ("sr1", "srpkd")

# ---- rounds 9 / 11 / 12
TRUNKS = ("pfasrdz", "pfat0pdz", "pfagtdz")
ROUTERS11 = ("sr1", "srpkd", "srdwkd", "t0p", "sr1a", "srpkda", "srdwkda", "srpkdh", "srpkdah", "srdwkdh", "srdwkdah")
PAIRS11 = {"anneal": (("sr1a", "sr1"), ("srpkda", "srpkd"), ("srdwkda", "srdwkd")),
           "part_first": (("srpkdh", "srpkd"), ("srdwkdh", "srdwkd"), ("srpkdah", "srpkda"), ("srdwkdah", "srdwkda")),
           "structure": (("srdwkd", "srpkd"),)}
TRUNK_PAIR = ("pfat0pdz", "pfagtdz")
NINE = (("srpkd9", "srpkd"), ("srpkd9a", "srpkda"), ("srpkd9h", "srpkdh"), ("srpkd9ah", "srpkdah"))
ROUTERS12 = ROUTERS11 + tuple(new for new, _ in NINE)
PAIRS12 = {"anneal": PAIRS11["anneal"] + (("srpkd9a", "srpkd9"),),
           "part_first": PAIRS11["part_first"] + (("srpkd9h", "srpkd9"), ("srpkd9ah", "srpkd9a")),
           "structure": PAIRS11["structure"], "nine": NINE}
TITLES11 = (("anneal", "退火效应"), ("part_first", "部位优先解码效应"), ("structure", "主干结构效应"), ("trunk", "主干效应"))
TITLES12 = (("nine", "九人效应"),) + TITLES11
LAYOUTS = {
    11: {"routers": ROUTERS11, "pairs": PAIRS11, "titles": TITLES11,
         "head": "# S45-SM 第 9、11 轮：主干 × 测试路由器（主行 local / global / RA，C；单 seed 3407，描述性）"},
    12: {"routers": ROUTERS12, "pairs": PAIRS12, "titles": TITLES12,
         "head": "# S45-SM 第 12 轮：主干 × 测试路由器，含九人路由器（主行 local / global / RA，C；单 seed 3407，描述性）"}}

# ---- round 10
D1_ROUND = 10
D1_SEQS = ("ylf_local", "ylf_global", "zgz_local", "zgz_global", "lyq_local")
D1_KEYS = ("occupancy_overlap_patch", "part_cross_share", "face_geodesic_median_mm", "coverage_onhand_0p5")


def entry(v, budget_ok):
    return {**{k: float(v["main_row"][k]) for k in KEYS}, "C": float(v["convergence"]["C"]),
            "budget_ok": bool(budget_ok), "class": v["class"]}


def flag(d, k):
    return "beyond" if abs(d) > NOISE[k] else "within"


def difference(a, b):
    out = {}
    for k in KEYS:
        d = a[k] - b[k]
        out[k] = {"diff": d, "flag": "beyond" if abs(d) > NOISE[k] else "within"}
    out["C_diff"] = a["C"] - b["C"]
    return out


# --------------------------------------------------------------------------- round 7
def grid7(summary7, summary5):
    arms, reg = summary7["arms"], registry()["arms"]
    g = {}
    for trunk, routes in TRUNKS7.items():
        if trunk in arms:
            g[(trunk, routes)] = {"net": trunk, **entry(arms[trunk], arms[trunk]["budget_ok"])}
    o = summary5["oracles"]["pfagtdz"]
    g[("pfagtdz", "label")] = {"net": "pfagtdz", **entry(o, False)}
    for net, spec in reg.items():
        if "trunk" in spec:
            require(net in arms, "a swap cell was not summarised: " + net)
            g[(spec["trunk"], spec["router"])] = {"net": net, **entry(arms[net], arms[net]["budget_ok"])}
    return g


def effects7(g):
    out = {"trunk": {}, "router": {}}
    for r in ROUTES7:
        base = g.get(("pfasrdz", r))
        for t in ("pfat0pdz", "pfagtdz"):
            x = g.get((t, r))
            if base and x:
                out["trunk"][f"{t}-pfasrdz@{r}"] = {**{k: {"diff": x[k] - base[k], "flag": flag(x[k] - base[k], k)}
                                                        for k in KEYS}, "C_diff": x["C"] - base["C"]}
    for t in TRUNKS7:
        base = g.get((t, "sr1"))
        for r in ("srpkd", "t0p"):
            x = g.get((t, r))
            if base and x:
                out["router"][f"{r}-sr1@{t}"] = {**{k: {"diff": x[k] - base[k], "flag": flag(x[k] - base[k], k)}
                                                     for k in KEYS}, "C_diff": x["C"] - base["C"]}
    return out


def trunk_choice(summary7):
    reg = registry()["arms"]
    pool = {n: v for n, v in summary7["arms"].items() if v["budget_ok"]}
    require(pool, "no in-budget network")
    conv = {n: v for n, v in pool.items() if v["convergence"]["C"] <= C_MAX}
    use = conv or pool
    best = min(use, key=lambda n: use[n]["main_row"]["ra"])
    trunk = reg[best].get("trunk", best)
    routes = TRUNKS7.get(trunk) or reg[trunk]["router"]
    return {"network": best, "trunk": trunk, "trunk_routes": routes, "converging_pool": bool(conv),
            "privileged": routes not in IN_BUDGET_ROUTERS}


def markdown7(g, eff, choice):
    lines = ["# S45-SM 第 7 轮：主干 × 测试路由器（主行 local / global / RA，C；单 seed 3407，描述性）", "",
             "| 主干（训练时的路由） | " + " | ".join(ROUTES7) + " |", "|---|" + "---|" * len(ROUTES7)]
    for t, routes in TRUNKS7.items():
        cells = []
        for r in ROUTES7:
            x = g.get((t, r))
            cells.append("" if x is None else
                         f"{x['local']:.2f} / {x['global']:.2f} / {x['ra']:.2f}，C {x['C']:.2f}（{x['net']}，{x['class']}）")
        lines.append(f"| {t}（{routes}） | " + " | ".join(cells) + " |")
    for name in ("trunk", "router"):
        lines += ["", f"{'主干' if name == 'trunk' else '路由器'}效应（噪声底 3.5 / 2.75 / 3.1 mm）：", ""]
        for key, v in eff[name].items():
            lines.append(f"- {key}：" + "，".join(f"{k} {v[k]['diff']:+.2f}（{v[k]['flag']}）" for k in KEYS)
                         + f"，C {v['C_diff']:+.2f}")
    lines += ["", f"写定的主干选择 T*：{choice['trunk']}（训练路由 {choice['trunk_routes']}；来自 {choice['network']}；"
              f"{'C ≤ 0.8 中' if choice['converging_pool'] else '预算内无 C ≤ 0.8，取'} RA 最低；"
              f"{'特权路由主干' if choice['privileged'] else '预算内路由器训练的主干'}）"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- rounds 11 / 12
def grid(summary, round_):
    layout = LAYOUTS[round_]
    require(summary["round"] == round_, f"not the round-{round_} summary")
    reg = registry()["arms"]
    g = {}
    for net, v in summary["arms"].items():
        trunk = trunk_of(reg[net]) or net
        router = reg[net]["router"]
        if trunk not in TRUNKS or router not in layout["routers"]:
            continue
        require((trunk, router) not in g, f"two networks for trunk {trunk} and router {router}")
        g[(trunk, router)] = {"net": net, **entry(v, v["budget_ok"])}
    return g


def effects(g, round_):
    layout = LAYOUTS[round_]
    eff = {name: {f"{new}-{base}@{t}": difference(g[(t, new)], g[(t, base)])
                  for new, base in pairs for t in TRUNKS if (t, new) in g and (t, base) in g}
           for name, pairs in layout["pairs"].items()}
    a, b = TRUNK_PAIR
    eff["trunk"] = {f"{a}-{b}@{r}": difference(g[(a, r)], g[(b, r)]) for r in layout["routers"] if (a, r) in g and (b, r) in g}
    return eff


def lowest_ra(d):
    return d[min(d, key=lambda k: d[k]["ra"])]["net"] if d else None


def best(g):
    pool = {k: v for k, v in g.items() if v["budget_ok"]}
    conv = {k: v for k, v in pool.items() if v["C"] <= C_MAX}
    return {"in_budget": lowest_ra(pool), "in_budget_converging": lowest_ra(conv)}


def cell(v):
    if v is None:
        return ""
    return (f"{v['local']:.2f} / {v['global']:.2f} / {v['ra']:.2f}，C {v['C']:.2f}（{v['net']}，{v['class']}"
            f"{'' if v['budget_ok'] else '，超预算'}）")


def markdown(g, eff, top, round_):
    layout = LAYOUTS[round_]
    lines = [layout["head"], "", "| 测试路由器 | " + " | ".join(TRUNKS) + " |", "|---|" + "---|" * len(TRUNKS)]
    for r in layout["routers"]:
        cells = [cell(g.get((t, r))) for t in TRUNKS]
        if any(cells):
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
    for name, title in layout["titles"]:
        lines += ["", f"{title}（噪声底 3.5 / 2.75 / 3.1 mm）：", ""]
        for key, d in eff[name].items():
            lines.append(f"- {key}：" + "，".join(f"{k} {d[k]['diff']:+.2f}（{d[k]['flag']}）" for k in KEYS) +
                         f"，C {d['C_diff']:+.2f}")
    lines += ["", f"预算内主行 RA 最低：{top['in_budget']}；其中 C ≤ {C_MAX}：{top['in_budget_converging']}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- round 10 (D1)
def base_of(spec):
    return spec.get("anneal_of") or spec.get("decode_of")


def d1_rows():
    routers = registry()["routers"]
    names = [n for n, s in routers.items() if s["round"] == D1_ROUND]
    require(names, "no round-10 router")
    bases = sorted({base_of(routers[n]) for n in names} | {"sr1"})
    out = {}
    for n in bases + names:
        d = read(results_dir() / "readouts_d1" / f"{n}.json")
        out[n] = {s: {k: float(d["sequences"][s][k]) for k in D1_KEYS} for s in D1_SEQS}
    diffs = {}
    for n in names:
        b = base_of(routers[n])
        diffs[n] = {"base": b, **{s: {k: out[n][s][k] - out[b][s][k] for k in D1_KEYS} for s in D1_SEQS}}
    return out, diffs


def d1_markdown(out, diffs):
    lines = ["# S45-SM 第 10 轮路由器 D1（块级占据 / 跨部位 / 面测地中位 mm / 在手覆盖；描述性）", "",
             "| 路由器 | " + " | ".join(D1_SEQS) + " |", "|---|" + "---|" * len(D1_SEQS)]
    for n, v in out.items():
        lines.append(f"| {n} | " + " | ".join(f"{v[s]['occupancy_overlap_patch']:.3f} / {100 * v[s]['part_cross_share']:.1f} % / "
                                            f"{v[s]['face_geodesic_median_mm']:.1f} / {v[s]['coverage_onhand_0p5']:.3f}"
                                            for s in D1_SEQS) + " |")
    lines += ["", "对基础路由器的差（占据 / 跨部位百分点）：", ""]
    for n, d in diffs.items():
        lines.append(f"- {n} − {d['base']}：" + "，".join(
            f"{s} {d[s]['occupancy_overlap_patch']:+.3f} / {100 * d[s]['part_cross_share']:+.1f}" for s in D1_SEQS))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- entry
def table(round_):
    """Compute and write the table of round 7 / 10 / 11 / 12; returns the JSON path."""
    verify_protocol()
    root = results_dir()
    if round_ == 7:
        target = root / "swap_table_r7.json"
        require(not target.exists(), "the swap table is read once")
        s7, s5 = read(root / "summary_r7.json"), read(root / "summary_r5.json")
        require(s7["round"] == 7 and s5["round"] == 5, "summary rounds")
        g = grid7(s7, s5)
        eff = effects7(g)
        choice = trunk_choice(s7)
        save(target, {"computed_utc": now(), "noise": NOISE, "grid": {f"{t}|{r}": v for (t, r), v in g.items()},
                      "effects": eff, "trunk_choice": choice, "single_seed_descriptive": True}, exclusive=True)
        text = markdown7(g, eff, choice)
        (root / "swap_table_r7.md").write_text(text)
    elif round_ in LAYOUTS:
        target = root / f"swap_table_r{round_}.json"
        require(not target.exists(), f"the round-{round_} table is read once")
        g = grid(read(root / f"summary_r{round_}.json"), round_)
        eff, top = effects(g, round_), best(g)
        save(target, {"computed_utc": now(), "grid": {f"{t}|{r}": v for (t, r), v in g.items()}, "effects": eff,
                      "best": top, "descriptive": True}, exclusive=True)
        text = markdown(g, eff, top, round_)
        (root / f"swap_table_r{round_}.md").write_text(text)
    elif round_ == D1_ROUND:
        target = root / f"d1_table_r{D1_ROUND}.json"
        require(not target.exists(), "the round-10 D1 table is read once")
        out, diffs = d1_rows()
        save(target, {"computed_utc": now(), "routers": out, "differences_from_base": diffs, "descriptive": True},
             exclusive=True)
        text = d1_markdown(out, diffs)
        (root / f"d1_table_r{D1_ROUND}.md").write_text(text)
    else:
        raise RuntimeError(f"no table for round {round_}")
    print(text, flush=True)
    return target
