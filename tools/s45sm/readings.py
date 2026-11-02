"""Written one-off readings (CPU only, run once each after every reading of their round; single seed 3407).

  rowcheck_r3a  every round-1 arm's latency-path row carries exactly the accuracy of its round-1 row (per sequence and pooled)
  repeat_r13    the written repeat: pairs (repeat, original) -> REPRODUCED iff the repeat cell is in budget and AT_TARGET /
                NEAR_TARGET, else NOT_REPRODUCED; the originals must match summary_r11.json
  cosine_r14    the trunk learning-rate decay: pairs (cosine, constant) -> ADOPT / REJECT / MIXED on the two in-budget cell pairs
Each network is classified by summary.classify (latency-path main row), unchanged. Writes <results>/<id>.json / .md exclusively.
"""
from __future__ import annotations

from .paths import NOISE, TH, now, read, require, results_dir, save
from .protocol import verify_protocol
from . import summary as S

ROWCHECK_ARMS = ("t0pdz", "pfat0pdz", "srdz", "pfasrdz")
KEYS = ("local", "global", "ra")
PASSING = ("AT_TARGET", "NEAR_TARGET")

REPEAT_ROUND = 13
REPEAT_PAIRS = (("xt0rep_srdwa", "xt0_srdwa"), ("pfat0pdz_rep", "pfat0pdz"))
REPEAT_WRITTEN = REPEAT_PAIRS[0][0]

COSINE_ROUND = 14
COSINE_CELLS = (("xt0cos_srdwa", "xt0_srdwa"), ("xt0repcos_srdwa", "xt0rep_srdwa"))
COSINE_NATIVE = (("pfat0pdz_cos", "pfat0pdz"), ("pfat0pdz_rep_cos", "pfat0pdz_rep"))
COSINE_PAIRS = COSINE_CELLS + COSINE_NATIVE


# --------------------------------------------------------------------------- rowcheck (block r3a)
def accuracy(row):
    seed = row["per_seed"]["3407"]
    return {"per_seq": seed["per_seq"], "n_frames": seed["n_frames"],
            **{k: row["two_seed_mean"][k] for k in ("overall", "local", "global")}}


def rowcheck():
    verify_protocol()
    target = results_dir() / "rowcheck_r3a.json"
    require(not target.exists(), "the block r3a check runs once")
    arms = {}
    for arm in ROWCHECK_ARMS:
        slow, fast = S.row(arm)[0], S.fast_row(arm)
        require(fast[2] == "r3a", "row measured outside block r3a: " + arm)
        arms[arm] = {"equal": accuracy(slow) == accuracy(fast[0]), "slow": accuracy(slow), "fast": accuracy(fast[0])}
    result = {"status": "PASS" if all(v["equal"] for v in arms.values()) else "FAIL", "checked_at_utc": now(),
              "arms": arms}
    save(target, result, exclusive=True)
    print(result["status"], {a: v["equal"] for a, v in arms.items()}, flush=True)
    require(result["status"] == "PASS", "latency-path rows differ from the round-1 rows")
    return target


# --------------------------------------------------------------------------- pairs (shared by 13 and 14)
def pair(rep, orig):
    out = {"repeat": {"class": rep["class"], "budget_ok": rep["budget_ok"], "main_row": rep["main_row"],
                      "C": rep["convergence"]["C"]},
           "original": {"class": orig["class"], "budget_ok": orig["budget_ok"], "main_row": orig["main_row"],
                        "C": orig["convergence"]["C"]}}
    out["diff"] = {k: {"diff": rep["main_row"][k] - orig["main_row"][k],
                       "flag": "beyond" if abs(rep["main_row"][k] - orig["main_row"][k]) > NOISE[k] else "within"}
                   for k in KEYS}
    out["diff"]["C"] = rep["convergence"]["C"] - orig["convergence"]["C"]
    out["mean"] = {**{k: (rep["main_row"][k] + orig["main_row"][k]) / 2 for k in KEYS},
                   "C": (rep["convergence"]["C"] + orig["convergence"]["C"]) / 2}
    return out


def _net_lines(nets, pairs):
    """One row per network of the pairs (repeat / cosine, original / constant): the original first."""
    lines = ["| 网络 | 主行 local / global / RA | C | 参数（含路由器） | 时延增量（块） | 类别 |", "|---|---|---|---|---|---|"]
    for rep, orig in pairs:
        for n in (orig, rep):
            v = nets[n]
            m = v["main_row"]
            lines.append(f"| {n} | {m['local']:.2f} / {m['global']:.2f} / {m['ra']:.2f} | {v['convergence']['C']:.2f} | "
                         f"{v['params_total_with_router'] / 1e6:.3f} M | {v['latency_extra_vs_s45ref_ms']:+.2f} ms"
                         f"（{v['latency_block']}） | {v['class']}{'' if v['budget_ok'] else '，超预算'} |")
    return lines


def _pair_lines(res):
    lines = []
    for key, p in res["pairs"].items():
        d, mu = p["diff"], p["mean"]
        lines.append(f"- {key}：" + "，".join(f"{k} {d[k]['diff']:+.2f}（{d[k]['flag']}）" for k in KEYS) +
                     f"，C {d['C']:+.3f}；平均 {mu['local']:.2f} / {mu['global']:.2f} / {mu['ra']:.2f}，C {mu['C']:.3f}")
    return lines


# --------------------------------------------------------------------------- repeat (round 13)
def repeat_reading(nets):
    pairs = {f"{r}-{o}": pair(nets[r], nets[o]) for r, o in REPEAT_PAIRS}
    w = nets[REPEAT_WRITTEN]
    verdict = "REPRODUCED" if w["budget_ok"] and w["class"] in PASSING else "NOT_REPRODUCED"
    return {"pairs": pairs, "reading": verdict}


def repeat_markdown(nets, res):
    lines = ["# S45-SM 第 13 轮：写定的重复（只换采样流；单 seed 3407，描述性）", ""]
    lines += _net_lines(nets, REPEAT_PAIRS)
    lines += ["", "重复 − 原始（噪声底 3.5 / 2.75 / 3.1 mm）与两条采样流的平均（描述）：", ""] + _pair_lines(res)
    lines += ["", f"判读：{res['reading']}（{REPEAT_WRITTEN} 在预算内且为 AT_TARGET / NEAR_TARGET 记 REPRODUCED）"]
    return "\n".join(lines) + "\n"


def repeat_read():
    verify_protocol()
    root = results_dir()
    target = root / f"repeat_r{REPEAT_ROUND}.json"
    require(not target.exists(), "the repeat is read once")
    ref = S.references()
    nets = {n: S.classify(n, ref) for p in REPEAT_PAIRS for n in p}
    earlier = read(root / "summary_r11.json")["arms"]
    consistent = {n: nets[n]["main_row"] == earlier[n]["main_row"] and nets[n]["class"] == earlier[n]["class"]
                  for _, n in REPEAT_PAIRS}
    res = repeat_reading(nets)
    save(target, {"computed_utc": now(), "round": REPEAT_ROUND, "noise": NOISE, "thresholds": TH, "networks": nets,
                  "originals_match_summary_r11": consistent, **res, "single_seed_descriptive": True}, exclusive=True)
    text = repeat_markdown(nets, res)
    (root / f"repeat_r{REPEAT_ROUND}.md").write_text(text)
    print(text, flush=True)
    return target


# --------------------------------------------------------------------------- cosine (round 14)
def cosine_reading(nets):
    pairs = {f"{c}-{k}": pair(nets[c], nets[k]) for c, k in COSINE_PAIRS}
    lower = [nets[c]["main_row"]["ra"] < nets[k]["main_row"]["ra"] for c, k in COSINE_CELLS]
    passing = [nets[c]["budget_ok"] and nets[c]["class"] in PASSING for c, _ in COSINE_CELLS]
    if all(lower) and all(passing):
        verdict = "ADOPT"
    elif not any(lower):
        verdict = "REJECT"
    else:
        verdict = "MIXED"
    at = [c for c, _ in COSINE_CELLS if nets[c]["budget_ok"] and nets[c]["class"] == "AT_TARGET"]
    return {"pairs": pairs, "reading": verdict, "at_target": at}


def cosine_markdown(nets, res):
    lines = ["# S45-SM 第 14 轮：主干学习率余弦衰减（单 seed 3407，两条采样流；描述性）", ""]
    lines += _net_lines(nets, COSINE_PAIRS)
    lines += ["", "余弦 − 恒定（噪声底 3.5 / 2.75 / 3.1 mm）与两者平均（描述）：", ""] + _pair_lines(res)
    lines += ["", f"判读：{res['reading']}；预算内 AT_TARGET 的余弦单元：{', '.join(res['at_target']) or '无'}"]
    return "\n".join(lines) + "\n"


def cosine_read():
    verify_protocol()
    root = results_dir()
    target = root / f"cosine_r{COSINE_ROUND}.json"
    require(not target.exists(), "the round-14 reading is computed once")
    ref = S.references()
    nets = {n: S.classify(n, ref) for p in COSINE_PAIRS for n in p}
    res = cosine_reading(nets)
    save(target, {"computed_utc": now(), "round": COSINE_ROUND, "noise": NOISE, "thresholds": TH, "networks": nets, **res,
                  "single_seed_descriptive": True}, exclusive=True)
    text = cosine_markdown(nets, res)
    (root / f"cosine_r{COSINE_ROUND}.md").write_text(text)
    print(text, flush=True)
    return target


READINGS = {"rowcheck_r3a": rowcheck, "repeat_r13": repeat_read, "cosine_r14": cosine_read}


def read_one(id_):
    require(id_ in READINGS, "unregistered reading " + repr(id_))
    return READINGS[id_]()
