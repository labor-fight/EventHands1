#!/usr/bin/env python3
"""T1A-UB round 5, the final round (CPU only; registers and writes driver jobs, never trains or reads a result).

The composition is the one the earlier pre-registrations wrote (docs/EXPERIMENT_RECORDS.md#t1aub-r4-prereg-20261102,
#t1aub-r4-amend1-20261102, #t1aub-r3-prereg-20261102):
  X  = candidate B's trunk (t0pnoeg20) + every training-side mechanism adopted in summary_r4_amend1.json
       (cos -> cosine schedule, dz -> depth augmentation, match -> trained on srdwkda's routes, n2 -> the N2 term through
       tools/t1aub/train2.py); with none adopted, X = the best in-budget network of rounds 1-2 (summary_r2.json
       `best_deployable`, else the lowest main-row RA in budget).
  The paired replicate is X's trunk on the dedicated sampling stream (`x5rep`). The legacy trunk is the round-4 arm itself
  when one mechanism is adopted, a new arm `x5` when two or more are, the round-1 trunk when none is.
  Deployed networks: cells `x5l` (legacy trunk) and `x5d` (dedicated trunk) with srdwkda, plus the round-2 arbitration when
  X is an arbitrated cell; their rows are measured with T1a in one exclusive block.
  Diagnostics (label-assisted or over budget; never the deployed result): `x5g` = X's legacy trunk with the label routes at
  test time (tools/t1aub/r5.py: the router's share of X's remaining gap with the trunk fixed), and X's legacy trunk with
  T0P's routes (the arm itself; read here unless round 1 already read it).
  Candidate-A transplant (descriptive; the round-4 pre-registration promised it when a mechanism is adopted): `a5` = t0ps16 +
  the adopted mechanisms except n2 (two forwards of the event graph do not fit), cell `a5_srdwa`; only when candidate A
  itself (t0ps16_srdwa) is within all three budgets in summary_r1.json (a cost criterion, not an accuracy reading).

    final.py plan                                       print the round-5 registries
    final.py check                                      generate the configs of every possible composition in memory
    final.py register --tag T1AUB-R5-PREREG-20261102    write the registries and freeze the round in one process
    final.py jobs --block B --out FILE                  the round's driver jobs (trainings, readings, block, decision)

Registration order (no job can see a registered round without a manifest, or a manifest naming a missing file): the
configs and side registries (registry2_r5 / registry5_r5 / train2_r5), then manifests/r5x.json (round 5; final.py,
decide_final.py, r5.py, evaluate7.py, those files), then
registry_r5.json, then manifests/r5.json (freeze.py's round manifest).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import r2  # noqa: E402
from tools.t1aub import r5  # noqa: E402

ROUND = 5
PREREG = "docs/EXPERIMENT_RECORDS.md#t1aub-r5-prereg-20261102"
B_TRUNK, A_TRUNK, ROUTER = "t0pnoeg20", "t0ps16", "srdwkda"
ORDER = ("cos", "n2", "dz", "match")
R4_ARMS = {"cos": "b4cos", "n2": "b4n2", "dz": "b4dz", "match": "b4match"}
N2_LAMBDA = 1.0
PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
NOTE = "T1A-UB r5 (docs/EXPERIMENT_RECORDS.md#t1aub-r5-prereg-20261102)"


def base_spec(arm):
    return {k: copy.deepcopy(v) for k, v in C.arm_spec(arm).items() if k not in ("round", "note", "phases")}


def with_mechanisms(spec, mechs):
    s = copy.deepcopy(spec)
    for m in mechs:
        if m == "cos":
            s["lr_schedule"] = "cosine"
        elif m == "dz":
            s["dz"] = True
        elif m == "match":
            s["route"] = {**s["route"], "router": ROUTER}
        else:
            C.require(m == "n2", "unknown mechanism " + repr(m))
    return s


def best_in_budget():
    s2 = C.read(C.OUT / "summary_r2.json")
    if s2["best_deployable"]:
        return s2["best_deployable"], "summary_r2 best_deployable (in budget, C <= 0.8, lowest RA)"
    pool = {n: v for n, v in s2["pool"].items() if v["budget_ok"]}
    C.require(pool, "no in-budget network in rounds 1-2")
    return min(pool, key=lambda n: (pool[n]["ra"], pool[n]["C"])), "lowest main-row RA in budget (none converges)"


def a_in_budget():
    n = C.read(C.OUT / "summary_r1.json")["networks"]["t0ps16_srdwa"]
    return bool(n["budget_ok"]), n["budget_parts"]


def plan(adopted=None, best=None, a_ok=None):
    """The round-5 registries: {"adopted", "x", "registry", "registry2", "train2", "a_transplant"}."""
    if adopted is None:
        adopted = C.read(C.OUT / "summary_r4_amend1.json")["adopted"]
    a_parts = None
    if a_ok is None and adopted:
        a_ok, a_parts = a_in_budget()
    adopted = [m for m in ORDER if m in adopted]
    reg = {"round": ROUND, "prereg": PREREG, "arms": {}, "cells": {}}
    reg2 = {"round": ROUND, "prereg": PREREG, "cells": {}}
    reg5 = {"round": ROUND, "prereg": PREREG, "cells": {}}
    t2 = {"round": ROUND, "arms": {}}
    if adopted:
        what = " + ".join(adopted)
        spec = with_mechanisms(base_spec(B_TRUNK), adopted)
        if len(adopted) == 1:
            legacy = R4_ARMS[adopted[0]]
        else:
            legacy = "x5"
            reg["arms"]["x5"] = {**spec, "note": f"candidate B's trunk (t0pnoeg20) + {what}; legacy_none stream"}
        reg["arms"]["x5rep"] = {**spec, "sampler": "dedicated",
                                "note": f"candidate B's trunk + {what}; dedicated stream (the paired replicate)"}
        if "n2" in adopted:
            for arm in reg["arms"]:
                t2["arms"][arm] = {"n2": N2_LAMBDA, "note": "S45-TR N2 fixed-point term, lambda 1 (as b4n2)"}
        reg["cells"]["x5l"] = {"trunk": legacy, "router": ROUTER, "note": f"X on the legacy stream: {legacy} + {ROUTER}"}
        reg["cells"]["x5d"] = {"trunk": "x5rep", "router": ROUTER, "note": f"X on the dedicated stream: x5rep + {ROUTER}"}
        a_mechs = [m for m in adopted if m != "n2"]
        if a_mechs and a_ok:
            reg["arms"]["a5"] = {**with_mechanisms(base_spec(A_TRUNK), a_mechs),
                                 "note": "candidate A's trunk (t0ps16) + " + " + ".join(a_mechs) + " (descriptive transplant)"}
            reg["cells"]["a5_srdwa"] = {"trunk": "a5", "router": ROUTER, "note": "a5 + srdwkda (as candidate A)"}
        x = {"source": "adopted", "legacy_trunk": legacy, "a_transplant": bool(a_mechs and a_ok),
             "a_in_budget": a_ok, "a_budget_parts": a_parts}
    else:
        if best is None:
            best, why = best_in_budget()
        else:
            why = "given"
        r1cells = C.registry()["cells"]
        cell = r1cells[best] if best in r1cells else r2.cell_spec(best)
        trunk = cell["trunk"]
        reg["arms"]["x5rep"] = {**base_spec(trunk), "sampler": "dedicated",
                                "note": f"{trunk} (the trunk of {best}) on the dedicated stream (the paired replicate)"}
        target = reg2 if "arb" in cell else reg
        extra = {"arb": copy.deepcopy(cell["arb"])} if "arb" in cell else {}
        target["cells"]["x5l"] = {"trunk": trunk, "router": cell["router"], **extra,
                                  "note": f"X = {best} ({why}), re-measured in the final block"}
        target["cells"]["x5d"] = {"trunk": "x5rep", "router": cell["router"], **extra,
                                  "note": f"X's trunk on the dedicated stream, deployed as {best}"}
        x = {"source": why, "network": best, "legacy_trunk": trunk}
    reg5["cells"]["x5g"] = {"trunk": x["legacy_trunk"], "router": "gt",
                            "note": "diagnostic: X's legacy trunk with the label routes at test time (not deployable)"}
    return {"adopted": adopted, "x": x, "registry": reg, "registry2": reg2, "registry5": reg5, "train2": t2}


# --------------------------------------------------------------------------- configs (in memory)
def _patched(p):
    """Context: C.registry / r2.registry also return the planned round-5 networks (nothing is written)."""
    class Patch:
        def __enter__(self):
            self.native, self.native2, self.native5 = C.registry, r2.registry, r5.registry

            def merged():
                out = self.native()
                out["rounds"][ROUND] = PREREG
                for what in ("arms", "cells"):
                    for n, s in p["registry"][what].items():
                        C.require(n not in out["arms"] and n not in out["cells"], "network registered twice: " + n)
                        out[what][n] = {**copy.deepcopy(s), "round": ROUND}
                return out

            def merged2():
                out = self.native2()
                for n, s in p["registry2"]["cells"].items():
                    C.require(n not in out["cells"], "network registered twice: " + n)
                    out["cells"][n] = {**copy.deepcopy(s), "round": ROUND}
                return out
            def merged5():
                out = self.native5()
                for n, s in p["registry5"]["cells"].items():
                    C.require(n not in out["cells"], "network registered twice: " + n)
                    out["cells"][n] = {**copy.deepcopy(s), "round": ROUND}
                return out
            C.registry, r2.registry, r5.registry = merged, merged2, merged5
            return self

        def __exit__(self, *exc):
            C.registry, r2.registry, r5.registry = self.native, self.native2, self.native5
    return Patch()


def config_texts(p):
    """[(path, text)] of every round-5 config: arms (both phases) and cells of both registries."""
    with _patched(p):
        files = C.round_files(ROUND) + (r2.round_files(ROUND) if p["registry2"]["cells"] else []) + r5.round_files(ROUND)
        for _, text in files:
            C.require("T1AUB:" in text, "config without its T1AUB block")
    return files


def check():
    """Every composition the rule can produce generates its configs (CPU, nothing written)."""
    out = {}
    for n in range(0, len(ORDER) + 1):
        for subset in itertools.combinations(ORDER, n):
            if not subset:
                continue
            p = plan(list(subset), a_ok=True)
            out["+".join(subset)] = len(config_texts(p))
    for best in ("t0pnoeg20_srdwa", "t0ps16_srdwa", "b_pt2", "a_ray"):
        p = plan([], best=best)
        out["none:" + best] = len(config_texts(p))
    return out


# --------------------------------------------------------------------------- registration
def _text(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def register(tag):
    from tools.t1aub import freeze as F
    C.verify_protocol()
    for name in ("registry_r5.json", "registry2_r5.json", "registry5_r5.json", "train2_r5.json"):
        C.require(not (C.PACKAGE / name).exists(), name + " exists")
    for name in ("r5.json", "r5x.json"):
        C.require(not (C.MANIFESTS / name).exists(), name + " exists")
    p = plan()
    digest, length = C.section(tag)
    configs = config_texts(p)
    side = []
    if p["registry2"]["cells"]:
        side.append((C.PACKAGE / "registry2_r5.json", _text(p["registry2"])))
    side.append((C.PACKAGE / "registry5_r5.json", _text(p["registry5"])))
    if p["train2"]["arms"]:
        side.append((C.PACKAGE / "train2_r5.json", _text(p["train2"])))
    for path, text in configs + side:
        C.require(not path.exists(), "exists: " + str(path))
    # 1. configs and side registries (no manifest names them yet; no registry reads round-5 configs yet)
    for path, text in configs + side:
        path.write_text(text)
    # 2. the files manifest of round 5 (makes round 5 count as frozen before its registry exists)
    files = [*(path for path, _ in configs + side), ROOT / "tools/t1aub/final.py", ROOT / "tools/t1aub/decide_final.py",
             ROOT / "tools/t1aub/train2.py", ROOT / "tools/t1aub/r2.py", ROOT / "tools/t1aub/evaluate2.py",
             ROOT / "tools/t1aub/r5.py", ROOT / "tools/t1aub/evaluate7.py", ROOT / "tests/test_t1aub_r5.py",
             ROOT / "semkine/t1aub_arb.py", ROOT / "semkine/s45tr_n2.py"]
    F.manifest("r5x", ROUND, tag, files, "FILES")
    # 3. the round-1-layer registry, 4. its round manifest (freeze.py's file list)
    reg_path = C.PACKAGE / "registry_r5.json"
    reg_path.write_text(_text(p["registry"]))
    C.require(C.sha(reg_path) == _sha_text(_text(p["registry"])), "registry write differs")
    r1_configs = [path for path, _ in configs if path.name.startswith("t1aub_") and path in
                  {q for q, _ in C.round_files(ROUND)}]
    F.manifest("r5", ROUND, tag, [reg_path, *r1_configs, *(ROOT / f for f in C.TRAIN_FILES), ROOT / "tests/test_t1aub.py",
                                  *(ROOT / f for f in F.DEPENDENCIES)], "ROUND")
    bad = [q for q, ok in {**C.check_configs(ROUND), **r5.check_configs(ROUND),
                           **(r2.check_configs(ROUND) if p["registry2"]["cells"] else {})}.items() if not ok]
    C.require(not bad, "configs differ from the generator: " + ", ".join(bad))
    C.verify_protocol()
    C.save(C.OUT / "final_plan.json", {**p, "prereg_tag": tag, "prereg_section_sha256": digest,
                                       "prereg_section_bytes": length, "registered_at_utc": C.now()}, exclusive=True)
    return p


# --------------------------------------------------------------------------- jobs
def _round_of(arm):
    return C.registry()["arms"][arm]["round"]


def _trunk(net):
    if net in C.registry()["arms"]:
        return net
    if r5.is_r5(net):
        return r5.cell_spec(net)["trunk"]
    return (r2.cell_spec(net) if r2.is_r2(net) else C.cell_spec(net))["trunk"]


def jobs(block):
    from tools.t1aub import jobs as JB
    p = C.read(C.OUT / "final_plan.json")
    reg = C.registry()
    arms = [a for a, s in reg["arms"].items() if s["round"] == ROUND]
    n2 = set(p["train2"]["arms"])
    out = []
    for j in JB.train_jobs(ROUND):
        arm = j["command"][j["command"].index("--arm") + 1] if "--arm" in j["command"] else None
        if arm in n2 and "tools/t1aub/train.py" in j["command"]:
            j["command"] = [("tools/t1aub/train2.py" if c == "tools/t1aub/train.py" else c) for c in j["command"]]
        j["note"] = NOTE
        out.append(j)
    cells = [c for c, s in reg["cells"].items() if s["round"] == ROUND] + list(p["registry2"]["cells"])
    # readings: every deployed cell, the oracle cell, and X's legacy trunk with T0P's routes unless round 1 read it
    leg = p["x"]["legacy_trunk"]
    t0p_net = [leg] if (_round_of(leg) > 1 and C.arm_spec(leg)["route"]["router"] == "t0p") else []
    nets = cells + t0p_net + list(p["registry5"]["cells"])
    row_nets = cells + t0p_net
    readings = []
    for net in nets:
        trunk = _trunk(net)
        dep = [f"ub:r{_round_of(trunk)}:acc:formal:{trunk}"]
        for mode, step in JB.TRACES:
            name = net if step == 6000 else f"{net}_{step}"
            key = f"ub:r5:ev:{net}:{mode}:{step}"
            out.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": NOTE, "deps": dep,
                        "command": [PY, "tools/t1aub/evaluate7.py", "trace", "--net", name, "--mode", mode]})
            readings.append(key)
        for task in JB.READOUTS:
            key = f"ub:r5:ro:{net}:{task}"
            deps = dep + ([f"ub:r5:ro:{net}:tl_e1"] if task == "tl_f1" else [])
            out.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": NOTE, "deps": deps,
                        "command": [PY, "tools/t1aub/evaluate7.py", "readout", "--net", net, "--task", task]})
            readings.append(key)
        for step in (5000, 5500):
            key = f"ub:r5:ro:{net}_{step}:tl_e1"
            out.append({"key": key, "kind": "gpu", "mem_gib": 15, "note": NOTE, "deps": dep,
                        "command": [PY, "tools/t1aub/evaluate7.py", "readout", "--net", f"{net}_{step}", "--task", "tl_e1"]})
            readings.append(key)
    rows = []
    for net in [f"t1aref_{block}", f"t1arp_{block}"] + row_nets:
        key = f"ub:r5:row:{block}:{net}"
        out.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": NOTE,
                    "deps": readings + (rows[-1:] or ["ub:r4:row2:b4:b4match", "ub:r4:decide:amend1:r2"]),
                    "command": [PY, "tools/t1aub/evaluate7.py", "row", "--net", net, "--block", block]})
        rows.append(key)
    # stage timings (tools/t1aub/breakdown7.py: the frozen breakdown.py after amendment 1) of T1a and X after the rows
    bds = [f"t1aref_{block}"] + [n for n in ("x5l", "x5d") if n in reg["cells"]]
    for net in bds:
        key = f"ub:r5:bd:{block}:{net}"
        out.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": NOTE, "deps": rows[-1:],
                    "command": [PY, "tools/t1aub/breakdown7.py", "--net", net, "--block", block]})
        rows.append(key)
    rep = [f"ub:r3:ev:t1arep:{mode}:{step}" for mode, step in JB.TRACES] + \
          [f"ub:r3:ro:t1arep:{task}" for task in JB.READOUTS] + [f"ub:r3:ro:t1arep_{s}:tl_e1" for s in (5000, 5500)]
    out.append({"key": "ub:r5:decide", "kind": "cpu", "note": NOTE, "deps": rows + rep + ["ub:r4:row2:b4:t1arep"],
                "command": [PY, "tools/t1aub/decide_final.py", "--x-legacy", "x5l", "--x-dedicated", "x5d", "--block", block]})
    if "rf6" in reg["arms"]:
        out += block6("b6")
    return out


def block6(block):
    """Round 6's main-row block (docs/EXPERIMENT_RECORDS.md#t1aub-r6-prereg-20261102, after the final round's decision):
    t1aref_<B>, rf6, rf6's stage timings, then tools/t1aub/decide6.py; rows through evaluate7.py (amendment 1)."""
    from tools.t1aub import jobs as JB
    note = "T1A-UB r6 (docs/EXPERIMENT_RECORDS.md#t1aub-r6-prereg-20261102; rows per amendment 1)"
    readings = [f"ub:r6:ev:rf6:{m}:{s}" for m, s in JB.TRACES] + [f"ub:r6:ro:rf6:{t}" for t in JB.READOUTS] + \
               [f"ub:r6:ro:rf6_{s}:tl_e1" for s in (5000, 5500)]
    jobs, prev = [], ["ub:r5:decide"]
    for net in (f"t1aref_{block}", "rf6"):
        key = f"ub:r6:row:{block}:{net}"
        jobs.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": note, "deps": readings + prev,
                     "command": [PY, "tools/t1aub/evaluate7.py", "row", "--net", net, "--block", block]})
        prev = [key]
    key = f"ub:r6:bd:{block}:rf6"
    jobs.append({"key": key, "kind": "exclusive", "mem_gib": 15, "note": note, "deps": prev,
                 "command": [PY, "tools/t1aub/breakdown7.py", "--net", "rf6", "--block", block]})
    jobs.append({"key": "ub:r6:decide", "kind": "cpu", "note": note,
                 "deps": [key, "ub:r4:row2:b4:b4match", "ub:r1:decide:r2"] + readings,
                 "command": [PY, "tools/t1aub/decide6.py", "--block", block, "--match-block", "b4"]})
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    sub.add_parser("plan")
    sub.add_parser("check")
    r = sub.add_parser("register")
    r.add_argument("--tag", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--block", required=True)
    j.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.what == "plan":
        print(_text(plan()), end="")
    elif a.what == "check":
        print(_text(check()), end="")
    elif a.what == "register":
        print(_text(register(a.tag)), end="")
    else:
        js = jobs(a.block)
        Path(a.out).write_text(json.dumps(js, indent=1) + "\n")
        print(len(js), "jobs ->", a.out)


if __name__ == "__main__":
    main()
