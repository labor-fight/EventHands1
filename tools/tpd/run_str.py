#!/usr/bin/env python3
"""TPD stage STR readings and rules (docs/EXPERIMENT_RECORDS.md#tpd-str-prereg-20261106), inside tools/tpd/ns_str.py:

    python -m tools.tpd.run_str accept --seed S --phase P    receipts/tpd_str_s<S>[_preflight].json: run_final's checks
                                                             (ROL's grid 40 | 500 / 1000 / 1500) after the start checks
                                                             of both ranks say "stream"
    python -m tools.tpd.run_str read --seed S --task T       PFO's engine in the streaming form, step 1500; tag tpd_str_s<S>
    python -m tools.tpd.run_str ref --task T                 rol5 3407 (deployed weights, zero training) read in the
                                                             streaming form; tag tpd_rol5_s3407__str (description)
    python -m tools.tpd.run_str screen                       decisions/str_screen.json (once): seed 3407 against rolk 3407
                                                             (same initial weights and packets, causal form); on STR_BETTER
                                                             or STR_EQUIV it appends seeds 3408 / 3409 to driver 5
    python -m tools.tpd.run_str table                        analysis/str.md, decisions/str_table.json (once): three seeds,
                                                             the final gate's rule applied as a description
Rules (written before any STR reading):
  STR_BETTER  amendment 3's screen against rolk 3407 passes (RA better by >= 0.3, or zgz_local R15 and F15 both lower by
              >= 0.05; E3 not worse by > 0.3)
  STR_EQUIV   otherwise, RA(str) - RA(rolk) <= +0.15 and E3 not worse by > 0.3 (the streaming form costs nothing)
  STR_WORSE   otherwise
The screen also requires the streaming form's asynchronous receipt (receipts/async_str_s3407.json) to be PASS.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_str  # noqa: E402

ns_str.enter()
from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd import run_final as RF  # noqa: E402
from tools.tpd.common import OUT, PY, S43_ENGINE, now, read, require, save, sha  # noqa: E402

ARM = ns_str.ARM
STEP = 1500
TASKS = ("e1", "e3", "pert", "f1")
REFERENCE = "tpd_rolk_s3407"
REF_SPEC = OUT / "specs" / "tpd_rol5_s3407.json"
EQUIV_RA, E3_TOL = 0.15, 0.3
RF.GRID = {"preflight": (40,), "formal": (500, 1000, STEP)}          # ROL's checkpoint grid (run_final reads it)


def spec(seed):
    receipt = read(OUT / "receipts" / f"{A.run_name(ARM, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = RF.checkpoint(ARM, seed, STEP)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(STEP)], "checkpoint changed since acceptance")
    tag = A.run_name(ARM, seed)
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(STEP)],
          "config": str(RF.eval_config(ARM, seed)), "trunk": "gshare", "module": "trunk4", "arm": ARM, "seed": seed,
          "causal": True, "stream": True}
    return tag, RF.write_once(OUT / "specs" / f"{tag}.json", sp)


def engine(tag, task, sp):
    from semkine import tpd_str as TS
    from tools.pfo import engine as E
    from tools.tpd import run_rol as RR
    require(task in TASKS, "task")
    RR.prepare(sp)
    TS.install()
    TS.engine_hooks()
    E.OUT = OUT
    out = E.run(tag, task, None, None, (), str(sp))
    save(Path(out) / "final_form.json", {"stream": True, "written_utc": now()}, exclusive=True)


def reading(seed, task):
    tag, sp = spec(seed)
    engine(tag, task, sp)


def ref(task):
    """rol5 3407 as its zgz readings used it (the spec file, checkpoint checked against its acceptance), streaming form."""
    sp = read(REF_SPEC)
    receipt = read(OUT / "receipts" / "tpd_rol5_s3407.json")
    require(sha(sp["checkpoint"]) == receipt["checkpoint_sha256_by_step"]["1500"], "rol5 3407 checkpoint changed")
    engine("tpd_rol5_s3407__str", task, REF_SPEC)


def accept(seed, phase):
    d = A.run_dir(ARM, seed, phase)
    starts = [read(d / f"final_start_check_rank{r}.json") for r in (0, 1)]
    require(all(x.get("stream") is True and x.get("init_equal") is True for x in starts), "start checks: " + str(starts))
    return RF.accept(ARM, seed, phase)


def mech():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT)
    return M


def rule(row, base):
    from tools.tpd.analyze_rol import screen as amend3
    v = amend3(row, base)
    if v["verdict"] == "SCREEN_PASS":
        verdict = "STR_BETTER"
    elif v["d_ra"] <= EQUIV_RA and v["d_e3"] <= E3_TOL:
        verdict = "STR_EQUIV"
    else:
        verdict = "STR_WORSE"
    return {**v, "amend3": v["verdict"], "verdict": verdict}


def follow_up_jobs():
    """Seeds 3408 / 3409 (formal fine-tunes from B2 of the seed), acceptance, readings, the three-seed table."""
    jobs, reads = [], []
    for seed in (3408, 3409):
        cfg = A.write_config(ARM, seed, "formal")
        jobs.append({"key": f"str:train:s{seed}", "kind": "pair", "mem_gib": 130, "deps": ["str:screen"],
                     "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY,
                                 str(ROOT / "tools/tpd/train_str.py"), "--config", str(cfg)],
                     "note": "TPD STR: seed %d (screen STR_BETTER / STR_EQUIV)" % seed})
        jobs.append({"key": f"str:accept:s{seed}", "kind": "cpu", "deps": [f"str:train:s{seed}"],
                     "command": [PY, "-m", "tools.tpd.run_str", "accept", "--seed", str(seed), "--phase", "formal"]})
        for task in TASKS:
            key = f"str:read:s{seed}:{task}"
            jobs.append({"key": key, "kind": "gpu", "deps": [f"str:accept:s{seed}"] + (
                [f"str:read:s{seed}:e1"] if task == "f1" else []),
                "command": [PY, "-m", "tools.tpd.run_str", "read", "--seed", str(seed), "--task", task]})
            reads.append(key)
    jobs.append({"key": "str:table", "kind": "cpu", "deps": reads, "command": [PY, "-m", "tools.tpd.run_str", "table"]})
    return jobs


def screen():
    out = OUT / "decisions" / "str_screen.json"
    require(not out.exists(), "written once")
    asy = read(OUT / "receipts" / "async_str_s3407.json")
    require(asy["status"] == "PASS", "the streaming form failed its asynchronous check")
    from tools.tpd.analyze_rol import compact
    M = mech()
    rx, r0 = M.row("tpd", A.run_name(ARM, 3407)), M.row("tpd", REFERENCE)
    v = rule(compact(rx), compact(r0))
    added = None
    if v["verdict"] in ("STR_BETTER", "STR_EQUIV"):
        jobs = follow_up_jobs()
        jf = OUT / "jobs" / "str_s23.json"
        require(not jf.exists(), "follow-up jobs written once")
        jf.write_text(json.dumps(jobs, indent=1) + "\n")
        env = dict(os.environ, TPD_DRIVER_DIR="driver5")
        subprocess.run([PY, "tools/tpd/chain4.py", "add", str(jf)], cwd=ROOT, env=env, check=True)
        added = {"file": str(jf), "jobs": [j["key"] for j in jobs], "driver": "driver5"}
    save(out, {"arm": ARM, "reference": REFERENCE, "screen": v, "rows": {ARM: rx, "reference": r0},
               "async_receipt": asy["status"], "follow_up": added, "written_at_utc": now()}, exclusive=True)
    print(v["verdict"], {k: round(v[k], 3) for k in ("d_ra", "d_e3", "d_R15", "d_F15")}, "->", added)


def table():
    out = OUT / "decisions" / "str_table.json"
    require(not out.exists(), "written once")
    from tools.tpd.analyze_syn import final_rule
    M = mech()
    seeds = (3407, 3408, 3409)
    rows = {s: M.row("tpd", A.run_name(ARM, s)) for s in seeds}
    causal = {s: M.row("tpd", f"tpd_rol5_s{s}__causal") for s in seeds}
    v = final_rule(rows)
    names = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
    L = ["## TPD STR：流式输入形式（rol5 配方在流式形式下微调；三种子，描述，不是目标写定的最终配置）", "",
         "| 行 | " + " | ".join(names) + " | E3 | 放大 | R15 / F15 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for label, r in ((f"rol5 {s}（零训练换成因果形式）", causal[s]), (f"str {s}", rows[s])):
            ret = r.get("retention")
            L.append(f"| {label} | " + " | ".join(f"{x:.2f}" for x in r["five"]) + f" | {r['e3']:.2f} | {r['amp'][0]:.2f} | "
                     + (f"{ret['R15']:.2f} / {ret['F15']:.2f}" if ret else "–") + " |")
    mean_c = [sum(causal[s]["five"][i] for s in seeds) / 3 for i in range(5)]
    L.append("| **str 三种子均值** | " + " | ".join(f"{x:.2f}" for x in v["mean_five"]) + " | | | |")
    L.append("| rol5 因果形式三种子均值 | " + " | ".join(f"{x:.2f}" for x in mean_c) + " | | | |")
    L.append("| S43（引擎三种子） | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | | | |")
    L += ["", f"最终门规则（作描述）：{v['verdict']}；对 S43 差 " + " / ".join(f"{g:+.2f}" for g in v["gap_to_s43"]) +
          f"；每种子 RA < 11.0：{v['seed_ra_below_11']}。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "str.md").write_text(text)
    save(out, {"rows": rows, "causal_reference": causal, **v, "description_only": True, "written_at_utc": now()},
         exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("accept")
    s.add_argument("--seed", type=int, required=True, choices=A.SEEDS)
    s.add_argument("--phase", required=True, choices=A.PHASES)
    s = sub.add_parser("read")
    s.add_argument("--seed", type=int, required=True, choices=A.SEEDS)
    s.add_argument("--task", required=True, choices=TASKS)
    s = sub.add_parser("ref")
    s.add_argument("--task", required=True, choices=("e1", "e3"))
    sub.add_parser("screen")
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "accept":
        return accept(a.seed, a.phase)
    if a.verb == "read":
        return reading(a.seed, a.task)
    if a.verb == "ref":
        return ref(a.task)
    return {"screen": screen, "table": table}[a.verb]()


if __name__ == "__main__":
    main()
