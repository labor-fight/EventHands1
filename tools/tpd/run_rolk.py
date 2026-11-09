#!/usr/bin/env python3
"""TPD stage ROL-k readings and rules (docs/EXPERIMENT_RECORDS.md#tpd-rolk-prereg-20261106), inside tools/tpd/ns_rolk.py:

    python -m tools.tpd.run_rolk accept --seed S --phase P     receipts/tpd_rolk_s<S>[_preflight].json (run_final's checks;
                                                               the checkpoints of ROL's grid: 40 | 500 / 1000 / 1500)
    python -m tools.tpd.run_rolk read --seed S --task T        PFO's engine in the causal form, step 1500; tag tpd_rolk_s<S>
    python -m tools.tpd.run_rolk screen                        decisions/rolk_screen.json (once): seed 3407 against rol5
                                                               3407 read in the causal form (zero training); on SCREEN_PASS
                                                               it appends seeds 3408 / 3409 to the driver's queue
    python -m tools.tpd.run_rolk table                         analysis/rolk.md, decisions/rolk_table.json (once): three
                                                               seeds, the final gate's rule applied as a description
Rules: screen = amendment 3's (tools/tpd/analyze_rol.py `screen`: RA better by >= 0.3, or zgz_local R15 and F15 both lower
by >= 0.05, and E3 not worse by > 0.3); table = tools/tpd/analyze_syn.py `final_rule` (five three-seed means below S43's
engine values and every seed's RA < 11.0), reported as a description: the goal's final configuration trains from scratch.
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

from tools.tpd import ns_rolk  # noqa: E402

ns_rolk.enter()
from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd import run_final as RF  # noqa: E402
from tools.tpd.common import OUT, PY, S43_ENGINE, now, read, require, save, sha  # noqa: E402

ARM = ns_rolk.ARM
STEP = 1500
TASKS = ("e1", "e3", "pert", "f1")
REFERENCE = "tpd_rol5_s3407__causal"
RF.GRID = {"preflight": (40,), "formal": (500, 1000, STEP)}          # ROL's checkpoint grid (run_final reads it)


def spec(seed):
    receipt = read(OUT / "receipts" / f"{A.run_name(ARM, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = RF.checkpoint(ARM, seed, STEP)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(STEP)], "checkpoint changed since acceptance")
    tag = A.run_name(ARM, seed)
    sp = {"kind": "routed", "checkpoint": str(ck), "checkpoint_sha256": receipt["checkpoint_sha256_by_step"][str(STEP)],
          "config": str(RF.eval_config(ARM, seed)), "trunk": "gshare", "module": "trunk4", "arm": ARM, "seed": seed,
          "causal": True}
    return tag, RF.write_once(OUT / "specs" / f"{tag}.json", sp)


def reading(seed, task):
    from semkine import tpd_causal as TC
    from tools.pfo import engine as E
    from tools.tpd import run_rol as RR
    require(task in TASKS, "task")
    tag, sp = spec(seed)
    RR.prepare(sp)
    TC.install()
    TC.engine_hooks()
    E.OUT = OUT
    out = E.run(tag, task, None, None, (), str(sp))
    save(Path(out) / "final_form.json", {"causal": True, "written_utc": now()}, exclusive=True)


def mech():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT)
    return M


def follow_up_jobs():
    """Seeds 3408 / 3409 (formal fine-tunes from B2 of the seed), their acceptance and readings, the three-seed table."""
    jobs, reads = [], []
    for seed in (3408, 3409):
        cfg = A.write_config(ARM, seed, "formal")
        jobs.append({"key": f"rolk:train:s{seed}", "kind": "pair", "mem_gib": 130, "deps": ["rolk:screen"],
                     "command": ["env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", PY,
                                 str(ROOT / "tools/tpd/train_rolk.py"), "--config", str(cfg)],
                     "note": "TPD ROL-k: rolk seed %d (screen passed)" % seed})
        jobs.append({"key": f"rolk:accept:s{seed}", "kind": "cpu", "deps": [f"rolk:train:s{seed}"],
                     "command": [PY, "-m", "tools.tpd.run_rolk", "accept", "--seed", str(seed), "--phase", "formal"]})
        for task in TASKS:
            key = f"rolk:read:s{seed}:{task}"
            jobs.append({"key": key, "kind": "gpu", "deps": [f"rolk:accept:s{seed}"] + (
                [f"rolk:read:s{seed}:e1"] if task == "f1" else []),
                "command": [PY, "-m", "tools.tpd.run_rolk", "read", "--seed", str(seed), "--task", task]})
            reads.append(key)
    jobs.append({"key": "rolk:table", "kind": "cpu", "deps": reads,
                 "command": [PY, "-m", "tools.tpd.run_rolk", "table"]})
    return jobs


def screen():
    out = OUT / "decisions" / "rolk_screen.json"
    require(not out.exists(), "written once")
    from tools.tpd.analyze_rol import compact, screen as rule
    M = mech()
    rx, r0 = M.row("tpd", A.run_name(ARM, 3407)), M.row("tpd", REFERENCE)
    v = rule(compact(rx), compact(r0))
    added = None
    if v["verdict"] == "SCREEN_PASS":
        jobs = follow_up_jobs()
        jf = OUT / "jobs" / "rolk_s23.json"
        require(not jf.exists(), "follow-up jobs written once")
        jf.write_text(json.dumps(jobs, indent=1) + "\n")
        env = dict(os.environ, TPD_DRIVER_DIR=os.environ.get("TPD_DRIVER_DIR", "driver4"))
        subprocess.run([PY, "tools/tpd/chain3.py", "add", str(jf)], cwd=ROOT, env=env, check=True)
        added = {"file": str(jf), "jobs": [j["key"] for j in jobs], "driver": env["TPD_DRIVER_DIR"]}
    save(out, {"arm": ARM, "reference": REFERENCE, "screen": v, "rows": {ARM: rx, "reference": r0},
               "follow_up": added, "written_at_utc": now()}, exclusive=True)
    print(v["verdict"], {k: round(v[k], 3) for k in ("d_ra", "d_e3", "d_R15", "d_F15")}, "->", added)


def table():
    out = OUT / "decisions" / "rolk_table.json"
    require(not out.exists(), "written once")
    from tools.tpd.analyze_syn import final_rule
    M = mech()
    seeds = (3407, 3408, 3409)
    rows = {s: M.row("tpd", A.run_name(ARM, s)) for s in seeds}
    ref = {s: M.row("tpd", f"tpd_rol5_s{s}__causal") for s in seeds}
    v = final_rule(rows)
    names = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
    L = ["## TPD ROL-k：B2 + rol5 微调在因果形式下训练（三种子；描述，不是目标写定的最终配置）", "",
         "| 行 | " + " | ".join(names) + " | E3 | 放大 | R15 / F15 |", "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for label, r in ((f"rol5 {s}（零训练换成因果形式）", ref[s]), (f"rolk {s}", rows[s])):
            ret = r.get("retention")
            L.append(f"| {label} | " + " | ".join(f"{x:.2f}" for x in r["five"]) + f" | {r['e3']:.2f} | {r['amp'][0]:.2f} | "
                     + (f"{ret['R15']:.2f} / {ret['F15']:.2f}" if ret else "–") + " |")
    L.append("| **rolk 三种子均值** | " + " | ".join(f"{x:.2f}" for x in v["mean_five"]) + " | | | |")
    L.append("| S43（引擎三种子） | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | | | |")
    L += ["", f"最终门规则（作描述）：{v['verdict']}；对 S43 差 " + " / ".join(f"{g:+.2f}" for g in v["gap_to_s43"]) +
          f"；每种子 RA < 11.0：{v['seed_ra_below_11']}。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "rolk.md").write_text(text)
    save(out, {"rows": rows, "reference": ref, **v, "description_only": True, "written_at_utc": now()}, exclusive=True)
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
    sub.add_parser("screen")
    sub.add_parser("table")
    a = p.parse_args(argv)
    if a.verb == "accept":
        return RF.accept(ARM, a.seed, a.phase)
    if a.verb == "read":
        return reading(a.seed, a.task)
    return {"screen": screen, "table": table}[a.verb]()


if __name__ == "__main__":
    main()
