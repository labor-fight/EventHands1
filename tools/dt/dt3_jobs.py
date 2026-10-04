#!/usr/bin/env python3
"""DT3 round: sched2 job JSON for the zero-training probes (docs/DT3_PREREG.md). Prints the job list to stdout and submits
nothing (a dry run by construction); fields as in tools/dt/jobs2.py / fa_jobs.py (id, kind, cwd, [after], prio, cores, mem_mib,
cmd, [env]). Every id starts with `dt3_`, every path is absolute (cwd = the live repo, so the staged code must be deployed
before a job runs; use `--after` for the deploy job).

    # evalx on runs with extra flags; --suffix is required so a recorded evalx_<...>.json is never overwritten
    python tools/dt/dt3_jobs.py evalx --runs dt_dz_l3_s3407 dt_dz_l3_s3408 --extra "--window-ms 150 --clip-run-start --tf --perturb" \\
        --suffix cs150 --prio 80 --id-prefix A2 > /tmp/j.json
    # the recorded AdaptiveFilter spec applied at window W to other runs (fa_jobs.py plus --window-ms)
    python tools/dt/dt3_jobs.py fapply --runs dt_dz_l3w128_s3407 --spec outputs/dt2/filter_sweep/recommended_spec.json \\
        --window-ms 200 --out-root outputs/dt3/filter_apply --prio 75 > /tmp/j.json
    # inference-time probes (tools/dt/infer_probe.py <sub> ...); seed-ens is ONE job over all --runs
    python tools/dt/dt3_jobs.py probe --sub ens-win --runs dt_dz_l3_s3407 dt_dz_l3_s3408 dt_dz_l3_s3409 \\
        --args "--window-ms 200 --windows 100,200,300 --clip-run-start" --prio 70 --after dt3_deploy > /tmp/j.json
    SCHED_PROG=outputs/dt3 python tools/tracking/sched2.py submit /tmp/j.json

    # gates: dt3_gate_cpu (pytest, device cpu) then dt3_gate_gpu (bit-identity repro on a GPU, after the cpu gate); the probes wait on
    # dt3_gate_gpu, a failure of either skips them (sched2 skips dependents). Submit the gates before the probes.
    python tools/dt/dt3_jobs.py gate --device cpu --cmd "bash /abs/dt3_gate_cpu.sh" > /tmp/g1.json
    python tools/dt/dt3_jobs.py gate --device gpu --cmd "bash /abs/dt3_gate_gpu.sh" --after dt3_gate_cpu --gpus 6 > /tmp/g2.json
    # then the probes with --after dt3_gate_gpu

GPU placement is sched2's (slots.json) unless `--gpus 6 7` pins job i to gpus[i % 2] (sched2's `gpu` field; only 4-7 are accepted).
An `--after` id that is not in the queue (nor finished) makes the job wait forever (a warning is printed for every `--after` id
not in the same output). `probe` jobs follow tools/dt/infer_probe.py (checked against the package-C stage): positional sub-command
{ens-win,adabn,tta-pol,seed-ens}, `--run-dir RD --ckpt last` (seed-ens: `--run-dirs RD...`) and the `--args` text appended
verbatim. `--out-dir` and `--tag` are NOT added: the script's own defaults apply (outputs/dt3/<sub>/<run>_<tag>.json, tag
`<core>_pw<window-ms>[_clip][_af<6hex>]` with core ensw<windows joined by ->, adabn_i<iters>_bs<bs>, ttapol, seedens), so clip /
filter / window variants never share a file name; give `--out-dir` / `--tag` in `--args` to override, and match the dt3_cmp path
templates to the names. Probe jobs: kind `probe`, cores 2, mem 4000 MiB (adabn 6000: batch-32 forward plus ~0.9 GB of host-side
inputs), ids `dt3_B{1,2,3,4}_<run>` / `dt3_B4_seed_ens` (`--id-prefix` to separate repeated settings of one sub-command).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

PY = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
REPO = "/data1/lyq/code/mesh/EventHands1"
#: sub-command -> (id prefix, mem_mib)
PROBES = {"ens-win": ("B1", 4000), "adabn": ("B2", 6000), "tta-pol": ("B3", 4000), "seed-ens": ("B4", 4000)}


def absp(p: str) -> str:
    return p if os.path.isabs(p) else f"{REPO}/{p}"


def run_dir(run: str) -> str:
    return f"{REPO}/outputs/semkine/{run}"


def job(jid, kind, prio, cores, mem, cmd, after=(), env=None) -> dict:
    j = {"id": jid, "kind": kind, "cwd": REPO}
    if after:
        j["after"] = list(after)
    j.update({"prio": prio, "cores": cores, "mem_mib": mem, "cmd": cmd})
    if env:
        j["env"] = dict(env)
    return j


def pin_gpus(jobs: list, gpus: list) -> list:
    """`--gpus G...`: job i is pinned to gpus[i % len] (sched2's `gpu` field: its only candidate, the memory accounting still
    applies). Only GPUs 4-7 exist for this round."""
    bad = [g for g in gpus if g not in (4, 5, 6, 7)]
    if bad:
        sys.exit(f"--gpus {bad}: only GPUs 4-7 may be used")
    for i, j in enumerate(jobs):
        j["gpu"] = gpus[i % len(gpus)]
    return jobs


def jid_of(prefix: str, run: str) -> str:
    """`dt3_<prefix>_<run>`; a prefix that already starts with dt3_ is not doubled."""
    prefix = prefix[4:] if prefix.startswith("dt3_") else prefix
    return f"dt3_{prefix}_{run}" if run else f"dt3_{prefix}"


def has_flag(tokens: list, flag: str) -> bool:
    return any(t == flag or t.startswith(flag + "=") for t in tokens)


def flag_value(tokens: list, flag: str):
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return None


def evalx_jobs(a) -> list:
    extra = shlex.split(a.extra)
    if has_flag(extra, "--suffix") or has_flag(extra, "--run-dir") or has_flag(extra, "--ckpt"):
        sys.exit("--extra must not carry --suffix / --run-dir / --ckpt (they are set by this tool)")
    if not a.suffix and not a.allow_empty_suffix:
        sys.exit("--suffix is required: without one evalx overwrites the recorded evalx_<split>_<ckpt>_<flags>.json of the run "
                 "(--allow-empty-suffix if that is really intended)")
    out = []
    for run in a.runs:
        cmd = [PY, f"{REPO}/tools/tracking/evalx.py", "eval", "--run-dir", run_dir(run), "--ckpt", "last"] + extra
        if a.suffix:
            cmd += ["--suffix", a.suffix]
        out.append(job(jid_of(a.id_prefix, run), "eval", a.prio, a.cores, a.mem, cmd, a.after, a.env))
    return out


def fapply_jobs(a) -> list:
    extra = shlex.split(a.extra)
    for f in ("--run-dir", "--ckpt", "--filter-spec", "--window-ms", "--out-dir", "--gains"):
        if has_flag(extra, f):
            sys.exit(f"--extra must not carry {f} (set by this tool)")
    out = []
    for run in a.runs:
        cmd = [PY, f"{REPO}/tools/dt/filter_eval.py", "--run-dir", run_dir(run), "--ckpt", "last",
               "--filter-spec", absp(a.spec), "--gains", a.gains, "--perturb", "--window-ms", str(a.window_ms)] + extra
        cmd += ["--out-dir", f"{absp(a.out_root)}/{run}"]
        out.append(job(jid_of(a.id_prefix or f"fa_w{a.window_ms}", run), "eval", a.prio, a.cores, a.mem, cmd, a.after, a.env))
    return out


def probe_jobs(a) -> list:
    prefix, mem = PROBES[a.sub]
    toks = shlex.split(a.args)
    for f in ("--run-dir", "--run-dirs", "--ckpt"):
        if has_flag(toks, f):
            sys.exit(f"--args must not carry {f} (set by this tool)")
    if not has_flag(toks, "--window-ms"):
        print("warning: --args has no --window-ms (the probes require the primary window)", file=sys.stderr)
    pre = a.id_prefix or prefix
    mem = a.mem or mem
    base = [PY, f"{REPO}/tools/dt/infer_probe.py", a.sub]
    if a.sub == "seed-ens":                                         # one job over all runs, no per-run name
        cmd = base + ["--run-dirs"] + [run_dir(r) for r in a.runs] + ["--ckpt", "last"] + toks
        return [job(jid_of(pre, "seed_ens"), "probe", a.prio, a.cores, mem, cmd, a.after, a.env)]
    return [job(jid_of(pre, run), "probe", a.prio, a.cores, mem,
                base + ["--run-dir", run_dir(run), "--ckpt", "last"] + toks, a.after, a.env) for run in a.runs]


def gate_jobs(a) -> list:
    """One `kind: cmd` job that runs `--cmd` and must exit 0; `--after` it from the probes: a failing gate ends `failed` and
    sched2 then skips every dependent. `--device cpu` adds sched2's `device: cpu` (launched with CUDA_VISIBLE_DEVICES="": the
    pytest gate); `--device gpu` has no `device` field (the repro gates are bit-identity checks that need the GPU), mem 2500 MiB,
    and `--gpus G` pins it (4-7; 5-7 preferred)."""
    cmd = shlex.split(a.gate_cmd)
    if not cmd:
        sys.exit("--cmd is empty")
    if a.device == "cpu" and a.gpus:
        sys.exit("--gpus makes no sense with --device cpu")
    mem = a.mem or (2500 if a.device == "gpu" else 1500)
    j = job(jid_of(a.id_prefix or f"gate_{a.device}", ""), "cmd", a.prio, a.cores, mem, cmd, a.after, a.env)
    if a.device == "cpu":
        j["device"] = "cpu"
    return [j]


def check_after(jobs: list) -> None:
    """sched2 treats an `after` id that is neither queued nor finished as pending forever, so say so for every id this output
    does not contain (it may legitimately be queued already)."""
    ids = {j["id"] for j in jobs}
    for j in jobs:
        for d in j.get("after", []):
            if d not in ids:
                print(f"warning: {j['id']}: --after {d} is not in this output; it must already be queued and end `done`, "
                      f"otherwise {j['id']} waits forever (a failed dependency skips it)", file=sys.stderr)
                ids.add(d)


def check(jobs: list) -> None:
    """Unique `dt3_` ids; a missing run directory (or checkpoint) is only a warning, the run may still be training."""
    ids = [j["id"] for j in jobs]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        sys.exit(f"duplicate job ids {dup} (repeated --runs? give a different --id-prefix)")
    for j in jobs:
        if not j["id"].startswith("dt3_"):
            sys.exit(f"id {j['id']!r} does not start with dt3_")
        c = j["cmd"]
        rds = [c[c.index("--run-dir") + 1]] if "--run-dir" in c else []
        if "--run-dirs" in c:
            i = c.index("--run-dirs") + 1
            while i < len(c) and not c[i].startswith("--"):
                rds.append(c[i])
                i += 1
        for rd in rds:
            if not Path(rd).is_dir():
                print(f"warning: {j['id']}: run directory {rd} does not exist", file=sys.stderr)
            elif not list(Path(rd).glob("*step=*.ckpt")):
                print(f"warning: {j['id']}: no checkpoint in {rd}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, prio, cores, mem):
        p.add_argument("--prio", type=int, default=prio)
        p.add_argument("--cores", type=int, default=cores)
        p.add_argument("--mem", type=int, default=mem, help="mem_mib estimate for the scheduler (probe: 0 = per-sub default)")
        p.add_argument("--after", nargs="*", default=[], help="job ids this job waits for (e.g. the deploy job)")
        p.add_argument("--env", nargs="*", default=[], metavar="K=V", help="extra environment of the job")
        p.add_argument("--gpus", nargs="+", type=int, default=None, help="pin the jobs round-robin to these GPUs (4-7)")
        p.add_argument("--summary", action="store_true", help="also print one line per job to stderr")

    e = sub.add_parser("evalx", help="evalx.py eval --run-dir RUN --ckpt last <extra> --suffix S, one job per run")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--extra", default="", help='evalx eval flags, e.g. "--window-ms 150 --tf --perturb"')
    e.add_argument("--suffix", default=None)
    e.add_argument("--allow-empty-suffix", action="store_true")
    e.add_argument("--id-prefix", default="evalx")
    common(e, 80, 1, 1500)
    f = sub.add_parser("fapply", help="filter_eval.py with a fixed AdaptiveFilter spec at window W, one job per run")
    f.add_argument("--runs", nargs="+", required=True)
    f.add_argument("--spec", required=True)
    f.add_argument("--window-ms", type=int, default=50)
    f.add_argument("--extra", default="")
    f.add_argument("--gains", default="0.5,1.0,0.5", help="the constant-gain companion triple (fa_jobs.py)")
    f.add_argument("--out-root", default="outputs/dt3/filter_apply")
    f.add_argument("--id-prefix", default=None, help="default fa_w<window-ms>")
    common(f, 75, 1, 2500)
    p = sub.add_parser("probe", help="infer_probe.py <sub> ...: ens-win | adabn | tta-pol | seed-ens")
    p.add_argument("--sub", required=True, choices=sorted(PROBES))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--args", default="", help='probe flags, e.g. "--window-ms 200 --windows 100,200,300"')
    p.add_argument("--id-prefix", default=None, help="default B1 / B2 / B3 / B4 by sub-command")
    common(p, 70, 2, 0)
    g = sub.add_parser("gate", help="one `kind: cmd` job that runs --cmd; --device cpu (pytest) or gpu (repro); probes --after it")
    g.add_argument("--cmd", dest="gate_cmd", required=True,
                   help='command line (shlex-split), e.g. "bash /abs/path/gate.sh"; exit 0 = gate passed')
    g.add_argument("--device", required=True, choices=["cpu", "gpu"],
                   help="cpu: sched2 device cpu (CUDA_VISIBLE_DEVICES empty); gpu: no device field, mem 2500, optional --gpus G")
    g.add_argument("--id-prefix", default=None, help="job id dt3_<this> (default gate_<device>: dt3_gate_cpu / dt3_gate_gpu)")
    common(g, 90, 1, 0)
    # "--extra --tf" / "--args --clip-run-start": argparse would take the value for an option, so glue it with "="
    argv = sys.argv[1:]
    argv = [f"{t}={argv[i + 1]}" if t in ("--extra", "--args", "--cmd") and i + 1 < len(argv) else t
            for i, t in enumerate(argv) if not (i > 0 and argv[i - 1] in ("--extra", "--args", "--cmd"))]
    a = ap.parse_args(argv)
    a.env = dict(kv.split("=", 1) for kv in a.env)
    jobs = {"evalx": evalx_jobs, "fapply": fapply_jobs, "probe": probe_jobs, "gate": gate_jobs}[a.cmd](a)
    if a.gpus:
        pin_gpus(jobs, a.gpus)
    check(jobs)
    check_after(jobs)
    json.dump(jobs, sys.stdout, indent=1)
    print()
    if a.summary:
        for j in jobs:
            print(f"{j['id']:48s} {j['kind']:6s} prio {j['prio']:3d} cores {j['cores']} mem {j['mem_mib']:5.0f} "
                  f"gpu {j.get('gpu', '-')} after {j.get('after', [])}: {' '.join(shlex.quote(c) for c in j['cmd'][1:])}", file=sys.stderr)


if __name__ == "__main__":
    main()
