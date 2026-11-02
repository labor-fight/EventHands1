#!/usr/bin/env python3
"""S45-SM: the single entry of the study's tools (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031 and each round's section).

    python tools/s45sm/cli.py <verb> ...        (or: python -m tools.s45sm.cli <verb> ...)

Every behaviour that used to be a script per variant is selected by the registry spec (tools/s45sm/registry_r<k>.json) of
the arm / router / cell the arguments name; see tools/s45sm/legacy.py for the old script -> verb table. Verbs:

  configs       --round K [--out DIR] | --check [--round K]   write / check the arm configs (YAML) of a round
  freeze        round --round K --tag T | files --name N --round K --tag T FILE... | identity --archive DIR [--report FILE]
  verify-protocol                                             re-hash every frozen file (identity manifest + legacy manifests)
  train         --arm A --phase P --config C                  fresh 6000-step (50-step preflight) training, two GPUs
  accept        --arm A --phase P                             engineering acceptance of a finished training -> receipt
  replay        --arm A                                       bit-for-bit comparison of a registered replay's preflight
  evaluate      --net N --mode {e1,e2,e3,row} [--step S]      zgz closed-loop readouts, one GPU among 2-7
  row           --net N --block B                             main row on the latency path, GPU 4 exclusive
  readouts      --net N --task T                              S45-TR readout tasks, one GPU
  bank          {run,verify} --arm A [--shard i --shards n]   own-loop state bank
  router-train  --router R --phase P                          train (or anneal) a router, two GPUs
  router-decode --router R                                    link a part-first decoding variant to its base router
  router-read   {d1,gate,ctx} --router R                      router readings, one GPU
  select        --id {r4,r6,r8,r6c} [--router R]              round router selections
  decide        [--round K] [--as-of-round K]                 round summary (default: the latest registered round)
  table         --round {7,10,11,12}                          descriptive tables
  read          --id {rowcheck_r3a,repeat_r13,cosine_r14}     written readings
  jobs          standard --round K [--block B] [--out F] | translate FILE   queue job blocks / old command translation
  plan          <argv of another verb>                        what a command would do (no execution)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

GPU_VERBS = ("evaluate", "row", "readouts", "router-read")
PAIR_VERBS = ("train", "router-train")
SELECT_IDS = ("r4", "r6", "r8", "r6c")
READ_IDS = ("rowcheck_r3a", "repeat_r13", "cosine_r14")
TABLE_ROUNDS = (7, 10, 11, 12)


def build_parser():
    p = argparse.ArgumentParser(prog="s45sm", description=__doc__, allow_abbrev=False,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)

    def verb(name, **kw):
        return sub.add_parser(name, allow_abbrev=False, **kw)
    c = verb("configs")
    c.add_argument("--round", type=int)
    c.add_argument("--check", action="store_true")
    c.add_argument("--out")
    c.add_argument("--root-alias", help="verification in a copy of the repository: the root the files on disk were written under")
    f = verb("freeze")
    fs = f.add_subparsers(dest="what", required=True)
    fr = fs.add_parser("round", allow_abbrev=False)
    fr.add_argument("--round", type=int, required=True)
    fr.add_argument("--tag", required=True, help="pre-registration marker tag, e.g. S45SM-R14-PREREG-20261101")
    ff = fs.add_parser("files", allow_abbrev=False)
    ff.add_argument("--name", required=True)
    ff.add_argument("--round", type=int, required=True)
    ff.add_argument("--tag", required=True)
    ff.add_argument("files", nargs="+")
    fi = fs.add_parser("identity", allow_abbrev=False)
    fi.add_argument("--archive", required=True)
    fi.add_argument("--report")
    verb("verify-protocol")
    t = verb("train")
    t.add_argument("--arm")
    t.add_argument("--phase", choices=("preflight", "formal"))
    t.add_argument("--config", required=True)
    for name in ("accept",):
        a = verb(name)
        a.add_argument("--arm", required=True)
        a.add_argument("--phase", required=True, choices=("preflight", "formal"))
    verb("replay").add_argument("--arm", required=True)
    e = verb("evaluate")
    e.add_argument("--net", required=True)
    e.add_argument("--mode", required=True, choices=("row", "e1", "e2", "e3"))
    e.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    r = verb("row")
    r.add_argument("--net", required=True)
    r.add_argument("--block", required=True, help="exclusive measurement block, e.g. r14")
    o = verb("readouts")
    o.add_argument("--net", required=True)
    o.add_argument("--task", required=True)
    b = verb("bank")
    b.add_argument("action", choices=("run", "verify"))
    b.add_argument("--arm", required=True)
    b.add_argument("--shard", type=int, default=0)
    b.add_argument("--shards", type=int, default=1)
    rt = verb("router-train")
    rt.add_argument("--router", required=True)
    rt.add_argument("--phase", required=True, choices=("preflight", "formal"))
    verb("router-decode").add_argument("--router", required=True)
    rr = verb("router-read")
    rr.add_argument("what", choices=("d1", "gate", "ctx"))
    rr.add_argument("--router", required=True)
    s = verb("select")
    s.add_argument("--id", required=True, choices=SELECT_IDS)
    s.add_argument("--router", help="r6c only")
    d = verb("decide")
    d.add_argument("--round", type=int)
    d.add_argument("--as-of-round", type=int, help="keep only arms registered in rounds <= K (reproduces a recorded summary)")
    verb("table").add_argument("--round", type=int, required=True, choices=TABLE_ROUNDS)
    verb("read").add_argument("--id", required=True, choices=READ_IDS)
    j = verb("jobs")
    js = j.add_subparsers(dest="what", required=True)
    jst = js.add_parser("standard", allow_abbrev=False)
    jst.add_argument("--round", type=int, required=True)
    jst.add_argument("--block")
    jst.add_argument("--out")
    jst.add_argument("--dry-run", action="store_true")
    jt = js.add_parser("translate", allow_abbrev=False)
    jt.add_argument("file")
    pl = verb("plan")
    pl.add_argument("argv", nargs=argparse.REMAINDER)
    return p


def parse(argv):
    return build_parser().parse_args(argv)


# --------------------------------------------------------------------------- plans (no execution)
def plan(argv):
    """What a command line would do, resolved from the registry only (nothing is read from the outputs, nothing runs)."""
    from . import registry as R
    from .paths import require
    from .routing import NEEDS
    ns = parse(argv)
    out = {"verb": ns.verb, "args": {k: v for k, v in vars(ns).items() if k not in ("verb", "argv")},
           "needs_gpu": ns.verb in GPU_VERBS or (ns.verb == "bank" and ns.action == "run") or
           (ns.verb == "router-read"), "gpu_pair": ns.verb in PAIR_VERBS,
           "exclusive_gpu": ns.verb == "row" or (ns.verb == "evaluate" and ns.mode == "row")}
    if ns.verb in ("train", "accept", "replay"):
        arm = ns.arm
        require(arm in R.registry()["arms"] or arm in R.oracles(), "unregistered S45-SM arm " + str(arm))
        out.update(family="oracle" if arm in R.oracles() else R.family(arm), round=(R.oracles() if arm in R.oracles()
                                                                                     else R.registry()["arms"])[arm]["round"])
        if ns.verb == "train":
            require(ns.phase in R.phases(arm), "phase not registered for the arm")
            require(Path(ns.config).resolve() == R.config_path(arm, ns.phase).resolve(), "Unregistered config path")
            out.update(config=str(R.config_path(arm, ns.phase)), output=str(R.output_dir(arm, ns.phase)))
            if arm not in R.oracles():
                out.update(sampler=R.sampler(arm), schedule=R.schedule(arm),
                           register_kinds=list(NEEDS[R.router_spec(R.arm_spec(arm)["router"])["kind"]]))
    elif ns.verb in ("evaluate", "row", "readouts"):
        net = ns.net
        if net == "s45ref" or net.startswith("s45ref_"):
            out.update(family="reference", config=None)
        else:
            arm, step, fam, spec = R.net_spec(net)
            out.update(family=fam, arm=arm, step=step if ns.verb == "readouts" else getattr(ns, "step", 6000),
                       config=str(R.config_path(arm, "formal")), register_kinds=(
                           ["srp", "dw", "hier"] if fam == "cell" else
                           ([] if fam == "oracle" else list(NEEDS[R.router_spec(spec["router"])["kind"]]))))
            if fam == "cell":
                out["trunk"] = R.trunk_of(spec)
                out["router"] = spec["router"]
    elif ns.verb in ("router-train", "router-decode", "router-read", "select"):
        router = getattr(ns, "router", None)
        if router:
            spec = R.router_spec(router)
            out.update(kind=spec["kind"], round=spec["round"])
    return out


# --------------------------------------------------------------------------- handlers
def _configs(ns):
    from . import configs as K
    if ns.check:
        result = K.check_configs(ns.round, ns.root_alias)
        bad = [k for k, v in result.items() if not v]
        print(f"{len(result) - len(bad)} / {len(result)} configs equal their generator's output")
        for k in bad:
            print("DIFFERENT:", k)
        return 1 if bad else 0
    from .paths import require
    require(ns.round is not None, "--round K is required to write configs")
    for path in K.write_configs(ns.round, ns.out, ns.root_alias):
        print(path)
    return 0


def _freeze(ns):
    from . import protocol as P
    if ns.what == "round":
        target, n = P.freeze_round(ns.round, ns.tag)
    elif ns.what == "files":
        target, n = P.freeze_files(ns.name, ns.round, ns.tag, ns.files)
    else:
        target, n = P.build_identity(ns.archive, ns.report), None
    print(target, "" if n is None else f"{n} files")
    return 0


def _verify_protocol(ns):
    from . import protocol as P
    manifests = P.verify_protocol()
    print(f"OK {len(manifests)} manifests; code identity {P.code_identity()}")
    return 0


def _train(ns):
    from . import train
    args = []
    for flag, value in (("--arm", ns.arm), ("--phase", ns.phase), ("--config", ns.config)):
        if value is not None:                       # a DDP child gets arm and phase through the environment instead
            args += [flag, value]
    train.main(args)
    return 0


def _accept(ns):
    from . import verify
    verify.accept(ns.arm, ns.phase)
    return 0


def _replay(ns):
    from . import verify
    verify.replay(ns.arm)
    return 0


def _evaluate(ns):
    from . import evaluate
    evaluate.evaluate(ns.net, ns.mode, ns.step)
    return 0


def _row(ns):
    from . import evaluate
    evaluate.row_fast(ns.net, ns.block)
    return 0


def _readouts(ns):
    from . import readouts
    readouts.readout(ns.net, ns.task)
    return 0


def _bank(ns):
    from . import bank
    from .paths import require
    from .registry import registry
    require(ns.arm in registry()["arms"], "unregistered arm")
    bank.run(ns.arm, ns.shard, ns.shards) if ns.action == "run" else bank.verify(ns.arm)
    return 0


def _router_train(ns):
    from . import router_train
    router_train.train(ns.router, ns.phase)
    return 0


def _router_decode(ns):
    from . import router_train
    router_train.decode_variant(ns.router)
    return 0


def _router_read(ns):
    from . import router_read
    {"d1": router_read.d1, "gate": router_read.gate, "ctx": router_read.ctx_probe}[ns.what](ns.router)
    return 0


def _select(ns):
    from . import selection
    from .paths import require
    if ns.id == "r6c":
        require(ns.router, "select r6c needs --router")
        selection.run_ctx(ns.router)
    else:
        require(ns.router is None, "--router is for r6c only")
        selection.run(ns.id)
    return 0


def _decide(ns):
    from . import summary
    from .registry import latest_round
    summary.summarize(ns.round if ns.round is not None else latest_round(), ns.as_of_round)
    return 0


def _table(ns):
    from . import tables
    tables.table(ns.round)
    return 0


def _read(ns):
    from . import readings
    readings.read_one(ns.id)
    return 0


def _jobs(ns):
    import json
    from . import jobs
    if ns.what == "translate":
        for key, command in jobs.translate_file(ns.file).items():
            print(key, " ".join(command))
        return 0
    plan_ = jobs.standard_round(ns.round, block=ns.block)
    if ns.out and not ns.dry_run:
        with open(ns.out, "x") as stream:
            stream.write(json.dumps(plan_, indent=2) + "\n")
    for j in plan_:
        print(j["key"], j["kind"], j["mem_gib"], j["deps"])
    return 0


def _plan(ns):
    import json
    print(json.dumps(plan(ns.argv), indent=2))
    return 0


HANDLERS = {"configs": _configs, "freeze": _freeze, "verify-protocol": _verify_protocol, "train": _train, "accept": _accept,
            "replay": _replay, "evaluate": _evaluate, "row": _row, "readouts": _readouts, "bank": _bank,
            "router-train": _router_train, "router-decode": _router_decode, "router-read": _router_read, "select": _select,
            "decide": _decide, "table": _table, "read": _read, "jobs": _jobs, "plan": _plan}


def main(argv=None):
    ns = parse(sys.argv[1:] if argv is None else argv)
    return HANDLERS[ns.verb](ns)


def _bootstrap():
    """`python tools/s45sm/cli.py ...`: relative imports need the package, and this directory must not shadow stdlib /
    third-party modules (it holds `train.py`, `evaluate.py`, ...)."""
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    for p in (str(root / "model"), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib
    return importlib.import_module(f"tools.{here.name}.cli")


if __name__ == "__main__":
    sys.exit((main if __package__ else _bootstrap().main)())
