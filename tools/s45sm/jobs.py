"""Standard job blocks for the shared queue (tools/s45rt/chain_r2.py `add`), as the CLI's commands (CPU only).

Only the blocks every round repeated are generated; a round's jobs were always assembled by hand from them (extra
dependencies between rounds, an arm that trains first, a bank), so nothing here replaces that assembly:
  train_block     preflight -> acceptance -> formal -> acceptance of one arm (`pair`: one GPU pair, 130 GiB; `cpu`: 4 GiB)
  evaluate_block  zgz E1 at steps 5000 / 5500 / 6000, E2, E3, and the readouts tl_e1 / tl_e3 / zl_e3, tl_f1 (after tl_e1),
                  tl_e1 at steps 5000 / 5500 (`gpu`, 15 GiB each) of one network
  row_block       the exclusive main-row measurements of a block: S45 itself (`s45ref_<block>`) first, then the networks
                  serially (`exclusive`, GPU 4)
  reading_job     a CPU job of a written reading / summary (`decide`, `read`, `table`)
`translate_file` maps an old jobs file to the single CLI (tools/s45sm/legacy.py).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import legacy
from .paths import ROOT, require
from .registry import arm_spec, config_path, oracles, registry, trunk_of

DEFAULT_PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
CLI = legacy.CLI
ALLOC = ["/usr/bin/env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"]
MEM = {"pair": 130, "cpu": 4, "gpu": 15, "exclusive": 15}
LATE_STEPS = (5000, 5500)
EVALUATIONS = (("e1", 5000), ("e1", 5500), ("e1", 6000), ("e2", 6000), ("e3", 6000))
READOUTS = ("tl_e1", "tl_e3", "zl_e3", "tl_f1")


def prereg(round_):
    """The ledger anchor a round's job notes point to (from the registry file)."""
    import json as _json
    reg = registry()["rounds"][round_]
    return _json.loads((ROOT / reg).read_text()).get("prereg")


def job(key, kind, deps, command, note, mem=None):
    return {"key": key, "kind": kind, "mem_gib": MEM[kind] if mem is None else mem, "deps": list(deps),
            "command": list(command), "note": note}


def _config(arm, phase):
    return str(config_path(arm, phase).relative_to(ROOT))


def train_block(round_, arm, note, python=DEFAULT_PYTHON, deps_pre=(), deps_formal=(), deps_accept_pre=(),
                deps_accept_formal=()):
    k = f"sm:r{round_}"
    pre, acc_pre = f"{k}:pre:{arm}", f"{k}:acc:pre:{arm}"
    formal, acc_formal = f"{k}:formal:{arm}", f"{k}:acc:formal:{arm}"
    return [
        job(pre, "pair", deps_pre, ALLOC + [python, CLI, "train", "--arm", arm, "--phase", "preflight",
                                           "--config", _config(arm, "preflight")], note),
        job(acc_pre, "cpu", [pre, *deps_accept_pre], [python, CLI, "accept", "--arm", arm, "--phase", "preflight"], note),
        job(formal, "pair", [acc_pre, *deps_formal], ALLOC + [python, CLI, "train", "--arm", arm, "--phase", "formal",
                                                              "--config", _config(arm, "formal")], note),
        job(acc_formal, "cpu", [formal, *deps_accept_formal], [python, CLI, "accept", "--arm", arm, "--phase", "formal"], note)]


def evaluate_block(round_, net, note, python=DEFAULT_PYTHON, deps=(), f1_deps=()):
    """`deps`: the acceptance job(s) of the network's (trunk's) formal training; `f1_deps` are added to tl_f1's
    dependency on tl_e1."""
    k = f"sm:r{round_}"
    out = []
    for mode, step in EVALUATIONS:
        cmd = [python, CLI, "evaluate", "--net", net, "--mode", mode] + (["--step", str(step)] if mode == "e1" else [])
        out.append(job(f"{k}:ev:{net}:{mode}:{step}", "gpu", deps, cmd, note))
    for task in READOUTS:
        dep = [f"{k}:rs:{net}:tl_e1", *f1_deps] if task == "tl_f1" else list(deps)
        out.append(job(f"{k}:rs:{net}:{task}", "gpu", dep, [python, CLI, "readouts", "--net", net, "--task", task], note))
    for step in LATE_STEPS:
        out.append(job(f"{k}:rs:{net}_{step}:tl_e1", "gpu", deps,
                       [python, CLI, "readouts", "--net", f"{net}_{step}", "--task", "tl_e1"], note))
    return out


def row_block(round_, block, nets, note, python=DEFAULT_PYTHON, deps_first=()):
    """S45 re-measured in `block`, then each network, one after the other (GPU 4 exclusive)."""
    k = f"sm:r{round_}"
    out, previous = [], None
    for net in [f"s45ref_{block}", *nets]:
        out.append(job(f"{k}:row:{net}", "exclusive", list(deps_first) if previous is None else [previous],
                       [python, CLI, "row", "--net", net, "--block", block], note))
        previous = out[-1]["key"]
    return out


def reading_job(round_, args, dep, note, python=DEFAULT_PYTHON, key=None):
    """`args`: the CLI arguments after the verb, e.g. ("decide", "--round", "3") or ("read", "--id", "repeat_r13")."""
    return job(key or f"sm:r{round_}:{'decide' if args[0] == 'decide' else 'read'}", "cpu", [dep] if isinstance(dep, str) else dep,
               [python, CLI, *args], note)


def acceptance_key(name):
    """The queue key of the acceptance of a network's formal training (the dependency of its evaluations)."""
    spec = oracles()[name] if name in oracles() else arm_spec(name)
    return f"sm:r{spec['round']}:acc:formal:{name}"


def standard_round(round_, python=DEFAULT_PYTHON, block=None, reading=None):
    """The standard blocks of a round's arms in registry order: a trained arm = train block + evaluate block, a cell =
    evaluate block on its trunk's acceptance; then the row block (block tag `r<round>` unless given) and an optional
    reading job (args after the verb)."""
    note = f"S45-SM r{round_} ({prereg(round_)})" if prereg(round_) else f"S45-SM r{round_}"
    reg = registry()["arms"]
    arms = [a for a, s in reg.items() if s["round"] == round_ and not s.get("phases") == ["preflight"]]
    out = []
    for arm in arms:
        spec = reg[arm]
        trunk = trunk_of(spec)
        if trunk is None:
            out += train_block(round_, arm, note, python)
            out += evaluate_block(round_, arm, note, python, [f"sm:r{round_}:acc:formal:{arm}"])
        else:
            out += evaluate_block(round_, arm, note, python, [acceptance_key(trunk)])
    if arms:
        out += row_block(round_, block or f"r{round_}", arms, note, python, [j["key"] for j in out])
    if reading:
        out.append(reading_job(round_, reading, out[-1]["key"] if out else [], note, python))
    return out


def translate_file(path):
    """{key: new command} for every job of an old jobs file that has a successor (obsolete scripts are skipped)."""
    out = {}
    for j in json.loads(Path(path).read_text()):
        try:
            out[j["key"]] = legacy.translate(j["command"])
        except legacy.Obsolete:
            continue
    return out
