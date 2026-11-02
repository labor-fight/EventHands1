"""Translation of the command lines the shared queue holds (one per old script) into the single CLI (CPU only).

The queue is append-only, so its 749 `sm:` commands name the scripts that existed before this package (`train4.py`,
`swap_run2.py` with `S45SM_ENTRY=evaluate2`, `kind_run2.py` with `S45SM_ENTRY=router_domains`, ...). `translate` maps such a
command to the equivalent `cli.py` command; it is used by the tests and by `cli.py jobs translate`, never by an entry.
Scripts without a successor are listed in OBSOLETE (one-off amendment bookkeeping, the never-used context variants and the
illustrations); they are archived with the old tree.
"""
from __future__ import annotations

import re

#: old script -> (verb and fixed leading arguments); arguments of the old command are appended unchanged
SCRIPT = {
    **{s: ("train",) for s in ("train", "train_a1", "train2", "train3", "train4", "train_oracle")},
    **{s: ("accept",) for s in ("accept", "accept2", "accept3", "accept4", "accept_oracle")},
    **{s: ("evaluate",) for s in ("evaluate", "evaluate2", "evaluate_oracle")},
    "evaluate_fast": ("row",),
    **{s: ("readouts",) for s in ("readouts", "readouts2", "readouts_oracle")},
    "bank": ("bank",),
    **{s: ("router-train",) for s in ("router_train", "router_train2", "router_train3", "router_train4", "router_train5",
                                       "router_train7")},
    "decode_variant": ("router-decode",),
    "router_domains": ("router-read", "d1"),
    "router_eval": ("router-read", "gate"),
    "router_ctx": ("router-read", "ctx"),
    "select_r4": ("select", "--id", "r4"),
    "select_r6": ("select", "--id", "r6"),
    "select_r8": ("select", "--id", "r8"),
    "select_r6c": ("select", "--id", "r6c"),
    "decide": ("decide",), "decide2": ("decide",), "decide3": ("decide",),
    "swap_table": ("table", "--round", "7"), "swap_table2": ("table", "--round", "11"),
    "swap_table3": ("table", "--round", "12"), "d1_table": ("table", "--round", "10"),
    "rowcheck": ("read", "--id", "rowcheck_r3a"), "repeat_read": ("read", "--id", "repeat_r13"),
    "cosine_read": ("read", "--id", "cosine_r14"),
    "replay3": ("replay",)}
#: wrappers: the entry named by S45SM_ENTRY decides the verb (the wrapper only registered kinds / patched resolve)
WRAPPERS = ("swap_run", "swap_run2", "kind_run", "kind_run2", "srp_run", "read_run")
ENTRY = {"train2": "train", "accept2": "accept", "evaluate2": "evaluate", "readouts2": "readouts", "evaluate_fast": "row",
         "router_domains": "router-read d1", "router_ctx": "router-read ctx", "decide3": "decide",
         # rounds 13 / 14 engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#s45sm-r13-amend1-20261101)
         "repeat_read": "read --id repeat_r13", "cosine_read": "read --id cosine_r14"}
OBSOLETE = ("a1_archive", "ctx_run", "toy_contraction", "toy_contraction_v2")
SCRIPT_RE = re.compile(r"tools/s45sm/([A-Za-z0-9_]+)\.py$")
CLI = "tools/s45sm/cli.py"


class Obsolete(RuntimeError):
    pass


def split(command):
    """(script name, interpreter tokens, environment assignments other than S45SM_ENTRY, S45SM_ENTRY or None, arguments)
    of an old queue command."""
    command = list(command)
    for i, token in enumerate(command):
        m = SCRIPT_RE.search(token)
        if m:
            break
    else:
        for name in OBSOLETE:                                  # e.g. a bash -c chain of a1_archive.py calls
            if f"tools/s45sm/{name}.py" in " ".join(command):
                raise Obsolete(name)
        raise ValueError("no tools/s45sm script in the command: " + " ".join(command))
    head, args = command[:i], command[i + 1:]
    entry, env, interpreter = None, [], []
    for token in head:
        if token == "/usr/bin/env":
            continue
        if token.startswith("S45SM_ENTRY="):
            entry = token.split("=", 1)[1]
        elif "=" in token and token.split("=", 1)[0].isupper():
            env.append(token)
        else:
            interpreter.append(token)
    return m.group(1), interpreter, env, entry, args


def verb_of(script, entry):
    """The CLI verb (and fixed leading arguments) an old command stands for."""
    if script in OBSOLETE:
        raise Obsolete(script)
    if script in WRAPPERS:
        if entry not in ENTRY:
            raise ValueError(f"{script} needs S45SM_ENTRY in {sorted(ENTRY)}")
        return tuple(ENTRY[entry].split())
    if script in SCRIPT:
        return SCRIPT[script]
    raise ValueError("unknown script " + script)


def translate(command):
    """The equivalent command of the single CLI for an old queue command (a list of strings): the interpreter and any
    environment assignments stay, S45SM_ENTRY and the script name become the verb."""
    script, interpreter, env, entry, args = split(command)
    verb = verb_of(script, entry)
    return (["/usr/bin/env", *env] if env else []) + interpreter + [CLI, *verb, *args]
