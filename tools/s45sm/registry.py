"""S45-SM registry: arms, routers and label-route oracles of every round, and the names derived from them.

One `registry_r<k>.json` per round (frozen with the round's manifest; never edited afterwards). Every behavioural
difference between arms is a registry field (docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031 and each round's
section), so the entries (train, evaluate, accept, ...) read the spec instead of existing once per variant:

arm fields    router, assoc (face_route | pfa | spfa), onhand_tau, dz, fast, bank {from, p}, sampler (dedicated),
              repeat_of, lr_schedule (cosine), schedule_of, replay_of, phases, conditional (select_r4),
              trunk (round 7) / pair_trunk (rounds >= 9) = a zero-training cell, pairing / note (informational)
router fields kind (t0p | sr | srp | dw | srph | dwh | t0ph), checkpoint(+sha256) (t0p), heldout, steps, dz, aug, kd,
              params, anneal_of + anneal_steps, decode_of
oracles       (key "oracles") base, assoc: S45-AT t1ddz with its association replaced by PFA (not deployable)

Families of a network name: `oracle`, `cell` (zero training: a trunk of another network + this arm's router), `replay`
(engineering replay, preflight only) and `router` (a trained arm: the router's routes enter at training time).
"""
from __future__ import annotations

import json

from .paths import (BANKS, CONFIGS, OUT, REGISTRY_DIR, ROOT, ROUTERS_DIR, SAMPLERS, SCHEDULES, read, recorded, require,
                    sha)

STEPS = (5000, 5500, 6000)
CELL_KEYS = ("trunk", "pair_trunk")
KINDS = ("t0p", "sr", "srp", "dw", "srph", "dwh", "t0ph")
ASSOCS = ("face_route", "pfa", "spfa")


def _files():
    return sorted(REGISTRY_DIR.glob("registry_r*.json"))


def registry():
    """Every registered round, merged (file order = lexicographic, as it fixes the key order of the summaries);
    arm / router names never repeat across rounds. Re-read on every call (tests patch it)."""
    arms, routers, rounds = {}, {}, {}
    for path in _files():
        r = json.loads(path.read_text())
        rounds[int(r["round"])] = str(path.relative_to(ROOT))
        for name, spec in r["routers"].items():
            require(name not in routers, "router registered twice: " + name)
            routers[name] = {**spec, "round": int(r["round"])}
        for name, spec in r["arms"].items():
            require(name not in arms, "arm registered twice: " + name)
            require(spec["router"] in routers, "arm uses an unregistered router: " + name)
            arms[name] = {**spec, "round": int(r["round"])}
    return {"arms": arms, "routers": routers, "rounds": rounds}


def oracles():
    out = {}
    for path in _files():
        r = json.loads(path.read_text())
        for name, spec in r.get("oracles", {}).items():
            require(name not in out, "oracle registered twice: " + name)
            out[name] = {**spec, "round": int(r["round"])}
    return out


def arm_spec(arm):
    arms = registry()["arms"]
    require(arm in arms, "unregistered S45-SM arm " + repr(arm))
    return arms[arm]


def router_spec(name):
    routers = registry()["routers"]
    require(name in routers, "unregistered S45-SM router " + repr(name))
    return routers[name]


def oracle_spec(arm):
    o = oracles()
    require(arm in o, "unregistered S45-SM oracle arm " + repr(arm))
    return o[arm]


def latest_round():
    return max(registry()["rounds"])


def trunk_of(spec):
    """The trunk a cell uses (round-7 key `trunk`, rounds >= 9 `pair_trunk`), or None for a trained arm."""
    return spec.get("pair_trunk", spec.get("trunk"))


def family(name):
    """oracle | cell | replay | router for a registered network name (arm or oracle)."""
    if name in oracles():
        return "oracle"
    spec = arm_spec(name)
    if trunk_of(spec) is not None:
        return "cell"
    return "replay" if spec.get("replay_of") is not None else "router"


def run_name(arm, phase="formal"):
    require(arm in registry()["arms"] or arm in oracles(), "unregistered S45-SM network " + repr(arm))
    return f"s45sm_{arm}_s3407" + ("_preflight" if phase == "preflight" else "")


def config_path(arm, phase):
    return CONFIGS / (run_name(arm, phase) + ".yaml")


def output_dir(arm, phase):
    return OUT / ("preflight" if phase == "preflight" else "runs") / run_name(arm)


def phases(name):
    spec = oracle_spec(name) if name in oracles() else arm_spec(name)
    return tuple(spec.get("phases", ("preflight", "formal")))


def router_dir(name, phase="formal"):
    return ROUTERS_DIR / (name if phase == "formal" else name + "_preflight")


def router_checkpoint(name, phase="formal"):
    spec = router_spec(name)
    if spec["kind"] == "t0p":
        require(phase == "formal", "T0P is a frozen S45-RT checkpoint")
        return ROOT / spec["checkpoint"]
    return router_dir(name, phase) / f"{name}_final.ckpt"


def router_ready(name):
    """The router checkpoint the arms read: S45-RT's registered SHA, or this study's completed formal training."""
    spec, ckpt = router_spec(name), router_checkpoint(name)
    if spec["kind"] == "t0p":
        require(sha(ckpt) == spec["checkpoint_sha256"], "T0P checkpoint changed")
    else:
        done = read(ckpt.with_suffix(".json"))
        require(done["status"] == "COMPLETE" and done["phase"] == "formal" and done["checkpoint_sha256"] == sha(ckpt),
                "router training not complete: " + name)
    return ckpt


def bank_dir(arm):
    return BANKS / arm


def bank_ready(arm):
    receipt = read(bank_dir(arm).parent / f"{arm}_receipt.json")
    require(receipt["status"] == "PASS", "state bank not verified: " + arm)
    for name, digest in receipt["files_sha256"].items():
        require(sha(bank_dir(arm) / name) == digest, "state bank changed: " + name)
    return bank_dir(arm)


def sampler(arm):
    s = arm_spec(arm).get("sampler", "legacy_none")
    require(s in SAMPLERS, "unregistered sampler " + repr(s))
    return s


def schedule(arm):
    s = arm_spec(arm).get("lr_schedule", "constant")
    require(s in SCHEDULES, "unregistered LR schedule " + repr(s))
    return s


def replayed(arm):
    spec = arm_spec(arm)
    require(spec.get("replay_of") is not None and spec.get("phases") == ["preflight"], "not a registered replay: " + arm)
    return spec["replay_of"]


# --------------------------------------------------------------------------- swap cells
def trunk_routes(trunk):
    """The routes a trunk was trained with: its arm's router, or "label" for a label-route oracle."""
    if trunk in oracles():
        return "label"
    return arm_spec(trunk)["router"]


def trunk_checkpoint(trunk, step):
    """The step checkpoint of a trunk (any network with a verified formal training), SHA-checked against its receipt."""
    require(int(step) in STEPS, f"unregistered step {step}")
    receipt = read(OUT / "receipts" / f"formal_{trunk}.json")
    require(receipt["status"] == "PASS" and receipt["arm"] == trunk and receipt["phase"] == "formal",
            "trunk has no verified formal training: " + trunk)
    run = run_name(trunk)
    checkpoint = OUT / "runs" / run / f"{run}-step={int(step)}.ckpt"
    require(str(receipt["output"]) == recorded(checkpoint.parent), "trunk run directory differs from its receipt")
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(int(step))],
            "trunk checkpoint changed since its receipt")
    return checkpoint


def cell_spec(arm):
    """A zero-training cell (a trunk with another router): round 7 names the trunk `trunk`, rounds >= 9 `pair_trunk`."""
    spec = arm_spec(arm)
    require(("trunk" in spec and spec["round"] == 7) or ("pair_trunk" in spec and spec["round"] >= 9),
            "not a swap cell: " + arm)
    trunk = trunk_of(spec)
    require(spec["router"] != trunk_routes(trunk), "a cell pairs its trunk with another router")
    require(spec["assoc"] == "pfa" and float(spec["onhand_tau"]) == 0.5 and spec["dz"] is True and
            spec.get("fast") is True and not spec.get("bank"), "cells are PFA, tau 0.5, dz, latency path, no bank")
    return spec


def cell_of(net):
    """The arm (or oracle) of a network name `<arm>` or `<arm>_5000` / `<arm>_5500` (the late-step readout names)."""
    known = registry()["arms"]
    if net in known or net in oracles():
        return net
    head, _, step = net.rpartition("_")
    require(step in ("5000", "5500") and (head in known or head in oracles()), "unregistered network " + net)
    return head


def net_spec(net):
    """(arm, step, family, spec) of a network name (`<net>_5000` / `_5500` = the late-step readout); oracles included."""
    arm = cell_of(net)
    step = int(net[len(arm) + 1:]) if net != arm else 6000
    if arm in oracles():
        return arm, step, "oracle", oracle_spec(arm)
    return arm, step, family(arm), arm_spec(arm)


def validate():
    """Cross-field checks of every registered arm / router (raises RuntimeError on the first inconsistency)."""
    reg = registry()
    for name, r in reg["routers"].items():
        require(r["kind"] in KINDS, f"router {name}: unregistered kind {r['kind']!r}")
        if "anneal_of" in r:
            require(r["anneal_of"] in reg["routers"] and int(r["anneal_steps"]) > 0, f"router {name}: anneal")
        if "decode_of" in r:
            require(r["decode_of"] in reg["routers"], f"router {name}: decode_of")
    for name, a in reg["arms"].items():
        require(a["assoc"] in ASSOCS, f"arm {name}: unregistered association {a['assoc']!r}")
        require(a["assoc"] != "spfa" or a.get("fast"), f"arm {name}: spfa runs on the latency path only")
        sampler(name)
        schedule(name)
        if a.get("sampler") == "dedicated":
            require(a.get("repeat_of") in reg["arms"], f"arm {name}: a dedicated stream needs repeat_of")
        if "schedule_of" in a:
            require(a["schedule_of"] in reg["arms"], f"arm {name}: schedule_of")
        if a.get("conditional") is not None:
            require(a["conditional"] == "select_r4", f"arm {name}: unknown condition {a['conditional']!r}")
        if a.get("bank"):
            require(a["bank"]["from"] in reg["arms"], f"arm {name}: bank source")
        if a.get("replay_of") is not None:
            require(a["replay_of"] in reg["arms"] and a.get("phases") == ["preflight"], f"arm {name}: replay")
        if trunk_of(a) is not None:
            require(trunk_of(a) in reg["arms"] or trunk_of(a) in oracles(), f"arm {name}: unregistered trunk")
    return True
