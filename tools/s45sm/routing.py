"""Router-kind registration, routing hooks, network resolution and swap receipts.

No monkeypatching of this package's functions and no `S45SM_ENTRY`: every entry reads the registry (`net_spec`) and
resolves what it evaluates through `resolve`. A zero-training cell resolves to its trunk's checkpoint and to its own
frozen config.

Router kinds (model side, `semkine/s45sm_{srp,dw,hier}.py`): t0p and sr are native; srp, dw and the part-first kinds
srph / dwh / t0ph are registered here, in this order (each wraps what is installed; each is idempotent). A network
registers only the kinds it needs, except a cell, which registers all of them as the swap wrappers always did (the
latency path of a cell then goes through the same wrapper calls as before).
"""
from __future__ import annotations

from pathlib import Path

from .configs import expected_state
from .paths import OUT, now, read, recorded, require, require_one_gpu, save, sha  # noqa: F401
from .protocol import config_frozen
from .registry import (arm_spec, config_path, family, net_spec, oracles, output_dir, router_ready, router_spec, run_name,
                       trunk_checkpoint, trunk_of, trunk_routes, bank_dir, bank_ready)

ALL_KINDS = ("srp", "dw", "hier")
NEEDS = {"t0p": (), "sr": (), "srp": ("srp",), "dw": ("dw",), "srph": ("srp", "hier"), "dwh": ("dw", "hier"),
         "t0ph": ("hier",)}
ENTRY_LABEL = {"evaluate": "evaluate2", "readouts": "readouts2", "row": "evaluate_fast"}


def register_kinds(kinds=ALL_KINDS):
    """Register the named router kinds in the frozen hooks, in the fixed order srp -> dw -> hier; idempotent."""
    from semkine import s45sm_dw as DW
    from semkine import s45sm_hier as H
    from semkine import s45sm_srp as SRP
    for kind, module in (("srp", SRP), ("dw", DW), ("hier", H)):
        if kind in kinds:
            module.register()


def unregister_kinds():
    from semkine import s45sm_dw as DW
    from semkine import s45sm_hier as H
    from semkine import s45sm_srp as SRP
    H.unregister()
    DW.unregister()
    SRP.unregister()


def kinds_for(router_kind, cell=False):
    require(router_kind in NEEDS, "unregistered router kind " + repr(router_kind))
    return ALL_KINDS if cell else NEEDS[router_kind]


def register_for_net(net):
    """Register what the network named `net` (arm, cell, `<arm>_5000`...) needs; returns the registered kinds."""
    if net.startswith("s45ref"):
        return ()
    arm, _, fam, spec = net_spec(net)
    if fam == "oracle":
        return ()
    kinds = kinds_for(router_spec(spec["router"])["kind"], cell=fam == "cell")
    register_kinds(kinds)
    return kinds


# --------------------------------------------------------------------------- hooks
def hook_module(s, force_fast=False):
    """The routing module of an arm: s45sm_soft for spfa, s45sm_fast for fast arms (or `force_fast`), else s45sm."""
    if s["ASSOC"] == "spfa":
        require(bool(s.get("FAST")), "spfa runs on the latency path only")
        from semkine import s45sm_soft as M
    elif s.get("FAST") or force_fast:
        from semkine import s45sm_fast as M
    else:
        from semkine import s45sm as M
    return M


def installed_routing():
    """The installed routing (any of the three modules) with its `fast` flag, or None."""
    from semkine import s45sm, s45sm_fast, s45sm_soft
    soft = s45sm_soft.installed()
    if soft is not None:
        return {**soft, "fast": True}
    state = s45sm.installed()
    fast = s45sm_fast.installed()
    return None if state is None else {**state, "fast": bool(fast and fast["fast"])}


def install_hook(cfg, force_fast=False):
    s = cfg["S45SM"]
    require(recorded(router_ready(s["ROUTER"])) == s["ROUTER_CHECKPOINT"], "router checkpoint path differs from the config")
    M = hook_module(s, force_fast)
    state = M.install(s["ROUTER_KIND"], s["ROUTER_CHECKPOINT"], s["ASSOC"], float(s["ONHAND_TAU"]))
    require({k: state[k] for k in ("router", "checkpoint", "assoc", "onhand_tau")} == expected_state(cfg),
            "installed routing differs from the config")
    return state


def install_eval_hook(cfg, force_fast=False):
    """Training hook + the evaluator attaches no ground-truth XAG source (the arms read none; as S45-RT T1p)."""
    state = install_hook(cfg, force_fast)
    from semkine import eval_track as ET
    ET.attach_xag_sources = lambda batch, *args, **kwargs: batch
    return state


def install_train_hook(cfg):
    """The bank prior sampler (training processes only); returns its identity or None."""
    b = cfg["S45SM"].get("BANK")
    if not b:
        return None
    require(recorded(bank_ready(b["FROM"])) == b["ROOT"], "bank root differs from the config")
    from semkine import s45tr_prior as P
    P.install_n4(b["ROOT"], float(b["P"]))
    return {"from": b["FROM"], "root": b["ROOT"], "p": float(b["P"])}


def install_oracle(cfg):
    from semkine import s45sm_oracle as O
    require(cfg["S45SM"]["ROUTER"] == "gt" and cfg["S45SM"]["ASSOC"] == "pfa", "not an oracle config")
    return O.install()


def is_oracle_config(cfg):
    return cfg["S45SM"]["ROUTER"] == "gt"


def install_for_config(cfg, force_fast=False, evaluation=True):
    """Register the router kind of the config and install its routing: the oracle's PFA association, or the router's.
    The oracle keeps the label sources in the batches (evaluation attaches them from the config); a router arm's
    evaluator attaches none."""
    if is_oracle_config(cfg):
        return install_oracle(cfg)
    kind = cfg["S45SM"]["ROUTER_KIND"]
    register_kinds(NEEDS[kind])
    return install_eval_hook(cfg, force_fast) if evaluation else install_hook(cfg, force_fast)


# --------------------------------------------------------------------------- resolution
def resolve(name, step):
    """(checkpoint, config) of network `name` (an arm, a cell, or an oracle) at `step`, checked against receipts."""
    if name in oracles():
        receipt = read(OUT / "receipts" / f"formal_{name}.json")
        require(receipt["status"] == "PASS", "oracle arm has no verified formal training")
        checkpoint = output_dir(name, "formal") / f"{run_name(name)}-step={step}.ckpt"
        require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
        return checkpoint, config_path(name, "formal")
    spec = arm_spec(name)
    if trunk_of(spec) is not None:
        config = config_path(name, "formal")
        config_frozen(config)
        return trunk_checkpoint(trunk_of(spec), step), config
    receipt = read(OUT / "receipts" / f"formal_{name}.json")
    require(receipt["status"] == "PASS", "arm has no verified formal training")
    checkpoint = output_dir(name, "formal") / f"{run_name(name)}-step={step}.ckpt"
    require(sha(checkpoint) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since its receipt")
    return checkpoint, config_path(name, "formal")


def swap_receipt_path(net, entry, what):
    return OUT / "swap_receipts" / f"{net}_{ENTRY_LABEL.get(entry, entry)}_{what}.json"


def swap_receipt(net, entry, step):
    """The content of a cell's swap receipt (without the time stamp): trunk, its routes, router, step, checkpoint, config."""
    arm = net_spec(net)[0]
    spec = arm_spec(arm)
    trunk = trunk_of(spec)
    config = config_path(arm, "formal")
    return {"net": net, "cell": arm, "entry": ENTRY_LABEL.get(entry, entry), "trunk": trunk,
            "trunk_routes": trunk_routes(trunk), "router": spec["router"], "step": step,
            "checkpoint_sha256": sha(trunk_checkpoint(trunk, step)), "config_sha256": sha(config)}


def write_swap_receipt(net, entry, what, step):
    """After a cell's evaluation / readout / row: swap_receipts/<net>_<entry>_<what>.json (exclusive)."""
    path = swap_receipt_path(net, entry, what)
    save(path, {**swap_receipt(net, entry, step), "completed_at_utc": now()}, exclusive=True)
    return path


def cell_net(net):
    """True when `net` names a zero-training cell (or its late-step readout name)."""
    return not net.startswith("s45ref") and net_spec(net)[2] == "cell"
