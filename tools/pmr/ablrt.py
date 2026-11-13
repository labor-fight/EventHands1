"""ABL runtime helpers (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111) shared by the thin wrappers train11 / run11 /
train_rol6 / run_rol6 / run_forms6: the router block per arm (r_s2d, an RDR router behind the "pfr" kind, or none = S45's
ray association), the engine loader for an arm without a router, and the two acceptance procedures of tools/pmr/run.py and
tools/pmr/run_rol2.py with their router check generalised to those three blocks (every other check unchanged).
"""
from __future__ import annotations

from pathlib import Path

from tools.pmr.common import OUT, now, read, require, save, sha


def safe_sha(path):
    return None if path is None else sha(path)


def install_routes(p):
    """The arm's router block: r_s2d or an RDR router (PFA, tau 0.5), or none (S45's own ray association)."""
    if p["ROUTER_KIND"] == "none":
        require(p["ROUTER"] == "none" and p["ASSOC"] == "ray" and p["ROUTER_CHECKPOINT"] is None, "ray-association block")
        return {"router": "none", "assoc": "ray", "sample_sources": "none (MODEL.XAG_ASSOC base)"}
    require(p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and (p["ROUTER"] == "r_s2d" or p["ROUTER"].startswith("rdr_")),
            "ABL router block")
    from semkine import pfo_router as PR
    from semkine import rdr_hooks as RH
    from semkine import s45sm as SM
    RH.enable()
    PR.register()
    state = SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    return {**state, "sample_sources": "semkine.s45sm.predicted_sources"}


def block_ok(p):
    if p["ROUTER_KIND"] == "none":
        return p["ROUTER"] == "none" and p["ASSOC"] == "ray"
    return p["ROUTER_KIND"] == "pfr" and p["ASSOC"] == "pfa" and (p["ROUTER"] == "r_s2d" or p["ROUTER"].startswith("rdr_"))


def router_ok(launch, cfg):
    p, r = cfg["PFO"], launch["routing"]
    if p["ROUTER_KIND"] == "none":
        return r.get("router") == "none" and r.get("assoc") == "ray" and launch["router_checkpoint_sha256"] is None
    return (r["router"] == "pfr" and r["assoc"] == "pfa" and float(r["onhand_tau"]) == 0.5
            and launch["router_checkpoint_sha256"] == sha(p["ROUTER_CHECKPOINT"]))


def patch_loader():
    """tools/pfo/nets.py `load` for an arm whose evaluation config has no router (S45SM.ROUTER_KIND none): the model with
    S45's ray association, no routing hook; every other arm goes to the native loader."""
    from tools.pfo import nets as N
    if getattr(N.load, "_abl", False):
        return
    native = N.load

    def load(name, route=None, tau=None, device=None, spec=None):
        import copy
        import torch
        from config import load_config
        s = spec or N.NETS[name]
        cfg = copy.deepcopy(load_config(str(s["config"])))
        if cfg.get("S45SM", {}).get("ROUTER_KIND") != "none":
            return native(name, route=route, tau=tau, device=device, spec=spec)
        require(route in (None, "router") and tau is None, "the ray-association arm has no test-time routes")
        require(str(cfg["MODEL"].get("XAG_ASSOC")) == "base", "ray association = MODEL.XAG_ASSOC base")
        from model import MNISTModel
        device = device or torch.device("cuda")
        cfg["DATA"].pop("XAG_SOURCE_ROOT", None)
        identity = {"net": name, "kind": "ray", "checkpoint": str(s["checkpoint"]), "config": str(s["config"]),
                    "routing": {"router": "none", "assoc": "ray"}, "route": "ray"}
        model = MNISTModel.load_from_checkpoint(str(s["checkpoint"]), cfg=cfg, map_location=device, strict=True)
        return N.Net(name, "ray", model.to(device).eval(), cfg, device, None, None, identity)
    load._abl = True
    N.load = load


def accept_base(arm, phase, seed):
    """tools/pmr/run.py `accept` over the union registry with the generalised router check."""
    from tools.pmr import run as R
    A = R.A
    design, variant = A.parse(arm)
    require(phase in R.GRID, "phase")
    name = A.run_name(arm, seed, phase)
    out = OUT / "receipts" / f"{name}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(arm, seed, phase)
    launch, meta, det = read(d / "pmr_launch.json"), read(d / "training_metadata.json"), read(d / "pmr_det.json")
    cfg = A.build_config(arm, seed, phase)
    mine, b2 = R.stream_rows(d), R.stream_rows(R.b2_run(phase, seed))
    ok = {"steps": all(R.checkpoint(arm, seed, st, phase).exists() for st in R.GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr_schedule": meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              arm, seed, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": router_ok(launch, cfg),
          "trunk": launch["trunk"] == {"design": design, "variant": variant, "module": A.DESIGNS[design][0]},
          "data_stream_equals_b2": mine == b2}
    receipt = {"run": name, "arm": arm, "seed": seed, "phase": phase, "checks": ok, "b2_run": str(R.b2_run(phase, seed)),
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(R.checkpoint(arm, seed, st, phase)) for st in R.GRID[phase]
                                             if R.checkpoint(arm, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


def accept_rol(name, seed, phase):
    """tools/pmr/run_rol2.py `accept` with the generalised router check."""
    from tools.pmr import arms_any2 as PA
    from tools.pmr import arms_rol2 as A
    from tools.pmr import run_rol2 as R
    base, rol = A.parse(name)
    require(phase in R.GRID, "phase")
    rn = A.run_name(name, seed, phase)
    out = OUT / "receipts" / f"{rn}.json"
    require(not out.exists(), "accepted once")
    d = A.run_dir(name, seed, phase)
    launch, meta, det = read(d / "pmr_launch.json"), read(d / "training_metadata.json"), read(d / "pmr_det.json")
    cfg = A.build_config(name, seed, phase)
    starts = [read(d / f"rol_start_check_rank{r}.json") for r in (0, 1)]
    timing = [read(d / f"rol_timing_rank{r}.json") for r in (0, 1)]
    streams = [read(d / f"rollout_stream_rank{r}.json") for r in (0, 1)]
    s = A.ROL[rol]
    design, variant = PA.parse(base)
    ck, ck_sha = A.init_checkpoint(base, seed), A.init_sha(base, seed)
    ok = {"steps": all(R.checkpoint(name, seed, st, phase).exists() for st in R.GRID[phase]),
          "seed": meta["seed"] == seed and launch["pfo"]["SEED"] == seed,
          "max_steps": meta["max_steps"] == cfg["TRAIN"]["MAX_STEPS"],
          "lr": meta["lr"] == cfg["TRAIN"]["LR"] and meta["lr_schedule"] == "cosine",
          "devices": meta["devices"] == 2 and meta["batch_size_per_gpu"] == 512,
          "config": launch["config_sha256"] == sha(launch["config"]) and Path(launch["config"]) == A.config_path(
              name, seed, phase).resolve(),
          "deterministic": bool(det["deterministic_algorithms"]) and det["trainer_deterministic"],
          "router": router_ok(launch, cfg),
          "trunk": launch["trunk"] == {"design": design, "variant": variant, "module": PA.DESIGNS[design][0]},
          "rollout": launch["rol"] == cfg["ROL"] and launch["rol"]["K"] == s["k"] and launch["rol"]["GRAD"] == s["grad"],
          "init_from_base": launch["init_from"] == ck and launch["init_from_sha256"] == ck_sha,
          "start_check": all(x["delta_bitwise_equal"] and x["init_equal"] and x["autocast"] and x["global_step"] == 0
                             for x in starts),
          "stream_recorded": all(len(x) == 2 and all(r["packets"] == s["k"] for r in x) for x in streams),
          "memory": max(x["max_reserved_gib"] for x in timing) <= R.MEM_MAX_GIB}
    receipt = {"run": rn, "arm": name, "seed": seed, "phase": phase, "checks": ok, "start_checks": starts,
               "peak_reserved_gib": [x["max_reserved_gib"] for x in timing],
               "status": "PASS" if all(ok.values()) else "FAIL", "accepted_at_utc": now(),
               "checkpoint_sha256_by_step": {str(st): sha(R.checkpoint(name, seed, st, phase)) for st in R.GRID[phase]
                                             if R.checkpoint(name, seed, st, phase).exists()}}
    save(out, receipt, exclusive=True)
    print(receipt["status"], ok, flush=True)
    require(receipt["status"] == "PASS", "acceptance failed")


__all__ = ["accept_base", "accept_rol", "block_ok", "install_routes", "patch_loader", "router_ok", "safe_sha"]
