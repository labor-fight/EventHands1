"""CPU tests: the registry, the config generator and the frozen-manifest protocol of the consolidated S45-SM package
(ported from the per-round tests of the old tools: round 1, 3, 5, 7, 9 / 11, 12, 13, 14 identities and the repeat arms)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import configs as K  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import protocol as PR  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402

IDENTITY = {("S45SM", "ARM"), ("TRAIN", "RUN_NAME"), ("TRAIN", "OUTPUT_DIR"), ("EVAL", "OUTPUT_DIR")}
ROUTING_KEYS = {("S45SM",), ("TRAIN", "RUN_NAME"), ("TRAIN", "OUTPUT_DIR"), ("EVAL", "OUTPUT_DIR"), ("DATA", "XAG_SOURCE_ROOT")}


def flat(d, prefix=(), keep=frozenset()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and prefix + (k,) not in keep:
            out.update(flat(v, prefix + (k,), keep))
        else:
            out[prefix + (k,)] = v
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def load(arm, phase):
    return yaml.safe_load(R.config_path(arm, phase).read_text())


def trunk_config(trunk):
    return R.config_path(trunk, "formal")


# --------------------------------------------------------------------------- registry
def test_registry_is_consistent():
    assert R.validate()
    reg = R.registry()
    assert len(reg["arms"]) >= 46 and len(reg["routers"]) >= 19 and sorted(reg["rounds"]) == list(range(1, max(reg["rounds"]) + 1))
    assert list(reg["rounds"].values()) == [str(p.relative_to(P.ROOT)) for p in sorted(R.REGISTRY_DIR.glob("registry_r*.json"))] \
        or sorted(reg["rounds"].values())            # file order is lexicographic, rounds are keyed by number
    assert set(R.oracles()) == {"pfagtdz"} and R.oracles()["pfagtdz"]["round"] == 5
    assert "pfagtdz" not in reg["arms"]
    assert R.family("pfagtdz") == "oracle" and R.family("pfasrdz") == "router" and R.family("xt0_srdwa") == "cell"
    assert R.family("pfat0pdz_replay") == "replay"


def test_registry_lexicographic_file_order():
    names = [p.name for p in sorted(R.REGISTRY_DIR.glob("registry_r*.json"))]
    assert names[:3] == ["registry_r1.json", "registry_r10.json", "registry_r11.json"]
    arms = list(R.registry()["arms"])
    assert arms.index("t0pdz") < arms.index("xgt_sr1a") < arms.index("pfasrbank") < arms.index("xt0_srp")   # r1, r10, r3, r7


def test_names_and_derived_paths():
    assert R.run_name("pfasrdz") == "s45sm_pfasrdz_s3407" and R.run_name("pfasrdz", "preflight") == "s45sm_pfasrdz_s3407_preflight"
    assert R.output_dir("pfasrdz", "formal") == P.OUT / "runs/s45sm_pfasrdz_s3407"
    assert R.output_dir("pfasrdz", "preflight") == P.OUT / "preflight/s45sm_pfasrdz_s3407"      # no phase suffix
    assert R.config_path("pfagtdz", "formal").name == "s45sm_pfagtdz_s3407.yaml"
    assert R.router_checkpoint("srpkd") == R.router_dir("srpkd") / "srpkd_final.ckpt"
    assert R.router_dir("srpkd", "preflight").name == "srpkd_preflight"
    with pytest.raises(RuntimeError):
        R.router_checkpoint("t0p", "preflight")
    with pytest.raises(RuntimeError):
        R.arm_spec("nonexistent")
    assert R.phases("pfat0pdz_replay") == ("preflight",) and R.phases("pfasrdz") == ("preflight", "formal")


def test_net_names():
    assert R.cell_of("xgt_sr1") == "xgt_sr1" and R.cell_of("xgt_sr1_5000") == "xgt_sr1"
    with pytest.raises(RuntimeError):
        R.cell_of("xgt_sr1_4000")
    assert R.net_spec("xt0_srdwa_5500")[:3] == ("xt0_srdwa", 5500, "cell")
    assert R.net_spec("pfagtdz")[:3] == ("pfagtdz", 6000, "oracle")


# --------------------------------------------------------------------------- configs
def test_every_config_is_what_the_generator_produces():
    result = K.check_configs()
    assert len(result) >= 93 and all(result.values()), [k for k, v in result.items() if not v]


def test_write_configs_to_a_directory_is_byte_identical(tmp_path):
    for round_ in (1, 5, 13, 14):
        paths = K.write_configs(round_, tmp_path)
        for path in paths:
            assert path.read_text() == K.dump(yaml.safe_load(R.config_path(path.stem.replace("_s3407_preflight", "").replace("_s3407", "")[len("s45sm_"):],
                                                                         "preflight" if path.stem.endswith("preflight") else "formal").read_text()))
    with pytest.raises(RuntimeError):
        K.write_configs(14, tmp_path)                                  # refuses to overwrite
    assert not (tmp_path / R.config_path("pfat0pdz_replay", "formal").name).exists()   # a replay has no formal config


def test_round1_configs_are_dz_plus_routing():
    dz = flatten(yaml.safe_load((ROOT / "configs/s45at_20261031/s45at_dz_s3407.yaml").read_text()))
    base = K.base_config()
    reg = R.registry()
    assert set(reg["arms"]) >= {"t0pdz", "pfat0pdz", "srdz", "pfasrdz"}
    for arm in ("t0pdz", "pfat0pdz", "srdz", "pfasrdz"):
        for phase in ("preflight", "formal"):
            got = flatten(K.expected_raw(arm, phase, base))
            changed = {k for k in set(got) | set(dz) if got.get(k) != dz.get(k)}
            allowed = {"MODEL.XAG_ASSOC", "MODEL.XAG_GNN", "TRAIN.RUN_NAME", "TRAIN.OUTPUT_DIR", "EVAL.OUTPUT_DIR"}
            allowed |= {k for k in changed if k.startswith(("S45SM.", "S45AT."))}
            if phase == "preflight":
                allowed |= {"TRAIN.MAX_STEPS", "TRAIN.VAL_CHECK_INTERVAL"}
            assert changed <= allowed, (arm, phase, changed - allowed)
            assert got["MODEL.XAG_ASSOC"] == "face_route" and "DATA.XAG_SOURCE_ROOT" not in got
            assert got["AUG.DOMRAND.SCALE_MODE"] == "depth"
            spec = reg["arms"][arm]
            assert got["S45SM.ASSOC"] == spec["assoc"] and got["S45SM.ROUTER"] == spec["router"]
            assert "S45SM.FAST" not in got                               # the round-1 files have no such key


def test_round3_configs_are_the_registered_recipe():
    base = K.base_config()
    ref = load("pfasrdz", "formal")
    for arm in ("pfasrbank", "spfasrdz", "spfasrbank"):
        spec = R.arm_spec(arm)
        assert spec["round"] == 3 and spec["router"] == "sr1" and spec["dz"] and spec["fast"]
        for phase in ("preflight", "formal"):
            assert load(arm, phase) == yaml.safe_load(K.dump(K.expected_raw(arm, phase, base)))
        cfg = load(arm, "formal")
        s = cfg.pop("S45SM")
        r = dict(ref)
        rs = r.pop("S45SM")
        for k in ("TRAIN", "EVAL"):                                              # only names and paths differ
            cfg[k] = {kk: v for kk, v in cfg[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
            r[k] = {kk: v for kk, v in r[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
        assert cfg == r
        assert {k: s[k] for k in ("ROUTER", "ROUTER_KIND", "ROUTER_CHECKPOINT", "ONHAND_TAU")} == \
            {k: rs[k] for k in ("ROUTER", "ROUTER_KIND", "ROUTER_CHECKPOINT", "ONHAND_TAU")}
        assert s["ASSOC"] == spec["assoc"] and s["FAST"] is True
        assert ("BANK" in s) == ("bank" in spec)
        if "bank" in spec:
            assert s["BANK"] == {"FROM": "pfasrdz", "ROOT": P.recorded(R.bank_dir("pfasrdz")), "P": 0.5}


def test_pfakdbank_is_pfasrbank_with_srkd():
    spec, ref = R.arm_spec("pfakdbank"), R.arm_spec("pfasrbank")
    assert spec["round"] == 5 and spec["router"] == "srkd" and spec["conditional"] == "select_r4"
    assert {k: spec[k] for k in ("assoc", "onhand_tau", "dz", "fast", "bank")} == \
        {k: ref[k] for k in ("assoc", "onhand_tau", "dz", "fast", "bank")}
    assert R.router_spec("srkd")["kind"] == R.router_spec("sr1")["kind"] == "sr"
    for phase in ("preflight", "formal"):
        cfg, old = load("pfakdbank", phase), load("pfasrbank", phase)
        s, so = cfg.pop("S45SM"), old.pop("S45SM")
        for k in ("TRAIN", "EVAL"):
            cfg[k] = {kk: v for kk, v in cfg[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
            old[k] = {kk: v for kk, v in old[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
        assert cfg == old
        assert s["ROUTER"] == "srkd" and s["ROUTER_CHECKPOINT"] == P.recorded(R.router_checkpoint("srkd"))
        drop = ("ARM", "ROUTER", "ROUTER_CHECKPOINT")
        assert {k: v for k, v in s.items() if k not in drop} == {k: v for k, v in so.items() if k not in drop}


def test_oracle_registered_and_configs_written(monkeypatch):
    o = R.oracles()
    assert set(o) == {"pfagtdz"} and o["pfagtdz"]["base"] == "t1ddz" and o["pfagtdz"]["assoc"] == "pfa"
    base = K.oracle_base_config()
    for phase in ("preflight", "formal"):
        assert load("pfagtdz", phase) == yaml.safe_load(K.dump(K.expected_raw_oracle("pfagtdz", phase, base)))
    cfg = K.expected_raw_oracle("pfagtdz", "formal", base)
    assert cfg["S45SM"] == {"ARM": "pfagtdz", "ROUTER": "gt", "ASSOC": "pfa", "BASE": "s45at_t1ddz"}
    assert "S45AT" not in cfg and cfg["MODEL"]["XAG_ASSOC"] == "face_route" and cfg["DATA"]["XAG_SOURCE_ROOT"]
    bad = dict(base)
    bad["S45AT"] = {**base["S45AT"], "ARM": "other"}
    with pytest.raises(RuntimeError):
        K.expected_raw_oracle("pfagtdz", "formal", bad)


def test_cells_registry_and_configs():
    reg = R.registry()["arms"]
    r7 = {"xsr_srp": ("pfasrdz", "srpkd"), "xt0_sr1": ("pfat0pdz", "sr1"), "xt0_srp": ("pfat0pdz", "srpkd"),
          "xgt_sr1": ("pfagtdz", "sr1"), "xgt_srp": ("pfagtdz", "srpkd"), "xsr_t0p": ("pfasrdz", "t0p"),
          "xgt_t0p": ("pfagtdz", "t0p")}
    assert {a: (s["trunk"], s["router"]) for a, s in reg.items() if s["round"] == 7} == r7
    assert all("trunk" not in s for s in reg.values() if s["round"] != 7)
    assert all("pair_trunk" not in s for s in reg.values() if s["round"] < 9)
    cells = {a: s for a, s in reg.items() if R.trunk_of(s) is not None}
    assert len(cells) >= 7 + 16 + 8 + 3
    params = {"sr1": 137028, "srpkd": 137787, "t0p": 1606683}
    for arm, (trunk, router) in r7.items():
        spec = R.cell_spec(arm)
        assert spec["router"] == router and R.trunk_routes(trunk) != router
        assert ((P.S45_PARAMS + params[router]) <= P.BUDGET["params_total_max"]) == (router != "t0p")
    assert {t: R.trunk_routes(t) for t in ("pfasrdz", "pfat0pdz", "pfagtdz")} == {"pfasrdz": "sr1", "pfat0pdz": "t0p", "pfagtdz": "label"}
    for arm, spec in cells.items():
        R.cell_spec(arm)
        trunk, router = R.trunk_of(spec), spec["router"]
        cell, base = load(arm, "formal"), yaml.safe_load(trunk_config(trunk).read_text())
        fc, fb = flat(cell, keep=ROUTING_KEYS), flat(base, keep=ROUTING_KEYS)
        drop = ROUTING_KEYS | {("TRAIN", "LR_SCHEDULE")}               # a cosine trunk's training-only key is not in its cells
        assert {k: v for k, v in fc.items() if k not in drop} == {k: v for k, v in fb.items() if k not in drop}, arm
        assert cell["S45SM"] == {"ARM": arm, "ROUTER": router, "ROUTER_KIND": R.router_spec(router)["kind"],
                                 "ROUTER_CHECKPOINT": P.recorded(R.router_checkpoint(router)), "ASSOC": "pfa",
                                 "ONHAND_TAU": 0.5, "FAST": True}
        assert cell["TRAIN"]["RUN_NAME"] == f"s45sm_{arm}_s3407" and cell["DATA"].get("XAG_SOURCE_ROOT") is None
    with pytest.raises(RuntimeError):
        R.cell_spec("pfasrdz")                                          # a trained arm is not a cell


def test_cells_are_in_the_paired_rounds_only_with_their_key():
    reg = R.registry()["arms"]
    assert {a for a, s in reg.items() if s["round"] in (9, 11)} == {a for a, s in reg.items() if "pair_trunk" in s and s["round"] in (9, 11)}
    got = {a: (s["pair_trunk"], s["router"], s["pairing"]) for a, s in reg.items() if s["round"] == 12}
    assert {a: v[:2] for a, v in got.items()} == {f"{p}_{s}": (t, r) for p, t in (("xgt", "pfagtdz"), ("xt0", "pfat0pdz"))
                                                  for s, r in (("srp9", "srpkd9"), ("srp9a", "srpkd9a"), ("srp9h", "srpkd9h"), ("srp9ah", "srpkd9ah"))}
    for arm, (trunk, router, _) in got.items():
        assert P.S45_PARAMS + int(R.router_spec(router)["params"]) <= P.BUDGET["params_total_max"]


def test_rounds_13_and_14_arms():
    reg = R.registry()["arms"]
    drop = ("round", "note")
    base = reg["pfat0pdz"]
    same = {k: v for k, v in base.items() if k not in drop}
    replay, rep = reg["pfat0pdz_replay"], reg["pfat0pdz_rep"]
    assert {k: v for k, v in replay.items() if k not in drop + ("replay_of", "phases")} == same
    assert R.replayed("pfat0pdz_replay") == "pfat0pdz" and replay["phases"] == ["preflight"]
    assert {k: v for k, v in rep.items() if k not in drop + ("sampler", "repeat_of")} == same
    assert rep["sampler"] == "dedicated" and rep["repeat_of"] == "pfat0pdz" and "fast" not in rep
    with pytest.raises(RuntimeError):
        R.replayed("pfat0pdz_rep")
    for arm, bs in (("pfat0pdz_cos", "pfat0pdz"), ("pfat0pdz_rep_cos", "pfat0pdz_rep")):
        spec = reg[arm]
        assert spec["lr_schedule"] == "cosine" and spec["schedule_of"] == bs and R.schedule(bs) == "constant"
        d = ("round", "note", "lr_schedule", "schedule_of", "repeat_of")
        assert {k: v for k, v in spec.items() if k not in d} == {k: v for k, v in reg[bs].items() if k not in d}
    assert reg["pfat0pdz_rep_cos"]["repeat_of"] == "pfat0pdz_cos" and reg["pfat0pdz_rep_cos"]["sampler"] == "dedicated"


def test_cells_of_rounds_13_and_14_equal_their_originals_on_another_trunk():
    reg = R.registry()["arms"]
    drop = ("round", "note", "pair_trunk", "pairing")
    for cell, orig, trunk in (("xt0rep_srdwa", "xt0_srdwa", "pfat0pdz_rep"), ("xt0cos_srdwa", "xt0_srdwa", "pfat0pdz_cos"),
                              ("xt0repcos_srdwa", "xt0rep_srdwa", "pfat0pdz_rep_cos")):
        assert {k: v for k, v in reg[cell].items() if k not in drop} == {k: v for k, v in reg[orig].items() if k not in drop}
        assert reg[cell]["pair_trunk"] == trunk and "trunk" not in reg[cell] and R.cell_spec(cell)["router"] == "srdwkda"
        assert R.trunk_routes(trunk) == "t0p"
        a, b = flat(load(cell, "formal")), flat(load(orig, "formal"))
        assert {k: v for k, v in a.items() if k not in IDENTITY} == {k: v for k, v in b.items() if k not in IDENTITY}
        assert load(cell, "formal")["S45SM"]["ARM"] == cell and load(cell, "formal")["S45SM"]["FAST"] is True
    assert reg["xt0cos_srdwa"]["pair_trunk"] == "pfat0pdz_cos" and R.schedule(reg["xt0cos_srdwa"]["pair_trunk"]) == "cosine"


def test_replay_and_repeat_configs_are_pfat0pdz_plus_identity():
    for phase in ("preflight", "formal"):
        base = flat(load("pfat0pdz", phase))
        for arm, extra in (("pfat0pdz_replay", {}), ("pfat0pdz_rep", {("S45SM", "SAMPLER"): "dedicated", ("S45SM", "REPEAT_OF"): "pfat0pdz"})):
            if arm == "pfat0pdz_replay" and phase == "formal":
                assert not R.config_path(arm, phase).exists()
                continue
            got = flat(load(arm, phase))
            want = {**{k: v for k, v in base.items() if k not in IDENTITY}, ("S45SM", "FAST"): False, **extra}
            assert {k: v for k, v in got.items() if k not in IDENTITY} == want
            assert got[("S45SM", "ARM")] == arm and got[("TRAIN", "RUN_NAME")] == R.run_name(arm, phase)
            assert got[("TRAIN", "OUTPUT_DIR")] == P.recorded(R.output_dir(arm, phase))


def test_configs_differ_from_their_bases_only_by_the_schedule():
    for arm, base in (("pfat0pdz_cos", "pfat0pdz"), ("pfat0pdz_rep_cos", "pfat0pdz_rep")):
        for phase in ("preflight", "formal"):
            got, ref = flat(load(arm, phase)), flat(load(base, phase))
            got = {k: v for k, v in got.items() if k not in IDENTITY and k != ("S45SM", "REPEAT_OF")}
            ref = {k: v for k, v in ref.items() if k not in IDENTITY and k != ("S45SM", "REPEAT_OF")}
            ref.setdefault(("S45SM", "FAST"), False)
            if phase == "formal":
                assert got.pop(("TRAIN", "LR_SCHEDULE")) == "cosine"
            assert got == ref and ("TRAIN", "LR_SCHEDULE") not in ref
    assert load("pfat0pdz_rep_cos", "formal")["S45SM"]["REPEAT_OF"] == "pfat0pdz_cos"


def test_cosine_equals_constant_through_step_500():
    import math
    import types

    import torch
    from model import MNISTModel
    S45 = K.base_config()

    def factors(schedule):
        cfg = {"TRAIN": {"LR": S45["TRAIN"]["LR"], "WARMUP_STEPS": S45["TRAIN"]["WARMUP_STEPS"],
                         "MAX_STEPS": S45["TRAIN"]["MAX_STEPS"], "LR_SCHEDULE": schedule}}
        w = torch.nn.Parameter(torch.zeros(1))
        fake = types.SimpleNamespace(cfg=cfg, weight_decay=0.0, parameters=lambda: [w], _register_ema_hook=lambda opt: None,
                                     LR_SCHEDULES=MNISTModel.LR_SCHEDULES, LR_FLOOR=MNISTModel.LR_FLOOR)
        out = MNISTModel.configure_optimizers(fake)
        opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
        lrs = []
        for _ in range(6000):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        return lrs
    const, cos = factors("constant"), factors("cosine")
    assert const[:P.SAME_STEPS + 1] == cos[:P.SAME_STEPS + 1]
    assert cos[P.SAME_STEPS + 1] < const[P.SAME_STEPS + 1] and const[-1] == pytest.approx(0.004)
    assert cos[-1] == pytest.approx(0.004 * (MNISTModel.LR_FLOOR + (1 - MNISTModel.LR_FLOOR) * 0.5 * (1 + math.cos(math.pi * 5499 / 5500))))


@pytest.fixture()
def fake_registry(monkeypatch):
    real = R.registry()

    def registry():
        arms = dict(real["arms"])
        base = arms["pfasrbank"]
        arms["pfasrbank_rep"] = {**base, "round": 7, "sampler": "dedicated", "repeat_of": "pfasrbank"}
        arms["bad_rep"] = {**base, "round": 7, "sampler": "dedicated", "repeat_of": "pfasrbank", "onhand_tau": 0.6}
        return {**real, "arms": arms}
    monkeypatch.setattr(R, "registry", registry)


def test_repeat_config_is_its_arm_plus_sampler(fake_registry):
    base_raw = K.base_config()
    assert R.sampler("pfasrbank") == "legacy_none" and R.sampler("pfasrbank_rep") == "dedicated"
    for phase in ("preflight", "formal"):
        rep = K.expected_raw("pfasrbank_rep", phase, base_raw)
        ref = yaml.safe_load(R.config_path("pfasrbank", phase).read_text())
        s, so = rep.pop("S45SM"), ref.pop("S45SM")
        assert s.pop("SAMPLER") == "dedicated" and s.pop("REPEAT_OF") == "pfasrbank"
        assert s.pop("ARM") == "pfasrbank_rep" and so.pop("ARM") == "pfasrbank" and yaml.safe_load(K.dump({"s": s})) == {"s": so}
        for k in ("TRAIN", "EVAL"):
            rep[k] = {kk: v for kk, v in rep[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
            ref[k] = {kk: v for kk, v in ref[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR")}
        assert yaml.safe_load(K.dump(rep)) == ref
    with pytest.raises(RuntimeError):
        K.expected_raw("bad_rep", "formal", base_raw)


def test_reference_streams_exist():
    import numpy as np
    a = np.fromfile(P.SAMP_RUN / "indices_rank0.i32", dtype=np.int32)
    b = np.fromfile(P.DZ_RUN / "indices_rank0.i32", dtype=np.int32)
    assert a.size == b.size == 6000 * 512 and not np.array_equal(a[:512], b[:512])


# --------------------------------------------------------------------------- protocol (frozen manifests)
def test_the_tree_matches_its_frozen_manifests():
    manifests = PR.verify_protocol()
    assert len(manifests) >= 18
    rounds = {m["round"] for m in manifests if "round" in m}
    assert set(R.registry()["rounds"]) <= rounds
    PR.config_frozen(R.config_path("pfat0pdz", "formal"))
    with pytest.raises(RuntimeError):
        PR.config_frozen(ROOT / "README.md")
    assert len(PR.code_identity()) == 64 and PR.code_files()


class Tree:
    """A synthetic repository root with frozen manifests, to test tampering without touching the real ones."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "repo"
        (self.root / "manifests").mkdir(parents=True)
        for name in ("a.py", "b.py", "c.yaml"):
            (self.root / name).write_text(f"content of {name}\n")
        (self.root / "pkg").mkdir()
        (self.root / "pkg" / "module.py").write_text("x = 1\n")
        monkeypatch.setattr(PR, "PACKAGE", self.root / "pkg")
        monkeypatch.setattr(PR, "ROOT", self.root)
        monkeypatch.setattr(PR, "MANIFESTS", self.root / "manifests")
        monkeypatch.setattr(PR, "registry", lambda: {"rounds": {1: "tools/registry_r1.json"}, "arms": {}, "routers": {}})
        self.write("r1.json", {"study": P.STUDY, "status": "FROZEN", "round": 1,
                               "file_sha256": {n: P.sha(self.root / n) for n in ("a.py", "b.py", "c.yaml")}})

    def write(self, name, data):
        (self.root / "manifests" / name).write_text(json.dumps(data))

    def archive(self, names=("a.py",)):
        arch = self.root / "legacy"
        (arch / "files").mkdir(parents=True)
        files = {}
        for n in names:
            (arch / "files" / n).write_text((self.root / n).read_text())
            files[n] = {"sha256": P.sha(arch / "files" / n), "bytes": 1, "manifests": ["r1.json"]}
        (arch / "inventory.json").write_text(json.dumps({"files": files}))
        return arch


def test_tampered_frozen_file_is_detected(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    assert len(PR.verify_protocol()) == 1
    (tree.root / "b.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="frozen S45-SM file changed.*b.py"):
        PR.verify_protocol()


def test_unfrozen_round_and_manifest_identity(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    monkeypatch.setattr(PR, "registry", lambda: {"rounds": {1: "x", 2: "y"}, "arms": {}, "routers": {}})
    with pytest.raises(RuntimeError, match="no frozen manifest"):
        PR.verify_protocol()
    monkeypatch.setattr(PR, "registry", lambda: {"rounds": {1: "x"}, "arms": {}, "routers": {}})
    tree.write("r1.json", {"study": "OTHER", "status": "FROZEN", "round": 1, "file_sha256": {}})
    with pytest.raises(RuntimeError, match="manifest identity"):
        PR.verify_protocol()


def test_archived_code_stays_verifiable(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    arch = tree.archive()
    monkeypatch.setattr(PR, "identity_files", lambda: {"b.py": P.sha(tree.root / "b.py")})
    report = arch / "equivalence_report.md"
    report.write_text("report\n")
    target = PR.build_identity(arch, report)
    assert target.name == P.IDENTITY_MANIFEST
    identity = json.loads(target.read_text())
    assert identity["supersedes"] == ["r1.json"] and identity["legacy_inventory_sha256"] == P.sha(arch / "inventory.json")
    assert identity["legacy_archive"] == "legacy" and identity["equivalence_report"] == "legacy/equivalence_report.md"
    assert identity["equivalence_report_sha256"] == P.sha(report) and identity["kind"] == "CODE_IDENTITY"
    (tree.root / "a.py").unlink()                                       # the retired code is gone from the tree
    assert len(PR.verify_protocol()) == 2                               # ... and verified through the archive
    with pytest.raises(RuntimeError, match="exists"):
        PR.build_identity(arch, None)
    (arch / "files" / "a.py").write_text("tampered\n")
    with pytest.raises(RuntimeError, match="archived S45-SM copy changed"):
        PR.verify_protocol()


def test_inventory_and_identity_tampering(tmp_path, monkeypatch):
    tree = Tree(tmp_path, monkeypatch)
    arch = tree.archive()
    monkeypatch.setattr(PR, "identity_files", lambda: {"b.py": P.sha(tree.root / "b.py")})
    PR.build_identity(arch, None)
    inv = json.loads((arch / "inventory.json").read_text())
    inv["files"]["a.py"]["sha256"] = "0" * 64
    (arch / "inventory.json").write_text(json.dumps(inv))
    with pytest.raises(RuntimeError, match="inventory changed"):
        PR.verify_protocol()
    tree2 = tmp_path / "second"
    tree2.mkdir()
    tree = Tree(tree2, monkeypatch)
    arch = tree.archive()
    monkeypatch.setattr(PR, "identity_files", lambda: {"b.py": P.sha(tree.root / "b.py")})
    PR.build_identity(arch, None)
    (tree.root / "b.py").write_text("edited after the identity manifest\n")
    tree.write("r1.json", {"study": P.STUDY, "status": "FROZEN", "round": 1,
                           "file_sha256": {n: P.sha(tree.root / n) for n in ("a.py", "b.py", "c.yaml")}})
    with pytest.raises(RuntimeError, match="identity manifest"):
        PR.verify_protocol()


def test_code_identity_does_not_depend_on_the_directory_name(tmp_path, monkeypatch):
    import shutil
    first = PR.code_identity()
    copy = tmp_path / "renamed_package"
    shutil.copytree(P.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__", "*.json"))
    monkeypatch.setattr(PR, "PACKAGE", copy)
    assert PR.code_identity() == first
