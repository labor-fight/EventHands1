"""T1A-UB round 5 (tools/t1aub/final.py, tools/t1aub/r5.py): CPU only, nothing written."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import final as F  # noqa: E402
from tools.t1aub import r5  # noqa: E402


def _jobs(p):
    native = C.read
    C.read = lambda path: p if str(path).endswith("final_plan.json") else native(path)
    try:
        with F._patched(p):
            return F.jobs("b5")
    finally:
        C.read = native


def test_every_composition_generates_configs():
    counts = F.check()
    assert len(counts) == 15 + 4 and all(n >= 5 for n in counts.values())


def test_single_mechanism_reuses_round4_trunk():
    p = F.plan(["dz"], a_ok=True)
    assert p["registry"]["cells"]["x5l"]["trunk"] == "b4dz" and "x5" not in p["registry"]["arms"]
    rep = p["registry"]["arms"]["x5rep"]
    assert rep["dz"] is True and rep["sampler"] == "dedicated" and rep["eg"] is False
    assert rep["readout"] == {"slot_dim": 20, "hidden": 256} and rep["route"]["router"] == "t0p"
    assert p["registry"]["arms"]["a5"]["dz"] is True and p["registry"]["arms"]["a5"]["readout"]["slot_dim"] == 16
    assert p["registry5"]["cells"]["x5g"] == {"trunk": "b4dz", "router": "gt", "note": p["registry5"]["cells"]["x5g"]["note"]}
    assert "a5" not in F.plan(["dz"], a_ok=False)["registry"]["arms"]


def test_oracle_config_and_jobs():
    p = F.plan(["cos", "n2", "dz"], a_ok=True)
    texts = {q.name: t for q, t in F.config_texts(p)}
    g = texts["t1aub_cell_x5g.yaml"]
    assert "ROUTER: gt" in g and "XAG_SOURCE_ROOT" in g and "ORACLE: true" in g and "LR_SCHEDULE" not in g
    assert "LR_SCHEDULE: cosine" in texts["t1aub_x5_s3407.yaml"] and "LR_SCHEDULE" not in texts["t1aub_x5_s3407_preflight.yaml"]
    js = _jobs(p)
    keys = [j["key"] for j in js]
    assert len(keys) == len(set(keys))
    for j in js:
        if j["kind"] == "pair":
            arm = j["command"][j["command"].index("--arm") + 1]
            assert ("tools/t1aub/train2.py" in j["command"]) == (arm in ("x5", "x5rep"))
        if j["kind"] in ("gpu", "exclusive"):
            tool = "tools/t1aub/breakdown7.py" if ":bd:" in j["key"] else "tools/t1aub/evaluate7.py"
            assert tool in j["command"]
    rows = [k for k in keys if ":row:" in k]
    assert rows[:2] == ["ub:r5:row:b5:t1aref_b5", "ub:r5:row:b5:t1arp_b5"] and not any("x5g" in k for k in rows)
    with F._patched(p):
        assert r5.is_r5("x5g") and r5.family("x5g") == "cell" and r5.cell_spec("x5g")["router"] == "gt"


def _final_case(tmp_path, monkeypatch, rows, Cs, lat_x=10.0):
    """decide_final.main on synthetic readings: rows {net: (local, global, ra)}, Cs {net: C}; T1a latency 15 ms."""
    import json
    from tools.t1aub import decide as D
    from tools.t1aub import decide_final as DF
    out = tmp_path / "out"
    for net, block in (("t1arep", "b4"), ("x5l", "b5"), ("x5d", "b5")):
        (out / "evaluations" / net / "row_step6000").mkdir(parents=True)
        (out / "evaluations" / net / "row_step6000/report.json").write_text(json.dumps({"block": block}))
    plan = {"x": {"legacy_trunk": "b4dz"}, "registry5": {"cells": {"x5g": {}}}}
    (out / "final_plan.json").write_text(json.dumps(plan))
    (out / "summary_r1.json").write_text(json.dumps({"block": "b1", "networks": {
        "t0ps16_srdwa": {"row": {"ra": 14.0}, "C": 0.77, "budget_ok": True},
        "gt": {"row": {"ra": 8.5}, "C": 0.5, "budget_ok": True}}}))
    (out / "summary_r2.json").write_text(json.dumps({"block": "b2", "cells": {
        "a_pt2": {"row": {"ra": 13.5}, "C": 0.79, "budget_ok": True}}}))
    (out / "summary_r4.json").write_text(json.dumps({"block": "b4", "mechanisms": {
        "dz": {"network": "b4dz_srdwa", "row": {"ra": 12.0}, "C": 0.85, "budget_ok": True}}}))
    monkeypatch.setattr(C, "OUT", out)
    monkeypatch.setattr(C, "verify_protocol", lambda: {})
    monkeypatch.setattr(r5, "patch_common", lambda: None)
    native_spec = C.arm_spec
    monkeypatch.setattr(C, "arm_spec", lambda a: {"route": {"router": "t0p"}} if a == "b4dz" else native_spec(a))

    def row(net, block):
        name = "t1a" if net == "t1a" else net
        lo, gl, ra = rows[name]
        m = {"local": {"mpjpe_ra_mm": lo, "mpvpe_ra_mm": lo}, "global": {"mpjpe_ra_mm": gl, "mpvpe_ra_mm": gl},
             "overall": {"mpjpe_ra_mm": ra}}
        lat = 15.0 if name in ("t1a", "t1arep") else lat_x
        return {"two_seed_mean": m}, {"params_total": 491000, "macs_total": 4.4e8, "latency_ms": lat}
    e1 = {s: {"local": 15.0, "global": 8.0} for s in C.STEPS}
    monkeypatch.setattr(D, "main_row", row)
    monkeypatch.setattr(D, "zgz", lambda n: {"e1": e1, "e2": e1[6000], "e3": e1[6000]})
    monkeypatch.setattr(D, "insample", lambda n: {s: {"a": 9.0} for s in C.STEPS})
    monkeypatch.setattr(D, "fixed_point", lambda n: {"C": Cs.get(n, 0.7), "gt": {"ra1": 4, "ra20": 9}, "loop": {"ra1": 9, "ra20": 9}})
    monkeypatch.setattr(D, "single_step", lambda n: {"tl_e3": 5.0, "zl_e3": 9.0})
    monkeypatch.setattr(D, "retention", lambda n: {"T10": 0.3, "R5": 0.3, "Fs": 0.3})
    DF.main(["--x-legacy", "x5l", "--x-dedicated", "x5d", "--block", "b5"])
    return json.loads((out / "summary_final.json").read_text())


def test_decide_final_reproduced(tmp_path, monkeypatch):
    s = _final_case(tmp_path, monkeypatch, {"t1a": (20.6, 9.9, 14.9), "t1arep": (21.0, 10.0, 15.2),
                                            "x5l": (15.0, 8.5, 11.5), "x5d": (15.5, 8.6, 11.8)}, {"x5l": 0.76, "x5d": 0.78})
    assert s["verdict"] == "REPRODUCED" and s["direction"] == "X_better"
    assert s["best_deployable"]["network"] == "x5l" and s["streams"]["dedicated"]["reference"] == "t1arep"
    assert abs(s["streams"]["legacy_none"]["delta_row"]["local"] - 5.6) < 1e-9 and "router_accuracy" in s["gap"]


def test_decide_final_directional_and_not(tmp_path, monkeypatch):
    s = _final_case(tmp_path, monkeypatch, {"t1a": (20.6, 9.9, 14.9), "t1arep": (21.0, 10.0, 15.2),
                                            "x5l": (15.0, 8.5, 11.5), "x5d": (19.0, 9.5, 14.0)}, {})
    assert s["verdict"] == "DIRECTIONAL" and s["direction"] == "X_better"


def test_decide_final_opposite_and_over_budget(tmp_path, monkeypatch):
    s = _final_case(tmp_path, monkeypatch, {"t1a": (20.6, 9.9, 14.9), "t1arep": (21.0, 10.0, 15.2),
                                            "x5l": (15.0, 8.5, 11.5), "x5d": (22.0, 10.5, 16.0)}, {}, lat_x=16.0)
    assert s["verdict"] == "NOT_REPRODUCED" and s["direction"] == "mixed"
    best = s["best_deployable"]
    assert best["network"] == "a_pt2" and "gt" not in best["pool"]
