"""CPU tests: summaries, tables and written readings of the consolidated S45-SM package (ported from the round 5 ceiling and
next-step rules, the round 7 / 11 / 12 table effects, the round 10 D1 table and the round 13 / 14 readings; new: the
`--as-of-round` arm filter and the schema levels of the summaries)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import readings as RD  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402
from tools.s45sm import summary as S  # noqa: E402
from tools.s45sm import tables as T  # noqa: E402


def fake(ra, C_, budget=True, cls="WEAK", local=None, glob=None):
    return {"main_row": {"local": local if local is not None else ra + 3, "global": glob if glob is not None else ra - 3, "ra": ra},
            "convergence": {"C": C_}, "budget_ok": budget, "class": cls}


# --------------------------------------------------------------------------- summary rules
def test_ceiling_class_and_next_step():
    gap = lambda l, g, r: {"local": l, "global": g, "ra": r}               # noqa: E731
    assert S.ceiling_class(gap(1.5, -2.0, 1.0), 0.8) == "AT"
    assert S.ceiling_class(gap(1.6, 0.0, 0.0), 0.5) == "NEAR"
    assert S.ceiling_class(gap(3.5, 2.75, 3.1), 0.5) == "NEAR"
    assert S.ceiling_class(gap(0.0, 0.0, 0.0), 0.81) == "CAPPED"
    assert S.ceiling_class(gap(3.6, 0.0, 0.0), 0.5) == "CAPPED"

    def arm(cls, ra, ok=True):
        return {"class": cls, "budget_ok": ok, "main_row": {"ra": ra}}
    assert S.next_step({"a": arm("WEAK", 12.0)}, {"o": "AT"}) == {"repeat": None, "line": "ROUTER"}
    assert S.next_step({"a": arm("WEAK", 12.0)}, {"o": "CAPPED"}) == {"repeat": None, "line": "ASSOCIATION"}
    assert S.next_step({"a": arm("NEAR_TARGET", 11.0), "b": arm("NEAR_TARGET", 10.9, False)}, {"o": "NEAR"}) == \
        {"repeat": "a", "line": "ROUTER"}
    assert S.next_step({"a": arm("NEAR_TARGET", 11.0), "b": arm("AT_TARGET", 10.0)}, {"o": "CAPPED"}) == \
        {"repeat": "b", "line": "ROBUSTNESS"}
    split = S.split({"local": 15.88, "global": 9.54, "ra": 12.48}, {"local": 12.0, "global": 6.0, "ra": 9.0})
    assert abs(split["local"]["router"] + split["local"]["association"] - split["local"]["total"]) < 1e-9


def test_classes_in_the_written_order():
    gap = lambda l, g, r: {"local": l, "global": g, "ra": r}               # noqa: E731
    R_ = lambda v: {5000: v, 5500: v, 6000: v}                              # noqa: E731
    assert S.classify_gaps(True, True, gap(1.5, 1.5, 1.5), R_(0.0)) == "AT_TARGET"
    assert S.classify_gaps(True, False, gap(1.5, 1.5, 1.5), R_(0.0)) == "NEAR_TARGET"          # over budget: not AT
    assert S.classify_gaps(True, True, gap(3.5, 2.75, 3.1), R_(0.0)) == "NEAR_TARGET"
    assert S.classify_gaps(False, True, gap(0.0, 0.0, 0.0), R_(0.7)) == "WEAK"                 # C > 0.8: not NEAR / MECHANISM
    assert S.classify_gaps(True, True, gap(9.0, 9.0, 9.0), R_(0.6)) == "MECHANISM"
    assert S.classify_gaps(True, True, gap(9.0, 9.0, 9.0), {5000: 0.6, 5500: 0.59, 6000: 0.7}) == "WEAK"
    assert S.classify_gaps(True, True, gap(9.0, 9.0, 9.0), R_(0.1)) == "NULL"


def test_schema_levels():
    assert [S.schema_of(r) for r in (1, 2, 3, 4, 5, 7, 14)] == [1, 3, 3, 3, 5, 5, 5]


def test_condition_holds(monkeypatch):
    assert S.condition_holds({"router": "sr1"}) is True
    with pytest.raises(RuntimeError, match="unknown condition"):
        S.condition_holds({"conditional": "other"})
    monkeypatch.setattr(S, "select_r4_selected", lambda: False)
    assert S.condition_holds({"conditional": "select_r4"}) is False


def test_summary_arm_set_follows_as_of_round(monkeypatch, tmp_path):
    """Arms of the latency-path schemas: the round's arms plus every arm with a row; `as_of_round` keeps rounds <= K."""
    monkeypatch.setenv("S45SM_RESULTS_DIR", str(tmp_path))
    rows = {a for a, s in R.registry()["arms"].items() if s.get("conditional") is None}      # pfakdbank was never trained
    monkeypatch.setattr(S, "_has_fast_row", lambda arm: arm in rows)
    monkeypatch.setattr(S, "verify_protocol", lambda: None)
    monkeypatch.setattr(S, "register_kinds", lambda: None)
    monkeypatch.setattr(S, "references", lambda: {"s45": {}, "t1d": {}, "t1ddz": {}})
    def entry(arm, ref, fast=True):
        return {**fake(12.0, 0.7, True, "WEAK"), "gap_to_t1ddz": {"local": 1.0, "global": 1.0, "ra": 1.0},
                "R": {"5000": 0.1, "5500": 0.1, "6000": 0.1}, "params_total_with_router": 596350,
                "latency_extra_vs_s45ref_ms": 0.1, "latency_block": "rX", "arm": arm}
    monkeypatch.setattr(S, "classify", entry)
    monkeypatch.setattr(S, "fast_row", lambda net: ({"local": {"mpjpe_ra_mm": 10.0}, "global": {"mpjpe_ra_mm": 7.0},
                                                      "overall": {"mpjpe_ra_mm": 9.0}}, None, "rX"))
    monkeypatch.setattr(S, "condition_holds", lambda spec: spec.get("conditional") is None)
    monkeypatch.setattr(S, "summarize_oracle", lambda net, ref: {
        "main_row": {"local": 9.0, "global": 6.0, "ra": 8.0}, "class": "AT", "gap_to_t1ddz": {"local": 0.0, "global": 0.0, "ra": 0.0},
        "R": {"5000": 1.0, "5500": 1.0, "6000": 1.0}, "convergence": {"C": 0.5}, "zgz": {"e3": {"local": 5.0, "global": 4.0}}})
    reg = R.registry()["arms"]
    full = S.summarize(12)
    got = json.loads(full.read_text())
    assert set(got["arms"]) == rows and got["untrained"] == {}          # unrestricted: every arm with a row, later rounds too
    assert {"xt0cos_srdwa", "pfat0pdz_rep"} <= set(got["arms"])
    (tmp_path / "summary_r12.json").unlink()
    (tmp_path / "report_r12.md").unlink()
    restricted = json.loads(S.summarize(12, as_of_round=7).read_text())
    assert set(restricted["arms"]) == {a for a in rows if reg[a]["round"] <= 7}
    assert restricted["next"]["line"] in ("ROUTER", "ASSOCIATION", "ROBUSTNESS")
    with pytest.raises(RuntimeError, match="read once"):
        S.summarize(12)
    r5 = json.loads(S.summarize(5, as_of_round=5).read_text())            # the conditional arm of round 5 was not trained
    assert set(r5["arms"]) == {a for a in rows if reg[a]["round"] <= 5} and set(r5["untrained"]) == {"pfakdbank"}
    assert set(r5["oracles"]) == {"pfagtdz"} and r5["next"]["line"] == "ROUTER"
    r3 = S.summarize(3, as_of_round=3)                                  # schema 2-4: no oracles, no next
    got3 = json.loads(r3.read_text())
    assert set(got3["arms"]) == {a for a, s in reg.items() if s["round"] <= 3} and "next" not in got3 and "oracles" not in got3
    assert (tmp_path / "report_r3.md").read_text().startswith("# S45-SM 第 3 轮汇总")


# --------------------------------------------------------------------------- round 7 table
def test_swap_table7_effects_and_trunk_choice():
    arms = {"pfasrdz": fake(12.48, 0.81, True), "pfat0pdz": fake(10.93, 0.67, False, "NEAR_TARGET"),
            "pfasrbank": fake(12.15, 0.84, True), "xsr_srp": fake(12.0, 0.83, True), "xt0_sr1": fake(11.9, 0.79, True),
            "xt0_srp": fake(11.0, 0.78, True, "NEAR_TARGET"), "xgt_sr1": fake(16.0, 0.95, True),
            "xgt_srp": fake(10.5, 0.85, True), "xsr_t0p": fake(11.5, 0.75, False), "xgt_t0p": fake(9.0, 0.6, False)}
    s7 = {"round": 7, "arms": arms}
    s5 = {"round": 5, "oracles": {"pfagtdz": {**fake(6.75, 0.49, False, "AT")}}}
    g = T.grid7(s7, s5)
    assert g[("pfagtdz", "label")]["net"] == "pfagtdz" and g[("pfat0pdz", "srpkd")]["net"] == "xt0_srp"
    assert len(g) == 10
    eff = T.effects7(g)
    d = eff["trunk"]["pfagtdz-pfasrdz@sr1"]
    assert d["ra"]["diff"] == pytest.approx(16.0 - 12.48) and d["ra"]["flag"] == "beyond"
    assert eff["router"]["srpkd-sr1@pfasrdz"]["ra"]["flag"] == "within"
    assert set(eff["router"]) == {"srpkd-sr1@pfasrdz", "t0p-sr1@pfasrdz", "srpkd-sr1@pfat0pdz", "t0p-sr1@pfat0pdz",
                                  "srpkd-sr1@pfagtdz", "t0p-sr1@pfagtdz"}
    assert set(eff["trunk"]) == {f"{t}-pfasrdz@{r}" for t in ("pfat0pdz", "pfagtdz") for r in ("sr1", "srpkd", "t0p")}
    choice = T.trunk_choice(s7)
    assert choice == {"network": "xt0_srp", "trunk": "pfat0pdz", "trunk_routes": "t0p", "converging_pool": True, "privileged": True}
    no_conv = {k: {**v, "convergence": {"C": 0.9}} for k, v in arms.items()}
    choice = T.trunk_choice({"round": 7, "arms": no_conv})
    assert choice["network"] == "xgt_srp" and choice["converging_pool"] is False and choice["trunk"] == "pfagtdz"
    only_arm = {"pfasrbank": fake(12.15, 0.79, True), "xgt_sr1": fake(16.0, 0.95, True)}
    assert T.trunk_choice({"round": 7, "arms": only_arm})["trunk_routes"] == "sr1"
    assert "主干" in T.markdown7(g, eff, T.trunk_choice(s7))


# --------------------------------------------------------------------------- rounds 9 / 11 and 12 tables
def fake_pair(ra, C_, budget=True, cls="WEAK"):
    return {"main_row": {"local": ra + 4, "global": ra - 3, "ra": ra}, "convergence": {"C": C_}, "budget_ok": budget, "class": cls}


def test_swap_table11_effects_and_best():
    arms = {"pfasrdz": fake_pair(12.48, 0.81), "pfat0pdz": fake_pair(10.93, 0.67, False, "NEAR_TARGET"),
            "pfasrbank": fake_pair(12.15, 0.84), "xt0_srp": fake_pair(11.92, 0.80), "xgt_srp": fake_pair(13.93, 0.76, cls="MECHANISM"),
            "xt0_srdw": fake_pair(11.5, 0.82), "xgt_srdw": fake_pair(13.0, 0.75), "xt0_srph": fake_pair(11.2, 0.79),
            "xgt_srph": fake_pair(18.0, 0.74), "xt0_srpah": fake_pair(11.0, 0.85), "xgt_srpah": fake_pair(12.6, 0.70)}
    g = T.grid({"round": 11, "arms": arms}, 11)
    assert ("pfasrbank", "sr1") not in g and g[("pfat0pdz", "t0p")]["net"] == "pfat0pdz"
    assert g[("pfagtdz", "srdwkd")]["net"] == "xgt_srdw" and len(g) == 10
    eff = T.effects(g, 11)
    d = eff["part_first"]["srpkdh-srpkd@pfagtdz"]
    assert d["ra"]["diff"] == pytest.approx(18.0 - 13.93) and d["ra"]["flag"] == "beyond"
    assert eff["structure"]["srdwkd-srpkd@pfat0pdz"]["ra"]["flag"] == "within"
    assert set(eff["trunk"]) == {"pfat0pdz-pfagtdz@srpkd", "pfat0pdz-pfagtdz@srdwkd", "pfat0pdz-pfagtdz@srpkdh", "pfat0pdz-pfagtdz@srpkdah"}
    assert eff["anneal"] == {} and set(eff["part_first"]) == {"srpkdh-srpkd@pfat0pdz", "srpkdh-srpkd@pfagtdz"}
    assert T.best(g) == {"in_budget": "xt0_srpah", "in_budget_converging": "xt0_srph"}
    assert "主干效应" in T.markdown(g, eff, T.best(g), 11)
    with pytest.raises(RuntimeError):
        T.grid({"round": 7, "arms": arms}, 11)


def test_swap_table12_nine_subject_effects():
    arms = {"xt0_srp": fake_pair(11.92, 0.80), "xgt_srp": fake_pair(13.93, 0.76), "xt0_srp9": fake_pair(11.0, 0.78),
            "xgt_srp9": fake_pair(17.5, 0.74), "xt0_srp9a": fake_pair(10.9, 0.81), "xt0_srpa": fake_pair(11.5, 0.79),
            "pfat0pdz": fake_pair(10.93, 0.67, False, "NEAR_TARGET")}
    g = T.grid({"round": 12, "arms": arms}, 12)
    assert g[("pfat0pdz", "srpkd9")]["net"] == "xt0_srp9" and len(g) == 7
    eff = T.effects(g, 12)
    assert set(eff["nine"]) == {"srpkd9-srpkd@pfat0pdz", "srpkd9-srpkd@pfagtdz", "srpkd9a-srpkda@pfat0pdz"}
    d = eff["nine"]["srpkd9-srpkd@pfagtdz"]
    assert d["ra"]["diff"] == pytest.approx(17.5 - 13.93) and d["ra"]["flag"] == "beyond"
    assert set(eff["anneal"]) == {"srpkda-srpkd@pfat0pdz", "srpkd9a-srpkd9@pfat0pdz"}
    assert T.best(g) == {"in_budget": "xt0_srp9a", "in_budget_converging": "xt0_srp9"}
    assert "九人效应" in T.markdown(g, eff, T.best(g), 12)
    with pytest.raises(RuntimeError):
        T.grid({"round": 11, "arms": arms}, 12)


def test_the_round_12_layout_extends_the_round_11_one():
    l11, l12 = T.LAYOUTS[11], T.LAYOUTS[12]
    assert l12["routers"][:len(l11["routers"])] == l11["routers"] and l12["routers"][len(l11["routers"]):] == ("srpkd9", "srpkd9a", "srpkd9h", "srpkd9ah")
    assert set(l12["pairs"]) == set(l11["pairs"]) | {"nine"} and [t for t, _ in l12["titles"]][:1] == ["nine"]


def test_d1_table(monkeypatch):
    def fake_read(path):
        name = Path(path).stem
        v = 0.4 + 0.01 * len(name)
        return {"sequences": {s: {"occupancy_overlap_patch": v, "part_cross_share": 0.3 - 0.001 * len(name),
                                  "face_geodesic_median_mm": 13.0, "coverage_onhand_0p5": 0.98} for s in T.D1_SEQS}}
    monkeypatch.setattr(T, "read", fake_read)
    out, diffs = T.d1_rows()
    assert set(diffs) == {"srpkda", "sr1a", "srdwkda", "srpkdh", "srpkdah", "srdwkdh", "srdwkdah", "t0ph"}
    assert {"sr1", "srpkd", "srdwkd", "t0p", "srpkda", "srdwkda"} <= set(out)
    d = diffs["srpkdah"]
    assert d["base"] == "srpkda" and d["ylf_local"]["occupancy_overlap_patch"] == pytest.approx(0.01)
    assert "对基础路由器的差" in T.d1_markdown(out, diffs)


def test_tables_write_exclusively_into_the_results_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("S45SM_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(T, "verify_protocol", lambda: None)
    (tmp_path / "summary_r11.json").write_text(json.dumps({"round": 11, "arms": {"xt0_srp": fake_pair(11.9, 0.8)}}))
    assert T.table(11).parent == tmp_path and (tmp_path / "swap_table_r11.md").is_file()
    with pytest.raises(RuntimeError, match="read once"):
        T.table(11)
    with pytest.raises(RuntimeError, match="no table"):
        T.table(9)


# --------------------------------------------------------------------------- readings of rounds 3a, 13 and 14
def net(ra, C_, cls, budget=True, block="r13"):
    return {"main_row": {"local": ra + 4, "global": ra - 3, "ra": ra}, "convergence": {"C": C_}, "class": cls,
            "budget_ok": budget, "params_total_with_router": 596350 if budget else 2065742,
            "latency_extra_vs_s45ref_ms": 0.1, "latency_block": block}


def test_repeat_reading():
    nets = {"xt0_srdwa": net(11.73, 0.79, "NEAR_TARGET"), "pfat0pdz": net(10.93, 0.67, "NEAR_TARGET", False),
            "xt0rep_srdwa": net(12.40, 0.82, "WEAK"), "pfat0pdz_rep": net(11.2, 0.70, "NEAR_TARGET", False)}
    res = RD.repeat_reading(nets)
    assert res["reading"] == "NOT_REPRODUCED"
    d = res["pairs"]["xt0rep_srdwa-xt0_srdwa"]
    assert d["diff"]["ra"]["diff"] == pytest.approx(0.67) and d["diff"]["ra"]["flag"] == "within"
    assert d["diff"]["C"] == pytest.approx(0.03) and d["mean"]["C"] == pytest.approx(0.805)
    nets["xt0rep_srdwa"] = net(11.0, 0.78, "NEAR_TARGET")
    assert RD.repeat_reading(nets)["reading"] == "REPRODUCED"
    nets["xt0rep_srdwa"] = net(9.5, 0.70, "AT_TARGET")
    assert RD.repeat_reading(nets)["reading"] == "REPRODUCED"
    nets["xt0rep_srdwa"] = net(11.0, 0.78, "NEAR_TARGET", budget=False)
    assert RD.repeat_reading(nets)["reading"] == "NOT_REPRODUCED"
    text = RD.repeat_markdown(nets, RD.repeat_reading(nets))
    assert "判读：NOT_REPRODUCED" in text
    assert text.index("| pfat0pdz |") < text.index("| pfat0pdz_rep |")            # the original first


def test_cosine_reading():
    nets = {"xt0_srdwa": net(11.73, 0.79, "NEAR_TARGET"), "xt0rep_srdwa": net(12.1, 0.81, "WEAK"),
            "pfat0pdz": net(10.93, 0.67, "NEAR_TARGET", False), "pfat0pdz_rep": net(11.0, 0.7, "NEAR_TARGET", False),
            "xt0cos_srdwa": net(11.2, 0.75, "NEAR_TARGET"), "xt0repcos_srdwa": net(11.5, 0.78, "NEAR_TARGET"),
            "pfat0pdz_cos": net(10.5, 0.6, "NEAR_TARGET", False), "pfat0pdz_rep_cos": net(10.6, 0.6, "AT_TARGET", False)}
    res = RD.cosine_reading(nets)
    assert res["reading"] == "ADOPT" and res["at_target"] == []
    assert res["pairs"]["xt0cos_srdwa-xt0_srdwa"]["diff"]["ra"]["diff"] == pytest.approx(11.2 - 11.73)
    nets["xt0repcos_srdwa"] = net(9.9, 0.7, "AT_TARGET")
    assert RD.cosine_reading(nets)["at_target"] == ["xt0repcos_srdwa"]
    nets["xt0repcos_srdwa"] = net(12.5, 0.78, "NEAR_TARGET")
    assert RD.cosine_reading(nets)["reading"] == "MIXED"
    nets["xt0cos_srdwa"] = net(12.0, 0.75, "NEAR_TARGET")
    assert RD.cosine_reading(nets)["reading"] == "REJECT"
    nets["xt0cos_srdwa"], nets["xt0repcos_srdwa"] = net(11.2, 0.82, "WEAK"), net(11.5, 0.78, "NEAR_TARGET")
    assert RD.cosine_reading(nets)["reading"] == "MIXED"
    assert "判读：MIXED" in RD.cosine_markdown(nets, RD.cosine_reading(nets))


def test_the_reading_registry():
    assert set(RD.READINGS) == {"rowcheck_r3a", "repeat_r13", "cosine_r14"}
    with pytest.raises(RuntimeError, match="unregistered reading"):
        RD.read_one("other")
    assert RD.REPEAT_PAIRS[0][1] == "xt0_srdwa" and RD.COSINE_PAIRS == RD.COSINE_CELLS + RD.COSINE_NATIVE
