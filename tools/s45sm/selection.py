"""Router selections: the rules written before the readings, as data, and their queries (CPU only, run once each).

Every rule reads the router's D1 reading (`readouts_d1/<router>.json`, router_read.d1) on the held-out subject's local
recording (ylf_local; zgz is listed as description only):
  r4  srkd replaces sr1 iff part_cross_share(srkd) <= part_cross_share(sr1) - 0.005 and occupancy_overlap_patch(srkd) >= that of sr1
  r6  srpkd replaces the round-5 router (srkd if r4 selected it, else sr1) iff occupancy >= reference + 0.02 and
      part_cross_share <= reference + 0.005
  r8  srdwkd replaces the router after round 6 with round 6's thresholds
  r6c per router: a context in 100 / 150 / 200 / 300 ms qualifies if occupancy >= the 50 ms value + 0.02 and part-cross
      share <= + 0.005 (from `readouts_ctx/<router>.json`, both of its checks required); best = highest occupancy, ties
      the shorter window, else none
Each writes <results>/select_<id>.json (`select_r6c_<router>.json`) exclusively; the queries read those files back.
"""
from __future__ import annotations

from .paths import now, read, require, results_dir, save, sha
from .protocol import verify_protocol
from .registry import router_ready

SEQ = "ylf_local"
KEYS = ("part_cross_share", "occupancy_overlap_patch", "face_geodesic_median_mm", "coverage_onhand_0p5")
MARGIN = 0.005
OCC_GAIN = 0.02
CROSS_SLACK = 0.005
CTX_CANDIDATES = (100, 150, 200, 300)
CTX_KEYS = ("occupancy_overlap_patch", "part_cross_share", "face_geodesic_median_mm", "coverage_onhand_0p5")


def d1(router):
    r = read(results_dir() / "readouts_d1" / f"{router}.json")
    require(r["router"] == router and r["router_checkpoint_sha256"] == sha(router_ready(router)),
            "D1 was read on another checkpoint: " + router)
    return r["sequences"]


def rule_r4(base, cand):
    b, c = base[SEQ], cand[SEQ]
    cross_ok = c["part_cross_share"] <= b["part_cross_share"] - MARGIN
    occupancy_ok = c["occupancy_overlap_patch"] >= b["occupancy_overlap_patch"]
    return {"cross_ok": bool(cross_ok), "occupancy_ok": bool(occupancy_ok), "selected": bool(cross_ok and occupancy_ok)}


def rule_r6(base, cand):
    b, c = base[SEQ], cand[SEQ]
    occupancy_ok = c["occupancy_overlap_patch"] >= b["occupancy_overlap_patch"] + OCC_GAIN
    cross_ok = c["part_cross_share"] <= b["part_cross_share"] + CROSS_SLACK
    return {"occupancy_ok": bool(occupancy_ok), "cross_ok": bool(cross_ok), "selected": bool(occupancy_ok and cross_ok)}


def _read_target(id_):
    r = read(results_dir() / RULES[id_]["file"])
    require(r["status"] == "COMPUTED", f"round-{id_[1:]} selection not computed")
    return r


def selected():
    """The round-4 decision as written in select_r4.json (raises if it has not been computed)."""
    return bool(_read_target("r4")["selected"])


def selected_router(round_):
    """The router after round 6 / 8, as written in select_r6.json / select_r8.json (raises if not computed)."""
    id_ = f"r{round_}"
    require(id_ in ("r6", "r8"), "router selections exist for rounds 6 and 8")
    return _read_target(id_)[RULES[id_]["out_key"]]


def reference_r4():
    return "sr1"


def reference_r6():
    return "srkd" if selected() else "sr1"


def reference_r8():
    return selected_router(6)


#: id -> candidate, reference router, rule, file, written key, constants, the ledger section the rule was written in
RULES = {
    "r4": {"candidate": "srkd", "reference": reference_r4, "rule": rule_r4, "file": "select_r4.json",
           "out_key": "router_for_round5", "constants": {"margin": MARGIN}, "reference_key": False,
           "prereg": "docs/EXPERIMENT_RECORDS.md#s45sm-r4-prereg-20261031"},
    "r6": {"candidate": "srpkd", "reference": reference_r6, "rule": rule_r6, "file": "select_r6.json",
           "out_key": "router_after_round6", "constants": {"occupancy_gain": OCC_GAIN, "cross_slack": CROSS_SLACK},
           "reference_key": True, "prereg": "docs/EXPERIMENT_RECORDS.md#s45sm-r6-prereg-20261031"},
    "r8": {"candidate": "srdwkd", "reference": reference_r8, "rule": rule_r6, "file": "select_r8.json",
           "out_key": "router_after_round8", "constants": {"occupancy_gain": OCC_GAIN, "cross_slack": CROSS_SLACK},
           "reference_key": True, "prereg": "docs/EXPERIMENT_RECORDS.md#s45sm-r8-prereg-20261031"},
}


def run(id_):
    """Compute one router selection (r4 / r6 / r8) once and write it."""
    spec = RULES[id_]
    verify_protocol()
    target = results_dir() / spec["file"]
    require(not target.exists(), f"the round-{id_[1:]} selection is computed once")
    ref, name = spec["reference"](), spec["candidate"]
    base, cand = d1(ref), d1(name)
    res = spec["rule"](base, cand)
    head = {"status": "COMPUTED", "computed_utc": now(), "rule": spec["prereg"], "sequence": SEQ, **spec["constants"]}
    if spec["reference_key"]:
        head["reference"] = ref
    out = {**head, **res, spec["out_key"]: name if res["selected"] else ref,
           ref: {k: base[SEQ][k] for k in KEYS}, name: {k: cand[SEQ][k] for k in KEYS},
           "descriptive_zgz_local": {r: {k: s["zgz_local"][k] for k in KEYS} for r, s in ((ref, base), (name, cand))}}
    save(target, out, exclusive=True)
    if spec["reference_key"]:
        print(name, "SELECTED" if res["selected"] else "NOT SELECTED", "over", ref, res, flush=True)
    else:
        print(name, "SELECTED" if res["selected"] else "NOT SELECTED", res, flush=True)
    return target


# --------------------------------------------------------------------------- context selection (r6c)
def ctx_target(router):
    return results_dir() / f"select_r6c_{router}.json"


def rule_ctx(probe):
    base = probe["by_context"]["50"][SEQ]
    rows = {}
    for c in CTX_CANDIDATES:
        m = probe["by_context"][str(c)][SEQ]
        occ_ok = m["occupancy_overlap_patch"] >= base["occupancy_overlap_patch"] + OCC_GAIN
        cross_ok = m["part_cross_share"] <= base["part_cross_share"] + CROSS_SLACK
        rows[str(c)] = {"occupancy_ok": bool(occ_ok), "cross_ok": bool(cross_ok), "qualifies": bool(occ_ok and cross_ok)}
    ok = [c for c in CTX_CANDIDATES if rows[str(c)]["qualifies"]]
    best = max(ok, key=lambda c: (probe["by_context"][str(c)][SEQ]["occupancy_overlap_patch"], -c)) if ok else None
    return best, rows


def selected_context(router):
    """The selected context in ms for `router`, or None (no selection file, or nothing qualified)."""
    path = ctx_target(router)
    if not path.exists():
        return None
    r = read(path)
    require(r["status"] == "COMPUTED", "selection not computed: " + router)
    return r["context_ms"]


def run_ctx(router):
    verify_protocol()
    out = ctx_target(router)
    require(not out.exists(), "the context selection is computed once per router")
    src = results_dir() / "readouts_ctx" / f"{router}.json"
    probe = read(src)
    checks = probe["checks"]
    status = "COMPUTED" if checks["d1"] and checks["ctx_equal"] else "CHECK_FAILED"
    best, rows = rule_ctx(probe) if status == "COMPUTED" else (None, {})
    desc = {c: {s: {k: probe["by_context"][c][s][k] for k in CTX_KEYS} for s in (SEQ, "zgz_local", "zgz_global")}
            for c in probe["by_context"]}
    save(out, {"status": status, "computed_utc": now(), "rule": "docs/EXPERIMENT_RECORDS.md#s45sm-r6c-prereg-20261031",
               "router": router, "probe": str(src), "probe_sha256": sha(src), "checks": checks, "sequence": SEQ,
               "occ_gain": OCC_GAIN, "cross_slack": CROSS_SLACK, "candidates": rows, "context_ms": best,
               "values": desc}, exclusive=True)
    print(router, status, "context_ms =", best, flush=True)
    return out
