"""CPU tests: the shared-queue job blocks and the translation of the old command lines (every `sm:` command of the queue
parses with the single CLI; the standard blocks equal the recorded jobs of rounds 3, 13 and 14)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import cli  # noqa: E402
from tools.s45sm import jobs as J  # noqa: E402
from tools.s45sm import legacy as L  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402

QUEUE = ROOT / "outputs/s45rt/20261031/queue_r2.json"
JOBS = P.OUT / "jobs"


def queue_jobs():
    if not QUEUE.is_file():
        pytest.skip("shared queue not present")
    data = json.loads(QUEUE.read_text())
    return [j for j in (data["jobs"] if isinstance(data, dict) else data) if j["key"].startswith("sm:")]


def test_every_queued_command_translates_and_parses():
    jobs = queue_jobs()
    assert len(jobs) >= 749
    obsolete, verbs = [], {}
    for j in jobs:
        try:
            command = L.translate(j["command"])
        except L.Obsolete:
            obsolete.append(j["key"])
            continue
        i = command.index(L.CLI)
        plan = cli.plan(command[i + 1:])
        verbs[plan["verb"]] = verbs.get(plan["verb"], 0) + 1
        assert plan["needs_gpu"] == (j["kind"] in ("gpu", "exclusive")) or plan["gpu_pair"] == (j["kind"] == "pair") or j["kind"] == "cpu", j["key"]
        assert not any(t.startswith("S45SM_ENTRY") for t in command)
    assert len(obsolete) == 3 and all(k.startswith(("sm:a1:arch", "sm:a1:archive")) or ":arch" in k for k in obsolete), obsolete
    assert verbs["train"] >= 31 and verbs["evaluate"] >= 100 and verbs["readouts"] >= 150 and verbs["row"] >= 17
    assert set(verbs) <= set(cli.HANDLERS)


def test_gpu_kinds_match_the_old_driver_classification():
    for j in queue_jobs():
        try:
            command = L.translate(j["command"])
        except L.Obsolete:
            continue
        plan = cli.plan(command[command.index(L.CLI) + 1:])
        if j["kind"] == "pair":
            assert plan["gpu_pair"], j["key"]
        elif j["kind"] == "exclusive":
            assert plan["exclusive_gpu"], j["key"]
        elif j["kind"] == "gpu":
            assert plan["needs_gpu"] and not plan["gpu_pair"], j["key"]
        else:
            assert not plan["needs_gpu"] and not plan["gpu_pair"], j["key"]


def test_translation_of_each_wrapper_form():
    py = "/p/python"
    cases = [
        ([py, "tools/s45sm/train4.py", "--arm", "a", "--phase", "formal", "--config", "c"], [py, L.CLI, "train", "--arm", "a", "--phase", "formal", "--config", "c"]),
        (["/usr/bin/env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", py, "tools/s45sm/train3.py", "--arm", "a"],
         ["/usr/bin/env", "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", py, L.CLI, "train", "--arm", "a"]),
        (["/usr/bin/env", "S45SM_ENTRY=evaluate2", py, "tools/s45sm/swap_run2.py", "--net", "x", "--mode", "e1"], [py, L.CLI, "evaluate", "--net", "x", "--mode", "e1"]),
        (["/usr/bin/env", "S45SM_ENTRY=readouts2", py, "tools/s45sm/swap_run.py", "--net", "x", "--task", "tl_e1"], [py, L.CLI, "readouts", "--net", "x", "--task", "tl_e1"]),
        (["/usr/bin/env", "S45SM_ENTRY=evaluate_fast", py, "tools/s45sm/swap_run2.py", "--net", "x", "--block", "b"], [py, L.CLI, "row", "--net", "x", "--block", "b"]),
        (["/usr/bin/env", "S45SM_ENTRY=router_domains", py, "tools/s45sm/kind_run2.py", "--router", "r"], [py, L.CLI, "router-read", "d1", "--router", "r"]),
        (["/usr/bin/env", "S45SM_ENTRY=decide3", py, "tools/s45sm/kind_run.py", "--round", "7"], [py, L.CLI, "decide", "--round", "7"]),
        ([py, "tools/s45sm/select_r6c.py", "--router", "r"], [py, L.CLI, "select", "--id", "r6c", "--router", "r"]),
        ([py, "tools/s45sm/swap_table3.py"], [py, L.CLI, "table", "--round", "12"]),
        ([py, "tools/s45sm/d1_table.py"], [py, L.CLI, "table", "--round", "10"]),
        ([py, "tools/s45sm/cosine_read.py"], [py, L.CLI, "read", "--id", "cosine_r14"]),
        ([py, "tools/s45sm/replay3.py", "--arm", "a"], [py, L.CLI, "replay", "--arm", "a"]),
        ([py, "tools/s45sm/bank.py", "run", "--arm", "a", "--shard", "1", "--shards", "6"], [py, L.CLI, "bank", "run", "--arm", "a", "--shard", "1", "--shards", "6"]),
        ([py, "tools/s45sm/decode_variant.py", "--router", "r"], [py, L.CLI, "router-decode", "--router", "r"])]
    for old, new in cases:
        assert L.translate(old) == new, old
    for script in L.OBSOLETE:
        with pytest.raises(L.Obsolete):
            L.translate([py, f"tools/s45sm/{script}.py"])
    with pytest.raises(ValueError, match="S45SM_ENTRY"):
        L.translate([py, "tools/s45sm/swap_run2.py", "--net", "x"])
    with pytest.raises(L.Obsolete):
        L.translate(["/bin/bash", "-c", f"{py} tools/s45sm/a1_archive.py --arm a --wait-key k"])
    with pytest.raises(ValueError):
        L.translate([py, "tools/s45sm/unknown_script.py"])


def recorded(round_):
    path = JOBS / f"r{round_}.json"
    if not path.is_file():
        pytest.skip("jobs file not present")
    return json.loads(path.read_text())


def compare(generated, recorded_jobs, tolerated_deps=()):
    rec = {j["key"]: j for j in recorded_jobs}
    for g in generated:
        r = rec[g["key"]]
        assert g["kind"] == r["kind"] and g["mem_gib"] == r["mem_gib"] and g["note"] == r["note"], g["key"]
        assert g["command"] == L.translate(r["command"]), g["key"]
        assert g["deps"] == r["deps"] or g["key"] in tolerated_deps, (g["key"], g["deps"], r["deps"])
    return len(generated)


def test_blocks_equal_the_recorded_round_14_jobs():
    rec = recorded(14)
    note = rec[0]["note"]
    n = compare(J.train_block(14, "pfat0pdz_cos", note), rec)
    n += compare(J.train_block(14, "pfat0pdz_rep_cos", note, deps_accept_pre=["sm:r13:acc:pre:pfat0pdz_rep"],
                               deps_accept_formal=["sm:r13:acc:formal:pfat0pdz_rep"]), rec)
    for net, dep in (("pfat0pdz_cos", "pfat0pdz_cos"), ("pfat0pdz_rep_cos", "pfat0pdz_rep_cos"), ("xt0cos_srdwa", "pfat0pdz_cos"),
                     ("xt0repcos_srdwa", "pfat0pdz_rep_cos")):
        n += compare(J.evaluate_block(14, net, note, deps=[f"sm:r14:acc:formal:{dep}"]), rec)
    nets = ["pfat0pdz_cos", "pfat0pdz_rep_cos", "xt0cos_srdwa", "xt0repcos_srdwa"]
    first = next(j for j in rec if j["key"] == "sm:r14:row:s45ref_r14")["deps"]
    n += compare(J.row_block(14, "r14", nets, note, deps_first=first), rec)
    n += compare([J.reading_job(14, ("read", "--id", "cosine_r14"), "sm:r14:row:xt0repcos_srdwa", note)], rec)
    assert n == len(rec) == 58


def test_blocks_equal_the_recorded_round_13_jobs():
    rec = recorded(13)
    note = rec[0]["note"]
    # the replay arm: a preflight, its acceptance and the replay check (the replay has its own job kinds)
    n = compare(J.train_block(13, "pfat0pdz_rep", note, deps_formal=["sm:r13:replay:pfat0pdz_replay"]), rec)
    for net, dep in (("pfat0pdz_rep", "pfat0pdz_rep"), ("xt0rep_srdwa", "pfat0pdz_rep")):
        n += compare(J.evaluate_block(13, net, note, deps=[f"sm:r13:acc:formal:{dep}"]), rec)
    first = next(j for j in rec if j["key"] == "sm:r13:row:s45ref_r13")["deps"]
    n += compare(J.row_block(13, "r13", ["pfat0pdz_rep", "xt0rep_srdwa"], note, deps_first=first), rec)
    n += compare([J.reading_job(13, ("read", "--id", "repeat_r13"), "sm:r13:row:xt0rep_srdwa", note)], rec)
    assert n == len(rec) - 3                                             # minus the replay arm's three jobs


def test_blocks_equal_the_recorded_round_3_jobs_up_to_the_tl_f1_acceptance_dependency():
    rec = recorded(3)
    note = rec[0]["note"]
    tolerated = {j["key"] for j in rec if j["key"].endswith(":tl_f1")}       # round 3 also made tl_f1 wait for the acceptance
    n = 0
    for arm in ("spfasrdz", "pfasrbank", "spfasrbank"):
        n += compare(J.train_block(3, arm, note, deps_pre=[next(j for j in rec if j["key"] == f"sm:r3:pre:{arm}")["deps"][0]]), rec)
        n += compare(J.evaluate_block(3, arm, note, deps=[f"sm:r3:acc:formal:{arm}"]), rec, tolerated)
    assert n == 3 * (4 + 11)
    assert J.evaluate_block(3, "spfasrdz", note, deps=["a"], f1_deps=["a"])[8]["deps"] == ["sm:r3:rs:spfasrdz:tl_e1", "a"]


def test_the_block_shapes():
    note = "n"
    train = J.train_block(14, "pfat0pdz_cos", note)
    assert [j["kind"] for j in train] == ["pair", "cpu", "pair", "cpu"] and [j["mem_gib"] for j in train] == [130, 4, 130, 4]
    assert train[0]["command"][:2] == J.ALLOC and train[1]["command"][:1] == [J.DEFAULT_PYTHON]
    ev = J.evaluate_block(14, "xt0cos_srdwa", note, deps=["acc"])
    assert len(ev) == 11 and {j["kind"] for j in ev} == {"gpu"} and ev[8]["deps"] == ["sm:r14:rs:xt0cos_srdwa:tl_e1"]
    rows = J.row_block(14, "r14", ["a", "b"], note, deps_first=["x"])
    assert [j["key"] for j in rows] == ["sm:r14:row:s45ref_r14", "sm:r14:row:a", "sm:r14:row:b"]
    assert rows[1]["deps"] == [rows[0]["key"]] and rows[2]["deps"] == [rows[1]["key"]] and {j["kind"] for j in rows} == {"exclusive"}


def test_standard_round_plans_the_registered_arms():
    plan = J.standard_round(14, block="r14", reading=("read", "--id", "cosine_r14"))
    keys = [j["key"] for j in plan]
    assert len(keys) == len(set(keys))
    assert "sm:r14:formal:pfat0pdz_cos" in keys and "sm:r14:ev:xt0repcos_srdwa:e3:6000" in keys and keys[-1] == "sm:r14:read"
    assert "sm:r14:row:xt0cos_srdwa" in keys and not any(k.startswith("sm:r14:pre:xt0") for k in keys)       # cells do not train
    plan13 = J.standard_round(13)
    assert not any(":pfat0pdz_replay" in j["key"] for j in plan13)             # a replay is not a standard arm
    assert J.acceptance_key("pfagtdz") == "sm:r5:acc:formal:pfagtdz" and J.acceptance_key("pfat0pdz_cos") == "sm:r14:acc:formal:pfat0pdz_cos"
    assert J.prereg(14) is None or J.prereg(14).startswith("docs/")


def test_translate_file_skips_obsolete_scripts():
    out = J.translate_file(JOBS / "r1a.json") if (JOBS / "r1a.json").is_file() else {}
    assert all(c[1] == L.CLI or L.CLI in c for c in out.values())
    assert not any("a1_archive" in " ".join(c) for c in out.values())
