"""CPU tests: acceptance of finished trainings, the bit-identity helpers and the receipts of the consolidated S45-SM package
(ported from the round 13 replay comparison and the round 14 same-through tests; new: the acceptance branches re-derive the
receipts on disk, tampering is detected, the receipt keys)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import protocol as PR  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402
from tools.s45sm import verify as V  # noqa: E402

IGNORE = {"wrapper_sha256", "train4_sha256", "verified_at_utc", "engineering_only_no_accuracy_result", "code_identity_sha256"}


def fake_run(path, *, objective="a", state="s"):
    path.mkdir(parents=True)
    for rank in range(2):
        (path / f"completion_rank{rank}.json").write_text(json.dumps(
            {"status": "COMPLETE", "step": 50, "batches": 50, "input_stream_sha256": "i", "indices_sha256": "x"}))
        (path / f"indices_rank{rank}.i32").write_bytes(b"\x01\x02\x03\x04")
        rows = [{"step": s, "objective_sha256": objective if s == 7 else "o", "gradients": {"w": s}} for s in range(1, 51)]
        (path / f"batches_rank{rank}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (path / f"pair_stream_rank{rank}.json").write_text(json.dumps([{"step": 0, "sha256": "p"}]))
        for s in range(51):
            (path / f"state_rank{rank}_step{s:06}.json").write_text(json.dumps(
                {"step": s, "model": state if s == 33 else "m", "optimizers": "opt", "schedulers": "sch",
                 "torch_cpu_rng": "c", "torch_cuda_rng": "g", "input_stream_sha256": "i", "batches": s}))
        for s in (25, 50):
            (path / f"validation_rank{rank}_step{s:06}.json").write_text(json.dumps(
                {"status": "PASS", "step": s, "batches": 126, "model_state_unchanged": True}))
    return path


def test_replay_compare(tmp_path):
    ref = fake_run(tmp_path / "ref")
    assert [r["state_snapshots"] for r in V.replay_compare(fake_run(tmp_path / "same"), ref)] == [51, 51]
    with pytest.raises(RuntimeError, match="journal rows differ at steps \\[7\\]"):
        V.replay_compare(fake_run(tmp_path / "obj", objective="b"), ref)
    with pytest.raises(RuntimeError, match="state snapshot 33"):
        V.replay_compare(fake_run(tmp_path / "state", state="t"), ref)
    tape = shutil.copytree(ref, tmp_path / "tape")
    (tape / "indices_rank1.i32").write_bytes(b"\x01\x02\x03\x05")
    with pytest.raises(RuntimeError, match="index tape"):
        V.replay_compare(tape, ref)
    pair = shutil.copytree(ref, tmp_path / "pair")
    (pair / "pair_stream_rank0.json").write_text(json.dumps([{"step": 0, "sha256": "q"}]))
    with pytest.raises(RuntimeError, match="paired-stream"):
        V.replay_compare(pair, ref)
    val = shutil.copytree(ref, tmp_path / "val")
    (val / "validation_rank1_step000050.json").write_text(json.dumps({"status": "PASS", "step": 50, "batches": 125, "model_state_unchanged": True}))
    with pytest.raises(RuntimeError, match="full validation"):
        V.replay_compare(val, ref)


def fake_formal(path, *, late="o", early="o", state="m"):
    path.mkdir(parents=True)
    for rank in range(2):
        rows = [{"step": s, "objective_sha256": (early if s == 300 else "o") if s <= 500 else late} for s in range(1, 601)]
        (path / f"batches_rank{rank}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (path / f"state_rank{rank}_step000500.json").write_text(json.dumps(
            {"step": 500, "model": state, "optimizers": "opt", "schedulers": "sch", "torch_cpu_rng": "c",
             "torch_cuda_rng": "g", "input_stream_sha256": "i", "batches": 500}))
    return path


def test_same_through(tmp_path):
    ref = fake_formal(tmp_path / "ref")
    assert V.same_through(fake_formal(tmp_path / "later", late="x"), ref) == 500
    with pytest.raises(RuntimeError, match="steps \\[300\\]"):
        V.same_through(fake_formal(tmp_path / "early", early="x"), ref)
    with pytest.raises(RuntimeError, match="step-500 state"):
        V.same_through(fake_formal(tmp_path / "state", state="z"), ref)
    short = tmp_path / "short"
    short.mkdir()
    for rank in range(2):
        (short / f"batches_rank{rank}.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError, match="shorter"):
        V.same_through(short, ref)


def test_replayed_names():
    assert R.replayed("pfat0pdz_replay") == "pfat0pdz"
    with pytest.raises(RuntimeError):
        R.replayed("pfat0pdz_rep")


# --------------------------------------------------------------------------- every acceptance branch re-derives its receipt
CASES = ("preflight_pfat0pdz", "preflight_pfasrbank", "preflight_spfasrdz", "preflight_pfat0pdz_rep", "preflight_pfagtdz",
         "preflight_pfat0pdz_cos", "preflight_pfat0pdz_rep_cos", "formal_pfat0pdz_cos", "formal_pfagtdz")


@pytest.mark.parametrize("case", CASES)
def test_verify_run_rederives_the_receipt(case):
    phase, _, arm = case.partition("_")
    rec = json.loads((P.OUT / "receipts" / f"{case}.json").read_text())
    got = V.verify_run(arm, phase)
    assert got == {k: v for k, v in rec.items() if k not in IGNORE}
    spec = R.arm_spec(arm) if arm in R.registry()["arms"] else None
    if spec is None:
        assert "full_packets_compared" in got["ranks"][0] and got["base_run"].endswith("s45at_t1ddz_s3407")
    elif R.sampler(arm) == "dedicated":
        assert got["index_run"].endswith("s45at_s45samp_s3407") and got["ranks"][0] == {"rank": 0, "packets_compared": False}
    elif spec.get("bank"):
        assert got["ranks"][0] == {"rank": 0, "packets_compared": False} and "index_run" not in got
    else:
        assert got["ranks"][0]["full_packets_compared"] > 0
    if R.schedule(arm) == "cosine" if spec else False:
        assert ("preflight_equals" in got) == (phase == "preflight") and ("formal_equals_base_through_step" in got) == (phase == "formal")


def test_replay_receipt_rederives():
    rec = json.loads((P.OUT / "receipts/replay_pfat0pdz_replay.json").read_text())
    assert V.replay_compare(R.output_dir("pfat0pdz_replay", "preflight"), R.output_dir("pfat0pdz", "preflight")) == rec["ranks"]


# --------------------------------------------------------------------------- tampering is detected
@pytest.fixture()
def copy_of_a_preflight(tmp_path, monkeypatch):
    real = R.output_dir("pfat0pdz", "preflight")
    copy = tmp_path / "run"
    shutil.copytree(real, copy)
    monkeypatch.setattr(V, "output_dir", lambda arm, phase: copy if (arm, phase) == ("pfat0pdz", "preflight") else R.output_dir(arm, phase))
    return copy


def edit(path, fn):
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_untouched_copy_passes(copy_of_a_preflight):
    assert V.verify_run("pfat0pdz", "preflight")["status"] == "PASS"


def test_changed_packet_hash_is_detected(copy_of_a_preflight):
    rows = [json.loads(s) for s in (copy_of_a_preflight / "batches_rank1.jsonl").read_text().splitlines()]
    rows[10]["small_fields_sha256"] = "0" * 64
    (copy_of_a_preflight / "batches_rank1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="Training packets differ"):
        V.verify_run("pfat0pdz", "preflight")


def test_changed_index_tape_is_detected(copy_of_a_preflight):
    data = bytearray((copy_of_a_preflight / "indices_rank0.i32").read_bytes())
    data[0] ^= 1
    (copy_of_a_preflight / "indices_rank0.i32").write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="Sample indices differ"):
        V.verify_run("pfat0pdz", "preflight")


def test_incomplete_rank_and_checkpoint_sha_are_detected(copy_of_a_preflight):
    edit(copy_of_a_preflight / "completion_rank1.json", lambda d: d.update(final_checkpoint_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="Final checkpoint SHA"):
        V.verify_run("pfat0pdz", "preflight")
    edit(copy_of_a_preflight / "completion_rank1.json", lambda d: d.update(status="RUNNING"))
    with pytest.raises(RuntimeError):
        V.verify_run("pfat0pdz", "preflight")


def test_runtime_differences_are_detected(copy_of_a_preflight):
    path = copy_of_a_preflight / "effective_runtime_rank0.json"
    original = path.read_text()
    edit(path, lambda d: d.update(s45sm_router_checkpoint_sha256="0" * 64))
    with pytest.raises(RuntimeError, match="Router checkpoint changed"):
        V.verify_run("pfat0pdz", "preflight")
    path.write_text(original)
    edit(path, lambda d: d["s45sm_routing"].update(assoc="face_route"))
    with pytest.raises(RuntimeError, match="Installed routing differs"):
        V.verify_run("pfat0pdz", "preflight")
    path.write_text(original)
    edit(path, lambda d: d.update(s45sm_fast=True))
    with pytest.raises(RuntimeError, match="Fast-path flag"):
        V.verify_run("pfat0pdz", "preflight")
    path.write_text(original)
    edit(path, lambda d: d["actual_math_flags"].update(cudnn_benchmark=True))
    with pytest.raises(RuntimeError, match="math flags"):
        V.verify_run("pfat0pdz", "preflight")


def test_unfinished_validation_is_detected(copy_of_a_preflight):
    edit(copy_of_a_preflight / "validation_rank0_step000050.json", lambda d: d.update(model_state_unchanged=False))
    with pytest.raises(RuntimeError, match="Full validation"):
        V.verify_run("pfat0pdz", "preflight")


# --------------------------------------------------------------------------- the receipt writers (into a scratch OUT)
def test_accept_writes_the_receipt_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUT", tmp_path)
    V.accept("pfat0pdz_cos", "preflight")
    rec = json.loads((tmp_path / "receipts/preflight_pfat0pdz_cos.json").read_text())
    assert rec["status"] == "PASS" and rec["wrapper_sha256"] == P.sha(P.PACKAGE / "train.py")
    assert rec["code_identity_sha256"] == PR.code_identity() and rec["engineering_only_no_accuracy_result"] is True
    assert rec["preflight_equals"] == "pfat0pdz"
    with pytest.raises(FileExistsError):
        V.accept("pfat0pdz_cos", "preflight")                          # exclusive
    with pytest.raises(RuntimeError, match="not registered"):
        V.accept("pfat0pdz_replay", "formal")                          # a replay has no formal phase
    with pytest.raises(RuntimeError, match="unregistered"):
        V.accept("no_such_arm", "preflight")


def test_a_failed_acceptance_is_recorded_and_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUT", tmp_path)
    monkeypatch.setattr(V, "verify_run", lambda arm, phase: (_ for _ in ()).throw(RuntimeError("synthetic difference")))
    with pytest.raises(RuntimeError, match="acceptance failed"):
        V.accept("pfat0pdz", "preflight")
    rec = json.loads((tmp_path / "receipts/preflight_pfat0pdz.json").read_text())
    assert rec["status"] == "FAIL" and rec["error"] == "synthetic difference"


def test_replay_writes_its_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "OUT", tmp_path)
    V.replay("pfat0pdz_replay")
    rec = json.loads((tmp_path / "receipts/replay_pfat0pdz_replay.json").read_text())
    assert rec["status"] == "PASS" and rec["replay_of"] == "pfat0pdz" and rec["checker_sha256"] == P.sha(P.PACKAGE / "verify.py")
    assert rec["code_identity_sha256"] == PR.code_identity()


# --------------------------------------------------------------------------- the formal-run binding
def test_check_binding_requires_this_code_identity(tmp_path, monkeypatch):
    from tools.s45sm import train as T
    monkeypatch.setattr(T, "OUT", tmp_path)
    monkeypatch.setattr(T, "verify_protocol", lambda: None)
    monkeypatch.setattr(T, "config_frozen", lambda config: None)
    (tmp_path / "receipts").mkdir()

    class Options:
        arm = "pfat0pdz"

    def write(**extra):
        (tmp_path / "receipts/preflight_pfat0pdz.json").write_text(json.dumps(
            {"status": "PASS", "wrapper_sha256": P.sha(T.__file__), **extra}))
    write(code_identity_sha256=PR.code_identity())
    T.check_binding(Options, "any")
    write()                                                             # a preflight accepted by the old tools
    with pytest.raises(RuntimeError, match="code changed"):
        T.check_binding(Options, "any")
    write(code_identity_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="code changed"):
        T.check_binding(Options, "any")
    (tmp_path / "receipts/preflight_pfat0pdz.json").write_text(json.dumps(
        {"status": "PASS", "wrapper_sha256": "0" * 64, "code_identity_sha256": PR.code_identity()}))
    with pytest.raises(RuntimeError, match="wrapper changed"):
        T.check_binding(Options, "any")
    (tmp_path / "receipts/preflight_pfat0pdz.json").write_text(json.dumps({"status": "FAIL"}))
    with pytest.raises(RuntimeError, match="not accepted"):
        T.check_binding(Options, "any")


def test_validate_recipe_detects_a_difference():
    from tools.s45sm import train as T
    import yaml
    cfg = yaml.safe_load(R.config_path("pfat0pdz", "formal").read_text())
    T.validate_recipe(cfg, "pfat0pdz", "formal")
    cfg["TRAIN"]["LR"] = 0.5
    with pytest.raises(RuntimeError, match="Unregistered difference"):
        T.validate_recipe(cfg, "pfat0pdz", "formal")
    orc = yaml.safe_load(R.config_path("pfagtdz", "formal").read_text())
    T.validate_recipe(orc, "pfagtdz", "formal")
    orc["S45SM"]["ASSOC"] = "face_route"
    with pytest.raises(RuntimeError, match="oracle identity"):
        T.validate_recipe(orc, "pfagtdz", "formal")


def test_fresh_claim_is_exclusive(tmp_path, monkeypatch):
    from tools.s45sm import train as T

    class Options:
        arm, phase = "pfat0pdz", "preflight"
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("x: 1\n")
    cfg = {"TRAIN": {"OUTPUT_DIR": str(tmp_path / "out")}, "_config_path": str(cfg_file)}
    monkeypatch.delenv("S45SM_CLAIM_NONCE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    T.fresh_claim(Options, cfg)
    try:
        assert (tmp_path / "out/.fresh_claim.json").is_file()
        monkeypatch.delenv("S45SM_CLAIM_NONCE")
        with pytest.raises(RuntimeError, match="Fresh-only"):
            T.fresh_claim(Options, cfg)
    finally:
        import os
        os.environ.pop("S45SM_CLAIM_NONCE", None)
        os.environ.pop("S45SM_CLAIM_PARENT", None)
    assert T.claim_prefix("pfagtdz") == "S45SMO_" and T.claim_prefix("pfat0pdz") == "S45SM_"
