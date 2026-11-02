"""Engineering acceptance of S45-SM trainings, bit-identity helpers and replay checks (CPU only, never scores).

`verify_run(arm, phase)` checks one finished run against the recorded training it must equal, chosen from the arm's
registry fields (an acceptance never scores):
  packets    (default; also round 1 and cosine arms' base checks) completion, sample indices, initial tensors and every
             journal row (index SHA, small-field SHA, full-packet SHA) equal to S45-AT dz's recorded training: the
             routes change inside the model only; runtime loader / math flags equal to dz's; routing as registered
  bank       `bank`: the priors change, so packets are not compared (index stream, initial tensors, runtime, bank)
  dedicated  `sampler: dedicated`: index stream equal to S45-AT s45samp's, initial tensors equal to dz's, loader contract
             equal to s45samp's, sampler / routing / bank / fast flag as registered; no packet comparison
  oracle     a label-route oracle: as `packets` against S45-AT t1ddz
  cosine     `lr_schedule: cosine` (on top of the branch above): the preflight equals the base arm's preflight bit for
             bit, the formal run equals the base's through step 500 (journal rows and the step-500 state)
The receipt keys per branch are the ones the per-variant verifiers wrote (`full_packets_compared` / `packets_compared`,
`base_run` / `index_run` + `init_run`, `preflight_equals`, `formal_equals_base_through_step`, `schedule_base`).
"""
from __future__ import annotations

from pathlib import Path

from .paths import (DZ_RUN, OUT, PACKAGE, SAME_STEPS, SAMP_RUN, T1DDZ_RUN, now, read, recorded, require, save, sha)
from .protocol import code_identity, verify_protocol
from .registry import (arm_spec, config_path, oracles, output_dir, phases, registry, replayed, run_name, schedule, sampler)
from .configs import expected_state

STATE_KEYS = ("step", "model", "optimizers", "schedulers", "torch_cpu_rng", "torch_cuda_rng", "input_stream_sha256",
              "batches")
REPLAY_STEPS = 50


def journal(directory, rank, steps=None):
    import json
    rows = [json.loads(s) for s in (Path(directory) / f"batches_rank{rank}.jsonl").read_text().splitlines()]
    return rows if steps is None else rows[:steps]


# --------------------------------------------------------------------------- bit-identity helpers
def same_through(run, ref, steps=SAME_STEPS):
    """Journal rows 1..steps and the step-`steps` state snapshot of two formal runs, per rank, bit for bit."""
    for rank in range(2):
        rows, ref_rows = journal(run, rank)[:steps], journal(ref, rank)[:steps]
        require(len(rows) == len(ref_rows) == steps, "journal shorter than the identity window")
        bad = [r.get("step") for r, o in zip(rows, ref_rows) if r != o]
        require(not bad, f"formal journal differs from its base at steps {bad[:5]}")
        name = f"state_rank{rank}_step{steps:06}.json"
        a, b = read(run / name), read(ref / name)
        bad = [k for k in STATE_KEYS if a.get(k) != b.get(k)]
        require(not bad, f"step-{steps} state differs from its base in {bad}")
    return steps


def replay_compare(run, ref):
    """Per-rank comparison of two preflight directories; raises RuntimeError on the first difference."""
    ranks = []
    for rank in range(2):
        done, old = read(run / f"completion_rank{rank}.json"), read(ref / f"completion_rank{rank}.json")
        require(done["status"] == old["status"] == "COMPLETE" and done["step"] == old["step"] == REPLAY_STEPS,
                "incomplete preflight")
        require(done["input_stream_sha256"] == old["input_stream_sha256"] and done["indices_sha256"] == old["indices_sha256"],
                "input stream differs")
        require((run / f"indices_rank{rank}.i32").read_bytes() == (ref / f"indices_rank{rank}.i32").read_bytes(),
                "index tape differs")
        rows, ref_rows = journal(run, rank), journal(ref, rank)
        require(len(rows) == len(ref_rows) == REPLAY_STEPS, "journal length differs")
        bad = [r.get("step") for r, o in zip(rows, ref_rows) if r != o]
        require(not bad, f"journal rows differ at steps {bad[:5]}")
        require(read(run / f"pair_stream_rank{rank}.json") == read(ref / f"pair_stream_rank{rank}.json"),
                "paired-stream records differ")
        for step in range(REPLAY_STEPS + 1):
            name = f"state_rank{rank}_step{step:06}.json"
            a, b = read(run / name), read(ref / name)
            bad = [k for k in STATE_KEYS if a.get(k) != b.get(k)]
            require(not bad, f"state snapshot {step} differs in {bad}")
        for step in (25, REPLAY_STEPS):
            name = f"validation_rank{rank}_step{step:06}.json"
            a, b = read(run / name), read(ref / name)
            require(a == b and a["status"] == "PASS" and a["model_state_unchanged"], "full validation differs")
        ranks.append({"rank": rank, "journal_rows": len(rows), "state_snapshots": REPLAY_STEPS + 1, "validations": 2})
    return ranks


# --------------------------------------------------------------------------- shared pieces of the run checks
def _steps(phase):
    return 50 if phase == "preflight" else 6000


def _complete(out, rank, steps, checkpoint_sha):
    done = read(out / f"completion_rank{rank}.json")
    require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
    require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")


def _indices(out, ref, rank, steps, what):
    import numpy as np
    actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
    expected = np.fromfile(ref / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
    require(actual.size == steps * 512 and np.array_equal(actual, expected), f"Sample indices differ from {what}")


def _initial(out, ref, rank, message):
    initial = read(out / f"state_rank{rank}_step000000.json")["model"]
    require(initial == read(ref / f"state_rank{rank}_step000000.json")["model"], message)


def _phase_files(out, arm, rank, phase):
    if phase == "preflight":
        for step in (25, 50):
            val = read(out / f"validation_rank{rank}_step{step:06}.json")
            require(val["status"] == "PASS" and val["model_state_unchanged"], "Full validation was not verified")
    else:
        for step in range(500, 6001, 500):
            require((out / f"{run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")


def _runtime(rt):
    require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == 3407, "Runtime changed")


def _finish(out, arm, phase, checkpoint_sha, result, message):
    import torch
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)), message)
    if phase == "formal":
        require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): sha(out / f"{run_name(arm)}-step={s}.ckpt") for s in (5000, 5500, 6000)}
    result["checkpoint_sha256"] = checkpoint_sha
    return result


def _packets_rank(out, ref, rank, steps, what):
    """Journal rows vs the reference's: index stream, small-field SHA and full-packet SHA; returns the number of full
    packets compared."""
    rows, ref_rows = journal(out, rank), journal(ref, rank, steps)
    require(len(rows) == steps, "Input journal denominator differs")
    small = full = full_compared = 0
    for row, old in zip(rows, ref_rows):
        require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
        small += row["small_fields_sha256"] != old["small_fields_sha256"]
        if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
            full_compared += 1
            full += row["full_packet_sha256"] != old["full_packet_sha256"]
    require(small == 0 and full == 0 and full_compared > 0, f"Training packets differ from {what}")
    return full_compared


def _routing(rt, cfg_state):
    got = {k: rt["s45sm_routing"][k] for k in ("router", "checkpoint", "assoc", "onhand_tau")}
    require(got == cfg_state, "Installed routing differs from the registered arm")
    require(rt["s45sm_router_checkpoint_sha256"] == sha(cfg_state["checkpoint"]), "Router checkpoint changed")


# --------------------------------------------------------------------------- the branches
def _verify_packets(arm, phase):
    import yaml
    spec = arm_spec(arm)
    require(spec["dz"], "round-1 arms are all dz-based")
    out, steps = output_dir(arm, phase), _steps(phase)
    cfg_state = expected_state(yaml.safe_load(config_path(arm, phase).read_text()))
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": recorded(out), "base_run": recorded(DZ_RUN), "ranks": []}
    for rank in range(2):
        _complete(out, rank, steps, checkpoint_sha)
        _indices(out, DZ_RUN, rank, steps, "S45-AT dz")
        _initial(out, DZ_RUN, rank, "Initial tensors differ from dz")
        compared = _packets_rank(out, DZ_RUN, rank, steps, "S45-AT dz (routes change in-model only)")
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(DZ_RUN / f"effective_runtime_rank{rank}.json")
        _runtime(rt)
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from dz")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from dz")
        _routing(rt, cfg_state)
        _phase_files(out, arm, rank, phase)
        result["ranks"].append({"rank": rank, "full_packets_compared": compared})
    return _finish(out, arm, phase, checkpoint_sha, result, "Nonfinite model")


def _verify_bank(arm, phase):
    import yaml
    cfg = yaml.safe_load(config_path(arm, phase).read_text())
    out, steps = output_dir(arm, phase), _steps(phase)
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": recorded(out), "base_run": recorded(DZ_RUN), "ranks": []}
    for rank in range(2):
        _complete(out, rank, steps, checkpoint_sha)
        _indices(out, DZ_RUN, rank, steps, "S45-AT dz")
        _initial(out, DZ_RUN, rank, "Initial tensors differ")
        rows, ref_rows = journal(out, rank), journal(DZ_RUN, rank, steps)
        require(len(rows) == steps and all(r["indices_sha256"] == o["indices_sha256"] for r, o in zip(rows, ref_rows)),
                "Index stream differs")
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(DZ_RUN / f"effective_runtime_rank{rank}.json")
        _runtime(rt)
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from dz")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from dz")
        b = cfg["S45SM"]["BANK"]
        require(rt["s45sm_bank"] == {"from": b["FROM"], "root": b["ROOT"], "p": float(b["P"])}, "Installed bank differs")
        _phase_files(out, arm, rank, phase)
        result["ranks"].append({"rank": rank, "packets_compared": False})
    return _finish(out, arm, phase, checkpoint_sha, result, "Nonfinite")


def _verify_dedicated(arm, phase):
    import yaml
    spec = arm_spec(arm)
    cfg = yaml.safe_load(config_path(arm, phase).read_text())
    out, steps = output_dir(arm, phase), _steps(phase)
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": recorded(out), "index_run": recorded(SAMP_RUN),
              "init_run": recorded(DZ_RUN), "ranks": []}
    for rank in range(2):
        _complete(out, rank, steps, checkpoint_sha)
        _indices(out, SAMP_RUN, rank, steps, "s45samp")
        rows, ref_rows = journal(out, rank), journal(SAMP_RUN, rank, steps)
        require(len(rows) == steps and all(r["indices_sha256"] == o["indices_sha256"] for r, o in zip(rows, ref_rows)),
                "Index stream differs from s45samp")
        _initial(out, DZ_RUN, rank, "Initial tensors differ")
        rt = read(out / f"effective_runtime_rank{rank}.json")
        samp_rt, dz_rt = read(SAMP_RUN / f"effective_runtime_rank{rank}.json"), read(DZ_RUN / f"effective_runtime_rank{rank}.json")
        _runtime(rt)
        require(rt["actual_train_loader"] == samp_rt["actual_train_loader"], "Loader contract differs from s45samp")
        require(rt["actual_math_flags"] == dz_rt["actual_math_flags"], "Strict math flags differ from dz")
        require(rt["s45sm_sampler"] == "dedicated", "Sampler not recorded as dedicated")
        require({k: rt["s45sm_routing"][k] for k in ("router", "checkpoint", "assoc", "onhand_tau")} == expected_state(cfg),
                "Installed routing differs")
        bank = cfg["S45SM"].get("BANK")
        want = None if bank is None else {"from": bank["FROM"], "root": bank["ROOT"], "p": float(bank["P"])}
        require(rt["s45sm_bank"] == want, "Installed bank differs")
        require(bool(rt.get("s45sm_fast")) == bool(spec.get("fast", False)), "Fast-path flag differs from the arm")
        _phase_files(out, arm, rank, phase)
        result["ranks"].append({"rank": rank, "packets_compared": False})
    return _finish(out, arm, phase, checkpoint_sha, result, "Nonfinite")


def _verify_oracle(arm, phase):
    from semkine import s45sm_oracle as O
    out, steps = output_dir(arm, phase), _steps(phase)
    checkpoint_sha = sha(out / "repro_final.ckpt")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": recorded(out), "base_run": recorded(T1DDZ_RUN), "ranks": []}
    for rank in range(2):
        _complete(out, rank, steps, checkpoint_sha)
        _indices(out, T1DDZ_RUN, rank, steps, "t1ddz")
        _initial(out, T1DDZ_RUN, rank, "Initial tensors differ")
        compared = _packets_rank(out, T1DDZ_RUN, rank, steps, "t1ddz")
        rt, ref_rt = read(out / f"effective_runtime_rank{rank}.json"), read(T1DDZ_RUN / f"effective_runtime_rank{rank}.json")
        _runtime(rt)
        require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from t1ddz")
        require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ from t1ddz")
        require(rt["s45sm_oracle"] == O.STATE, "Installed oracle association differs")
        _phase_files(out, arm, rank, phase)
        result["ranks"].append({"rank": rank, "full_packets_compared": compared})
    return _finish(out, arm, phase, checkpoint_sha, result, "Nonfinite model")


def _verify_cosine(arm, phase, result):
    base = arm_spec(arm)["schedule_of"]
    require(schedule(base) == "constant", "the base of a cosine arm runs the constant schedule")
    out, ref = output_dir(arm, phase), output_dir(base, phase)
    meta = read(out / "training_metadata.json")
    require(meta["lr_schedule"] == ("cosine" if phase == "formal" else "constant"), "recorded LR schedule differs")
    if phase == "preflight":
        replay_compare(out, ref)
        result["preflight_equals"] = base
    else:
        result["formal_equals_base_through_step"] = same_through(out, ref)
        result["schedule_base"] = base
    return result


def verify_run(arm, phase):
    """Engineering acceptance of one finished run (see the module docstring); raises RuntimeError on any difference."""
    require(phase in ("preflight", "formal"), "unregistered phase")
    if arm in oracles():
        return _verify_oracle(arm, phase)
    spec = arm_spec(arm)
    if sampler(arm) == "dedicated":
        result = _verify_dedicated(arm, phase)
    else:
        result = _verify_bank(arm, phase) if spec.get("bank") else _verify_packets(arm, phase)
        rt0 = read(output_dir(arm, phase) / "effective_runtime_rank0.json")
        require(bool(rt0.get("s45sm_fast")) == bool(spec.get("fast", False)), "Fast-path flag differs from the arm")
    if schedule(arm) == "cosine":
        result = _verify_cosine(arm, phase, result)
    return result


# --------------------------------------------------------------------------- the two receipt writers
def accept(arm, phase):
    """Verify the run and write receipts/<phase>_<arm>.json (exclusive); a failed check is recorded and raises."""
    require(arm in registry()["arms"] or arm in oracles(), "unregistered S45-SM arm " + arm)
    require(phase in phases(arm), f"phase {phase} is not registered for {arm}")
    verify_protocol()
    try:
        receipt = verify_run(arm, phase)
    except RuntimeError as exc:
        receipt = {"status": "FAIL", "arm": arm, "phase": phase, "error": str(exc)}
    receipt.update(wrapper_sha256=sha(PACKAGE / "train.py"), code_identity_sha256=code_identity(), verified_at_utc=now(),
                   engineering_only_no_accuracy_result=True)
    save(OUT / "receipts" / f"{phase}_{arm}.json", receipt, exclusive=True)
    require(receipt["status"] == "PASS", "acceptance failed: " + phase + " " + arm)
    print("PASS", phase, arm)


def replay(arm):
    """An arm registered with replay_of / phases [preflight]: its preflight must equal the replayed arm's bit for bit."""
    verify_protocol()
    base = replayed(arm)
    run, ref = output_dir(arm, "preflight"), output_dir(base, "preflight")
    receipt = {"arm": arm, "replay_of": base, "run": recorded(run), "reference": recorded(ref)}
    try:
        receipt.update(status="PASS", ranks=replay_compare(run, ref))
    except RuntimeError as exc:
        receipt.update(status="FAIL", error=str(exc))
    receipt.update(checker_sha256=sha(__file__), code_identity_sha256=code_identity(), verified_at_utc=now(),
                   engineering_only_no_accuracy_result=True)
    save(OUT / "receipts" / f"replay_{arm}.json", receipt, exclusive=True)
    require(receipt["status"] == "PASS", "replay differs: " + receipt.get("error", ""))
    print("PASS replay", arm, "=", base)
