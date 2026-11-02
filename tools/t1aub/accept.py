#!/usr/bin/env python3
"""Engineering acceptance of a finished T1A-UB training (CPU only; never scores).

    python tools/t1aub/accept.py --arm A --phase {preflight,formal}     -> outputs/t1aub/20261102/receipts/<phase>_<arm>.json

Checked against the recorded training the arm must equal (T1a's preflight / formal run, S45's run for the initial tensors
of an arm without the event graph, S45-AT s45samp's run for the index stream of a `dedicated` arm):
  completion and the final checkpoint; per-rank sample indices; initial model tensors equal to the base's except the
  arm's registered keys; the input journal (index SHA every step; small-field SHA every step unless `dz` changes the
  augmentation; full-packet SHA where recorded for same-input arms); loader contract and strict math flags; full
  validations (preflight) or the fixed checkpoint grid (formal).
A replay arm (`replay_of: t1a`) must equal T1a's preflight bit for bit: every journal row (predictions, objectives,
gradients), all 51 state snapshots and both validations, per rank.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402

STATE_KEYS = ("step", "model", "optimizers", "schedulers", "torch_cpu_rng", "torch_cuda_rng", "input_stream_sha256",
              "batches")


def journal(directory, rank, steps=None):
    rows = [json.loads(s) for s in (Path(directory) / f"batches_rank{rank}.jsonl").read_text().splitlines()]
    return rows if steps is None else rows[:steps]


def exemptions(spec):
    keys = []
    if spec.get("readout"):
        keys += ["xyz.slot_proj.", "xyz.fc1.", "xyz.fc2."]
    if spec.get("gtfeat"):
        keys += ["xyz.rel.0.weight"]
    return tuple(keys)


def bases(arm, phase):
    """(index/journal base run, initial-tensor base run, loader base run)."""
    spec = C.arm_spec(arm)
    dedicated = C.sampler(arm) == "dedicated"
    stream = C.S45AT_SAMP_RUN if dedicated else (C.T1A_PREFLIGHT if phase == "preflight" else C.T1A_RUN)
    init = C.T1A_RUN if spec.get("eg", True) else C.S45_RUN
    return stream, init, stream


def check_initial(out, init, rank, allowed):
    initial = C.read(out / f"state_rank{rank}_step000000.json")["model"]
    base = C.read(init / f"state_rank{rank}_step000000.json")["model"]

    def exempt(key):
        return any(key == a or (a.endswith(".") and key.startswith(a)) for a in allowed)
    shared = {k for k in base if not exempt(k)}
    C.require(shared <= set(initial), "base tensors missing from the arm's initial state")
    bad = sorted(k for k in shared if initial[k] != base[k])
    C.require(not bad, "initial tensors differ from the base: " + ", ".join(bad[:5]))
    extra = sorted(k for k in initial if k not in base)
    C.require(all(exempt(k) for k in extra), "unregistered new tensors: " + ", ".join(extra[:5]))
    changed = sorted(k for k in base if exempt(k) and k in initial)
    return {"new_tensors": extra, "registered_tensors_present": changed}


def replay_compare(out, ref):
    ranks = []
    for rank in range(2):
        rows, ref_rows = journal(out, rank), journal(ref, rank)
        C.require(len(rows) == len(ref_rows) == 50, "replay journal length differs")
        bad = [r.get("step") for r, o in zip(rows, ref_rows) if r != o]
        C.require(not bad, f"replay journal rows differ at steps {bad[:5]}")
        for step in range(51):
            a = C.read(out / f"state_rank{rank}_step{step:06}.json")
            b = C.read(ref / f"state_rank{rank}_step{step:06}.json")
            diff = [k for k in STATE_KEYS if a.get(k) != b.get(k)]
            C.require(not diff, f"replay state snapshot {step} differs in {diff}")
        for step in (25, 50):
            a = C.read(out / f"validation_rank{rank}_step{step:06}.json")
            b = C.read(ref / f"validation_rank{rank}_step{step:06}.json")
            C.require(a == b and a["status"] == "PASS", "replay validation differs")
        ranks.append({"rank": rank, "journal_rows": 50, "state_snapshots": 51, "validations": 2})
    return ranks


def verify(arm, phase):
    import numpy as np
    import torch
    spec = C.arm_spec(arm)
    C.require(phase in C.phases(arm), "phase not registered")
    out = C.output_dir(arm, phase)
    steps = 50 if phase == "preflight" else 6000
    stream, init, loader_base = bases(arm, phase)
    checkpoint_sha = C.sha(out / "repro_final.ckpt")
    same_packets = not spec.get("dz") and not (spec.get("gtfeat") or (spec.get("route") or {}).get("router") == "gt")
    same_small = not spec.get("dz")
    result = {"status": "PASS", "arm": arm, "phase": phase, "output": str(out), "stream_base": str(stream),
              "init_base": str(init), "ranks": []}
    for rank in range(2):
        done = C.read(out / f"completion_rank{rank}.json")
        C.require(done["status"] == "COMPLETE" and done["step"] == done["batches"] == steps, "Incomplete rank")
        C.require(done["final_checkpoint_sha256"] == checkpoint_sha, "Final checkpoint SHA differs from completion")
        actual = np.fromfile(out / f"indices_rank{rank}.i32", dtype=np.int32)
        ref = np.fromfile(stream / f"indices_rank{rank}.i32", dtype=np.int32)[:steps * 512]
        C.require(actual.size == steps * 512 and np.array_equal(actual, ref), "Sample indices differ from the stream base")
        init_info = check_initial(out, init, rank, exemptions(spec))
        rows, ref_rows = journal(out, rank), journal(stream, rank, steps)
        C.require(len(rows) == steps, "Input journal denominator differs")
        small = full = full_compared = 0
        for row, old in zip(rows, ref_rows):
            C.require(row["step"] == old["step"] and row["indices_sha256"] == old["indices_sha256"], "Index stream differs")
            small += row["small_fields_sha256"] != old["small_fields_sha256"]
            if old.get("full_packet_sha256") is not None and row.get("full_packet_sha256") is not None:
                full_compared += 1
                full += row["full_packet_sha256"] != old["full_packet_sha256"]
        if same_small:
            C.require(small == 0, "training packets (small fields) differ from the stream base")
        if same_packets:
            C.require(full == 0 and full_compared > 0, "full training packets differ from the stream base")
        rt = C.read(out / f"effective_runtime_rank{rank}.json")
        ref_rt = C.read(loader_base / f"effective_runtime_rank{rank}.json")
        C.require(rt["world_size"] == 2 and rt["precision"] == "bf16" and rt["effective_seed"] == C.SEED, "Runtime changed")
        C.require(rt["actual_train_loader"] == ref_rt["actual_train_loader"], "Loader contract differs from the stream base")
        C.require(rt["actual_math_flags"] == ref_rt["actual_math_flags"], "Strict math flags differ")
        C.require(rt["code_identity_sha256"] == C.code_identity(), "T1A-UB code changed since this run started")
        if phase == "preflight":
            for step in (25, 50):
                val = C.read(out / f"validation_rank{rank}_step{step:06}.json")
                C.require(val["status"] == "PASS" and val["batches"] == C.VALIDATION["batches"]
                          and val["model_state_unchanged"], "Full validation was not verified")
            C.require(rows[25]["step"] == 26, "Training did not return after validation")
        else:
            for step in range(500, 6001, 500):
                C.require((out / f"{C.run_name(arm)}-step={step}.ckpt").is_file(), "Missing fixed checkpoint grid")
        result["ranks"].append({"rank": rank, "packets_small_fields_differing": small, "packets_full_differing": full,
                                "packets_full_compared": full_compared, **init_info,
                                "params_model": rt.get("params_model")})
    obj = torch.load(out / "repro_final.ckpt", map_location="cpu")
    C.require(all(bool(torch.isfinite(v).all()) for v in obj["state_dict"].values() if torch.is_tensor(v)),
              "Nonfinite final model")
    if phase == "formal":
        C.require(obj["global_step"] == 6000, "Final checkpoint step differs")
        result["checkpoint_sha256_by_step"] = {str(s): C.sha(out / f"{C.run_name(arm)}-step={s}.ckpt") for s in C.STEPS}
    if spec.get("replay_of"):
        result["replay"] = replay_compare(out, C.T1A_PREFLIGHT)
        result["t1a_replay_exact"] = True
    result.update(checkpoint=str(out / "repro_final.ckpt"), checkpoint_sha256=checkpoint_sha,
                  wrapper_sha256=C.sha(ROOT / "tools/t1aub/train.py"), code_identity_sha256=C.code_identity(),
                  verified_at_utc=C.now(), engineering_only_no_accuracy_result=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args(argv)
    C.verify_protocol()
    dst = C.OUT / "receipts" / f"{a.phase}_{a.arm}.json"
    C.require(not dst.exists(), "receipt exists: " + str(dst))
    try:
        result = verify(a.arm, a.phase)
    except Exception as exc:
        C.save(C.OUT / "receipts" / f"{a.phase}_{a.arm}.FAILED.json",
               {"status": "FAILED", "arm": a.arm, "phase": a.phase, "error": f"{type(exc).__name__}: {exc}",
                "verified_at_utc": C.now()})
        raise
    C.save(dst, result, exclusive=True)
    print("PASS", dst, flush=True)


if __name__ == "__main__":
    main()
