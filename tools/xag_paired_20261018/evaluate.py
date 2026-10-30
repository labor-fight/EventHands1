#!/usr/bin/env python3
"""Original zgz scoring and full-model costs, with explicit new-run identity adapters.

The registered input directory, XAG registration/contract guard, paired binding
read and final output path are adapted in a bounded context. The original model,
packet builder, scoring, latency anchor/timing and MAC/parameter functions remain
unchanged. No old registry, alias, binding or result file is written.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xag_paired_20261018 import verify_final as V

POLICY_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "3407",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
              "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1"}


def configure_environment():
    previous = os.environ.get("PYTHONHASHSEED")
    os.environ.update(POLICY_ENV)
    if previous != "3407":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


def set_policy():
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def math_flags():
    import torch
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}


class RunInput:
    """Read the new paired binding where the original tool requests its binding."""
    def __init__(self, directory):
        self.directory = directory
        self.binding_reads = 0

    @property
    def name(self):
        return self.directory.name

    def glob(self, pattern):
        return self.directory.glob(pattern)

    def __truediv__(self, name):
        if str(name) == "xag_binding.json":
            self.binding_reads += 1
            return self.directory / "paired_binding_rank0.json"
        return self.directory / name


class EvaluationRoot:
    def __init__(self, run, output):
        self.run = run
        self.input = RunInput(V.path(run["output"]))
        self.output = output
        self.output_redirects = 0
        self.input_redirects = 0

    def __truediv__(self, name):
        name = str(name)
        if name == f"outputs/semkine/{self.run['name']}":
            self.input_redirects += 1
            return self.input
        if name == f"outputs/semkine/{self.run['prefix']}_main_row_step6000.json":
            self.output_redirects += 1
            return self.output
        return ROOT / name


@contextmanager
def registration_context(original, protocol_path, protocol, run, output):
    # Import the old modules before temporarily adapting their explicit guards;
    # their original OUT and all disk files remain preserved and are restored.
    from tools.xag import prepare as old_prepare
    from tools.xag import run as old_run
    saved = original.REPO, old_prepare.OUT, old_prepare.verify, old_run.check_checkpoint
    redirect = EvaluationRoot(run, output)
    protocol_digest = V.sha(protocol_path)
    def guard():
        V.require(V.sha(protocol_path) == protocol_digest, "Paired protocol changed during evaluation")
        V.verify_protocol(protocol_path)
    def checkpoint_guard(registered, record=True):
        V.require(record is False and registered == run, "Unexpected evaluator registration request")
        guard()
        V.check_checkpoint(protocol_path, protocol, registered)
    try:
        original.REPO = redirect
        old_prepare.OUT = Path(protocol_path).parent
        old_prepare.verify = guard
        old_run.check_checkpoint = checkpoint_guard
        yield redirect
    finally:
        original.REPO, old_prepare.OUT, old_prepare.verify, old_run.check_checkpoint = saved


def check_row(row, run, checkpoint):
    V.require(set(row["per_seed"]) == {"3407"}, "Expected one seed3407 row")
    seed = row["per_seed"]["3407"]
    V.require(seed["step"] == 6000 and seed["n_frames"] == 2590
              and seed["sequences"] == ["zgz_global", "zgz_local"]
              and set(seed["per_seq"]) == {"zgz_global", "zgz_local"}
              and Path(seed["ckpt"]).resolve() == checkpoint.resolve(), "Incomplete or mismatched original zgz row")
    for block in ("overall", "local", "global"):
        V.require(seed[block] == row["two_seed_mean"][block]
                  and all(math.isfinite(seed[block][key]) for key in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")),
                  "Nonfinite or non-single-seed accuracy")
    for seq, block, count in (("zgz_global", "global", 1386), ("zgz_local", "local", 1204)):
        V.require(all(value * count / count == seed[block][key] for key, value in seed["per_seq"][seq].items()),
                  "Per-sequence and original weighted local/global metrics differ")
    V.require(row["step_policy"] == row["protocol"]["step_policy"] == "step6000"
              and row["protocol"]["split"] == "val_core" and row["protocol"]["subject"] is None
              and row["protocol"]["step_ms"] == 50 and row["protocol"]["manifest"] == "splits_semkine.json",
              "Original full zgz reporting protocol changed")
    V.require(seed["xag_binding"]["training_binding"] == V.read(V.path(run["output"]) / "paired_binding_rank0.json"),
              "Original row does not identify the new paired training")
    V.require(row["latency_packets"] == 600 and all(math.isfinite(row[k]) and row[k] > 0 for k in (
        "latency_ms_scaled_full1p75", "latency_ms_raw", "anchor_full_raw_ms", "macs_forward_packet", "params_total")),
        "Full-model original cost contract incomplete")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True); parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, default=6000); parser.add_argument("--seeds", default="3407")
    parser.add_argument("--split", default="val_core"); parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    V.require(args.step == 6000 and args.seeds == "3407" and args.split == "val_core", "Only single3407 fixed6000 complete zgz is allowed")
    protocol_path = V.path(args.protocol).resolve(); protocol = V.verify_protocol(protocol_path)
    run = V.resolve_run(protocol, args.run)
    output_dir = V.path(args.output_dir).resolve()
    V.require(output_dir.is_relative_to(V.OUT) and not args.output_dir.is_symlink(), "Write only in new paired study")
    output = output_dir / f"{run['prefix']}_main_row_step6000.json"
    receipt_path = output.with_name(output.stem + "_receipt.json")
    V.require(not any(p.exists() or p.is_symlink() for p in (output, receipt_path)), "Preserve existing row and receipt")
    configure_environment()
    set_policy()
    # Full input/init/optimizer contract is checked before any GPU evaluation.
    contract = V.verify_run(protocol_path, protocol, run)
    checkpoint, _ = V.check_checkpoint(protocol_path, protocol, run)
    from tools import make_s36_row as original
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_argv = sys.argv
    try:
        with registration_context(original, protocol_path, protocol, run, output) as redirect:
            sys.argv = [str(ROOT / "tools/make_s36_row.py"), "--run", run["prefix"], "--step", "6000", "--seeds", "3407", "--split", "val_core"]
            original.main()
            V.require(redirect.output_redirects == redirect.input_redirects == redirect.input.binding_reads == 1,
                      "Original input/binding/final-write adaptation was not unique")
    finally:
        sys.argv = previous_argv
    row = V.read(output); check_row(row, run, checkpoint)
    V.verify_protocol(protocol_path)
    flags = math_flags(); V.require(flags == V.MATH, "Strict math flags changed during original evaluation")
    receipt = {"status": "COMPLETE", "runtime_policy": "XAG_PAIRED_C1_STRICT_20261018", "seed": 3407, "step": 6000,
               "row": str(output.relative_to(ROOT)), "row_sha256": V.sha(output), "wrapper_sha256": V.sha(__file__),
               "original_evaluator_sha256": V.sha(ROOT / "tools/make_s36_row.py"),
               "original_reporter_sha256": V.sha(ROOT / "tools/report_table.py"),
               "protocol_sha256": V.sha(protocol_path), "checkpoint_sha256": V.sha(checkpoint),
               "config_sha256": V.sha(V.path(run["config"])), "source_manifest_sha256": protocol["source_manifest_sha256"],
               "old_source_binding_sha256": V.sha(V.path(run["output"]) / "xag_binding.json"),
               "paired_training_binding_sha256": V.sha(V.path(run["output"]) / "paired_binding_rank0.json"),
               "contract_verification_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
               "environment": {k: os.environ[k] for k in POLICY_ENV}, "math_flags": flags, "output_redirection_count": 1,
               "adaptation": {"new_run_directory": run["output"], "new_registered_protocol": str(protocol_path),
                              "binding_read": "xag_binding.json request -> paired_binding_rank0.json read-only; old source binding preserved",
                              "scoring_cost_model_packet_functions_changed": False},
               "oracle": run["cell"].startswith(("a1", "a2")),
               "scope": "Original full2590 recursive accuracy and newly measured strict full-model latency/MAC/params; single seed, no score selection"}
    V.save_new(receipt_path, receipt)
    print("paired original-evaluator receipt", receipt_path)


if __name__ == "__main__":
    main()
