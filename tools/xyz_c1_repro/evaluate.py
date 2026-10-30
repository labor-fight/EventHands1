#!/usr/bin/env python3
"""Run the original main-row evaluator with an explicit deterministic runtime.

Only the evaluator's final output path is redirected. Checkpoint identity,
packet construction, scoring, latency and MAC calculation use the original
tools.make_s36_row functions. Existing rows are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

POLICY_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "3407",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1",
}
previous_hash_seed = os.environ.get("PYTHONHASHSEED")
os.environ.update(POLICY_ENV)
if previous_hash_seed != "3407":
    os.execv(sys.executable, [sys.executable, *sys.argv])

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.chdir(ROOT)

import torch
from tools import make_s36_row as original


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_policy():
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


class EvaluationRoot:
    """Redirect exactly the final JSON write; leave every input path intact."""

    def __init__(self, relative_output, output):
        self.relative_output = relative_output
        self.output = output
        self.redirects = 0

    def __truediv__(self, name):
        if str(name) == self.relative_output:
            self.redirects += 1
            return self.output
        return ROOT / name


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    options, remaining = parser.parse_known_args()
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--run", required=True)
    check.add_argument("--step", type=original.step_policy_arg, default="selection")
    check.add_argument("--split", default="val_core")
    check.add_argument("--subject", default=None)
    check.add_argument("--seeds", default="3407")
    args, _ = check.parse_known_args(remaining)
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.run):
        raise ValueError("Run must be a plain registered prefix")
    if args.seeds != "3407" or args.step != 6000 or args.split != "val_core" or args.subject:
        raise ValueError("This release uses only seed3407 fixed6000 full zgz")
    options.output_dir = options.output_dir.resolve()
    allowed = [ROOT / "outputs/xyz_c1_repro/20261018",
               ROOT / "outputs/xag/reviews/reconfirm_metrics_20261018"]
    if not any(options.output_dir.is_relative_to(parent) for parent in allowed):
        raise ValueError("Write only inside the new dated reproduction/reconfirmation directories")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    tag = original.row_tag(args.split, args.subject, args.step)
    name = f"{args.run}_main_row{tag}.json"
    output = options.output_dir / name
    receipt_path = options.output_dir / (name.removesuffix(".json") + "_receipt.json")
    if output.exists() or output.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Preserve prior evaluation; choose a fresh output directory")
    set_policy()
    redirect = EvaluationRoot(f"outputs/semkine/{name}", output)
    original.REPO = redirect
    sys.argv = [str(ROOT / "tools/make_s36_row.py"), *remaining]
    original.main()
    if redirect.redirects != 1 or not output.is_file():
        raise RuntimeError("Original evaluator final-write redirection was not unique")
    row = json.loads(output.read_text())
    if set(row["per_seed"]) != {"3407"} or row["per_seed"]["3407"]["step"] != 6000:
        raise RuntimeError("Unexpected evaluation identity")
    seed = row["per_seed"]["3407"]
    if (seed["sequences"] != ["zgz_global", "zgz_local"] or seed["n_frames"] != 2590
            or set(seed["per_seq"]) != {"zgz_global", "zgz_local"}):
        raise RuntimeError("Evaluation did not cover both complete zgz sequences")
    for block in ("overall", "local", "global"):
        for metric in original.METRICS:
            if not math.isfinite(seed[block][metric]):
                raise FloatingPointError("Nonfinite accuracy is not a verified result")
    receipt = {
        "status": "COMPLETE", "runtime_policy": "XYZ_C1_REPRO_20261018_STRICT",
        "row": str(output.relative_to(ROOT)), "row_sha256": sha(output),
        "wrapper_sha256": sha(__file__),
        "original_evaluator_sha256": sha(ROOT / "tools/make_s36_row.py"),
        "original_reporter_sha256": sha(ROOT / "tools/report_table.py"),
        "output_redirection_count": redirect.redirects,
        "seed": 3407, "step": 6000, "torch": torch.__version__,
        "environment": {k: os.environ[k] for k in POLICY_ENV},
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "accuracy_and_cost": "Both generated by unchanged make_s36_row functions under the declared strict runtime",
        "preservation": "Only the original evaluator final JSON output path was redirected; old rows unchanged",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("strict evaluation receipt", receipt_path)


if __name__ == "__main__":
    main()
