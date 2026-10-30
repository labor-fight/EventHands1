#!/usr/bin/env python3
"""Fresh single-seed XAG retraining paired to the sealed strict C1 baseline.

Reuse the frozen C1 audit/runtime implementation. Only the two pre-existing
XAG model switches and their immutable GT proxy source fields are added. Native
semkine.train.main still constructs the dataset, model, objective and optimizer.
No resume, command-line recipe overrides, deterministic fallback or batch shrink.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
# Import sets the exact sealed environment before torch import. Its hash-seed
# re-exec re-enters THIS script, including native Lightning DDP child re-execs.
from tools.xyz_c1_repro import train as BASE

HELPER = ROOT / "tools/xyz_c1_repro/train.py"
HELPER_SHA = "c8034de8df1048ed53e62039854807b3526dc7894e8269b2660a8f3015921311"
CANONICAL_CONFIG = ROOT / "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"
CANONICAL_CONFIG_SHA = "2b0a2ec99d76c57360a57bcc0b2cd4e7e496cb7a60f60ce17c044852db7a91fc"
CANONICAL_BASELINE = ROOT / "outputs/xyz_c1_repro/baseline.json"
OUT = ROOT / "outputs/xag_paired/20261018"
SOURCE_ROOT = ROOT / "outputs/xag/sources"
SOURCE_MANIFEST = ROOT / "outputs/xag/source_manifest.json"
SOURCE_FIELDS = tuple(name for name in BASE.RAW_FIELDS if name.startswith("xag_"))
CELLS = {
    "a0f1": ("base", "global_residual"),
    "a1f0": ("part_route", "legacy"),
    "a1f1": ("part_route", "global_residual"),
    "a2f0": ("face_route", "legacy"),
    "a2f1": ("face_route", "global_residual"),
}


def validate_recipe(cfg, cell, phase):
    """Compare every resolved key with canonical; allow named changes only."""
    if BASE.file_sha(HELPER) != HELPER_SHA or BASE.file_sha(CANONICAL_CONFIG) != CANONICAL_CONFIG_SHA:
        raise ValueError("sealed C1 helper/config changed")
    expected = BASE.load_config(CANONICAL_CONFIG)
    expected.pop("_config_path")
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    # Deep copy by JSON keeps the caller's loaded config unmodified.
    expected = json.loads(json.dumps(expected))
    expected["MODEL"].update(zip(("XAG_ASSOC", "XAG_GNN"), CELLS[cell]))
    expected["DATA"]["XAG_SOURCE_ROOT"] = str(SOURCE_ROOT)
    name = f"xag_paired_{cell}_s3407"
    if phase == "formal":
        expected["TRAIN"]["RUN_NAME"] = name
        expected["TRAIN"]["OUTPUT_DIR"] = str(OUT / "runs" / name)
        expected["EVAL"]["OUTPUT_DIR"] = str(OUT / "runs" / name / "eval")
    else:
        budget = cfg["TRAIN"]["MAX_STEPS"]
        if budget not in (20, 50):
            raise ValueError("preflight is limited to 20/50 optimizer steps")
        val_interval = cfg["TRAIN"]["VAL_CHECK_INTERVAL"]
        if val_interval not in (20, 25, 50, 500):
            raise ValueError("unregistered preflight validation interval")
        if not cfg["TRAIN"]["RUN_NAME"].startswith(name):
            raise ValueError("preflight run identity differs from cell")
        for section in ("TRAIN", "EVAL"):
            if not Path(cfg[section]["OUTPUT_DIR"]).resolve().is_relative_to(OUT / "preflight"):
                raise ValueError("preflight outputs must stay in new preflight namespace")
            expected[section]["OUTPUT_DIR"] = cfg[section]["OUTPUT_DIR"]
        expected["TRAIN"].update(MAX_STEPS=budget, VAL_CHECK_INTERVAL=val_interval,
                                  RUN_NAME=cfg["TRAIN"]["RUN_NAME"])
    if actual != expected:
        def flatten(v, prefix=""):
            if isinstance(v, dict):
                return {key: item for k, x in v.items() for key, item in flatten(x, prefix + k + ".").items()}
            return {prefix[:-1]: v}
        a, e = flatten(actual), flatten(expected)
        different = sorted(k for k in set(a) | set(e) if a.get(k) != e.get(k))
        raise ValueError("unapproved canonical recipe differences: " + ", ".join(different))


def check_formal_binding(options, config_path):
    """New E -> P -> A graph; no edits or reuse of C1 launch acceptance."""
    if not options.acceptance:
        raise ValueError("new paired formal launch acceptance required")
    acceptance = json.loads(Path(options.acceptance).read_text())
    protocol = json.loads(Path(options.protocol).read_text())
    bindings = acceptance.get("bindings", {})
    key = str(Path(config_path).resolve())
    required = {
        "wrapper_sha256": BASE.file_sha(__file__),
        "helper_sha256": BASE.file_sha(HELPER),
        "canonical_config_sha256": BASE.file_sha(CANONICAL_CONFIG),
        "canonical_baseline_sha256": BASE.file_sha(CANONICAL_BASELINE),
        "source_manifest_sha256": BASE.file_sha(SOURCE_MANIFEST),
        "protocol_sha256": BASE.file_sha(options.protocol),
    }
    if acceptance.get("status") != "PASS" or any(bindings.get(k) != v for k, v in required.items()):
        raise ValueError("paired launch status/source/protocol/baseline binding mismatch")
    if (bindings.get("config_sha256", {}).get(key) != BASE.file_sha(config_path)
            or bindings.get("sampler_by_config", {}).get(key) != "legacy_none"
            or bindings.get("cell_by_config", {}).get(key) != options.cell
            or bindings.get("modes_by_config", {}).get(key) != list(CELLS[options.cell])):
        raise ValueError("paired config/sampler/cell/mode binding mismatch")
    evidence_sha = acceptance.get("preflight_evidence_sha256")
    if not evidence_sha or evidence_sha != protocol.get("preflight_acceptance_sha256"):
        raise ValueError("paired preflight evidence binding mismatch")
    evidence_path = protocol.get("preflight_evidence_path")
    if not evidence_path or BASE.file_sha(evidence_path) != evidence_sha:
        raise ValueError("paired preflight evidence physical file mismatch")
    if json.loads(Path(evidence_path).read_text()).get("status") != "PASS":
        raise ValueError("paired preflight evidence not PASS")


def fresh_output_claim(options, cfg):
    """Atomically own a new run directory before native main can write to it.

    A native rank1 child may join only its direct live parent's exact claim.
    The nonce uses OS entropy and never consumes Python/NumPy/torch RNG state.
    Existing directories/files/symlinks (including broken links) cannot be reused.
    """
    output = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim_path = output / ".fresh_claim.json"
    identity = {
        "schema": "xag-paired-fresh-output-v1", "output_dir": str(output),
        "cell": options.cell, "phase": options.phase, "sampler": options.sampler,
        "config_path": str(Path(cfg["_config_path"]).resolve()),
        "config_sha256": BASE.file_sha(cfg["_config_path"]),
        "protocol_path": str(Path(options.protocol).resolve()),
        "protocol_sha256": BASE.file_sha(options.protocol),
        "wrapper_sha256": BASE.file_sha(__file__),
    }
    nonce = os.environ.get("XAG_PAIRED_CLAIM_NONCE")
    if nonce:
        # Lightning's native two-rank launcher Popen is the direct parent of rank1.
        # A stale claim in an ordinary direct invocation must never bypass fresh.
        if (os.environ.get("LOCAL_RANK") != "1"
                or os.environ.get("XAG_PAIRED_CLAIM_PARENT_PID") != str(os.getppid())):
            raise ValueError("only native rank1 may inherit its direct parent's fresh-output claim")
        if output.is_symlink() or not output.is_dir() or claim_path.is_symlink():
            raise ValueError("owned fresh output/claim path changed")
        if os.environ.get("XAG_PAIRED_CLAIM_PATH") != str(claim_path):
            raise ValueError("inherited output claim path differs")
        if BASE.file_sha(claim_path) != os.environ.get("XAG_PAIRED_CLAIM_SHA256"):
            raise ValueError("inherited output claim bytes changed")
        claim = json.loads(claim_path.read_text())
        expected = {**identity, "nonce": nonce, "parent_pid": os.getppid()}
        if claim != expected:
            raise ValueError("inherited output claim config/protocol/parent identity differs")
        return claim
    if os.environ.get("LOCAL_RANK") not in (None, "0"):
        raise ValueError("rank1 requires its parent's explicit fresh-output claim")
    # lexists sees a dangling symlink; is_dir/exists alone would miss it.
    if os.path.lexists(output):
        raise FileExistsError("fresh-only output already exists: " + str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)  # atomic exclusion between competing parents
    claim = {**identity, "nonce": secrets.token_hex(32), "parent_pid": os.getpid()}
    with claim_path.open("x") as handle:
        handle.write(json.dumps(claim, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for key, value in (("NONCE", claim["nonce"]), ("PARENT_PID", str(claim["parent_pid"])),
                       ("PATH", str(claim_path)), ("SHA256", BASE.file_sha(claim_path))):
        os.environ["XAG_PAIRED_CLAIM_" + key] = value
    return claim


class PairedAudit(BASE.ReproAudit):
    def on_fit_start(self, trainer, model):
        validate_recipe(model.cfg, self.options.cell, self.options.phase)
        # BASE's formal hook resolves its binding function in BASE's globals.
        # Replace only that policy hook, preserving the frozen numerical callback.
        BASE.check_formal_binding = check_formal_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        # The native source audit writes its shared provenance on rank0 only.
        # All ranks must see that file before recording its exact byte hash.
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        path = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        receipt = json.loads(path.read_text())
        receipt.update(
            wrapper_sha256=BASE.file_sha(__file__), reused_helper_sha256=BASE.file_sha(HELPER),
            cell=self.options.cell, xag_modes=list(CELLS[self.options.cell]),
            canonical_baseline=str(CANONICAL_BASELINE),
            canonical_baseline_sha256=BASE.file_sha(CANONICAL_BASELINE),
            source_manifest_sha256=BASE.file_sha(SOURCE_MANIFEST),
            fresh_output_claim_path=os.environ["XAG_PAIRED_CLAIM_PATH"],
            fresh_output_claim_sha256=os.environ["XAG_PAIRED_CLAIM_SHA256"],
            full_packet_hash_contract="BASE.RAW_FIELDS with XAG fields None, identical to canonical C1; actual source fields separately hashed",
            original_xag_binding_boundary="xag_binding.json retains old source-implementation provenance; paired_binding_rank*.json is this new launch identity",
        )
        BASE.save(path, receipt)
        BASE.save(self.out / f"paired_binding_rank{trainer.global_rank}.json", {
            "cell": self.options.cell, "sampler": "legacy_none", "phase": self.options.phase,
            "config": model.cfg["_config_path"], "config_sha256": BASE.file_sha(model.cfg["_config_path"]),
            "protocol": self.options.protocol, "protocol_sha256": BASE.file_sha(self.options.protocol),
            "acceptance": self.options.acceptance,
            "acceptance_sha256": BASE.file_sha(self.options.acceptance) if self.options.acceptance else None,
            "launch_acceptance_sha256": BASE.file_sha(self.options.acceptance) if self.options.acceptance else None,
            "wrapper_sha256": BASE.file_sha(__file__), "helper_sha256": BASE.file_sha(HELPER),
            "reused_helper_sha256": BASE.file_sha(HELPER),
            "canonical_baseline_sha256": BASE.file_sha(CANONICAL_BASELINE),
            "source_manifest_sha256": BASE.file_sha(SOURCE_MANIFEST),
            "frozen_sources_sha256": BASE.file_sha(ROOT / "outputs/xag/frozen_sources.json"),
            "original_xag_source_binding": str(self.out / "xag_binding.json"),
            "original_xag_source_binding_sha256": BASE.file_sha(self.out / "xag_binding.json"),
            "fresh_output_claim_sha256": os.environ["XAG_PAIRED_CLAIM_SHA256"],
        })


def parse_options(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cell", required=not bool(os.environ.get("XAG_PAIRED_CELL")),
                        default=os.environ.get("XAG_PAIRED_CELL"), choices=tuple(CELLS))
    parser.add_argument("--phase", required=not bool(os.environ.get("XAG_PAIRED_PHASE")),
                        default=os.environ.get("XAG_PAIRED_PHASE"), choices=("preflight", "formal"))
    parser.add_argument("--sampler", choices=("legacy_none",), default="legacy_none")
    parser.add_argument("--protocol", required=not bool(os.environ.get("XAG_PAIRED_PROTOCOL")),
                        default=os.environ.get("XAG_PAIRED_PROTOCOL"))
    parser.add_argument("--acceptance", default=os.environ.get("XAG_PAIRED_ACCEPTANCE") or None)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main():
    options = parse_options()
    # Native main shrinks sys.argv to --config before Lightning spawns ranks.
    # Export only execution identity so every child restores the same new gate.
    for key, value in (("CELL", options.cell), ("PHASE", options.phase),
                       ("PROTOCOL", options.protocol), ("ACCEPTANCE", options.acceptance or "")):
        os.environ["XAG_PAIRED_" + key] = value
    if not Path(options.protocol).is_file():
        raise ValueError("existing explicit new paired protocol required")
    if Path(options.protocol).resolve() == (ROOT / "outputs/xyz_c1_repro/20261018/protocol.json"):
        raise ValueError("do not reuse C1 protocol for new paired experiment")
    cfg = BASE.load_config(options.config)
    validate_recipe(cfg, options.cell, options.phase)
    if options.phase == "formal":
        check_formal_binding(options, options.config)
    fresh_output_claim(options, cfg)

    torch, original, pl = BASE.torch, BASE.ORIGINAL, BASE.pl
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

    original_loader = original.DataLoader
    def controlled_loader(*args, **kwargs):
        if kwargs.get("shuffle") is True:
            kwargs["generator"] = None  # overrides the native XAG dedicated branch
        return original_loader(*args, **kwargs)
    original.DataLoader = controlled_loader

    original_trainer = pl.Trainer
    class ControlledTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            kwargs["deterministic"] = True
            kwargs["benchmark"] = False
            kwargs["strategy"] = BASE.DDPStrategy(
                find_unused_parameters=False, process_group_backend="nccl", bucket_cap_mb=25,
                gradient_as_bucket_view=False, broadcast_buffers=True)
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), PairedAudit(options)]
            kwargs["enable_progress_bar"] = False
            super().__init__(*args, **kwargs)
    original.pl.Trainer = ControlledTrainer

    def before_transfer(self, batch, dataloader_idx):
        if self.training:
            step = int(self.global_step) + 1
            raw = BASE.selected_step(step, options.phase, int(self.cfg["TRAIN"]["MAX_STEPS"]),
                                     int(self.trainer.num_training_batches))
            canonical_view = SimpleNamespace(**{
                name: None if name in SOURCE_FIELDS else getattr(batch, name, None) for name in BASE.RAW_FIELDS})
            self._repro_cpu_fingerprint = {
                "step": step, "small_fields_sha256": BASE.packet_sha(batch, BASE.FIELDS),
                "full_packet_sha256": BASE.packet_sha(canonical_view, BASE.RAW_FIELDS) if raw else None,
                "events": int(batch.events.shape[0]),
                "xag_full_packet_sha256": BASE.packet_sha(batch, BASE.RAW_FIELDS) if raw else None,
                "xag_source_fields_sha256": BASE.packet_sha(batch, SOURCE_FIELDS) if raw else None,
            }
        return batch
    original.MNISTModel.on_before_batch_transfer = before_transfer
    forward = original.MNISTModel.forward_packet
    def capture_prediction(self, batch, *args, **kwargs):
        result = forward(self, batch, *args, **kwargs)
        if self.training:
            self._repro_last_prediction = result.detach()
        return result
    original.MNISTModel.forward_packet = capture_prediction
    sys.argv = [sys.argv[0], "--config", options.config]
    original.main()


if __name__ == "__main__":
    main()
