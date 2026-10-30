#!/usr/bin/env python3
"""Fresh XCL training with the original C1 strict numerical and input contract."""
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
from tools.xyz_c1_repro import train as B
from tools.xcl.common import (BASELINE, CANONICAL, CELLS, OUT, SOURCE_MANIFEST,
                             path, read, require, save, sha, verify_protocol)

def validate_recipe(cfg, cell, phase):
    expected = json.loads(json.dumps(B.load_config(CANONICAL)))
    expected.pop("_config_path")
    assoc, gnn, geom, _ = CELLS[cell]
    expected["MODEL"].update(XAG_ASSOC=assoc, XAG_GNN=gnn, XYZ_GEOM=geom)
    expected["DATA"]["XAG_SOURCE_ROOT"] = str(ROOT / "outputs/xag/sources")
    expected["XCL"] = {"ENABLED": True, "CELL": cell}
    name = f"xcl_{cell}_s3407"
    if phase == "preflight":
        require(cfg["TRAIN"]["MAX_STEPS"] == 50 and cfg["TRAIN"]["VAL_CHECK_INTERVAL"] == 25,
                "Preflight requires 50 steps with full validation at25/50")
        require(path(cfg["TRAIN"]["OUTPUT_DIR"]).resolve().is_relative_to(OUT / "preflight"),
                "Preflight output outside XCL namespace")
        require(cfg["TRAIN"]["RUN_NAME"].startswith(name + "_preflight"), "Wrong preflight name")
        expected["TRAIN"].update(MAX_STEPS=50, VAL_CHECK_INTERVAL=25,
                                 RUN_NAME=cfg["TRAIN"]["RUN_NAME"], OUTPUT_DIR=cfg["TRAIN"]["OUTPUT_DIR"])
        expected["EVAL"]["OUTPUT_DIR"] = cfg["EVAL"]["OUTPUT_DIR"]
    else:
        expected["TRAIN"].update(RUN_NAME=name, OUTPUT_DIR=str(OUT / "runs" / name))
        expected["EVAL"]["OUTPUT_DIR"] = str(OUT / "runs" / name / "eval")
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected, "Unregistered difference from the C1 recipe")

def check_binding(options, config_path):
    p = verify_protocol(options.protocol)
    require(options.acceptance, "Formal training requires passed five-cell preflight")
    a = read(options.acceptance)
    require(a["status"] == "PASS" and a["protocol_sha256"] == sha(options.protocol), "Invalid preflight binding")
    require(set(a["cells"]) == set(CELLS), "All five cells must pass preflight")
    run = next(r for r in p["runs"] if r["cell"] == options.cell and not r["reuse"])
    require(path(run["config"]).resolve() == Path(config_path).resolve()
            and run["config_sha256"] == sha(config_path), "Registered formal config differs")
    require(a["wrapper_sha256"] == sha(__file__), "Training wrapper changed since preflight")
    require(a["depth_binding_sha256"] == sha(OUT / "depth_binding.json"), "Formal depth binding differs")

def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"cell": options.cell, "phase": options.phase,
                "config_sha256": sha(cfg["_config_path"]), "protocol_sha256": sha(options.protocol)}
    nonce = os.environ.get("XCL_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("XCL_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(not directory.is_symlink() and not claim.is_symlink(), "Claim path was replaced")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(XCL_CLAIM_NONCE=nonce, XCL_CLAIM_PARENT=str(os.getpid()))

class XCLAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        validate_recipe(model.cfg, self.options.cell, self.options.phase)
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(xcl_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__),
                 baseline_sha256=sha(BASELINE), source_manifest_sha256=sha(SOURCE_MANIFEST),
                 cell=self.options.cell, shared_packet_contract="XAG source fields removed; canonical C1 RAW_FIELDS unchanged")
        save(f, r)

    def on_validation_start(self, trainer, model):
        self.validation_batches = self.validation_samples = 0
        self.validation_before = B.normalized(model.state_dict())

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1
        self.validation_samples += batch.batch_size

    def on_validation_end(self, trainer, model):
        require(self.validation_batches == 126 and self.validation_samples == 64094,
                "Full validation denominator changed")
        same = self.validation_before == B.normalized(model.state_dict())
        require(same, "Validation changed model parameters/buffers")
        save(self.out / f"validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json", {
            "status": "PASS", "step": int(trainer.global_step), "batches": self.validation_batches,
            "samples_with_original_distributed_padding": self.validation_samples,
            "model_state_unchanged": same})

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cell", choices=tuple(CELLS), default=os.environ.get("XCL_CELL"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("XCL_PHASE"))
    parser.add_argument("--protocol", default=os.environ.get("XCL_PROTOCOL"))
    parser.add_argument("--acceptance", default=os.environ.get("XCL_ACCEPTANCE") or None)
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    options.sampler = "legacy_none"
    require(options.cell in CELLS and options.phase in ("preflight", "formal") and options.protocol, "Explicit XCL identity required")
    for key in ("cell", "phase", "protocol", "acceptance"):
        os.environ["XCL_" + key.upper()] = str(getattr(options, key) or "")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.cell, options.phase)
    protocol = verify_protocol(options.protocol)
    if options.cell == "a3f0":
        require((OUT / "depth_binding.json").is_file(), "A3 requires accepted immutable depth identity")
    require(sha(options.config) == protocol["file_sha256"][str(Path(options.config).resolve().relative_to(ROOT))], "Config not frozen")
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    torch, original, pl = B.torch, B.ORIGINAL, B.pl
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
    native_loader = original.DataLoader
    def loader(*args, **kwargs):
        if kwargs.get("shuffle") is True:
            kwargs["generator"] = None
        return native_loader(*args, **kwargs)
    original.DataLoader = loader
    native_trainer = pl.Trainer
    class Trainer(native_trainer):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False, enable_progress_bar=False,
                          strategy=B.DDPStrategy(find_unused_parameters=False, process_group_backend="nccl",
                            bucket_cap_mb=25, gradient_as_bucket_view=False, broadcast_buffers=True))
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), XCLAudit(options)]
            super().__init__(*args, **kwargs)
    original.pl.Trainer = Trainer
    source_fields = tuple(n for n in B.RAW_FIELDS if n.startswith("xag_"))
    def before_transfer(self, batch, dataloader_idx):
        if self.training:
            step = int(self.global_step) + 1
            selected = B.selected_step(step, options.phase, int(self.cfg["TRAIN"]["MAX_STEPS"]), int(self.trainer.num_training_batches))
            common = SimpleNamespace(**{n: None if n in source_fields else getattr(batch, n, None) for n in B.RAW_FIELDS})
            self._repro_cpu_fingerprint = {
                "step": step, "small_fields_sha256": B.packet_sha(batch, B.FIELDS),
                "full_packet_sha256": B.packet_sha(common, B.RAW_FIELDS) if selected else None,
                "xcl_source_fields_sha256": B.packet_sha(batch, (*source_fields, "xag_depth")) if selected else None,
                "events": int(batch.events.shape[0])}
        return batch
    original.MNISTModel.on_before_batch_transfer = before_transfer
    forward = original.MNISTModel.forward_packet
    def capture(self, batch, *args, **kwargs):
        result = forward(self, batch, *args, **kwargs)
        if self.training:
            self._repro_last_prediction = result.detach()
        return result
    original.MNISTModel.forward_packet = capture
    sys.argv = [sys.argv[0], "--config", options.config]
    original.main()

if __name__ == "__main__":
    main()
