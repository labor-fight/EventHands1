#!/usr/bin/env python3
"""Fresh S45-AT training under S45's strict numerical and input contract (docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031).

Same policy as tools/s45ct/train.py and tools/s45rt/train.py (deterministic algorithms, fixed DDP strategy, per-step
input journal / index tape / snapshots via tools/xyz_c1_repro/train.py's ReproAudit, the XAG provenance module
registered under the retired `tools.xag.train_audit` name). The only differences from the arm's base (S45 or S45-TR
T1d) are its registered config change, the t1dnoylf training-split hook (tools/s45at/common.py install_train_hook),
and for s45samp the dedicated sampling stream (`torch.Generator().manual_seed(3407)` for the shuffled loader, as the
strict wrapper's `dedicated` sampler).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xyz_c1_repro import train as B  # noqa: E402  (sets the strict environment, re-execs once)
import yaml  # noqa: E402
from tools.frozen import accepts  # noqa: E402
from tools.s45at.common import (ARMS, OUT, PROTOCOL, XAG_ARMS, base_config, config_frozen, config_path,  # noqa: E402
                                expected_raw, install_train_hook, now, read, require, save, sha, verify_protocol)

VALIDATION = {"batches": 126, "samples": 64094}


class XAGTrainAudit(B.pl.Callback):
    """Provenance only: which XAG source build the run read; never changes training."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            root = model.cfg["DATA"].get("XAG_SOURCE_ROOT")
            save(self.output_dir / "xag_binding.json", {
                "arm": model.cfg["S45AT"]["ARM"], "config_sha256": sha(model.cfg["_config_path"]),
                "xag_assoc": model.cfg["MODEL"].get("XAG_ASSOC"), "xag_gnn": model.cfg["MODEL"].get("XAG_GNN"),
                "xag_source_root": root,
                "xag_source_build_summary_sha256": sha(Path(root) / "build_summary.json") if root else None,
                "recorded_at_utc": now()})


def write(path, rows):
    save(path, rows)


def register():
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module


def validate_recipe(cfg, arm, phase):
    expected = expected_raw(arm, phase, yaml.safe_load(base_config(arm).read_text()))
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected, "Unregistered difference from the base recipe + the registered S45-AT change")


def check_binding(options, config):
    verify_protocol()
    config_frozen(config)
    pre = read(OUT / "receipts" / f"preflight_{options.arm}.json")
    require(pre["status"] == "PASS" and pre["protocol_sha256"] == sha(PROTOCOL), "Arm preflight not accepted")
    require(accepts(__file__, pre["wrapper_sha256"]), "Training wrapper changed since preflight")


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": sha(cfg["_config_path"]),
                "protocol_sha256": sha(PROTOCOL)}
    nonce = os.environ.get("S45AT_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("S45AT_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(S45AT_CLAIM_NONCE=nonce, S45AT_CLAIM_PARENT=str(os.getpid()))


class S45ATAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        arm = self.options.arm
        validate_recipe(model.cfg, arm, self.options.phase)
        require(type(model.xyz).__module__ == "semkine.xyz_mesh" and type(model.xyz).__name__ == "XYZMeshNet",
                "S45-AT arms build S45's XYZMeshNet")
        if arm in XAG_ARMS:
            require(model.xag_assoc_mode == "face_route", "T1d-based arms route with the XAG face route")
        require(self.options.excluded == ARMS[arm].get("exclude_subject"), "training-split hook differs")
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(s45at_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__), s45at_arm=self.options.arm,
                 s45at_excluded=self.options.excluded, s45at_sampler=self.options.sampler,
                 s45at_train_samples=int(len(trainer.train_dataloader.dataset))
                 if hasattr(trainer.train_dataloader, "dataset") else None)
        save(f, r)

    def on_validation_start(self, trainer, model):
        self.validation_batches = self.validation_samples = 0
        self.validation_before = B.normalized(model.state_dict())

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1
        self.validation_samples += batch.batch_size

    def on_validation_end(self, trainer, model):
        if self.options.arm != "w200":
            require(self.validation_batches == VALIDATION["batches"] and self.validation_samples == VALIDATION["samples"],
                    "Full validation denominator changed")
        else:
            require(self.validation_batches > 0, "Empty validation")
        same = self.validation_before == B.normalized(model.state_dict())
        require(same, "Validation changed model parameters/buffers")
        save(self.out / f"validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json", {
            "status": "PASS", "step": int(trainer.global_step), "batches": self.validation_batches,
            "samples_with_original_distributed_padding": self.validation_samples, "model_state_unchanged": same})


def main():
    register()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=tuple(ARMS), default=os.environ.get("S45AT_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("S45AT_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    require(options.arm in ARMS and options.phase in ("preflight", "formal"), "Explicit S45-AT identity required")
    options.sampler = ARMS[options.arm].get("sampler", "legacy_none")
    options.protocol = str(PROTOCOL)
    options.acceptance = None
    os.environ.update(S45AT_ARM=options.arm, S45AT_PHASE=options.phase)
    require(Path(options.config).resolve() == config_path(options.arm, options.phase).resolve(), "Unregistered config path")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    verify_protocol()
    config_frozen(options.config)
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    options.excluded = install_train_hook(cfg)
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
            kwargs["generator"] = None if options.sampler == "legacy_none" else torch.Generator().manual_seed(3407)
        return native_loader(*args, **kwargs)
    original.DataLoader = loader
    native_trainer = pl.Trainer

    class Trainer(native_trainer):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False, enable_progress_bar=False,
                          strategy=B.DDPStrategy(find_unused_parameters=False, process_group_backend="nccl",
                                                 bucket_cap_mb=25, gradient_as_bucket_view=False, broadcast_buffers=True))
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), S45ATAudit(options)]
            super().__init__(*args, **kwargs)
    original.pl.Trainer = Trainer

    def before_transfer(self, batch, dataloader_idx):
        if self.training:
            step = int(self.global_step) + 1
            selected = B.selected_step(step, options.phase, int(self.cfg["TRAIN"]["MAX_STEPS"]),
                                       int(self.trainer.num_training_batches))
            self._repro_cpu_fingerprint = {
                "step": step, "small_fields_sha256": B.packet_sha(batch, B.FIELDS),
                "full_packet_sha256": B.packet_sha(batch, B.RAW_FIELDS) if selected else None,
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
