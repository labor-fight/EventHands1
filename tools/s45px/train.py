#!/usr/bin/env python3
"""Fresh S45-PX training under S45's strict numerical and input contract.

Same policy as tools/s45tr/train.py (deterministic algorithms, legacy_none sampler, fixed DDP strategy, per-step input
journal / index tape / snapshots via tools/xyz_c1_repro/train.py's ReproAudit). The only difference from S45 is the
arm's added vertex position (semkine/s45px.py), installed in both DDP ranks before the model is built; its map is zero
at the start, so step 1 must reproduce S45's recorded step 1 bit for bit (checked by tools/s45px/accept.py).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xyz_c1_repro import train as B  # noqa: E402  (sets the strict environment, re-execs once)
import yaml  # noqa: E402
from tools.frozen import accepts  # noqa: E402
from tools.s45px.common import (ARMS, OUT, PROTOCOL, S45_CONFIG, S45_CONFIG_SHA256, S45TR, config_frozen,  # noqa: E402
                                config_path, expected_raw, install_hook, read, require, save, sha, verify_protocol)


def validate_recipe(cfg, arm, phase):
    require(accepts(S45_CONFIG, S45_CONFIG_SHA256), "S45 config changed")
    expected = expected_raw(arm, phase, yaml.safe_load(S45_CONFIG.read_text()))
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected, "Unregistered difference from S45's recipe + the registered S45-PX change")


def check_binding(options, config):
    from tools.s45tr import common as TR
    verify_protocol()
    config_frozen(config)
    for arm in ARMS:
        pre = read(OUT / "receipts" / f"preflight_{arm}.json")
        require(pre["status"] == "PASS" and pre["protocol_sha256"] == sha(PROTOCOL)
                and pre["s45_step1_exact_and_departs_at_step2"] is True, "Preflight not accepted: " + arm)
    require(accepts(__file__, read(OUT / "receipts" / f"preflight_{options.arm}.json")["wrapper_sha256"]),
            "Training wrapper changed since preflight")
    TR.verify_protocol()
    replay = read(S45TR / "receipts" / "preflight_s45replay.json")
    require(replay["status"] == "PASS" and replay["s45_code_replay_exact"] is True,
            "S45-TR P0 (current code replays S45 training exactly) not on record")


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": sha(cfg["_config_path"]),
                "protocol_sha256": sha(PROTOCOL)}
    nonce = os.environ.get("S45PX_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("S45PX_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(S45PX_CLAIM_NONCE=nonce, S45PX_CLAIM_PARENT=str(os.getpid()))


class S45PXAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        from semkine import s45px
        validate_recipe(model.cfg, self.options.arm, self.options.phase)
        require(type(model.xyz) is s45px.PXXYZMeshNet and model.xyz.px_frame == ARMS[self.options.arm]
                and s45px.installed() == ARMS[self.options.arm], "arm did not build its PXXYZMeshNet")
        self.zero_map = bool(float(model.xyz.sign_in.weight.detach().abs().max()) == 0.0)
        require(self.zero_map, "the added map is not zero at the start")
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(s45px_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__), arm=self.options.arm,
                 s45px_frame=model.xyz.px_frame, s45px_new_map_zero_at_fit_start=self.zero_map,
                 s45px_hook_sha256=sha(ROOT / "semkine/s45px.py"),
                 xyz_class=type(model.xyz).__module__ + "." + type(model.xyz).__name__)
        save(f, r)

    def on_validation_start(self, trainer, model):
        self.validation_batches = self.validation_samples = 0
        self.validation_before = B.normalized(model.state_dict())

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        self.validation_batches += 1
        self.validation_samples += batch.batch_size

    def on_validation_end(self, trainer, model):
        require(self.validation_batches == 126 and self.validation_samples == 64094, "Full validation denominator changed")
        same = self.validation_before == B.normalized(model.state_dict())
        require(same, "Validation changed model parameters/buffers")
        save(self.out / f"validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json", {
            "status": "PASS", "step": int(trainer.global_step), "batches": self.validation_batches,
            "samples_with_original_distributed_padding": self.validation_samples, "model_state_unchanged": same})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=tuple(ARMS), default=os.environ.get("S45PX_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("S45PX_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    options.sampler = "legacy_none"
    options.protocol = str(PROTOCOL)
    options.acceptance = None
    require(options.arm in ARMS and options.phase in ("preflight", "formal"), "Explicit S45-PX identity required")
    os.environ.update(S45PX_ARM=options.arm, S45PX_PHASE=options.phase)
    require(Path(options.config).resolve() == config_path(options.arm, options.phase).resolve(), "Unregistered config path")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    verify_protocol()
    config_frozen(options.config)
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    install_hook(cfg)
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
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), S45PXAudit(options)]
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
