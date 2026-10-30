#!/usr/bin/env python3
"""Fresh C1SF training under the canonical C1 strict numerical and input contract.

Same wrapper policy as tools/xcl/train.py (deterministic algorithms, legacy_none sampler, fixed DDP strategy,
per-step input journal / index tape / snapshots via tools/xyz_c1_repro/train.py's ReproAudit). The only
difference from C1 is the arm's registered config change; arm ZR also routes the model constructor to
`semkine.c1sf_zrel.ZRelXYZMeshNet` (both DDP ranks run this file, so both install it).
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
from tools.c1sf.common import (ARMS, CANONICAL, OUT, PROTOCOL, config_path, expected_raw, read,  # noqa: E402
                               require, save, sha, verify_protocol, zn_mode)


def validate_recipe(cfg, arm, phase):
    expected = expected_raw(arm, phase, yaml.safe_load(CANONICAL.read_text()))
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected, "Unregistered difference from the C1 recipe + registered C1SF change")


def check_binding(options, config):
    p = verify_protocol(options.protocol)
    require(sha(config) == p["file_sha256"][str(Path(config).resolve().relative_to(ROOT))], "Config not frozen")
    pre = read(OUT / "receipts" / f"preflight_{options.arm}.json")
    require(pre["status"] == "PASS" and pre["protocol_sha256"] == sha(options.protocol), "Arm preflight not accepted")
    replay = read(OUT / "receipts" / "preflight_c1replay.json")
    require(replay["status"] == "PASS" and replay["c1_code_replay_exact"] is True,
            "Current code does not replay C1 training exactly; formal pairing premise fails")
    require(pre["wrapper_sha256"] == sha(__file__), "Training wrapper changed since preflight")


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": sha(cfg["_config_path"]),
                "protocol_sha256": sha(options.protocol)}
    nonce = os.environ.get("C1SF_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("C1SF_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(C1SF_CLAIM_NONCE=nonce, C1SF_CLAIM_PARENT=str(os.getpid()))


class C1SFAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        validate_recipe(model.cfg, self.options.arm, self.options.phase)
        if zn_mode(model.cfg) == "root_relative":
            from semkine.c1sf_zrel import ZRelXYZMeshNet
            require(type(model.xyz) is ZRelXYZMeshNet, "Arm ZR built without the root-relative depth channel")
        else:
            from semkine.xyz_mesh import XYZMeshNet
            require(type(model.xyz).__name__ == "XYZMeshNet" and type(model.xyz).__module__ == XYZMeshNet.__module__,
                    "Non-ZR arm must build the plain C1 XYZMeshNet")
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(c1sf_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__), arm=self.options.arm,
                 registered_change=ARMS[self.options.arm], xyz_class=type(model.xyz).__module__ + "." + type(model.xyz).__name__)
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
    parser.add_argument("--arm", choices=tuple(ARMS), default=os.environ.get("C1SF_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("C1SF_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    options.sampler = "legacy_none"
    options.protocol = str(PROTOCOL)
    options.acceptance = None
    require(options.arm in ARMS and options.phase in ("preflight", "formal"), "Explicit C1SF identity required")
    os.environ.update(C1SF_ARM=options.arm, C1SF_PHASE=options.phase)
    require(Path(options.config).resolve() == config_path(options.arm, options.phase).resolve(), "Unregistered config path")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    protocol = verify_protocol(options.protocol)
    require(sha(options.config) == protocol["file_sha256"][str(Path(options.config).resolve().relative_to(ROOT))],
            "Config not frozen")
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    if zn_mode(cfg) == "root_relative":
        from semkine import c1sf_zrel
        c1sf_zrel.install(cfg["MANO"]["NPZ"])
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
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), C1SFAudit(options)]
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
