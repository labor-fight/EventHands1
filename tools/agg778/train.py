#!/usr/bin/env python3
"""AGG778 Stage D: fresh paired training of one aggregation variant under C1's strict numerical and input contract.

The XCL wrapper pattern (tools/xcl/train.py) on the C1 strict helper (tools/xyz_c1_repro/train.py): deterministic
algorithms, TF32 off, the legacy_none sampler (shuffle without a dedicated generator), the C1 DDP strategy, the C1
`ReproAudit` journal (initial state, per-step indices / packet fingerprints, snapshots every 500 steps). Fresh-only,
fail-closed, no resume. The arm's configuration must equal `d_common.expected_config` exactly.

    CUDA_VISIBLE_DEVICES=4,5 python tools/agg778/train.py --arm agg778_matpt_a3_s3407 --phase preflight \
        --protocol outputs/agg778/protocol.json --config configs/agg778/preflight/agg778_matpt_a3_s3407_preflight.yaml
"""
from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.xyz_c1_repro import train as B  # noqa: E402  (exports the strict environment, re-execs for PYTHONHASHSEED)
from tools.agg778.d_common import ARMS, expected_config, paths, read, require, save, sha  # noqa: E402


def validate_recipe(cfg, arm, phase):
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected_config(arm, phase), "Unregistered difference from the arm's control recipe")


def verify_protocol(protocol):
    p = read(protocol)
    bad = [k for k, v in p["files"].items() if sha(ROOT / k) != v]
    require(not bad, f"AGG778 frozen sources changed: {bad}")
    return p


def check_binding(options, config_path):
    verify_protocol(options.protocol)
    require(options.acceptance, "Formal training requires a passed preflight of this arm")
    a = read(options.acceptance)
    require(a["status"] == "PASS" and a["arm"] == options.arm, "Invalid preflight acceptance")
    require(a["protocol_sha256"] == sha(options.protocol), "Preflight was accepted under another protocol")
    require(a["wrapper_sha256"] == sha(__file__), "Training wrapper changed since preflight")
    require(Path(config_path).resolve() == paths(options.arm)["config"].resolve(), "Formal config path differs")


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase,
                "config_sha256": sha(cfg["_config_path"]), "protocol_sha256": sha(options.protocol)}
    nonce = os.environ.get("AGG778_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("AGG778_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank 1 requires the parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(AGG778_CLAIM_NONCE=nonce, AGG778_CLAIM_PARENT=str(os.getpid()))


class AggAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        validate_recipe(model.cfg, self.options.arm, self.options.phase)
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(agg778_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__), arm=self.options.arm,
                 variant=ARMS[self.options.arm]["variant"], recipe=ARMS[self.options.arm]["cond"])
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
    parser.add_argument("--arm", choices=tuple(ARMS), default=os.environ.get("AGG778_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("AGG778_PHASE"))
    parser.add_argument("--protocol", default=os.environ.get("AGG778_PROTOCOL"))
    parser.add_argument("--acceptance", default=os.environ.get("AGG778_ACCEPTANCE") or None)
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    options.sampler = "legacy_none"
    require(options.arm in ARMS and options.phase in ("preflight", "formal") and options.protocol, "Explicit identity")
    for key in ("arm", "phase", "protocol", "acceptance"):
        os.environ["AGG778_" + key.upper()] = str(getattr(options, key) or "")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    protocol = verify_protocol(options.protocol)
    rel = str(Path(options.config).resolve().relative_to(ROOT))
    require(protocol["files"].get(rel) == sha(options.config), "Config not frozen in the protocol")
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
            kwargs["generator"] = None                                                  # legacy_none, as C1 / XCL
        return native_loader(*args, **kwargs)
    original.DataLoader = loader
    native_trainer = pl.Trainer

    class Trainer(native_trainer):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False, enable_progress_bar=False,
                          strategy=B.DDPStrategy(find_unused_parameters=False, process_group_backend="nccl",
                                                 bucket_cap_mb=25, gradient_as_bucket_view=False,
                                                 broadcast_buffers=True))
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), AggAudit(options)]
            super().__init__(*args, **kwargs)
    original.pl.Trainer = Trainer
    source_fields = tuple(n for n in B.RAW_FIELDS if n.startswith("xag_"))

    def before_transfer(self, batch, dataloader_idx):
        if self.training:
            step = int(self.global_step) + 1
            selected = B.selected_step(step, options.phase, int(self.cfg["TRAIN"]["MAX_STEPS"]),
                                       int(self.trainer.num_training_batches))
            common = SimpleNamespace(**{n: None if n in source_fields else getattr(batch, n, None) for n in B.RAW_FIELDS})
            self._repro_cpu_fingerprint = {
                "step": step, "small_fields_sha256": B.packet_sha(batch, B.FIELDS),
                "full_packet_sha256": B.packet_sha(common, B.RAW_FIELDS) if selected else None,
                "source_fields_sha256": B.packet_sha(batch, (*source_fields, "xag_depth")) if selected else None,
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
