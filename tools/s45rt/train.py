#!/usr/bin/env python3
"""Fresh S45-RT T1p training under S45's strict numerical and input contract.

Same policy as tools/s45tr/train.py + tools/s45tr/train_xag.py for S45-TR T1b (deterministic algorithms,
legacy_none sampler, fixed DDP strategy, per-step input journal / index tape / snapshots via tools/xyz_c1_repro/
train.py's ReproAudit; the XAG provenance module registered under the retired `tools.xag.train_audit` name; the
pair-stream audit hashes the base packet fields only, since T1p's packets carry no ground-truth source fields). The
only difference from S45 is `MODEL.XAG_ASSOC: face_route` with the frozen T0P's routes (semkine/s45rt_router.py),
installed in both DDP ranks before the model is built. The formal run also requires T0P's gate to have passed.
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
from tools.s45rt.common import OUT, S45_CONFIG, now, read, require, save, sha, verify_protocol  # noqa: E402
from tools.s45rt.plan2 import (ARMS, ROUTER_CHECKPOINT, ROUTER_DIR, config_frozen, config_path, expected_raw,  # noqa: E402
                               install_hook)

PROTOCOL = OUT / "protocol.json"


class XAGTrainAudit(B.pl.Callback):
    """Provenance only: which XAG settings the run used; never changes training."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer, model):
        if trainer.global_rank == 0:
            root = model.cfg["DATA"].get("XAG_SOURCE_ROOT")
            save(self.output_dir / "xag_binding.json", {
                "arm": model.cfg["S45RT"]["ARM"], "config_sha256": sha(model.cfg["_config_path"]),
                "xag_assoc": model.cfg["MODEL"]["XAG_ASSOC"], "xag_gnn": model.cfg["MODEL"]["XAG_GNN"],
                "xag_source_root": root, "router_checkpoint_sha256": sha(ROUTER_CHECKPOINT), "recorded_at_utc": now()})


def write(path, rows):
    save(path, rows)


def register():
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module
    B.ORIGINAL.XYZPairStreamCallback = B.ORIGINAL.U1aPairStreamCallback


def validate_recipe(cfg, arm, phase):
    expected = expected_raw(arm, phase, yaml.safe_load(S45_CONFIG.read_text()))
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    require(actual == expected, "Unregistered difference from S45's recipe + the registered S45-RT change")


def check_binding(options, config):
    verify_protocol()
    config_frozen(config)
    pre = read(OUT / "receipts" / f"preflight_{options.arm}.json")
    require(pre["status"] == "PASS" and pre["protocol_sha256"] == sha(PROTOCOL), "Arm preflight not accepted")
    require(accepts(__file__, pre["wrapper_sha256"]), "Training wrapper changed since preflight")
    gate = read(ROUTER_DIR / "gate.json")
    require(gate["status"] == "PASS" and gate["t0p_checkpoint_sha256"] == sha(ROUTER_CHECKPOINT),
            "T0P gate not passed for this router")


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": sha(cfg["_config_path"]),
                "protocol_sha256": sha(PROTOCOL)}
    nonce = os.environ.get("S45RT_CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("S45RT_CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(S45RT_CLAIM_NONCE=nonce, S45RT_CLAIM_PARENT=str(os.getpid()))


def installed_router():
    from semkine import s45rt_router
    return s45rt_router.installed()


class S45RTAudit(B.ReproAudit):
    def on_fit_start(self, trainer, model):
        validate_recipe(model.cfg, self.options.arm, self.options.phase)
        require(type(model.xyz).__module__ == "semkine.xyz_mesh" and type(model.xyz).__name__ == "XYZMeshNet",
                "T1p builds S45's XYZMeshNet")
        require(model.xag_assoc_mode == "face_route", "T1p routes with the XAG face route")
        state = installed_router()
        require(state is not None and state["checkpoint"] == str(ROUTER_CHECKPOINT), "T0P routing not installed")
        B.check_formal_binding = check_binding
        super().on_fit_start(trainer, model)

    def on_train_start(self, trainer, model):
        trainer.strategy.barrier()
        super().on_train_start(trainer, model)
        f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
        r = read(f)
        r.update(s45rt_wrapper_sha256=sha(__file__), c1_helper_sha256=sha(B.__file__), arm=self.options.arm,
                 s45rt_router=installed_router(), router_checkpoint_sha256=sha(ROUTER_CHECKPOINT),
                 router_module_sha256=sha(ROOT / "semkine/s45rt_router.py"))
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
    register()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", choices=ARMS, default=os.environ.get("S45RT_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("S45RT_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    options.sampler = "legacy_none"
    options.protocol = str(PROTOCOL)
    options.acceptance = None
    require(options.arm in ARMS and options.phase in ("preflight", "formal"), "Explicit S45-RT identity required")
    os.environ.update(S45RT_ARM=options.arm, S45RT_PHASE=options.phase)
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
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), S45RTAudit(options)]
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
