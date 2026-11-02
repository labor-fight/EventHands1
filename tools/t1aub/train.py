#!/usr/bin/env python3
"""Fresh T1A-UB training of one arm under S45's strict numerical and input contract (the one T1a was trained under).

The policy of tools/s45tr/train.py and tools/s45sm/train.py: deterministic algorithms, fixed DDP strategy, per-step input
journal / index tape / snapshots through tools/xyz_c1_repro/train.py's ReproAudit, the retired `tools.xag.train_audit`
name provided for configs with MODEL.XAG_* keys. The arm's hooks (tools/t1aub/common.py install_for_config: T1a's event
graph, this study's subclass, the routes) are installed in both DDP ranks before the model and the datasets are built.
Router arms' packets carry no XAG source (the pair-stream audit hashes the base packet fields only); `sampler: dedicated`
draws the shuffled loader from torch.Generator().manual_seed(3407) (S45-AT s45samp's stream). The process config must equal
the registered arm's generated YAML; a formal run requires the arm's accepted preflight under the same wrapper and code
identity. Output directories are fresh-only (no resume).

    python tools/t1aub/train.py --arm A --phase {preflight,formal} --config C       (two GPUs; rank 1 re-enters via env)
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


def write(path, rows):
    from tools.t1aub.common import save
    save(path, rows)


def audit_classes(B, C):
    """The audit callbacks (they derive from B's Lightning classes, so they exist only once B is imported)."""

    class XAGTrainAudit(B.pl.Callback):
        """Provenance only: the routes and, when the run reads XAG sources, which build; never changes training."""

        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

        def on_fit_start(self, trainer, model):
            if trainer.global_rank == 0:
                root = model.cfg["DATA"].get("XAG_SOURCE_ROOT")
                C.save(self.output_dir / "xag_binding.json", {
                    "arm": model.cfg["T1AUB"]["ARM"], "config_sha256": C.sha(model.cfg["_config_path"]),
                    "xag_assoc": model.cfg["MODEL"].get("XAG_ASSOC"), "xag_source_root": root,
                    "xag_source_build_summary_sha256": C.sha(Path(root) / "build_summary.json") if root else None,
                    "route": model.cfg["T1AUB"].get("ROUTE"), "recorded_at_utc": C.now()})

    class Audit(B.ReproAudit):
        def on_fit_start(self, trainer, model):
            validate_recipe(model.cfg, self.options.arm, self.options.phase)
            check_model(model, self.options)
            B.check_formal_binding = check_binding
            super().on_fit_start(trainer, model)

        def on_train_start(self, trainer, model):
            trainer.strategy.barrier()
            super().on_train_start(trainer, model)
            f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
            r = C.read(f)
            r.update(t1aub_wrapper_sha256=C.sha(__file__), c1_helper_sha256=C.sha(B.__file__), arm=self.options.arm,
                     code_identity_sha256=C.code_identity(), installed=self.options.installed,
                     xyz_class=type(model.xyz).__module__ + "." + type(model.xyz).__name__,
                     params_model=int(sum(p.numel() for p in model.parameters())), sampler=self.options.sampler)
            C.save(f, r)

        def on_validation_start(self, trainer, model):
            self.validation_batches = self.validation_samples = 0
            self.validation_before = B.normalized(model.state_dict())

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            self.validation_batches += 1
            self.validation_samples += batch.batch_size

        def on_validation_end(self, trainer, model):
            C.require(self.validation_batches == C.VALIDATION["batches"]
                      and self.validation_samples == C.VALIDATION["samples"], "Full validation denominator changed")
            same = self.validation_before == B.normalized(model.state_dict())
            C.require(same, "Validation changed model parameters/buffers")
            C.save(self.out / f"validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json", {
                "status": "PASS", "step": int(trainer.global_step), "batches": self.validation_batches,
                "samples_with_original_distributed_padding": self.validation_samples, "model_state_unchanged": same})

    return XAGTrainAudit, Audit


def validate_recipe(cfg, arm, phase):
    from tools.t1aub import common as C
    expected = C.expected_raw(arm, phase)
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    C.require(actual == expected, "Unregistered difference from S45 + the registered T1A-UB arm")


def check_model(model, options):
    from tools.t1aub import common as C
    from semkine import t1aub as M
    C.require(type(model.xyz).__module__ == "semkine.t1aub" and type(model.xyz).__name__ == "T1aubXYZMeshNet",
              "T1A-UB arms build this study's XYZMeshNet subclass")
    b = model.cfg["T1AUB"]
    C.require(bool(model.xyz.t1aub_eg) == bool(b["EG"]), "event graph presence differs from T1AUB.EG")
    C.require(M.installed() == options.installed["model"], "model hook differs from the installed one")
    route = b.get("ROUTE")
    if route:
        C.require(model.xag_assoc_mode == "face_route", "routes enter through the XAG face_route entry")
    if route and route["ROUTER"] == "gt":
        from semkine import s45sm_oracle as O
        C.require(O.installed() == O.STATE, "PFA label-route oracle not installed")
    elif route:
        from semkine import s45sm as SM
        st = SM.installed()
        C.require(st is not None and st["router"] == route["ROUTER_KIND"] and st["checkpoint"] == route["ROUTER_CHECKPOINT"]
                  and st["assoc"] == route["ASSOC"] and float(st["onhand_tau"]) == float(route["ONHAND_TAU"]),
                  "router routes not installed as registered")
    if b.get("GTFEAT"):
        C.require(model.xag_assoc_mode == "face_route" and bool(model.xyz.t1aub_gtfeat), "gtfeat not installed")


def check_binding(options, config):
    from tools.t1aub import common as C
    C.verify_protocol()
    C.config_frozen(config)
    pre = C.read(C.OUT / "receipts" / f"preflight_{options.arm}.json")
    C.require(pre["status"] == "PASS", "Arm preflight not accepted")
    C.require(pre["wrapper_sha256"] == C.sha(__file__), "Training wrapper changed since preflight")
    C.require(pre.get("code_identity_sha256") == C.code_identity(), "T1A-UB code changed since the preflight was accepted")
    replay = C.read(C.OUT / "receipts" / "preflight_t1areplay.json")
    C.require(replay["status"] == "PASS" and replay.get("t1a_replay_exact") is True,
              "this wrapper does not replay T1a's preflight exactly; formal pairing premise fails")


def fresh_claim(options, cfg):
    from tools.t1aub import common as C
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": C.sha(cfg["_config_path"])}
    nonce = os.environ.get("T1AUB_CLAIM_NONCE")
    if nonce:
        C.require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get("T1AUB_CLAIM_PARENT") == str(os.getppid()),
                  "Only a direct native DDP child can join an existing fresh claim")
        C.require(C.read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    C.require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    C.require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    C.save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update(T1AUB_CLAIM_NONCE=nonce, T1AUB_CLAIM_PARENT=str(os.getpid()))


def register_audit_module(XAGTrainAudit):
    """semkine/train.py imports the retired `tools.xag.train_audit` for configs with MODEL.XAG_*: provide it."""
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module


def main(argv=None):
    from tools.xyz_c1_repro import train as B          # sets the strict environment, re-executes the process once
    from tools.t1aub import common as C
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", default=os.environ.get("T1AUB_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"), default=os.environ.get("T1AUB_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args(argv)
    C.require(options.arm in C.registry()["arms"], "Explicit T1A-UB arm required")
    C.require(options.phase in C.phases(options.arm), "phase not registered for the arm")
    XAGTrainAudit, Audit = audit_classes(B, C)
    register_audit_module(XAGTrainAudit)
    spec = C.arm_spec(options.arm)
    options.sampler = C.sampler(options.arm)
    options.protocol = str(C.MANIFESTS / f"r{spec['round']}.json")
    options.acceptance = None
    os.environ.update(T1AUB_ARM=options.arm, T1AUB_PHASE=options.phase)
    C.require(Path(options.config).resolve() == C.config_path(options.arm, options.phase).resolve(), "Unregistered config path")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    C.verify_protocol()
    C.config_frozen(options.config)
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    options.installed = C.install_for_config(cfg, evaluation=False)
    route = cfg["T1AUB"].get("ROUTE")
    if route and route["ROUTER"] != "gt":
        B.ORIGINAL.XYZPairStreamCallback = B.ORIGINAL.U1aPairStreamCallback        # packets carry no XAG sources
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
            kwargs["generator"] = None if options.sampler == "legacy_none" else torch.Generator().manual_seed(C.SEED)
        return native_loader(*args, **kwargs)
    original.DataLoader = loader
    native_trainer = pl.Trainer

    class Trainer(native_trainer):
        def __init__(self, *args, **kwargs):
            kwargs.update(deterministic=True, benchmark=False, enable_progress_bar=False,
                          strategy=B.DDPStrategy(find_unused_parameters=False, process_group_backend="nccl",
                                                 bucket_cap_mb=25, gradient_as_bucket_view=False, broadcast_buffers=True))
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), Audit(options)]
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
