"""Fresh S45-SM training of one arm, under S45's strict numerical and input contract
(docs/EXPERIMENT_RECORDS.md#s45sm-prereg-20261031 and each round's pre-registration).

One wrapper for every arm: the policy of tools/s45at/train.py and tools/s45rt/train.py (deterministic algorithms, fixed DDP
strategy, per-step input journal / index tape / snapshots via tools/xyz_c1_repro/train.py's ReproAudit, the XAG provenance
module registered under the retired `tools.xag.train_audit` name), and, chosen by the arm's registry spec:
  router arms  S45-AT dz + `face_route` with the routes of a frozen events-only router installed in both DDP ranks before the
               model is built (the router kinds the arm needs are registered first); the paired-stream audit hashes the
               base packet fields only (`U1aPairStreamCallback`: the packets carry no XAG source); `sampler: dedicated` draws
               the shuffled loader from `torch.Generator().manual_seed(3407)` (S45-AT s45samp's stream, seed 3407 stays), the
               default keeps the legacy None generator; `bank` installs the own-loop prior sampler; `lr_schedule: cosine` is
               part of the checked recipe; `phases` limits the phases an arm may run
  oracles      S45-AT t1ddz's config with the PFA association installed (semkine/s45sm_oracle.py); the batches keep the XAG
               sources (no pair-stream patch) and the provenance records the XAG source build
The recipe the process was started with must equal the registered arm's generated YAML (configs.expected_for). A formal run
requires the arm's accepted preflight receipt, which must carry this wrapper's SHA and this package's code identity: a preflight
accepted under other code does not authorise a formal run. Output directories are fresh-only (no resume).
The module is import-safe (no strict environment); `main()` imports tools.xyz_c1_repro.train first, which exports the strict
environment and re-executes the process once when PYTHONHASHSEED != 3407.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import sys
import types

import yaml

from .configs import base_config, dump, expected_for, expected_state, oracle_base_config
from .paths import MANIFESTS, OUT, ROOT, ROOT_ALIAS, VALIDATION, now, read, require, save, sha
from .protocol import code_identity, config_frozen, verify_protocol
from .registry import arm_spec, config_path, oracles, phases, registry, router_spec, sampler
from .routing import NEEDS, install_hook, install_oracle, install_train_hook, installed_routing, register_kinds


VERB = "train"


def write(path, rows):
    save(path, rows)


def validate_recipe(cfg, arm, phase):
    """The config the process runs equals the registered arm's generated YAML."""
    expected = expected_for(arm, phase)
    if ROOT_ALIAS:                                  # verification in a copy of the repository only
        expected = yaml.safe_load(dump(expected))
    actual = {k: v for k, v in cfg.items() if k != "_config_path"}
    if arm in oracles():
        require(actual == expected, "Unregistered difference from S45-AT t1ddz + the registered S45-SM oracle identity")
    else:
        require(actual == expected, "Unregistered difference from S45-AT dz + the registered S45-SM routing")


def check_binding(options, config):
    """Formal-run binding: the protocol, the frozen config, and the accepted preflight of this wrapper and code identity."""
    verify_protocol()
    config_frozen(config)
    pre = read(OUT / "receipts" / f"preflight_{options.arm}.json")
    require(pre["status"] == "PASS", "Arm preflight not accepted")
    require(pre["wrapper_sha256"] == sha(__file__), "Training wrapper changed since preflight")
    require(pre.get("code_identity_sha256") == code_identity(),
            "S45-SM code changed since the preflight was accepted (re-run the preflight under the current code)")


def claim_prefix(arm):
    return "S45SMO_" if arm in oracles() else "S45SM_"


def fresh_claim(options, cfg):
    directory = Path(cfg["TRAIN"]["OUTPUT_DIR"]).absolute()
    claim = directory / ".fresh_claim.json"
    identity = {"arm": options.arm, "phase": options.phase, "config_sha256": sha(cfg["_config_path"])}
    prefix = claim_prefix(options.arm)
    nonce = os.environ.get(prefix + "CLAIM_NONCE")
    if nonce:
        require(os.environ.get("LOCAL_RANK") == "1" and os.environ.get(prefix + "CLAIM_PARENT") == str(os.getppid()),
                "Only a direct native DDP child can join an existing fresh claim")
        require(read(claim) == {**identity, "nonce": nonce, "parent_pid": os.getppid()}, "Fresh claim identity differs")
        return
    require(os.environ.get("LOCAL_RANK") in (None, "0"), "Rank1 requires parent claim")
    require(not os.path.lexists(directory), "Fresh-only output already exists: " + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(exist_ok=False)
    nonce = secrets.token_hex(32)
    save(claim, {**identity, "nonce": nonce, "parent_pid": os.getpid()}, exclusive=True)
    os.environ.update({prefix + "CLAIM_NONCE": nonce, prefix + "CLAIM_PARENT": str(os.getpid())})


def audit_classes(B):
    """The audit callbacks (they derive from B's Lightning classes, so they exist only once B is imported)."""

    class XAGTrainAudit(B.pl.Callback):
        """Provenance only: routing settings and, when the run reads XAG sources, which build; never changes training."""

        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)

        def on_fit_start(self, trainer, model):
            if trainer.global_rank == 0:
                root = model.cfg["DATA"].get("XAG_SOURCE_ROOT")
                record = {"arm": model.cfg["S45SM"]["ARM"], "config_sha256": sha(model.cfg["_config_path"]),
                          "xag_assoc": model.cfg["MODEL"].get("XAG_ASSOC"), "xag_gnn": model.cfg["MODEL"].get("XAG_GNN"),
                          "xag_source_root": root}
                if model.cfg["S45SM"]["ARM"] in oracles():
                    record["xag_source_build_summary_sha256"] = sha(Path(root) / "build_summary.json") if root else None
                else:
                    record["routing"] = model.cfg["S45SM"]
                record["recorded_at_utc"] = now()
                save(self.output_dir / "xag_binding.json", record)

    class _Audit(B.ReproAudit):
        kind = None

        def on_fit_start(self, trainer, model):
            validate_recipe(model.cfg, self.options.arm, self.options.phase)
            self.check_model(model)
            B.check_formal_binding = check_binding
            super().on_fit_start(trainer, model)

        def on_train_start(self, trainer, model):
            trainer.strategy.barrier()
            super().on_train_start(trainer, model)
            f = self.out / f"effective_runtime_rank{trainer.global_rank}.json"
            r = read(f)
            r.update(self.runtime_record())
            r.update(c1_helper_sha256=sha(B.__file__), arm=self.options.arm, code_identity_sha256=code_identity())
            save(f, r)

        def on_validation_start(self, trainer, model):
            self.validation_batches = self.validation_samples = 0
            self.validation_before = B.normalized(model.state_dict())

        def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
            self.validation_batches += 1
            self.validation_samples += batch.batch_size

        def on_validation_end(self, trainer, model):
            require(self.validation_batches == VALIDATION["batches"] and self.validation_samples == VALIDATION["samples"],
                    "Full validation denominator changed")
            same = self.validation_before == B.normalized(model.state_dict())
            require(same, "Validation changed model parameters/buffers")
            save(self.out / f"validation_rank{trainer.global_rank}_step{trainer.global_step:06}.json", {
                "status": "PASS", "step": int(trainer.global_step), "batches": self.validation_batches,
                "samples_with_original_distributed_padding": self.validation_samples, "model_state_unchanged": same})

    class RouterArmAudit(_Audit):
        def check_model(self, model):
            require(type(model.xyz).__module__ == "semkine.xyz_mesh" and type(model.xyz).__name__ == "XYZMeshNet",
                    "S45-SM arms build S45's XYZMeshNet")
            require(model.xag_assoc_mode == "face_route", "S45-SM routes through the XAG face_route entry")
            state = installed_routing()
            require(state is not None and {k: state[k] for k in ("router", "checkpoint", "assoc", "onhand_tau")}
                    == expected_state(model.cfg), "S45-SM routing not installed as registered")

        def runtime_record(self):
            state = installed_routing()
            return {"s45sm_wrapper_sha256": sha(__file__), "s45sm_routing": {k: v for k, v in state.items() if k != "fast"},
                    "s45sm_router_checkpoint_sha256": sha(state["checkpoint"]), "s45sm_fast": state["fast"],
                    "s45sm_bank": self.options.bank, "s45sm_module_sha256": sha(ROOT / "semkine/s45sm.py"),
                    "s45sm_fast_module_sha256": sha(ROOT / "semkine/s45sm_fast.py"),
                    "s45sm_soft_module_sha256": sha(ROOT / "semkine/s45sm_soft.py"),
                    "s45sm_srp_module_sha256": sha(ROOT / "semkine/s45sm_srp.py"), "s45sm_sampler": self.options.sampler}

    class OracleAudit(_Audit):
        def check_model(self, model):
            from semkine import s45sm_oracle as O
            require(type(model.xyz).__module__ == "semkine.xyz_mesh" and type(model.xyz).__name__ == "XYZMeshNet",
                    "oracle arms build S45's XYZMeshNet")
            require(model.xag_assoc_mode == "face_route", "the label routes enter through the XAG face_route entry")
            require(O.installed() == O.STATE, "PFA oracle association not installed")

        def runtime_record(self):
            from semkine import s45sm_oracle as O
            return {"s45sm_oracle_wrapper_sha256": sha(__file__), "s45sm_oracle": O.installed(),
                    "s45sm_oracle_module_sha256": sha(ROOT / "semkine/s45sm_oracle.py"),
                    "s45sm_fast_module_sha256": sha(ROOT / "semkine/s45sm_fast.py")}

    return XAGTrainAudit, RouterArmAudit, OracleAudit


def register_audit_module(XAGTrainAudit):
    """semkine/train.py still imports the retired `tools.xag.train_audit`: provide it under that name."""
    package = sys.modules.get("tools.xag") or types.ModuleType("tools.xag")
    package.__path__ = []
    module = types.ModuleType("tools.xag.train_audit")
    module.XAGTrainAudit, module.write = XAGTrainAudit, write
    package.train_audit = module
    sys.modules["tools.xag"], sys.modules["tools.xag.train_audit"] = package, module


def parse(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arm", default=os.environ.get("S45SM_ARM") or os.environ.get("S45SMO_ARM"))
    parser.add_argument("--phase", choices=("preflight", "formal"),
                        default=os.environ.get("S45SM_PHASE") or os.environ.get("S45SMO_PHASE"))
    parser.add_argument("--config", required=True)
    options = parser.parse_args(argv)
    require(options.arm in registry()["arms"] or options.arm in oracles(), "Explicit S45-SM identity required")
    require(options.phase in phases(options.arm), "phase not registered for the arm")
    return options


def main(argv=None):
    from tools.xyz_c1_repro import train as B          # sets the strict environment, re-executes the process once
    XAGTrainAudit, RouterArmAudit, OracleAudit = audit_classes(B)
    register_audit_module(XAGTrainAudit)
    options = parse(argv)
    oracle = options.arm in oracles()
    spec = oracles()[options.arm] if oracle else arm_spec(options.arm)
    options.sampler = "legacy_none" if oracle else sampler(options.arm)
    options.protocol = str(MANIFESTS / f"r{spec['round']}.json")
    options.acceptance = None
    os.environ.update({claim_prefix(options.arm) + "ARM": options.arm, claim_prefix(options.arm) + "PHASE": options.phase})
    require(Path(options.config).resolve() == config_path(options.arm, options.phase).resolve(), "Unregistered config path")
    cfg = B.load_config(options.config)
    validate_recipe(cfg, options.arm, options.phase)
    verify_protocol()
    config_frozen(options.config)
    if options.phase == "formal":
        check_binding(options, options.config)
    fresh_claim(options, cfg)
    if oracle:
        install_oracle(cfg)
        audit = OracleAudit
    else:
        register_kinds(NEEDS[router_spec(spec["router"])["kind"]])
        install_hook(cfg)
        options.bank = install_train_hook(cfg)
        B.ORIGINAL.XYZPairStreamCallback = B.ORIGINAL.U1aPairStreamCallback        # packets carry no XAG sources
        audit = RouterArmAudit
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
            # built after semkine/train.py parsed its argv and before fit: Lightning relaunches rank 1 from sys.argv at
            # fit time, so the verb goes back in here (rank 1 re-enters `cli.py train`; arm and phase via the environment)
            sys.argv = [sys.argv[0], VERB, "--config", options.config]
            kwargs.update(deterministic=True, benchmark=False, enable_progress_bar=False,
                          strategy=B.DDPStrategy(find_unused_parameters=False, process_group_backend="nccl",
                                                 bucket_cap_mb=25, gradient_as_bucket_view=False, broadcast_buffers=True))
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), audit(options)]
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
    # semkine/train.py parses this argv (it knows no verb); the Trainer above restores the verb for Lightning's relaunch
    sys.argv = [sys.argv[0], "--config", options.config]
    original.main()
