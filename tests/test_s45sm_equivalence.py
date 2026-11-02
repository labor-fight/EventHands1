"""CPU tests: the consolidated S45-SM package against the code it replaced (V6 moved-code equivalence).

The old per-variant scripts are read from the repository's `tools/s45sm/` while they are still there (before the migration),
else from the archive the identity manifest names (outputs/s45sm/20261031/legacy_code/files/tools/s45sm/). Two kinds of check:
  - AST fingerprints: every function that was moved without change has, after stripping its docstring and mapping the
    module aliases / constants that were renamed by the move, the same syntax tree as before;
  - numeric differential tests (only while the old modules can be imported): augmentation, both distillation losses, the
    annealing schedule, the config generators of every arm and oracle, the registries, the oracle / bank / schedule helpers.
A check whose old code is not available is skipped, never passed."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import paths as P  # noqa: E402

PKG = P.PACKAGE
ALIASES = {"C", "D", "D2", "SEL", "R4", "R6", "R1", "RP", "T2"}               # tools.s45sm modules the old code imported as
CALL_RESULTS_DIR = {"OUT"}                                                    # the readers' OUT became results_dir()


def legacy_dir():
    live = ROOT / "tools/s45sm"
    if (live / "common.py").is_file() and "def expected_raw(" in (live / "common.py").read_text():
        return live
    manifest = P.MANIFESTS / P.IDENTITY_MANIFEST
    if manifest.is_file():
        info = json.loads(manifest.read_text())
        archive = Path(info["legacy_archive"])
        archive = archive if archive.is_absolute() else ROOT / archive
        if (archive / "files/tools/s45sm/common.py").is_file():
            return archive / "files/tools/s45sm"
    return None


LEGACY = legacy_dir()
needs_legacy = pytest.mark.skipif(LEGACY is None, reason="the old tools are neither in the tree nor archived")


def old_tree(module):
    return ast.parse((LEGACY / f"{module}.py").read_text())


def new_tree(module):
    return ast.parse((PKG / f"{module}.py").read_text())


def function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef,)) and node.name == name:
            return node
    raise KeyError(name)


class Normalise(ast.NodeTransformer):
    def __init__(self, renames, old):
        self.renames, self.old = renames, old

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if self.old and isinstance(node.value, ast.Name) and node.value.id in ALIASES:
            return ast.copy_location(ast.Name(id=self.renames.get(node.attr, node.attr), ctx=node.ctx), node)
        return node

    def visit_Name(self, node):
        if self.old and node.id in CALL_RESULTS_DIR:
            return ast.copy_location(ast.Call(func=ast.Name(id="results_dir", ctx=ast.Load()), args=[], keywords=[]), node)
        return ast.copy_location(ast.Name(id=self.renames.get(node.id, node.id), ctx=node.ctx), node)

    def visit_arg(self, node):
        self.generic_visit(node)
        node.annotation = None
        return node


def fingerprint(fn, renames, old):
    fn = ast.parse(ast.unparse(fn)).body[0]                       # a private copy
    fn.name = "f"
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        fn.body = fn.body[1:]
    fn = Normalise(renames, old).visit(fn)
    fn.returns = None
    return ast.dump(fn)


#: (old module, old function, new module, new function, renames of constants / names old -> new)
MOVED = [
    ("router_eval", "auc", "router_read", "auc", {}),
    ("router_eval", "overlap", "router_read", "overlap", {}),
    ("router_eval", "metrics", "router_read", "gate_metrics", {}),
    ("router_ctx", "d1_check", "router_read", "d1_check", {}),
    ("router_train3", "dataset", "router_train", "dataset", {}),
    ("router_train3", "augment", "router_train", "augment", {}),
    ("router_train2", "augment", "router_train", "augment", {}),
    ("router_train3", "patch_parts", "router_train", "patch_parts", {}),
    ("router_train3", "teacher_part_prob", "router_train", "teacher_part_prob", {}),
    ("router_train3", "kd_loss", "router_train", "kd_loss_part", {}),
    ("router_train4", "kd_loss", "router_train", "kd_loss_patch", {}),
    ("router_train7", "factor", "router_train", "factor", {}),
    ("decode_variant", "decode_spec", "router_train", "decode_spec", {}),
    ("decide", "trace", "summary", "trace", {}),
    ("decide", "zgz", "summary", "zgz", {}),
    ("decide", "insample_path", "summary", "insample_path", {}),
    ("decide", "insample", "summary", "insample", {}),
    ("decide", "convergence", "summary", "convergence", {}),
    ("decide", "router_cost", "summary", "router_cost", {}),
    ("decide3", "condition_holds", "summary", "condition_holds", {"selected": "select_r4_selected"}),
    ("decide3", "ceiling_class", "summary", "ceiling_class", {}),
    ("decide3", "split", "summary", "split", {}),
    ("decide3", "next_step", "summary", "next_step", {}),
    ("swap_table", "entry", "tables", "entry", {}),
    ("swap_table", "flag", "tables", "flag", {}),
    ("swap_table", "effects", "tables", "effects7", {"TRUNKS": "TRUNKS7", "ROUTES": "ROUTES7"}),
    ("swap_table", "trunk_choice", "tables", "trunk_choice", {"TRUNKS": "TRUNKS7"}),
    ("swap_table", "markdown", "tables", "markdown7", {"ROUTES": "ROUTES7", "TRUNKS": "TRUNKS7"}),
    ("swap_table", "grid", "tables", "grid7", {"TRUNKS": "TRUNKS7"}),
    ("swap_table2", "difference", "tables", "difference", {}),
    ("swap_table2", "lowest_ra", "tables", "lowest_ra", {}),
    ("swap_table2", "best", "tables", "best", {}),
    ("swap_table2", "cell", "tables", "cell", {}),
    ("d1_table", "base_of", "tables", "base_of", {}),
    ("d1_table", "rows", "tables", "d1_rows", {"ROUND": "D1_ROUND", "SEQS": "D1_SEQS", "KEYS": "D1_KEYS"}),
    ("repeat_read", "pair", "readings", "pair", {}),
    ("repeat_read", "reading", "readings", "repeat_reading", {"PAIRS": "REPEAT_PAIRS", "WRITTEN": "REPEAT_WRITTEN"}),
    ("cosine_read", "reading", "readings", "cosine_reading", {"PAIRS": "COSINE_PAIRS", "CELLS": "COSINE_CELLS"}),
    ("rowcheck", "accuracy", "readings", "accuracy", {}),
    ("common4", "same_through", "verify", "same_through", {}),
    ("replay3", "compare", "verify", "replay_compare", {"STEPS": "REPLAY_STEPS"}),
    ("common", "router_dir", "registry", "router_dir", {}),
    ("common", "router_checkpoint", "registry", "router_checkpoint", {}),
    ("common", "router_ready", "registry", "router_ready", {}),
    ("common3", "sampler", "registry", "sampler", {}),
    ("common4", "schedule", "registry", "schedule", {}),
    ("common2", "bank_dir", "registry", "bank_dir", {}),
    ("common2", "bank_ready", "registry", "bank_ready", {}),
    ("select_r4", "rule", "selection", "rule_r4", {}),
    ("select_r6", "rule", "selection", "rule_r6", {}),
    ("select_r6c", "rule", "selection", "rule_ctx", {"CANDIDATES": "CTX_CANDIDATES"}),
    ("select_r6c", "selected_context", "selection", "selected_context", {"target": "ctx_target"}),
    ("bank", "sequences", "bank", "sequences", {}),
    ("freeze", "section", "protocol", "section", {}),
    ("oracle", "oracle_spec", "registry", "oracle_spec", {}),
    ("common", "arm_spec", "registry", "arm_spec", {}),
    ("common", "router_spec", "registry", "router_spec", {}),
    ("common", "expected_state", "configs", "expected_state", {}),
]


@needs_legacy
@pytest.mark.parametrize("case", MOVED, ids=[f"{m[0]}.{m[1]}->{m[2]}.{m[3]}" for m in MOVED])
def test_moved_function_has_the_same_syntax_tree(case):
    old_mod, old_name, new_mod, new_name, renames = case
    old = function(old_tree(old_mod), old_name)
    new = function(new_tree(new_mod), new_name)
    assert fingerprint(old, renames, True) == fingerprint(new, renames, False), f"{old_mod}.{old_name} differs from {new_mod}.{new_name}"


@needs_legacy
def test_the_fingerprint_notices_a_change():
    old = function(old_tree("router_eval"), "auc")
    mutated = ast.parse(ast.unparse(old).replace("1)", "2)", 1)).body[0]
    assert fingerprint(old, {}, True) != fingerprint(mutated, {}, True)


def test_no_module_of_the_package_names_its_own_directory():
    sibling = "s45sm_" + "v2"                                     # the directory the package is introduced in (built here so the
    for path in PKG.glob("*.py"):                                 # migration's textual rename leaves this test alone)
        assert sibling not in path.read_text(), path.name        # the rename of that directory then changes no module


def old_module(name):
    if LEGACY is None or LEGACY != ROOT / "tools/s45sm":
        pytest.skip("the old modules are importable only while they are still tools.s45sm")
    return importlib.import_module(f"tools.s45sm.{name}")


@needs_legacy
def test_registries_are_equal():
    from tools.s45sm import registry as R
    C, O = old_module("common"), old_module("oracle")
    assert json.dumps(R.registry()) == json.dumps(C.registry()) and json.dumps(R.oracles()) == json.dumps(O.oracles())
    for arm in C.registry()["arms"]:
        assert R.run_name(arm) == C.run_name(arm) and R.config_path(arm, "formal") == C.config_path(arm, "formal")
        assert R.output_dir(arm, "preflight") == C.output_dir(arm, "preflight")
    assert R.router_ready("t0p") == C.router_ready("t0p")


@needs_legacy
def test_config_generators_equal_the_layered_old_ones_for_every_arm():
    from tools.s45sm import configs as K
    from tools.s45sm import registry as R
    C, C2, C3, C4, O = (old_module(n) for n in ("common", "common2", "common3", "common4", "oracle"))
    base, obase = yaml.safe_load(C.S45_CONFIG.read_text()), yaml.safe_load(O.T1DDZ_CONFIG.read_text())
    n = 0
    for arm, spec in R.registry()["arms"].items():
        old = C.expected_raw if spec["round"] == 1 else (C2.expected_raw2 if spec["round"] < 7 else (C3.expected_raw3 if spec["round"] < 14 else C4.expected_raw4))
        for phase in ("preflight", "formal"):
            assert K.expected_raw(arm, phase, base) == old(arm, phase, base), (arm, phase)
            assert K.dump(K.expected_raw(arm, phase, base)) == K.dump(old(arm, phase, base))
            n += 1
    for arm in R.oracles():
        for phase in ("preflight", "formal"):
            assert K.expected_raw_oracle(arm, phase, obase) == O.expected_raw(arm, phase, obase)
            n += 1
    assert n >= 93


@needs_legacy
def test_old_layers_that_do_not_apply_are_refused_or_differ():
    """The round-1 files need `expected_raw` (no FAST key): the later layers add it, so they never produced the round-1 files."""
    from tools.s45sm import configs as K
    C, C2 = old_module("common"), old_module("common2")
    base = yaml.safe_load(C.S45_CONFIG.read_text())
    assert C2.expected_raw2("t0pdz", "formal", base) != C.expected_raw("t0pdz", "formal", base)
    assert K.expected_raw("t0pdz", "formal", base) == C.expected_raw("t0pdz", "formal", base)


@needs_legacy
def test_augment_and_both_distillation_losses_equal_the_old_ones():
    from tools.s45sm import router_train as RTR
    R2, R3, R4 = old_module("router_train2"), old_module("router_train3"), old_module("router_train4")
    from test_s45sm_routers import batch
    aug = {"rot_deg": 60, "scale": [0.75, 1.33], "keep": [0.3, 1.0]}
    for seed in range(3):
        b = batch(seed)
        a = R2.augment(b, aug, torch.Generator().manual_seed(seed))
        c = RTR.augment(b, aug, torch.Generator().manual_seed(seed))
        for k, v in a.__dict__.items():
            if torch.is_tensor(v):
                assert torch.equal(v, getattr(c, k)), k
    g = torch.Generator().manual_seed(0)
    B, N = 2, 32
    mask = torch.rand(B, N, generator=g) > 0.2
    logits = torch.randn(B, N, 279, generator=g)
    t = {"patch": logits, "onhand": torch.randn(B, N, generator=g) * 3, "coord": torch.randn(B, N, 3, generator=g)}
    s = {"patch": torch.randn(B, N, 279, generator=g), "onhand": torch.randn(B, N, generator=g), "coord": torch.randn(B, N, 3, generator=g), "mask": mask}
    kd = {"T": 2.0, "lambda_patch": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    old_total, old_parts = R4.kd_loss(s, t, kd)
    new_total, new_parts = RTR.kd_loss_patch(s, t, kd)
    assert torch.equal(old_total, new_total) and all(torch.equal(old_parts[k], new_parts[k]) for k in old_parts)
    from semkine import s45rt_router as RT
    teacher = RT.load(old_module("common").router_ready("t0p"))[0]
    pp = R3.patch_parts(teacher)
    assert torch.equal(pp, RTR.patch_parts(teacher))
    kd = {"T": 2.0, "lambda_part": 1.0, "lambda_coord": 0.5, "lambda_onhand": 0.5}
    s2 = {"mask": mask, "part": torch.randn(B, N, 16, generator=g), "onhand": torch.randn(B, N, generator=g), "coord": torch.randn(B, N, 3, generator=g)}
    t2 = {"patch": torch.randn(B, N, teacher.n_patches, generator=g), "onhand": torch.randn(B, N, generator=g) * 3, "coord": torch.randn(B, N, 3, generator=g)}
    old_total, old_parts = R3.kd_loss(s2, t2, pp, kd)
    new_total, new_parts = RTR.kd_loss_part(s2, t2, pp, kd)
    assert torch.equal(old_total, new_total) and all(torch.equal(old_parts[k], new_parts[k]) for k in old_parts)


@needs_legacy
def test_the_annealing_schedule_and_modules_equal_the_old_ones():
    from tools.s45sm import router_train as RTR
    A = old_module("router_train7")
    assert [RTR.factor(s, 2000) for s in range(2001)] == [A.factor(s, 2000) for s in range(2001)]
    assert RTR.ANNEAL_WARMUP == A.ANNEAL_WARMUP and RTR.LR == A.LR and RTR.WARMUP == old_module("router_train3").WARMUP
    from semkine import s45sm as SM
    from semkine import s45sm_srp as SRP
    npz = ROOT / "assets/mano_right.npz"
    for new_cls, old_cls, net in ((RTR.SRAnneal, A.SRAnneal, lambda: SM.build(npz)), (RTR.PatchAnneal, A.PatchAnneal, lambda: SRP.build(npz))):
        sched = {}
        for label, cls in (("new", new_cls), ("old", old_cls)):
            torch.manual_seed(0)
            module = cls(net(), None, None, None) if label == "new" or cls is A.PatchAnneal else cls(net())
            module.anneal_steps = 500
            cfg = module.configure_optimizers()
            opt, sc, lrs = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"], []
            for _ in range(500):
                lrs.append(opt.param_groups[0]["lr"])
                opt.step()
                sc.step()
            sched[label] = lrs
        assert sched["new"] == sched["old"]


@needs_legacy
def test_old_and_new_decisions_agree_on_the_selection_rules():
    from tools.s45sm import selection as SEL
    old = {"r4": old_module("select_r4").rule, "r6": old_module("select_r6").rule}
    for cross in (0.2, 0.2887, 0.2937, 0.2987, 0.3):
        for occ in (0.46, 0.4671, 0.48, 0.5):
            base = {"ylf_local": {"part_cross_share": 0.2937, "occupancy_overlap_patch": 0.4671}}
            cand = {"ylf_local": {"part_cross_share": cross, "occupancy_overlap_patch": occ}}
            assert SEL.rule_r4(base, cand) == old["r4"](base, cand) and SEL.rule_r6(base, cand) == old["r6"](base, cand)


@needs_legacy
def test_old_and_new_summary_helpers_agree():
    from tools.s45sm import summary as S
    D, D3 = old_module("decide"), old_module("decide3")
    for l in (0.0, 1.5, 1.6, 3.5, 3.6):
        for C_ in (0.5, 0.8, 0.81):
            gap = {"local": l, "global": l - 1, "ra": l + 0.1}
            assert S.ceiling_class(gap, C_) == D3.ceiling_class(gap, C_)
    assert (S.TH, S.NOISE, S.STEPS) == (D.TH, D.NOISE, D.STEPS)
    assert S.TARGET == old_module("common").TARGET and S.BUDGET == old_module("common").BUDGET
