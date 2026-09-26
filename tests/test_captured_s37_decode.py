"""CPU/static ownership tests for capture infrastructure; no graphs/data/weights.

The parent owns separate authorized GPU primitive and complete own-H parity
checks. These tests cannot certify CUDA numerical equivalence or timing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semkine.captured_s37_decode import CapturedS37Decode, _compile_host_parent_forward


class _OneIndex:
    def forward(self):
        transform_chain = ["root", "child"]
        i = 1
        return transform_chain[self.parents[i]]


class _NoIndex:
    def forward(self):
        return self.parents


class _TwoIndices:
    def forward(self):
        transform_chain = ["root", "child"]
        i = 1
        return transform_chain[self.parents[i]], transform_chain[self.parents[i]]


def test_actual_mano_ast_has_exactly_one_reversible_replacement():
    spec = importlib.util.spec_from_file_location("cd0_mano_source", ROOT / "model/mano_layer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Definitions only; no ManoLayer construction.
    code, metadata = _compile_host_parent_forward(module.ManoLayer.forward)
    assert metadata["replacement_count"] == 1
    assert metadata["all_other_ast_nodes_preserved"]
    assert metadata["original_ast_sha256"] != metadata["adapted_ast_sha256"]
    assert isinstance(code, type(compile("", "test", "exec")))


def test_ast_replacement_uses_given_host_parent_and_keeps_other_operations():
    code, metadata = _compile_host_parent_forward(_OneIndex.forward)
    namespace = dict(_OneIndex.forward.__globals__)
    namespace["_captured_mano_parents"] = (-1, 0)
    exec(code, namespace)
    # A pure-Python sentinel: adapted code must not index self.parents.
    result = namespace["forward"](SimpleNamespace(parents=None))
    assert result == "root"
    assert metadata["replacement_count"] == 1


@pytest.mark.parametrize("function", [_NoIndex.forward, _TwoIndices.forward])
def test_source_mismatch_fails_closed(function):
    with pytest.raises(ValueError, match="exactly one"):
        _compile_host_parent_forward(function)


def _lifecycle_fixture():
    # Exercise installation/cleanup without allocating a tensor or graph.
    context = object.__new__(CapturedS37Decode)
    context.owner_thread = threading.get_ident()
    context._installed = False
    context._busy = False
    context._original_fk = lambda *args: "original_fk"
    context._original_decode = lambda *args: "original_decode"
    context.model = SimpleNamespace(_fk=context._original_fk, _decode_active=context._original_decode)
    context._validate_runtime = lambda: None
    return context


def test_context_restores_methods_after_body_exception_and_restore_is_idempotent():
    context = _lifecycle_fixture()
    with pytest.raises(LookupError, match="body failure"):
        with context:
            assert context.model._fk == context.fk
            assert context.model._decode_active == context.decode_active
            raise LookupError("body failure")
    assert context.model._fk is context._original_fk
    assert context.model._decode_active is context._original_decode
    context.restore()


def test_restore_preserves_foreign_replacement_but_cleans_owned_other_method():
    context = _lifecycle_fixture().install()
    foreign = lambda *args: "foreign"
    context.model._fk = foreign
    with pytest.raises(RuntimeError, match="foreign method"):
        context.restore()
    assert context.model._fk is foreign
    assert context.model._decode_active is context._original_decode
    assert not context._installed


def test_cross_thread_rejection_precedes_cuda_queries():
    context = object.__new__(CapturedS37Decode)
    context.owner_thread = threading.get_ident() + 1
    with mock.patch.object(torch.cuda, "current_stream", side_effect=AssertionError("touched CUDA")):
        with pytest.raises(RuntimeError, match="another thread"):
            context._validate_runtime()


def test_cpu_model_rejected_before_any_cuda_call():
    model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device("cpu"))]))
    with mock.patch.object(torch.cuda, "current_stream", side_effect=AssertionError("touched CUDA")):
        with pytest.raises(ValueError, match="CUDA model"):
            CapturedS37Decode(model)

