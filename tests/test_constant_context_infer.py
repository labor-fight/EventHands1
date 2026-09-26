"""CC1 exact inference contracts; no real checkpoint/data/GPU job is started here.

Run explicitly with CPU library thread limits and CC1_TEST_DEVICE=cpu or the
reserved cuda device. CC0_TEST_DEVICE/SC0_TEST_DEVICE are accepted as fallbacks.
This file was supplied for the parent's unified Debug, not executed by author.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semkine.constant_context_infer import (
    ConstantContextBias, ConstantContextInference, constant_context_mask,
    enable_constant_context_inference,
)
from semkine.event_gnn import EventGNN


DEVICE = torch.device(os.environ.get("CC1_TEST_DEVICE", os.environ.get(
    "CC0_TEST_DEVICE", os.environ.get("SC0_TEST_DEVICE", "cpu"))))


def _load_reference(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SC0 = _load_reference("cc1_sc0_reference", "scratch/goal_20260926/sc0/adapter.py")
CC0 = _load_reference("cc1_cc0_reference", "scratch/goal_20260926/cc0/constant.py")


@pytest.fixture(scope="module", autouse=True)
def _four_threads_and_explicit_device():
    torch.set_num_threads(4)
    torch.empty(0, device=DEVICE)  # Fail if requested device is unavailable.


def _encoder(max_nodes=40):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(20260926)
        encoder = EventGNN(hidden=128, feat_dim=64, max_nodes=max_nodes,
                           k=8, n_layers=3, window=32)
    return encoder.to(DEVICE).eval()


def _reference(max_nodes=40, nonzero=True):
    result = SC0.attach_context(_encoder(max_nodes), "L", seed=413)
    result.context = CC0.ConstantContext(device=DEVICE)
    if nonzero:
        with torch.no_grad():
            result.context.bias.copy_(torch.linspace(-.371, .529, 128, device=DEVICE))
            result.context.bias[0] = 0.0
            result.context.bias[1] = -0.0
    return result.eval()


def _packet(counts):
    rows, ptr = [], [0]
    for batch, count in enumerate(counts):
        i = torch.arange(count, dtype=torch.float32, device=DEVICE)
        rows.append(torch.stack([torch.full_like(i, batch), (i + 3).remainder(240),
                                 (7 * i).remainder(180), i / max(count, 1) * .049,
                                 i.remainder(2)], -1))
        ptr.append(ptr[-1] + count)
    return (torch.cat(rows), torch.tensor(ptr, dtype=torch.long, device=DEVICE),
            torch.full((len(counts),), .05, device=DEVICE))


def _byte_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _tuple_equal(actual, expected):
    assert len(actual) == len(expected) == 6
    for a, b in zip(actual, expected):
        _byte_equal(a, b)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("counts", [[65, 4, 0], [0, 0], [1], [2, 0, 7], [0, 1, 40]])
@pytest.mark.parametrize("nonzero", [False, True])
def test_full_tuple_fp32_bf16_empty_mixed_padding_and_cap(autocast, counts, nonzero):
    reference = _reference(nonzero=nonzero)
    optimized = enable_constant_context_inference(copy.deepcopy(reference))
    args = _packet(counts)
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=autocast):
        expected = reference(*args, return_nodes=True)
        actual = optimized(*args, return_nodes=True)
        plain = optimized(*args)
    _tuple_equal(actual, expected)
    _byte_equal(plain, actual[0])
    if sum(counts) == 0:
        assert actual[0].dtype == reference.embed.weight.dtype
        assert actual[1].shape[1] == 0
    else:
        assert actual[1].shape[1] == 40


@pytest.mark.parametrize("autocast", [False, True])
def test_mask_and_residual_match_old_for_every_degree_and_sparse_mask(autocast):
    mask = torch.ones(3, 42, dtype=torch.bool, device=DEVICE)
    mask[1] = False
    mask[1, [3, 7, 8, 12, 19, 25, 30, 32, 38, 41]] = True
    mask[2] = False
    p = torch.arange(mask.numel() * 3, device=DEVICE).reshape(*mask.shape, 3).float() / 129
    edges, _ = SC0.build_context_edges(p, mask, "L")
    actual_mask = constant_context_mask(mask)
    _byte_equal(actual_mask, edges[2])
    assert set(actual_mask[0].sum(-1).tolist()) == set(range(9))
    assert actual_mask[1, 3].sum().item() == 0
    h = torch.arange(mask.numel() * 128, device=DEVICE).reshape(*mask.shape, 128).float().sin()
    old = CC0.ConstantContext(device=DEVICE)
    new = ConstantContextBias(device=DEVICE)
    with torch.no_grad():
        old.bias.copy_(torch.linspace(-.371, .529, 128, device=DEVICE))
        old.bias[0] = 0.0
        old.bias[1] = -0.0
        new.bias.copy_(old.bias)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=autocast):
            expected = old(h, *edges)
            actual = new(h, actual_mask)
    _byte_equal(actual, expected)


@pytest.mark.parametrize("autocast", [False, True])
def test_sparse_sample_mask_full_forward_preserves_original_fields(autocast):
    reference = _reference(max_nodes=40)
    def sparse_sample(self, events, ptr):
        src, mask = EventGNN._sample(self, events, ptr)
        mask[:, :3] = False
        mask[:, 5::3] = False
        return src, mask
    reference._sample = MethodType(sparse_sample, reference)
    optimized = enable_constant_context_inference(copy.deepcopy(reference))
    with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=autocast):
        expected = reference(*_packet([40, 0]), return_nodes=True)
        actual = optimized(*_packet([40, 0]), return_nodes=True)
    _tuple_equal(actual, expected)


def test_loaded_install_preserves_state_keys_objects_and_strict_reload():
    old = _reference()
    candidate = copy.deepcopy(old)
    names = set(candidate.state_dict())
    parameter_ids = {name: id(p) for name, p in candidate.named_parameters()}
    proj, layers = candidate.proj, tuple(candidate.layers)
    identity = id(candidate)
    optimized = enable_constant_context_inference(candidate)
    assert id(optimized) == identity
    assert set(optimized.state_dict()) == names
    assert optimized.proj is proj
    assert all(a is b for a, b in zip(optimized.layers, layers))
    assert {name: id(p) for name, p in optimized.named_parameters()} == parameter_ids
    all_parameters = list(optimized.named_parameters(remove_duplicate=False))
    assert len(all_parameters) == len({id(p) for _, p in all_parameters})
    assert all(not p.requires_grad for p in optimized.parameters())
    fresh = enable_constant_context_inference(_encoder())
    fresh.load_state_dict(old.state_dict(), strict=True)
    assert set(fresh.state_dict()) == names
    with torch.no_grad():
        _tuple_equal(fresh(*_packet([40, 4]), return_nodes=True), old(*_packet([40, 4]), return_nodes=True))


@pytest.mark.parametrize("counts,expected_calls", [([0, 0], 0), ([40, 0], 1), ([1], 1)])
def test_one_projection_no_context_edges_and_no_temporary_readout_mutation(counts, expected_calls):
    optimized = enable_constant_context_inference(_reference())
    calls = []
    flags = []
    def projection_hook(module, args, output):
        calls.append(1)
        flags.append(optimized.readout)
    hooks = [optimized.proj.register_forward_hook(projection_hook)]
    for layer in optimized.layers:
        hooks.append(layer.register_forward_pre_hook(lambda module, args: flags.append(optimized.readout)))
    try:
        with mock.patch.object(SC0, "build_context_edges", side_effect=AssertionError("unused context graph called")):
            optimized(*_packet(counts), return_nodes=True)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(calls) == expected_calls
    assert all(flag is False for flag in flags)
    assert optimized.readout is False
    assert optimized._edges.__func__ is EventGNN._edges
    assert not hasattr(optimized, "_context_p")


def test_inference_only_explicit_install_and_no_scratch_dependency():
    optimized = enable_constant_context_inference(_encoder())
    assert isinstance(optimized, ConstantContextInference)
    with pytest.raises(RuntimeError, match="inference-only"):
        optimized.train(True)
    optimized.eval()
    with pytest.raises(TypeError):
        enable_constant_context_inference(optimized)
    with torch.enable_grad():
        assert not optimized(*_packet([4])).requires_grad
    source = (ROOT / "semkine/constant_context_infer.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "scratch" not in (node.module or "")
        if isinstance(node, ast.Import):
            assert all("scratch" not in alias.name for alias in node.names)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ConstantContextInference")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    for node in ast.walk(forward):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
            assert node.attr != "readout"
        if isinstance(node, ast.Name):
            assert node.id != "build_context_edges"


def test_rejects_nonconstant_context_without_mutating_it():
    message_encoder = SC0.attach_context(_encoder(), "L", seed=1)
    old_class = message_encoder.__class__
    old_context = message_encoder.context
    with pytest.raises(ValueError, match="only the CC0 bias"):
        enable_constant_context_inference(message_encoder)
    assert message_encoder.__class__ is old_class
    assert message_encoder.context is old_context
    assert message_encoder.readout is True
