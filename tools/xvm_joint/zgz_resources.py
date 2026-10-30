"""Stateful XJ packet cost helpers for the unchanged main-table formatter.

No dataset is opened here. The caller supplies chronological, pre-staged
packets (the main-table latency sequence is lyq_local) and the separately
measured original EventHands-Full anchor. Complete ``forward_packet`` work,
including the XJ CPU field/geometry construction and history commit, is timed.
Disk reads and staging packets on the device retain the existing table's
exclusion. This is a packet-service benchmark, not a stream I/O benchmark.

THOP counts all standard layers invoked in the adapter, including both the
canonical S43 and local CNN/MLPs. As in tools/make_s36_row.py, functional MANO,
finite differences, raster/SDF, assignment/scatter and linear solves are not
MAC-profiled; the figure is explicitly a standard-layer lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn


FULL_PUBLISHED_MS = 1.75
MAC_NOTE = (
    "whole forward_packet, THOP standard layers including canonical S43 and "
    "72-record local CNN/MLPs when present (global control has no local operator); "
    "lower bound under the existing main-table convention. "
    "Functional MANO FK (including finite-difference calls), projection/Jacobian "
    "products, raster/SDF, grid sampling, assignment/scatter and SPD linear solve "
    "are excluded. This is not a complete arithmetic-operation count."
)


@dataclass(frozen=True)
class PacketCase:
    batch: Any
    reset: bool = False
    run_start_ms: int = 0
    sequence_id: Any = None


def _sync(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _terminal_digest(value):
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _reset_case(model, case, previous):
    # Raw XJ owns an optional absolute run clock. The S39 wrapper follows the
    # original evaluator's reset_state(previous), which also initializes its
    # stateful SO(3)/velocity filter. Do not bypass that filter initialization.
    if callable(getattr(model, "reset_run", None)):
        model.reset_run(start_ms=int(case.run_start_ms), sequence_id=case.sequence_id)
    elif callable(getattr(model, "reset_state", None)):
        model.reset_state(previous)
    else:
        raise TypeError("A run reset interface is required")


@torch.no_grad()
def _replay(model, cases: Sequence[PacketCase], guard_receipts=None):
    from tools.s43_rel.access_guard import InferenceAccessGuard
    previous = None
    with InferenceAccessGuard() as guard:
        for index, case in enumerate(cases):
            if index == 0 or case.reset:
                previous = case.batch.prev_state.detach().clone()
                _reset_case(model, case, previous)
            batch = copy.copy(case.batch)
            # Never carry the original benchmark packet's static previous state
            # into later windows. Each arm/benchmark uses its own emitted state.
            batch.prev_state = previous
            previous = model.forward_packet(batch)
    if guard_receipts is not None:
        guard_receipts.append(guard.receipt())
    return previous


@torch.no_grad()
def benchmark_packet_replay(model, cases: Iterable[PacketCase], *, device,
                            anchor_raw_ms: float, max_packets: int = 600,
                            passes: int = 3, warmup_packets: int = 100):
    """Minimum pass-mean latency with identical causal replay on every pass.

    Call this serially on an otherwise unoccupied GPU. Packet creation and
    staging precede this function. Original model history is restored even on
    failure; the benchmark must not change the accuracy replay's initial state.
    The caller's supplied first/run-start previous state is the declared run
    initialization, rather than a new target read performed by this helper.
    """
    if not math.isfinite(anchor_raw_ms) or anchor_raw_ms <= 0:
        raise ValueError("A measured positive EventHands-Full anchor is required")
    if min(max_packets, passes) <= 0 or warmup_packets < 0:
        raise ValueError("Positive packet/pass counts and nonnegative warmup required")
    cases = list(cases)[:int(max_packets)]
    if not cases:
        raise ValueError("Cannot benchmark an empty packet sequence")
    required = ("snapshot_state", "restore_state", "forward_packet")
    if any(not callable(getattr(model, key, None)) for key in required):
        raise TypeError("XJ stateful adapter interface is incomplete")
    if not any(callable(getattr(model, key, None)) for key in ("reset_run", "reset_state")):
        raise TypeError("XJ stateful adapter reset interface is incomplete")
    snapshot = model.snapshot_state()
    means, terminal_hashes, guard_receipts = [], [], []
    counts = [int(case.batch.events.shape[0]) for case in cases]
    try:
        if warmup_packets:
            _replay(model, cases[:min(len(cases), int(warmup_packets))], guard_receipts)
            _sync(device)
        for _ in range(int(passes)):
            model.restore_state(snapshot)
            _sync(device)
            started = time.perf_counter()
            terminal = _replay(model, cases, guard_receipts)
            _sync(device)
            means.append((time.perf_counter()-started)*1000./len(cases))
            # Diagnostics and CPU transfers happen outside the timed region.
            if not torch.isfinite(terminal).all():
                raise ArithmeticError("Nonfinite benchmark output")
            terminal_hashes.append(_terminal_digest(terminal))
        raw = min(means)
        return {
            "latency_ms_raw": raw,
            "latency_ms_scaled_full1p75": raw / float(anchor_raw_ms) * FULL_PUBLISHED_MS,
            "anchor_full_raw_ms": float(anchor_raw_ms),
            "latency_packets": len(cases),
            "latency_mean_events_per_packet": int(np.mean(counts)),
            "latency_pass_means_ms": means,
            "latency_pass_terminal_state_sha256": terminal_hashes,
            "latency_pass_terminal_states_equal": len(set(terminal_hashes)) == 1,
            "latency_warmup_packets": min(len(cases), int(warmup_packets)),
            "latency_inference_access_audits": guard_receipts,
            "latency_scope": "complete stateful XJ forward_packet, own-state feedback and run resets; packet disk I/O/device staging excluded as in existing table",
            "latency_sequence": "caller-supplied lyq_local training sequence; no zgz timing or accuracy reads in this helper",
        }
    finally:
        model.restore_state(snapshot)


class _PacketWrap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model.forward_packet(batch)


def parameter_counts(model):
    """Count all persistent parameters, including the frozen canonical branch."""
    total = sum(parameter.numel() for parameter in model.parameters())
    core = model
    while not hasattr(core, "base") and isinstance(getattr(core, "core", None), nn.Module):
        core = core.core
    local_module, base_module = getattr(core, "local", None), getattr(core, "base", None)
    local = sum(parameter.numel() for parameter in local_module.parameters()) if isinstance(local_module, nn.Module) else 0
    base = sum(parameter.numel() for parameter in base_module.parameters()) if isinstance(base_module, nn.Module) else 0
    return {"params_total": int(total), "params_local": int(local), "params_canonical": int(base)}


@torch.no_grad()
def profile_packet_layers(model, case: PacketCase, *, priming_cases: Sequence[PacketCase] = ()):
    """Profile a fresh disposable adapter on a synthetic saturated packet.

    The passed model must be a separate cost-only instance: THOP temporarily
    installs operation counters. Do not reuse it for the accuracy replay. Only
    CPU execution is accepted, matching the existing main-table MAC method.
    """
    from thop import profile
    from tools.s43_rel.access_guard import InferenceAccessGuard
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("The standard-layer MAC fixture runs on CPU")
    counts = parameter_counts(model)
    previous = _replay(model, list(priming_cases)) if priming_cases else None
    if not priming_cases or case.reset:
        previous = case.batch.prev_state.detach().clone()
        _reset_case(model, case, previous)
    batch = copy.copy(case.batch)
    batch.prev_state = previous
    with InferenceAccessGuard() as profile_guard:
        macs, _ = profile(_PacketWrap(model.eval()), inputs=(batch,), verbose=False)
    if not math.isfinite(float(macs)) or macs <= 0:
        raise ArithmeticError("Standard-layer profile did not produce finite positive MACs")
    return {
        **counts,
        "macs_forward_packet": float(macs),
        "macs_forward_packet_note": MAC_NOTE,
        "macs_fixture_events": int(batch.events.shape[0]),
        "macs_fixture_scope": "synthetic saturated packet; no dataset or GT source read",
        "macs_local_records_upper_bound": 72 if counts["params_local"] else 0,
        "macs_inference_access_audit": profile_guard.receipt(),
    }


def synthetic_profile_cases(*, window_ms: int = 50, count: int = 3,
                            n_events: int = 4096, seed: int = 0):
    """The original table's spatial/count fixture, with causal priming packets.

    Packet clock fields use the same relative-window convention as
    semkine.eval_track.make_eval_packet. The stateful adapter owns the causal
    absolute clock, advanced by 50 ms per update after its first full window.
    """
    from semkine.events import EventPacketBatch
    if window_ms not in (50, 200) or count < 1 or n_events < 1:
        raise ValueError("Declared W50/W200 packet fixture and positive sizes required")
    generator = torch.Generator().manual_seed(int(seed))
    previous = torch.zeros(1, 51)
    previous[:, 2] = .45
    K = torch.tensor([[[603.4507, 0., 325.09183], [0., 602.95654, 242.09796], [0., 0., 1.]]])
    result = []
    for index in range(int(count)):
        event = torch.zeros(n_events, 5)
        event[:, 1] = (122+30*torch.randn(n_events, generator=generator)).clamp(0, 239).round()
        event[:, 2] = (91+30*torch.randn(n_events, generator=generator)).clamp(0, 179).round()
        # ET encodes integer us by float32 multiplication; a generic linspace
        # in seconds would fail that exact round-trip timestamp contract.
        integer_us = torch.linspace(0., window_ms*1000-1, n_events).round().to(torch.int64)
        event[:, 3] = integer_us.float()*np.float32(1e-6)
        event[:, 4] = (torch.rand(n_events, generator=generator) > .5).float()
        batch = EventPacketBatch(events=event, ptr=torch.tensor([0, n_events]),
            sequence_id=torch.zeros(1, dtype=torch.long), t_start_us=torch.zeros(1, dtype=torch.long),
            t_end_us=torch.tensor([window_ms*1000]), delta_t_s=torch.tensor([window_ms/1000.]),
            is_sequence_start=torch.tensor([index == 0]), is_sequence_end=torch.zeros(1, dtype=torch.bool),
            target=torch.zeros_like(previous), prev_state=previous.clone(), betas=torch.zeros(1, 10),
            camera_K=K.clone(), lnes=None)
        result.append(PacketCase(batch, reset=index == 0, run_start_ms=0, sequence_id=0))
    return result


def measure_resources(make_model, cases: Iterable[PacketCase], *, device,
                      anchor_raw_ms: float, window_ms: int = 50,
                      max_packets: int = 600, passes: int = 3,
                      warmup_packets: int = 100):
    """Factory entry point; ``make_model(device)`` binds arm/protocol/weights.

    Timing and profiling always use distinct model instances. A protocol/arm
    can be captured by the caller's closure. Thus a S39 filter wrapper is timed
    together with its core, while the W50 adapter follows the identical API.
    The caller must stage each case on ``device`` before entering this helper.
    """
    cases = list(cases)
    model = make_model(torch.device(device)).eval()
    timing = benchmark_packet_replay(model, cases, device=device,
        anchor_raw_ms=anchor_raw_ms, max_packets=max_packets, passes=passes,
        warmup_packets=warmup_packets)
    timed_counts = parameter_counts(model)
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    cpu_model = make_model(torch.device("cpu")).eval()
    fixtures = synthetic_profile_cases(window_ms=window_ms)
    profile = profile_packet_layers(cpu_model, fixtures[-1], priming_cases=fixtures[:-1])
    if parameter_counts(cpu_model) != timed_counts:
        raise RuntimeError("Timing and CPU profile model parameter identities differ")
    del cpu_model
    return {**timing, **profile, "resource_model_instances": "separate disposable timing/CPU-profile instances", "resource_profile_window_ms": int(window_ms)}
