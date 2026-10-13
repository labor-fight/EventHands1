"""Causal-history adapter around the unchanged canonical tracking evaluator."""
from __future__ import annotations

from inspect import signature

from semkine import eval_track as ET

ORIGINAL_TRACK = ET.track_sequence


def track_sequence(*args, **kwargs):
    bound = signature(ORIGINAL_TRACK).bind(*args, **kwargs)
    bound.apply_defaults()
    params = dict(bound.arguments)
    model = params["model"]
    if not hasattr(model, "reset_time_history"):
        return ORIGINAL_TRACK(*args, **kwargs)
    step_ms = int(params["step_ms"])
    window_ms = int(params["window_ms"] or step_ms)
    if step_ms != window_ms:
        raise RuntimeError("XC formal evaluation fixes query interval = raw window (W50)")
    original_hook = params.get("state_hook")
    state = {"run_step": 0}

    def hook(stage, previous, prediction):
        if stage == "run_start":
            state["run_step"] = 0
            model.reset_time_history()
            if original_hook is not None:
                original_hook(stage, previous, prediction)
            return None
        if original_hook is not None:
            prediction = original_hook(stage, previous, prediction)
        interval = ((max(step_ms, window_ms) - 1) if state["run_step"] == 0 else step_ms) * .001
        model.set_time_history(previous.detach().clone(), interval,
                               query_start_offset_s=.001)
        state["run_step"] += 1
        return prediction

    params["state_hook"] = hook
    try:
        return ORIGINAL_TRACK(**params)
    finally:
        model.reset_time_history()


def install():
    ET.track_sequence = track_sequence
