"""T1A-UB engineering fix (docs/EXPERIMENT_RECORDS.md#t1aub-amend1-rows-20261102): S45-SM's router cache per device.

semkine/s45sm.py `_net(device)` caches `_STATE["net"].to(device)` per device, but `nn.Module.to` moves the one loaded router
in place: after a CUDA forward, a CPU forward (make_main_row's MAC count) moves the router the CUDA entry still points to, and
the next CUDA forward (evaluate.row's tail-latency pass) fails with tensors on two devices. `install()` replaces `_net` by a
cache of per-device deep copies of the loaded router (the same weights; numerically identical outputs). Opt-in, in-process.
"""
from __future__ import annotations

import copy

_STATE = {"native": None}


def install():
    from semkine import s45sm as SM
    if _STATE["native"] is not None:
        return
    _STATE["native"] = SM._net

    def _net(device):
        on = SM._STATE["on"]
        net = on.get(device)
        if net is None:
            net = copy.deepcopy(SM._STATE["net"]).to(device)
            on[device] = net
        return net
    SM._net = _net


def uninstall():
    from semkine import s45sm as SM
    if _STATE["native"] is not None:
        SM._net = _STATE["native"]
        _STATE["native"] = None
