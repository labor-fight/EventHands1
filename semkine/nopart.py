"""ESMNet without the part head p_i (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113).

A router checkpoint saved with `nopart: True` (tools/nopart/router_sim.py) was trained without the part loss: its part
logits are untrained and must never be read. Loading such a checkpoint through semkine/pfo_router.py `load` (the single
entry every router path uses: S45-SM's `_load_router`, the RDR hook's wrapper, the readings) replaces every part's
candidate-face table by all faces, so both decoders -- semkine/s45sm.py `SRNet.route` and semkine/s45sm_fast.py `route`
-- return the face whose rest-template centroid is nearest to c_i over the whole hand (the same `project_bary` follows).
The argmax of the untrained part logits then selects a table that holds every face, i.e. it no longer matters.
Checkpoints without the flag load unchanged. Opt-in: nothing happens until `enable()`.
"""
from __future__ import annotations

import torch

_STATE = {"on": False}


def unconstrained(net):
    """Every part's candidate list = all faces (valid everywhere): decoding by c_i alone."""
    n_faces = int(net.faces.shape[0])
    k = int(net.part_faces.shape[0])
    dev = net.part_faces.device
    net.part_faces = torch.arange(n_faces, device=dev, dtype=net.part_faces.dtype).unsqueeze(0).expand(k, n_faces).contiguous()
    net.part_faces_valid = torch.ones(k, n_faces, dtype=torch.bool, device=dev)
    net.nopart = True
    return net


def enable():
    if _STATE["on"]:
        return
    from semkine import pfo_router as PR
    native = PR.load

    def load(path, map_location="cpu"):
        net, obj = native(path, map_location=map_location)
        if obj.get("nopart"):
            unconstrained(net)
        return net, obj

    PR.load = load
    _STATE["on"] = True


__all__ = ["enable", "unconstrained"]
