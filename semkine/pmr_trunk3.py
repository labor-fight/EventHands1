r"""PMR stage S3b rounds rungs, engineering amendment 3 (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109): D1's R with
fewer rounds of P. MODEL.ENCODER_LAYERS cannot change (semkine/xag.py pins the C1 interfaces 128 x 3 / 32 / 256 whenever
the PFA routes are on, so the rungs registered with ENCODER_LAYERS 1 / 2 stop at model construction); here R is built
as registered (three rounds, R's draws) and then keeps its first k rounds, an opt-in subclass of semkine/pmr_trunk.py's
`PMRNet` (unchanged), installed in place of `semkine.xyz_mesh.XYZMeshNet` (`install("r1" | "r2")`).
"""
from __future__ import annotations

from torch import nn

from . import pmr_trunk as PM
from . import xyz_mesh as XM

ROUNDS = {"r1": 1, "r2": 2}
_STATE = {"variant": None}


class PMRRoundsNet(PM.PMRNet):
    variant = "r"
    keep_rounds = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= self.keep_rounds < len(self.rounds):
            raise ValueError("keep 1 or 2 of R's rounds")
        self.rounds = nn.ModuleList(list(self.rounds)[: self.keep_rounds])
        self.rounds_n = self.keep_rounds


def install(variant):
    if variant not in ROUNDS:
        raise ValueError(f"PMR rounds rung in {tuple(ROUNDS)}")
    if XM.XYZMeshNet is not PM._NATIVE["cls"]:
        raise RuntimeError("XYZMeshNet already patched")
    XM.XYZMeshNet = type(f"PMRRoundsNet_{variant}", (PMRRoundsNet,), {"variant": "r", "keep_rounds": ROUNDS[variant]})
    _STATE["variant"] = variant
    return variant


def uninstall():
    XM.XYZMeshNet = PM._NATIVE["cls"]
    _STATE["variant"] = None


def installed():
    return _STATE["variant"]


__all__ = ["PMRRoundsNet", "ROUNDS", "install", "installed", "uninstall"]
