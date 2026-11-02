r"""S45-SM PFA with the label face routes: the upper bound of the prior-free association (not deployable).

PFA (semkine/s45sm.py, dense form semkine/s45sm_fast.py) with S45-TR T1d's ground-truth XAG face routes in place of a
router: `semkine.xag.sample_sources` stays native (the batch's XAG pseudo-sources, as T1d / S45-AT t1ddz read them),
S45's ray association is skipped (empty association) and every event with a valid source is assigned to its source
face with q = 1 at the camera depth of that surface point on the previous-state mesh; the rest go to the residual
slot. Same config as S45-AT t1ddz (MODEL.XAG_ASSOC: face_route + DATA.XAG_SOURCE_ROOT); only the association changes.
Opt-in: `install()` / `uninstall()`.
"""
from __future__ import annotations

from . import s45sm as SM
from . import s45sm_fast as FAST
from . import xag as XAG
from . import xyz_mesh as XM

_STATE = {"on": False}
STATE = {"router": "gt", "checkpoint": None, "assoc": "pfa", "onhand_tau": None}


def install():
    if XAG.sample_sources is not SM._NATIVE["sample_sources"] or XM.associate_rays is not SM._NATIVE["associate_rays"] \
            or XAG.route_association is not SM._NATIVE["route_association"]:
        raise RuntimeError("another hook already patched XAG / associate_rays")
    XAG.route_association = FAST.pfa_association
    XM.associate_rays = FAST.empty_association
    _STATE["on"] = True
    return dict(STATE)


def uninstall():
    XAG.route_association = SM._NATIVE["route_association"]
    XM.associate_rays = SM._NATIVE["associate_rays"]
    _STATE["on"] = False


def installed():
    if not _STATE["on"] or XAG.route_association is not FAST.pfa_association:
        return None
    return dict(STATE)


__all__ = ["STATE", "install", "installed", "uninstall"]
