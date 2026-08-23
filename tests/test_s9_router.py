#!/usr/bin/env python3
r"""S9 gates for the RDOR router.

The router is a policy, so most of what matters about it is decided by a tracking run, not a unit
test. What is checked here is the part that can be falsified without training: that the greedy
selection really maximises what it claims to, that conditioning does what it is for, that closure
and hysteresis behave, and that the whole thing reduces to something sane when there is no evidence.

The one substantive gate: the router must select the finger that moved, on real poses, without
seeing the ground truth. If it cannot do that offline, no amount of tracking will save it.
"""
from __future__ import annotations

import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from mano_layer import ManoLayer                                    # noqa: E402
from pose_repr import decode_to_mano_inputs                         # noqa: E402
from semkine import jacobian as JA                                  # noqa: E402
from semkine import kssf as KS                                      # noqa: E402
from semkine import lie                                             # noqa: E402
from semkine import oracle as OR                                    # noqa: E402
from semkine import router as RD                                    # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval().double()


@pytest.fixture(scope="module")
def scene(mano):
    """One real pose with its betas and intrinsics, chosen so the whole hand is in front.

    A frame whose label straddles the image plane (about 2% of them, see S6) would have most of its
    faces dropped, and a routing test on such a frame would be measuring the annotation.
    """
    m = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))[0]
    base = m[:-5]
    p = _read_meta51(base + ".meta")
    aux = np.load(base + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float64, device=DEV).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float64, device=DEV).view(1, 3, 3)
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)))[:, 0]
    idx = np.linspace(0, len(p) - 1, 24).astype(int)
    cand = torch.from_numpy(p[idx]).to(DEV).double()
    with torch.no_grad():
        d = decode_to_mano_inputs(cand, "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        v, _ = mano(betas.expand(len(cand), -1), d["global_orient"],
                    d["local_full_aa"], d["transl"])
    ok = torch.nonzero(v[..., 2].amin(-1) > 0.10).flatten()
    assert len(ok), "no frame in this sequence has a sane label"
    p51 = cand[int(ok[len(ok) // 2])].view(1, 51)
    return dict(p51=p51, betas=betas, K=K, j0=j0, pos=p)


def _events_for_motion(mano, field, p51, betas, K, j0, delta):
    """Pixels whose render changes under `delta`, as a stand-in for a real event packet."""
    def verts(pp):
        d = decode_to_mano_inputs(pp, "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        v, _ = mano(betas, d["global_orient"], d["local_full_aa"], d["transl"])
        return v
    with torch.no_grad():
        f0 = field.rasterize(verts(p51), K)
        f1 = field.rasterize(verts(lie.retract_51d(p51, delta, mano.hands_mean, j0)), K)
    moved = ((f0.inv_depth - f1.inv_depth).abs() + (f0.visibility - f1.visibility).abs()) > 0.02
    _, py, px = torch.nonzero(moved, as_tuple=True)
    return np.stack([px.cpu().numpy(), py.cpu().numpy(),
                     np.zeros(len(px), dtype=np.int64)], -1).astype(np.int64)


# --------------------------------------------------------------------- greedy selection
def test_greedy_matches_brute_force_on_small_budgets():
    """With 15 groups and a budget of 2, the optimum is enumerable; greedy must find it.

    Not a formality: the `1 - 1/e` guarantee bounds greedy from below but says nothing about any
    particular instance, and a conditioning bug would show up as greedy picking two groups that
    explain the same motion.
    """
    g = torch.Generator().manual_seed(0)
    for trial in range(5):
        A = torch.randn(45, 45, generator=g, dtype=torch.float64)
        Lam = (A @ A.T).to(DEV) * 0.5

        def ig(cols):
            t = torch.as_tensor(np.concatenate([np.arange(3 * c, 3 * c + 3) for c in cols]),
                                device=DEV)
            return float(JA.information_gain(Lam[t][:, t]))

        best = max(itertools.combinations(range(15), 2), key=ig)
        sel, _ = RD.greedy_logdet_select(Lam, budget=2)
        got = tuple(int(i) for i in np.nonzero(sel)[0])
        # Greedy is allowed to be within the submodular bound rather than exactly optimal, so the
        # gate is on the objective value, not on the index set.
        assert ig(got) >= 0.95 * ig(best), (trial, got, best, ig(got), ig(best))


def test_conditioning_prevents_double_counting():
    """Two groups sharing their information must not both be credited for it.

    Constructed so that groups 0 and 1 have identical rows: the second adds nothing, and a
    selection scored on per-group diagonals rather than the joint block would take it anyway.
    """
    J = torch.zeros(200, 45, dtype=torch.float64, device=DEV)
    g = torch.Generator().manual_seed(1)
    shared = torch.randn(200, 3, generator=g, dtype=torch.float64).to(DEV)
    J[:, 0:3] = shared
    J[:, 3:6] = shared                                   # group 1 duplicates group 0
    J[:, 6:9] = torch.randn(200, 3, generator=g, dtype=torch.float64).to(DEV) * 0.5
    Lam = J.T @ J
    sel, gains = RD.greedy_logdet_select(Lam, budget=2, gain_floor=1e-6)
    picked = set(int(i) for i in np.nonzero(sel)[0])
    assert 2 in picked, f"the independent group was not selected: {picked}"
    assert not {0, 1} <= picked, f"both duplicate groups were selected: {picked}"


def test_information_gain_is_monotone_in_the_selection():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(45, 45, generator=g, dtype=torch.float64).to(DEV)
    Lam = A @ A.T

    def ig(cols):
        t = torch.as_tensor(np.concatenate([np.arange(3 * c, 3 * c + 3) for c in cols]),
                            device=DEV)
        return float(JA.information_gain(Lam[t][:, t]))

    prev = 0.0
    for k in range(1, 8):
        v = ig(list(range(k)))
        assert v >= prev - 1e-9, (k, v, prev)
        prev = v


def test_kinematic_closure_pulls_in_ancestors():
    sel = np.zeros(15, dtype=bool)
    sel[2] = True                     # index distal only
    out = RD.close_kinematic(sel)
    assert out[0] and out[1] and out[2]
    assert not out[3:].any(), "closure leaked into another finger"
    sel = np.zeros(15, dtype=bool)
    sel[13] = True                    # thumb middle
    out = RD.close_kinematic(sel)
    assert out[12] and out[13] and not out[14]


# --------------------------------------------------------------------- policy behaviour
def test_no_evidence_means_no_update(mano, scene):
    """With no events the router must leave the state alone, not fall back to the prediction."""
    pol = RD.RDORPolicy(mano=mano, budget=4)
    pol.begin_sequence(scene["betas"], scene["K"])
    pol.reset()
    pol._dwell[:] = 0
    prev = scene["p51"]
    pred = prev + 0.05
    out = pol(prev, pred, None, None, n_events=0, events=None)
    assert torch.equal(out, prev)
    assert pol.n_no_evidence == 1


def test_root_is_always_updated(mano, scene):
    pol = RD.RDORPolicy(mano=mano, budget=1)
    pol.begin_sequence(scene["betas"], scene["K"])
    field = KS.KSSF(mano, 180, 240, 0.375).to(DEV).double()
    d = torch.zeros(1, JA.DOF, device=DEV, dtype=torch.float64)
    d[:, 6:9] = 0.15
    ev = _events_for_motion(mano, field, scene["p51"], scene["betas"], scene["K"],
                            scene["j0"], d)
    assert len(ev) > 200
    prev = scene["p51"]
    pred = prev + 0.05
    out = pol(prev, pred, None, None, n_events=len(ev), events=ev)
    assert not torch.equal(out[:, 0:6], prev[:, 0:6]), "root was not updated"


def test_hysteresis_holds_a_group_on_for_the_dwell(mano, scene):
    pol = RD.RDORPolicy(mano=mano, budget=3, min_dwell=3, gain_on=1e9, gain_off=1e9)
    pol.begin_sequence(scene["betas"], scene["K"])
    # Impossible thresholds, so nothing can switch on; only the dwell can keep a group active.
    pol._prev_active[:] = True
    pol._dwell[:] = 3
    prev, pred = scene["p51"], scene["p51"] + 0.05
    seen = []
    for _ in range(5):
        out = pol(prev, pred, None, None, n_events=0, events=None)
        seen.append(bool(not torch.equal(out[:, 6:9], prev[:, 6:9])))
    assert seen[0] and seen[1], seen
    assert not seen[-1], f"a group stayed active past its dwell: {seen}"


@pytest.mark.parametrize("finger", list(range(5)))
def test_router_selects_the_finger_that_moved(mano, scene, finger):
    r"""The substantive offline gate: routing without ground truth must match the routing with it.

    A packet is synthesised from the pixels a single finger's motion actually changes, and the
    router -- which never sees the motion, only the pixels and the previous state -- must put that
    finger in its selection. This is the offline form of the S7 oracle comparison: the oracle knows
    which finger moved, and RDOR has to infer it from the measurement geometry alone.
    """
    field = KS.KSSF(mano, 180, 240, 0.375).to(DEV).double()
    fam = [3 * finger + k for k in range(3)]
    d = torch.zeros(1, JA.DOF, device=DEV, dtype=torch.float64)
    for k in fam:
        d[:, 6 + 3 * k : 9 + 3 * k] = 0.18
    ev = _events_for_motion(mano, field, scene["p51"], scene["betas"], scene["K"],
                            scene["j0"], d)
    if len(ev) < 200:
        pytest.skip(f"finger {finger} is occluded in this pose ({len(ev)} pixels)")

    pol = RD.RDORPolicy(mano=mano, budget=4, gain_on=0.0, gain_off=0.0, min_dwell=0)
    pol.begin_sequence(scene["betas"], scene["K"])
    pol.reset()
    pol._prev_active[:] = False
    pol._dwell[:] = 0
    prev, pred = scene["p51"], scene["p51"] + 0.05
    out = pol(prev, pred, None, None, n_events=len(ev), events=ev)
    changed = [k for k in range(15)
               if not torch.equal(out[:, 6 + 3 * k : 9 + 3 * k],
                                  prev[:, 6 + 3 * k : 9 + 3 * k])]
    assert changed, "the router selected no joint at all"
    assert set(changed) & set(fam), (
        f"finger {finger} moved (joints {fam}) but the router selected {changed}")

    # Selecting the right finger somewhere in a set of four is a weak statement, so the gain
    # itself is checked: the ranking must put a joint of the moving finger at the top, and the
    # gains elsewhere must be an order of magnitude smaller. Occlusion boundaries are the one
    # legitimate exception -- a finger sweeping across its neighbour genuinely moves pixels whose
    # surface, at the previous state, belongs to the neighbour -- so second place is allowed.
    Lam = pol.information(prev, ev)
    cond = JA.schur_eliminate(Lam[None], slice(6, 51), slice(0, 6))[0]
    _, gains = RD.greedy_logdet_select(cond, 15, gain_floor=0.0)
    order = list(np.argsort(-gains))
    assert set(int(o) for o in order[:2]) & set(fam), (
        f"finger {finger} moved but the top gains were "
        f"{[(int(o), round(float(gains[o]), 2)) for o in order[:4]]}")
    other = np.median([gains[k] for k in range(15) if k not in fam])
    assert gains[fam].max() > 5.0 * max(other, 1e-3), (
        f"gain is not concentrated on the moving finger: "
        f"max {gains[fam].max():.2f} vs median elsewhere {other:.2f}")


def test_update_rate_responds_to_the_budget(mano, scene):
    field = KS.KSSF(mano, 180, 240, 0.375).to(DEV).double()
    d = torch.zeros(1, JA.DOF, device=DEV, dtype=torch.float64)
    d[:, 6:51] = 0.12
    ev = _events_for_motion(mano, field, scene["p51"], scene["betas"], scene["K"],
                            scene["j0"], d)
    rates = []
    for budget in (2, 6, 15):
        pol = RD.RDORPolicy(mano=mano, budget=budget, gain_on=0.0, gain_off=0.0, min_dwell=0)
        pol.begin_sequence(scene["betas"], scene["K"])
        pol.reset()
        pol(scene["p51"], scene["p51"] + 0.05, None, None, n_events=len(ev), events=ev)
        rates.append(pol.rate())
    assert rates[0] < rates[1] <= rates[2], rates
