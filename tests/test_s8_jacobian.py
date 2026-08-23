#!/usr/bin/env python3
r"""S8 gates for the projection Jacobian and the observability audit.

The Jacobian is verified three ways, deliberately redundantly, because an error here is invisible
downstream: a wrong `J` yields a Fisher matrix that is still symmetric positive definite and a
router that still looks decisive, just about the wrong joints.

1. **Against `ManoLayer` itself, via autograd.** The reference differentiates the production layer
   through the S3 retraction, so agreement means the analytic form describes the model the loss is
   actually fitting -- including the pose blendshapes that the articulated-tracking literature's
   Jacobian predates.
2. **Against central differences of the forward model.** Catches anything both implementations
   could get wrong together, in particular a sign or pivot convention error.
3. **Against finite rotations, not just infinitesimal ones.** The predicted displacement must match
   the true one to second order as the step shrinks; a first-order-only match would pass a loose
   tolerance while hiding a wrong constant.

Then the counterfactual audit the plan registers: eliminating the root by Schur complement must
collapse the finger information when only the root is observable, and a single moving finger must
concentrate the information in that finger's own columns.
"""
from __future__ import annotations

import glob
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
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W, SCALE = 180, 240, 0.375


@pytest.fixture(scope="module")
def mano():
    # float64 throughout: this module is checked against finite differences, and a float32 central
    # difference cannot resolve a 1e-6 relative error.
    m = ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval()
    return m.double()


@pytest.fixture(scope="module")
def sample(mano):
    """A small batch of real poses with their betas and intrinsics."""
    metas = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))
    P, Bt, Km = [], [], []
    for m in metas[::17][:3]:
        base = m[:-5]
        p = _read_meta51(base + ".meta")
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        b = torch.tensor(aux["betas"], dtype=torch.float64, device=DEV).view(1, -1)
        k = torch.tensor(aux["camera_K"], dtype=torch.float64, device=DEV).view(1, 3, 3)
        idx = np.linspace(0, len(p) - 1, 4).astype(int)
        pp = torch.from_numpy(p[idx]).to(DEV).double()
        P.append(pp)
        Bt.append(b.expand(len(pp), -1))
        Km.append(k.expand(len(pp), -1, -1))
    p51, betas, K = torch.cat(P), torch.cat(Bt), torch.cat(Km)
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)))[:, 0]
    state = lie.state_from_51d(p51, mano.hands_mean, j0)
    fk = JA.forward_kinematics(mano, state, betas)
    return dict(p51=p51, betas=betas, K=K, j0=j0, state=state, fk=fk)


def _mano_verts(mano, p51, betas):
    dec = decode_to_mano_inputs(p51, "mano_full_axis_angle",
                                mano.hands_components, mano.hands_mean)
    v, _ = mano(betas, dec["global_orient"], dec["local_full_aa"], dec["transl"])
    return v


# ------------------------------------------------------------------ forward model
def test_forward_kinematics_reproduces_mano(mano, sample):
    """The rebuilt chain must reproduce `ManoLayer` to numerical precision.

    This is what licenses the `G_0 = [R_root | p_root + R_root j0]` identity, and with it the claim
    that a root rotation pivots at `p_root`.

    The floor is not float64 epsilon but MANO's own: `batch_rodrigues` adds 1e-8 before taking the
    axis norm, which biases every rotation by about that much. S3 measured the same offset on the
    51D round trip. Anything above ~1e-7 m would be a genuine kinematics error.
    """
    fk, p51, betas = sample["fk"], sample["p51"], sample["betas"]
    ref = _mano_verts(mano, p51, betas)
    B, V = ref.shape[:2]
    b = torch.arange(B, device=DEV)[:, None].expand(B, V).reshape(-1)
    v = torch.arange(V, device=DEV)[None].expand(B, V).reshape(-1)
    got = JA.skin(fk, mano.weights[v], b, v).reshape(B, V, 3)
    assert float((got - ref).abs().max()) < 1e-7, float((got - ref).abs().max())


def test_pose_blendshapes_are_not_negligible(mano, sample):
    """Quantify the term the classical articulated Jacobian drops, rather than assuming it away."""
    fk = sample["fk"]
    B = sample["p51"].shape[0]
    g = torch.Generator(device="cpu").manual_seed(0)
    n = 3000
    b = torch.randint(0, B, (n,), generator=g).to(DEV)
    v = torch.randint(0, 778, (n,), generator=g).to(DEV)
    rep = JA.jacobian_report(mano, fk, b, v)
    # Both approximations are recorded in the log; the gate is only that they were measured and are
    # in a plausible range, since the analytic path uses the exact form regardless.
    assert 0.0 <= rep["posedirs_rel_median"] < 1.0
    assert 0.0 <= rep["blended_approx_rel_median"] < 1.0
    print("\n  jacobian_report:", {k: round(v, 6) if isinstance(v, float) else v
                                   for k, v in rep.items()})


# ------------------------------------------------------------------ analytic vs autograd
def test_analytic_matches_autograd(mano, sample):
    fk, p51, betas = sample["fk"], sample["p51"], sample["betas"]
    B = p51.shape[0]
    g = torch.Generator(device="cpu").manual_seed(1)
    n = 40
    b = torch.randint(0, B, (n,), generator=g).to(DEV)
    v = torch.randint(0, 778, (n,), generator=g).to(DEV)
    _, J = JA.vertex_jacobian(mano, fk, b, v, include_posedirs=True)
    Jref = JA.autograd_vertex_jacobian(mano, p51, betas, b, v)
    err = (J - Jref).abs().max()
    rel = err / Jref.abs().max().clamp_min(1e-12)
    assert float(rel) < 1e-8, f"analytic vs autograd relative error {float(rel):.3e}"


def test_analytic_matches_central_differences(mano, sample):
    """Central differences of the retraction, independent of both implementations."""
    fk, p51, betas, j0 = sample["fk"], sample["p51"], sample["betas"], sample["j0"]
    B = p51.shape[0]
    g = torch.Generator(device="cpu").manual_seed(2)
    n = 24
    b = torch.randint(0, B, (n,), generator=g).to(DEV)
    v = torch.randint(0, 778, (n,), generator=g).to(DEV)
    _, J = JA.vertex_jacobian(mano, fk, b, v)

    h = 1e-6
    worst = 0.0
    for a in range(JA.DOF):
        d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
        d[:, a] = h
        vp = _mano_verts(mano, lie.retract_51d(p51, d, mano.hands_mean, j0), betas)
        vm = _mano_verts(mano, lie.retract_51d(p51, -d, mano.hands_mean, j0), betas)
        fd = (vp[b, v] - vm[b, v]) / (2 * h)
        worst = max(worst, float((fd - J[:, :, a]).abs().max()))
    assert worst < 1e-6, f"worst central-difference disagreement {worst:.3e} m/unit"


@pytest.mark.parametrize("axis", [3, 4, 5, 6, 20, 50])
def test_prediction_is_second_order_accurate(mano, sample, axis):
    """Halving the step must quarter the residual, which pins the constant and not just the sign.

    Restricted to the vertices the axis actually moves. On a vertex a joint barely reaches, the
    true quadratic term falls below MANO's own 2e-8 m `batch_rodrigues` bias and the ratio measures
    round-off rather than truncation -- the test would then fail on a perfectly correct Jacobian.
    """
    fk, p51, betas, j0 = sample["fk"], sample["p51"], sample["betas"], sample["j0"]
    B = p51.shape[0]
    b = torch.zeros(778, dtype=torch.long, device=DEV)
    v = torch.arange(778, device=DEV)
    X0, J = JA.vertex_jacobian(mano, fk, b, v)
    resp = J[:, :, axis].norm(dim=-1)
    v = v[torch.topk(resp, 50).indices]
    b = b[:50]
    X0, J = X0[torch.topk(resp, 50).indices], J[torch.topk(resp, 50).indices]
    errs = []
    for h in (4e-2, 2e-2, 1e-2):
        d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
        d[:, axis] = h
        X1 = _mano_verts(mano, lie.retract_51d(p51, d, mano.hands_mean, j0), betas)[b, v]
        errs.append(float((X1 - X0 - h * J[:, :, axis]).abs().max()))
    # The smallest error must stay an order of magnitude above MANO's ~2e-8 m bias, or the ratios
    # below would be measuring round-off instead of truncation.
    assert errs[-1] > 3e-7, errs
    # Two successive halvings should each cut the error by ~4; allow slack for round-off.
    assert errs[0] / max(errs[1], 1e-18) > 3.2, errs
    assert errs[1] / max(errs[2], 1e-18) > 3.2, errs


def test_projection_jacobian_matches_central_differences(mano, sample):
    fk, K = sample["fk"], sample["K"]
    n = 500
    X = fk.A[:, 0, :, 3][:, None, :] + torch.randn(fk.A.shape[0], n, 3,
                                                   device=DEV, dtype=torch.float64) * 0.05
    X = X.reshape(-1, 3)
    X[:, 2] = X[:, 2].abs().clamp_min(0.2)
    bi = torch.arange(fk.A.shape[0], device=DEV)[:, None].expand(-1, n).reshape(-1)
    uv, d = JA.project_jacobian(X, K, bi, SCALE)
    h = 1e-7
    for c in range(3):
        e = torch.zeros_like(X)
        e[:, c] = h
        up, _ = JA.project_jacobian(X + e, K, bi, SCALE)
        um, _ = JA.project_jacobian(X - e, K, bi, SCALE)
        fd = (up - um) / (2 * h)
        assert float((fd - d[:, :, c]).abs().max()) < 1e-4


# ------------------------------------------------------------------ residual row
def test_residual_row_is_the_normal_projection_of_the_pixel_jacobian(mano, sample):
    """`dr/dx = -n^T J_uv`, and the residual must vanish when the pixel is the projection."""
    fk, K = sample["fk"], sample["K"]
    B = fk.A.shape[0]
    field = KS.KSSF(mano, H, W, SCALE).to(DEV).double()
    verts = JA.skin(fk,
                    mano.weights[torch.arange(778, device=DEV).repeat(B)],
                    torch.arange(B, device=DEV)[:, None].expand(B, 778).reshape(-1),
                    torch.arange(778, device=DEV).repeat(B)).reshape(B, 778, 3)
    with torch.no_grad():
        f = field.rasterize(verts, K)
    ys, xs = torch.nonzero(f.face_id.reshape(B, -1) >= 0, as_tuple=True)
    pick = torch.randperm(len(ys), device=DEV)[:2000]
    bi, flat = ys[pick], xs[pick]
    py, px = flat // W, flat % W
    fid = f.face_id[bi, py, px]
    bary = f.bary[bi, py, px]
    nrm = f.sdf_normal[bi, py, px]
    pix = torch.stack([px, py], -1).double()

    out = JA.query_jacobian(mano, fk, K, SCALE, bi, fid, bary, normal=nrm, pixel=pix)
    # The queried point reprojects into the pixel it was rasterised from, to within a pixel.
    assert float((out.uv - pix).abs().max()) < 1.5
    expect = -(nrm[:, None, :] @ out.J_uv)[:, 0, :]
    assert torch.allclose(out.J_r, expect, atol=1e-12)


# ------------------------------------------------------------------ observability audit
def _verts_of(mano, p51, betas):
    return _mano_verts(mano, p51, betas)


def _simulate_events(mano, field, p51, betas, K, delta, j0, min_events=200):
    r"""Where would events fire if the state moved by `delta`, and what do they constrain?

    An event says "brightness changed at this pixel", and brightness changes where a surface edge
    sweeps across it. So the event locations are taken to be the pixels whose rendered silhouette
    or inverse depth actually changes between the two states -- not a hand-picked region of
    interest, which would beg the question the audit is asking.

    Everything the estimator sees is evaluated at the *previous* state: the face, the barycentric
    coordinates and the contour normal all come from the field before the motion, exactly as they
    would at inference. Only the set of firing pixels comes from the pair.
    """
    v0 = _verts_of(mano, p51, betas)
    v1 = _verts_of(mano, lie.retract_51d(p51, delta, mano.hands_mean, j0), betas)
    with torch.no_grad():
        f0 = field.rasterize(v0, K)
        f1 = field.rasterize(v1, K)
    moved = ((f0.inv_depth - f1.inv_depth).abs() + (f0.visibility - f1.visibility).abs()) > 0.02
    # A residual is only defined where the field supplies a face and a contour normal.
    usable = moved & (f0.face_id >= 0) & (f0.sdf_normal.norm(dim=-1) > 0.5)
    bi, py, px = torch.nonzero(usable, as_tuple=True)
    if bi.numel() < min_events:
        return None
    state = lie.state_from_51d(p51, mano.hands_mean, j0)
    fk = JA.forward_kinematics(mano, state, betas)
    out = JA.query_jacobian(mano, fk, K, SCALE, bi, f0.face_id[bi, py, px],
                            f0.bary[bi, py, px],
                            normal=f0.sdf_normal[bi, py, px],
                            pixel=torch.stack([px, py], -1).to(p51.dtype))
    return JA.fisher(out.J_r, batch_index=bi, n_batch=p51.shape[0]), bi


@pytest.fixture(scope="module")
def field(mano):
    return KS.KSSF(mano, H, W, SCALE).to(DEV).double()


def _fisher_on_mask(mano, field, p51, betas, K, j0, f0, mask):
    """Fisher information from events supported only on `mask` pixels."""
    usable = mask & (f0.face_id >= 0) & (f0.sdf_normal.norm(dim=-1) > 0.5)
    bi, py, px = torch.nonzero(usable, as_tuple=True)
    if bi.numel() < 100:
        return None, None
    state = lie.state_from_51d(p51, mano.hands_mean, j0)
    fk = JA.forward_kinematics(mano, state, betas)
    out = JA.query_jacobian(mano, fk, K, SCALE, bi, f0.face_id[bi, py, px],
                            f0.bary[bi, py, px], normal=f0.sdf_normal[bi, py, px],
                            pixel=torch.stack([px, py], -1).to(p51.dtype))
    return JA.fisher(out.J_r, batch_index=bi, n_batch=p51.shape[0]), bi


def test_event_support_not_motion_is_what_the_information_depends_on(mano, sample, field):
    r"""The plan's root-Schur counterfactual, restated so that it is falsifiable.

    As originally registered the gate was "under root-only motion, finger information gain drops
    60% after eliminating the root". Measured, it drops 5%, and it should: the Fisher matrix
    `Lambda = sum_i J_i^T R_i^-1 J_i` is a function of *where the events are* and of the local
    geometry, and contains no reference whatsoever to which degree of freedom happened to move. A
    rigid wrist rotation sweeps edges across the fingers too, and at those pixels a finger rotation
    really would change the residual, so the finger columns are genuinely informative and are not
    collinear with the root columns. No Schur complement can remove information that is there.

    The well-posed version of the same intent is about support. Events confined to the palm and
    wrist must carry little information about finger angles, and events on a finger must carry a
    lot -- and it is that contrast, not the identity of the moving joint, that a router can act on.
    This is the property S9 actually needs, and the version that is gated here.
    """
    p51, betas, K, j0 = sample["p51"], sample["betas"], sample["K"], sample["j0"]
    with torch.no_grad():
        f0 = field.rasterize(_verts_of(mano, p51, betas), K)
    names = (f0.lbs_indices.long(), f0.lbs_weights)
    # "Palm" = pixels whose dominant skinning joint is the wrist; "finger g" = pixels dominated by
    # one of that finger's three joints.
    dom = torch.gather(names[0], -1, names[1].argmax(-1, keepdim=True))[..., 0]
    palm = (dom == 0) & (f0.face_id >= 0)
    Lam_palm, _ = _fisher_on_mask(mano, field, p51, betas, K, j0, f0, palm)
    assert Lam_palm is not None, "no palm-dominated pixels"
    ig_palm = JA.group_information(Lam_palm, eliminate_root=True)

    drops = []
    for fg in range(5):
        fam = [1 + 3 * fg + d for d in range(3)]
        m = torch.zeros_like(palm)
        for k in fam:
            m |= dom == k
        Lam_f, _ = _fisher_on_mask(mano, field, p51, betas, K, j0, f0, m & (f0.face_id >= 0))
        if Lam_f is None:
            continue
        ig_f = JA.group_information(Lam_f, eliminate_root=True)
        for k in fam:
            a = ig_f[f"j{k - 1}"]
            b = ig_palm[f"j{k - 1}"]
            ok = torch.isfinite(a) & torch.isfinite(b) & (a > 1e-6)
            if bool(ok.any()):
                drops.append((b[ok] / a[ok].clamp_min(1e-12)))
    assert drops, "no usable finger supports"
    med = float(torch.cat(drops).median())
    assert med < 0.40, (
        f"palm-only events retain {100 * med:.1f}% of the information that events on the finger "
        f"itself carry about that finger; the registered contrast is a 60% drop")


def test_normal_equations_recover_the_true_motion(mano, sample, field):
    r"""The decisive test of the whole S8 chain: solve for the motion and compare to the truth.

    This is what the plan's counterfactual was reaching for. Attribution -- "did the wrist move or
    did a finger move" -- is a property of the *residuals*, not of the information matrix, and it
    is settled by solving the normal equations `Lambda dx = J^T r`. If the Jacobian, the residual
    convention and the information assembly are all consistent, a root-only motion must come back
    as a root-only solution and a finger motion as that finger's, with no tuning anywhere.
    """
    p51, betas, K, j0 = sample["p51"], sample["betas"], sample["K"], sample["j0"]
    B = p51.shape[0]
    state = lie.state_from_51d(p51, mano.hands_mean, j0)
    fk = JA.forward_kinematics(mano, state, betas)
    with torch.no_grad():
        f0 = field.rasterize(_verts_of(mano, p51, betas), K)

    def solve(d_true):
        bi, py, px = torch.nonzero((f0.face_id >= 0)
                                  & (f0.sdf_normal.norm(dim=-1) > 0.5), as_tuple=True)
        fid, bary = f0.face_id[bi, py, px], f0.bary[bi, py, px]
        nrm = f0.sdf_normal[bi, py, px]
        cur = JA.query_jacobian(mano, fk, K, SCALE, bi, fid, bary)
        st1 = lie.state_from_51d(lie.retract_51d(p51, d_true, mano.hands_mean, j0),
                                 mano.hands_mean, j0)
        fk1 = JA.forward_kinematics(mano, st1, betas)
        nxt = JA.query_jacobian(mano, fk1, K, SCALE, bi, fid, bary)
        # The observed quantity: how far the surface point moved along the contour normal.
        r = (nrm * (nxt.uv - cur.uv)).sum(-1)
        Jr = (nrm[:, None, :] @ cur.J_uv)[:, 0, :]
        got = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
        for b in range(B):
            m = bi == b
            Jb, rb = Jr[m].double(), r[m].double()
            A = Jb.T @ Jb
            # A pseudo-inverse rather than a solve: a joint hidden from this view contributes no
            # rows at all, so `A` is genuinely singular and the right answer for those directions
            # is "no update", which is what zeroing the null space gives.
            got[b] = torch.linalg.pinv(A, rtol=1e-8) @ (Jb.T @ rb)
        return got

    g = torch.Generator(device="cpu").manual_seed(3)
    # Root-only motion.
    d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
    d[:, 0:6] = torch.randn(B, 6, generator=g).to(DEV).double() * 0.01
    got = solve(d)
    joint_leak = got[:, 6:].norm(dim=-1) / got.norm(dim=-1).clamp_min(1e-12)
    assert float(joint_leak.median()) < 0.25, (
        f"a root-only motion was attributed {100 * float(joint_leak.median()):.0f}% to the joints")

    # Single-finger motion: the recovered solution must peak on the finger that moved.
    hits = 0
    for fg in range(5):
        d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
        fam = [3 * fg + k for k in range(3)]
        for k in fam:
            d[:, 6 + 3 * k : 9 + 3 * k] = 0.08
        got = solve(d)
        per = got[:, 6:].reshape(B, 15, 3).norm(dim=-1)
        hits += int(sum(1 for b in range(B) if int(per[b].argmax()) in fam))
    assert hits / (5 * B) >= 0.8, f"finger recovered in only {hits}/{5 * B} trials"


@pytest.mark.parametrize("finger", list(range(5)))
def test_single_finger_motion_is_localised_to_that_finger(mano, sample, field, finger):
    r"""Moving one finger must put the information in that finger's own columns.

    Ranked after root elimination, so this is a statement about articulation and not about the
    incidental hand motion a finger rotation also produces.
    """
    p51, betas, K, j0 = sample["p51"], sample["betas"], sample["K"], sample["j0"]
    B = p51.shape[0]
    g = torch.Generator(device="cpu").manual_seed(11 + finger)
    hits, total = 0, 0
    family = [3 * finger + d for d in range(3)]
    for _ in range(5):
        d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
        for k in family:
            d[:, 6 + 3 * k : 9 + 3 * k] = torch.randn(B, 3, generator=g).to(DEV).double() * 0.10
        res = _simulate_events(mano, field, p51, betas, K, d, j0)
        if res is None:
            continue
        Lam, bi = res
        ig = torch.stack([JA.group_information(Lam, eliminate_root=True)[f"j{k}"]
                          for k in range(15)], -1)                       # (B, 15)
        present = torch.bincount(bi, minlength=B) > 50
        for b in range(B):
            if not bool(present[b]):
                continue
            total += 1
            if int(ig[b].argmax()) in family:
                hits += 1
    assert total >= 4, f"only {total} usable trials"
    assert hits / total >= 0.8, f"top-ranked group was in the moving finger only {hits}/{total}"


def test_fisher_is_symmetric_positive_semidefinite(mano, sample, field):
    p51, betas, K, j0 = sample["p51"], sample["betas"], sample["K"], sample["j0"]
    B = p51.shape[0]
    d = torch.zeros(B, JA.DOF, device=DEV, dtype=torch.float64)
    d[:, 3:6] = 0.05
    d[:, 6:51] = 0.05
    res = _simulate_events(mano, field, p51, betas, K, d, j0)
    assert res is not None
    Lam, _ = res
    assert float((Lam - Lam.transpose(-1, -2)).abs().max()) < 1e-9
    ev = torch.linalg.eigvalsh(Lam)
    assert float(ev.min()) > -1e-6 * float(ev.max().clamp_min(1.0))
    # And the Schur complement of a PSD matrix must itself be PSD.
    for name, sl in JA.GROUPS[1:]:
        S = JA.schur_eliminate(Lam, sl, JA.GROUPS[0][1])
        e = torch.linalg.eigvalsh(0.5 * (S + S.transpose(-1, -2)))
        assert float(e.min()) > -1e-6 * float(e.abs().max().clamp_min(1.0)), name


def test_query_point_matches_kssf_surface_point(mano, sample):
    """`query_jacobian`'s point must be the same surface point KSSF reports."""
    fk, K = sample["fk"], sample["K"]
    B = fk.A.shape[0]
    verts = JA.skin(fk,
                    mano.weights[torch.arange(778, device=DEV).repeat(B)],
                    torch.arange(B, device=DEV)[:, None].expand(B, 778).reshape(-1),
                    torch.arange(778, device=DEV).repeat(B)).reshape(B, 778, 3)
    field = KS.KSSF(mano, H, W, SCALE).to(DEV).double()
    with torch.no_grad():
        f = field.rasterize(verts, K)
    bi, py, px = torch.nonzero(f.face_id >= 0, as_tuple=True)
    keep = torch.randperm(len(bi), device=DEV)[:1000]
    bi, py, px = bi[keep], py[keep], px[keep]
    fid, bary = f.face_id[bi, py, px], f.bary[bi, py, px]
    out = JA.query_jacobian(mano, fk, K, SCALE, bi, fid, bary)
    ref = torch.stack([KS.surface_point(f, mano, verts[int(b)], fid[i : i + 1], bary[i : i + 1])[0]
                       for i, b in enumerate(bi[:64].tolist())])
    assert float((out.X[:64] - ref).abs().max()) < 1e-10
