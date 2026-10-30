"""KA step 1 (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028): atlas table, flexion sign, lifting invariance under rigid
camera changes, AUC helper and the synthetic family AUC on toy features. CPU, no recorded data."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.kin_atlas import common as C  # noqa: E402
from tools.kin_atlas import synth as S  # noqa: E402

K_CAM = np.array([[603.45, 0, 325.09], [0, 602.96, 242.10], [0, 0, 1.0]])
B0 = np.zeros(10, np.float32)


def _base():
    st = np.zeros(51)
    st[3:6] = [2.2, 0.1, -0.2]
    return st


def test_atlas_table_ranges_and_tips():
    t = C.atlas_table()
    assert t.shape == (778,) and t.min() >= 0 and t.max() < 16 * 32
    part = t // 32
    w = C.mano_npz()["weights"]
    assert np.array_equal(part, np.argmax(w, axis=1))
    for f, js in C.FINGER_JOINTS.items():
        assert part[C.TIP_VERTEX[f]] in js


def test_flex_moves_tip_towards_palm_normal():
    st = np.zeros(51)
    st[2] = 0.5
    for f in C.FINGERS:
        tip = C.OP_TIPS[C.FINGERS.index(f)]
        _, j0 = C.fk(st[None], B0)
        _, j1 = C.fk(S.flex(st, f, np.radians(10))[None], B0)
        _, jm = C.fk(S.flex(st, f, np.radians(-10))[None], B0)
        assert np.linalg.norm(j1[0, tip] - j0[0, tip]) > 0.003
        assert not np.allclose(j1[0, tip], jm[0, tip])


def test_amplitude_hits_target():
    s0 = S.place(_base(), B0, S.SETTINGS[0], (0, 0))
    th = S.amplitude(s0, B0, "index", +1)
    _, j0 = C.fk(s0[None], B0)
    _, j1 = C.fk(S.flex(s0, "index", th)[None], B0)
    assert abs(np.linalg.norm(j1[0, 8] - j0[0, 8]) - S.TIP_MOTION_M) < 2e-4


def test_lifting_is_invariant_to_rigid_camera_change():
    """Events placed exactly at the projections of vertices visible in two camera settings land in the same atlas
    cells (>= 90 %; neighbouring vertices closer than a pixel may swap) -- the operator's defining property."""
    K = C.event_K(K_CAM)
    tab = C.atlas_table()
    sa = S.place(_base(), B0, S.SETTINGS[0], (0, 0))
    sb = S.place(_base(), B0, S.SETTINGS[9], (5, -3))
    va, _ = C.fk(sa[None], B0)
    vb, _ = C.fk(sb[None], B0)
    both = np.flatnonzero(C.visible(va[0], K) & C.visible(vb[0], K))
    assert len(both) > 100
    pick = both[::3]
    agree = []
    for v in (va[0], vb[0]):
        uv = C.project(v[pick], K)
        counts, off = C.lift_counts(uv[:, 0], uv[:, 1], np.ones(len(pick), int), v, K, tab)
        assert off == 0.0
        agree.append(counts[:, 1])
    hist_true = np.bincount(tab[pick], minlength=512)
    for a in agree:
        assert np.abs(a - hist_true).sum() / (2 * len(pick)) <= 0.10


def test_representations_empty_and_shapes():
    f = C.representations(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), S.place(_base(), B0, S.SETTINGS[0], (0, 0)), B0, K_CAM)
    for rep in C.REPS:
        assert np.all(f[rep] == 0)
    assert f["atlas_kin"].shape == (1024,) and f["box24"].shape == (1152,)


def test_auc_from():
    assert C.auc_from([1.0], [0.0, 0.5]) == 1.0
    assert C.auc_from([0.0], [0.5, 1.0]) == 0.0
    assert C.auc_from([0.5], [0.5]) == 0.5


def test_family_auc_toy():
    rng = np.random.default_rng(0)
    rows = [(c, m, si, r) for c in range(3) for m in range(len(S.MOTIONS)) for si in range(len(S.SETTINGS))
            for r in ((1, 2) if si == 0 else (1,))]
    A = {k: np.array([row[i] for row in rows]) for i, k in enumerate(("context", "motion", "setting", "real"))}
    proto = rng.normal(size=(3, len(S.MOTIONS), 32))
    F = np.stack([proto[c, m] + 0.01 * rng.normal(size=32) for c, m, _, _ in rows])
    F /= np.linalg.norm(F, axis=1, keepdims=True)
    good = S.family_auc(A, F, "oop")
    assert min(good.values()) > 0.99
    R = rng.normal(size=F.shape)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    rnd = np.mean(list(S.family_auc(A, R, "oop").values()))
    assert 0.3 < rnd < 0.7
    sh = np.mean(list(S.family_auc(A, F, "oop", shuffle_rng=np.random.default_rng(1)).values()))
    assert sh < 0.7
