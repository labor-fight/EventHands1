"""PMR CPU tests (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk.py: the node frames equal MANO's own FK rotations; the moments equal a direct computation; every
  variant starts with a zero increment; the shared parameters start equal across the three variants at one seed; the
  node statistics are event-additive (the statistics of a packet are the mass-weighted combination of its halves').
- The S0 bypass audit (structural, at the node inputs): with every message zeroed, the increment of finger joint j has
  zero gradient with respect to the inputs of every part node i != j and the hand node, and the root / translation has
  zero gradient with respect to every part node; with the messages on, every joint depends on every node (the hub joins
  every pair within two rounds); the mlp control depends on every node (the bypass). Checked with the zero-initialised
  last layers re-drawn (at zero every gradient is zero).
- The arms' configs differ from B2's (PFO2 fgs, same seed and phase) only in the registered keys; the installed trunk
  builds inside MNISTModel and its forward_packet runs on a synthetic packet.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from tools.pmr import arms as A  # noqa: E402

BASE_KEYS = {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.TRUNK", "PFO.SECTION", "PFO.STUDY",
             "PMR.DESIGN", "PMR.VARIANT", "PMR.MODULE"}


def _mano():
    from mano_layer import ManoLayer
    return ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=False).eval()


def _inputs(B=3, N=200, seed=0):
    """A posed hand at ~0.5 m and events routed to random faces (PFA form: q = 1 on-hand, 0 off-hand)."""
    g = torch.Generator().manual_seed(seed)
    mano = _mano()
    prev = torch.zeros(B, 51)
    prev[:, 2] = 0.5
    prev[:, 3:6] = torch.randn(B, 3, generator=g) * 0.3
    prev[:, 6:] = torch.randn(B, 45, generator=g) * 0.2
    with torch.no_grad():
        verts, _ = mano(torch.zeros(B, 10), prev[:, 3:6], prev[:, 6:] + mano.hands_mean, prev[:, :3])
    F = mano.f.long()
    face = torch.randint(0, F.shape[0], (B, N), generator=g)
    bary = torch.rand(B, N, 3, generator=g)
    bary = bary / bary.sum(-1, keepdim=True)
    vid3 = F[face]
    pt = (bary.unsqueeze(-1) * verts[torch.arange(B).view(B, 1, 1), vid3]).sum(2)
    z = pt[..., 2]
    d = torch.cat([pt[..., :2] / z.unsqueeze(-1) + 0.004 * torch.randn(B, N, 2, generator=g), torch.ones(B, N, 1)], -1)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[-1, N // 2:] = False
    q = (torch.rand(B, N, generator=g) > 0.15).float()
    assoc = {"vid": torch.cat([vid3, vid3[..., :1]], -1), "A": torch.cat([bary, torch.zeros(B, N, 1)], -1),
             "q": q, "z": z * q, "valid": q > 0, "hit": q > 0}
    tok = torch.rand(B, N, 7, generator=g) * mask.unsqueeze(-1)
    return mano, prev, (tok, d, mask, verts, assoc, prev[:, 3:51])


def _net(variant, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type(f"T_{variant}", (PM.PMRNet,), {"variant": variant})(mano.weights, mano.f)


def _redraw_zero_layers(net):
    torch.manual_seed(5)
    for name in ("head_j2", "head_h2", "fc2"):
        mod = getattr(net, name, None)
        if mod is not None:
            for p in mod.parameters():
                torch.nn.init.normal_(p, std=0.05)


def test_frames_equal_mano_fk_rotations():
    mano, prev, args = _inputs()
    B = prev.shape[0]
    pose = torch.cat([prev[:, 3:6], prev[:, 6:] + mano.hands_mean], 1).view(B, 16, 3)
    rot = PM.rodrigues(pose)
    from mano_layer import batch_rodrigues
    assert torch.allclose(rot, batch_rodrigues(pose.reshape(-1, 3)).view(B, 16, 3, 3), atol=1e-5)
    G = [rot[:, 0]]
    for j in range(1, 16):
        G.append(G[PM.PARENT[j]] @ rot[:, j])
    F = PM.node_frames(prev[:, 3:51].double(), mano.hands_mean.double())
    assert torch.allclose(F[:, 0], torch.eye(3, dtype=torch.float64).expand(B, 3, 3))
    for j in range(1, 16):
        assert torch.allclose(F[:, j].float(), G[PM.PARENT[j]], atol=1e-5)
    # the frames are the rotation parts of MANO's own chain (posed joint offsets rotate with the parent frame)
    v_shaped = mano.v_template.unsqueeze(0).expand(B, -1, -1)
    J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)
    with torch.no_grad():
        _, j21 = mano(torch.zeros(B, 10), prev[:, 3:6], prev[:, 6:] + mano.hands_mean, prev[:, :3])
    posed16 = j21[:, [int(np.where(mano.mano16_tips_to_openpose21.cpu().numpy() == k)[0][0]) for k in range(16)]] \
        if hasattr(mano, "mano16_tips_to_openpose21") else None
    if posed16 is not None:
        for j in (1, 4, 13):
            off = posed16[:, j] - posed16[:, 0]
            rest = J[:, j] - J[:, 0]
            assert torch.allclose(off, (F[:, j].float() @ rest.unsqueeze(-1)).squeeze(-1), atol=1e-4)


def test_moments_equal_direct_computation():
    _, _, args = _inputs()
    net = _net("r").eval()
    with torch.no_grad():
        n = net.node_inputs(*args)
    for b, p in ((0, 5), (1, 0), (2, 14)):
        om, P, res, J, Fr = n["omega"][b, :, p], n["P"][b], n["res"][b], n["J"][b, p], n["F"][b, p]
        M = om.sum()
        lev, rl = (P - J) @ Fr, res @ Fr
        T = (om[:, None, None] * lev[:, :, None] * rl[:, None, :]).sum(0) / M / (PM.L_LEV * PM.L_RES)
        Gm = (om[:, None, None] * lev[:, :, None] * lev[:, None, :]).sum(0) / M / PM.L_LEV ** 2
        iu = torch.triu_indices(3, 3)
        ref = torch.cat([torch.log1p(M).view(1), (om[:, None] * rl).sum(0) / M / PM.L_RES,
                         (om[:, None] * lev).sum(0) / M / PM.L_LEV, T.flatten(), Gm[iu[0], iu[1]], Fr[2]])
        assert torch.allclose(n["sp"][b, p], ref, atol=1e-9)
    assert n["sp"].shape[-1] == PM.PART_STATS and n["sh"].shape[-1] == PM.HUB_STATS


@pytest.mark.parametrize("variant", PM.VARIANTS)
def test_initial_increment_is_zero(variant):
    _, _, args = _inputs()
    net = _net(variant).train()
    delta, stats, _ = net(*args)
    assert delta.shape == (3, 51) and torch.count_nonzero(delta) == 0
    assert set(stats) >= {"xyz_valid", "xyz_hit", "xyz_resid_mass"}


def test_shared_parameters_start_equal_across_variants():
    sd = {v: _net(v).state_dict() for v in PM.VARIANTS}
    common = set(sd["r"]) & set(sd["noprop"]) & set(sd["mlp"])
    assert {"phi.0.weight", "enc_part.weight", "enc_hub.weight"} <= common
    for k in common:
        assert torch.equal(sd["r"][k], sd["noprop"][k]) and torch.equal(sd["r"][k], sd["mlp"][k]), k
    assert not any(k.startswith(("rounds.0.msg", "gate", "edge_emb", "fc")) for k in sd["noprop"])
    assert not any(k.startswith(("rounds", "head", "gate", "edge_emb")) for k in sd["mlp"])


def _jac(net, x_part, x_hub, messages):
    """d output / d node inputs: (51, 17) summed absolute gradients (per output, per node)."""
    x_part = x_part.clone().requires_grad_(True)
    x_hub = x_hub.clone().requires_grad_(True)
    h0 = net.encode(x_part, x_hub)
    if net.variant == "mlp":
        out = net.fc2(torch.relu(net.fc1(h0.reshape(h0.shape[0], -1))))
    else:
        J = torch.zeros(x_part.shape[0], 16, 3) + torch.randn(16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.5])
        use = net.variant == "r" and messages
        out = net.readout(net.propagate(h0, net.gates(J) if use else None, messages=use))
    G = torch.zeros(51, 17)
    for o in range(51):
        gp, gh = torch.autograd.grad(out[:, o].sum(), (x_part, x_hub), retain_graph=True, allow_unused=True)
        G[o, :16] = gp.abs().sum((0, 2)) if gp is not None else 0
        G[o, 16] = gh.abs().sum() if gh is not None else 0
    return G


def _node_inputs(net):
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    return n["x_part"].float(), n["x_hub"].float()


@pytest.mark.parametrize("variant,messages", [("r", False), ("noprop", True)])
def test_bypass_audit_messages_zeroed(variant, messages):
    net = _net(variant).train()
    _redraw_zero_layers(net)
    G = _jac(net, *_node_inputs(net), messages=messages)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert rows[:, j].sum() > 0, f"joint {j} must read its own node"
        others = [i for i in range(17) if i != j]
        assert torch.count_nonzero(rows[:, others]) == 0, f"joint {j} reads another node without messages"
    hub_rows = G[0:6]
    assert hub_rows[:, 16].sum() > 0 and torch.count_nonzero(hub_rows[:, :16]) == 0
    assert torch.count_nonzero(G[:, 0]) == 0                      # the palm node feeds the outputs only through P


def test_bypass_audit_messages_on_reach_every_node():
    net = _net("r").train()
    _redraw_zero_layers(net)
    G = _jac(net, *_node_inputs(net), messages=True)
    assert (G > 0).all(), "with >= 2 rounds the hub joins every pair: every output depends on every node"


def test_mlp_control_is_the_bypass():
    net = _net("mlp").train()
    _redraw_zero_layers(net)
    G = _jac(net, *_node_inputs(net), messages=True)
    assert (G > 0).all()


def test_statistics_are_event_additive():
    """Mass-weighted means over a packet = mass-weighted combination of the two halves' means (sums add)."""
    _, _, (tok, d, mask, verts, assoc, prev48) = _inputs(B=1, N=240)
    net = _net("r").eval()
    half = torch.zeros_like(mask)
    half[:, ::2] = True

    def stats(m):
        with torch.no_grad():
            g = net.geometry(mask & m, d, verts, assoc, prev48, False, torch.float64)
            return g, net.statistics(g)
    g_all, (M, sp, Mh, sh) = stats(torch.ones_like(mask))
    _, (M1, sp1, Mh1, sh1) = stats(half)
    _, (M2, sp2, Mh2, sh2) = stats(~half)
    assert torch.allclose(M, M1 + M2)
    # first moments (mean residual, mean lever) and the second moments combine linearly in the mass
    for k in range(1, 22):
        comb = (M1 * sp1[..., k] + M2 * sp2[..., k]) / (M1 + M2).clamp_min(1e-12)
        assert torch.allclose(sp[..., k] * (M > 0), comb * (M > 0), atol=1e-9), k


@pytest.mark.parametrize("arm", tuple(A.arms()))
@pytest.mark.parametrize("phase", A.PHASES)
def test_configs_differ_from_b2_only_in_registered_keys(arm, phase):
    from tools.pfo2 import arms as P2

    def flat(dct, pre=""):
        out = {}
        for k, v in dct.items():
            out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
        return out
    a, b = flat(A.build_config(arm, 3407, phase)), flat(P2.build_config("fgs", 3407, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff == BASE_KEYS
    assert a["PFO.ROUTER"] == "r_s2d" and a["PFO.ROUTER_KIND"] == "pfr" and a["PFO.ASSOC"] == "pfa"
    assert a["PMR.VARIANT"] == A.parse(arm)[1]


def test_installed_trunk_runs_inside_the_model():
    from semkine import xyz_mesh as XM
    from model import MNISTModel
    cfg = A.build_config("d1r", 3407, "preflight")
    cfg["_config_path"] = "test"
    PM.install("r")
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg).eval()
    finally:
        PM.uninstall()
    assert XM.XYZMeshNet is PM._NATIVE["cls"]
    assert isinstance(model.xyz, PM.PMRNet) and model.xyz.variant == "r"
    names = {k for k, _ in model.named_parameters()}
    assert any(k.startswith("xyz.rounds.2.msg") for k in names) and not any(k.startswith("xyz.fc1") for k in names)


def test_smoke_phase_is_preflight_shortened():
    a, b = A.build_config("d1r", 3407, A.SMOKE), A.build_config("d1r", 3407, "preflight")
    assert a["TRAIN"]["MAX_STEPS"] == 20 and a["PFO"]["PHASE"] == A.SMOKE and a["TRAIN"]["RUN_NAME"].endswith("_smoke")
    assert {k: v for k, v in a["MODEL"].items()} == b["MODEL"] and a["LOSS"] == b["LOSS"] and a["DATA"] == b["DATA"]


def test_screen_rules_on_synthetic_numbers():
    from tools.pmr.analyze import rules

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    b2 = row(14.8, 8.6, 11.47, 5.6)
    ok = rules(b2, row(14.5, 8.4, 11.3, 5.6), row(16.0, 9.2, 12.5, 6.2), row(14.6, 8.5, 11.4, 5.5))
    assert ok["verdict"] == "SCREEN_PASS"
    # (a) the gap is there overall but local and global disagree
    assert not rules(b2, row(14.5, 8.4, 11.3, 5.6), row(17.0, 8.3, 12.5, 6.0), row(14.6, 8.5, 11.4, 5.5))["a_necessary"]
    # (b) the concat + MLP control is better by more than 0.3
    assert not rules(b2, row(14.5, 8.4, 11.3, 5.6), row(16.0, 9.2, 12.5, 6.2), row(14.0, 8.0, 10.9, 5.5))["b_sufficient"]
    # (c) R worse than B2 by more than 0.3
    assert not rules(b2, row(15.5, 8.9, 11.9, 5.6), row(17.0, 9.6, 13.0, 6.2), row(15.6, 8.9, 12.0, 5.5))["c_no_regress"]
    # health: a control's single step differs from R's by more than 1.0 (a collapsed control)
    bad = rules(b2, row(14.5, 8.4, 11.3, 5.6), row(16.0, 9.2, 12.5, 6.8), row(14.6, 8.5, 11.4, 5.5))
    assert not bad["health"] and bad["verdict"] == "SCREEN_FAIL"


def test_screen_job_list_is_well_formed():
    from tools.pmr import jobs as Jb
    J = Jb.screen("d1", "s1d1", True)
    keys = [j["key"] for j in J]
    assert keys[0] == "s1:d1:freeze" and keys[-1] == "s1:d1:screen"
    trains = [j for j in J if j["kind"] == "pair"]
    assert len(trains) == 6 and all("tools/pmr/train.py" in " ".join(j["command"]) for j in trains)
    assert sum(j["kind"] == "exclusive" for j in J) == 4
    assert all(d in keys[:i] for i, j in enumerate(J) for d in j["deps"])
