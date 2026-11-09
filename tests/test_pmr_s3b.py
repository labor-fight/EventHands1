"""PMR stage S3b ladder, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s3b-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk2.py: every rung starts with a zero increment; its shared parameters start equal to R's at one seed;
  rconst's gates are the constant 0.5 and read no geometry; rnohub's root / translation depend on the hand node only even
  with the messages on, and its fingers still read other parts; rnomesh forms no residual (the residual moments are
  zero) and keeps R's parameter set.
- tools/pmr/arms2.py: a rung's config differs from D1's R config (same seed and phase) only in the PMR block,
  MODEL.ENCODER_LAYERS for the rounds rungs, and names / directories / identity keys.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk2 as P2  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _node_inputs, _redraw_zero_layers  # noqa: E402


def _rung(ladder, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type(f"L_{ladder}", (P2.PMRLadderNet,), {"variant": "r", "ladder": ladder})(mano.weights, mano.f)


@pytest.mark.parametrize("ladder", P2.LADDER)
def test_rung_starts_at_zero_with_rs_shared_parameters(ladder):
    _, _, args = _inputs()
    net = _rung(ladder).train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    r, mine = _net("r").state_dict(), net.state_dict()
    common = set(r) & set(mine)
    assert {"phi.0.weight", "enc_part.weight", "rounds.0.slf.weight", "head_j1.weight"} <= common
    assert all(torch.equal(r[k], mine[k]) for k in common)
    extra = set(mine) - set(r)
    assert extra == ({"gate_const"} if ladder == "rconst" else set())


def test_rconst_gates_are_constant_and_geometry_free():
    net = _rung("rconst")
    J1 = torch.randn(2, 16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.5])
    J2 = torch.randn(2, 16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.6])
    g1, g2 = net.gates(J1), net.gates(J2)
    assert torch.equal(g1, g2) and torch.allclose(g1, torch.full_like(g1, 0.5))
    assert not hasattr(net, "edge_emb") and not hasattr(net, "gate")


def test_rnohub_root_reads_only_the_hand_node():
    net = _rung("rnohub").train()
    _redraw_zero_layers(net)
    assert all(len(rd.msg) == 2 for rd in net.rounds)
    G = _jac(net, *_node_inputs(net), messages=True)
    assert torch.count_nonzero(G[0:6, :16]) == 0 and G[0:6, 16].sum() > 0
    assert torch.count_nonzero(G[6:, 16]) == 0                      # no part reads the hand node
    mcp_index = 6 + 3 * (1 - 1)                                       # joint 1 (index MCP) reads its tree neighbours
    assert G[mcp_index:mcp_index + 3, 0].sum() > 0 and G[mcp_index:mcp_index + 3, 2].sum() > 0


def test_rnomesh_has_no_residual_and_finite_outputs():
    _, _, args = _inputs()
    net = _rung("rnomesh").eval()
    with torch.no_grad():
        n = net.node_inputs(*args)
        delta, _, _ = net(*args)
    assert torch.count_nonzero(n["res"]) == 0
    sp = n["sp"]
    assert torch.count_nonzero(sp[..., 1:4]) == 0 and torch.count_nonzero(sp[..., 7:16]) == 0   # mean res, lever x res
    assert torch.count_nonzero(sp[..., 4:7]) > 0                                                  # the event cloud
    assert torch.isfinite(delta).all()
    assert {k for k, _ in net.named_parameters()} == {k for k, _ in _net("r").named_parameters()}


@pytest.mark.parametrize("arm", ("d1l1", "d1l2", "d1g0", "d1h0", "d1a0"))
@pytest.mark.parametrize("phase", ("preflight", "formal"))
def test_rung_configs_differ_from_r_only_in_registered_keys(arm, phase):
    from tools.pmr import arms as A
    from tools.pmr import arms2 as A2

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
        return out
    a, b = flat(A2.build_config(arm, 3407, phase)), flat(A.build_config("d1r", 3407, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    allowed = {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.SECTION", "PMR.DESIGN",
               "PMR.VARIANT", "PMR.MODULE", "PMR.BASE_ARM", "MODEL.ENCODER_LAYERS"}
    assert diff <= allowed and {"PMR.DESIGN", "TRAIN.RUN_NAME"} <= diff
    assert ("MODEL.ENCODER_LAYERS" in diff) == (arm in ("d1l1", "d1l2"))
    assert a["PFO.ROUTER"] == "r_s2d" and a["LOSS.TYPE"] == "so3_trans_fk" and a["TRAIN.MAX_STEPS"] == b["TRAIN.MAX_STEPS"]


@pytest.mark.parametrize("ladder", P2.LADDER)
def test_every_parameter_takes_part_in_the_backward(ladder):
    """DDP is built with find_unused_parameters=False: a parameter outside the graph would stop the training."""
    _, _, args = _inputs()
    net = _rung(ladder).train()
    delta, _, _ = net(*args)
    delta.square().sum().backward()
    missing = [k for k, p in net.named_parameters() if p.grad is None]
    assert not missing, missing
