#!/usr/bin/env python3
"""S37 variants (docs/S37_EXPERIMENT_RECORDS.md [EDGE6]): deeper EdgeConv and per-joint root fusion.

What must hold:

  * each config differs from `s37_routed` in its arm keys and the run naming, nowhere else;
  * `ROOT_FUSION: per_joint` replaces the concat root head by sixteen `[f; e_j] -> 64 -> 6` MLPs whose
    outputs are summed: the batched module equals the sum of its per-joint heads, each head reads
    only its own joint, every head reaches the root, and each slice starts like an `nn.Linear`;
  * the finger decoders keep reading only their own joint's evidence;
  * an event-free packet still returns `prev` bitwise;
  * with the key absent the S37 parameter set is untouched (its checkpoint still loads strictly).
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tests"))

from model import MNISTModel  # noqa: E402
from semkine.routed_readout import PerJointRootFusion  # noqa: E402
from test_s37_routed_readout import HIDDEN, _cfg, _events, _packet, _test_prev  # noqa: E402

EV = 2 * HIDDEN + 1


def _diff(arm, *keys):
    return subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_routed_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / f"{arm}_s3407.yaml"),
         "--allow", *keys, "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)


def test_rootfuse_differs_from_s37_only_in_root_fusion():
    r = _diff("s37_rootfuse", "MODEL.ROOT_FUSION")
    assert r.returncode == 0, r.stdout + r.stderr


def test_edge6_differs_from_s37_only_in_depth_and_micro_batch():
    # 256 x 2 cards x 2 accumulated = the S37 effective batch 1024; validation every 1000 optimiser steps
    r = _diff("s37_edge6", "MODEL.ENCODER_LAYERS", "TRAIN.BATCH_SIZE_PER_GPU",
              "TRAIN.ACCUMULATE_GRAD_BATCHES", "TRAIN.VAL_CHECK_INTERVAL")
    assert r.returncode == 0, r.stdout + r.stderr
    from config import load_config
    s37 = load_config(str(ROOT / "configs/semkine/s37_routed_s3407.yaml"))["TRAIN"]
    e6 = load_config(str(ROOT / "configs/semkine/s37_edge6_s3407.yaml"))["TRAIN"]
    eff = lambda t: t["BATCH_SIZE_PER_GPU"] * t["DEVICES"] * t.get("ACCUMULATE_GRAD_BATCHES", 1)  # noqa: E731
    assert eff(e6) == eff(s37) == 1024
    assert e6["VAL_CHECK_INTERVAL"] / e6["ACCUMULATE_GRAD_BATCHES"] == s37["VAL_CHECK_INTERVAL"]


def _fuse_cfg():
    cfg = copy.deepcopy(_cfg())
    cfg["MODEL"]["ROOT_FUSION"] = "per_joint"
    return cfg


@pytest.fixture(scope="module")
def fused():
    torch.manual_seed(0)
    return MNISTModel(_fuse_cfg()).eval()


def test_per_joint_replaces_the_concat_root_head(fused):
    assert not hasattr(fused, "root_head")
    rf = fused.root_fusion_head
    feat, hid = fused.cfg["MODEL"]["ENCODER_FEAT"], fused.cfg["MODEL"]["ACTIVE_HIDDEN"]
    assert (rf.n_joints, rf.in_dim, rf.hidden, rf.out_dim) == (16, feat + EV, hid, 6)


def test_each_slice_starts_like_an_nn_linear():
    rf = PerJointRootFusion(512, 257, 64)
    for w, b, fan in ((rf.w1, rf.b1, 769), (rf.w2, rf.b2, 64)):
        bound = 1.0 / math.sqrt(fan)
        assert float(w.abs().max()) <= bound and float(b.abs().max()) <= bound
        assert float(w.abs().max()) > 0.9 * bound          # uniform over the full nn.Linear range
    assert not torch.equal(rf.w1[0], rf.w1[1]), "every joint has its own weights"


def test_root_is_the_sum_of_sixteen_per_joint_mlps(fused):
    torch.manual_seed(1)
    feat = torch.randn(3, 64)
    ev = torch.randn(3, 16, EV)
    prev = _test_prev().expand(3, -1).clone()
    rf = fused.root_fusion_head
    with torch.no_grad():
        out = fused._decode_active(feat, prev, ev)
        want = sum(rf.head(j, feat, ev[:, j]) for j in range(16))
    assert torch.allclose(out[:, :6], want, atol=1e-5)


def test_each_root_head_reads_only_its_own_joint_and_all_reach_the_root(fused):
    torch.manual_seed(2)
    rf = fused.root_fusion_head
    feat = torch.randn(1, 64, requires_grad=True)
    ev = torch.randn(1, 16, EV, requires_grad=True)
    for j in range(16):
        g_ev, = torch.autograd.grad(rf.head(j, feat, ev[:, j]).sum(), (ev,))
        others = torch.ones(16, dtype=torch.bool)
        others[j] = False
        assert torch.all(g_ev[0, others] == 0) and g_ev[0, j].abs().sum() > 0
    out = fused._decode_active(feat, _test_prev(), ev)
    fused.zero_grad()
    out[0, :6].sum().backward()
    assert feat.grad.abs().sum() > 0
    assert torch.all(ev.grad[0].abs().sum(-1) > 0), "the root must read all sixteen evidence rows"
    for p in (rf.w1, rf.b1, rf.w2, rf.b2):
        assert torch.all(p.grad.flatten(1).abs().sum(1) > 0), "every joint's head must get a gradient"


def test_finger_heads_still_read_only_their_own_evidence(fused):
    torch.manual_seed(3)
    feat = torch.randn(1, 64, requires_grad=True)
    ev = torch.randn(1, 16, EV, requires_grad=True)
    out = fused._decode_active(feat, _test_prev(), ev)
    for k in range(15):
        g_feat, g_ev = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (feat, ev), retain_graph=True)
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_feat == 0) and torch.all(g_ev[0, others] == 0)


def test_per_joint_empty_packet_returns_prev_bitwise(fused):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    assert torch.equal(fused.forward_packet(pk), pk.prev_state)


def test_per_joint_needs_the_routed_readout():
    cfg = _fuse_cfg()
    cfg["MODEL"]["ROUTED_READOUT"] = False
    with pytest.raises(ValueError, match="ROUTED_READOUT"):
        MNISTModel(cfg)


def test_six_rounds_build_six_edgeconv_layers():
    cfg = copy.deepcopy(_cfg())
    cfg["MODEL"]["ENCODER_LAYERS"] = 6
    m = MNISTModel(cfg)
    assert len(m.event_encoder.layers) == 6


def test_default_keeps_the_s37_parameter_set():
    from config import load_config
    cfg = load_config(str(ROOT / "configs/semkine/s37_routed_s3407.yaml"))
    m = MNISTModel(cfg)
    assert m.root_fusion == "concat" and hasattr(m, "root_head")
    sel = ROOT / "outputs/semkine/s37_routed_s3407/selection_val_core_step50.json"
    if not sel.exists():
        pytest.skip("S37 zgz run not present on this machine")
    ckpt = ROOT / json.loads(sel.read_text())["selected"]["ckpt"]
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    assert set(sd) == set(m.state_dict()), "ROOT_FUSION default changed the S37 key set"
