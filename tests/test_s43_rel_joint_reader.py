"""R1 basis invariance and weighted-support contracts; no data reads."""
import pytest
import torch
from torch import nn
from tools.s43_rel.model import TypedSupportAdapter
from tools.s43_rel.joint_reader import JointGramSupportAdapter, joint_gram_features


def factors(dtype=torch.float64):
    generator = torch.Generator().manual_seed(3407)
    return (torch.randn(2, 8, 51, generator=generator, dtype=dtype),
            torch.rand(2, 8, 24, generator=generator, dtype=dtype))


def test_exact_sum_of_slot_grams_reference():
    a, lam = factors()
    C, diagonal = joint_gram_features(a, lam)
    grams = torch.einsum("bnh,bnj,bnk->bhjk", lam, a, a)
    torch.testing.assert_close(C, grams.sum(1), rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(diagonal, grams.diagonal(dim1=-2, dim2=-1).transpose(1, 2), rtol=1e-13, atol=1e-13)
    assert torch.all(diagonal >= 0)
    assert torch.linalg.eigvalsh(C).min() >= -1e-12


def test_contour_point_permutation_sign_and_split_invariant():
    a, lam = factors()
    C, diagonal = joint_gram_features(a, lam)
    order = torch.tensor([4, 7, 0, 1, 6, 5, 3, 2])
    Cp, dp = joint_gram_features(a[:, order], lam[:, order])
    Cs, ds = joint_gram_features(-a, lam)
    Csplit, dsplit = joint_gram_features(a.repeat_interleave(2, 1), lam.repeat_interleave(2, 1) / 2.)
    for expected, actual in ((C, Cp), (diagonal, dp), (C, Cs), (diagonal, ds), (C, Csplit), (diagonal, dsplit)):
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)


def test_unsupported_point_geometry_cannot_enter_features():
    a, lam = factors()
    lam[:, 3] = 0
    C, diagonal = joint_gram_features(a, lam)
    altered = a.clone(); altered[:, 3] = 10000.
    Ca, da = joint_gram_features(altered, lam)
    assert torch.equal(C, Ca) and torch.equal(diagonal, da)


def test_same_parameter_count_and_zero_initialization():
    point, joint = TypedSupportAdapter(), JointGramSupportAdapter()
    assert sum(p.numel() for p in point.parameters()) == sum(p.numel() for p in joint.parameters()) == 45100
    assert point.state_dict().keys() == joint.state_dict().keys()
    a, lam = factors(torch.float32)
    output = joint(torch.ones(2, 51, 24), a, lam, torch.ones(2, 3))
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert joint.output.weight.grad.abs().sum() > 0


def test_zero_evidence_and_zero_joint_action_hard_gate_all_biases():
    reader = JointGramSupportAdapter()
    with torch.no_grad():
        reader.output.weight.fill_(1.); reader.output.bias.fill_(7.)
    f, condition = torch.ones(2, 51, 24), torch.ones(2, 3)
    a, lam = factors(torch.float32)
    assert torch.count_nonzero(reader(f, a, torch.zeros_like(lam), condition)) == 0
    assert torch.count_nonzero(reader(f, torch.zeros_like(a), lam, condition)) == 0
    with pytest.raises(ValueError, match="nonnegative"):
        reader(f, a, -lam, condition)


def test_factory_checkpoint_reader_identity_and_R0_default(monkeypatch, tmp_path):
    from tools.s43_rel import model as module
    class Base(nn.Module):
        sparse_trunk = True
        active_head = False
        transl_head = None
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(2,3,3,padding=1)
            self.rn = nn.Module(); self.rn.fc = nn.Linear(3,51)
    monkeypatch.setattr(module,"load_legacy",lambda *args,**kwargs:(Base(),{}))
    cfg = {"REL":{"ARM":"full","SEED":3407}}
    points,_ = module.load_relation_model(cfg)
    assert points.reader == "points"
    r0path = tmp_path / "r0.ckpt"
    torch.save({"schema":"s43_relation_training_v1","arm":"full", "model_state_dict":points.state_dict()},r0path)
    restored,_ = module.load_relation_model(cfg,r0path)
    assert restored.reader == "points"
    joint_cfg = {"REL":dict(cfg["REL"],SUPPORT_READER="joint_gram")}
    with pytest.raises(ValueError,match="SUPPORT_READER"):
        module.load_relation_model(joint_cfg,r0path)
    joint,_ = module.load_relation_model(joint_cfg)
    jointpath = tmp_path / "joint.ckpt"
    torch.save({"schema":"s43_relation_training_v1","arm":"full","reader":"joint_gram", "model_state_dict":joint.state_dict()},jointpath)
    loaded,_ = module.load_relation_model(joint_cfg,jointpath)
    assert loaded.reader == "joint_gram"
    with pytest.raises(ValueError,match="SUPPORT_READER"):
        module.load_relation_model(cfg,jointpath)
