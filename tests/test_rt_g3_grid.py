"""Root tracking G3: the grid readout is off by default (S37 bitwise) and, when on, a well-formed
per-packet function (gradients reach it, empty packets give zero, packets do not leak into each other)."""
import torch

from semkine.event_gnn import EventGNN
from tests.test_x1001_e9_sampling import _packets


def _delta_t(ptr):
    return torch.full((ptr.numel() - 1,), 0.05)


def test_off_is_s37_bitwise():
    torch.manual_seed(0)
    a = EventGNN(hidden=32, feat_dim=64, max_nodes=128)
    torch.manual_seed(0)
    b = EventGNN(hidden=32, feat_dim=64, max_nodes=128, grid_cell=0)
    assert not hasattr(b, "grid_cnn") and b.proj[0].in_features == 64
    ev, ptr = _packets([10, 300, 0, 64])
    a.eval(), b.eval()
    assert torch.equal(a(ev, ptr, _delta_t(ptr)), b(ev, ptr, _delta_t(ptr)))


def test_on_trains_and_every_parameter_gets_a_gradient():
    torch.manual_seed(0)
    enc = EventGNN(hidden=32, feat_dim=64, max_nodes=128, grid_cell=8)
    assert (enc.grid_h, enc.grid_w) == (23, 30) and enc.proj[0].in_features == 2 * 32 + 2 * 32
    ev, ptr = _packets([500, 300, 0, 64, 7], seed=1)
    out = enc(ev, ptr, _delta_t(ptr))
    assert out.shape == (5, 64) and torch.isfinite(out).all()
    assert torch.equal(out[2], torch.zeros(64))                       # empty packet is exactly zero
    out.pow(2).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert any(float(p.grad.abs().sum()) > 0 for n, p in enc.named_parameters() if n.startswith("grid_cnn"))


def test_on_packets_are_independent_in_eval_and_deterministic():
    torch.manual_seed(0)
    enc = EventGNN(hidden=32, feat_dim=64, max_nodes=128, grid_cell=8).eval()
    ev, ptr = _packets([400, 90, 0, 250], seed=2)
    with torch.no_grad():
        full = enc(ev, ptr, _delta_t(ptr))
        again = enc(ev, ptr, _delta_t(ptr))
        for b in range(4):
            sub = ev[ptr[b]:ptr[b + 1]].clone()
            sub[:, 0] = 0
            one = enc(sub, torch.tensor([0, sub.shape[0]]), torch.full((1,), 0.05))
            assert torch.allclose(one[0], full[b], atol=1e-5), b
    assert torch.equal(full, again)


def test_all_empty_batch():
    enc = EventGNN(hidden=32, feat_dim=64, max_nodes=128, grid_cell=8)
    out = enc(torch.zeros(0, 5), torch.tensor([0, 0, 0]), torch.full((2,), 0.05))
    assert out.shape == (2, 64) and torch.equal(out, torch.zeros(2, 64))


def test_on_is_bitwise_deterministic_on_cuda():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    torch.manual_seed(0)
    enc = EventGNN(hidden=32, feat_dim=64, max_nodes=2048, grid_cell=8).cuda().eval()
    ev, ptr = _packets([20000, 3000, 0, 700] * 8, seed=3)
    ev, ptr = ev.cuda(), ptr.cuda()
    with torch.no_grad():
        outs = [enc(ev, ptr, _delta_t(ptr).cuda()) for _ in range(3)]
    assert all(torch.equal(outs[0], o) for o in outs[1:])
