"""Research contract: exact causal prefix cache, not old-checkpoint accuracy."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semkine.event_gnn import EventGNN
from semkine.streaming import CausalEventEncoder


def fixture(n=91):
    g = torch.Generator().manual_seed(91)
    xy = torch.randint(0, 12, (n, 2), generator=g).float()
    p = torch.randint(0, 2, (n, 1), generator=g).float()
    ts = torch.cumsum(torch.randint(0, 130, (n,), generator=g), 0).long()
    return torch.cat((xy, p), -1), ts


def make():
    torch.manual_seed(47)
    enc = EventGNN(height=180, width=240, hidden=8, feat_dim=16, k=3,
                   n_layers=3, max_nodes=24, window=5)
    return CausalEventEncoder(enc, max_events_per_append=512), enc


def check_equal(a, b):
    assert len(a) == len(b) == 5
    for x, y in zip(a, b):
        if x.dtype == torch.bool:
            assert torch.equal(x, y)
        else:
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('sizes', [[91], [1] * 91, [7, 11, 2, 39, 32]])
def test_complete_prefix_matches_chunking_even_after_readout_eviction(sizes):
    run, _ = make()
    xy, t = fixture()
    state, k = None, 0
    for size in sizes:
        state = run.append(xy[k:k+size], t[k:k+size], state=state, stream_id=4)
        k += size
        check_equal(run.readout(state, int(t[k-1])+1),
                    run.full_reference(xy[:k], t[:k], int(t[k-1])+1, stream_id=4))


def test_time_origin_shift_long_epoch_ties_and_missing_polarity():
    run, _ = make()
    xy = torch.tensor([[5., 6., 0.], [5., 6., 0.], [5., 6., 1.],
                       [5., 6., 1.], [5., 6., 0.]])
    t = torch.tensor([0, 0, 0, 1, 8], dtype=torch.int64)
    a = run.append(xy[:2], t[:2], stream_id=9)
    a = run.append(xy[2:], t[2:], state=a, stream_id=9)
    shift = 9_000_000_000_000
    b = run.append(xy, t+shift, stream_id=9)
    check_equal(run.readout(a, 9), run.readout(b, shift+9))
    check_equal(run.readout(a, 9), run.full_reference(xy, t, 9, stream_id=9))


def test_future_changes_do_not_mutate_already_emitted_outputs_or_state():
    run, _ = make()
    xy, t = fixture()
    state = run.append(xy[:20], t[:20])
    before = tuple(z.clone() for z in run.readout(state, int(t[19])+1))
    run.append(xy[20:], t[20:], state=state)
    check_equal(before, run.readout(state, int(t[19])+1))


def test_empty_gap_and_no_ancient_edge():
    run, _ = make()
    xy, t = fixture(2)
    state = run.append(xy, t)
    assert run.append(xy[:0], t[:0], state=state) is state
    empty = run.readout(state, int(t[-1])+100_000)
    assert not bool(empty[-1].any())
    next_t = t[:1]+1_000_000
    after = run.append(xy[:1], next_t, state=state)
    fresh = run.append(xy[:1], next_t)
    # Tokens can carry a different inter-event gap from an entirely fresh stream;
    # the full-prefix reference, not reset output, is the semantic authority.
    check_equal(run.readout(after, int(next_t[0])+1),
                run.full_reference(torch.cat([xy, xy[:1]]), torch.cat([t, next_t]),
                                   int(next_t[0])+1))
    assert fresh is not None


@pytest.mark.parametrize('bad', ['nan', 'pixel', 'polarity', 'order', 'dtype', 'stream', 'budget'])
def test_invalid_inputs_fail_explicitly(bad):
    run, _ = make()
    xy, t = fixture(5)
    state = None
    sid = 0
    if bad == 'nan': xy[0, 0] = float('nan')
    elif bad == 'pixel': xy[0, 0] = -1
    elif bad == 'polarity': xy[0, 2] = 0.5
    elif bad == 'order': t[2] = t[1]-1
    elif bad == 'dtype': t = t.float()
    elif bad == 'stream':
        state = run.append(xy[:1], t[:1], stream_id=2)
        xy, t = xy[1:], t[1:]
        sid = 3
    elif bad == 'budget': xy, t = fixture(513)
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        run.append(xy, t, state=state, stream_id=sid)


def test_backward_optimizer_invalidates_cache_without_silent_detach():
    run, model = make()
    xy, t = fixture()
    state = run.append(xy[:40], t[:40])
    state = run.append(xy[40:], t[40:], state=state)
    out = run.readout(state, int(t[-1])+1)[0]
    loss = (out-0.37).square().mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    old = model.embed.weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(old, model.embed.weight)
    with pytest.raises((ValueError, RuntimeError)):
        run.readout(state, int(t[-1])+1)


def test_same_observations_gradient_parity():
    run, model = make()
    xy, t = fixture(37)
    state = run.append(xy[:11], t[:11])
    state = run.append(xy[11:], t[11:], state=state)
    run.readout(state, int(t[-1])+1)[0].square().sum().backward()
    inc = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    run.full_reference(xy, t, int(t[-1])+1)[0].square().sum().backward()
    for g, p in zip(inc, model.parameters()):
        torch.testing.assert_close(g, p.grad, atol=1e-5, rtol=1e-5)


def test_input_buffer_reuse_does_not_corrupt_either_cache():
    run, _ = make()
    xy, t = fixture(71)
    original_xy, original_t = xy.clone(), t.clone()
    state = run.append(xy, t)
    before = tuple(z.clone() for z in run.readout(state, int(t[-1])+1))
    xy.fill_(0)
    t.fill_(0)
    check_equal(before, run.readout(state, int(original_t[-1])+1))
    state = run.append(original_xy[-3:], original_t[-3:]+10000, state=state)
    check_equal(run.readout(state, int(original_t[-1])+10001),
        run.full_reference(torch.cat([original_xy, original_xy[-3:]]),
                           torch.cat([original_t, original_t[-3:]+10000]),
                           int(original_t[-1])+10001))


def test_every_cached_layer_matches_independently_recomputed_prefix():
    run, model = make()
    xy, t = fixture(117)
    state = run.append(xy[:41], t[:41])
    state = run.append(xy[41:], t[41:], state=state)
    reference = []
    hooks = [model.embed.register_forward_hook(
        lambda _m, _i, out: reference.append(torch.relu(out)))]
    for layer in model.layers:
        hooks.append(layer.register_forward_hook(
            lambda _m, inputs, out: reference.append((inputs[0]+out)[0])))
    try:
        run.full_reference(xy, t, int(t[-1])+1)
    finally:
        for hook in hooks: hook.remove()
    assert len(reference) == len(state.recent_layers)
    for cached, full in zip(state.recent_layers, reference):
        torch.testing.assert_close(cached, full[-model.window:], atol=1e-5, rtol=1e-5)


def test_tokens_have_hand_computed_zero_age_and_first_seen_semantics():
    run, _ = make()
    xy = torch.tensor([[5.,6.,0.], [5.,6.,1.], [5.,6.,0.], [7.,6.,0.]])
    t = torch.tensor([100, 100, 110, 120])
    token, _, _ = run._tokens(xy, t, t[:0], t[:0], None)
    torch.testing.assert_close(token[:, 3], torch.tensor([0.,0.,10/50000,10/50000]))
    torch.testing.assert_close(token[:, 5], torch.tensor([1.,1.,10/50000,1.]))
    torch.testing.assert_close(token[:, 6], torch.tensor([1.,0.,10/50000,1.]))
    torch.testing.assert_close(token[:, 4], torch.log1p(torch.tensor([0.,0.,10.,10.])))


def test_long_gap_has_zero_messages_not_only_shared_reference_parity():
    run, model = make()
    xy, t = fixture(6)
    old = run.append(xy[:5], t[:5])
    tnew = t[-1:] + 100000
    token, _, _ = run._tokens(xy[-1:], tnew, old.sae_keys, old.sae_timestamps,
                              old.last_timestamp)
    expected = torch.relu(model.embed(token))
    new = run.append(xy[-1:], tnew, state=old)
    for cached in new.recent_layers:
        torch.testing.assert_close(cached[-1:], expected, atol=0, rtol=0)


def test_fixed_linear_backend_cpu_reference_layers_and_state_invalidation():
    run, model = make()
    xy, t = fixture(71)
    expected = run.full_reference(xy, t, int(t[-1])+1)
    run.linear_backend = "fixed_fp32"
    state = run.append(xy, t)
    actual, layers = run.full_reference(xy, t, int(t[-1])+1, return_layers=True)
    check_equal(expected, actual)
    assert len(layers) == len(model.layers)+1
    for cached, full in zip(state.recent_layers, layers):
        torch.testing.assert_close(cached, full[-model.window:], atol=1e-5, rtol=1e-5)
    run.linear_backend = "torch"
    with pytest.raises(ValueError, match="configuration changed"):
        run.readout(state, int(t[-1])+1)
    with pytest.raises(ValueError, match="linear_backend"):
        CausalEventEncoder(model, linear_backend="unknown")
