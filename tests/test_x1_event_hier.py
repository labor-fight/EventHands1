#!/usr/bin/env python3
"""X1 (`event_hier`): contract tests for the sparse spatial-hierarchy absolute encoder.

What must hold (research_state/lit/CANDIDATES.md §2 C1 branch A, §4 X1):

  * the structure is the registered one: S37's tokens and node sample, level sizes 2048 / 512 /
    128 / 32 and widths 64 / 128 / 256 / 512 at the X1 configuration;
  * an event-free packet is exactly zero, few- and many-event packets are finite;
  * packets never leak into each other; reordering simultaneous events changes nothing unless it
    changes which event sits on the centroid stride (node order *is* part of the semantics, as in
    S37: the sampler and the centroid choice are index based);
  * gradients reach every parameter, finitely, and one Adam step moves every parameter;
  * nothing grows with the canvas: k-NN indices are bitwise invariant to an integer translation in a
    10x larger canvas, and forward memory is identical for 240x180 and 2400x1800;
  * no state can enter: no extra node channels, no node hand-off, and the X1 model's output does
    not depend on `prev_state`;
  * on a real training packet the X1 model gives a finite (B, 51) and a finite, back-propagating
    SO(3)+FK loss; the S37 checkpoint still loads strictly; the X1 config differs from S37 only
    where its header says.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402
from semkine import event_hier as EH  # noqa: E402
from semkine.encoder import TOKEN_DIM, event_tokens  # noqa: E402
from semkine.event_gnn import EventGNN  # noqa: E402
from semkine.event_hier import EventHierEncoder  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402
from semkine.frontends import build_frontend  # noqa: E402

X1_CFG = ROOT / "configs/semkine/x1_hier_abs_s3407.yaml"
S37_CFG = ROOT / "configs/semkine/s37_routed_s3407.yaml"
C0_CFG = ROOT / "configs/semkine/c0_s37_so3fk_s3407.yaml"
S37_CKPT = ROOT / "outputs/semkine/s37_routed_s3407/s37_routed_s3407-step=2500.ckpt"
DATA = ROOT / "data/hand_data51"


def _batch(counts, seed=0, integer=True, x0=0.0, y0=0.0, width=240, height=180, dt=0.05):
    """A ragged batch of random time-sorted packets inside a `width x height` box at `(x0, y0)`."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for b, n in enumerate(counts):
        e = torch.zeros(n, 5)
        e[:, EV_BATCH] = b
        x = torch.rand(n, generator=g) * (width - 1)
        y = torch.rand(n, generator=g) * (height - 1)
        if integer:
            x, y = x.floor(), y.floor()
        e[:, EV_X], e[:, EV_Y] = x + x0, y + y0
        e[:, EV_T] = torch.sort(torch.rand(n, generator=g) * dt).values
        e[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        parts.append(e)
    ev = torch.cat(parts) if parts else torch.zeros(0, 5)
    c = torch.tensor(counts, dtype=torch.long)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), c.cumsum(0)])
    return ev, ptr, torch.full((len(counts),), dt)


def _small(**kw):
    torch.manual_seed(0)
    args = dict(hidden=16, feat_dim=32, k=8, n_layers=2, max_nodes=256, t_px=10.0)
    args.update(kw)
    return EventHierEncoder(**args).eval()


# ------------------------------------------------------------------ structure
def test_level_sizes_and_widths_are_the_registered_ones():
    torch.manual_seed(0)
    enc = EventHierEncoder(hidden=64, feat_dim=512, k=16, n_layers=2, max_nodes=2048).eval()
    assert enc.level_sizes() == (2048, 512, 128, 32)
    assert enc.widths == (64, 128, 256, 512)
    assert len(enc.level0) == 2 and len(enc.level_convs) == 2 and len(enc.sa) == 3
    assert enc.proj[0].in_features == 1024 and enc.proj[2].out_features == 512
    seen = []
    hooks = [m.register_forward_hook(lambda _m, _i, o: seen.append(tuple(o.shape)))
             for m in enc.sa]
    ev, ptr, dt = _batch([3000, 40])
    with torch.no_grad():
        out = enc(ev, ptr, dt)
    for hk in hooks:
        hk.remove()
    assert seen == [(2, 512, 128), (2, 128, 256), (2, 32, 512)]
    assert out.shape == (2, 512)


def test_nodes_are_s37s_tokens_and_sample():
    """Same uniform-stride node set as `EventGNN` and the same seven-scalar tokens."""
    enc = _small(max_nodes=64)
    gnn = EventGNN(hidden=8, feat_dim=16, max_nodes=64)
    ev, ptr, dt = _batch([0, 5, 64, 65, 500, 0, 1])
    s_h, m_h = enc._sample(ev, ptr)
    s_g, m_g = gnn._sample(ev, ptr)
    assert torch.equal(s_h, s_g) and torch.equal(m_h, m_g)
    assert m_h.sum(1).tolist() == [0, 5, 64, 64, 64, 0, 1]
    tok = event_tokens(ev, ptr, dt, 180, 240)
    assert tok.shape[1] == TOKEN_DIM == enc.embed.in_features


# ------------------------------------------------------------------ outputs
def test_empty_batch_and_event_free_packets_are_exactly_zero():
    enc = _small()
    with torch.no_grad():
        z = enc(torch.zeros(0, 5), torch.tensor([0, 0, 0]), torch.tensor([0.05, 0.05]))
        assert z.shape == (2, 32) and torch.equal(z, torch.zeros_like(z))
        ev, ptr, dt = _batch([40, 0, 7])
        out = enc(ev, ptr, dt)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert out[0].abs().sum() > 0 and out[2].abs().sum() > 0


@pytest.mark.parametrize("counts", [[1], [5], [50], [3000], [5, 50, 3000, 1]])
def test_few_and_many_events_give_finite_features(counts):
    enc = _small(max_nodes=2048, k=16)
    ev, ptr, dt = _batch(counts, seed=len(counts))
    with torch.no_grad():
        out = enc(ev, ptr, dt)
    assert out.shape == (len(counts), 32)
    assert torch.isfinite(out).all()
    assert (out.abs().sum(1) > 0).all()


def test_packets_do_not_leak_and_batch_order_only_permutes():
    enc = _small()
    ev_a, ptr_a, dt_a = _batch([300], seed=1)
    ev_b, ptr_b, dt_b = _batch([120], seed=2)
    ev_b[:, EV_BATCH] = 1
    ev = torch.cat([ev_a, ev_b])
    ptr = torch.tensor([0, 300, 420])
    dt = torch.tensor([0.05, 0.05])
    with torch.no_grad():
        both = enc(ev, ptr, dt)
        alone = enc(ev_a, ptr_a, dt_a)
        ev_ba = torch.cat([ev_b.clone(), ev_a.clone()])
        ev_ba[:120, EV_BATCH], ev_ba[120:, EV_BATCH] = 0, 1
        swapped = enc(ev_ba, torch.tensor([0, 120, 420]), dt)
    assert torch.allclose(both[0], alone[0], atol=1e-6, rtol=1e-5)
    assert torch.allclose(both, swapped.flip(0), atol=1e-6, rtol=1e-5)


def _tied_packet(first, n=200):
    """One packet with continuous coordinates (no distance ties) and three simultaneous events at
    positions `first, first + 1, first + 2`, each on its own pixel. `n` stays below the test
    encoder's `max_nodes`: above it the stride sampler keeps events by index, which is a third
    (documented, S37-identical) way storage order enters."""
    ev, ptr, dt = _batch([n], seed=5, integer=False)
    ev[first + 1, EV_T] = ev[first, EV_T]
    ev[first + 2, EV_T] = ev[first, EV_T]
    ev[first:first + 3, EV_X] = torch.tensor([30.5, 90.5, 150.5])
    ev[first:first + 3, EV_Y] = torch.tensor([40.5, 100.5, 60.5])
    return ev, ptr, dt


def _swap(ev, i, j):
    out = ev.clone()
    out[[i, j]] = ev[[j, i]]
    return out


def test_reordering_simultaneous_events_off_the_centroid_stride_changes_nothing():
    """Swapping two simultaneous, non-leading events that are not on the centroid stride keeps every
    token and every neighbour set, so the feature must be bitwise unchanged: the encoder has no
    hidden dependence on storage order beyond the documented one."""
    enc = _small()
    ev, ptr, dt = _tied_packet(first=4)          # positions 4 (leading), 5, 6
    assert 5 % enc.STRIDE and 6 % enc.STRIDE
    assert ev.shape[0] <= enc.max_nodes          # every event is a node: same node set
    with torch.no_grad():
        a = enc(ev, ptr, dt)
        b = enc(_swap(ev, 5, 6), ptr, dt)
    assert torch.equal(a, b)


def test_centroids_follow_storage_order():
    """The other half of the semantics, pinned rather than hidden: the centroid of SA1 is whichever
    event sits on the stride, so moving a different simultaneous event onto slot 4 changes the
    feature even though the event set, its time order and every token are the same (S37's sampler is
    index based in exactly the same way)."""
    enc = _small()
    ev, ptr, dt = _tied_packet(first=3)          # positions 3 (leading), 4 (centroid), 5
    tok_a = event_tokens(ev, ptr, dt, 180, 240)
    tok_b = event_tokens(_swap(ev, 4, 5), ptr, dt, 180, 240)
    assert torch.equal(tok_a[[4, 5]], tok_b[[5, 4]])          # tokens travel with their events
    with torch.no_grad():
        a = enc(ev, ptr, dt)
        b = enc(_swap(ev, 4, 5), ptr, dt)
    assert not torch.equal(a, b)


# ------------------------------------------------------------------ training signal
def test_gradients_reach_every_parameter_and_are_finite():
    enc = _small().train()
    ev, ptr, dt = _batch([600, 350, 45], seed=3)
    out = enc(ev, ptr, dt)
    w = torch.randn_like(out)
    (out * w).sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} receives no gradient"


def test_one_adam_step_moves_every_parameter():
    enc = _small().train()
    before = {n: p.detach().clone() for n, p in enc.named_parameters()}
    opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    ev, ptr, dt = _batch([600, 350, 45], seed=4)
    out = enc(ev, ptr, dt)
    (out * torch.randn_like(out)).sum().backward()
    opt.step()
    for n, p in enc.named_parameters():
        assert not torch.equal(p.detach(), before[n]), f"{n} did not move"
        assert torch.isfinite(p).all(), n


# ------------------------------------------------------------------ canvas scale law
def _record_knn(monkeypatch):
    calls = []
    real = EH.knn

    def rec(*a, **kw):
        out = real(*a, **kw)
        calls.append(out)
        return out

    monkeypatch.setattr(EH, "knn", rec)
    return calls


def test_knn_indices_do_not_depend_on_the_canvas(monkeypatch):
    """The same events translated by (+1000, +700) px inside a 2400 x 1800 canvas get bitwise the
    same neighbours at every level: the search sees relative pixel geometry only, never the frame."""
    calls = _record_knn(monkeypatch)
    small = EventHierEncoder(height=180, width=240, hidden=16, feat_dim=32, k=16,
                             max_nodes=2048).eval()
    big = EventHierEncoder(height=1800, width=2400, hidden=16, feat_dim=32, k=16,
                           max_nodes=2048).eval()
    big.load_state_dict(small.state_dict())
    ev, ptr, dt = _batch([3000, 700], seed=6)
    ev_big = ev.clone()
    ev_big[:, EV_X] += 1000.0
    ev_big[:, EV_Y] += 700.0
    with torch.no_grad():
        small(ev, ptr, dt)
        n_small = len(calls)
        big(ev_big, ptr, dt)
    # level-0 graph, SA1 groups, level-1 graph, SA2 groups, level-2 graph, SA3 groups
    assert n_small == 6 and len(calls) == 12
    for (i_a, v_a), (i_b, v_b) in zip(calls[:6], calls[6:]):
        assert torch.equal(i_a, i_b) and torch.equal(v_a, v_b)


class _AllocTracker(torch.utils._python_dispatch.TorchDispatchMode):
    """Peak bytes of the storages allocated inside the region, plus every op's output signature.

    CPU tensors have no `max_memory_allocated`, so liveness is polled through storage weak
    references at every op; a storage freed between two ops is released at the next one, which is
    the same for both runs being compared.
    """

    def __init__(self):
        super().__init__()
        from torch.multiprocessing.reductions import StorageWeakRef
        self._weak = StorageWeakRef
        self.refs, self.live, self.peak, self.signature = {}, 0, 0, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for key in [k for k, (r, _) in self.refs.items() if r.expired()]:
            self.live -= self.refs.pop(key)[1]
        for t in torch.utils._pytree.tree_flatten(out)[0]:
            if isinstance(t, torch.Tensor):
                st = t.untyped_storage()
                self.signature.append((str(func), tuple(t.shape), str(t.dtype)))
                key = (st.data_ptr(), st.nbytes())
                if st.nbytes() and key not in self.refs:
                    self.refs[key] = (self._weak(st), st.nbytes())
                    self.live += st.nbytes()
                    self.peak = max(self.peak, self.live)
        return out


@pytest.mark.parametrize("mode", ["translate", "scale"])
def test_forward_memory_does_not_grow_with_the_canvas(mode):
    """Same events on a 240 x 180 and a 2400 x 1800 canvas (translated, or spread 10x): the op-level
    allocation sequence is identical, hence so is peak memory (registered tolerance: < 5 %)."""
    torch.manual_seed(0)
    small = EventHierEncoder(height=180, width=240, hidden=16, feat_dim=32, k=16,
                             max_nodes=2048).eval()
    big = EventHierEncoder(height=1800, width=2400, hidden=16, feat_dim=32, k=16,
                           max_nodes=2048).eval()
    big.load_state_dict(small.state_dict())
    ev, ptr, dt = _batch([3000, 700], seed=7)
    ev_big = ev.clone()
    if mode == "translate":
        ev_big[:, EV_X] += 1000.0
        ev_big[:, EV_Y] += 700.0
    else:
        ev_big[:, EV_X] *= 10.0
        ev_big[:, EV_Y] *= 10.0
    runs = []
    for enc, e in ((small, ev), (big, ev_big)):
        with torch.no_grad(), _AllocTracker() as tr:
            enc(e, ptr, dt)
        runs.append(tr)
    a, b = runs
    assert a.signature == b.signature
    assert abs(b.peak - a.peak) / a.peak < 0.05, (a.peak, b.peak)


def test_no_dense_layers_and_no_canvas_sized_parameter():
    small = _small()
    big = _small(height=1800, width=2400)
    assert not [m for m in small.modules() if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
    assert {k: v.shape for k, v in small.state_dict().items()} == \
        {k: v.shape for k, v in big.state_dict().items()}


# ------------------------------------------------------------------ no state
def test_state_free_contract():
    with pytest.raises(ValueError):
        EventHierEncoder(extra_channels=2)
    enc = _small()
    ev, ptr, dt = _batch([30])
    with pytest.raises(ValueError):
        enc(ev, ptr, dt, extra=torch.zeros(30, 2))
    with pytest.raises(NotImplementedError):
        enc(ev, ptr, dt, return_nodes=True)
    with torch.no_grad():
        # a zero-width `extra` is what `tools/make_s36_row.py` builds from `in_dim - 7`
        assert torch.equal(enc(ev, ptr, dt, extra=torch.zeros(30, 0)), enc(ev, ptr, dt))


def _x1_model_cfg(**model_over):
    from config import load_config
    cfg = load_config(str(X1_CFG))
    cfg["MODEL"].update(model_over)
    return cfg


def test_prev_render_cannot_reach_event_hier():
    with pytest.raises(ValueError):
        MNISTModel(_x1_model_cfg(PREV_RENDER=True))


def test_frontend_registry_builds_event_hier_and_filters_its_key():
    enc = build_frontend("event_hier", hidden=16, feat_dim=32, k=4, n_layers=1, max_nodes=64,
                         t_px=5.0, window=8, cell=16)
    assert isinstance(enc, EventHierEncoder)
    assert (enc.k, enc.n_layers, enc.max_nodes, enc.t_px) == (4, 1, 64, 5.0)
    # the model passes `t_px` to every frontend; the others must never see it
    assert isinstance(build_frontend("event_gnn", hidden=16, feat_dim=32, t_px=5.0), EventGNN)
    build_frontend("raw_scan", hidden=16, feat_dim=32, t_px=5.0)
    build_frontend("sparse_cell", hidden=16, feat_dim=32, t_px=5.0)


def test_so3fk_builds_mano_only_when_needed():
    """`LOSS.TYPE: so3_trans_fk` builds MANO on an arm with no conditioning path; `mse_51d` does
    not, so no existing absolute arm gains `mano.*` state_dict keys."""
    m = MNISTModel(_x1_model_cfg())
    assert hasattr(m, "mano") and any(k.startswith("mano.") for k in m.state_dict())
    cfg = _x1_model_cfg()
    cfg["LOSS"]["TYPE"] = "mse_51d"
    m2 = MNISTModel(cfg)
    assert not hasattr(m2, "mano") and not any(k.startswith("mano.") for k in m2.state_dict())


# ------------------------------------------------------------------ real data, full model
@pytest.fixture(scope="module")
def real_batch():
    if not (DATA / "splits_semkine.json").exists():
        pytest.skip("training data not present on this machine")
    from config import load_config
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = load_config(str(X1_CFG))
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    ds = build_dataset(cfg, "train", comps, train=True)
    idx = [int(round(v)) for v in np.linspace(0, len(ds) - 1, 4)]
    return cfg, collate_packets([ds[i] for i in idx])


def test_x1_model_on_a_real_training_packet(real_batch):
    cfg, batch = real_batch
    torch.manual_seed(0)
    m = MNISTModel(cfg)
    assert m.encoder_name == "event_hier" and isinstance(m.event_encoder, EventHierEncoder)
    assert m.conv1 is None and m.rn is None and hasattr(m, "mano")
    assert not hasattr(m, "prev_mlp") and not hasattr(m, "root_head")
    assert not (m.predict_delta or m.prevpos_embed or m.prev_render or m.routed or m.active_head)
    m.train()
    pred = m.forward_packet(batch)
    assert pred.shape == (batch.batch_size, 51) and torch.isfinite(pred).all()
    loss, parts = m._compute_loss(pred, batch.target, batch.betas)
    assert torch.isfinite(loss) and float(loss) > 0
    for key in ("loss_rot", "loss_trans", "loss_fk", "loss_abs_fk"):
        assert torch.isfinite(parts[key]), key
    obj = loss.float().log10() if m.log10_loss else loss       # what training_step returns
    obj.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert m.pose_head.weight.grad.abs().sum() > 0
    assert m.event_encoder.embed.weight.grad.abs().sum() > 0


def test_x1_output_does_not_read_prev_state(real_batch):
    """Structurally absolute: replacing `prev_state` by noise changes the output by exactly zero."""
    import dataclasses
    cfg, batch = real_batch
    torch.manual_seed(0)
    m = MNISTModel(cfg).eval()
    with torch.no_grad():
        a = m.forward_packet(batch)
        b = m.forward_packet(dataclasses.replace(batch, prev_state=torch.randn_like(batch.prev_state)))
    assert torch.equal(a, b)


# ------------------------------------------------------------------ compatibility and config
def test_s37_checkpoint_still_loads_strictly():
    if not S37_CKPT.exists():
        pytest.skip("S37 checkpoint not present on this machine")
    from config import load_config
    m = MNISTModel(load_config(str(S37_CFG)))
    sd = torch.load(str(S37_CKPT), map_location="cpu")["state_dict"]
    m.load_state_dict(sd, strict=True)


def test_x1_config_differs_from_s37_only_where_declared():
    import yaml
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"), str(S37_CFG), str(X1_CFG),
         "--allow", "MODEL", "LOSS", "TRAIN.MAX_STEPS", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME",
         "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    s37 = yaml.safe_load(S37_CFG.read_text())
    x1 = yaml.safe_load(X1_CFG.read_text())
    ma, mb = s37["MODEL"], x1["MODEL"]
    changed = {k for k in set(ma) & set(mb) if ma[k] != mb[k]}
    assert changed == {"BACKBONE", "ENCODER", "ENCODER_HIDDEN", "ENCODER_LAYERS", "ENCODER_K",
                       "PREDICT_DELTA", "PREVPOS_EMBED", "ROUTED_READOUT", "ZERO_EVENT_GATE",
                       "ACTIVE_HEAD"}
    assert set(ma) - set(mb) == {"ROUTE_BAND_PX", "ROUTE_FRONT_K", "ENCODER_WINDOW",
                                 "ENCODER_T_SCALE", "ACTIVE_HIDDEN"}
    assert set(mb) - set(ma) == {"ENCODER_HIER_T_PX"}
    assert mb["ENCODER"] == "event_hier" and mb["ENCODER_K"] == 16
    assert mb["ENCODER_MAX_NODES"] == 2048 and mb["ENCODER_HIDDEN"] == 64
    assert not any(mb[k] for k in ("PREDICT_DELTA", "PREVPOS_EMBED", "PREV_RENDER",
                                   "ROUTED_READOUT", "ZERO_EVENT_GATE", "ACTIVE_HEAD"))
    assert x1["TRAIN"]["MAX_STEPS"] == 3000 and x1["TRAIN"]["RUN_NAME"] == "x1_hier_abs_s3407"
    want_loss = {"TYPE": "so3_trans_fk", "ROT_WEIGHT": 1.0, "TRANS_WEIGHT": 1.0, "FK_WEIGHT": 2.0,
                 "ABS_FK_WEIGHT": 1.0, "TRANS_BETA": 0.01}
    assert {k: x1["LOSS"][k] for k in want_loss} == want_loss
    for k in ("LAMBDA_POSE", "LAMBDA_T", "LAMBDA_R", "NORMALIZER", "LOG10"):
        assert x1["LOSS"][k] == s37["LOSS"][k], k
    if C0_CFG.exists():
        assert x1["LOSS"] == yaml.safe_load(C0_CFG.read_text())["LOSS"]
