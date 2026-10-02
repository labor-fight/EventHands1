#!/usr/bin/env python3
"""U1a full forward, frozen-state and gradient debug on real training packets."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R/'model')]
from model import MNISTModel
from semkine.dataset import build_dataset
from semkine.events import collate_packets
from pose_repr import decode_to_mano_inputs

torch.set_num_threads(4)
cfgs = [yaml.safe_load((R/f'configs/u1a/u1a_{m}_debug_s3407.yaml').read_text()) for m in ('shared','untied')]
models = []
for cfg in cfgs:
    m = MNISTModel(copy.deepcopy(cfg))
    m.init_from_abs_checkpoint(cfg['MODEL']['INIT_FROM'])
    models.append(m.cuda())
components = np.load(cfgs[0]['MANO']['NPZ'])['hands_components'].astype(np.float32)
ds = build_dataset(cfgs[0], 'train', components, train=True)
torch.manual_seed(17)
batch = collate_packets([ds[i] for i in range(8)]).to('cuda')
stats = {}
for name, m in zip(('shared','untied'), models):
    m.train()
    assert not m.event_encoder.training
    before = {k:v.clone() for k,v in m.event_encoder.state_dict().items()}
    out = m.forward_packet(batch)
    assert out.shape == (8,51) and torch.isfinite(out).all()
    # Exercise heads beyond their all-zero starting point, rather than testing a constant function.
    heads = [m.u1a_readout.shared_head] if name == 'shared' else list(m.u1a_readout.node_heads)
    with torch.no_grad():
        for head in heads:
            head[-1].weight.fill_(0.001)
    p = batch.prev_state.clone()
    out = m.forward_packet(batch)
    raw = m.u1a_last_raw.clone()
    batch.prev_state = p + 0.13
    m.forward_packet(batch)
    assert torch.equal(raw[:,3:], m.u1a_last_raw[:,3:]), 'root/finger depend on prev/routing'
    batch.prev_state = p
    loss = m.forward_packet(batch).square().mean()
    loss.backward()
    grads = [p.grad for p in m.u1a_readout.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    assert all(p.grad is None for p in m.event_encoder.parameters())
    assert all(torch.equal(v, m.event_encoder.state_dict()[k]) for k,v in before.items())
    dec = decode_to_mano_inputs(out, m.pose_repr, m.mano.hands_components, m.mano.hands_mean)
    assert torch.allclose(dec['local_full_aa'], out[:,6:] + m.mano.hands_mean)
    stats[name] = {'finite_forward': True, 'raw_prev_independent': True, 'frozen_encoder': True,
                   'all_head_gradients_finite': True, 'mean_added_once': True,
                   'params': sum(p.numel() for p in m.parameters())}
    m.eval()
    # An empty packet must hold all 51 output coordinates.
    empty = collate_packets([ds[0]]).to('cuda')
    empty.events = empty.events[:0]
    empty.ptr = torch.zeros(2, dtype=torch.long, device='cuda')
    empty.prev_state = p[:1]
    assert torch.equal(m.forward_packet(empty), empty.prev_state.float())
    stats[name]['empty_hold'] = True
    empty_item = copy.deepcopy(ds[0])
    empty_item.events = empty_item.events[:0]
    mixed = collate_packets([empty_item, ds[1]]).to('cuda')
    assert torch.equal(m.forward_packet(mixed)[0], mixed.prev_state[0].float())
    stats[name]['mixed_empty_hold'] = True
    m.train()
    opt = torch.optim.Adam(m.u1a_readout.parameters(), lr=0.004)
    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        pred = m.forward_packet(batch)
        actual, parts = m._compute_loss(pred, batch.target, batch.betas)
        objective = actual.float().log10() if m.log10_loss else actual
        assert torch.isfinite(objective)
        losses.append(float(actual.detach()))
        objective.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.u1a_readout.parameters())
        opt.step()
    assert min(losses[1:]) < losses[0], 'actual S38 objective cannot decrease'
    assert all(torch.equal(v, m.event_encoder.state_dict()[k]) for k,v in before.items())
    stats[name]['actual_loss_start'] = losses[0]
    stats[name]['actual_loss_min'] = min(losses)
    stats[name]['actual_loss_end'] = losses[-1]
path = R/'outputs/u1a/debug_contract.json'
path.write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
