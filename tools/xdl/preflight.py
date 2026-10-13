#!/usr/bin/env python3
"""Independent XDL implementation/resource preflight; never formal training.

Standalone uses real raw packets and the exact 2048-node architecture. DDP
uses two tiny real batches with asymmetric empty/no-self versus live/self rows.
All nonzero output-head fixtures are labelled; the formal zero init is retained.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'model'))

import torch
from torch import nn
from config import load_config
from tools.xdl.data import build_train_dataset, collate_recovery, counter_rng, packet_hash
from semkine.xyz_recovery_study import RecoveryXYZModel

SOURCES = ('tools/xdl/preflight.py', 'tools/xdl/train.py', 'tools/xdl/data.py',
           'semkine/xyz_recovery_study.py', 'semkine/xyz_observation.py',
           'semkine/event_gnn.py', 'semkine/xyz_mesh.py', 'model/model.py')
# Registered before collecting GPU comparison values. Parameter units differ.
TOLERANCES = {'translation_max_abs_m': 0.0005, 'rotation_max_abs_rad': 0.002,
              'log10_loss_abs': 0.001, 'gradient_relative_l2': 0.03,
              'gradient_max_abs_fraction': 0.03, 'gradient_max_abs_floor': 1e-6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def move(batch, device):
    return {key: ([p.to(device) for p in value] if key == 'prefix' else value.to(device))
            for key, value in batch.items()}


def make_model(arm, device, *, step=2000, active_head=True):
    cfg = load_config(REPO / f'configs/xdl/xdl_{arm}_s3407.yaml')
    torch.manual_seed(3407)
    model = RecoveryXYZModel(cfg).to(device)
    model._trainer = types.SimpleNamespace(global_step=step)
    model.train()
    if active_head:
        with torch.no_grad():
            model.xyz.fc2.weight.normal_(mean=0., std=.002)
    return model


def gradients(model):
    values = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise AssertionError(f'missing/nonfinite main gradient: {name}')
            values[name] = parameter.grad.detach().float().cpu().clone()
    return values


def run_main(model, batch):
    model.zero_grad(set_to_none=True)
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction, target, betas, packed = model._predict_batch(batch)
        loss = model._compute_loss(prediction, target, betas)[0].float().log10()
    loss.backward()
    return {'prev': packed.prev_state.detach().float().cpu(),
            'prediction': prediction.detach().float().cpu(),
            'loss': float(loss.detach()), 'grads': gradients(model)}


def full_burn(model, batch):
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        previous = None
        for prefix in batch['prefix']:
            packet = prefix if previous is None else dataclasses.replace(prefix, prev_state=previous)
            previous = model.forward_packet(packet).detach()
    model.train()
    model.reset_recovery_context()
    return previous


def difference(a, b):
    previous = (a['prev'] - b['prev']).abs()
    prediction = (a['prediction'] - b['prediction']).abs()
    ga = torch.cat([a['grads'][name].flatten() for name in sorted(a['grads'])])
    gb = torch.cat([b['grads'][name].flatten() for name in sorted(b['grads'])])
    gd = ga - gb
    report = {'prev_translation_max_abs_m': float(previous[:, :3].max()),
              'prev_rotation_max_abs_rad': float(previous[:, 3:].max()),
              'prediction_translation_max_abs_m': float(prediction[:, :3].max()),
              'prediction_rotation_max_abs_rad': float(prediction[:, 3:].max()),
              'log10_loss_abs': abs(a['loss'] - b['loss']),
              'gradient_relative_l2': float(gd.norm() / ga.norm().clamp_min(1e-12)),
              'gradient_max_abs': float(gd.abs().max()),
              'gradient_max_allowed': TOLERANCES['gradient_max_abs_floor'] + TOLERANCES['gradient_max_abs_fraction'] * float(ga.abs().max())}
    report['pass'] = (
        max(report['prev_translation_max_abs_m'], report['prediction_translation_max_abs_m']) <= TOLERANCES['translation_max_abs_m']
        and max(report['prev_rotation_max_abs_rad'], report['prediction_rotation_max_abs_rad']) <= TOLERANCES['rotation_max_abs_rad']
        and report['log10_loss_abs'] <= TOLERANCES['log10_loss_abs']
        and report['gradient_relative_l2'] <= TOLERANCES['gradient_relative_l2']
        and report['gradient_max_abs'] <= report['gradient_max_allowed'])
    return report


def empty_packet(packet):
    return dataclasses.replace(packet, events=packet.events[:0], ptr=torch.zeros_like(packet.ptr))


def standalone(report):
    device = torch.device('cuda:0')
    cfg = load_config(REPO / 'configs/xdl/xdl_q1_s3407.yaml')
    dataset = build_train_dataset(cfg)
    cpu_batch = collate_recovery([dataset[2000 * 1024 + row] for row in range(4)])
    report['real_main_sha256'] = packet_hash(cpu_batch['main'])
    batch = move(cpu_batch, device)
    selected = [r for r in range(4) if counter_rng(3407, 2000, r, 'self').random() < .5]
    report['selected_rows'] = selected
    report['checks'] = {}
    report['subset_vs_fullburn'] = {}
    teacher = make_model('q0', device)
    a = run_main(teacher, batch)
    full_burn(teacher, batch)  # ignored by the teacher main, as in original XD
    b = run_main(teacher, batch)
    comparison = difference(a, b)
    report['teacher_unused_burn_equivalence'] = comparison
    assert comparison['pass']
    report['checks']['teacher_has_zero_burn'] = teacher.xdl_burn_rows == 0
    del a, b, teacher
    torch.cuda.empty_cache()
    for arm in ('f1', 'q1', 'n1'):
        model = make_model(arm, device)
        subset = run_main(model, batch)
        predicted = full_burn(model, batch)
        prior = batch['main'].prev_state.clone()
        prior[selected] = predicted[selected].to(prior)
        supplied = dataclasses.replace(batch['main'], prev_state=prior)
        complete = run_main(model, supplied)
        comparison = difference(subset, complete)
        comparison.update(actual_self_rows=model.xdl_self_rows, actual_burn_rows=model.xdl_burn_rows)
        report['subset_vs_fullburn'][arm] = comparison
        assert comparison['pass'], (arm, comparison)
        expected_burn = 3 * (batch['main'].batch_size if model.xdl_burn_mode == 'full' else len(selected))
        assert model.xdl_self_rows == len(selected) and model.xdl_burn_rows == expected_burn
        report['checks'][f'{arm}_main_grad_present_finite_after_detached_burn'] = True
        del model, subset, complete
        torch.cuda.empty_cache()
    null = make_model('n1', device)
    null_input = {}
    hook = null.xyz.fc1.register_forward_pre_hook(lambda _, inputs: null_input.update(input=inputs[0].detach().clone()))
    result = run_main(null, batch['main'])
    report['checks']['N1_final544_plus64_exactzero'] = not bool(null_input['input'][:, :608].count_nonzero())
    observation_grad = {name: grad for name, grad in result['grads'].items()
                        if name.startswith('observation_bank.') or (name.startswith('xyz.') and not name.startswith('xyz.fc'))}
    report['checks']['N1_all_observation_grad_present_and_exactzero'] = all(not bool(g.count_nonzero()) for g in observation_grad.values())
    alternative = batch['main'].events.clone()
    alternative[:, 1] = (alternative[:, 1] + 77) % 240
    alternative[:, 2] = (alternative[:, 2] + 53) % 180
    null.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        original = null.forward_packet(batch['main'])
        changed = null.forward_packet(dataclasses.replace(batch['main'], events=alternative))
        empty = null.forward_packet(empty_packet(batch['main']))
    report['checks']['N1_nonempty_changed_event_output_bitwise_same'] = bool(torch.equal(original, changed))
    report['checks']['N1_empty_output_exact_float32_prior'] = bool(torch.equal(empty, batch['main'].prev_state.float()))
    hook.remove()
    del null, result
    # F/Q only differ in query centers; same parameter initialization and support.
    hashes = []
    for arm in ('f0', 'q0'):
        model = make_model(arm, device, active_head=False)
        model.xyz_keep_aux = True
        initial = hashlib.sha256()
        for name, param in model.named_parameters():
            initial.update(name.encode()); initial.update(param.detach().cpu().numpy().tobytes())
        hashes.append(initial.hexdigest())
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model.forward_packet(batch['main'])
            all_empty = model.forward_packet(empty_packet(batch['main']))
        report['checks'][f'{arm}_empty_exact_hold'] = bool(torch.equal(all_empty, batch['main'].prev_state.float()))
        # Replicate camera project in camera pixels, then apply renderer scale.
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            _, joints = model._fk(batch['main'].prev_state.float(), batch['main'].betas.float())
            inverse = torch.argsort(model.mano.mano16_tips_to_openpose21)
            xyz = joints[:, inverse[:16]]
            K = batch['main'].camera_K.float()
            full_pixels = torch.stack([K[:, None, 0, 0] * xyz[..., 0] / xyz[..., 2] + K[:, None, 0, 2],
                                       K[:, None, 1, 1] * xyz[..., 1] / xyz[..., 2] + K[:, None, 1, 2]], -1)
            expected = full_pixels * float(model.cfg['MODEL']['RENDER_SCALE'])
            live = xyz[..., 2] > 1e-6
            actual = model.xyz_last['xdl_raw_uv']
            discrepancy = float((actual[live] - expected[live]).abs().max())
        report.setdefault('scaled_K_projection_max_abs_px', {})[arm] = discrepancy
        report['checks'][f'{arm}_scaled_K_projection_correct'] = discrepancy < 1e-3
        report.setdefault('sampling_src_sha256', {})[arm] = hashlib.sha256(model.xyz_last['xdl_src'].detach().cpu().numpy().tobytes()).hexdigest()
        del model
    report['checks']['FQ_common_initial_parameter_sha256'] = len(set(hashes)) == 1
    report['common_initial_parameter_sha256'] = hashes[0]
    report['checks']['FQ_common_stride_support'] = len(set(report['sampling_src_sha256'].values())) == 1
    assert all(report['checks'].values()), report['checks']
    report['max_memory_allocated_gb'] = torch.cuda.max_memory_allocated() / 2**30
    report['status'] = 'pass'


class _LossFixture(nn.Module):
    def __init__(self, study):
        super().__init__()
        self.study = study

    def forward(self, batch):
        prediction, target, betas, _ = self.study._predict_batch(batch)
        return self.study._compute_loss(prediction, target, betas)[0].float().log10()


def ddp_asymmetric(report, output):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=120))
    device = torch.device('cuda', rank)
    cfg = load_config(REPO / 'configs/xdl/xdl_q1_s3407.yaml')
    dataset = build_train_dataset(cfg)
    # rank0 uses only teacher coin rows; rank1 only self coin rows at plateau.
    rows = [row for row in range(rank, 1024, 2)
            if (counter_rng(3407, 2000, row, 'self').random() < .5) == (rank == 1)][:2]
    batch = move(collate_recovery([dataset[2000 * 1024 + row] for row in rows]), device)
    if rank == 0:
        batch['main'] = empty_packet(batch['main'])
        batch['prefix'] = [empty_packet(p) for p in batch['prefix']]
    results = []
    for arm in ('q1', 'n1'):
        study = make_model(arm, device)
        wrapper = DistributedDataParallel(_LossFixture(study), device_ids=[rank], find_unused_parameters=False)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=1e-4)
        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = wrapper(batch)
            loss.backward()
            grads = gradients(study)
            observations = {name: grad for name, grad in grads.items()
                            if name.startswith('observation_bank.') or (name.startswith('xyz.') and not name.startswith('xyz.fc'))}
            event_grad_zero = all(not bool(g.count_nonzero()) for g in observations.values())
            if arm == 'n1':
                assert event_grad_zero
            else:
                assert not event_grad_zero
            assert bool(torch.isfinite(loss))
            optimizer.step()
            results.append({'arm': arm, 'iteration': iteration, 'rank': rank, 'global_rows': rows,
                            'main_all_empty': rank == 0, 'self_rows_total': study.xdl_self_rows,
                            'burn_rows_total': study.xdl_burn_rows, 'loss_finite': True,
                            'all_grad_present_finite_after_allreduce': True,
                            'observations_all_zero_grad_after_allreduce': event_grad_zero})
        assert study.xdl_self_rows == (0 if rank == 0 else 4)
        assert study.xdl_burn_rows == (0 if rank == 0 else 12)
        dist.barrier()
        del wrapper, optimizer, study
        torch.cuda.empty_cache()
    report.update(status='pass', rank=rank, rows=results,
                  max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                  scope='two iterations of Q1 and N1, rank0 true-empty/no-self and rank1 real-events/self, bf16, active output head fixture')
    save(output.with_name(output.stem + f'_rank{rank}.json'), report)
    dist.destroy_process_group()


def timing(report):
    """Measure actual rank-sized plateau burn; no optimizer updates or results."""
    from tools.xdl.train import make_loader
    cfg = load_config(REPO / 'configs/xdl/xdl_q1_s3407.yaml')
    _, loader = make_loader(cfg, start_step=2000, max_steps=2004, workers=4, distributed=True)
    iterator = iter(loader)
    cpu_batch = next(iterator)
    iterator._shutdown_workers()
    device = torch.device('cuda:0')
    before = torch.cuda.memory_allocated()
    batch = move(cpu_batch, device)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    prefix_bytes = sum(t.numel() * t.element_size() for p in cpu_batch['prefix']
                       for t in p.__dict__.values() if torch.is_tensor(t))
    main_bytes = sum(t.numel() * t.element_size() for t in cpu_batch['main'].__dict__.values() if torch.is_tensor(t))
    model = make_model('q1', device)
    report.update(per_rank_batch=512, global_step_fixture=2000,
                  batch_data_cuda_allocated_bytes=after - before,
                  prefix_tensor_bytes=prefix_bytes, main_tensor_bytes=main_bytes,
                  main_event_count=int(batch['main'].events.shape[0]),
                  self_row_count=sum(counter_rng(3407, 2000, int(r), 'self').random() < .5 for r in batch['global_row'].cpu().tolist()))
    times = {}
    for self_enabled in (False, True):
        model.xdl_self = self_enabled
        label = 'self_plateau' if self_enabled else 'main_no_burn'
        values = []
        for repetition in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction, target, betas, _ = model._predict_batch(batch)
                loss = model._compute_loss(prediction, target, betas)[0].float().log10()
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            if repetition:
                values.append(elapsed)
            print(f'XDL independent timing {label} repetition={repetition} seconds={elapsed:.6f}', flush=True)
        gradients(model)
        times[label] = values
    extra = sum(times['self_plateau']) / len(times['self_plateau']) - sum(times['main_no_burn']) / len(times['main_no_burn'])
    report.update(status='pass', measured_gpu_forward_backward_times_s=times,
                  plateau_selected_burn_extra_wall_s=extra,
                  max_memory_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                  scope='one real rank-sized B512 batch, activehead fixture, 1warmup+3 timings each; no optimizer/no throughput-result claim')


def equivalence512(report):
    """Strict rank-sized equivalence under learned and high-response fixtures."""
    from tools.xdl.train import make_loader
    cfg = load_config(REPO / 'configs/xdl/xdl_q1_s3407.yaml')
    _, loader = make_loader(cfg, start_step=2000, max_steps=2004, workers=4, distributed=True)
    iterator = iter(loader)
    cpu_batch = next(iterator)
    iterator._shutdown_workers()
    batch = move(cpu_batch, torch.device('cuda:0'))
    selected = [i for i, row in enumerate(batch['global_row'].cpu().tolist())
                if counter_rng(3407, 2000, int(row), 'self').random() < .5]
    fixtures = {
        'health_step20': REPO / 'outputs/xdl/preflight/q1_health/xdl_q1_health-step=20.ckpt',
        'oldC0_mapped_high_response': REPO / 'outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt',
    }
    report.update(real_main_sha256=packet_hash(cpu_batch['main']), selected_rows=len(selected),
                  per_rank_batch=512, strict_selected_state_and_prediction_max_abs=1e-5,
                  fixture_only_no_formal_warmstart=True, fixture_results={})
    for name, path in fixtures.items():
        model = make_model('q1', torch.device('cuda:0'), active_head=False)
        state = torch.load(path, map_location='cpu')['state_dict']
        if name == 'health_step20':
            model.load_state_dict(state, strict=True)
        else:
            mapped = {k: value for k, value in state.items() if k.startswith('xyz.') and k != 'xyz.fc1.weight'}
            mismatch = model.load_state_dict(mapped, strict=False)
            assert not mismatch.unexpected_keys
            with torch.no_grad():
                old = state['xyz.fc1.weight'].to(model.xyz.fc1.weight)
                model.xyz.fc1.weight.zero_()
                model.xyz.fc1.weight[:, :544].copy_(old[:, :544])
                model.xyz.fc1.weight[:, 611:659].copy_(old[:, 544:592])
            # fc1.bias and fc2 are among the copied same-shape legacy weights.
        called = []
        original = model.forward_packet
        def capture(packet):
            output = original(packet)
            called.append(output.detach().float().cpu())
            return output
        model.forward_packet = capture
        subset = run_main(model, batch)
        selected_burn_predictions = called[:3]
        called.clear()
        predictions = full_burn(model, batch)
        complete_burn_predictions = called[:3]
        prior = batch['main'].prev_state.clone()
        prior[selected] = predictions[selected].to(prior)
        main = dataclasses.replace(batch['main'], prev_state=prior)
        complete = run_main(model, main)
        comparison = difference(subset, complete)
        per_step = [float(((small[selected] if small.shape[0] == batch['main'].batch_size else small) - big[selected]).abs().max())
                    for small, big in zip(selected_burn_predictions, complete_burn_predictions)]
        terminal = float((subset['prev'][selected] - complete['prev'][selected]).abs().max())
        final_prediction = float((subset['prediction'][selected] - complete['prediction'][selected]).abs().max())
        strict = max(per_step + [terminal, final_prediction]) <= 1e-5
        row = {'fixture_checkpoint': str(path), 'checkpoint_sha256': sha(path),
               'actual_burn_mode': model.xdl_burn_mode,
               'burn_predictions_max_abs_each_step': per_step,
               'terminal_selected_previous51_max_abs': terminal,
               'selected_main_prediction51_max_abs': final_prediction,
               'loss_gradient_comparison': comparison,
               'strict_state_and_prediction_pass': strict,
               'pass': strict and comparison['pass']}
        report['fixture_results'][name] = row
        print(name, json.dumps(row), flush=True)
        del model, subset, complete, called
        torch.cuda.empty_cache()
    report['status'] = 'pass' if all(row['pass'] for row in report['fixture_results'].values()) else 'FAIL'
    report['scope'] = 'B512 rank0 real rawW50, actual chosen262 versus full512 three burns; bf16; learned20 and legacy-mapped high-response fixtures; neither fixture initializes formal training'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('standalone', 'ddp', 'timing', 'equivalence512'), default='standalone')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    cards = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
    if (args.mode in ('standalone', 'timing', 'equivalence512') and cards != [2]) or (args.mode == 'ddp' and cards != [2, 3]):
        raise ValueError('this independent preflight uses only GPU2 standalone or GPU2,3 DDP')
    start = time.monotonic()
    report = {'status': 'not_completed', 'formal_training': False,
              'reviewer': '/root/xd_train_impl independent of model implementer',
              'fixture_output_head': 'fc2.weight std=.002 for gradient exposure; formal remains zero',
              'source_sha256': {p: sha(REPO / p) for p in SOURCES},
              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
              'tolerances_registered_before_values': TOLERANCES,
              'mode': args.mode, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    output = Path(args.output)
    try:
        if args.mode == 'standalone':
            standalone(report)
        elif args.mode == 'ddp':
            ddp_asymmetric(report, output)
        elif args.mode == 'equivalence512':
            equivalence512(report)
        else:
            timing(report)
        report['elapsed_s'] = time.monotonic() - start
        if args.mode != 'ddp':
            save(output, report)
        else:
            save(output.with_name(output.stem + f'_rank{os.environ["LOCAL_RANK"]}.json'), report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report.update(status='FAIL', elapsed_s=time.monotonic() - start, traceback=traceback.format_exc())
        save(output if args.mode != 'ddp' else output.with_name(output.stem + f'_rank{os.environ.get("LOCAL_RANK", "unknown")}.json'), report)
        raise


if __name__ == '__main__':
    main()
