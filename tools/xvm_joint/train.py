"""Frozen-S43 conditioned Stage1 training for the isolated XJ local operator.

This entry point is opt-in. Source/config/protocol/weight/split verification
precedes every real source reader and every optimizer. Geometry/input forward
is guarded against target-pose and actual-shape reads. Only the local network
is trained; this is not the fully fine-tuned joint method or a final main row.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')
    temporary.replace(path)


def tensor_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def supervision_digest(labels):
    """Bind actual supervision bytes after the separate reader returned them."""
    digest = hashlib.sha256()
    for name in ('target0', 'target1', 'beta_target', 'mask'):
        array = np.ascontiguousarray(getattr(labels, name))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def advance_chain(previous, *tokens):
    digest = hashlib.sha256(bytes.fromhex(previous) if previous else b'')
    for token in tokens:
        digest.update(str(token).encode())
    return digest.hexdigest()


def require_real_gpu_engineering(config_path, sourcefreeze_sha256):
    """The direct training entry point enforces the same gate as the launcher."""
    receipt_path = Path(config_path).resolve().parent / 'engineering/real_gpu.json'
    if not receipt_path.exists():
        raise RuntimeError('Real GPU forward/backward engineering receipt is required before training')
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('status') != 'PASS' or receipt.get('sourcefreeze_sha256') != sourcefreeze_sha256:
        raise RuntimeError('Real GPU engineering is not PASS for this exact source freeze')
    if receipt.get('actual_backward_calls', 0) < 1 or receipt.get('optimizer_steps') != 0:
        raise RuntimeError('Real GPU engineering must bind actual backward with zero optimizer steps')
    return receipt_path


def native_lnes_ms(events, start_ms, cutoff_ms, *, height=180, width=240):
    """Original W50 millisecond LNES from XJ ``[x,y,absolute_us,p]`` rows.

    The old ``native_lnes`` helper uses a different event-column layout, so
    it must not be called directly on an XJ packet. The first millisecond
    retains its original zero code and same-slot events combine by max.
    """
    if int(cutoff_ms) - int(start_ms) != 50:
        raise ValueError('Canonical absolute branch uses exactly W50')
    events = np.asarray(events)
    image = np.zeros((height, width, 2), np.float32)
    if len(events):
        if events.ndim != 2 or events.shape[1] != 4 or events.dtype.kind not in 'iu':
            raise ValueError('XJ packet requires integer [x,y,absolute_us,polarity]')
        relative_ms = events[:, 2] // 1000 - int(start_ms)
        if np.any(relative_ms < 0) or np.any(relative_ms >= 50):
            raise ValueError('Events exceed the causal absolute-branch window')
        np.maximum.at(image, (events[:, 1], events[:, 0], events[:, 3]),
                      relative_ms.astype(np.float32) / np.float32(50))
    return image


def source_record_weights(source_ids, mask):
    """Fixed reciprocal candidate counts; learned association cannot erase loss."""
    if source_ids.shape != mask.shape:
        raise ValueError('Source IDs and supervision mask differ')
    weights = torch.zeros_like(mask, dtype=torch.float32)
    for lane in range(len(source_ids)):
        active = mask[lane].bool() & (source_ids[lane] >= 0)
        if not bool(active.any()):
            continue
        ids = source_ids[lane, active]
        _, inverse, counts = torch.unique(ids, sorted=True, return_inverse=True, return_counts=True)
        weights[lane, active] = counts[inverse].float().reciprocal()
    return weights.detach()


def camera_valid(base, poses, beta0, *, minimum_depth_m=1e-4):
    """Finite state and all MANO vertices in front of the camera, per lane."""
    if not bool(torch.isfinite(poses).all()):
        raise ArithmeticError('Nonfinite state; do not hide it with camera fallback')
    with torch.no_grad():
        vertices, _ = base._fk(poses.detach().float(), beta0)
        finite = torch.isfinite(vertices).flatten(1).all(-1)
        front = (vertices[..., 2] > float(minimum_depth_m)).all(-1)
    return finite & front


def configure_device(device, config, *, engineering=False):
    device = torch.device(device)
    record = {'device': str(device), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')}
    if device.type == 'cpu':
        if not engineering:
            raise RuntimeError('This preregistered training stage requires an authorized GPU')
        return device, record
    if device.type != 'cuda' or device.index not in (None, 0):
        raise RuntimeError('Use logical cuda:0 under one explicitly selected physical GPU')
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    physical = None
    for row in rows:
        index, uuid = [part.strip() for part in row.split(',')]
        if selected in (index, uuid):
            physical = int(index)
            record['gpu_uuid'] = uuid
    if physical not in range(2, 8):
        raise RuntimeError('Exactly one physical GPU2-7 must be selected; GPU0-1 are forbidden')
    if not engineering and physical != int(config['physical_gpu']):
        raise RuntimeError('Physical training GPU differs from frozen configuration')
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    memory_limit_gib = float(os.environ.get('XJ_GPU_MEMORY_LIMIT_GIB', '16'))
    total_gib = properties.total_memory / (1024 ** 3)
    if memory_limit_gib <= 0 or memory_limit_gib >= total_gib:
        raise RuntimeError('Invalid per-process GPU memory limit')
    torch.cuda.set_per_process_memory_fraction(memory_limit_gib / total_gib, device)
    record.update(physical_gpu=physical, gpu_name=properties.name,
                  total_memory_gib=total_gib, process_memory_cap_gib=memory_limit_gib)
    return device, record


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    if saved.get('cuda') is not None:
        torch.cuda.set_rng_state_all(saved['cuda'])


def gradient_report(model):
    groups = {}
    finite = True
    for name, parameter in model.named_parameters():
        group = name.split('.')[0]
        row = groups.setdefault(group, {'parameters': 0, 'with_grad': 0, 'grad_norm_squared': 0.0})
        row['parameters'] += parameter.numel()
        if parameter.grad is not None:
            row['with_grad'] += parameter.numel()
            finite = finite and bool(torch.isfinite(parameter.grad).all())
            row['grad_norm_squared'] += float(parameter.grad.detach().double().square().sum())
    for row in groups.values():
        row['grad_norm'] = math.sqrt(row.pop('grad_norm_squared'))
    return {'finite': finite, 'groups': groups,
            'total_norm': math.sqrt(sum(row['grad_norm'] ** 2 for row in groups.values()))}


def reference_perturbation(previous, config):
    probability = float(config.get('reference_perturb_probability', 0.0))
    if not 0. <= probability <= 1.:
        raise ValueError('Reference perturbation probability must be in[0,1]')
    keys = ('reference_translation_std_m', 'reference_root_std_rad', 'reference_fingers_std_rad')
    if probability > 0 and any(key not in config for key in keys):
        raise ValueError('Enabled reference perturbations require the registered standard-deviation keys')
    selected = torch.rand((len(previous),), device=previous.device) < probability
    scales = previous.new_tensor([float(config.get(keys[0], 0.0))] * 3
                                  + [float(config.get(keys[1], 0.0))] * 3
                                  + [float(config.get(keys[2], 0.0))] * 45)
    clip_sigma = float(config.get('reference_clip_sigma', 3.0))
    if clip_sigma <= 0 or bool((scales < 0).any()):
        raise ValueError('Positive clipping and nonnegative reference-noise scales required')
    noise = torch.randn_like(previous).clamp(-clip_sigma, clip_sigma) * scales
    noise = noise * selected[:, None]
    return previous + noise, selected, noise


class Stage1Runtime:
    """Own-state prediction and only-then supervision for one chronological batch."""
    def __init__(self, config, device, *, engineering=False):
        from tools.s43_integration.model import load_legacy
        from tools.s43_rel.mean_shape_replay import InferenceAccessGuard
        from .data import XJRunLanes, XJSupervisionReader
        from .geometry import JointGeometry, GeometryConfig
        from .model import JointLocalOperator
        self.config, self.device = config, device
        self.base, self.base_config = load_legacy(config['base_checkpoint'], config['base_config'], device)
        self.base = self.base.float().eval()
        self.base.render_fp32 = True
        if getattr(self.base, 'root_compose', 'add') != 'add':
            raise RuntimeError('Frozen XJ Stage1 physical Jacobians require canonical additive root composition')
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.local = JointLocalOperator().to(device).float()
        self.local.train()
        self.geometry = JointGeometry(self.base, GeometryConfig(
            max_supports=int(config['num_supports']),
            candidates_per_support=int(config['candidates_per_support']),
            radius_m=float(config['radius_m'])))
        self.guard_class = InferenceAccessGuard
        with InferenceAccessGuard() as guard:
            self.lanes = XJRunLanes(config['data_root'], subjects=tuple(config['subjects']),
                                    batch_size=int(config['batch_size']), seed=int(config['seed']),
                                    warmup_steps=2, log_counts=False)
        self.setup_input_audit = guard.receipt()
        self.supervision = XJSupervisionReader(config['data_root'])
        self.previous = None
        self.absolute_previous = None
        self.batches = 0
        self.source_chain = ''
        self.supervision_chain = ''
        self.engineering = engineering

    def step(self):
        from .data import stack_inputs
        from .model import local_record_loss
        device = self.device
        timings = {}
        sync(device)
        started = time.perf_counter()
        with self.guard_class() as guard:
            batch = self.lanes.peek_batch()
            arrays = stack_inputs(batch)
            if not arrays['valid'].all():
                raise RuntimeError('Training lane unexpectedly padded; this stage uses infinite complete-window lanes')
            f0 = torch.from_numpy(arrays['field0']).to(device)
            f1 = torch.from_numpy(arrays['field1']).to(device)
            K = torch.from_numpy(arrays['K_source']).to(device)
            beta0 = torch.from_numpy(arrays['beta_input']).to(device)
            initial = torch.from_numpy(arrays['initial_state']).to(device)
            reset = torch.from_numpy(arrays['reset']).to(device)
            exposure = torch.from_numpy(arrays['both_snapshots_valid']).to(device)
            if self.previous is None:
                self.previous = initial.clone()
                self.absolute_previous = initial.clone()
            previous = torch.where(reset[:, None], initial, self.previous).detach()
            absolute_previous = torch.where(reset[:, None], initial, self.absolute_previous).detach()
            images = np.stack([native_lnes_ms(item['events'], item['start_ms'], item['cutoff_ms']) for item in batch])
            lnes = torch.from_numpy(images).to(device)
            sync(device)
            timings['source_field_s'] = time.perf_counter() - started
            stamp = time.perf_counter()
            with torch.no_grad():
                absolute_raw = self.base(lnes, previous, betas=beta0, camera_K=K).float()
                absolute_valid = camera_valid(self.base, absolute_raw, beta0)
                absolute = torch.where(absolute_valid[:, None], absolute_raw, previous)
            # The perturbation is a local-reference intervention. The absolute
            # proposal always came from the unperturbed own feedback above.
            reference, perturbed, noise = reference_perturbation(previous, self.config)
            reference_valid = camera_valid(self.base, reference, beta0)
            reference = torch.where(reference_valid[:, None], reference, previous)
            sync(device)
            timings['absolute_s'] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            prepared = self.geometry.prepare(f0, f1, reference, absolute, K, both_exposed=exposure)
            sync(device)
            timings['geometry_s'] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            delta0, delta1, aux = self.local(**prepared.model_kwargs)
            proposed = absolute + delta1
            proposed_valid = camera_valid(self.base, proposed, beta0)
            empty = torch.as_tensor([len(item['events']) == 0 for item in batch], device=device)
            pred = torch.where((proposed_valid & ~empty)[:, None], proposed, previous)
            if not bool(torch.isfinite(pred).all()):
                raise ArithmeticError('Nonfinite emitted state')
            sync(device)
            timings['local_network_s'] = time.perf_counter() - stamp
        # This is the FIRST label/actual-beta read in this step. Neither a GT
        # start nor a target mesh was available to fields, geometry or local.
        stamp = time.perf_counter()
        labels = self.supervision.read(batch, inputs_complete=True)
        supervision_token = supervision_digest(labels)
        mask = torch.from_numpy(labels.mask.copy()).to(device)
        if bool(mask.any()):
            gt0 = torch.from_numpy(labels.target0.copy()).to(device)
            gt1 = torch.from_numpy(labels.target1.copy()).to(device)
            target_beta = torch.from_numpy(labels.beta_target.copy()).to(device)
            # Safe placeholders for warmup rows: they are NOT teacher inputs
            # and remain excluded from every supervised loss and target read.
            target0 = torch.where(mask[:, None], gt0, reference.detach())
            target1 = torch.where(mask[:, None], gt1, absolute.detach())
            targets = self.geometry.targets(prepared, target0, target1)
            supervised_records = targets['supervised'].bool() & mask[:, None]
            fixed_weights = source_record_weights(prepared.model_kwargs['source_ids'], supervised_records)
            validity_weights = source_record_weights(
                prepared.model_kwargs['source_ids'], prepared.model_kwargs['candidate_mask'].bool() & mask[:, None])
            # NLL uses fixed, detached source weights, never learned alpha.
            losses = local_record_loss(aux, targets['target_a_px'], targets['target_d_px'],
                                       prepared.rho, supervised_records,
                                       validity_target=None, record_weights=fixed_weights)
            validity_per = F.binary_cross_entropy_with_logits(
                aux['validity_logits'], targets['validity_target'].to(aux['validity_logits']), reduction='none')
            validity_bce = (validity_per * validity_weights).sum() / validity_weights.sum().clamp_min(1)
            _, pj = self.base._fk(pred[mask].float(), beta0[mask])
            with torch.no_grad():
                _, gj = self.base._fk(gt1[mask].float(), target_beta[mask])
            terminal_mse = (pj - gj).square().mean()
            loss = (losses['local_nll'] + float(self.config.get('validity_loss_weight', .1)) * validity_bce
                    + float(self.config.get('terminal_joint_loss_weight', 1000.)) * terminal_mse)
            loss_values = dict(local_nll=float(losses['local_nll'].detach()),
                               validity_bce=float(validity_bce.detach()),
                               terminal_joint_mse_m2=float(terminal_mse.detach()),
                               total=float(loss.detach()))
            record_count = int(supervised_records.sum())
        else:
            loss = None
            loss_values = None
            record_count = 0
        sync(device)
        timings['supervision_loss_s'] = time.perf_counter() - stamp
        metrics = dict(batch_id=batch.batch_id, input_token=batch.token,
                       supervision_token=supervision_token,
                       source_hash_chain=advance_chain(getattr(self, 'source_chain', ''), batch.token),
                       supervision_hash_chain=advance_chain(getattr(self, 'supervision_chain', ''),
                                                            batch.token, supervision_token),
                       supervised_outputs=int(mask.sum()), supervised_records=record_count,
                       event_count=sum(len(item['events']) for item in batch),
                       reset_count=int(reset.sum()), empty_count=int(empty.sum()),
                       absolute_camera_invalid=int((~absolute_valid).sum()),
                       reference_camera_invalid=int((~reference_valid).sum()),
                       local_camera_invalid=int((~proposed_valid).sum()),
                       perturbed_reference_lanes=int(perturbed.sum()),
                       perturbation_norm=float(noise.norm(dim=-1).mean()),
                       active_records=int(aux['active'].sum()),
                       mean_association_mass=float(aux['association_mass'].detach().mean()),
                       mean_validity=float(aux['validity_score'].detach().mean()),
                       mean_update_norm=float(delta1.detach().norm(dim=-1).mean()),
                       state_norm=float(pred.detach().norm(dim=-1).mean()),
                       loss=loss_values, timings=timings, input_audit=guard.receipt(),
                       rows=[dict(recording=item['recording'], run_id=item['run_id'],
                                  start_ms=item['start_ms'], cutoff_ms=item['cutoff_ms'],
                                  packet_ordinal=item['packet_ordinal'], source_range=item['source_range'])
                             for item in batch])
        return dict(batch=batch, pred=pred, absolute=absolute, absolute_previous=absolute_previous,
                    loss=loss, metrics=metrics, labels_mask=mask, aux=aux,
                    prepared=prepared, inputs_complete=True)

    def commit(self, result):
        self.previous = result['pred'].detach().clone()
        self.absolute_previous = result['absolute'].detach().clone()
        self.lanes.commit_batch(result['batch'])
        self.batches += 1
        self.source_chain = result['metrics']['source_hash_chain']
        self.supervision_chain = result['metrics']['supervision_hash_chain']

    def state_dict(self):
        return dict(lanes=self.lanes.state_dict(),
                    previous=None if self.previous is None else self.previous.detach().cpu(),
                    absolute_previous=None if self.absolute_previous is None else self.absolute_previous.detach().cpu(),
                    batches=self.batches, supervision_reads=self.supervision.reads,
                    source_hash_chain=self.source_chain, supervision_hash_chain=self.supervision_chain)

    def load_state_dict(self, saved):
        self.lanes.load_state_dict(saved['lanes'])
        self.previous = None if saved['previous'] is None else saved['previous'].to(self.device)
        self.absolute_previous = (None if saved['absolute_previous'] is None
                                  else saved['absolute_previous'].to(self.device))
        self.batches = int(saved['batches'])
        self.supervision.reads = int(saved.get('supervision_reads', 0))
        self.source_chain = saved.get('source_hash_chain', '')
        self.supervision_chain = saved.get('supervision_hash_chain', '')


def save_checkpoint(path, runtime, optimizer, *, updates, supervised_outputs,
                    config_path, freeze_path, initial_local_digest):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(schema='xvm_joint_stage1_checkpoint_v1',
                 stage='local_capability', updates=int(updates), supervised_outputs=int(supervised_outputs),
                 local=runtime.local.state_dict(), optimizer=None if optimizer is None else optimizer.state_dict(),
                 runtime=runtime.state_dict(), rng=capture_rng(),
                 config=str(Path(config_path).resolve()), config_sha256=sha(config_path),
                 sourcefreeze=str(Path(freeze_path).resolve()), sourcefreeze_sha256=sha(freeze_path),
                 initial_local_digest=initial_local_digest,
                 local_digest=tensor_digest(runtime.local.state_dict()),
                 source_hash_chain=runtime.source_chain, supervision_hash_chain=runtime.supervision_chain,
                 frozen_absolute_branch=True, no_GT_state_feedback=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)
    return dict(path=str(path), sha256=sha(path), updates=int(updates),
                supervised_outputs=int(supervised_outputs), local_digest=state['local_digest'])


def run(config_path, freeze_path, device, *, engineering=False, receipt_path=None, resume=None):
    from .control import verify
    # Verification occurs before model weight opening or real source readers.
    checked = verify(freeze_path, config_path)
    config_path, freeze_path = Path(config_path).resolve(), Path(freeze_path).resolve()
    config = json.loads(config_path.read_text())
    if not engineering:
        require_real_gpu_engineering(config_path, checked['sourcefreeze_sha256'])
    seed = int(config['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device, device_record = configure_device(device, config, engineering=engineering)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    output = Path(config['out'])
    output.mkdir(parents=True, exist_ok=True)
    if not engineering and not resume and any((output / name).exists() for name in ('manifest.json', 'trace.jsonl', 'last.ckpt')):
        raise FileExistsError('This training output already contains a run; no silent fresh restart')
    runtime = Stage1Runtime(config, device, engineering=engineering)
    initial_digest = tensor_digest(runtime.local.state_dict())
    optimizer = None if engineering else torch.optim.Adam(runtime.local.parameters(), lr=float(config['lr']))
    updates = supervised_outputs = 0
    if resume:
        if engineering:
            raise ValueError('Engineering does not resume training checkpoints')
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if (saved.get('schema') != 'xvm_joint_stage1_checkpoint_v1'
                or saved['config_sha256'] != sha(config_path)
                or saved['sourcefreeze_sha256'] != sha(freeze_path)):
            raise RuntimeError('Checkpoint belongs to another frozen execution identity')
        runtime.local.load_state_dict(saved['local'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        runtime.load_state_dict(saved['runtime'])
        restore_rng(saved['rng'])
        updates, supervised_outputs = int(saved['updates']), int(saved['supervised_outputs'])
        initial_digest = saved['initial_local_digest']
    manifest = dict(schema='xvm_joint_stage1_manifest_v1', started_utc=datetime.now(timezone.utc).isoformat(),
                    pid=os.getpid(), config=str(config_path), config_sha256=sha(config_path),
                    sourcefreeze=str(freeze_path), **checked, device=device_record,
                    stage='local_capability', frozen_absolute_branch=True,
                    base_checkpoint=config['base_checkpoint'], base_checkpoint_sha256=sha(config['base_checkpoint']),
                    base_config=config['base_config'], root_compose='add',
                    local_metadata=runtime.local.metadata(), initial_local_digest=initial_digest,
                    trainable_parameters=sum(p.numel() for p in runtime.local.parameters()),
                    frozen_global_parameters=sum(p.numel() for p in runtime.base.parameters()),
                    data=runtime.lanes.manifest(), setup_input_audit=runtime.setup_input_audit,
                    loss='local joint a/d NLL +0.1 source-weighted validityBCE +1000 terminal21joint MSE m2',
                    GT_read_contract='after all inputs, absolute prediction, candidate geometry and local forward',
                    feedback='corrected previous output; local reference perturbation is not history',
                    optimizer='none' if engineering else 'Adam', precision='float32')
    if not engineering:
        dump(output / ('manifest.json' if not resume else f'resume_{updates:06d}.json'), manifest)
        dump(output / 'status.json', dict(status='RUNNING', pid=os.getpid(), updates=updates,
                                          supervised_outputs=supervised_outputs, sourcefreeze_sha256=sha(freeze_path)))
    start = time.perf_counter()
    engineering_steps = []
    last_checkpoint = None
    try:
        while updates < (1 if engineering else int(config['max_updates'])):
            result = runtime.step()
            loss = result['loss']
            metrics = result['metrics']
            if loss is None:
                runtime.commit(result)
                if engineering:
                    engineering_steps.append(metrics)
                else:
                    metrics.update(update=updates, cumulative_supervised_outputs=supervised_outputs,
                                   optimizer_step=False, warmup=True, wall_s=time.perf_counter() - start)
                    with (output / 'trace.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + '\n')
                continue
            if not bool(torch.isfinite(loss)):
                raise ArithmeticError('Nonfinite training loss')
            if optimizer is None:
                runtime.local.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            sync(device)
            stamp = time.perf_counter()
            loss.backward()
            sync(device)
            metrics['timings']['backward_s'] = time.perf_counter() - stamp
            grads = gradient_report(runtime.local)
            if not grads['finite']:
                raise ArithmeticError('Nonfinite local gradient')
            if any(p.grad is not None for p in runtime.base.parameters()):
                raise AssertionError('Frozen canonical absolute branch received parameter gradients')
            clipped = float(torch.nn.utils.clip_grad_norm_(runtime.local.parameters(), float(config['grad_clip'])))
            if not math.isfinite(clipped):
                raise ArithmeticError('Nonfinite clipped gradient norm')
            stamp = time.perf_counter()
            if optimizer is not None:
                optimizer.step()
            sync(device)
            metrics['timings']['optimizer_s'] = time.perf_counter() - stamp
            updates += 1
            supervised_outputs += metrics['supervised_outputs']
            metrics.update(update=updates, cumulative_supervised_outputs=supervised_outputs,
                           optimizer_step=optimizer is not None, gradient=grads,
                           unclipped_gradient_norm=clipped, wall_s=time.perf_counter() - start)
            runtime.commit(result)
            if engineering:
                engineering_steps.append(metrics)
                if grads['total_norm'] <= 0:
                    raise RuntimeError('Real backward produced no trainable local gradient')
                if metrics['supervised_records'] < 1:
                    raise RuntimeError('Real engineering requires at least one supported supervision record')
                for group in ('mean_head', 'covariance_head'):
                    if grads['groups'].get(group, {}).get('grad_norm', 0.) <= 0:
                        raise RuntimeError('Real backward did not reach required local prediction head: ' + group)
                break
            with (output / 'trace.jsonl').open('a') as stream:
                stream.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
            if updates == 1 or updates % int(config['log_every']) == 0:
                print(json.dumps({'update': updates, 'supervised_outputs': supervised_outputs,
                                  'loss': metrics['loss'], 'grad_norm': grads['total_norm'],
                                  'camera_invalid': metrics['local_camera_invalid'],
                                  'timings': metrics['timings'], 'wall_s': metrics['wall_s']}, ensure_ascii=False), flush=True)
                dump(output / 'status.json', dict(status='RUNNING', pid=os.getpid(), updates=updates,
                                                  supervised_outputs=supervised_outputs,
                                                  last_loss=metrics['loss'], last_gradient_norm=grads['total_norm'],
                                                  last_input_token=metrics['input_token'],
                                                  elapsed_s=metrics['wall_s'], sourcefreeze_sha256=sha(freeze_path)))
            checkpoint_every = int(config['checkpoint_every'])
            initial_every = int(config.get('initial_checkpoint_every', checkpoint_every))
            early_checkpoint = updates <= checkpoint_every and updates % initial_every == 0
            if early_checkpoint or updates % checkpoint_every == 0 or updates == int(config['max_updates']):
                saved_path = output / f'checkpoint_{updates:06d}.ckpt'
                last_checkpoint = save_checkpoint(saved_path, runtime, optimizer, updates=updates,
                    supervised_outputs=supervised_outputs, config_path=config_path,
                    freeze_path=freeze_path, initial_local_digest=initial_digest)
                # Atomic replacement preserves a real complete checkpoint, not
                # merely a filename pointing to a potentially partial writer.
                temporary = output / 'last.ckpt.tmp'
                with saved_path.open('rb') as src, temporary.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                temporary.replace(output / 'last.ckpt')
        if engineering:
            if tensor_digest(runtime.local.state_dict()) != initial_digest:
                raise RuntimeError('Zero-optimizer engineering unexpectedly changed local weights')
            receipt = dict(status='PASS', sourcefreeze_sha256=sha(freeze_path),
                           schema='xvm_joint_real_engineering_v1', manifest=manifest,
                           actual_forward_batches=len(engineering_steps), actual_backward_calls=1,
                           optimizer_steps=0, local_weights_unchanged=tensor_digest(runtime.local.state_dict()) == initial_digest,
                           steps=engineering_steps,
                           initial_zero_tail_note='At exact zero-residual initialization CNN gradient may be zero; prediction heads must receive finite nonzero gradient.',
                           elapsed_s=time.perf_counter() - start)
            destination = (Path(receipt_path) if receipt_path else config_path.parent / 'engineering' /
                           ('real_gpu.json' if device.type == 'cuda' else 'real_cpu.json'))
            if destination.exists():
                raise FileExistsError('Engineering receipt exists; preserve earlier evidence')
            dump(destination, receipt)
            print(json.dumps({'status': receipt['status'], 'receipt': str(destination),
                              'forward_batches': len(engineering_steps), 'backward_calls': 1, 'optimizer_steps': 0}, ensure_ascii=False), flush=True)
            return receipt
        complete = dict(status='COMPLETE', completed_utc=datetime.now(timezone.utc).isoformat(),
                        updates=updates, supervised_outputs=supervised_outputs,
                        sourcefreeze_sha256=sha(freeze_path), checkpoint=last_checkpoint,
                        local_digest=tensor_digest(runtime.local.state_dict()),
                        initial_local_digest=initial_digest, stage='local_capability',
                        no_final_accuracy_claim=True, elapsed_s=time.perf_counter() - start)
        dump(output / 'complete.json', complete)
        dump(output / 'status.json', complete)
        return complete
    except BaseException as exc:
        failure = dict(status='FAILED_PRESERVED', exception=repr(exc), traceback=traceback.format_exc(),
                       updates=updates, supervised_outputs=supervised_outputs,
                       sourcefreeze_sha256=sha(freeze_path), last_checkpoint=last_checkpoint,
                       local_digest=tensor_digest(runtime.local.state_dict()),
                       elapsed_s=time.perf_counter() - start)
        failure_path = ((Path(receipt_path).with_name(Path(receipt_path).stem + '_failure.json')) if receipt_path
                        else config_path.parent / 'engineering/real_failure.json') if engineering else output / 'failure.json'
        dump(failure_path, failure)
        if not engineering:
            dump(output / 'status.json', failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('engineering', 'train'), nargs='?', default='train')
    parser.add_argument('--config', required=True)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', help='Engineering receipt JSON override only')
    parser.add_argument('--resume')
    args = parser.parse_args()
    if args.out and args.command != 'engineering':
        parser.error('--out is reserved for engineering receipts; training output is frozen in config')
    run(args.config, args.source_freeze, args.device, engineering=args.command == 'engineering',
        receipt_path=args.out, resume=args.resume)


if __name__ == '__main__':
    main()
