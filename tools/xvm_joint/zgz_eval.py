"""Fixed-final XJ evaluation through the original zgz recursive loop.

This is a new, explicitly authorised evaluation entry point.  It preserves
the original split order, rng0 initial perturbations and scored 50ms schedule.
The original training and SELECT evaluation modules remain frozen/read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from .control import verify as verify_training
from .evaluation_geometry import FastJointGeometry
from .geometry import GeometryConfig
from .model import JointLocalOperator
from .train import configure_device, dump, sha, tensor_digest
from .zgz_model import XJPacketModel


ROOT = Path(__file__).resolve().parents[2]
METRICS = ('mpjpe_ra_mm', 'mpvpe_ra_mm', 'mpjpe_abs_mm', 'mpvpe_abs_mm', 'root_abs_mm')


def entry_path(entry):
    return Path(entry['path'] if isinstance(entry, dict) else entry).resolve()


def verify_entry(entry, name):
    if not isinstance(entry, dict) or 'path' not in entry or 'sha256' not in entry:
        raise RuntimeError('A SHA-bound identity is required for '+name)
    if sha(entry['path']) != entry['sha256']:
        raise RuntimeError('Changed identity: '+name)


def verify_identity(protocol_path, sourcefreeze_path):
    protocol_path, sourcefreeze_path = Path(protocol_path).resolve(), Path(sourcefreeze_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    frozen = json.loads(sourcefreeze_path.read_text())
    if protocol.get('schema') != 'xj_zgz_eval_protocol_v1' or frozen.get('schema') != 'xj_zgz_eval_sourcefreeze_v1':
        raise RuntimeError('Unexpected fixed zgz protocol/freeze schema')
    if entry_path(frozen['protocol']) != protocol_path:
        raise RuntimeError('Wrong evaluation protocol path')
    verify_entry(frozen['protocol'], 'zgz protocol')
    for name, digest in frozen['files'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT/name
        if sha(path) != digest:
            raise RuntimeError('Changed evaluation executable: '+name)
    for name, entry in frozen['identities'].items():
        verify_entry(entry, name)
    for name in ('config', 'checkpoint', 'training_sourcefreeze', 'base_checkpoint', 'base_config'):
        verify_entry(protocol[name], name)
    verify_training(entry_path(protocol['training_sourcefreeze']), entry_path(protocol['config']))
    if protocol.get('subjects') != ['zgz'] or int(protocol.get('init_rng', -1)) != 0:
        raise RuntimeError('Only the declared zgz/rng0 formal evaluation is allowed')
    cfg = json.loads(entry_path(protocol['config']).read_text())
    if (entry_path(protocol['base_checkpoint']) != Path(cfg['base_checkpoint']).resolve()
            or entry_path(protocol['base_config']) != Path(cfg['base_config']).resolve()):
        raise RuntimeError('Canonical S43 identity differs from completed training')
    return protocol, frozen, cfg


def official_evalx():
    name = '_xj_fixed_zgz_original_evalx'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT/'tools/tracking/evalx.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def make_model(protocol, arm='primary', mode='w50', device='cpu'):
    """Construct exactly the fixed model; does not open a recording or label."""
    if arm not in ('primary', 'global') or mode not in ('w50', 's39'):
        raise ValueError('Declared primary/global and w50/s39 arms only')
    config = json.loads(entry_path(protocol['config']).read_text())
    from tools.s43_integration.model import load_legacy
    base, cfg = load_legacy(config['base_checkpoint'], config['base_config'], device)
    base = base.float().eval()
    base.render_fp32 = True
    local = None
    if arm == 'primary':
        verify_entry(protocol['checkpoint'], 'fixed final local checkpoint')
        checkpoint = torch.load(entry_path(protocol['checkpoint']), map_location='cpu', weights_only=False)
        complete = json.loads((entry_path(protocol['config']).parent/'train/complete.json').read_text())
        if (int(checkpoint.get('updates', -1)) != 12000
                or checkpoint.get('sourcefreeze_sha256') != protocol['training_sourcefreeze']['sha256']
                or complete.get('status') != 'COMPLETE'
                or complete['checkpoint']['sha256'] != protocol['checkpoint']['sha256']):
            raise RuntimeError('Only the completed fixed12000 checkpoint is allowed')
        local = JointLocalOperator().to(device).float().eval()
        local.load_state_dict(checkpoint['local'], strict=True)
        if tensor_digest(local.state_dict()) != complete['local_digest']:
            raise RuntimeError('Local tensor identity differs from completed training')
    gc = GeometryConfig(max_supports=int(config['num_supports']),
                        candidates_per_support=int(config['candidates_per_support']),
                        radius_m=float(config['radius_m']))
    model = XJPacketModel(base, local, FastJointGeometry(base, gc), variant=arm).to(device).eval()
    if mode == 's39':
        from .zgz_filter import S39PacketFilter
        source = protocol['filter_spec']
        if isinstance(source, dict) and 'path' in source:
            verify_entry(source, 'fixed S39 filter spec')
            source = json.loads(entry_path(source).read_text())
        elif isinstance(source, str):
            source = json.loads(Path(source).read_text())
        model = S39PacketFilter(model, source).to(device).eval()
    return model, cfg, base


def sequence_inventory(data_root):
    """Metadata only, original val_core order and original legacy directories."""
    from semkine.dataset import sequences_for_split
    root = Path(data_root)
    return sequences_for_split(root, 'val_core', root/'splits_semkine.json')


def valid_runs(data_root, legacy_dir, sequence):
    with np.load(Path(data_root)/legacy_dir/(sequence+'_aux.npz'), allow_pickle=True) as aux:
        return np.asarray(aux['valid_runs_ms'], np.int64).reshape(-1, 2).copy()


def initial_rng_for_sequence(cfg, data_root, inventory, sequence, *, sample_noise=None):
    """Consume the original prefix's *metadata-counted* initial noise draws.

    No prior sequence pose metadata or betas are opened.  In the original loop
    a valid run shorter than one complete50ms packet consumes no noise draw.
    """
    if sample_noise is None:
        sample_noise = official_evalx().ET.sample_init_noise
    rng = np.random.default_rng(0)
    consumed = 0
    for name, directory in inventory:
        if name == sequence:
            return rng, consumed
        for start, stop in valid_runs(data_root, directory, name):
            if int(stop)-int(start) >= 50:
                sample_noise(cfg, rng, 1.)
                consumed += 1
    raise ValueError('Sequence absent from original inventory')


def packet_input_token(batch):
    """Bind only legal forward input bytes; target/actual beta are not read."""
    digest = hashlib.sha256()
    for name in ('events', 'ptr', 't_start_us', 't_end_us', 'prev_state', 'camera_K'):
        value = getattr(batch, name).detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


class AuditedPacketModel(nn.Module):
    encoder_name = 'xj_fixed_zgz_audited_packet'
    xyz_assoc_prev = 'self'

    def __init__(self, core, trace_path, sequence):
        super().__init__()
        self.core = core
        self.trace_path, self.sequence = Path(trace_path), sequence
        self.frames = 0
        self.run_id = -1
        self.source_chain = ''
        self.started = time.perf_counter()

    def reset_state(self, previous):
        self.run_id += 1
        self.core.reset_state(previous)

    def set_hand_context(self, betas, camera_K):
        # The original evaluator supplies actual beta here for old models.
        # New prediction code only uses its per-packet K and explicit beta0.
        return None

    @torch.no_grad()
    def forward_packet(self, batch):
        from tools.s43_rel.access_guard import InferenceAccessGuard
        with InferenceAccessGuard() as guard:
            token = packet_input_token(batch)
            prediction = self.core.forward_packet(batch)
        self.source_chain = hashlib.sha256(
            (bytes.fromhex(self.source_chain) if self.source_chain else b'')+token.encode()).hexdigest()
        row = dict(frame=self.frames, sequence=self.sequence, run_id=self.run_id,
                   input_token=token, source_hash_chain=self.source_chain,
                   prediction_digest=tensor_digest({'prediction': prediction}),
                   packet_audit=dict(self.core.last_audit), input_audit=guard.receipt())
        with self.trace_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.frames += 1
        if self.frames % 100 == 0:
            print(json.dumps(dict(sequence=self.sequence, frames=self.frames,
                                  elapsed_s=time.perf_counter()-self.started)), flush=True)
        return prediction


@torch.no_grad()
def metric_arrays(base, predictions, targets, beta_target, *, device='cpu', chunk=256):
    """Prediction FK uses beta0; GT FK uses its actual score-only hand shape."""
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 51:
        raise ValueError('Matching prediction and target physical51 arrays required')
    beta_target = np.asarray(beta_target, np.float32).reshape(1, 10)
    result = {key: [] for key in METRICS}
    for first in range(0, len(predictions), chunk):
        p = torch.from_numpy(predictions[first:first+chunk].astype(np.float32)).to(device)
        g = torch.from_numpy(targets[first:first+chunk].astype(np.float32)).to(device)
        beta0 = p.new_zeros((len(p), 10))
        beta = torch.from_numpy(beta_target.copy()).to(device).expand(len(p), -1)
        pv, pj = base._fk(p, beta0)
        gv, gj = base._fk(g, beta)
        means = lambda v: v.norm(dim=-1).mean(-1)*1000.
        values = dict(mpjpe_ra_mm=means((pj-pj[:, :1])-(gj-gj[:, :1])),
                      mpvpe_ra_mm=means((pv-pv[:, :1])-(gv-gv[:, :1])),
                      mpjpe_abs_mm=means(pj-gj), mpvpe_abs_mm=means(pv-gv),
                      root_abs_mm=(pj[:, 0]-gj[:, 0]).norm(dim=-1)*1000.)
        for key, value in values.items():
            if not bool(torch.isfinite(value).all()):
                raise ArithmeticError('Nonfinite metric; do not drop this endpoint')
            result[key].append(value.cpu().numpy().astype(np.float64))
    return {key: np.concatenate(values) for key, values in result.items()}


def evaluate_worker(protocol_path, sourcefreeze_path, *, arm, mode, sequence, device, out):
    protocol, frozen, config = verify_identity(protocol_path, sourcefreeze_path)
    if arm not in ('primary', 'global') or mode not in ('w50', 's39') or sequence not in protocol['sequences']:
        raise ValueError('Undeclared evaluation worker')
    output = Path(out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('Worker evidence exists; no hidden repeated zgz evaluation')
    random.seed(int(protocol['seed']));np.random.seed(int(protocol['seed']));torch.manual_seed(int(protocol['seed']))
    torch.set_num_threads(int(frozen.get('execution', {}).get('cpu_threads', 2)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    device, device_info = configure_device(device, config, engineering=True)
    if device.type != 'cuda' or device_info['physical_gpu'] not in (2, 3):
        raise RuntimeError('This zgz evaluation uses only physical GPU2/3')
    evalx = official_evalx()
    inventory = sequence_inventory(protocol['data_root'])
    if [name for name, _ in inventory] != protocol['sequences']:
        raise RuntimeError('Original val_core inventory/order differs from the frozen protocol')
    mapping = dict(inventory)
    runs = valid_runs(protocol['data_root'], mapping[sequence], sequence)
    expected = sum(max(0, (int(stop)-int(start))//50) for start, stop in runs)
    if expected != int(protocol['expected_frames'][sequence]):
        raise RuntimeError('Metadata scored-frame inventory differs from frozen declaration')
    model, cfg, base = make_model(protocol, arm, mode, device)
    rng, prefix_noise_draws = initial_rng_for_sequence(cfg, protocol['data_root'], inventory, sequence)
    audited = AuditedPacketModel(model, output/'trace.jsonl', sequence)
    before = tensor_digest(model.state_dict())
    binding = dict(schema='xj_zgz_worker_binding_v1', arm=arm, mode=mode, sequence=sequence,
                   protocol_sha256=sha(protocol_path), sourcefreeze_sha256=sha(sourcefreeze_path),
                   checkpoint_sha256=protocol['checkpoint']['sha256'], device=device_info,
                   expected_frames=expected, init_rng=0, earlier_sequence_run_noise_draws=prefix_noise_draws,
                   initialization='original evalx GT51 plus cfg TRACK noise once per valid run; then own feedback',
                   beta_contract='prediction beta0, metrics GT beta_target; input actual beta never read',
                   first_window_scored=True, local_first_query_exposure=False,
                   process_id=os.getpid(), initial_model_digest=before)
    dump(output/'binding.json', binding)
    started = time.perf_counter()
    try:
        raw = evalx.run_sequence(audited, cfg, Path(protocol['data_root']), mapping[sequence], sequence,
                                device, rng, mode='model', wmode='fixed', win=50 if mode=='w50' else 200,
                                clip_run_start=mode=='s39', count_mode='last50')
        if len(raw['pred']) != expected or audited.frames != expected:
            raise RuntimeError('Incomplete scored-frame coverage; preserve failure')
        if tensor_digest(model.state_dict()) != before:
            raise RuntimeError('Evaluation changed frozen model state')
        metrics = metric_arrays(base, raw['pred'], raw['gt'], raw['betas'], device=device)
        artifact = {key: np.asarray(value) for key, value in raw.items() if key != 'fstate'}
        artifact.update({'metric_'+key: value for key, value in metrics.items()})
        artifact['beta_input'] = np.zeros(10, np.float32)
        np.savez_compressed(output/'frames.npz', **artifact)
        summary = dict(status='COMPLETE', arm=arm, mode=mode, sequence=sequence,
                       eval_protocol_sha256=sha(protocol_path), eval_sourcefreeze_sha256=sha(sourcefreeze_path),
                       binding=binding, n_frames=expected,
                       metrics={key: float(value.mean()) for key, value in metrics.items()},
                       frames_sha256=sha(output/'frames.npz'), trace_sha256=sha(output/'trace.jsonl'),
                       final_model_digest=tensor_digest(model.state_dict()), source_hash_chain=audited.source_chain,
                       elapsed_s=time.perf_counter()-started, no_warmup_exclusion=True,
                       no_confidence_filtering=True, scope='single-seed fixed-final zgz evaluation, no selection/training')
        dump(output/'summary.json', summary)
        return summary
    except BaseException as exc:
        dump(output/'failure.json', dict(status='FAILED_PRESERVED', error=repr(exc),
             traceback=traceback.format_exc(), frames_completed=audited.frames, source_hash_chain=audited.source_chain,
             elapsed_s=time.perf_counter()-started))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'source-freeze', 'arm', 'mode', 'sequence', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    result = evaluate_worker(args.protocol, args.source_freeze, arm=args.arm, mode=args.mode,
                             sequence=args.sequence, device=args.device, out=args.out)
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
