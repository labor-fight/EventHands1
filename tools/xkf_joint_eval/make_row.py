#!/usr/bin/env python3
"""Full2590 stable own-loop row with honest joint-continuation clocks."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split, splits_manifest
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel
from tools import make_s36_row as C
from tools.xdl import make_row as X
from tools.xkf import debug_eval as E, make_row as F
from tools.xkf.runtime import LongWindowEval
from tools.xkf_repair.evaluate import setup
from tools.xkf_fusion.evaluate import sources
from tools.xkf_fusion.bn_probe import parameter_digest
from tools.xkf_joint_eval.checkpoint import validate_joint_endpoint
from tools.xkf_fusion import make_row_centroid as MC

POLICY = dict(gain_scale=[.5, .25, .5], count_gate='ramp1000',
              translation_reference='joint_centroid', motion_count_gate='ramp1000')
SOURCE_FILES = (
    'semkine/xkf_centroid.py', 'semkine/xkf_motion_gate.py',
    'semkine/xkf_stable_geometry.py', 'semkine/xkf_motion_gate_runtime.py',
    'tools/xkf_fusion/incremental_checkpoint.py', 'tools/xkf_fusion/make_row_incremental.py',
    'tools/xkf_fusion/make_row_centroid.py', 'tools/xkf_fusion/bn_probe.py',
    'tools/xkf_fusion/motion_data.py', 'tools/xkf_fusion/finetune_motion.py',
    'tools/xkf_joint/model.py', 'tools/xkf_joint/train.py', 'tools/xkf_joint/release.py', 'tools/xkf_joint/verify.py',
    'tools/xkf_joint_eval/checkpoint.py', 'tools/xkf_joint_eval/make_row.py',
)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True, choices=(3407, 3408, 3409))
    for name in ('config', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    setup()
    torch.set_num_threads(4)
    E.XKFModel = F.XKFModel = XKFStableMotionGateModel
    source = {**sources(), **{name: E.sha(ROOT / name) for name in SOURCE_FILES}}
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(args.config, args.checkpoint, device)
    if identity['seed'] != args.seed:
        raise ValueError('wrong checkpoint seed')
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity['training_clocks'] = validate_joint_endpoint(model, saved, args.checkpoint)
    del saved
    model.configure_fusion(POLICY)
    model.configure_stability('per_sample_model')
    MotionCountFusionPolicy(POLICY)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    parameters = parameter_digest(model)
    buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()}
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    data_root = Path(cfg['DATA']['ROOT'])
    inputs = [args.config, args.checkpoint, Path(splits_manifest(cfg))]
    inputs.extend(args.checkpoint.parent / name for name in ('training_metadata.json', 'train_finish.json'))
    inputs.extend(args.checkpoint.parent / f'{kind}_rank{rank}.json'
                  for kind in ('freeze_audit', 'actual_exposure') for rank in (0, 1))
    hashes = {str(path.resolve()): E.sha(path) for path in inputs}
    protocol = dict(formal=False, scope='complete2590 joint continuation, debug2800 or full6200; not original global6000',
                    seeds=[args.seed], n_seeds=1, exact_optimizer_step=identity['global_step'],
                    parameter_training_clocks=identity['training_clocks'],
                    split='val_core', step_ms=50, window_ms=50, direct_window_ms=200,
                    noise_rng_seed=0, noise_scale=1., tf32=False, policy=POLICY,
                    input_sha256=hashes, source_sha256=source, finite_subset_reduction=False,
                    stability_mode='per_sample_model', calibration_scope='fixed declared policy')
    X.write_json(str(args.output) + '.protocol.json', protocol)
    rng, raw = np.random.default_rng(0), []
    for seq, directory in sequences_for_split(data_root, 'val_core', splits_manifest(cfg)):
        runs = F.runs_of(data_root, directory, seq)
        scored = [(index, a, b) for index, (a, b) in enumerate(runs) if b - a >= 50]
        index, records, run_ids = -1, [], []

        def hook(event, prev, pred):
            nonlocal index
            if event == 'run_start':
                index += 1
                return None
            if not bool(torch.isfinite(pred).all()):
                raise FloatingPointError('nonfinite full sequence prediction')
            records.append(pred.detach().cpu().numpy()[0])
            run_ids.append(scored[index][0])
            return pred

        with LongWindowEval(runs, 200):
            result = ET.track_sequence(model, mano, cfg, data_root, directory, seq, 50,
                                       device, rng, 1., 1., None, window_ms=50, state_hook=hook)
        expected = {'zgz_local': 1204, 'zgz_global': 1386}[seq]
        if (result is None or result['n_frames'] != expected or len(records) != expected
                or index + 1 != len(scored)):
            raise ValueError('incomplete sequence or run assignment')
        np.savez_compressed(str(args.output) + f'.{seq}.npz', predictions=np.stack(records),
                            run_ids=np.asarray(run_ids))
        raw.append(result)
        X.write_json(str(args.output) + '.raw_accuracy.json', dict(identity=identity,
                     protocol=protocol, raw_sequences=raw, complete=len(raw) == 2,
                     finite_subset_reduction=False))
    accuracy = X.accuracy_aggregate(raw)
    accuracy.update(step=identity['global_step'], ckpt=str(args.checkpoint), checkpoint_sha256=identity['checkpoint_sha256'])
    latency, latency_audit = F.latency(model, cfg, device, 200)
    MC.XKFCentroidModel = XKFStableMotionGateModel
    macs, params, mac_audit = MC.profile_policy_macs(
        cfg, model, Path(str(args.output) + '.mac_profile.json'))
    latency_audit['fixed_policy'] = model.fusion_policy.spec
    if parameter_digest(model) != parameters:
        raise RuntimeError('inference changed parameters')
    if any(not torch.equal(value.detach().cpu(), buffers[name]) for name, value in model.named_buffers()):
        raise RuntimeError('inference changed a buffer')
    del model, mano
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    anchor = C.latency_anchor(device)
    if source != {name: E.sha(ROOT / name) for name in source}:
        raise RuntimeError('producer sources changed')
    if hashes != {str(path.resolve()): E.sha(path) for path in inputs}:
        raise RuntimeError('evaluation inputs changed')
    X.write_json(args.output, dict(per_seed={str(args.seed): accuracy},
                 two_seed_mean={key: accuracy[key] for key in ('overall', 'local', 'global')},
                 latency_ms_scaled_full1p75=latency / anchor * C.FULL_PUBLISHED_MS,
                 latency_ms_raw=latency, anchor_full_raw_ms=anchor, latency_context_audit=latency_audit,
                 macs_forward_packet=macs, params_total=params, protocol=protocol,
                 mac_profile_audit=mac_audit, stability_mode='per_sample_model',
                 numeric_status='PASS', finite_subset_reduction=False,
                 provenance=dict(identity, source_sha256=source)))
    print(json.dumps(dict(output=str(args.output), formal=False)), flush=True)


if __name__ == '__main__':
    main()
