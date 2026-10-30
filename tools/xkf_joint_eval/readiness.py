#!/usr/bin/env python3
"""Original readiness formulas, real initial weights, actual repaired fusion.

Only the fused inference output is replaced by the declared stable deployment
forward. Original cohort selection, thresholds, gradient capability test and
current-stage self/burn exposure checks remain unchanged.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import torch
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel
from tools.xkf import debug_eval as E
from tools.xkf_repair import evaluate as RE
from tools.xkf_joint_eval.checkpoint import validate_joint_endpoint
from tools.xkf_joint_eval.make_row import POLICY, SOURCE_FILES
from tools.xkf_fusion.evaluate import sources
from tools.xdl.make_row import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    RE.setup()
    torch.set_num_threads(4)
    E.XKFModel = XKFStableMotionGateModel
    source = {**sources(), **{name: E.sha(ROOT / name) for name in SOURCE_FILES},
              'tools/xkf_joint_eval/readiness.py': E.sha(__file__)}
    model, cfg, identity = E.load_model(args.config, args.checkpoint, torch.device('cpu'))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    clock = validate_joint_endpoint(model, saved, args.checkpoint)
    del model, saved
    initial = args.checkpoint.parent / 'initial_state.ckpt'
    metadata = json.loads((args.checkpoint.parent / 'training_metadata.json').read_text())
    if E.sha(initial) != metadata['initial_checkpoint_sha256']:
        raise ValueError('actual original initial weights differ')
    inputs = {str(path.resolve()): E.sha(path) for path in (args.config, args.checkpoint, initial)}
    original_load, original_outputs, original_gradient = E.load_model, E.outputs, E.gradient_audit

    def load(config, checkpoint, device):
        model, cfg, identity = original_load(config, checkpoint, device)
        model.configure_fusion(POLICY)
        model.configure_stability('per_sample_model')
        return model, cfg, identity

    def outputs(model, packet, x_i, z_d):
        value = original_outputs(model, packet, x_i, z_d)
        if bool(packet.target.any()):
            raise ValueError('fused inference must receive zero target')
        value['fused'] = model.forward_packet(packet)
        return value

    def gradient(model, batch):
        # This original backward capability check uses a disposable model after
        # every inference measurement. The canonical inference mode is not a
        # training algorithm; use the exact original differentiable I path here.
        model.configure_stability('off')
        try:
            return original_gradient(model, batch)
        finally:
            model.configure_stability('per_sample_model')

    E.load_model, E.outputs, E.gradient_audit = load, outputs, gradient
    argv = sys.argv
    for checkpoint, output, compare in (
        (initial, args.output_dir / 'initial.json', None),
        (args.checkpoint, args.output_dir / 'current.json', args.output_dir / 'initial.json'),
    ):
        sys.argv = [argv[0], '--config', str(args.config), '--checkpoint', str(checkpoint),
                    '--output', str(output), '--device', 'cpu', '--cpu-threads', '4',
                    '--train-batches', '2', '--batch-size', '4', '--train-start-index', '0',
                    '--check-rows', '8', '--clips', '1', '--steps', '10']
        if compare is not None:
            sys.argv += ['--compare', str(compare), '--gradients']
        E.main()
    sys.argv = argv
    current = json.loads((args.output_dir / 'current.json').read_text())
    baseline = json.loads((args.output_dir / 'initial.json').read_text())
    if current['readiness'] != E.readiness(current, baseline):
        raise ValueError('original readiness formula differs')
    if source != {name: E.sha(ROOT / name) for name in source}:
        raise ValueError('source changed during readiness')
    if any(E.sha(path) != digest for path, digest in inputs.items()):
        raise ValueError('weights or config changed during readiness')
    write_json(args.output_dir / 'result.json', dict(
        kind='XKF_JOINT_ORIGINAL_READINESS_ACTUAL_FUSION', formal=False,
        automatic_full_training=False, identity=identity, parameter_training_clocks=clock,
        policy=POLICY, source_sha256=source, input_sha256=inputs,
        readiness=current['readiness'], original_dual_branch_backward_capability=current['gradients'],
        current_stage_exposure=current['training_identity'],
        artifacts={name: dict(path=str(args.output_dir / name), sha256=E.sha(args.output_dir / name))
                   for name in ('initial.json', 'current.json')},
        scope='All original readiness formulas and fixed cohorts; actual policy forward only replaces fused inference. '
              'Joint update clocks and active branches require the separate actual training audit.'))


if __name__ == '__main__':
    main()
