#!/usr/bin/env python3
"""Real CPU integration evidence for the prepared joint entry; never train a run."""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import torch
from config import load_config
from mano_layer import ManoLayer
from tools.xdl.data import counter_rng, packet_hash
from tools.xkf_fusion.motion_data import MotionDataset, arm_config, collate_incremental
from tools.xkf_joint.model import JointFrozenStatisticsModel, ORIGIN_GLOBAL
from tools.xkf_joint.train import parent_audit, all_buffer_hashes, sources
from tools.xkf_joint.release import sha, require
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), 'refusing to overwrite CPU validation')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    source = sources()
    parents, inputs = {}, {}
    for seed in (3407, 3408, 3409):
        config = ROOT / f'configs/xkf_repair/repair_s{seed}.yaml'
        checkpoint = ROOT / f'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune/raw_s{seed}/xkf_ionly_raw_s{seed}-step=2600.ckpt'
        cfg = arm_config(load_config(str(config)), 'raw')
        model = JointFrozenStatisticsModel(cfg)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        restored = parent_audit(model, saved, checkpoint)
        require(len(restored['parameter_order']) == 96 and len(restored['all_buffers_sha256']) == 65,
                'parent coverage differs')
        parents[str(seed)] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                                 global_step=2600, i_adam_step=2400, d_adam_step=2400,
                                 named_Adam_slots=96, strict_model_restore=True)
        inputs.update({str(config): sha(config), str(checkpoint): sha(checkpoint)})
        if seed == 3407:
            step_model, step_saved, step_cfg = model, saved, cfg
    model, saved, cfg = step_model, step_saved, step_cfg
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    dataset = MotionDataset(cfg, start_step=2600, max_steps=2601)
    row = next(i for i in range(20) if counter_rng(3407, 2600, i, 'self').random() < .5)
    batch = collate_incremental([dataset[row]])
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
    require(all(int(optimizer.state[p]['step']) == 2400 for p in model.parameters()), 'CPU restore clock differs')
    before = all_buffer_hashes(model)
    calls = []
    hook = model.direct.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    model._trainer = SimpleNamespace(global_step=ORIGIN_GLOBAL)
    losses = {}
    model.log = lambda name, value, **unused: losses.update({name: float(value.detach().float())})
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model.training_step(batch, 0)
    loss.backward()
    hook.remove()
    gradients = {}
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'missing/nonfinite gradient: ' + name)
        gradients[name] = float(parameter.grad.float().norm())
    require(len(gradients) == 96 and all(any(v > 0 for n, v in gradients.items() if n.startswith('direct.') == direct)
                                      for direct in (False, True)), 'missing active branch gradients')
    require(calls == [False, False, False, True] and model.xdl_self_rows == 1 and model.xdl_burn_rows == 3,
            'actual original full self-burn differs')
    optimizer.step()
    require(all(int(optimizer.state[p]['step']) == 2401 for p in model.parameters()), 'CPU updated Adam clock differs')
    after = all_buffer_hashes(model)
    require(before == after and len(before) == 65, 'a named buffer changed')
    require(not hasattr(model, 'fusion_policy'), 'deployment policy was added to training')
    actual = dict(seed=3407, precision='bf16', all_96_gradients_finite=True,
                  both_branches_nonzero_gradients=True, all_96_Adam_steps_before=2400,
                  all_96_Adam_steps_after=2401, all_65_named_buffers_bitwise_unchanged=True,
                  direct_call_grad_enabled=calls, actual_self_rows=int(model.xdl_self_rows),
                  actual_burn_rows=int(model.xdl_burn_rows), loss_finite=bool(torch.isfinite(loss)),
                  no_deployment_adapter_in_training=True, gradient_norms=gradients, losses=losses,
                  buffer_sha256_before=before, buffer_sha256_after=after,
                  sample_identity=batch['sampling_identity'], main_packet_sha256=packet_hash(batch['main']))
    require(source == sources() and all(sha(p) == digest for p, digest in inputs.items()), 'CPU validation inputs/source changed')
    result = dict(kind='XKF_JOINT_CPU_IMPLEMENTATION_VALIDATION_V1', status='PASS', device='cpu',
                  formal=False, automatic_full_training=False, source_sha256=source, input_sha256=inputs,
                  parents=parents, actual_step=actual,
                  scope='Real single CPU optimizer update discarded in memory; no training checkpoint or quality claim. '
                        'This producer still requires independent source review and the complete CPU test suite.')
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', output=str(output / 'result.json'), trained_run=False)), flush=True)


if __name__ == '__main__':
    main()
