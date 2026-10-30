"""Real CPU parity of fixed effective gains; no fitted policy is selected."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'model')]
import torch
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_paired_statistics import PairedStatisticsPolicy, XKFPairedStatisticsModel
from tools.xkf_fusion import evaluate_tuned as Q
from tools.xkf_fusion.make_row_incremental import POLICY
from tools.xkf_fusion.bn_probe import parameter_digest


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    Q.RE.setup()
    Q.E.XKFModel = XKFPairedStatisticsModel
    run = ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/raw_s3407'
    model, cfg, identity = Q.E.load_model(run/'config_resolved.yaml', run/'xkf_donly_raw_s3407-step=2400.ckpt', torch.device('cpu'))
    for module in model.modules():
        if isinstance(module, Q.ManoLayer):
            module.fast_chain = True
    parameters = parameter_digest(model)
    buffers = {k: v.detach().cpu().clone() for k, v in model.named_buffers()}
    bank, banksha = Q.R.D.load_bank()
    clips, _ = Q.R.schedule(bank, 10)
    clip = clips[0]
    meta = bank['sequences'][clip['sequence']]
    data = Q.R.D.load_data(cfg, clip['sequence'], meta)
    cache = Q.PacketCache(model, data, torch.device('cpu'))
    initial = Q.R.initial_states(torch.from_numpy(data['gt'][clip['initial_gt_ms']]).float())
    row = meta['rows'][clip['row_indices'][0]]
    effective = MotionCountFusionPolicy(POLICY).gains(model.gains('fused'), torch.tensor([1000]))[0].tolist()
    fixed = dict(POLICY, gain_scale=[1., 1., 1.], fixed_effective_gains=effective)
    outputs = {}
    for label, cls, spec, batch in [('old_batch1', MotionCountFusionPolicy, POLICY, 1),
                                    ('new_batch1', PairedStatisticsPolicy, fixed, 1),
                                    ('new_batch9', PairedStatisticsPolicy, fixed, 9)]:
        model.fusion_policy = cls(spec)
        predictions = []
        for start in range(0, 9, batch):
            packet, cached = cache.packet(row, initial[start:start+batch], [])
            if cached or bool(packet.target.any()):
                raise ValueError('smoke must use actual D with zero inference target')
            Q.R.D.reset(model)
            predictions.append(model.forward_packet(packet))
        outputs[label] = torch.cat(predictions)
    if not torch.equal(outputs['old_batch1'], outputs['new_batch1']):
        raise AssertionError('effective coefficients changed existing full-forward arithmetic')
    if not torch.equal(outputs['old_batch1'], outputs['new_batch9']):
        raise AssertionError('fixed coefficient batch9 differs from actual batch1')
    if parameter_digest(model) != parameters or any(not torch.equal(v.cpu(), buffers[k]) for k, v in model.named_buffers()):
        raise AssertionError('inference mutated model tensors')
    result = dict(status='PASS', identity=identity, bank_sha256=banksha,
                  scope='one actual nonempty packet x9 declared initial conditions; compatibility only, no fit or effectiveness claim',
                  old_new_batch1_bitwise=True, old_batch1_new_batch9_bitwise=True,
                  parameters_and_buffers_unchanged=True, no_direct_cache=True,
                  fixed_effective_gains=effective, policy_activated=False,
                  source_sha256={name: Q.E.sha(ROOT/name) for name in (
                      'semkine/xkf_paired_statistics.py', 'tools/xkf_fusion/smoke_paired_statistics.py')})
    out = ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune/paired_statistics_runtime_smoke.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], output=str(out))))


if __name__ == '__main__':
    main()
