#!/usr/bin/env python3
"""Full raw/FK audit for explicit centroid candidates and training-only affine.

Extends the unchanged r3 assessor's cohort/gate definitions. Replays centroid
translation independently with FK and validates all training-only fit rows.
BN experiments retain strict I/hold parity; changed D/fused controls are named
explicitly and never compared as though their BN buffers were unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
from tools.xkf_fusion.assess import Assessor, near, require, sha, write


class CentroidAssessor(Assessor):
    def calibration(self, descriptor, seed, clips):
        if descriptor is None:
            return None
        require(self.digest(descriptor['path']) == descriptor['sha256'], 'training calibration hash mismatch')
        if descriptor['sha256'] in self.calibrations:
            return self.calibrations[descriptor['sha256']]
        path = self.path(descriptor['path']); doc = self.read(path)
        require(doc['seed'] == seed and doc['fit_rows'] == 2304 and doc['alpha'] in (0., .5), 'wrong training affine cohort')
        require(doc['alpha'] == self.expected_alpha, 'affine coefficients and evaluated BN variant differ')
        require(doc['checkpoint_sha256'] == self.parent['arms']['repair'][str(seed)]['training']['checkpoint_sha256'], 'affine fitted from another checkpoint')
        buffer_path = self.study / f'bn/s{seed}/alpha_{doc["alpha"]}/bn_buffers.pt'
        require(self.digest(buffer_path) == doc['bn_buffers_sha256'], 'affine BN buffers changed')
        declaration = self.read(path.parent / 'protocol.json')
        require(self.digest(path.parent / 'protocol.json') == doc['protocol_sha256'], 'training protocol changed')
        require(declaration['partial_smoke'] is False and declaration['ridge'] == .001, 'partial or changed fit')
        require(declaration['identity']['checkpoint_sha256'] == doc['checkpoint_sha256'], 'training checkpoint differs')
        for name, digest in declaration['source_sha256'].items():
            require(self.digest(name) == digest, 'training fit source changed: ' + name)
        manifest = ROOT / 'data/hand_data51/splits_semkine.json'
        require(self.digest(manifest) == declaration['split_manifest_sha256'], 'training manifest changed')
        names = self.read(manifest)['train']['trials']
        require(names == declaration['canonical_train_sequences'] and len(names) == 72, 'noncanonical training sequences')
        rawpath = path.parent / f'train_predictions_alpha{doc["alpha"]}.npz'
        require(self.digest(rawpath) == doc['fit_source_sha256'], 'training raw predictions changed')
        with np.load(rawpath, allow_pickle=False) as raw:
            require(raw['direct_state'].shape == raw['truth'].shape == (2304,51), 'training fit rows incomplete')
            require(set(raw['sequence']) == set(names) and not any(x.startswith('zgz_') for x in raw['sequence']), 'development GT in training fit')
            require(raw['sequence'].tolist() == [r['sequence'] for r in declaration['rows']]
                    and raw['end_ms'].tolist() == [r['end_ms'] for r in declaration['rows']], 'training row identities changed')
            x, residual = raw['direct_state'][:,:3], raw['truth'][:,:3] - raw['direct_state'][:,:3]
            mean, scale = x.mean(0), np.maximum(x.std(0),1e-8)
            z = (x-mean)/scale; bias = residual.mean(0)
            weights = np.linalg.solve(z.T@z + len(x)*.001*np.eye(3), z.T@(residual-bias))
            delta = (weights/scale[:,None]).T
        near(doc['spec']['matrix'],np.eye(3)+delta,'training affine refit matrix',atol=1e-10,rtol=1e-10)
        near(doc['spec']['bias'],bias-delta@mean,'training affine refit bias',atol=1e-10,rtol=1e-10)
        # Normalize only the assessor's in-memory lookup; the artifact is a
        # single shared coefficient set, never an actual clip-specific fit.
        result = dict(doc,clip_specs={c['clip_id']:doc['spec'] for c in clips})
        self.calibrations[descriptor['sha256']] = result
        return result

    def raw_verify(self, raw, old, protocol, clip, clip_index, beta):
        bn = 'bn_calibration_by_seed' in protocol['spec']
        if bn:
            # Preserve original inputs and I/hold parity while acknowledging
            # that applying BN .5 intentionally changes all D-based baselines.
            old = {key: value.copy() for key,value in old.items()}
            for mode in ('d','fused','d_zero','fused_keep_t'):
                index = protocol['modes'].index(mode)
                old['predictions'][:,:,index] = raw['predictions'][:,:,index]
        original = self.fuse_states
        policies = protocol['resolved_clip_policies'][clip['clip_id']]
        cursor = 0
        def independent_replay(a,b,k):
            nonlocal cursor
            policy = policies[cursor];cursor += 1
            fused = original(a,b,k)
            if policy.get('translation_reference') == 'joint_centroid':
                states = self.torch.cat((a,b,fused)).clone();states[:,:3]=0.
                betas = self.torch.from_numpy(beta).float().expand(len(states),-1)
                with self.torch.no_grad():
                    _, joints = self.fk_model._fk(states,betas)
                ci,cd,cf = joints.mean(1).chunk(3)
                fused[:,:3] = (1-k[:,:3])*(a[:,:3]+ci)+k[:,:3]*(b[:,:3]+cd)-cf
                fused = self.torch.where((k==0).all(1,keepdim=True),a,fused)
            else:
                require(policy.get('translation_reference') == 'state_t','undeclared geometry')
            return fused
        self.fuse_states = independent_replay
        try:
            value = super().raw_verify(raw,old,protocol,clip,clip_index,beta)
            require(cursor == len(policies),'candidate fusion replay incomplete')
            return value
        finally:
            self.fuse_states = original

    def seed(self, seed):
        protocol = self.read(self.rollout_root/f's{seed}'/'protocol.json')
        require(protocol['spec']['geometry'] == 'all21_joint_centroid_relative_to_state_translation', 'wrong centroid definition')
        self.expected_alpha = .5 if 'bn_calibration_by_seed' in protocol['spec'] else 0.
        result = super().seed(seed)
        if self.expected_alpha:
            descriptor = protocol['spec']['bn_calibration_by_seed'][str(seed)]
            require(self.digest(descriptor['path']) == descriptor['sha256'] == result['identity']['bn_buffers_sha256'], 'evaluated BN buffer identity mismatch')
        result['raw_verification']['fusion_algebra_replayed'] = 'independent centroid FK or original state-t algebra for every declared candidate'
        result['raw_verification']['bn_control_contract'] = 'BN0: original six; BN.5: original inputs/I/hold, D-based original modes intentionally use new BN'
        for value in result['candidates'].values():
            value['candidate_original_readiness']['known_loop_failure'] = any(
                not x['mean_abs_pass'] or not x['endpoint_wrist_pass'] for x in value['original_candidate_loop_checks'].values())
        return result

    def run(self):
        result = super().run()
        result['kind'] = 'XKF_CENTROID_RAW_ASSESSMENT'
        result['centroid_assessor_sha256'] = sha(__file__)
        result['calibration_scope'] = 'one shared affine from canonical 9-person training set; no development calibration'
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=CentroidAssessor(ROOT/'outputs/xkf/fusion_20261015_r3',args.rollout_dir.resolve(),True).run()
    write(args.output,result)
    import json
    print(json.dumps({'status':result['status'],'errors':result['validation_errors'],'missing':result['missing'],
                      'candidates':result['all_three_seed_candidates']}),flush=True)
    return 2 if result['status']=='INVALID' else 0


if __name__ == '__main__':
    raise SystemExit(main())
