#!/usr/bin/env python3
"""Independent raw audit for new I/whole-update count gates, preserving old gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from tools.xkf_fusion.assess import Assessor,require,write,sha
from tools.xkf_fusion import assess_tuned_centroid as T

SPEC=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/motion_gate/spec.json'

def validate_motion_spec(spec,declaration):
    require(spec==declaration,'motion-gate spec differs from its own immutable declaration')
    require(spec['formal'] is False and spec['checkpoint_step']==2400 and spec['arm']=='raw','wrong motion-gate endpoint')
    require(spec['geometry']=='all21_joint_centroid_relative_to_state_translation','wrong motion-gate geometry')
    common={'gain_scale':[.5,.25,.5],'translation_reference':'joint_centroid'}
    expected=[dict(common,id='prior_centroid',count_gate='ramp1000'),
              dict(common,id='motion_ramp_centroid',count_gate='ramp1000',motion_count_gate='ramp1000'),
              dict(common,id='whole_ramp_centroid',count_gate=None,update_count_gate='ramp1000')]
    require(spec['candidates']==expected and 'bn_calibration_by_seed' not in spec,'motion-gate candidate set changed')

def independent_motion(previous,proposal,gate):
    """Float64 scipy geodesic replay; no production interpolation helper."""
    p,q=previous.numpy(),proposal.numpy();g=np.asarray(gate,dtype=np.float64).reshape(-1)
    require(len(g)==len(p) and np.isfinite(g).all() and ((g>=0)&(g<=1)).all(),'invalid independent motion gate')
    result=p.astype(np.float64)+g[:,None]*(q-p).astype(np.float64)
    rp=Rotation.from_rotvec(p[:,3:6]);rq=Rotation.from_rotvec(q[:,3:6])
    roots=(rp*Rotation.from_rotvec(g[:,None]*(rp.inv()*rq).as_rotvec())).as_rotvec()
    norm=np.maximum(np.linalg.norm(roots,axis=1,keepdims=True),1e-9)
    alternate=roots*(1-2*np.pi/norm)
    use=(np.linalg.norm(alternate-p[:,3:6],axis=1)<np.linalg.norm(roots-p[:,3:6],axis=1)) & (norm[:,0]>1e-6)
    result[:,3:6]=np.where(use[:,None],alternate,roots)
    result=np.where((g==0)[:,None],p,np.where((g==1)[:,None],q,result)).astype(np.float32)
    return previous.new_tensor(result)

class MotionGateAssessor(T.TunedCentroidAssessor):
    def endpoint_identity(self,identity,protocol,summary,seed):
        # All exact2400/Adam/I/BN/EMA audits remain; only the declaration
        # validator is substituted inside this single-threaded analysis.
        with patch.object(T,'DECLARATION',SPEC),patch.object(T,'validate_spec',validate_motion_spec):
            return super().endpoint_identity(identity,protocol,summary,seed)

    def raw_verify(self,raw,old,protocol,clip,clip_index,beta):
        require('bn_calibration_by_seed' not in protocol['spec'],'no BN intervention allowed')
        old={k:v.copy() for k,v in old.items()}
        for mode in ('d','fused','d_zero','fused_keep_t'):
            mi=protocol['modes'].index(mode);old['predictions'][:,:,mi]=raw['predictions'][:,:,mi]
        m=len(protocol['modes']);initial=raw['initial']
        previous=np.concatenate([np.broadcast_to(initial[:,None],(9,m,51))[None],raw['predictions'][:-1]],0)
        policies=protocol['resolved_clip_policies'][clip['clip_id']];cursor=0;original=self.fuse_states
        gate=np.repeat(np.clip(raw['n_raw_events']/1000.,0,1),9)
        def replay(a,b,k):
            nonlocal cursor
            policy=policies[cursor];cursor+=1
            incoming=self.torch.from_numpy(previous[:,:,protocol['modes'].index(policy['id'])].reshape(-1,51)).float()
            motion=independent_motion(incoming,a,gate) if policy.get('motion_count_gate') else a
            fused=original(motion,b,k)
            states=self.torch.cat([motion,b,fused]).clone();states[:,:3]=0.
            betas=self.torch.from_numpy(beta).float().expand(len(states),-1)
            with self.torch.no_grad():_,joints=self.fk_model._fk(states,betas)
            ci,cd,cf=joints.mean(1).chunk(3)
            fused[:,:3]=(1-k[:,:3])*(motion[:,:3]+ci)+k[:,:3]*(b[:,:3]+cd)-cf
            fused=self.torch.where((k==0).all(1,keepdim=True),motion,fused)
            if policy.get('update_count_gate'):
                require(policy.get('count_gate') is None,'whole update double-gates D')
                fused=independent_motion(incoming,fused,gate)
            return fused
        self.fuse_states=replay
        try:
            value=Assessor.raw_verify(self,raw,old,protocol,clip,clip_index,beta)
            require(cursor==3,'all motion-gate candidates must be independently replayed')
        finally:self.fuse_states=original
        priorpath=ROOT/f'outputs/xkf/fusion_20261015_r3/direct_finetune/evaluations/raw/s{protocol["seed"]}/raw/{clip_index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
        require(self.digest(priorpath)==self.prior_assessment['input_sha256'][str(priorpath.resolve())],'prior centroid raw changed')
        with np.load(priorpath) as z:
            for key in ('initial','truth','clean','packet_ids','end_ms','lnes_window_ms','n_raw_events'):
                require(np.array_equal(raw[key],z[key]),'new motion-gate input differs from paired raw2400 source')
            prior=z['predictions'][:,:,self.prior_mode_index].copy()
        current=raw['predictions'][:,:,protocol['modes'].index('prior_centroid')]
        self.prior_parity.append({'clip_id':clip['clip_id'],'bitwise_equal':bool(np.array_equal(current,prior)),
            'translation_max_abs_mm':float(np.abs(current[...,:3]-prior[...,:3]).max()*1000),
            'pose_aa_max_abs':float(np.abs(current[...,3:]-prior[...,3:]).max())})
        return value

    def seed(self,seed):
        self.prior_assessment=self.read(ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/assessment_raw.json')
        require(self.prior_assessment['status']=='ASSESSED' and not self.prior_assessment['validation_errors'],'prior assessment invalid')
        summary=self.read(ROOT/f'outputs/xkf/fusion_20261015_r3/direct_finetune/evaluations/raw/s{seed}/summary.json')
        self.prior_mode_index=summary['modes'].index('uncalibrated_centroid_t05_r025_f05_ramp');self.prior_parity=[]
        result=super().seed(seed)
        require(result['identity']==summary['identity'],'motion gate changed checkpoint/config identity')
        result['prior_centroid_pairing']={'all_bitwise_equal':all(v['bitwise_equal'] for v in self.prior_parity),'per_clip':self.prior_parity,
            'scope':'same policy/data, recomputed own-loop; batch-context discrepancy retained and never used to drop clips'}
        result['raw_verification']['fusion_algebra_replayed']='independent scipy SO3 interpolation, CPU centroid FK and inner fusion, then optional independent whole-update interpolation'
        result['raw_verification']['effective_gains_semantics']='whole_ramp stores inner ungated gains; final update gate independently replayed from W50 count'
        return result

    def run(self):
        result=super().run();result['kind']='XKF_D2400_MOTION_GATE_RAW_ASSESSMENT';result['motion_gate_assessor_sha256']=sha(__file__)
        result['calibration_scope']='none; all three declared motion-gate candidates use unchanged raw D'
        for row in result['seeds'].values():
            if row is not None:
                row['raw_verification']['fusion_algebra_replayed']='independent scipy SO3 motion interpolation, centroid FK, inner fusion and optional whole-update interpolation'
        return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rollout-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=MotionGateAssessor(ROOT/'outputs/xkf/fusion_20261015_r3',args.rollout_dir.resolve(),True).run();write(args.output,result)
    print(json.dumps({'status':result['status'],'errors':result['validation_errors'],'missing':result['missing'],'candidates':result['all_three_seed_candidates']}),flush=True)
    return 2 if result['status']=='INVALID' else 0

if __name__=='__main__':raise SystemExit(main())
