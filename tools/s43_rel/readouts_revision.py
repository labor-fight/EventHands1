"""R1 proposal: invariant, fixed-MANO-basis prototype readout adaptation.

This changes ONLY the E12 diagnostic reader. It never changes the sensor core,
geometry, S43 production adapter, observations, labels, history or optimizers.
Run arithmetic checks with --self-check. Task scores require an explicit R1
correction preregistration path; the original negative results remain intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import mechanisms as M
from .readouts import load_head

SENSOR_D = 288
F_D = 51 * 24
POINTS = 128
A_D = POINTS * 51
LAMBDA_D = POINTS * 24
OLD_D = SENSOR_D + F_D + A_D + LAMBDA_D
NEW_D = SENSOR_D + F_D * 2
R1_ARMS = ('full', 'no_order', 'tore4')
SCALE_FLOOR = 1.0
CLIP = 8.0


def joint_diagonal_support(a, lam):
    """diagG[j,h] = sum_l lam[l,h] * a[l,j]^2 in fixed physical MANO axes.

    Each nonnegative column is sorted before float64 reduction, so a point
    permutation yields bitwise-identical output, including reduction order.
    This is a support feature, never an inverse Gram or uncertainty estimate.
    """
    a = np.asarray(a, np.float64)
    lam = np.asarray(lam, np.float64)
    if a.ndim != 2 or lam.ndim != 2 or len(a) != len(lam):
        raise ValueError('a[N,J], lambda[N,H] with common N required')
    if not np.isfinite(a).all() or not np.isfinite(lam).all() or np.any(lam < 0):
        raise ValueError('finite a and nonnegative finite lambda required')
    with np.errstate(over='ignore', invalid='ignore'):
        terms = lam[:, None, :] * np.square(a[:, :, None])
    if not np.isfinite(terms).all():
        raise ValueError('joint support product overflow')
    # Canonical summation is an implementation invariant, not a statistical
    # intervention or a change of the summand values.
    result = np.sort(terms, axis=0).sum(axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError('joint support sum overflow')
    return result


def project_typed(f, a, lam):
    """Direct-Z f plus supported joint-basis diagonal, no bare static a slots."""
    f = np.asarray(f, np.float64)
    a, lam = np.asarray(a, np.float64), np.asarray(lam, np.float64)
    if f.ndim != 2 or a.shape[1:] != (f.shape[0],) or lam.shape[1:] != (f.shape[1],):
        raise ValueError('f[J,H], a[N,J], lambda[N,H] required')
    if not np.isfinite(f).all():
        raise ValueError('nonfinite direct-Z f')
    diagonal = joint_diagonal_support(a, lam)
    live = np.any(lam > 0, axis=0)
    # A channel with no support remains exactly zero even for a malformed
    # nonzero f. For valid direct-Z packages this gate is already an identity.
    f = np.where(live[None], f, 0.0)
    return np.concatenate((M.signedlog(f), np.log1p(diagonal).ravel())).astype(np.float32)


def project_cached(features):
    """Convert the unchanged E12 cache; cached direct signedlog(f) stays exact.

    Original a/lambda were stored after float32 log transforms. Their inverse
    is deterministic but cannot recover the pre-cache float64 values exactly;
    this finite-storage approximation is disclosed and shared by all R1 arms.
    No original event, target, history or source selection is rebuilt.
    """
    x = np.asarray(features)
    if x.ndim != 2 or x.shape[1] != OLD_D or not np.isfinite(x).all():
        raise ValueError(f'finite original E12 cache [B,{OLD_D}] required')
    out = np.empty((len(x), NEW_D), np.float32)
    out[:, :SENSOR_D+F_D] = x[:, :SENSOR_D+F_D]
    for i in range(len(x)):
        logs_a = x[i, SENSOR_D+F_D:SENSOR_D+F_D+A_D].astype(np.float64).reshape(POINTS,51)
        logs_lam = x[i, SENSOR_D+F_D+A_D:].astype(np.float64).reshape(POINTS,24)
        if np.any(logs_lam < 0):
            raise ValueError('negative cached log1p(lambda)')
        a = np.sign(logs_a)*np.expm1(np.abs(logs_a))
        lam = np.expm1(logs_lam)
        diagonal = joint_diagonal_support(a,lam)
        out[i, SENSOR_D+F_D:] = np.log1p(diagonal).ravel()
        live = np.any(lam > 0,axis=0)
        ff = out[i,SENSOR_D:SENSOR_D+F_D].reshape(51,24)
        ff[:,~live] = 0.
    return out


class RevisionNormalize:
    """Zero-preserving physical log/RMS normalization in fixed semantic blocks.

    There is no per-contour-slot standard deviation or subtraction of a static
    mean. The FIT-only RMS has a physical log-space floor of one. Clip at eight
    before fixed dimension balancing. Baseline LN/enhanced readers are untouched.
    Context normalization is exactly the original E12 context rule.
    """
    def fit(self,x,c):
        x=np.asarray(x,np.float64)
        if x.shape[1] != NEW_D:raise ValueError('R1 fixed feature width mismatch')
        self.scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),SCALE_FLOOR)
        self.blocks=((0,SENSOR_D),(SENSOR_D,SENSOR_D+F_D),(SENSOR_D+F_D,NEW_D))
        self.cm=c.mean(0);self.cs=c.std(0)
        self.cscale=np.sqrt(max(1,int(np.sum(self.cs>=1e-6))))
        self.cs[self.cs<1e-6]=1.
        return self
    def transform(self,x,c):
        result=np.clip(np.asarray(x)/self.scale,-CLIP,CLIP).astype(np.float32)
        for start,stop in self.blocks:
            result[:,start:stop] /= np.sqrt((stop-start)*len(self.blocks))
        context=(np.clip((c-self.cm)/self.cs,-8.,8.)/self.cscale).astype(np.float32)
        return result,context


class RevisionHead(M.RegressionHead):
    """Same RBF and JL512/128x2 MLP optimization budget; new normalization only."""
    def fit(self,x,c,y):
        import torch
        self.normal=RevisionNormalize().fit(x,c)
        xx,cc=self.normal.transform(x,c)
        self.curves=[]
        if self.kind=='rbf':
            self.train=np.concatenate((xx,cc),1).astype(np.float64)
            d=M.dist(self.train,self.train);v=d[np.triu_indices(len(d),1)]
            self.bandwidth=max(float(np.median(v[v>1e-12])) if np.any(v>1e-12) else 1.,1e-6)
            self.coef=np.linalg.solve(np.exp(-d/self.bandwidth)+.01*np.eye(len(d)),y)
            self.params=int(self.coef.size)
            return self
        rng=np.random.default_rng(M.SEED)
        self.projection=(rng.standard_normal((x.shape[1],512))/np.sqrt(512)).astype(np.float32)
        inputs=np.concatenate((xx@self.projection,cc),1)
        torch.manual_seed(M.SEED)
        self.model=torch.nn.Sequential(torch.nn.Linear(inputs.shape[1],128),torch.nn.SiLU(),
            torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,63)).to(self.device)
        self.ym=y.mean(0);self.ys=np.maximum(y.std(0),1.)
        xt=torch.as_tensor(inputs,device=self.device)
        yt=torch.as_tensor((y-self.ym)/self.ys,device=self.device,dtype=torch.float32)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        for step in range(300):
            ix=torch.as_tensor(rng.integers(0,len(x),32),device=self.device)
            optimizer.zero_grad();loss=torch.nn.functional.mse_loss(self.model(xt[ix]),yt[ix])
            loss.backward();optimizer.step()
            if step%25==0 or step==299:self.curves.append([step+1,float(loss.detach().cpu())])
        self.params=sum(p.numel() for p in self.model.parameters())
        return self


def self_check():
    rng=np.random.default_rng(M.SEED)
    a=rng.normal(size=(128,51));lam=rng.lognormal(size=(128,24));f=rng.normal(size=(51,24))
    order=rng.permutation(128)
    expected=joint_diagonal_support(a,lam)
    actual=joint_diagonal_support(a[order],lam[order])
    if not np.array_equal(expected,actual):raise AssertionError('point permutation changed joint support')
    if not np.array_equal(project_typed(f,a,lam),project_typed(f,a[order],lam[order])):
        raise AssertionError('point permutation changed projected typed feature')
    zero=project_typed(f,a,np.zeros_like(lam))
    if np.count_nonzero(zero):raise AssertionError('zero lambda did not produce exact zero typed package')
    if not np.array_equal(project_typed(f,a,lam)[:F_D],M.signedlog(f).astype(np.float32)):
        raise AssertionError('supported direct-Z f was changed')
    np.testing.assert_allclose(expected,np.einsum('nh,nj,nj->jh',lam,a,a),rtol=2e-15,atol=1e-12)
    x=np.zeros((4,NEW_D),np.float32);x[:,SENSOR_D]=[0.,1e-6,-1e-6,0.]
    c=np.zeros((4,72),np.float32)
    normal=RevisionNormalize().fit(x,c)
    if np.any(normal.scale<SCALE_FLOOR):raise AssertionError('physical scale floor failed')
    z,_=normal.transform(np.zeros_like(x),c)
    if np.count_nonzero(z):raise AssertionError('normalizer destroys zero support')
    return dict(point_permutation_bitwise=True,lambda_zero_typed_exact_zero=True,
        supported_direct_f_unchanged=True,joint_diagonal_formula=True,physical_RMS_floor=True,
        zero_preserving_normalization=True,old_feature_dimensions=OLD_D,new_feature_dimensions=NEW_D,
        MLP_parameters='same JL512 output + 128x2 + 63, context72; unchanged optimizer/steps/batches',
        scope='arithmetic only, no data/labels/task scores loaded')


def run_domain(domain,out,device):
    original=M.OUT/domain
    with np.load(M.OUT/f'{domain}_features.npz',allow_pickle=False) as cache:values={k:cache[k] for k in cache.files}
    old_report=json.loads((original/'metrics.json').read_text())
    fit=values['partition']=='FIT7';select=values['partition']=='SELECT'
    features={}
    for variant in ('original','destroy'):
        for arm in R1_ARMS:
            name=arm if variant=='original' else arm+'_destroy'
            features[variant,arm]=project_cached(values[name])
    report={};heads={};predictions={}
    for kind in ('rbf','mlp'):
        report[kind]={}
        for variant in ('original','destroy'):
            report[kind][variant]={}
            for arm in R1_ARMS:
                x=features[variant,arm]
                head=RevisionHead(kind,device).fit(x[fit],values['context'][fit],values['delta'][fit])
                pred=head.predict(x,values['context'])
                row={part:M.risks(pred[mask],values['prior_joints'][mask],values['target_joints'][mask],values['recording'][mask])
                    for part,mask in (('FIT7',fit),('SELECT',select))}
                row.update(params=head.params,feature_dimensions=NEW_D,curves=head.curves,
                    physical_scale_min=float(head.normal.scale.min()),physical_scale_max=float(head.normal.scale.max()))
                report[kind][variant][arm]=row
                head.save(out/f'{kind}_{variant}_{arm}.pt')
                predictions[f'{kind}__{variant}__{arm}']=pred.astype(np.float32)
                if variant=='original':heads[kind,arm]=head
                M.log(f'R1 {domain} {kind}/{variant}/{arm} SELECTabs={row["SELECT"]["abs_mm"]:.3f}')
            for baseline in ('lnes','enhanced','a_only','context_only'):
                report[kind][variant][baseline]=old_report['E1'][kind][variant][baseline]
    e2={}
    for key in values:
        if not key.startswith('e2__') or not key.endswith('__fresh'):continue
        condition=key.split('__')[1];e2[condition]={}
        c=values[f'e2__{condition}__context'];prior=values[f'e2__{condition}__prior_joints']
        fresh=project_cached(values[key]);stale=project_cached(values[f'e2__{condition}__stale'])
        for kind in ('rbf','mlp'):
            e2[condition][kind]={}
            for mode,x in (('fresh',fresh),('stale',stale)):
                pred=heads[kind,'full'].predict(x,c)
                e2[condition][kind][mode]=M.risks(pred[select],prior[select],values['target_joints'][select],values['recording'][select])
            for baseline in ('sensor_only','perturbed_prior'):
                e2[condition][kind][baseline]=old_report['E2'][condition][kind][baseline]
    support={kind:all(report[kind]['original']['full']['SELECT']['abs_mm']<=
        .98*report[kind]['original'][arm]['SELECT']['abs_mm'] for arm in ('no_order','enhanced')) for kind in ('rbf','mlp')}
    np.savez_compressed(out/'predictions.npz',**predictions)
    result=dict(E1=report,E2=e2,E1_support=all(support.values()),E1_support_by_head=support,
        scope='R1 prototype reader adaptation only; original negatives remain valid for original readers.',
        baselines='Original LN/enhanced/A-only/context-only heads and scores retained without refitting.',
        storage_approximation='Invert original float32 signedlog(a)/log1p(lambda), retain cached direct signedlog(f) exactly; no new prefix/history/source.',
        no_uncertainty_claim=True,no_new_statistic_claim=True)
    M.dump(out/'metrics.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--prereg',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--out',type=Path,default=M.OUT/'R1')
    args=parser.parse_args()
    checks=self_check()
    if args.self_check:print(json.dumps(checks,indent=2))
    if not args.run:return
    if args.prereg is None or not args.prereg.exists() or args.prereg.resolve()==M.PROTOCOL.resolve():
        raise RuntimeError('explicit separate correction R1 preregistration required before task scores')
    if 'R1' not in args.prereg.read_text():raise RuntimeError('correction preregistration does not identify R1')
    import torch
    torch.set_num_threads(1)
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=dict(correction='R1',preregistration=str(args.prereg),prereg_sha256=M.sha(args.prereg),
        original_protocol_sha256=M.sha(M.PROTOCOL),source_sha256=M.sha(__file__),checks=checks,
        fixed_typed_basis='MANO51 x ordered(p,q,tau,age)24',
        features='A288 + direct signedlog(f)1224 + log1p(sum_point lambda*a^2)1224',
        floor=SCALE_FLOOR,clip=CLIP,normalization='uncentered FIT RMS per feature, physical floor1, fixed equal block budgets; original context normalization',
        seed=M.SEED,MLP=dict(JL=512,hidden=[128,128],steps=300,batch=32,lr=.001),
        changed='Prototype E12 reader projection/normalization only',
        unchanged='A/Z/tau/gamma/events/history/K/beta/targets/seed/head steps',
        parallel_S43_revision='The same unified R1 preregistration separately changes the S43 support reader to joint_gram; this script changes only the offline experimental reader.',
        started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    manifest_path=args.out/'implementation_manifest.json'
    if manifest_path.exists():raise FileExistsError('R1 outputs already exist; do not overwrite a scored revision')
    M.dump(manifest_path,manifest)
    results={}
    for domain in ('existing','synthetic'):
        dest=args.out/domain;dest.mkdir(parents=True,exist_ok=True)
        results[domain]=run_domain(domain,dest,args.device)
    M.dump(args.out/'summary.json',dict(correction='R1',complete=True,checks=checks,
        E1_support={d:r['E1_support'] for d,r in results.items()},
        metrics={d:str(args.out/d/'metrics.json') for d in results},
        claim='Reader adaptation, not new sensor-statistic validation; original reports retained.'))


if __name__=='__main__':main()
