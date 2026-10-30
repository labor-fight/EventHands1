"""Preregistered E1/E2 finite-head diagnostics, separate from S43 training.

Every 50ms mechanism snippet resets at the SAME observation boundary in every
arm. This local intervention experiment is not continuous-history deployment.
Targets first enter supervision/scoring; public or frozen two-step predictions
alone define construction geometry. Time destruction is explicitly nonphysical.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .core import RelConfig, RelationMemory
from .data import enhanced_channels, last_lnes, sensor_A_grid
from .geometry import ManoGeometry, event_intrinsics
from tools.s43_br.data import (FIT7, SELECT, PUBLIC_INITIAL_STATE,
                               load_existing_samples, CHECKPOINT, CONFIG)
from tools.s43_integration import fast_geometry as FK

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_rel_20261027/mechanisms/e12'
PROTOCOL = ROOT / 'outputs/s43_rel_20261027/protocol.json'
ARMS = ('context_only', 'lnes', 'enhanced', 'a_only', 'no_order', 'full', 'tore4')
SEED = 3407
CORE = RelConfig(backend='cpp')
POOL = (6, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda a: a.tolist() if isinstance(a, np.ndarray) else
        a.item() if isinstance(a, np.generic) else str(a)) + '\n')


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def pool(image):
    h, w = POOL
    return np.asarray(image).reshape(h, 180//h, w, 240//w, -1).mean((1, 3)).ravel()


def signedlog(value):
    value = np.asarray(value)
    return (np.sign(value) * np.log1p(np.abs(value))).ravel()


def typed_feature(A, package):
    return np.concatenate((pool(sensor_A_grid(A)), signedlog(package['f']),
                            signedlog(package['a']), np.log1p(package['lam']).ravel())).astype(np.float32)


def context(prev, beta, K):
    # Both readers receive precisely the same deployable conditioning fields.
    return np.r_[prev, beta, np.asarray(K).ravel()/640., 1., 1.].astype(np.float32)


def retract(prev, kind, amount, finger):
    result = np.asarray(prev, np.float64).copy()
    if kind == 'root':
        result[0] += amount
    else:
        with np.load(ROOT/'assets/mano_right.npz') as data:
            mean = data['hands_mean']
        joint, axis, sign = finger['joint'], finger['axis'], finger['sign']
        start = 6 + (joint-1)*3
        old = Rotation.from_rotvec(result[start:start+3]+mean[(joint-1)*3:joint*3])
        delta = np.eye(3)[axis]*amount*sign
        result[start:start+3] = (old*Rotation.from_rotvec(delta)).as_rotvec()-mean[(joint-1)*3:joint*3]
    return result.astype(np.float32)


def choose_finger():
    """Select MCP flexion using only public FK: maximal wrist-tip shortening."""
    state = PUBLIC_INITIAL_STATE.copy(); beta = np.zeros(10, np.float32)
    original = FK.fk_joints(state, beta)
    candidates = []
    for axis in range(3):
        for sign in (-1, 1):
            selected = dict(joint=1, axis=axis, sign=sign)
            altered = retract(state, 'finger', .3, selected)
            joints = FK.fk_joints(altered, beta)
            candidates.append(dict(selected, shortening_mm=float((np.linalg.norm(original[8]-original[0])-
                np.linalg.norm(joints[8]-joints[0]))*1000),
                index_mpjpe_mm=float(np.linalg.norm(joints[5:9]-original[5:9], axis=-1).mean()*1000),
                all_mpjpe_mm=float(np.linalg.norm(joints-original, axis=-1).mean()*1000)))
    best = max(candidates, key=lambda v: v['shortening_mm'])
    if best['shortening_mm'] <= 0 or best['index_mpjpe_mm'] < .5:
        raise RuntimeError('public FK did not identify a real flexion direction')
    return dict(selected={k:best[k] for k in ('joint','axis','sign')}, candidates=candidates,
                rule='public mean, MCP joint1, +/-0.3rad: maximal wrist-to-index-tip shortening; no events/labels')


def core_events(native):
    # Existing adapter is [t_us,x,y,p]. All science uses integer-ms groups.
    ev = np.asarray(native, np.int64)
    return np.column_stack((ev[:, 1], ev[:, 2], ev[:, 0]//1000, ev[:, 3]))


def memory(events):
    result = RelationMemory(CORE)
    result.ingest(events, 50, 0, time_unit='ms', source_range=(0, len(events)))
    return result


def destroy_times(events, seed):
    """Preserve pixel/p count, first/last and all four coarse-bin totals exactly."""
    out = np.array(events, copy=True)
    if not len(out):
        return out, dict(changed=0, invariants=True)
    slot = (out[:, 1]*240+out[:, 0])*2+out[:, 3]
    first = np.full(180*240*2, 50, np.int64)
    last = np.full_like(first, -1)
    np.minimum.at(first, slot, out[:, 2]); np.maximum.at(last, slot, out[:, 2])
    bins = 4*out[:, 2]//50
    low = np.maximum((bins*50+3)//4, first[slot]+1)
    high = np.minimum(((bins+1)*50+3)//4-1, last[slot]-1)
    interior = (out[:, 2] > first[slot]) & (out[:, 2] < last[slot]) & (high >= low)
    rng = np.random.default_rng(seed)
    out[interior, 2] = rng.integers(low[interior], high[interior]+1)
    changed = int(np.count_nonzero(out[:, 2] != events[:, 2]))
    # Check protected values before reordering; identities stay on their pixel/p.
    f2, l2 = np.full_like(first, 50), np.full_like(last, -1)
    np.minimum.at(f2, slot, out[:, 2]); np.maximum.at(l2, slot, out[:, 2])
    valid = np.array_equal(first, f2) and np.array_equal(last, l2) and np.array_equal(bins, 4*out[:, 2]//50)
    if not valid:
        raise AssertionError('protected interior-time intervention changed its declared marginals')
    return out[np.argsort(out[:, 2], kind='stable')], dict(changed=changed, invariants=valid)


def tore_events(events, k=4):
    if not len(events):
        return events.copy()
    slot = (events[:, 1]*240+events[:, 0])*2+events[:, 3]
    order = np.argsort(slot, kind='stable')
    sorted_slots = slot[order]
    end = np.r_[np.flatnonzero(sorted_slots[1:] != sorted_slots[:-1])+1, len(order)]
    start = np.r_[0, end[:-1]]
    select = np.concatenate([order[max(a,b-k):b] for a,b in zip(start,end)])
    return events[np.sort(select)]


def read_binding(mem, binding, mode='full'):
    if mode == 'no_order':
        reverse = mem.topology.reverse[binding.edge_ids]
        ids = np.union1d(binding.edge_ids, reverse)
        snap = mem.readout(50, ids)
        direct, rev = np.searchsorted(ids, binding.edge_ids), np.searchsorted(ids, reverse)
        # Fixed p,q slots: the protocol's spatial-order ablation, NOT F_abpq.
        snap = dataclasses.replace(snap, edge_ids=binding.edge_ids, Z=.5*(snap.Z[direct]+snap.Z[rev]))
    else:
        snap = mem.readout(50, binding.edge_ids)
    return snap, binding.read(snap)


def encode(events, prev, beta, K, geometry, binding=None):
    started = time.perf_counter(); mem = memory(events)
    binding = binding or geometry.bind(prev, beta, event_intrinsics(K), mem.topology)
    snap, package = read_binding(mem, binding)
    _, unordered = read_binding(mem, binding, 'no_order')
    tore = memory(tore_events(events))
    tsnap, tpackage = read_binding(tore, binding)
    features = dict(context_only=np.zeros(1,np.float32),lnes=pool(last_lnes(events, 0, 50)),
        enhanced=pool(enhanced_channels(events, 0, 50, snap.A)),
        a_only=pool(sensor_A_grid(snap.A)),
        full=typed_feature(snap.A, package), no_order=typed_feature(snap.A, unordered),
        tore4=typed_feature(tsnap.A, tpackage))
    return features, mem, binding, dict(seconds=time.perf_counter()-started,
        source_events=len(events), tore_events=len(tore_events(events)), edges=len(binding.edge_ids),
        bytes=mem.stats()['resident_bytes'], support_points=int(package['support'].sum()))


class CausalHistory:
    """GPU optional replacement of the read-only canonical two-step history."""
    def __init__(self, device):
        import torch
        from tools.s43_integration.model import load_legacy
        self.torch = torch; self.device = device
        self.model, _ = load_legacy(device=device)

    def predict(self, first_images, second_images, betas, K_cam, batch_size=8):
        torch = self.torch; older, prev = [], []
        with torch.inference_mode():
            for i in range(0, len(first_images), batch_size):
                n = min(batch_size, len(first_images)-i)
                tensor = lambda x: torch.as_tensor(x, device=self.device, dtype=torch.float32)
                b = tensor(np.repeat(np.asarray(betas)[None], n, axis=0))
                K = tensor(np.repeat(np.asarray(K_cam)[None], n, axis=0))
                initial = tensor(np.repeat(PUBLIC_INITIAL_STATE[None], n, axis=0))
                a = self.model(tensor(first_images[i:i+n]), initial, betas=b, camera_K=K)
                z = self.model(tensor(second_images[i:i+n]), a, betas=b, camera_K=K)
                older.append(a.cpu().numpy()); prev.append(z.cpu().numpy())
        return np.concatenate(older), np.concatenate(prev)


def full_render(states, beta, K):
    vertices, _ = FK.fk_geometry(states, beta)
    vertices = vertices.reshape(-1, 778, 3).astype(np.float64)
    if vertices[..., 2].min() <= .03:
        raise RuntimeError('synthetic MANO intersects near plane')
    h = vertices @ event_intrinsics(K).T
    xy = np.ascontiguousarray(h[..., :2]/h[..., 2:], np.float64)
    faces = np.ascontiguousarray(FK._mano().f.numpy(), np.int64)
    masks = np.zeros((len(vertices), 180, 240), np.uint8)
    rc = FK._raster_library().s43_constant_coverage(
        xy.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), faces.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        len(vertices), 778, len(faces), 180, 240,
        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    if rc:
        raise RuntimeError('full-resolution threshold renderer failed')
    return masks


def threshold_events(masks, rng, threshold):
    levels = np.log(np.where(masks, .7, .15))
    reference = levels[0]+rng.uniform(-threshold*.1, threshold*.1, (180,240))
    output = []
    for t in range(50):
        change = levels[t+1]-reference
        count = np.floor(np.abs(change)/threshold).astype(np.int64)
        direction = change > 0
        reference += np.sign(change)*count*threshold
        yy,xx = np.nonzero(count)
        if len(xx):
            output.append(np.column_stack((np.repeat(xx,count[yy,xx]), np.repeat(yy,count[yy,xx]),
                np.full(count.sum(),t), np.repeat(direction[yy,xx],count[yy,xx]))))
        # Public independent, sparse sensor noise; not conditioned on motion.
        n = int(rng.poisson(180*240*.000005))
        if n:
            output.append(np.column_stack((rng.integers(0,240,n),rng.integers(0,180,n),
                                            np.full(n,t),rng.integers(0,2,n))))
    return np.concatenate(output).astype(np.int64) if output else np.empty((0,4),np.int64)


def synthetic_samples(finger, contexts=30):
    samples = []
    K = np.array([[800.,0,320.],[0,800.,240.],[0,0,1]],np.float32)
    for axis_id, axis in enumerate(('root', 'finger')):
        for index in range(contexts):
            rng = np.random.default_rng(np.random.SeedSequence([SEED,axis_id,index,0]))
            public = PUBLIC_INITIAL_STATE.copy()
            public[:3] = [rng.uniform(-.018,.018),rng.uniform(-.025,.005),rng.uniform(.5,.64)]
            public[3:6] = rng.uniform(-.2,.2,3)
            public[6:] = rng.normal(0,.025,45)
            beta = rng.normal(0,.3,10).astype(np.float32)
            partition = 'FIT7' if index < 16 else 'VALID' if index < 22 else 'SELECT'
            for cls, amplitude in enumerate((-.015,0.,.015) if axis=='root' else (-.3,0.,.3)):
                states = np.stack([retract(public,axis,amplitude*t/50.,finger) for t in range(51)])
                masks = full_render(states,beta,K)
                for replicate, threshold in enumerate((.2,.3)):
                    rng_obs = np.random.default_rng(np.random.SeedSequence([SEED,axis_id,index,cls,replicate,1]))
                    ev = threshold_events(masks,rng_obs,threshold)
                    native = np.column_stack((ev[:,2]*1000,ev[:,0],ev[:,1],ev[:,3]))
                    samples.append(dict(events=native,prev_state=public.copy(),prev_older_state=public.copy(),
                        betas=beta.copy(),K_cam=K.copy(),target=states[-2].copy(),subject=f'context{index:02d}',
                        recording=f'synthetic_{axis}_{index:02d}',partition=partition,domain=axis,end_ms=49,
                        sample_id=f'{axis}/context{index:02d}/class{cls}/rep{replicate}',
                        context_id=index,motion_class=cls,threshold=threshold,axis=axis,
                        history_source='known public synthetic boundary geometry, not inferred GT history'))
            log(f'synthetic generator {axis} context {index+1}/{contexts}')
    return samples


def feature_job(task):
    sample, index, finger, do_e2 = task
    import torch
    geometry = ManoGeometry()
    e = core_events(sample['events'])
    prev, beta, K = sample['prev_state'], sample['betas'], sample['K_cam']
    original, mem, binding, costs = encode(e,prev,beta,K,geometry)
    changed, audit = destroy_times(e,SEED+index)
    destroyed, _, _, _ = encode(changed,prev,beta,K,geometry,binding)
    jprev = FK.fk_joints(prev,beta); jtarget = FK.fk_joints(sample['target'],beta)
    result = dict(features=original,destroyed=destroyed,context=context(prev,beta,K),
        prior_joints=jprev,target_joints=jtarget,delta=(jtarget-jprev).ravel()*1000,
        partition=sample['partition'],recording=sample['recording'],domain=sample['domain'],
        sample_id=sample.get('sample_id',f"{sample['recording']}/{sample['end_ms']}"),
        source_sha256=hashlib.sha256(e.tobytes()).hexdigest(),source_events=len(e),
        costs=costs,time_audit=audit,history_source=sample['history_source'])
    if do_e2:
        result['e2'] = {}
        rebuilt = memory(e)
        for kind, amount in (('root',.005),('root',.015),('finger',.1),('finger',.3)):
            perturbed = retract(prev,kind,amount,finger)
            fresh_binding = geometry.bind(perturbed,beta,event_intrinsics(K),mem.topology)
            fs, fresh = read_binding(mem,fresh_binding)
            rs, raw = read_binding(rebuilt,fresh_binding)
            parity = all(np.allclose(fresh[k],raw[k],rtol=1e-4,atol=1e-5) for k in ('f','a','lam'))
            if not parity:
                raise AssertionError('same raw/same core/rebind parity failed')
            key = f'{kind}_{amount:g}'
            result['e2'][key] = dict(fresh=typed_feature(fs.A,fresh),stale=original['full'],
                context=context(perturbed,beta,K),prior_joints=FK.fk_joints(perturbed,beta),
                raw_parity=parity,parity_max=float(max(np.max(np.abs(fresh[k]-raw[k])) for k in ('f','a','lam'))))
    return result


def extract(samples, path, finger, workers=8, do_e2=True):
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i,result in enumerate(executor.map(feature_job,[(s,i,finger,do_e2) for i,s in enumerate(samples)])):
            records.append(result)
            if (i+1)%16==0:
                log(f'feature {path.stem} {i+1}/{len(samples)}')
    values = {arm:np.stack([r['features'][arm] for r in records]) for arm in ARMS}
    values.update({arm+'_destroy':np.stack([r['destroyed'][arm] for r in records]) for arm in ARMS})
    for key in ('context','prior_joints','target_joints','delta','partition','recording','domain','sample_id'):
        values[key] = np.stack([r[key] for r in records])
    if do_e2:
        for cond in records[0]['e2']:
            for key in ('fresh','stale','context','prior_joints'):
                values[f'e2__{cond}__{key}'] = np.stack([r['e2'][cond][key] for r in records])
    np.savez_compressed(path,**values)
    manifest = [{k:v for k,v in r.items() if k not in ('features','destroyed','context','prior_joints','target_joints','delta','e2')} for r in records]
    dump(path.with_suffix('.manifest.json'),dict(samples=manifest,
        raw_parity=all(all(q['raw_parity'] for q in r.get('e2',{}).values()) for r in records),
        raw_parity_max=max([q['parity_max'] for r in records for q in r.get('e2',{}).values()] or [0.]),
        total_changed_timestamps=sum(r['time_audit']['changed'] for r in records),
        protected_invariants=all(r['time_audit']['invariants'] for r in records),
        reset_contract='isolated50ms same boundary reset in all arms; not persistent deployment'))
    return values


class Normalize:
    def fit(self,x,c):
        self.mu=x.mean(0);self.sd=x.std(0);self.cm=c.mean(0);self.cs=c.std(0)
        self.xscale=np.sqrt(max(1,(self.sd>=1e-6).sum()));self.cscale=np.sqrt(max(1,(self.cs>=1e-6).sum()))
        self.sd[self.sd<1e-6]=1;self.cs[self.cs<1e-6]=1
        return self
    def transform(self,x,c):
        return (np.clip((x-self.mu)/self.sd,-8,8)/self.xscale).astype(np.float32), (np.clip((c-self.cm)/self.cs,-8,8)/self.cscale).astype(np.float32)


def dist(a,b):
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None]-2*a@b.T,0.)


class RegressionHead:
    def __init__(self,kind,device):
        self.kind,self.device=kind,device
    def fit(self,x,c,y):
        import torch
        self.normal=Normalize().fit(x,c);xx,cc=self.normal.transform(x,c)
        self.curves=[]
        if self.kind=='rbf':
            self.train=np.concatenate((xx,cc),1).astype(np.float64)
            d=dist(self.train,self.train);v=d[np.triu_indices(len(d),1)]
            self.bandwidth=max(float(np.median(v[v>1e-12])) if np.any(v>1e-12) else 1.,1e-6)
            self.coef=np.linalg.solve(np.exp(-d/self.bandwidth)+.01*np.eye(len(d)),y)
            self.params=int(self.coef.size);return self
        rng=np.random.default_rng(SEED)
        self.projection=(rng.standard_normal((x.shape[1],512))/np.sqrt(512)).astype(np.float32)
        inputs=np.concatenate((xx@self.projection,cc),1)
        torch.manual_seed(SEED)
        self.model=torch.nn.Sequential(torch.nn.Linear(inputs.shape[1],128),torch.nn.SiLU(),
            torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,63)).to(self.device)
        self.ym=y.mean(0);self.ys=np.maximum(y.std(0),1.)
        xt=torch.as_tensor(inputs,device=self.device);yt=torch.as_tensor((y-self.ym)/self.ys,device=self.device,dtype=torch.float32)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        for step in range(300):
            ix=torch.as_tensor(rng.integers(0,len(x),32),device=self.device)
            optimizer.zero_grad();loss=torch.nn.functional.mse_loss(self.model(xt[ix]),yt[ix]);loss.backward();optimizer.step()
            if step%25==0 or step==299:self.curves.append([step+1,float(loss.detach().cpu())])
        self.params=sum(p.numel() for p in self.model.parameters());return self
    def predict(self,x,c):
        xx,cc=self.normal.transform(x,c)
        if self.kind=='rbf':
            a=np.concatenate((xx,cc),1).astype(np.float64)
            return np.exp(-dist(a,self.train)/self.bandwidth)@self.coef
        import torch
        with torch.inference_mode():
            a=np.concatenate((xx@self.projection,cc),1)
            return self.model(torch.as_tensor(a,device=self.device)).cpu().numpy()*self.ys+self.ym
    def save(self,path):
        import torch
        state=dict(self.__dict__)
        state['normal']=dict(self.normal.__dict__)
        if self.kind=='mlp':state['model']=self.model.cpu().state_dict();self.model.to(self.device)
        torch.save(state,path)


def risks(delta,prior,target,recordings):
    prediction=prior+delta.reshape(-1,21,3)/1000.
    absolute=np.linalg.norm(prediction-target,axis=-1).mean(-1)*1000
    relative=np.linalg.norm((prediction-prediction[:,:1])-(target-target[:,:1]),axis=-1).mean(-1)*1000
    result=dict(abs_mm=float(absolute.mean()),ra_mm=float(relative.mean()),
        mse_mm2=float(np.mean(((prediction-target)*1000)**2)),n=len(prediction),
        per_recording={str(r):dict(abs_mm=float(absolute[recordings==r].mean()),
            ra_mm=float(relative[recordings==r].mean())) for r in np.unique(recordings)})
    return result


def train_evaluate(values,out,device):
    out.mkdir(parents=True,exist_ok=True)
    fit=values['partition']=='FIT7';select=values['partition']=='SELECT'
    if not fit.any() or not select.any():raise ValueError('no FIT or SELECT rows')
    results,heads={},{}
    for kind in ('rbf','mlp'):
        results[kind]={}
        for variant in ('original','destroy'):
            results[kind][variant]={}
            for arm in ARMS:
                name=arm if variant=='original' else arm+'_destroy'
                head=RegressionHead(kind,device).fit(values[name][fit],values['context'][fit],values['delta'][fit])
                pred=head.predict(values[name],values['context'])
                row={part:risks(pred[mask],values['prior_joints'][mask],values['target_joints'][mask],values['recording'][mask])
                     for part,mask in [('FIT7',fit),('SELECT',select)]}
                row.update(params=head.params,feature_dimensions=values[name].shape[1],curves=head.curves)
                results[kind][variant][arm]=row
                head.save(out/f'{kind}_{variant}_{arm}.pt')
                if variant=='original':heads[kind,arm]=head
                log(f'head {out.name} {kind}/{variant}/{arm} SELECTabs={row["SELECT"]["abs_mm"]:.3f}')
    e2={}
    for key in values:
        if not key.startswith('e2__') or not key.endswith('__fresh'):continue
        cond=key.split('__')[1];e2[cond]={}
        c=values[f'e2__{cond}__context'];prior=values[f'e2__{cond}__prior_joints']
        for kind in ('rbf','mlp'):
            e2[cond][kind]={}
            for mode in ('fresh','stale','sensor_only'):
                x=values['enhanced'] if mode=='sensor_only' else values[f'e2__{cond}__{mode}']
                head=heads[kind,'enhanced' if mode=='sensor_only' else 'full']
                pred=head.predict(x,c)
                e2[cond][kind][mode]=risks(pred[select],prior[select],values['target_joints'][select],values['recording'][select])
            e2[cond][kind]['perturbed_prior']=risks(np.zeros((select.sum(),63)),prior[select],values['target_joints'][select],values['recording'][select])
    supports={}
    for kind in ('rbf','mlp'):
        rows=results[kind]['original'];full=rows['full']['SELECT']['abs_mm']
        supports[kind]=all(full<=.98*rows[a]['SELECT']['abs_mm'] for a in ('no_order','enhanced'))
    report=dict(E1=results,E1_support=all(supports.values()),E1_support_by_head=supports,E2=e2,
        claim='Task-risk diagnostic; same-target time destruction is nonphysical and is not a matched physical-pair proof.',
        synthetic_matching='No event-marginal matched different-target physical pair construction is claimed; missing evidence remains missing.',
        independent_unit='recording/subject on existing; context on synthetic, not windows or observation repeats',
        MLP='Fixed JL512 then two hidden128 SiLU layers, Adam.001 batch32 300updates; MSE jointdelta proper mean regression',
        RBF='FIT-normalized event/context blocks, median FIT pair-distance bandwidth, ridge.01')
    dump(out/'metrics.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--stage',choices=('all','existing','synthetic','heads'),default='all')
    parser.add_argument('--engineering-contexts',type=int,default=0)
    args=parser.parse_args()
    if not PROTOCOL.exists():raise RuntimeError('mechanism protocol must be frozen before execution')
    import torch
    torch.set_num_threads(1);np.random.seed(SEED);torch.manual_seed(SEED)
    args.out.mkdir(parents=True,exist_ok=True)
    finger=choose_finger()
    manifest=dict(protocol_sha256=sha(PROTOCOL),source_sha256=sha(__file__),seed=SEED,
        fixed_sources={p:sha(ROOT/p) for p in ('assets/mano_right.npz','tools/s43_rel/core.py','tools/s43_rel/core_cpu.cpp',
            'tools/s43_rel/geometry.py','tools/s43_rel/data.py','tools/s43_br/data.py')},
        selected_finger=finger,feature_pool=[6,8],typed_order=['A6pool288','f1224','a6528','lambda3072'],
        MLP_JL=512,MLP_hidden=[128,128],MLP_steps=300,MLP_batch=32,MLP_lr=.001,
        history='canonical original S43 two causal W50 steps from public initial pose; no GT',
        snippet_reset='all arms reset common local50ms boundary; no persistent-deployment claim',
        time_destroy='nonphysical same-target diagnostic preserving pixel-p count/first/last/coarse4',
        cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    dump(args.out/'implementation_manifest.json',manifest)
    if args.stage in ('all','existing'):
        path=args.out/'existing_features.npz'
        if not path.exists():
            log('loading label-independent existing windows and canonical causal history')
            history=CausalHistory(args.device)
            samples=load_existing_samples(subjects=FIT7+SELECT,samples_per_recording=16,history_provider=history)
            del history;torch.cuda.empty_cache() if args.device.startswith('cuda') else None
            log(f'loaded {len(samples)} windows')
            extract(samples,path,finger['selected'],args.workers,True)
            del samples
    if args.stage in ('all','synthetic'):
        path=args.out/('synthetic_engineering_features.npz' if args.engineering_contexts else 'synthetic_features.npz')
        if not path.exists():
            samples=synthetic_samples(finger['selected'],args.engineering_contexts or 30)
            extract(samples,path,finger['selected'],args.workers,False)
            del samples
    if args.stage in ('all','heads','existing','synthetic') and not args.engineering_contexts:
        for domain in ('existing','synthetic'):
            path=args.out/f'{domain}_features.npz'
            if path.exists() and not (args.out/domain/'metrics.json').exists():
                with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
                train_evaluate(values,args.out/domain,args.device)
    log('E12 requested stages complete')


if __name__=='__main__':main()
