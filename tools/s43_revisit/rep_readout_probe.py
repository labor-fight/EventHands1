"""Stage1 P2: CPU-only, preregistered same-cache readout/compression ladder.

Writes a pre-score implementation amendment before materializing cache arrays.
Never recomputes geometry, reads raw recordings, opens zgz, or edits old files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .representation_audit import ROOT, E12, sha256, write_json

ARMS = ("context_only", "lnes", "enhanced", "a_only",
        "production_joint_gram", "slot_gram", "point_factor", "geometry_only")
PROTOCOL_SHA = "e81f43be5391128ed3449965fcaf4c5c209583ddd67c9f216c0d81dedc13bdc7"
SEED = 3407


def slog(x):
    return np.sign(x)*np.log1p(np.abs(x))


def cache_factors(full):
    if full.ndim != 2 or full.shape[1] != 11112:
        raise ValueError("expected frozen A288/f1224/a6528/lambda3072 cache")
    a_log = full[:,1512:8040].astype(np.float64).reshape(-1,128,51)
    ll = full[:,8040:].astype(np.float64).reshape(-1,128,24)
    if np.any(ll < 0):
        raise ValueError("negative cached support")
    return np.sign(a_log)*np.expm1(np.abs(a_log)), np.expm1(ll)


def make_features(arm, cache, a, lam):
    """Return logged features and fixed weighted semantic slices."""
    full = cache["full"]
    n = len(full)
    if arm == "context_only":
        return np.zeros((n,1),np.float32), [(0,1,1.)]
    if arm in ("lnes", "enhanced", "a_only"):
        x = cache[arm].astype(np.float32,copy=True)
        # These baseline caches already contain the prescribed image channels;
        # pool values are retained. No second log is applied to logged counts/A.
        return x, [(0,x.shape[1],1.)]
    if arm == "geometry_only":
        return full[:,1512:8040].copy(), [(0,6528,1.)]
    if arm == "point_factor":
        return full.copy(), [(0,288,1/3), (288,1512,1/3),
                             (1512,8040,1/6), (8040,11112,1/6)]
    if arm == "production_joint_gram":
        diagonal = np.einsum("bnj,bnh->bjh",a*a,lam,optimize=True)
        summed = np.einsum("bnj,bnk,bn->bjk",a,a,lam.sum(-1),optimize=True)
        support = np.concatenate((slog(summed).reshape(n,-1),
                                  np.log1p(diagonal).reshape(n,-1)),1)
    elif arm == "slot_gram":
        ij = np.triu_indices(51)
        support = np.empty((n,24*len(ij[0])),np.float32)
        for i in range(n):
            gh = np.einsum("nh,nj,nk->hjk",lam[i],a[i],a[i],optimize=True)
            support[i] = slog(gh[:,ij[0],ij[1]]).ravel()
    else:
        raise ValueError(arm)
    x = np.concatenate((full[:,:1512],support),1).astype(np.float32)
    return x, [(0,288,1/3), (288,1512,1/3), (1512,x.shape[1],1/3)]


class Normalizer:
    def fit(self, x, context, blocks):
        self.scale=np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2,axis=0)),1.)
        self.blocks=blocks
        # The old E12 context normalization is deliberately unchanged.
        self.cm=context.mean(0); self.cs=context.std(0)
        self.context_dim_scale=np.sqrt(max(1,int(np.sum(self.cs>=1e-6))))
        self.cs[self.cs<1e-6]=1.
        return self

    def transform(self,x,context):
        feature=np.clip(x/self.scale,-8,8).astype(np.float32)
        for start,stop,budget in self.blocks:
            feature[:,start:stop]*=np.sqrt(budget/(stop-start))
        cond=(np.clip((context-self.cm)/self.cs,-8,8)/self.context_dim_scale).astype(np.float32)
        return feature,cond


def distances(a,b):
    return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None]-2*a@b.T,0)


class Head:
    def __init__(self,kind):
        self.kind=kind

    def fit(self,x,c,y,blocks):
        import torch
        self.normal=Normalizer().fit(x,c,blocks)
        xx,cc=self.normal.transform(x,c)
        self.curves=[]
        if self.kind=="rbf":
            self.train=np.concatenate((xx,cc),1).astype(np.float64)
            distance=distances(self.train,self.train)
            positive=distance[np.triu_indices(len(y),1)]
            positive=positive[positive>1e-12]
            self.bandwidth=max(float(np.median(positive)) if len(positive) else 1.,1e-6)
            self.coef=np.linalg.solve(np.exp(-distance/self.bandwidth)+.01*np.eye(len(y)),y)
            self.params=int(self.coef.size)
            self.storage_bytes=int(self.train.nbytes+self.coef.nbytes)
            return self
        rng=np.random.default_rng(SEED)
        self.projection=(rng.standard_normal((x.shape[1],512))/np.sqrt(512)).astype(np.float32)
        inputs=np.concatenate((xx@self.projection,cc),1)
        torch.manual_seed(SEED)
        self.model=torch.nn.Sequential(torch.nn.Linear(inputs.shape[1],128),torch.nn.SiLU(),
            torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,63))
        self.ym=y.mean(0); self.ys=np.maximum(y.std(0),1.)
        xt=torch.as_tensor(inputs); yt=torch.as_tensor((y-self.ym)/self.ys,dtype=torch.float32)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        for step in range(300):
            ix=torch.as_tensor(rng.integers(0,len(x),32))
            optimizer.zero_grad()
            loss=torch.nn.functional.mse_loss(self.model(xt[ix]),yt[ix])
            loss.backward();optimizer.step()
            if step%25==0 or step==299:
                self.curves.append([step+1,float(loss.detach())])
        self.params=sum(p.numel() for p in self.model.parameters())
        self.storage_bytes=int(self.projection.nbytes+sum(p.numel()*p.element_size() for p in self.model.parameters()))
        return self

    def predict(self,x,c):
        xx,cc=self.normal.transform(x,c)
        if self.kind=="rbf":
            return np.exp(-distances(np.concatenate((xx,cc),1).astype(np.float64),self.train)/self.bandwidth)@self.coef
        import torch
        with torch.inference_mode():
            return self.model(torch.as_tensor(np.concatenate((xx@self.projection,cc),1))).numpy()*self.ys+self.ym

    def save(self,path):
        import torch
        state=dict(self.__dict__)
        state["normal"]=dict(self.normal.__dict__)
        if self.kind=="mlp":
            state["model"]=self.model.state_dict()
        torch.save(state,path)


def risks(delta,prior,target,recordings):
    prediction=prior+delta.reshape(-1,21,3)/1000.
    absolute=np.linalg.norm(prediction-target,axis=-1).mean(-1)*1000.
    ra=np.linalg.norm((prediction-prediction[:,:1])-(target-target[:,:1]),axis=-1).mean(-1)*1000.
    return dict(abs_mm=float(absolute.mean()),ra_mm=float(ra.mean()),
        mse_mm2=float(np.mean(((prediction-target)*1000.)**2)),n=len(prediction),
        per_recording={str(r):dict(n=int(np.sum(recordings==r)),
            abs_mm=float(absolute[recordings==r].mean()),ra_mm=float(ra[recordings==r].mean()))
            for r in np.unique(recordings)})


def decision(report):
    result={}
    for arm in ("slot_gram","point_factor"):
        heads={}
        for kind in ("rbf","mlp"):
            base=report[kind]["production_joint_gram"]["SELECT"]
            cand=report[kind][arm]["SELECT"]
            gains={r:base["per_recording"][r]["ra_mm"]-cand["per_recording"][r]["ra_mm"]
                   for r in base["per_recording"]}
            margin=max(.25,.02*base["ra_mm"])
            gain=base["ra_mm"]-cand["ra_mm"]
            passed=gain>=margin and sum(v>0 for v in gains.values())>=3 and min(gains.values())>=-.5
            heads[kind]=dict(passed=bool(passed),RA_gain_mm=gain,required_gain_mm=margin,
                             per_recording_RA_gain_mm=gains)
        result[arm]=dict(evidence_of_task_cost_at_this_compression_and_budget=all(v["passed"] for v in heads.values()),
                         readers=heads)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prereg",type=Path,required=True)
    p.add_argument("--out",type=Path,default=ROOT/"outputs/s43_revisit_20261028/rep_p2")
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if sha256(args.prereg)!=PROTOCOL_SHA:
        raise ValueError("not the frozen Stage1 protocol")
    if os.environ.get("CUDA_VISIBLE_DEVICES",None)!="":
        raise RuntimeError("Stage1 requires explicit CUDA_VISIBLE_DEVICES=''")
    if args.out.exists():
        raise FileExistsError(args.out)
    protocol=json.loads(args.prereg.read_text())
    if tuple(protocol["P2"]["arms"])!=ARMS:
        raise ValueError("arm mismatch")
    before={name:sha256(ROOT/name) for name in protocol["dependencies"]}
    if before!=protocol["dependencies"]:
        raise RuntimeError("an old dependency/cache no longer matches preregistration")
    import torch
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True)
    implementation=dict(
        schema="s43_revisit_rep_p2_pre_score_implementation_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),preregistration_sha256=PROTOCOL_SHA,
        source_sha256=sha256(__file__),old_dependencies_sha256=before,
        new_dependency_sha256={"tools/s43_revisit/representation_audit.py":
            sha256(ROOT/"tools/s43_revisit/representation_audit.py")},
        amendment="Pre-score implementation detail, no protocol file changed.",
        primary="SELECT frame-mean J0-aligned MPJPE, mm; absolute/MSE/FIT/per-recording also retained",
        arms=list(ARMS),typed_blocks={"A":1/3,"f":1/3,"support":1/3},
        support_blocks={"production_joint_gram":"C+diagG as one joint support block",
                        "slot_gram":"24 upper-triangle G_h as one support block",
                        "point_factor":"cached signedlog(a) and log1p(lambda) each half of support block"},
        baseline_blocks="one feature block; stored image-channel values retained, no duplicate log transform",
        geometry_only="original cached signedlog(a), no events; context remains shared",
        normalization="FIT-only uncentered RMS floor1; clip8; scale each block by sqrt(budget/dim); context exactly old E12 rule",
        rbf="full normalized coordinates; ridge.01; positive FIT squared-distance median bandwidth",
        mlp="fixed Gaussian JL512;128x2 SiLU;300Adam updates;batch32;lr.001;seed3407;float32CPU",
        cache_inversion="float32 logged a/lambda inverse to float64, shared by both Gram arms; cannot restore pre-cache exact64",
        caveats=["No point m, direction-resolved Z, raw events or new geometry available in this cache.",
                 "A larger feature dimension is not a task result; MLP JL can discard distinctions.",
                 "RBF has identical fitted coefficient count but retained exemplar dimension/storage varies.",
                 "No uncertainty/calibration claim; both heads are finite conditional-mean regressors.",
                 "SELECT subjects were seen by canonical S43; held out only from these heads.",
                 "P1 first arithmetic receipt predates Stage1 freeze and is engineering, not preregistered task evidence."],
        prior_scores_opened=False,new_scores_computed_at_manifest_write=False,
        threads=args.threads,cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"])
    write_json(args.out/"rep_p2_implementation.json",implementation)
    cache_path=ROOT/protocol["partition"]["cache"]
    needed=("full","lnes","enhanced","a_only","context","delta","prior_joints","target_joints",
            "partition","recording","sample_id")
    with np.load(cache_path,allow_pickle=False) as archive:
        cache={k:archive[k] for k in needed}
    fit=cache["partition"]=="FIT7"; select=cache["partition"]=="SELECT"
    assert (int(fit.sum()),int(select.sum()))==(224,64)
    assert not any(str(x).startswith("zgz") for x in cache["recording"])
    a,lam=cache_factors(cache["full"])
    report={"rbf":{},"mlp":{}}; predictions={}; started=time.perf_counter()
    for arm in ARMS:
        x,blocks=make_features(arm,cache,a,lam)
        if not np.isfinite(x).all():
            raise RuntimeError("nonfinite feature")
        for kind in ("rbf","mlp"):
            t=time.perf_counter()
            head=Head(kind).fit(x[fit],cache["context"][fit],cache["delta"][fit],blocks)
            pred=head.predict(x,cache["context"])
            row={part:risks(pred[mask],cache["prior_joints"][mask],cache["target_joints"][mask],cache["recording"][mask])
                 for part,mask in (("FIT7",fit),("SELECT",select))}
            row.update(feature_dimensions=int(x.shape[1]),feature_bytes_per_window=int(x.shape[1]*4),
                       fitted_parameters=head.params,reader_storage_bytes=head.storage_bytes,
                       fixed_JL_parameters=int(head.projection.size) if kind=="mlp" else 0,
                       blocks=blocks,curves=head.curves,seconds=time.perf_counter()-t,
                       normalization_scale_range=[float(head.normal.scale.min()),float(head.normal.scale.max())])
            report[kind][arm]=row
            head.save(args.out/f"{kind}_{arm}.pt")
            predictions[f"{kind}__{arm}"]=pred.astype(np.float32)
            write_json(args.out/"partial_metrics.json",report)
            print(f"{kind}/{arm}: FIT RA {row['FIT7']['ra_mm']:.4f}, SELECT RA {row['SELECT']['ra_mm']:.4f}",flush=True)
    after={name:sha256(ROOT/name) for name in before}
    new_unchanged=all(sha256(ROOT/k)==v for k,v in implementation["new_dependency_sha256"].items())
    if before!=after or sha256(__file__)!=implementation["source_sha256"] or not new_unchanged:
        raise RuntimeError("a dependency/cache/probe changed during P2")
    np.savez_compressed(args.out/"predictions.npz",sample_id=cache["sample_id"],partition=cache["partition"],
                        recording=cache["recording"],**predictions)
    result=dict(schema="s43_revisit_rep_p2_results_v1",execution_complete=True,
        preregistration_sha256=PROTOCOL_SHA,implementation_sha256=sha256(args.out/"rep_p2_implementation.json"),
        source_sha256=implementation["source_sha256"],old_source_cache_sha256_before=before,
        source_files_unchanged={k:after[k]==v for k,v in before.items()},metrics=report,
        compression_evidence=decision(report),seconds=time.perf_counter()-started,
        new_zgz_opened=False,old_results_modified=False,scientific_interpretation="finite-reader compression diagnosis, not an information-sufficiency theorem")
    write_json(args.out/"summary.json",result)
    print(json.dumps(result["compression_evidence"],ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
