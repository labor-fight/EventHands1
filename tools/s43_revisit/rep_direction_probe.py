"""Posthoc Stage1 direction attribution, three new CPU-only matched readers.

This is explicitly after the primary P2 scores and before the extension scores.
It never changes the main preregistration or its negative compression conclusion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .representation_audit import ROOT, sha256, write_json
from .rep_readout_probe import (Head, cache_factors, make_features, risks, PROTOCOL_SHA)

EXTENSION_SHA = "0d4935e20423972d39a80f8d551b11fc848aac2bd0aa11eabe7f2120b70a7aa0"
ARMS = ("f_only", "support_only", "cached_no_order_same_support")


def difference(a,b):
    flat=(a-b).reshape(len(a),-1)
    denom=np.linalg.norm(a.reshape(len(a),-1),axis=1)
    return dict(max_absolute=float(np.abs(flat).max()),
        max_per_window_relative_L2=float(np.max(np.linalg.norm(flat,axis=1)/np.maximum(denom,1e-30))),
        bitwise_equal=bool(np.array_equal(a,b)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prereg",type=Path,default=ROOT/"outputs/s43_revisit_20261028/stage1_direction_extension.json")
    p.add_argument("--out",type=Path,default=ROOT/"outputs/s43_revisit_20261028/rep_direction")
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if sha256(args.prereg)!=EXTENSION_SHA:
        raise ValueError("not the registered posthoc direction extension")
    if os.environ.get("CUDA_VISIBLE_DEVICES",None)!="":
        raise RuntimeError("CPU-only stage requires explicit empty CUDA_VISIBLE_DEVICES")
    if args.out.exists():
        raise FileExistsError(args.out)
    extension=json.loads(args.prereg.read_text())
    parent=args.prereg.parent
    protocol_path=parent/extension["parent_protocol"]
    if sha256(protocol_path)!=PROTOCOL_SHA:
        raise ValueError("Stage1 protocol mismatch")
    primary_path=parent/extension["parent_P2_summary"]
    if sha256(primary_path)!=extension["parent_P2_summary_sha256"]:
        raise ValueError("primary P2 result changed")
    protocol=json.loads(protocol_path.read_text())
    primary=json.loads(primary_path.read_text())
    before={name:sha256(ROOT/name) for name in protocol["dependencies"]}
    if before!=protocol["dependencies"]:
        raise RuntimeError("old source/cache mismatch")
    new_sources={name:sha256(ROOT/name) for name in (
        "tools/s43_revisit/representation_audit.py","tools/s43_revisit/rep_readout_probe.py",
        "tools/s43_revisit/rep_direction_probe.py")}
    if new_sources["tools/s43_revisit/rep_readout_probe.py"]!=primary["source_sha256"]:
        raise RuntimeError("P2 reader source changed")
    args.out.mkdir(parents=True)
    write_json(args.out/"implementation_manifest.json",dict(
        schema="s43_revisit_direction_pre_score_implementation_v1",created_utc=datetime.now(timezone.utc).isoformat(),
        posthoc_after_P2=True,extension_sha256=EXTENSION_SHA,parent_protocol_sha256=PROTOCOL_SHA,
        primary_summary_sha256=sha256(primary_path),source_sha256=new_sources,
        old_source_cache_sha256=before,removed_block_budget_redistributed=False,
        normalization_and_readers="Import unchanged frozen P2 implementation; common5337layout and blocks",
        new_scores_computed_at_manifest_write=False))
    import torch
    torch.set_num_threads(args.threads)
    needed=("full","no_order","context","delta","prior_joints","target_joints","partition","recording","sample_id")
    with np.load(ROOT/protocol["partition"]["cache"],allow_pickle=False) as archive:
        cache={k:archive[k] for k in needed}
    fit=cache["partition"]=="FIT7"; select=cache["partition"]=="SELECT"
    assert (int(fit.sum()),int(select.sum()))==(224,64)
    assert not any(str(r).startswith("zgz") for r in cache["recording"])
    a,lam=cache_factors(cache["full"])
    na,nlam=cache_factors(cache["no_order"])
    matching=dict(a=difference(a,na),lambda_values=difference(lam,nlam),
                  cached_logged_f=difference(cache["full"][:,288:1512],cache["no_order"][:,288:1512]),
                  noorder_logged_f_L2=float(np.linalg.norm(cache["no_order"][:,288:1512])),
                  full_logged_f_L2=float(np.linalg.norm(cache["full"][:,288:1512])),
                  evaluated_support="Always rebuilt from full a/lambda, identical in support_only and no_order; identical to P2 production")
    if not np.array_equal(a,na):
        write_json(args.out/"matching_failed.json",matching)
        raise RuntimeError("unexpected different geometry in full/no_order cache; do not interpret as matched intervention")
    write_json(args.out/"support_matching.json",matching)
    full,blocks=make_features("production_joint_gram",cache,a,lam)
    assert full.shape==(288,5337)
    report={"rbf":{},"mlp":{}}; predictions={}; start=time.perf_counter()
    for arm in ARMS:
        x=full.copy()
        if arm=="f_only":x[:,1512:]=0
        elif arm=="support_only":x[:,288:1512]=0
        else:x[:,288:1512]=cache["no_order"][:,288:1512]
        for kind in ("rbf","mlp"):
            head=Head(kind).fit(x[fit],cache["context"][fit],cache["delta"][fit],blocks)
            pred=head.predict(x,cache["context"])
            row={part:risks(pred[mask],cache["prior_joints"][mask],cache["target_joints"][mask],cache["recording"][mask])
                 for part,mask in (("FIT7",fit),("SELECT",select))}
            row.update(feature_dimensions=5337,blocks=blocks,fitted_parameters=head.params,
                       reader_storage_bytes=head.storage_bytes,curves=head.curves)
            report[kind][arm]=row;head.save(args.out/f"{kind}_{arm}.pt")
            predictions[f"{kind}__{arm}"]=pred.astype(np.float32)
            write_json(args.out/"partial_metrics.json",report)
            print(f"{kind}/{arm}: FIT RA {row['FIT7']['ra_mm']:.4f}, SELECT RA {row['SELECT']['ra_mm']:.4f}",flush=True)
    # Preserve the already trained full comparator with explicit artifact identity.
    comparisons={}
    for kind in ("rbf","mlp"):
        base=primary["metrics"][kind]["production_joint_gram"]["SELECT"]
        comparisons[kind]={}
        for arm in ("support_only","cached_no_order_same_support","f_only"):
            alternative=report[kind][arm]["SELECT"]
            comparisons[kind][f"full_minus_{arm}"]={metric:base[metric]-alternative[metric] for metric in ("abs_mm","ra_mm")}
            comparisons[kind][f"full_minus_{arm}"]["per_recording"]={r:{metric:base["per_recording"][r][metric]-alternative["per_recording"][r][metric]
                for metric in ("abs_mm","ra_mm")} for r in base["per_recording"]}
    after={name:sha256(ROOT/name) for name in before}
    new_after={name:sha256(ROOT/name) for name in new_sources}
    if before!=after or new_sources!=new_after:
        raise RuntimeError("source/cache changed during direction attribution")
    np.savez_compressed(args.out/"predictions.npz",sample_id=cache["sample_id"],partition=cache["partition"],
                        recording=cache["recording"],**predictions)
    write_json(args.out/"summary.json",dict(schema="s43_revisit_direction_posthoc_results_v1",
        execution_complete=True,posthoc_after_P2=True,extension_sha256=EXTENSION_SHA,
        primary_protocol_sha256=PROTOCOL_SHA,primary_summary_sha256=sha256(primary_path),
        source_sha256=new_sources,source_files_unchanged={k:after[k]==v for k,v in before.items()},
        matching=matching,metrics=report,full_P2_comparator={kind:primary["metrics"][kind]["production_joint_gram"] for kind in ("rbf","mlp")},
        comparisons_full_minus_alternative=comparisons,seconds=time.perf_counter()-start,
        new_zgz_opened=False,new_success_gate=False,
        scope="Posthoc matched finite-reader direction diagnosis; no unique-statistic or physical-identifiability claim"))
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
