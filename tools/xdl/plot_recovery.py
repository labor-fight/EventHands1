#!/usr/bin/env python3
"""Hash-bound standalone XDL scientific diagnostic figures, without inference.

Formal new figures require the complete five-arm result audit. Undefined or
nonfinite mandatory plotted scores invalidate a figure; no finite-subset means.
"""
from __future__ import annotations
import argparse
import datetime
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'model'))
from tools.xdl import diagnostics as D
from tools.xdl import geometry as G
from tools.xdl.prepare import sha,write_json

ARMS=('f0','q0','f1','q1','n1')
LEGACY=('xa_c0_s3407','xa_c1_s3407')
SEQUENCES=('zgz_global','zgz_local')
BLOCKS=('translation','root','finger')
METRICS={'translation':('wrist_mm','Wrist error (mm)','Translation: ±50 mm'),
         'root':('root_deg','Root error (deg)','Root: ±20 deg'),
         'finger':('finger_deg','Mean local rotation error (deg)','Local joints: ±20 deg')}
STYLE={'f0':('#6C757D','o','-'),'q0':('#4C78A8','o','-'),'f1':('#F58518','o','-'),
       'q1':('#54A24B','o','-'),'n1':('#E45756','o','-'),
       'c0':('#161616','s','--'),'c1':('#8F63A8','D','--')}


def verified_json(path,expected_sha=None):
    raw=Path(path).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and digest!=expected_sha:raise ValueError(f'JSON SHA mismatch: {path}')
    return json.loads(raw),digest


def run_arm(run):return run.split('_')[1]


def run_label(run):
    arm=run_arm(run)
    return ('Legacy '+arm.upper()) if run.startswith('xa_') else arm.upper()


def select_runs(audit,legacy_only=False):
    if not legacy_only:
        if not audit.get('new_stage_complete') or any(audit.get('new_runs',{}).get(a,{}).get('status')!='COMPLETE' for a in ARMS):
            raise ValueError('formal new figures require all five complete, valid arms')
        new=[f'xdl_{a}_s3407' for a in ARMS]
    else:new=[]
    if any(audit.get('legacy_runs',{}).get(r,{}).get('status')!='COMPLETE' for r in LEGACY):
        raise ValueError('legacy figure anchors must both be complete and valid')
    return new+list(LEGACY)


def bank_point(records,seq,block,expected):
    rows=[r for r in records if r['seq']==seq and r['spec']['block']==block and r['spec'].get('scale')=='large']
    if len(rows)!=expected:raise ValueError(f'bank denominator mismatch for {seq}/{block}: {len(rows)} !={expected}')
    metric=METRICS[block][0]
    rho=[r['paired'][block]['rho'] for r in rows];error=[r['endpoint'][metric] for r in rows]
    if not all(r['finite'] for r in rows) or any(v is None for v in rho+error) or not np.isfinite(rho).all() or not np.isfinite(error).all():
        raise ValueError(f'nonfinite/undefined mandatory plot scores for {seq}/{block}; no finite subset')
    return {'n_queries':len(rows),'median_rho':float(np.median(rho)),'mean_gt_error':float(np.mean(error)),
            'metric':metric,'n_packets':len({r.get('packet_id') for r in rows})}


def natural_curve(records,seq,metric):
    rows=[r for r in records if r['seq']==seq and r['spec']['block']=='natural']
    if len(rows)!=24:raise ValueError(f'natural figure requires all24 common clips in {seq}: got{len(rows)}')
    elapsed=[];errors=[];ids=[]
    for row in rows:
        time=np.asarray(row['times_ms'],np.float64);value=np.asarray(row['errors'][metric],np.float64)
        if len(time)!=21 or value.shape!=(21,) or not np.all(np.diff(time)>0):raise ValueError('natural figure requires21 states and strictly increasing actual times')
        if not row['finite'] or not np.isfinite(time).all() or not np.isfinite(value).all():raise ValueError('nonfinite mandatory natural trajectory; no finite subset')
        elapsed.append(time-time[0]);errors.append(value);ids.append(row.get('clip_id'))
    e=np.stack(elapsed);v=np.stack(errors)
    return {'n_clips':24,'n_states':21,'mean_actual_elapsed_ms':e.mean(0).tolist(),'mean_error':v.mean(0).tolist(),
            'actual_elapsed_ms_by_clip':e.tolist(),'errors_by_clip':v.tolist(),'clip_ids':ids,
            'time_aggregation':'arithmetic mean actual elapsed timestamp at each query ordinal; no synthetic50ms replacement or interpolation',
            'sample_scope':'24 paired diagnostic clips, not independent inferential replicates; no CI'}


def audit_record(audit,run):return audit['legacy_runs'][run] if run.startswith('xa_') else audit['new_runs'][run_arm(run)]


def load_section(audit,run,section):
    record=audit_record(audit,run);entry=record['sections'][section]
    if entry['status']!='COMPLETE' or not all(entry['checks'].values()):raise ValueError(f'invalid audited section: {run}/{section}')
    info=entry['raw'];path=Path(info['path'])
    if sha(path)!=info['sha256']:raise ValueError(f'raw SHA changed: {path}')
    payload=D.load_raw(path);metadata=payload['metadata']
    if metadata['bank_sha256']!=audit['bank']['sha256'] or metadata['identity']!=record['identity'] or metadata.get('prediction_batch_size')!=1 or metadata.get('partial_smoke',True):
        raise ValueError(f'raw metadata disagrees with complete serial audit: {run}/{section}')
    return payload['data'],{'path':str(path),'sha256':info['sha256'],'section':section,'run':run,
                            'checkpoint_sha256':metadata['identity']['checkpoint_sha256']}


def validate_packet_keys(bank,records):
    expected={(s,p['packet_id'],spec['id']) for s,m in bank['sequences'].items() if s in SEQUENCES for p in m['bank'] for spec in p['specs']
              if spec['block'] in BLOCKS and spec.get('scale')=='large'}
    observed=[(r['seq'],r['packet_id'],r['spec']['id']) for r in records if r['seq'] in SEQUENCES and r['spec']['block'] in BLOCKS and r['spec'].get('scale')=='large']
    if len(observed)!=len(expected) or len(set(observed))!=len(observed) or set(observed)!=expected:
        raise ValueError('large-bank plotted keys differ from frozen physical probe IDs')


def validate_clip_keys(bank,records):
    expected={(s,c['clip_id']) for s,m in bank['sequences'].items() if s in SEQUENCES for c in m['clips']}
    observed=[(r['seq'],r['clip_id']) for r in records if r['seq'] in SEQUENCES and r['spec']['block']=='natural']
    if len(observed)!=len(expected) or len(set(observed))!=len(observed) or set(observed)!=expected:
        raise ValueError('natural plotted keys differ from all48 frozen diagnostic clips')


def figure_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,
                         'legend.fontsize':10,'svg.fonttype':'none','savefig.facecolor':'white','axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'grid.linewidth':.6})


def legend(runs):
    handles=[]
    for run in runs:
        color,marker,line=STYLE[run_arm(run)];handles.append(Line2D([0],[0],color=color,marker=marker,linestyle=line,label=run_label(run),markersize=6))
    return handles


def save_figure(fig,directory,name):
    paths=[]
    for suffix in ('png','svg'):
        path=directory/f'{name}.{suffix}';fig.savefig(path,dpi=220,bbox_inches='tight');paths.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    plt.close(fig);return paths


def scatter_figure(points,runs,directory,legacy_only):
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.8))
    for i,seq in enumerate(SEQUENCES):
        for j,block in enumerate(BLOCKS):
            ax=axes[i,j]
            for run in runs:
                point=points[run][seq][block];color,marker,_=STYLE[run_arm(run)]
                ax.scatter(point['median_rho'],point['mean_gt_error'],color=color,marker=marker,s=65,edgecolors='white',linewidths=.8,zorder=4)
            ax.axvline(1,color='#9BA1A6',linestyle=':',linewidth=1,zorder=1)
            ax.set_title(seq.replace('zgz_','').capitalize()+' — '+METRICS[block][2])
            ax.set_xlabel('Median paired contraction ρ');ax.set_ylabel(METRICS[block][1]);ax.set_ylim(0,max(1.,max(points[r][seq][block]['mean_gt_error'] for r in runs)*1.12));ax.set_xlim(0,max(1.05,max(points[r][seq][block]['median_rho'] for r in runs)*1.10))
    scope='Legacy frozen weights, seed3407' if legacy_only else 'XDL five arms + legacy anchors, seed3407'
    fig.suptitle(scope+'\nPaired contraction and actual-GT endpoint accuracy',fontsize=14,y=.99)
    fig.legend(handles=legend(runs),loc='upper center',bbox_to_anchor=(.5,.915),ncol=min(len(runs),7),frameon=False)
    fig.text(.5,.015,'ρ < 1 means reduced response to the same-event prior perturbation; it does not prove GT accuracy or closed-loop recovery.\nEach point uses every frozen large probe (24 packets/sequence); no uncertainty intervals or selected subsets.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,top=.835,bottom=.14,hspace=.42,wspace=.33)
    return save_figure(fig,directory,('legacy' if legacy_only else 'xdl')+'_large_bank_rho_gt')


def curve_figure(curves,runs,directory,legacy_only):
    fig,axes=plt.subplots(2,2,figsize=(11.4,8.0))
    for i,seq in enumerate(SEQUENCES):
        for j,(metric,title) in enumerate((('wrist_mm','GT wrist error (mm)'),('mpjpe_ra_mm','GT RA-MPJPE (mm)'))):
            ax=axes[i,j]
            for run in runs:
                value=curves[run][seq][metric];color,marker,line=STYLE[run_arm(run)]
                ax.plot(value['mean_actual_elapsed_ms'],value['mean_error'],color=color,linestyle=line,linewidth=1.8,
                        marker=marker,markersize=3.2,markevery=4,label=run_label(run))
            ax.set_title(seq.replace('zgz_','').capitalize()+' — '+title);ax.set_ylabel(title)
            ax.set_xlabel('Mean actual elapsed time at each query (ms)');ax.set_ylim(bottom=0);ax.set_xlim(left=0)
    scope='Legacy frozen weights, seed3407' if legacy_only else 'XDL five arms + legacy anchors, seed3407'
    fig.suptitle(scope+'\nOwn-state recovery from the common C0 natural prior',fontsize=14,y=.99)
    fig.legend(handles=legend(runs),loc='upper center',bbox_to_anchor=(.5,.915),ncol=min(len(runs),7),frameon=False)
    fig.text(.5,.015,'All24 paired clips/sequence and all21 states retained. After initialization, each model uses its own predictions.\nCurves are arithmetic means, with actual49/50ms timing preserved; no CI or independent-sample inference.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,top=.835,bottom=.14,hspace=.38,wspace=.28)
    return save_figure(fig,directory,('legacy' if legacy_only else 'xdl')+'_natural_recovery_gt')


def plot(audit_path=REPO/'outputs/xdl/results_audit.json',audit_sha=None,directory=None,legacy_only=False):
    audit_path=Path(audit_path);actual=None
    directory=Path(directory or audit_path.parent/'figures');directory.mkdir(parents=True,exist_ok=True)
    name='legacy' if legacy_only else 'xdl';failure_path=directory/f'{name}_figure_failure.json'
    points={};curves={};sources=[]
    try:
        audit,actual=verified_json(audit_path,audit_sha);runs=select_runs(audit,legacy_only);bank_path=Path(audit['bank']['path'])
        bank,_=verified_json(bank_path,audit['bank']['sha256'])
        for run in runs:
            data,source=load_section(audit,run,'bank');sources.append(source);validate_packet_keys(bank,data['rows']);points[run]={}
            for seq in SEQUENCES:
                points[run][seq]={}
                for block in BLOCKS:
                    expected=sum(spec['block']==block and spec.get('scale')=='large' for p in bank['sequences'][seq]['bank'] for spec in p['specs'])
                    points[run][seq][block]=bank_point(data['rows'],seq,block,expected)
            del data;gc.collect()
            data,source=load_section(audit,run,'recovery');sources.append(source);validate_clip_keys(bank,data['rows']);curves[run]={}
            for seq in SEQUENCES:curves[run][seq]={metric:natural_curve(data['rows'],seq,metric) for metric in ('wrist_mm','mpjpe_ra_mm')}
            del data;gc.collect()
        # All raw validation precedes all figure mutations, including mandatory finiteness.
        figure_style();artifacts=scatter_figure(points,runs,directory,legacy_only)+curve_figure(curves,runs,directory,legacy_only)
        payload={'status':'COMPLETE','scope':'legacy-only descriptive figures' if legacy_only else 'five complete fresh runs, seed3407; exploratory comparison',
                 'audit':{'path':str(audit_path),'sha256':actual},'bank':audit['bank'],'producer':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
                 'runs':runs,'sources':sources,'figures':artifacts,'scatter_points':points,'natural_curves':curves,
                 'numeric_policy':'all required frozen probes and clips; nonfinite mandatory scores FAIL, no finite-subset statistics',
                 'claim_limits':'rho is sensitivity contraction, not GT accuracy; actual-GT recovery after common C0 initialization, then own-state loops; no CI/p/repeat or causal mechanism declaration',
                 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        write_json(directory/f'{name}_figures_manifest.json',payload)
        print(json.dumps({'manifest':str(directory/f'{name}_figures_manifest.json'),'figures':[a['path'] for a in artifacts]},ensure_ascii=False),flush=True)
        return payload
    except Exception as error:
        write_json(failure_path,{'status':'FAIL','reason':type(error).__name__,'message':str(error),'audit_sha256':actual,'finite_subset_reduction':False,'sources_loaded':sources})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',default=str(REPO/'outputs/xdl/results_audit.json'));p.add_argument('--audit-sha256');p.add_argument('--out');p.add_argument('--legacy-only',action='store_true')
    args=p.parse_args();plot(args.audit,args.audit_sha256,args.out,args.legacy_only)
