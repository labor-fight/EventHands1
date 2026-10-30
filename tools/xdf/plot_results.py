#!/usr/bin/env python3
"""Plot diagnostic evidence; never mix probe scores with official model rows."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def read(s):return json.loads((ROOT/'outputs/xdf'/s).read_text())
def main():
    interface=read('interface/summary.json');objective=read('objective/eval_clean.json');roll=read('interface/rollout/summary.json')
    fig,axs=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
    colors=['#15616d','#f4a261'];labels=['F: full','B: no legacy','Wrong legacy','State only','Wrong full'];arms=['F','B','Wlegacy','state','wrongFull']
    for j,seed in enumerate((1701,1702)):
        vals=[next(r['metrics']['check_dz0']['wrist_mm'] for r in interface['results'] if r['interface']==a and r['seed']==seed) for a in arms]
        axs[0].scatter(np.arange(5)+(j-.5)*.18,vals,color=colors[j],s=40,label=f'head seed {seed}')
    q=interface['original_Q1']['check_dz0'];axs[0].axhline(q['wrist_mm'],ls='--',c='#555555',label='Original Q1')
    axs[0].axhline(q['hold_wrist_mm'],ls=':',c='#555555',label='Hold incoming')
    axs[0].set(xticks=np.arange(5),xticklabels=labels,title='A  Legacy content matters',ylabel='Check clean wrist error (mm)',ylim=(0,28));axs[0].tick_params(axis='x',rotation=35);axs[0].legend(fontsize=7,loc='upper left')
    for j,seed in enumerate((1701,1702)):
        vals=[objective['metrics']['zgz_local'][f'z{w}_s{seed}_r0']['metrics']['wrist_mm']['mean'] for w in (1,2)]
        axs[1].plot([0,1],vals,marker='o',color=colors[j],label=str(seed))
    orig=objective['metrics']['zgz_local']['q1'];axs[1].axhline(orig['metrics']['wrist_mm']['mean'],ls='--',c='#555555',label='Original Q1')
    axs[1].axhline(orig['hold_wrist_mm'],ls=':',c='#555555',label='Hold incoming')
    axs[1].set(xticks=[0,1],xticklabels=['L0: z weight 1','L1: z weight 2'],title='B  More z weight is insufficient',ylabel='Local clean wrist error (mm)',ylim=(0,105));axs[1].legend(fontsize=7)
    for j,seq in enumerate(('zgz_local','zgz_global')):
        vals=[roll['sequences'][seq][f'own/{m}']['auc21_actual_time']['wrist_mm']['mean'] for m in ('q1','F_s1701','F_s1702')]
        axs[2].plot(np.arange(3),vals,marker='o',color=colors[j],label=seq)
    axs[2].set(xticks=np.arange(3),xticklabels=['Q1','F 1701','F 1702'],title='C  Local gain, global damage',ylabel='Own-trajectory wrist AUC (mm)',ylim=(0,120));axs[2].legend(fontsize=8)
    for ax in axs:ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    out=ROOT/'docs/assets/xdf_20261014';out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'diagnostic_summary.svg');fig.savefig(out/'diagnostic_summary.png',dpi=170)
    print(str(out/'diagnostic_summary.png'))
if __name__=='__main__':main()
