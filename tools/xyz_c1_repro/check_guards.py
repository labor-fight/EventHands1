#!/usr/bin/env python3
"""CPU-only rejection checks for the dedicated fresh-training entry contract."""
import argparse,contextlib,copy,hashlib,io,json,os,sys
from pathlib import Path
from types import SimpleNamespace
os.environ['PYTHONHASHSEED']='3407'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.xyz_c1_repro import train as W

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--protocol',required=True)
    ap.add_argument('--output-directory',required=True);args=ap.parse_args()
    out=Path(args.output_directory).resolve();out.mkdir(exist_ok=False)
    results={};argv=sys.argv[:]
    for override in('--seed=3408','--max-steps=20','--batch-size=1','--num-workers=1','--resume=/tmp/checkpoint.ckpt','--no-logger'):
        sys.argv=[str(Path(W.__file__)), '--phase','preflight','--protocol',args.protocol,'--config',args.config,override]
        with contextlib.redirect_stderr(io.StringIO())as stderr:
            try:W.main();results[override]={'rejected':False}
            except SystemExit as exc:results[override]={'rejected':exc.code==2,'exit_code':exc.code,'reason':stderr.getvalue()}
    sys.argv=argv
    protocol=out/'fixture_protocol.json';protocol.write_text(json.dumps({'preflight_acceptance_sha256':'1'*64}))
    config=str(Path(args.config).resolve());acceptance=out/'fixture_acceptance.json'
    value={'status':'PASS','preflight_evidence_sha256':'1'*64,
        'bindings':{'wrapper_sha256':W.file_sha(W.__file__),'protocol_sha256':W.file_sha(protocol),
        'config_sha256':{config:W.file_sha(config)},'sampler_by_config':{config:'legacy_none'}}}
    options=SimpleNamespace(acceptance=str(acceptance),protocol=str(protocol),sampler='legacy_none')
    acceptance.write_text(json.dumps(value));W.check_formal_binding(options,config)
    results['complete_binding_positive_control']={'accepted':True}
    for key in('status','wrapper_sha256','protocol_sha256','config_sha256','sampler_by_config','preflight_evidence_sha256'):
        bad=copy.deepcopy(value)
        if key=='status':bad[key]='FAIL'
        elif key=='preflight_evidence_sha256':bad[key]='2'*64
        elif key=='config_sha256':bad['bindings'][key][config]='2'*64
        elif key=='sampler_by_config':bad['bindings'][key][config]='dedicated'
        else:bad['bindings'][key]='2'*64
        acceptance.write_text(json.dumps(bad))
        try:W.check_formal_binding(options,config);results['bad_'+key]={'rejected':False}
        except ValueError as exc:results['bad_'+key]={'rejected':True,'reason':str(exc)}
    acceptance.write_text(json.dumps(value))
    passed=all(x.get('rejected',x.get('accepted',False))for x in results.values())
    report={'status':'PASS'if passed else'FAIL','scope':'CPU-only argument rejection and launch binding validation; original main not reached by negative CLI cases, no training or optimizer updates.',
        'checks':results,'wrapper_sha256':W.file_sha(W.__file__),'checker_sha256':W.file_sha(__file__)}
    (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(results)}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
