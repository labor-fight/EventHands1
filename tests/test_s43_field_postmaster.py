"""Outer chain only advances after exact training/audit gates; no real jobs."""
from pathlib import Path
import sys
import pytest
from tools.s43_revisit import s43_field_postmaster as P

@pytest.mark.parametrize('good',[True,False])
def test_postmaster_fixed_gate_collect_wait_report_order(tmp_path,monkeypatch,good):
 T=P.T;monkeypatch.setattr(T,'OUT',tmp_path/'train');monkeypatch.setattr(T,'ROOT',tmp_path)
 freeze=tmp_path/'freeze.json';freeze.write_text('{}');h=T.sha(freeze)
 monkeypatch.setattr(T,'verify_frozen',lambda *a,**k:None)
 report=tmp_path/'tools/s43_revisit/report_full_field.py';report.parent.mkdir(parents=True);report.write_text('# fake report')
 target=tmp_path/'evaluation';calls=[]
 def execute(cmd,name,logs):
  calls.append(name)
  if name=='training_and_audit':
   T.dump(T.OUT/'all_final_weights_complete.json',dict(status='COMPLETE',sourcefreeze_sha256=h,supervised_outputs_per_arm=6144000 if good else 32,arms=list(T.ARMS)))
   T.dump(T.OUT/'audits/full.json',dict(status='PASS',sourcefreeze_sha256=h))
  elif name=='sealed_collection':
   assert '--wait' in cmd and cmd[cmd.index('--out')+1]==str(target)
   T.dump(target/'evaluation_complete.json',dict(status='COMPLETE'))
  elif name=='saved_artifact_report':assert '--write-docs' in cmd
  return dict(returncode=0,command=cmd)
 monkeypatch.setattr(P,'execute',execute)
 monkeypatch.setattr(sys,'argv',['postmaster','--pilot-controller-pid','123','--source-freeze',str(freeze),'--eval-root',str(target)])
 if good:
  P.main();assert calls==['training_and_audit','sealed_collection','saved_artifact_report'];assert (T.OUT/'training_collection_reporting_complete.json').exists()
 else:
  with pytest.raises(RuntimeError):P.main()
  assert calls==['training_and_audit'];assert (T.OUT/'postmaster_failure.json').exists()
